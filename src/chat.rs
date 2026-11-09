//! Chat templates from a model's `tokenizer_config.json`, rendered the way transformers does.

use std::ops::Range;
use std::path::Path;

use anyhow::{Context, Result, bail};
use minijinja::syntax::SyntaxConfig;
use minijinja::value::Serde;
use minijinja::{Environment, Error, ErrorKind, context};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One turn of a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// A model's chat template plus the special tokens it refers to.
pub struct Template {
    env: Environment<'static>,
    bos: String,
    /// The token that ends an assistant turn. The model must learn to emit it, or it never stops.
    pub eos: String,
}

impl Template {
    /// Reads the template from `chat_template.jinja` or the model's tokenizer config.
    /// Returns `None` for base models that ship without one.
    pub fn load(model_dir: &Path, config: &Value) -> Result<Option<Self>> {
        let separate = std::fs::read_to_string(model_dir.join("chat_template.jinja")).ok();
        let Some(source) = separate.or_else(|| inline_template(&config["chat_template"])) else {
            return Ok(None);
        };
        Self::new(
            source,
            token(&config["bos_token"]),
            token(&config["eos_token"]),
        )
        .map(Some)
    }

    fn new(source: String, bos: String, eos: String) -> Result<Self> {
        let mut env = Environment::new();
        // transformers renders with these two on, and templates are written to expect them.
        let syntax = SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .expect("default delimiters are valid");
        env.set_syntax(syntax);
        // Templates call Python string methods such as `.strip()` and `.startswith()`.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> Result<String, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        env.add_template_owned("chat", source)
            .context("the model's chat template doesn't parse")?;
        Ok(Self { env, bos, eos })
    }

    /// Renders a whole conversation as training text, without a trailing generation prompt.
    pub fn render(&self, messages: &[Message]) -> Result<String> {
        let template = self
            .env
            .get_template("chat")
            .expect("added in Template::new");
        let text = template.render(context! {
            messages => Serde(messages),
            bos_token => &self.bos,
            eos_token => &self.eos,
            add_generation_prompt => false,
        })?;
        Ok(text)
    }
}

/// Reads `tokenizer_config.json`, which holds the chat template and the special tokens.
pub fn tokenizer_config(model_dir: &Path) -> Result<Value> {
    let path = model_dir.join("tokenizer_config.json");
    let text = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "can't read {}; the model download may be incomplete",
            path.display()
        )
    })?;
    serde_json::from_str(&text).with_context(|| format!("{} is not valid JSON", path.display()))
}

/// Byte ranges of each assistant reply in `text`, including the end-of-turn token right after it.
///
/// Rendering prefixes of the conversation would be the textbook way, but templates such as Qwen3's
/// render earlier turns differently once later ones exist. Finding each message in order works
/// for every template that copies replies through unchanged, which is all the ones we support.
/// Other turns are only stepped over, so a reply that repeats an earlier question isn't mistaken for it.
pub fn assistant_spans(text: &str, messages: &[Message], eos: &str) -> Result<Vec<Range<usize>>> {
    let mut spans = Vec::new();
    let mut cursor = 0;
    for (i, message) in messages.iter().enumerate() {
        // Templates move system prompts around; Mistral's merges it into the last user turn.
        if message.role == "system" {
            continue;
        }
        let content = message.content.trim();
        let found = text[cursor..].find(content);
        let assistant = message.role == "assistant";
        let start = match found {
            Some(offset) => cursor + offset,
            None if assistant => bail!(
                "the chat template rewrote reply {} instead of copying it; remove `<think>` sections \
                 from earlier replies, or split the conversation into one example per reply",
                i + 1
            ),
            None => continue,
        };
        let mut end = start + content.len();
        if text[end..].starts_with(eos) {
            end += eos.len();
        }
        if assistant {
            spans.push(start..end);
        }
        cursor = end;
    }
    Ok(spans)
}

/// Old configs store a list of named templates; the one named "default" is for chat.
fn inline_template(value: &Value) -> Option<String> {
    match value {
        Value::String(source) => Some(source.clone()),
        Value::Array(named) => named
            .iter()
            .find(|t| t["name"] == "default")
            .and_then(|t| t["template"].as_str())
            .map(String::from),
        _ => None,
    }
}

/// Special tokens appear either as plain strings or as `{"content": ...}` objects.
pub fn token(value: &Value) -> String {
    value
        .as_str()
        .or_else(|| value["content"].as_str())
        .unwrap_or_default()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMOLLM2: &str = "tests/fixtures/smollm2";

    fn chat(turns: &[(&str, &str)]) -> Vec<Message> {
        turns
            .iter()
            .map(|(role, content)| Message {
                role: role.to_string(),
                content: content.to_string(),
            })
            .collect()
    }

    #[test]
    fn renders_like_transformers() {
        let dir = Path::new(SMOLLM2);
        let config = tokenizer_config(dir).unwrap();
        let template = Template::load(dir, &config).unwrap().unwrap();
        let text = template
            .render(&chat(&[("user", "Hi"), ("assistant", "Hello!")]))
            .unwrap();
        assert_eq!(
            text,
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n"
        );
    }

    #[test]
    fn spans_cover_replies_and_their_end_token_only() {
        // Shaped like Mistral's template, which moves the system prompt into the last user turn.
        let text = "[INST] Yes [/INST]Yes</s>[INST] Be brief.\n\nok [/INST]Sure</s>";
        let messages = chat(&[
            ("system", "Be brief."),
            ("user", "Yes"),
            ("assistant", "Yes"),
            ("user", "ok"),
            ("assistant", " Sure "),
        ]);
        let spans = assistant_spans(text, &messages, "</s>").unwrap();
        let replies: Vec<&str> = spans.into_iter().map(|s| &text[s]).collect();
        assert_eq!(replies, ["Yes</s>", "Sure</s>"]);
    }

    #[test]
    fn python_string_methods_and_raise_exception_work() {
        let source =
            "{% if messages[0].role != 'user' %}{{ raise_exception('user first') }}{% endif %}\
                      {{ messages[0].content.strip().upper() }}"
                .to_string();
        let template = Template::new(source, String::new(), String::new()).unwrap();
        assert_eq!(template.render(&chat(&[("user", " hi ")])).unwrap(), "HI");
        let error = template.render(&chat(&[("system", "x")])).unwrap_err();
        assert!(format!("{error:#}").contains("user first"));
    }
}

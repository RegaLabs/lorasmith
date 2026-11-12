//! Turns one line of a data file into tokens, in the format the model expects.

use std::ops::Range;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::chat::{self, Message, Template};
use crate::data::Example;

/// The shapes a line of training data can take.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Format {
    Messages,
    PromptCompletion,
    Text,
}

#[derive(Deserialize)]
struct Conversation {
    messages: Vec<Message>,
}

#[derive(Deserialize)]
struct PromptCompletion {
    prompt: String,
    completion: String,
}

#[derive(Deserialize)]
struct Text {
    text: String,
}

/// The parts of a model folder needed to turn text into tokens.
pub struct Encoder {
    tokenizer: Tokenizer,
    template: Option<Template>,
    eos: String,
}

/// Which format a line is in, judged by its keys.
pub fn detect(value: &Value) -> Result<Format> {
    let has = |key| value.get(key).is_some();
    if has("messages") {
        Ok(Format::Messages)
    } else if has("prompt") && has("completion") {
        Ok(Format::PromptCompletion)
    } else if has("text") {
        Ok(Format::Text)
    } else {
        bail!("expected a `messages` list, `prompt` and `completion`, or `text`")
    }
}

/// `format` in the words an error message uses for it.
pub fn describe(format: Format) -> &'static str {
    match format {
        Format::Messages => "a `messages` list",
        Format::PromptCompletion => "`prompt` and `completion`",
        Format::Text => "`text`",
    }
}

impl Encoder {
    /// Reads the tokenizer and chat template of the model in `model_dir`.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(anyhow::Error::msg)
            .with_context(|| {
                format!(
                    "can't load {}; the model download may be incomplete",
                    path.display()
                )
            })?;
        let config = chat::tokenizer_config(model_dir)?;
        Ok(Self {
            tokenizer,
            template: Template::load(model_dir, &config)?,
            eos: chat::token(&config["eos_token"]),
        })
    }

    /// Tokenizes one line and marks the tokens to learn from.
    pub fn encode(&self, format: Format, value: Value) -> Result<Example> {
        match format {
            Format::Messages => {
                let line: Conversation = serde_json::from_value(value).context(
                    "`messages` must be a list of objects with string `role` and `content`",
                )?;
                self.conversation(&line.messages)
            }
            Format::PromptCompletion => {
                let line: PromptCompletion = serde_json::from_value(value)
                    .context("`prompt` and `completion` must be strings")?;
                self.prompt_completion(line)
            }
            Format::Text => {
                let line: Text =
                    serde_json::from_value(value).context("`text` must be a string")?;
                let text = line.text + &self.eos;
                let everything = 0..text.len();
                self.tokenize(&text, std::slice::from_ref(&everything), true)
            }
        }
    }

    fn conversation(&self, messages: &[Message]) -> Result<Example> {
        let Some(template) = &self.template else {
            bail!("this model has no chat template; use `text` lines, or pick an instruct model");
        };
        let text = template.render(messages)?;
        let spans = chat::assistant_spans(&text, messages, &template.eos)?;
        // Templates already contain every special token, so the tokenizer must not add more.
        self.tokenize(&text, &spans, false)
    }

    /// Instruct models get the pair as a chat, because that is how Ollama will prompt them later.
    fn prompt_completion(&self, line: PromptCompletion) -> Result<Example> {
        if self.template.is_some() {
            let turn = |role: &str, content| Message {
                role: role.to_string(),
                content,
            };
            return self.conversation(&[
                turn("user", line.prompt),
                turn("assistant", line.completion),
            ]);
        }
        let text = line.prompt.clone() + &line.completion + &self.eos;
        let completion = line.prompt.len()..text.len();
        self.tokenize(&text, std::slice::from_ref(&completion), true)
    }

    /// Tokens overlapping any of `spans` count toward the loss.
    fn tokenize(&self, text: &str, spans: &[Range<usize>], special: bool) -> Result<Example> {
        let encoding = self
            .tokenizer
            .encode(text, special)
            .map_err(anyhow::Error::msg)?;
        let train = encoding
            .get_offsets()
            .iter()
            .map(|&(start, end)| spans.iter().any(|s| start < s.end && end > s.start))
            .collect();
        Ok(Example {
            tokens: encoding.get_ids().to_vec(),
            train,
        })
    }
}

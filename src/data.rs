//! Training examples from a JSONL file.

use std::ops::Range;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::chat::{self, Message, Template};

/// One tokenized example. `train[i]` says whether token `i` counts toward the loss.
#[derive(Debug, Clone)]
pub struct Example {
    pub tokens: Vec<u32>,
    pub train: Vec<bool>,
}

/// Everything read from a data file, with counts worth telling the user about.
#[derive(Debug, Default)]
pub struct Dataset {
    pub examples: Vec<Example>,
    /// Examples cut down to the length limit.
    pub truncated: usize,
    /// Examples left with nothing to learn from, usually because truncation removed the answer.
    pub skipped: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Format {
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
struct Encoder {
    tokenizer: Tokenizer,
    template: Option<Template>,
    eos: String,
}

/// Reads `path` and tokenizes every line for the model in `model_dir`, keeping at most `max_len` tokens each.
pub fn load(path: &Path, model_dir: &Path, max_len: usize) -> Result<Dataset> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("can't read {}; check the path", path.display()))?;
    let encoder = Encoder::load(model_dir)?;
    let mut first: Option<(Format, usize)> = None;
    let mut dataset = Dataset::default();

    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let number = index + 1;
        let at = || format!("{} line {number}", path.display());
        let value: Value = serde_json::from_str(line).with_context(|| {
            format!(
                "{}: not valid JSON; each line must be one JSON object",
                at()
            )
        })?;
        let format = detect(&value).with_context(at)?;
        let (expected, first_line) = *first.get_or_insert((format, number));
        if format != expected {
            bail!(
                "{}: expected {} like line {first_line}; use one format per file",
                at(),
                describe(expected)
            );
        }
        let mut example = encoder.encode(format, value).with_context(at)?;
        if example.tokens.len() > max_len {
            example.tokens.truncate(max_len);
            example.train.truncate(max_len);
            dataset.truncated += 1;
        }
        // The first token is never predicted, so it can't be the only one marked for training.
        if example.train.iter().skip(1).any(|&t| t) {
            dataset.examples.push(example);
        } else {
            dataset.skipped += 1;
        }
    }
    if dataset.examples.is_empty() {
        bail!(
            "no usable examples in {}; lines must hold `messages`, `prompt` and `completion`, or `text`",
            path.display()
        );
    }
    Ok(dataset)
}

fn detect(value: &Value) -> Result<Format> {
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

fn describe(format: Format) -> &'static str {
    match format {
        Format::Messages => "a `messages` list",
        Format::PromptCompletion => "`prompt` and `completion`",
        Format::Text => "`text`",
    }
}

impl Encoder {
    fn load(model_dir: &Path) -> Result<Self> {
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

    fn encode(&self, format: Format, value: Value) -> Result<Example> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const SMOLLM2: &str = "tests/fixtures/smollm2";

    fn jsonl(name: &str, lines: &[&str]) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("loraforge-{name}-{}.jsonl", std::process::id()));
        std::fs::write(&path, lines.join("\n")).unwrap();
        path
    }

    fn trained_text(example: &Example) -> String {
        let tokenizer = Tokenizer::from_file(Path::new(SMOLLM2).join("tokenizer.json")).unwrap();
        let ids: Vec<u32> = example
            .tokens
            .iter()
            .zip(&example.train)
            .filter(|(_, train)| **train)
            .map(|(id, _)| *id)
            .collect();
        tokenizer.decode(&ids, false).unwrap()
    }

    fn error(path: &Path) -> String {
        format!("{:#}", load(path, Path::new(SMOLLM2), 512).unwrap_err())
    }

    #[test]
    fn conversations_train_on_assistant_replies_only() {
        let line = concat!(
            r#"{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}, "#,
            r#"{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "See you"}]}"#,
        );
        let path = jsonl("messages", &[line]);
        let dataset = load(&path, Path::new(SMOLLM2), 512).unwrap();
        assert_eq!(
            trained_text(&dataset.examples[0]),
            "Hello!<|im_end|>See you<|im_end|>"
        );
    }

    #[test]
    fn prompt_completion_trains_on_the_completion_as_a_chat() {
        let path = jsonl("pc", &[r#"{"prompt": "2+2?", "completion": "4"}"#]);
        let dataset = load(&path, Path::new(SMOLLM2), 512).unwrap();
        assert_eq!(trained_text(&dataset.examples[0]), "4<|im_end|>");
    }

    #[test]
    fn text_trains_on_everything_and_ends_with_eos() {
        let path = jsonl(
            "text",
            &[
                r#"{"text": "The sky is blue."}"#,
                "",
                r#"{"text": "Grass is green."}"#,
            ],
        );
        let dataset = load(&path, Path::new(SMOLLM2), 512).unwrap();
        assert_eq!(dataset.examples.len(), 2);
        assert_eq!(
            trained_text(&dataset.examples[1]),
            "Grass is green.<|im_end|>"
        );
    }

    #[test]
    fn long_examples_are_cut_and_answerless_ones_skipped() {
        let long_answer =
            r#"{"prompt": "Count", "completion": "one two three four five six seven eight"}"#;
        let long_prompt = format!(
            r#"{{"prompt": "{}", "completion": "ok"}}"#,
            "word ".repeat(100)
        );
        let path = jsonl("truncate", &[long_answer, &long_prompt]);
        let dataset = load(&path, Path::new(SMOLLM2), 32).unwrap();
        assert_eq!(dataset.examples.len(), 1);
        assert_eq!(dataset.examples[0].tokens.len(), 32);
        assert_eq!((dataset.truncated, dataset.skipped), (2, 1));
    }

    #[test]
    fn bad_lines_are_reported_with_their_number() {
        let mixed = jsonl(
            "mixed",
            &[r#"{"text": "a"}"#, r#"{"prompt": "b", "completion": "c"}"#],
        );
        assert!(
            error(&mixed).contains("line 2: expected `text` like line 1"),
            "{}",
            error(&mixed)
        );
        let broken = jsonl("broken", &[r#"{"text": "a"}"#, "", r#"{"text": "b""#]);
        assert!(
            error(&broken).contains("line 3: not valid JSON"),
            "{}",
            error(&broken)
        );
        let typed = jsonl("typed", &[r#"{"messages": "hello"}"#]);
        assert!(
            error(&typed).contains("line 1: `messages` must be a list"),
            "{}",
            error(&typed)
        );
    }
}

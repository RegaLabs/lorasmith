//! Training examples from a JSONL file.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::encode::{Encoder, Format, describe, detect};

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

/// Tokenizes every line of `text`, read from `path`, for the model in `model_dir`, keeping
/// at most `max_len` tokens each.
pub fn load(text: &str, path: &Path, model_dir: &Path, max_len: usize) -> Result<Dataset> {
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

/// Reads a data file as UTF-8 text. Separate from `load` so a file saved in the wrong
/// encoding is reported before the model downloads.
pub fn read(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("can't read {}; check the path", path.display()))?;
    let resave = format!(
        "save it as UTF-8 (Notepad: File > Save As > Encoding UTF-8; PowerShell: \
         Get-Content {0} | Set-Content -Encoding utf8 fixed.jsonl)",
        path.display()
    );
    if bytes.starts_with(&[0xFF, 0xFE]) || bytes.starts_with(&[0xFE, 0xFF]) {
        bail!("{} is saved as UTF-16; {resave}", path.display());
    }
    let text = String::from_utf8(bytes)
        .map_err(|_| anyhow::anyhow!("{} isn't UTF-8 text; {resave}", path.display()))?;
    // Notepad and PowerShell 5.1 often start UTF-8 files with a byte-order mark, which JSON parsers reject.
    Ok(match text.strip_prefix('\u{feff}') {
        Some(rest) => rest.to_string(),
        None => text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tokenizers::Tokenizer;

    const SMOLLM2: &str = "tests/fixtures/smollm2";

    fn jsonl(name: &str, lines: &[&str]) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("lorasmith-{name}-{}.jsonl", std::process::id()));
        std::fs::write(&path, lines.join("\n")).unwrap();
        path
    }

    fn trained_text(example: &Example) -> String {
        trained_text_in(SMOLLM2, example)
    }

    fn trained_text_in(model_dir: &str, example: &Example) -> String {
        let tokenizer = Tokenizer::from_file(Path::new(model_dir).join("tokenizer.json")).unwrap();
        let ids: Vec<u32> = example
            .tokens
            .iter()
            .zip(&example.train)
            .filter(|(_, train)| **train)
            .map(|(id, _)| *id)
            .collect();
        tokenizer.decode(&ids, false).unwrap()
    }

    fn load_file(path: &Path, model_dir: &str, max_len: usize) -> Result<Dataset> {
        load(&read(path)?, path, Path::new(model_dir), max_len)
    }

    fn error(path: &Path) -> String {
        format!("{:#}", load_file(path, SMOLLM2, 512).unwrap_err())
    }

    #[test]
    fn conversations_train_on_assistant_replies_only() {
        let line = concat!(
            r#"{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}, "#,
            r#"{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "See you"}]}"#,
        );
        let path = jsonl("messages", &[line]);
        let dataset = load_file(&path, SMOLLM2, 512).unwrap();
        assert_eq!(
            trained_text(&dataset.examples[0]),
            "Hello!<|im_end|>See you<|im_end|>"
        );
    }

    // Spans are byte ranges, so they only line up if the tokenizer reports byte offsets too.
    #[test]
    fn non_ascii_conversations_train_on_the_reply_only() {
        let line = concat!(
            r#"{"messages": [{"role": "user", "content": "Grüße, 日本語?"}, "#,
            r#"{"role": "assistant", "content": "こんにちは 🌍 café"}]}"#,
        );
        let path = jsonl("non-ascii-messages", &[line]);
        let dataset = load_file(&path, SMOLLM2, 512).unwrap();
        assert_eq!(
            trained_text(&dataset.examples[0]),
            "こんにちは 🌍 café<|im_end|>"
        );
    }

    #[test]
    fn non_ascii_base_model_prompts_train_on_the_completion_only() {
        let base_model = "tests/fixtures/sentencepiece";
        let line = r#"{"prompt": "Grüße, 日本語?", "completion": " こんにちは 🌍 café"}"#;
        let path = jsonl("non-ascii-pc", &[line]);
        let dataset = load_file(&path, base_model, 512).unwrap();
        // The sentencepiece decoder drops the leading space of whatever it decodes first.
        assert_eq!(
            trained_text_in(base_model, &dataset.examples[0]),
            "こんにちは 🌍 café</s>"
        );
    }

    #[test]
    fn prompt_completion_trains_on_the_completion_as_a_chat() {
        let path = jsonl("pc", &[r#"{"prompt": "2+2?", "completion": "4"}"#]);
        let dataset = load_file(&path, SMOLLM2, 512).unwrap();
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
        let dataset = load_file(&path, SMOLLM2, 512).unwrap();
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
        let dataset = load_file(&path, SMOLLM2, 32).unwrap();
        assert_eq!(dataset.examples.len(), 1);
        assert_eq!(dataset.examples[0].tokens.len(), 32);
        assert_eq!((dataset.truncated, dataset.skipped), (2, 1));
    }

    #[test]
    fn a_utf8_byte_order_mark_is_ignored() {
        let path = jsonl("bom", &["\u{feff}{\"text\": \"a\"}", r#"{"text": "b"}"#]);
        let dataset = load_file(&path, SMOLLM2, 512).unwrap();
        assert_eq!(dataset.examples.len(), 2);
    }

    #[test]
    fn utf16_files_get_re_save_advice() {
        let path = jsonl("utf16", &[]);
        let mut bytes = vec![0xFF, 0xFE];
        bytes.extend(r#"{"text": "a"}"#.encode_utf16().flat_map(u16::to_le_bytes));
        std::fs::write(&path, bytes).unwrap();
        let message = error(&path);
        assert!(
            message.contains("saved as UTF-16; save it as UTF-8"),
            "{message}"
        );
        assert!(!message.contains("check the path"), "{message}");
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

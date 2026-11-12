//! The vocabulary of SentencePiece tokenizers (Mistral, Llama 2) for llama.cpp's "llama"
//! tokenizer, following `_create_vocab_sentencepiece` in its converter. llama.cpp merges
//! pieces by score, and the scores exist only in `tokenizer.model`, a protobuf file.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value as Json;

use super::Value;
use super::vocab::{CONTROL, Keys, NORMAL, UNUSED, USER_DEFINED, looks_special};

const UNKNOWN: i32 = 2;
const BYTE: i32 = 6;

struct Piece {
    text: String,
    score: f32,
    kind: i32,
}

enum Payload<'a> {
    Number(u64),
    Bytes(&'a [u8]),
}

/// The tokens with their scores and types as GGUF keys, and the tokens themselves.
/// `tokenizer` is the parsed `tokenizer.json`, which lists the tokens added on top.
pub fn keys(model_dir: &Path, tokenizer: &Json, vocab_size: usize) -> Result<(Vec<String>, Keys)> {
    let path = model_dir.join("tokenizer.model");
    let bytes = std::fs::read(&path).with_context(|| format!("can't read {}", path.display()))?;
    let pieces = pieces(&bytes).with_context(|| format!("can't use {}", path.display()))?;

    let mut tokens: Vec<String> = (0..vocab_size).map(|i| format!("[PAD{i}]")).collect();
    let mut scores = vec![-10000.0; vocab_size];
    let mut types = vec![UNUSED; vocab_size];
    for (id, piece) in pieces.into_iter().enumerate().take(vocab_size) {
        tokens[id] = piece.text;
        scores[id] = piece.score;
        // llama.cpp has no use for SentencePiece's own "user defined" mark.
        types[id] = match piece.kind {
            UNKNOWN | CONTROL | UNUSED | BYTE => piece.kind,
            _ => NORMAL,
        };
    }
    // llama.cpp reads these from `added_tokens_decoder` in tokenizer_config.json, which
    // transformers 5 no longer writes. tokenizer.json has the same list in every version.
    for token in tokenizer["added_tokens"].as_array().into_iter().flatten() {
        let content = token["content"].as_str().unwrap_or_default();
        let Some(id) = token["id"].as_u64().map(|id| id as usize) else {
            bail!("tokenizer.json has the added token `{content}` without an id");
        };
        if id >= vocab_size {
            continue;
        }
        if token["special"] == true || looks_special(content) {
            tokens[id] = content.to_string();
            types[id] = CONTROL;
        } else {
            // llama.cpp matches user-defined tokens against raw text, where a space is a space.
            tokens[id] = content.replace('\u{2581}', " ");
            types[id] = USER_DEFINED;
        }
        scores[id] = -1000.0;
    }

    let keys = vec![
        (
            "tokenizer.ggml.model".to_string(),
            Value::Str("llama".into()),
        ),
        (
            "tokenizer.ggml.pre".to_string(),
            Value::Str("default".into()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            Value::Strs(tokens.clone()),
        ),
        ("tokenizer.ggml.scores".to_string(), Value::F32s(scores)),
        ("tokenizer.ggml.token_type".to_string(), Value::I32s(types)),
    ];
    Ok((tokens, keys))
}

/// The pieces of a serialized SentencePiece `ModelProto`: its repeated field 1, each a
/// message of text (1), score (2) and type (3). Everything else in the file is skipped.
fn pieces(mut model: &[u8]) -> Result<Vec<Piece>> {
    let mut pieces = Vec::new();
    while !model.is_empty() {
        let (1, Payload::Bytes(mut message)) = field(&mut model)? else {
            continue;
        };
        let mut piece = Piece {
            text: String::new(),
            score: 0.0,
            kind: NORMAL,
        };
        while !message.is_empty() {
            match field(&mut message)? {
                (1, Payload::Bytes(text)) => {
                    piece.text =
                        String::from_utf8(text.to_vec()).context("a piece is not valid UTF-8")?;
                }
                (2, Payload::Bytes(score)) => {
                    let score = score.try_into().context("a score is not 4 bytes")?;
                    piece.score = f32::from_le_bytes(score);
                }
                (3, Payload::Number(kind)) => piece.kind = kind as i32,
                _ => {}
            }
        }
        pieces.push(piece);
    }
    ensure!(!pieces.is_empty(), "it holds no pieces");
    Ok(pieces)
}

/// Takes one protobuf field off the front of `input`: its number and its payload.
fn field<'a>(input: &mut &'a [u8]) -> Result<(u64, Payload<'a>)> {
    let tag = varint(input)?;
    let payload = match tag & 7 {
        0 => Payload::Number(varint(input)?),
        1 => Payload::Bytes(take(input, 8)?),
        2 => {
            let len = varint(input)? as usize;
            Payload::Bytes(take(input, len)?)
        }
        5 => Payload::Bytes(take(input, 4)?),
        other => bail!("it is not a SentencePiece model (protobuf wire type {other})"),
    };
    Ok((tag >> 3, payload))
}

fn varint(input: &mut &[u8]) -> Result<u64> {
    let mut value = 0;
    for shift in (0..64).step_by(7) {
        let byte = take(input, 1)?[0];
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    bail!("it is not a SentencePiece model (a number runs past 64 bits)")
}

fn take<'a>(input: &mut &'a [u8], len: usize) -> Result<&'a [u8]> {
    ensure!(input.len() >= len, "the file is cut short");
    let (head, rest) = input.split_at(len);
    *input = rest;
    Ok(head)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIR: &str = "tests/fixtures/sentencepiece";

    fn json(file: &str) -> Json {
        let text = std::fs::read_to_string(Path::new(DIR).join(file)).unwrap();
        serde_json::from_str(&text).unwrap()
    }

    #[test]
    fn pieces_read_as_sentencepiece_reports_them() {
        let pieces = pieces(&std::fs::read(Path::new(DIR).join("tokenizer.model")).unwrap());
        let pieces = pieces.unwrap();
        assert_eq!(pieces.len(), 300);
        let piece = |id: usize| (pieces[id].text.as_str(), pieces[id].score, pieces[id].kind);
        assert_eq!(piece(0), ("<unk>", 0.0, UNKNOWN));
        assert_eq!(piece(2), ("</s>", 0.0, CONTROL));
        assert_eq!(piece(3), ("[INST]", 0.0, USER_DEFINED));
        assert_eq!(piece(259), ("<0xFF>", 0.0, BYTE));
        assert_eq!(piece(270), ("\u{2581}l", -10.0, NORMAL));
    }

    #[test]
    fn vocabulary_matches_what_llama_cpp_writes() {
        let (tokens, keys) = keys(Path::new(DIR), &json("tokenizer.json"), 320).unwrap();
        let expected = json("expected.json");
        let strings: Vec<&str> = expected["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap())
            .collect();
        assert_eq!(tokens, strings);
        let numbers = |key: &str| -> Vec<f64> {
            expected[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect()
        };
        let Value::F32s(scores) = &keys[3].1 else {
            panic!("scores come fourth")
        };
        let Value::I32s(types) = &keys[4].1 else {
            panic!("types come fifth")
        };
        let scores: Vec<f64> = scores.iter().map(|s| f64::from(*s)).collect();
        let types: Vec<f64> = types.iter().map(|t| f64::from(*t)).collect();
        assert_eq!(scores, numbers("scores"));
        assert_eq!(types, numbers("types"));
        assert_eq!(keys[0].1, Value::Str("llama".into()));
    }

    #[test]
    fn tokens_beyond_the_embeddings_are_left_out() {
        let (tokens, _) = keys(Path::new(DIR), &json("tokenizer.json"), 280).unwrap();
        assert_eq!(tokens.len(), 280);
        assert_eq!(tokens[279], "n");
    }

    #[test]
    fn a_broken_model_file_is_an_error_not_a_panic() {
        let bytes = std::fs::read(Path::new(DIR).join("tokenizer.model")).unwrap();
        assert!(pieces(&bytes[..bytes.len() - 1]).is_err());
        assert!(pieces(b"not a protobuf file at all").is_err());
        assert!(pieces(&[]).is_err());
    }
}

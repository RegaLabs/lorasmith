//! The tokenizer as GGUF metadata, following what llama.cpp's `convert_hf_to_gguf.py`
//! writes for the same model: its "gpt2" tokenizer for byte-level BPE, "llama" for SentencePiece.

use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use serde_json::Value as Json;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use super::{Value, sentencepiece};
use crate::chat;
use crate::config::Config;

/// llama.cpp names each pre-tokenizer by hashing the tokens of this text.
const CHECK_TEXT: &str = include_str!("pretokenizer_check.txt");
/// `hash<TAB>name` pairs, copied from llama.cpp (MIT licensed).
const PRETOKENIZERS: &str = include_str!("pretokenizers.tsv");

/// GGUF metadata keys with their values, in the order they are written.
pub(super) type Keys = Vec<(String, Value)>;

pub(super) const NORMAL: i32 = 1;
pub(super) const CONTROL: i32 = 3;
pub(super) const USER_DEFINED: i32 = 4;
pub(super) const UNUSED: i32 = 5;

/// Tokenizer keys for the GGUF header.
pub fn metadata(model_dir: &Path, config: &Config) -> Result<Vec<(String, Value)>> {
    let path = model_dir.join("tokenizer.json");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("can't read {}", path.display()))?;
    let json: Json = serde_json::from_str(&text)
        .with_context(|| format!("{} is not valid JSON", path.display()))?;
    let tokenizer = Tokenizer::from_str(&text).map_err(anyhow::Error::msg)?;
    let tokenizer_config = chat::tokenizer_config(model_dir)?;
    let model_config: Json =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json"))?)?;

    // Like llama.cpp's converter, take a SentencePiece model when there is one.
    let bpe = json["model"]["type"] == "BPE";
    let (tokens, mut keys) = if bpe && model_dir.join("tokenizer.model").exists() {
        let (tokens, mut keys) = sentencepiece::keys(model_dir, &json, config.vocab_size)?;
        if let Some(prefix) = tokenizer_config["add_prefix_space"].as_bool() {
            let key = "tokenizer.ggml.add_space_prefix".to_string();
            keys.push((key, Value::Bool(prefix)));
        }
        (tokens, keys)
    } else if bpe && json["model"]["byte_fallback"] != true {
        byte_level(&json, &tokenizer, config.vocab_size)?
    } else {
        bail!(
            "GGUF export supports byte-level BPE tokenizers (Llama 3, Qwen, SmolLM) and \
             SentencePiece ones that come with a `tokenizer.model` file (Mistral, Llama 2). \
             If the base model has a `tokenizer.model`, copy it into {} and train again. \
             The adapter and merged model are still usable",
            model_dir.display()
        );
    };
    let ids: HashMap<&str, u32> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i as u32))
        .collect();

    let mut special = HashMap::new();
    for (kind, key) in [
        ("bos", "bos_token_id"),
        ("eos", "eos_token_id"),
        ("unk", "unknown_token_id"),
        ("pad", "padding_token_id"),
    ] {
        let from_tokenizer = ids
            .get(chat::token(&tokenizer_config[format!("{kind}_token")]).as_str())
            .copied();
        let from_config = model_config[format!("{kind}_token_id")]
            .as_u64()
            .map(|id| id as u32);
        if let Some(id) = from_tokenizer.or(from_config) {
            special.insert(kind, id);
            keys.push((format!("tokenizer.ggml.{key}"), Value::U32(id)));
        }
    }
    // Whether the tokenizer adds BOS or EOS on its own is easiest to learn by asking it.
    let sample = tokenizer.encode("a", true).map_err(anyhow::Error::msg)?;
    let sample = sample.get_ids();
    let add_bos = special
        .get("bos")
        .is_some_and(|bos| sample.first() == Some(bos));
    let add_eos = special
        .get("eos")
        .is_some_and(|eos| sample.last() == Some(eos));
    keys.push((
        "tokenizer.ggml.add_bos_token".to_string(),
        Value::Bool(add_bos),
    ));
    keys.push((
        "tokenizer.ggml.add_eos_token".to_string(),
        Value::Bool(add_eos),
    ));
    if let Some(template) = chat::template_source(model_dir, &tokenizer_config) {
        keys.push(("tokenizer.chat_template".to_string(), Value::Str(template)));
    }
    Ok(keys)
}

fn byte_level(
    json: &Json,
    tokenizer: &Tokenizer,
    vocab_size: usize,
) -> Result<(Vec<String>, Keys)> {
    let (tokens, types) = tokens(json, vocab_size)?;
    let keys = vec![
        (
            "tokenizer.ggml.model".to_string(),
            Value::Str("gpt2".into()),
        ),
        (
            "tokenizer.ggml.pre".to_string(),
            Value::Str(pretokenizer(tokenizer)?),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            Value::Strs(tokens.clone()),
        ),
        ("tokenizer.ggml.token_type".to_string(), Value::I32s(types)),
        (
            "tokenizer.ggml.merges".to_string(),
            Value::Strs(merges(json)?),
        ),
    ];
    Ok((tokens, keys))
}

/// Every id up to the model's vocabulary size, with its llama.cpp token type.
/// Models often reserve more embedding rows than the tokenizer uses; those become padding.
fn tokens(json: &Json, vocab_size: usize) -> Result<(Vec<String>, Vec<i32>)> {
    let mut tokens: Vec<String> = (0..vocab_size).map(|i| format!("[PAD{i}]")).collect();
    let mut types = vec![UNUSED; vocab_size];
    let vocab = json["model"]["vocab"]
        .as_object()
        .context("tokenizer.json has no vocabulary")?;
    let added = json["added_tokens"].as_array().cloned().unwrap_or_default();
    let entries = vocab
        .iter()
        .map(|(token, id)| (token.as_str(), id.as_u64(), NORMAL))
        .chain(added.iter().map(|t| {
            let content = t["content"].as_str().unwrap_or_default();
            let kind = if t["special"] == true || looks_special(content) {
                CONTROL
            } else {
                USER_DEFINED
            };
            (content, t["id"].as_u64(), kind)
        }));
    for (token, id, kind) in entries {
        let id = id.context("tokenizer.json has a token without an id")? as usize;
        if id >= vocab_size {
            bail!("token `{token}` has id {id}, beyond the model's {vocab_size} embeddings");
        }
        tokens[id] = token.to_string();
        types[id] = kind;
    }
    Ok((tokens, types))
}

/// Some models forget to mark control tokens as special; llama.cpp catches them by shape.
pub(super) fn looks_special(token: &str) -> bool {
    (token.starts_with("<|") && token.ends_with("|>")) || matches!(token, "<pad>" | "<mask>")
}

/// Merges as "left right" strings. Newer tokenizer files store pairs, which may contain
/// spaces; llama.cpp expects those spaces shifted to U+0120 as GPT-2's byte encoding does.
fn merges(json: &Json) -> Result<Vec<String>> {
    let list = json["model"]["merges"]
        .as_array()
        .context("tokenizer.json has no merges")?;
    list.iter()
        .map(|merge| match merge {
            Json::String(pair) => Ok(pair.clone()),
            Json::Array(pair) if pair.len() == 2 => {
                let part = |p: &Json| p.as_str().unwrap_or_default().replace(' ', "\u{120}");
                Ok(format!("{} {}", part(&pair[0]), part(&pair[1])))
            }
            _ => bail!("tokenizer.json has merges in an unknown format"),
        })
        .collect()
}

fn pretokenizer(tokenizer: &Tokenizer) -> Result<String> {
    let ids = tokenizer
        .encode(CHECK_TEXT, true)
        .map_err(anyhow::Error::msg)?;
    // Python's str() of a list of ints, which is what llama.cpp hashes.
    let hash: String = Sha256::digest(format!("{:?}", ids.get_ids()))
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    PRETOKENIZERS
        .lines()
        .find_map(|line| line.strip_prefix(&hash)?.strip_prefix('\t'))
        .map(String::from)
        .with_context(|| {
            format!(
                "llama.cpp doesn't know this tokenizer's pre-tokenizer (fingerprint {hash}), \
                 so a GGUF file would tokenize text wrongly. The adapter and merged model are still usable"
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smollm2() -> (Vec<(String, Value)>, HashMap<String, Value>) {
        let dir = Path::new("tests/fixtures/smollm2");
        let config =
            Config::parse(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        let keys = metadata(dir, &config).unwrap();
        let map = keys.iter().cloned().collect();
        (keys, map)
    }

    #[test]
    fn smollm2_matches_what_llama_cpp_writes() {
        let (_, keys) = smollm2();
        assert_eq!(keys["tokenizer.ggml.pre"], Value::Str("smollm".into()));
        assert_eq!(keys["tokenizer.ggml.eos_token_id"], Value::U32(2));
        assert_eq!(keys["tokenizer.ggml.add_bos_token"], Value::Bool(false));
        let Value::Strs(tokens) = &keys["tokenizer.ggml.tokens"] else {
            panic!()
        };
        let Value::I32s(types) = &keys["tokenizer.ggml.token_type"] else {
            panic!()
        };
        assert_eq!(tokens.len(), 49152);
        assert_eq!((tokens[1].as_str(), types[1]), ("<|im_start|>", CONTROL));
        assert_eq!(types[200], NORMAL);
        assert!(
            matches!(&keys["tokenizer.chat_template"], Value::Str(t) if t.contains("im_start"))
        );
    }

    #[test]
    fn sentencepiece_models_use_the_llama_tokenizer() {
        let dir = Path::new("tests/fixtures/sentencepiece");
        let config =
            Config::parse(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        let keys: HashMap<String, Value> = metadata(dir, &config).unwrap().into_iter().collect();
        assert_eq!(keys["tokenizer.ggml.model"], Value::Str("llama".into()));
        assert_eq!(keys["tokenizer.ggml.pre"], Value::Str("default".into()));
        assert!(matches!(&keys["tokenizer.ggml.scores"], Value::F32s(s) if s.len() == 320));
        assert!(!keys.contains_key("tokenizer.ggml.merges"));
        assert_eq!(keys["tokenizer.ggml.bos_token_id"], Value::U32(1));
        assert_eq!(keys["tokenizer.ggml.eos_token_id"], Value::U32(2));
        assert_eq!(keys["tokenizer.ggml.unknown_token_id"], Value::U32(0));
        assert_eq!(keys["tokenizer.ggml.add_bos_token"], Value::Bool(true));
        assert_eq!(keys["tokenizer.ggml.add_eos_token"], Value::Bool(false));
    }

    #[test]
    fn sentencepiece_without_its_model_file_says_what_to_do() {
        let dir = std::env::temp_dir().join(format!("lorasmith-spm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for file in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            std::fs::copy(
                Path::new("tests/fixtures/sentencepiece").join(file),
                dir.join(file),
            )
            .unwrap();
        }
        let config =
            Config::parse(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        let error = metadata(&dir, &config).unwrap_err().to_string();
        assert!(error.contains("tokenizer.model"), "{error}");
    }

    #[test]
    fn merge_pairs_with_spaces_use_the_byte_encoding() {
        let json: Json = serde_json::json!({"model": {"merges": [["a b", "c"], "d e"]}});
        assert_eq!(merges(&json).unwrap(), ["a\u{120}b c", "d e"]);
    }
}

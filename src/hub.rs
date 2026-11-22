//! Finds a model by local folder, short alias or Hugging Face repo id, downloading what's missing.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use hf_hub::Cache;
use hf_hub::api::sync::{ApiBuilder, ApiError, ApiRepo};

use crate::config::Config;

/// Short names in the style of Ollama, for instruct models that train well on a laptop.
const ALIASES: &[(&str, &str)] = &[
    ("smollm2:135m", "HuggingFaceTB/SmolLM2-135M-Instruct"),
    ("smollm2:360m", "HuggingFaceTB/SmolLM2-360M-Instruct"),
    ("smollm2:1.7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("qwen2.5:0.5b", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen2.5:1.5b", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("qwen2.5:3b", "Qwen/Qwen2.5-3B-Instruct"),
    ("qwen2.5:7b", "Qwen/Qwen2.5-7B-Instruct"),
    ("qwen3:0.6b", "Qwen/Qwen3-0.6B"),
    ("qwen3:1.7b", "Qwen/Qwen3-1.7B"),
    ("qwen3:4b", "Qwen/Qwen3-4B"),
    ("qwen3:8b", "Qwen/Qwen3-8B"),
    ("llama3.2:1b", "meta-llama/Llama-3.2-1B-Instruct"),
    ("llama3.2:3b", "meta-llama/Llama-3.2-3B-Instruct"),
    ("mistral:7b", "mistralai/Mistral-7B-Instruct-v0.3"),
];

/// Files besides the weights that the merged model folder should carry along.
const EXTRAS: &[&str] = &[
    // SentencePiece tokenizers keep the piece scores GGUF needs only in this file.
    "tokenizer.model",
    "generation_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "LICENSE",
    "LICENSE.txt",
];

/// A model ready to load: where its files are and what it is called on the Hub.
pub struct Model {
    pub id: String,
    pub dir: PathBuf,
    pub config: Config,
}

/// Resolves `name` to a local folder, downloading config, tokenizer and weights if needed.
/// The config is read before any weights download, so unsupported models fail fast, and
/// so does a model that `on_config` turns down, for example for being too big.
pub fn fetch(
    name: &str,
    progress: bool,
    on_config: impl FnOnce(&Config) -> Result<()>,
) -> Result<Model> {
    if Path::new(name).is_dir() {
        let config = read_config(&Path::new(name).join("config.json"))?;
        on_config(&config)?;
        return Ok(Model {
            id: name.to_string(),
            dir: PathBuf::from(name),
            config,
        });
    }
    let id = resolve(name)?;
    let token = std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| Cache::from_env().token());
    let has_token = token.is_some();
    let api = ApiBuilder::from_env()
        .with_progress(progress)
        .with_token(token)
        .build()
        .context("can't set up the Hugging Face client")?;
    let repo = api.model(id.to_string());
    let files: Vec<String> = match repo.info() {
        Ok(info) => info.siblings.into_iter().map(|s| s.rfilename).collect(),
        Err(error) => return cached(id, error, on_config),
    };
    download(id, &repo, &files, has_token, on_config)
}

fn resolve(name: &str) -> Result<&str> {
    if let Some((_, id)) = ALIASES.iter().find(|(alias, _)| *alias == name) {
        return Ok(id);
    }
    if name.contains('/') {
        return Ok(name);
    }
    let known: Vec<&str> = ALIASES.iter().map(|(alias, _)| *alias).collect();
    bail!(
        "`{name}` is not a folder, a known short name or a Hugging Face id like `Qwen/Qwen2.5-0.5B-Instruct`.\n\
         Short names: {}",
        known.join(", ")
    )
}

fn download(
    id: &str,
    repo: &ApiRepo,
    files: &[String],
    has_token: bool,
    on_config: impl FnOnce(&Config) -> Result<()>,
) -> Result<Model> {
    let get = |file: &str| {
        repo.get(file).map_err(|error| {
            let (status, cause) = http_status(error);
            anyhow::anyhow!(download_hint(id, file, status, has_token, &cause))
        })
    };
    let config_path = get("config.json")?;
    let config = read_config(&config_path)?;
    on_config(&config)?;
    for file in ["tokenizer.json", "tokenizer_config.json"] {
        if !files.iter().any(|f| f == file) {
            bail!(
                "{id} has no {file}; lorasmith needs a fast tokenizer. Try another upload of this model"
            );
        }
        get(file)?;
    }
    for file in files.iter().filter(|f| EXTRAS.contains(&f.as_str())) {
        get(file)?;
    }
    for file in weight_files(id, files, &get)? {
        get(&file)?;
    }
    let dir = config_path
        .parent()
        .expect("downloaded files live in a folder")
        .to_path_buf();
    Ok(Model {
        id: id.to_string(),
        dir,
        config,
    })
}

/// Gated repos publish their file list to anyone, so only the file download itself is
/// refused; the HTTP status is the one sign of that. Returns the status and the error text.
fn http_status(error: ApiError) -> (Option<u16>, String) {
    match error {
        ApiError::RequestError(request) => {
            let cause = request.to_string();
            (request.into_response().map(|r| r.status()), cause)
        }
        other => (None, other.to_string()),
    }
}

/// What to tell the user when `file` can't be downloaded from `id`. `has_token` says whether
/// a token was sent: Hugging Face answers 401 to a gated download both without one and with
/// one it doesn't accept, and telling someone who set HF_TOKEN to set it doesn't help.
fn download_hint(
    id: &str,
    file: &str,
    status: Option<u16>,
    has_token: bool,
    cause: &str,
) -> String {
    let license = format!("https://huggingface.co/{id}");
    let tokens = "https://huggingface.co/settings/tokens";
    match status {
        Some(401) if has_token => format!(
            "Hugging Face didn't accept your token for {id}, so it won't send {file} ({cause}).
             Check that HF_TOKEN holds a current read token from {tokens}, and that you accepted              the license at {license}. Or pick an ungated model such as qwen2.5:1.5b"
        ),
        Some(401) => format!(
            "{id} is gated, so Hugging Face won't send {file} ({cause}).
             Accept its license at {license}, create a read token at {tokens}, set HF_TOKEN to              it and run again. Or pick an ungated model such as qwen2.5:1.5b"
        ),
        Some(403) => format!(
            "your token has no access to {id} yet, so Hugging Face won't send {file} ({cause}).
             Accept its license at {license} and wait for approval if it asks for one. A              fine-grained token also needs read access to public gated repos. Or pick an              ungated model such as qwen2.5:1.5b"
        ),
        _ => format!(
            "can't download {file} from {id} ({cause}); check your connection and try again"
        ),
    }
}

/// Lets training run offline when an earlier download is complete.
fn cached(
    id: &str,
    error: ApiError,
    on_config: impl FnOnce(&Config) -> Result<()>,
) -> Result<Model> {
    let Some(config_path) = Cache::from_env().model(id.to_string()).get("config.json") else {
        bail!(
            "can't reach {id} on Hugging Face ({error}).\n\
             Check the name at https://huggingface.co/{id}. Gated models such as Llama and Mistral \
             need you to accept their license there and set HF_TOKEN"
        );
    };
    let config = read_config(&config_path)?;
    on_config(&config)?;
    Ok(Model {
        id: id.to_string(),
        config,
        dir: config_path
            .parent()
            .expect("cached files live in a folder")
            .to_path_buf(),
    })
}

/// Sharded models list their shards in an index. Some repos also carry the same weights in
/// another layout (Mistral's `consolidated.safetensors`), which we must not download twice.
fn weight_files(
    id: &str,
    files: &[String],
    get: &impl Fn(&str) -> Result<PathBuf>,
) -> Result<Vec<String>> {
    const INDEX: &str = "model.safetensors.index.json";
    if files.iter().any(|f| f == "model.safetensors") {
        return Ok(vec!["model.safetensors".to_string()]);
    }
    if !files.iter().any(|f| f == INDEX) {
        bail!("{id} has no model.safetensors weights; try another upload of this model");
    }
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(get(INDEX)?)?)
        .with_context(|| format!("{id} has a broken {INDEX}"))?;
    let mut shards: Vec<String> = index["weight_map"]
        .as_object()
        .with_context(|| format!("{id} has a broken {INDEX}"))?
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    shards.sort();
    shards.dedup();
    Ok(shards)
}

fn read_config(path: &Path) -> Result<Config> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("can't read {}", path.display()))?;
    Config::parse(&text).with_context(|| format!("can't use {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_resolve_to_hub_ids() {
        assert_eq!(
            resolve("qwen2.5:0.5b").unwrap(),
            "Qwen/Qwen2.5-0.5B-Instruct"
        );
        assert_eq!(
            resolve("someone/their-model").unwrap(),
            "someone/their-model"
        );
        let error = resolve("qwen9:1b").unwrap_err().to_string();
        assert!(error.contains("smollm2:135m"), "{error}");
    }

    #[test]
    fn local_folders_are_used_as_they_are() {
        let model = fetch("tests/fixtures/qwen3", false, |_| Ok(())).unwrap();
        assert_eq!(model.config.model_type, "qwen3");
    }

    #[test]
    fn a_model_the_caller_turns_down_is_not_fetched() {
        let error = fetch("tests/fixtures/qwen3", false, |_| bail!("too big")).err();
        assert_eq!(error.map(|e| e.to_string()), Some("too big".to_string()));
    }

    #[test]
    fn a_refused_download_says_how_to_get_access() {
        let id = "meta-llama/Llama-3.2-1B-Instruct";
        let hint = |status, has_token| download_hint(id, "config.json", status, has_token, "x");
        for (status, has_token) in [(401, false), (401, true), (403, true)] {
            let hint = hint(Some(status), has_token);
            assert!(
                hint.contains(&format!("https://huggingface.co/{id}")),
                "{hint}"
            );
            assert!(hint.contains("qwen2.5:1.5b"), "{hint}");
            assert!(!hint.contains("connection"), "{hint}");
        }
        assert!(hint(Some(401), false).contains("set HF_TOKEN"));
        assert!(hint(Some(401), true).contains("didn't accept your token"));
        assert!(hint(Some(403), true).contains("no access"));
        for status in [None, Some(500)] {
            assert!(hint(status, true).contains("check your connection"));
        }
    }

    #[test]
    #[ignore = "downloads about 270 MB"]
    fn downloads_a_small_model() {
        let model = fetch("smollm2:135m", false, |_| Ok(())).unwrap();
        assert!(model.dir.join("model.safetensors").exists());
        assert!(model.dir.join("tokenizer.json").exists());
    }
}

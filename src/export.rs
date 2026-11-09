//! Writes a trained model in layouts other tools load unchanged: a PEFT adapter for
//! Hugging Face and vLLM, a merged model folder, and a Modelfile for Ollama.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use burn::module::Module;
use burn::store::{
    BurnToPyTorchAdapter, FloatCastAdapter, ModuleAdapter, ModuleSnapshot, SafetensorsStore,
};
use burn::tensor::{DType, Device, Tensor, TensorData};
use safetensors::{Dtype, tensor::TensorView};
use serde_json::json;

use crate::model::Llama;

/// File types that hold the base model's weights, which the merged model replaces.
const WEIGHTS: &[&str] = &[
    ".safetensors",
    ".bin",
    ".pth",
    ".pt",
    ".gguf",
    ".index.json",
];

/// Writes the adapter, the merged model under `merged/`, `model.gguf` and a `Modelfile` into `dir`.
/// `base` names the base model for the adapter config, `model_dir` holds its files.
/// Returns why no GGUF file was written, if one wasn't; the Modelfile then points at `merged/`.
pub fn all(
    model: Llama,
    base: &str,
    model_dir: &Path,
    dir: &Path,
) -> Result<Option<anyhow::Error>> {
    std::fs::create_dir_all(dir).with_context(|| format!("can't create {}", dir.display()))?;
    // Merging builds a second full set of weights. RAM has room for that; a small GPU,
    // still holding the trained model, often doesn't.
    let model = model.to_device(&Device::flex());
    adapter(&model, base, dir)?;
    merged(model, model_dir, &dir.join("merged"))?;
    // Ollama runs GGUF on every machine, but runs a safetensors folder only where it has MLX.
    let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("model");
    let gguf = crate::gguf::convert(&dir.join("merged"), name, &dir.join("model.gguf"));
    let from = if gguf.is_ok() {
        "./model.gguf"
    } else {
        "./merged"
    };
    std::fs::write(dir.join("Modelfile"), format!("FROM {from}\n"))
        .context("can't write the Modelfile")?;
    Ok(gguf.err())
}

/// Writes `adapter_config.json` and `adapter_model.safetensors` as HF PEFT does.
pub fn adapter(model: &Llama, base: &str, dir: &Path) -> Result<()> {
    let mut tensors = Vec::new();
    let mut targets = Vec::new();
    let mut rank_and_scale = None;
    for (name, linear) in crate::lora::projections(model) {
        let Some(lora) = linear.weight.adapter() else {
            continue;
        };
        // Burn keeps `x @ a @ b` with a: [in, r] and b: [r, out]; PEFT stores both transposed.
        let key = format!("base_model.model.{name}");
        tensors.push((format!("{key}.lora_A.weight"), lora.a.val().transpose()));
        tensors.push((format!("{key}.lora_B.weight"), lora.b.val().transpose()));
        let module = name.rsplit('.').next().expect("names have a module part");
        if !targets.iter().any(|t| t == module) {
            targets.push(module.to_string());
        }
        rank_and_scale = Some((lora.a.dims()[1], lora.scale));
    }
    let Some((rank, scale)) = rank_and_scale else {
        bail!("the model has no LoRA adapters to export");
    };
    write_safetensors(tensors, &dir.join("adapter_model.safetensors"))?;

    let config = json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base,
        "r": rank,
        "lora_alpha": scale * rank as f64,
        "lora_dropout": 0.0,
        "target_modules": targets,
        "bias": "none",
        "fan_in_fan_out": false,
        "use_rslora": false,
        "inference_mode": true,
    });
    let text = serde_json::to_string_pretty(&config).expect("plain JSON values serialize");
    std::fs::write(dir.join("adapter_config.json"), text).context("can't write adapter_config.json")
}

/// Folds the adapters into the base weights and writes a plain HF model folder,
/// copying the tokenizer and config files from `model_dir`.
pub fn merged(model: Llama, model_dir: &Path, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("can't create {}", dir.display()))?;
    for entry in std::fs::read_dir(model_dir)
        .with_context(|| format!("can't read {}", model_dir.display()))?
    {
        let path = entry?.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if path.is_file() && !name.starts_with('.') && !WEIGHTS.iter().any(|w| name.ends_with(w)) {
            std::fs::copy(&path, dir.join(name))
                .with_context(|| format!("can't copy {} into {}", name, dir.display()))?;
        }
    }

    // transformers refuses safetensors files whose "format" metadata isn't "pt".
    let to_hf = BurnToPyTorchAdapter.chain(FloatCastAdapter::to(published_dtype(model_dir)));
    let mut store = SafetensorsStore::from_file(dir.join("model.safetensors"))
        .with_to_adapter(to_hf)
        .with_key_remapping(r"^(embed_tokens|layers|norm)", "model.$1")
        .clear_metadata()
        .metadata("format", "pt")
        .overwrite(true);
    model
        .materialize()
        .save_into(&mut store)
        .with_context(|| format!("can't write the merged model to {}", dir.display()))
}

/// The precision the base model was published in. Training runs in f32, but writing the
/// merged model that way would make it twice the size of the original for no gain.
fn published_dtype(model_dir: &Path) -> DType {
    let config: serde_json::Value = std::fs::read_to_string(model_dir.join("config.json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default();
    // transformers 5 writes `dtype`; older versions wrote `torch_dtype`.
    match config["dtype"].as_str().or(config["torch_dtype"].as_str()) {
        Some("bfloat16") => DType::BF16,
        Some("float16") => DType::F16,
        _ => DType::F32,
    }
}

fn write_safetensors(tensors: Vec<(String, Tensor<2>)>, path: &Path) -> Result<()> {
    let data: Vec<(String, Vec<usize>, TensorData)> = tensors
        .into_iter()
        .map(|(name, tensor)| {
            (
                name,
                tensor.dims().to_vec(),
                tensor.into_data().convert::<f32>(),
            )
        })
        .collect();
    let views = data
        .iter()
        .map(|(name, shape, data)| {
            let view = TensorView::new(Dtype::F32, shape.clone(), data.as_bytes())?;
            Ok((name.as_str(), view))
        })
        .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
    let metadata = HashMap::from([("format".to_string(), "pt".to_string())]);
    safetensors::serialize_to_file(views, Some(metadata), path)
        .with_context(|| format!("can't write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Example;
    use crate::train::{self, Options};
    use burn::tensor::{Int, Tolerance};
    use std::path::PathBuf;

    const BASE: &str = "tests/fixtures/llama";

    fn trained(device: &Device) -> Llama {
        let (_, model) = crate::load::model(Path::new(BASE), device).unwrap();
        let model = train::attach(model, &Options::default());
        let example = Example {
            tokens: vec![3, 1, 4, 1, 5, 9, 2, 6],
            train: vec![true; 8],
        };
        let options = Options {
            learning_rate: 1e-2,
            epochs: 5,
            batch_size: 1,
            ..Options::default()
        };
        train::train(model, &[example], &[], &options, device, |_| ())
    }

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("loraforge-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn logits(model: &Llama, device: &Device) -> TensorData {
        let tokens =
            Tensor::<2, Int>::from_data(TensorData::new(vec![2i64, 7, 1, 8], [1, 4]), device);
        model.forward(tokens).into_data()
    }

    #[test]
    fn merged_model_reloads_with_the_same_outputs() {
        let device = Device::flex().autodiff();
        let model = trained(&device);
        let expected = logits(&model, &device);
        let dir = temp("merged");
        merged(model, Path::new(BASE), &dir).unwrap();

        let (_, reloaded) = crate::load::model(&dir, &device).unwrap();
        logits(&reloaded, &device).assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-4));
        assert!(dir.join("config.json").exists());
    }

    #[test]
    fn modelfile_falls_back_to_the_merged_folder_without_gguf() {
        let device = Device::flex().autodiff();
        let dir = temp("all");
        // The tiny fixture has no tokenizer, so no GGUF file can be written.
        let skipped = all(trained(&device), "tiny-llama", Path::new(BASE), &dir).unwrap();
        assert!(skipped.is_some());
        assert_eq!(
            std::fs::read_to_string(dir.join("Modelfile")).unwrap(),
            "FROM ./merged\n"
        );
    }

    #[test]
    fn merged_model_keeps_the_published_precision() {
        assert_eq!(
            published_dtype(Path::new("tests/fixtures/smollm2")),
            DType::BF16
        );
        assert_eq!(published_dtype(Path::new(BASE)), DType::F32);
    }

    #[test]
    fn adapter_follows_the_peft_layout() {
        let device = Device::flex().autodiff();
        let model = trained(&device);
        let dir = temp("adapter");
        adapter(&model, "tiny-llama", &dir).unwrap();

        let config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("adapter_config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(config["r"], 16);
        assert_eq!(config["lora_alpha"], 32.0);
        assert_eq!(config["target_modules"].as_array().unwrap().len(), 7);

        // PEFT computes W[out, in] + (alpha / r) * B @ A, which must equal our effective weight.
        let bytes = std::fs::read(dir.join("adapter_model.safetensors")).unwrap();
        let file = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let load = |name: &str| {
            let view = file
                .tensor(&format!(
                    "base_model.model.model.layers.1.mlp.up_proj.{name}"
                ))
                .unwrap();
            let values: Vec<f32> = view
                .data()
                .chunks(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            Tensor::<2>::from_data(
                TensorData::new(values, [view.shape()[0], view.shape()[1]]),
                &device,
            )
        };
        let weight = &model.layers[1].mlp.up_proj.weight;
        let peft =
            weight.base().transpose() + load("lora_B.weight").matmul(load("lora_A.weight")) * 2.0;
        let ours = weight.val().transpose();
        assert!(
            weight
                .adapter()
                .unwrap()
                .b
                .val()
                .abs()
                .max()
                .into_scalar::<f32>()
                > 0.0
        );
        peft.into_data()
            .assert_approx_eq::<f32>(&ours.into_data(), Tolerance::absolute(1e-5));
    }
}

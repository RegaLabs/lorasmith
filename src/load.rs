//! Loading Hugging Face checkpoints from a local folder.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burn::store::{
    FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use burn::tensor::{Device, FloatDType};

use crate::config::Config;
use crate::model::Llama;

/// Reads `config.json` and every `*.safetensors` shard in `dir` into a ready model whose
/// weights are stored, and so computed with, as `dtype`.
pub fn model(dir: &Path, device: &Device, dtype: FloatDType) -> Result<(Config, Llama)> {
    let path = dir.join("config.json");
    let json =
        fs::read_to_string(&path).with_context(|| format!("can't read {}", path.display()))?;
    let config = Config::parse(&json).with_context(|| format!("in {}", path.display()))?;
    let mut model = Llama::new(&config, device);
    weights(&mut model, dir, dtype)?;
    Ok((config, model))
}

/// Fills `model` with the weights in `dir`, cast to `dtype`. Fails if any weight is missing
/// from every shard.
pub fn weights(model: &mut Llama, dir: &Path, dtype: FloatDType) -> Result<()> {
    let mut applied = BTreeSet::new();
    let mut missing = BTreeSet::new();
    for shard in shards(dir)? {
        // Checkpoints usually ship in bf16, which WGSL GPUs can't compute in.
        let mut store = SafetensorsStore::from_file(&shard)
            .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(dtype.into())))
            .with_key_remapping(r"^model\.", "")
            .allow_partial(true);
        let result = model
            .load_from(&mut store)
            .with_context(|| format!("can't read {}", shard.display()))?;
        if let Some(error) = result.errors.first() {
            bail!("{} doesn't fit this model: {error:?}", shard.display());
        }
        applied.extend(result.applied);
        missing.extend(result.missing.into_iter().map(|(path, _)| path));
    }
    let missing: Vec<_> = missing.difference(&applied).collect();
    if !missing.is_empty() {
        bail!(
            "{} is missing {} weights, starting with `{}`. The download may be incomplete.",
            dir.display(),
            missing.len(),
            missing[0]
        );
    }
    Ok(())
}

fn shards(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut shards: Vec<PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("can't open {}", dir.display()))?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    if shards.is_empty() {
        bail!("no .safetensors files in {}", dir.display());
    }
    shards.sort();
    Ok(shards)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tiny;
    use burn::store::BurnToPyTorchAdapter;
    use burn::tensor::{DType, Int, Tensor, TensorData};

    fn logits(model: &Llama, device: &Device) -> Vec<f32> {
        let tokens =
            Tensor::<2, Int>::from_data(TensorData::new(vec![3i64, 1, 4, 1], [1, 4]), device);
        model
            .forward(tokens)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
    }

    #[test]
    fn loads_back_a_checkpoint_saved_in_hugging_face_layout() {
        let device = Device::flex();
        let config = tiny("qwen2");
        let original = Llama::new(&config, &device);
        let dir = std::env::temp_dir().join(format!("loraforge-load-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let mut store = SafetensorsStore::from_file(dir.join("model.safetensors"))
            .with_to_adapter(BurnToPyTorchAdapter)
            .with_key_remapping(r"^(embed_tokens|layers|norm)", "model.$1");
        original.save_into(&mut store).unwrap();

        let mut loaded = Llama::new(&config, &device);
        weights(&mut loaded, &dir, FloatDType::F32).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!(logits(&original, &device), logits(&loaded, &device));
    }

    #[test]
    fn bf16_checkpoints_load_in_the_precision_asked_for() {
        let device = Device::flex();
        let config = tiny("llama");
        let dir = std::env::temp_dir().join(format!("loraforge-bf16-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let mut store = SafetensorsStore::from_file(dir.join("model.safetensors"))
            .with_to_adapter(BurnToPyTorchAdapter.chain(FloatCastAdapter::to(DType::BF16)))
            .with_key_remapping(r"^(embed_tokens|layers|norm)", "model.$1");
        Llama::new(&config, &device).save_into(&mut store).unwrap();

        for dtype in [FloatDType::F32, FloatDType::F16] {
            let mut loaded = Llama::new(&config, &device);
            weights(&mut loaded, &dir, dtype).unwrap();
            let up_proj = loaded.layers[0].mlp.up_proj.weight.val();
            assert_eq!(up_proj.dtype(), DType::from(dtype));
            assert_eq!(loaded.embed_tokens.weight.val().dtype(), DType::from(dtype));
            assert_eq!(loaded.is_half(), dtype == FloatDType::F16);
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn refuses_a_folder_without_weights() {
        let dir = std::env::temp_dir().join(format!("loraforge-empty-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let mut model = Llama::new(&tiny("llama"), &Device::flex());
        let error = weights(&mut model, &dir, FloatDType::F32).unwrap_err();
        fs::remove_dir_all(&dir).unwrap();
        assert!(error.to_string().contains("no .safetensors"));
    }
}

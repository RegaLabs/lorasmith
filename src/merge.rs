//! Folds trained adapters into the published weights, one tensor at a time, so merging
//! never needs a whole model in memory.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use burn::tensor::{Device, Tensor, TensorData};
use serde_json::json;

use crate::model::Llama;
use crate::shards::{self, Source};

/// What training adds to one weight, in the [out, in] layout Hugging Face stores:
/// `scale * b @ a`.
pub struct Update {
    b: Tensor<2>,
    a: Tensor<2>,
    scale: f64,
}

/// The update for every adapted projection of `model`, by the name of the weight it
/// changes, without the `model.` prefix that some checkpoints leave out.
pub fn updates(model: &Llama) -> HashMap<String, Update> {
    let cpu = Device::flex();
    // Burn keeps `x @ a @ b` with a: [in, r] and b: [r, out], both transposed from here.
    let stored = |factor: Tensor<2>| Tensor::from_data(factor.transpose().into_data(), &cpu);
    let mut all = HashMap::new();
    for (name, layer) in crate::lora::projections(model) {
        if let Some(lora) = layer.weight.adapter() {
            let update = Update {
                b: stored(lora.b.val()),
                a: stored(lora.a.val()),
                scale: lora.scale,
            };
            all.insert(format!("{}.weight", key(&name)), update);
        }
    }
    all
}

/// Writes every weight in `model_dir` into the safetensors file `path`, with `updates`
/// added to the weights they name. All others are copied byte for byte.
pub fn write(updates: &HashMap<String, Update>, model_dir: &Path, path: &Path) -> Result<()> {
    let mut tensors = shards::tensors(model_dir)?;
    let found = tensors
        .iter()
        .filter(|(name, ..)| updates.contains_key(key(name)))
        .count();
    ensure!(
        found == updates.len(),
        "{} holds {found} of the {} weights that were trained, so there is nothing to merge \
         the rest into. The adapter is saved; download the model again and rerun to get a \
         merged model.",
        model_dir.display(),
        updates.len()
    );
    // The order safetensors itself writes: the widest values first, then by name.
    tensors.sort_by_key(|(_, _, source)| Reverse(source.dtype().size()));

    // transformers refuses safetensors files whose "format" metadata isn't "pt".
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_string(), json!({ "format": "pt" }));
    let mut end = 0;
    for (name, shape, source) in &tensors {
        let start = end;
        end += shape.iter().product::<usize>() * source.dtype().size();
        let dtype = shards::dtype_name(source.dtype()).expect("shards reads only named dtypes");
        let entry = json!({ "dtype": dtype, "shape": shape, "data_offsets": [start, end] });
        header.insert(name.clone(), entry);
    }
    let mut header = serde_json::to_vec(&header).expect("plain JSON values serialize");
    // Readers expect the weights to start on an 8-byte boundary.
    header.resize(header.len().next_multiple_of(8), b' ');

    let failed = || format!("can't write {}", path.display());
    let mut out = BufWriter::new(File::create(path).with_context(failed)?);
    out.write_all(&(header.len() as u64).to_le_bytes())
        .with_context(failed)?;
    out.write_all(&header).with_context(failed)?;
    for (name, shape, source) in &tensors {
        let bytes = match updates.get(key(name)) {
            Some(update) => {
                fold(update, shape, source).with_context(|| format!("can't merge {name}"))?
            }
            None => shards::bytes(source)?,
        };
        out.write_all(&bytes).with_context(failed)?;
    }
    out.flush().with_context(failed)
}

fn key(name: &str) -> &str {
    name.strip_prefix("model.").unwrap_or(name)
}

/// The stored weight plus its update, in the precision it was stored in.
fn fold(update: &Update, shape: &[usize], source: &Source) -> Result<Vec<u8>> {
    let grown = [update.b.dims()[0], update.a.dims()[1]];
    ensure!(
        shape == grown,
        "it is {shape:?} and the adapter is for {grown:?}"
    );
    let change = update
        .b
        .clone()
        .matmul(update.a.clone())
        .mul_scalar(update.scale)
        .into_data();
    let change = change
        .as_slice::<f32>()
        .map_err(|error| anyhow::anyhow!("{error:?}"))?;
    let mut values = shards::read(source)?;
    for (value, change) in values.iter_mut().zip(change) {
        *value += change;
    }
    let merged = TensorData::new(values, shape.to_vec()).convert_dtype(source.dtype());
    Ok(merged.as_bytes().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tiny;
    use burn::store::SafetensorsStore;
    use burn::store::{BurnToPyTorchAdapter, FloatCastAdapter, ModuleAdapter, ModuleSnapshot};
    use burn::tensor::DType;
    use std::path::PathBuf;

    const BASE: &str = "tests/fixtures/llama";
    const DOWN_PROJ: &str = "layers.0.mlp.down_proj.weight";

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lorasmith-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// An update for the weight `name` in `dir` that adds 0.5 * (1 * 0.25) to every value.
    fn update(dir: &Path, name: &str) -> HashMap<String, Update> {
        let tensors = shards::tensors(dir).unwrap();
        let (_, shape, _) = tensors.iter().find(|(n, ..)| key(n) == name).unwrap();
        let cpu = Device::flex();
        let update = Update {
            b: Tensor::ones([shape[0], 1], &cpu),
            a: Tensor::full([1, shape[1]], 0.25, &cpu),
            scale: 0.5,
        };
        HashMap::from([(name.to_string(), update)])
    }

    #[test]
    fn only_the_adapted_weight_changes() {
        let base = Path::new(BASE);
        let dir = temp("merge");
        // No model is loaded here: the published files and the adapter are all it takes.
        write(
            &update(base, DOWN_PROJ),
            base,
            &dir.join("model.safetensors"),
        )
        .unwrap();

        let published = shards::tensors(base).unwrap();
        let merged = shards::tensors(&dir).unwrap();
        assert_eq!(published.len(), merged.len());
        for ((name, shape, before), (merged_name, merged_shape, after)) in
            published.iter().zip(&merged)
        {
            assert_eq!((name, shape), (merged_name, merged_shape));
            if key(name) == DOWN_PROJ {
                let expected: Vec<f32> = shards::read(before)
                    .unwrap()
                    .iter()
                    .map(|v| v + 0.125)
                    .collect();
                assert_eq!(shards::read(after).unwrap(), expected);
            } else {
                assert_eq!(
                    shards::bytes(before).unwrap(),
                    shards::bytes(after).unwrap(),
                    "{name}"
                );
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn weights_stay_in_the_precision_they_were_published_in() {
        let published = temp("merge-bf16");
        let mut store = SafetensorsStore::from_file(published.join("model.safetensors"))
            .with_to_adapter(BurnToPyTorchAdapter.chain(FloatCastAdapter::to(DType::BF16)))
            .with_key_remapping(r"^(embed_tokens|layers|norm)", "model.$1");
        Llama::new(&tiny("llama"), &Device::flex())
            .save_into(&mut store)
            .unwrap();
        let dir = temp("merge-bf16-out");
        write(
            &update(&published, DOWN_PROJ),
            &published,
            &dir.join("model.safetensors"),
        )
        .unwrap();

        let before = shards::tensors(&published).unwrap();
        let after = shards::tensors(&dir).unwrap();
        for ((name, _, before), (_, _, after)) in before.iter().zip(&after) {
            assert_eq!(after.dtype(), DType::BF16, "{name}");
            if key(name) == DOWN_PROJ {
                let before = shards::read(before).unwrap();
                let after = shards::read(after).unwrap();
                // bf16 keeps about three digits, so 0.125 lands within a percent.
                for (before, after) in before.iter().zip(&after) {
                    assert!((after - before - 0.125).abs() < 2e-3, "{before} {after}");
                }
            }
        }
        std::fs::remove_dir_all(&published).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn refuses_a_folder_that_lacks_a_trained_weight() {
        let base = Path::new(BASE);
        let dir = temp("merge-missing");
        let mut updates = update(base, DOWN_PROJ);
        let stray = updates.remove(DOWN_PROJ).unwrap();
        updates.insert("layers.99.mlp.down_proj.weight".to_string(), stray);
        let error = write(&updates, base, &dir.join("model.safetensors")).unwrap_err();
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(error.to_string().contains("0 of the 1 weights"), "{error}");
    }
}

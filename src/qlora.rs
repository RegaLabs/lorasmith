//! QLoRA: the projections are held in 4 bits under full-precision adapters, so the same
//! GPU fits a much larger model.

use std::path::Path;

use anyhow::Result;
use burn::module::{Module, Quantizer};
use burn::tensor::quantization::{Calibration, QuantValue, ScaleDtype};
use burn::tensor::{DType, Device, FloatDType};

use crate::lora;
use crate::model::Llama;

/// The most neighbouring weights that share one scale. A single scale for a whole
/// projection lets its few large weights flatten all the others: Qwen2.5-0.5B's loss goes
/// from 5.1 to 19.9. At 32 it stays at 5.1 and a weight takes 5 bits, the layout of GGUF's
/// Q4_0.
pub const BLOCK: u8 = 32;

/// Loads the model in `dir` onto `device` with every projection packed into 4 bits.
/// Embeddings, norms and the output head stay in `dtype`.
pub fn load(dir: &Path, device: &Device, dtype: FloatDType) -> Result<Llama> {
    // Packed in system RAM first. A GPU that has once held the full weights keeps that
    // memory reserved, which gives back most of what packing saves.
    let (_, model) = crate::load::model(dir, &Device::flex(), dtype)?;
    // The model arrives from a device without autodiff, where nothing is trainable.
    // `train` lets the adapters attached next be.
    Ok(pack(model, device).to_device(device).train())
}

/// Packs the projections of `model` into 4 bits, in the layout `device` computes with.
pub fn pack(model: Llama, device: &Device) -> Llama {
    // Burn only cuts a row of weights into whole blocks. Published models are 32 wide many
    // times over, but smaller ones, like the test fixtures, need a smaller block.
    let fits = |block: &u8| {
        let block = usize::from(*block);
        let mut widths = lora::projections(&model)
            .into_iter()
            .map(|(_, layer)| layer.weight.dims()[1]);
        widths.all(|width| width.is_multiple_of(block))
    };
    let block = [BLOCK, 16, 8, 4, 2, 1]
        .into_iter()
        .find(fits)
        .expect("one divides every width");
    // GPUs store packed weights differently from the CPU, and a packed weight can move to
    // another device but can't change layout on the way.
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q4S)
        .per_block([block], ScaleDtype::F32);
    let mut quantizer = Quantizer::new(Calibration::MinMax, scheme);
    model.quantize_weights_group(&mut quantizer, lora::targets())
}

/// Whether the projections are packed, and so no longer hold the published weights.
pub fn is_packed(model: &Llama) -> bool {
    lora::projections(model)
        .first()
        .is_some_and(|(_, layer)| matches!(layer.weight.base().dtype(), DType::QFloat(_)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Example;
    use crate::train::{self, Options};
    use burn::tensor::{Int, Tensor, TensorData, Tolerance};

    const BASE: &str = "tests/fixtures/llama";

    fn logits(model: &Llama, device: &Device) -> Vec<f32> {
        let tokens =
            Tensor::<2, Int>::from_data(TensorData::new(vec![2i64, 7, 1, 8], [1, 4]), device);
        model
            .forward(tokens)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
    }

    fn distance(a: &[f32], b: &[f32]) -> f32 {
        let squares = a.iter().zip(b).map(|(a, b)| (a - b).powi(2));
        squares.sum::<f32>().sqrt()
    }

    #[test]
    fn only_the_projections_are_packed() {
        let device = Device::flex().autodiff();
        let (_, full) = crate::load::model(Path::new(BASE), &device, FloatDType::F32).unwrap();
        assert!(!is_packed(&full));
        let model = load(Path::new(BASE), &device, FloatDType::F32).unwrap();
        assert!(is_packed(&model));
        for (name, layer) in lora::projections(&model) {
            let dtype = layer.weight.base().dtype();
            assert!(matches!(dtype, DType::QFloat(_)), "{name} is {dtype:?}");
        }
        assert_eq!(
            model.embed_tokens.weight.val().into_data(),
            full.embed_tokens.weight.val().into_data()
        );
    }

    #[test]
    fn a_scale_per_block_stays_closer_than_one_per_projection() {
        let device = Device::flex();
        let (_, full) = crate::load::model(Path::new(BASE), &device, FloatDType::F32).unwrap();
        let expected = logits(&full, &device);
        let whole = device
            .settings()
            .quantization
            .scheme
            .with_value(QuantValue::Q4S);
        let mut quantizer = Quantizer::new(Calibration::MinMax, whole);
        let per_projection = full
            .clone()
            .quantize_weights_group(&mut quantizer, lora::targets());
        let per_block = pack(full, &device);

        let norm = distance(&expected, &vec![0.0; expected.len()]);
        let blocks = distance(&logits(&per_block, &device), &expected) / norm;
        let projections = distance(&logits(&per_projection, &device), &expected) / norm;
        assert!(blocks < 0.2, "per block is off by {blocks}");
        assert!(blocks < projections, "{blocks} vs {projections}");
    }

    #[test]
    fn training_changes_the_adapters_and_leaves_the_packed_base_alone() {
        let device = Device::flex().autodiff();
        let model = load(Path::new(BASE), &device, FloatDType::F32).unwrap();
        let before = logits(&model, &device);
        let model = train::attach(model, &Options::default());
        assert_eq!(before, logits(&model, &device));
        let base = |model: &Llama| model.layers[0].mlp.down_proj.weight.base().into_data();
        let packed = base(&model);

        let example = Example {
            tokens: vec![3, 1, 4, 1, 5, 9, 2, 6],
            train: vec![false, false, true, true, true, true, true, true],
        };
        let options = Options {
            learning_rate: 1e-2,
            epochs: 20,
            batch_size: 1,
            ..Options::default()
        };
        let mut losses = Vec::new();
        let model = train::train(model, &[example], &[], &options, &device, |p| {
            losses.push(p.loss)
        })
        .unwrap();

        assert!(losses[19] < losses[0] * 0.85, "{losses:?}");
        assert!(is_packed(&model));
        assert_eq!(base(&model), packed);
        assert!(!model.embed_tokens.weight.val().is_require_grad());
    }

    #[test]
    fn merging_folds_the_adapters_into_the_published_weights() {
        let device = Device::flex().autodiff();
        let model = load(Path::new(BASE), &device, FloatDType::F32).unwrap();
        let model = train::attach(model, &Options::default());
        // B starts at zero, and merging adapters that do nothing would prove nothing.
        let model = lora::set_factors(model, |path, shape| {
            let size = shape.iter().product::<usize>();
            path.ends_with(".lora.b")
                .then(|| TensorData::new(vec![0.01f32; size], shape.to_vec()))
        });
        let lora = model.layers[0].mlp.down_proj.weight.adapter().unwrap();
        let change = lora.a.val().matmul(lora.b.val()).mul_scalar(lora.scale);
        let dir = std::env::temp_dir().join(format!("lorasmith-qlora-{}", std::process::id()));
        crate::export::all(model, "tiny-llama", Path::new(BASE), &dir).unwrap();

        let down_proj = |dir: &Path| {
            let (_, model) = crate::load::model(dir, &device, FloatDType::F32).unwrap();
            model.layers[0].mlp.down_proj.weight.val()
        };
        let merged = down_proj(&dir.join("merged"));
        std::fs::remove_dir_all(&dir).unwrap();
        let expected = down_proj(Path::new(BASE)) + change;
        merged
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(1e-5));
    }

    #[test]
    fn the_config_counts_the_weights_that_get_packed() {
        let device = Device::flex();
        let (config, model) =
            crate::load::model(Path::new(BASE), &device, FloatDType::F32).unwrap();
        let packed: usize = lora::projections(&model)
            .iter()
            .map(|(_, layer)| layer.weight.dims().iter().product::<usize>())
            .sum();
        assert_eq!(config.projection_parameters(), packed);
    }
}

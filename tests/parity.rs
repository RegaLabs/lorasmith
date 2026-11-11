//! Our model must produce the same logits as Hugging Face transformers for the same weights.
//! Fixtures come from `tools/make_fixtures.py`.

use std::fs;
use std::path::Path;

use burn::tensor::{Device, FloatDType, Int, Tensor, TensorData};
use serde::Deserialize;

#[derive(Deserialize)]
struct Expected {
    tokens: Vec<Vec<i64>>,
    logits: Vec<f32>,
}

fn check(name: &str) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let expected: Expected =
        serde_json::from_str(&fs::read_to_string(dir.join("expected.json")).unwrap()).unwrap();
    let device = Device::flex();
    let (_, model) = loraforge::load::model(&dir, &device, FloatDType::F32).unwrap();

    let shape = [expected.tokens.len(), expected.tokens[0].len()];
    let tokens = expected.tokens.concat();
    let tokens = Tensor::<2, Int>::from_data(TensorData::new(tokens, shape), &device);
    let logits: Vec<f32> = model
        .forward(tokens)
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();

    assert_eq!(logits.len(), expected.logits.len(), "{name}");
    let worst = logits
        .iter()
        .zip(&expected.logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(worst < 1e-4, "{name}: logits differ by up to {worst}");
}

#[test]
fn llama_with_llama3_rope_scaling() {
    check("llama");
}

#[test]
fn mistral() {
    check("mistral");
}

#[test]
fn qwen2_with_tied_embeddings_and_attention_bias() {
    check("qwen2");
}

#[test]
fn qwen3_with_qk_norm_and_custom_head_dim() {
    check("qwen3");
}

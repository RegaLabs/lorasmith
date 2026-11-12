//! A half-precision base under full-precision adapters must train like a full-precision
//! model, and export a model built on the published weights.

use std::path::Path;

use burn::tensor::{DType, Device, FloatDType, Int, Tensor, TensorData, Tolerance};
use lorasmith::data::Example;
use lorasmith::model::Llama;
use lorasmith::train::{self, Options};
use lorasmith::{export, load, lora};

const BASE: &str = "tests/fixtures/llama";
const STEPS: usize = 8;

/// The fixture with adapters that start the same every time. `attach` draws them from a
/// random generator all tests share, so two models made by it alone can't be compared.
fn fresh(dtype: FloatDType, device: &Device) -> Llama {
    let (_, model) = load::model(Path::new(BASE), device, dtype).unwrap();
    let model = train::attach(model, &Options::default());
    lora::set_factors(model, |path, shape| {
        let size = shape.iter().product::<usize>();
        let a = (0..size).map(|i| ((i * 37 % 101) as f32 / 101.0 - 0.5) * 0.3);
        path.ends_with(".lora.a")
            .then(|| TensorData::new(a.collect(), shape.to_vec()))
    })
}

/// Takes `STEPS` steps from fresh adapters. Returns the model and the loss before each step.
fn trained(dtype: FloatDType, mut loss_scale: f64, device: &Device) -> (Llama, Vec<f32>) {
    let example = Example {
        tokens: vec![3, 1, 4, 1, 5, 9, 2, 6],
        train: vec![false, false, true, true, true, true, true, true],
    };
    let mut model = fresh(dtype, device);
    let mut optimizer = train::optimizer();
    let mut losses = Vec::new();
    for _ in 0..STEPS {
        let loss;
        (model, loss) = train::step(
            model,
            &mut optimizer,
            &[&example],
            1e-2,
            1.0,
            &mut loss_scale,
            device,
        );
        losses.push(loss);
    }
    (model, losses)
}

fn logits(model: &Llama, device: &Device) -> TensorData {
    let tokens = Tensor::<2, Int>::from_data(TensorData::new(vec![2i64, 7, 1, 8], [1, 4]), device);
    model.forward(tokens).into_data().convert::<f32>()
}

#[test]
fn half_precision_trains_like_full_precision() {
    let device = Device::flex().autodiff();
    let (_, full) = trained(FloatDType::F32, 1.0, &device);
    let (model, half) = trained(FloatDType::F16, 1024.0, &device);

    assert!(half[STEPS - 1] < half[0] * 0.9, "{half:?}");
    for (half, full) in half.iter().zip(&full) {
        assert!((half - full).abs() < full * 0.01, "{half} vs {full}");
    }
    let weight = &model.layers[0].self_attn.q_proj.weight;
    assert_eq!(weight.base().dtype(), DType::F16);
    assert_eq!(weight.adapter().unwrap().a.val().dtype(), DType::F32);
}

#[test]
fn the_loss_scale_cancels_out() {
    let device = Device::flex().autodiff();
    let (_, plain) = trained(FloatDType::F32, 1.0, &device);
    let (_, scaled) = trained(FloatDType::F32, 1024.0, &device);
    for (scaled, plain) in scaled.iter().zip(&plain) {
        assert!((scaled - plain).abs() < 1e-5, "{scaled} vs {plain}");
    }
}

#[test]
fn gradients_too_big_for_half_precision_skip_the_step() {
    let device = Device::flex().autodiff();
    let untouched = lora::factors(&fresh(FloatDType::F16, &device));
    let example = Example {
        tokens: vec![3, 1, 4, 1],
        train: vec![true; 4],
    };
    let mut loss_scale = 1e30;
    let (model, loss) = train::step(
        fresh(FloatDType::F16, &device),
        &mut train::optimizer(),
        &[&example],
        1e-2,
        1.0,
        &mut loss_scale,
        &device,
    );
    assert!(loss.is_finite());
    assert_eq!(loss_scale, 5e29);
    assert!(lora::factors(&model) == untouched);
}

#[test]
fn merging_a_half_precision_model_starts_from_the_published_weights() {
    let device = Device::flex().autodiff();
    let (model, _) = trained(FloatDType::F16, 1024.0, &device);
    let expected = logits(&model, &device);
    let dir = std::env::temp_dir().join(format!("lorasmith-half-{}", std::process::id()));
    export::merged(model, Path::new(BASE), &dir).unwrap();

    let (_, published) = load::model(Path::new(BASE), &device, FloatDType::F32).unwrap();
    let (_, merged) = load::model(&dir, &device, FloatDType::F32).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(
        merged.embed_tokens.weight.val().into_data(),
        published.embed_tokens.weight.val().into_data()
    );
    assert_ne!(
        merged.layers[0].self_attn.q_proj.weight.val().into_data(),
        published.layers[0]
            .self_attn
            .q_proj
            .weight
            .val()
            .into_data()
    );
    logits(&merged, &device).assert_approx_eq::<f32>(&expected, Tolerance::absolute(2e-2));
}

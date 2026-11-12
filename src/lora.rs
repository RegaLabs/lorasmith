//! LoRA adapters on every projection, initialized and clipped the way Hugging Face PEFT
//! and the Trainer do, so their well-known hyperparameters carry over unchanged.

use std::collections::HashMap;

use burn::module::{Lora, Module, ModuleMapper, Param, ParamGroup, ParamId};
use burn::nn::Linear;
use burn::optim::GradientsParams;
use burn::tensor::module::linear;
use burn::tensor::{Device, Distribution, FloatDType, Tensor, TensorData};

use crate::model::Llama;

/// Every projection inside the blocks gets an adapter, as the QLoRA paper recommends.
/// Embeddings and the output head stay as they are.
const TARGETS: &str = r"\.(q|k|v|o|gate|up|down)_proj\.weight$";

/// The weights that get adapters, which are also the ones QLoRA packs into 4 bits.
pub fn targets() -> ParamGroup {
    ParamGroup::from_regex(TARGETS).expect("TARGETS is a valid regex")
}

/// Freezes `model` and attaches fresh adapters. As in PEFT, A is uniform in ±1/√in and
/// B is zero, so the output doesn't change until training starts.
pub fn attach(model: Llama, rank: usize, alpha: f64) -> Llama {
    // The adapters stay in f32 over a half-precision base: one optimizer step changes them
    // by less than f16 can tell apart.
    let lora = Lora::new(rank, alpha)
        .set_param_group(targets())
        .set_dtype(FloatDType::F32);
    let model = model.apply_lora(lora);
    // Burn draws A from Normal(0, 1/rank), which for a 576-wide layer is 2.6 times PEFT's
    // spread and makes the same learning rate act that much stronger.
    set_factors(model, |path, shape| {
        if !path.ends_with(".lora.a") {
            return None;
        }
        let bound = 1.0 / (shape[0] as f64).sqrt();
        let a = Tensor::<2>::random(
            [shape[0], shape[1]],
            Distribution::Uniform(-bound, bound),
            &Device::flex(),
        );
        Some(a.into_data())
    })
}

/// Replaces adapter factors. `values` gets each parameter's path, such as
/// `layers.0.self_attn.q_proj.weight.lora.a`, and its shape, and returns new data or `None`.
pub fn set_factors(
    model: Llama,
    values: impl FnMut(&str, &[usize]) -> Option<TensorData>,
) -> Llama {
    let mut setter = Setter {
        path: Vec::new(),
        values,
    };
    model.map(&mut setter)
}

struct Setter<F> {
    path: Vec<String>,
    values: F,
}

impl<F: FnMut(&str, &[usize]) -> Option<TensorData>> ModuleMapper for Setter<F> {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let path = self.path.join(".");
        let shape = param.lazy_shape().to_vec();
        match (self.values)(&path, &shape) {
            Some(data) => param.map(|old| {
                let trainable = old.is_require_grad();
                Tensor::from_data(data, &old.device()).set_require_grad(trainable)
            }),
            None => param,
        }
    }
}

/// Applies `layer` to `x` the way PEFT does: x·W + s·((x·A)·B).
///
/// Burn's own forward builds the full W + s·A·B for every call, and autodiff then keeps a
/// weight-sized gradient for it: two extra copies of every projection, enough to push a
/// 0.5B model out of a 4 GB GPU. This way the extra memory is only rank-sized.
pub fn forward(layer: &Linear, x: Tensor<3>) -> Tensor<3> {
    let Some(lora) = layer.weight.adapter() else {
        return layer.forward(x);
    };
    let dtype = FloatDType::from(x.dtype());
    let bias = layer.bias.as_ref().map(|b| b.val());
    let base = linear(x.clone(), layer.weight.base(), bias);
    let x = x.cast(FloatDType::F32);
    let low_rank = linear(linear(x, lora.a.val(), None), lora.b.val(), None);
    base + low_rank.mul_scalar(lora.scale).cast(dtype)
}

/// The current adapter factors by path, in the form [`set_factors`] takes them back.
pub fn factors(model: &Llama) -> HashMap<String, TensorData> {
    let mut all = HashMap::new();
    for (name, layer) in projections(model) {
        let Some(lora) = layer.weight.adapter() else {
            continue;
        };
        let path = format!("{}.weight.lora", name.trim_start_matches("model."));
        all.insert(format!("{path}.a"), lora.a.val().into_data());
        all.insert(format!("{path}.b"), lora.b.val().into_data());
    }
    all
}

/// Every linear layer inside the blocks, under its Hugging Face name.
pub fn projections(model: &Llama) -> Vec<(String, &Linear)> {
    let mut all = Vec::new();
    for (i, block) in model.layers.iter().enumerate() {
        let (attn, mlp) = (&block.self_attn, &block.mlp);
        for (name, linear) in [
            ("self_attn.q_proj", &attn.q_proj),
            ("self_attn.k_proj", &attn.k_proj),
            ("self_attn.v_proj", &attn.v_proj),
            ("self_attn.o_proj", &attn.o_proj),
            ("mlp.gate_proj", &mlp.gate_proj),
            ("mlp.up_proj", &mlp.up_proj),
            ("mlp.down_proj", &mlp.down_proj),
        ] {
            all.push((format!("model.layers.{i}.{name}"), linear));
        }
    }
    all
}

/// Scales all adapter gradients together so their combined norm is at most `max_norm`,
/// like `torch.nn.utils.clip_grad_norm_`. Burn's own clipping limits each tensor separately,
/// which lets hundreds of adapter tensors through at many times the intended norm.
///
/// `loss_scale` is what the loss was multiplied by before backpropagation, which is divided
/// out again here. Returns `None` when the gradients aren't finite and must not be applied.
pub fn clip_grad_norm(
    model: &Llama,
    mut grads: GradientsParams,
    loss_scale: f64,
    max_norm: f64,
) -> Option<GradientsParams> {
    let ids: Vec<ParamId> = projections(model)
        .iter()
        .filter_map(|(_, linear)| linear.weight.adapter())
        .flat_map(|lora| [lora.a.id, lora.b.id])
        .collect();
    let squares: Vec<Tensor<1>> = ids
        .iter()
        .filter_map(|id| grads.get::<2>(*id))
        .map(|grad| grad.powi_scalar(2).sum())
        .collect();
    if squares.is_empty() {
        return Some(grads);
    }
    let norm = Tensor::cat(squares, 0).sum().sqrt().into_scalar::<f32>() as f64 / loss_scale;
    if !norm.is_finite() {
        return None;
    }
    let scale = (max_norm / (norm + 1e-6)).min(1.0) / loss_scale;
    for id in ids {
        if let Some(grad) = grads.remove::<2>(id) {
            grads.register(id, grad.mul_scalar(scale));
        }
    }
    Some(grads)
}

/// Moves the adapters of `trained` onto `base`, another copy of the same model without any.
pub fn transplant(trained: &Llama, base: Llama) -> Llama {
    let adapters = projections(trained);
    let Some(lora) = adapters
        .iter()
        .find_map(|(_, layer)| layer.weight.adapter())
    else {
        return base;
    };
    let rank = lora.a.dims()[1];
    let base = attach(base, rank, lora.scale * rank as f64);
    let factors = factors(trained);
    set_factors(base, |path, _| factors.get(path).cloned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tiny;

    #[test]
    fn a_starts_uniform_like_peft_and_b_at_zero() {
        let device = Device::flex().autodiff();
        let model = attach(Llama::new(&tiny("llama"), &device), 4, 8.0);
        let lora = model.layers[0].mlp.down_proj.weight.adapter().unwrap();
        let bound = 1.0 / 48f32.sqrt();
        let a = lora.a.val();
        assert_eq!(a.dims(), [48, 4]);
        assert!(a.clone().abs().max().into_scalar::<f32>() <= bound);
        assert!(a.clone().abs().max().into_scalar::<f32>() > bound / 2.0);
        assert!(a.is_require_grad());
        assert_eq!(lora.b.val().abs().max().into_scalar::<f32>(), 0.0);
    }

    #[test]
    fn factors_round_trip_through_set_factors() {
        let device = Device::flex().autodiff();
        let model = attach(Llama::new(&tiny("llama"), &device), 4, 8.0);
        let saved = factors(&model);
        assert_eq!(saved.len(), 2 * 7 * 2);
        let other = attach(Llama::new(&tiny("llama"), &device), 4, 8.0);
        let restored = set_factors(other, |path, _| saved.get(path).cloned());
        assert_eq!(factors(&restored), saved);
    }

    #[test]
    fn set_factors_sees_adapter_paths() {
        let device = Device::flex().autodiff();
        let mut seen = Vec::new();
        let model = attach(Llama::new(&tiny("llama"), &device), 4, 8.0);
        set_factors(model, |path, _| {
            seen.push(path.to_string());
            None
        });
        assert!(
            seen.contains(&"layers.1.self_attn.q_proj.weight.lora.b".to_string()),
            "{seen:?}"
        );
    }
}

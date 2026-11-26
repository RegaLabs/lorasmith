//! Backpropagation one decoder layer at a time. Attention goes through [`crate::attention`]
//! instead of keeping seq² probabilities, the loss through chunks of positions instead of
//! keeping every position's logits, and with `checkpoint` each layer runs again during the
//! backward pass instead of keeping its activations from the forward one.
//!
//! A layer's gradients come from backpropagating `sum(output * grad)` through that layer
//! alone, where `grad` is the loss's gradient against its output: the same numbers one
//! backward pass through the whole model hands on. Burn can't backpropagate through a graph
//! twice, so every piece builds a graph of its own.

use burn::optim::GradientsParams;
use burn::tensor::{Int, Tensor};

use crate::batch::{self, Batch};
use crate::model::{Block, Llama};
use crate::rope::Rope;
use crate::{attention, lora};

/// The loss of `batch` and the adapters' gradients for it, with the loss multiplied by
/// `loss_scale` before backpropagation. With `checkpoint`, only each layer's input is kept
/// between the forward and backward passes.
pub fn gradients(
    model: &Llama,
    batch: Batch,
    loss_scale: f64,
    checkpoint: bool,
) -> (f32, GradientsParams) {
    let rope = model.rope();
    // The embeddings are frozen, so nothing before the first layer needs a gradient.
    let mut x = model.embed_tokens.forward(batch.inputs).detach();
    let mut inputs = Vec::new();
    let mut kept = Vec::new();
    for block in &model.layers {
        let layer = Layer::run(block, &x, rope);
        let next = layer.output.clone().detach();
        if checkpoint {
            inputs.push(x);
        } else {
            kept.push(layer);
        }
        x = next;
    }
    let (loss, mut grad) = head(model, x, batch.positions, batch.targets, loss_scale);
    let mut params = GradientsParams::new();
    for (i, block) in model.layers.iter().enumerate().rev() {
        let layer = kept
            .pop()
            .unwrap_or_else(|| Layer::run(block, &inputs[i], rope));
        grad = layer.backward(block, grad, &mut params);
    }
    (loss, params)
}

/// One layer as two graphs: from its input to the attention's queries, keys and values, and
/// from the attention's output and the input (through the residual) to the layer's output.
struct Layer {
    input_to_qkv: Tensor<3>,
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    attended: Tensor<4>,
    logsumexp: Tensor<4>,
    input_to_output: Tensor<3>,
    output: Tensor<3>,
}

impl Layer {
    fn run(block: &Block, x: &Tensor<3>, rope: &Rope) -> Self {
        let input_to_qkv = x.clone().detach().require_grad();
        let (q, k, v) = block.attention_inputs(input_to_qkv.clone(), rope);
        let (out, logsumexp) = attention::forward(&plain(&q), &plain(&k), &plain(&v));
        let attended = out.autodiff().require_grad();
        let input_to_output = x.clone().detach().require_grad();
        let output = block.finish(input_to_output.clone(), attended.clone());
        Self {
            input_to_qkv,
            q,
            k,
            v,
            attended,
            logsumexp,
            input_to_output,
            output,
        }
    }

    /// Adds the gradients of `block`'s adapters to `params`, from `grad`, the loss's gradient
    /// against the layer's output. Returns the loss's gradient against the layer's input.
    fn backward(self, block: &Block, grad: Tensor<3>, params: &mut GradientsParams) -> Tensor<3> {
        let mut grads = (self.output * grad).sum().backward();
        let d_attended = self
            .attended
            .grad_remove(&mut grads)
            .expect("the output depends on the attention");
        let residual = self
            .input_to_output
            .grad_remove(&mut grads)
            .expect("the output depends on the input");
        absorb(params, GradientsParams::from_grads(grads, block), block);

        let qkv = (&plain(&self.q), &plain(&self.k), &plain(&self.v));
        let out = plain(&self.attended);
        let (dq, dk, dv) = attention::backward(qkv, &out, &self.logsumexp, &d_attended);
        let inputs = (self.q * dq).sum() + (self.k * dk).sum() + (self.v * dv).sum();
        let mut grads = inputs.backward();
        let through_attention = self
            .input_to_qkv
            .grad_remove(&mut grads)
            .expect("the queries depend on the input");
        absorb(params, GradientsParams::from_grads(grads, block), block);
        residual + through_attention
    }
}

/// The mean loss from the last layer's output `x`, and the loss's gradient against `x`
/// times `loss_scale`, a chunk of positions at a time.
fn head(
    model: &Llama,
    x: Tensor<3>,
    positions: Tensor<1, Int>,
    targets: Tensor<1, Int>,
    loss_scale: f64,
) -> (f32, Tensor<3>) {
    let [rows, len, width] = x.dims();
    let count = targets.dims()[0];
    let mut loss = 0.0;
    let mut grad: Option<Tensor<3>> = None;
    for range in batch::chunks(count, model.vocab_size()) {
        let leaf = x.clone().detach().require_grad();
        let hidden = model
            .norm
            .forward(leaf.clone())
            .reshape([rows * len, width]);
        let chunk = (
            positions.clone().slice(range.clone()),
            targets.clone().slice(range),
        );
        let part = batch::loss_sum(model, hidden, chunk.0, chunk.1).div_scalar(count as f64);
        loss += part.clone().into_scalar::<f32>();
        let mut grads = part.mul_scalar(loss_scale).backward();
        let part = leaf
            .grad_remove(&mut grads)
            .expect("the loss depends on the hidden states");
        grad = Some(match grad {
            Some(sum) => sum + part,
            None => part,
        });
    }
    (
        loss,
        grad.expect("every batch trains at least one position"),
    )
}

/// Moves the gradients of `block`'s adapters from `found` into `params`.
fn absorb(params: &mut GradientsParams, mut found: GradientsParams, block: &Block) {
    for (_, linear) in lora::block_projections(block) {
        let Some(adapter) = linear.weight.adapter() else {
            continue;
        };
        for id in [adapter.a.id, adapter.b.id] {
            if let Some(grad) = found.remove::<2>(id) {
                params.register(id, grad);
            }
        }
    }
}

fn plain(t: &Tensor<4>) -> Tensor<4> {
    t.clone().without_autodiff()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tiny;
    use crate::data::Example;
    use crate::train::{Options, attach};
    use burn::module::ParamId;
    use burn::tensor::{Device, Tolerance};

    fn ids(model: &Llama) -> Vec<ParamId> {
        lora::projections(model)
            .iter()
            .filter_map(|(_, linear)| linear.weight.adapter())
            .flat_map(|lora| [lora.a.id, lora.b.id])
            .collect()
    }

    /// A tiny model whose adapters' B isn't zero, so every gradient has something to show.
    fn model(device: &Device) -> Llama {
        let model = attach(Llama::new(&tiny("qwen2"), device), &Options::default());
        lora::set_factors(model, |path, shape| {
            let size = shape.iter().product::<usize>();
            let values = (0..size)
                .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.01)
                .collect();
            path.ends_with(".lora.b")
                .then(|| burn::tensor::TensorData::new(values, shape.to_vec()))
        })
    }

    fn batch(device: &Device) -> Batch {
        let examples = [
            Example {
                tokens: (0..23).map(|i| (i * 5 + 1) % 64).collect(),
                train: (0..23).map(|i| i > 4).collect(),
            },
            Example {
                tokens: (0..17).map(|i| (i * 3 + 2) % 64).collect(),
                train: vec![true; 17],
            },
        ];
        batch::tensors(&[&examples[0], &examples[1]], device)
    }

    #[test]
    fn layer_by_layer_matches_one_backward_pass() {
        let device = Device::flex().autodiff();
        let model = model(&device);
        let loss = batch::loss(&model, batch(&device));
        let expected_loss: f32 = loss.clone().into_scalar();
        let expected = GradientsParams::from_grads(loss.mul_scalar(8.0).backward(), &model);

        for checkpoint in [false, true] {
            let (loss, grads) = gradients(&model, batch(&device), 8.0, checkpoint);
            assert!(
                (loss - expected_loss).abs() < 1e-6,
                "{loss} vs {expected_loss}"
            );
            for id in ids(&model) {
                let ours = grads.get::<2>(id).expect("every adapter gets a gradient");
                let theirs = expected.get::<2>(id).unwrap();
                ours.into_data()
                    .assert_approx_eq::<f32>(&theirs.into_data(), Tolerance::absolute(1e-5));
            }
        }
    }

    #[test]
    fn checkpointing_changes_nothing() {
        let device = Device::flex().autodiff();
        let model = model(&device);
        let (loss, kept) = gradients(&model, batch(&device), 1.0, false);
        let (again, recomputed) = gradients(&model, batch(&device), 1.0, true);
        assert_eq!(loss, again);
        for id in ids(&model) {
            let a = kept.get::<2>(id).unwrap().into_data();
            assert_eq!(a, recomputed.get::<2>(id).unwrap().into_data());
        }
    }
}

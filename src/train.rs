//! The LoRA training loop.

use std::f64::consts::PI;

use burn::optim::{AdamWConfig, GradientsParams, ModuleOptimizer};
use burn::tensor::{Device, Int, Tensor, TensorData, activation};

use crate::data::Example;
use crate::lora;
use crate::model::Llama;

/// Training settings. The defaults are the common choices for LoRA with Hugging Face PEFT.
#[derive(Debug, Clone)]
pub struct Options {
    pub rank: usize,
    pub alpha: f64,
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
    /// Gradients are scaled down together when their combined norm exceeds this.
    pub max_grad_norm: f64,
    /// Picks the adapters' starting values and the order of examples, so runs repeat exactly.
    pub seed: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            learning_rate: 2e-4,
            epochs: 3,
            batch_size: 4,
            max_grad_norm: 1.0,
            seed: 42,
        }
    }
}

pub struct Progress {
    pub step: usize,
    pub total: usize,
    pub loss: f32,
}

/// About how many optimizer steps a small dataset needs before the adapter has learned it.
/// SmolLM2-135M learned a 36-example persona in 72 steps at lr 2e-4, and nothing in 27.
const TARGET_STEPS: usize = 75;

/// Epochs for `examples` examples: enough for about `TARGET_STEPS` steps, but never fewer
/// than 3 passes, and at most 10 so a handful of examples isn't memorized word for word.
pub fn epochs_for(examples: usize, batch_size: usize) -> usize {
    let steps_per_epoch = examples.div_ceil(batch_size).max(1);
    TARGET_STEPS.div_ceil(steps_per_epoch).clamp(3, 10)
}

/// Freezes `model` and attaches fresh adapters. The output doesn't change until training starts.
pub fn attach(model: Llama, options: &Options) -> Llama {
    Device::flex().seed(options.seed);
    lora::attach(model, options.rank, options.alpha)
}

/// The optimizer `train` uses: AdamW with PyTorch's defaults and no weight decay,
/// which is what the Hugging Face Trainer uses for LoRA.
pub fn optimizer() -> ModuleOptimizer {
    AdamWConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(0.0)
        .init()
}

/// Trains the adapters on `examples`, reshuffled every epoch.
/// `device` must be an autodiff device holding `model`.
pub fn train(
    mut model: Llama,
    examples: &[Example],
    options: &Options,
    device: &Device,
    mut report: impl FnMut(Progress),
) -> Llama {
    let mut optimizer = optimizer();
    let total = examples.len().div_ceil(options.batch_size) * options.epochs;
    let mut step = 0;
    for epoch in 0..options.epochs {
        let order = shuffled(examples.len(), options.seed.wrapping_add(epoch as u64));
        for indices in order.chunks(options.batch_size) {
            let batch: Vec<&Example> = indices.iter().map(|&i| &examples[i]).collect();
            let lr = learning_rate(options, step, total);
            let loss;
            (model, loss) = self::step(
                model,
                &mut optimizer,
                &batch,
                lr,
                options.max_grad_norm,
                device,
            );
            step += 1;
            report(Progress { step, total, loss });
        }
    }
    model
}

/// One optimizer step on one batch. Returns the updated model and the batch's loss before the step.
pub fn step(
    model: Llama,
    optimizer: &mut ModuleOptimizer,
    batch: &[&Example],
    learning_rate: f64,
    max_grad_norm: f64,
    device: &Device,
) -> (Llama, f32) {
    let loss = loss(&model, batch_tensors(batch, device));
    let value: f32 = loss.clone().into_scalar();
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    let grads = lora::clip_grad_norm(&model, grads, max_grad_norm);
    (optimizer.step(learning_rate, model, grads), value)
}

/// Indices `0..len` in a random order that depends only on `seed` (splitmix64 + Fisher-Yates).
fn shuffled(len: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..len).collect();
    let mut state = seed;
    for i in (1..len).rev() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        order.swap(i, (z % (i as u64 + 1)) as usize);
    }
    order
}

/// What one batch trains on: padded inputs, and for every position that counts toward
/// the loss, its index in the flattened `[batch * seq]` grid and the token it should predict.
struct Batch {
    inputs: Tensor<2, Int>,
    positions: Tensor<1, Int>,
    targets: Tensor<1, Int>,
}

/// Shifts tokens by one so each position predicts the next, padding to the longest example.
fn batch_tensors(batch: &[&Example], device: &Device) -> Batch {
    let len = batch.iter().map(|e| e.tokens.len() - 1).max().unwrap_or(0);
    let (mut inputs, mut positions, mut targets) = (Vec::new(), Vec::new(), Vec::new());
    for (row, example) in batch.iter().enumerate() {
        let n = example.tokens.len() - 1;
        inputs.extend(example.tokens[..n].iter().map(|&t| t as i64));
        inputs.extend(std::iter::repeat_n(0, len - n));
        for i in 0..n {
            if example.train[i + 1] {
                positions.push((row * len + i) as i64);
                targets.push(example.tokens[i + 1] as i64);
            }
        }
    }
    let count = positions.len();
    Batch {
        inputs: Tensor::from_data(TensorData::new(inputs, [batch.len(), len]), device),
        positions: Tensor::from_data(TensorData::new(positions, [count]), device),
        targets: Tensor::from_data(TensorData::new(targets, [count]), device),
    }
}

/// Mean cross-entropy over the trained positions. Logits are computed for those positions
/// only: with a 150k-token vocabulary, logits for every position of a long batch take gigabytes.
fn loss(model: &Llama, batch: Batch) -> Tensor<1> {
    let hidden = model.hidden(batch.inputs);
    let [rows, len, width] = hidden.dims();
    let picked = hidden
        .reshape([rows * len, width])
        .select(0, batch.positions);
    let log_probs = activation::log_softmax(model.logits(picked), 1);
    let chosen: Tensor<2> = log_probs.gather(1, batch.targets.unsqueeze_dim(1));
    chosen.mean().neg()
}

/// Short linear warmup, then cosine decay to zero.
fn learning_rate(options: &Options, step: usize, total: usize) -> f64 {
    let warmup = (total / 20).max(1);
    if step < warmup {
        return options.learning_rate * (step + 1) as f64 / warmup as f64;
    }
    let progress = (step - warmup) as f64 / (total - warmup).max(1) as f64;
    options.learning_rate * 0.5 * (1.0 + (PI * progress).cos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tiny;

    fn logits(model: &Llama, device: &Device) -> Vec<f32> {
        let tokens = Tensor::<2, Int>::from_data(TensorData::new(vec![1i64, 2, 3], [1, 3]), device);
        model
            .forward(tokens)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
    }

    #[test]
    fn fresh_adapters_leave_the_output_unchanged() {
        let device = Device::flex().autodiff();
        let model = Llama::new(&tiny("llama"), &device);
        let before = logits(&model, &device);
        let model = attach(model, &Options::default());
        assert_eq!(before, logits(&model, &device));
    }

    #[test]
    fn training_lowers_the_loss_and_keeps_the_base_frozen() {
        let device = Device::flex().autodiff();
        let model = attach(Llama::new(&tiny("qwen2"), &device), &Options::default());
        let embeddings = model.embed_tokens.weight.val().into_data();
        let example = Example {
            tokens: vec![5, 9, 13, 17, 21, 25, 29, 33],
            train: vec![false, false, true, true, true, true, true, true],
        };
        let options = Options {
            learning_rate: 1e-2,
            epochs: 40,
            batch_size: 2,
            ..Options::default()
        };
        let mut losses = Vec::new();
        let model = train(model, &[example.clone(), example], &options, &device, |p| {
            losses.push(p.loss)
        });

        assert_eq!(losses.len(), 40);
        assert!(
            losses[39] < losses[0] * 0.5,
            "loss went {} -> {}",
            losses[0],
            losses[39]
        );
        assert_eq!(embeddings, model.embed_tokens.weight.val().into_data());
    }

    #[test]
    fn small_datasets_get_more_epochs() {
        assert_eq!(epochs_for(36, 4), 9);
        assert_eq!(epochs_for(4, 4), 10);
        assert_eq!(epochs_for(1000, 4), 3);
    }

    #[test]
    fn shuffling_is_a_repeatable_permutation() {
        let order = shuffled(10, 42);
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
        assert_eq!(order, shuffled(10, 42));
        assert_ne!(order, shuffled(10, 43));
        assert_ne!(order, sorted);
    }

    #[test]
    fn padding_and_prompt_tokens_are_masked_out() {
        let device = Device::flex();
        let batch = [
            Example {
                tokens: vec![1, 2, 3, 4],
                train: vec![false, false, true, true],
            },
            Example {
                tokens: vec![7, 8],
                train: vec![true, true],
            },
        ];
        let batch = batch_tensors(&[&batch[0], &batch[1]], &device);
        let ints = |t: Tensor<1, Int>| t.into_data().convert::<i64>().try_to_vec::<i64>().unwrap();
        assert_eq!(batch.inputs.dims(), [2, 3]);
        assert_eq!(ints(batch.positions), [1, 2, 3]);
        assert_eq!(ints(batch.targets), [3, 4, 8]);
    }
}

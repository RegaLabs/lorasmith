//! The LoRA training loop.

use std::collections::HashMap;
use std::f64::consts::PI;

use anyhow::{Result, ensure};
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams, ModuleOptimizer};
use burn::tensor::{Device, TensorData};

use crate::batch;
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

pub struct Progress<'a> {
    pub step: usize,
    pub total: usize,
    pub epoch: usize,
    pub loss: f32,
    /// Whether this step finished an epoch.
    pub epoch_done: bool,
    /// On the last step of each epoch: the loss on held-out examples, if there are any.
    pub validation: Option<f32>,
    /// The model as trained so far, for example to save a checkpoint.
    pub model: &'a Llama,
}

/// Holds out about a tenth of `examples`, picked by `seed`, to measure how training
/// generalizes. Fewer than 20 examples are too few to spare any.
pub fn split(examples: Vec<Example>, seed: u64) -> (Vec<Example>, Vec<Example>) {
    let held = if examples.len() < 20 {
        0
    } else {
        (examples.len() / 10).min(200)
    };
    // A different order than the first epoch's, which uses `seed` itself.
    let order = shuffled(examples.len(), !seed);
    let mut slots: Vec<Option<Example>> = examples.into_iter().map(Some).collect();
    let validation = order[..held]
        .iter()
        .map(|&i| slots[i].take().expect("shuffled yields each index once"))
        .collect();
    (slots.into_iter().flatten().collect(), validation)
}

/// Mean loss per trained token over `examples`, without training on them.
pub fn evaluate(model: &Llama, examples: &[Example], batch_size: usize, device: &Device) -> f32 {
    // Without autodiff, no gradient graph is kept for a loss nobody differentiates.
    let model = model.valid();
    let device = device.clone().without_autodiff();
    let device = &device;
    let (mut sum, mut count) = (0.0f64, 0usize);
    for chunk in examples.chunks(batch_size) {
        let refs: Vec<&Example> = chunk.iter().collect();
        let batch = batch::tensors(&refs, device);
        let tokens = batch.targets.dims()[0];
        let loss: f32 = batch::loss(&model, batch).into_scalar();
        sum += loss as f64 * tokens as f64;
        count += tokens;
    }
    (sum / count.max(1) as f64) as f32
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

/// What the loss of a half-precision model is first multiplied by, see [`step`]. From 256
/// to 1024, gradients on an Intel GPU match full precision to a thousandth. At 16 the
/// smallest are still lost, and at 65536 Qwen2.5-0.5B's no longer fit.
const HALF_LOSS_SCALE: f64 = 1024.0;

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

/// Trains the adapters on `examples`, reshuffled every epoch. With `validation` examples,
/// it ends with the adapters from the epoch that did best on them, like the Hugging Face
/// Trainer's `load_best_model_at_end`. `device` must be an autodiff device holding `model`.
/// Fails when the loss has stopped being a number by the last step.
pub fn train(
    mut model: Llama,
    examples: &[Example],
    validation: &[Example],
    options: &Options,
    device: &Device,
    mut report: impl FnMut(Progress<'_>),
) -> Result<Llama> {
    let mut optimizer = optimizer();
    let half = model.is_half();
    let mut loss_scale = if half { HALF_LOSS_SCALE } else { 1.0 };
    let mut last_loss = 0.0;
    let per_epoch = examples.len().div_ceil(options.batch_size);
    let total = per_epoch * options.epochs;
    let mut step = 0;
    let mut best: Option<(f32, HashMap<String, TensorData>)> = None;
    for epoch in 0..options.epochs {
        let order = shuffled(examples.len(), options.seed.wrapping_add(epoch as u64));
        for (i, indices) in order.chunks(options.batch_size).enumerate() {
            let batch: Vec<&Example> = indices.iter().map(|&i| &examples[i]).collect();
            let lr = learning_rate(options, step, total);
            let loss;
            (model, loss) = self::step(
                model,
                &mut optimizer,
                &batch,
                lr,
                options.max_grad_norm,
                &mut loss_scale,
                device,
            );
            step += 1;
            last_loss = loss;
            let epoch_done = i + 1 == per_epoch;
            let checked = epoch_done && !validation.is_empty();
            let score = checked.then(|| evaluate(&model, validation, options.batch_size, device));
            if let Some(score) = score
                && best.as_ref().is_none_or(|(lowest, _)| score < *lowest)
            {
                best = Some((score, lora::factors(&model)));
            }
            report(Progress {
                step,
                total,
                epoch: epoch + 1,
                loss,
                epoch_done,
                validation: score,
                model: &model,
            });
        }
    }
    ensure!(
        last_loss.is_finite(),
        "the loss stopped being a number, so this model would only write nonsense. \
         Try a lower --learning-rate{}",
        if half {
            ", or --cpu to train in full precision"
        } else {
            ""
        }
    );
    if let Some((_, factors)) = best {
        model = lora::set_factors(model, |path, _| factors.get(path).cloned());
    }
    Ok(model)
}

/// One optimizer step on one batch. Returns the updated model and the batch's loss before the step.
///
/// The loss is multiplied by `loss_scale` on the way back through the model, because small
/// gradients vanish in half precision on a GPU: unscaled, SmolLM2's arrive at a third of
/// their size. When scaling makes one too big for half precision instead, the step is
/// skipped and `loss_scale` halved for the next one. Full precision needs none: pass 1.
pub fn step(
    model: Llama,
    optimizer: &mut ModuleOptimizer,
    batch: &[&Example],
    learning_rate: f64,
    max_grad_norm: f64,
    loss_scale: &mut f64,
    device: &Device,
) -> (Llama, f32) {
    let loss = batch::loss(&model, batch::tensors(batch, device));
    let value: f32 = loss.clone().into_scalar();
    let grads = GradientsParams::from_grads(loss.mul_scalar(*loss_scale).backward(), &model);
    match lora::clip_grad_norm(&model, grads, *loss_scale, max_grad_norm) {
        Some(grads) => (optimizer.step(learning_rate, model, grads), value),
        None => {
            *loss_scale = (*loss_scale / 2.0).max(1.0);
            (model, value)
        }
    }
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
    use burn::tensor::{Int, Tensor, TensorData};

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
        let model = train(
            model,
            &[example.clone(), example],
            &[],
            &options,
            &device,
            |p| losses.push(p.loss),
        )
        .unwrap();

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
    fn a_loss_that_stops_being_a_number_is_an_error() {
        let device = Device::flex().autodiff();
        let model = attach(Llama::new(&tiny("qwen2"), &device), &Options::default());
        let example = Example {
            tokens: vec![5, 9, 13, 17, 21, 25, 29, 33],
            train: vec![true; 8],
        };
        // A step this large pushes the adapters past what f32 can hold.
        let options = Options {
            learning_rate: 1e30,
            epochs: 4,
            batch_size: 1,
            ..Options::default()
        };
        let mut losses = Vec::new();
        let result = train(model, &[example], &[], &options, &device, |p| {
            losses.push(p.loss)
        });

        assert!(!losses[3].is_finite(), "losses were {losses:?}");
        let error = result.expect_err("training must not hand back the model");
        assert!(format!("{error:#}").contains("--learning-rate"));
    }

    #[test]
    fn a_tenth_is_held_out_from_twenty_examples_on() {
        let examples = |n: u32| -> Vec<Example> {
            (0..n)
                .map(|i| Example {
                    tokens: vec![i, i],
                    train: vec![true, true],
                })
                .collect()
        };
        let (train, validation) = split(examples(30), 42);
        assert_eq!((train.len(), validation.len()), (27, 3));
        assert!(
            validation
                .iter()
                .all(|v| train.iter().all(|t| t.tokens != v.tokens))
        );
        assert_eq!(split(examples(19), 42).1.len(), 0);
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
}

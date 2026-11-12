//! Which examples train, in what order, for how long and how fast.

use std::f64::consts::PI;

use crate::data::Example;
use crate::train::Options;

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

/// About how many optimizer steps a small dataset needs before the adapter has learned it.
/// SmolLM2-135M learned a 36-example persona in 72 steps at lr 2e-4, and nothing in 27.
const TARGET_STEPS: usize = 75;

/// Epochs for `examples` examples: enough for about `TARGET_STEPS` steps, but never fewer
/// than 3 passes, and at most 10 so a handful of examples isn't memorized word for word.
pub fn epochs_for(examples: usize, batch_size: usize) -> usize {
    let steps_per_epoch = examples.div_ceil(batch_size).max(1);
    TARGET_STEPS.div_ceil(steps_per_epoch).clamp(3, 10)
}

/// Indices `0..len` in a random order that depends only on `seed` (splitmix64 + Fisher-Yates).
pub fn shuffled(len: usize, seed: u64) -> Vec<usize> {
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

/// The learning rate for `step` of `total`: a short linear warmup, then cosine decay to zero.
pub fn learning_rate(options: &Options, step: usize, total: usize) -> f64 {
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

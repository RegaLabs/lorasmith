//! Turning examples into tensors, and the loss training minimizes.

use std::ops::Range;

use burn::tensor::{Device, FloatDType, Int, Tensor, TensorData, activation};

use crate::data::Example;
use crate::model::Llama;

/// What one batch trains on: padded inputs, and for every position that counts toward
/// the loss, its index in the flattened `[batch * seq]` grid and the token it should predict.
pub struct Batch {
    pub inputs: Tensor<2, Int>,
    pub positions: Tensor<1, Int>,
    pub targets: Tensor<1, Int>,
}

/// Shifts tokens by one so each position predicts the next, padding to the longest example.
pub fn tensors(batch: &[&Example], device: &Device) -> Batch {
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

/// Logits one chunk of the loss may hold, 512 MB in f32. With Qwen's 150k-token vocabulary,
/// 16k trained positions would otherwise take 10 GB per logits-sized tensor.
const CHUNK_LOGITS: usize = 128 << 20;

/// Mean cross-entropy over the trained positions. Logits are computed for those positions
/// only, a chunk at a time, see [`chunks`].
pub fn loss(model: &Llama, batch: Batch) -> Tensor<1> {
    let hidden = model.hidden(batch.inputs);
    let [rows, len, width] = hidden.dims();
    let hidden = hidden.reshape([rows * len, width]);
    let count = batch.targets.dims()[0];
    let sums = chunks(count, model.vocab_size()).map(|range| {
        let positions = batch.positions.clone().slice(range.clone());
        loss_sum(
            model,
            hidden.clone(),
            positions,
            batch.targets.clone().slice(range),
        )
    });
    Tensor::cat(sums.collect(), 0)
        .sum()
        .div_scalar(count as f64)
}

/// Ranges of trained positions that each hold at most [`CHUNK_LOGITS`] logits.
pub fn chunks(count: usize, vocab: usize) -> impl Iterator<Item = Range<usize>> {
    let size = (CHUNK_LOGITS / vocab.max(1)).max(1);
    (0..count)
        .step_by(size)
        .map(move |start| start..count.min(start + size))
}

/// Summed cross-entropy of predicting `targets` at `positions` of the flattened final hidden
/// states `[batch * seq, hidden]`.
pub fn loss_sum(
    model: &Llama,
    hidden: Tensor<2>,
    positions: Tensor<1, Int>,
    targets: Tensor<1, Int>,
) -> Tensor<1> {
    // Half-precision logits are too coarse for the loss: exp overflows and small
    // probabilities round to zero.
    let logits = model
        .logits(hidden.select(0, positions))
        .cast(FloatDType::F32);
    let log_probs = activation::log_softmax(logits, 1);
    let chosen: Tensor<2> = log_probs.gather(1, targets.unsqueeze_dim(1));
    chosen.sum().neg()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_every_position_once() {
        assert_eq!(
            chunks(5, CHUNK_LOGITS).collect::<Vec<_>>(),
            [0..1, 1..2, 2..3, 3..4, 4..5]
        );
        let mut whole = chunks(5, 1);
        assert_eq!((whole.next(), whole.next()), (Some(0..5), None));
        assert_eq!(
            chunks(5, CHUNK_LOGITS / 2).collect::<Vec<_>>(),
            [0..2, 2..4, 4..5]
        );
        assert_eq!(chunks(0, 10).count(), 0);
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
        let batch = tensors(&[&batch[0], &batch[1]], &device);
        let ints = |t: Tensor<1, Int>| t.into_data().convert::<i64>().try_to_vec::<i64>().unwrap();
        assert_eq!(batch.inputs.dims(), [2, 3]);
        assert_eq!(ints(batch.positions), [1, 2, 3]);
        assert_eq!(ints(batch.targets), [3, 4, 8]);
    }
}

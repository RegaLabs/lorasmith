//! Exact causal attention for training, in memory that grows with the sequence instead of its
//! square.
//!
//! Burn has no attention backward for the CPU, so its autodiff keeps every layer's
//! `[heads, seq, seq]` probabilities until the backward pass: 68 GB for Qwen2.5-0.5B at 4096
//! tokens. Here the queries go through in blocks. The forward pass keeps only each query's
//! log-sum-exp, and the backward pass rebuilds one block of probabilities at a time from it,
//! as FlashAttention does. Both take tensors without autodiff, see [`crate::backprop`].

use burn::tensor::{Bool, FloatDType, Tensor, s};

/// Score entries a block may hold, 128 MB in f32. Bigger blocks only save a little time; a few
/// blocks of this size are alive at once.
const BLOCK_SCORES: usize = 32 << 20;

/// Causal attention over queries, keys and values `[batch, heads, seq, head_dim]` with the keys
/// and values already repeated for every query head. Returns the output, in the inputs'
/// precision, and each query's log-sum-exp of its scores `[batch, heads, seq, 1]` in f32.
pub fn forward(q: &Tensor<4>, k: &Tensor<4>, v: &Tensor<4>) -> (Tensor<4>, Tensor<4>) {
    forward_in_blocks(q, k, v, BLOCK_SCORES)
}

fn forward_in_blocks(
    q: &Tensor<4>,
    k: &Tensor<4>,
    v: &Tensor<4>,
    budget: usize,
) -> (Tensor<4>, Tensor<4>) {
    let dtype = FloatDType::from(q.dtype());
    let (q, k, v) = (f32(q), f32(k), f32(v));
    let mut outs = Vec::new();
    let mut sums = Vec::new();
    for (start, rows) in blocks(&q, budget) {
        let scores = scores(&q, &k, start, rows);
        let max = scores.clone().max_dim(3);
        let exp = (scores - max.clone()).exp();
        let sum = exp.clone().sum_dim(3);
        let probs = exp / sum.clone();
        outs.push(probs.matmul(v.clone().narrow(2, 0, start + rows)));
        sums.push(max + sum.log());
    }
    (Tensor::cat(outs, 2).cast(dtype), Tensor::cat(sums, 2))
}

/// The gradients of [`forward`]'s output against `q`, `k` and `v`, given `out` and `logsumexp`
/// from it and `d_out`, the loss's gradient against `out`. Returned in the inputs' precision.
pub fn backward(
    qkv: (&Tensor<4>, &Tensor<4>, &Tensor<4>),
    out: &Tensor<4>,
    logsumexp: &Tensor<4>,
    d_out: &Tensor<4>,
) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
    backward_in_blocks(qkv, out, logsumexp, d_out, BLOCK_SCORES)
}

fn backward_in_blocks(
    (q, k, v): (&Tensor<4>, &Tensor<4>, &Tensor<4>),
    out: &Tensor<4>,
    logsumexp: &Tensor<4>,
    d_out: &Tensor<4>,
    budget: usize,
) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
    let dtype = FloatDType::from(q.dtype());
    let (q, k, v, d_out) = (f32(q), f32(k), f32(v), f32(d_out));
    let scale = scale(&q);
    // Each query's sum of probability times its gradient, which every score's gradient shares.
    let shared = (d_out.clone() * f32(out)).sum_dim(3);
    let mut dq = Vec::new();
    let mut dk = k.zeros_like();
    let mut dv = v.zeros_like();
    for (start, rows) in blocks(&q, budget) {
        let seen = start + rows;
        let (keys, values) = (k.clone().narrow(2, 0, seen), v.clone().narrow(2, 0, seen));
        let lse = logsumexp.clone().narrow(2, start, rows);
        let probs = (scores(&q, &k, start, rows) - lse).exp();
        let grad = d_out.clone().narrow(2, start, rows);
        let d_probs = grad.clone().matmul(values.transpose());
        let d_scores = probs.clone() * (d_probs - shared.clone().narrow(2, start, rows));
        let queries = q.clone().narrow(2, start, rows);
        dq.push(d_scores.clone().matmul(keys).mul_scalar(scale));
        let d_keys = d_scores.transpose().matmul(queries).mul_scalar(scale);
        let d_values = probs.transpose().matmul(grad);
        dk = add_front(dk, d_keys);
        dv = add_front(dv, d_values);
    }
    (
        Tensor::cat(dq, 2).cast(dtype),
        dk.cast(dtype),
        dv.cast(dtype),
    )
}

/// Scaled scores of queries `start..start + rows` against every key they may see, with the
/// keys after each query at minus infinity: `[batch, heads, rows, start + rows]`.
fn scores(q: &Tensor<4>, k: &Tensor<4>, start: usize, rows: usize) -> Tensor<4> {
    let seen = start + rows;
    let queries = q.clone().narrow(2, start, rows);
    let keys = k.clone().narrow(2, 0, seen);
    let scores = queries.matmul(keys.transpose()).mul_scalar(scale(q));
    let [batch, heads, ..] = scores.dims();
    // Query `start + r` sees keys up to `start + r`; `tril_mask` marks the keys after that.
    let future = Tensor::<2, Bool>::tril_mask([rows, seen], start as i64, &scores.device())
        .unsqueeze::<4>()
        .expand([batch, heads, rows, seen]);
    scores.mask_fill(future, f32::NEG_INFINITY)
}

/// Query blocks as `(start, rows)`, as many rows each as keep `budget` scores against the
/// whole sequence.
fn blocks(q: &Tensor<4>, budget: usize) -> impl Iterator<Item = (usize, usize)> {
    let [batch, heads, seq, _] = q.dims();
    let rows = (budget / (batch * heads * seq).max(1)).clamp(1, seq.max(1));
    (0..seq)
        .step_by(rows)
        .map(move |start| (start, rows.min(seq - start)))
}

fn scale(q: &Tensor<4>) -> f64 {
    1.0 / (q.dims()[3] as f64).sqrt()
}

fn f32(t: &Tensor<4>) -> Tensor<4> {
    t.clone().cast(FloatDType::F32)
}

/// `total` with `part` added to its first positions along the sequence.
fn add_front(total: Tensor<4>, part: Tensor<4>) -> Tensor<4> {
    let seen = part.dims()[2];
    let front = total.clone().narrow(2, 0, seen) + part;
    total.slice_assign(s![.., .., 0..seen, ..], front)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::module::attention;
    use burn::tensor::ops::AttentionModuleOptions;
    use burn::tensor::{Device, Distribution, Tolerance};

    fn random(shape: [usize; 4], device: &Device) -> Tensor<4> {
        Tensor::random(shape, Distribution::Normal(0.0, 1.0), device)
    }

    fn causal(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> Tensor<4> {
        let options = AttentionModuleOptions {
            is_causal: true,
            ..Default::default()
        };
        attention(q, k, v, None, None, None, options)
    }

    #[test]
    fn blocks_cover_the_sequence_in_order() {
        let q = Tensor::<4>::zeros([1, 2, 10, 4], &Device::flex());
        assert_eq!(blocks(&q, 1000).collect::<Vec<_>>(), [(0, 10)]);
        assert_eq!(blocks(&q, 80).collect::<Vec<_>>(), [(0, 4), (4, 4), (8, 2)]);
        assert_eq!(blocks(&q, 1).count(), 10);
    }

    #[test]
    fn forward_matches_burn_attention_in_any_blocks() {
        let device = Device::flex();
        let [q, k, v] = [0, 1, 2].map(|_| random([2, 3, 37, 8], &device));
        let expected = causal(q.clone(), k.clone(), v.clone()).into_data();
        // One block, blocks of 5 rows with a short last one, and one row each.
        for budget in [1 << 20, 2 * 3 * 37 * 5, 1] {
            let (out, _) = forward_in_blocks(&q, &k, &v, budget);
            out.into_data()
                .assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-5));
        }
    }

    #[test]
    fn gradients_match_autodiff() {
        let device = Device::flex().autodiff();
        let [q, k, v] = [0, 1, 2].map(|_| random([1, 2, 29, 8], &device).require_grad());
        let weights = random([1, 2, 29, 8], &device);
        let loss = (causal(q.clone(), k.clone(), v.clone()) * weights.clone()).sum();
        let grads = loss.backward();

        let plain = |t: &Tensor<4>| t.clone().inner();
        let qkv = (&plain(&q), &plain(&k), &plain(&v));
        for budget in [1 << 20, 2 * 29 * 4, 1] {
            let (out, lse) = forward_in_blocks(qkv.0, qkv.1, qkv.2, budget);
            let (dq, dk, dv) = backward_in_blocks(qkv, &out, &lse, &plain(&weights), budget);
            for (theirs, ours) in [(&q, dq), (&k, dk), (&v, dv)] {
                let expected = theirs.grad(&grads).unwrap().into_data();
                ours.into_data()
                    .assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-4));
            }
        }
    }
}

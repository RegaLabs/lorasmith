//! Rotary position embeddings, laid out the way Hugging Face models were trained with.

use std::f64::consts::PI;

use burn::module::Module;
use burn::tensor::{Device, FloatDType, Tensor, TensorData};

use crate::config::Config;

/// Cos and sin tables in Hugging Face's `rotate_half` layout. Burn's own RoPE interleaves pairs
/// instead, which would silently scramble pretrained q/k weights.
#[derive(Module, Debug)]
pub struct Rope {
    cos: Tensor<2>,
    sin: Tensor<2>,
}

impl Rope {
    /// Tables for every position the model supports.
    pub fn new(config: &Config, device: &Device) -> Self {
        let freqs = inverse_frequencies(config);
        let half = freqs.len();
        let positions = config.max_position_embeddings;
        let mut cos = Vec::with_capacity(positions * half * 2);
        let mut sin = Vec::with_capacity(positions * half * 2);
        for pos in 0..positions {
            let angles = freqs.iter().map(|f| pos as f64 * f);
            let row: Vec<f64> = angles.clone().chain(angles).collect();
            cos.extend(row.iter().map(|a| a.cos() as f32));
            sin.extend(row.iter().map(|a| a.sin() as f32));
        }
        let shape = [positions, half * 2];
        Self {
            cos: Tensor::from_data(TensorData::new(cos, shape), device),
            sin: Tensor::from_data(TensorData::new(sin, shape), device),
        }
    }

    /// Rotates queries or keys `[batch, heads, seq, head_dim]` by their position.
    pub fn apply(&self, x: Tensor<4>) -> Tensor<4> {
        let [_, _, seq, dim] = x.dims();
        // The tables are f32, and multiplying doesn't convert them to a half-precision `x`.
        let dtype = FloatDType::from(x.dtype());
        let table = |t: &Tensor<2>| t.clone().narrow(0, 0, seq).cast(dtype).unsqueeze::<4>();
        let (cos, sin) = (table(&self.cos), table(&self.sin));
        let first = x.clone().narrow(3, 0, dim / 2);
        let second = x.clone().narrow(3, dim / 2, dim / 2);
        let rotated = Tensor::cat(vec![second.neg(), first], 3);
        x * cos + rotated * sin
    }
}

/// The rotation speed of each pair of dimensions, with Llama 3's scaling if the model uses it.
pub fn inverse_frequencies(config: &Config) -> Vec<f64> {
    let dim = config.head_dim();
    let base: Vec<f64> = (0..dim / 2)
        .map(|i| 1.0 / config.rope_theta().powf(2.0 * i as f64 / dim as f64))
        .collect();
    let Some(s) = config.llama3_scaling() else {
        return base;
    };
    // Llama 3 leaves fast-spinning frequencies alone, slows the slow ones by `factor`,
    // and blends in between. Matches `_compute_llama3_parameters` in transformers.
    let old_context = s.original_max_position_embeddings as f64;
    let short = old_context / s.high_freq_factor;
    let long = old_context / s.low_freq_factor;
    base.into_iter()
        .map(|freq| {
            let wavelength = 2.0 * PI / freq;
            if wavelength < short {
                freq
            } else if wavelength > long {
                freq / s.factor
            } else {
                let smooth = (old_context / wavelength - s.low_freq_factor)
                    / (s.high_freq_factor - s.low_freq_factor);
                (1.0 - smooth) * freq / s.factor + smooth * freq
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tiny;

    #[test]
    fn llama3_scaling_keeps_high_and_slows_low_frequencies() {
        let mut config = tiny("llama");
        config.head_dim = Some(128);
        config.rope_theta = Some(500_000.0);
        let plain = inverse_frequencies(&config);
        config.rope_scaling = Some(crate::config::RopeParameters {
            rope_type: "llama3".to_string(),
            rope_theta: None,
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        });
        let scaled = inverse_frequencies(&config);
        assert_eq!(scaled[0], plain[0]);
        assert!((scaled[63] - plain[63] / 8.0).abs() < 1e-12);
    }
}

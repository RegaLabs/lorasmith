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
        let positions = config.context();
        // YaRN scales cos and sin up, as transformers' `attention_scaling` does.
        let scale = config.yarn().map_or(1.0, |yarn| yarn.attention_factor);
        let mut cos = Vec::with_capacity(positions * half * 2);
        let mut sin = Vec::with_capacity(positions * half * 2);
        for pos in 0..positions {
            let angles = freqs.iter().map(|f| pos as f64 * f);
            let row: Vec<f64> = angles.clone().chain(angles).collect();
            cos.extend(row.iter().map(|a| (a.cos() * scale) as f32));
            sin.extend(row.iter().map(|a| (a.sin() * scale) as f32));
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

/// The rotation speed of each pair of dimensions, with Llama 3's or YaRN's scaling if the
/// model uses it.
pub fn inverse_frequencies(config: &Config) -> Vec<f64> {
    let dim = config.head_dim();
    let base: Vec<f64> = (0..dim / 2)
        .map(|i| 1.0 / config.rope_theta().powf(2.0 * i as f64 / dim as f64))
        .collect();
    if let Some(yarn) = config.yarn() {
        return crate::yarn::frequencies(base, dim, config.rope_theta(), &yarn);
    }
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
            legacy_type: String::new(),
            rope_theta: None,
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
            ..Default::default()
        });
        let scaled = inverse_frequencies(&config);
        assert_eq!(scaled[0], plain[0]);
        assert!((scaled[63] - plain[63] / 8.0).abs() < 1e-12);
    }

    #[derive(serde::Deserialize)]
    struct YarnCase {
        config: serde_json::Value,
        inv_freq: Vec<f64>,
        attention_factor: f64,
        positions: Vec<usize>,
        cos: Vec<Vec<f32>>,
        sin: Vec<Vec<f32>>,
    }

    /// Fixtures from `tools/make_yarn_fixture.py`, computed by transformers.
    #[test]
    fn yarn_matches_transformers() {
        let text = std::fs::read_to_string("tests/fixtures/yarn/rope.json").unwrap();
        let cases: std::collections::BTreeMap<String, YarnCase> =
            serde_json::from_str(&text).unwrap();
        for (name, case) in cases {
            let config = Config::parse(&case.config.to_string()).unwrap();
            let ours = inverse_frequencies(&config);
            assert_eq!(ours.len(), case.inv_freq.len(), "{name}");
            for (i, (a, b)) in ours.iter().zip(&case.inv_freq).enumerate() {
                // transformers computes them in f32.
                assert!(
                    (a - b).abs() <= 1e-6 * b,
                    "{name}: frequency {i}: {a} vs {b}"
                );
            }
            let factor = config.yarn().unwrap().attention_factor;
            assert!((factor - case.attention_factor).abs() < 1e-12, "{name}");

            let rope = Rope::new(&config, &Device::flex());
            let table = |t: &Tensor<2>, row: usize| {
                let values = t.clone().narrow(0, row, 1).into_data();
                values.try_to_vec::<f32>().unwrap()
            };
            for (i, &pos) in case.positions.iter().enumerate() {
                let pairs = table(&rope.cos, pos)
                    .into_iter()
                    .zip(&case.cos[i])
                    .chain(table(&rope.sin, pos).into_iter().zip(&case.sin[i]));
                for (a, b) in pairs {
                    // transformers turns the angle in f32, which drifts by up to a few
                    // ten-thousandths by position 4095.
                    assert!((a - b).abs() < 1e-3, "{name}: position {pos}: {a} vs {b}");
                }
            }
        }
    }
}

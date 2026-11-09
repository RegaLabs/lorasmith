//! A Llama-shaped decoder whose field names match Hugging Face, so weights load without renaming.

use std::f64::consts::PI;

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Device, Int, Tensor, TensorData, activation};

use crate::config::Config;

#[derive(Module, Debug)]
pub struct Llama {
    pub embed_tokens: Embedding,
    pub layers: Vec<Block>,
    pub norm: RmsNorm,
    /// `None` when the output projection reuses `embed_tokens`.
    pub lm_head: Option<Linear>,
    rope: Rope,
}

#[derive(Module, Debug)]
pub struct Block {
    pub input_layernorm: RmsNorm,
    pub self_attn: Attention,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: Mlp,
}

#[derive(Module, Debug)]
pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: Option<RmsNorm>,
    pub k_norm: Option<RmsNorm>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

#[derive(Module, Debug)]
pub struct Mlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

/// Cos and sin tables in Hugging Face's `rotate_half` layout. Burn's own RoPE interleaves pairs
/// instead, which would silently scramble pretrained q/k weights.
#[derive(Module, Debug)]
struct Rope {
    cos: Tensor<2>,
    sin: Tensor<2>,
}

impl Llama {
    /// Builds the model with random weights. Load real ones with [`crate::load::weights`].
    pub fn new(config: &Config, device: &Device) -> Self {
        let linear =
            |d_in, d_out, bias| LinearConfig::new(d_in, d_out).with_bias(bias).init(device);
        let norm = |dim| {
            RmsNormConfig::new(dim)
                .with_epsilon(config.rms_norm_eps)
                .init(device)
        };
        let (hidden, head_dim) = (config.hidden_size, config.head_dim());
        let (heads, kv_heads) = (config.num_attention_heads, config.kv_heads());
        let bias = config.attention_bias();

        let block = || Block {
            input_layernorm: norm(hidden),
            self_attn: Attention {
                q_proj: linear(hidden, heads * head_dim, bias),
                k_proj: linear(hidden, kv_heads * head_dim, bias),
                v_proj: linear(hidden, kv_heads * head_dim, bias),
                o_proj: linear(heads * head_dim, hidden, false),
                q_norm: config.qk_norm().then(|| norm(head_dim)),
                k_norm: config.qk_norm().then(|| norm(head_dim)),
                heads,
                kv_heads,
                head_dim,
            },
            post_attention_layernorm: norm(hidden),
            mlp: Mlp {
                gate_proj: linear(hidden, config.intermediate_size, false),
                up_proj: linear(hidden, config.intermediate_size, false),
                down_proj: linear(config.intermediate_size, hidden, false),
            },
        };

        Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, hidden).init(device),
            layers: (0..config.num_hidden_layers).map(|_| block()).collect(),
            norm: norm(hidden),
            lm_head: (!config.tie_word_embeddings)
                .then(|| linear(hidden, config.vocab_size, false)),
            rope: Rope::new(config, device),
        }
    }

    /// Token ids `[batch, seq]` in, next-token logits `[batch, seq, vocab]` out.
    pub fn forward(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        let mut x = self.embed_tokens.forward(tokens);
        for block in &self.layers {
            x = block.forward(x, &self.rope);
        }
        let x = self.norm.forward(x);
        match &self.lm_head {
            Some(head) => head.forward(x),
            None => x.matmul(self.embed_tokens.weight.val().transpose().unsqueeze()),
        }
    }
}

impl Block {
    fn forward(&self, x: Tensor<3>, rope: &Rope) -> Tensor<3> {
        let x = x.clone()
            + self
                .self_attn
                .forward(self.input_layernorm.forward(x), rope);
        x.clone() + self.mlp.forward(self.post_attention_layernorm.forward(x))
    }
}

impl Attention {
    fn forward(&self, x: Tensor<3>, rope: &Rope) -> Tensor<3> {
        let [batch, seq, _] = x.dims();
        let split = |t: Tensor<3>, heads| {
            t.reshape([batch, seq, heads, self.head_dim])
                .swap_dims(1, 2)
        };
        let mut q = split(self.q_proj.forward(x.clone()), self.heads);
        let mut k = split(self.k_proj.forward(x.clone()), self.kv_heads);
        let v = split(self.v_proj.forward(x), self.kv_heads);

        if let (Some(q_norm), Some(k_norm)) = (&self.q_norm, &self.k_norm) {
            q = q_norm.forward(q);
            k = k_norm.forward(k);
        }
        let (q, k) = (rope.apply(q), rope.apply(k));

        let repeat = self.heads / self.kv_heads;
        let options = AttentionModuleOptions {
            is_causal: true,
            ..Default::default()
        };
        let out = attention(
            q,
            self.repeat_kv(k, repeat),
            self.repeat_kv(v, repeat),
            None,
            None,
            options,
        );
        self.o_proj.forward(
            out.swap_dims(1, 2)
                .reshape([batch, seq, self.heads * self.head_dim]),
        )
    }

    /// Grouped-query attention shares each key/value head between several query heads.
    fn repeat_kv(&self, t: Tensor<4>, repeat: usize) -> Tensor<4> {
        if repeat == 1 {
            return t;
        }
        let [batch, kv_heads, seq, head_dim] = t.dims();
        t.unsqueeze_dim::<5>(2)
            .expand([batch, kv_heads, repeat, seq, head_dim])
            .reshape([batch, kv_heads * repeat, seq, head_dim])
    }
}

impl Mlp {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let gate = activation::silu(self.gate_proj.forward(x.clone()));
        self.down_proj.forward(gate * self.up_proj.forward(x))
    }
}

impl Rope {
    fn new(config: &Config, device: &Device) -> Self {
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

    fn apply(&self, x: Tensor<4>) -> Tensor<4> {
        let [_, _, seq, dim] = x.dims();
        let cos = self.cos.clone().narrow(0, 0, seq).unsqueeze::<4>();
        let sin = self.sin.clone().narrow(0, 0, seq).unsqueeze::<4>();
        let first = x.clone().narrow(3, 0, dim / 2);
        let second = x.clone().narrow(3, dim / 2, dim / 2);
        let rotated = Tensor::cat(vec![second.neg(), first], 3);
        x * cos + rotated * sin
    }
}

pub(crate) fn inverse_frequencies(config: &Config) -> Vec<f64> {
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

    fn logits(model: &Llama, tokens: [i64; 6], device: &Device) -> Vec<f32> {
        let tokens = Tensor::<2, Int>::from_data(TensorData::new(tokens.to_vec(), [1, 6]), device);
        model
            .forward(tokens)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
    }

    #[test]
    fn every_supported_architecture_runs() {
        let device = Device::flex();
        for model_type in ["llama", "mistral", "qwen2", "qwen3"] {
            let mut config = tiny(model_type);
            config.tie_word_embeddings = model_type == "qwen2";
            let model = Llama::new(&config, &device);
            assert_eq!(model.num_params(), config.parameters(), "{model_type}");
            let out = logits(&model, [1, 2, 3, 4, 5, 6], &device);
            assert_eq!(out.len(), 6 * config.vocab_size, "{model_type}");
            assert!(out.iter().all(|v| v.is_finite()), "{model_type}");
        }
    }

    #[test]
    fn later_tokens_do_not_change_earlier_logits() {
        let device = Device::flex();
        let model = Llama::new(&tiny("llama"), &device);
        let a = logits(&model, [1, 2, 3, 4, 5, 6], &device);
        let b = logits(&model, [1, 2, 3, 9, 9, 9], &device);
        let first_three = 3 * 64;
        for (x, y) in a[..first_three].iter().zip(&b[..first_three]) {
            assert!((x - y).abs() < 1e-5);
        }
        assert_ne!(a[first_three..], b[first_three..]);
    }

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

//! A Llama-shaped decoder whose field names match Hugging Face, so weights load without renaming.

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::module::{attention, linear};
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{DType, Device, Int, Tensor, activation};

use crate::config::Config;
use crate::lora;
use crate::rope::Rope;

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
        self.logits(self.hidden(tokens))
    }

    /// Final hidden states `[batch, seq, hidden]`, before the output head.
    pub fn hidden(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        let mut x = self.embed_tokens.forward(tokens);
        for block in &self.layers {
            x = block.forward(x, &self.rope);
        }
        self.norm.forward(x)
    }

    /// Vocabulary logits for hidden states of any leading shape.
    pub fn logits<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        match &self.lm_head {
            Some(head) => head.forward(x),
            None => linear(x, self.embed_tokens.weight.val().transpose(), None),
        }
    }

    /// Whether the weights are held in f16, which training and merging must allow for.
    pub fn is_half(&self) -> bool {
        self.embed_tokens.weight.val().dtype() == DType::F16
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
        let mut q = split(lora::forward(&self.q_proj, x.clone()), self.heads);
        let mut k = split(lora::forward(&self.k_proj, x.clone()), self.kv_heads);
        let v = split(lora::forward(&self.v_proj, x), self.kv_heads);

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
        lora::forward(
            &self.o_proj,
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
        let gate = activation::silu(lora::forward(&self.gate_proj, x.clone()));
        lora::forward(&self.down_proj, gate * lora::forward(&self.up_proj, x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tiny;
    use burn::tensor::TensorData;

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
}

//! The parts of a Hugging Face `config.json` that decide the model's shape.

use anyhow::{Result, bail};
use serde::Deserialize;

const SUPPORTED: &[&str] = &["llama", "mistral", "qwen2", "qwen3"];

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub rms_norm_eps: f64,
    pub rope_theta: Option<f64>,
    pub rope_scaling: Option<RopeParameters>,
    /// transformers 5 moved `rope_theta` and `rope_scaling` in here. Hub models use both layouts.
    pub rope_parameters: Option<RopeParameters>,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub attention_bias: Option<bool>,
}

/// Llama 3 stretches its RoPE frequencies. Other scaling schemes are refused in [`Config::check`].
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(alias = "type")]
    pub rope_type: String,
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub factor: f64,
    #[serde(default)]
    pub low_freq_factor: f64,
    #[serde(default)]
    pub high_freq_factor: f64,
    #[serde(default)]
    pub original_max_position_embeddings: usize,
}

impl Config {
    pub fn parse(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)?;
        config.check()?;
        Ok(config)
    }

    /// Refuses anything we would load into the wrong shape instead of training garbage.
    pub fn check(&self) -> Result<()> {
        if !SUPPORTED.contains(&self.model_type.as_str()) {
            bail!(
                "model type `{}` is not supported yet. Supported: {}. \
                 Try a Llama, Mistral or Qwen model.",
                self.model_type,
                SUPPORTED.join(", ")
            );
        }
        if let Some(rope) = self.rope()
            && !matches!(rope.rope_type.as_str(), "llama3" | "default")
        {
            bail!(
                "RoPE scaling `{}` is not supported yet. Try a model without long-context scaling.",
                rope.rope_type
            );
        }
        if !self.num_attention_heads.is_multiple_of(self.kv_heads()) {
            bail!("config.json is inconsistent: attention heads must be a multiple of KV heads");
        }
        Ok(())
    }

    fn rope(&self) -> Option<&RopeParameters> {
        self.rope_parameters.as_ref().or(self.rope_scaling.as_ref())
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|rope| rope.rope_theta)
            .or(self.rope_theta)
            .unwrap_or(10_000.0)
    }

    pub fn llama3_scaling(&self) -> Option<&RopeParameters> {
        self.rope().filter(|rope| rope.rope_type == "llama3")
    }

    pub fn kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Qwen2 has q/k/v biases without saying so in its config.
    pub fn attention_bias(&self) -> bool {
        self.attention_bias.unwrap_or(self.model_type == "qwen2")
    }

    /// Qwen3 normalizes queries and keys per head before RoPE.
    pub fn qk_norm(&self) -> bool {
        self.model_type == "qwen3"
    }

    /// Number of weights in the model, known before downloading any of them.
    pub fn parameters(&self) -> usize {
        let (hidden, head_dim) = (self.hidden_size, self.head_dim());
        let (q, kv) = (
            self.num_attention_heads * head_dim,
            self.kv_heads() * head_dim,
        );
        let bias = if self.attention_bias() { q + 2 * kv } else { 0 };
        let norms = if self.qk_norm() { 2 * head_dim } else { 0 };
        let attention = 2 * hidden * q + 2 * hidden * kv + bias + norms;
        let block = attention + 3 * hidden * self.intermediate_size + 2 * hidden;
        let embeddings = self.vocab_size * hidden;
        let head = if self.tie_word_embeddings {
            0
        } else {
            embeddings
        };
        embeddings + self.num_hidden_layers * block + hidden + head
    }
}

#[cfg(test)]
pub fn tiny(model_type: &str) -> Config {
    Config {
        model_type: model_type.to_string(),
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 48,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: None,
        rms_norm_eps: 1e-6,
        rope_theta: None,
        rope_scaling: None,
        rope_parameters: None,
        max_position_embeddings: 128,
        tie_word_embeddings: false,
        attention_bias: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_real_qwen_config() {
        let json = r#"{
            "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
            "vocab_size": 151936, "hidden_size": 896, "intermediate_size": 4864,
            "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06, "rope_theta": 1000000.0, "rope_scaling": null,
            "max_position_embeddings": 32768, "tie_word_embeddings": true
        }"#;
        let config = Config::parse(json).unwrap();
        assert_eq!(config.head_dim(), 64);
        assert!(config.attention_bias());
        assert!(config.tie_word_embeddings);
        assert_eq!(config.rope_theta(), 1_000_000.0);
    }

    #[test]
    fn reads_transformers_5_rope_parameters() {
        let json = r#"{
            "model_type": "llama", "vocab_size": 64, "hidden_size": 32,
            "intermediate_size": 48, "num_hidden_layers": 2, "num_attention_heads": 4,
            "rms_norm_eps": 1e-06, "max_position_embeddings": 128,
            "rope_parameters": {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0,
                "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                "original_max_position_embeddings": 64}
        }"#;
        let config = Config::parse(json).unwrap();
        assert_eq!(config.rope_theta(), 500_000.0);
        assert_eq!(config.llama3_scaling().unwrap().factor, 8.0);
    }

    #[test]
    fn refuses_unknown_architectures_with_a_hint() {
        let mut config = tiny("llama");
        config.model_type = "gemma2".to_string();
        let message = config.check().unwrap_err().to_string();
        assert!(message.contains("gemma2") && message.contains("Try"));
    }
}

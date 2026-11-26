//! YaRN, which stretches a model's context by slowing its slowest RoPE frequencies, as
//! Qwen2.5 and Qwen3 do past 32k tokens.

use std::f64::consts::PI;

use anyhow::{Result, bail};

use crate::config::{Config, RopeParameters};

/// YaRN's settings with the defaults transformers fills in.
#[derive(Debug, Clone, PartialEq)]
pub struct Yarn {
    pub factor: f64,
    /// The context the model was pretrained on, before stretching.
    pub original_context: usize,
    /// What cos and sin are multiplied by, which keeps attention as sharp as before the stretch.
    pub attention_factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
}

impl Config {
    /// The model's YaRN settings, if it uses YaRN. Defaults as in `_compute_yarn_parameters`.
    pub fn yarn(&self) -> Option<Yarn> {
        let rope = self.rope().filter(|rope| rope.kind() == "yarn")?;
        let original_context = match rope.original_max_position_embeddings {
            0 => self.max_position_embeddings,
            n => n,
        };
        let factor = if rope.factor > 0.0 {
            rope.factor
        } else {
            self.max_position_embeddings as f64 / original_context as f64
        };
        let attention_factor = match rope.attention_factor {
            Some(given) => given,
            None if factor <= 1.0 => 1.0,
            None => 0.1 * factor.ln() + 1.0,
        };
        Some(Yarn {
            factor,
            original_context,
            attention_factor,
            beta_fast: rope.beta_fast.filter(|b| *b != 0.0).unwrap_or(32.0),
            beta_slow: rope.beta_slow.filter(|b| *b != 0.0).unwrap_or(1.0),
        })
    }

    /// The longest sequence the model handles: what it was trained on, or with YaRN that
    /// stretched `factor` times, as vLLM counts it.
    pub fn context(&self) -> usize {
        match self.yarn() {
            Some(yarn) => self
                .max_position_embeddings
                .max((yarn.factor * yarn.original_context as f64) as usize),
            None => self.max_position_embeddings,
        }
    }

    /// Adds YaRN that stretches the context `factor` times, for a model pretrained without
    /// scaling. Refuses a model that already scales its RoPE, which would be stretched twice.
    pub fn stretch(&mut self, factor: f64) -> Result<()> {
        if let Some(rope) = self.rope().filter(|rope| rope.kind() != "default") {
            bail!(
                "this model already stretches its context with `{}` RoPE scaling. \
                 Drop --rope-scaling to train with that.",
                rope.kind()
            );
        }
        let yarn = RopeParameters {
            rope_type: "yarn".to_string(),
            rope_theta: Some(self.rope_theta()),
            factor,
            original_max_position_embeddings: self.max_position_embeddings,
            ..RopeParameters::default()
        };
        // transformers 5 keeps the RoPE settings in `rope_parameters`, earlier ones in `rope_scaling`.
        if self.rope_parameters.is_some() {
            self.rope_parameters = Some(yarn);
        } else {
            self.rope_scaling = Some(yarn);
        }
        Ok(())
    }

    /// `json`, the text of a `config.json`, with this config's YaRN settings written into it the
    /// way Qwen documents them, so Hugging Face, vLLM and the GGUF converter all see them.
    /// Without YaRN it comes back unchanged.
    pub fn write_yarn(&self, json: &str) -> Result<String> {
        let Some(yarn) = self.yarn() else {
            return Ok(json.to_string());
        };
        let mut value: serde_json::Value = serde_json::from_str(json)?;
        let key = if value["rope_parameters"].is_object() {
            "rope_parameters"
        } else {
            "rope_scaling"
        };
        if !value[key].is_object() {
            value[key] = serde_json::json!({});
        }
        let rope = &mut value[key];
        rope["rope_type"] = "yarn".into();
        rope["factor"] = yarn.factor.into();
        rope["original_max_position_embeddings"] = yarn.original_context.into();
        if let Some(object) = rope.as_object_mut() {
            object.remove("type");
        }
        Ok(serde_json::to_string_pretty(&value)?)
    }
}

/// YaRN keeps pairs that turn many times within the original context at their speed, slows
/// those that turn less than once by `factor`, and ramps linearly in between. Matches
/// `_compute_yarn_parameters` in transformers.
pub fn frequencies(base: Vec<f64>, dim: usize, theta: f64, yarn: &Yarn) -> Vec<f64> {
    let pair_turning = |turns: f64| {
        dim as f64 * (yarn.original_context as f64 / (turns * 2.0 * PI)).ln() / (2.0 * theta.ln())
    };
    let low = pair_turning(yarn.beta_fast).floor().max(0.0);
    let high = pair_turning(yarn.beta_slow).ceil().min(dim as f64 - 1.0);
    let width = if high == low { 0.001 } else { high - low };
    base.into_iter()
        .enumerate()
        .map(|(i, freq)| {
            let ramp = ((i as f64 - low) / width).clamp(0.0, 1.0);
            freq / yarn.factor * ramp + freq * (1.0 - ramp)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tiny;

    #[test]
    fn reads_qwen_yarn_with_transformers_defaults() {
        let json = r#"{
            "model_type": "qwen2", "vocab_size": 64, "hidden_size": 32,
            "intermediate_size": 48, "num_hidden_layers": 2, "num_attention_heads": 4,
            "rms_norm_eps": 1e-06, "max_position_embeddings": 32768,
            "rope_scaling": {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
        }"#;
        let config = Config::parse(json).unwrap();
        let yarn = config.yarn().unwrap();
        assert_eq!((yarn.beta_fast, yarn.beta_slow), (32.0, 1.0));
        assert!((yarn.attention_factor - (1.0 + 0.1 * 4f64.ln())).abs() < 1e-12);
        assert_eq!(config.context(), 131072);
        assert!(config.check_max_len(131072).is_ok());
    }

    #[test]
    fn stretching_adds_yarn_over_the_old_context() {
        let mut config = tiny("qwen2");
        let message = config.check_max_len(512).unwrap_err().to_string();
        assert!(message.contains("--rope-scaling yarn"), "{message}");
        config.stretch(4.0).unwrap();
        assert_eq!(config.context(), 512);
        assert_eq!(config.yarn().unwrap().original_context, 128);
        assert!(config.check_max_len(512).is_ok());
        let message = config.stretch(2.0).unwrap_err().to_string();
        assert!(
            message.contains("already") && message.contains("yarn"),
            "{message}"
        );
    }

    #[test]
    fn writes_yarn_into_whichever_rope_key_the_config_uses() {
        let mut config = tiny("qwen2");
        config.stretch(4.0).unwrap();
        let read = |text: String| serde_json::from_str::<serde_json::Value>(&text).unwrap();
        let old = read(
            config
                .write_yarn(r#"{"rope_scaling": null, "a": 1}"#)
                .unwrap(),
        );
        assert_eq!(old["rope_scaling"]["rope_type"], "yarn");
        assert_eq!(old["rope_scaling"]["factor"], 4.0);
        assert_eq!(old["rope_scaling"]["original_max_position_embeddings"], 128);
        assert_eq!(old["a"], 1);
        let new = r#"{"rope_parameters": {"rope_type": "default", "type": "default", "rope_theta": 5.0}}"#;
        let new = read(config.write_yarn(new).unwrap());
        assert_eq!(new["rope_parameters"]["rope_type"], "yarn");
        assert_eq!(new["rope_parameters"]["rope_theta"], 5.0);
        assert!(new["rope_parameters"].get("type").is_none());
        assert!(new.get("rope_scaling").is_none());
        assert_eq!(tiny("llama").write_yarn("{}").unwrap(), "{}");
    }
}

//! Converts a Hugging Face model folder into one GGUF file, the way llama.cpp's
//! `convert_hf_to_gguf.py` does for Llama, Mistral, Qwen2 and Qwen3 models.

use std::path::Path;

use anyhow::{Context, Result};

use super::shards::{self, Source};
use super::{Info, Kind, Value, vocab};
use crate::config::Config;

/// Writes `model_dir` as a Q8_0 GGUF file at `out`, calling the model `name`.
pub fn convert(model_dir: &Path, name: &str, out: &Path) -> Result<()> {
    let text =
        std::fs::read_to_string(model_dir.join("config.json")).context("can't read config.json")?;
    let config = Config::parse(&text)?;
    // llama.cpp runs Mistral as Llama, and expects Llama's rotary halves interleaved.
    let arch = match config.model_type.as_str() {
        "qwen2" => "qwen2",
        "qwen3" => "qwen3",
        _ => "llama",
    };
    let mut metadata = hyperparameters(&config, arch, name);
    metadata.extend(vocab::metadata(model_dir, &config)?);

    let mut tensors: Vec<(Info, Source)> = Vec::new();
    for (hf_name, shape, source) in shards::tensors(model_dir)? {
        if hf_name.ends_with("rotary_emb.inv_freq") {
            continue;
        }
        let name = gguf_name(&hf_name)
            .with_context(|| format!("{hf_name} has no llama.cpp equivalent"))?;
        // Q8_0 needs whole 32-value blocks along each row.
        let kind = match shape.last() {
            Some(width) if shape.len() >= 2 && name.ends_with(".weight") && width % 32 == 0 => {
                Kind::Q8_0
            }
            _ if shape.len() >= 2 && name.ends_with(".weight") => Kind::F16,
            _ => Kind::F32,
        };
        tensors.push((Info { name, shape, kind }, source));
    }
    if let Some(factors) = llama3_rope_factors(&config) {
        let info = Info {
            name: "rope_freqs.weight".into(),
            shape: vec![factors.len()],
            kind: Kind::F32,
        };
        tensors.push((info, Source::Values(factors)));
    }

    let infos: Vec<Info> = tensors.iter().map(|(info, _)| info.clone()).collect();
    super::write(out, &metadata, &infos, |i| {
        let (info, source) = &tensors[i];
        let mut values = shards::read(source)?;
        if arch == "llama" && info.name.contains("attn_q.") {
            values = permute(&values, info.shape[0], config.num_attention_heads);
        } else if arch == "llama" && info.name.contains("attn_k.") {
            values = permute(&values, info.shape[0], config.kv_heads());
        }
        Ok(super::encode(info.kind, values))
    })
}

fn hyperparameters(config: &Config, arch: &str, name: &str) -> Vec<(String, Value)> {
    let key = |k: &str| format!("{arch}.{k}");
    let mut keys = vec![
        ("general.architecture".to_string(), Value::Str(arch.into())),
        ("general.name".to_string(), Value::Str(name.into())),
        // 7 is llama.cpp's "mostly Q8_0": 2-D weights in Q8_0, norms and biases in f32.
        ("general.file_type".to_string(), Value::U32(7)),
        ("general.quantization_version".to_string(), Value::U32(2)),
        (key("vocab_size"), Value::U32(config.vocab_size as u32)),
        (
            key("context_length"),
            Value::U32(config.max_position_embeddings as u32),
        ),
        (
            key("embedding_length"),
            Value::U32(config.hidden_size as u32),
        ),
        (
            key("feed_forward_length"),
            Value::U32(config.intermediate_size as u32),
        ),
        (
            key("block_count"),
            Value::U32(config.num_hidden_layers as u32),
        ),
        (
            key("attention.head_count"),
            Value::U32(config.num_attention_heads as u32),
        ),
        (
            key("attention.head_count_kv"),
            Value::U32(config.kv_heads() as u32),
        ),
        (
            key("attention.layer_norm_rms_epsilon"),
            Value::F32(config.rms_norm_eps as f32),
        ),
        (
            key("rope.freq_base"),
            Value::F32(config.rope_theta() as f32),
        ),
        (
            key("rope.dimension_count"),
            Value::U32(config.head_dim() as u32),
        ),
    ];
    if config.head_dim.is_some() {
        keys.push((
            key("attention.key_length"),
            Value::U32(config.head_dim() as u32),
        ));
        keys.push((
            key("attention.value_length"),
            Value::U32(config.head_dim() as u32),
        ));
    }
    keys
}

fn gguf_name(hf: &str) -> Option<String> {
    match hf {
        "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        "model.norm.weight" => return Some("output_norm.weight".into()),
        "lm_head.weight" => return Some("output.weight".into()),
        _ => {}
    }
    let rest = hf.strip_prefix("model.layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    let (module, param) = rest.rsplit_once('.')?;
    let module = match module {
        "input_layernorm" => "attn_norm",
        "post_attention_layernorm" => "ffn_norm",
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        "self_attn.q_norm" => "attn_q_norm",
        "self_attn.k_norm" => "attn_k_norm",
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        _ => return None,
    };
    Some(format!("blk.{layer}.{module}.{param}"))
}

/// Reorders each head's rows from Hugging Face's two rotary halves to llama.cpp's
/// interleaved pairs, as `LlamaModel.permute` does.
fn permute(values: &[f32], rows: usize, heads: usize) -> Vec<f32> {
    let width = values.len() / rows;
    let half = rows / heads / 2;
    let mut out = Vec::with_capacity(values.len());
    for head in 0..heads {
        for j in 0..half {
            for side in 0..2 {
                let row = (head * 2 + side) * half + j;
                out.extend_from_slice(&values[row * width..(row + 1) * width]);
            }
        }
    }
    out
}

/// Llama 3 stretches low RoPE frequencies; llama.cpp reads the stretch per frequency from a tensor.
fn llama3_rope_factors(config: &Config) -> Option<Vec<f32>> {
    let rope = config.llama3_scaling()?;
    let dim = config.head_dim();
    let old_context = rope.original_max_position_embeddings as f64;
    let (low_wavelen, high_wavelen) = (
        old_context / rope.low_freq_factor,
        old_context / rope.high_freq_factor,
    );
    let factors = (0..dim / 2).map(|i| {
        let freq = 1.0 / config.rope_theta().powf(2.0 * i as f64 / dim as f64);
        let wavelen = 2.0 * std::f64::consts::PI / freq;
        if wavelen < high_wavelen {
            1.0
        } else if wavelen > low_wavelen {
            rope.factor as f32
        } else {
            let smooth = (old_context / wavelen - rope.low_freq_factor)
                / (rope.high_freq_factor - rope.low_freq_factor);
            (1.0 / ((1.0 - smooth) / rope.factor + smooth)) as f32
        }
    });
    Some(factors.collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permute_interleaves_rotary_halves_per_head() {
        // numpy: reshape(heads, 2, half, width).swapaxes(1, 2) on rows 0..8 with 2 heads.
        let rows: Vec<f32> = (0..8).map(|v| v as f32).collect();
        assert_eq!(
            permute(&rows, 8, 2),
            [0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]
        );
        let wide: Vec<f32> = (0..8).map(|v| v as f32).collect();
        assert_eq!(
            permute(&wide, 4, 1),
            [0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]
        );
    }

    #[test]
    fn names_follow_llama_cpp() {
        assert_eq!(
            gguf_name("model.layers.3.self_attn.o_proj.weight").unwrap(),
            "blk.3.attn_output.weight"
        );
        assert_eq!(
            gguf_name("model.layers.0.self_attn.q_proj.bias").unwrap(),
            "blk.0.attn_q.bias"
        );
        assert_eq!(
            gguf_name("model.layers.1.self_attn.k_norm.weight").unwrap(),
            "blk.1.attn_k_norm.weight"
        );
        assert_eq!(gguf_name("lm_head.weight").unwrap(), "output.weight");
        assert!(gguf_name("model.layers.0.mlp.experts.weight").is_none());
    }

    #[test]
    fn rope_factors_undo_our_llama3_scaling() {
        let text = std::fs::read_to_string("tests/fixtures/llama/config.json").unwrap();
        let config = Config::parse(&text).unwrap();
        let factors = llama3_rope_factors(&config).unwrap();
        let scaled = crate::rope::inverse_frequencies(&config);
        for (i, (factor, scaled)) in factors.iter().zip(scaled).enumerate() {
            let base = 1.0
                / config
                    .rope_theta()
                    .powf(2.0 * i as f64 / config.head_dim() as f64);
            assert!(
                (base / scaled - *factor as f64).abs() < 1e-4,
                "frequency {i}"
            );
        }
        assert!(factors.iter().any(|f| *f > 1.0));
    }
}

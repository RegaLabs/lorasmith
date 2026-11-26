# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "safetensors"]
# ///
"""Writes YaRN RoPE tables and a tiny YaRN model with its logits to tests/fixtures/.

Run with `uv run tools/make_yarn_fixture.py`. The Rust tests compare against these files,
so `cargo test` never needs Python.

- tests/fixtures/yarn/rope.json: transformers' inverse frequencies, attention factor and the
  cos/sin its Qwen2 rotary embedding computes, for Qwen2.5's documented setting and for one
  with every optional YaRN key set.
- tests/fixtures/qwen2-yarn: a tiny random Qwen2 stretched 4 times past a 16-token context,
  run on 40 tokens so positions beyond the original context count.
"""

import json
import pathlib

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"
POSITIONS = [0, 1, 7, 100, 1000, 4095]
CASES = {
    "qwen2.5": dict(
        model_type="qwen2",
        vocab_size=64,
        hidden_size=896,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=14,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        max_position_embeddings=32768,
        rope_theta=1000000.0,
        rope_scaling=dict(type="yarn", factor=4.0, original_max_position_embeddings=32768),
    ),
    "custom": dict(
        model_type="qwen2",
        vocab_size=64,
        hidden_size=1024,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        rms_norm_eps=1e-6,
        max_position_embeddings=5120,
        rope_theta=10000.0,
        rope_scaling=dict(
            rope_type="yarn",
            factor=2.5,
            original_max_position_embeddings=2048,
            attention_factor=1.3,
            beta_fast=16.0,
            beta_slow=2.0,
        ),
    ),
}


def rope_case(fields):
    config = AutoConfig.for_model(**fields)
    inv_freq, attention_factor = ROPE_INIT_FUNCTIONS["yarn"](config)
    rotary = Qwen2RotaryEmbedding(config)
    positions = torch.tensor([POSITIONS])
    cos, sin = rotary(torch.zeros(1, dtype=torch.float32), positions)
    plain = {k: v for k, v in fields.items() if k != "rope_scaling"}
    plain["rope_scaling"] = fields["rope_scaling"]
    return {
        "config": plain,
        "inv_freq": inv_freq.tolist(),
        "attention_factor": attention_factor,
        "positions": POSITIONS,
        "cos": cos[0].tolist(),
        "sin": sin[0].tolist(),
    }


def tiny_model():
    config = AutoConfig.for_model(
        model_type="qwen2",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        rope_scaling=dict(type="yarn", factor=4.0, original_max_position_embeddings=16),
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).eval()
    # Fresh models start with norms at 1 and biases at 0, which would hide loading bugs.
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.2)
    folder = OUT / "qwen2-yarn"
    model.save_pretrained(folder, safe_serialization=True)
    tokens = [[(7 * i + 3) % 64 for i in range(40)]]
    with torch.no_grad():
        logits = model(torch.tensor(tokens)).logits
    expected = {"tokens": tokens, "logits": logits.flatten().tolist()}
    (folder / "expected.json").write_text(json.dumps(expected))
    print(f"wrote {folder}")


(OUT / "yarn").mkdir(parents=True, exist_ok=True)
tables = {name: rope_case(fields) for name, fields in CASES.items()}
(OUT / "yarn" / "rope.json").write_text(json.dumps(tables))
print(f"wrote {OUT / 'yarn' / 'rope.json'}")
tiny_model()

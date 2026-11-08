# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "safetensors"]
# ///
"""Writes tiny random Hugging Face models and their logits to tests/fixtures/.

Run with `uv run tools/make_fixtures.py`. The Rust parity tests compare against these files,
so `cargo test` never needs Python.
"""

import json
import pathlib

import torch
from transformers import AutoConfig, AutoModelForCausalLM

OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"
TOKENS = [[3, 1, 4, 1, 5, 9, 2, 6]]
COMMON = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=48,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
    rms_norm_eps=1e-6,
)
MODELS = {
    "llama": dict(
        model_type="llama",
        rope_theta=500000.0,
        rope_scaling=dict(
            rope_type="llama3",
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=64,
        ),
    ),
    "mistral": dict(model_type="mistral"),
    "qwen2": dict(model_type="qwen2", tie_word_embeddings=True, rope_theta=1000000.0),
    "qwen3": dict(model_type="qwen3", head_dim=16, rope_theta=1000000.0),
}

torch.manual_seed(0)
for name, extra in MODELS.items():
    config = AutoConfig.for_model(**{**COMMON, **extra})
    model = AutoModelForCausalLM.from_config(config).eval()
    # Fresh models start with norms at 1 and biases at 0, which would hide loading bugs.
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.2)
    folder = OUT / name
    model.save_pretrained(folder, safe_serialization=True)
    with torch.no_grad():
        logits = model(torch.tensor(TOKENS)).logits
    expected = {"tokens": TOKENS, "logits": logits.flatten().tolist()}
    (folder / "expected.json").write_text(json.dumps(expected))
    print(f"wrote {folder}")

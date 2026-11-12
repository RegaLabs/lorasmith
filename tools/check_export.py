# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "peft", "safetensors"]
# ///
"""Checks that a lorasmith run loads in Hugging Face PEFT and transformers unchanged.

Run with `uv run tools/check_export.py <base model dir> <run dir>`. It compares the logits of
base + PEFT adapter against the merged model, and prints the largest relative difference.
"""

import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_dir, run_dir = sys.argv[1], sys.argv[2]
tokens = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]])

base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32).eval()
with torch.no_grad():
    base_logits = base(tokens).logits
    adapted = PeftModel.from_pretrained(base, run_dir).eval()
    adapter_logits = adapted(tokens).logits
    merged = AutoModelForCausalLM.from_pretrained(f"{run_dir}/merged", dtype=torch.float32).eval()
    merged_logits = merged(tokens).logits

change = (adapter_logits - base_logits).abs().max().item()
# Relative, because float32 rounding grows with depth and the size of the logits.
difference = ((adapter_logits - merged_logits).abs().max() / merged_logits.abs().max()).item()
print(f"adapter changes the logits by up to {change:.4f}")
print(f"adapter vs merged model: max relative difference {difference:.2e}")
sys.exit(0 if change > 1e-3 and difference < 1e-4 else 1)

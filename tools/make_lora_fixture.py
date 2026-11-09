# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "peft", "safetensors"]
# ///
"""Writes a reference LoRA training run to tests/fixtures/lora/, made with HF PEFT.

Run with `uv run tools/make_lora_fixture.py` after tools/make_fixtures.py. It trains the tiny
Llama fixture for three steps the way the Hugging Face Trainer does (global gradient clipping,
then AdamW) and saves the adapter before and after, plus the loss of every step.
"""

import json
import pathlib

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

ROOT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"
OUT = ROOT / "lora"
EXAMPLES = [
    {"tokens": [3, 1, 4, 1, 5, 9, 2, 6], "train": [False, False, True, True, True, True, True, True]},
    {"tokens": [2, 7, 1, 8, 2], "train": [False, True, True, True, True]},
]
# The gradient norm starts near 0.15, so a cap of 0.1 makes clipping matter in every step.
RANK, ALPHA, LEARNING_RATE, MAX_GRAD_NORM, STEPS = 4, 8.0, 1e-2, 0.1, 3

torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(ROOT / "llama", dtype=torch.float32)
targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
config = LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=targets, lora_dropout=0.0, task_type="CAUSAL_LM")
model = get_peft_model(model, config)
# B starts at zero, which would leave A without gradients in the first step.
with torch.no_grad():
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.normal_(0.0, 0.1)
model.save_pretrained(OUT / "start")
(OUT / "start" / "README.md").unlink()

length = max(len(e["tokens"]) for e in EXAMPLES)
pad = lambda values, fill: values + [fill] * (length - len(values))
input_ids = torch.tensor([pad(e["tokens"], 0) for e in EXAMPLES])
attention_mask = torch.tensor([pad([1] * len(e["tokens"]), 0) for e in EXAMPLES])
labels = torch.tensor([pad([t if m else -100 for t, m in zip(e["tokens"], e["train"])], -100) for e in EXAMPLES])

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, eps=1e-8, weight_decay=0.0)
losses, norms = [], []
for _ in range(STEPS):
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    loss.backward()
    norms.append(torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM).item())
    optimizer.step()
    optimizer.zero_grad()
    losses.append(loss.item())
model.save_pretrained(OUT / "end")
(OUT / "end" / "README.md").unlink()

expected = {
    "examples": EXAMPLES,
    "rank": RANK,
    "alpha": ALPHA,
    "learning_rate": LEARNING_RATE,
    "max_grad_norm": MAX_GRAD_NORM,
    "losses": losses,
    "grad_norms_before_clipping": norms,
}
(OUT / "expected.json").write_text(json.dumps(expected, indent=1))
print(f"losses {losses}\ngradient norms {norms}")

# AGENTS.md

Rules for everyone who writes code here, human or AI. Short on purpose. Read it before you touch anything.

## What we're building

lorasmith fine-tunes an open LLM on your own data with one command, on the machine you already have.

```bash
lorasmith train qwen2.5:1.5b my-data.jsonl
ollama create my-data -f runs/my-data/Modelfile
ollama run my-data
```

North star: **under 10 minutes from install to chatting with your fine-tuned model, on a normal laptop, no NVIDIA required.** When two choices compete, pick the one that gets closer to that.

## Scope

In:
- LoRA and QLoRA training of Llama-shaped models: Llama, Mistral, Qwen2, Qwen3 (and their fine-tunes, such as SmolLM).
- Downloading models and tokenizers from Hugging Face.
- Exporting results that other tools load unchanged.

Out, for now: inference server, chat UI, RLHF/DPO, multi-GPU, full fine-tuning, other architectures.

If it isn't in scope, open an issue before writing it. Saying no is how this stays small.

## Stack

- Rust stable, edition 2024, MSRV 1.95 (Burn's).
- Burn 0.22+ handles tensors, autodiff, LoRA and QLoRA. We don't write kernels or LoRA math.
- Backend is picked at runtime: wgpu (Vulkan, Metal, DX12) first, CPU fallback. CUDA and ROCm are opt-in cargo features, never required.
- Allowed dependencies: `burn`, `tokenizers`, `hf-hub`, `safetensors`, `serde`, `serde_json`, `clap`, `minijinja`, `minijinja-contrib` (chat templates call Python string methods), `sha2` (llama.cpp names a pre-tokenizer by a SHA-256 fingerprint), `anyhow`. Anything else needs a one-line reason in the PR.
- No Python at runtime. Python is allowed only in `tools/`, for generating test fixtures with HF PEFT.

## Product rules

- `lorasmith train <model> <data>` works with nothing else. Every flag has a sane default.
- Models: HF repo ids (`Qwen/Qwen2.5-1.5B-Instruct`) or short aliases (`qwen2.5:1.5b`).
- Data: JSONL, format detected per file: `{"messages": [...]}`, `{"prompt": ..., "completion": ...}` or `{"text": ...}`. The chat template comes from the model's `tokenizer_config.json`.
- Each run writes one folder with:
  - a PEFT adapter (`adapter_config.json`, `adapter_model.safetensors`) for HF, vLLM and friends,
  - a merged model in plain HF safetensors layout,
  - a `Modelfile` that points Ollama at the merged model.
- Tensor names and file layouts match Hugging Face exactly. Other tools must not need a converter.
- Unsupported model or not enough memory: fail before downloading gigabytes, and say what to try instead (smaller model, `--qlora`, `--cpu`).
- Every error tells the user what to do next, not only what broke.
- Output is a few lines a person actually wants to read. No spinners when stdout isn't a terminal.

## Code rules

- Write the boring version. Plain functions and structs. If someone new can't follow it in one read, simplify it.
- One crate: library in `src/lib.rs`, thin CLI in `src/main.rs`. Use modules, not crates. Split only when it really hurts.
- No trait without two real implementations today. No custom macros. No `unsafe`. Generics only where Burn needs them.
- Files stay under ~300 lines. Functions fit on a screen.
- Comments explain *why*. If code needs a comment to explain *what* it does, rename things instead.
- No narration comments (`// Step 1: load the model`), no banner or separator comments, no emoji, no comments that restate the signature.
- Public items get a one or two line doc comment.
- Errors: `anyhow` with `.context("...")`. No custom error enums until some caller actually matches on them.
- No `unwrap()` outside tests. `expect("reason")` only for real invariants.
- No dead code, commented-out code, or TODOs without an issue link.
- `cargo fmt` and `cargo clippy --all-targets -- -D warnings` pass. Default lints only.

## Tests

- Unit tests live next to the code and use a tiny random Llama config on CPU, so `cargo test` takes seconds with no downloads and no GPU.
- Parity with HF PEFT is checked against committed fixtures made by `tools/`. Tests never run Python.
- Tests that download real models are `#[ignore]` and run manually or in a nightly job.
- A bug fix comes with a test that fails without it.

## Working here

- Small PRs that do one thing. Branches: `feat/...`, `fix/...`, `docs/...`, `chore/...`.
- Conventional Commits, imperative mood. Say why when it isn't obvious.
- Before pushing: `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test`.
- Agents own the module in their task. Don't edit unrelated files, add dependencies, reformat code you didn't change, or edit this file as a side effect.
- If a rule here is wrong or in the way, change it in its own PR with a reason. The rules serve the project.

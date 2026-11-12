# lorasmith

[![ci](https://github.com/RegaLabs/lorasmith/actions/workflows/ci.yml/badge.svg)](https://github.com/RegaLabs/lorasmith/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/lorasmith.svg)](https://crates.io/crates/lorasmith)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Fine-tune an open LLM on your own data with one command. Any GPU or none at all. No Python, no CUDA setup, no 15 GB Docker image.

```bash
lorasmith train qwen2.5:0.5b my-data.jsonl
ollama create my-data -f runs/my-data/Modelfile
ollama run my-data
```

> **Status:** version 0.1.0, early. Training, download and export work on real models, and the result runs in Ollama. It has been run on Linux only, on an NVIDIA Quadro M2200 (Vulkan), an Intel HD 630 and the CPU. Metal, DirectX 12 and AMD cards should work through wgpu but haven't been tried: [reports are welcome](CONTRIBUTING.md).

## Why

Training a LoRA adapter today means PyTorch, CUDA versions that have to line up, half a dozen Hugging Face libraries, and a machine with an NVIDIA card. lorasmith is a single binary built on [Burn](https://github.com/tracel-ai/burn). It runs on Vulkan, Metal, DirectX 12 or plain CPU, and it writes files that Hugging Face, vLLM, Ollama and llama.cpp already understand.

## Install

With [Rust](https://rustup.rs) 1.95 or newer:

```bash
cargo install --locked lorasmith
```

That is the release on [crates.io](https://crates.io/crates/lorasmith). It has one gap for now: a crate there can't depend on a git commit, so it is built on the latest Burn pre-release, where `--qlora` trains but saves little or no GPU memory. To get `--qlora` in full, install from this repository instead:

```bash
cargo install --locked --git https://github.com/RegaLabs/lorasmith
```

From a checkout, `cargo install --locked --path .` does the same. The gap closes when Burn 0.23 is released with the two fixes it needs ([tracel-ai/burn#6465](https://github.com/tracel-ai/burn/pull/6465) and [#6503](https://github.com/tracel-ai/burn/pull/6503)); until then `Cargo.toml` pins the Burn commit that has both. Either way the first build compiles Burn and takes several minutes.

Every version tag also gets archives for Linux (x86_64, glibc 2.35 or newer), macOS (Apple Silicon and Intel) and Windows (x86_64), each holding the one `lorasmith` binary to unpack and put on your `PATH`. None is published yet.

Three optional cargo features: `cuda` and `rocm` train through NVIDIA's and AMD's own stacks instead of wgpu, and `autotune` benchmarks GPU kernels on first use, which is about 20% faster but needs more memory than a 4 GB GPU has to spare for a 0.5B model.

## Quick start

Put your examples in a JSONL file, one per line:

```json
{"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "Arr, I be Captain Byte."}]}
{"messages": [{"role": "user", "content": "What's 2+2?"}, {"role": "assistant", "content": "Four doubloons, matey."}]}
```

Then train, and chat with the result:

```bash
lorasmith train qwen2.5:0.5b pirate.jsonl
ollama create pirate -f runs/pirate/Modelfile
ollama run pirate
```

The model is downloaded from Hugging Face the first time. Before that, lorasmith checks that its weights can fit on your device, and says what to try if they can't.

## Usage

```bash
lorasmith devices                                   # which GPU it will use
lorasmith train smollm2:135m my-data.jsonl          # a short name
lorasmith train Qwen/Qwen3-0.6B my-data.jsonl       # any Hugging Face id
lorasmith train ./my-local-model my-data.jsonl --epochs 5 --cpu
lorasmith train qwen2.5:0.5b my-data.jsonl --qlora  # 4-bit base, for a small GPU
```

Every flag has a default, so `lorasmith train <model> <data>` is enough.

| Flag | Default | What it does |
|---|---|---|
| `--out <dir>` | `runs/<data file name>` | Where the results go. The default never overwrites an earlier run: the next one gets `-2`, `-3` and so on. |
| `--epochs <n>` | 3 to 10 | Passes over the data. By default small datasets get extra passes, so the adapter has about 75 steps to learn from. |
| `--batch-size <n>` | 4 | Examples per step. Lower it when memory runs out. |
| `--max-len <n>` | 1024 | Longer examples are cut to this many tokens. |
| `--learning-rate <x>` | 0.0002 | The usual LoRA value. |
| `--rank <n>` | 16 | LoRA rank. Higher learns more and uses more memory. Alpha is twice the rank. |
| `--qlora` | off | Holds the model in 4 bits while training. Needs far less GPU memory; the merged model is still built on the full weights. |
| `--cpu` | off | Trains on the CPU even when a GPU works. |

With several GPUs, `CUBECL_WGPU_DEFAULT_DEVICE='IntegratedGpu(0)'` or `'DiscreteGpu(1)'` picks one; `lorasmith devices` lists them.

### Models

Llama, Mistral, Qwen2 and Qwen3, and their fine-tunes such as SmolLM. Give a Hugging Face id, a local folder, or one of these short names:

`smollm2:135m`, `smollm2:360m`, `smollm2:1.7b`, `qwen2.5:0.5b`, `qwen2.5:1.5b`, `qwen2.5:3b`, `qwen2.5:7b`, `qwen3:0.6b`, `qwen3:1.7b`, `qwen3:4b`, `qwen3:8b`, `llama3.2:1b`, `llama3.2:3b`, `mistral:7b`

Llama and Mistral are gated: accept their license on Hugging Face and set `HF_TOKEN`.

### Data

Each line of the data file is one example, in one of three shapes. The shape is detected per file.

```json
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
{"prompt": "2+2?", "completion": "4"}
{"text": "Plain text to continue."}
```

Conversations use the model's own chat template, and only the assistant's replies are learned. Training matches Hugging Face PEFT step for step, so the usual LoRA settings carry over.

With 20 or more examples, a tenth is held out: training prints the loss on them after every epoch and keeps the adapter from the epoch that did best. If training stops early, `runs/<data name>/checkpoint/` holds the adapter from the last finished epoch.

### What a run writes

`runs/<data name>/` holds:

- `adapter_config.json` and `adapter_model.safetensors`: a PEFT adapter for Hugging Face, vLLM and friends,
- `merged/`: the base model with the adapter folded in, in plain Hugging Face layout,
- `model.gguf`: the same merged model in Q8_0 (half the size of float16, near-identical output) for llama.cpp, Ollama and LM Studio,
- `Modelfile`: points Ollama at `model.gguf`.

GGUF export covers byte-level BPE tokenizers (Llama 3, Qwen, SmolLM) and SentencePiece ones whose repo has a `tokenizer.model` file (Mistral, Llama 2). For any other tokenizer the Modelfile points at `merged/`, which Ollama runs only on Apple Silicon and NVIDIA RTX 20 series or newer.

## Memory and speed

Measured on a 2017 laptop: Quadro M2200 with 4 GB under Vulkan, 16 GB of RAM. The data is 16 chat examples of about 45 tokens each, at the default batch of 4, for 40 steps unless the row says otherwise. Times include loading the model and writing the adapter, merged model and GGUF file.

| Model | Flags | GPU memory at peak | Whole run |
|---|---|---|---|
| Qwen2.5-0.5B | none | 2.9 GB | 1m35s |
| Qwen2.5-0.5B | `--qlora` | 1.8 GB | 1m47s |
| Qwen2.5-1.5B | `--qlora` | 3.9 GB | 14m |
| Qwen2.5-1.5B | `--qlora --batch-size 1` | 3.0 GB | 3m01s for 16 steps |

What those numbers mean in practice:

- **Float16 GPUs train in half precision**, which takes about a third less memory for the same losses: Qwen2.5-0.5B took 2.2 GB instead of 3.2 GB on the Intel HD 630. Older NVIDIA cards under Vulkan, like the Quadro here, and the CPU train in float32.
- **`--qlora` cuts the weights, not the rest.** On the GPU, Qwen2.5-0.5B's weights take 2.2 GB, or 1.1 GB with `--qlora`. Both runs reach the same loss.
- **Longer examples and bigger batches cost memory on top of the weights.** On Qwen2.5-0.5B a batch of 180 tokens in total takes about 0.6 GB more, 360 tokens about 1.1 GB, 480 tokens about 1.7 GB.
- **A 1.5B model only just fits 4 GB.** At batch 4 the GPU is 95% full even with 45-token examples, and a GPU that full is erratic: the first four steps have taken anywhere from 3 to 10 minutes, the later ones about 5 seconds each. With `--batch-size 1` it takes 2 seconds a step from the first one.
- **System RAM matters too.** With `--qlora` the model is packed in RAM before it moves to the GPU: Qwen2.5-1.5B needs 8.3 GB for about a minute, Qwen2.5-0.5B about 5 GB.

When memory runs out, lorasmith says so and suggests, in this order: `--qlora`, `--batch-size 1`, a lower `--max-len`, a smaller model, or `--cpu` to use system RAM instead.

## Limits

- Early software: version 0.1.0, no prebuilt binaries yet.
- Run on Linux with two GPUs and the CPU so far. macOS and Windows build in CI on a release tag, which hasn't happened yet.
- Built on a Burn that isn't released. A checkout pins the exact commit; the crates.io release uses the latest Burn pre-release and loses most of what `--qlora` saves.
- A stopped run can't be resumed. Its last finished epoch is kept as an adapter in `checkpoint/`.
- No inference server, chat UI, multi-GPU training, full fine-tuning or other architectures. That is on purpose: lorasmith trains, exports and gets out of the way.

## Roadmap

1. ~~Project rules and skeleton~~
2. ~~Llama-shaped model in Burn, loaded from HF safetensors~~
3. ~~Dataset loading and chat templates~~
4. ~~Training loop with LoRA~~
5. ~~Export: PEFT adapter, merged model, GGUF, Ollama `Modelfile`~~
6. ~~Model download and short aliases~~
7. ~~QLoRA, half precision and memory checks~~
8. Release binaries for Linux, macOS and Windows: ~~release workflow~~, first tagged release

## Contributing

Issues and pull requests are welcome: [CONTRIBUTING.md](CONTRIBUTING.md) says how to build, test and send one. The rules the code follows are in [AGENTS.md](AGENTS.md), and they apply to people and AI agents alike. Security problems go through [SECURITY.md](SECURITY.md).

`tools/` holds Python scripts that make test fixtures and check exports against Hugging Face PEFT; the Rust build and tests never need them. `tools/package.sh` packs the release archive for the machine it runs on, and `.github/workflows/release.yml` runs it on every platform when a version tag is pushed.

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.

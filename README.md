# loraforge

Fine-tune an open LLM on your own data with one command. Any GPU or none at all. No Python, no CUDA setup, no 15 GB Docker image.

```bash
loraforge train qwen2.5:0.5b my-data.jsonl
ollama create my-data -f runs/my-data/Modelfile
ollama run my-data
```

> **Status:** early work in progress. Training, download and export work on real models. Models train in float32, so a 4 GB GPU fits about 0.5B parameters until QLoRA lands. Importing into Ollama has not been tested yet.

## Why

Training a LoRA adapter today means PyTorch, CUDA versions that have to line up, half a dozen Hugging Face libraries, and a machine with an NVIDIA card. loraforge is a single binary built on [Burn](https://github.com/tracel-ai/burn). It runs on Vulkan, Metal, DirectX 12 or plain CPU, and it writes files that Hugging Face, vLLM, Ollama and llama.cpp already understand.

## Usage

```bash
loraforge devices                                   # which GPU it will use
loraforge train smollm2:135m my-data.jsonl          # a short name
loraforge train Qwen/Qwen3-0.6B my-data.jsonl       # any Hugging Face id
loraforge train ./my-local-model my-data.jsonl --epochs 5 --cpu
```

Models can be Llama, Mistral, Qwen2 or Qwen3, and their fine-tunes. Short names: `smollm2:135m`, `smollm2:360m`, `smollm2:1.7b`, `qwen2.5:0.5b`, `qwen2.5:1.5b`, `qwen2.5:3b`, `qwen2.5:7b`, `qwen3:0.6b`, `qwen3:1.7b`, `qwen3:4b`, `qwen3:8b`, `llama3.2:1b`, `llama3.2:3b`, `mistral:7b`. Llama and Mistral are gated: accept their license on Hugging Face and set `HF_TOKEN`.

Each line of the data file is one example, in one of three shapes:

```json
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
{"prompt": "2+2?", "completion": "4"}
{"text": "Plain text to continue."}
```

Conversations use the model's own chat template, and only the assistant's replies are learned. A run writes `runs/<data name>/` with:

- `adapter_config.json` and `adapter_model.safetensors`: a PEFT adapter for Hugging Face, vLLM and friends,
- `merged/`: the base model with the adapter folded in, in plain Hugging Face layout,
- `Modelfile`: points Ollama at `merged/`.

## Roadmap

1. ~~Project rules and skeleton~~
2. ~~Llama-shaped model in Burn, loaded from HF safetensors~~
3. ~~Dataset loading and chat templates~~
4. ~~Training loop with LoRA~~
5. ~~Export: PEFT adapter, merged model, Ollama `Modelfile`~~
6. ~~Model download and short aliases~~
7. QLoRA, half precision and memory checks
8. Release binaries for Linux, macOS and Windows

## Contributing

Read [AGENTS.md](AGENTS.md) first. It applies to people and AI agents alike. `tools/` holds Python scripts that make test fixtures and check exports against Hugging Face PEFT; the Rust build and tests never need them.

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.

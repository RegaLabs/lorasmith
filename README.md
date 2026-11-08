# loraforge

Fine-tune an open LLM on your own data with one command. Any GPU or none at all. No Python, no CUDA setup, no 15 GB Docker image.

```bash
loraforge train qwen2.5:1.5b my-data.jsonl
ollama create my-model -f runs/latest/Modelfile
ollama run my-model
```

> **Status:** early work in progress. Nothing above runs yet. Follow the roadmap below.

## Why

Training a LoRA adapter today means PyTorch, CUDA versions that have to line up, half a dozen Hugging Face libraries, and a machine with an NVIDIA card. loraforge is a single binary built on [Burn](https://github.com/tracel-ai/burn). It runs on Vulkan, Metal, DirectX 12 or plain CPU, and it writes files that Hugging Face, vLLM, Ollama and llama.cpp already understand.

## Roadmap

1. Project rules and skeleton
2. Llama-shaped model in Burn, loaded from HF safetensors
3. Dataset loading and chat templates
4. Training loop with LoRA
5. Export: PEFT adapter, merged model, Ollama `Modelfile`
6. Model download and short aliases
7. QLoRA and memory checks
8. Release binaries for Linux, macOS and Windows

## Contributing

Read [AGENTS.md](AGENTS.md) first. It applies to people and AI agents alike.

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option.

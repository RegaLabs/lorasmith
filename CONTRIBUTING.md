# Contributing to lorasmith

Thanks for wanting to help. This page is the short version; [AGENTS.md](AGENTS.md) holds the
rules the code follows, and they apply to every pull request.

## Reporting a problem

Open an issue with the command you ran, what it printed, and what you expected. The output of
`lorasmith devices` and your operating system help a lot: most surprises so far have been
about one GPU or driver.

Reports from hardware we haven't run on are especially welcome: Apple Silicon, AMD cards and
Windows. "It worked" is useful too.

Security problems go through [SECURITY.md](SECURITY.md), not a public issue.

## What fits

lorasmith does one job: LoRA and QLoRA training of Llama-shaped models, and export in layouts
other tools load unchanged. An inference server, a chat UI, multi-GPU training, full
fine-tuning and other architectures are out of scope for now. If your idea is outside that, or
you aren't sure, open an issue before writing code, so nobody builds something that can't be
merged.

## Building and testing

You need [Rust](https://rustup.rs) 1.95 or newer. No GPU and no downloads are needed for the
tests: they run on a tiny random model on the CPU and take seconds once the build is done.

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
```

All three must pass before a pull request can be merged, and CI runs them again. Tests that
download a real model or need a GPU are marked `#[ignore]`; run them with
`cargo test -- --ignored` if your change touches that path.

The first build takes a while, because it compiles [Burn](https://github.com/tracel-ai/burn).
`Cargo.toml` pins Burn to one commit until Burn 0.23 is released; please don't change that pin
in an unrelated pull request.

`tools/` holds the Python scripts that made the test fixtures with Hugging Face PEFT. You only
need Python if you change a fixture. The Rust build and tests never run it.

## Pull requests

- Keep each one small and about one thing.
- Branch names: `feat/...`, `fix/...`, `docs/...` or `chore/...`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org), in the
  imperative mood. Say why when it isn't obvious.
- A bug fix comes with a test that fails without it.
- A new dependency needs a one-line reason in the pull request.
- If you measured something (memory, time, loss), say on what hardware and with what data.

## Licence

lorasmith is licensed under either of [Apache License 2.0](LICENSE-APACHE) or
[MIT](LICENSE-MIT), at the user's option. By opening a pull request you agree that your
contribution is licensed the same way, with no extra terms.

## Conduct

Be kind and assume good faith. The details are in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

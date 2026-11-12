# /// script
# requires-python = ">=3.10"
# dependencies = ["sentencepiece", "transformers"]
# ///
"""Writes a tiny SentencePiece tokenizer and the vocabulary llama.cpp would store for it
to tests/fixtures/sentencepiece/.

Run with `uv run tools/make_sentencepiece_fixture.py`. The expected values follow
`_create_vocab_sentencepiece` in llama.cpp's converter, read through the real SentencePiece
library, so the Rust reader is checked against both without `cargo test` needing Python.

One deliberate difference: llama.cpp takes added tokens from `added_tokens_decoder` in
tokenizer_config.json, which transformers 5 no longer writes. tokenizer.json lists the same
tokens in every version, so that is what loraforge and this script read.
"""

import io
import json
import pathlib

import sentencepiece
from transformers import LlamaTokenizer, MistralConfig

OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sentencepiece"
# More embeddings than tokens, as fine-tunes that reserve rows have.
VOCAB_SIZE = 320
TEXT = [
    "the quick brown fox jumps over the lazy dog",
    "fine tune a small model on your own data",
    "loraforge trains adapters on a laptop",
    "what is the capital of france",
    "the answer is paris",
] * 20
NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6


def looks_special(token: str) -> bool:
    return token in ("<pad>", "<mask>") or (token.startswith("<|") and token.endswith("|>"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(TEXT),
        model_writer=model,
        model_type="bpe",
        vocab_size=300,
        byte_fallback=True,
        character_coverage=1.0,
        # Llama and Mistral don't normalize either, and the rule table would be 240 KB.
        normalization_rule_name="identity",
        hard_vocab_limit=False,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=-1,
        user_defined_symbols=["[INST]"],
        minloglevel=2,
    )
    (OUT / "tokenizer.model").write_bytes(model.getvalue())

    tokenizer = LlamaTokenizer.from_pretrained(str(OUT), add_bos_token=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[/INST]"]})
    # Not marked special, but shaped like a control token; and one with a SentencePiece space.
    tokenizer.add_tokens(["<|note|>", "▁lora▁forge"])
    tokenizer.save_pretrained(OUT)
    MistralConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    ).save_pretrained(OUT)

    pieces = sentencepiece.SentencePieceProcessor(model_file=str(OUT / "tokenizer.model"))
    tokens = [f"[PAD{i}]" for i in range(VOCAB_SIZE)]
    scores = [-10000.0] * VOCAB_SIZE
    types = [UNUSED] * VOCAB_SIZE
    for i in range(pieces.vocab_size()):
        tokens[i] = pieces.IdToPiece(i)
        scores[i] = pieces.GetScore(i)
        types[i] = (
            UNKNOWN if pieces.IsUnknown(i)
            else CONTROL if pieces.IsControl(i)
            else UNUSED if pieces.IsUnused(i)
            else BYTE if pieces.IsByte(i)
            else NORMAL
        )
    for token in json.loads((OUT / "tokenizer.json").read_text())["added_tokens"]:
        i, content = token["id"], token["content"]
        if token["special"] or looks_special(content):
            types[i] = CONTROL
        else:
            content = content.replace("▁", " ")
            types[i] = USER_DEFINED
        scores[i] = -1000.0
        tokens[i] = content

    expected = {"tokens": tokens, "scores": scores, "types": types}
    (OUT / "expected.json").write_text(json.dumps(expected, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

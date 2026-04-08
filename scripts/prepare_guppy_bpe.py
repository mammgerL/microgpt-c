#!/usr/bin/env python3
import argparse
import json
import os
import random
import subprocess
import sys
from array import array


SPECIAL_TOKENS = ["<pad>", "<|user|>", "<|assistant|>", "<|end|>"]


def train_tokenizer(texts, vocab_size, tokenizer_path):
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    except ImportError as exc:
        raise SystemExit(
            "tokenizers is required. Install it with: pip install tokenizers"
        ) from exc

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.save(tokenizer_path)
    return tokenizer


def write_bin(path, ids, dtype):
    buf = array("H" if dtype == "u16" else "I", ids)
    with open(path, "wb") as f:
        buf.tofile(f)


def main():
    parser = argparse.ArgumentParser(description="Prepare Guppy BPE token data for microgpt-c")
    parser.add_argument("--out-dir", default="data/guppy_bpe")
    parser.add_argument("--samples", type=int, default=60000)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binary", default="./microgpt_mac", help="Path to compiled microgpt_mac binary")
    args = parser.parse_args()

    if args.samples <= 0:
        raise SystemExit("samples must be positive")
    if not (0.0 < args.eval_ratio < 0.5):
        raise SystemExit("eval-ratio must be between 0 and 0.5")
    if args.vocab_size < len(SPECIAL_TOKENS):
        raise SystemExit("vocab-size is too small for required special tokens")

    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, "guppy_raw.txt")
    tokenizer_path = os.path.join(args.out_dir, "tokenizer.json")
    train_bin_path = os.path.join(args.out_dir, "train.bin")
    eval_bin_path = os.path.join(args.out_dir, "eval.bin")
    meta_path = os.path.join(args.out_dir, "meta.json")

    if not os.path.exists(args.binary):
        raise SystemExit(f"binary not found: {args.binary}. Run `make` first.")

    subprocess.run(
        [args.binary, "guppy-data", raw_path, str(args.samples)],
        check=True,
    )

    with open(raw_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    rng = random.Random(args.seed)
    rng.shuffle(lines)
    n_eval = max(1, int(len(lines) * args.eval_ratio))
    eval_lines = lines[:n_eval]
    train_lines = lines[n_eval:]

    print(f"training tokenizer on {len(lines):,} chat samples...")
    tokenizer = train_tokenizer(lines, args.vocab_size, tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    dtype = "u16" if vocab_size < 65536 else "u32"

    def encode_all(split_lines):
        ids = []
        for text in split_lines:
            ids.extend(tokenizer.encode(text).ids)
        return ids

    train_ids = encode_all(train_lines)
    eval_ids = encode_all(eval_lines)
    write_bin(train_bin_path, train_ids, dtype)
    write_bin(eval_bin_path, eval_ids, dtype)

    vocab = tokenizer.get_vocab()
    meta = {
        "name": "guppy_bpe",
        "samples": len(lines),
        "train_samples": len(train_lines),
        "eval_samples": len(eval_lines),
        "vocab_size": vocab_size,
        "dtype": dtype,
        "special_tokens": {
            "pad_id": vocab["<pad>"],
            "user_id": vocab["<|user|>"],
            "assistant_id": vocab["<|assistant|>"],
            "end_id": vocab["<|end|>"],
        },
        "tokenizer_path": "tokenizer.json",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"saved tokenizer: {tokenizer_path}")
    print(f"saved train ids:  {train_bin_path} ({len(train_ids):,} tokens)")
    print(f"saved eval ids:   {eval_bin_path} ({len(eval_ids):,} tokens)")
    print(f"saved meta:       {meta_path}")
    print(f"vocab_size={vocab_size} dtype={dtype}")


if __name__ == "__main__":
    main()

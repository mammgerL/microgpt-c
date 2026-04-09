#!/usr/bin/env python3
import argparse
import json
import os
from array import array


SPECIAL_TOKENS = ["<pad>", "<|im_start|>", "<|im_end|>"]


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
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
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


def normalize(text):
    return " ".join(text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip().split()).lower()


def format_sample(user, assistant):
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"


def main():
    parser = argparse.ArgumentParser(description="Prepare the original GuppyLM HF dataset for microgpt-c")
    parser.add_argument("--out-dir", default="data/guppy_hf")
    parser.add_argument("--dataset", default="arman-bd/guppylm-60k-generic")
    parser.add_argument("--vocab-size", type=int, default=4096)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "datasets is required. Install it with: pip install datasets"
        ) from exc

    os.makedirs(args.out_dir, exist_ok=True)
    raw_train_path = os.path.join(args.out_dir, "train.txt")
    raw_eval_path = os.path.join(args.out_dir, "eval.txt")
    tokenizer_path = os.path.join(args.out_dir, "tokenizer.json")
    train_bin_path = os.path.join(args.out_dir, "train.bin")
    eval_bin_path = os.path.join(args.out_dir, "eval.bin")
    meta_path = os.path.join(args.out_dir, "meta.json")

    ds = load_dataset(args.dataset)
    train_rows = ds["train"]
    eval_rows = ds["test"] if "test" in ds else ds["validation"]

    def to_text(row):
        user = normalize(row["input"])
        assistant = normalize(row["output"])
        return format_sample(user, assistant)

    train_texts = [to_text(row) for row in train_rows]
    eval_texts = [to_text(row) for row in eval_rows]

    with open(raw_train_path, "w", encoding="utf-8") as f:
        for line in train_texts:
            f.write(line + "\n")
    with open(raw_eval_path, "w", encoding="utf-8") as f:
        for line in eval_texts:
            f.write(line + "\n")

    print(f"training tokenizer on {len(train_texts) + len(eval_texts):,} original guppy samples...")
    tokenizer = train_tokenizer(train_texts + eval_texts, args.vocab_size, tokenizer_path)
    vocab = tokenizer.get_vocab()
    vocab_size = tokenizer.get_vocab_size()
    dtype = "u16" if vocab_size < 65536 else "u32"

    def encode_all(lines):
        ids = []
        for text in lines:
            ids.extend(tokenizer.encode(text).ids)
        return ids

    train_ids = encode_all(train_texts)
    eval_ids = encode_all(eval_texts)
    write_bin(train_bin_path, train_ids, dtype)
    write_bin(eval_bin_path, eval_ids, dtype)

    meta = {
        "name": "guppy_hf",
        "dataset": args.dataset,
        "train_samples": len(train_texts),
        "eval_samples": len(eval_texts),
        "vocab_size": vocab_size,
        "dtype": dtype,
        "special_tokens": {
            "pad_id": vocab["<pad>"],
            "im_start_id": vocab["<|im_start|>"],
            "im_end_id": vocab["<|im_end|>"],
        },
        "tokenizer_path": "tokenizer.json",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"saved raw train:  {raw_train_path}")
    print(f"saved raw eval:   {raw_eval_path}")
    print(f"saved tokenizer:  {tokenizer_path}")
    print(f"saved train ids:  {train_bin_path} ({len(train_ids):,} tokens)")
    print(f"saved eval ids:   {eval_bin_path} ({len(eval_ids):,} tokens)")
    print(f"saved meta:       {meta_path}")
    print(f"vocab_size={vocab_size} dtype={dtype}")


if __name__ == "__main__":
    main()

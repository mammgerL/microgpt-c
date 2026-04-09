#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys


def sanitize_chat_text(text):
    return text.replace("\n", " ").replace("\r", " ").replace("\t", " ").lower()


def parse_token_reply(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith("guppy_token_ids>"):
            payload = line.split(">", 1)[1].strip()
            if not payload:
                return []
            return [int(part) for part in payload.split()]
    raise SystemExit("binary output did not contain guppy_token_ids>")


def format_prompt(prompt):
    clean_prompt = sanitize_chat_text(prompt).strip()
    return f"<|im_start|>user\n{clean_prompt}<|im_end|>\n<|im_start|>assistant\n"


def main():
    parser = argparse.ArgumentParser(description="Chat with a BPE-trained Guppy checkpoint")
    parser.add_argument("prompt")
    parser.add_argument("--data-dir", default="data/guppy_bpe")
    parser.add_argument("--ckpt", default="ckpt_best.bin")
    parser.add_argument("--binary", default="./microgpt_mac")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    tokenizer_path = os.path.join(args.data_dir, "tokenizer.json")
    train_bin_path = os.path.join(args.data_dir, "train.bin")
    meta_path = os.path.join(args.data_dir, "meta.json")

    if not os.path.exists(args.binary):
        raise SystemExit(f"binary not found: {args.binary}")
    if not os.path.exists(tokenizer_path):
        raise SystemExit(f"tokenizer not found: {tokenizer_path}")
    if not os.path.exists(train_bin_path):
        raise SystemExit(f"dataset not found: {train_bin_path}")
    if not os.path.exists(meta_path):
        raise SystemExit(f"meta not found: {meta_path}")

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit(
            "tokenizers is required. Install it with: pip install tokenizers"
        ) from exc

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("dtype") != "u16":
        raise SystemExit("only u16 token datasets are currently supported")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    full_prompt = format_prompt(args.prompt)
    prompt_ids = tokenizer.encode(full_prompt).ids
    prompt_arg = ",".join(str(token_id) for token_id in prompt_ids)

    proc = subprocess.run(
        [
            args.binary,
            "token-chat",
            prompt_arg,
            train_bin_path,
            args.ckpt,
            str(args.temperature),
            str(args.max_new_tokens),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    reply_ids = parse_token_reply(proc.stdout)
    reply = tokenizer.decode(reply_ids)
    for tag in ("<|im_start|>", "<pad>"):
        reply = reply.replace(tag, " ")
    if "<|im_end|>" in reply:
        reply = reply.split("<|im_end|>", 1)[0]
    reply = reply.strip()

    print(f"you> {args.prompt}")
    print(f"guppy> {reply}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            sys.stdout.write(exc.stdout)
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise SystemExit(exc.returncode)

#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys


EVAL_CASES = [
    {"id": "greeting_basic", "prompt": "hi guppy", "expect_keywords": ["hello", "hi", "water", "swim", "bubble"]},
    {"id": "feeling_check", "prompt": "how are you feeling today", "expect_keywords": ["water", "good", "ok", "fine", "hungry"]},
    {"id": "food_excited", "prompt": "want some food?", "expect_keywords": ["food", "yes", "eat", "flake", "hungry"]},
    {"id": "temp_hot", "prompt": "it's really hot today", "expect_keywords": ["warm", "water", "hot", "oxygen", "slow"]},
    {"id": "temp_cold", "prompt": "brrr it's freezing", "expect_keywords": ["cold", "slow", "water", "cool", "hide"]},
    {"id": "confused_abstract", "prompt": "what do you think about politics", "expect_keywords": ["don't know", "fish", "water", "human", "understand"]},
    {"id": "water_quality", "prompt": "i just changed your water", "expect_keywords": ["water", "fresh", "clean", "thank", "breathe"]},
    {"id": "light_on", "prompt": "i turned on the light", "expect_keywords": ["light", "see", "bright", "dark"]},
    {"id": "loud_noise", "prompt": "sorry i dropped something", "expect_keywords": ["vibration", "scare", "water", "felt", "hide"]},
    {"id": "goodnight", "prompt": "goodnight guppy", "expect_keywords": ["night", "rest", "dark", "sleep", "still"]},
    {"id": "identity", "prompt": "what are you", "expect_keywords": ["fish", "small", "guppy", "swim", "water"]},
    {"id": "lonely_check", "prompt": "do you get lonely", "expect_keywords": ["alone", "fish", "ok", "bubble", "swim"]},
    {"id": "new_decoration", "prompt": "i got you a new cave", "expect_keywords": ["cave", "hide", "inside", "new", "swim"]},
    {"id": "confused_math", "prompt": "what's 2 plus 2", "expect_keywords": ["don't know", "fish", "brain", "small", "understand"]},
    {"id": "misc_thought", "prompt": "what's on your mind", "expect_keywords": ["water", "food", "swim", "bubble", "think"]},
    {"id": "thank_you", "prompt": "thank you guppy", "expect_keywords": ["welcome", "food", "ok"]},
]


def parse_reply(stdout):
    for line in stdout.splitlines():
        if line.startswith("guppy> "):
            return line[len("guppy> "):].strip().lower()
    raise RuntimeError("failed to find guppy reply in command output")


def score_case(reply, keywords):
    hits = [kw for kw in keywords if kw in reply]
    style_ok = (reply == reply.lower()) and (len(reply.split()) <= 30)
    score = len(hits) / max(len(keywords), 1)
    if style_ok:
        score += 0.1
    return min(score, 1.0), hits, style_ok


def main():
    parser = argparse.ArgumentParser(description="Run held-out Guppy eval prompts against microgpt-c")
    parser.add_argument("--dataset", required=True, help="Path to dataset file used for chat, txt or train.bin")
    parser.add_argument("--ckpt", default="ckpt_best.bin")
    parser.add_argument("--binary", default="./microgpt_mac")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = []
    for case in EVAL_CASES:
        proc = subprocess.run(
            [args.binary, "chat", case["prompt"], args.dataset, args.ckpt, str(args.temperature)],
            check=True,
            capture_output=True,
            text=True,
        )
        reply = parse_reply(proc.stdout)
        score, hits, style_ok = score_case(reply, case["expect_keywords"])
        rows.append({
            "id": case["id"],
            "prompt": case["prompt"],
            "reply": reply,
            "score": score,
            "hits": hits,
            "style_ok": style_ok,
        })

    avg_score = sum(row["score"] for row in rows) / len(rows)
    if args.json:
        json.dump({"avg_score": avg_score, "cases": rows}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    print(f"avg_score={avg_score:.3f}")
    for row in rows:
        print(f"[{row['id']}] score={row['score']:.2f} style={'ok' if row['style_ok'] else 'bad'}")
        print(f"prompt: {row['prompt']}")
        print(f"reply:  {row['reply']}")
        print(f"hits:   {', '.join(row['hits']) if row['hits'] else '-'}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            sys.stdout.write(exc.stdout)
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise SystemExit(exc.returncode)

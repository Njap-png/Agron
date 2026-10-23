"""Generate short samples from a trained Agron checkpoint.

Usage:
  python sample_agron.py --model /content/agron-domain-run/final
  python sample_agron.py --model teddyoas/Agron --output samples.json
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    {
        "name": "python",
        "prompt": "def merge_intervals(intervals):\n    \"\"\"Merge overlapping intervals.\"\"\"\n",
    },
    {
        "name": "javascript",
        "prompt": "// Debounce a function so it runs once after 300ms\nfunction debounce(fn, delay) {\n",
    },
    {
        "name": "cyber",
        "prompt": (
            "### Instruction:\nHow do you detect a reverse shell on a Linux host?\n\n"
            "### Response:\n"
        ),
    },
    {
        "name": "general",
        "prompt": "The capital of France is",
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="teddyoas/Agron")
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    samples = []
    for item in PROMPTS:
        inputs = tok(item["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        samples.append(
            {"name": item["name"], "prompt": item["prompt"], "completion": text}
        )
        print(f"\n=== {item['name']} ===\n{item['prompt']}{text}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"model": args.model, "samples": samples}, f, indent=2)
        print("\nsaved", args.output)


if __name__ == "__main__":
    main()

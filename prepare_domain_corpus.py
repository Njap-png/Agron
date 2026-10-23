"""Build a programming-language + cybersecurity pretraining corpus for Agron.

The corpus is written as flat ``uint16`` token streams (vocab 16384 fits in 16
bits) so pretraining can memory-map it with zero preprocessing overhead.

Sources
-------
code    codeparrot/github-code (public parquet shards, filtered by file extension)
cyber   Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset
replay  Salesforce/wikitext wikitext-2-raw-v1 (the base model's original corpus,
        kept at a small fraction so general-English ability does not collapse)

Outputs (in ``--out-dir``)
--------------------------
train.bin               uint16 training tokens
val_code.bin            uint16 held-out code tokens
val_cyber.bin           uint16 held-out cybersecurity tokens
val_replay.bin          uint16 held-out WikiText tokens
meta.json               provenance, sizes and token counts per source

Usage
-----
  python prepare_domain_corpus.py --out-dir /content/data/agron-domain
"""
import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_FILE = os.path.join(ROOT, "tokenizer.json")

# Extension -> language label. Only these files enter the code corpus.
EXT_LANG = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".java": "Java",
    ".c": "C",
    ".h": "C",
    ".cpp": "C++",
    ".cc": "C++",
    ".hpp": "C++",
    ".cs": "C#",
    ".go": "Go",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".php": "PHP",
    ".sh": "Shell",
    ".sql": "SQL",
    ".scala": "Scala",
    ".lua": "Lua",
    ".pl": "Perl",
}

SKIP_PATH_MARKERS = (
    "node_modules/",
    "vendor/",
    "third_party/",
    "bower_components/",
    "/dist/",
    "/build/",
    "/.git/",
)

GITHUB_CODE_REPO = "codeparrot/github-code"
GITHUB_CODE_SHARDS = 1126
CYBER_REPO = "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset"
CYBER_FILE = "CyberSec-Dataset_escaped.jsonl"


def log(msg):
    print(f"[prepare] {msg}", flush=True)


def download_shards(indices, cache_dir, token):
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download

    def one(i):
        name = f"data/train-{i:05d}-of-{GITHUB_CODE_SHARDS:05d}.parquet"
        return hf_hub_download(
            GITHUB_CODE_REPO, name, repo_type="dataset", cache_dir=cache_dir, token=token
        )

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(one, indices))
    log(f"downloaded {len(paths)} github-code shards in {time.time() - t0:.1f}s")
    return paths


def collect_code(shard_paths, caps, got, per_lang, max_file_chars=80_000, min_file_chars=200):
    """Collect code documents per language up to ``caps`` characters.

    ``got`` (chars) and ``per_lang`` (docs) are mutated in place so the cap is
    enforced across every shard batch, not per batch.
    """
    import pyarrow.parquet as pq

    docs = []
    for path in shard_paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=512, columns=["content", "path"]):
            for text, fpath in zip(
                batch.column("content").to_pylist(), batch.column("path").to_pylist()
            ):
                if not text or not fpath:
                    continue
                lang = EXT_LANG.get(os.path.splitext(fpath)[1].lower())
                if lang is None or lang not in caps:
                    continue
                if any(m in fpath for m in SKIP_PATH_MARKERS):
                    continue
                n = len(text)
                if n < min_file_chars or n > max_file_chars:
                    continue
                if got[lang] >= caps[lang]:
                    continue
                docs.append((lang, text))
                got[lang] += n
                per_lang[lang] += 1
            if all(got[lang] >= caps[lang] for lang in caps):
                break
        log(
            "code progress: "
            + ", ".join(f"{l}={got[l] / 1e6:.1f}MB" for l in sorted(got))
        )
        if all(got[lang] >= caps[lang] for lang in caps):
            break
    return docs


def collect_cyber(path, cap_chars):
    docs = []
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            text = (
                "### System:\n"
                + (ex.get("system") or "").strip()
                + "\n\n### Instruction:\n"
                + (ex.get("user") or "").strip()
                + "\n\n### Response:\n"
                + (ex.get("assistant") or "").strip()
            )
            if len(text) < 200:
                continue
            docs.append(("cyber", text))
            total += len(text)
            if total >= cap_chars:
                break
    log(f"cyber: {len(docs):,} docs, {total / 1e6:.1f}MB")
    return docs, total


def collect_replay(cap_chars, token):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", token=token)
    docs = []
    total = 0
    for split in ("train",):
        for row in ds[split]:
            text = row["text"].strip()
            if len(text) < 100:
                continue
            docs.append(("replay", text))
            total += len(text)
            if total >= cap_chars:
                break
    val_texts = [r["text"] for r in ds["validation"] if r["text"].strip()]
    log(f"replay: {len(docs):,} docs, {total / 1e6:.1f}MB | val {len(val_texts)} docs")
    return docs, total, val_texts


def tokenize_and_write(pairs, tok, out_path, batch_docs=256):
    """Tokenize ``(source, text)`` pairs and append uint16 tokens with EOS."""
    per_source = Counter()
    n_tokens = 0
    eos = tok.eos_token_id
    with open(out_path, "wb") as f:
        for start in range(0, len(pairs), batch_docs):
            batch = pairs[start : start + batch_docs]
            ids = tok(
                [text for _, text in batch],
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            flat = []
            for (source, _), row in zip(batch, ids):
                flat.extend(row)
                flat.append(eos)
                per_source[source] += len(row) + 1
            arr = np.asarray(flat, dtype=np.uint16)
            f.write(arr.tobytes())
            n_tokens += arr.size
            if (start // batch_docs) % 40 == 0:
                log(f"tokenized {start + len(batch):,}/{len(pairs):,} docs -> {n_tokens:,} tokens")
    return n_tokens, dict(per_source)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="/content/data/agron-domain")
    ap.add_argument("--cache-dir", default="/content/data/hf-cache")
    ap.add_argument("--code-chars", type=float, default=150e6, help="total code characters")
    ap.add_argument("--cyber-chars", type=float, default=120e6, help="cyber characters")
    ap.add_argument("--replay-chars", type=float, default=15e6, help="WikiText replay characters")
    ap.add_argument("--max-shards", type=int, default=40)
    ap.add_argument("--shards-per-batch", type=int, default=4)
    ap.add_argument("--val-docs-per-source", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    import torch

    from transformers import PreTrainedTokenizerFast

    os.makedirs(args.out_dir, exist_ok=True)
    tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)
    tok.pad_token = "<pad>"
    tok.bos_token = "<s>"
    tok.eos_token = "</s>"
    tok.unk_token = "<unk>"

    caps = {
        "Python": int(args.code_chars * 0.24),
        "JavaScript": int(args.code_chars * 0.16),
        "TypeScript": int(args.code_chars * 0.12),
        "Java": int(args.code_chars * 0.11),
        "C++": int(args.code_chars * 0.11),
        "C": int(args.code_chars * 0.08),
        "Go": int(args.code_chars * 0.07),
        "Rust": int(args.code_chars * 0.06),
        "C#": int(args.code_chars * 0.05),
    }
    log(f"code caps: {json.dumps({k: round(v / 1e6, 1) for k, v in caps.items()})} MB")

    from huggingface_hub import hf_hub_download

    cyber_path = hf_hub_download(
        CYBER_REPO, CYBER_FILE, repo_type="dataset", cache_dir=args.cache_dir, token=args.token
    )

    code_docs = []
    got = defaultdict(int)
    per_lang = Counter()
    shard_idx = 0
    while shard_idx < args.max_shards:
        idxs = list(range(shard_idx, shard_idx + args.shards_per_batch))
        paths = download_shards(idxs, args.cache_dir, args.token)
        new_docs = collect_code(paths, caps, got, per_lang)
        code_docs.extend(new_docs)
        shard_idx += len(paths)
        log(f"after {shard_idx} shards: {len(code_docs):,} code docs")
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass
        if all(got.get(lang, 0) >= caps[lang] for lang in caps):
            break

    cyber_docs, _ = collect_cyber(cyber_path, args.cyber_chars)
    replay_docs, _, replay_val = collect_replay(args.replay_chars, args.token)

    rng = random.Random(args.seed)
    val = {}
    train = []
    for source, docs in (("code", code_docs), ("cyber", cyber_docs), ("replay", replay_docs)):
        rng.shuffle(docs)
        n_val = min(args.val_docs_per_source, max(1, len(docs) // 50))
        val[source] = docs[:n_val]
        train.extend(docs[n_val:])
    rng.shuffle(train)
    log(f"train docs: {len(train):,} | val docs: {sum(len(v) for v in val.values()):,}")

    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer": "teddyoas/Agron tokenizer.json",
        "seed": args.seed,
        "sources": {
            "code": {"dataset": GITHUB_CODE_REPO, "shards": shard_idx},
            "cyber": {"dataset": CYBER_REPO, "file": CYBER_FILE},
            "replay": {"dataset": "Salesforce/wikitext", "config": "wikitext-2-raw-v1"},
        },
        "caps_chars": {
            "code_total": int(args.code_chars),
            "cyber": int(args.cyber_chars),
            "replay": int(args.replay_chars),
        },
        "val": {},
    }

    for source, pairs in val.items():
        path = os.path.join(args.out_dir, f"val_{source}.bin")
        n, _ = tokenize_and_write(pairs, tok, path)
        meta["val"][source] = {"tokens": n, "docs": len(pairs)}
        log(f"wrote {path} ({n:,} tokens)")

    train_path = os.path.join(args.out_dir, "train.bin")
    t0 = time.time()
    n_train, per_source_tokens = tokenize_and_write(train, tok, train_path)
    log(f"wrote {train_path} ({n_train:,} tokens) in {time.time() - t0:.1f}s")
    meta["train"] = {"tokens": n_train, "docs": len(train), "tokens_per_source": per_source_tokens}
    meta["disk_bytes"] = os.path.getsize(train_path) + sum(
        os.path.getsize(os.path.join(args.out_dir, f"val_{s}.bin")) for s in val
    )
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log("done")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

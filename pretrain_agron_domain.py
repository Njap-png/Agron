"""Continue-pretraining Agron on programming languages + cybersecurity.

Starts from the published Agron weights (``model.safetensors``) and trains on the
uint16 token streams produced by ``prepare_domain_corpus.py``. WikiText replay is
mixed in by the corpus builder so the base model's general-English behaviour is
not erased.

The held-out ``val_code.bin`` / ``val_cyber.bin`` / ``val_replay.bin`` streams are
evaluated before training, periodically during training, and after training so the
domain shift is measurable per source.

Usage
-----
  python pretrain_agron_domain.py --data-dir /content/data/agron-domain \\
      --output-dir /content/agron-domain-run --steps 3000
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

from train_agron import build_published_model, build_tokenizer
from agron_nets.utils import count_parameters, format_parameters

ROOT = os.path.dirname(os.path.abspath(__file__))
VAL_SOURCES = ("code", "cyber", "replay")


class TokenBlockDataset(Dataset):
    """Flat uint16 token stream viewed as non-overlapping ``block_size`` chunks."""

    def __init__(self, path, block_size):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.n_blocks = len(self.tokens) // block_size

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        chunk = np.asarray(
            self.tokens[idx * self.block_size : (idx + 1) * self.block_size], dtype=np.int64
        )
        x = torch.from_numpy(chunk)
        return {"input_ids": x, "labels": x.clone(), "attention_mask": torch.ones_like(x)}


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def evaluate_sources(model, data_dir, block_size, batch_size, max_batches, device):
    was_training = model.training
    model.eval()
    results = {}
    for name in VAL_SOURCES:
        path = os.path.join(data_dir, f"val_{name}.bin")
        if not os.path.exists(path):
            continue
        ds = TokenBlockDataset(path, block_size)
        loader = make_loader(ds, batch_size, False, 0)
        losses = []
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            losses.append(float(model(**batch).loss))
        mean = sum(losses) / max(1, len(losses))
        results[name] = {
            "loss": round(mean, 4),
            "perplexity": round(math.exp(min(mean, 20)), 2),
            "blocks": len(losses),
        }
    if was_training:
        model.train()
    return results


class SourceEvalCallback(TrainerCallback):
    """Evaluate the held-out domain streams every ``every`` optimizer steps."""

    def __init__(self, data_dir, block_size, batch_size, max_batches, every):
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.every = every
        self.history = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.every <= 0 or state.global_step == 0 or state.global_step % self.every:
            return
        res = evaluate_sources(
            model, self.data_dir, self.block_size, self.batch_size, self.max_batches, args.device
        )
        entry = {"step": state.global_step, **res}
        self.history.append(entry)
        print(f"[eval] step {state.global_step}: {json.dumps(res)}", flush=True)
        with open(os.path.join(args.output_dir, "eval_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/content/data/agron-domain")
    ap.add_argument("--output-dir", default="/content/agron-domain-run")
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--logging-steps", type=int, default=25)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.data_dir, "meta.json")) as f:
        corpus_meta = json.load(f)

    tok = build_tokenizer()
    model = build_published_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"model: {format_parameters(count_parameters(model))} params | device: {device}")

    train_ds = TokenBlockDataset(os.path.join(args.data_dir, "train.bin"), args.block_size)
    tokens_per_step = args.batch_size * args.grad_accum * args.block_size
    print(
        f"train blocks: {len(train_ds):,} | {len(train_ds) * args.block_size:,} tokens "
        f"| {tokens_per_step:,} tokens/step | {args.steps} steps = "
        f"{args.steps * tokens_per_step / 1e6:.1f}M tokens"
    )

    before = evaluate_sources(
        model, args.data_dir, args.block_size, args.eval_batch_size, args.eval_batches, device
    )
    print("before:", json.dumps(before))
    with open(os.path.join(args.output_dir, "eval_before.json"), "w") as f:
        json.dump(before, f, indent=2)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup,
        weight_decay=args.weight_decay,
        max_grad_norm=args.grad_clip,
        fp16=torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        dataloader_num_workers=args.num_workers,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    )

    callback = SourceEvalCallback(
        args.data_dir, args.block_size, args.eval_batch_size, args.eval_batches, args.eval_every
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=None,
        callbacks=[callback],
    )

    t0 = time.time()
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    elapsed = time.time() - t0

    after = evaluate_sources(
        model, args.data_dir, args.block_size, args.eval_batch_size, args.eval_batches, device
    )
    print("after:", json.dumps(after))

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)

    report = {
        "status": "completed",
        "model": "teddyoas/Agron (continued pretraining)",
        "data_dir": args.data_dir,
        "corpus": corpus_meta,
        "hyperparameters": {
            k: getattr(args, k)
            for k in (
                "block_size",
                "batch_size",
                "grad_accum",
                "steps",
                "lr",
                "warmup",
                "weight_decay",
                "grad_clip",
                "seed",
            )
        },
        "tokens_per_step": tokens_per_step,
        "total_tokens": args.steps * tokens_per_step,
        "train_wall_seconds": round(elapsed, 1),
        "tokens_per_second": round(args.steps * tokens_per_step / max(elapsed, 1e-9), 1),
        "eval_before": before,
        "eval_after": after,
        "eval_history": callback.history,
        "train_loss_history": [
            {k: e[k] for k in ("step", "loss", "learning_rate", "grad_norm") if k in e}
            for e in trainer.state.log_history
            if "loss" in e
        ],
        "final_checkpoint": final_dir,
    }
    with open(os.path.join(args.output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ("corpus", "eval_history", "train_loss_history")}, indent=2))


if __name__ == "__main__":
    main()

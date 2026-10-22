"""
Train an Agron model FROM SCRATCH (random initialization).

Every architecture, optimiser, scheduler and algorithm from ``agron_nets`` is
available on the command line.

Usage examples:
  python train_from_scratch.py --size small --steps 500
  python train_from_scratch.py --size small --arch mamba --optimizer muon \\
      --recipe orthogonal --steps 1000
  python train_from_scratch.py --arch moe --optimizer lion --scheduler wsd \\
      --algorithm gradient_clip --algorithm ema --steps 2000
  python train_from_scratch.py --size medium --dataset file --file corpus.txt --steps 1000
  python train_from_scratch.py --arch gru --engine hf --steps 200   # legacy HF Trainer
"""
import argparse
import json
import os

import torch
from datasets import Dataset, load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from agron_nets.training import add_training_args, build_model_from_args, trainer_from_args
from agron_nets.utils import count_parameters, format_parameters

# Base architecture presets
PRESETS = {
    "small": {"vocab_size": 16384, "hidden_size": 256, "intermediate_size": 1024, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 4},
    "current": {"vocab_size": 16384, "hidden_size": 512, "intermediate_size": 2048, "num_hidden_layers": 8, "num_attention_heads": 4, "num_key_value_heads": 4},
    "medium": {"vocab_size": 16384, "hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 12, "num_attention_heads": 6, "num_key_value_heads": 6},
}

ROOT = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_FILE = os.path.join(ROOT, "tokenizer.json")


def _work_dir():
    """Writable scratch dir for training artifacts."""
    for cand in [os.environ.get("AGRON_WORK_DIR"), "/content", os.path.join(ROOT, "workspace")]:
        if not cand:
            continue
        try:
            os.makedirs(cand, exist_ok=True)
            probe = os.path.join(cand, ".agron_write_probe")
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            return cand
        except Exception:
            continue
    return ROOT


def tokenize_dataset(tok, text_list, block_size=256):
    tokenized = tok(text_list, truncation=False)["input_ids"]
    chunks = []
    buf = []
    for ids in tokenized:
        buf.extend(ids)
        while len(buf) >= block_size:
            chunks.append(buf[:block_size])
            buf = buf[block_size:]
    if len(buf) >= block_size // 2:
        chunks.append(buf + [tok.pad_token_id] * (block_size - len(buf)))
    if not chunks:
        raise RuntimeError("no training chunks produced")
    return Dataset.from_dict({"input_ids": chunks, "labels": [c[:] for c in chunks]})


def make_loader(dataset, tok, batch_size=8, shuffle=True, num_workers=0):
    """Torch DataLoader whose batches are dicts the Agron engine understands."""

    def collate(rows):
        input_ids = torch.tensor([r["input_ids"] for r in rows], dtype=torch.long)
        labels = input_ids.clone()
        labels = labels.masked_fill(labels == tok.pad_token_id, -100)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=True,
    )


def load_texts(args):
    if args.dataset == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        train_texts = [x["text"] for x in ds["train"] if x["text"].strip()]
        eval_texts = [x["text"] for x in ds["validation"] if x["text"].strip()]
        return train_texts, eval_texts
    if not args.file or not os.path.exists(args.file):
        raise RuntimeError("Provide existing --file for --dataset file")
    with open(args.file, "r", encoding="utf-8") as f:
        texts = [ln.strip() for ln in f if ln.strip()]
    split = max(1, int(len(texts) * 0.95))
    return texts[:split], texts[split:]


def main():
    ap = argparse.ArgumentParser(description="Train an Agron architecture from scratch.")
    ap.add_argument("--size", choices=list(PRESETS), default="small")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--dataset", choices=["wikitext", "file"], default="wikitext")
    ap.add_argument("--file", type=str, default=None, help="text file for --dataset file (one doc per line or paragraph)")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--engine", choices=["agron", "hf"], default="agron",
                    help="agron = compose optimiser/scheduler/algorithms (default); hf = legacy Trainer")
    ap.add_argument("--num-workers", type=int, default=0)
    add_training_args(ap)
    args = ap.parse_args()

    tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)
    tok.pad_token = "<pad>"; tok.bos_token = "<s>"; tok.eos_token = "</s>"; tok.unk_token = "<unk>"

    model, config = build_model_from_args(args, preset=PRESETS[args.size])
    print(f"Initialized from-scratch model: {args.size} / {config.model_type} | "
          f"{format_parameters(count_parameters(model))} params")

    train_texts, eval_texts = load_texts(args)
    train_ds = tokenize_dataset(tok, train_texts, block_size=args.block_size)
    print(f"Training chunks: {len(train_ds):,}")

    out_dir = args.output or os.path.join(_work_dir(), f"agron-scratch-{args.size}-{config.model_type}")
    os.makedirs(out_dir, exist_ok=True)

    if args.engine == "hf":
        if config.model_type != "llama":
            raise SystemExit("--engine hf only supports the llama architecture; use --engine agron")
        ta = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=args.batch_size,
            max_steps=args.steps,
            learning_rate=args.lr if args.lr is not None else 5e-05,
            optim="adamw_torch_fused",
            fp16=True if torch.cuda.is_available() else False,
            logging_steps=max(1, args.log_every),
            save_strategy="steps",
            save_steps=max(100, args.steps // 2),
            save_total_limit=2,
            report_to=[],
            remove_unused_columns=True,
            seed=args.seed,
        )
        collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
        trainer = Trainer(model=model, args=ta, train_dataset=train_ds, data_collator=collator, processing_class=tok)
        trainer.train()
    else:
        trainer = trainer_from_args(args, model, output_dir=out_dir)
        eval_loader = None
        if eval_texts:
            eval_ds = tokenize_dataset(tok, eval_texts, block_size=args.block_size)
            eval_loader = make_loader(eval_ds, tok, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers)
        report = trainer.fit(
            make_loader(train_ds, tok, batch_size=args.batch_size, num_workers=args.num_workers),
            eval_dataloader=eval_loader,
        )
        print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2, default=str))

    final = os.path.join(out_dir, "final")
    os.makedirs(final, exist_ok=True)
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print("From-scratch training complete:", final)


if __name__ == "__main__":
    main()

"""
Train an Agron base LLM FROM SCRATCH (random initialization).

Usage examples:
  python train_from_scratch.py --size small --steps 500
  python train_from_scratch.py --size medium --dataset wikitext --steps 2000
  python train_from_scratch.py --size current --dataset file --file mycorpus.txt --steps 1000
"""
import os
import argparse
import torch
from datasets import load_dataset, Dataset
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Base architecture presets
PRESETS = {
    "small": {"vocab_size": 16384, "hidden_size": 256, "intermediate_size": 1024, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 4},
    "current": {"vocab_size": 16384, "hidden_size": 512, "intermediate_size": 2048, "num_hidden_layers": 8, "num_attention_heads": 4, "num_key_value_heads": 4},
    "medium": {"vocab_size": 16384, "hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 12, "num_attention_heads": 6, "num_key_value_heads": 6},
}

ROOT = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_FILE = os.path.join(ROOT, "tokenizer.json")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(PRESETS), default="small")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--dataset", choices=["wikitext", "file"], default="wikitext")
    ap.add_argument("--file", type=str, default=None, help="text file for --dataset file (one doc per line or paragraph)")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)
    tok.pad_token = "<pad>"; tok.bos_token = "<s>"; tok.eos_token = "</s>"; tok.unk_token = "<unk>"

    cfg = PRESETS[args.size]
    cfg.update({"max_position_embeddings": 512, "bos_token_id": 0, "eos_token_id": 2, "pad_token_id": 1, "tie_word_embeddings": False})
    model = LlamaForCausalLM(LlamaConfig(**cfg))
    print("Initialized from-scratch model:", args.size, "params:", model.num_parameters())

    # data
    if args.dataset == "wikitext":
        from datasets import load_dataset as ld
        ds = ld("Salesforce/wikitext", "wikitext-2-raw-v1")
        texts = [x["text"] for x in ds["train"] if x["text"].strip()]
    else:
        if not args.file or not os.path.exists(args.file):
            raise RuntimeError("Provide existing --file for --dataset file")
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [ln.strip() for ln in f if ln.strip()]

    train_ds = tokenize_dataset(tok, texts, block_size=256)

    out_dir = args.output or os.path.join("/content", f"agron-scratch-{args.size}")
    os.makedirs(out_dir, exist_ok=True)

    ta = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=8,
        max_steps=args.steps,
        learning_rate=5e-05,
        optim="adamw_torch_fused",
        fp16=True if torch.cuda.is_available() else False,
        logging_steps=25,
        save_strategy="steps",
        save_steps=max(100, args.steps // 2),
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=True,
        seed=42,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    trainer = Trainer(model=model, args=ta, train_dataset=train_ds, data_collator=collator, processing_class=tok)
    trainer.train()

    final = os.path.join(out_dir, "final")
    os.makedirs(final, exist_ok=True)
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print("From-scratch training complete:", final)


if __name__ == "__main__":
    main()
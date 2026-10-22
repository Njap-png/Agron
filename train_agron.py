"""Fine-tune the published Agron model (or any architecture) with the full zoo.

Examples:
  python train_agron.py --steps 500                     # continue the published LLaMA
  python train_agron.py --arch hybrid --recipe stable --steps 1000
  python train_agron.py --arch mamba --optimizer muon --scheduler wsd \\
      --algorithm gradient_clip --algorithm ema --steps 2000
  python train_agron.py --engine hf --steps 500         # legacy HF Trainer
"""
import argparse
import json
import os

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from agron_nets.training import add_training_args, build_model_from_args, trainer_from_args
from agron_nets.utils import count_parameters, format_parameters
from train_from_scratch import PRESETS, load_texts, make_loader, tokenize_dataset

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.json")
OUTPUT_DIR = os.getenv("AGRON_OUTPUT_DIR", os.path.join(MODEL_DIR, "agron-model"))


def build_tokenizer():
    tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)
    tok.pad_token = "<pad>"
    tok.bos_token = "<s>"
    tok.eos_token = "</s>"
    tok.unk_token = "<unk>"
    tok.mask_token = "<mask>"
    return tok


def build_published_model():
    """The exact config/weights that were published to the Hub."""
    cfg = LlamaConfig(
        vocab_size=16384,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(cfg)
    weights = os.path.join(MODEL_DIR, "model.safetensors")
    if os.path.exists(weights):
        from safetensors.torch import load_file

        model.load_state_dict(load_file(weights))
        print("Loaded published weights from", weights)
    return model


def main():
    ap = argparse.ArgumentParser(description="Continue training an Agron model.")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--dataset", choices=["wikitext", "file"], default="wikitext")
    ap.add_argument("--file", type=str, default=None)
    ap.add_argument("--output", type=str, default=OUTPUT_DIR)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--engine", choices=["agron", "hf"], default="agron")
    ap.add_argument("--init", choices=["published", "scratch"], default="published",
                    help="start from the published LLaMA weights or fresh weights")
    ap.add_argument("--num-workers", type=int, default=2)
    add_training_args(ap)
    args = ap.parse_args()

    tok = build_tokenizer()

    arch = args.arch or "llama"
    if args.init == "published" and arch == "llama":
        model = build_published_model()
        config = model.config
    else:
        if args.init == "published":
            print(f"--init published is only meaningful for --arch llama; initialising {arch} from scratch")
        model, config = build_model_from_args(args, preset=PRESETS["current"])
    print(f"Model: {config.model_type} | {format_parameters(count_parameters(model))} params")

    train_texts, eval_texts = load_texts(args)
    train_ds = tokenize_dataset(tok, train_texts, block_size=args.block_size)
    eval_ds = tokenize_dataset(tok, eval_texts, block_size=args.block_size) if eval_texts else None
    print(f"Training chunks: {len(train_ds):,}")

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    if args.engine == "hf":
        if config.model_type != "llama":
            raise SystemExit("--engine hf only supports the llama architecture; use --engine agron")
        ta = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            max_steps=args.steps,
            learning_rate=args.lr if args.lr is not None else 5e-05,
            lr_scheduler_type="linear",
            warmup_steps=args.warmup,
            optim="adamw_torch_fused",
            weight_decay=args.weight_decay if args.weight_decay is not None else 0.0,
            fp16=True if torch.cuda.is_available() else False,
            gradient_accumulation_steps=args.grad_accum,
            max_grad_norm=args.grad_clip or 0.0,
            logging_steps=max(1, args.log_every),
            logging_strategy="steps",
            save_strategy="steps",
            save_steps=max(100, args.steps // 2),
            save_total_limit=2,
            dataloader_num_workers=args.num_workers,
            remove_unused_columns=True,
            report_to=[],
            seed=args.seed,
            do_train=True,
        )
        collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
        trainer = Trainer(model=model, args=ta, train_dataset=train_ds, data_collator=collator,
                          processing_class=tok)
        trainer.train()
    else:
        trainer = trainer_from_args(args, model, output_dir=out_dir)
        report = trainer.fit(
            make_loader(train_ds, tok, batch_size=args.batch_size, num_workers=args.num_workers),
            eval_dataloader=(
                make_loader(eval_ds, tok, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
                if eval_ds is not None else None
            ),
        )
        print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2, default=str))

    final_dir = os.path.join(out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    print("Training complete. Final model saved to", final_dir)


if __name__ == "__main__":
    main()

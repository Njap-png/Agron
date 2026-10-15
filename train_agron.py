import os
import torch
from datasets import load_dataset
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.json")
OUTPUT_DIR = os.getenv("AGRON_OUTPUT_DIR", "/content/agron-model")

def build_tokenizer():
    tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)
    tok.pad_token = "<pad>"
    tok.bos_token = "<s>"
    tok.eos_token = "</s>"
    tok.unk_token = "<unk>"
    tok.mask_token = "<mask>"
    return tok

def build_model():
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
    # Load current published weights if available
    weights = os.path.join(MODEL_DIR, "model.safetensors")
    if os.path.exists(weights):
        from safetensors.torch import load_file
        state = load_file(weights)
        model.load_state_dict(state)
        print("Loaded existing weights from", weights)
    return model

def main():
    tok = build_tokenizer()
    model = build_model()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def tokenize(examples):
        out = tok(examples["text"])
        return {"input_ids": out["input_ids"]}

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"], num_proc=2)

    block_size = 512

    def group_texts(examples):
        concatenated = {"input_ids": sum(examples["input_ids"], [])}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            "input_ids": [concatenated["input_ids"][i:i+block_size] for i in range(0, total_length, block_size)]
        }
        result["labels"] = [row[:] for row in result["input_ids"]]
        return result

    lm_ds = tokenized.map(group_texts, batched=True, num_proc=2)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        max_steps=500,
        learning_rate=5e-05,
        lr_scheduler_type="linear",
        warmup_steps=0,
        optim="adamw_torch_fused",
        weight_decay=0.0,
        fp16=True,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        logging_steps=25,
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        dataloader_num_workers=2,
        remove_unused_columns=True,
        report_to=[],
        seed=42,
        do_train=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=lm_ds["train"],
        data_collator=data_collator,
        processing_class=tok,
    )

    # Ensure generation_config exists if needed
    trainer.train()
    final_dir = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    print("Training complete. Final model saved to", final_dir)

if __name__ == "__main__":
    main()

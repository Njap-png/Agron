import os
import torch
import gradio as gr
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

ROOT = os.path.dirname(os.path.abspath(__file__))

def build_tokenizer():
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(ROOT, "tokenizer.json"))
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
    from safetensors.torch import load_file
    state = load_file(os.path.join(ROOT, "model.safetensors"))
    model.load_state_dict(state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, device

TOKENIZER = build_tokenizer()
MODEL, DEVICE = build_model()

def _normalize_history(history):
    pairs = []
    if history and isinstance(history[0], dict):
        current_user = None
        for m in history:
            role = m.get("role")
            content = m.get("content")
            if role == "user":
                current_user = content
            elif role == "assistant" and current_user is not None:
                pairs.append((current_user, content))
                current_user = None
    else:
        for h in history:
            if isinstance(h, (list, tuple)) and len(h) >= 2:
                pairs.append((h[0], h[1]))
    return pairs

def chat_fn(message, history, max_new_tokens, temperature, top_p):
    prompt = ""
    for user_msg, assistant_msg in _normalize_history(history):
        prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
    prompt += f"User: {message}\nAssistant:"
    inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        output = MODEL.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            pad_token_id=1,
            eos_token_id=2,
        )
    new_tokens = output[0][input_len:]
    response = TOKENIZER.decode(new_tokens, skip_special_tokens=True)
    return response.strip()

with gr.Blocks(title="Agron Chat") as demo:
    gr.Markdown("# Agron — Self-Improving Multi-Agent LLM\nChat with the Agron base model (8-layer LLaMA, ~50M params).")
    with gr.Row():
        max_new_tokens = gr.Slider(1, 256, value=80, step=1, label="Max new tokens")
        temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
    gr.ChatInterface(
        fn=chat_fn,
        additional_inputs=[max_new_tokens, temperature, top_p],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

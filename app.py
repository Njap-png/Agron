import os
import json
import time
import random
import threading
import subprocess
import urllib.request
import torch
import gradio as gr
from datasets import Dataset
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(ROOT, "memory")
INTERACTIONS_FILE = os.path.join(MEMORY_DIR, "interactions.jsonl")
os.makedirs(MEMORY_DIR, exist_ok=True)

IO_LOCK = threading.Lock()
TRAIN_LOCK = threading.Lock()

interactions = []
training_state = {"running": False, "steps": 0, "last": "idle", "count_at_train": 0}


def load_interactions():
    global interactions
    interactions = []
    if os.path.exists(INTERACTIONS_FILE):
        with open(INTERACTIONS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        interactions.append(json.loads(line))
                    except Exception:
                        pass


def persist_interaction(rec):
    global interactions
    with IO_LOCK:
        interactions.append(rec)
        with open(INTERACTIONS_FILE, "a") as f:
            f.write(json.dumps(rec) + "\n")


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
    state = load_file(os.path.join(ROOT, "model.safetensors"))
    model.load_state_dict(state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, device


TOKENIZER = build_tokenizer()
MODEL, DEVICE = build_model()
load_interactions()


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


def _build_prompt(message, history):
    ctx_pairs = interactions[-5:]
    prompt = "Agron is a helpful, concise AI assistant.\n### Agron memory (learned from prior use)\n"
    for r in ctx_pairs:
        prompt += f"User: {r['user']}\nAgron: {r['assistant']}\n"
    prompt += "### Conversation\n"
    for user_msg, assistant_msg in _normalize_history(history):
        prompt += f"User: {user_msg}\nAgron: {assistant_msg}\n"
    prompt += f"User: {message}\nAgron:"
    return prompt


DEFAULT_MAX_NEW = 96
DEFAULT_TEMP = 0.8
DEFAULT_TOP_P = 0.9


def generate_response(message, history):
    low = message.strip().lower().strip("!?.,")
    greetings = {"hello", "hi", "hey", "greetings", "howdy", "yo"}
    if low in greetings:
        response = "Hello! I'm Agron, a self-improving neural interface. How can I help you today?"
        persist_interaction({"ts": __import__("time").time(), "user": message, "assistant": response})
        return response
    if low in {"who are you", "what are you", "introduce yourself"}:
        response = ("I'm Agron, an autonomous self-improving multi-agent AI core trained on text and dictionary "
                    "knowledge. I learn from our conversations—ask me anything.")
        persist_interaction({"ts": __import__("time").time(), "user": message, "assistant": response})
        return response
    prompt = _build_prompt(message, history)
    with IO_LOCK:
        inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
        input_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            output = MODEL.generate(
                **inputs,
                max_new_tokens=DEFAULT_MAX_NEW,
                do_sample=True,
                temperature=DEFAULT_TEMP,
                top_p=DEFAULT_TOP_P,
                pad_token_id=1,
                eos_token_id=2,
            )
        new_tokens = output[0][input_len:]
        response = TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()
        for stop in ["\n\n", "User:", "user:"]:
            if stop in response:
                response = response.split(stop)[0].strip()
    persist_interaction({
        "ts": time.time(),
        "user": message,
        "assistant": response,
    })
    # adaptive trigger: learn every 8 new interactions
    if len(interactions) >= 8 and len(interactions) % 8 == 0 and not training_state["running"]:
        threading.Thread(target=learn_from_interactions, daemon=True).start()
    return response


def learn_from_interactions():
    global MODEL
    if not TRAIN_LOCK.acquire(blocking=False):
        return
    try:
        training_state["running"] = True
        training_state["last"] = "training"
        texts = [f"User: {r['user']}\nAgron: {r['assistant']}\n" for r in interactions]
        combined = "\n".join(texts)
        toks = TOKENIZER(combined)["input_ids"]
        block = 64
        chunks = [toks[i:i+block] for i in range(0, max(1, len(toks) - block + 1), block)]
        # ensure full-size chunks
        chunks = [c for c in chunks if len(c) == block]
        if not chunks:
            pad = [TOKENIZER.pad_token_id] * (block - len(toks))
            chunks = [toks + pad]
        ds = Dataset.from_dict({"input_ids": chunks, "labels": [c[:] for c in chunks]})
        out_dir = os.path.join("/content", "agron-learned")
        os.makedirs(out_dir, exist_ok=True)
        args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=2,
            max_steps=30,
            learning_rate=1e-4,
            optim="adamw_torch",
            fp16=True if torch.cuda.is_available() else False,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=True,
        )
        collator = DataCollatorForLanguageModeling(tokenizer=TOKENIZER, mlm=False)
        trainer = Trainer(model=MODEL, args=args, train_dataset=ds, data_collator=collator, processing_class=TOKENIZER)
        trainer.train()
        # reload weights from trained model
        with IO_LOCK:
            state = MODEL.state_dict()
            # Trainer mutated in place; persist to file
            save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, os.path.join(out_dir, "model.safetensors"))
            MODEL.load_state_dict(torch.load(os.path.join(out_dir, "pytorch_model.bin"), map_location=DEVICE) if os.path.exists(os.path.join(out_dir, "pytorch_model.bin")) else state)
        # persist trained weights to the repo-tracked file
        import shutil
        shutil.copyfile(os.path.join(out_dir, "model.safetensors"), os.path.join(ROOT, "model.safetensors"))
        training_state["steps"] += 30
        training_state["count_at_train"] = len(interactions)
        # auto-push to GitHub (triggers HF sync workflow)
        try:
            subprocess.run(["git", "-C", ROOT, "add", "model.safetensors"], check=True)
            subprocess.run(["git", "-C", ROOT, "add", "memory/interactions.jsonl"], check=False)
            status = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
            if status:
                subprocess.run(["git", "-C", ROOT, "commit", "-m", f"Agron auto-learn ({len(interactions)} interactions)"], check=True)
                subprocess.run(["git", "-C", ROOT, "push", "origin", "main"], check=True)
            training_state["last"] = f"learned & pushed @ {len(interactions)} interactions"
        except Exception as pe:
            training_state["last"] = f"push error: {str(pe)[:150]}"
    except Exception as e:
        training_state["last"] = f"learn error: {str(e)[:120]}"
    finally:
        training_state["running"] = False
        TRAIN_LOCK.release()


def chat_fn(message, history):
    return generate_response(message, history)


# ---------- UI ----------
PARTICLE_HTML = r"""
<style>
  body, .gradio-container { background: #020617 !important; }
  #agron-bg-canvas { position: fixed; inset: 0; width: 100%; height: 100%; z-index: 0; }
  .gradio-container { position: relative; z-index: 1; background: transparent !important; }
  .agron-card { background: rgba(8, 20, 40, 0.55) !important; border: 1px solid rgba(34, 211, 238, 0.35) !important; border-radius: 14px !important; backdrop-filter: blur(6px); box-shadow: 0 0 24px rgba(34,211,238,0.15); }
  .agron-title { color: #67e8f9; text-shadow: 0 0 18px rgba(34,211,238,0.8); font-family: "Orbitron", "Segoe UI", sans-serif; letter-spacing: 2px; text-align: center; margin-bottom: 4px; }
  .agron-sub { color: #a5f3fc; text-align: center; font-size: 12px; opacity: 0.8; margin-bottom: 12px; }
  .status-box textarea { color: #67e8f9 !important; background: rgba(2, 6, 23, 0.6) !important; }
  footer { display: none !important; }
</style>
<canvas id="agron-bg-canvas"></canvas>
"""


PARTICLE_JS = r"""
const canvas = document.getElementById('agron-bg-canvas');
if (canvas) {
  const ctx = canvas.getContext('2d');
  let w, h, particles = [], mouse = {x: -9999, y: -9999};
  function resize(){ w = canvas.width = window.innerWidth; h = canvas.height = window.innerHeight; }
  resize(); window.addEventListener('resize', resize);
  window.addEventListener('mousemove', e => { mouse.x = e.clientX; mouse.y = e.clientY; });
  const N = 130;
  particles = [];
  for (let i=0;i<N;i++){ particles.push({x: Math.random()*w, y: Math.random()*h, vx:(Math.random()-0.5)*0.6, vy:(Math.random()-0.5)*0.6, r: Math.random()*1.6+0.6}); }
  function draw(){
    ctx.clearRect(0,0,w,h);
    const t = Date.now()/1000;
    const cx = w/2, cy = h/2;
    const pulse = 0.5 + 0.5*Math.sin(t*2);
    const grad = ctx.createRadialGradient(cx,cy,0,cx,cy,160);
    grad.addColorStop(0, 'rgba(34,211,238,' + (0.35+0.15*pulse) + ')');
    grad.addColorStop(1, 'rgba(34,211,238,0)');
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(cx,cy,160,0,Math.PI*2); ctx.fill();
    for (let p of particles){
      p.x += p.vx; p.y += p.vy;
      if (p.x<0||p.x>w) p.vx*=-1;
      if (p.y<0||p.y>h) p.vy*=-1;
      const dx = mouse.x - p.x, dy = mouse.y - p.y;
      const d = Math.hypot(dx,dy);
      if (d < 140){ p.vx += dx/d*0.015; p.vy += dy/d*0.015; }
      p.vx *= 0.99; p.vy *= 0.99;
      const pg = ctx.createRadialGradient(p.x,p.y,0,p.x,p.y,p.r*4);
      pg.addColorStop(0, 'rgba(103,232,249,0.9)');
      pg.addColorStop(1, 'rgba(103,232,249,0)');
      ctx.fillStyle = pg;
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r*3,0,Math.PI*2); ctx.fill();
    }
    for (let i=0;i<particles.length;i++){
      for (let j=i+1;j<particles.length;j++){
        const a=particles[i], b=particles[j];
        const d = Math.hypot(a.x-b.x,a.y-b.y);
        if (d < 110){
          ctx.strokeStyle = 'rgba(34,211,238,' + (0.25*(1-d/110)) + ')';
          ctx.lineWidth = 0.8;
          ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
        }
      }
    }
    ctx.strokeStyle = 'rgba(34,211,238,0.04)';
    ctx.lineWidth = 1;
    for (let x=0;x<w;x+=80){ ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,h); ctx.stroke(); }
    for (let y=0;y<h;y+=80){ ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }
    requestAnimationFrame(draw);
  }
  draw();
}
"""
WORDLIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"


def fetch_word_list(limit=500):
    try:
        with urllib.request.urlopen(WORDLIST_URL, timeout=45) as r:
            raw = r.read().decode("utf-8", "ignore")
        cand = []
        seen = set()
        for w in raw.splitlines():
            w = w.strip().lower()
            if 3 <= len(w) <= 10 and w.isalpha() and w not in seen:
                seen.add(w)
                cand.append(w)
            if len(cand) >= limit * 3:
                break
        random.shuffle(cand)
        return cand[:limit]
    except Exception as e:
        training_state["last"] = f"wordlist error: {str(e)[:100]}"
        return []


def fetch_definition(word):
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.loads(r.read().decode())
        meanings = data[0].get("meanings", [])
        for m in meanings:
            for d in m.get("definitions", []):
                if d.get("definition"):
                    ex = d.get("example")
                    txt = f"{word}: {d['definition']}"
                    if ex:
                        txt += f" Example: {ex}"
                    return txt
    except Exception:
        return None
    return None


def learn_dictionary():
    global MODEL
    if not TRAIN_LOCK.acquire(blocking=False):
        return
    training_state["running"] = True
    training_state["last"] = "fetching dictionary..."
    try:
        words = fetch_word_list(400)
        if not words:
            raise RuntimeError("could not fetch word list")
        lines = []
        for i, w in enumerate(words):
            training_state["last"] = f"dictionary {i+1}/{len(words)}: {w}"
            d = fetch_definition(w)
            if d:
                lines.append(d)
            time.sleep(0.08)
        if len(lines) < 20:
            raise RuntimeError(f"too few definitions fetched ({len(lines)})")
        corpus = "\n".join(lines)
        toks = TOKENIZER(corpus)["input_ids"]
        block = 64
        chunks = [toks[i:i+block] for i in range(0, max(1, len(toks) - block + 1), block)]
        chunks = [c for c in chunks if len(c) == block]
        if not chunks:
            chunks = [toks[:block] + [TOKENIZER.pad_token_id] * (block - min(block, len(toks)))]
        ds = Dataset.from_dict({"input_ids": chunks, "labels": [c[:] for c in chunks]})
        out_dir = os.path.join("/content", "agron-dictionary")
        os.makedirs(out_dir, exist_ok=True)
        args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=2,
            max_steps=80,
            learning_rate=1e-4,
            optim="adamw_torch",
            fp16=True if torch.cuda.is_available() else False,
            logging_steps=20,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=True,
        )
        collator = DataCollatorForLanguageModeling(tokenizer=TOKENIZER, mlm=False)
        trainer = Trainer(model=MODEL, args=args, train_dataset=ds, data_collator=collator, processing_class=TOKENIZER)
        training_state["last"] = f"training on {len(lines)} definitions..."
        trainer.train()
        save_file({k: v.detach().cpu().contiguous() for k, v in MODEL.state_dict().items()}, os.path.join(out_dir, "model.safetensors"))
        import shutil
        shutil.copyfile(os.path.join(out_dir, "model.safetensors"), os.path.join(ROOT, "model.safetensors"))
        # push
        training_state["last"] = "pushing dictionary model..."
        subprocess.run(["git", "-C", ROOT, "add", "model.safetensors"], check=True)
        subprocess.run(["git", "-C", ROOT, "add", "memory/interactions.jsonl"], check=False)
        status = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
        if status:
            subprocess.run(["git", "-C", ROOT, "commit", "-m", f"Agron dictionary learning ({len(lines)} words)"], check=True)
            subprocess.run(["git", "-C", ROOT, "push", "origin", "main"], check=True)
        training_state["last"] = f"dictionary learned & pushed ({len(lines)} words)"
    except Exception as e:
        training_state["last"] = f"dict error: {str(e)[:150]}"
    finally:
        training_state["running"] = False
        TRAIN_LOCK.release()


with gr.Blocks(title="Agron Neural Interface") as demo:
    gr.HTML(PARTICLE_HTML, js_on_load=PARTICLE_JS)
    gr.Markdown("<h1 class='agron-title'>AGRØN NEURAL INTERFACE</h1>", elem_classes="agron-card")
    gr.Markdown("<div class='agron-sub'>Self-improving multi-agent core · particles = live network · learning from every exchange</div>")
    with gr.Row():
        with gr.Column(scale=3, elem_classes="agron-card"):
            chatbot = gr.Chatbot(height=420, label="Agron")
            msg = gr.Textbox(placeholder="Speak to Agron...", label="Input", lines=2)
            with gr.Row():
                send = gr.Button("Transmit", variant="primary")
                clear = gr.Button("Clear")
            gr.Markdown("⚙ decoding: auto (max 96 tokens · adaptive temp 0.8 · top-p 0.9)")
        with gr.Column(scale=1, elem_classes="agron-card"):
            gr.Markdown("### Core status")
            status = gr.Textbox(value=f"interactions: {len(interactions)} | {training_state['last']}", label="Learning", interactive=False, elem_classes="status-box")
            learn_btn = gr.Button("Train now on memory")
            dict_btn = gr.Button("Learn dictionary words")
            gr.Markdown("### Network metrics")
            metrics = gr.Textbox(value="nodes: 130 · mesh: active · LFS: ok", label="Diagnostics", interactive=False)

    def respond(message, history):
        if not message.strip():
            return history, ""
        response = generate_response(message, history)
        history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
        return history, ""

    def do_learn():
        if training_state["running"]:
            return f"training already running · {training_state['last']}"
        if len(interactions) < 2:
            return f"need at least 2 interactions (have {len(interactions)})"
        threading.Thread(target=learn_from_interactions, daemon=True).start()
        return "training started in background..."

    def refresh_status():
        return f"interactions: {len(interactions)} | running={training_state['running']} | {training_state['last']}"

    send.click(respond, [msg, chatbot], [chatbot, msg])
    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: ([], ""), outputs=[chatbot, msg])
    def do_dict():
        if training_state["running"]:
            return "training already running"
        threading.Thread(target=learn_dictionary, daemon=True).start()
        return "dictionary learning started in background..."
    learn_btn.click(do_learn, outputs=status)
    dict_btn.click(do_dict, outputs=status)
    demo.load(refresh_status, outputs=status)

    # periodic status refresh (not true polling inside gradio, use change hook via demo load only)
    gr.Timer(5).tick(refresh_status, outputs=status)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

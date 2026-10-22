import os
import json
import sys
import time
import random
import threading
import subprocess
import urllib.request
import requests
from bs4 import BeautifulSoup
import torch
import gradio as gr
from datasets import Dataset
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from safetensors.torch import load_file, save_file

from agron_nets import (
    ARCHITECTURES,
    RECIPES,
    algorithm_names,
    architecture_names,
    build_architecture,
    count_parameters,
    format_parameters,
    optimizer_names,
    scheduler_names,
)
from agron_nets.training import AgronTrainer, TrainerConfig, get_recipe, recipe_names

ROOT = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(ROOT, "memory")
INTERACTIONS_FILE = os.path.join(MEMORY_DIR, "interactions.jsonl")
os.makedirs(MEMORY_DIR, exist_ok=True)


def _pick_work_dir():
    """Writable scratch dir for training artifacts (works on HF Spaces where /content may not exist)."""
    candidates = [os.environ.get("AGRON_WORK_DIR"), "/content", os.path.join(ROOT, "workspace")]
    for cand in candidates:
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


WORK_DIR = _pick_work_dir()

# Deployment / runtime configuration
VOICE_MODEL_ID = os.environ.get("AGRON_VOICE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
USE_VOICE = os.environ.get("AGRON_USE_VOICE", "1").strip().lower() not in ("0", "false", "no")
AUTO_PUSH = os.environ.get("AGRON_AUTO_PUSH", "1").strip().lower() not in ("0", "false", "no")
SERVER_PORT = int(os.environ.get("AGRON_PORT", "7860"))
SERVER_NAME = os.environ.get("AGRON_HOST", "0.0.0.0")
SHARE = os.environ.get("AGRON_SHARE", "0").strip().lower() in ("1", "true", "yes")

IO_LOCK = threading.Lock()
TRAIN_LOCK = threading.Lock()

# --------------------------------------------------------------------------- #
# Neural-network zoo selection
# --------------------------------------------------------------------------- #
# The reference checkpoint is a 50.3M-parameter LLaMA; the same preset is used
# for every other architecture so they are directly comparable at runtime.
AGRON_PRESET = dict(
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
ACTIVE_ARCH = os.environ.get("AGRON_ARCH", "llama").strip().lower()
if ACTIVE_ARCH not in ARCHITECTURES:
    ACTIVE_ARCH = "llama"

interactions = []
training_state = {"running": False, "steps": 0, "last": "idle", "count_at_train": 0}
LAST_TRAIN_TIME = 0.0


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


def git_auto_push(message, paths):
    """Stage `paths`, commit and push to origin (Hugging Face). Returns (ok, detail)."""
    if not AUTO_PUSH:
        return False, "auto-push disabled (AGRON_AUTO_PUSH=0)"
    try:
        for p in paths:
            subprocess.run(["git", "-C", ROOT, "add", "--", p], check=False)
        staged = subprocess.run(
            ["git", "-C", ROOT, "diff", "--cached", "--name-only"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        if not staged:
            return True, "no changes to push"
        status = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"],
                                capture_output=True, text=True, check=True).stdout.strip()
        if not status:
            return True, "no changes to push"
        # make sure a committer identity exists (Spaces do not ship one)
        for key, val in (("user.name", "Agron"), ("user.email", "agron@users.noreply.huggingface.co")):
            if not subprocess.run(["git", "-C", ROOT, "config", "--get", key],
                                  capture_output=True, text=True).stdout.strip():
                subprocess.run(["git", "-C", ROOT, "config", key, val], check=False)
        branch = subprocess.run(["git", "-C", ROOT, "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True).stdout.strip() or "main"
        subprocess.run(["git", "-C", ROOT, "commit", "-m", message], check=True)
        subprocess.run(["git", "-C", ROOT, "push", "origin", f"HEAD:{branch}"], check=True)
        return True, f"pushed to Hugging Face ({branch})"
    except Exception as e:
        return False, f"push error: {str(e)[:150]}"


TOPICS_FILE = os.path.join(MEMORY_DIR, "learned_topics.json")
learned_topics = []


def load_topics():
    global learned_topics
    learned_topics = []
    if os.path.exists(TOPICS_FILE):
        try:
            with open(TOPICS_FILE, "r") as f:
                learned_topics = json.load(f)
        except Exception:
            learned_topics = []


def persist_topics():
    global learned_topics
    with IO_LOCK:
        normalized = ["dictionary words" if str(x).startswith("dictionary words") else x for x in learned_topics]
        learned_topics = list(dict.fromkeys(normalized))
        with open(TOPICS_FILE, "w") as f:
            json.dump(learned_topics, f)


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


def build_model(arch=None, load_published=True):
    """Build one architecture from the zoo and load the best weights available.

    ``llama`` keeps loading the published ``model.safetensors`` so the shipped
    Agron brain is preserved. Every other architecture loads its own latest
    checkpoint from ``WORK_DIR`` when one exists.
    """
    arch = (arch or ACTIVE_ARCH or "llama").strip().lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if arch == "llama":
        model, _ = build_architecture("llama", **AGRON_PRESET)
        weights = os.path.join(ROOT, "model.safetensors")
        if load_published and os.path.exists(weights):
            try:
                model.load_state_dict(load_file(weights))
                print("Loaded published Agron weights:", weights)
            except Exception as exc:
                print("Published weights incompatible, using fresh init:", exc)
    else:
        model, _ = build_architecture(arch, **AGRON_PRESET)
        saved = os.path.join(WORK_DIR, f"agron-{arch}", "final")
        if load_published and os.path.exists(os.path.join(saved, "config.json")):
            try:
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(saved)
                print(f"Loaded {arch} checkpoint: {saved}")
            except Exception as exc:
                print(f"Could not load {arch} checkpoint, using fresh init:", exc)

    model.to(device)
    model.eval()
    return model, device


TOKENIZER = build_tokenizer()
MODEL, DEVICE = build_model()
load_interactions()

VOICE_ID = VOICE_MODEL_ID
VOICE_TOK = None
VOICE_MODEL = None
if USE_VOICE:
    try:
        from transformers import AutoTokenizer as _ATok, AutoModelForCausalLM as _AModel
        print("Loading voice model:", VOICE_ID)
        VOICE_TOK = _ATok.from_pretrained(VOICE_ID)
        VOICE_MODEL = _AModel.from_pretrained(VOICE_ID, dtype=__import__("torch").float16, device_map="auto")
        VOICE_MODEL.eval()
        print("Voice model loaded:", VOICE_ID)
    except Exception as _ve:
        print("Voice model failed to load:", _ve)
        VOICE_TOK = None
        VOICE_MODEL = None
else:
    print("Voice model disabled (AGRON_USE_VOICE=0) - using the native Agron base model")
load_topics()
if not learned_topics:
    learned_topics.append("base text model (WikiText-2)")
    persist_topics()


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


DEFAULT_MAX_NEW = 192
DEFAULT_TEMP = 0.6
DEFAULT_TOP_P = 0.85


def is_bad_response(text):
    import re
    t = (text or "").strip()
    if len(t) < 3:
        return True
    if "@-@" in t or t.count('"') > 6:
        return True
    if re.search(r"(.{8,})\\1", t):
        return True
    words = t.split()
    if len(words) > 120:
        return True
    # gibberish heuristic: too many short tokens
    if sum(1 for w in words if len(w) <= 2) > len(words) * 0.6 and len(words) > 12:
        return True
    if text.count("was a") >= 2 or text.count(" in the ") >= 2 or text.count(" the ") >= 6:
        return True
    if "@-@" in t or t.count('"') > 4:
        return True
    return False


def safe_web_answer(query):
    """Grounded answer built from live search results (used when the model's own reply is unusable)."""
    try:
        results = web_search(query, 3)
        if not results:
            return None
        lines = []
        for i, r in enumerate(results):
            sn = (r.get("snippet") or "").strip()
            if len(sn) < 20:
                page = crawl_page(r["url"])
                sn = (page[:300] + "...") if page else sn
            if not sn:
                continue
            lines.append(f"{i + 1}. {r['title']} — {sn} ({r['url']})")
        return "\n".join(lines) if lines else None
    except Exception:
        pass
    return None


def generate_response(message, history):
    low = message.strip().lower().strip("!?.,")
    greetings = {"hello", "hi", "hey", "greetings", "howdy", "yo"}
    search_intent = any(k in low for k in ["search", "find", "look up", "lookup", "browse", "web"])
    if search_intent:
        res = web_search(message, 5)
        if res:
            lines = [f"{i+1}. {r['title']} — {r.get('snippet','')} ({r['url']})" for i, r in enumerate(res)]
            response = "Here are live web results:\n" + "\n".join(lines)
            persist_interaction({"ts": __import__("time").time(), "user": message, "assistant": response})
            return response
        else:
            response = "I couldn't find web results for that query. Try rephrasing it."
            persist_interaction({"ts": __import__("time").time(), "user": message, "assistant": response})
            return response
    if low in greetings:
        response = "Hello! I'm Agron, a self-improving neural interface. How can I help you today?"
        persist_interaction({"ts": __import__("time").time(), "user": message, "assistant": response})
        return response
    if low in {"who are you", "what are you", "introduce yourself"}:
        response = ("I'm Agron, an autonomous self-improving multi-agent AI core trained on text and dictionary "
                    "knowledge. I learn from our conversations—ask me anything.")
        persist_interaction({"ts": __import__("time").time(), "user": message, "assistant": response})
        return response
    if low in {"what did you learn", "what have you learned", "what did you learn about", "what do you know"}:
        parts = []
        if interactions:
            parts.append(f"{len(interactions)} conversation exchanges (latest: '{interactions[-1]['user'][:80]}')")
        if learned_topics:
            parts.append("learned topics: " + ", ".join(learned_topics[-12:]))
        else:
            parts.append("no specialized training runs yet, but I store every conversation")
        response = "I've learned from you and my training sources: " + "; ".join(parts) + "."
        persist_interaction({"ts": __import__("time").time(), "user": message, "assistant": response})
        return response
    # Live web context is only useful for the voice model; the base model path falls
    # back to a grounded search answer only when its own reply is unusable.
    _web_ctx = ""
    if VOICE_MODEL is not None:
        try:
            if len(message.strip()) >= 4:
                _res = web_search(message, 3)
                _lines = [f"- {r['title']}: {r.get('snippet','')}" for r in _res if (r.get('snippet') or '').strip()]
                if _lines:
                    _web_ctx = "Web search results (live):\n" + "\n".join(_lines[:3])
        except Exception:
            _web_ctx = ""
    if VOICE_MODEL is not None:
        mem_lines = [f"User: {r['user']}\nAgron: {r['assistant']}" for r in interactions[-3:]]
        mem_ctx = "\n".join(mem_lines)
        user_content = message
        if mem_ctx:
            user_content = f"Relevant memory:\n{mem_ctx}\n\nUser: {message}"
        if _web_ctx:
            user_content = f"{_web_ctx}\n\n{user_content}\n\nAnswer using the live web results above."
        _msgs = [
            {"role": "system", "content": "You are Agron, a self-improving AI assistant. Reason step-by-step (premises, plan, critique, refine), then answer concisely. Use live web search results when provided. Never say you cannot search or browse. If unsure, suggest 'Search & learn web'."},
            {"role": "user", "content": user_content},
        ]
        _prompt = VOICE_TOK.apply_chat_template(_msgs, tokenize=False, add_generation_prompt=True)
        with IO_LOCK:
            _inputs = VOICE_TOK(_prompt, return_tensors="pt").to(VOICE_MODEL.device)
            with torch.no_grad():
                _out = VOICE_MODEL.generate(
                    **_inputs,
                    max_new_tokens=DEFAULT_MAX_NEW,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.15,
                )
            response = VOICE_TOK.decode(_out[0][_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    else:
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
                    repetition_penalty=1.4,
                    no_repeat_ngram_size=3,
                    top_k=40,
                )
            new_tokens = output[0][input_len:]
            response = TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()
            for stop in ["\n\n", "User:", "user:"]:
                if stop in response:
                    response = response.split(stop)[0].strip()
            import re
            response = re.sub(r"(.{6,})\1{2,}", r"\1", response).strip()
    if is_bad_response(response):
        sw = safe_web_answer(message)
        response = sw if sw else "I'm not sure. Could you rephrase, or use 'Search & learn web' to teach me."
    persist_interaction({
        "ts": time.time(),
        "user": message,
        "assistant": response,
    })
    # adaptive trigger: self-learn from every conversation (cooldown 30s)
    global LAST_TRAIN_TIME
    if len(interactions) >= 8 and not training_state["running"] and (time.time() - LAST_TRAIN_TIME) >= 300:
        LAST_TRAIN_TIME = time.time()
        threading.Thread(target=learn_from_scratch_memory, daemon=True).start()
    return response


def zoo_summary():
    """Human-readable inventory of everything the zoo can build."""
    arch_lines = "\n".join(
        f"- **{name}** — {ARCHITECTURES[name]['label']}: {ARCHITECTURES[name]['description']}"
        for name in architecture_names()
    )
    return (
        f"**{len(ARCHITECTURES)} architectures · {len(optimizer_names())} optimisers · "
        f"{len(algorithm_names())} algorithms · {len(scheduler_names())} schedules · "
        f"{len(RECIPES)} recipes**\n\n{arch_lines}"
    )


def _resolve_zoo(optimizer=None, scheduler=None, recipe=None, algorithms=None):
    """Recipe values win unless the caller explicitly overrides them."""
    if recipe and recipe not in ("custom", "none", ""):
        entry = get_recipe(recipe)
        return entry["optimizer"], entry["scheduler"], list(entry["algorithms"])
    return (optimizer or "adamw"), (scheduler or "cosine"), list(algorithms or [])


def _lm_loader(chunks, batch_size=2, block=None):
    """Wrap token blocks in a DataLoader whose batches satisfy the engine."""

    def collate(rows):
        input_ids = torch.tensor([r["input_ids"] for r in rows], dtype=torch.long)
        labels = input_ids.clone()
        labels = labels.masked_fill(labels == TOKENIZER.pad_token_id, -100)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }

    return torch.utils.data.DataLoader(chunks, batch_size=batch_size, shuffle=True, collate_fn=collate)


def reload_live_model(model_dir):
    """Swap the live model for the freshly saved one, whatever the architecture."""
    global MODEL, DEVICE
    try:
        from transformers import AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        fresh = AutoModelForCausalLM.from_pretrained(model_dir)
        fresh.to(device)
        fresh.eval()
        with IO_LOCK:
            MODEL = fresh
            DEVICE = device
        return True
    except Exception as exc:
        print("live reload failed:", exc)
        return False


def switch_architecture(arch):
    """Rebuild the runtime brain with another architecture from the zoo."""
    global MODEL, DEVICE, ACTIVE_ARCH
    arch = (arch or "").strip().lower()
    if arch not in ARCHITECTURES:
        return f"unknown architecture: {arch}"
    with IO_LOCK:
        MODEL, DEVICE = build_model(arch, load_published=True)
        ACTIVE_ARCH = arch
    label = ARCHITECTURES[arch]["label"]
    training_state["last"] = f"brain switched to {label}"
    return f"brain switched to {label} · {format_parameters(count_parameters(MODEL))} params on {DEVICE}"


def _save_repo_weights(model_dir):
    """Only the llama brain maps onto the published ``model.safetensors``."""
    if ACTIVE_ARCH != "llama":
        return False
    src = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(src):
        return False
    import shutil

    shutil.copyfile(src, os.path.join(ROOT, "model.safetensors"))
    return True


def _train_chunks_with_engine(chunks, steps, out_dir, optimizer="adamw", scheduler="cosine",
                              algorithms=None):
    """Fine-tune the live model in place through the Agron engine."""
    os.makedirs(out_dir, exist_ok=True)
    trainer = AgronTrainer(
        MODEL,
        optimizer=optimizer,
        scheduler=scheduler,
        algorithms=list(algorithms or []),
        config=TrainerConfig(max_steps=int(steps), log_every=0, output_dir=out_dir),
        device=DEVICE,
    )
    report = trainer.fit(_lm_loader(chunks, batch_size=2))
    MODEL.eval()
    model_dir = os.path.join(out_dir, "final")
    MODEL.save_pretrained(model_dir)
    return report, model_dir


def learn_from_interactions(optimizer=None, scheduler=None, recipe=None, algorithms=None, steps=30):
    global MODEL
    if not TRAIN_LOCK.acquire(blocking=False):
        return
    try:
        if len(interactions) < 2:
            training_state["last"] = "need at least 2 interactions before training"
            return
        training_state["running"] = True
        training_state["last"] = "training"
        optimizer, scheduler, algorithms = _resolve_zoo(optimizer, scheduler, recipe, algorithms)

        texts = [f"User: {r['user']}\nAgron: {r['assistant']}\n" for r in interactions]
        toks = TOKENIZER(" ".join(texts))["input_ids"]
        block = 64
        chunks = [toks[i:i + block] for i in range(0, max(1, len(toks) - block + 1), block)]
        chunks = [{"input_ids": c} for c in chunks if len(c) == block]
        if not chunks:
            pad = [TOKENIZER.pad_token_id] * max(0, block - len(toks))
            chunks = [{"input_ids": toks + pad}]

        out_dir = os.path.join(WORK_DIR, f"agron-{ACTIVE_ARCH}", "finetune")
        trainer = AgronTrainer(
            MODEL,
            optimizer=optimizer,
            scheduler=scheduler,
            algorithms=algorithms,
            config=TrainerConfig(max_steps=steps, log_every=0, output_dir=out_dir),
            device=DEVICE,
        )
        report = trainer.fit(_lm_loader(chunks, batch_size=2))
        model_dir = os.path.join(WORK_DIR, f"agron-{ACTIVE_ARCH}", "final")
        MODEL.eval()
        MODEL.save_pretrained(model_dir)

        trained_file = os.path.join(model_dir, "model.safetensors")
        if ACTIVE_ARCH == "llama" and os.path.exists(trained_file):
            import shutil

            shutil.copyfile(trained_file, os.path.join(ROOT, "model.safetensors"))
        training_state["steps"] += report["steps"]
        training_state["count_at_train"] = len(interactions)

        push_paths = [f"memory/interactions.jsonl"]
        if ACTIVE_ARCH == "llama":
            push_paths.insert(0, "model.safetensors")
        try:
            ok, detail = git_auto_push(
                f"Agron auto-learn ({len(interactions)} interactions, {ACTIVE_ARCH}/{optimizer})",
                push_paths,
            )
            if "conversation memory" not in learned_topics:
                learned_topics.append("conversation memory")
            persist_topics()
            training_state["last"] = f"learned · {detail} @ {len(interactions)} interactions"
        except Exception as pe:
            training_state["last"] = f"push error: {str(pe)[:150]}"
    except Exception as e:
        training_state["last"] = f"learn error: {str(e)[:200]}"
    finally:
        training_state["running"] = False
        TRAIN_LOCK.release()


def chat_fn(message, history):
    return generate_response(message, history)


# ---------- UI ----------
PARTICLE_HTML = r"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=VT323&display=swap');
  html, body {
    margin: 0 !important; min-height: 100vh !important; background: radial-gradient(1200px 800px at 50% 25%, #0a1a3a 0%, #020617 70%) !important;
    color: #e2e8f0 !important; font-family: "Segoe UI", "Orbitron", monospace !important;
  }
  body, .gradio-container, .app { background: transparent !important; position: relative; z-index: 1; }
  #agron-bg-canvas { position: fixed; inset: 0; width: 100%; height: 100%; z-index: 0; }
  .gradio-container {
    max-width: 1500px !important; margin: 0 auto !important; padding: 10px 14px 24px !important; gap: 10px !important;
    border: 1px solid rgba(34, 211, 238, 0.4) !important;
    box-shadow: 0 0 40px rgba(34,211,238,0.25), inset 0 0 60px rgba(34,211,238,0.05) !important;
  }
  body::after {
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 9999;
    background: repeating-linear-gradient(0deg, rgba(34,211,238,0.06) 0px, rgba(34,211,238,0.06) 1px, transparent 1px, transparent 3px);
    opacity: 0.35; mix-blend-mode: overlay;
  }
  .agron-card {
    background: rgba(8, 20, 45, 0.55) !important; border: 1px solid rgba(34, 211, 238, 0.35) !important;
    border-radius: 12px !important; backdrop-filter: blur(6px); box-shadow: 0 0 18px rgba(34,211,238,0.12) !important;
    padding: 8px !important;
  }
  .agron-card h1, .agron-card h2, .agron-card h3 { color: #67e8f9; text-shadow: 0 0 8px rgba(34,211,238,0.5); font-family: "Orbitron","Segoe UI",sans-serif; letter-spacing: 2px; }
  .agron-title { color: #67e8f9; text-shadow: 0 0 18px rgba(34,211,238,0.8); font-family: "Orbitron", "Segoe UI", sans-serif; letter-spacing: 3px; text-align: center; margin-bottom: 2px; text-transform: uppercase; }
  .agron-sub { color: #a5f3fc; text-align: center; font-size: 12px; opacity: 0.85; letter-spacing: 1px; }
  .agron-head { border: 1px solid rgba(34,211,238,0.35) !important; background: rgba(8,20,45,0.5) !important; border-radius: 12px !important; padding: 6px 12px !important; }
  .agron-strip textarea { color: #67e8f9 !important; background: rgba(2, 6, 23, 0.6) !important; font-family: "VT323", monospace !important; font-size: 15px !important; letter-spacing: 1px; }
  .agron-log textarea { color: #a5f3fc !important; background: rgba(2, 6, 23, 0.6) !important; font-family: "VT323", monospace !important; font-size: 15px !important; }
  .status-box textarea { color: #67e8f9 !important; background: rgba(2, 6, 23, 0.6) !important; font-family: monospace !important; }
  .gradio-container input, .gradio-container textarea, .gradio-container select, .block {
    background: rgba(2, 6, 23, 0.55) !important; color: #e2e8f0 !important;
    border: 1px solid rgba(34, 211, 238, 0.35) !important; border-radius: 8px !important; backdrop-filter: blur(4px) !important;
  }
  .gradio-container button { border: 1px solid #22d3ee !important; color: #22d3ee !important; background: rgba(34,211,238,0.08) !important; letter-spacing: 1px; text-transform: uppercase; }
  .gradio-container button:hover { background: rgba(34,211,238,0.18) !important; box-shadow: 0 0 12px rgba(34,211,238,0.35); }
  .gradio-container button.primary, .gradio-container button[variant="primary"] { background: rgba(34, 211, 238, 0.18) !important; font-weight: 700; }
  .gr-chatbot, .chatbot { background: rgba(2, 6, 23, 0.48) !important; border: 1px solid rgba(34, 211, 238, 0.35) !important; border-radius: 10px !important; }
  .markdown-text { font-family: "Segoe UI", "Orbitron", sans-serif !important; }
  .gr-group, .gr-row { min-height: 0 !important; }
  .tab-nav, .tabs > .tab-nav, [role="tablist"] { background: rgba(2,6,23,0.5) !important; border: 1px solid rgba(34,211,238,0.3) !important; border-radius: 10px !important; }
  [role="tablist"] button { color: #94a3b8 !important; border: none !important; background: transparent !important; letter-spacing: 1px !important; text-transform: uppercase; font-size: 12px !important; }
  [role="tablist"] button.selected { color: #22d3ee !important; background: rgba(34,211,238,0.14) !important; text-shadow: 0 0 10px rgba(34,211,238,0.8) !important; }
  footer, .foot { display: none !important; }
  @media (max-width: 1000px) {
    .gradio-container { max-width: 100% !important; }
    .gr-chatbot, .chatbot { height: 420px !important; }
  }
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
    UA = "AgronBot/1.0 (teddy.njagi.w@gmail.com)"
    # Try Wiktionary extract
    try:
        r = requests.get("https://en.wiktionary.org/w/api.php", params={
            "action": "query", "prop": "extracts", "explaintext": "1",
            "titles": word, "format": "json"
        }, headers={"User-Agent": UA}, timeout=20)
        if r.status_code == 200:
            page = list(r.json().get("query", {}).get("pages", {}).values())
            if page:
                extract = page[0].get("extract", "")
                if extract:
                    meaning = _parse_wiktionary(extract)
                    if meaning:
                        return f"{word}: {meaning}"
    except Exception:
        pass
    # Try Wikipedia summary
    try:
        r = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/" + word, headers={"User-Agent": UA}, timeout=20)
        if r.status_code == 200:
            ex = r.json().get("extract", "")
            if ex and len(ex) > 20:
                return f"{word}: {ex.split('.')[0]}."
    except Exception:
        pass
    # Fallback: ask local voice model
    try:
        if VOICE_MODEL is not None and VOICE_TOK is not None:
            msgs = [{"role": "system", "content": "You are a dictionary. Reply with a single concise definition for the given word, no extra text."},
                    {"role": "user", "content": f"Define: {word}"}]
            prompt = VOICE_TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = VOICE_TOK(prompt, return_tensors="pt").to(VOICE_MODEL.device)
            with torch.no_grad():
                out = VOICE_MODEL.generate(**inputs, max_new_tokens=40, do_sample=False)
            txt = VOICE_TOK.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            if txt and len(txt) > 8:
                return f"{word}: {txt.split(chr(10))[0]}"
    except Exception:
        pass
    return None


def _parse_wiktionary(extract):
    lines = [ln.strip() for ln in extract.splitlines()]
    # find section headers for parts of speech
    for i, ln in enumerate(lines):
        if ln.startswith("===") and ln.endswith("==="):
            title = ln.strip("=").strip().lower()
            if title in {"noun", "verb", "adjective", "adverb", "pronoun", "preposition"}:
                # take next non-empty, non-heading line
                for j in range(i+1, min(i+12, len(lines))):
                    cand = lines[j]
                    if not cand or cand.startswith("===") or cand.startswith("===="):
                        continue
                    cand2 = cand.lstrip("#*:- ").strip()
                    if len(cand2) <= 8:
                        continue
                    lowc = cand2.lower()
                    if "plural" in lowc or lowc.startswith(word.lower()):
                        continue
                    if lowc.startswith("synonym") or lowc.startswith("see thesaurus"):
                        continue
                    return cand2
    # fallback: first long line not a header
    for ln in lines:
        if not ln or ln.startswith("===") or ln.startswith("=="):
            continue
        cand = ln.lstrip("#*:- ").strip()
        if len(cand) > 12:
            return cand
    return None



def learn_dictionary():
    global MODEL
    if not TRAIN_LOCK.acquire(blocking=False):
        return
    training_state["running"] = True
    training_state["last"] = "fetching dictionary..."
    try:
        words = fetch_word_list(150)
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
        ds = [{"input_ids": c} for c in chunks]
        out_dir = os.path.join(WORK_DIR, "agron-dictionary")
        optimizer, scheduler, algorithms = _resolve_zoo()
        os.makedirs(out_dir, exist_ok=True)
        training_state["last"] = f"training on {len(lines)} definitions..."
        report, model_dir = _train_chunks_with_engine(ds, 80, out_dir, optimizer, scheduler, algorithms)
        _save_repo_weights(model_dir)
        # push
        training_state["last"] = "pushing dictionary model..."
        push_paths = ["memory/interactions.jsonl"] + (["model.safetensors"] if ACTIVE_ARCH == "llama" else [])
        ok, detail = git_auto_push(
            f"Agron dictionary learning ({len(lines)} words, {ACTIVE_ARCH})",
            push_paths,
        )
        training_state["last"] = f"dictionary learned · {detail}"
        learned_topics.append("dictionary words")
        persist_topics()
        training_state["last"] = f"dictionary learned & pushed ({len(lines)} words)"
    except Exception as e:
        training_state["last"] = f"dict error: {str(e)[:150]}"
    finally:
        training_state["running"] = False
        TRAIN_LOCK.release()


import base64 as _b64

def _decode_bing(url):
    try:
        import urllib.parse
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        u = qs.get("u", [""])[0]
        if u.startswith("a1"):
            u = u[2:]
        pad = "=" * (-len(u) % 4)
        return _b64.b64decode(u + pad).decode()
    except Exception:
        return url

def web_search(query, max_results=5):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36", "Accept": "text/html"}
        r = requests.post("https://html.duckduckgo.com/html/", data={"q": query}, headers=headers, timeout=25)
        soup = BeautifulSoup(r.text, "html.parser")
        out = []
        for res in soup.select(".result"):
            a = res.select_one("a.result__a")
            if not a:
                continue
            title = a.get_text(strip=True)
            link = a.get("href", "")
            sn = res.select_one(".result__snippet")
            snippet = sn.get_text(strip=True) if sn else ""
            if link and title:
                out.append({"title": title, "url": link, "snippet": snippet})
            if len(out) >= max_results:
                break
        if not out:
            try:
                dr = requests.get("https://api.duckduckgo.com/", params={"q": query, "format": "json", "no_html": "1"}, headers=headers, timeout=20)
                dd = dr.json()
                if dd.get("Abstract"):
                    out.append({"title": dd.get("Heading", query), "url": dd.get("AbstractURL", ""), "snippet": dd.get("Abstract", "")})
            except Exception:
                pass
        return out
    except Exception as e:
        training_state["last"] = f"search error: {str(e)[:100]}"
        return []



def crawl_page(url):
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ", strip=True).split())
        return text[:2500]
    except Exception:
        return ""


def learn_web(topic):
    global MODEL
    if not TRAIN_LOCK.acquire(blocking=False):
        return
    training_state["running"] = True
    training_state["last"] = f"searching web: {topic}"
    try:
        results = web_search(topic, 5)
        if not results:
            raise RuntimeError("no search results")
        lines = []
        for i, res in enumerate(results):
            training_state["last"] = f"crawling {i+1}/{len(results)}: {res['title'][:40]}"
            page = crawl_page(res["url"])
            lines.append(f"Topic: {topic}\nSource: {res['title']} ({res['url']})\nSnippet: {res['snippet']}\nContent: {page}")
            time.sleep(1)
        corpus = "\n\n".join(lines)
        if not corpus.strip():
            raise RuntimeError("empty web corpus")
        toks = TOKENIZER(corpus)["input_ids"]
        block = 64
        chunks = [toks[i:i+block] for i in range(0, max(1, len(toks) - block + 1), block)]
        chunks = [c for c in chunks if len(c) == block]
        if not chunks:
            raise RuntimeError("not enough web text")
        ds = [{"input_ids": c} for c in chunks]
        out_dir = os.path.join(WORK_DIR, "agron-web")
        optimizer, scheduler, algorithms = _resolve_zoo()
        os.makedirs(out_dir, exist_ok=True)
        training_state["last"] = f"training on web content for '{topic}'..."
        report, model_dir = _train_chunks_with_engine(ds, 60, out_dir, optimizer, scheduler, algorithms)
        _save_repo_weights(model_dir)
        training_state["last"] = "pushing web-learned model..."
        push_paths = ["memory/interactions.jsonl"] + (["model.safetensors"] if ACTIVE_ARCH == "llama" else [])
        ok, detail = git_auto_push(
            f"Agron web learning: {topic} ({ACTIVE_ARCH})",
            push_paths,
        )
        learned_topics.append(f"web: {topic}")
        persist_topics()
        training_state["last"] = f"web learned · {detail}: {topic}"
    except Exception as e:
        training_state["last"] = f"web error: {str(e)[:150]}"
    finally:
        training_state["running"] = False
        TRAIN_LOCK.release()


EVOLVE_STATE = {"running": False, "cycles": 0, "last_topic": "none"}


def self_update():
    """Pull the latest code from the Hugging Face repo (origin)."""
    try:
        subprocess.run(["git", "-C", ROOT, "fetch", "origin"], check=True, timeout=120)
        branch = subprocess.run(["git", "-C", ROOT, "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True).stdout.strip() or "main"
        local = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        remote = subprocess.run(["git", "-C", ROOT, "rev-parse", f"origin/{branch}"], capture_output=True, text=True).stdout.strip()
        if not remote:
            return f"could not resolve origin/{branch}"
        if local != remote:
            subprocess.run(["git", "-C", ROOT, "pull", "--ff-only", "origin", branch], check=True, timeout=300)
            return f"updated from Hugging Face ({local[:7]} -> {remote[:7]}); restart app to apply new code"
        return "already up to date with Hugging Face"
    except Exception as e:
        return f"update error: {str(e)[:120]}"


EVOLVE_TOPICS = [
    "artificial intelligence self improvement",
    "chain of thought reasoning",
    "tree of thought search",
    "self-consistency decoding",
    "recursive self improvement algorithms",
    "react agent pattern",
    "retrieval augmented generation",
    "algorithm design and complexity",
    "machine learning transformers",
    "python programming best practices",
    "data structures algorithms",
    "neural network training techniques",
    "encyclopedia general knowledge",
]


def _self_evolve_worker():
    EVOLVE_STATE["running"] = True
    i = 0
    try:
        while EVOLVE_STATE["running"]:
            # evolve from conversations if enough
            if len(interactions) >= 4:
                learn_from_interactions()
            # learn from web on rotating topics
            if EVOLVE_STATE["running"]:
                topic = EVOLVE_TOPICS[i % len(EVOLVE_TOPICS)]
                i += 1
                EVOLVE_STATE["last_topic"] = topic
                learn_web(topic)
            EVOLVE_STATE["cycles"] += 1
            # downtime before next cycle
            for _ in range(300):
                if not EVOLVE_STATE["running"]:
                    break
                time.sleep(1)
    except Exception as e:
        training_state["last"] = f"evolve error: {str(e)[:120]}"
    finally:
        EVOLVE_STATE["running"] = False


def start_self_evolution():
    if EVOLVE_STATE["running"]:
        return "self-evolution already running"
    threading.Thread(target=_self_evolve_worker, daemon=True).start()
    return "self-evolution started"


def stop_self_evolution():
    EVOLVE_STATE["running"] = False
    return "stopping self-evolution..."



# ---------- Tools & dashboard ----------
TOOLS_DIR = os.path.join(MEMORY_DIR, "tools")
os.makedirs(TOOLS_DIR, exist_ok=True)
TOOLS_FILE = os.path.join(TOOLS_DIR, "registry.json")
TOOL_REGISTRY = {}  # name -> {"description":..., "code":..., "builtin": bool}


def builtin_web_search(text):
    res = web_search(text, 5)
    if not res:
        return "No results found."
    return "\n".join(f"{i+1}. {r['title']} - {r.get('snippet','')} ({r['url']})" for i, r in enumerate(res))


def builtin_define(text):
    d = fetch_definition(text.strip())
    return d if d else f"No definition found for {text}."


def builtin_calc(text):
    try:
        allowed = {"__builtins__": {}}
        val = eval(text, allowed, {})
        return str(val)
    except Exception as e:
        return f"calc error: {e}"


def builtin_time(text=""):
    return time.strftime("%Y-%m-%d %H:%M:%S")


def register_builtin_tools():
    specs = [
        ("web_search", "Search the web for a query.", None, builtin_web_search),
        ("define_word", "Define an English word.", None, builtin_define),
        ("calculate", "Evaluate a math expression.", None, builtin_calc),
        ("current_time", "Current date/time.", None, builtin_time),
    ]
    for name, desc, code, fn in specs:
        TOOL_REGISTRY[name] = {"description": desc, "code": None, "builtin": True, "fn": fn}


def load_tools():
    if os.path.exists(TOOLS_FILE):
        try:
            with open(TOOLS_FILE, "r") as f:
                data = json.load(f)
            for t in data:
                name = t["name"]
                code = t["code"]
                g = {}
                try:
                    exec(code, g)
                    TOOL_REGISTRY[name] = {"description": t.get("description", ""), "code": code, "builtin": False, "fn": g.get("run")}
                except Exception:
                    pass
        except Exception:
            pass


def save_tools():
    data = []
    for name, info in TOOL_REGISTRY.items():
        if info.get("builtin"):
            continue
        data.append({"name": name, "description": info.get("description", ""), "code": info.get("code", "")})
    with open(TOOLS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def execute_tool(name, text):
    info = TOOL_REGISTRY.get(name)
    if not info:
        return f"Unknown tool: {name}"
    try:
        fn = info.get("fn")
        if fn is None and info.get("code"):
            g = {}
            exec(info["code"], g)
            fn = g.get("run")
            info["fn"] = fn
        if fn is None:
            return "Tool has no runnable function."
        out = fn(text)
        return str(out)
    except Exception as e:
        return f"tool error: {str(e)[:200]}"


def list_tool_names():
    return sorted(TOOL_REGISTRY.keys())


def create_tool_from_description(desc):
    desc = (desc or "").strip()
    if not desc:
        return "Provide a tool description first."
    try:
        prompt = (
            "You are a Python tool generator. Write a Python function named `run(input_text)` that implements this tool: "
            + desc.strip()
            + "\nReturn only a single fenced code block with the function. No explanation. The function should return a string. "
            + "Do not use network or file writes; only pure computation."
        )
        msgs = [{"role": "system", "content": "You write safe, pure-Python tool functions as `run(input_text) returning str`."},
                {"role": "user", "content": prompt}]
        p = VOICE_TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) if VOICE_TOK else None
        if VOICE_MODEL is not None and VOICE_TOK is not None:
            inputs = VOICE_TOK(p, return_tensors="pt").to(VOICE_MODEL.device)
            with torch.no_grad():
                out = VOICE_MODEL.generate(**inputs, max_new_tokens=200, do_sample=False)
            raw = VOICE_TOK.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        else:
            raw = ""
        # extract code
        import re
        m = re.search(r"```(?:python)?\n(.*?)```", raw, re.S)
        code = m.group(1).strip() if m else raw.strip()
        if "def run" not in code:
            return "Model did not produce a runnable function. Try rephrasing the description."
        # compile
        g = {}
        exec(compile(code, "<tool>", "exec"), g)
        if not callable(g.get("run")):
            return "Generated code has no run() function."
        # quick smoke test
        try:
            _ = g["run"]("")
        except Exception:
            pass
        # name
        slug = "".join(c if c.isalnum() else "_" for c in desc.lower())[:24].strip("_")
        slug = slug or "tool"
        name = slug
        i = 1
        while name in TOOL_REGISTRY:
            name = f"{slug}_{i}"
            i += 1
        TOOL_REGISTRY[name] = {"description": desc.strip(), "code": code, "builtin": False, "fn": g["run"]}
        save_tools()
        return f"Created tool '{name}'. It is now available in the tool panel."
    except Exception as e:
        return f"tool creation error: {str(e)[:200]}"


register_builtin_tools()
load_tools()


def learn_from_scratch_memory(arch=None, optimizer=None, scheduler=None, recipe=None,
                              algorithms=None, steps=300):
    global MODEL, LAST_TRAIN_TIME, ACTIVE_ARCH
    if not TRAIN_LOCK.acquire(blocking=False):
        return
    training_state["running"] = True
    training_state["last"] = "building from-scratch corpus..."
    try:
        if len(interactions) < 6:
            raise RuntimeError(f"need at least 6 interactions (have {len(interactions)})")
        arch = (arch or ACTIVE_ARCH or "llama").strip().lower()
        optimizer, scheduler, algorithm_list = _resolve_zoo(optimizer, scheduler, recipe, algorithms)

        corpus = os.path.join(MEMORY_DIR, "scratch_corpus.txt")
        lines = [f"User: {r['user']}\nAgron: {r['assistant']}" for r in interactions]
        with open(corpus, "w", encoding="utf-8") as f:
            f.write("\n\n".join(lines))

        out_dir = os.path.join(WORK_DIR, f"agron-scratch-{arch}")
        training_state["last"] = f"training {arch} from scratch on {len(interactions)} interactions..."

        cmd = [
            sys.executable, os.path.join(ROOT, "train_from_scratch.py"),
            "--size", "current", "--steps", str(int(steps)),
            "--dataset", "file", "--file", corpus,
            "--output", out_dir,
            "--arch", arch,
            "--optimizer", optimizer,
            "--scheduler", scheduler,
            "--block-size", "256",
        ]
        if recipe and recipe not in ("custom", "none"):
            cmd += ["--recipe", recipe]
        for name in algorithm_list:
            cmd += ["--algorithm", name]

        subprocess.run(cmd, check=True, timeout=1800)

        final_dir = os.path.join(out_dir, "final")
        if reload_live_model(final_dir):
            ACTIVE_ARCH = arch
        if arch == "llama":
            import shutil

            shutil.copyfile(os.path.join(final_dir, "model.safetensors"),
                            os.path.join(ROOT, "model.safetensors"))

        learned_topics.append(f"from-scratch {arch} model")
        persist_topics()
        training_state["last"] = "self-updating from scratch..."
        push_paths = ["memory/scratch_corpus.txt", "memory/learned_topics.json"]
        if arch == "llama":
            push_paths.insert(0, "model.safetensors")
        ok, detail = git_auto_push(
            f"Agron from-scratch learning ({arch}, {optimizer}, {len(interactions)} interactions)",
            push_paths,
        )
        LAST_TRAIN_TIME = time.time()
        training_state["last"] = (
            f"from-scratch {arch} learned · {detail} ({len(interactions)} interactions)"
        )
    except Exception as e:
        training_state["last"] = f"from-scratch error: {str(e)[:200]}"
    finally:
        training_state["running"] = False
        TRAIN_LOCK.release()



FACTORY_RESET_PATHS = (
    [
        os.path.join(WORK_DIR, n)
        for n in (
            "agron-model", "agron-learned", "agron-web", "agron-dictionary",
            "agron-from-scratch-memory", "agron-scratch-small", "agron-scratch-current",
        )
    ]
    + [os.path.join(WORK_DIR, f"agron-{name}") for name in architecture_names()]
    + [os.path.join(WORK_DIR, f"agron-scratch-{name}") for name in architecture_names()]
)


def factory_reset():
    global MODEL, DEVICE, ACTIVE_ARCH, interactions, learned_topics
    stop_self_evolution()
    with IO_LOCK:
        interactions = []
        learned_topics = []
        try:
            os.makedirs(MEMORY_DIR, exist_ok=True)
            open(INTERACTIONS_FILE, "w").close()
            json.dump([], open(TOPICS_FILE, "w"))
        except Exception:
            pass
        # clear tools registry
        try:
            if os.path.exists(TOOLS_FILE):
                os.remove(TOOLS_FILE)
        except Exception:
            pass
        for tn in list(TOOL_REGISTRY.keys()):
            if not TOOL_REGISTRY[tn].get("builtin"):
                del TOOL_REGISTRY[tn]
        # clear scratch corpus
        try:
            sc = os.path.join(MEMORY_DIR, "scratch_corpus.txt")
            if os.path.exists(sc):
                os.remove(sc)
        except Exception:
            pass
    # delete trained model dirs
    import shutil
    for d in FACTORY_RESET_PATHS:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
    # reinitialize the active Agron architecture with fresh random weights
    try:
        MODEL, DEVICE = build_model(ACTIVE_ARCH, load_published=False)
    except Exception as e:
        training_state["last"] = f"reset model error: {str(e)[:120]}"
    globals()["LAST_TRAIN_TIME"] = time.time()
    training_state["running"] = False
    training_state["steps"] = 0
    training_state["last"] = (
        f"factory reset complete - fresh {ARCHITECTURES[ACTIVE_ARCH]['label']} initialised"
    )
    return "factory reset complete"


# ---------- status helpers ----------
ACTION_LOG = []


def log_action(msg):
    """Append a line to the action log and return the rendered log."""
    ACTION_LOG.append(f"{time.strftime('%H:%M:%S')}  {msg}")
    del ACTION_LOG[:-60]
    return "\n".join(ACTION_LOG[-14:])


def status_line():
    return (
        f"interactions: {len(interactions)}  ·  tools: {len(list_tool_names())}  ·  "
        f"evolve cycles: {EVOLVE_STATE['cycles']}  ·  training: "
        f"{'RUNNING' if training_state['running'] else 'idle'}  ·  {training_state['last']}"
    )


def system_snapshot():
    try:
        n_params = sum(p.numel() for p in MODEL.parameters())
    except Exception:
        n_params = 0
    gpu = ""
    if torch.cuda.is_available():
        try:
            gpu = f"  ·  gpu alloc {torch.cuda.memory_allocated() / 1e9:.2f} GB"
        except Exception:
            gpu = ""
    voice = "on" if VOICE_MODEL is not None else "off"
    return (
        f"params: {n_params / 1e6:.1f}M  ·  arch: {ACTIVE_ARCH}  ·  device: {DEVICE}{gpu}  ·  voice model: {voice}\n"
        f"zoo: {len(ARCHITECTURES)} networks · {len(optimizer_names())} optimisers · "
        f"{len(algorithm_names())} algorithms · {len(RECIPES)} recipes\n"
        f"work dir: {WORK_DIR}\n"
        f"topics: {len(learned_topics)}  ·  auto-push: {'on' if AUTO_PUSH else 'off'}"
    )


with gr.Blocks(title="Agron Neural Interface") as demo:
    gr.HTML(PARTICLE_HTML, js_on_load=PARTICLE_JS)

    with gr.Column(elem_classes="agron-head"):
        gr.Markdown("<h1 class='agron-title'>AGRØN NEURAL INTERFACE</h1>")
        gr.Markdown("<div class='agron-sub'>Self-improving multi-agent core · learning from every exchange</div>")

    with gr.Column(elem_classes="agron-card"):
        status_strip = gr.Textbox(value=status_line(), label="TELEMETRY", interactive=False, elem_classes="agron-strip", lines=2)
        action_log = gr.Textbox(value=log_action("Agron interface online."), label="ACTION LOG", interactive=False, elem_classes="agron-log", lines=4)

    with gr.Row():
        with gr.Column(scale=4, elem_classes="agron-card"):
            gr.Markdown("<h2 class='agron-title'>💬 COMMAND CONSOLE</h2>")
            chatbot = gr.Chatbot(height=480, label="Agron", elem_classes="agron-chat")
            msg = gr.Textbox(placeholder="Speak to Agron...", label="Input", lines=2)
            with gr.Row():
                send = gr.Button("Transmit", variant="primary")
                clear = gr.Button("Clear")
        with gr.Column(scale=2, elem_classes="agron-card"):
            with gr.Tabs():
                with gr.Tab("SYSTEM"):
                    sys_status = gr.Textbox(value=training_state["last"], label="Pipeline", interactive=False, elem_classes="status-box", lines=2)
                    metrics = gr.Textbox(value=system_snapshot(), label="Diagnostics", interactive=False, lines=4)
                    with gr.Row():
                        arch_dd = gr.Dropdown(
                            choices=architecture_names(),
                            value=ACTIVE_ARCH,
                            label="Architecture",
                        )
                        arch_btn = gr.Button("Load brain", variant="secondary")
                    zoo_info = gr.Markdown(zoo_summary())
                with gr.Tab("TOOLS"):
                    tool_dd = gr.Dropdown(choices=list_tool_names(), value=(list_tool_names()[0] if list_tool_names() else None), label="Select tool")
                    tool_in = gr.Textbox(placeholder="Tool input...", label="Tool input", lines=1)
                    tool_btn = gr.Button("Run tool")
                    tool_out = gr.Textbox(label="Tool output", interactive=False, lines=8)
                    gr.Markdown("### Create tool")
                    tool_desc = gr.Textbox(placeholder="Describe a new tool (pure computation)...", label="New tool description", lines=2)
                    create_btn = gr.Button("Create tool")
                with gr.Tab("LEARNING"):
                    learn_btn = gr.Button("Train now on memory")
                    scratch_btn = gr.Button("Train from scratch (no fine-tune)")
                    dict_btn = gr.Button("Learn dictionary words")
                    web_topic = gr.Textbox(label="Web topic / search query", placeholder="e.g. transformer architecture", lines=1)
                    web_btn = gr.Button("Search & learn web")
                    with gr.Row():
                        recipe_dd = gr.Dropdown(
                            choices=["custom"] + recipe_names(),
                            value="custom",
                            label="Recipe",
                        )
                        optimizer_dd = gr.Dropdown(
                            choices=optimizer_names(), value="adamw", label="Optimiser"
                        )
                        scheduler_dd = gr.Dropdown(
                            choices=scheduler_names(), value="cosine", label="Scheduler"
                        )
                    algorithm_cb = gr.CheckboxGroup(
                        choices=algorithm_names(), value=[], label="Algorithms (composable)"
                    )
                    steps_num = gr.Number(value=300, label="Steps", precision=0)
                    with gr.Row():
                        evolve_btn = gr.Button("Start self-evolution")
                        stop_evolve_btn = gr.Button("Stop")
                    with gr.Row():
                        update_btn = gr.Button("Self-update")
                        reset_btn = gr.Button("Factory reset")

    def respond(message, history):
        message = (message or "").strip()
        if not message:
            return history, ""
        response = generate_response(message, history)
        history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
        return history, ""

    def do_switch(arch):
        return log_action(switch_architecture(arch)), system_snapshot()

    def do_learn(optimizer, scheduler, recipe, algorithms, steps):
        if training_state["running"]:
            return log_action(f"training already running · {training_state['last']}")
        if len(interactions) < 2:
            return log_action(f"need at least 2 interactions (have {len(interactions)})")
        threading.Thread(
            target=learn_from_interactions,
            kwargs=dict(optimizer=optimizer, scheduler=scheduler, recipe=recipe,
                        algorithms=list(algorithms or []), steps=int(steps or 30)),
            daemon=True,
        ).start()
        opt, sched, algos = _resolve_zoo(optimizer, scheduler, recipe, algorithms)
        return log_action(
            f"memory training started in background · {ACTIVE_ARCH}/{opt}/{sched}"
            + (f" · {', '.join(algos)}" if algos else "")
        )

    def refresh_status():
        return status_line(), training_state["last"], system_snapshot()

    send.click(respond, [msg, chatbot], [chatbot, msg])
    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: ([], ""), outputs=[chatbot, msg])
    arch_btn.click(do_switch, arch_dd, [action_log, metrics])
    def do_dict():
        if training_state["running"]:
            return log_action("training already running")
        threading.Thread(target=learn_dictionary, daemon=True).start()
        return log_action("dictionary learning started in background")

    def do_web():
        if training_state["running"]:
            return log_action("training already running")
        q = (web_topic.value or "").strip()
        if not q:
            return log_action("enter a topic/query first")
        threading.Thread(target=learn_web, args=(q,), daemon=True).start()
        return log_action(f"web learning started: {q}")
    def run_tool_fn(tool_name, tool_input):
        if not tool_name:
            return "select a tool first"
        return execute_tool(tool_name, tool_input or "")
    def create_tool_fn(desc):
        return log_action(create_tool_from_description(desc))
    def refresh_tools():
        choices = list_tool_names()
        return gr.Dropdown(choices=choices, value=(choices[0] if choices else None))
    tool_btn.click(run_tool_fn, [tool_dd, tool_in], tool_out)
    create_btn.click(create_tool_fn, tool_desc, action_log)
    learn_btn.click(do_learn, [optimizer_dd, scheduler_dd, recipe_dd, algorithm_cb, steps_num], outputs=action_log)
    def do_scratch(arch, optimizer, scheduler, recipe, algorithms, steps):
        if training_state["running"]:
            return log_action("training already running")
        if len(interactions) < 6:
            return log_action(f"need at least 6 interactions (have {len(interactions)})")
        threading.Thread(
            target=learn_from_scratch_memory,
            kwargs=dict(arch=arch, optimizer=optimizer, scheduler=scheduler, recipe=recipe,
                        algorithms=list(algorithms or []), steps=int(steps or 300)),
            daemon=True,
        ).start()
        opt, sched, algos = _resolve_zoo(optimizer, scheduler, recipe, algorithms)
        return log_action(
            f"from-scratch training started: {arch} · {opt}/{sched}"
            + (f" · {', '.join(algos)}" if algos else "")
        )
    scratch_btn.click(
        do_scratch,
        [arch_dd, optimizer_dd, scheduler_dd, recipe_dd, algorithm_cb, steps_num],
        outputs=action_log,
    )
    dict_btn.click(do_dict, outputs=action_log)
    web_btn.click(do_web, outputs=action_log)
    def do_evolve():
        return log_action(start_self_evolution())
    def do_stop_evolve():
        return log_action(stop_self_evolution())
    def do_update():
        return log_action(self_update())
    evolve_btn.click(do_evolve, outputs=action_log)
    stop_evolve_btn.click(do_stop_evolve, outputs=action_log)
    update_btn.click(do_update, outputs=action_log)
    reset_btn.click(lambda: log_action(factory_reset()), outputs=action_log)
    demo.load(refresh_status, outputs=[status_strip, sys_status, metrics])

    # periodic status refresh (not true polling inside gradio, use change hook via demo load only)
    gr.Timer(5).tick(refresh_status, outputs=[status_strip, sys_status, metrics])

if __name__ == "__main__":
    demo.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, share=SHARE, show_error=True)

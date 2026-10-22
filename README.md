---
language:
- en
license: mit
tags:
- agent
- multi-agent
- autonomous
- self-improving
- recursive
- learn-from-scratch
- huggingface
pipeline_tag: text-generation
---

# Agron — Self-Improving Multi-Agent AI

**Agron** is an autonomous, self-improving AI agent system that **learns from scratch**.
It starts with empty memory and bootstrap rules, then recursively improves through:

1. Observation of task outcomes
2. Memory consolidation and knowledge graph growth
3. Tool/strategy evolution and validation
4. Safe self-modification with tests
5. Optional synchronization to Hugging Face

## Agent Team

| Agent | Role | Specialty |
|---|---|---|
| Coordinator | Orchestrator | Decomposes work, delegates, synthesizes |
| Learner | Knowledge | Ingests, distills, updates memory from scratch |
| Builder | Improvement | Generates new tools/strategies recursively |
| Guardian | Safety | Validates changes, prevents regressions |

## Learn From Scratch

Agron begins with:
- Empty episodic memory
- No imported knowledge
- Seed evaluation rules only
- Bootstrap loop enabled

First cycles prioritize:
- Environment probing
- Capability discovery
- Minimal safe tool creation
- Curriculum expansion
- Test-backed improvement

## Recursive Self-Improvement

- XP / generation tracking
- Gap analysis against current capabilities
- Candidate generation with validation
- Test-gated commits
- Misalignment/secret checks before push
- Rollback on failed validation

## Hugging Face Sync

Agron can push validated improvements to:
`https://huggingface.co/teddyoas/Agron`

## Trained LLM

Agron's initial LLM has now been trained **from scratch** on WikiText-2 using an 8-layer, 512-dim, 4-head LLaMA-style architecture (about **50.3M parameters**). Training artifacts and weights are committed in this repository, including `model.safetensors`, tokenizer files, and the checkpoints.

Future recursive self-improvement can fine-tune or grow this base model.

## Interface

The Gradio app is a HUD-style neural interface:

- **Command Console** — the chat surface, backed by the base model plus an optional
  instruction-tuned voice model (`AGRON_VOICE_MODEL`, on by default, ~6 GB)
- **Telemetry strip** — live interaction, tool and training counters
- **Action log** — timestamped results of every command, kept separate from the
  telemetry strip so it is never overwritten by the status refresh
- **SYSTEM / TOOLS / LEARNING tabs** — diagnostics, the tool registry (run and
  create), and all learning controls

The page scrolls normally, so every control stays reachable on small viewports.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `AGRON_VOICE_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | Instruction model used for chat |
| `AGRON_USE_VOICE` | `1` | Set to `0` to run on the native base model only |
| `AGRON_AUTO_PUSH` | `1` | Set to `0` to disable auto-push of learned weights |
| `AGRON_WORK_DIR` | `/content` | Scratch dir for training artifacts |
| `AGRON_PORT` / `AGRON_HOST` | `7860` / `0.0.0.0` | Bind address |
| `AGRON_SHARE` | `0` | Set to `1` for a public gradio.live share link |

## Cybersecurity Focus

Agron and its spawned agents are constrained to **cybersecurity exploitation and analysis only**:

- Exploitation (authorized targets, CVEs, payloads)
- Vulnerability analysis and research
- Reverse engineering and malware analysis
- Network / host / application reconnaissance
- Defensive hardening and detection engineering

Agents can be **spawned in parallel** for tasks using the built-in multi-agent orchestration.

## LLM + Agent Spawning

Agron is an **LLM** at its core, not only an agent system. Its model weights are trained and published in this repository, and the runtime can use the model to power (or seed) specialized **cybersecurity agents** that are spawned in parallel for tasks:

- ExploitAgent
- AnalysisAgent
- ReconAgent
- VulnAgent
- DefenseAgent

Agents can be created on demand for task decomposition, then run concurrently, while the backbone remains the Agron LLM.

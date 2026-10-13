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

Agron's initial LLM has now been trained **from scratch** on WikiText-2 using a 4-layer, 256-dim, 4-head LLaMA-style architecture (about **8.4M parameters**). Training artifacts and weights are committed in this repository, including `model.safetensors`, tokenizer files, and the checkpoint.

Future recursive self-improvement can fine-tune or grow this base model.

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

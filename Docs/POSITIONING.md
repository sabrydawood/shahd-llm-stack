# Positioning — What Shahd Is, and What Exists Around It

## The category

**Shahd is a from-scratch, fully-owned LLM stack in TypeScript** — not "an AI model."

The distinction matters. A *model* is a file of weights; judged as a model, any small
CPU-trained network loses instantly to anything served from a datacenter. A *stack* is the
machinery that produces, serves, and improves models: data collection → tokenizer →
training → instruction tuning → agent serving. Shahd ships models (see the releases), but
the models are the **proof that the stack works**, not the product's ceiling. Every layer
of that machinery is implemented in this repository and owned outright:

- Tensor + reverse-mode autograd engine (gradcheck-verified) — not TensorFlow.js
- Native SIMD compute kernels (Go/C, AVX2/FMA, F64 and F32 paths) behind a pluggable backend seam
- Byte-level BPE tokenizer with special-token handling
- A data foundry: licensed multi-source collection, tiering, dedup, per-kind tables, a control panel
- Pretraining and SFT (chat template, loss masking, tool-use + thinking recipes, multi-turn stitching)
- Sequence-parallel training over a worker pool with deterministic gradient reduction
- Self-describing checkpoints (weights + optimizer + RNG + tokenizer + config) with exact resume
- An agent serving loop: tool calls, a think-scratchpad, and a reasoning trace that streams live and is persisted per reply

## The landscape (verified July 2026)

### In JavaScript/TypeScript

| Project | What it is | Owns the math? | Trains? | Beyond pretraining? |
|---------|-----------|----------------|---------|---------------------|
| [transformers.js](https://github.com/huggingface/transformers.js) | Run pretrained ONNX models in JS | — (ONNX runtime) | ❌ inference only | ❌ |
| [WebLLM](https://github.com/mlc-ai/web-llm) | WebGPU in-browser inference | — (MLC/TVM) | ❌ inference only | ❌ |
| [homemade-gpt-js](https://github.com/trekhleb/homemade-gpt-js) | minGPT re-implemented in ~300 lines of TS | ❌ TensorFlow.js kernels | ✅ toy scale | ❌ |
| [micrograd-ts](https://github.com/trekhleb/micrograd-ts) | Educational scalar autograd (~200 lines) | ✅ but scalar-valued | ❌ can't train a real transformer | ❌ |
| **Shahd** | Full lifecycle: data → train → SFT → agent serving | ✅ tensor-grade + native kernels | ✅ pretrain + SFT + resume | ✅ tools, thinking, multi-turn, trace, releases |

The pattern: the popular JS AI projects are **inference-only** consumers of models trained
elsewhere; the few that train ride on TensorFlow.js (Google's C++/WASM/WebGL kernels — the
math is not owned); the owned-autograd projects are scalar educational toys.

### Spiritual relatives outside TypeScript

- [femtoGPT](https://github.com/keyvank/femtoGPT) (Rust) — the closest philosophical
  cousin: GPT from scratch with an owned tensor library. Stops at pretraining; no SFT,
  tools, agent loop, or serving stack.
- [minGPT](https://github.com/karpathy/minGPT) / nanoGPT /
  [microgpt](https://karpathy.github.io/2026/02/12/microgpt/) (Python) — the canonical
  educational trainers. Python, PyTorch (or pure-Python at toy scale), pretraining-focused.
- llm.c (C) — GPT-2 training in raw C/CUDA. Kernels-and-loop, not a lifecycle.

No public project in any language combines the full owned lifecycle in one codebase the
way integrated lab-internal stacks do — and in TypeScript specifically, the field is
effectively empty.

## Why the TypeScript field is empty

Performance economics. Python owns the training world because CUDA-backed kernels are
free to pick up; anyone wanting JS either serves ONNX (inference-only) or sits on
TensorFlow.js and gives up owning the math. Shahd takes the third path — TypeScript for
the entire system, with its own native Go/C SIMD kernels reached over FFI for the hot
loops. That is the deliberate trade that makes the stack ownable end-to-end.

## Naming guidance

- **Use:** "a from-scratch, fully-owned LLM stack in TypeScript" — or short: *an owned
  LLM stack in TypeScript*. For the shipped weights: "checkpoints" or "models" plural,
  named by tier (MicroChat, NanoChat…).
- **Avoid:** "an AI model" (invites a losing size comparison and undersells every layer
  except the weights), "an LLM" (the L overclaims parameter count), "a framework" (it is
  not a library you import — it is a complete, opinionated system).

One-line description for the repository and release pages:

> **Shahd — a from-scratch, fully-owned LLM stack in TypeScript: data engine, tokenizer,
> training, SFT, and agent serving with a visible reasoning trace. The released models are
> the proof it works end-to-end.**

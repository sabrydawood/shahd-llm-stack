# Model Scaling Reference

A practical reference for model sizes: how many parameters, how much data, and what hardware each
needs to **train** and to **run** (inference). Two parts: the in-repo model family (small, CPU-to-modest-GPU)
and a real-world reference for large production-scale models. Figures for large models are
order-of-magnitude references under stated assumptions, not exact guarantees.

Three things are always separate — never one file:

- **Data** — the training corpus (input). Lives in storage; read in mini-batches during training.
- **Training** — the compute process that turns data into weights. Runs on CPU/GPU; owns neither.
- **Model (checkpoint)** — the trained weights (output). Loaded once to run.

---

## 1. The in-repo model family

Parameter counts verified by building each config: `params ≈ (Vocab × Embed) + 16 × Layers × Embed²`.
Head dimension is 64 (standard) except the two smallest tiers. **The columns match the dashboard Train
panel exactly**, so each row is a ready preset. `Batch` and `Steps` are the Chinchilla ~20-tokens/param
**floor** (`steps = tokens ÷ (batch × context)`) — a lower bound; real training uses several times
more. `Corpus MB` is a practical starting size — we currently have ~60 MB collected, so the larger
tiers need much more data (see §3).

| Tier | Embed | Layers | Heads | Context | Vocab | Batch | Steps | Corpus MB | Params | Trains on |
|------|-------|--------|-------|---------|-------|-------|-------|-----------|--------|-----------|
| Seed | 96 | 3 | 4 | 96 | 512 | 16 | ~6,000 | 2 | 0.49M | CPU |
| Nano | 128 | 4 | 4 | 256 | 512 | 16 | ~5,000 | 3 | 1.1M | CPU |
| Micro | 256 | 6 | 4 | 512 | 1,024 | 16 | ~16,000 | 8 | 6.6M | CPU (slow) |
| Mini | 512 | 8 | 8 | 1,024 | 4,096 | 32 | ~22,000 | 30 | 36M | small GPU |
| Small | 768 | 12 | 12 | 2,048 | 16,384 | 64 | ~19,000 | 80 | 126M | 8–12 GB GPU |
| Base | 1,024 | 24 | 16 | 4,096 | 32,000 | 128 | ~17,000 | 200 | 435M | 16–24 GB GPU |
| Large | 2,048 | 32 | 32 | 8,192 | 50,000 | 256 | ~22,000 | 500 | 2.25B | 40–80 GB GPU |

Rough equivalents by size: Small ≈ GPT-2 small (124M), Base ≈ GPT-2 medium/large, Large ≈ a small
modern LLM.

Training memory ≈ 4× the weights (weights + gradients + optimizer m/v), plus activations. The current
compute path is Float64 (8 bytes), which **doubles** memory; large tiers require switching to Float32 /
mixed precision (`Compute.Precision: "F32"`) and a GPU backend.

### Steps vs tokens

`Steps` is not intrinsic to a model — it is `tokens ÷ (batch × context)`, and the real measure of "how
much training" is tokens seen (`steps × batch × context`). The table's Steps are the Chinchilla minimum
(~20 tokens/param) at each tier's batch. The step counts stay modest for the big tiers only because
their batch is large — at a tiny CPU batch the same token budget needs 10–20× more steps, which is
exactly why the larger tiers need a GPU: big batches cut the step count and each step runs far faster.

---

## 2. From a base model to a chat model (SFT)

A base (pretrained) model only **autocompletes**. Making it one that **replies and follows a
conversation** is a second stage — **SFT (instruction/chat tuning)** — plus optional tool-use and
thinking on top. Three stages, each reading a different **data kind** (kept in separate tables):

| Stage | What it adds | Data kind(s) | How (dashboard) |
|-------|--------------|--------------|-----------------|
| Pretrain | language + code patterns | code (+ knowledge) | Train ▸ Mode **Pretrain** |
| **SFT (chat)** | reply in the chat format, call tools, think, then stop | **conversation** (+ code) | Train ▸ Mode **Chat / SFT** |
| RL (optional) | prefer better answers | conversation | rejection sampling |

**The single most important input for "talks well" is conversation data** (OASST/OASST2 dialogue),
used in the SFT stage. More + more diverse dialogue → better conversational behavior, up to the
model's scale ceiling.

### Chat-model recipe by tier

| Chat tier | Base tier | SFT steps | Conversation examples | Realistically expect |
|-----------|-----------|-----------|-----------------------|----------------------|
| Seed-chat | Seed/Nano | ~500–800 | 1k–5k | learns the FORMAT (replies + stops); output mostly incoherent |
| Micro-chat | Micro | ~2k–4k | 10k–50k | short on-topic replies on seen patterns; frequent errors |
| Mini-chat | Mini | ~8k–15k | 50k–200k | simple coherent Q&A + tool calls; not fluent |
| Small-chat | Small | ~20k+ | 200k–1M | basic assistant on narrow tasks (GPT-2-class); needs a GPU |
| *fluent + senior-level code* | *7B+* | *100k+* | *millions* | *emergent at scale — not reachable from scratch on modest hardware* |

### The data mix (per kind), set from the dashboard

Data types live in separate tables, so a run picks how much of each:

- **Pretrain**: `Code MB` (documents_code) + `Knowledge MB` (documents_knowledge). Code-only for a
  code base; add knowledge for general language.
- **Chat (SFT)**: `Conversations` (documents_conversation) + `Code samples` (documents_code). Set
  Code samples to 0 for a pure-chat model; set Conversations to 0 for a pure-code assistant.

### Honest ceiling

At the tiers that run on modest hardware (Seed–Mini), SFT teaches the **format + the tool/thinking
mechanism** — the model replies, stops, and can call tools — but it will **not** be fluent or write
senior-level code; that is emergent at billions of parameters + trillions of tokens (see §3). The
architecture is complete, so under-training is acceptable; to improve conversation within the ceiling,
collect **more + more diverse conversation data** and raise SFT steps + `Conversations`.

---

## 3. Real-world large models

Inference weight memory is exact math: `bytes-per-parameter × parameters`. FP16 = 2 B, INT8 = 1 B,
INT4 ≈ 0.5 B. Training data uses the Chinchilla compute-optimal rule (~20 tokens/parameter); modern
models are often "over-trained" on far more (Llama-2 used ~2T tokens at every size; Llama-3.1 405B used
~15T). Raw text is roughly ~4 bytes/token, but is usually curated from 10–100× more crawled data.

| Params | Train tokens (opt. → modern) | Weights FP16 | Weights INT4 | Inference GPU (FP16) | Training hardware (reference) |
|--------|------------------------------|--------------|--------------|----------------------|-------------------------------|
| **1.5B** | 30B → ~0.3–1T | 3 GB | ~1 GB | 1× 8 GB | 1× A100, days |
| **7B** | 140B → 1–15T | 14 GB | ~4 GB | 1× 16–24 GB | ~256× A100 · ~30 d (~184k A100-hrs) |
| **13B** | 260B → ~2T | 26 GB | ~7 GB | 1× 48 GB or 2× 24 GB | ~2× the 7B cost (~369k A100-hrs) |
| **34B** | 680B → ~2T | 68 GB | ~17 GB | 2× 48 GB or 1× 80 GB | ~5–8× the 7B cost |
| **70B** | 1.4T → 2–15T | 140 GB | ~35 GB | 2× 80 GB | ~2000× A100 · ~35 d (~1.7M A100-hrs) |
| **175B** | 3.5T → ~0.3–1T | 350 GB | ~88 GB | 8× 80 GB | ~1000s of A100 · weeks · ~$5–12M |
| **405B** | 8.1T → ~15T | 810 GB | ~200 GB | 8–16× 80 GB | ~16k× H100 · ~2 months |
| **1T+** | ~20T+ | ~2 TB | ~500 GB | 16–32× 80 GB (MoE lowers active cost) | large clusters · months |

**Training needs far more hardware than inference.** Mixed-precision training holds ~16–18 bytes per
parameter (weights + grads + Adam states + master copy), so a 7B model needs ~120 GB just for state —
sharded across GPUs (ZeRO / FSDP) — even though its FP16 weights (14 GB) run inference on one card.
4-bit quantization collapses the inference bar: a 7B runs on a laptop, a 70B on ~2× 24 GB.

---

## 4. Context length and the KV-cache

Long context (256k, 1M) is mostly a **KV-cache memory** problem, not a weights problem. During
generation the model caches keys+values for every token:

```
KV-cache bytes ≈ 2 (K,V) × Layers × KVdim × Context × bytes-per-value
```

Example for a 7B-class model (32 layers, hidden 4096, FP16). With standard multi-head attention
(KVdim = 4096) it is ~0.5 MB per token:

| Context | Full MHA (KVdim 4096) | GQA (KVdim 1024) | GQA + INT8 KV |
|---------|-----------------------|------------------|---------------|
| 8k | 4 GB | 1 GB | 0.5 GB |
| 32k | 16 GB | 4 GB | 2 GB |
| 128k | 64 GB | 16 GB | 8 GB |
| 256k | 128 GB | 32 GB | 16 GB |
| 1M | 512 GB | 128 GB | 64 GB |

This is why long-context models rely on **GQA** (fewer KV heads), **KV-cache quantization**, **paged
attention**, and **sliding-window / sparse attention** — otherwise a single 1M-token request would need
more memory than the model weights themselves.

---

## 5. Rules of thumb

```
Weights (inference)   = bytes-per-param × params          (FP16 = 2 B, INT4 ≈ 0.5 B)
Training state memory ≈ 16–18 bytes × params              (weights + grads + Adam + master)
Compute-optimal data  ≈ 20 tokens per parameter           (Chinchilla; modern models use much more)
Tokens seen           = steps × batch × context           (steps = tokens ÷ (batch × context))
Capacity (smartness)  = width (embed) × depth (layers)    → needs GPU beyond ~100M params
Long-context cost      = KV-cache, not weights             → mitigated by GQA + quantization
```

Practical path for this project: stay small on CPU (Seed–Micro), move to a GPU for Mini and up, and
switch weights + training to Float32 / mixed precision before scaling past ~100M parameters.

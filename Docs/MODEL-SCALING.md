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
Head dimension is 64 (standard) except the two smallest tiers.

| Tier | Embed | Layers | Heads | Context | Vocab | Params | Trains on | Comparable to |
|------|-------|--------|-------|---------|-------|--------|-----------|---------------|
| Seed | 96 | 3 | 4 | 96 | 512 | 0.49M | CPU (minutes) | toy |
| Nano | 128 | 4 | 4 | 256 | 512 | 1.1M | CPU (minutes) | experiment |
| Micro | 256 | 6 | 4 | 512 | 1,024 | 6.6M | CPU (~1 hr) | tiny |
| Mini | 512 | 8 | 8 | 1,024 | 4,096 | 36M | small GPU | GPT-2 nano |
| Small | 768 | 12 | 12 | 2,048 | 16,384 | 126M | 8–12 GB GPU | ≈ GPT-2 small (124M) |
| Base | 1,024 | 24 | 16 | 4,096 | 32,000 | 435M | 16–24 GB GPU | ≈ GPT-2 medium/large |
| Large | 2,048 | 32 | 32 | 8,192 | 50,000 | 2.25B | 40–80 GB GPU | ≈ a small modern LLM |

Training memory ≈ 4× the weights (weights + gradients + optimizer m/v), plus activations. The current
compute path is Float64 (8 bytes), which **doubles** memory; large tiers require switching to Float32 /
mixed precision (`Compute.Precision: "F32"`) and a GPU backend.

---

## 2. Real-world large models

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

## 3. Context length and the KV-cache

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

## 4. Rules of thumb

```
Weights (inference)   = bytes-per-param × params          (FP16 = 2 B, INT4 ≈ 0.5 B)
Training state memory ≈ 16–18 bytes × params              (weights + grads + Adam + master)
Compute-optimal data  ≈ 20 tokens per parameter           (Chinchilla; modern models use much more)
Capacity (smartness)  = width (embed) × depth (layers)    → needs GPU beyond ~100M params
Training amount        = steps × corpus                    → needs time (and compute)
Long-context cost      = KV-cache, not weights             → mitigated by GQA + quantization
```

Practical path for this project: stay small on CPU (Seed–Micro), move to a GPU for Mini and up, and
switch weights + training to Float32 / mixed precision before scaling past ~100M parameters.

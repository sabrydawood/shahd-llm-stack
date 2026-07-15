# Shahd (شهد)

A **from-scratch, fully-owned, code-specialized language model** written in TypeScript (Bun) with
Go compute kernels. No open-source model weights, no fine-tuning of someone else's brain — the
autograd engine, transformer, tokenizer, training loop, safety, tools, and reasoning are all built
here and are meant to be evolved by their owner. Helper libraries (Zod, etc.) are fine; the *model*
is owned.

> **Status (honest):** the engine is complete and gradcheck-verified (96 tests green). The shipped
> models are tiny (0.2M–1.1M params) trained on toy corpora — real capability needs scale (params ×
> data × compute), which the roadmap below builds toward. See [Docs/ROADMAP.md](Docs/ROADMAP.md).

## Quickstart

```bash
bun install
bun run ci          # typecheck + length + naming + lint + gradcheck + all tests
bun test            # tests only
bun run gradcheck   # finite-difference gradient check (the numerical oracle)

bun run train       # train a char-level model on a built-in code sample, then sample
bun run sample      # sample from a saved checkpoint
bun run Scripts/TrainOnCorpus.ts     # train on the permissive seed corpus
bun run Scripts/PhaseSevenDemo.ts    # capstone: corpus -> tools agent -> speculative -> safety
bun run Scripts/ComputeSpike.ts 512  # benchmark TS vs Go FFI matmul

bun run foundry:dashboard            # control plane: collect data -> train -> chat (http://localhost:8090)
```

## Architecture (layered, `Brain/`)

| Layer                     | What it does                                                                                              |
| ------------------------- | --------------------------------------------------------------------------------------------------------- |
| `Config/`               | Zod-validated config;`Constants.ts` is the single source of defaults; derive + hash + freeze            |
| `Tensor/` `Autograd/` | Float64 tensors + reverse-mode autograd (topological backward);`GradCheck` is the oracle                |
| `Ops/`                  | Core differentiable ops (matmul, softmax, norms, RoPE, SwiGLU, cross-entropy…)                           |
| `Nn/`                   | `Shahd` transformer: embeddings, multi-head attention (+RoPE/GQA), MLP, blocks, weight tying            |
| `Optim/`                | Adam/AdamW, grad clipping, LR schedule                                                                    |
| `Tokenizer/`            | Char + byte-level BPE (no-OOV), code-aware pretokenization, special tokens                                |
| `Data/`                 | Corpus pipeline: license allowlist, MinHash dedup, quality filter, decontamination, FIM,`CorpusBuilder` |
| `Training/`             | Training step, gradient accumulation, eval loop (bits-per-byte), train loop                               |
| `Sampling/`             | Temperature/top-k/top-p/min-p sampling, generation,**KV-cache** (numerically exact)                 |
| `Checkpoint/`           | Self-describing checkpoints (weights + optimizer + RNG + config + hash)                                   |
| `Safety/`               | Controllable content-safety filter + resource limits +`GuardedGenerate` (absolute priority)             |
| `Sft/`                  | Chat template + loss masking, task taxonomy, tool-use exemplars                                           |
| `Eval/` `Rl/`         | pass@k, sandboxed code executor, RLVR rejection sampling                                                  |
| `Serving/`              | Tool protocol,**rich tool system** (`Tools/`), agent loop, OpenAI-compatible server               |
| `Reasoning/`            | Speculative decoding, self-consistency, tree-of-thoughts, thinking-mode                                   |
| `ComputeBackend/`       | Pluggable numeric backend seam: TS, Go subprocess, Go FFI (in-process, 2–8× on CPU)                     |

## Conventions (enforced by CI)

1. **PascalCase** everything we declare (`check:naming` + ESLint).
2. **≤ 600 lines** per file (`check:length`).
3. **DRY-strict** — no duplicated function/constant; check before writing.
4. **Central constants**, config validated by Zod in one place.
5. **Acyclic imports**, explicit named barrels (no `index.ts`).

Full rules: [Docs/CONVENTIONS.md](Docs/CONVENTIONS.md). Design: [Docs/ARCHITECTURE.md](Docs/ARCHITECTURE.md).
File-by-file layout: [Docs/STRUCTURE.md](Docs/STRUCTURE.md). Where it's going: [Docs/ROADMAP.md](Docs/ROADMAP.md).

## Data

The **Data Foundry** is the tiered, inspectable dataset layer and the collection engine that fills it:
multiple licensed sources (GitHub code, OASST/Stack Exchange dialogue, Wikipedia knowledge, GSM8K
reasoning, local folders of books), each routed to its own per-kind table, with honest dedup
accounting and resumable collection. Drive it from the dashboard (`bun run foundry:dashboard`).
See [Docs/DATA-FOUNDRY.md](Docs/DATA-FOUNDRY.md).

## Safety

Safety and performance are absolute priorities living in dedicated, controllable places
(`Brain/Safety/`, `Config.Safety`, `Config.Limits`, `Config.Tools`). The model must never assist harm;
the content filter and capability gates can be strengthened or disabled centrally via config.

## Community

Want to take part — ideas, data sources, testing, or just following along? Join the Discord:
**<https://discord.gg/v4ACAn5CKf>**

## License

Proprietary — © Sabry. All rights reserved. See [LICENSE](LICENSE).

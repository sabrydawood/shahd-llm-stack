// Incremental (KV-cached) forward + generation — the inference-only twin of the training forward
// (deliberately a separate path, not a flag on the training forward). It processes ONE token row
// in plain numerics (tape off), attending over the growing KvCache instead of recomputing past
// K/V. Verified numerically identical to the Tensor forward by CachedForward.Test.ts.

import type { Shahd } from "../Nn/Shahd.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import type { NumArray } from "../Tensor/Tensor.ts";
import type { SamplingOptions } from "./Sampler.ts";
import { KvCache } from "./KvCache.ts";
import { SampleFromLogits } from "./Sampler.ts";
import { WithTapeOff } from "../Tensor/Tape.ts";

// Weight buffers arrive in the run's storage precision (NumArray); the row scratch and outputs
// stay f64 — inference math in doubles is free accuracy.
function MatMulRow(X: Float64Array, W: NumArray, K: number, N: number): Float64Array {
  const Out = new Float64Array(N);
  for (let Kk = 0; Kk < K; Kk++) {
    const Xk = X[Kk];
    if (Xk === 0) continue;
    const Base = Kk * N;
    for (let Nn = 0; Nn < N; Nn++) Out[Nn] += Xk * W[Base + Nn];
  }
  return Out;
}

function LayerNormRow(X: Float64Array, Gamma: NumArray, Beta: NumArray, Eps: number, N: number): Float64Array {
  let Mu = 0;
  for (let J = 0; J < N; J++) Mu += X[J];
  Mu /= N;
  let Var = 0;
  for (let J = 0; J < N; J++) {
    const D = X[J] - Mu;
    Var += D * D;
  }
  Var /= N;
  const Is = 1 / Math.sqrt(Var + Eps);
  const Out = new Float64Array(N);
  for (let J = 0; J < N; J++) Out[J] = Gamma[J] * ((X[J] - Mu) * Is) + Beta[J];
  return Out;
}

/** Process one token at absolute Position, updating the cache; returns next-token logits [VocabSize]. */
export function CachedForwardStep(Model: Shahd, Cache: KvCache, TokenId: number, Position: number): Float64Array {
  const Cfg = Model.Config;
  if (
    Cfg.Model.PositionScheme !== "Learned" ||
    Cfg.Model.NormKind !== "LayerNorm" ||
    Cfg.Model.MlpKind !== "Relu" ||
    Cfg.Derived.KvHeads !== Cfg.Model.NumHeads
  ) {
    throw new Error(
      "CachedForwardStep: the KV-cache path currently supports only the Learned/LayerNorm/Relu/MHA " +
        "architecture; use the uncached Generate for the modern (RoPE/RMSNorm/SwiGLU/GQA) stack.",
    );
  }
  const E = Cfg.Model.EmbedDim;
  const NumHeads = Cfg.Model.NumHeads;
  const HeadDim = Cfg.Derived.HeadDim;
  const Scale = Cfg.Derived.AttentionScale;
  const Wte = Model.Embedding.Wte.Data;
  const Wpe = Model.Embedding.Wpe!.Data; // non-null: guard above ensures learned positions

  const X = new Float64Array(E);
  for (let J = 0; J < E; J++) X[J] = Wte[TokenId * E + J] + Wpe[Position * E + J];

  for (let L = 0; L < Model.Blocks.length; L++) {
    const Block = Model.Blocks[L];
    const Attn = Block.Attn;
    const Xn = LayerNormRow(X, Block.Ln1.Gamma.Data, Block.Ln1.Beta!.Data, Block.Ln1.Eps, E);

    const AttnOut = new Float64Array(E);
    for (let Hd = 0; Hd < NumHeads; Hd++) {
      const Q = MatMulRow(Xn, Attn.WqHeads[Hd].Data, E, HeadDim);
      const K = MatMulRow(Xn, Attn.WkHeads[Hd].Data, E, HeadDim);
      const V = MatMulRow(Xn, Attn.WvHeads[Hd].Data, E, HeadDim);
      Cache.Append(L, Hd, K, V);

      const CachedK = Cache.GetKeys(L, Hd);
      const CachedV = Cache.GetValues(L, Hd);
      const T = CachedK.length / HeadDim;
      const Scores = new Float64Array(T);
      let Max = -Infinity;
      for (let Ti = 0; Ti < T; Ti++) {
        let Dot = 0;
        const KBase = Ti * HeadDim;
        for (let D = 0; D < HeadDim; D++) Dot += Q[D] * CachedK[KBase + D];
        const Scaled = Dot * Scale;
        Scores[Ti] = Scaled;
        if (Scaled > Max) Max = Scaled;
      }
      let Sum = 0;
      for (let Ti = 0; Ti < T; Ti++) {
        const Ex = Math.exp(Scores[Ti] - Max);
        Scores[Ti] = Ex;
        Sum += Ex;
      }
      const Ctx = new Float64Array(HeadDim);
      for (let Ti = 0; Ti < T; Ti++) {
        const Weight = Scores[Ti] / Sum;
        const VBase = Ti * HeadDim;
        for (let D = 0; D < HeadDim; D++) Ctx[D] += Weight * CachedV[VBase + D];
      }
      const Proj = MatMulRow(Ctx, Attn.WoHeads[Hd].Data, HeadDim, E);
      for (let J = 0; J < E; J++) AttnOut[J] += Proj[J];
    }
    for (let J = 0; J < E; J++) X[J] += AttnOut[J]; // residual

    const Xn2 = LayerNormRow(X, Block.Ln2.Gamma.Data, Block.Ln2.Beta!.Data, Block.Ln2.Eps, E);
    const Hidden = Cfg.Derived.MlpHidden;
    const H1 = MatMulRow(Xn2, Block.Mlp.Wfc!.Data, E, Hidden);
    const Bfc = Block.Mlp.Bfc!.Data;
    for (let J = 0; J < Hidden; J++) {
      const Val = H1[J] + Bfc[J];
      H1[J] = Val > 0 ? Val : 0; // ReLU
    }
    const H2 = MatMulRow(H1, Block.Mlp.Wproj!.Data, Hidden, E);
    const Bproj = Block.Mlp.Bproj!.Data;
    for (let J = 0; J < E; J++) X[J] += H2[J] + Bproj[J]; // residual + bias
  }

  const Xf = LayerNormRow(X, Model.LnFinal.Gamma.Data, Model.LnFinal.Beta!.Data, Model.LnFinal.Eps, E);

  const VocabSize = Cfg.Model.VocabSize;
  const Bias = Model.LmHeadBias.Data;
  const Logits = new Float64Array(VocabSize);
  if (Model.WeightTying) {
    // Head = transpose(Wte): logits[v] = sum_j Xf[j] * Wte[v*E + j]
    for (let Vi = 0; Vi < VocabSize; Vi++) {
      let S = 0;
      const WBase = Vi * E;
      for (let J = 0; J < E; J++) S += Xf[J] * Wte[WBase + J];
      Logits[Vi] = S + Bias[Vi];
    }
  } else {
    if (Model.LmHead === null) throw new Error("CachedForwardStep: untied model missing LmHead");
    const Head = Model.LmHead.Data; // [E, VocabSize]
    for (let Vi = 0; Vi < VocabSize; Vi++) {
      let S = 0;
      for (let J = 0; J < E; J++) S += Xf[J] * Head[J * VocabSize + Vi];
      Logits[Vi] = S + Bias[Vi];
    }
  }
  return Logits;
}

/** KV-cached autoregressive generation. Valid up to BlockSize tokens (learned absolute positions). */
export function CachedGenerate(
  Model: Shahd,
  PromptIds: number[],
  MaxNewTokens: number,
  Options: SamplingOptions,
  Rng: SeededRng,
): number[] {
  return WithTapeOff(() => {
    const BlockSize = Model.Config.Model.BlockSize;
    const VocabSize = Model.Config.Model.VocabSize;
    // Mirrors the empty-prompt guard below: an oversized prompt must fail loudly too, not silently
    // `break` on the first generation step and hand back PromptIds unchanged with zero new tokens.
    if (PromptIds.length >= BlockSize) {
      throw new Error(
        `CachedGenerate: prompt length ${PromptIds.length} already fills or exceeds BlockSize ${BlockSize}; no new tokens can be generated`,
      );
    }
    const Cache = new KvCache(Model.Blocks.length, Model.Config.Model.NumHeads, Model.Config.Derived.HeadDim, BlockSize);
    const Ids = [...PromptIds];

    let LastLogits: Float64Array | null = null;
    for (let P = 0; P < Ids.length && P < BlockSize; P++) {
      LastLogits = CachedForwardStep(Model, Cache, Ids[P], P);
    }
    for (let S = 0; S < MaxNewTokens; S++) {
      if (LastLogits === null) throw new Error("CachedGenerate: empty prompt");
      const Position = Ids.length;
      if (Position >= BlockSize) break; // absolute position table exhausted (see KvCache note)
      const Next = SampleFromLogits(LastLogits, 0, VocabSize, Options, Rng);
      Ids.push(Next);
      LastLogits = CachedForwardStep(Model, Cache, Next, Position);
    }
    return Ids;
  });
}

// Causal multi-head self-attention via PER-HEAD weight matrices whose projected outputs are
// SUMMED (equivalent to concat-then-Wo, verified in REVIEW.md), reusing the 2D ops with NO new
// slice/concat primitives. Supports Grouped-Query Attention: NumHeads query heads share KvHeads
// key/value heads (GroupSize = NumHeads/KvHeads query heads per KV head). MHA when KvHeads =
// NumHeads. Attention scale comes from DeriveConfig (1/sqrt(HeadDim)); RoPE optional.

import type { Tensor } from "../Tensor/Tensor.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import { MatMul, Transpose, Scale, CausalMask, SoftmaxRows, Add, ApplyRope } from "../Ops/OpsBarrel.ts";
import { InitWeight } from "./InitPolicy.ts";

export class MultiHeadAttention {
  WqHeads: Tensor[] = []; // [NumHeads] of [EmbedDim, HeadDim]
  WoHeads: Tensor[] = []; // [NumHeads] of [HeadDim, EmbedDim] (residual projection)
  WkHeads: Tensor[] = []; // [KvHeads] of [EmbedDim, HeadDim] (shared across a query group)
  WvHeads: Tensor[] = []; // [KvHeads] of [EmbedDim, HeadDim]
  NumHeads: number;
  KvHeads: number;
  GroupSize: number;
  AttentionScale: number;
  UseRope: boolean;

  constructor(Config: ResolvedConfig, Rng: SeededRng) {
    const { EmbedDim, NumHeads } = Config.Model;
    const { HeadDim, AttentionScale, KvHeads } = Config.Derived;
    this.NumHeads = NumHeads;
    this.KvHeads = KvHeads;
    this.GroupSize = NumHeads / KvHeads;
    this.AttentionScale = AttentionScale;
    this.UseRope = Config.Model.PositionScheme === "Rope";
    for (let H = 0; H < NumHeads; H++) {
      this.WqHeads.push(InitWeight(EmbedDim, HeadDim, Rng, Config));
      this.WoHeads.push(InitWeight(HeadDim, EmbedDim, Rng, Config, true)); // scaled residual init
    }
    for (let Kv = 0; Kv < KvHeads; Kv++) {
      this.WkHeads.push(InitWeight(EmbedDim, HeadDim, Rng, Config));
      this.WvHeads.push(InitWeight(EmbedDim, HeadDim, Rng, Config));
    }
  }

  Forward(X: Tensor): Tensor {
    // Project each KV head once (shared by its query group), and transpose K ONCE here — under GQA the
    // same K feeds GroupSize query heads, so re-transposing it inside the per-head loop was redundant
    // work (and redundant autograd nodes) that partly cancelled GQA's own compute saving. The shared
    // transpose node correctly accumulates gradient from every query head in the group.
    const KsT: Tensor[] = [];
    const Vs: Tensor[] = [];
    for (let Kv = 0; Kv < this.KvHeads; Kv++) {
      let K = MatMul(X, this.WkHeads[Kv]);
      if (this.UseRope) K = ApplyRope(K, 0);
      KsT.push(Transpose(K));
      Vs.push(MatMul(X, this.WvHeads[Kv]));
    }

    let Output: Tensor | null = null;
    for (let H = 0; H < this.NumHeads; H++) {
      const KvIndex = Math.floor(H / this.GroupSize);
      let Q = MatMul(X, this.WqHeads[H]); // [T, HeadDim]
      if (this.UseRope) Q = ApplyRope(Q, 0);
      let Scores = MatMul(Q, KsT[KvIndex]); // [T, T] — reuse the shared transpose
      Scores = Scale(Scores, this.AttentionScale);
      Scores = CausalMask(Scores);
      const Weights = SoftmaxRows(Scores);
      const HeadOut = MatMul(Weights, Vs[KvIndex]); // [T, HeadDim]
      const Projected = MatMul(HeadOut, this.WoHeads[H]); // [T, EmbedDim]
      Output = Output === null ? Projected : Add(Output, Projected);
    }
    if (Output === null) throw new Error("MultiHeadAttention: NumHeads must be >= 1");
    return Output;
  }

  Parameters(): Tensor[] {
    return [...this.WqHeads, ...this.WkHeads, ...this.WvHeads, ...this.WoHeads];
  }
}

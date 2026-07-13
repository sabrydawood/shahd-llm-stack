// Causal multi-head self-attention via PER-HEAD weight matrices whose projected outputs are
// SUMMED (mathematically equivalent to concat-then-Wo, verified in REVIEW.md), so it reuses the
// existing 2D ops with NO new slice/concat primitives. The attention scale comes from
// DeriveConfig (1/sqrt(HeadDim)) — never re-computed here (closes L4).

import type { Tensor } from "../Tensor/Tensor.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import { MatMul, Transpose, Scale, CausalMask, SoftmaxRows, Add } from "../Ops/OpsBarrel.ts";
import { InitWeight } from "./InitPolicy.ts";

export class MultiHeadAttention {
  WqHeads: Tensor[] = []; // each [EmbedDim, HeadDim]
  WkHeads: Tensor[] = [];
  WvHeads: Tensor[] = [];
  WoHeads: Tensor[] = []; // each [HeadDim, EmbedDim] (residual projection)
  NumHeads: number;
  AttentionScale: number;

  constructor(Config: ResolvedConfig, Rng: SeededRng) {
    const { EmbedDim, NumHeads } = Config.Model;
    const { HeadDim, AttentionScale } = Config.Derived;
    this.NumHeads = NumHeads;
    this.AttentionScale = AttentionScale;
    for (let H = 0; H < NumHeads; H++) {
      this.WqHeads.push(InitWeight(EmbedDim, HeadDim, Rng, Config));
      this.WkHeads.push(InitWeight(EmbedDim, HeadDim, Rng, Config));
      this.WvHeads.push(InitWeight(EmbedDim, HeadDim, Rng, Config));
      this.WoHeads.push(InitWeight(HeadDim, EmbedDim, Rng, Config, true)); // scaled residual init
    }
  }

  Forward(X: Tensor): Tensor {
    let Output: Tensor | null = null;
    for (let H = 0; H < this.NumHeads; H++) {
      const Q = MatMul(X, this.WqHeads[H]); // [T, HeadDim]
      const K = MatMul(X, this.WkHeads[H]);
      const V = MatMul(X, this.WvHeads[H]);
      let Scores = MatMul(Q, Transpose(K)); // [T, T]
      Scores = Scale(Scores, this.AttentionScale);
      Scores = CausalMask(Scores);
      const Weights = SoftmaxRows(Scores);
      const HeadOut = MatMul(Weights, V); // [T, HeadDim]
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

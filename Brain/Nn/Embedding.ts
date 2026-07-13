// Token embeddings, plus learned absolute position embeddings when PositionScheme is "Learned".
// When "Rope", positions are applied inside attention (RoPE) and there is no Wpe.

import type { Tensor } from "../Tensor/Tensor.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import { Add, EmbeddingLookup } from "../Ops/OpsBarrel.ts";
import { InitWeight } from "./InitPolicy.ts";

export class Embedding {
  Wte: Tensor; // token embedding table [VocabSize, EmbedDim]
  Wpe: Tensor | null; // position table [BlockSize, EmbedDim]; null when using RoPE

  constructor(Config: ResolvedConfig, Rng: SeededRng) {
    const { VocabSize, BlockSize, EmbedDim, PositionScheme } = Config.Model;
    this.Wte = InitWeight(VocabSize, EmbedDim, Rng, Config);
    this.Wpe = PositionScheme === "Learned" ? InitWeight(BlockSize, EmbedDim, Rng, Config) : null;
  }

  Forward(Ids: number[]): Tensor {
    const Tokens = EmbeddingLookup(this.Wte, Ids);
    if (this.Wpe === null) return Tokens;
    const Positions: number[] = [];
    for (let I = 0; I < Ids.length; I++) Positions.push(I);
    return Add(Tokens, EmbeddingLookup(this.Wpe, Positions));
  }

  Parameters(): Tensor[] {
    return this.Wpe === null ? [this.Wte] : [this.Wte, this.Wpe];
  }
}

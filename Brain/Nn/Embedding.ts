// Token + learned absolute position embeddings. Forward maps token ids -> summed token/position
// vectors [T, EmbedDim]. (Learned absolute positions are the simple Phase-1 choice; RoPE is a
// Phase-2 architecture-lock decision per CAPABILITIES.md.)

import type { Tensor } from "../Tensor/Tensor.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import { Add, EmbeddingLookup } from "../Ops/OpsBarrel.ts";
import { InitWeight } from "./InitPolicy.ts";

export class Embedding {
  Wte: Tensor; // token embedding table [VocabSize, EmbedDim]
  Wpe: Tensor; // position embedding table [BlockSize, EmbedDim]

  constructor(Config: ResolvedConfig, Rng: SeededRng) {
    const { VocabSize, BlockSize, EmbedDim } = Config.Model;
    this.Wte = InitWeight(VocabSize, EmbedDim, Rng, Config);
    this.Wpe = InitWeight(BlockSize, EmbedDim, Rng, Config);
  }

  Forward(Ids: number[]): Tensor {
    const Positions: number[] = [];
    for (let I = 0; I < Ids.length; I++) Positions.push(I);
    return Add(EmbeddingLookup(this.Wte, Ids), EmbeddingLookup(this.Wpe, Positions));
  }

  Parameters(): Tensor[] {
    return [this.Wte, this.Wpe];
  }
}

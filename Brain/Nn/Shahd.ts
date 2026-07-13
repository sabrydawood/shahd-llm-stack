// The full Shahd model: token/position embeddings -> N transformer blocks -> final LayerNorm
// -> LM head producing next-token logits [T, VocabSize]. Supports weight tying (LM head =
// transpose of the token embedding) and asserts the context bound (closes the Wpe OOB risk).

import type { Tensor } from "../Tensor/Tensor.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import { MatMul, AddBias, Transpose } from "../Ops/OpsBarrel.ts";
import { Zeros } from "../Tensor/TensorFactories.ts";
import { Embedding } from "./Embedding.ts";
import { Block } from "./Block.ts";
import { LayerNormModule } from "./LayerNormModule.ts";
import { InitWeight } from "./InitPolicy.ts";

export class Shahd {
  Config: ResolvedConfig;
  Embedding: Embedding;
  Blocks: Block[] = [];
  LnFinal: LayerNormModule;
  LmHead: Tensor | null; // null when weights are tied (head = transpose of token embedding)
  LmHeadBias: Tensor; // [1, VocabSize]
  WeightTying: boolean;

  constructor(Config: ResolvedConfig, Rng: SeededRng) {
    const { EmbedDim, VocabSize, NumLayers, LayerNormEps } = Config.Model;
    this.Config = Config;
    this.Embedding = new Embedding(Config, Rng);
    for (let L = 0; L < NumLayers; L++) this.Blocks.push(new Block(Config, Rng));
    this.LnFinal = new LayerNormModule(EmbedDim, LayerNormEps);
    this.WeightTying = Config.Model.WeightTying;
    this.LmHead = this.WeightTying ? null : InitWeight(EmbedDim, VocabSize, Rng, Config);
    this.LmHeadBias = Zeros(1, VocabSize);
  }

  /** ids[T] -> logits[T, VocabSize]. */
  Forward(Ids: number[]): Tensor {
    if (Ids.length > this.Config.Model.BlockSize) {
      throw new Error(
        `Shahd.Forward: sequence length ${Ids.length} exceeds BlockSize ${this.Config.Model.BlockSize}`,
      );
    }
    let X = this.Embedding.Forward(Ids);
    for (const CurrentBlock of this.Blocks) X = CurrentBlock.Forward(X);
    X = this.LnFinal.Forward(X);

    let HeadMatrix: Tensor;
    if (this.WeightTying) {
      HeadMatrix = Transpose(this.Embedding.Wte); // [VocabSize, EmbedDim] -> [EmbedDim, VocabSize]
    } else {
      if (this.LmHead === null) throw new Error("Shahd: LmHead missing on an untied model");
      HeadMatrix = this.LmHead;
    }
    return AddBias(MatMul(X, HeadMatrix), this.LmHeadBias);
  }

  /** All trainable parameters in a stable order (consumed by the optimizer and checkpoints). */
  Parameters(): Tensor[] {
    const Params: Tensor[] = [...this.Embedding.Parameters()];
    for (const CurrentBlock of this.Blocks) Params.push(...CurrentBlock.Parameters());
    Params.push(...this.LnFinal.Parameters());
    if (this.LmHead !== null) Params.push(this.LmHead);
    Params.push(this.LmHeadBias);
    return Params;
  }
}

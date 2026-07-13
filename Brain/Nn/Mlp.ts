// Position-wise feed-forward network: Linear -> ReLU -> Linear. The down-projection (Wproj) is
// a residual-stream projection, so it uses the scaled-residual init.

import type { Tensor } from "../Tensor/Tensor.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import { MatMul, AddBias, ReLU } from "../Ops/OpsBarrel.ts";
import { Zeros } from "../Tensor/TensorFactories.ts";
import { InitWeight } from "./InitPolicy.ts";

export class Mlp {
  Wfc: Tensor; // [EmbedDim, MlpHidden]
  Bfc: Tensor; // [1, MlpHidden]
  Wproj: Tensor; // [MlpHidden, EmbedDim]
  Bproj: Tensor; // [1, EmbedDim]

  constructor(Config: ResolvedConfig, Rng: SeededRng) {
    const { EmbedDim } = Config.Model;
    const Hidden = Config.Derived.MlpHidden;
    this.Wfc = InitWeight(EmbedDim, Hidden, Rng, Config);
    this.Bfc = Zeros(1, Hidden);
    this.Wproj = InitWeight(Hidden, EmbedDim, Rng, Config, true); // scaled residual init
    this.Bproj = Zeros(1, EmbedDim);
  }

  Forward(X: Tensor): Tensor {
    const Hidden = ReLU(AddBias(MatMul(X, this.Wfc), this.Bfc));
    return AddBias(MatMul(Hidden, this.Wproj), this.Bproj);
  }

  Parameters(): Tensor[] {
    return [this.Wfc, this.Bfc, this.Wproj, this.Bproj];
  }
}

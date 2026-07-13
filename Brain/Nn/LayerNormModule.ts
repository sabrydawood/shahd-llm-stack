// A LayerNorm with learnable Gamma/Beta parameters wrapping the pure Ops/LayerNorm op.

import { Tensor } from "../Tensor/Tensor.ts";
import { Filled, Zeros } from "../Tensor/TensorFactories.ts";
import { LayerNorm } from "../Ops/OpsBarrel.ts";

export class LayerNormModule {
  Gamma: Tensor; // [1, Dim], init to 1
  Beta: Tensor; // [1, Dim], init to 0
  Eps: number;

  constructor(Dim: number, Eps: number) {
    this.Gamma = Filled(1, Dim, 1);
    this.Beta = Zeros(1, Dim);
    this.Eps = Eps;
  }

  Forward(X: Tensor): Tensor {
    return LayerNorm(X, this.Gamma, this.Beta, this.Eps);
  }

  Parameters(): Tensor[] {
    return [this.Gamma, this.Beta];
  }
}

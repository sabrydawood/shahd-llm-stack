// Normalization layer that dispatches on Config.Model.NormKind: LayerNorm (gamma + beta) or
// RmsNorm (gamma only). Replaces the old LayerNormModule so the norm is config-selectable.

import type { Tensor } from "../Tensor/Tensor.ts";
import { Filled, Zeros } from "../Tensor/TensorFactories.ts";
import { LayerNorm, RmsNorm } from "../Ops/OpsBarrel.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";

export class NormLayer {
  Gamma: Tensor;
  Beta: Tensor | null; // present only for LayerNorm
  Eps: number;
  Kind: "LayerNorm" | "RmsNorm";

  constructor(Dim: number, Config: ResolvedConfig) {
    this.Kind = Config.Model.NormKind;
    this.Eps = Config.Model.LayerNormEps;
    this.Gamma = Filled(1, Dim, 1);
    this.Beta = this.Kind === "LayerNorm" ? Zeros(1, Dim) : null;
  }

  Forward(X: Tensor): Tensor {
    if (this.Kind === "RmsNorm") return RmsNorm(X, this.Gamma, this.Eps);
    if (this.Beta === null) throw new Error("NormLayer: LayerNorm missing Beta");
    return LayerNorm(X, this.Gamma, this.Beta, this.Eps);
  }

  Parameters(): Tensor[] {
    return this.Beta === null ? [this.Gamma] : [this.Gamma, this.Beta];
  }
}

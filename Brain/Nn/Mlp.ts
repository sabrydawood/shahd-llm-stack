// Position-wise feed-forward, config-selectable by MlpKind:
//   Relu  : Linear -> ReLU -> Linear (with biases) — the Phase-1 default.
//   SwiGlu: (Silu(x@W1) ⊙ (x@W3)) @ W2  — the Llama-family gated MLP.
//   GeGlu : same, with GELU as the gate.
// The down-projection uses the scaled-residual init.

import type { Tensor } from "../Tensor/Tensor.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import { MatMul, AddBias, ReLU, Silu, Gelu, Mul } from "../Ops/OpsBarrel.ts";
import { Zeros } from "../Tensor/TensorFactories.ts";
import { InitWeight } from "./InitPolicy.ts";

export class Mlp {
  Kind: "Relu" | "SwiGlu" | "GeGlu";
  // Relu path
  Wfc: Tensor | null = null;
  Bfc: Tensor | null = null;
  Wproj: Tensor | null = null;
  Bproj: Tensor | null = null;
  // Gated path
  W1: Tensor | null = null; // gate projection
  W3: Tensor | null = null; // up projection
  W2: Tensor | null = null; // down projection

  constructor(Config: ResolvedConfig, Rng: SeededRng) {
    const { EmbedDim } = Config.Model;
    const Hidden = Config.Derived.MlpHidden;
    this.Kind = Config.Model.MlpKind;
    if (this.Kind === "Relu") {
      this.Wfc = InitWeight(EmbedDim, Hidden, Rng, Config);
      this.Bfc = Zeros(1, Hidden);
      this.Wproj = InitWeight(Hidden, EmbedDim, Rng, Config, true);
      this.Bproj = Zeros(1, EmbedDim);
    } else {
      this.W1 = InitWeight(EmbedDim, Hidden, Rng, Config);
      this.W3 = InitWeight(EmbedDim, Hidden, Rng, Config);
      this.W2 = InitWeight(Hidden, EmbedDim, Rng, Config, true);
    }
  }

  Forward(X: Tensor): Tensor {
    if (this.Kind === "Relu") {
      if (this.Wfc === null || this.Bfc === null || this.Wproj === null || this.Bproj === null) {
        throw new Error("Mlp: Relu weights missing");
      }
      const Hidden = ReLU(AddBias(MatMul(X, this.Wfc), this.Bfc));
      return AddBias(MatMul(Hidden, this.Wproj), this.Bproj);
    }
    if (this.W1 === null || this.W3 === null || this.W2 === null) {
      throw new Error("Mlp: gated weights missing");
    }
    const GateRaw = MatMul(X, this.W1);
    const Gate = this.Kind === "SwiGlu" ? Silu(GateRaw) : Gelu(GateRaw);
    const Up = MatMul(X, this.W3);
    return MatMul(Mul(Gate, Up), this.W2);
  }

  Parameters(): Tensor[] {
    const Params: Tensor[] = [];
    for (const T of [this.Wfc, this.Bfc, this.Wproj, this.Bproj, this.W1, this.W3, this.W2]) {
      if (T !== null) Params.push(T);
    }
    return Params;
  }
}

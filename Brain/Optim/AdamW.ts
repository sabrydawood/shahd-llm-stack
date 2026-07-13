// AdamW = Adam with DECOUPLED weight decay (Loshchilov & Hutter). Reuses Adam's moment math via
// inheritance (rule #4: no duplication) and only adds the decay step: theta -= Lr*WeightDecay*theta,
// applied before the Adam update. Decay is applied to weight MATRICES only (Rows>1), not to
// biases / LayerNorm gamma/beta ([1,N]) — the standard no-decay-on-1D-params practice.

import { Adam } from "./Adam.ts";
import type { Tensor } from "../Tensor/Tensor.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";

export class AdamW extends Adam {
  WeightDecay: number;

  constructor(Params: Tensor[], Config: ResolvedConfig) {
    super(Params, Config);
    this.WeightDecay = Config.Optimizer.WeightDecay;
  }

  override Step(Lr: number): void {
    for (const P of this.Params) {
      if (P.Rows > 1) {
        const Decay = Lr * this.WeightDecay;
        for (let I = 0; I < P.Size; I++) P.Data[I] -= Decay * P.Data[I];
      }
    }
    super.Step(Lr);
  }
}

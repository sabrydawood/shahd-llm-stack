// Adam optimizer (Kingma & Ba, 2014), bias-corrected, with eps OUTSIDE the sqrt as the paper
// specifies. Holds per-parameter first/second moment buffers (M, V) + a step counter — all
// public so the Checkpoint module can serialize them (resuming without them restarts momentum
// and causes a loss spike).

import type { Tensor } from "../Tensor/Tensor.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";

/** Structural type both Adam and AdamW satisfy (used by the training loop). */
export type Optimizer = {
  Params: Tensor[];
  M: Float64Array[];
  V: Float64Array[];
  StepCount: number;
  Step(Lr: number): void;
  ZeroGrad(): void;
};

export class Adam {
  Params: Tensor[];
  M: Float64Array[];
  V: Float64Array[];
  StepCount = 0;
  Beta1: number;
  Beta2: number;
  Eps: number;

  constructor(Params: Tensor[], Config: ResolvedConfig) {
    this.Params = Params;
    this.M = Params.map((P) => new Float64Array(P.Size));
    this.V = Params.map((P) => new Float64Array(P.Size));
    this.Beta1 = Config.Optimizer.Beta1;
    this.Beta2 = Config.Optimizer.Beta2;
    this.Eps = Config.Optimizer.Epsilon;
  }

  /** One optimizer step at learning rate Lr (supplied per-step by the LR schedule). */
  Step(Lr: number): void {
    this.StepCount++;
    const BiasCorr1 = 1 - Math.pow(this.Beta1, this.StepCount);
    const BiasCorr2 = 1 - Math.pow(this.Beta2, this.StepCount);
    for (let Pi = 0; Pi < this.Params.length; Pi++) {
      const P = this.Params[Pi];
      const M = this.M[Pi];
      const V = this.V[Pi];
      for (let I = 0; I < P.Size; I++) {
        const G = P.Grad[I];
        M[I] = this.Beta1 * M[I] + (1 - this.Beta1) * G;
        V[I] = this.Beta2 * V[I] + (1 - this.Beta2) * G * G;
        P.Data[I] -= (Lr * (M[I] / BiasCorr1)) / (Math.sqrt(V[I] / BiasCorr2) + this.Eps);
      }
    }
  }

  ZeroGrad(): void {
    for (const P of this.Params) P.ZeroGrad();
  }
}

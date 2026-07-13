// Gradient-accumulation batching: run BatchSize independent sequences, letting gradients
// accumulate into the shared params, then divide the ACCUMULATED GRADIENT by BatchSize (NEVER
// the learning rate — with Adam's eps-bounded update the two are not interchangeable; dividing
// the LR would shift the step size ~BatchSize×). Closes REVIEW.md L3.

import type { Shahd } from "../Nn/Shahd.ts";
import type { Optimizer } from "../Optim/OptimBarrel.ts";
import type { DataLoader } from "../Data/DataLoader.ts";
import { ForwardBackward } from "./TrainingStep.ts";

export function AccumulateGradients(
  Model: Shahd,
  Optimizer: Optimizer,
  Loader: DataLoader,
  BatchSize: number,
): number {
  Optimizer.ZeroGrad();
  let TotalLoss = 0;
  for (let B = 0; B < BatchSize; B++) {
    const { Ids, Targets } = Loader.GetSequence();
    TotalLoss += ForwardBackward(Model, Ids, Targets);
  }
  const Inv = 1 / BatchSize;
  for (const P of Optimizer.Params) {
    for (let I = 0; I < P.Size; I++) P.Grad[I] *= Inv;
  }
  return TotalLoss * Inv;
}

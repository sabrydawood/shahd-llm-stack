// One forward + backward over a single sequence. Accumulates gradients into the model
// parameters (via the ops' += semantics) and returns the scalar loss value.

import type { Shahd } from "../Nn/Shahd.ts";
import { CrossEntropy } from "../Ops/OpsBarrel.ts";
import { Backward } from "../Autograd/Backward.ts";

export function ForwardBackward(Model: Shahd, Ids: number[], Targets: number[]): number {
  const Logits = Model.Forward(Ids);
  const Loss = CrossEntropy(Logits, Targets);
  Backward(Loss);
  return Loss.Data[0];
}

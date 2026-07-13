// Validation loss over `Iterations` sequences with the tape off (no autograd). Also reports
// bits-per-byte (loss in nats / ln2) — for char-level on mostly-ASCII code this ~= bits/char,
// and it is the tokenizer-invariant metric to track across the future char->BPE transition.

import type { Shahd } from "../Nn/Shahd.ts";
import type { DataLoader } from "../Data/DataLoader.ts";
import { CrossEntropy } from "../Ops/OpsBarrel.ts";
import { WithTapeOff } from "../Tensor/Tape.ts";

export type EvalResult = { Loss: number; BitsPerByte: number };

export function EvalLoss(Model: Shahd, Loader: DataLoader, Iterations: number): EvalResult {
  return WithTapeOff(() => {
    let Total = 0;
    for (let I = 0; I < Iterations; I++) {
      const { Ids, Targets } = Loader.GetSequence();
      Total += CrossEntropy(Model.Forward(Ids), Targets).Data[0];
    }
    const Loss = Total / Iterations;
    return { Loss, BitsPerByte: Loss / Math.LN2 };
  });
}

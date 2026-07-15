// Validation loss over `Iterations` sequences with the tape off (no autograd). Also reports
// BitsPerByte (loss in nats / ln2) — despite the name this is bits-PER-TOKEN, since Loss is
// CrossEntropy's mean nats-per-token (see Brain/Ops/CrossEntropy.ts) and EvalLoss has no
// tokenizer here to recover true byte counts. It equals bits-per-byte only for the char-level
// tokenizer, where one token == one byte. Now that BPE is in active use (BytePairEncoder via
// Scripts/TrainOnFoundry.ts, Scripts/TrainSftChat.ts), a BPE token spans a variable number of
// bytes, so ValBpb for those runs is NOT a true bits/byte figure — treat it as bits/token.

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
    return { Loss, BitsPerByte: Loss / Math.LN2 }; // bits-per-TOKEN; see file header caveat
  });
}

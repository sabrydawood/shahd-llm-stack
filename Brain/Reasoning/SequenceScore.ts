// Model-backed sequence scoring — the real value function the reasoning-search primitives were
// missing. TreeOfThoughtsSearch and best-of-N take an injected Score/rank; until now the only supplier
// was a toy placeholder in tests. SequenceLogProb turns the MODEL ITSELF into the scorer: the average
// per-token log-probability it assigns to a candidate (higher = the model finds it more coherent). One
// teacher-forced forward pass with the autograd tape OFF (inference only). This is what makes ToT and
// best-of-N reranking genuinely model-backed rather than decorative.

import type { Shahd } from "../Nn/Shahd.ts";
import { WithTapeOff } from "../Tensor/Tape.ts";

/** Average natural-log per-token probability the model assigns to `Ids` (row t of Forward predicts token
 *  t+1). Higher (closer to 0) = more confident. Returns 0 for sequences shorter than 2 tokens. */
export function SequenceLogProb(Model: Shahd, Ids: number[]): number {
  if (Ids.length < 2) return 0;
  return WithTapeOff(() => {
    const Logits = Model.Forward(Ids); // [T, V]
    const V = Logits.Cols;
    let LogProb = 0;
    for (let T = 0; T < Ids.length - 1; T++) {
      const Base = T * V;
      let Max = -Infinity;
      for (let J = 0; J < V; J++) Max = Math.max(Max, Logits.Data[Base + J]);
      let Sum = 0;
      for (let J = 0; J < V; J++) Sum += Math.exp(Logits.Data[Base + J] - Max);
      const Target = Ids[T + 1]!;
      LogProb += Logits.Data[Base + Target]! - Max - Math.log(Sum); // log softmax at the realized token
    }
    return LogProb / (Ids.length - 1);
  });
}

/** Best-of-N reranking: return the index of the candidate the model scores most confident (ties -> the
 *  first). A test-time-compute lever — sample N, keep the model's own top pick — needing no training. */
export function BestOfNByLogProb(Model: Shahd, Candidates: number[][]): number {
  let BestIdx = 0;
  let BestScore = -Infinity;
  for (let I = 0; I < Candidates.length; I++) {
    const Score = SequenceLogProb(Model, Candidates[I]!);
    if (Score > BestScore) {
      BestScore = Score;
      BestIdx = I;
    }
  }
  return BestIdx;
}

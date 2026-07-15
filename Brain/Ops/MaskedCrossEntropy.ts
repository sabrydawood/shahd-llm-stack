// Softmax cross-entropy over rows, but only positions where Mask[i] is true contribute (loss and
// gradient). This is the SFT loss: we train ONLY on the assistant's response tokens, not the
// system/user prompt (Phase 4). Reduces to CrossEntropy when every position is masked-in.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function MaskedCrossEntropy(Logits: Tensor, Targets: number[], Mask: boolean[]): Tensor {
  const T = Logits.Rows;
  const V = Logits.Cols;
  if (Targets.length !== T || Mask.length !== T) {
    throw new Error(`MaskedCrossEntropy: Targets/Mask length must equal rows ${T}`);
  }
  const Out = new Tensor(1, 1, undefined, [Logits]);
  const Probs = new Float64Array(T * V);

  let Count = 0;
  let Loss = 0;
  for (let I = 0; I < T; I++) {
    let Max = -Infinity;
    for (let J = 0; J < V; J++) Max = Math.max(Max, Logits.Data[I * V + J]);
    let Sum = 0;
    for (let J = 0; J < V; J++) {
      const E = Math.exp(Logits.Data[I * V + J] - Max);
      Probs[I * V + J] = E;
      Sum += E;
    }
    for (let J = 0; J < V; J++) Probs[I * V + J] /= Sum;
    if (Mask[I]) {
      const Target = Targets[I];
      if (Target < 0 || Target >= V || !Number.isInteger(Target)) {
        throw new Error(`MaskedCrossEntropy: target ${Target} at position ${I} out of range [0,${V}) — likely a tokenizer/vocab mismatch`);
      }
      Loss += -Math.log(Probs[I * V + Target] + 1e-12);
      Count++;
    }
  }
  const Denom = Count > 0 ? Count : 1;
  Out.Data[0] = Loss / Denom;

  if (Tape.On) {
    Out.BackwardFn = () => {
      const G = Out.Grad[0] / Denom;
      for (let I = 0; I < T; I++) {
        if (!Mask[I]) continue;
        const Target = Targets[I];
        for (let J = 0; J < V; J++) {
          Logits.Grad[I * V + J] += G * (Probs[I * V + J] - (J === Target ? 1 : 0));
        }
      }
    };
  }
  return Out;
}

// Softmax cross-entropy over rows: Logits[T,V], Targets length T -> scalar mean loss.
// softmax is FUSED into the loss so the gradient is the numerically-clean (probs - onehot)/T.
// Kept separate from SoftmaxRows (rule #4) — the fused backward is a different mechanism.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function CrossEntropy(Logits: Tensor, Targets: number[]): Tensor {
  const T = Logits.Rows;
  const V = Logits.Cols;
  if (Targets.length !== T) {
    throw new Error(`CrossEntropy: Targets length ${Targets.length} != rows ${T}`);
  }
  const Out = new Tensor(1, 1, undefined, [Logits]);
  const Probs = new Float64Array(T * V);

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
    Loss += -Math.log(Probs[I * V + Targets[I]] + 1e-12);
  }
  Out.Data[0] = Loss / T;

  if (Tape.On) {
    Out.BackwardFn = () => {
      const G = Out.Grad[0] / T;
      for (let I = 0; I < T; I++) {
        const Target = Targets[I];
        for (let J = 0; J < V; J++) {
          Logits.Grad[I * V + J] += G * (Probs[I * V + J] - (J === Target ? 1 : 0));
        }
      }
    };
  }
  return Out;
}

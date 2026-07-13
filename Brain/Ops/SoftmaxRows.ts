// Row-wise softmax on X[M,N] (max-subtracted for numerical stability). Backward is the exact
// softmax Jacobian-vector product: dX_j = p_j * (dOut_j - sum_k p_k*dOut_k). Kept SEPARATE
// from CrossEntropy (rule #4): although both involve softmax, their backward passes are
// genuinely different mechanisms.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function SoftmaxRows(X: Tensor): Tensor {
  const M = X.Rows;
  const N = X.Cols;
  const Out = new Tensor(M, N, undefined, [X]);

  for (let I = 0; I < M; I++) {
    let Max = -Infinity;
    for (let J = 0; J < N; J++) Max = Math.max(Max, X.Data[I * N + J]);
    let Sum = 0;
    for (let J = 0; J < N; J++) {
      const E = Math.exp(X.Data[I * N + J] - Max);
      Out.Data[I * N + J] = E;
      Sum += E;
    }
    for (let J = 0; J < N; J++) Out.Data[I * N + J] /= Sum;
  }

  if (Tape.On) {
    Out.BackwardFn = () => {
      for (let I = 0; I < M; I++) {
        let Dot = 0;
        for (let J = 0; J < N; J++) Dot += Out.Data[I * N + J] * Out.Grad[I * N + J];
        for (let J = 0; J < N; J++) {
          X.Grad[I * N + J] += Out.Data[I * N + J] * (Out.Grad[I * N + J] - Dot);
        }
      }
    };
  }
  return Out;
}

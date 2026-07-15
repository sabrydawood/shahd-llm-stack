// Row-wise softmax on X[M,N] (max-subtracted for numerical stability). Backward is the exact
// softmax Jacobian-vector product: dX_j = p_j * (dOut_j - sum_k p_k*dOut_k). Kept SEPARATE
// from CrossEntropy (rule #4): although both involve softmax, their backward passes are
// genuinely different mechanisms.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";
import { ComputeRowSoftmax } from "./Internal/RowSoftmax.ts";

export function SoftmaxRows(X: Tensor): Tensor {
  const M = X.Rows;
  const N = X.Cols;
  const Out = new Tensor(M, N, undefined, [X]);

  for (let I = 0; I < M; I++) {
    ComputeRowSoftmax(X.Data, I * N, Out.Data, I * N, N);
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

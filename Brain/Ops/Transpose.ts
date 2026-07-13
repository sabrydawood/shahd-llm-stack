// Transpose X[M,N] -> [N,M]. Backward: dX[i,j] += dOut[j,i].

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function Transpose(X: Tensor): Tensor {
  const M = X.Rows;
  const N = X.Cols;
  const Out = new Tensor(N, M, undefined, [X]);
  for (let I = 0; I < M; I++) {
    for (let J = 0; J < N; J++) Out.Data[J * M + I] = X.Data[I * N + J];
  }

  if (Tape.On) {
    Out.BackwardFn = () => {
      for (let I = 0; I < M; I++) {
        for (let J = 0; J < N; J++) X.Grad[I * N + J] += Out.Grad[J * M + I];
      }
    };
  }
  return Out;
}

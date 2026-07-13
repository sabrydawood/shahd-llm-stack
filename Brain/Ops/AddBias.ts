// Add a row-vector bias Bias[1,N] to every row of X[M,N]. Backward: dX passes through;
// dBias is the column-sum of dOut over all rows.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function AddBias(X: Tensor, Bias: Tensor): Tensor {
  const M = X.Rows;
  const N = X.Cols;
  if (Bias.Rows !== 1 || Bias.Cols !== N) {
    throw new Error(`AddBias: bias must be [1,${N}], got ${Bias.Rows}x${Bias.Cols}`);
  }
  const Out = new Tensor(M, N, undefined, [X, Bias]);
  for (let I = 0; I < M; I++) {
    for (let J = 0; J < N; J++) Out.Data[I * N + J] = X.Data[I * N + J] + Bias.Data[J];
  }

  if (Tape.On) {
    Out.BackwardFn = () => {
      for (let I = 0; I < M; I++) {
        for (let J = 0; J < N; J++) {
          const G = Out.Grad[I * N + J];
          X.Grad[I * N + J] += G;
          Bias.Grad[J] += G;
        }
      }
    };
  }
  return Out;
}

// ReLU nonlinearity: max(0, x). Backward: gradient passes only where the input was positive.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function ReLU(X: Tensor): Tensor {
  const Out = new Tensor(X.Rows, X.Cols, undefined, [X]);
  for (let I = 0; I < X.Size; I++) Out.Data[I] = X.Data[I] > 0 ? X.Data[I] : 0;

  if (Tape.On) {
    Out.BackwardFn = () => {
      for (let I = 0; I < X.Size; I++) {
        if (X.Data[I] > 0) X.Grad[I] += Out.Grad[I];
      }
    };
  }
  return Out;
}

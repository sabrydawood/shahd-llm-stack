// Multiply a tensor by a scalar constant. Backward: dX += Factor * dOut.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function Scale(X: Tensor, Factor: number): Tensor {
  const Out = new Tensor(X.Rows, X.Cols, undefined, [X]);
  for (let I = 0; I < X.Size; I++) Out.Data[I] = X.Data[I] * Factor;

  if (Tape.On) {
    Out.BackwardFn = () => {
      for (let I = 0; I < X.Size; I++) X.Grad[I] += Factor * Out.Grad[I];
    };
  }
  return Out;
}

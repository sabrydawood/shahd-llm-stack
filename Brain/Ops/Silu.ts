// SiLU / Swish activation: x * sigmoid(x). The gate function in SwiGLU MLPs (Llama family).
// Backward: d/dx = sig*(1 + x*(1 - sig)), sig = sigmoid(x).

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function Silu(X: Tensor): Tensor {
  const Out = new Tensor(X.Rows, X.Cols, undefined, [X]);
  for (let I = 0; I < X.Size; I++) {
    const V = X.Data[I];
    const Sig = 1 / (1 + Math.exp(-V));
    Out.Data[I] = V * Sig;
  }

  if (Tape.On) {
    Out.BackwardFn = () => {
      for (let I = 0; I < X.Size; I++) {
        const V = X.Data[I];
        const Sig = 1 / (1 + Math.exp(-V));
        X.Grad[I] += Sig * (1 + V * (1 - Sig)) * Out.Grad[I];
      }
    };
  }
  return Out;
}

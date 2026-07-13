// Elementwise (Hadamard) product of two same-shape tensors. Needed by gated MLPs (SwiGLU/GeGLU).
// Backward: dA += dOut ⊙ B ; dB += dOut ⊙ A.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function Mul(A: Tensor, B: Tensor): Tensor {
  if (A.Rows !== B.Rows || A.Cols !== B.Cols) {
    throw new Error(`Mul: shape mismatch ${A.Rows}x${A.Cols} vs ${B.Rows}x${B.Cols}`);
  }
  const Out = new Tensor(A.Rows, A.Cols, undefined, [A, B]);
  for (let I = 0; I < A.Size; I++) Out.Data[I] = A.Data[I] * B.Data[I];

  if (Tape.On) {
    Out.BackwardFn = () => {
      for (let I = 0; I < A.Size; I++) {
        A.Grad[I] += Out.Grad[I] * B.Data[I];
        B.Grad[I] += Out.Grad[I] * A.Data[I];
      }
    };
  }
  return Out;
}

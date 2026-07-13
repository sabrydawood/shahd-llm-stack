// Global-norm gradient clipping: rescale ALL gradients so their combined L2 norm <= MaxNorm.
// A standard stability guard for from-scratch transformers. Returns the pre-clip norm (for logging).

import type { Tensor } from "../Tensor/Tensor.ts";

export function ClipGradGlobalNorm(Params: Tensor[], MaxNorm: number): number {
  let SumSquares = 0;
  for (const P of Params) {
    for (let I = 0; I < P.Size; I++) SumSquares += P.Grad[I] * P.Grad[I];
  }
  const Norm = Math.sqrt(SumSquares);

  if (Norm > MaxNorm) {
    const Factor = MaxNorm / (Norm + 1e-6);
    for (const P of Params) {
      for (let I = 0; I < P.Size; I++) P.Grad[I] *= Factor;
    }
  }
  return Norm;
}

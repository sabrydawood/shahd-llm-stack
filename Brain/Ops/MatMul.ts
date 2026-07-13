// Matrix multiply: A[M,K] @ B[K,N] -> [M,N]. Backward: dA += dOut @ Bᵀ ; dB += Aᵀ @ dOut.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function MatMul(A: Tensor, B: Tensor): Tensor {
  const M = A.Rows;
  const K = A.Cols;
  const N = B.Cols;
  if (B.Rows !== K) {
    throw new Error(`MatMul: inner dims mismatch ${A.Rows}x${A.Cols} @ ${B.Rows}x${B.Cols}`);
  }
  const Out = new Tensor(M, N, undefined, [A, B]);

  for (let I = 0; I < M; I++) {
    for (let P = 0; P < K; P++) {
      const AVal = A.Data[I * K + P];
      if (AVal === 0) continue;
      for (let J = 0; J < N; J++) Out.Data[I * N + J] += AVal * B.Data[P * N + J];
    }
  }

  if (Tape.On) {
    Out.BackwardFn = () => {
      for (let I = 0; I < M; I++) {
        for (let J = 0; J < N; J++) {
          const G = Out.Grad[I * N + J];
          if (G === 0) continue;
          for (let P = 0; P < K; P++) {
            A.Grad[I * K + P] += G * B.Data[P * N + J];
            B.Grad[P * N + J] += G * A.Data[I * K + P];
          }
        }
      }
    };
  }
  return Out;
}

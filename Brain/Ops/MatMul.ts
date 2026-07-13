// Matrix multiply: A[M,K] @ B[K,N] -> [M,N]. Backward: dA += dOut @ Bᵀ ; dB += Aᵀ @ dOut.
//
// Compute routing (M2): when a compute backend is active (Config.Compute selects Go FFI / f32 / GPU),
// both the forward and the two backward matmuls route through it; when none is active (the DEFAULT),
// the inline f64 loops run — zero seam overhead, bit-identical, and gradient-checkable. The autograd
// tape always stays here in TypeScript; only the numeric matmul bodies are delegated.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";
import { GetActiveBackend } from "../ComputeBackend/BackendSelector.ts";

function Transpose(Data: Float64Array, Rows: number, Cols: number): Float64Array {
  const Out = new Float64Array(Rows * Cols);
  for (let R = 0; R < Rows; R++) {
    for (let C = 0; C < Cols; C++) Out[C * Rows + R] = Data[R * Cols + C];
  }
  return Out;
}

export function MatMul(A: Tensor, B: Tensor): Tensor {
  const M = A.Rows;
  const K = A.Cols;
  const N = B.Cols;
  if (B.Rows !== K) {
    throw new Error(`MatMul: inner dims mismatch ${A.Rows}x${A.Cols} @ ${B.Rows}x${B.Cols}`);
  }
  const Out = new Tensor(M, N, undefined, [A, B]);

  const Backend = GetActiveBackend();
  if (Backend === null) {
    // Inline f64 fast path (default): bit-identical, gradcheck-exact, no allocation beyond Out.
    for (let I = 0; I < M; I++) {
      for (let P = 0; P < K; P++) {
        const AVal = A.Data[I * K + P];
        if (AVal === 0) continue;
        for (let J = 0; J < N; J++) Out.Data[I * N + J] += AVal * B.Data[P * N + J];
      }
    }
  } else {
    Out.Data.set(Backend.MatMul(A.Data, B.Data, M, K, N));
  }

  if (Tape.On) {
    Out.BackwardFn = () => {
      const Back = GetActiveBackend();
      if (Back === null) {
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
        return;
      }
      // dA = Out.Grad @ Bᵀ  (M,N)@(N,K) -> (M,K)
      const DA = Back.MatMul(Out.Grad, Transpose(B.Data, K, N), M, N, K);
      for (let I = 0; I < M * K; I++) A.Grad[I] += DA[I];
      // dB = Aᵀ @ Out.Grad  (K,M)@(M,N) -> (K,N)
      const DB = Back.MatMul(Transpose(A.Data, M, K), Out.Grad, K, M, N);
      for (let I = 0; I < K * N; I++) B.Grad[I] += DB[I];
    };
  }
  return Out;
}

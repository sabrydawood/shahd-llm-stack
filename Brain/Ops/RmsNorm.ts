// RMSNorm per row: y = Gamma * x / sqrt(mean(x^2) + Eps). Like LayerNorm but with NO mean
// subtraction and NO beta (used by Llama/Mistral/Qwen; cheaper, often as good). Backward:
//   dL/dx_k = inv*Gamma_k*g_k - (inv^3 / N) * x_k * sum_j(g_j*Gamma_j*x_j)
// where inv = 1/sqrt(mean(x^2)+Eps).

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function RmsNorm(X: Tensor, Gamma: Tensor, Eps: number): Tensor {
  const M = X.Rows;
  const N = X.Cols;
  if (Gamma.Rows !== 1 || Gamma.Cols !== N) throw new Error(`RmsNorm: Gamma must be [1,${N}]`);
  const Out = new Tensor(M, N, undefined, [X, Gamma]);
  const Inv = new Float64Array(M);

  for (let I = 0; I < M; I++) {
    let SumSq = 0;
    for (let J = 0; J < N; J++) {
      const V = X.Data[I * N + J];
      SumSq += V * V;
    }
    const Is = 1 / Math.sqrt(SumSq / N + Eps);
    Inv[I] = Is;
    for (let J = 0; J < N; J++) Out.Data[I * N + J] = Gamma.Data[J] * X.Data[I * N + J] * Is;
  }

  if (Tape.On) {
    Out.BackwardFn = () => {
      for (let I = 0; I < M; I++) {
        const Is = Inv[I];
        let A = 0; // sum_j g_j * Gamma_j * x_j
        for (let J = 0; J < N; J++) {
          A += Out.Grad[I * N + J] * Gamma.Data[J] * X.Data[I * N + J];
        }
        const Coef = (Is * Is * Is) / N;
        for (let J = 0; J < N; J++) {
          const G = Out.Grad[I * N + J];
          Gamma.Grad[J] += G * X.Data[I * N + J] * Is;
          X.Grad[I * N + J] += Is * Gamma.Data[J] * G - Coef * X.Data[I * N + J] * A;
        }
      }
    };
  }
  return Out;
}

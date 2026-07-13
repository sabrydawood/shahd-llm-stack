// LayerNorm per row: y = Gamma * (x - mu)/sqrt(var + Eps) + Beta, with Gamma/Beta as [1,N].
// Uses population variance (divide by N). Backward is the exact standard closed form:
//   dX_i = (invStd/N) * [N*dXHat_i - sum(dXHat) - xHat_i*sum(dXHat*xHat)]
// (the hardest op backward to get right — verified against nano-gpt.ts and by GradCheck).

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

export function LayerNorm(X: Tensor, Gamma: Tensor, Beta: Tensor, Eps: number): Tensor {
  const M = X.Rows;
  const N = X.Cols;
  if (Gamma.Rows !== 1 || Gamma.Cols !== N || Beta.Rows !== 1 || Beta.Cols !== N) {
    throw new Error(`LayerNorm: Gamma/Beta must be [1,${N}]`);
  }
  const Out = new Tensor(M, N, undefined, [X, Gamma, Beta]);
  const XHat = new Float64Array(M * N);
  const InvStd = new Float64Array(M);

  for (let I = 0; I < M; I++) {
    let Mu = 0;
    for (let J = 0; J < N; J++) Mu += X.Data[I * N + J];
    Mu /= N;
    let Var = 0;
    for (let J = 0; J < N; J++) {
      const D = X.Data[I * N + J] - Mu;
      Var += D * D;
    }
    Var /= N;
    const Is = 1 / Math.sqrt(Var + Eps);
    InvStd[I] = Is;
    for (let J = 0; J < N; J++) {
      const Xh = (X.Data[I * N + J] - Mu) * Is;
      XHat[I * N + J] = Xh;
      Out.Data[I * N + J] = Gamma.Data[J] * Xh + Beta.Data[J];
    }
  }

  if (Tape.On) {
    Out.BackwardFn = () => {
      for (let I = 0; I < M; I++) {
        let SumDXHat = 0;
        let SumDXHatXHat = 0;
        const DXHat = new Float64Array(N);
        for (let J = 0; J < N; J++) {
          const Dy = Out.Grad[I * N + J];
          const Dxh = Dy * Gamma.Data[J];
          DXHat[J] = Dxh;
          SumDXHat += Dxh;
          SumDXHatXHat += Dxh * XHat[I * N + J];
          Gamma.Grad[J] += Dy * XHat[I * N + J];
          Beta.Grad[J] += Dy;
        }
        const Is = InvStd[I];
        for (let J = 0; J < N; J++) {
          X.Grad[I * N + J] += (Is / N) * (N * DXHat[J] - SumDXHat - XHat[I * N + J] * SumDXHatXHat);
        }
      }
    };
  }
  return Out;
}

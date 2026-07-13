// The default (and today's only) compute backend: pure-TypeScript matmul, synchronous. Matches
// the math in Ops/MatMul.ts so results are bit-identical.

import type { ComputeBackend } from "./ComputeBackend.ts";

export class TsBackend implements ComputeBackend {
  MatMul(A: Float64Array, B: Float64Array, M: number, K: number, N: number): Float64Array {
    const Out = new Float64Array(M * N);
    for (let I = 0; I < M; I++) {
      const IK = I * K;
      const IN = I * N;
      for (let P = 0; P < K; P++) {
        const AV = A[IK + P];
        if (AV === 0) continue;
        const PN = P * N;
        for (let J = 0; J < N; J++) Out[IN + J] += AV * B[PN + J];
      }
    }
    return Out;
  }
}

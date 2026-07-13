// Async matmul entry point (M5): use the GPU when a WebGPU runtime is present, otherwise fall back
// to the CPU Float32 backend. This is the async counterpart to the sync ComputeBackend seam — it
// exists because WebGPU is inherently async and cannot sit behind the sync forward path yet. Use it
// for batch/offline matmul today; it is the foundation a future async-forward would train on.

import { TsBackendF32 } from "./TsBackendF32.ts";
import { WebGpuAvailable, WebGpuMatMul } from "./WebGpuMatMul.ts";

export type AsyncMatMulOptions = { PreferGpu?: boolean };

/** Out[M,N] = A[M,K] @ B[K,N] in f32: GPU when available (and preferred), else CPU f32 fallback. */
export async function ComputeMatMulAsync(
  A: Float64Array,
  B: Float64Array,
  M: number,
  K: number,
  N: number,
  Options: AsyncMatMulOptions = {},
): Promise<Float64Array> {
  if ((Options.PreferGpu ?? true) && WebGpuAvailable()) {
    try {
      return await WebGpuMatMul(A, B, M, K, N);
    } catch {
      // GPU present but failed — fall back to CPU rather than break the caller.
    }
  }
  return new TsBackendF32().MatMul(A, B, M, K, N);
}

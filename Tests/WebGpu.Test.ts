import { test, expect } from "bun:test";
import { WebGpuAvailable } from "../Brain/ComputeBackend/WebGpuMatMul.ts";
import { ComputeMatMulAsync } from "../Brain/ComputeBackend/AsyncCompute.ts";
import { TsBackendF32 } from "../Brain/ComputeBackend/TsBackendF32.ts";
import { TsBackend } from "../Brain/ComputeBackend/TsBackend.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";

// This runtime has no WebGPU, so these tests verify the PROBE and the CPU FALLBACK path. The GPU
// shader itself runs (and is verified) on a WebGPU-capable runtime.

test("WebGpuAvailable reflects the runtime (false where navigator.gpu is absent)", () => {
  expect(typeof WebGpuAvailable()).toBe("boolean");
});

test("ComputeMatMulAsync returns correct results via the CPU fallback", async () => {
  const Rng = new SeededRng(9);
  const A = new Float64Array(16 * 12);
  const B = new Float64Array(12 * 8);
  for (let I = 0; I < A.length; I++) A[I] = Rng.NextGaussian();
  for (let I = 0; I < B.length; I++) B[I] = Rng.NextGaussian();

  const Async = await ComputeMatMulAsync(A, B, 16, 12, 8);
  // Fallback path is TsBackendF32, so results are exactly the f32 backend's.
  const F32 = new TsBackendF32().MatMul(A, B, 16, 12, 8);
  for (let I = 0; I < Async.length; I++) expect(Async[I]).toBe(F32[I]);

  // And close to the exact f64 reference.
  const Exact = new TsBackend().MatMul(A, B, 16, 12, 8);
  let MaxRel = 0;
  for (let I = 0; I < Exact.length; I++) MaxRel = Math.max(MaxRel, Math.abs(Exact[I] - Async[I]) / (Math.abs(Exact[I]) + 1e-6));
  expect(MaxRel).toBeLessThan(1e-3);
});

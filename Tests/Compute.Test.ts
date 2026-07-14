import { test, expect, afterEach } from "bun:test";
import { TsBackend } from "../Brain/ComputeBackend/TsBackend.ts";
import { TsBackendF32 } from "../Brain/ComputeBackend/TsBackendF32.ts";
import {
  GetActiveBackend,
  SetActiveBackend,
  ActivateFromConfig,
} from "../Brain/ComputeBackend/BackendSelector.ts";
import { MatMul } from "../Brain/Ops/MatMul.ts";
import { Tensor } from "../Brain/Tensor/Tensor.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";

// STRUCTURAL RESET (advisor): never let a thrown assertion leave a non-default backend active and
// pollute the gradcheck-bearing tests elsewhere in the suite.
afterEach(() => SetActiveBackend(null));

function RandTensor(Rows: number, Cols: number, Rng: SeededRng): Tensor {
  const T = new Tensor(Rows, Cols);
  for (let I = 0; I < T.Size; I++) T.Data[I] = Rng.NextGaussian();
  return T;
}

test("default active backend is null (the inline f64 fast path)", () => {
  expect(GetActiveBackend()).toBe(null);
});

test("TsBackendF32 matches TsBackend within f32 tolerance and leaves inputs untouched", () => {
  const Rng = new SeededRng(1);
  const A = new Float64Array(64 * 48);
  const B = new Float64Array(48 * 32);
  for (let I = 0; I < A.length; I++) A[I] = Rng.NextGaussian();
  for (let I = 0; I < B.length; I++) B[I] = Rng.NextGaussian();
  const ASnapshot = Float64Array.from(A);
  const Exact = new TsBackend().MatMul(A, B, 64, 48, 32);
  const F32 = new TsBackendF32().MatMul(A, B, 64, 48, 32);
  let MaxRel = 0;
  for (let I = 0; I < Exact.length; I++) {
    MaxRel = Math.max(MaxRel, Math.abs(Exact[I] - F32[I]) / (Math.abs(Exact[I]) + 1e-6));
  }
  expect(MaxRel).toBeLessThan(1e-3);
  expect(A).toEqual(ASnapshot); // f32 path copied inputs, did not mutate them
});

test("routing MatMul through TsBackend is bit-identical to the inline path", () => {
  const Rng = new SeededRng(2);
  const A = RandTensor(10, 7, Rng);
  const B = RandTensor(7, 5, Rng);
  const Inline = MatMul(A, B); // Active === null
  SetActiveBackend(new TsBackend());
  const Routed = MatMul(A, B);
  for (let I = 0; I < Inline.Size; I++) expect(Routed.Data[I]).toBe(Inline.Data[I]);
});

test("routing MatMul through TsBackendF32 stays within f32 tolerance, then toggles back", () => {
  const Rng = new SeededRng(3);
  const A = RandTensor(12, 9, Rng);
  const B = RandTensor(9, 6, Rng);
  const Exact = MatMul(A, B);
  SetActiveBackend(new TsBackendF32());
  const F32 = MatMul(A, B);
  let MaxRel = 0;
  for (let I = 0; I < Exact.Size; I++) MaxRel = Math.max(MaxRel, Math.abs(Exact.Data[I] - F32.Data[I]) / (Math.abs(Exact.Data[I]) + 1e-6));
  expect(MaxRel).toBeLessThan(1e-3);
  SetActiveBackend(null); // runtime toggle back to CPU f64
  const Back = MatMul(A, B);
  for (let I = 0; I < Exact.Size; I++) expect(Back.Data[I]).toBe(Exact.Data[I]);
});

test("routed backward gradients match the inline backward within f64 rounding", () => {
  const Rng = new SeededRng(4);
  const A = RandTensor(6, 5, Rng);
  const B = RandTensor(5, 4, Rng);
  const InlineOut = MatMul(A, B);
  for (let I = 0; I < InlineOut.Size; I++) InlineOut.Grad[I] = 1;
  InlineOut.BackwardFn?.();
  const InlineDA = Float64Array.from(A.Grad);
  const InlineDB = Float64Array.from(B.Grad);

  A.Grad.fill(0);
  B.Grad.fill(0);
  SetActiveBackend(new TsBackend());
  const RoutedOut = MatMul(A, B);
  for (let I = 0; I < RoutedOut.Size; I++) RoutedOut.Grad[I] = 1;
  RoutedOut.BackwardFn?.();
  for (let I = 0; I < A.Size; I++) expect(Math.abs(A.Grad[I] - InlineDA[I])).toBeLessThan(1e-9);
  for (let I = 0; I < B.Size; I++) expect(Math.abs(B.Grad[I] - InlineDB[I])).toBeLessThan(1e-9);
});

test("ActivateFromConfig honors the config and falls back to CPU when a backend is missing", () => {
  // Test backends EXPLICITLY (not via the default) so this stays portable regardless of which backend
  // is the configured default and whether the Go FFI DLL is present on the CI machine.
  const Ts = ActivateFromConfig(LoadConfig({ Overrides: { Compute: { Backend: "Ts" } }, UseCli: false, UseEnv: false }));
  expect(Ts.Chosen).toContain("Ts/F64");
  expect(GetActiveBackend()).toBe(null); // Ts f64 => inline

  const F32 = ActivateFromConfig(LoadConfig({ Overrides: { Compute: { Backend: "Ts", Precision: "F32" } }, UseCli: false, UseEnv: false }));
  expect(F32.Chosen).toContain("F32");
  expect(GetActiveBackend()).not.toBe(null);

  // GPU is not built; with FallbackToCpu it must not throw and must drop to CPU.
  const Gpu = ActivateFromConfig(LoadConfig({ Overrides: { Compute: { Backend: "Gpu", FallbackToCpu: true } }, UseCli: false, UseEnv: false }));
  expect(Gpu.FellBack).toBe(true);
});

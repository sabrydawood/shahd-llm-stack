import { test, expect } from "bun:test";
import { Tensor } from "../Brain/Tensor/Tensor.ts";
import { Tape } from "../Brain/Tensor/Tape.ts";
import { GradCheck } from "../Brain/Autograd/GradCheck.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";
import { ApplyRope, RmsNorm, Gelu, Silu, Mul } from "../Brain/Ops/OpsBarrel.ts";

function RandTensor(Rows: number, Cols: number, Rng: SeededRng, Scale = 1): Tensor {
  const T = new Tensor(Rows, Cols);
  for (let I = 0; I < T.Size; I++) T.Data[I] = Rng.NextGaussian() * Scale;
  return T;
}

function RandVec(N: number, Rng: SeededRng): Float64Array {
  const Out = new Float64Array(N);
  for (let I = 0; I < N; I++) Out[I] = Rng.NextGaussian();
  return Out;
}

function ProjectScalar(X: Tensor, R: Float64Array): Tensor {
  const Out = new Tensor(1, 1, undefined, [X]);
  let Sum = 0;
  for (let I = 0; I < X.Size; I++) Sum += X.Data[I] * R[I];
  Out.Data[0] = Sum;
  if (Tape.On) {
    Out.BackwardFn = () => {
      const G = Out.Grad[0];
      for (let I = 0; I < X.Size; I++) X.Grad[I] += G * R[I];
    };
  }
  return Out;
}

test("RoPE gradient is correct and preserves norm per pair", () => {
  const Rng = new SeededRng(1);
  const X = RandTensor(4, 8, Rng); // T=4, HeadDim=8
  const R = RandVec(32, Rng);
  expect(GradCheck([X], () => ProjectScalar(ApplyRope(X, 2), R)).Passed).toBe(true);

  // Rotation preserves each pair's magnitude.
  const Y = ApplyRope(X, 0);
  const M0 = X.Data[0] * X.Data[0] + X.Data[1] * X.Data[1];
  const M1 = Y.Data[0] * Y.Data[0] + Y.Data[1] * Y.Data[1];
  expect(Math.abs(M0 - M1)).toBeLessThan(1e-9);
});

test("RmsNorm gradient is correct (X and Gamma)", () => {
  const Rng = new SeededRng(2);
  const X = RandTensor(3, 6, Rng);
  const Gamma = new Tensor(1, 6);
  Gamma.Data.fill(1);
  for (let I = 0; I < 6; I++) Gamma.Data[I] += 0.1 * Rng.NextGaussian();
  const R = RandVec(18, Rng);
  expect(GradCheck([X, Gamma], () => ProjectScalar(RmsNorm(X, Gamma, 1e-5), R)).Passed).toBe(true);
});

test("Gelu gradient is correct", () => {
  const Rng = new SeededRng(3);
  const X = RandTensor(3, 5, Rng);
  const R = RandVec(15, Rng);
  expect(GradCheck([X], () => ProjectScalar(Gelu(X), R)).Passed).toBe(true);
});

test("Silu gradient is correct", () => {
  const Rng = new SeededRng(4);
  const X = RandTensor(3, 5, Rng);
  const R = RandVec(15, Rng);
  expect(GradCheck([X], () => ProjectScalar(Silu(X), R)).Passed).toBe(true);
});

test("Mul (Hadamard) gradient is correct", () => {
  const Rng = new SeededRng(5);
  const A = RandTensor(3, 4, Rng);
  const B = RandTensor(3, 4, Rng);
  const R = RandVec(12, Rng);
  expect(GradCheck([A, B], () => ProjectScalar(Mul(A, B), R)).Passed).toBe(true);
});

import { test, expect } from "bun:test";
import { Zeros, Filled, RandN } from "../Brain/Tensor/TensorFactories.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";

test("tensor factories produce correct shapes", () => {
  const Z = Zeros(2, 3);
  expect(Z.Rows).toBe(2);
  expect(Z.Cols).toBe(3);
  expect(Z.Size).toBe(6);
  expect([...Z.Shape]).toEqual([2, 3]);
  expect(Array.from(Z.Data)).toEqual([0, 0, 0, 0, 0, 0]);

  const F = Filled(1, 4, 2.5);
  expect(Array.from(F.Data)).toEqual([2.5, 2.5, 2.5, 2.5]);
});

test("ZeroGrad clears the gradient buffer", () => {
  const Z = Zeros(2, 2);
  Z.Grad.fill(3);
  Z.ZeroGrad();
  expect(Array.from(Z.Grad)).toEqual([0, 0, 0, 0]);
});

test("RandN is reproducible with the same seed and scaled", () => {
  const A = RandN(4, 4, 0.02, new SeededRng(1));
  const B = RandN(4, 4, 0.02, new SeededRng(1));
  expect(Array.from(A.Data)).toEqual(Array.from(B.Data));
  const Mean = A.Data.reduce((Sum, X) => Sum + X, 0) / A.Size;
  expect(Math.abs(Mean)).toBeLessThan(0.05);
});

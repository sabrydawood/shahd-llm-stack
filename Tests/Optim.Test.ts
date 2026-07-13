import { test, expect } from "bun:test";
import { Tensor } from "../Brain/Tensor/Tensor.ts";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { Adam } from "../Brain/Optim/Adam.ts";
import { AdamW } from "../Brain/Optim/AdamW.ts";
import { ComputeLr } from "../Brain/Optim/LrSchedule.ts";
import { ClipGradGlobalNorm } from "../Brain/Optim/GradClip.ts";

const Isolated = { UseCli: false, UseEnv: false } as const;

test("Adam drives a simple quadratic to its minimum", () => {
  const X = new Tensor(1, 5);
  X.Data.set([1, -2, 3, -4, 5]);
  const Config = LoadConfig({ Overrides: { Optimizer: { Kind: "Adam", LearningRate: 0.1 } }, ...Isolated });
  const Opt = new Adam([X], Config);
  for (let S = 0; S < 800; S++) {
    for (let I = 0; I < X.Size; I++) X.Grad[I] = 2 * X.Data[I]; // grad of sum(x^2)
    Opt.Step(0.1);
  }
  for (let I = 0; I < X.Size; I++) expect(Math.abs(X.Data[I])).toBeLessThan(0.02);
});

test("AdamW decays weight matrices toward zero when gradients are zero", () => {
  const W = new Tensor(2, 2);
  W.Data.fill(1);
  const Config = LoadConfig({
    Overrides: { Optimizer: { Kind: "AdamW", WeightDecay: 0.1, LearningRate: 0.01 } },
    ...Isolated,
  });
  const Opt = new AdamW([W], Config);
  const Before = W.Data[0];
  for (let S = 0; S < 20; S++) {
    W.ZeroGrad();
    Opt.Step(0.01);
  }
  expect(W.Data[0]).toBeLessThan(Before);
  expect(Number.isFinite(W.Data[0])).toBe(true);
});

test("LR schedule warms up linearly then cosine-decays to the floor", () => {
  const Config = LoadConfig({
    Overrides: {
      Optimizer: { LearningRate: 1 },
      Schedule: { Kind: "Cosine", WarmupSteps: 10, MaxSteps: 100, MinLrRatio: 0.1 },
    },
    ...Isolated,
  });
  expect(ComputeLr(0, Config)).toBeCloseTo(0.1, 6); // (0+1)/10
  expect(ComputeLr(9, Config)).toBeCloseTo(1.0, 6); // peak at end of warmup
  expect(ComputeLr(100, Config)).toBeCloseTo(0.1, 6); // floor
  const Mid = ComputeLr(55, Config);
  expect(Mid).toBeLessThan(1.0);
  expect(Mid).toBeGreaterThan(0.1);
});

test("GradClip rescales gradients to the max global norm", () => {
  const A = new Tensor(1, 2);
  A.Grad.set([3, 4]); // L2 norm = 5
  const Norm = ClipGradGlobalNorm([A], 1);
  expect(Norm).toBeCloseTo(5, 6);
  expect(Math.hypot(A.Grad[0], A.Grad[1])).toBeLessThanOrEqual(1.0001);
});

import { test, expect } from "bun:test";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";

const Isolated = { UseCli: false, UseEnv: false } as const;

test("LoadConfig derives HeadDim and AttentionScale from the single home (L4-safe)", () => {
  const Config = LoadConfig({ ConfigPath: "Configs/Phase1Small.Config.json", ...Isolated });
  expect(Config.Model.EmbedDim).toBe(128);
  expect(Config.Model.NumHeads).toBe(4);
  expect(Config.Derived.HeadDim).toBe(32);
  expect(Config.Derived.AttentionScale).toBeCloseTo(1 / Math.sqrt(32), 12);
  expect(Config.Derived.MlpHidden).toBe(512);
});

test("LoadConfig rejects EmbedDim not divisible by NumHeads (L4 guard)", () => {
  expect(() =>
    LoadConfig({ Overrides: { Model: { EmbedDim: 10, NumHeads: 3 } }, ...Isolated }),
  ).toThrow();
});

test("resolved config is deep-frozen", () => {
  const Config = LoadConfig(Isolated);
  expect(Object.isFrozen(Config)).toBe(true);
  expect(Object.isFrozen(Config.Model)).toBe(true);
  expect(Object.isFrozen(Config.Derived)).toBe(true);
});

test("ConfigHash is stable for identical inputs and changes with content", () => {
  const A = LoadConfig(Isolated);
  const B = LoadConfig(Isolated);
  expect(A.ConfigHash).toBe(B.ConfigHash);
  const C = LoadConfig({ Overrides: { Model: { NumLayers: 8 } }, ...Isolated });
  expect(C.ConfigHash).not.toBe(A.ConfigHash);
});

test("scaled residual init shrinks the init scale with depth", () => {
  const Config = LoadConfig({
    Overrides: { Model: { UseScaledResidualInit: true, InitScale: 0.02, NumLayers: 2 } },
    ...Isolated,
  });
  expect(Config.Derived.ResidualInitScale).toBeCloseTo(0.02 / Math.sqrt(4), 12);
});

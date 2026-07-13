import { test, expect } from "bun:test";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams } from "../Brain/Random/SeededRng.ts";
import { Shahd } from "../Brain/Nn/NnBarrel.ts";
import { CrossEntropy } from "../Brain/Ops/OpsBarrel.ts";
import { GradCheck } from "../Brain/Autograd/GradCheck.ts";
import type { ConfigOverride } from "../Brain/Config/ConfigTypes.ts";

function BuildTiny(Overrides: ConfigOverride): Shahd {
  const Config = LoadConfig({ Overrides, UseCli: false, UseEnv: false });
  const Rng = CreateRngStreams(Config.Training.Seed);
  return new Shahd(Config, Rng.InitRng);
}

test("Shahd forward produces logits of shape [T, VocabSize]", () => {
  const Model = BuildTiny({ Model: { EmbedDim: 8, NumLayers: 2, NumHeads: 2, BlockSize: 8, VocabSize: 10 } });
  const Logits = Model.Forward([1, 4, 2, 7, 3]);
  expect(Logits.Rows).toBe(5);
  expect(Logits.Cols).toBe(10);
});

test("Shahd rejects sequences longer than BlockSize", () => {
  const Model = BuildTiny({ Model: { EmbedDim: 8, NumLayers: 1, NumHeads: 2, BlockSize: 4, VocabSize: 10 } });
  expect(() => Model.Forward([1, 2, 3, 4, 5])).toThrow();
});

test("full-model backward is gradcheck-correct (tied weights, multi-head)", () => {
  const Model = BuildTiny({
    Model: { EmbedDim: 8, NumLayers: 1, NumHeads: 2, BlockSize: 8, VocabSize: 10, MlpRatio: 2, WeightTying: true },
  });
  const Ids = [1, 4, 2, 7, 3];
  const Targets = [4, 2, 7, 3, 9];
  const Result = GradCheck(Model.Parameters(), () => CrossEntropy(Model.Forward(Ids), Targets), {
    Tolerance: 1e-3,
  });
  expect(Result.Passed).toBe(true);
});

test("full-model backward is gradcheck-correct (untied weights, single head)", () => {
  const Model = BuildTiny({
    Model: { EmbedDim: 6, NumLayers: 1, NumHeads: 1, BlockSize: 8, VocabSize: 7, MlpRatio: 2, WeightTying: false },
  });
  const Ids = [0, 3, 1, 5];
  const Targets = [3, 1, 5, 6];
  const Result = GradCheck(Model.Parameters(), () => CrossEntropy(Model.Forward(Ids), Targets), {
    Tolerance: 1e-3,
  });
  expect(Result.Passed).toBe(true);
});

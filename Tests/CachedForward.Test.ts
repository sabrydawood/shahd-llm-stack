import { test, expect } from "bun:test";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams, SeededRng } from "../Brain/Random/SeededRng.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { WithTapeOff } from "../Brain/Tensor/Tape.ts";
import { KvCache } from "../Brain/Sampling/KvCache.ts";
import { CachedForwardStep, CachedGenerate } from "../Brain/Sampling/CachedForward.ts";
import { Generate } from "../Brain/Sampling/Generate.ts";
import type { ConfigOverride } from "../Brain/Config/ConfigTypes.ts";

function Build(Model: ConfigOverride["Model"]): Shahd {
  const Config = LoadConfig({ Overrides: { Model }, UseCli: false, UseEnv: false });
  return new Shahd(Config, CreateRngStreams(Config.Training.Seed).InitRng);
}

test("cached step logits are numerically identical to the full Tensor forward (tied, multi-head)", () => {
  const Model = Build({ EmbedDim: 16, NumLayers: 2, NumHeads: 2, BlockSize: 16, VocabSize: 12, MlpRatio: 2, WeightTying: true });
  const Seq = [1, 5, 2, 8, 3, 0, 7];
  const V = Model.Config.Model.VocabSize;
  const Full = WithTapeOff(() => Model.Forward(Seq));
  const Cache = new KvCache(Model.Blocks.length, Model.Config.Model.NumHeads, Model.Config.Derived.HeadDim, Model.Config.Model.BlockSize);
  for (let P = 0; P < Seq.length; P++) {
    const Logits = CachedForwardStep(Model, Cache, Seq[P], P);
    for (let Vi = 0; Vi < V; Vi++) {
      expect(Math.abs(Logits[Vi] - Full.Data[P * V + Vi])).toBeLessThan(1e-9);
    }
  }
});

test("cached step logits match the full forward (untied, single head)", () => {
  const Model = Build({ EmbedDim: 12, NumLayers: 1, NumHeads: 1, BlockSize: 16, VocabSize: 9, MlpRatio: 2, WeightTying: false });
  const Seq = [2, 0, 5, 1, 3];
  const V = Model.Config.Model.VocabSize;
  const Full = WithTapeOff(() => Model.Forward(Seq));
  const Cache = new KvCache(Model.Blocks.length, Model.Config.Model.NumHeads, Model.Config.Derived.HeadDim, Model.Config.Model.BlockSize);
  for (let P = 0; P < Seq.length; P++) {
    const Logits = CachedForwardStep(Model, Cache, Seq[P], P);
    for (let Vi = 0; Vi < V; Vi++) {
      expect(Math.abs(Logits[Vi] - Full.Data[P * V + Vi])).toBeLessThan(1e-9);
    }
  }
});

test("CachedGenerate matches uncached Generate (greedy) within BlockSize", () => {
  const Model = Build({ EmbedDim: 16, NumLayers: 2, NumHeads: 2, BlockSize: 16, VocabSize: 12, MlpRatio: 2 });
  const Prompt = [1, 5, 2];
  const Greedy = { Temperature: 0, TopK: 0, TopP: 1 };
  const Uncached = Generate(Model, Prompt, 8, Greedy, new SeededRng(1));
  const Cached = CachedGenerate(Model, Prompt, 8, Greedy, new SeededRng(1));
  expect(Cached).toEqual(Uncached);
});

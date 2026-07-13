import { test, expect } from "bun:test";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams } from "../Brain/Random/SeededRng.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { SpeculativeSample } from "../Brain/Reasoning/ReasoningBarrel.ts";
import { ProbsFromLogits, SampleFromDistribution } from "../Brain/Sampling/Distribution.ts";
import { SampleFromLogits } from "../Brain/Sampling/Sampler.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";

function TinyModel(NumLayers: number, Seed: number): Shahd {
  const Config = LoadConfig({
    Overrides: { Model: { VocabSize: 20, EmbedDim: 16, NumLayers, NumHeads: 2, BlockSize: 32 } },
    UseCli: false,
    UseEnv: false,
  });
  return new Shahd(Config, CreateRngStreams(Seed).InitRng);
}

test("refactored sampler still matches a direct distribution sample (behavior preserved)", () => {
  const Rng1 = new SeededRng(5);
  const Rng2 = new SeededRng(5);
  const Logits = new Float64Array(20);
  for (let I = 0; I < 20; I++) Logits[I] = new SeededRng(I + 1).NextGaussian();
  const Options = { Temperature: 0.8, TopK: 5, TopP: 0.9 };
  const ViaSampler = SampleFromLogits(Logits, 0, 20, Options, Rng1);
  const ViaDistribution = SampleFromDistribution(ProbsFromLogits(Logits, 0, 20, Options), Rng2);
  expect(ViaSampler).toBe(ViaDistribution); // same single RNG draw, same selection
});

test("speculative sampling with the target as its own draft accepts every proposal", () => {
  const Model = TinyModel(2, 7);
  const Options = { Temperature: 0.9, TopK: 0, TopP: 1 };
  const Result = SpeculativeSample(Model, Model, [1, 2, 3], 12, Options, new SeededRng(3), 4);
  expect(Result.Ids.length).toBe(3 + 12);
  expect(Result.AcceptedTokens).toBe(Result.DraftTokens); // q == p => accept probability 1
  expect(Result.TargetCalls).toBeLessThan(12); // fewer target passes than tokens produced
});

test("speculative sampling with a weaker draft still produces a full, valid sequence", () => {
  const Target = TinyModel(2, 7);
  const Draft = TinyModel(1, 11);
  const Options = { Temperature: 1, TopK: 10, TopP: 0.95 };
  const Result = SpeculativeSample(Target, Draft, [4, 5, 6], 10, Options, new SeededRng(2), 4);
  expect(Result.Ids.length).toBe(3 + 10);
  expect(Result.AcceptedTokens).toBeLessThanOrEqual(Result.DraftTokens);
  for (const Id of Result.Ids) expect(Id).toBeGreaterThanOrEqual(0);
});

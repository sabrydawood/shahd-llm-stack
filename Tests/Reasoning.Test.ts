import { test, expect } from "bun:test";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams } from "../Brain/Random/SeededRng.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { Generate } from "../Brain/Sampling/Generate.ts";
import { ChatTokens } from "../Brain/Sft/ChatTemplate.ts";
import {
  SpeculativeDecodeGreedy,
  SplitThinking,
  StripThinking,
  WrapThinking,
  MajorityVote,
  SelfConsistency,
  TreeOfThoughtsSearch,
  ExtractAnswer,
  NormalizeAnswer,
  AnswerKey,
  SequenceLogProb,
  BestOfNByLogProb,
} from "../Brain/Reasoning/ReasoningBarrel.ts";

const GreedyOpts = { Temperature: 0, TopK: 0, TopP: 1 };

function TinyModel(NumLayers: number, Seed: number): Shahd {
  const Config = LoadConfig({
    Overrides: { Model: { VocabSize: 20, EmbedDim: 16, NumLayers, NumHeads: 2, BlockSize: 32 } },
    UseCli: false,
    UseEnv: false,
  });
  return new Shahd(Config, CreateRngStreams(Seed).InitRng);
}

test("speculative greedy is bit-identical to plain greedy (same model as its own draft)", () => {
  const Model = TinyModel(2, 7);
  const Prompt = [1, 2, 3];
  const Plain = Generate(Model, Prompt, 12, GreedyOpts, CreateRngStreams(0).SamplingRng);
  const Spec = SpeculativeDecodeGreedy(Model, Model, Prompt, 12, 4);
  expect(Spec.Ids).toEqual(Plain); // exactness within the context window
  expect(Spec.AcceptedTokens).toBe(Spec.DraftTokens); // self-draft => 100% acceptance
  expect(Spec.TargetCalls).toBeLessThan(12); // fewer target passes than tokens generated
});

test("speculative greedy matches the target even with a weaker draft model", () => {
  const Target = TinyModel(2, 7);
  const Draft = TinyModel(1, 11); // different weights => imperfect draft
  const Prompt = [4, 5, 6];
  const Plain = Generate(Target, Prompt, 10, GreedyOpts, CreateRngStreams(0).SamplingRng);
  const Spec = SpeculativeDecodeGreedy(Target, Draft, Prompt, 10, 4);
  expect(Spec.Ids).toEqual(Plain); // still the target's exact greedy output
  expect(Spec.AcceptedTokens).toBeLessThanOrEqual(Spec.DraftTokens);
});

test("thinking mode splits hidden reasoning from the visible answer", () => {
  const Text = WrapThinking("first I compute 2+2", "The answer is 4.");
  const Split = SplitThinking(Text);
  expect(Split.HadThinking).toBe(true);
  expect(Split.Thinking).toBe("first I compute 2+2");
  expect(Split.Answer).toBe("The answer is 4.");
  expect(StripThinking(Text)).toBe("The answer is 4.");
  expect(SplitThinking("no think tags here").HadThinking).toBe(false);
  expect(Text.startsWith(ChatTokens.Think)).toBe(true);
});

test("majority vote tallies and breaks ties by first appearance", () => {
  const V = MajorityVote(["4", "4", "5", "4", "5"], (A) => A);
  expect(V.Winner).toBe("4");
  expect(V.Count).toBe(3);
  expect(V.Total).toBe(5);
  const Tie = MajorityVote(["b", "a", "a", "b"], (A) => A);
  expect(Tie.Winner).toBe("b"); // "b" seen first at the tied count
});

test("self-consistency votes over sampled generations", () => {
  const Outputs = ["ans=7", "ans=7", "ans=9"];
  let I = 0;
  const Vote = SelfConsistency(() => Outputs[I++], 3, (T) => T.split("=")[1]);
  expect(Vote.Winner).toBe("7");
});

test("answer extraction pulls the <answer> span, ignores thinking, and normalizes", () => {
  expect(ExtractAnswer(WrapThinking("compute 6*7", "<answer>42</answer>"))).toBe("42");
  expect(ExtractAnswer("some reasoning\nfinal line")).toBe("final line"); // fallback: last non-empty line
  expect(NormalizeAnswer("  42.0 ")).toBe("42");
  expect(NormalizeAnswer("The Answer.")).toBe("the answer");
  expect(AnswerKey("<answer>1,000</answer>")).toBe("1000");
});

test("self-consistency votes together across trivial formatting differences (normalized key)", () => {
  const Outputs = ["<answer>42</answer>", "<answer>42.</answer>", "<answer> 42 </answer>", "<answer>7</answer>"];
  let I = 0;
  const Vote = SelfConsistency(() => Outputs[I++]!, 4, ExtractAnswer, NormalizeAnswer);
  expect(NormalizeAnswer(Vote.Winner)).toBe("42");
  expect(Vote.Count).toBe(3); // the three 42-variants counted as one answer
});

test("model-backed sequence scoring is finite and best-of-N returns the model's top-scoring candidate", () => {
  const Model = TinyModel(2, 3);
  const Score = SequenceLogProb(Model, [1, 2, 3, 4, 5]);
  expect(Number.isFinite(Score)).toBe(true);
  expect(Score).toBeLessThanOrEqual(0); // average log-prob is <= 0
  const Cands = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
  const Idx = BestOfNByLogProb(Model, Cands);
  const Scores = Cands.map((C) => SequenceLogProb(Model, C));
  expect(Scores[Idx]).toBe(Math.max(...Scores)); // returns the argmax by model confidence
});

test("tree of thoughts beam search finds the highest-scoring path", () => {
  // Expand appends digits; Score rewards paths whose digits ascend. Best 2-step path from [] is 8,9.
  const Expand = (Path: string[]): string[] => (Path.length >= 2 ? [] : ["1", "5", "9", "8"]);
  const Score = (Path: string[]): number => Path.reduce((Acc, D) => Acc + Number(D), 0);
  const Best = TreeOfThoughtsSearch([], Expand, Score, { Beam: 2, Depth: 2 });
  expect(Best.Path).toEqual(["9", "9"]);
  expect(Best.Score).toBe(18);
});

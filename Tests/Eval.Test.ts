import { test, expect } from "bun:test";
import { PassAtK } from "../Brain/Eval/PassAtK.ts";
import { RunCode } from "../Brain/Eval/CodeExecutor.ts";
import { EvaluateProblem } from "../Brain/Eval/EvalHarness.ts";
import { CollectPassing } from "../Brain/Rl/RejectionSampling.ts";
import { CodingProblems } from "../Brain/Eval/ProblemSet.ts";

test("PassAtK matches known values", () => {
  expect(PassAtK(5, 0, 1)).toBe(0);
  expect(PassAtK(5, 5, 1)).toBe(1);
  expect(PassAtK(2, 1, 1)).toBeCloseTo(0.5, 10);
  expect(PassAtK(4, 2, 2)).toBeCloseTo(1 - 1 / 6, 10); // 1 - C(2,2)/C(4,2)
});

test("CodeExecutor runs passing/failing code and enforces a timeout", () => {
  expect(RunCode("if (1 + 1 !== 2) throw new Error('bad');").Passed).toBe(true);
  expect(RunCode("throw new Error('boom');").Passed).toBe(false);
  const Timed = RunCode("while (true) {}", 800);
  expect(Timed.Passed).toBe(false); // killed by the timeout
}, 20000);

test("EvaluateProblem and CollectPassing count correct candidates", () => {
  const Problem = { Name: "add", Tests: "if (add(2, 3) !== 5) throw new Error('fail');" };
  const Good = "function add(a, b) { return a + b; }";
  const Bad = "function add(a, b) { return a - b; }";
  const Result = EvaluateProblem(Problem, [Good, Bad, Good]);
  expect(Result.Correct).toBe(2);
  expect(Result.PassAt1).toBeCloseTo(PassAtK(3, 2, 1), 10);
  expect(CollectPassing(Problem, [Good, Bad]).length).toBe(1);
  expect(CollectPassing(Problem, [Good, Good, Good]).length).toBe(1); // distinct passers only (dedup)
}, 20000);

test("every ProblemSet reference solution passes its own tests (genuine executable ground truth)", () => {
  for (const P of CodingProblems) {
    expect(RunCode(`${P.Reference}\n${P.Tests}`).Passed).toBe(true);
  }
}, 60000);

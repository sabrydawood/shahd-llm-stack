import { test, expect } from "bun:test";
import { CreateGsmProvider, GsmRowToDoc } from "../Foundry/GsmProvider.ts";
import type { SourceInput } from "../Foundry/FoundryBarrel.ts";
import { ExtractAnswer, NormalizeAnswer } from "../Brain/Reasoning/ReasoningBarrel.ts";

const Row = {
  question: "Janet has 16 eggs. She eats 3 and bakes with 4. She sells the rest at $2 each. How much does she make?",
  answer: "She has 16 - 3 - 4 = <<16-3-4=9>>9 eggs left.\nShe makes 9 * 2 = $<<9*2=18>>18.\n#### 18",
};

test("GsmRowToDoc formats a problem into the canonical think/answer shape the reasoning infra reads", () => {
  const Doc = GsmRowToDoc(Row, "gsm8k:train:0")!;
  expect(Doc).not.toBeNull();
  expect(Doc.Source).toBe("gsm8k");
  expect(Doc.License).toBe("MIT");
  expect(Doc.Content).toContain("<|think|>");
  expect(Doc.Content).toContain("<|endthink|>");
  expect(Doc.Content).not.toContain("<<"); // calculator annotations stripped
  // The final number lands where ExtractAnswer/NormalizeAnswer can read it — so STaR/self-consistency
  // can VERIFY it, not just train on it. This is the whole point of the canonical shape.
  const AssistantTurn = Doc.Content.split("Assistant:")[1]!;
  expect(NormalizeAnswer(ExtractAnswer(AssistantTurn))).toBe("18");
});

test("GsmRowToDoc rejects malformed rows (no question / no #### marker)", () => {
  expect(GsmRowToDoc({ answer: "#### 5" }, "p")).toBeNull(); // no question
  expect(GsmRowToDoc({ question: "q", answer: "no final marker here" }, "p")).toBeNull(); // no ####
  expect(GsmRowToDoc({ question: "q", answer: "reason\n#### " }, "p")).toBeNull(); // empty final
});

test("GSM8K provider streams reasoning docs in batches (injected fetch, no network)", async () => {
  const Jsonl = [JSON.stringify(Row), JSON.stringify({ ...Row, question: "Q2 " + Row.question }), ""].join("\n");
  const Batches: SourceInput[] = [];
  const Provider = CreateGsmProvider({ FetchText: async () => Jsonl, BatchSize: 10, OnRepoReady: async (_S, Docs) => { Batches.push(...Docs); } });
  expect(Provider.Semantics).toBe("bounded");
  const Ret = await Provider.Fetch("train", 100);
  expect(Ret.length).toBe(0); // streaming mode
  expect(Batches.length).toBe(2);
  expect(Batches.every((D) => D.Lang === "text-en" && D.Origin === "curated")).toBe(true);
});

// GSM8K reasoning provider (data engine, Phase 2) — the first source aimed squarely at THINKING, not
// just talking or reading. GSM8K is grade-school math word problems with step-by-step solutions and a
// checkable final number; it is MIT-licensed (openai/grade-school-math) and downloaded as plain JSONL
// from GitHub raw (datasets-server is unreachable here; the raw file is not). Each problem is stored in
// the model's OWN canonical reasoning shape — User question, then an Assistant turn with the reasoning
// inside <|think|>…<|endthink|> and the final number inside <answer>…</answer> — so it plugs directly
// into the SFT template, ExtractAnswer/NormalizeAnswer, and the STaR/ProblemSet verifier (which lacked
// real verifiable problems at toy scale). kind = "instruction". The fetch is injected (mock in tests).

import type { WebProvider, RepoSink } from "./WebSource.ts";
import type { SourceInput } from "./Ingest.ts";
import { FetchWithBackoff, HttpError } from "./HttpBackoff.ts";
import { WrapThinkingAnswer } from "../Brain/Reasoning/ReasoningBarrel.ts";

// openai/grade-school-math is MIT. The socratic variant restructures the same problems; we take the
// standard train/test splits (question + worked answer ending in "#### <number>").
const RawBase = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data";
export const GsmSplits = { train: `${RawBase}/train.jsonl`, test: `${RawBase}/test.jsonl` } as const;

export type GsmOptions = {
  FetchText?: (Url: string) => Promise<string>; // injected in tests; real GitHub-raw fetch by default
  BatchSize?: number;
  Sleep?: (Ms: number) => Promise<void>; // injected in tests so backoff never really waits
  OnRepoStart?: (Name: string) => void;
  OnRepoReady?: RepoSink;
  Log?: (Message: string) => void;
};

type GsmRow = { question?: string; answer?: string };

async function DefaultFetchText(Url: string, Sleep?: (Ms: number) => Promise<void>): Promise<string> {
  return FetchWithBackoff(
    async () => {
      const Response = await fetch(Url, { headers: { "User-Agent": "shahd-foundry" } });
      if (!Response.ok) throw new HttpError(Response.status);
      return Response.text();
    },
    { Sleep },
  );
}

// Turn one GSM8K row into a canonical-format reasoning document. The worked answer carries inline
// calculator annotations like "<<16-3-4=9>>" and ends with "#### 9"; we strip the annotations, split off
// the final number, and rebuild the assistant turn as <|think|>reasoning<|endthink|><answer>9</answer>.
// Returns null for a malformed row (missing question/answer or no final marker).
export function GsmRowToDoc(Row: GsmRow, Provenance: string): SourceInput | null {
  const Question = typeof Row.question === "string" ? Row.question.trim() : "";
  const Answer = typeof Row.answer === "string" ? Row.answer : "";
  if (Question === "" || Answer === "") return null;
  const HashIdx = Answer.lastIndexOf("####");
  if (HashIdx === -1) return null; // no checkable final answer
  const Final = Answer.slice(HashIdx + 4).trim();
  const Reasoning = Answer.slice(0, HashIdx).replace(/<<[^>]*>>/g, "").trim(); // drop calc annotations
  if (Final === "" || Reasoning === "") return null;
  return {
    Source: "gsm8k",
    License: "MIT",
    Lang: "text-en",
    Content: `User: ${Question}\n\nAssistant: ${WrapThinkingAnswer(Reasoning, Final)}`,
    Provenance,
    Origin: "curated", // an MIT dataset we vetted — quality-gated, license recorded not re-checked
  };
}

export function CreateGsmProvider(Options: GsmOptions = {}): WebProvider {
  const FetchText = Options.FetchText ?? ((Url: string): Promise<string> => DefaultFetchText(Url, Options.Sleep));
  const BatchSize = Options.BatchSize ?? 200;
  const Log = Options.Log ?? ((Message: string): void => console.log(Message));

  return {
    Name: "gsm8k",
    Semantics: "bounded", // a fixed dataset — a full collect exhausts it; re-runs only dedup
    Fetch: async (Query: string, Limit: number): Promise<SourceInput[]> => {
      // Query selects the split: "train" (default), "test", or "all".
      const Which = (Query || "train").trim().toLowerCase();
      const Splits = Which === "test" ? (["test"] as const) : Which === "all" ? (["train", "test"] as const) : (["train"] as const);

      const Docs: SourceInput[] = [];
      for (const Split of Splits) {
        if (Docs.length >= Limit) break;
        Options.OnRepoStart?.(`GSM8K ${Split}`); // working signal + Stop boundary before the download
        Log(`[gsm8k] downloading ${Split} split…`);
        const Text = await FetchText(GsmSplits[Split]);
        const Lines = Text.split("\n");
        Log(`[gsm8k] ${Split}: ${Lines.length} rows; formatting reasoning docs (max ${Limit})…`);
        for (let I = 0; I < Lines.length; I++) {
          if (Docs.length >= Limit) break;
          const Trimmed = Lines[I]!.trim();
          if (Trimmed.length === 0) continue;
          try {
            const Doc = GsmRowToDoc(JSON.parse(Trimmed) as GsmRow, `gsm8k:${Split}:${I}`);
            if (Doc !== null) Docs.push(Doc);
          } catch {
            // a malformed line is skipped, never fatal
          }
        }
      }
      Log(`[gsm8k] formatted ${Docs.length} reasoning docs`);

      if (Options.OnRepoReady === undefined) return Docs; // batch mode (CLI): return all
      for (let Start = 0; Start < Docs.length; Start += BatchSize) {
        Options.OnRepoStart?.(`GSM8K batch ${Math.floor(Start / BatchSize) + 1}`); // Stop between batches
        await Options.OnRepoReady(`gsm8k-${Math.floor(Start / BatchSize)}`, Docs.slice(Start, Start + BatchSize));
      }
      Log(`[gsm8k] done: ${Docs.length} reasoning docs stored`);
      return [];
    },
  };
}

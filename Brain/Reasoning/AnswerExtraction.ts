// Answer extraction + normalization (Phase 8, reasoning). The canonical answer format is
// <answer>…</answer>: the model is prompted (see DefaultThinkingSystemPrompt) to think inside
// <|think|>…<|endthink|> and then put its FINAL answer in an <answer> span. These helpers pull that
// answer out and canonicalize it so that semantically-identical answers ("42", "42.", " 42\n") collapse
// to the same key — which is exactly what self-consistency voting needs to count agreeing samples.
// Without a canonical form, majority voting counts trivially-different formatting as different answers
// and its accuracy gain evaporates.

import { StripThinking, WrapThinking } from "./ThinkingMode.ts";

export const AnswerTags = { Open: "<answer>", Close: "</answer>" } as const;

// The ONE canonical system prompt that ties training to serving: SFT data is generated with it and the
// serving agent renders with it, so the model is asked to do at inference exactly what it was taught —
// think inside <|think|>…<|endthink|>, then give the final answer inside <answer>…</answer> (which
// ExtractAnswer parses). Keeping this in one place is what makes A2 (prompt) / A3 (extract) / A4 (vote)
// consistent instead of silently drifting apart.
export const DefaultThinkingSystemPrompt =
  "You are Shahd, a helpful coding assistant. Think step by step inside <|think|>…<|endthink|>, then give your final answer inside <answer>…</answer>.";

/** Wrap reasoning + answer in the canonical think+answer format the model is trained on and the
 *  extractor parses: <|think|>reasoning<|endthink|><answer>answer</answer>. */
export function WrapThinkingAnswer(Thinking: string, Answer: string): string {
  return WrapThinking(Thinking, `${AnswerTags.Open}${Answer}${AnswerTags.Close}`);
}

/** Pull the final answer out of a generation: the <answer>…</answer> span if present (thinking removed
 *  first so the scratchpad is never mistaken for the answer), else a sensible fallback — the last
 *  non-empty line (the common "final answer on its own line" shape), else the whole cleaned text. */
export function ExtractAnswer(Text: string): string {
  const Clean = StripThinking(Text);
  const Start = Clean.indexOf(AnswerTags.Open);
  if (Start !== -1) {
    const From = Start + AnswerTags.Open.length;
    const End = Clean.indexOf(AnswerTags.Close, From);
    return (End === -1 ? Clean.slice(From) : Clean.slice(From, End)).trim();
  }
  const Lines = Clean.split(/\r?\n/).map((L) => L.trim()).filter((L) => L.length > 0);
  return Lines.length > 0 ? Lines[Lines.length - 1]! : Clean.trim();
}

/** Canonicalize an answer for voting: trim, collapse whitespace, lowercase, drop trailing sentence
 *  punctuation, and reduce a pure number to a stable numeric form ("42.0"/"+42"/"1,000" -> "42"/"1000"). */
export function NormalizeAnswer(Answer: string): string {
  const A = Answer.trim().toLowerCase().replace(/\s+/g, " ").replace(/[.!?;:,]+$/g, "").trim();
  if (A === "") return A;
  if (/^[+\-]?[\d,]*\.?\d+$/.test(A)) {
    const Num = Number(A.replace(/,/g, ""));
    if (Number.isFinite(Num)) return String(Num);
  }
  return A;
}

/** Extract then normalize — the voting key used by self-consistency (KeyOf). */
export function AnswerKey(Text: string): string {
  return NormalizeAnswer(ExtractAnswer(Text));
}

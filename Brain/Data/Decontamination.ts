// Eval-set decontamination (Phase 3). If training docs share a long exact n-gram with any held-out
// eval doc, the eval number measures memorization, not generalization — so drop those train docs
// BEFORE training (REVIEW.md: contaminated held-out perplexity makes the Phase-3 go/no-go bogus).

import { CodePretokenize } from "../Tokenizer/CodePretokenizer.ts";

// Code-aware windows (not a naive whitespace split): a code doc's real token boundaries are
// identifiers/numbers/punctuation runs, so this matches the granularity contamination actually
// leaks at (e.g. renamed-variable near-copies still share the same punctuation/number n-grams).
function WordNgrams(Text: string, N: number): Set<string> {
  const Toks = CodePretokenize(Text.toLowerCase())
    .map((T) => T.trim())
    .filter((T) => T.length > 0);
  const Out = new Set<string>();
  for (let I = 0; I + N <= Toks.length; I++) Out.add(Toks.slice(I, I + N).join(" "));
  return Out;
}

export type DecontaminationResult = { Kept: number[]; Removed: number[] };

/** Remove train docs sharing any N-word n-gram with the eval set. */
export function Decontaminate(TrainDocs: string[], EvalDocs: string[], NgramSize = 13): DecontaminationResult {
  const EvalNgrams = new Set<string>();
  for (const Doc of EvalDocs) {
    for (const Gram of WordNgrams(Doc, NgramSize)) EvalNgrams.add(Gram);
  }

  const Kept: number[] = [];
  const Removed: number[] = [];
  for (let I = 0; I < TrainDocs.length; I++) {
    let Contaminated = false;
    for (const Gram of WordNgrams(TrainDocs[I], NgramSize)) {
      if (EvalNgrams.has(Gram)) {
        Contaminated = true;
        break;
      }
    }
    if (Contaminated) Removed.push(I);
    else Kept.push(I);
  }
  return { Kept, Removed };
}

// Split a corpus into train/val. Two strategies:
//   • TrainValSplit — a positional cut on an already-encoded flat stream. Correct ONLY for a single
//     document (Phase-0/1 smoke); on a multi-document corpus it leaks (a doc can straddle the cut, and
//     val is the unshuffled tail). Kept for the single-document/basic path.
//   • SplitAndEncodeDocuments — the REAL multi-document path: shuffle documents, split at the DOCUMENT
//     boundary, and encode each side separately with a hard EOS token between documents. This removes
//     the leak AND gives the model an unambiguous document-boundary signal (not a soft "\n\n").

import type { SeededRng } from "../Random/SeededRng.ts";

export type SplitData = { Train: number[]; Val: number[] };

export function TrainValSplit(Data: number[], ValFraction: number): SplitData {
  if (ValFraction <= 0 || ValFraction >= 1) {
    throw new Error(`TrainValSplit: ValFraction must be in (0,1), got ${ValFraction}`);
  }
  const Cut = Math.floor(Data.length * (1 - ValFraction));
  return { Train: Data.slice(0, Cut), Val: Data.slice(Cut) };
}

/** Fisher-Yates shuffle of a COPY, driven by the given RNG stream (deterministic given the seed). */
export function ShuffleCopy<T>(Items: readonly T[], Rng: SeededRng): T[] {
  const Out = [...Items];
  for (let I = Out.length - 1; I > 0; I--) {
    const J = Math.floor(Rng.NextFloat() * (I + 1));
    const Tmp = Out[I];
    Out[I] = Out[J];
    Out[J] = Tmp;
  }
  return Out;
}

// Encode a list of documents into one token stream, appending a hard EOS token id after EACH document
// so the model sees an unambiguous boundary (not a soft, BPE-mergeable "\n\n" it has to infer).
function EncodeWithEos(Documents: readonly string[], Encode: (Text: string) => number[], EosId: number): number[] {
  const Out: number[] = [];
  for (const Doc of Documents) {
    for (const Id of Encode(Doc)) Out.push(Id);
    Out.push(EosId);
  }
  return Out;
}

/**
 * Split documents at the DOCUMENT level (shuffled first), then encode each split SEPARATELY with an EOS
 * token between documents. Fixes two real defects at once:
 *   • No positional train/val LEAK — a document can never straddle the cut, and val is a random sample of
 *     WHOLE documents, not the unshuffled tail (which skews toward whatever source sorts last).
 *   • A real document-boundary signal (EOS) instead of a soft "\n\n" the model must guess at.
 * The tokenizer/Encode must already cover every document (build it over ALL docs first — vocab coverage
 * is not leakage). Requires >= 2 documents (one for each side); both sides are guaranteed non-empty.
 */
export function SplitAndEncodeDocuments(
  Documents: readonly string[],
  ValFraction: number,
  Rng: SeededRng,
  Encode: (Text: string) => number[],
  EosId: number,
): SplitData {
  if (ValFraction <= 0 || ValFraction >= 1) {
    throw new Error(`SplitAndEncodeDocuments: ValFraction must be in (0,1), got ${ValFraction}`);
  }
  if (Documents.length < 2) {
    throw new Error(`SplitAndEncodeDocuments: need >= 2 documents for a document-level split, got ${Documents.length}`);
  }
  const Shuffled = ShuffleCopy(Documents, Rng);
  const Cut = Math.min(Documents.length - 1, Math.max(1, Math.floor(Documents.length * (1 - ValFraction))));
  return {
    Train: EncodeWithEos(Shuffled.slice(0, Cut), Encode, EosId),
    Val: EncodeWithEos(Shuffled.slice(Cut), Encode, EosId),
  };
}

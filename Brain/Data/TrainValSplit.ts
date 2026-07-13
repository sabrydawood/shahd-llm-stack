// Split an encoded corpus into train/val. This positional split is correct for a single
// document (Phase-0/1 smoke). For a REAL multi-file code corpus (Phase 3), splitting must be
// file/repo-level with near-duplicate removal BEFORE the split, or the val curve measures
// memorization, not generalization (REVIEW.md) — that lives in the Phase-3 data pipeline.

export type SplitData = { Train: number[]; Val: number[] };

export function TrainValSplit(Data: number[], ValFraction: number): SplitData {
  if (ValFraction <= 0 || ValFraction >= 1) {
    throw new Error(`TrainValSplit: ValFraction must be in (0,1), got ${ValFraction}`);
  }
  const Cut = Math.floor(Data.length * (1 - ValFraction));
  return { Train: Data.slice(0, Cut), Val: Data.slice(Cut) };
}

// Next-token selection from a single logits row, with the full set of sampling controls:
// temperature (0 => greedy/argmax), top-k, and top-p (nucleus). Sampling draws from the
// SamplingRng stream. These are MECHANISMS — buildable at any scale; output quality depends on
// what training put in the weights (CAPABILITIES.md).

import type { SeededRng } from "../Random/SeededRng.ts";

export type SamplingOptions = {
  Temperature: number; // <= 0 => greedy argmax
  TopK: number; // 0 => disabled
  TopP: number; // 1 => disabled
};

export const DefaultSampling: SamplingOptions = { Temperature: 1, TopK: 0, TopP: 1 };

/** Pick a next-token id from Logits[Offset .. Offset+VocabSize) using the given options + RNG. */
export function SampleFromLogits(
  Logits: Float64Array,
  Offset: number,
  VocabSize: number,
  Options: SamplingOptions,
  Rng: SeededRng,
): number {
  if (Options.Temperature <= 0) {
    let Best = 0;
    let BestVal = -Infinity;
    for (let J = 0; J < VocabSize; J++) {
      const Val = Logits[Offset + J];
      if (Val > BestVal) {
        BestVal = Val;
        Best = J;
      }
    }
    return Best;
  }

  // Temperature-scaled softmax (max-subtracted).
  const Probs = new Float64Array(VocabSize);
  let Max = -Infinity;
  for (let J = 0; J < VocabSize; J++) {
    const Scaled = Logits[Offset + J] / Options.Temperature;
    Probs[J] = Scaled;
    if (Scaled > Max) Max = Scaled;
  }
  let Sum = 0;
  for (let J = 0; J < VocabSize; J++) {
    const E = Math.exp(Probs[J] - Max);
    Probs[J] = E;
    Sum += E;
  }
  for (let J = 0; J < VocabSize; J++) Probs[J] /= Sum;

  // Candidate ids sorted by probability (descending), then top-k, then top-p.
  let Indices: number[] = [];
  for (let J = 0; J < VocabSize; J++) Indices.push(J);
  Indices.sort((A, B) => Probs[B] - Probs[A]);

  if (Options.TopK > 0 && Options.TopK < Indices.length) Indices = Indices.slice(0, Options.TopK);

  if (Options.TopP < 1) {
    const Kept: number[] = [];
    let Cumulative = 0;
    for (const Idx of Indices) {
      Kept.push(Idx);
      Cumulative += Probs[Idx];
      if (Cumulative >= Options.TopP) break;
    }
    Indices = Kept;
  }

  // Renormalize over the kept set and sample.
  let KeptSum = 0;
  for (const Idx of Indices) KeptSum += Probs[Idx];
  let R = Rng.NextFloat() * KeptSum;
  for (const Idx of Indices) {
    R -= Probs[Idx];
    if (R <= 0) return Idx;
  }
  return Indices[Indices.length - 1];
}

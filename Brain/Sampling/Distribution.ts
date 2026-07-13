// The next-token probability distribution, factored out so both the sampler and speculative sampling
// share ONE implementation of temperature + top-k + top-p (rule #4). ProbsFromLogits returns a
// full-vocab distribution (zeros outside the kept set, renormalized over it); SampleFromDistribution
// draws from it in descending-probability order (the reference sampler's order, so behavior is
// unchanged). SamplingOptions lives here (the lowest layer that needs it); Sampler re-exports it.

import type { SeededRng } from "../Random/SeededRng.ts";

export type SamplingOptions = {
  Temperature: number; // <= 0 => greedy argmax (handled by the caller, not here)
  TopK: number; // 0 => disabled
  TopP: number; // 1 => disabled
};

/** Temperature-scaled softmax with top-k then top-p, renormalized over the kept set. */
export function ProbsFromLogits(Logits: Float64Array, Offset: number, VocabSize: number, Options: SamplingOptions): Float64Array {
  const Temperature = Options.Temperature > 0 ? Options.Temperature : 1;
  const Probs = new Float64Array(VocabSize);
  let Max = -Infinity;
  for (let J = 0; J < VocabSize; J++) {
    const Scaled = Logits[Offset + J] / Temperature;
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

  const Out = new Float64Array(VocabSize);
  let KeptSum = 0;
  for (const Idx of Indices) KeptSum += Probs[Idx];
  if (KeptSum > 0) for (const Idx of Indices) Out[Idx] = Probs[Idx] / KeptSum;
  return Out;
}

/** Sample an index from a probability vector (descending-probability walk; one RNG draw). */
export function SampleFromDistribution(Probs: Float64Array, Rng: SeededRng): number {
  const Indices: number[] = [];
  for (let J = 0; J < Probs.length; J++) if (Probs[J] > 0) Indices.push(J);
  Indices.sort((A, B) => Probs[B] - Probs[A]);
  let R = Rng.NextFloat();
  for (const Idx of Indices) {
    R -= Probs[Idx];
    if (R <= 0) return Idx;
  }
  return Indices.length > 0 ? Indices[Indices.length - 1] : 0;
}

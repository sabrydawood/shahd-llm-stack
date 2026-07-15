// Next-token selection from a single logits row. Temperature (0 => greedy/argmax), top-k, and top-p
// (nucleus). The stochastic path delegates to the shared Distribution helper so the exact same
// temperature/top-k/top-p logic backs both sampling and speculative sampling (rule #4). These are
// MECHANISMS — output quality depends on what training put in the weights (CAPABILITIES.md).

import type { SeededRng } from "../Random/SeededRng.ts";
import type { SamplingOptions } from "./Distribution.ts";
import { ArgmaxOf, ProbsFromLogits, SampleFromDistribution } from "./Distribution.ts";

export type { SamplingOptions } from "./Distribution.ts";

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
    return ArgmaxOf(Logits, Offset, VocabSize);
  }
  return SampleFromDistribution(ProbsFromLogits(Logits, Offset, VocabSize, Options), Rng);
}

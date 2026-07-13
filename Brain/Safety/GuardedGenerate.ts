// The SAFE generation entry point for product use. Enforces, in one controllable place:
//   1. Safety on the prompt (block harmful requests before spending any compute).
//   2. Performance/resource Limits (cap new tokens + context length).
//   3. Safety on the generated output.
// Raw Generate stays the bare mechanism; product/serving code should call this instead.

import type { Shahd } from "../Nn/Shahd.ts";
import type { Tokenizer } from "../Tokenizer/TokenizerTypes.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import type { SamplingOptions } from "../Sampling/Sampler.ts";
import { Generate } from "../Sampling/Generate.ts";
import { SafetyPolicy } from "./SafetyPolicy.ts";

export function GuardedGenerate(
  Model: Shahd,
  Tokenizer: Tokenizer,
  Prompt: string,
  MaxNewTokens: number,
  Options: SamplingOptions,
  Rng: SeededRng,
  Config: ResolvedConfig,
): string {
  const Policy = new SafetyPolicy(Config);

  // 1. Safety on the prompt.
  Policy.EnforceInput(Prompt);

  // 2. Resource limits: cap generation length and context.
  const CappedNewTokens = Math.min(MaxNewTokens, Config.Limits.MaxNewTokens);
  let PromptIds = Tokenizer.Encode(Prompt);
  if (PromptIds.length > Config.Limits.MaxContextTokens) {
    PromptIds = PromptIds.slice(-Config.Limits.MaxContextTokens);
  }

  const GeneratedIds = Generate(Model, PromptIds, CappedNewTokens, Options, Rng);
  const Text = Tokenizer.Decode(GeneratedIds);

  // 3. Safety on the output.
  Policy.EnforceOutput(Text);

  return Text;
}

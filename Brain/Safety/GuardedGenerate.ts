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
import { Generate, GenerateAsync } from "../Sampling/Generate.ts";
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

/**
 * Streaming sibling of GuardedGenerate: emits the completion as decoded deltas via OnDelta as tokens
 * are produced, and returns the full completion (the NEW text only, excluding the prompt). Prompt
 * safety + resource caps are enforced up front. Output safety is enforced INCREMENTALLY — the
 * growing text is vetted BEFORE each delta is emitted, so blocked content is never delivered (the
 * stream stops at the first token that trips the guard, matching the block-before-delivery guarantee
 * of the non-streaming path). A trailing U+FFFD from a not-yet-complete multibyte (byte-level BPE)
 * token is held back until its bytes arrive. ShouldStop lets a caller abort early (e.g. the client
 * disconnected) so no compute is wasted generating for nobody.
 */
export async function GuardedGenerateStream(
  Model: Shahd,
  Tokenizer: Tokenizer,
  Prompt: string,
  MaxNewTokens: number,
  Options: SamplingOptions,
  Rng: SeededRng,
  Config: ResolvedConfig,
  OnDelta: (Delta: string) => void,
  ShouldStop?: () => boolean,
): Promise<string> {
  const Policy = new SafetyPolicy(Config);
  Policy.EnforceInput(Prompt);

  const CappedNewTokens = Math.min(MaxNewTokens, Config.Limits.MaxNewTokens);
  let PromptIds = Tokenizer.Encode(Prompt);
  if (PromptIds.length > Config.Limits.MaxContextTokens) {
    PromptIds = PromptIds.slice(-Config.Limits.MaxContextTokens);
  }

  const Incomplete = String.fromCharCode(0xfffd); // U+FFFD: a not-yet-complete multibyte char
  const NewIds: number[] = [];
  let Emitted = 0;
  const EmitVetted = (Text: string): void => {
    if (Text.length <= Emitted) return;
    Policy.EnforceOutput(Prompt + Text); // vet BEFORE delivering — throws (and aborts) if unsafe
    OnDelta(Text.slice(Emitted));
    Emitted = Text.length;
  };

  await GenerateAsync(
    Model,
    PromptIds,
    CappedNewTokens,
    Options,
    Rng,
    (Id: number): void => {
      NewIds.push(Id);
      let Stable = Tokenizer.Decode(NewIds);
      while (Stable.endsWith(Incomplete)) Stable = Stable.slice(0, -1); // hold back an incomplete char
      EmitVetted(Stable);
    },
    ShouldStop,
  );

  const Completion = Tokenizer.Decode(NewIds);
  Policy.EnforceOutput(Prompt + Completion); // final full-text vet (unconditional), then flush the tail
  if (Completion.length > Emitted) OnDelta(Completion.slice(Emitted));
  return Completion;
}

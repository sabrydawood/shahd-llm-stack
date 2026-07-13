// Autoregressive generation loop (tape off — no autograd graph). Recomputes the forward pass
// over the trailing BlockSize context each step. A KV-cache (KvCache.ts) will make this
// incremental; for now this is correct-but-O(T) per token, matching nano-gpt.ts.

import type { Shahd } from "../Nn/Shahd.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import type { SamplingOptions } from "./Sampler.ts";
import { WithTapeOff } from "../Tensor/Tape.ts";
import { SampleFromLogits } from "./Sampler.ts";

export function Generate(
  Model: Shahd,
  PromptIds: number[],
  MaxNewTokens: number,
  Options: SamplingOptions,
  Rng: SeededRng,
): number[] {
  return WithTapeOff(() => {
    const Ids = [...PromptIds];
    const BlockSize = Model.Config.Model.BlockSize;
    const VocabSize = Model.Config.Model.VocabSize;
    for (let S = 0; S < MaxNewTokens; S++) {
      const Context = Ids.slice(-BlockSize);
      const Logits = Model.Forward(Context);
      const LastRowOffset = (Logits.Rows - 1) * VocabSize;
      Ids.push(SampleFromLogits(Logits.Data, LastRowOffset, VocabSize, Options, Rng));
    }
    return Ids;
  });
}

// Autoregressive generation loop (tape off — no autograd graph). Recomputes the forward pass
// over the trailing BlockSize context each step. Two entry points share one per-token step (DRY):
//   Generate       — synchronous, returns all ids (training samples, tests, non-streaming serving).
//   GenerateAsync  — yields to the event loop between tokens so buffered SSE/WebSocket sends flush,
//                    enabling REAL token-by-token streaming; fires OnToken as each token is produced.

import type { Shahd } from "../Nn/Shahd.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import type { SamplingOptions } from "./Sampler.ts";
import { WithTapeOff } from "../Tensor/Tape.ts";
import { SampleFromLogits } from "./Sampler.ts";

// One decode step: forward the trailing context and sample the next token id. Tape off (inference).
function StepToken(Model: Shahd, Ids: number[], BlockSize: number, VocabSize: number, Options: SamplingOptions, Rng: SeededRng): number {
  return WithTapeOff(() => {
    const Context = Ids.slice(-BlockSize);
    const Logits = Model.Forward(Context);
    const LastRowOffset = (Logits.Rows - 1) * VocabSize;
    return SampleFromLogits(Logits.Data, LastRowOffset, VocabSize, Options, Rng);
  });
}

export function Generate(
  Model: Shahd,
  PromptIds: number[],
  MaxNewTokens: number,
  Options: SamplingOptions,
  Rng: SeededRng,
): number[] {
  const Ids = [...PromptIds];
  const BlockSize = Model.Config.Model.BlockSize;
  const VocabSize = Model.Config.Model.VocabSize;
  for (let S = 0; S < MaxNewTokens; S++) {
    Ids.push(StepToken(Model, Ids, BlockSize, VocabSize, Options, Rng));
  }
  return Ids;
}

export async function GenerateAsync(
  Model: Shahd,
  PromptIds: number[],
  MaxNewTokens: number,
  Options: SamplingOptions,
  Rng: SeededRng,
  OnToken: (Id: number) => void, // fired after each newly sampled token
): Promise<number[]> {
  const Ids = [...PromptIds];
  const BlockSize = Model.Config.Model.BlockSize;
  const VocabSize = Model.Config.Model.VocabSize;
  for (let S = 0; S < MaxNewTokens; S++) {
    const Next = StepToken(Model, Ids, BlockSize, VocabSize, Options, Rng);
    Ids.push(Next);
    OnToken(Next);
    // Yield a macrotask so buffered stream writes reach the client between tokens (real streaming).
    await new Promise<void>((Resolve) => setTimeout(Resolve, 0));
  }
  return Ids;
}

// Batch source for training/eval. GetSequence returns one (Ids, Targets) window of length
// BlockSize (Targets = Ids shifted by one), sampled with the DataRng stream. The grad-
// accumulation loop calls this BatchSize times per optimizer step. Named GetSequence (not
// GetBatch) because it returns a single sequence, per REVIEW.md.

import type { SeededRng } from "../Random/SeededRng.ts";

export type Sequence = { Ids: number[]; Targets: number[] };

export interface DataLoader {
  GetSequence(): Sequence;
}

export class InMemoryDataLoader implements DataLoader {
  private Data: number[];
  private BlockSize: number;
  private Rng: SeededRng;

  constructor(Data: number[], BlockSize: number, Rng: SeededRng) {
    this.Data = Data;
    this.BlockSize = BlockSize;
    this.Rng = Rng;
    if (Data.length < BlockSize + 2) {
      throw new Error(`InMemoryDataLoader: corpus (${Data.length} tokens) too small for BlockSize ${BlockSize}`);
    }
  }

  GetSequence(): Sequence {
    const MaxStart = this.Data.length - this.BlockSize - 1;
    // +1 so the final valid start position (Start === MaxStart) is reachable — NextFloat() is [0, 1),
    // so without it Start could never land on MaxStart itself.
    const Start = Math.floor(this.Rng.NextFloat() * (MaxStart + 1));
    const Ids = this.Data.slice(Start, Start + this.BlockSize);
    const Targets = this.Data.slice(Start + 1, Start + 1 + this.BlockSize);
    return { Ids, Targets };
  }
}

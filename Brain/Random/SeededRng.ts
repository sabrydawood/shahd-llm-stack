// Deterministic, seedable RNG (a 32-bit LCG) with named independent streams. Using SEPARATE
// streams for init / data-sampling / dropout / generation-sampling closes the ablation-
// contamination trap from REVIEW.md: changing one concern's number of draws (e.g. adding
// layers changes how many init draws happen) must NOT shift another concern's sequence.

/** Mix a base seed with a stream index so nearby seeds produce decorrelated streams. */
function MixSeed(Seed: number, StreamIndex: number): number {
  let X = ((Seed >>> 0) ^ Math.imul(StreamIndex + 1, 0x9e3779b9)) >>> 0;
  X = Math.imul(X ^ (X >>> 16), 0x45d9f3b) >>> 0;
  X = Math.imul(X ^ (X >>> 16), 0x45d9f3b) >>> 0;
  return (X ^ (X >>> 16)) >>> 0;
}

export class SeededRng {
  private State: number;

  constructor(Seed: number) {
    this.State = Seed >>> 0;
  }

  /** Snapshot the internal state (for checkpoint save). */
  GetState(): number {
    return this.State;
  }

  /** Restore a snapshotted state (for reproducible resume). */
  SetState(State: number): void {
    this.State = State >>> 0;
  }

  /** Next 32-bit unsigned integer. */
  NextUint32(): number {
    this.State = (Math.imul(this.State, 1664525) + 1013904223) >>> 0;
    return this.State;
  }

  /** Next float in [0, 1). */
  NextFloat(): number {
    return this.NextUint32() / 4294967296;
  }

  /** Standard-normal sample via Box-Muller (guarded log to avoid -Infinity). */
  NextGaussian(): number {
    const U = this.NextFloat();
    const V = this.NextFloat();
    return Math.sqrt(-2 * Math.log(U + 1e-12)) * Math.cos(2 * Math.PI * V);
  }
}

/** The named streams threaded through training. Each is seeded from the base via MixSeed. */
export type RngStreams = {
  InitRng: SeededRng; // weight initialization
  DataRng: SeededRng; // batch sampling
  DropoutRng: SeededRng; // dropout masks (Phase 2/3)
  SamplingRng: SeededRng; // generation-time token sampling
};

export function CreateRngStreams(Seed: number): RngStreams {
  return {
    InitRng: new SeededRng(MixSeed(Seed, 0)),
    DataRng: new SeededRng(MixSeed(Seed, 1)),
    DropoutRng: new SeededRng(MixSeed(Seed, 2)),
    SamplingRng: new SeededRng(MixSeed(Seed, 3)),
  };
}

// Key/Value cache for incremental decoding: stores, per layer and head, the K and V vectors of
// every token processed so far (flat, preallocated for up to BlockSize tokens). Lets the cached
// forward attend a new token over all past tokens without recomputing their K/V.
//
// NOTE: with learned ABSOLUTE position embeddings the cache is valid only up to BlockSize tokens
// (Wpe has exactly BlockSize rows). Going beyond that cheaply is a reason to adopt RoPE in
// Phase 2 (CAPABILITIES.md). BlockSize is exactly the bound this cache preallocates to, so the two
// limits stay in lockstep by construction.

export class KvCache {
  private NumHeads: number;
  private HeadDim: number;
  private BlockSize: number;
  private Keys: Float64Array[]; // index = Layer*NumHeads + Head; buffer of BlockSize*HeadDim
  private Values: Float64Array[];
  private Lengths: number[]; // tokens appended so far, per (layer, head)

  constructor(NumLayers: number, NumHeads: number, HeadDim: number, BlockSize: number) {
    this.NumHeads = NumHeads;
    this.HeadDim = HeadDim;
    this.BlockSize = BlockSize;
    this.Keys = [];
    this.Values = [];
    this.Lengths = [];
    for (let I = 0; I < NumLayers * NumHeads; I++) {
      this.Keys.push(new Float64Array(BlockSize * HeadDim));
      this.Values.push(new Float64Array(BlockSize * HeadDim));
      this.Lengths.push(0);
    }
  }

  Append(Layer: number, Head: number, K: Float64Array, V: Float64Array): void {
    const Idx = Layer * this.NumHeads + Head;
    const Pos = this.Lengths[Idx];
    if (Pos >= this.BlockSize) throw new Error("KvCache.Append: BlockSize exhausted for this (layer, head)");
    const Base = Pos * this.HeadDim;
    const KArr = this.Keys[Idx];
    const VArr = this.Values[Idx];
    for (let D = 0; D < this.HeadDim; D++) {
      KArr[Base + D] = K[D];
      VArr[Base + D] = V[D];
    }
    this.Lengths[Idx] = Pos + 1;
  }

  GetKeys(Layer: number, Head: number): Float64Array {
    const Idx = Layer * this.NumHeads + Head;
    return this.Keys[Idx].subarray(0, this.Lengths[Idx] * this.HeadDim);
  }

  GetValues(Layer: number, Head: number): Float64Array {
    const Idx = Layer * this.NumHeads + Head;
    return this.Values[Idx].subarray(0, this.Lengths[Idx] * this.HeadDim);
  }
}

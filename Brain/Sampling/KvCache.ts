// Key/Value cache for incremental decoding: stores, per layer and head, the K and V vectors of
// every token processed so far (flat, growing by HeadDim per token). Lets the cached forward
// attend a new token over all past tokens without recomputing their K/V.
//
// NOTE: with learned ABSOLUTE position embeddings the cache is valid only up to BlockSize tokens
// (Wpe has exactly BlockSize rows). Going beyond that cheaply is a reason to adopt RoPE in
// Phase 2 (CAPABILITIES.md).

export class KvCache {
  private NumHeads: number;
  private HeadDim: number;
  private Keys: number[][]; // index = Layer*NumHeads + Head
  private Values: number[][];

  constructor(NumLayers: number, NumHeads: number, HeadDim: number) {
    this.NumHeads = NumHeads;
    this.HeadDim = HeadDim;
    this.Keys = [];
    this.Values = [];
    for (let I = 0; I < NumLayers * NumHeads; I++) {
      this.Keys.push([]);
      this.Values.push([]);
    }
  }

  Append(Layer: number, Head: number, K: Float64Array, V: Float64Array): void {
    const Idx = Layer * this.NumHeads + Head;
    const KArr = this.Keys[Idx];
    const VArr = this.Values[Idx];
    for (let D = 0; D < this.HeadDim; D++) {
      KArr.push(K[D]);
      VArr.push(V[D]);
    }
  }

  GetKeys(Layer: number, Head: number): number[] {
    return this.Keys[Layer * this.NumHeads + Head];
  }

  GetValues(Layer: number, Head: number): number[] {
    return this.Values[Layer * this.NumHeads + Head];
  }
}

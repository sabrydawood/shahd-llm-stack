// The 2D tensor + autograd node. Every op produces a Tensor that remembers its parents (Prev)
// and how to push gradient to them (BackwardFn); Backward() walks that graph in reverse.
// Data/Grad are flat Float64Array (row-major, index = Row*Cols + Col). Shape is kept
// rank-general (number[]) for the future ComputeBackend seam, but Phase-1 ops treat it as 2D
// via the Rows/Cols fields.

import { Tape } from "./Tape.ts";

function NoBackward(): void {}

export class Tensor {
  Data: Float64Array;
  Grad: Float64Array;
  Rows: number;
  Cols: number;
  readonly Shape: readonly number[];
  Prev: Tensor[];
  BackwardFn: () => void;

  constructor(Rows: number, Cols: number, Data?: Float64Array, Prev: Tensor[] = []) {
    this.Rows = Rows;
    this.Cols = Cols;
    this.Shape = [Rows, Cols];
    this.Data = Data ?? new Float64Array(Rows * Cols);
    this.Grad = new Float64Array(Rows * Cols);
    // Only retain the graph when the tape is on (skipped during sampling/eval).
    this.Prev = Tape.On ? Prev : [];
    this.BackwardFn = NoBackward;
  }

  /** Number of elements (Rows * Cols). */
  get Size(): number {
    return this.Data.length;
  }

  /** Reset this node's gradient to zero. */
  ZeroGrad(): void {
    this.Grad.fill(0);
  }
}

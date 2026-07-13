// Constructors for leaf tensors. RandN takes an explicit SeededRng (never a hidden global) so
// every weight's randomness source is visible at the call site and reproducible.

import { Tensor } from "./Tensor.ts";
import type { SeededRng } from "../Random/SeededRng.ts";

export function Zeros(Rows: number, Cols: number): Tensor {
  return new Tensor(Rows, Cols);
}

export function Filled(Rows: number, Cols: number, Value: number): Tensor {
  const Result = new Tensor(Rows, Cols);
  Result.Data.fill(Value);
  return Result;
}

export function Ones(Rows: number, Cols: number): Tensor {
  return Filled(Rows, Cols, 1);
}

/** Gaussian(0, Scale^2) weights drawn from the given RNG stream (typically InitRng). */
export function RandN(Rows: number, Cols: number, Scale: number, Rng: SeededRng): Tensor {
  const Result = new Tensor(Rows, Cols);
  const Data = Result.Data;
  for (let I = 0; I < Data.length; I++) {
    Data[I] = Rng.NextGaussian() * Scale;
  }
  return Result;
}

/** Wrap an existing flat buffer (row-major) as a tensor. The buffer is used as-is (not copied). */
export function FromArray(Rows: number, Cols: number, Data: Float64Array): Tensor {
  if (Data.length !== Rows * Cols) {
    throw new Error(`FromArray: buffer length ${Data.length} != ${Rows}*${Cols}`);
  }
  return new Tensor(Rows, Cols, Data);
}

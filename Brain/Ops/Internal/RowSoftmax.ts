// Shared row-wise softmax math (max-subtract/exp/sum/normalize) used by the FORWARD pass of
// SoftmaxRows, CrossEntropy, and MaskedCrossEntropy. Internal helper only — not re-exported from
// OpsBarrel.ts. Each op keeps its own distinct BackwardFn (rule #4: this is pure plumbing, not a
// gradient mechanism, so centralizing it does not violate the "keep different backward math
// separate" rule).

import type { NumArray } from "../../Tensor/Tensor.ts";

/** Softmax one row of `Input` (length N, starting at InputOffset) into `Output` at OutputOffset.
 *  Both buffers arrive in their owner's storage precision; the exp/sum math itself runs in f64. */
export function ComputeRowSoftmax(
  Input: NumArray,
  InputOffset: number,
  Output: NumArray,
  OutputOffset: number,
  N: number,
): void {
  let Max = -Infinity;
  for (let J = 0; J < N; J++) Max = Math.max(Max, Input[InputOffset + J]);
  let Sum = 0;
  for (let J = 0; J < N; J++) {
    const E = Math.exp(Input[InputOffset + J] - Max);
    Output[OutputOffset + J] = E;
    Sum += E;
  }
  for (let J = 0; J < N; J++) Output[OutputOffset + J] /= Sum;
}

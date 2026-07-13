// Finite-difference gradient checker — the REAL numerical correctness oracle for the engine
// (REVIEW.md L5). For each input element it compares the analytic gradient (from Backward)
// against a central finite difference of the loss. This is the standing gate every new op must
// pass before it is trusted, because a subtly-wrong backward still lets the loss go down.

import type { Tensor } from "../Tensor/Tensor.ts";
import { Backward } from "./Backward.ts";
import { WithTapeOff } from "../Tensor/Tape.ts";

export type GradCheckResult = {
  MaxAbsError: number;
  MaxRelError: number;
  Passed: boolean;
};

export type GradCheckOptions = {
  Epsilon?: number; // perturbation size (default 1e-6, tuned for f64)
  Tolerance?: number; // pass if MaxRelError <= Tolerance OR MaxAbsError <= Tolerance
};

/**
 * Check the gradients of `Inputs` for the scalar loss produced by `Forward`.
 * `Forward` must rebuild the graph from the (mutable) `Inputs` each call and return a [1,1] loss.
 */
export function GradCheck(
  Inputs: Tensor[],
  Forward: () => Tensor,
  Options: GradCheckOptions = {},
): GradCheckResult {
  const Epsilon = Options.Epsilon ?? 1e-6;
  const Tolerance = Options.Tolerance ?? 1e-4;

  // Analytic gradients: zero the leaves, forward+backward once.
  for (const Input of Inputs) Input.ZeroGrad();
  const Loss = Forward();
  if (Loss.Size !== 1) {
    throw new Error(`GradCheck: Forward must return a scalar [1,1] loss, got ${Loss.Rows}x${Loss.Cols}`);
  }
  Backward(Loss);

  let MaxAbsError = 0;
  let MaxRelError = 0;

  for (const Input of Inputs) {
    for (let I = 0; I < Input.Size; I++) {
      const Original = Input.Data[I];

      Input.Data[I] = Original + Epsilon;
      const LossPlus = WithTapeOff(() => Forward().Data[0]);
      Input.Data[I] = Original - Epsilon;
      const LossMinus = WithTapeOff(() => Forward().Data[0]);
      Input.Data[I] = Original;

      const Numeric = (LossPlus - LossMinus) / (2 * Epsilon);
      const Analytic = Input.Grad[I];
      const AbsError = Math.abs(Numeric - Analytic);
      const RelError = AbsError / (Math.abs(Numeric) + Math.abs(Analytic) + 1e-12);

      if (AbsError > MaxAbsError) MaxAbsError = AbsError;
      if (RelError > MaxRelError) MaxRelError = RelError;
    }
  }

  return {
    MaxAbsError,
    MaxRelError,
    Passed: MaxRelError <= Tolerance || MaxAbsError <= Tolerance,
  };
}

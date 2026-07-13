import { test, expect } from "bun:test";
import { Tensor } from "../Brain/Tensor/Tensor.ts";
import { Tape } from "../Brain/Tensor/Tape.ts";
import { GradCheck } from "../Brain/Autograd/GradCheck.ts";

// A minimal manual op: loss = sum(x_i^2). Correct backward is dLoss/dx_i = 2*x_i.
// `Wrong` deliberately uses x_i instead, to prove GradCheck actually catches bad gradients.
function SumOfSquares(X: Tensor, Wrong: boolean): Tensor {
  const Out = new Tensor(1, 1, undefined, [X]);
  let Sum = 0;
  for (let I = 0; I < X.Size; I++) Sum += X.Data[I] * X.Data[I];
  Out.Data[0] = Sum;
  if (Tape.On) {
    Out.BackwardFn = () => {
      const G = Out.Grad[0];
      for (let I = 0; I < X.Size; I++) {
        X.Grad[I] += G * (Wrong ? X.Data[I] : 2 * X.Data[I]);
      }
    };
  }
  return Out;
}

function MakeInput(): Tensor {
  const X = new Tensor(2, 3);
  X.Data.set([0.5, -1.2, 0.3, 0.9, -0.4, 1.1]);
  return X;
}

test("GradCheck passes for a correct backward pass", () => {
  const X = MakeInput();
  const Result = GradCheck([X], () => SumOfSquares(X, false));
  expect(Result.Passed).toBe(true);
  expect(Result.MaxAbsError).toBeLessThan(1e-5);
});

test("GradCheck catches a deliberately wrong backward pass", () => {
  const X = MakeInput();
  const Result = GradCheck([X], () => SumOfSquares(X, true));
  expect(Result.Passed).toBe(false);
});

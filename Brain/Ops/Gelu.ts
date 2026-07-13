// GELU activation (tanh approximation, as used by GPT-2/many code models). Backward is the
// analytic derivative of the approximation.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

const C = Math.sqrt(2 / Math.PI); // 0.79788456...
const A3 = 0.044715;

export function Gelu(X: Tensor): Tensor {
  const Out = new Tensor(X.Rows, X.Cols, undefined, [X]);
  for (let I = 0; I < X.Size; I++) {
    const V = X.Data[I];
    const Inner = C * (V + A3 * V * V * V);
    Out.Data[I] = 0.5 * V * (1 + Math.tanh(Inner));
  }

  if (Tape.On) {
    Out.BackwardFn = () => {
      for (let I = 0; I < X.Size; I++) {
        const V = X.Data[I];
        const Inner = C * (V + A3 * V * V * V);
        const T = Math.tanh(Inner);
        const DInner = C * (1 + 3 * A3 * V * V);
        const D = 0.5 * (1 + T) + 0.5 * V * (1 - T * T) * DInner;
        X.Grad[I] += D * Out.Grad[I];
      }
    };
  }
  return Out;
}

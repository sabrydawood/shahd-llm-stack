// Rotary Position Embedding (RoPE). Rotates adjacent dim pairs (2i, 2i+1) of X[T,D] by an angle
// proportional to the token's absolute position and the pair's frequency. Applied to Q and K
// inside attention (D = HeadDim, even). RoPE encodes RELATIVE position in the dot product, so
// unlike learned absolute positions it extends past the training context — which also removes
// the KV-cache-beyond-BlockSize limitation (KvCache.ts note). The rotation is orthogonal, so its
// backward is the rotation by the negative angle.

import { Tensor } from "../Tensor/Tensor.ts";
import { Tape } from "../Tensor/Tape.ts";

/** Rotate X[T,D] (D even). Row r uses absolute position PositionStart+r. */
export function ApplyRope(X: Tensor, PositionStart: number, Base = 10000): Tensor {
  const T = X.Rows;
  const D = X.Cols;
  if (D % 2 !== 0) throw new Error(`ApplyRope: head dim ${D} must be even`);
  const Half = D / 2;
  const Out = new Tensor(T, D, undefined, [X]);

  // Precompute cos/sin per (row, pair) so forward and backward share them.
  const Cos = new Float64Array(T * Half);
  const Sin = new Float64Array(T * Half);
  for (let R = 0; R < T; R++) {
    const Position = PositionStart + R;
    for (let I = 0; I < Half; I++) {
      const Freq = Math.pow(Base, (-2 * I) / D);
      const Angle = Position * Freq;
      Cos[R * Half + I] = Math.cos(Angle);
      Sin[R * Half + I] = Math.sin(Angle);
    }
  }

  for (let R = 0; R < T; R++) {
    for (let I = 0; I < Half; I++) {
      const C = Cos[R * Half + I];
      const S = Sin[R * Half + I];
      const X0 = X.Data[R * D + 2 * I];
      const X1 = X.Data[R * D + 2 * I + 1];
      Out.Data[R * D + 2 * I] = X0 * C - X1 * S;
      Out.Data[R * D + 2 * I + 1] = X0 * S + X1 * C;
    }
  }

  if (Tape.On) {
    Out.BackwardFn = () => {
      for (let R = 0; R < T; R++) {
        for (let I = 0; I < Half; I++) {
          const C = Cos[R * Half + I];
          const S = Sin[R * Half + I];
          const G0 = Out.Grad[R * D + 2 * I];
          const G1 = Out.Grad[R * D + 2 * I + 1];
          // Backward = rotation by -angle applied to the upstream gradient.
          X.Grad[R * D + 2 * I] += G0 * C + G1 * S;
          X.Grad[R * D + 2 * I + 1] += -G0 * S + G1 * C;
        }
      }
    };
  }
  return Out;
}

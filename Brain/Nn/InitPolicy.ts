// The single home for weight-initialization policy (rule #4). In particular the scaled-residual
// init (1/sqrt(2*NumLayers) on the attention-output and MLP-output projections) is applied here
// and nowhere else, so it can never be re-derived inline and silently forgotten as depth grows.
// The residual scale itself is computed once in DeriveConfig; this just selects it.

import { RandN } from "../Tensor/TensorFactories.ts";
import type { Tensor } from "../Tensor/Tensor.ts";
import type { SeededRng } from "../Random/SeededRng.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";

/**
 * Initialize a weight matrix. Residual-projection weights (attention Wo, MLP down-projection)
 * pass IsResidualProjection=true to get the depth-scaled init; all others use the base InitScale.
 */
export function InitWeight(
  Rows: number,
  Cols: number,
  Rng: SeededRng,
  Config: ResolvedConfig,
  IsResidualProjection = false,
): Tensor {
  const Scale = IsResidualProjection ? Config.Derived.ResidualInitScale : Config.Model.InitScale;
  return RandN(Rows, Cols, Scale, Rng);
}

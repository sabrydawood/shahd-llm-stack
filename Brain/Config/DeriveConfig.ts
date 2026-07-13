// The ONLY home for derived quantities (rule #4 single-source). Critically, HeadDim and the
// attention scale live here and nowhere else — this closes REVIEW.md's L4 landmine by
// construction: no op site ever re-computes 1/sqrt(...) inline, so multi-head can never
// silently use the wrong scale.

import type { ShahdConfig, DerivedConfig } from "./ConfigTypes.ts";

export function DeriveConfig(Config: ShahdConfig): DerivedConfig {
  const { EmbedDim, NumHeads, NumLayers, MlpRatio, InitScale, UseScaledResidualInit } = Config.Model;

  const HeadDim = EmbedDim / NumHeads; // integer — guaranteed by ValidateConfig's superRefine

  return {
    HeadDim,
    AttentionScale: 1 / Math.sqrt(HeadDim), // scale by per-head key dim, NOT full EmbedDim (L4)
    MlpHidden: Math.round(EmbedDim * MlpRatio),
    ResidualInitScale: UseScaledResidualInit ? InitScale / Math.sqrt(2 * NumLayers) : InitScale,
  };
}

// The config type contracts (single-source, rule #4). Types are INFERRED from the Zod schema
// (z.infer) so the type and the validator can never drift apart.

import type { z } from "zod";
import type { ShahdConfigSchema } from "./ValidateConfig.ts";

/** A fully-specified, validated Shahd config (before derivation). */
export type ShahdConfig = z.infer<typeof ShahdConfigSchema>;

/** A deep-partial override that LoadConfig merges onto the defaults. */
export type ConfigOverride = DeepPartial<ShahdConfig>;

/** Values computed once from a validated config — the ONLY home for HeadDim / AttentionScale. */
export type DerivedConfig = {
  HeadDim: number; // EmbedDim / NumHeads
  AttentionScale: number; // 1 / sqrt(HeadDim)  ← the L4-safe scale
  MlpHidden: number; // round(EmbedDim * MlpRatio)
  ResidualInitScale: number; // InitScale / sqrt(2 * NumLayers) when scaled-residual init is on
};

/** The frozen, self-describing config threaded through the whole system. */
export type ResolvedConfig = ShahdConfig & {
  readonly Derived: DerivedConfig;
  readonly ConfigHash: string; // stable hash of the validated config; embedded in checkpoints
};

export type DeepPartial<T> = {
  [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K];
};

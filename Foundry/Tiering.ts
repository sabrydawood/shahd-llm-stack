// Tier classification (M3) — the one place a document's fate is decided, reusing the existing Data
// filters (rule #4: no reimplementation). Rules, in order:
//   general web  -> Raw (isolated, inspect-only) regardless of license (license is unverified).
//   non-permissive -> Rejected (with the license as the reason).
//   low quality  -> Rejected (with the quality reasons).
//   otherwise    -> Filtered (permissive + clean, eligible for training).

import { IsPermissive } from "../Brain/Data/LicenseManifest.ts";
import { ScoreCodeQuality } from "../Brain/Data/QualityFilter.ts";
import type { Tier, Origin } from "./DocumentRecord.ts";

export type TierDecision = { Tier: Tier; QualityScore: number; RejectReason: string | null };

export function ClassifyDocument(License: string, Content: string, Origin: Origin): TierDecision {
  const Quality = ScoreCodeQuality(Content);
  if (Origin === "web-general") {
    return { Tier: "Raw", QualityScore: Quality.Score, RejectReason: "general web: isolated for inspection, not training-eligible" };
  }
  if (!IsPermissive(License)) {
    return { Tier: "Rejected", QualityScore: Quality.Score, RejectReason: `non-permissive license: ${License}` };
  }
  if (!Quality.Passed) {
    return { Tier: "Rejected", QualityScore: Quality.Score, RejectReason: `low quality: ${Quality.Reasons.join("; ")}` };
  }
  return { Tier: "Filtered", QualityScore: Quality.Score, RejectReason: null };
}

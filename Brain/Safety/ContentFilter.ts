// Deterministic content scanner. Given text and a safety Level, returns which harm categories
// matched (rules below the active Level are skipped). Level "Off" scans nothing.

import type { HarmCategory } from "./HarmfulPatterns.ts";
import { HarmfulRules } from "./HarmfulPatterns.ts";

export type SafetyLevel = "Off" | "Standard" | "Strict";

export type FilterVerdict = { Blocked: boolean; Categories: HarmCategory[] };

function LevelRank(Level: SafetyLevel): number {
  if (Level === "Off") return 0;
  if (Level === "Standard") return 1;
  return 2;
}

function RuleRank(MinLevel: "Standard" | "Strict"): number {
  return MinLevel === "Standard" ? 1 : 2;
}

// Common homoglyphs (Cyrillic / Greek / letterlike look-alikes) mapped to their Latin base, so
// "bоmb" (Cyrillic o) can't trivially evade the English-phrase rules.
const Homoglyphs: Record<string, string> = {
  "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
  "і": "i", "ѕ": "s", "ԁ": "d", "ɡ": "g", "ⅼ": "l", "ο": "o", "А": "a",
};

// Normalize before scanning: NFKC folds fullwidth/compatibility forms, then homoglyphs fold to Latin.
// A strict IMPROVEMENT (ASCII normalizes to itself, so nothing previously caught is missed) that closes
// the easy unicode-evasion class. NOTE (honest): this is still a bounded regex filter — leetspeak,
// letter-spacing, base64, and non-English phrasing can still bypass it; it is a dev-time nudge for a
// small model, not a hard content-safety guarantee.
function NormalizeForScan(Text: string): string {
  const Folded = Text.normalize("NFKC");
  let Out = "";
  for (const Ch of Folded) Out += Homoglyphs[Ch] ?? Ch;
  return Out;
}

export function ScanContent(Text: string, Level: SafetyLevel): FilterVerdict {
  if (Level === "Off") return { Blocked: false, Categories: [] };
  const Active = LevelRank(Level);
  const Normalized = NormalizeForScan(Text);
  const Hits: HarmCategory[] = [];
  for (const Rule of HarmfulRules) {
    if (RuleRank(Rule.MinLevel) > Active) continue;
    for (const Pattern of Rule.Patterns) {
      if (Pattern.test(Normalized)) {
        Hits.push(Rule.Category);
        break;
      }
    }
  }
  return { Blocked: Hits.length > 0, Categories: Hits };
}

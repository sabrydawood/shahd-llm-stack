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

export function ScanContent(Text: string, Level: SafetyLevel): FilterVerdict {
  if (Level === "Off") return { Blocked: false, Categories: [] };
  const Active = LevelRank(Level);
  const Hits: HarmCategory[] = [];
  for (const Rule of HarmfulRules) {
    if (RuleRank(Rule.MinLevel) > Active) continue;
    for (const Pattern of Rule.Patterns) {
      if (Pattern.test(Text)) {
        Hits.push(Rule.Category);
        break;
      }
    }
  }
  return { Blocked: Hits.length > 0, Categories: Hits };
}

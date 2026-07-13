// The config-driven safety gate. Wraps the content filter with the Enabled/Level knobs from
// config, and provides enforcement helpers that throw when input/output is blocked. This is the
// one place product code goes through to be safe — strengthen/weaken via config + HarmfulPatterns.

import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import { ScanContent } from "./ContentFilter.ts";
import type { FilterVerdict, SafetyLevel } from "./ContentFilter.ts";

export class SafetyBlockedError extends Error {
  Categories: string[];
  constructor(Where: string, Categories: string[]) {
    super(`Safety: blocked ${Where} (categories: ${Categories.join(", ") || "none"})`);
    this.name = "SafetyBlockedError";
    this.Categories = Categories;
  }
}

export class SafetyPolicy {
  private Enabled: boolean;
  private Level: SafetyLevel;

  constructor(Config: ResolvedConfig) {
    this.Enabled = Config.Safety.Enabled;
    this.Level = Config.Safety.Level;
  }

  Check(Text: string): FilterVerdict {
    if (!this.Enabled) return { Blocked: false, Categories: [] };
    return ScanContent(Text, this.Level);
  }

  EnforceInput(Text: string): void {
    const Verdict = this.Check(Text);
    if (Verdict.Blocked) throw new SafetyBlockedError("prompt", Verdict.Categories);
  }

  EnforceOutput(Text: string): void {
    const Verdict = this.Check(Text);
    if (Verdict.Blocked) throw new SafetyBlockedError("output", Verdict.Categories);
  }
}

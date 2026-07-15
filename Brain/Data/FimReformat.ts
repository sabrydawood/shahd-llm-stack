// Fill-in-the-Middle reformatting (Phase 3). Teaches the model to complete code given both a
// prefix AND a suffix (essential for real IDE autocomplete, not just left-to-right). A fraction
// of pretraining documents are rewritten into PSM/SPM form with sentinel tokens; the tokenizer
// reserves these as special tokens (added alongside the Phase-4 chat tokens).

import { SafeBoundary } from "../Tokenizer/Internal/SurrogateBoundary.ts";

export const FimTokens = {
  Prefix: "<|fim_prefix|>",
  Middle: "<|fim_middle|>",
  Suffix: "<|fim_suffix|>",
} as const;

export type FimMode = "Psm" | "Spm"; // prefix-suffix-middle vs suffix-prefix-middle

/** Split Text into prefix/middle/suffix at two random cut points and reformat for FIM. */
export function ToFim(Text: string, Cut1: number, Cut2: number, Mode: FimMode = "Psm"): string {
  // Snap both cut points off a surrogate-pair boundary — an astral character (emoji etc.) must never
  // be split between prefix/middle or middle/suffix.
  const A = SafeBoundary(Text, Math.max(0, Math.min(Cut1, Cut2)));
  const B = SafeBoundary(Text, Math.min(Text.length, Math.max(Cut1, Cut2)));
  const Prefix = Text.slice(0, A);
  const Middle = Text.slice(A, B);
  const Suffix = Text.slice(B);
  if (Mode === "Spm") {
    return `${FimTokens.Prefix}${FimTokens.Suffix}${Suffix}${FimTokens.Middle}${Prefix}${Middle}`;
  }
  return `${FimTokens.Prefix}${Prefix}${FimTokens.Suffix}${Suffix}${FimTokens.Middle}${Middle}`;
}

// Recover the original document from a PSM-formatted FIM string (for round-trip verification).
// SPM is a training-only re-ordering and is not reconstructed here (prefix/middle are adjacent).
export function FromFim(Fim: string): string {
  const P = Fim.indexOf(FimTokens.Prefix);
  const S = Fim.indexOf(FimTokens.Suffix);
  const M = Fim.indexOf(FimTokens.Middle);
  if (P === -1 || S === -1 || M === -1 || !(P < S && S < M)) {
    throw new Error("FromFim: expected a PSM-formatted FIM string");
  }
  const Prefix = Fim.slice(P + FimTokens.Prefix.length, S);
  const Suffix = Fim.slice(S + FimTokens.Suffix.length, M);
  const Middle = Fim.slice(M + FimTokens.Middle.length);
  return Prefix + Middle + Suffix;
}

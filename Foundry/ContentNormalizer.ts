// Clean ingested source before it is stored (M10). It strips the repeated LICENSE/COPYRIGHT header
// boilerplate that sits atop most files in a repo (the same 15 lines on thousands of files — pure
// noise that would bias the model toward emitting copyright banners) while KEEPING real code
// comments/docstrings (valuable signal a code model should learn). Only a LEADING comment block that
// looks like a license header is removed; ordinary leading comments are left alone. SanitizeText
// then drops characters a Postgres text column cannot store, so one binary-ish file never aborts a
// whole ingest run.

const LicenseKeyword = /copyright|licensed under|SPDX-License|permission is hereby granted|all rights reserved|this (program|file|software) is free|@license|http:\/\/www\.apache\.org\/licenses|MIT License|BSD License|GNU General Public/i;

const Nul = 0x00;
const HighSurrogateStart = 0xd800;
const HighSurrogateEnd = 0xdbff;
const LowSurrogateStart = 0xdc00;
const LowSurrogateEnd = 0xdfff;
const Replacement = String.fromCharCode(0xfffd); // U+FFFD REPLACEMENT CHARACTER

/** Remove a leading license/copyright header comment block; keep everything else verbatim. */
export function StripLicenseHeader(Content: string): string {
  const Body = Content.replace(/^﻿/, ""); // drop a BOM if present

  // A leading block comment: /* … */  (C/JS/TS/Go/Rust/Java/CSS …)
  const Block = Body.match(/^\s*\/\*[\s\S]*?\*\/[ \t]*\r?\n?/);
  if (Block !== null && LicenseKeyword.test(Block[0])) {
    return Body.slice(Block[0].length).replace(/^\s+/, "");
  }

  // A run of leading line comments: // …  or  # …  (JS/TS, Python/Ruby/Shell …)
  const Lines = Body.match(/^(\s*(\/\/|#|;;).*\r?\n)+/);
  if (Lines !== null && LicenseKeyword.test(Lines[0])) {
    return Body.slice(Lines[0].length).replace(/^\s+/, "");
  }

  return Content;
}

/**
 * Remove characters a Postgres `text` column cannot store: NUL (U+0000) is forbidden outright, and
 * lone (unpaired) UTF-16 surrogates aren't valid UTF-8 and also error on insert. NUL is dropped;
 * lone surrogates become U+FFFD. Valid surrogate pairs (real emoji / astral chars) stay intact.
 * Single pass over code units (no regex on control chars — keeps this source byte-clean).
 */
export function SanitizeText(Content: string): string {
  let Out = "";
  for (let I = 0; I < Content.length; I++) {
    const Code = Content.charCodeAt(I);
    if (Code === Nul) continue;
    if (Code >= HighSurrogateStart && Code <= HighSurrogateEnd) {
      const Next = Content.charCodeAt(I + 1);
      if (Next >= LowSurrogateStart && Next <= LowSurrogateEnd) {
        Out += Content[I] + Content[I + 1]; // valid pair — keep both
        I++;
      } else {
        Out += Replacement; // lone high surrogate
      }
      continue;
    }
    if (Code >= LowSurrogateStart && Code <= LowSurrogateEnd) {
      Out += Replacement; // lone low surrogate
      continue;
    }
    Out += Content[I];
  }
  return Out;
}

// Clean ingested source before it is stored (M10). It strips the repeated LICENSE/COPYRIGHT header
// boilerplate that sits atop most files in a repo (the same 15 lines on thousands of files — pure
// noise that would bias the model toward emitting copyright banners) while KEEPING real code
// comments/docstrings (valuable signal a code model should learn). Only a LEADING comment block that
// looks like a license header is removed; ordinary leading comments are left alone.

const LicenseKeyword = /copyright|licensed under|SPDX-License|permission is hereby granted|all rights reserved|this (program|file|software) is free|@license|http:\/\/www\.apache\.org\/licenses|MIT License|BSD License|GNU General Public/i;

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

// Precision-first SPDX license detection from raw LICENSE text (Phase 3b). GitHub's own matcher
// (Licensee) returns NOASSERTION for any file that deviates from a template — which is exactly where
// the danger lives: open-core "Apache text + a commercial rider", "MIT + Commons Clause", or
// multi-license "Portions of this software are licensed as follows" splits. Presence of permissive
// wording is therefore NOT sufficient. A file is promoted to permissive ONLY when it is essentially
// ONE canonical permissive license and nothing else — proven by three gates that must all pass:
//   1. NO disqualifier  — no copyleft/commercial/mixing marker anywhere in the text.
//   2. Signature match  — the distinctive spans of exactly one permissive license are all present.
//   3. Coverage/size    — the canonical license makes up most of the file (rejects appended riders
//                         and giant bundled-dependency LICENSE files), or an Apache size cap.
// Wrong-promoting a copyleft/commercial repo is the legal risk this whole path exists to prevent, so
// every ambiguous case defaults to non-permissive (recall is deliberately traded away for precision).

export type SpdxResult = { Spdx: string | null; Permissive: boolean; Note: string };

/** Fold to lowercase alphanumeric words separated by single spaces — kills punctuation/quote/newline
 *  variation so template matching is robust across formatting. */
export function NormalizeLicense(Text: string): string {
  return Text.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

// Any of these anywhere in the normalized text => NOT permissive, regardless of other wording. Covers
// copyleft (gpl catches gpl/agpl/lgpl as substrings), share-alike, source-available/commercial, and
// the open-core mixing phrases that make a file deviate from a pure template. "all rights reserved" is
// deliberately absent — BSD licenses legitimately contain it.
const Disqualifiers: readonly string[] = [
  "gpl", // gpl, agpl, lgpl (substring)
  "gnu general public",
  "gnu affero",
  "gnu lesser",
  "affero",
  "mozilla public",
  "mpl 2 0",
  "creative commons",
  "commons clause",
  "eclipse public",
  "business source",
  "busl",
  "server side public",
  "sspl",
  "elastic license",
  "functional source",
  "sustainable use",
  "licensed as follows", // open-core split ("Portions of this software are licensed as follows"). NOT
  // bare "portions of" — the MIT license body itself says "substantial portions of the Software".
  "modified version of", // dify: "modified version of the Apache License"
  "community license", // lobehub
  "this license applies to parts", // astro: vendored-code carve-out
  "unless otherwise specified",
  "unless otherwise noted",
  "unless a directory", // joplin
  "except as documented below", // lodash
  "additionally it includes", // remote-jobs (ISC + MIT + SIL)
  "customer agreement", // serverless
  "protected by the copyright laws", // developer-roadmap
  "licensing transition", // mcp/servers (MIT -> Apache, in flux)
  "externally maintained libraries", // node (bundle)
];

// A permissive template: every Signature span must be present (normalized). Order matters — more
// specific templates (BSD-3 before BSD-2, 0BSD before ISC) are checked first. Coverage = CanonNorm /
// file-normalized-length must be >= MinCoverage (the canonical license is most of the file). Apache
// has two valid forms (a ~500-char notice and the ~10k full text), so it uses MaxNormLen instead.
type Template = {
  Spdx: string;
  Signatures: readonly string[];
  CanonNorm?: number; // normalized length of the canonical license text
  MinCoverage?: number; // require CanonNorm / normLen >= this
  MaxNormLen?: number; // alternative size guard (Apache)
};

const MitGrant = "permission is hereby granted free of charge to any person obtaining a copy of this software";
const IscGrant = "permission to use copy modify and or distribute this software for any purpose with or without fee is hereby granted";
const BsdGrant = "redistribution and use in source and binary forms with or without modification are permitted provided that the following conditions are met";

const Templates: readonly Template[] = [
  {
    Spdx: "MIT",
    Signatures: [MitGrant, "to deal in the software without restriction", "the software is provided as is without warranty of any kind"],
    CanonNorm: 1129,
    MinCoverage: 0.72,
  },
  {
    Spdx: "BSD-3-Clause",
    Signatures: [BsdGrant, "endorse or promote products"],
    CanonNorm: 1398,
    MinCoverage: 0.72,
  },
  {
    Spdx: "BSD-2-Clause",
    Signatures: [BsdGrant, "this software is provided by the copyright"],
    CanonNorm: 1295,
    MinCoverage: 0.72,
  },
  {
    Spdx: "0BSD",
    Signatures: [IscGrant, "zero clause"],
    CanonNorm: 606,
    MinCoverage: 0.7,
  },
  {
    Spdx: "ISC",
    Signatures: [IscGrant],
    CanonNorm: 740,
    MinCoverage: 0.6,
  },
  {
    Spdx: "Apache-2.0",
    Signatures: ["apache license version 2 0", "www apache org licenses license 2 0"],
    MaxNormLen: 15000,
  },
  {
    Spdx: "Unlicense",
    Signatures: ["this is free and unencumbered software released into the public domain"],
    CanonNorm: 1160,
    MinCoverage: 0.6,
  },
  {
    Spdx: "Zlib",
    Signatures: ["altered source versions must be plainly marked as such", "this software is provided as is"],
    CanonNorm: 750,
    MinCoverage: 0.55,
  },
];

/** Classify raw LICENSE text. Permissive:true only when the text is provably a single clean
 *  permissive license; every ambiguous/mixed/copyleft/commercial case returns Permissive:false. */
export function DetectSpdx(Text: string): SpdxResult {
  const Norm = NormalizeLicense(Text);
  if (Norm.length < 40) return { Spdx: null, Permissive: false, Note: "empty/too short" };

  for (const Bad of Disqualifiers) {
    if (Norm.includes(Bad)) return { Spdx: null, Permissive: false, Note: `disqualifier: "${Bad}"` };
  }

  for (const T of Templates) {
    if (!T.Signatures.every((S) => Norm.includes(S))) continue;
    if (T.MinCoverage !== undefined && T.CanonNorm !== undefined) {
      const Coverage = T.CanonNorm / Norm.length;
      if (Coverage < T.MinCoverage) return { Spdx: T.Spdx, Permissive: false, Note: `low coverage ${Coverage.toFixed(2)} (bundled/appended?)` };
    }
    if (T.MaxNormLen !== undefined && Norm.length > T.MaxNormLen) {
      return { Spdx: T.Spdx, Permissive: false, Note: `oversized ${Norm.length} > ${T.MaxNormLen} (bundle?)` };
    }
    return { Spdx: T.Spdx, Permissive: true, Note: "clean permissive template" };
  }

  return { Spdx: null, Permissive: false, Note: "no permissive template matched" };
}

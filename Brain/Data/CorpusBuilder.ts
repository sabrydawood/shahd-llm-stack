// The Phase-3 corpus pipeline, assembled end-to-end (rule #4: this ties the existing stages
// together, it does not reimplement them). Raw source documents flow through:
//   license allowlist -> quality filter -> near-duplicate removal -> eval decontamination
//   -> optional FIM reformatting -> concatenation into a training-ready token/text stream,
// with a provenance manifest for exactly what ended up in the training set and stats for every
// stage's drop count. Pure (no fs): the caller supplies documents; scripts do the I/O.

import { LicenseManifest, IsPermissive } from "./LicenseManifest.ts";
import { ScoreCodeQuality } from "./QualityFilter.ts";
import type { QualityOptions } from "./QualityFilter.ts";
import { DedupedIndices } from "./NearDedup.ts";
import { Decontaminate } from "./Decontamination.ts";
import { ToFim } from "./FimReformat.ts";
import type { FimMode } from "./FimReformat.ts";
import type { SeededRng } from "../Random/SeededRng.ts";

export type SourceDocument = {
  Source: string; // repo/dataset name
  License: string; // SPDX id
  Path: string; // path within the source
  Content: string;
  IngestedAt?: string; // ISO timestamp; falls back to Options.IngestedAt
};

export type CorpusBuilderOptions = {
  DedupThreshold?: number; // Jaccard >= this => near-duplicate (default 0.8)
  Quality?: QualityOptions; // forwarded to ScoreCodeQuality
  EvalDocs?: string[]; // decontaminate training docs against these
  DecontaminationNgram?: number; // word n-gram size for contamination (default 13)
  FimFraction?: number; // fraction of kept docs rewritten to FIM (needs FimRng)
  FimMode?: FimMode; // PSM (default) or SPM
  FimRng?: SeededRng; // deterministic cut points
  DocumentSeparator?: string; // joins docs into Text (default "\n\n")
  IngestedAt?: string; // default provenance timestamp
};

export type CorpusStats = {
  Input: number;
  DroppedNonPermissive: number;
  DroppedLowQuality: number;
  DroppedNearDuplicate: number;
  DroppedContaminated: number;
  FimRewritten: number;
  Kept: number;
  TotalBytes: number;
};

export type BuiltCorpus = {
  Documents: string[]; // final kept contents (some FIM-rewritten)
  Manifest: LicenseManifest; // provenance ledger of exactly what's in the corpus
  Stats: CorpusStats;
  Text: string; // training-ready concatenation
};

export function BuildCorpus(Sources: SourceDocument[], Options: CorpusBuilderOptions = {}): BuiltCorpus {
  const DedupThreshold = Options.DedupThreshold ?? 0.8;
  const Separator = Options.DocumentSeparator ?? "\n\n";
  const DefaultIngestedAt = Options.IngestedAt ?? "unspecified";
  // Byte counts must be actual UTF-8 bytes, not UTF-16 code units (string.length) — non-ASCII text
  // (Arabic, CJK, emoji) undercounts badly with .length, skewing manifest provenance and corpus stats.
  const Utf8Encoder = new TextEncoder();

  // 1) License allowlist — the only documents legally safe to train on.
  const Permissive = Sources.filter((D) => IsPermissive(D.License));
  const DroppedNonPermissive = Sources.length - Permissive.length;

  // 2) Quality filter — drop minified/generated/binary-ish junk.
  const Quality = Permissive.filter((D) => ScoreCodeQuality(D.Content, Options.Quality).Passed);
  const DroppedLowQuality = Permissive.length - Quality.length;

  // 3) Near-duplicate removal — keep one representative per MinHash group.
  const KeepIdx = new Set(DedupedIndices(Quality.map((D) => D.Content), DedupThreshold));
  const Deduped = Quality.filter((_D, I) => KeepIdx.has(I));
  const DroppedNearDuplicate = Quality.length - Deduped.length;

  // 4) Eval decontamination — remove docs sharing a long n-gram with the held-out set.
  let Clean = Deduped;
  let DroppedContaminated = 0;
  if (Options.EvalDocs !== undefined && Options.EvalDocs.length > 0) {
    const Result = Decontaminate(Deduped.map((D) => D.Content), Options.EvalDocs, Options.DecontaminationNgram ?? 13);
    Clean = Result.Kept.map((I) => Deduped[I]);
    DroppedContaminated = Result.Removed.length;
  }

  // 5) Optional FIM reformatting on a deterministic fraction.
  const FimFraction = Options.FimFraction ?? 0;
  const FimRng = Options.FimRng;
  let FimRewritten = 0;
  const Manifest = new LicenseManifest();
  const Documents = Clean.map((D) => {
    let Content = D.Content;
    if (FimFraction > 0 && FimRng !== undefined && FimRng.NextFloat() < FimFraction && Content.length > 4) {
      const Cut1 = Math.floor(FimRng.NextFloat() * Content.length);
      const Cut2 = Math.floor(FimRng.NextFloat() * Content.length);
      Content = ToFim(Content, Cut1, Cut2, Options.FimMode ?? "Psm");
      FimRewritten++;
    }
    Manifest.Add({ Source: D.Source, License: D.License, Path: D.Path, Bytes: Utf8Encoder.encode(Content).length, IngestedAt: D.IngestedAt ?? DefaultIngestedAt });
    return Content;
  });

  const Text = Documents.join(Separator);
  const Stats: CorpusStats = {
    Input: Sources.length,
    DroppedNonPermissive,
    DroppedLowQuality,
    DroppedNearDuplicate,
    DroppedContaminated,
    FimRewritten,
    Kept: Documents.length,
    TotalBytes: Utf8Encoder.encode(Text).length,
  };
  return { Documents, Manifest, Stats, Text };
}

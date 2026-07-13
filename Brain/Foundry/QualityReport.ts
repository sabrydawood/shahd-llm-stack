// Aggregate the Foundry into an inspectable quality report (M3) — the "what is in the data and how
// good is it" answer that feeds the dashboard (M4). Counts per tier / license / language, filtered
// (training-eligible) bytes, and a 10-bucket quality-score histogram.

import type { DocumentRecord, Tier } from "./DocumentRecord.ts";

export type FoundryReport = {
  Total: number;
  ByTier: Record<Tier, number>;
  ByLicense: Record<string, number>;
  ByLang: Record<string, number>;
  FilteredBytes: number; // bytes eligible for training
  QualityHistogram: number[]; // 10 buckets over [0,1]
};

export function BuildReport(Docs: DocumentRecord[]): FoundryReport {
  const ByTier: Record<Tier, number> = { Filtered: 0, Raw: 0, Rejected: 0 };
  const ByLicense: Record<string, number> = {};
  const ByLang: Record<string, number> = {};
  const QualityHistogram = new Array<number>(10).fill(0);
  let FilteredBytes = 0;

  for (const Doc of Docs) {
    ByTier[Doc.Tier]++;
    ByLicense[Doc.License] = (ByLicense[Doc.License] ?? 0) + 1;
    ByLang[Doc.Lang] = (ByLang[Doc.Lang] ?? 0) + 1;
    if (Doc.Tier === "Filtered") FilteredBytes += Doc.Bytes;
    const Bucket = Math.min(9, Math.max(0, Math.floor(Doc.QualityScore * 10)));
    QualityHistogram[Bucket]++;
  }

  return { Total: Docs.length, ByTier, ByLicense, ByLang, FilteredBytes, QualityHistogram };
}

/** A compact human-readable rendering of the report (for the CLI / logs). */
export function RenderReportText(Report: FoundryReport): string {
  const Lines = [
    `Foundry: ${Report.Total} documents`,
    `  tiers    : Filtered=${Report.ByTier.Filtered} Raw=${Report.ByTier.Raw} Rejected=${Report.ByTier.Rejected}`,
    `  filtered : ${Report.FilteredBytes} training-eligible bytes`,
    `  licenses : ${JSON.stringify(Report.ByLicense)}`,
    `  langs    : ${JSON.stringify(Report.ByLang)}`,
    `  quality  : ${Report.QualityHistogram.join(" ")} (buckets 0.0→1.0)`,
  ];
  return Lines.join("\n");
}

// Run the Data Foundry over the seed corpus into an in-memory store, print the quality report, and
// show the training-eligible export — the whole M3 core end-to-end with no database required.
//
//   bun run Scripts/RunFoundry.ts

import { readFileSync, existsSync } from "node:fs";
import { InMemoryDocumentStore, IngestDocuments, ExportTrainingText, BuildReport, RenderReportText } from "../Brain/Foundry/FoundryBarrel.ts";
import type { SourceInput } from "../Brain/Foundry/FoundryBarrel.ts";

type ManifestEntry = { Source: string; License: string; Path: string; Lang?: string };

const Manifest = JSON.parse(readFileSync("Corpus/Manifest.json", "utf8")) as { Documents: ManifestEntry[] };
const Inputs: SourceInput[] = Manifest.Documents
  .filter((E) => existsSync(E.Path))
  .map((E) => ({ Source: E.Source, License: E.License, Lang: E.Lang ?? "unknown", Content: readFileSync(E.Path, "utf8"), Provenance: E.Path, Origin: "local" as const }));

const Store = new InMemoryDocumentStore();
const Stats = await IngestDocuments(Inputs, Store, "2026-07-13T00:00:00.000Z");
console.log(`ingested ${Stats.Ingested} documents -> Filtered=${Stats.ByTier.Filtered} Raw=${Stats.ByTier.Raw} Rejected=${Stats.ByTier.Rejected}\n`);

console.log(RenderReportText(BuildReport(await Store.All())));

const Rejected = await Store.ByTier("Rejected");
console.log("\nrejected (with reasons):");
for (const Doc of Rejected) console.log(`  ${Doc.Provenance}: ${Doc.RejectReason}`);

const Text = await ExportTrainingText(Store);
console.log(`\ntraining-eligible export: ${Text.length} bytes from the Filtered tier`);

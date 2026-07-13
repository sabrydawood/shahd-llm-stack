// Integration smoke test for the Postgres Foundry (M3b) — needs a live database.
//   docker compose up -d && bun run Scripts/FoundryMigrate.ts && bun run Scripts/FoundrySmoke.ts
// Ingests the seed corpus into Postgres, prints the quality report, and runs a similarity query.

import { readFileSync, existsSync } from "node:fs";
import { IngestDocuments, BuildReport, RenderReportText, HashingEmbedding } from "../Brain/Foundry/FoundryBarrel.ts";
import type { SourceInput } from "../Brain/Foundry/FoundryBarrel.ts";
import { PostgresDocumentStore } from "../Brain/Foundry/PostgresDocumentStore.ts";

type ManifestEntry = { Source: string; License: string; Path: string; Lang?: string };

const Url = process.env["DATABASE_URL"] ?? "postgres://shahd:shahd@localhost:5432/shahd";
const Store = new PostgresDocumentStore(Url);

const Manifest = JSON.parse(readFileSync("Corpus/Manifest.json", "utf8")) as { Documents: ManifestEntry[] };
const Inputs: SourceInput[] = Manifest.Documents
  .filter((E) => existsSync(E.Path))
  .map((E) => ({ Source: E.Source, License: E.License, Lang: E.Lang ?? "unknown", Content: readFileSync(E.Path, "utf8"), Provenance: E.Path, Origin: "local" as const }));

const Stats = await IngestDocuments(Inputs, Store, "2026-07-13T00:00:00.000Z");
console.log(`ingested ${Stats.Ingested} -> stored ${await Store.Count()}\n`);
console.log(RenderReportText(BuildReport(await Store.All())));

const Similar = await Store.FindSimilar(HashingEmbedding("export function add(a, b) { return a + b; }"), 3);
console.log("\nmost similar to an add() snippet:");
for (const Hit of Similar) console.log(`  ${Hit.Score.toFixed(3)}  ${Hit.Doc.Provenance}`);

await Store.Close();

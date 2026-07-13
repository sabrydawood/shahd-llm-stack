// Serve the Foundry inspection dashboard. Reads the store from the environment (Bun loads .env): if
// DATABASE_URL is set it shows the Postgres data by default; otherwise it ingests the seed corpus
// into an in-memory store. Override with --Store=memory|postgres.
//   bun run foundry:dashboard            # uses .env (Postgres if DATABASE_URL set)
//   then open http://localhost:8090

import { readFileSync, existsSync } from "node:fs";
import { IngestDocuments, StartDashboard } from "../Foundry/FoundryBarrel.ts";
import type { SourceInput } from "../Foundry/FoundryBarrel.ts";
import { ResolveStore } from "./FoundryEnv.ts";
import { ReadArg } from "./ScriptArgs.ts";

type ManifestEntry = { Source: string; License: string; Path: string; Lang?: string };

const { Store, Kind } = ResolveStore();
if (Kind === "memory") {
  // No database configured — populate an in-memory store from the seed corpus so there's something to see.
  const Manifest = JSON.parse(readFileSync("Corpus/Manifest.json", "utf8")) as { Documents: ManifestEntry[] };
  const Inputs: SourceInput[] = Manifest.Documents
    .filter((E) => existsSync(E.Path))
    .map((E) => ({ Source: E.Source, License: E.License, Lang: E.Lang ?? "unknown", Content: readFileSync(E.Path, "utf8"), Provenance: E.Path, Origin: "local" as const }));
  await IngestDocuments(Inputs, Store, "2026-07-13T00:00:00.000Z");
}

const Port = Number(ReadArg("--Port=", "8090"));
StartDashboard(Store, Port);
console.log(`Data Foundry dashboard: http://localhost:${Port}  (${await Store.Count()} documents, ${Kind})`);

// Ingest the seed corpus into an in-memory Foundry and serve the inspection dashboard.
//   bun run Scripts/StartDashboard.ts   (then open http://localhost:8090)

import { readFileSync, existsSync } from "node:fs";
import { InMemoryDocumentStore, IngestDocuments, StartDashboard } from "../Brain/Foundry/FoundryBarrel.ts";
import type { DocumentStore, SourceInput } from "../Brain/Foundry/FoundryBarrel.ts";
import { PostgresDocumentStore } from "../Brain/Foundry/PostgresDocumentStore.ts";
import { ReadArg } from "./ScriptArgs.ts";

type ManifestEntry = { Source: string; License: string; Path: string; Lang?: string };

// With --Postgres (and DATABASE_URL) read the already-migrated Postgres store; otherwise ingest the
// seed corpus into an in-memory store.
const UsePostgres = process.argv.slice(2).includes("--Postgres");
let Store: DocumentStore;
if (UsePostgres) {
  Store = new PostgresDocumentStore(process.env["DATABASE_URL"] ?? "postgres://postgres:postgres@localhost:5432/shahd");
} else {
  const Manifest = JSON.parse(readFileSync("Corpus/Manifest.json", "utf8")) as { Documents: ManifestEntry[] };
  const Inputs: SourceInput[] = Manifest.Documents
    .filter((E) => existsSync(E.Path))
    .map((E) => ({ Source: E.Source, License: E.License, Lang: E.Lang ?? "unknown", Content: readFileSync(E.Path, "utf8"), Provenance: E.Path, Origin: "local" as const }));
  const Memory = new InMemoryDocumentStore();
  await IngestDocuments(Inputs, Memory, "2026-07-13T00:00:00.000Z");
  Store = Memory;
}

const Port = Number(ReadArg("--Port=", "8090"));
StartDashboard(Store, Port);
console.log(`Data Foundry dashboard: http://localhost:${Port}  (${await Store.Count()} documents, ${UsePostgres ? "postgres" : "in-memory"})`);

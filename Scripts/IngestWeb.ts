// Ingest from the web into the Foundry (M6). Gated by Config.Data.WebEnabled. GitHub-permissive
// files become training-eligible (after license/quality tiering); general web search results are
// stored in the isolated Raw tier. Provide credentials via env; nothing reaches the network without
// them beyond public GitHub (rate-limited).
//
//   GITHUB_TOKEN=... BRAVE_API_KEY=... bun run Scripts/IngestWeb.ts --Query="language:typescript license:mit stars:>100" --Store=postgres

import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { InMemoryDocumentStore, IngestFromWeb, CreateGitHubProvider, CreateWebSearchProvider, BuildReport, RenderReportText } from "../Brain/Foundry/FoundryBarrel.ts";
import type { WebProvider, SearchBackend, DocumentStore } from "../Brain/Foundry/FoundryBarrel.ts";
import { PostgresDocumentStore } from "../Brain/Foundry/PostgresDocumentStore.ts";
import { ReadArg } from "./ScriptArgs.ts";

// Brave Search backend (used only if BRAVE_API_KEY is set); swap for any search API by shape.
function BraveSearch(Key: string): SearchBackend {
  return async (Query, Limit) => {
    const Response = await fetch(`https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(Query)}&count=${Limit}`, {
      headers: { "X-Subscription-Token": Key, Accept: "application/json" },
    });
    if (!Response.ok) throw new Error(`Brave ${Response.status}`);
    const Data = (await Response.json()) as { web?: { results?: { url: string; title: string }[] } };
    return (Data.web?.results ?? []).map((R) => ({ Url: R.url, Title: R.title }));
  };
}

const Config = LoadConfig({ UseCli: true, UseEnv: false });
if (!Config.Data.WebEnabled) {
  console.log("web ingestion disabled (Config.Data.WebEnabled=false) — nothing fetched.");
  process.exit(0);
}

const Store: DocumentStore = ReadArg("--Store=", "memory") === "postgres"
  ? new PostgresDocumentStore(process.env["DATABASE_URL"] ?? "postgres://postgres:postgres@localhost:5432/shahd")
  : new InMemoryDocumentStore();

const Providers: WebProvider[] = [CreateGitHubProvider({ Token: process.env["GITHUB_TOKEN"] })];
const BraveKey = process.env["BRAVE_API_KEY"];
if (BraveKey !== undefined) Providers.push(CreateWebSearchProvider({ Search: BraveSearch(BraveKey) }));

const Queries = [ReadArg("--Query=", "language:typescript license:mit stars:>500")];
console.log(`ingesting from: ${Providers.map((P) => P.Name).join(", ")}  |  query: ${JSON.stringify(Queries[0])}`);

const Stats = await IngestFromWeb(Providers, Queries, Store, new Date().toISOString(), Number(ReadArg("--PerQuery=", "5")), Config.Data.EmbeddingDim);
console.log(`ingested ${Stats.Ingested} -> Filtered=${Stats.ByTier.Filtered} Raw=${Stats.ByTier.Raw} Rejected=${Stats.ByTier.Rejected}\n`);
console.log(RenderReportText(BuildReport(await Store.All())));

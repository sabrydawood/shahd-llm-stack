// Web ingestion orchestrator (M6). A WebProvider fetches documents already tagged with an Origin;
// IngestFromWeb runs the providers over a set of queries and feeds the results through the normal
// Foundry ingestion (so tiering/license/quality/dedup apply identically to web and local data). A
// provider that throws is skipped (non-fatal) so one bad source can't sink the whole run. Providers
// are injected — the network fetch and any search/API key live in the provider, never here.

import type { DocumentStore } from "./DocumentStore.ts";
import type { SourceInput, IngestStats } from "./Ingest.ts";
import { IngestDocuments } from "./Ingest.ts";

export type WebProvider = {
  Name: string;
  Fetch: (Query: string, Limit: number) => Promise<SourceInput[]>;
};

export async function IngestFromWeb(
  Providers: WebProvider[],
  Queries: string[],
  Store: DocumentStore,
  IngestedAt: string,
  PerQuery = 10,
  EmbeddingDim = 256,
): Promise<IngestStats> {
  const Collected: SourceInput[] = [];
  for (const Provider of Providers) {
    for (const Query of Queries) {
      try {
        Collected.push(...(await Provider.Fetch(Query, PerQuery)));
      } catch {
        // A provider/query failure is non-fatal — skip it and keep going.
      }
    }
  }
  return IngestDocuments(Collected, Store, IngestedAt, EmbeddingDim);
}

// The Foundry's persistence boundary (M3). The model never depends on this; only the data-curation
// layer does. Async so a Postgres/pgvector implementation fits the same interface as the in-memory
// one used in tests. FindSimilar powers semantic near-dup and "find similar" over embeddings.

import type { DocumentRecord, Tier } from "./DocumentRecord.ts";

export type SimilarHit = { Doc: DocumentRecord; Score: number };

export interface DocumentStore {
  Upsert(Doc: DocumentRecord): Promise<void>;
  All(): Promise<DocumentRecord[]>;
  ByTier(Tier: Tier): Promise<DocumentRecord[]>;
  FindSimilar(Embedding: number[], Limit: number): Promise<SimilarHit[]>;
  Count(): Promise<number>;
}

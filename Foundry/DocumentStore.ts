// The Foundry's persistence boundary (M3). The model never depends on this; only the data-curation
// layer does. Async so a Postgres/pgvector implementation fits the same interface as the in-memory
// one used in tests. FindSimilar powers semantic near-dup and "find similar" over embeddings.

import type { DocumentRecord, Tier } from "./DocumentRecord.ts";

export type SimilarHit = { Doc: DocumentRecord; Score: number };

// Per-repo rollup for the dashboard (accordion list + "already learned" skip).
export type RepoSummary = { Source: string; Files: number; Bytes: number };

// Aggregate counts for the dashboard cards — computed WITHOUT loading document content.
export type FoundryStats = {
  Total: number;
  ByTier: Record<Tier, number>;
  ByLang: Record<string, number>;
  ByLicense: Record<string, number>;
  FilteredBytes: number;
};

export interface DocumentStore {
  Upsert(Doc: DocumentRecord): Promise<void>;
  All(): Promise<DocumentRecord[]>;
  ByTier(Tier: Tier): Promise<DocumentRecord[]>;
  FindSimilar(Embedding: number[], Limit: number): Promise<SimilarHit[]>;
  Count(): Promise<number>;
  /** Distinct repo/source names already ingested (used to skip re-learning). */
  Sources(): Promise<string[]>;
  /** Per-repo file counts + bytes, most files first (the dashboard's repo list). */
  RepoSummaries(): Promise<RepoSummary[]>;
  /** Documents belonging to one repo/source (accordion contents). */
  DocumentsBySource(Source: string, Limit: number): Promise<DocumentRecord[]>;
  /** Aggregate dashboard stats (counts by tier/lang/license + filtered bytes), computed efficiently. */
  Stats(): Promise<FoundryStats>;
}

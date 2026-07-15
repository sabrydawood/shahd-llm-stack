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
  /** Documents in a tier; Limit caps the read (used for per-kind training size control). */
  ByTier(Tier: Tier, Limit?: number): Promise<DocumentRecord[]>;
  FindSimilar(Embedding: number[], Limit: number): Promise<SimilarHit[]>;
  Count(): Promise<number>;
  /** Distinct repo/source names already ingested (used to skip re-learning). */
  Sources(): Promise<string[]>;
  /** Per-repo file counts + bytes, most files first (the dashboard's repo list). */
  RepoSummaries(): Promise<RepoSummary[]>;
  /** Documents belonging to one repo/source (accordion contents). */
  DocumentsBySource(Source: string, Limit: number): Promise<DocumentRecord[]>;
  /** One document's full record by id (content hash) — powers the file viewer. Null if absent. */
  DocumentById(Id: string): Promise<DocumentRecord | null>;
  /** Aggregate dashboard stats (counts by tier/lang/license + filtered bytes), computed efficiently. */
  Stats(): Promise<FoundryStats>;
  /**
   * Relabel a source's NOASSERTION docs once its real license is verified (license-backfill). Docs
   * that also pass quality (>= MinQuality) are promoted to Filtered with NewLicense and reject_reason
   * cleared; the rest keep their tier but get NewLicense + a low-quality reason. Scoped to
   * license='NOASSERTION' so only the previously-unresolved rows are touched. Returns the split.
   */
  ReclassifyBySource(Source: string, NewLicense: string, MinQuality: number): Promise<{ Promoted: number; KeptLowQuality: number }>;
}

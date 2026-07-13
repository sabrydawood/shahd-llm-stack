// In-memory DocumentStore (M3) — the dependency-free implementation used by tests and small local
// runs. Same interface as the Postgres store, so ingestion/report/export code is validated without a
// live database. Dedup is by Id (content hash); FindSimilar is a linear cosine scan (fine at this
// scale — pgvector handles scale in the Postgres implementation).

import type { DocumentStore, SimilarHit } from "./DocumentStore.ts";
import type { DocumentRecord, Tier } from "./DocumentRecord.ts";
import { CosineSimilarity } from "./Embedding.ts";

export class InMemoryDocumentStore implements DocumentStore {
  private Docs = new Map<string, DocumentRecord>();

  async Upsert(Doc: DocumentRecord): Promise<void> {
    this.Docs.set(Doc.Id, Doc);
  }

  async All(): Promise<DocumentRecord[]> {
    return [...this.Docs.values()];
  }

  async ByTier(Tier: Tier): Promise<DocumentRecord[]> {
    return [...this.Docs.values()].filter((D) => D.Tier === Tier);
  }

  async FindSimilar(Embedding: number[], Limit: number): Promise<SimilarHit[]> {
    return [...this.Docs.values()]
      .map((Doc) => ({ Doc, Score: CosineSimilarity(Embedding, Doc.Embedding) }))
      .sort((A, B) => B.Score - A.Score)
      .slice(0, Limit);
  }

  async Count(): Promise<number> {
    return this.Docs.size;
  }
}

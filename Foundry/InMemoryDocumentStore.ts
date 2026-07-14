// In-memory DocumentStore (M3) — the dependency-free implementation used by tests and small local
// runs. Same interface as the Postgres store, so ingestion/report/export code is validated without a
// live database. Dedup is by Id (content hash); FindSimilar is a linear cosine scan (fine at this
// scale — pgvector handles scale in the Postgres implementation).

import type { DocumentStore, SimilarHit, RepoSummary, FoundryStats } from "./DocumentStore.ts";
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

  async Sources(): Promise<string[]> {
    return [...new Set([...this.Docs.values()].map((D) => D.Source))];
  }

  async RepoSummaries(): Promise<RepoSummary[]> {
    const By = new Map<string, RepoSummary>();
    for (const Doc of this.Docs.values()) {
      const Entry = By.get(Doc.Source) ?? { Source: Doc.Source, Files: 0, Bytes: 0 };
      Entry.Files++;
      Entry.Bytes += Doc.Bytes;
      By.set(Doc.Source, Entry);
    }
    return [...By.values()].sort((A, B) => B.Files - A.Files);
  }

  async DocumentsBySource(Source: string, Limit: number): Promise<DocumentRecord[]> {
    return [...this.Docs.values()].filter((D) => D.Source === Source).slice(0, Limit);
  }

  async DocumentById(Id: string): Promise<DocumentRecord | null> {
    return this.Docs.get(Id) ?? null;
  }

  async ReclassifyBySource(Source: string, NewLicense: string, MinQuality: number): Promise<{ Promoted: number; KeptLowQuality: number }> {
    let Promoted = 0;
    let KeptLowQuality = 0;
    for (const Doc of this.Docs.values()) {
      if (Doc.Source !== Source || Doc.License !== "NOASSERTION") continue;
      Doc.License = NewLicense;
      if (Doc.QualityScore >= MinQuality) {
        Doc.Tier = "Filtered";
        Doc.RejectReason = null;
        Promoted++;
      } else {
        Doc.RejectReason = `low quality (score < ${MinQuality})`;
        KeptLowQuality++;
      }
    }
    return { Promoted, KeptLowQuality };
  }

  async Stats(): Promise<FoundryStats> {
    const ByTier: Record<Tier, number> = { Filtered: 0, Raw: 0, Rejected: 0 };
    const ByLang: Record<string, number> = {};
    const ByLicense: Record<string, number> = {};
    let FilteredBytes = 0;
    for (const Doc of this.Docs.values()) {
      ByTier[Doc.Tier]++;
      ByLang[Doc.Lang] = (ByLang[Doc.Lang] ?? 0) + 1;
      ByLicense[Doc.License] = (ByLicense[Doc.License] ?? 0) + 1;
      if (Doc.Tier === "Filtered") FilteredBytes += Doc.Bytes;
    }
    return { Total: this.Docs.size, ByTier, ByLang, ByLicense, FilteredBytes };
  }
}

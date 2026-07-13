// Postgres + pgvector implementation of DocumentStore (M3b). Same interface as the in-memory store
// (which the CI tests cover), so ingestion / reports / dashboard work unchanged against a real
// database. FindSimilar uses pgvector cosine distance. Integration-verified via Scripts/FoundrySmoke
// (needs a running Postgres — `docker compose up` + `bun run Scripts/FoundryMigrate.ts`); it is not
// unit-tested here because CI has no database.

import postgres from "postgres";
import { drizzle } from "drizzle-orm/postgres-js";
import { eq, sql } from "drizzle-orm";
import { Documents } from "./FoundrySchema.ts";
import type { DocumentRow, DocumentInsert } from "./FoundrySchema.ts";
import type { DocumentStore, SimilarHit } from "./DocumentStore.ts";
import type { DocumentRecord, Tier } from "./DocumentRecord.ts";
import { CosineSimilarity } from "./Embedding.ts";

function ToRow(Doc: DocumentRecord): DocumentInsert {
  return {
    id: Doc.Id,
    tier: Doc.Tier,
    origin: Doc.Origin,
    source: Doc.Source,
    license: Doc.License,
    lang: Doc.Lang,
    content: Doc.Content,
    bytes: Doc.Bytes,
    quality: Doc.QualityScore,
    contentHash: Doc.ContentHash,
    embedding: Doc.Embedding,
    rejectReason: Doc.RejectReason,
    provenance: Doc.Provenance,
    ingestedAt: Doc.IngestedAt,
  };
}

function FromRow(Row: DocumentRow): DocumentRecord {
  return {
    Id: Row.id,
    Tier: Row.tier as Tier,
    Origin: Row.origin as DocumentRecord["Origin"],
    Source: Row.source,
    License: Row.license,
    Lang: Row.lang,
    Content: Row.content,
    Bytes: Row.bytes,
    QualityScore: Row.quality,
    ContentHash: Row.contentHash,
    Embedding: Row.embedding,
    RejectReason: Row.rejectReason,
    Provenance: Row.provenance,
    IngestedAt: Row.ingestedAt,
  };
}

export class PostgresDocumentStore implements DocumentStore {
  private Sql: ReturnType<typeof postgres>;
  private Db: ReturnType<typeof drizzle>;

  constructor(Url: string) {
    this.Sql = postgres(Url);
    this.Db = drizzle(this.Sql);
  }

  async Upsert(Doc: DocumentRecord): Promise<void> {
    const Row = ToRow(Doc);
    await this.Db.insert(Documents).values(Row).onConflictDoUpdate({ target: Documents.id, set: Row });
  }

  async All(): Promise<DocumentRecord[]> {
    return (await this.Db.select().from(Documents)).map(FromRow);
  }

  async ByTier(Tier: Tier): Promise<DocumentRecord[]> {
    return (await this.Db.select().from(Documents).where(eq(Documents.tier, Tier))).map(FromRow);
  }

  async FindSimilar(Embedding: number[], Limit: number): Promise<SimilarHit[]> {
    // In-app cosine ranking (portable: no pgvector required). For large-scale search, swap the
    // embedding column to pgvector and push this ordering into SQL.
    const All = await this.All();
    return All.map((Doc) => ({ Doc, Score: CosineSimilarity(Embedding, Doc.Embedding) }))
      .sort((A, B) => B.Score - A.Score)
      .slice(0, Limit);
  }

  async Count(): Promise<number> {
    const Result = await this.Db.select({ Count: sql<number>`count(*)::int` }).from(Documents);
    return Number(Result[0]?.Count ?? 0);
  }

  async Close(): Promise<void> {
    await this.Sql.end();
  }
}

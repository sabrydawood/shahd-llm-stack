// Postgres + pgvector implementation of DocumentStore (M3b). Same interface as the in-memory store
// (which the CI tests cover), so ingestion / reports / dashboard work unchanged against a real
// database. FindSimilar uses pgvector cosine distance. Integration-verified via Scripts/FoundrySmoke
// (needs a running Postgres — `docker compose up` + `bun run Scripts/FoundryMigrate.ts`); it is not
// unit-tested here because CI has no database.

import postgres from "postgres";
import { drizzle } from "drizzle-orm/postgres-js";
import { eq, sql, desc } from "drizzle-orm";
import { Documents } from "./FoundrySchema.ts";
import type { DocumentRow, DocumentInsert } from "./FoundrySchema.ts";
import type { DocumentStore, SimilarHit, RepoSummary, FoundryStats } from "./DocumentStore.ts";
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

  async Sources(): Promise<string[]> {
    return (await this.Db.selectDistinct({ Source: Documents.source }).from(Documents)).map((R) => R.Source);
  }

  async RepoSummaries(): Promise<RepoSummary[]> {
    const Rows = await this.Db
      .select({ Source: Documents.source, Files: sql<number>`count(*)::int`, Bytes: sql<number>`coalesce(sum(bytes),0)::bigint` })
      .from(Documents)
      .groupBy(Documents.source)
      .orderBy(desc(sql`count(*)`));
    return Rows.map((R) => ({ Source: R.Source, Files: Number(R.Files), Bytes: Number(R.Bytes) }));
  }

  async DocumentsBySource(Source: string, Limit: number): Promise<DocumentRecord[]> {
    return (await this.Db.select().from(Documents).where(eq(Documents.source, Source)).limit(Limit)).map(FromRow);
  }

  async Stats(): Promise<FoundryStats> {
    const CountExpr = sql<number>`count(*)::int`;
    const Tiers = await this.Db.select({ Key: Documents.tier, Count: CountExpr }).from(Documents).groupBy(Documents.tier);
    const Langs = await this.Db.select({ Key: Documents.lang, Count: CountExpr }).from(Documents).groupBy(Documents.lang);
    const Licenses = await this.Db.select({ Key: Documents.license, Count: CountExpr }).from(Documents).groupBy(Documents.license);
    const Filtered = await this.Db.select({ Bytes: sql<number>`coalesce(sum(bytes),0)::bigint` }).from(Documents).where(eq(Documents.tier, "Filtered"));

    const ByTier: Record<Tier, number> = { Filtered: 0, Raw: 0, Rejected: 0 };
    for (const Row of Tiers) if (Row.Key in ByTier) ByTier[Row.Key as Tier] = Number(Row.Count);
    const ByLang: Record<string, number> = {};
    for (const Row of Langs) ByLang[Row.Key] = Number(Row.Count);
    const ByLicense: Record<string, number> = {};
    for (const Row of Licenses) ByLicense[Row.Key] = Number(Row.Count);
    const Total = ByTier.Filtered + ByTier.Raw + ByTier.Rejected;
    return { Total, ByTier, ByLang, ByLicense, FilteredBytes: Number(Filtered[0]?.Bytes ?? 0) };
  }

  async Close(): Promise<void> {
    await this.Sql.end();
  }
}

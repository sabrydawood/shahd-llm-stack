// Postgres implementation of DocumentStore (M3b). Same interface as the in-memory store (which the CI
// tests cover), so ingestion / reports / dashboard work unchanged against a real database. The table
// is a constructor parameter (default "documents") so each data kind gets its own physically-separate
// table (documents_code, documents_conversation, …) via the same store class. Integration-verified via
// Scripts/FoundrySmoke; not unit-tested here because CI has no database.

import postgres from "postgres";
import { drizzle } from "drizzle-orm/postgres-js";
import { eq, sql, desc } from "drizzle-orm";
import { DocumentsTable } from "./FoundrySchema.ts";
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
  private Table: ReturnType<typeof DocumentsTable>;
  private OwnsSql: boolean;

  // Accepts a connection URL (opens+owns its own client) OR a shared postgres client (so many per-kind
  // stores can share ONE connection pool instead of opening one each — see FoundryStores).
  constructor(UrlOrSql: string | ReturnType<typeof postgres>, TableName = "documents") {
    this.OwnsSql = typeof UrlOrSql === "string";
    this.Sql = typeof UrlOrSql === "string" ? postgres(UrlOrSql) : UrlOrSql;
    this.Db = drizzle(this.Sql);
    this.Table = DocumentsTable(TableName);
  }

  async Upsert(Doc: DocumentRecord): Promise<void> {
    const Row = ToRow(Doc);
    await this.Db.insert(this.Table).values(Row).onConflictDoUpdate({ target: this.Table.id, set: Row });
  }

  async All(): Promise<DocumentRecord[]> {
    return (await this.Db.select().from(this.Table)).map(FromRow);
  }

  async ByTier(Tier: Tier, Limit?: number): Promise<DocumentRecord[]> {
    const Query = this.Db.select().from(this.Table).where(eq(this.Table.tier, Tier));
    const Rows = Limit !== undefined ? await Query.limit(Limit) : await Query;
    return Rows.map(FromRow);
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
    const Result = await this.Db.select({ Count: sql<number>`count(*)::int` }).from(this.Table);
    return Number(Result[0]?.Count ?? 0);
  }

  async Sources(): Promise<string[]> {
    return (await this.Db.selectDistinct({ Source: this.Table.source }).from(this.Table)).map((R) => R.Source);
  }

  async RepoSummaries(): Promise<RepoSummary[]> {
    const Rows = await this.Db
      .select({ Source: this.Table.source, Files: sql<number>`count(*)::int`, Bytes: sql<number>`coalesce(sum(bytes),0)::bigint` })
      .from(this.Table)
      .groupBy(this.Table.source)
      .orderBy(desc(sql`count(*)`));
    return Rows.map((R) => ({ Source: R.Source, Files: Number(R.Files), Bytes: Number(R.Bytes) }));
  }

  async DocumentsBySource(Source: string, Limit: number): Promise<DocumentRecord[]> {
    return (await this.Db.select().from(this.Table).where(eq(this.Table.source, Source)).limit(Limit)).map(FromRow);
  }

  async DocumentById(Id: string): Promise<DocumentRecord | null> {
    const Rows = await this.Db.select().from(this.Table).where(eq(this.Table.id, Id)).limit(1);
    return Rows[0] ? FromRow(Rows[0]) : null;
  }

  async ReclassifyBySource(Source: string, NewLicense: string, MinQuality: number): Promise<{ Promoted: number; KeptLowQuality: number }> {
    // Scoped to license='NOASSERTION' so only the previously-unresolved rows are relabeled. The id
    // (which encodes the old license) is intentionally left as-is — it is a dedup key, and SkipRepo
    // by source prevents any future re-ingest that would collide with it.
    const Promote = await this.Db
      .update(this.Table)
      .set({ tier: "Filtered", license: NewLicense, rejectReason: null })
      .where(sql`${this.Table.source} = ${Source} and ${this.Table.license} = 'NOASSERTION' and ${this.Table.quality} >= ${MinQuality}`)
      .returning({ Id: this.Table.id });
    const Keep = await this.Db
      .update(this.Table)
      .set({ license: NewLicense, rejectReason: `low quality (score < ${MinQuality})` })
      .where(sql`${this.Table.source} = ${Source} and ${this.Table.license} = 'NOASSERTION' and ${this.Table.quality} < ${MinQuality}`)
      .returning({ Id: this.Table.id });
    return { Promoted: Promote.length, KeptLowQuality: Keep.length };
  }

  async Stats(): Promise<FoundryStats> {
    const CountExpr = sql<number>`count(*)::int`;
    const Tiers = await this.Db.select({ Key: this.Table.tier, Count: CountExpr }).from(this.Table).groupBy(this.Table.tier);
    const Langs = await this.Db.select({ Key: this.Table.lang, Count: CountExpr }).from(this.Table).groupBy(this.Table.lang);
    const Licenses = await this.Db.select({ Key: this.Table.license, Count: CountExpr }).from(this.Table).groupBy(this.Table.license);
    const Filtered = await this.Db.select({ Bytes: sql<number>`coalesce(sum(bytes),0)::bigint` }).from(this.Table).where(eq(this.Table.tier, "Filtered"));

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
    if (this.OwnsSql) await this.Sql.end(); // don't close a shared client we didn't open
  }
}

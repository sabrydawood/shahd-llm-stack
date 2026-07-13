// Drizzle schema for the Data Foundry's Postgres store (M3b). One `documents` table. The embedding
// is stored as JSONB (number[]), which works on ANY Postgres with no extension — portable and
// verifiable on a stock server. Similarity ranking is done in-app (cosine); for large-scale vector
// search, swap the column to pgvector (docker-compose ships the pgvector image) and push the ranking
// into SQL. The JSONB path keeps the adapter dependency-light and universally runnable.

import { pgTable, text, integer, real, jsonb } from "drizzle-orm/pg-core";

export const EmbeddingDimensions = 256;

export const Documents = pgTable("documents", {
  id: text("id").primaryKey(),
  tier: text("tier").notNull(),
  origin: text("origin").notNull(),
  source: text("source").notNull(),
  license: text("license").notNull(),
  lang: text("lang").notNull(),
  content: text("content").notNull(),
  bytes: integer("bytes").notNull(),
  quality: real("quality_score").notNull(),
  contentHash: text("content_hash").notNull(),
  embedding: jsonb("embedding").$type<number[]>().notNull(),
  rejectReason: text("reject_reason"),
  provenance: text("provenance").notNull(),
  ingestedAt: text("ingested_at").notNull(),
});

export type DocumentRow = typeof Documents.$inferSelect;
export type DocumentInsert = typeof Documents.$inferInsert;

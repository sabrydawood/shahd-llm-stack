// Create the documents table (M3b). Portable: JSONB embedding, no extension required, runs on any
// Postgres. Run once against a live database:
//   docker compose up -d   (or point DATABASE_URL at an existing Postgres)
//   bun run Scripts/FoundryMigrate.ts

import postgres from "postgres";

const Url = process.env["DATABASE_URL"] ?? "postgres://postgres:postgres@localhost:5432/shahd";
const Sql = postgres(Url);

await Sql`
  CREATE TABLE IF NOT EXISTS documents (
    id            text PRIMARY KEY,
    tier          text NOT NULL,
    origin        text NOT NULL,
    source        text NOT NULL,
    license       text NOT NULL,
    lang          text NOT NULL,
    content       text NOT NULL,
    bytes         integer NOT NULL,
    quality_score real NOT NULL,
    content_hash  text NOT NULL,
    embedding     jsonb NOT NULL,
    reject_reason text,
    provenance    text NOT NULL,
    ingested_at   text NOT NULL
  )`;
await Sql`CREATE INDEX IF NOT EXISTS documents_tier_idx ON documents (tier)`;

await Sql.end();
console.log("migrated: documents table ready (portable JSONB embedding)");

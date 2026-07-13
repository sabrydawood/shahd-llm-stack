# Data Foundry

The Foundry is the tiered, inspectable dataset layer. Documents are classified into three tiers,
each kept with full provenance so the corpus is auditable and improvable:

- **Filtered** — permissive license + passed the quality filter → eligible for training.
- **Raw** — general web / unverified → kept for inspection only, never trained until licensed.
- **Rejected** — non-permissive or low-quality → kept with the reason.

It runs two ways: **in-memory** (no setup, used by tests and quick local runs) and **Postgres +
pgvector** (persistent, semantic search, dashboard-backed). Both sit behind the same `DocumentStore`
interface, so ingestion, reports, export, and the dashboard are identical either way.

## In-memory (no database)

```bash
bun run foundry            # ingest the seed corpus, print the quality report + export
bun run foundry:dashboard  # serve the inspection dashboard at http://localhost:8090
```

## Postgres + pgvector

```bash
docker compose up -d              # start Postgres 16 + pgvector (see docker-compose.yml)
cp .env.example .env              # DATABASE_URL=postgres://shahd:shahd@localhost:5432/shahd
bun run foundry:migrate           # create the vector extension + documents table
bun run foundry:smoke             # ingest the seed corpus into Postgres + a similarity query
```

The dashboard can be pointed at Postgres by constructing a `PostgresDocumentStore(DATABASE_URL)` and
passing it to `StartDashboard` — the same handler serves either store.

## Schema

One `documents` table (see `Brain/Foundry/FoundrySchema.ts`): id (content hash), tier, origin,
source, license, lang, content, bytes, quality_score, content_hash, `embedding vector(256)`,
reject_reason, provenance, ingested_at. The embedding dimension must equal `Config.Data.EmbeddingDim`.

## Web ingestion

Network ingestion is behind `Config.Data.WebEnabled` (off by default). Callers tag inputs with an
`Origin`: `local` and `web-permissive` are license-checked and can reach the Filtered tier;
`web-general` is always routed to the isolated **Raw** tier (inspect-only, never training-eligible)
regardless of license, because the license is unverified.

// One-off maintenance: retro-clean documents ingested BEFORE the license-header strip landed.
// Scans rows that still carry a leading LICENSE/COPYRIGHT banner, strips it, and re-ingests the
// cleaned content through the normal path (IngestDocuments — which recomputes id = hash(origin +
// license + content), content hash, embedding, and bytes), then deletes the old row. Because the id
// is content-derived, a cleaned row gets a NEW primary key, so the stale dirty row must be removed
// explicitly. Idempotent: a second run finds nothing. Upsert-then-delete ordering means a crash
// mid-run can at worst leave a duplicate (fixed by re-running), never lose data.
//
//   bun run Scripts/NormalizeCorpus.ts            # migrate in place
//   bun run Scripts/NormalizeCorpus.ts --DryRun   # count only, no writes

import postgres from "postgres";
import { IngestDocuments } from "../Foundry/FoundryBarrel.ts";
import type { SourceInput, Origin } from "../Foundry/FoundryBarrel.ts";
import { StripLicenseHeader } from "../Foundry/ContentNormalizer.ts";
import { ResolveStore } from "./FoundryEnv.ts";
import { ReadFlag } from "./ScriptArgs.ts";

type DirtyRow = { id: string; content: string; source: string; license: string; lang: string; provenance: string; origin: string; ingested_at: string };

const Url = process.env["DATABASE_URL"];
if (Url === undefined || Url === "") {
  throw new Error("NormalizeCorpus needs DATABASE_URL (this migration only applies to the Postgres store)");
}
const DryRun = ReadFlag("--DryRun");
const BatchSize = 300;

const { Store, Kind } = ResolveStore();
if (Kind !== "postgres") throw new Error(`expected postgres store, got ${Kind}`);
const Sql = postgres(Url);

// Rows that plausibly begin with a license banner (SQL pre-filter; the real decision is made in JS
// by StripLicenseHeader, which only strips a genuine LEADING license block).
function SelectDirty(AfterId: string): Promise<DirtyRow[]> {
  return Sql<DirtyRow[]>`
    select id, content, source, license, lang, provenance, origin, ingested_at
    from documents
    where id > ${AfterId}
      and ( (content like '/*%' and (content ilike '%copyright%' or content ilike '%licensed under%' or content ilike '%SPDX-License%'))
         or ((content like '//%' or content like '#%') and (content ilike '%copyright%' or content ilike '%licensed under%')) )
    order by id
    limit ${BatchSize}`;
}

let AfterId = "";
let Scanned = 0;
let Migrated = 0;
let Unchanged = 0;

for (;;) {
  const Rows = await SelectDirty(AfterId);
  if (Rows.length === 0) break;
  for (const Row of Rows) {
    AfterId = Row.id; // advance the cursor past every scanned row (incl. unchanged ones)
    Scanned++;
    const Cleaned = StripLicenseHeader(Row.content);
    if (Cleaned === Row.content) {
      Unchanged++; // copyright lives in the body, not a leading banner — leave it alone
      continue;
    }
    if (DryRun) {
      Migrated++;
      continue;
    }
    const Input: SourceInput = { Source: Row.source, License: Row.license, Lang: Row.lang, Content: Cleaned, Provenance: Row.provenance, Origin: Row.origin as Origin };
    await IngestDocuments([Input], Store, Row.ingested_at); // upsert the cleaned row under its new id
    await Sql`delete from documents where id = ${Row.id}`; // then drop the stale dirty row
    Migrated++;
  }
  console.log(`  scanned=${Scanned}  migrated=${Migrated}  unchanged=${Unchanged}`);
}

const Total = (await Sql`select count(*)::int c from documents`)[0]?.["c"] ?? 0;
console.log(`${DryRun ? "[dry-run] would migrate" : "migrated"} ${Migrated} rows (scanned ${Scanned}, ${Unchanged} left as body-only). Corpus now ${Total} docs.`);
await Sql.end();
process.exit(0);

// One-time license backfill (Phase 3b). GitHub's search API tags many repos "NOASSERTION" (its own
// matcher could not classify the LICENSE file), so the Foundry rejected every file from them on
// license grounds — even the ones that are actually MIT/BSD/Apache. This re-fetches each such repo's
// real LICENSE text, runs the strict local SpdxDetector, and — ONLY when the text is provably a
// single clean permissive license — promotes that repo's quality-passing docs from Rejected to
// Filtered. Copyleft / commercial / mixed / unresolved repos stay rejected (the legal guardrail).
// Dry-run by default; pass --Apply to write. Needs DATABASE_URL + GITHUB_TOKEN.
//
//   bun run Scripts/BackfillLicenses.ts            # dry run: report what WOULD change
//   bun run Scripts/BackfillLicenses.ts --Apply    # write the reclassification

import postgres from "postgres";
import { PostgresDocumentStore } from "../Foundry/PostgresDocumentStore.ts";
import { DetectSpdx } from "../Foundry/SpdxDetector.ts";
import { FetchRepoLicense } from "../Foundry/GitHubHttp.ts";
import { DatabaseUrl, GitHubToken } from "./FoundryEnv.ts";
import { ReadFlag } from "./ScriptArgs.ts";

const Apply = ReadFlag("--Apply");
const MinQuality = 0.6; // matches ScoreCodeQuality's pass Threshold — promote only quality-passing docs
const Url = DatabaseUrl();
const Token = GitHubToken();

const Sql = postgres(Url);
const Sources = await Sql<{ source: string; n: number }[]>`
  select source, count(*)::int n from documents where license = 'NOASSERTION' group by source order by n desc`;
console.log(`NOASSERTION: ${Sources.length} repos, ${Sources.reduce((A, R) => A + R.n, 0)} docs. Mode: ${Apply ? "APPLY" : "DRY-RUN"}\n`);

const Store = new PostgresDocumentStore(Url);
let PromoteRepos = 0;
let PromoteDocs = 0;
let KeepRepos = 0;
for (const { source, n } of Sources) {
  let Lic: Awaited<ReturnType<typeof FetchRepoLicense>> = null;
  try {
    Lic = await FetchRepoLicense(source, Token);
  } catch (Caught) {
    console.log(`KEEP  ${String(n).padStart(5)}  ${source}  (fetch error: ${(Caught as Error).message})`);
    KeepRepos++;
    continue;
  }
  if (Lic === null) {
    console.log(`KEEP  ${String(n).padStart(5)}  ${source}  (no LICENSE file)`);
    KeepRepos++;
    continue;
  }
  const Det = DetectSpdx(Lic.Text);
  if (!Det.Permissive || Det.Spdx === null) {
    console.log(`KEEP  ${String(n).padStart(5)}  ${source}  (${Det.Note})`);
    KeepRepos++;
    continue;
  }
  if (Apply) {
    const { Promoted, KeptLowQuality } = await Store.ReclassifyBySource(source, Det.Spdx, MinQuality);
    console.log(`PROMOTE ${String(Promoted).padStart(5)}  ${source}  -> ${Det.Spdx}  (kept ${KeptLowQuality} low-quality rejected)`);
    PromoteDocs += Promoted;
  } else {
    console.log(`PROMOTE ${String(n).padStart(5)}  ${source}  -> ${Det.Spdx}  (dry-run; before quality gate)`);
    PromoteDocs += n;
  }
  PromoteRepos++;
}
console.log(`\n${Apply ? "APPLIED" : "DRY-RUN"}: promoted ${PromoteRepos} repos (${PromoteDocs} docs${Apply ? "" : " before quality gate"}); kept ${KeepRepos} repos rejected.`);
if (!Apply) console.log("Re-run with --Apply to write the changes.");
await Store.Close();
await Sql.end();

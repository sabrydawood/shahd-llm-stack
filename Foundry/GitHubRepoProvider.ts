// Whole-repo GitHub provider (M7): search repositories, then for each download the ENTIRE repo
// (tarball), assess its level, and — if it meets the minimum level — ingest EVERY substantive source
// file so the model learns the whole project, not a scattered sample. Repos below the minimum level
// are skipped (understand the level, learn from the good ones). Both fetchers are injected (mock in
// tests; real GitHub by default) — a token is recommended for the tarball rate limit.

import type { WebProvider } from "./WebSource.ts";
import type { SourceInput } from "./Ingest.ts";
import type { HttpJson, FetchBytes } from "./GitHubHttp.ts";
import { DefaultGitHubJson, DefaultGitHubBytes } from "./GitHubHttp.ts";
import { FetchRepoFiles } from "./RepoArchive.ts";
import { AssessRepo, LevelRank, EmptyAssessment } from "./RepoQuality.ts";
import type { RepoLevel, RepoIngestInfo } from "./RepoQuality.ts";
import { LangForPath } from "./CodeFileFilter.ts";

type RepoItem = { full_name: string; default_branch: string; license: { spdx_id: string } | null };
type SearchResult = { items?: RepoItem[] };

export type GitHubRepoOptions = {
  Token?: string;
  Http?: HttpJson; // repo search (JSON)
  FetchBytes?: FetchBytes; // repo tarball (bytes)
  MaxFilesPerRepo?: number; // cap so a monorepo can't dominate (generous by default)
  MaxBytesPerRepo?: number; // byte budget per repo
  MinLevel?: RepoLevel; // skip repos below this level
  SkipRepo?: (Repo: string) => boolean; // skip already-learned repos (before download)
  OnRepo?: (Info: RepoIngestInfo) => void; // progress/reporting hook
};

export function CreateGitHubRepoProvider(Options: GitHubRepoOptions = {}): WebProvider {
  const Http = Options.Http ?? DefaultGitHubJson(Options.Token);
  const Fetch = Options.FetchBytes ?? DefaultGitHubBytes(Options.Token);
  const Limits = { MaxFiles: Options.MaxFilesPerRepo, MaxBytes: Options.MaxBytesPerRepo };
  const MinLevel = Options.MinLevel ?? "medium";

  return {
    Name: "github-repo",
    Fetch: async (Query: string, Limit: number): Promise<SourceInput[]> => {
      const Search = (await Http(`https://api.github.com/search/repositories?q=${encodeURIComponent(Query)}&per_page=${Limit}`)) as SearchResult;
      const Out: SourceInput[] = [];
      for (const Repo of Search.items ?? []) {
        const License = Repo.license?.spdx_id ?? "unknown";
        if (Options.SkipRepo?.(Repo.full_name) === true) {
          Options.OnRepo?.({ Repo: Repo.full_name, License, Assessment: EmptyAssessment, Ingested: false, Reason: "already learned" });
          continue; // don't re-download a repo we already ingested
        }
        const Files = await FetchRepoFiles(`https://api.github.com/repos/${Repo.full_name}/tarball/${Repo.default_branch}`, Fetch, Limits);
        const Assessment = AssessRepo(Files);
        const Ingested = LevelRank[Assessment.Level] >= LevelRank[MinLevel];
        Options.OnRepo?.({ Repo: Repo.full_name, License, Assessment, Ingested });
        if (!Ingested) continue; // skip low-level repos entirely
        for (const File of Files) {
          Out.push({
            Source: Repo.full_name,
            License,
            Lang: LangForPath(File.Path),
            Content: File.Content,
            Provenance: `https://github.com/${Repo.full_name}/blob/${Repo.default_branch}/${File.Path}`,
            Origin: "web-permissive",
          });
        }
      }
      return Out;
    },
  };
}

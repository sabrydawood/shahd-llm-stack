// GitHub web provider (M6): pulls SUBSTANTIVE code files from repositories via the GitHub API,
// tagging each with the repo's detected SPDX license and Origin "web-permissive". File selection
// goes through CodeFileFilter (rule #4): non-source paths (tests/examples/dist/vendor/docs/dotfiles),
// declarations/minified/generated/config files, and markup/style are excluded, source dirs are
// preferred, and each candidate's content is gated (no stubs, blobs, or license-header-only files) —
// so it does NOT ingest the config/benchmark junk that first-N-by-tree-order pulled in. HTTP is
// injected (a mock in tests; the real fetch, with an optional token, by default).

import type { WebProvider } from "./WebSource.ts";
import type { SourceInput } from "./Ingest.ts";
import type { HttpJson } from "./GitHubHttp.ts";
import { DefaultGitHubJson } from "./GitHubHttp.ts";
import { IsSubstantiveCodePath, IsSubstantiveCodeContent, RankCodePath, LangForPath } from "./CodeFileFilter.ts";

export type { HttpJson } from "./GitHubHttp.ts";

type RepoItem = { full_name: string; default_branch: string; license: { spdx_id: string } | null };
type SearchResult = { items?: RepoItem[] };
type TreeEntry = { path: string; type: string };
type TreeResult = { tree?: TreeEntry[] };
type ContentResult = { content?: string; encoding?: string };

function DecodeContent(Content: ContentResult): string {
  return Content.encoding === "base64" && Content.content !== undefined
    ? Buffer.from(Content.content, "base64").toString("utf8")
    : (Content.content ?? "");
}

export type GitHubOptions = { Token?: string; Http?: HttpJson; FilesPerRepo?: number };

export function CreateGitHubProvider(Options: GitHubOptions = {}): WebProvider {
  const Http = Options.Http ?? DefaultGitHubJson(Options.Token);
  const FilesPerRepo = Options.FilesPerRepo ?? 3;
  return {
    Name: "github",
    Fetch: async (Query: string, Limit: number): Promise<SourceInput[]> => {
      const Search = (await Http(`https://api.github.com/search/repositories?q=${encodeURIComponent(Query)}&per_page=${Limit}`)) as SearchResult;
      const Out: SourceInput[] = [];
      for (const Repo of Search.items ?? []) {
        const License = Repo.license?.spdx_id ?? "unknown";
        const Tree = (await Http(`https://api.github.com/repos/${Repo.full_name}/git/trees/${Repo.default_branch}?recursive=1`)) as TreeResult;
        // Substantive source paths, source dirs first.
        const Candidates = (Tree.tree ?? [])
          .filter((E) => E.type === "blob" && IsSubstantiveCodePath(E.path))
          .sort((A, B) => RankCodePath(A.path) - RankCodePath(B.path));
        // Fetch candidates until FilesPerRepo pass the content gate (bounded to limit API calls).
        const MaxAttempts = Math.min(Candidates.length, FilesPerRepo * 6);
        let Kept = 0;
        for (let I = 0; I < MaxAttempts && Kept < FilesPerRepo; I++) {
          const File = Candidates[I];
          const Text = DecodeContent((await Http(`https://api.github.com/repos/${Repo.full_name}/contents/${encodeURIComponent(File.path)}`)) as ContentResult);
          if (!IsSubstantiveCodeContent(Text)) continue;
          Out.push({
            Source: Repo.full_name,
            License,
            Lang: LangForPath(File.path),
            Content: Text,
            Provenance: `https://github.com/${Repo.full_name}/blob/${Repo.default_branch}/${File.path}`,
            Origin: "web-permissive",
          });
          Kept++;
        }
      }
      return Out;
    },
  };
}

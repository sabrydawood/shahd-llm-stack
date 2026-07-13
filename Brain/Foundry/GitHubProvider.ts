// GitHub web provider (M6): pulls code files from repositories via the GitHub API, tagging each with
// the repo's detected SPDX license and Origin "web-permissive" — so the normal tiering keeps only
// permissive ones (the rest are Rejected with a reason). HTTP is injected (a mock in tests; the real
// fetch, with an optional token, by default), so this is testable without the network.

import type { WebProvider } from "./WebSource.ts";
import type { SourceInput } from "./Ingest.ts";

export type HttpJson = (Url: string) => Promise<unknown>;

type RepoItem = { full_name: string; default_branch: string; license: { spdx_id: string } | null };
type SearchResult = { items?: RepoItem[] };
type TreeEntry = { path: string; type: string };
type TreeResult = { tree?: TreeEntry[] };
type ContentResult = { content?: string; encoding?: string };

const CodeExtension = /\.(ts|tsx|js|jsx|go|py|rs|java|c|cpp|h|rb|php|cs|kt|swift|scala)$/;
const LangByExtension: Record<string, string> = {
  ts: "typescript", tsx: "typescript", js: "javascript", jsx: "javascript", go: "go", py: "python",
  rs: "rust", java: "java", c: "c", cpp: "cpp", h: "c", rb: "ruby", php: "php", cs: "csharp",
  kt: "kotlin", swift: "swift", scala: "scala",
};

function LangOf(Path: string): string {
  const Ext = Path.split(".").pop() ?? "";
  return LangByExtension[Ext] ?? "unknown";
}

function DefaultHttp(Token?: string): HttpJson {
  return async (Url: string): Promise<unknown> => {
    const Headers: Record<string, string> = { "User-Agent": "shahd-foundry", Accept: "application/vnd.github+json" };
    if (Token !== undefined) Headers["Authorization"] = `Bearer ${Token}`;
    const Response = await fetch(Url, { headers: Headers });
    if (!Response.ok) throw new Error(`GitHub API ${Response.status}`);
    return Response.json();
  };
}

export type GitHubOptions = { Token?: string; Http?: HttpJson; FilesPerRepo?: number };

export function CreateGitHubProvider(Options: GitHubOptions = {}): WebProvider {
  const Http = Options.Http ?? DefaultHttp(Options.Token);
  const FilesPerRepo = Options.FilesPerRepo ?? 3;
  return {
    Name: "github",
    Fetch: async (Query: string, Limit: number): Promise<SourceInput[]> => {
      const Search = (await Http(`https://api.github.com/search/repositories?q=${encodeURIComponent(Query)}&per_page=${Limit}`)) as SearchResult;
      const Out: SourceInput[] = [];
      for (const Repo of Search.items ?? []) {
        const License = Repo.license?.spdx_id ?? "unknown";
        const Tree = (await Http(`https://api.github.com/repos/${Repo.full_name}/git/trees/${Repo.default_branch}?recursive=1`)) as TreeResult;
        const Files = (Tree.tree ?? []).filter((E) => E.type === "blob" && CodeExtension.test(E.path)).slice(0, FilesPerRepo);
        for (const File of Files) {
          const Content = (await Http(`https://api.github.com/repos/${Repo.full_name}/contents/${encodeURIComponent(File.path)}`)) as ContentResult;
          const Text = Content.encoding === "base64" && Content.content !== undefined
            ? Buffer.from(Content.content, "base64").toString("utf8")
            : (Content.content ?? "");
          Out.push({
            Source: Repo.full_name,
            License,
            Lang: LangOf(File.path),
            Content: Text,
            Provenance: `https://github.com/${Repo.full_name}/blob/${Repo.default_branch}/${File.path}`,
            Origin: "web-permissive",
          });
        }
      }
      return Out;
    },
  };
}

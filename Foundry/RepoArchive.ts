// Download and extract a WHOLE repository in one request (M7). The GitHub tarball endpoint returns
// the entire repo as one gzipped tar, so we fetch it once (instead of hundreds of per-file API
// calls) and keep every substantive source file — the model learns from complete projects, not a
// scattered sample. Extraction reuses CodeFileFilter for path + content quality.

import { parseTarGzip } from "nanotar";
import type { FetchBytes } from "./GitHubHttp.ts";
import { IsSubstantiveCodePath, IsSubstantiveCodeContent } from "./CodeFileFilter.ts";

export type RepoFile = { Path: string; Content: string };

/** GitHub tarballs prefix every entry with "{repo}-{sha}/"; drop that top-level directory. */
function StripRoot(Name: string): string {
  const Slash = Name.indexOf("/");
  return Slash === -1 ? Name : Name.slice(Slash + 1);
}

/** Fetch a repo tarball and return its substantive source files (path-filtered + content-gated). */
export async function FetchRepoFiles(TarballUrl: string, Fetch: FetchBytes, MaxFiles = 400): Promise<RepoFile[]> {
  const Items = await parseTarGzip(await Fetch(TarballUrl));
  const Out: RepoFile[] = [];
  for (const Item of Items) {
    if (Item.type !== "file") continue;
    const Path = StripRoot(Item.name);
    if (!IsSubstantiveCodePath(Path)) continue;
    const Content = Item.text ?? new TextDecoder().decode(Item.data);
    if (!IsSubstantiveCodeContent(Content)) continue;
    Out.push({ Path, Content });
    if (Out.length >= MaxFiles) break;
  }
  return Out;
}

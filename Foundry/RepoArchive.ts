// Download and extract a WHOLE repository in one request (M7). The GitHub tarball endpoint returns
// the entire repo as one gzipped tar, so we fetch it once (instead of hundreds of per-file API
// calls) and keep every substantive source file — the model learns from complete projects, not a
// scattered sample. Extraction reuses CodeFileFilter for path + content quality.

import { parseTarGzip } from "nanotar";
import type { FetchBytes } from "./GitHubHttp.ts";
import { IsSubstantiveCodePath, IsSubstantiveCodeContent } from "./CodeFileFilter.ts";
import { StripLicenseHeader } from "./ContentNormalizer.ts";

export type RepoFile = { Path: string; Content: string };

// Generous defaults so a large but organized repo is learned WHOLE; the byte budget only guards
// against a pathological monorepo. MaxContentBytes caps a single file (larger ones are dropped, not
// truncated). Raise them for truly massive repos.
export type RepoLimits = { MaxFiles?: number; MaxBytes?: number; MaxContentBytes?: number };
export const DefaultRepoLimits: Required<RepoLimits> = { MaxFiles: 8000, MaxBytes: 64_000_000, MaxContentBytes: 512_000 };

/** GitHub tarballs prefix every entry with "{repo}-{sha}/"; drop that top-level directory. */
function StripRoot(Name: string): string {
  const Slash = Name.indexOf("/");
  return Slash === -1 ? Name : Name.slice(Slash + 1);
}

/** Fetch a repo tarball and return its substantive source files (path-filtered + content-gated). */
export async function FetchRepoFiles(TarballUrl: string, Fetch: FetchBytes, Limits: RepoLimits = {}): Promise<RepoFile[]> {
  const MaxFiles = Limits.MaxFiles ?? DefaultRepoLimits.MaxFiles;
  const MaxBytes = Limits.MaxBytes ?? DefaultRepoLimits.MaxBytes;
  const MaxContentBytes = Limits.MaxContentBytes ?? DefaultRepoLimits.MaxContentBytes;
  const Items = await parseTarGzip(await Fetch(TarballUrl));
  const Out: RepoFile[] = [];
  let Bytes = 0;
  for (const Item of Items) {
    if (Item.type !== "file") continue;
    const Path = StripRoot(Item.name);
    if (!IsSubstantiveCodePath(Path)) continue;
    const Content = StripLicenseHeader(Item.text ?? new TextDecoder().decode(Item.data));
    if (!IsSubstantiveCodeContent(Content, MaxContentBytes)) continue;
    Out.push({ Path, Content });
    Bytes += Content.length;
    if (Out.length >= MaxFiles || Bytes >= MaxBytes) break;
  }
  return Out;
}

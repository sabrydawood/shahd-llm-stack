// Shared GitHub HTTP fetchers (rule #4: one home for auth + fetch). Both the file-sampling provider
// and the whole-repo provider use these. Injected in tests; the real fetch (with an optional token)
// by default. JSON for search/tree/contents; bytes for the repo tarball.

export type HttpJson = (Url: string) => Promise<unknown>;
export type FetchBytes = (Url: string) => Promise<Uint8Array>;

function AuthHeaders(Token: string | undefined, Accept: string): Record<string, string> {
  const Headers: Record<string, string> = { "User-Agent": "shahd-foundry", Accept };
  if (Token !== undefined) Headers["Authorization"] = `Bearer ${Token}`;
  return Headers;
}

export function DefaultGitHubJson(Token?: string): HttpJson {
  return async (Url: string): Promise<unknown> => {
    const Response = await fetch(Url, { headers: AuthHeaders(Token, "application/vnd.github+json") });
    if (!Response.ok) throw new Error(`GitHub API ${Response.status}`);
    return Response.json();
  };
}

export function DefaultGitHubBytes(Token?: string): FetchBytes {
  return async (Url: string): Promise<Uint8Array> => {
    const Response = await fetch(Url, { headers: AuthHeaders(Token, "application/vnd.github+json") });
    if (!Response.ok) throw new Error(`GitHub API ${Response.status}`);
    return new Uint8Array(await Response.arrayBuffer());
  };
}

// Fetch a repo's LICENSE file (raw text) via GET /repos/{owner}/{repo}/license. Returns the SPDX id
// GitHub's own matcher assigned (often "NOASSERTION" — GitHub can't classify a deviant file) plus the
// decoded license text, which is what a stricter local matcher (SpdxDetector) then classifies. Null
// when the repo has no license file (404) or on any API error — the caller treats that as unresolved.
export type RepoLicense = { Spdx: string; Text: string };
export async function FetchRepoLicense(FullName: string, Token?: string): Promise<RepoLicense | null> {
  const Response = await fetch(`https://api.github.com/repos/${FullName}/license`, { headers: AuthHeaders(Token, "application/vnd.github+json") });
  if (!Response.ok) return null;
  const Body = (await Response.json()) as { license?: { spdx_id?: string }; content?: string; encoding?: string };
  const Text = Body.content !== undefined && Body.encoding === "base64" ? Buffer.from(Body.content, "base64").toString("utf8") : "";
  return { Spdx: Body.license?.spdx_id ?? "NOASSERTION", Text };
}

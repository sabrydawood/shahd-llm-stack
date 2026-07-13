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

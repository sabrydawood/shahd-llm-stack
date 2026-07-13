// General web-search provider (M6): searches, fetches each result page, converts it to text, and
// tags it Origin "web-general" — which tiering ALWAYS routes to the isolated Raw tier (inspect-only,
// never training-eligible, license unverified). The Search backend is REQUIRED and injected (plug in
// Brave/Bing/SerpAPI/etc.), so this never silently fakes results and carries no hard network/API
// dependency; page Fetch is injected too (real fetch by default).

import type { WebProvider } from "./WebSource.ts";
import type { SourceInput } from "./Ingest.ts";
import { HtmlToText } from "./HtmlToText.ts";

export type SearchHit = { Url: string; Title: string };
export type SearchBackend = (Query: string, Limit: number) => Promise<SearchHit[]>;
export type PageFetch = (Url: string) => Promise<string>;

export type WebSearchOptions = { Search: SearchBackend; Fetch?: PageFetch; MaxChars?: number };

export function CreateWebSearchProvider(Options: WebSearchOptions): WebProvider {
  const Fetch = Options.Fetch ?? (async (Url: string): Promise<string> => (await fetch(Url)).text());
  const MaxChars = Options.MaxChars ?? 20000;
  return {
    Name: "web-search",
    Fetch: async (Query: string, Limit: number): Promise<SourceInput[]> => {
      const Hits = await Options.Search(Query, Limit);
      const Out: SourceInput[] = [];
      for (const Hit of Hits) {
        try {
          const Text = HtmlToText(await Fetch(Hit.Url)).slice(0, MaxChars);
          if (Text.length > 0) {
            Out.push({ Source: "web-search", License: "unknown", Lang: "unknown", Content: Text, Provenance: Hit.Url, Origin: "web-general" });
          }
        } catch {
          // Skip a page that fails to fetch.
        }
      }
      return Out;
    },
  };
}

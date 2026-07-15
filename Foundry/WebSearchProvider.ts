// General web-search provider (M6): searches, fetches each result page, converts it to text, and
// tags it Origin "web-general" — which tiering ALWAYS routes to the isolated Raw tier (inspect-only,
// never training-eligible, license unverified). The Search backend is REQUIRED and injected (plug in
// Brave/Bing/SerpAPI/etc.), so this never silently fakes results and carries no hard network/API
// dependency; page Fetch is injected too (real fetch by default).

import type { WebProvider } from "./WebSource.ts";
import type { SourceInput } from "./Ingest.ts";
import { HtmlToText } from "./HtmlToText.ts";

export type SearchHit = { Url: string; Title: string };
export type SearchBackend = (Query: string, Limit: number, Signal?: AbortSignal) => Promise<SearchHit[]>;
export type PageFetch = (Url: string, Signal?: AbortSignal) => Promise<string>;

export type WebSearchOptions = { Search: SearchBackend; Fetch?: PageFetch; MaxChars?: number };

// Read a response body with a hard byte ceiling instead of buffering the whole page: the search
// backend returns arbitrary third-party URLs, so an unbounded `.text()` lets one huge/streaming page
// spike memory before MaxChars ever truncates. Stop reading once enough bytes are in to fill MaxChars
// after HtmlToText strips markup (~8x headroom for tags/whitespace), then decode only what was read.
async function FetchCapped(Url: string, MaxBytes: number, Signal?: AbortSignal): Promise<string> {
  const Response = await fetch(Url, { signal: Signal });
  const Body = Response.body;
  if (Body === null) return (await Response.text()).slice(0, MaxBytes);
  const Reader = Body.getReader();
  const Decoder = new TextDecoder();
  let Text = "";
  for (;;) {
    const { done: Done, value: Value } = await Reader.read();
    if (Done) break;
    if (Value !== undefined) {
      Text += Decoder.decode(Value, { stream: true });
      if (Text.length >= MaxBytes) {
        await Reader.cancel();
        break;
      }
    }
  }
  return Text;
}

export function CreateWebSearchProvider(Options: WebSearchOptions): WebProvider {
  const MaxChars = Options.MaxChars ?? 20000;
  const Fetch = Options.Fetch ?? ((Url: string, Signal?: AbortSignal): Promise<string> => FetchCapped(Url, MaxChars * 8, Signal));
  return {
    Name: "web-search",
    Fetch: async (Query: string, Limit: number, Signal?: AbortSignal): Promise<SourceInput[]> => {
      const Hits = await Options.Search(Query, Limit, Signal);
      const Out: SourceInput[] = [];
      for (const Hit of Hits) {
        try {
          const Text = HtmlToText(await Fetch(Hit.Url, Signal)).slice(0, MaxChars);
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

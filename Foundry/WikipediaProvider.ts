// Wikipedia article provider (Phase 8) — broad general-knowledge text, strong in Arabic and English.
// Pulls random articles' plain-text extracts via the MediaWiki API (reliable JSON, no scraping) and
// stores them as curated documents. License is CC-BY-SA (share-alike) — NOT on the permissive code
// allowlist; it is included as an explicitly Sabry-approved general-text source (Origin "curated"),
// with the license recorded on every document for provenance. The JSON fetch is injected (mock in
// tests; real MediaWiki API by default). Random-article dups across requests are deduped downstream
// by content hash.

import type { WebProvider, RepoSink } from "./WebSource.ts";
import type { SourceInput } from "./Ingest.ts";

export type WikipediaOptions = {
  FetchJson?: (Url: string) => Promise<unknown>; // injected in tests; real MediaWiki API by default
  BatchSize?: number;
  MinChars?: number; // skip stubs / near-empty extracts
  OnRepoStart?: (Name: string) => void; // working signal + Stop boundary (throws on abort)
  OnRepoReady?: RepoSink;
  Log?: (Message: string) => void;
};

type WikiPage = { title?: string; extract?: string; pageid?: number };
type WikiResponse = { query?: { pages?: Record<string, WikiPage> } };

async function DefaultFetchJson(Url: string): Promise<unknown> {
  const Response = await fetch(Url, { headers: { "User-Agent": "shahd-foundry (educational LM research)" } });
  if (!Response.ok) throw new Error(`Wikipedia API ${Response.status}`);
  return Response.json();
}

export function CreateWikipediaProvider(Options: WikipediaOptions = {}): WebProvider {
  const FetchJson = Options.FetchJson ?? DefaultFetchJson;
  const BatchSize = Options.BatchSize ?? 100;
  const MinChars = Options.MinChars ?? 200;
  const Log = Options.Log ?? ((Message: string): void => console.log(Message));

  return {
    Name: "wikipedia",
    Fetch: async (Query: string, Limit: number): Promise<SourceInput[]> => {
      const Lang = (Query || "en").trim().toLowerCase();
      // Validate the language code STRICTLY (a real one is just letters, optionally hyphenated, e.g.
      // "en", "pt-br"). Without this, a value like "127.0.0.1:6379/x" makes the request host become
      // attacker-controlled — a server-side request forgery (SSRF) — since a URL's authority ends at
      // the first "/". An allow-list of the expected shape closes it without touching legitimate use.
      if (!/^[a-z]{2,10}(-[a-z]{2,10})?$/.test(Lang)) {
        throw new Error(`Wikipedia: invalid language code "${Lang}" (expected e.g. "en", "ar", "pt-br")`);
      }
      const Api = `https://${Lang}.wikipedia.org/w/api.php?action=query&generator=random&grnnamespace=0&grnlimit=10&prop=extracts&explaintext=1&exlimit=max&format=json&origin=*`;
      const Docs: SourceInput[] = [];
      let Batch: SourceInput[] = [];
      const Flush = async (): Promise<void> => {
        if (Batch.length > 0 && Options.OnRepoReady !== undefined) {
          Options.OnRepoStart?.(`Wikipedia batch (${Docs.length}/${Limit})`); // Stop between batches
          await Options.OnRepoReady(`wikipedia-${Lang}`, Batch);
          Batch = [];
        }
      };

      // The random generator returns ~10 articles/request (and can repeat), so request until Limit —
      // with a safety cap so an all-stub/all-dup run can't loop forever.
      const MaxRequests = Math.ceil(Limit / 10) + 10;
      for (let Request = 0; Request < MaxRequests && Docs.length < Limit; Request++) {
        Options.OnRepoStart?.(`Wikipedia ${Lang} (${Docs.length}/${Limit})`);
        let Json: WikiResponse;
        try {
          Json = (await FetchJson(Api)) as WikiResponse;
        } catch (Caught) {
          Log(`[wiki] fetch error: ${(Caught as Error).message}`);
          continue;
        }
        const Pages = Json.query?.pages ?? {};
        for (const Key of Object.keys(Pages)) {
          if (Docs.length >= Limit) break;
          const Page = Pages[Key]!;
          const Extract = typeof Page.extract === "string" ? Page.extract.trim() : "";
          if (Extract.length < MinChars) continue; // skip stubs
          const Doc: SourceInput = {
            Source: `wikipedia-${Lang}`,
            License: "CC-BY-SA-4.0",
            Lang: `text-${Lang}`,
            Content: `${Page.title ?? ""}\n\n${Extract}`.trim(),
            Provenance: `wikipedia:${Lang}:${Page.pageid ?? Key}`,
            Origin: "curated",
          };
          Docs.push(Doc);
          Batch.push(Doc);
          if (Batch.length >= BatchSize) await Flush();
        }
      }
      await Flush();
      Log(`[wiki] ${Docs.length} ${Lang} articles collected`);
      return Options.OnRepoReady !== undefined ? [] : Docs;
    },
  };
}

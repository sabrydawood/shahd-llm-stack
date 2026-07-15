// OpenAssistant (OASST) conversation provider (Phase 8) — the first GENERAL/CONVERSATION data source,
// so the model can learn to talk, not just read code. OASST is Apache-2.0 (permissive) and
// multilingual (English + Arabic + more), and it is CURATED human dialogue, so it feeds the Foundry
// like any other permissive source: each prompter->assistant exchange becomes one document
// (Lang="text-<lang>", License="Apache-2.0"), tiered + deduped identically. The gz download is
// injected (mock in tests; real HuggingFace fetch by default) so this stays testable offline.

import { gunzipSync } from "node:zlib";
import type { WebProvider, RepoSink } from "./WebSource.ts";
import type { SourceInput } from "./Ingest.ts";

export const Oasst1Url = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz";
export const Oasst2Url = "https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/main/2023-11-05_oasst2_ready.trees.jsonl.gz";
const DefaultUrl = Oasst1Url;

export type OasstOptions = {
  Url?: string;
  FetchBytes?: (Url: string) => Promise<Uint8Array>; // injected in tests; real HF download by default
  BatchSize?: number; // conversations stored per incremental batch
  OnRepoStart?: (Name: string) => void; // "working" signal + Stop boundary (throws on abort)
  OnRepoReady?: RepoSink; // incremental storage, batch by batch
  Log?: (Message: string) => void;
};

// One OASST message node (prompter or assistant) with its nested replies. Only the fields we read.
type OasstNode = { text?: string; role?: string; lang?: string; message_id?: string; replies?: OasstNode[] };
type OasstTree = { prompt?: OasstNode };

async function DefaultFetch(Url: string): Promise<Uint8Array> {
  // The HF/xethub CDN occasionally drops a long (~34MB) download mid-stream ("socket connection closed
  // unexpectedly"); the download itself is fine, it is just a transient blip. Retry with backoff on a
  // fresh connection so one dropped socket doesn't fail the whole collect run. A real 4xx/5xx (not ok)
  // is not retried — only network/socket errors are.
  const MaxAttempts = 4;
  let LastError = "";
  for (let Attempt = 1; Attempt <= MaxAttempts; Attempt++) {
    try {
      const Response = await fetch(Url, { headers: { "User-Agent": "shahd-foundry" } }); // follows the HF 302 redirect
      if (!Response.ok) throw new Error(`OASST fetch ${Response.status}`); // a status error is not transient — fail fast
      return new Uint8Array(await Response.arrayBuffer());
    } catch (Caught) {
      LastError = (Caught as Error).message;
      if (LastError.includes("OASST fetch ")) throw Caught; // HTTP status error: do not retry
      console.warn(`[oasst] download attempt ${Attempt}/${MaxAttempts} failed: ${LastError}`);
      if (Attempt < MaxAttempts) await new Promise((Resolve) => setTimeout(Resolve, 1500 * Attempt)); // 1.5s, 3s, 4.5s
    }
  }
  throw new Error(`OASST download failed after ${MaxAttempts} attempts: ${LastError}`);
}

// Walk a message tree: each prompter -> its first assistant reply is one conversation; recurse into
// the assistant's follow-up prompters for multi-turn threads. Filtered by language, capped at Limit.
function Extract(Node: OasstNode | undefined, LangFilter: string, Limit: number, Out: SourceInput[]): void {
  if (Out.length >= Limit || Node === undefined || Node.role !== "prompter" || typeof Node.text !== "string") return;
  const Lang = Node.lang ?? "unknown";
  const Replies = Array.isArray(Node.replies) ? Node.replies : [];
  const Assistant = Replies.find((R) => R.role === "assistant" && typeof R.text === "string");
  if (Assistant === undefined) return;
  if (LangFilter === "all" || Lang === LangFilter) {
    Out.push({
      Source: `oasst-${Lang}`,
      License: "Apache-2.0",
      Lang: `text-${Lang}`,
      Content: `User: ${Node.text}\n\nAssistant: ${Assistant.text}`,
      Provenance: `oasst:${Node.message_id ?? "?"}`,
      Origin: "curated", // a whole dataset we vetted (Apache-2.0) — quality-gated, license recorded not re-checked
    });
  }
  for (const Next of Assistant.replies ?? []) Extract(Next, LangFilter, Limit, Out); // deeper turns
}

export function CreateOasstProvider(Options: OasstOptions = {}): WebProvider {
  const Url = Options.Url ?? DefaultUrl;
  const FetchBytes = Options.FetchBytes ?? DefaultFetch;
  const BatchSize = Options.BatchSize ?? 200;
  const Log = Options.Log ?? ((Message: string): void => console.log(Message));

  return {
    Name: "oasst",
    Fetch: async (Query: string, Limit: number): Promise<SourceInput[]> => {
      const LangFilter = (Query || "all").trim().toLowerCase();
      Options.OnRepoStart?.("OASST dataset"); // working signal + Stop boundary before the (slow) download
      Log(`[oasst] downloading ${Url} …`);
      const Gz = await FetchBytes(Url);
      const Text = gunzipSync(Buffer.from(Gz)).toString("utf8");
      const Lines = Text.split("\n");
      Log(`[oasst] ${Lines.length} trees; extracting lang=${LangFilter} conversations (max ${Limit})…`);

      const Docs: SourceInput[] = [];
      for (const Line of Lines) {
        if (Docs.length >= Limit) break;
        const Trimmed = Line.trim();
        if (Trimmed.length === 0) continue;
        try {
          Extract((JSON.parse(Trimmed) as OasstTree).prompt, LangFilter, Limit, Docs);
        } catch {
          // a malformed line is skipped, never fatal
        }
      }
      Log(`[oasst] extracted ${Docs.length} conversations; storing in batches of ${BatchSize}`);

      if (Options.OnRepoReady === undefined) return Docs; // batch mode (CLI): return all
      for (let Start = 0; Start < Docs.length; Start += BatchSize) {
        Options.OnRepoStart?.(`OASST batch ${Math.floor(Start / BatchSize) + 1}`); // Stop between batches
        await Options.OnRepoReady(`oasst-${LangFilter}-${Math.floor(Start / BatchSize)}`, Docs.slice(Start, Start + BatchSize));
      }
      Log(`[oasst] done: ${Docs.length} conversations stored`);
      return [];
    },
  };
}

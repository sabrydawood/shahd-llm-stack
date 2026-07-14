// Foundry ingestion + export flow (M3). IngestDocuments classifies each input into a tier, hashes
// its content (stable id + exact-dup key), embeds it, and upserts it to the store. The web-fetching
// itself is NOT here — callers pass inputs tagged with an Origin (local / web-permissive /
// web-general), so ingestion stays pure and testable and the risky network fetch is an injected
// concern. ExportTrainingText materializes only the training-eligible (Filtered) tier.

import { createHash } from "node:crypto";
import type { DocumentStore } from "./DocumentStore.ts";
import type { DocumentRecord, Origin, Tier } from "./DocumentRecord.ts";
import { ClassifyDocument } from "./Tiering.ts";
import { HashingEmbedding } from "./Embedding.ts";
import { SanitizeText } from "./ContentNormalizer.ts";

export type SourceInput = {
  Source: string;
  License: string;
  Lang: string;
  Content: string;
  Provenance: string;
  Origin: Origin;
};

export type IngestStats = { Ingested: number; ByTier: Record<Tier, number>; Failed: number };

export async function IngestDocuments(
  Inputs: SourceInput[],
  Store: DocumentStore,
  IngestedAt: string,
  EmbeddingDim = 256,
): Promise<IngestStats> {
  const ByTier: Record<Tier, number> = { Filtered: 0, Raw: 0, Rejected: 0 };
  let Ingested = 0;
  let Failed = 0;
  for (const Input of Inputs) {
    // Sanitize FIRST so id/hash/embedding/bytes are all computed on the exact bytes we will store —
    // and so a NUL/lone-surrogate from a binary-ish file can never make the store reject the row.
    const Content = SanitizeText(Input.Content);
    const Decision = ClassifyDocument(Input.License, Content, Input.Origin);
    const ContentHash = createHash("sha256").update(Content).digest("hex").slice(0, 32);
    // The dedup/primary key includes Origin + License so provenance-distinct copies of identical
    // bytes do NOT overwrite each other (a web-general Raw copy must not clobber a local Filtered
    // one, and a Rejected doc must not be "laundered" into Filtered by re-tagging its license).
    const Id = createHash("sha256").update(`${Input.Origin}\0${Input.License}\0${Content}`).digest("hex").slice(0, 32);
    const Record: DocumentRecord = {
      Id,
      Tier: Decision.Tier,
      Origin: Input.Origin,
      Source: Input.Source,
      License: Input.License,
      Lang: Input.Lang,
      Content,
      Bytes: Buffer.byteLength(Content, "utf8"),
      QualityScore: Decision.QualityScore,
      ContentHash,
      Embedding: HashingEmbedding(Content, EmbeddingDim),
      RejectReason: Decision.RejectReason,
      Provenance: Input.Provenance,
      IngestedAt,
    };
    // One bad row must not abort a whole multi-repo Learn run: log and continue.
    try {
      await Store.Upsert(Record);
      ByTier[Decision.Tier]++;
      Ingested++;
    } catch (Caught) {
      Failed++;
      console.warn(`IngestDocuments: skipped ${Input.Provenance}: ${(Caught as Error).message}`);
    }
  }
  return { Ingested, ByTier, Failed };
}

/** Materialize the training-eligible (Filtered) tier as one training-ready text blob. */
export async function ExportTrainingText(Store: DocumentStore, Separator = "\n\n"): Promise<string> {
  const Filtered = await Store.ByTier("Filtered");
  return Filtered.map((Doc) => Doc.Content).join(Separator);
}

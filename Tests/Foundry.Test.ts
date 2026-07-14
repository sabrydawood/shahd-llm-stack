import { test, expect } from "bun:test";
import {
  ClassifyDocument,
  HashingEmbedding,
  CosineSimilarity,
  InMemoryDocumentStore,
  IngestDocuments,
  ExportTrainingText,
  BuildReport,
  RenderReportText,
} from "../Foundry/FoundryBarrel.ts";
import type { SourceInput, DocumentRecord } from "../Foundry/FoundryBarrel.ts";

const Clean = "export function add(a, b) {\n  return a + b;\n}\n";
const GoSnippet = "package main\n\nfunc Max(a, b int) int {\n\tif a > b {\n\t\treturn a\n\t}\n\treturn b\n}\n";
const CSnippet = "#include <stdio.h>\nint main(void) {\n  printf(\"hi\\n\");\n  return 0;\n}\n";
const Minified = "var a=1;" + "x=x+1;".repeat(90);

test("tiering: permissive+clean -> Filtered, non-permissive -> Rejected, general web -> Raw", () => {
  expect(ClassifyDocument("MIT", Clean, "local").Tier).toBe("Filtered");
  expect(ClassifyDocument("GPL-3.0", Clean, "local").Tier).toBe("Rejected");
  expect(ClassifyDocument("MIT", Minified, "local").Tier).toBe("Rejected"); // low quality
  const Web = ClassifyDocument("MIT", Clean, "web-general");
  expect(Web.Tier).toBe("Raw"); // isolated regardless of license
  expect(String(Web.RejectReason)).toContain("isolated");
});

test("embedding: identical text is more similar than unrelated text", () => {
  const A = HashingEmbedding(Clean);
  const B = HashingEmbedding(Clean + " // a comment");
  const C = HashingEmbedding("def multiply(x, y):\n    return x * y\n");
  expect(CosineSimilarity(A, A)).toBeCloseTo(1, 6);
  expect(CosineSimilarity(A, B)).toBeGreaterThan(CosineSimilarity(A, C));
});

test("ingest tiers documents into the store and export yields only Filtered", async () => {
  const Store = new InMemoryDocumentStore();
  const Inputs: SourceInput[] = [
    { Source: "s", License: "MIT", Lang: "ts", Content: Clean, Provenance: "a.ts", Origin: "local" },
    { Source: "s", License: "GPL-3.0", Lang: "c", Content: CSnippet, Provenance: "b.c", Origin: "local" },
    { Source: "s", License: "MIT", Lang: "js", Content: Minified, Provenance: "m.js", Origin: "local" },
    { Source: "web", License: "unknown", Lang: "ts", Content: "const q = fetchFromWeb();\n", Provenance: "http://x", Origin: "web-general" },
  ];
  const Stats = await IngestDocuments(Inputs, Store, "2026-07-13T00:00:00.000Z");
  expect(Stats.Ingested).toBe(4);
  expect(Stats.ByTier.Filtered).toBe(1);
  expect(Stats.ByTier.Rejected).toBe(2);
  expect(Stats.ByTier.Raw).toBe(1);

  const Text = await ExportTrainingText(Store);
  expect(Text).toContain("export function add"); // filtered content present
  expect(Text).not.toContain("fetchFromWeb"); // isolated web content excluded from training
});

test("quality report aggregates tiers, licenses, and langs", async () => {
  const Store = new InMemoryDocumentStore();
  await IngestDocuments(
    [
      { Source: "s", License: "MIT", Lang: "ts", Content: Clean, Provenance: "a.ts", Origin: "local" },
      { Source: "s", License: "Apache-2.0", Lang: "go", Content: GoSnippet, Provenance: "c.go", Origin: "local" },
      { Source: "s", License: "GPL-3.0", Lang: "c", Content: CSnippet, Provenance: "b.c", Origin: "local" },
    ],
    Store,
    "2026-07-13T00:00:00.000Z",
  );
  const Report = BuildReport(await Store.All());
  expect(Report.Total).toBe(3);
  expect(Report.ByTier.Filtered).toBe(2);
  expect(Report.ByLicense["GPL-3.0"]).toBe(1);
  expect(Report.FilteredBytes).toBeGreaterThan(0);
  expect(RenderReportText(Report)).toContain("Filtered=2");
});

test("dedup by content hash: re-ingesting identical content does not duplicate", async () => {
  const Store = new InMemoryDocumentStore();
  const One: SourceInput = { Source: "s", License: "MIT", Lang: "ts", Content: Clean, Provenance: "a.ts", Origin: "local" };
  await IngestDocuments([One, One], Store, "2026-07-13T00:00:00.000Z");
  expect(await Store.Count()).toBe(1);
});

test("ingest is resilient: one failing Upsert doesn't abort the batch", async () => {
  // A store that rejects the 2nd row (mimics Postgres refusing an unstorable value): the run must
  // continue and report the failure, not throw and lose the other rows.
  class FlakyStore extends InMemoryDocumentStore {
    Calls = 0;
    override async Upsert(Doc: DocumentRecord): Promise<void> {
      this.Calls++;
      if (this.Calls === 2) throw new Error("simulated store failure");
      await super.Upsert(Doc);
    }
  }
  const Store = new FlakyStore();
  const Inputs: SourceInput[] = [
    { Source: "s", License: "MIT", Lang: "ts", Content: Clean, Provenance: "a.ts", Origin: "local" },
    { Source: "s", License: "MIT", Lang: "go", Content: GoSnippet, Provenance: "b.go", Origin: "local" },
    { Source: "s", License: "MIT", Lang: "c", Content: CSnippet, Provenance: "c.c", Origin: "local" },
  ];
  const Stats = await IngestDocuments(Inputs, Store, "2026-07-13T00:00:00.000Z");
  expect(Stats.Ingested).toBe(2);
  expect(Stats.Failed).toBe(1);
  expect(await Store.Count()).toBe(2);
});

test("ReclassifyBySource promotes quality NOASSERTION docs to Filtered, keeps low-quality rejected", async () => {
  const Store = new InMemoryDocumentStore();
  await IngestDocuments(
    [
      { Source: "acme/lib", License: "NOASSERTION", Lang: "ts", Content: Clean, Provenance: "a.ts", Origin: "web-permissive" },
      { Source: "acme/lib", License: "NOASSERTION", Lang: "js", Content: Minified, Provenance: "m.js", Origin: "web-permissive" }, // low quality
      { Source: "other/lib", License: "NOASSERTION", Lang: "go", Content: GoSnippet, Provenance: "g.go", Origin: "web-permissive" }, // different repo — untouched
    ],
    Store,
    "2026-07-13T00:00:00.000Z",
  );
  expect((await Store.ByTier("Rejected")).length).toBe(3); // all NOASSERTION => Rejected on license

  const Res = await Store.ReclassifyBySource("acme/lib", "MIT", 0.6);
  expect(Res).toEqual({ Promoted: 1, KeptLowQuality: 1 });
  const Filtered = await Store.ByTier("Filtered");
  expect(Filtered.length).toBe(1);
  expect(Filtered[0]!.License).toBe("MIT");
  expect(Filtered[0]!.RejectReason).toBeNull();
  // the other repo is untouched (scoped by source); low-quality doc stays Rejected with a quality reason
  expect((await Store.ByTier("Rejected")).length).toBe(2);
  const StillRejected = (await Store.ByTier("Rejected")).find((D) => D.Source === "acme/lib");
  expect(String(StillRejected?.RejectReason)).toContain("low quality");
});

test("provenance-aware dedup: identical content from different origins coexist (no tier clobber)", async () => {
  const Store = new InMemoryDocumentStore();
  await IngestDocuments(
    [
      { Source: "local", License: "MIT", Lang: "ts", Content: Clean, Provenance: "a.ts", Origin: "local" },
      { Source: "web", License: "unknown", Lang: "ts", Content: Clean, Provenance: "http://x", Origin: "web-general" },
    ],
    Store,
    "2026-07-13T00:00:00.000Z",
  );
  expect(await Store.Count()).toBe(2); // the web copy does NOT overwrite the local one
  expect((await Store.ByTier("Filtered")).length).toBe(1); // local MIT stays training-eligible
  expect((await Store.ByTier("Raw")).length).toBe(1); // general web stays isolated
});

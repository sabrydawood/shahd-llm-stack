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
} from "../Brain/Foundry/FoundryBarrel.ts";
import type { SourceInput } from "../Brain/Foundry/FoundryBarrel.ts";

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

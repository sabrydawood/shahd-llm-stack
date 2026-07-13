import { test, expect } from "bun:test";
import { InMemoryDocumentStore, IngestDocuments } from "../Foundry/FoundryBarrel.ts";
import { CreateDashboardHandler } from "../Foundry/Dashboard.ts";
import type { SourceInput } from "../Foundry/FoundryBarrel.ts";

async function SeededHandler() {
  const Store = new InMemoryDocumentStore();
  const Inputs: SourceInput[] = [
    { Source: "s", License: "MIT", Lang: "ts", Content: "export function add(a, b) {\n  return a + b;\n}\n", Provenance: "a.ts", Origin: "local" },
    { Source: "s", License: "GPL-3.0", Lang: "c", Content: "int main(void) {\n  return 0;\n}\n", Provenance: "b.c", Origin: "local" },
  ];
  await IngestDocuments(Inputs, Store, "2026-07-13T00:00:00.000Z");
  return CreateDashboardHandler(Store);
}

test("dashboard serves the HTML page", async () => {
  const Handler = await SeededHandler();
  const Res = await Handler(new Request("http://x/"));
  expect(Res.headers.get("Content-Type")).toContain("text/html");
  expect(await Res.text()).toContain("Data Foundry");
});

test("dashboard /api/report returns tier counts", async () => {
  const Handler = await SeededHandler();
  const Report = (await (await Handler(new Request("http://x/api/report"))).json()) as { Total: number; ByTier: Record<string, number> };
  expect(Report.Total).toBe(2);
  expect(Report.ByTier.Filtered).toBe(1);
  expect(Report.ByTier.Rejected).toBe(1);
});

test("dashboard /api/documents filters by tier and includes reject reasons", async () => {
  const Handler = await SeededHandler();
  const Docs = (await (await Handler(new Request("http://x/api/documents?tier=Rejected"))).json()) as { tier: string; rejectReason: string; preview: string }[];
  expect(Docs.length).toBe(1);
  expect(Docs[0].tier).toBe("Rejected");
  expect(Docs[0].rejectReason).toContain("non-permissive");
});

test("dashboard 404s unknown paths", async () => {
  const Handler = await SeededHandler();
  expect((await Handler(new Request("http://x/nope"))).status).toBe(404);
});

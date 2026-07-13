import { test, expect } from "bun:test";
import { InMemoryDocumentStore, IngestDocuments, CreateDashboardHandler } from "../Foundry/FoundryBarrel.ts";
import type { SourceInput, LearnFn } from "../Foundry/FoundryBarrel.ts";

async function SeededStore() {
  const Store = new InMemoryDocumentStore();
  const Inputs: SourceInput[] = [
    { Source: "acme/good", License: "MIT", Lang: "ts", Content: "export function add(a, b) {\n  return a + b;\n}\n", Provenance: "https://github.com/acme/good/blob/main/src/Add.ts", Origin: "web-permissive" },
    { Source: "acme/good", License: "MIT", Lang: "ts", Content: "export const pi = 3.14159;\n", Provenance: "https://github.com/acme/good/blob/main/src/Pi.ts", Origin: "web-permissive" },
    { Source: "bad/repo", License: "GPL-3.0", Lang: "c", Content: "int main(void) {\n  return 0;\n}\n", Provenance: "https://github.com/bad/repo/blob/main/m.c", Origin: "web-permissive" },
  ];
  await IngestDocuments(Inputs, Store, "2026-07-13T00:00:00.000Z");
  return Store;
}

test("dashboard serves the control-panel page", async () => {
  const Handler = CreateDashboardHandler(await SeededStore());
  const Res = await Handler(new Request("http://x/"));
  expect(Res.headers.get("Content-Type")).toContain("text/html");
  expect(await Res.text()).toContain("Data Foundry");
});

test("/api/stats aggregates tiers/langs efficiently", async () => {
  const Handler = CreateDashboardHandler(await SeededStore());
  const Stats = (await (await Handler(new Request("http://x/api/stats"))).json()) as { Total: number; ByTier: Record<string, number>; ByLang: Record<string, number> };
  expect(Stats.Total).toBe(3);
  expect(Stats.ByTier.Filtered).toBe(2);
  expect(Stats.ByTier.Rejected).toBe(1);
  expect(Stats.ByLang.ts).toBe(2);
});

test("/api/repos + /api/documents power the per-repo accordion", async () => {
  const Handler = CreateDashboardHandler(await SeededStore());
  const Repos = (await (await Handler(new Request("http://x/api/repos"))).json()) as { Source: string; Files: number }[];
  expect(Repos[0].Source).toBe("acme/good"); // most files first
  expect(Repos[0].Files).toBe(2);
  const Docs = (await (await Handler(new Request("http://x/api/documents?source=acme/good"))).json()) as { path: string; tier: string }[];
  expect(Docs.length).toBe(2);
  expect(Docs[0].path).toContain("Add.ts");
});

test("POST /api/learn starts a run and SSE streams its progress", async () => {
  const MockLearn: LearnFn = async (_Settings, OnEvent) => {
    OnEvent({ kind: "repo", repo: "acme/new", level: "high", files: 7, bytes: 2000, ingested: true, reason: null });
    OnEvent({ kind: "done", ingested: 7 });
  };
  const Handler = CreateDashboardHandler(await SeededStore(), MockLearn);
  const Post = await Handler(new Request("http://x/api/learn", { method: "POST", body: JSON.stringify({ Source: "github", Query: "q" }) }));
  expect(Post.status).toBe(202);
  const Stream = await Handler(new Request("http://x/api/learn/stream"));
  const Text = await Stream.text();
  expect(Text).toContain("acme/new");
  expect(Text).toContain('"kind":"done"');
});

test("dashboard is read-only when no Learn runner is wired", async () => {
  const Handler = CreateDashboardHandler(await SeededStore());
  const Config = (await (await Handler(new Request("http://x/api/config"))).json()) as { learnEnabled: boolean };
  expect(Config.learnEnabled).toBe(false);
  const Post = await Handler(new Request("http://x/api/learn", { method: "POST", body: "{}" }));
  expect(Post.status).toBe(501);
});

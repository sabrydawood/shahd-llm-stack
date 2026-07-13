import { test, expect } from "bun:test";
import {
  InMemoryDocumentStore,
  IngestFromWeb,
  CreateGitHubProvider,
  CreateWebSearchProvider,
  HtmlToText,
} from "../Foundry/FoundryBarrel.ts";
import type { HttpJson, SearchBackend, PageFetch } from "../Foundry/FoundryBarrel.ts";

// Mock GitHub HTTP: one MIT repo, one code file (base64), so no real network is used.
const MockHttp: HttpJson = async (Url) => {
  if (Url.includes("/search/repositories")) {
    return { items: [{ full_name: "acme/util", default_branch: "main", license: { spdx_id: "MIT" } }] };
  }
  if (Url.includes("/git/trees/")) {
    return { tree: [{ path: "src/Add.ts", type: "blob" }, { path: "README.md", type: "blob" }] };
  }
  if (Url.includes("/contents/")) {
    return { encoding: "base64", content: Buffer.from("export function add(a, b) {\n  return a + b;\n}\n").toString("base64") };
  }
  return {};
};

test("GitHub provider tags files web-permissive with the repo license", async () => {
  const Provider = CreateGitHubProvider({ Http: MockHttp });
  const Docs = await Provider.Fetch("language:typescript", 5);
  expect(Docs.length).toBe(1); // only the .ts file, not README.md
  expect(Docs[0].Origin).toBe("web-permissive");
  expect(Docs[0].License).toBe("MIT");
  expect(Docs[0].Lang).toBe("typescript");
  expect(Docs[0].Content).toContain("export function add");
});

test("HtmlToText strips tags, script/style, and entities", () => {
  const Text = HtmlToText("<style>x{}</style><h1>Hi &amp; bye</h1><script>ignore()</script><p>body</p>");
  expect(Text).toBe("Hi & bye body");
});

test("web search provider tags pages web-general (isolated Raw tier)", async () => {
  const Search: SearchBackend = async () => [{ Url: "http://x/a", Title: "A" }, { Url: "http://x/b", Title: "B" }];
  const Fetch: PageFetch = async (Url) => `<html><body>content of ${Url}</body></html>`;
  const Provider = CreateWebSearchProvider({ Search, Fetch });
  const Docs = await Provider.Fetch("anything", 2);
  expect(Docs.length).toBe(2);
  expect(Docs.every((D) => D.Origin === "web-general")).toBe(true);
  expect(Docs[0].Content).toContain("content of http://x/a");
});

test("IngestFromWeb tiers GitHub-permissive to Filtered and general web to isolated Raw", async () => {
  const Store = new InMemoryDocumentStore();
  const GitHub = CreateGitHubProvider({ Http: MockHttp });
  const WebSearch = CreateWebSearchProvider({
    Search: async () => [{ Url: "http://x/page", Title: "P" }],
    Fetch: async () => "<p>some general web prose that is not code</p>",
  });
  const Stats = await IngestFromWeb([GitHub, WebSearch], ["query"], Store, "2026-07-13T00:00:00.000Z");
  expect(Stats.ByTier.Filtered).toBe(1); // the MIT .ts file
  expect(Stats.ByTier.Raw).toBe(1); // the general web page (never training-eligible)
});

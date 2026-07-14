import { test, expect } from "bun:test";
import {
  InMemoryDocumentStore,
  IngestFromWeb,
  CreateGitHubProvider,
  CreateWebSearchProvider,
  HtmlToText,
} from "../Foundry/FoundryBarrel.ts";
import type { HttpJson, SearchBackend, PageFetch } from "../Foundry/FoundryBarrel.ts";
import { IsSubstantiveCodePath, IsSubstantiveCodeContent } from "../Foundry/CodeFileFilter.ts";
import { StripLicenseHeader, SanitizeText } from "../Foundry/ContentNormalizer.ts";

// A substantive code file (passes the content gate); the tiny stub the old test used would now be
// correctly rejected as too small.
const SampleCode = `import { readFileSync } from "node:fs";

export type Config = { name: string; retries: number };

export function loadConfig(path: string): Config {
  const raw = readFileSync(path, "utf8");
  const parsed = JSON.parse(raw) as Partial<Config>;
  return {
    name: parsed.name ?? "default",
    retries: typeof parsed.retries === "number" ? parsed.retries : 3,
  };
}

export function withRetries<T>(fn: () => T, times: number): T {
  let lastError: unknown;
  for (let i = 0; i < times; i++) {
    try {
      return fn();
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError;
}
`;

// Mock GitHub HTTP: one MIT repo whose tree has a substantive src file plus junk that must be filtered.
const MockHttp: HttpJson = async (Url) => {
  if (Url.includes("/search/repositories")) {
    return { items: [{ full_name: "acme/util", default_branch: "main", license: { spdx_id: "MIT" } }] };
  }
  if (Url.includes("/git/trees/")) {
    return {
      tree: [
        { path: "benchmarks/demo.css", type: "blob" }, // junk dir + markup -> excluded
        { path: "src/types.d.ts", type: "blob" }, // declaration file -> excluded
        { path: "README.md", type: "blob" }, // not code -> excluded
        { path: "src/Config.ts", type: "blob" }, // the real source file -> kept
      ],
    };
  }
  if (Url.includes("/contents/")) {
    return { encoding: "base64", content: Buffer.from(SampleCode).toString("base64") };
  }
  return {};
};

test("GitHub provider keeps only substantive source (filters markup, declarations, junk dirs)", async () => {
  const Provider = CreateGitHubProvider({ Http: MockHttp });
  const Docs = await Provider.Fetch("language:typescript", 5);
  expect(Docs.length).toBe(1); // only src/Config.ts — the .css/.d.ts/.md are all filtered out
  expect(Docs[0].Origin).toBe("web-permissive");
  expect(Docs[0].License).toBe("MIT");
  expect(Docs[0].Lang).toBe("typescript");
  expect(Docs[0].Provenance).toContain("src/Config.ts");
  expect(Docs[0].Content).toContain("loadConfig");
});

test("code file filter rejects junk paths and thin content, keeps real source", () => {
  expect(IsSubstantiveCodePath("src/Parser.ts")).toBe(true);
  expect(IsSubstantiveCodePath("packages/core/lib/Engine.go")).toBe(true);
  expect(IsSubstantiveCodePath("benchmarks/big-table/demo.css")).toBe(false); // junk dir + markup
  expect(IsSubstantiveCodePath("src/types.d.ts")).toBe(false); // declaration
  expect(IsSubstantiveCodePath(".eslint-plugin-local/x.ts")).toBe(false); // dot-dir
  expect(IsSubstantiveCodePath("test/foo.test.ts")).toBe(false); // test dir + .test
  expect(IsSubstantiveCodePath("README.md")).toBe(false); // not code
  expect(IsSubstantiveCodeContent('declare module "*.txt" { const c: string }\n')).toBe(false); // stub
  expect(IsSubstantiveCodeContent(SampleCode)).toBe(true);
});

test("StripLicenseHeader drops the license banner but keeps code + comments", () => {
  const WithHeader = "/*-----\n * Copyright (c) Microsoft Corporation. All rights reserved.\n * Licensed under the MIT License.\n *-----*/\n\nimport { x } from \"y\";\n// a real comment\nexport const z = 1;\n";
  const Stripped = StripLicenseHeader(WithHeader);
  expect(Stripped).not.toContain("Copyright");
  expect(Stripped.startsWith("import { x }")).toBe(true);
  expect(Stripped).toContain("// a real comment"); // meaningful code comment is kept
  // a leading NON-license comment (a docstring) is preserved untouched
  const Doc = "// This module parses configuration files.\nexport function parse() {}\n";
  expect(StripLicenseHeader(Doc)).toBe(Doc);
});

test("SanitizeText drops NUL and lone surrogates (Postgres-unstorable) but keeps valid text", () => {
  expect(SanitizeText("a" + String.fromCharCode(0) + "b")).toBe("ab"); // NUL removed
  const LoneHigh = "x" + String.fromCharCode(0xd800) + "y";
  expect(SanitizeText(LoneHigh)).toBe("x" + String.fromCharCode(0xfffd) + "y"); // lone surrogate -> U+FFFD
  const Emoji = "z" + String.fromCharCode(0xd83d, 0xde00) + "w"; // valid surrogate pair (😀)
  expect(SanitizeText(Emoji)).toBe(Emoji); // real astral char preserved
  expect(SanitizeText("plain code();")).toBe("plain code();"); // ordinary text untouched
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

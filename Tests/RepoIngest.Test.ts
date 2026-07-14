import { test, expect } from "bun:test";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createTarGzip } from "nanotar";
import {
  FetchRepoFiles,
  AssessRepo,
  CreateGitHubRepoProvider,
  CreateLocalRepoProvider,
  InMemoryDocumentStore,
  IngestFromWeb,
  IsSubstantiveCodePath,
  LangForPath,
} from "../Foundry/FoundryBarrel.ts";
import type { HttpJson, FetchBytes, RepoFile } from "../Foundry/FoundryBarrel.ts";

// A substantive source file (>300 chars, passes the content gate), distinct per name.
function Code(Name: string): string {
  return `import { readFileSync, writeFileSync } from "node:fs";\n\nexport function ${Name}(path: string): string {\n  const raw = readFileSync(path, "utf8");\n  const lines = raw.split("\\n").filter((l) => l.trim().length > 0);\n  const normalized = lines.map((l) => l.trimEnd()).join("\\n");\n  return normalized;\n}\n\nexport function ${Name}ToFile(source: string, dest: string): void {\n  writeFileSync(dest, ${Name}(source));\n}\n\nexport const ${Name}Version = "1.0.0";\n`;
}

function Tar(Files: { name: string; data: string }[]): Promise<Uint8Array> {
  return createTarGzip(Files);
}

test("FetchRepoFiles extracts substantive source and skips junk/markup/declarations", async () => {
  const Gz = await Tar([
    { name: "repo-sha/src/Parser.ts", data: Code("parse") },
    { name: "repo-sha/lib/Engine.go", data: Code("engine") },
    { name: "repo-sha/benchmarks/demo.css", data: "form { margin: 0; }\n" }, // junk dir + markup
    { name: "repo-sha/src/types.d.ts", data: "export type X = number;\n" }, // declaration file
    { name: "repo-sha/README.md", data: "# hi\n" }, // not code
    { name: "repo-sha/.eslintrc.js", data: "module.exports = {};\n" }, // dotfile config
  ]);
  const Fetch: FetchBytes = async () => Gz;
  const Files = await FetchRepoFiles("http://tar", Fetch);
  expect(Files.map((F) => F.Path).sort()).toEqual(["lib/Engine.go", "src/Parser.ts"]);
});

test("AssessRepo grades a structured, quality repo 'high' and a thin one 'low'", () => {
  const Good: RepoFile[] = ["A", "B", "C", "D", "E"].map((N) => ({ Path: `src/${N}.ts`, Content: Code(N) }));
  expect(AssessRepo(Good).Level).toBe("high");
  const Thin: RepoFile[] = [{ Path: "main.py", Content: Code("m") }];
  expect(AssessRepo(Thin).Level).toBe("low");
});

test("whole-repo provider ingests every file of a qualifying repo and skips low-level ones", async () => {
  const GoodTar = await Tar(["A", "B", "C", "D", "E"].map((N) => ({ name: `good-sha/src/${N}.ts`, data: Code(N) })));
  const ThinTar = await Tar([{ name: "thin-sha/main.py", data: Code("m") }]);

  const Http: HttpJson = async () => ({
    items: [
      { full_name: "acme/good", default_branch: "main", license: { spdx_id: "MIT" } },
      { full_name: "acme/thin", default_branch: "main", license: { spdx_id: "MIT" } },
    ],
  });
  const BytesFetcher: FetchBytes = async (Url) => (Url.includes("acme/good") ? GoodTar : ThinTar);

  const Provider = CreateGitHubRepoProvider({ Http, FetchBytes: BytesFetcher, MinLevel: "medium" });
  const Docs = await Provider.Fetch("q", 5);
  expect(Docs.length).toBe(5); // all 5 files of the good repo
  expect(Docs.every((D) => D.Source === "acme/good")).toBe(true); // thin repo skipped
  expect(Docs.every((D) => D.Origin === "web-permissive" && D.License === "MIT")).toBe(true);
});

test("incremental collect: OnRepoReady stores each repo before the next is downloaded", async () => {
  const Tar1 = await Tar(["A", "B", "C"].map((N) => ({ name: `r1-sha/src/${N}.ts`, data: Code(N) })));
  const Tar2 = await Tar(["D", "E"].map((N) => ({ name: `r2-sha/src/${N}.ts`, data: Code(N) })));
  const Http: HttpJson = async () => ({
    items: [
      { full_name: "acme/r1", default_branch: "main", license: { spdx_id: "MIT" } },
      { full_name: "acme/r2", default_branch: "main", license: { spdx_id: "MIT" } },
    ],
  });
  const BytesFetcher: FetchBytes = async (Url) => (Url.includes("acme/r1") ? Tar1 : Tar2);
  const Order: string[] = [];
  const Ready: { repo: string; count: number }[] = [];
  const Provider = CreateGitHubRepoProvider({
    Http,
    FetchBytes: BytesFetcher,
    MinLevel: "medium",
    OnRepo: (Info) => Order.push("assess:" + Info.Repo),
    OnRepoReady: async (Source, Files) => {
      Order.push("store:" + Source);
      Ready.push({ repo: Source, count: Files.length });
    },
  });
  const Ret = await Provider.Fetch("q", 5);
  expect(Ret.length).toBe(0); // incremental mode: nothing collected/returned
  expect(Ready).toEqual([{ repo: "acme/r1", count: 3 }, { repo: "acme/r2", count: 2 }]);
  // each repo is STORED before the next is ASSESSED — proves store-as-you-go, not collect-all-then-store
  expect(Order).toEqual(["assess:acme/r1", "store:acme/r1", "assess:acme/r2", "store:acme/r2"]);
});

test("resilient collect: one repo failing (e.g. rate-limit) does not abort the whole run", async () => {
  // The exact failure mode that silently killed collection: a tarball fetch throwing mid-loop. The
  // run must log+skip the bad repo and still store the good ones — not abandon everything.
  const GoodTar = await Tar(["A", "B", "C"].map((N) => ({ name: `ok-sha/src/${N}.ts`, data: Code(N) })));
  const Http: HttpJson = async () => ({
    items: [
      { full_name: "acme/boom", default_branch: "main", license: { spdx_id: "MIT" } },
      { full_name: "acme/ok", default_branch: "main", license: { spdx_id: "MIT" } },
    ],
  });
  const BytesFetcher: FetchBytes = async (Url) => {
    if (Url.includes("acme/boom")) throw new Error("GitHub API 403"); // secondary rate limit
    return GoodTar;
  };
  const Reasons: (string | null)[] = [];
  const Logs: string[] = [];
  const Provider = CreateGitHubRepoProvider({
    Http,
    FetchBytes: BytesFetcher,
    MinLevel: "medium",
    OnRepo: (Info) => Reasons.push(Info.Reason ?? null),
    Log: (M) => Logs.push(M),
  });
  const Docs = await Provider.Fetch("q", 5);
  expect(Docs.length).toBe(3); // the good repo's files survived the bad repo's failure
  expect(Docs.every((D) => D.Source === "acme/ok")).toBe(true);
  expect(Reasons.some((R) => R !== null && R.includes("error: GitHub API 403"))).toBe(true); // failure reported, not swallowed
  expect(Logs.some((L) => L.includes("ERROR acme/boom"))).toBe(true); // and logged to the console trail
  expect(Logs.some((L) => L.includes("errored=1") && L.includes("stored=1"))).toBe(true); // summary is auditable
});

test("NOASSERTION license is resolved from the real LICENSE: permissive -> detected SPDX, else unchanged", async () => {
  const Tar1 = await Tar(["A", "B", "C"].map((N) => ({ name: `mit-sha/src/${N}.ts`, data: Code(N) })));
  const Tar2 = await Tar(["D", "E", "F"].map((N) => ({ name: `gpl-sha/src/${N}.ts`, data: Code(N) })));
  const Http: HttpJson = async () => ({
    items: [
      { full_name: "acme/mitrepo", default_branch: "main", license: { spdx_id: "NOASSERTION" } },
      { full_name: "acme/gplrepo", default_branch: "main", license: { spdx_id: "NOASSERTION" } },
    ],
  });
  const BytesFetcher: FetchBytes = async (Url) => (Url.includes("acme/mitrepo") ? Tar1 : Tar2);
  const MitText = 'MIT License\n\nCopyright (c) 2025 X\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.\n';
  const Provider = CreateGitHubRepoProvider({
    Http,
    FetchBytes: BytesFetcher,
    MinLevel: "medium",
    FetchLicense: async (Name) => (Name.includes("mitrepo") ? { Spdx: "NOASSERTION", Text: MitText } : { Spdx: "NOASSERTION", Text: "GNU GENERAL PUBLIC LICENSE Version 3" }),
  });
  const Docs = await Provider.Fetch("q", 5);
  const Mit = Docs.filter((D) => D.Source === "acme/mitrepo");
  const Gpl = Docs.filter((D) => D.Source === "acme/gplrepo");
  expect(Mit.length).toBe(3);
  expect(Mit.every((D) => D.License === "MIT")).toBe(true); // verified permissive -> promoted to real SPDX
  expect(Gpl.every((D) => D.License === "NOASSERTION")).toBe(true); // copyleft stays unresolved -> Rejected on license
});

test("expanded languages: css/html/vue/sql are recognized as code (M8)", () => {
  expect(IsSubstantiveCodePath("src/styles/main.css")).toBe(true);
  expect(IsSubstantiveCodePath("src/Page.vue")).toBe(true);
  expect(IsSubstantiveCodePath("db/schema.sql")).toBe(true);
  expect(IsSubstantiveCodePath("web/index.html")).toBe(true);
  expect(LangForPath("x.css")).toBe("css");
  expect(LangForPath("x.vue")).toBe("vue");
});

test("local repo provider learns from OUR own code as 'owned' (trained on despite license)", async () => {
  const Root = mkdtempSync(join(tmpdir(), "shahd-repo-"));
  try {
    mkdirSync(join(Root, "src"), { recursive: true });
    for (const N of ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]) writeFileSync(join(Root, "src", `${N}.ts`), Code(N));
    writeFileSync(join(Root, "src", "styles.css"), "body { margin: 0; padding: 0; color: #333; background: #fff; }\n".repeat(12));
    const Store = new InMemoryDocumentStore();
    const Provider = CreateLocalRepoProvider({ Roots: [Root], License: "proprietary", MinLevel: "medium" });
    const Stats = await IngestFromWeb([Provider], [""], Store, "2026-07-13T00:00:00.000Z");
    expect(Stats.ByTier.Filtered).toBeGreaterThanOrEqual(5); // owned code is training-eligible despite "proprietary"
    const Docs = await Store.All();
    expect(Docs.every((D) => D.Origin === "owned")).toBe(true);
    expect(Docs.some((D) => D.Lang === "css")).toBe(true); // css ingested (expanded languages)
  } finally {
    rmSync(Root, { recursive: true, force: true });
  }
});

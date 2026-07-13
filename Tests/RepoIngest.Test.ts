import { test, expect } from "bun:test";
import { createTarGzip } from "nanotar";
import { FetchRepoFiles, AssessRepo, CreateGitHubRepoProvider } from "../Foundry/FoundryBarrel.ts";
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

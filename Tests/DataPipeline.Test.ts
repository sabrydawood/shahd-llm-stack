import { test, expect } from "bun:test";
import { LicenseManifest } from "../Brain/Data/LicenseManifest.ts";
import { NearDuplicateGroups, DedupedIndices } from "../Brain/Data/NearDedup.ts";
import { ScoreCodeQuality } from "../Brain/Data/QualityFilter.ts";
import { ToFim, FromFim, FimTokens } from "../Brain/Data/FimReformat.ts";
import { Decontaminate } from "../Brain/Data/Decontamination.ts";

test("license manifest filters to permissive and summarizes", () => {
  const Manifest = new LicenseManifest();
  Manifest.Add({ Source: "repoA", License: "MIT", Path: "a.ts", Bytes: 100, IngestedAt: "2026-07-13" });
  Manifest.Add({ Source: "repoB", License: "GPL-3.0", Path: "b.ts", Bytes: 50, IngestedAt: "2026-07-13" });
  expect(Manifest.PermissiveOnly().length).toBe(1);
  expect(Manifest.Summary().TotalBytes).toBe(150);
  expect(LicenseManifest.FromJson(Manifest.ToJson()).Entries.length).toBe(2);
});

test("near-dedup groups near-identical documents and keeps distinct ones", () => {
  const A = "function add(a, b) { return a + b; } // small helper used across the codebase";
  const B = "function add(a, b) { return a + b; } // small helper used across the whole codebase";
  const C = "class Server { constructor() { this.port = 8080; } listen() { startAll(); } }";
  const Groups = NearDuplicateGroups([A, B, C], 0.6);
  const GroupOf = (Index: number): number => Groups.findIndex((G) => G.includes(Index));
  expect(GroupOf(0)).toBe(GroupOf(1));
  expect(GroupOf(2)).not.toBe(GroupOf(0));
  expect(DedupedIndices([A, B, C], 0.6).length).toBe(2);
});

test("quality filter keeps clean code and drops minified/binary junk", () => {
  const Clean = "function add(a, b) {\n  return a + b;\n}\n\nconst x = add(1, 2);\n";
  expect(ScoreCodeQuality(Clean).Passed).toBe(true);
  expect(ScoreCodeQuality("a".repeat(2000)).Passed).toBe(false);
  let Binary = "";
  for (let I = 0; I < 200; I++) Binary += String.fromCharCode(I);
  expect(ScoreCodeQuality(Binary).Passed).toBe(false);
});

test("FIM PSM round-trips to the original", () => {
  const Doc = "function f(x) {\n  return x * 2;\n}\n";
  const Fim = ToFim(Doc, 10, 25, "Psm");
  expect(Fim).toContain(FimTokens.Prefix);
  expect(Fim).toContain(FimTokens.Middle);
  expect(FromFim(Fim)).toBe(Doc);
});

test("decontamination removes train docs overlapping the eval set", () => {
  const EvalSet = ["the quick brown fox jumps over the lazy dog again and once more today please"];
  const Train = [
    "the quick brown fox jumps over the lazy dog again and once more today please",
    "unrelated snippet about servers databases networking stacks here now",
  ];
  const Result = Decontaminate(Train, EvalSet, 13);
  expect(Result.Removed).toContain(0);
  expect(Result.Kept).toContain(1);
});

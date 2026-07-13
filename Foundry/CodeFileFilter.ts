// Selecting GOOD code files from a repository (M6 quality fix). Naively taking the first files in
// git-tree order pulls in dotfiles, config, benchmarks, generated stubs, and markup — exactly the
// low-value data that dilutes a CODE model. These pure predicates keep only substantive source:
//   - a code-only extension allowlist (markup/style excluded — Shahd is code-specialized),
//   - path exclusions for non-source dirs (tests, examples, dist, vendor, docs, dotfiles…),
//   - file exclusions for declarations/minified/generated/config/lockfiles,
//   - a content gate rejecting tiny stubs, license-header-only files, embedded blobs, and symbol soup.
// A rank puts real source dirs (src/lib/…) first.

const CodeExtension = /\.(ts|tsx|js|jsx|mjs|cjs|go|py|rs|java|kt|swift|scala|c|cc|cpp|cxx|h|hpp|rb|php|cs|dart|lua|ex|exs)$/i;

const LangByExtension: Record<string, string> = {
  ts: "typescript", tsx: "typescript", js: "javascript", jsx: "javascript", mjs: "javascript", cjs: "javascript",
  go: "go", py: "python", rs: "rust", java: "java", kt: "kotlin", swift: "swift", scala: "scala",
  c: "c", cc: "cpp", cpp: "cpp", cxx: "cpp", h: "c", hpp: "cpp", rb: "ruby", php: "php", cs: "csharp",
  dart: "dart", lua: "lua", ex: "elixir", exs: "elixir",
};

// A path segment that means "not substantive product source".
const ExcludeSegment = new Set([
  "node_modules", "dist", "build", "out", "bin", "obj", "target", "vendor", "third_party", "deps",
  "coverage", "__tests__", "__mocks__", "__snapshots__", "__pycache__", "test", "tests", "spec", "specs",
  "e2e", "example", "examples", "sample", "samples", "demo", "demos", "benchmark", "benchmarks", "bench",
  "fixture", "fixtures", "mock", "mocks", "doc", "docs", "website", "site", "generated", "gen", "proto",
  "migration", "migrations", "locale", "locales", "i18n", "asset", "assets", "public", "static",
]);

const ExcludeFile = /(\.min\.[a-z0-9]+$|\.d\.ts$|\.test\.[a-z0-9]+$|\.spec\.[a-z0-9]+$|[.-]lock\.[a-z]+$|\.lock$|\.generated\.[a-z0-9]+$|\.gen\.[a-z0-9]+$|\.pb\.[a-z0-9]+$|\.snap$|\.map$)$/i;
const ConfigFile = /(^|\/)([^/]*\.config\.[a-z0-9]+|\.[a-z].*rc(\.[a-z]+)?|babel|webpack|rollup|vite\.config|jest\.config|tsconfig[^/]*\.json|setup\.py|conftest\.py)$/i;

export function LangForPath(Path: string): string {
  const Ext = (Path.split(".").pop() ?? "").toLowerCase();
  return LangByExtension[Ext] ?? "unknown";
}

/** Is this repo path a substantive source-code file worth training on? */
export function IsSubstantiveCodePath(Path: string): boolean {
  if (!CodeExtension.test(Path)) return false;
  if (ExcludeFile.test(Path)) return false;
  if (ConfigFile.test(Path)) return false;
  for (const Segment of Path.split("/")) {
    if (Segment.startsWith(".")) return false; // dotfiles / dot-dirs
    if (ExcludeSegment.has(Segment.toLowerCase())) return false;
  }
  return true;
}

/** Lower = preferred: real source directories first. */
export function RankCodePath(Path: string): number {
  return /(^|\/)(src|lib|source|app|packages|internal|pkg|core)\//i.test(Path) ? 0 : 1;
}

/** Does the fetched content look like substantive, human-written code (not a stub/blob/comment header)? */
export function IsSubstantiveCodeContent(Text: string): boolean {
  const Trimmed = Text.trim();
  if (Trimmed.length < 300) return false; // stub / near-empty
  if (Trimmed.length > 100_000) return false; // likely generated/huge
  if (/data:(font|image|application)|;base64,/i.test(Text)) return false; // embedded blobs
  // Strip block + line comments; require enough real code left (not just a license header).
  const NoComments = Text
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/^[ \t]*\/\/.*$/gm, " ")
    .replace(/^[ \t]*#.*$/gm, " ")
    .trim();
  if (NoComments.length < 200) return false;
  let Alpha = 0;
  for (let I = 0; I < Text.length; I++) {
    const Code = Text.charCodeAt(I);
    if ((Code >= 65 && Code <= 90) || (Code >= 97 && Code <= 122)) Alpha++;
  }
  return Alpha / Text.length >= 0.25;
}

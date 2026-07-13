// Code-aware pretokenization: split text into chunks BEFORE byte-level BPE so merges never cross
// a chunk boundary. Chunks: optional-leading-space + word / number / punctuation run, or a
// whitespace run kept whole (so indentation stays together — important for code, per
// CAPABILITIES.md). This is deliberately simple for Phase 1; deeper AST/structure-aware
// segmentation is a Phase-3 lever.

export function CodePretokenize(Text: string): string[] {
  const Pattern = / ?[A-Za-z_]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+/g;
  const Chunks: string[] = [];
  let Match: RegExpExecArray | null;
  while ((Match = Pattern.exec(Text)) !== null) Chunks.push(Match[0]);
  return Chunks;
}

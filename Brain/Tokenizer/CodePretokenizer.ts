// Code-aware pretokenization: split text into chunks BEFORE byte-level BPE so merges never cross
// a chunk boundary. Chunks: optional-leading-space + word / number / punctuation run, or a
// whitespace run kept whole (so indentation stays together — important for code, per
// CAPABILITIES.md). This is deliberately simple for Phase 1; deeper AST/structure-aware
// segmentation is a Phase-3 lever.

import { SafeBoundary } from "./Internal/SurrogateBoundary.ts";

// Cap on a single pretoken chunk: an unbounded run (thousands of blank lines / trailing spaces / a
// giant numeric literal) would make BPE's per-chunk merge loop O(n^2) — a CPU-DoS on ordinary-looking
// input. Splitting oversized chunks keeps encode cost linear; real code chunks are far below the cap,
// so normal tokenization is unchanged.
const MaxChunkChars = 256;

export function CodePretokenize(Text: string): string[] {
  const Pattern = / ?[A-Za-z_]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+/g;
  const Chunks: string[] = [];
  let Match: RegExpExecArray | null;
  while ((Match = Pattern.exec(Text)) !== null) {
    const Chunk = Match[0];
    if (Chunk.length <= MaxChunkChars) {
      Chunks.push(Chunk);
    } else {
      // Snap each split point off a surrogate-pair boundary — an oversized chunk (e.g. a long emoji
      // run) must never be sliced between a high and low surrogate.
      let Pos = 0;
      while (Pos < Chunk.length) {
        const End = SafeBoundary(Chunk, Math.min(Pos + MaxChunkChars, Chunk.length));
        Chunks.push(Chunk.slice(Pos, End));
        Pos = End;
      }
    }
  }
  return Chunks;
}

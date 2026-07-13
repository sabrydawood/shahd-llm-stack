// Offline BPE merge training: learn a merge table from a corpus. Works over UNIQUE pretoken
// pieces weighted by frequency (much less work than rescanning the whole corpus). For very
// large corpora this could be made incremental (heap + linked list) — noted as a future
// optimization; this straightforward version is fine for Phase-1 offline runs.

import { CodePretokenize } from "./CodePretokenizer.ts";
import type { BpeModel } from "./BytePairEncoder.ts";

const Utf8Encoder = new TextEncoder();

type Piece = { Ids: number[]; Count: number };

export function TrainBpe(
  Corpus: string,
  NumMerges: number,
  Pretokenize: (Text: string) => string[] = CodePretokenize,
): BpeModel {
  const Frequency = new Map<string, number>();
  for (const Chunk of Pretokenize(Corpus)) {
    Frequency.set(Chunk, (Frequency.get(Chunk) ?? 0) + 1);
  }
  const Pieces: Piece[] = [];
  for (const [Chunk, Count] of Frequency) {
    Pieces.push({ Ids: Array.from(Utf8Encoder.encode(Chunk)), Count });
  }

  const Merges: [number, number][] = [];
  for (let M = 0; M < NumMerges; M++) {
    // Weighted adjacent-pair counts across all unique pieces.
    const PairCount = new Map<string, number>();
    for (const Piece of Pieces) {
      for (let I = 0; I < Piece.Ids.length - 1; I++) {
        const Key = `${Piece.Ids[I]},${Piece.Ids[I + 1]}`;
        PairCount.set(Key, (PairCount.get(Key) ?? 0) + Piece.Count);
      }
    }
    if (PairCount.size === 0) break;

    let BestKey = "";
    let BestCount = -1;
    for (const [Key, Count] of PairCount) {
      if (Count > BestCount) {
        BestCount = Count;
        BestKey = Key;
      }
    }

    const Parts = BestKey.split(",");
    const A = Number(Parts[0]);
    const B = Number(Parts[1]);
    const NewId = 256 + Merges.length;
    Merges.push([A, B]);

    // Replace every occurrence of (A,B) in all pieces.
    for (const Piece of Pieces) {
      const Next: number[] = [];
      let I = 0;
      while (I < Piece.Ids.length) {
        if (I < Piece.Ids.length - 1 && Piece.Ids[I] === A && Piece.Ids[I + 1] === B) {
          Next.push(NewId);
          I += 2;
        } else {
          Next.push(Piece.Ids[I]);
          I += 1;
        }
      }
      Piece.Ids = Next;
    }
  }

  return { Merges };
}

// Apply-time byte-level BPE. Operates on raw UTF-8 BYTES (not JS UTF-16 chars), so it has the
// no-OOV guarantee on any input — unicode identifiers, emoji in strings, mixed encodings all
// round-trip (REVIEW.md flagged the char-level approach as the wrong starting point). Base vocab
// is the 256 byte values; merges add ids 256+. Merge *training* is offline (BpeMergeTrainer).

import type { Tokenizer } from "./TokenizerTypes.ts";
import { CodePretokenize } from "./CodePretokenizer.ts";

const Utf8Encoder = new TextEncoder();
const Utf8Decoder = new TextDecoder();

export type BpeModel = { Merges: [number, number][] };

export class BytePairEncoder implements Tokenizer {
  VocabSize: number;
  private RankOf: Map<string, number>; // "a,b" -> merge rank
  private IdToBytes: number[][]; // id -> the byte sequence it expands to
  private Pretokenize: (Text: string) => string[];

  constructor(Model: BpeModel, Pretokenize: (Text: string) => string[] = CodePretokenize) {
    this.Pretokenize = Pretokenize;
    this.RankOf = new Map();
    this.IdToBytes = [];
    for (let ByteVal = 0; ByteVal < 256; ByteVal++) this.IdToBytes.push([ByteVal]);
    for (let Rank = 0; Rank < Model.Merges.length; Rank++) {
      const Pair = Model.Merges[Rank];
      this.RankOf.set(`${Pair[0]},${Pair[1]}`, Rank);
      this.IdToBytes.push([...this.IdToBytes[Pair[0]], ...this.IdToBytes[Pair[1]]]);
    }
    this.VocabSize = 256 + Model.Merges.length;
  }

  private EncodeChunk(Bytes: number[]): number[] {
    const Ids = Bytes.slice();
    for (;;) {
      let BestRank = Infinity;
      let BestPos = -1;
      for (let I = 0; I < Ids.length - 1; I++) {
        const Rank = this.RankOf.get(`${Ids[I]},${Ids[I + 1]}`);
        if (Rank !== undefined && Rank < BestRank) {
          BestRank = Rank;
          BestPos = I;
        }
      }
      if (BestPos === -1) break;
      Ids.splice(BestPos, 2, 256 + BestRank);
    }
    return Ids;
  }

  Encode(Text: string): number[] {
    const Out: number[] = [];
    for (const Chunk of this.Pretokenize(Text)) {
      for (const Id of this.EncodeChunk(Array.from(Utf8Encoder.encode(Chunk)))) Out.push(Id);
    }
    return Out;
  }

  Decode(Ids: number[]): string {
    const Bytes: number[] = [];
    for (const Id of Ids) {
      const Seq = this.IdToBytes[Id];
      if (Seq === undefined) throw new Error(`BytePairEncoder: id ${Id} out of range`);
      for (const B of Seq) Bytes.push(B);
    }
    return Utf8Decoder.decode(new Uint8Array(Bytes));
  }
}

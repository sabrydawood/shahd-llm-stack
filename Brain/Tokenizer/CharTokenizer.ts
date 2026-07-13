// Character-level tokenizer (Phase-0/tiny mode). Vocab is the sorted set of distinct code
// points in a corpus. Unlike byte-level BPE it has no OOV guarantee, so Encode throws on an
// unseen character (explicit, not a silent bad index).

import type { Tokenizer } from "./TokenizerTypes.ts";

export class CharTokenizer implements Tokenizer {
  VocabSize: number;
  private Stoi: Map<string, number>;
  private Itos: string[];

  constructor(Chars: string[]) {
    this.Itos = Chars;
    this.VocabSize = Chars.length;
    this.Stoi = new Map();
    for (let I = 0; I < Chars.length; I++) this.Stoi.set(Chars[I], I);
  }

  static FromCorpus(Corpus: string): CharTokenizer {
    const Distinct = Array.from(new Set(Array.from(Corpus))).sort();
    return new CharTokenizer(Distinct);
  }

  Encode(Text: string): number[] {
    const Ids: number[] = [];
    for (const Ch of Array.from(Text)) {
      const Id = this.Stoi.get(Ch);
      if (Id === undefined) {
        throw new Error(`CharTokenizer: unseen character ${JSON.stringify(Ch)}`);
      }
      Ids.push(Id);
    }
    return Ids;
  }

  Decode(Ids: number[]): string {
    let Out = "";
    for (const Id of Ids) {
      const Ch = this.Itos[Id];
      if (Ch === undefined) throw new Error(`CharTokenizer: id ${Id} out of range`);
      Out += Ch;
    }
    return Out;
  }
}

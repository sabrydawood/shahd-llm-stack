// Character-level tokenizer (Phase-0/tiny mode). Vocab is the sorted set of distinct code
// points in a corpus. Unlike byte-level BPE it has no OOV guarantee, so by default Encode throws
// on an unseen character (explicit, not a silent bad index) — the right behaviour for TRAINING.
// In Lenient mode (serving), an unseen character is mapped to a fallback id instead of throwing,
// so an experimental model never crashes the chat on a character it happens not to have in vocab.

import type { Tokenizer } from "./TokenizerTypes.ts";

export type CharTokenizerOptions = { Lenient?: boolean };

export class CharTokenizer implements Tokenizer {
  VocabSize: number;
  private Stoi: Map<string, number>;
  private Itos: string[];
  private Lenient: boolean;
  private FallbackId: number; // used only in Lenient mode: prefer space/newline, else id 0

  constructor(Chars: string[], Options: CharTokenizerOptions = {}) {
    this.Itos = Chars;
    this.VocabSize = Chars.length;
    this.Stoi = new Map();
    for (let I = 0; I < Chars.length; I++) this.Stoi.set(Chars[I], I);
    this.Lenient = Options.Lenient ?? false;
    this.FallbackId = this.Stoi.get(" ") ?? this.Stoi.get("\n") ?? 0;
  }

  static FromCorpus(Corpus: string): CharTokenizer {
    const Distinct = Array.from(new Set(Array.from(Corpus))).sort();
    return new CharTokenizer(Distinct);
  }

  /** The vocab char list (for persisting in a checkpoint so it can be rebuilt without the corpus). */
  GetVocabChars(): string[] {
    return this.Itos.slice();
  }

  Encode(Text: string): number[] {
    const Ids: number[] = [];
    for (const Ch of Array.from(Text)) {
      const Id = this.Stoi.get(Ch);
      if (Id === undefined) {
        if (this.Lenient) {
          Ids.push(this.FallbackId); // serving: substitute rather than crash on an unseen char
          continue;
        }
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

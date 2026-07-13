// The tokenizer contract (single-source, rule #4). The model layer only depends on this
// interface, so Char and byte-level BPE tokenizers are interchangeable.

export interface Tokenizer {
  VocabSize: number;
  Encode(Text: string): number[];
  Decode(Ids: number[]): string;
}

import { test, expect } from "bun:test";
import { TrainBpe } from "../Brain/Tokenizer/BpeMergeTrainer.ts";
import { BytePairEncoder } from "../Brain/Tokenizer/BytePairEncoder.ts";

test("byte-level BPE round-trips ASCII and non-ASCII (no OOV)", () => {
  const Corpus =
    "function add(a, b) { return a + b; }\n".repeat(20) + "const cafe = 'a b c';\n".repeat(5);
  const Model = TrainBpe(Corpus, 100);
  const Encoder = new BytePairEncoder(Model);

  // Includes text never seen in training (unicode identifier, emoji) — byte-level must still round-trip.
  for (const Text of ["function add", "return a + b", "café", "日本語 🎉", "xyz_123"]) {
    expect(Encoder.Decode(Encoder.Encode(Text))).toBe(Text);
  }
  expect(Encoder.VocabSize).toBe(256 + Model.Merges.length);
});

test("BPE compresses frequently-seen text below its raw byte length", () => {
  const Corpus = "abcabcabc ".repeat(80);
  const Model = TrainBpe(Corpus, 50);
  const Encoder = new BytePairEncoder(Model);
  const Ids = Encoder.Encode("abcabc");
  expect(Ids.length).toBeLessThan(6); // "abcabc" is 6 raw bytes; merges shorten it
  expect(Encoder.Decode(Ids)).toBe("abcabc");
});

test("empty merge table degenerates to raw bytes", () => {
  const Encoder = new BytePairEncoder({ Merges: [] });
  expect(Encoder.VocabSize).toBe(256);
  const Ids = Encoder.Encode("Hi!");
  expect(Ids).toEqual([72, 105, 33]); // ASCII byte values
  expect(Encoder.Decode(Ids)).toBe("Hi!");
});

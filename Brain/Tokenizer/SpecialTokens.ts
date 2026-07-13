// Reserved special tokens. Byte-level BPE reserves ids 0..255 for raw bytes and merges above
// that; special tokens (if enabled) are appended above the merge range. The chat/SFT tokens
// (system/user/assistant) are added in Phase 4 — this is their single home.

export const SpecialTokens = {
  Bos: "<|bos|>",
  Eos: "<|eos|>",
  Pad: "<|pad|>",
  Unk: "<|unk|>",
} as const;

export type SpecialTokenName = keyof typeof SpecialTokens;

export const SpecialTokenValues: readonly string[] = Object.values(SpecialTokens);

// System utility tools: clock, hashing, deterministic ids/randomness. Time and randomness come
// from injected providers (Context.Clock / Context.Rng) so runs are reproducible; both fall back
// to real sources when no provider is wired.

import { createHash } from "node:crypto";
import type { Tool, ToolContext } from "./ToolTypes.ts";
import { Err, RequireString, OptionalString, OptionalNumber } from "./ToolArgs.ts";

function NextU32(Context: ToolContext | undefined): number {
  if (Context?.Rng !== undefined) return Context.Rng.NextU32() >>> 0;
  return Math.floor(Math.random() * 0x100000000) >>> 0;
}

// Current time. Args: none. Uses Context.Clock when injected.
export const CurrentTimeTool: Tool = {
  Name: "current_time",
  Description: "Return the current time (epoch ms + ISO string).",
  Args: "{}",
  Execute: (_Arguments, Context) => {
    const EpochMs = Context?.Clock !== undefined ? Context.Clock() : Date.now();
    return { epochMs: EpochMs, iso: new Date(EpochMs).toISOString() };
  },
};

// Hash text. Args: { text: string, algo?: 'sha256'|'sha1'|'md5' }.
export const HashTool: Tool = {
  Name: "hash",
  Description: "Hash text with a named digest.",
  Args: "{ text: string, algo?: 'sha256'|'sha1'|'md5' }",
  Execute: (Arguments) => {
    const Algo = OptionalString(Arguments, "algo", "sha256");
    if (!["sha256", "sha1", "md5"].includes(Algo)) return Err(`unsupported algo: ${Algo}`);
    return { hash: createHash(Algo).update(RequireString(Arguments, "text")).digest("hex"), algo: Algo };
  },
};

// A deterministic RFC-4122-shaped v4 id from the injected RNG. Args: none.
export const UuidTool: Tool = {
  Name: "uuid",
  Description: "Generate a v4-shaped unique id.",
  Args: "{}",
  Execute: (_Arguments, Context) => {
    const Bytes = new Uint8Array(16);
    for (let I = 0; I < 16; I += 4) {
      const Word = NextU32(Context);
      Bytes[I] = (Word >>> 24) & 0xff;
      Bytes[I + 1] = (Word >>> 16) & 0xff;
      Bytes[I + 2] = (Word >>> 8) & 0xff;
      Bytes[I + 3] = Word & 0xff;
    }
    Bytes[6] = (Bytes[6] & 0x0f) | 0x40; // version 4
    Bytes[8] = (Bytes[8] & 0x3f) | 0x80; // variant
    const Hex = [...Bytes].map((B) => B.toString(16).padStart(2, "0")).join("");
    return { uuid: `${Hex.slice(0, 8)}-${Hex.slice(8, 12)}-${Hex.slice(12, 16)}-${Hex.slice(16, 20)}-${Hex.slice(20)}` };
  },
};

// Random integer in [min, max]. Args: { min?: number, max?: number }. Uses Context.Rng.
export const RandomIntTool: Tool = {
  Name: "random_int",
  Description: "Uniform random integer in [min, max].",
  Args: "{ min?: number, max?: number }",
  Execute: (Arguments, Context) => {
    const Min = Math.trunc(OptionalNumber(Arguments, "min", 0));
    const Max = Math.trunc(OptionalNumber(Arguments, "max", 100));
    if (Max < Min) return Err("max must be >= min");
    const Span = Max - Min + 1;
    // Rejection sampling for uniformity: NextU32() % Span is biased toward the low end unless 2^32 is
    // an exact multiple of Span. Redraw any value >= the largest multiple of Span that still fits in
    // 2^32, so every outcome in [Min, Max] is equally likely. When Span itself exceeds the 32-bit draw
    // range, a single draw can never uniformly cover it anyway, so fall back to the plain modulo.
    const TwoPow32 = 0x100000000;
    if (Span > TwoPow32) return { result: Min + (NextU32(Context) % Span) };
    const Limit = Math.floor(TwoPow32 / Span) * Span;
    let Draw = NextU32(Context);
    while (Draw >= Limit) Draw = NextU32(Context);
    return { result: Min + (Draw % Span) };
  },
};

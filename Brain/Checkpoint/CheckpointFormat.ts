// The versioned, self-describing checkpoint format (single-source contract, rule #4). A
// checkpoint carries the FULL resolved config + its hash + weights + optimizer state (m/v/step)
// + RNG stream states + tokenizer state, so a run resumes bit-identically and can never silently
// load into a mismatched architecture. Float64 buffers are base64-encoded (compact + exact).

import type { ShahdConfig } from "../Config/ConfigTypes.ts";

export const CheckpointFormatVersion = 1;

export type TensorState = { Rows: number; Cols: number; Data: string }; // Data = base64 of f64
export type OptimizerStateDump = { M: string[]; V: string[]; StepCount: number };
export type RngStateDump = { Init: number; Data: number; Dropout: number; Sampling: number };

export type Checkpoint = {
  FormatVersion: number;
  ConfigHash: string;
  Config: ShahdConfig;
  Params: TensorState[];
  Optimizer: OptimizerStateDump;
  Rng: RngStateDump;
  TokenizerState: unknown | null; // populated once a tokenizer with persistent state (BPE) is used
  Meta: Record<string, unknown>;
};

/** Base64-encode a Float64Array's raw bytes. */
export function EncodeFloat64(Arr: Float64Array): string {
  return Buffer.from(Arr.buffer, Arr.byteOffset, Arr.byteLength).toString("base64");
}

/** Decode base64 back into a fresh (aligned) Float64Array. */
export function DecodeFloat64(B64: string): Float64Array {
  const Bytes = Buffer.from(B64, "base64");
  const Out = new Float64Array(Bytes.byteLength / 8);
  new Uint8Array(Out.buffer).set(Bytes);
  return Out;
}

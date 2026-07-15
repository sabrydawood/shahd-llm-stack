// The versioned, self-describing checkpoint format (single-source contract, rule #4). A
// checkpoint carries the FULL resolved config + its hash + weights + optimizer state (m/v/step)
// + RNG stream states + tokenizer state, so a run resumes bit-identically and can never silently
// load into a mismatched architecture. Float64 buffers are base64-encoded (compact + exact).

import { createHash } from "node:crypto";
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
  Checksum?: string; // sha256 over the numeric payload; optional so pre-checksum checkpoints still load
};

/** Deterministic sha256 over the numeric payload (weights + optimizer moments + step + per-tensor
 *  shape). Catches silent corruption/truncation of a checkpoint's tensors that the shape-metadata
 *  checks alone can't — a same-length bit-flip loads as valid-but-wrong weights without this. */
export function ChecksumPayload(Params: TensorState[], Optimizer: OptimizerStateDump): string {
  const Hash = createHash("sha256");
  for (const P of Params) {
    Hash.update(`${P.Rows}x${P.Cols}:`);
    Hash.update(P.Data);
  }
  for (const M of Optimizer.M) Hash.update(M);
  for (const V of Optimizer.V) Hash.update(V);
  Hash.update(String(Optimizer.StepCount));
  return Hash.digest("hex");
}

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

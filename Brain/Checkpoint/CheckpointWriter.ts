// Serialize a training state to a self-describing checkpoint file (weights + optimizer m/v/step
// + RNG stream states + full config + hash + tokenizer state + meta).

import { writeFileSync, mkdirSync, existsSync, renameSync, copyFileSync } from "node:fs";
import { dirname } from "node:path";
import type { Shahd } from "../Nn/Shahd.ts";
import type { Optimizer } from "../Optim/OptimBarrel.ts";
import type { RngStreams } from "../Random/SeededRng.ts";
import type { ResolvedConfig, ShahdConfig } from "../Config/ConfigTypes.ts";
import { CheckpointFormatVersion, EncodeFloat64, ChecksumPayload } from "./CheckpointFormat.ts";
import type { Checkpoint, TensorState } from "./CheckpointFormat.ts";

function ExtractShahdConfig(Config: ResolvedConfig): ShahdConfig {
  return {
    Model: Config.Model,
    Optimizer: Config.Optimizer,
    Schedule: Config.Schedule,
    Training: Config.Training,
    Tokenizer: Config.Tokenizer,
    Safety: Config.Safety,
    Limits: Config.Limits,
    Tools: Config.Tools,
    Compute: Config.Compute,
    Data: Config.Data,
  };
}

/** Build the serializable checkpoint object (no I/O) — storage (file or Postgres) is a caller concern. */
export function BuildCheckpoint(
  Model: Shahd,
  Optimizer: Optimizer,
  Rng: RngStreams,
  Meta: Record<string, unknown> = {},
  TokenizerState: unknown | null = null,
): Checkpoint {
  const Params: TensorState[] = Model.Parameters().map((P) => ({
    Rows: P.Rows,
    Cols: P.Cols,
    Data: EncodeFloat64(P.Data),
  }));

  const OptimizerDump = {
    M: Optimizer.M.map(EncodeFloat64),
    V: Optimizer.V.map(EncodeFloat64),
    StepCount: Optimizer.StepCount,
  };
  return {
    FormatVersion: CheckpointFormatVersion,
    ConfigHash: Model.Config.ConfigHash,
    Config: ExtractShahdConfig(Model.Config),
    Params,
    Optimizer: OptimizerDump,
    Rng: {
      Init: Rng.InitRng.GetState(),
      Data: Rng.DataRng.GetState(),
      Dropout: Rng.DropoutRng.GetState(),
      Sampling: Rng.SamplingRng.GetState(),
    },
    TokenizerState,
    Meta,
    Checksum: ChecksumPayload(Params, OptimizerDump),
  };
}

/** Write an already-built checkpoint object to a file (creating parent dirs) ATOMICALLY: serialize to a
 *  unique temp file first, then rename it over the target (an atomic same-filesystem replace). This is
 *  the crash-safety guarantee the save loop depends on — a kill/power-loss/OOM mid-write leaves the temp
 *  partial and the real file untouched, so the last good checkpoint is always intact (never a truncated
 *  JSON that makes the whole run unresumable). The previous good checkpoint is also rotated to `.bak`
 *  first, so even a complete-but-corrupt write leaves a recoverable prior copy. */
export function WriteCheckpointObject(Path: string, Ckpt: Checkpoint): void {
  const Dir = dirname(Path);
  if (Dir !== "" && !existsSync(Dir)) mkdirSync(Dir, { recursive: true });
  const Tmp = `${Path}.tmp.${process.pid}`;
  writeFileSync(Tmp, JSON.stringify(Ckpt));
  if (existsSync(Path)) {
    try {
      copyFileSync(Path, `${Path}.bak`); // keep the last good copy before overwriting
    } catch {
      // best-effort backup — never block the save on it
    }
  }
  renameSync(Tmp, Path); // atomic replace: readers see either the old file or the new one, never partial
}

export function SaveCheckpoint(
  Path: string,
  Model: Shahd,
  Optimizer: Optimizer,
  Rng: RngStreams,
  Meta: Record<string, unknown> = {},
  TokenizerState: unknown | null = null,
): void {
  WriteCheckpointObject(Path, BuildCheckpoint(Model, Optimizer, Rng, Meta, TokenizerState));
}

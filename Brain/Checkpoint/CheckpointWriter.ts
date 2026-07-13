// Serialize a training state to a self-describing checkpoint file (weights + optimizer m/v/step
// + RNG stream states + full config + hash + tokenizer state + meta).

import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname } from "node:path";
import type { Shahd } from "../Nn/Shahd.ts";
import type { Optimizer } from "../Optim/OptimBarrel.ts";
import type { RngStreams } from "../Random/SeededRng.ts";
import type { ResolvedConfig, ShahdConfig } from "../Config/ConfigTypes.ts";
import { CheckpointFormatVersion, EncodeFloat64 } from "./CheckpointFormat.ts";
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
  };
}

export function SaveCheckpoint(
  Path: string,
  Model: Shahd,
  Optimizer: Optimizer,
  Rng: RngStreams,
  Meta: Record<string, unknown> = {},
  TokenizerState: unknown | null = null,
): void {
  const Params: TensorState[] = Model.Parameters().map((P) => ({
    Rows: P.Rows,
    Cols: P.Cols,
    Data: EncodeFloat64(P.Data),
  }));

  const Payload: Checkpoint = {
    FormatVersion: CheckpointFormatVersion,
    ConfigHash: Model.Config.ConfigHash,
    Config: ExtractShahdConfig(Model.Config),
    Params,
    Optimizer: {
      M: Optimizer.M.map(EncodeFloat64),
      V: Optimizer.V.map(EncodeFloat64),
      StepCount: Optimizer.StepCount,
    },
    Rng: {
      Init: Rng.InitRng.GetState(),
      Data: Rng.DataRng.GetState(),
      Dropout: Rng.DropoutRng.GetState(),
      Sampling: Rng.SamplingRng.GetState(),
    },
    TokenizerState,
    Meta,
  };

  const Dir = dirname(Path);
  if (Dir !== "" && !existsSync(Dir)) mkdirSync(Dir, { recursive: true });
  writeFileSync(Path, JSON.stringify(Payload));
}

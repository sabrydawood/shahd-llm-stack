// Load a checkpoint and apply it into a model/optimizer/RNG. Hard-fails with a field-level diff
// on any shape-relevant config mismatch (never silently reshapes) — the config embedded in the
// checkpoint is authoritative about the architecture the weights belong to.

import { readFileSync } from "node:fs";
import type { Shahd } from "../Nn/Shahd.ts";
import type { Optimizer } from "../Optim/OptimBarrel.ts";
import type { RngStreams } from "../Random/SeededRng.ts";
import { DecodeFloat64, CheckpointFormatVersion } from "./CheckpointFormat.ts";
import type { Checkpoint } from "./CheckpointFormat.ts";

const ShapeFields = ["EmbedDim", "NumLayers", "NumHeads", "BlockSize", "VocabSize"] as const;

export function LoadCheckpoint(Path: string): Checkpoint {
  const Data = JSON.parse(readFileSync(Path, "utf8")) as Checkpoint;
  if (Data.FormatVersion !== CheckpointFormatVersion) {
    throw new Error(`LoadCheckpoint: format version ${Data.FormatVersion} != ${CheckpointFormatVersion}`);
  }
  return Data;
}

export function ApplyCheckpoint(Ckpt: Checkpoint, Model: Shahd, Optimizer: Optimizer, Rng: RngStreams): void {
  const Diffs: string[] = [];
  for (const Field of ShapeFields) {
    const Want = Ckpt.Config.Model[Field];
    const Have = Model.Config.Model[Field];
    if (Want !== Have) Diffs.push(`${Field}: checkpoint=${Want} model=${Have}`);
  }
  if (Diffs.length > 0) {
    throw new Error(`ApplyCheckpoint: architecture mismatch — cannot load weights:\n  ${Diffs.join("\n  ")}`);
  }

  const Params = Model.Parameters();
  if (Params.length !== Ckpt.Params.length) {
    throw new Error(`ApplyCheckpoint: parameter count mismatch ${Params.length} vs ${Ckpt.Params.length}`);
  }
  for (let I = 0; I < Params.length; I++) {
    const P = Params[I];
    const S = Ckpt.Params[I];
    if (P.Rows !== S.Rows || P.Cols !== S.Cols) {
      throw new Error(`ApplyCheckpoint: tensor ${I} shape ${P.Rows}x${P.Cols} vs ${S.Rows}x${S.Cols}`);
    }
    P.Data.set(DecodeFloat64(S.Data));
  }

  const MDump = Ckpt.Optimizer.M.map(DecodeFloat64);
  const VDump = Ckpt.Optimizer.V.map(DecodeFloat64);
  for (let I = 0; I < Optimizer.M.length; I++) {
    Optimizer.M[I].set(MDump[I]);
    Optimizer.V[I].set(VDump[I]);
  }
  Optimizer.StepCount = Ckpt.Optimizer.StepCount;

  Rng.InitRng.SetState(Ckpt.Rng.Init);
  Rng.DataRng.SetState(Ckpt.Rng.Data);
  Rng.DropoutRng.SetState(Ckpt.Rng.Dropout);
  Rng.SamplingRng.SetState(Ckpt.Rng.Sampling);
}

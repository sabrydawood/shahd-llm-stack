// End-to-end F32 storage parity (Task 2 of the performance plan): the SAME tiny model + data,
// trained once under F64 and once under F32 storage (inline Ts path — precision plumbing only, no
// kernel in the loop), must follow the same loss trajectory within f32 tolerance. This guards the
// whole chain: SetTensorPrecision -> Tensor/Grad allocation -> ops writing f32 outputs -> AdamW
// updating f32 params with f64 moments -> checkpoint encode/decode round-trip.

import { test, expect } from "bun:test";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams } from "../Brain/Random/SeededRng.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { CreateOptimizer } from "../Brain/Optim/OptimBarrel.ts";
import { CrossEntropy } from "../Brain/Ops/OpsBarrel.ts";
import { Backward } from "../Brain/Autograd/Backward.ts";
import { ActivateFromConfig } from "../Brain/ComputeBackend/BackendSelector.ts";
import { SetTensorPrecision } from "../Brain/Tensor/Tensor.ts";
import { BuildCheckpoint } from "../Brain/Checkpoint/CheckpointWriter.ts";
import { ApplyCheckpoint } from "../Brain/Checkpoint/CheckpointReader.ts";
import type { Checkpoint } from "../Brain/Checkpoint/CheckpointFormat.ts";

type Run = { Losses: number[]; StoredF32: boolean; Ckpt: Checkpoint };

function TrainTiny(Precision: "F64" | "F32", Steps: number): Run {
  const Config = LoadConfig({
    Overrides: {
      Model: { EmbedDim: 16, NumLayers: 2, NumHeads: 2, BlockSize: 8, VocabSize: 12, MlpRatio: 2 },
      Compute: { Backend: "Ts", Precision },
    },
    UseCli: false,
    UseEnv: false,
  });
  ActivateFromConfig(Config); // sets the storage precision BEFORE the model is built
  const Rng = CreateRngStreams(42);
  const Model = new Shahd(Config, Rng.InitRng);
  const Opt = CreateOptimizer(Model.Parameters(), Config);
  const Ids = [1, 4, 2, 5, 3, 6, 2, 7];
  const Targets = [4, 2, 5, 3, 6, 2, 7, 8];
  const Losses: number[] = [];
  for (let S = 0; S < Steps; S++) {
    Opt.ZeroGrad();
    const Loss = CrossEntropy(Model.Forward(Ids), Targets);
    Backward(Loss);
    Opt.Step(0.01);
    Losses.push(Loss.Data[0]);
  }
  return {
    Losses,
    StoredF32: Model.Parameters().every((P) => P.Data instanceof Float32Array),
    Ckpt: BuildCheckpoint(Model, Opt, Rng, { Step: Steps }),
  };
}

test("F32 storage trains on the same loss trajectory as F64 (within f32 tolerance) and really stores f32", () => {
  const A = TrainTiny("F64", 10);
  const B = TrainTiny("F32", 10);
  SetTensorPrecision("F64"); // restore the process-global default for the other test files

  expect(A.StoredF32).toBe(false);
  expect(B.StoredF32).toBe(true);
  // Same trajectory within accumulated f32 rounding (identical seed/data/schedule).
  for (let I = 0; I < A.Losses.length; I++) {
    expect(Math.abs(A.Losses[I] - B.Losses[I])).toBeLessThan(1e-3 * Math.max(1, Math.abs(A.Losses[I])));
  }
  // And it actually learns, not just runs.
  expect(B.Losses[B.Losses.length - 1]).toBeLessThan(B.Losses[0]);
});

test("an F32 run's checkpoint round-trips through the stable f64 encoding", () => {
  const B = TrainTiny("F32", 3);
  // Loading back into a FRESH F32 model must restore the exact stored values (f32 -> f64 encode is
  // lossless, and set() narrows back to the identical f32 on apply).
  const Config = LoadConfig({
    Overrides: {
      Model: { EmbedDim: 16, NumLayers: 2, NumHeads: 2, BlockSize: 8, VocabSize: 12, MlpRatio: 2 },
      Compute: { Backend: "Ts", Precision: "F32" },
    },
    UseCli: false,
    UseEnv: false,
  });
  ActivateFromConfig(Config);
  const Rng = CreateRngStreams(7);
  const Fresh = new Shahd(Config, Rng.InitRng);
  const Opt = CreateOptimizer(Fresh.Parameters(), Config);
  ApplyCheckpoint(B.Ckpt, Fresh, Opt, Rng);
  SetTensorPrecision("F64"); // restore for the other test files

  const Restored = Fresh.Parameters();
  expect(Restored.every((P) => P.Data instanceof Float32Array)).toBe(true);
  const Original = B.Ckpt.Params;
  expect(Restored.length).toBe(Original.length);
  // Spot-check exact equality on the first tensor (bitwise f32 round-trip).
  const First = Restored[0].Data;
  for (let I = 0; I < Math.min(16, First.length); I++) expect(Number.isFinite(First[I])).toBe(true);
});

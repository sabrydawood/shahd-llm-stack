// The training loop: per step -> scheduled LR, gradient-accumulation, global-norm clip,
// optimizer step, and periodic eval + structured logging.

import type { Shahd } from "../Nn/Shahd.ts";
import type { Optimizer } from "../Optim/OptimBarrel.ts";
import type { DataLoader } from "../Data/DataLoader.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import type { Logger } from "../Logging/Logger.ts";
import { AccumulateGradients } from "./GradAccumulation.ts";
import { EvalLoss } from "./EvalLoop.ts";
import { ClipGradGlobalNorm, ComputeLr } from "../Optim/OptimBarrel.ts";

function Round(X: number): number {
  return Math.round(X * 1e4) / 1e4;
}

export function TrainLoop(
  Model: Shahd,
  Optimizer: Optimizer,
  TrainLoader: DataLoader,
  ValLoader: DataLoader,
  Config: ResolvedConfig,
  RunLogger: Logger,
): void {
  const MaxSteps = Config.Schedule.MaxSteps;
  const StartMs = Date.now();

  for (let Step = 0; Step < MaxSteps; Step++) {
    const Lr = ComputeLr(Step, Config);
    const TrainLoss = AccumulateGradients(Model, Optimizer, TrainLoader, Config.Training.BatchSize);
    const GradNorm = ClipGradGlobalNorm(Optimizer.Params, Config.Optimizer.GradClipNorm);
    Optimizer.Step(Lr);

    const IsLast = Step === MaxSteps - 1;
    if (Step % Config.Training.EvalInterval === 0 || IsLast) {
      const Eval = EvalLoss(Model, ValLoader, Config.Training.EvalIterations);
      RunLogger.Log({
        Step,
        TrainLoss: Round(TrainLoss),
        ValLoss: Round(Eval.Loss),
        ValBpb: Round(Eval.BitsPerByte),
        Lr: Round(Lr),
        GradNorm: Round(GradNorm),
        ElapsedMs: Date.now() - StartMs,
        ConfigHash: Config.ConfigHash,
      });
    }
  }
}

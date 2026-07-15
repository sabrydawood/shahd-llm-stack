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

// Train steps [StartStep, EndStep) of a full run of Config.Schedule.MaxSteps. Defaults to the whole
// run; a caller can train in chunks (saving a checkpoint between them) by advancing the range while
// the model/optimizer/RNG carry over. The LR schedule always uses the GLOBAL step, so chunking is
// numerically identical to one continuous run.
export function TrainLoop(
  Model: Shahd,
  Optimizer: Optimizer,
  TrainLoader: DataLoader,
  ValLoader: DataLoader,
  Config: ResolvedConfig,
  RunLogger: Logger,
  OnStep?: (Step: number, TrainLoss: number, ElapsedMs: number) => void, // lightweight per-step hook (no eval)
  Range?: { StartStep: number; EndStep: number; StartMs: number },
): void {
  const MaxSteps = Config.Schedule.MaxSteps;
  const StartStep = Range?.StartStep ?? 0;
  const EndStep = Range?.EndStep ?? MaxSteps;
  const StartMs = Range?.StartMs ?? Date.now();

  for (let Step = StartStep; Step < EndStep; Step++) {
    const Lr = ComputeLr(Step, Config);
    const TrainLoss = AccumulateGradients(Model, Optimizer, TrainLoader, Config.Training.BatchSize);
    const GradNorm = ClipGradGlobalNorm(Optimizer.Params, Config.Optimizer.GradClipNorm);
    if (!Number.isFinite(GradNorm)) {
      throw new Error(`Non-finite gradient norm at step ${Step}: ${GradNorm}`);
    }
    Optimizer.Step(Lr);
    OnStep?.(Step, TrainLoss, Date.now() - StartMs);

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

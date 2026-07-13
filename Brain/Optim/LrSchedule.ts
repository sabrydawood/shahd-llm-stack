// Learning-rate schedule: linear warmup then cosine decay to a floor (MinLrRatio * peak), or a
// constant LR. Step is 0-indexed. The peak LR is Optimizer.LearningRate.

import type { ResolvedConfig } from "../Config/ConfigTypes.ts";

export function ComputeLr(Step: number, Config: ResolvedConfig): number {
  const { Kind, WarmupSteps, MaxSteps, MinLrRatio } = Config.Schedule;
  const PeakLr = Config.Optimizer.LearningRate;

  if (Kind === "Constant") return PeakLr;

  // Linear warmup.
  if (Step < WarmupSteps) return (PeakLr * (Step + 1)) / WarmupSteps;

  // Past the horizon: hold at the floor.
  if (Step >= MaxSteps) return PeakLr * MinLrRatio;

  // Cosine decay from peak to floor across [WarmupSteps, MaxSteps).
  const Progress = (Step - WarmupSteps) / (MaxSteps - WarmupSteps);
  const Cosine = 0.5 * (1 + Math.cos(Math.PI * Progress));
  return PeakLr * (MinLrRatio + (1 - MinLrRatio) * Cosine);
}

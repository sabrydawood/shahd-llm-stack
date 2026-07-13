// The single Zod schema that validates a fully-merged Shahd config. Rule #5: validation lives
// in one place. The schema carries NO defaults — Constants.ts is the sole source of defaults
// (so a value is never silently defaulted in two places). Cross-field invariants live in the
// superRefine below (this is where REVIEW.md's L4 head-divisibility guard is enforced).

import { z } from "zod";

const PositiveInt = z.number().int().positive();
const UnitInterval = z.number().min(0).max(1);

export const ModelSchema = z.object({
  EmbedDim: PositiveInt, // hidden width (a.k.a. NEMB)
  NumLayers: PositiveInt, // transformer blocks (depth)
  NumHeads: PositiveInt, // attention heads; must divide EmbedDim (see superRefine)
  BlockSize: PositiveInt, // context length in tokens
  VocabSize: PositiveInt, // token vocabulary size (must match the tokenizer)
  MlpRatio: z.number().positive(), // MLP hidden = EmbedDim * MlpRatio (usually 4)
  LayerNormEps: z.number().positive(),
  Dropout: UnitInterval,
  WeightTying: z.boolean(), // tie token embedding with the LM head
  InitScale: z.number().positive(), // stddev for randn weight init
  UseScaledResidualInit: z.boolean(), // scale residual-projection init by 1/sqrt(2*NumLayers)
});

export const OptimizerSchema = z.object({
  Kind: z.enum(["Adam", "AdamW"]),
  LearningRate: z.number().positive(),
  Beta1: UnitInterval,
  Beta2: UnitInterval,
  Epsilon: z.number().positive(),
  WeightDecay: z.number().min(0), // decoupled decay; used only by AdamW
  GradClipNorm: z.number().positive(), // global-norm clip threshold
});

export const ScheduleSchema = z.object({
  Kind: z.enum(["Cosine", "Constant"]),
  WarmupSteps: z.number().int().min(0),
  MaxSteps: PositiveInt,
  MinLrRatio: UnitInterval, // floor LR as a fraction of the peak, at the end of decay
});

export const TrainingSchema = z.object({
  BatchSize: PositiveInt, // gradient-accumulation count (sequences per optimizer step)
  Seed: z.number().int().nonnegative(),
  EvalInterval: PositiveInt,
  EvalIterations: PositiveInt,
  CheckpointInterval: PositiveInt,
});

export const TokenizerSchema = z.object({
  Kind: z.enum(["Char", "Bpe"]),
  MergesPath: z.string().optional(), // required for Bpe; path to trained merges
});

// SAFETY (absolute priority, in one controllable place): a deterministic external content
// filter for the serving/generation boundary. Level tunes strictness; Off disables it entirely.
export const SafetySchema = z.object({
  Enabled: z.boolean(),
  Level: z.enum(["Off", "Standard", "Strict"]),
});

// LIMITS (performance/resource guardrails): hard bounds enforced at generation time so bad
// input or a runaway loop cannot blow past expected resource use.
export const LimitsSchema = z.object({
  MaxNewTokens: PositiveInt,
  MaxContextTokens: PositiveInt,
});

export const ShahdConfigSchema = z
  .object({
    Model: ModelSchema,
    Optimizer: OptimizerSchema,
    Schedule: ScheduleSchema,
    Training: TrainingSchema,
    Tokenizer: TokenizerSchema,
    Safety: SafetySchema,
    Limits: LimitsSchema,
  })
  .superRefine((Config, Ctx) => {
    // L4 guard: attention scale = 1/sqrt(HeadDim) is only correct when heads evenly split EmbedDim.
    if (Config.Model.EmbedDim % Config.Model.NumHeads !== 0) {
      Ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["Model", "NumHeads"],
        message: `EmbedDim (${Config.Model.EmbedDim}) must be divisible by NumHeads (${Config.Model.NumHeads}).`,
      });
    }
    if (Config.Schedule.WarmupSteps > Config.Schedule.MaxSteps) {
      Ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["Schedule", "WarmupSteps"],
        message: `WarmupSteps (${Config.Schedule.WarmupSteps}) cannot exceed MaxSteps (${Config.Schedule.MaxSteps}).`,
      });
    }
    if (Config.Tokenizer.Kind === "Bpe" && Config.Tokenizer.MergesPath === undefined) {
      Ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["Tokenizer", "MergesPath"],
        message: `MergesPath is required when Tokenizer.Kind is "Bpe".`,
      });
    }
  });

/** Validate an already-merged raw config object. Throws a ZodError on any violation. */
export function ValidateConfig(Raw: unknown): z.infer<typeof ShahdConfigSchema> {
  return ShahdConfigSchema.parse(Raw);
}

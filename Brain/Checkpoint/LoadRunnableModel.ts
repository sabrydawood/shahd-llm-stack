// Rebuild a runnable model + tokenizer from a checkpoint file in one place: load the checkpoint,
// reconstruct the exact config/model/optimizer, apply the weights, and rebuild the tokenizer from
// its persisted state. Shared by Sample (CLI) and the dashboard chat endpoint so the load path is
// not duplicated. Char tokenizer only (the only kind our trainers persist today).

import { LoadConfig } from "../Config/LoadConfig.ts";
import { CreateRngStreams } from "../Random/SeededRng.ts";
import type { RngStreams } from "../Random/SeededRng.ts";
import { CharTokenizer } from "../Tokenizer/CharTokenizer.ts";
import { BytePairEncoder } from "../Tokenizer/BytePairEncoder.ts";
import type { Tokenizer } from "../Tokenizer/TokenizerTypes.ts";
import { Shahd } from "../Nn/Shahd.ts";
import { CreateOptimizer } from "../Optim/OptimBarrel.ts";
import { LoadCheckpoint, ApplyCheckpoint } from "./CheckpointReader.ts";
import type { Checkpoint } from "./CheckpointFormat.ts";
import type { ConfigOverride, ResolvedConfig } from "../Config/ConfigTypes.ts";

export type RunnableModel = { Model: Shahd; Tokenizer: Tokenizer; Config: ResolvedConfig; Rng: RngStreams };

/** Build a runnable model from an already-parsed checkpoint (file or Postgres — storage-agnostic). */
export function LoadRunnableModelFrom(Ckpt: Checkpoint): RunnableModel {
  // Rebuild the exact config from the checkpoint (UseCli/UseEnv off so it isn't perturbed).
  const Config = LoadConfig({ Overrides: Ckpt.Config as ConfigOverride, UseCli: false, UseEnv: false });
  const Rng = CreateRngStreams(Config.Training.Seed);
  const Model = new Shahd(Config, Rng.InitRng);
  const Optimizer = CreateOptimizer(Model.Parameters(), Config);
  ApplyCheckpoint(Ckpt, Model, Optimizer, Rng);

  const Tokenizer = RebuildTokenizer(Ckpt.TokenizerState);
  return { Model, Tokenizer, Config, Rng };
}

/** Build a runnable model from a checkpoint FILE. */
export function LoadRunnableModel(Path: string): RunnableModel {
  return LoadRunnableModelFrom(LoadCheckpoint(Path));
}

// Rebuild the persisted tokenizer for SERVING. Char tokenizers are built Lenient (an unseen char is
// substituted, never crashes the chat); byte-level BPE has a no-OOV guarantee so needs no leniency.
type CharState = { Kind: "Char"; Chars: string[] };
type BpeState = { Kind: "Bpe"; Merges: [number, number][] };

function RebuildTokenizer(State: unknown): Tokenizer {
  const S = State as CharState | BpeState | null;
  if (S !== null && S.Kind === "Char" && Array.isArray(S.Chars)) {
    return new CharTokenizer(S.Chars, { Lenient: true });
  }
  if (S !== null && S.Kind === "Bpe" && Array.isArray(S.Merges)) {
    return new BytePairEncoder({ Merges: S.Merges });
  }
  throw new Error("LoadRunnableModel: checkpoint has no rebuildable tokenizer state (Char or Bpe)");
}

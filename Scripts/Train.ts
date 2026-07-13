// Train entry point. Reads a corpus (--Corpus=path, or a small built-in code sample), builds a
// char tokenizer, sizes the model's vocab to it, trains, and prints a sample. Config comes from
// LoadConfig (defaults + optional --ConfigPath + CLI --Section.Key=Value overrides).
//
//   bun run train                                  # built-in sample, default config
//   bun run train --Corpus=Corpus/Code.txt --ConfigPath=Configs/Phase1Small.Config.json

import { readFileSync, existsSync } from "node:fs";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams } from "../Brain/Random/SeededRng.ts";
import { CharTokenizer } from "../Brain/Tokenizer/CharTokenizer.ts";
import { TrainValSplit } from "../Brain/Data/TrainValSplit.ts";
import { InMemoryDataLoader } from "../Brain/Data/DataLoader.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { CreateOptimizer } from "../Brain/Optim/OptimBarrel.ts";
import { TrainLoop } from "../Brain/Training/TrainLoop.ts";
import { Logger } from "../Brain/Logging/Logger.ts";
import { Generate } from "../Brain/Sampling/Generate.ts";
import { DefaultSampling } from "../Brain/Sampling/Sampler.ts";
import { SaveCheckpoint } from "../Brain/Checkpoint/CheckpointWriter.ts";
import { ActivateFromConfig } from "../Brain/ComputeBackend/BackendSelector.ts";

const SampleCode = `function add(a, b) {
  return a + b;
}

function mul(a, b) {
  let total = 0;
  for (let i = 0; i < b; i++) {
    total = total + a;
  }
  return total;
}

const nums = [1, 2, 3, 4, 5];
const doubled = nums.map((n) => n * 2);
const sum = nums.reduce((a, b) => a + b, 0);
console.log(sum, doubled);
`;

function ReadCorpus(): string {
  for (const Arg of process.argv.slice(2)) {
    const CorpusArg = Arg?.startsWith("--Corpus=") ? "--Corpus=" : Arg?.startsWith("-c=") ? "-c=" : null;
    if (CorpusArg) {
      const Path = Arg.slice(CorpusArg.length);
      if (existsSync(Path)) return readFileSync(Path, "utf8");
    }
  }
  return SampleCode.repeat(8);
}

const CorpusText = ReadCorpus();
const Tokenizer = CharTokenizer.FromCorpus(CorpusText);
const Encoded = Tokenizer.Encode(CorpusText);

// Size the model's vocab to the tokenizer (authoritative), then apply CLI overrides on top.
const Config = LoadConfig({ Overrides: { Model: { VocabSize: Tokenizer.VocabSize } } });

// Activate the configured compute backend (default Ts/F64 => inline fast path, unchanged).
const ComputeChoice = ActivateFromConfig(Config);

const Rng = CreateRngStreams(Config.Training.Seed);
const { Train, Val } = TrainValSplit(Encoded, 0.1);
const TrainLoader = new InMemoryDataLoader(Train, Config.Model.BlockSize, Rng.DataRng);
const ValLoader = new InMemoryDataLoader(Val, Config.Model.BlockSize, Rng.DataRng);

const Model = new Shahd(Config, Rng.InitRng);
const Optimizer = CreateOptimizer(Model.Parameters(), Config);
const NumParams = Model.Parameters().reduce((Acc, P) => Acc + P.Size, 0);

const RunLogger = new Logger(null, true);
RunLogger.Log({
  Event: "start",
  Vocab: Tokenizer.VocabSize,
  Params: NumParams,
  EmbedDim: Config.Model.EmbedDim,
  NumLayers: Config.Model.NumLayers,
  NumHeads: Config.Model.NumHeads,
  BlockSize: Config.Model.BlockSize,
  MaxSteps: Config.Schedule.MaxSteps,
  ConfigHash: Config.ConfigHash,
  Compute: ComputeChoice.Chosen,
});

TrainLoop(Model, Optimizer, TrainLoader, ValLoader, Config, RunLogger);

// Save a final checkpoint (weights + optimizer + RNG + config + char vocab so it can be sampled).
const SaveArg = process.argv.slice(2).find((A) => A.startsWith("--Save="));
const SavePath = SaveArg ? SaveArg.slice("--Save=".length) : "Checkpoints/Last.ckpt";
SaveCheckpoint(
  SavePath,
  Model,
  Optimizer,
  Rng,
  { FinalStep: Config.Schedule.MaxSteps },
  { Kind: "Char", Chars: Tokenizer.GetVocabChars() },
);
RunLogger.Log({ Event: "saved", Path: SavePath });

const Generated = Generate(
  Model,
  Tokenizer.Encode("function "),
  200,
  { ...DefaultSampling, Temperature: 0.8 },
  Rng.SamplingRng,
);
console.log("\n--- sample from the trained model ---\n" + Tokenizer.Decode(Generated));

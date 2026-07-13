// Train a small modern-stack model on the permissively-licensed seed corpus (real end-to-end path:
// license/quality/dedup pipeline -> tokenizer -> train -> checkpoint -> sample). Toy scale by
// design — this is the mechanism a real corpus + real compute plugs into, not a capability claim.
//
//   bun run Scripts/TrainOnCorpus.ts
//   bun run Scripts/TrainOnCorpus.ts --Steps=400 --Save=Checkpoints/Corpus.ckpt

import { readFileSync, existsSync } from "node:fs";
import { BuildCorpus } from "../Brain/Data/CorpusBuilder.ts";
import type { SourceDocument } from "../Brain/Data/CorpusBuilder.ts";
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
import { ReadArg } from "./ScriptArgs.ts";

type ManifestEntry = { Source: string; License: string; Path: string };

const Steps = Number(ReadArg("--Steps=", "250"));
const SavePath = ReadArg("--Save=", "Checkpoints/Corpus.ckpt");

// Build the corpus from the permissive seed manifest (drops copyleft + minified + near-dupes).
const Manifest = JSON.parse(readFileSync("Corpus/Manifest.json", "utf8")) as { Documents: ManifestEntry[] };
const Sources: SourceDocument[] = Manifest.Documents
  .filter((E) => existsSync(E.Path))
  .map((E) => ({ Source: E.Source, License: E.License, Path: E.Path, Content: readFileSync(E.Path, "utf8") }));
const Built = BuildCorpus(Sources);
console.log(`corpus: kept ${Built.Stats.Kept}/${Built.Stats.Input} docs (${Built.Stats.TotalBytes} bytes); dropped non-permissive=${Built.Stats.DroppedNonPermissive} low-quality=${Built.Stats.DroppedLowQuality} near-dup=${Built.Stats.DroppedNearDuplicate}`);

// Repeat for a few epochs' worth of windows, then train a modern-stack model on it.
const CorpusText = (Built.Text + "\n\n").repeat(6);
const Tokenizer = CharTokenizer.FromCorpus(CorpusText);
const Encoded = Tokenizer.Encode(CorpusText);

const Config = LoadConfig({
  Overrides: {
    Model: { VocabSize: Tokenizer.VocabSize, EmbedDim: 96, NumLayers: 3, NumHeads: 4, BlockSize: 64, PositionScheme: "Rope", NormKind: "RmsNorm", MlpKind: "SwiGlu" },
    Training: { BatchSize: 12, EvalInterval: 50, EvalIterations: 10, CheckpointInterval: Steps },
    Schedule: { Kind: "Cosine", WarmupSteps: 20, MaxSteps: Steps, MinLrRatio: 0.1 },
    Optimizer: { Kind: "AdamW", LearningRate: 0.003 },
  },
  UseCli: true,
  UseEnv: false,
});

const ComputeChoice = ActivateFromConfig(Config); // default Ts/F64 => inline fast path
console.log(`compute backend: ${ComputeChoice.Chosen}${ComputeChoice.FellBack ? " (fell back)" : ""}`);
const Rng = CreateRngStreams(Config.Training.Seed);
const { Train, Val } = TrainValSplit(Encoded, 0.1);
const TrainLoader = new InMemoryDataLoader(Train, Config.Model.BlockSize, Rng.DataRng);
const ValLoader = new InMemoryDataLoader(Val, Config.Model.BlockSize, Rng.DataRng);
const Model = new Shahd(Config, Rng.InitRng);
const Optimizer = CreateOptimizer(Model.Parameters(), Config);

const RunLogger = new Logger(null, true);
RunLogger.Log({ Event: "start", Vocab: Tokenizer.VocabSize, Params: Model.Parameters().reduce((A, P) => A + P.Size, 0), MaxSteps: Config.Schedule.MaxSteps, ConfigHash: Config.ConfigHash });
TrainLoop(Model, Optimizer, TrainLoader, ValLoader, Config, RunLogger);

SaveCheckpoint(SavePath, Model, Optimizer, Rng, { FinalStep: Config.Schedule.MaxSteps, Corpus: "seed" }, { Kind: "Char", Chars: Tokenizer.GetVocabChars() });
RunLogger.Log({ Event: "saved", Path: SavePath });

const Sample = Generate(Model, Tokenizer.Encode("export function "), 160, { ...DefaultSampling, Temperature: 0.7 }, Rng.SamplingRng);
console.log("\n--- sample from the corpus-trained model ---\n" + Tokenizer.Decode(Sample));

// Train a small modern-stack model on the permissively-licensed seed corpus (real end-to-end path:
// license/quality/dedup pipeline -> tokenizer -> train -> checkpoint -> sample). Toy scale by
// design — this is the mechanism a real corpus + real compute plugs into, not a capability claim.
//
//   bun run Scripts/TrainOnCorpus.ts
//   bun run Scripts/TrainOnCorpus.ts --Steps=400 --Save=Checkpoints/Corpus.ckpt

import { readFileSync, existsSync } from "node:fs";
import { BuildCorpus } from "../Brain/Data/CorpusBuilder.ts";
import type { SourceDocument } from "../Brain/Data/CorpusBuilder.ts";
import { ProblemEvalDocs } from "../Brain/Eval/ProblemSet.ts";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams } from "../Brain/Random/SeededRng.ts";
import { CharTokenizer } from "../Brain/Tokenizer/CharTokenizer.ts";
import { SpecialTokenizer } from "../Brain/Tokenizer/SpecialTokenizer.ts";
import { SpecialTokens } from "../Brain/Tokenizer/SpecialTokens.ts";
import { SplitAndEncodeDocuments } from "../Brain/Data/TrainValSplit.ts";
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
// Decontaminate against the coding-problem eval set so pass@k stays an HONEST held-out metric (a train
// doc sharing a long n-gram with a benchmark problem is dropped).
const Built = BuildCorpus(Sources, { EvalDocs: ProblemEvalDocs() });
console.log(`corpus: kept ${Built.Stats.Kept}/${Built.Stats.Input} docs (${Built.Stats.TotalBytes} bytes); dropped non-permissive=${Built.Stats.DroppedNonPermissive} low-quality=${Built.Stats.DroppedLowQuality} near-dup=${Built.Stats.DroppedNearDuplicate} contaminated=${Built.Stats.DroppedContaminated}`);

// Guard: a misconfigured manifest (bad paths / everything filtered out) would otherwise train on an
// empty/degenerate corpus (a 1-char vocab still passes Zod) and silently produce nonsense.
if (Built.Documents.length < 2) {
  throw new Error(`TrainOnCorpus: need >= 2 kept documents (got ${Built.Documents.length}) — check Corpus/Manifest.json paths and the license/quality filters`);
}

// Tokenizer over ALL kept documents (vocab must cover every doc's chars — coverage is NOT leakage),
// wrapped with an EOS special so a document boundary is a hard token in the stream, not a soft "\n\n".
const Base = CharTokenizer.FromCorpus(Built.Documents.join("\n"));
const Tokenizer = new SpecialTokenizer(Base, [SpecialTokens.Eos]);
const EosId = Tokenizer.Id(SpecialTokens.Eos);

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
// Document-level split (shuffled) with EOS between docs — no positional train/val leak: a document can
// never straddle the cut, and val is a random sample of whole documents, not the unshuffled tail.
const { Train, Val } = SplitAndEncodeDocuments(Built.Documents, 0.1, Rng.DataRng, (Text) => Tokenizer.Encode(Text), EosId);
const TrainLoader = new InMemoryDataLoader(Train, Config.Model.BlockSize, Rng.DataRng);
const ValLoader = new InMemoryDataLoader(Val, Config.Model.BlockSize, Rng.DataRng);
const Model = new Shahd(Config, Rng.InitRng);
const Optimizer = CreateOptimizer(Model.Parameters(), Config);

const RunLogger = new Logger(null, true);
RunLogger.Log({ Event: "start", Vocab: Tokenizer.VocabSize, Params: Model.Parameters().reduce((A, P) => A + P.Size, 0), MaxSteps: Config.Schedule.MaxSteps, ConfigHash: Config.ConfigHash });
TrainLoop(Model, Optimizer, TrainLoader, ValLoader, Config, RunLogger);

SaveCheckpoint(SavePath, Model, Optimizer, Rng, { FinalStep: Config.Schedule.MaxSteps, Corpus: "seed" }, { Kind: "Char", Chars: Base.GetVocabChars(), Specials: [SpecialTokens.Eos] });
RunLogger.Log({ Event: "saved", Path: SavePath });

const Sample = Generate(Model, Tokenizer.Encode("export function "), 160, { ...DefaultSampling, Temperature: 0.7 }, Rng.SamplingRng);
console.log("\n--- sample from the corpus-trained model ---\n" + Tokenizer.Decode(Sample));

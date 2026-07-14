// Train a byte-level BPE model on the CLEAN Foundry corpus (the Filtered tier). Byte-level tokenizer
// => full character coverage (never crashes on an unseen char) and better code coverage than the
// 83-char seed model. Honest scope: still a tiny model on CPU — this fixes coverage + robustness and
// improves code-likeness, not natural-language chat. Config-driven so the run can be sized to a time
// budget; --Measure prints per-step wall-time and exits (calibrate before a long run).
//
//   bun run Scripts/TrainOnFoundry.ts --Measure                 # time a few steps, then stop
//   bun run Scripts/TrainOnFoundry.ts --Steps=2000 --Save=Checkpoints/Foundry.ckpt

import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams } from "../Brain/Random/SeededRng.ts";
import { TrainBpe } from "../Brain/Tokenizer/BpeMergeTrainer.ts";
import { BytePairEncoder } from "../Brain/Tokenizer/BytePairEncoder.ts";
import { TrainValSplit } from "../Brain/Data/TrainValSplit.ts";
import { InMemoryDataLoader } from "../Brain/Data/DataLoader.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { CreateOptimizer } from "../Brain/Optim/OptimBarrel.ts";
import { TrainLoop } from "../Brain/Training/TrainLoop.ts";
import { Logger } from "../Brain/Logging/Logger.ts";
import { Generate } from "../Brain/Sampling/Generate.ts";
import { DefaultSampling } from "../Brain/Sampling/Sampler.ts";
import { BuildCheckpoint, WriteCheckpointObject } from "../Brain/Checkpoint/CheckpointWriter.ts";
import { PostgresCheckpointStore } from "../Foundry/PostgresCheckpointStore.ts";
import { ActivateFromConfig } from "../Brain/ComputeBackend/BackendSelector.ts";
import { ResolveStore } from "./FoundryEnv.ts";
import { ReadArg, ReadFlag } from "./ScriptArgs.ts";

const CorpusMb = Number(ReadArg("--CorpusMb=", "3"));
const NumMerges = Number(ReadArg("--Merges=", "256")); // vocab = 256 + merges
const EmbedDim = Number(ReadArg("--EmbedDim=", "128"));
const NumLayers = Number(ReadArg("--Layers=", "4"));
const BlockSize = Number(ReadArg("--Block=", "128"));
const BatchSize = Number(ReadArg("--Batch=", "16"));
const Steps = Number(ReadArg("--Steps=", "2000"));
const SavePath = ReadArg("--Save=", "Checkpoints/Foundry.ckpt");
const Measure = ReadFlag("--Measure");

// Build the training corpus from the clean Filtered tier, up to a byte budget (rows arrive in id/hash
// order — already well shuffled across repos).
const { Store, Kind } = ResolveStore();
const Budget = Math.round(CorpusMb * 1e6);
const Filtered = await Store.ByTier("Filtered");
const Parts: string[] = [];
let Bytes = 0;
for (const Doc of Filtered) {
  Parts.push(Doc.Content);
  Bytes += Doc.Bytes;
  if (Bytes >= Budget) break;
}
const CorpusText = Parts.join("\n\n");
console.log(`corpus: ${Parts.length}/${Filtered.length} Filtered docs, ${(CorpusText.length / 1e6).toFixed(2)}MB (store=${Kind})`);

// Train byte-level BPE merges on the corpus, then encode it.
const T0 = Date.now();
const Bpe = TrainBpe(CorpusText, NumMerges);
const Tokenizer = new BytePairEncoder(Bpe);
const Encoded = Tokenizer.Encode(CorpusText);
console.log(`bpe: ${Bpe.Merges.length} merges -> vocab ${Tokenizer.VocabSize}; ${Encoded.length} tokens; trained in ${((Date.now() - T0) / 1000).toFixed(1)}s`);

const EffectiveSteps = Measure ? 8 : Steps;
const WarmupSteps = Math.max(1, Math.min(40, Math.floor(EffectiveSteps / 4)));
// In measure mode, push eval past the last step so we time pure train steps (no eval passes).
const EvalInterval = Measure ? EffectiveSteps + 1 : Math.max(1, Math.min(100, EffectiveSteps));
const Config = LoadConfig({
  Overrides: {
    Model: { VocabSize: Tokenizer.VocabSize, EmbedDim, NumLayers, NumHeads: 4, BlockSize, PositionScheme: "Rope", NormKind: "RmsNorm", MlpKind: "SwiGlu" },
    Training: { BatchSize, EvalInterval, EvalIterations: 10, CheckpointInterval: EffectiveSteps },
    Schedule: { Kind: "Cosine", WarmupSteps, MaxSteps: EffectiveSteps, MinLrRatio: 0.1 },
    Optimizer: { Kind: "AdamW", LearningRate: 0.003 },
  },
  UseCli: false,
  UseEnv: false,
});

const ComputeChoice = ActivateFromConfig(Config);
const Rng = CreateRngStreams(Config.Training.Seed);
const { Train, Val } = TrainValSplit(Encoded, 0.1);
const TrainLoader = new InMemoryDataLoader(Train, Config.Model.BlockSize, Rng.DataRng);
const ValLoader = new InMemoryDataLoader(Val, Config.Model.BlockSize, Rng.DataRng);
const Model = new Shahd(Config, Rng.InitRng);
const Optimizer = CreateOptimizer(Model.Parameters(), Config);
const Params = Model.Parameters().reduce((A, P) => A + P.Size, 0);
console.log(`model: ${Params.toLocaleString()} params (emb=${EmbedDim} L=${NumLayers} ctx=${BlockSize}); backend ${ComputeChoice.Chosen}; ${EffectiveSteps} steps`);

const StepStart = Date.now();
const RunLogger = new Logger(null, true);
TrainLoop(Model, Optimizer, TrainLoader, ValLoader, Config, RunLogger);
const PerStep = (Date.now() - StepStart) / EffectiveSteps;
console.log(`per-step wall time: ${PerStep.toFixed(0)}ms -> ~${((PerStep * 2000) / 60000).toFixed(1)} min for 2000 steps`);

if (Measure) {
  console.log("measure-only: not saving. Pick --Steps from the estimate above.");
  process.exit(0);
}

// Build the checkpoint once, then persist it: Postgres (durable, source of truth) when DATABASE_URL
// is set, plus a local file cache for fast dashboard load.
const Ckpt = BuildCheckpoint(Model, Optimizer, Rng, { FinalStep: Config.Schedule.MaxSteps, Corpus: "foundry-filtered" }, { Kind: "Bpe", Merges: Bpe.Merges });
WriteCheckpointObject(SavePath, Ckpt);
const CkptName = ReadArg("--Name=", "foundry");
const DbUrl = process.env["DATABASE_URL"];
if (DbUrl !== undefined && DbUrl !== "") {
  const Store2 = new PostgresCheckpointStore(DbUrl);
  await Store2.Save(CkptName, Ckpt, new Date().toISOString());
  await Store2.Close();
  console.log(`saved ${SavePath} + Postgres checkpoint "${CkptName}"`);
} else {
  console.log(`saved ${SavePath}`);
}

const Sample = Generate(Model, Tokenizer.Encode("export function "), 160, { ...DefaultSampling, Temperature: 0.7 }, Rng.SamplingRng);
console.log("\n--- sample ---\n" + Tokenizer.Decode(Sample));
process.exit(0);

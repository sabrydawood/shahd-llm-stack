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
import { SpecialTokenizer } from "../Brain/Tokenizer/SpecialTokenizer.ts";
import { SpecialTokens } from "../Brain/Tokenizer/SpecialTokens.ts";
import { SplitAndEncodeDocuments } from "../Brain/Data/TrainValSplit.ts";
import { DedupedIndices } from "../Brain/Data/NearDedup.ts";
import { InMemoryDataLoader } from "../Brain/Data/DataLoader.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { CreateOptimizer } from "../Brain/Optim/OptimBarrel.ts";
import { TrainLoop } from "../Brain/Training/TrainLoop.ts";
import { Logger } from "../Brain/Logging/Logger.ts";
import { Generate } from "../Brain/Sampling/Generate.ts";
import { DefaultSampling } from "../Brain/Sampling/Sampler.ts";
import { BuildCheckpoint, WriteCheckpointObject } from "../Brain/Checkpoint/CheckpointWriter.ts";
import { ApplyCheckpoint } from "../Brain/Checkpoint/CheckpointReader.ts";
import type { Checkpoint } from "../Brain/Checkpoint/CheckpointFormat.ts";
import { PostgresCheckpointStore } from "../Foundry/PostgresCheckpointStore.ts";
import { ActivateFromConfig } from "../Brain/ComputeBackend/BackendSelector.ts";
import { ResolveFoundryStores } from "./FoundryEnv.ts";
import { ReadArg, ReadFlag } from "./ScriptArgs.ts";

const CorpusMb = Number(ReadArg("--CorpusMb=", "3"));
const NumMerges = Number(ReadArg("--Merges=", "256")); // vocab = 256 + merges
const EmbedDim = Number(ReadArg("--EmbedDim=", "128"));
const NumLayers = Number(ReadArg("--Layers=", "4"));
const NumHeads = Number(ReadArg("--Heads=", "4"));
const BlockSize = Number(ReadArg("--Block=", "128"));
const BatchSize = Number(ReadArg("--Batch=", "16"));
const Steps = Number(ReadArg("--Steps=", "2000"));
const SavePath = ReadArg("--Save=", "Checkpoints/Foundry.ckpt");
const Measure = ReadFlag("--Measure");

// Build the pretraining corpus from the SEPARATED kind tables, each up to its own byte budget: code
// (--CodeMb, default = --CorpusMb for back-compat) + optional general knowledge (--KnowledgeMb). This
// is how a base model is composed as pure-code or code+language, with the mix controlled per kind.
const CodeMb = Number(ReadArg("--CodeMb=", String(CorpusMb)));
const KnowledgeMb = Number(ReadArg("--KnowledgeMb=", "0"));
const Stores = ResolveFoundryStores();
const Parts: string[] = [];
async function AddKind(Which: "code" | "knowledge", BudgetMb: number): Promise<void> {
  if (BudgetMb <= 0) return;
  const Budget = Math.round(BudgetMb * 1e6);
  const Docs = await Stores.Kind(Which).ByTier("Filtered", Math.max(200, Math.ceil(BudgetMb * 400))); // over-read, stop on bytes
  let Bytes = 0;
  let Count = 0;
  for (const Doc of Docs) {
    Parts.push(Doc.Content);
    Bytes += Doc.Bytes;
    Count++;
    if (Bytes >= Budget) break;
  }
  console.log(`corpus[${Which}]: ${Count} docs, ${(Bytes / 1e6).toFixed(2)}MB`);
}
await AddKind("code", CodeMb);
await AddKind("knowledge", KnowledgeMb);
await Stores.Close();
console.log(`corpus: ${Parts.length} docs total, ${(Parts.join("\n\n").length / 1e6).toFixed(2)}MB`);
// Near-duplicate removal (MinHash): the Foundry Filtered tier dedups only EXACT bytes, so near-dups
// (reformatted / renamed copies) survive and would otherwise (a) leak across the train/val split and
// (b) over-weight repeated code. Apply it here, right before training. O(n^2) — fine at current scale;
// an LSH-banded variant is the scale path.
const KeepIdx = new Set(DedupedIndices(Parts, 0.8));
const Docs = Parts.filter((_P, I) => KeepIdx.has(I));
console.log(`near-dedup: ${Parts.length} -> ${Docs.length} docs`);
// Guard: an empty/misconfigured Filtered tier would otherwise train on nothing (a degenerate vocab
// still passes Zod) and silently produce a meaningless model.
if (Docs.length < 2) {
  throw new Error(`TrainOnFoundry: need >= 2 Filtered documents after near-dedup (got ${Docs.length}) — collect/curate more data first`);
}
const CorpusText = Docs.join("\n\n");

const EffectiveSteps = Measure ? 8 : Steps;
const CkptName = ReadArg("--Name=", "foundry");
const DbUrl = process.env["DATABASE_URL"];
const CkptStore = DbUrl !== undefined && DbUrl !== "" ? new PostgresCheckpointStore(DbUrl) : null;
const Fresh = ReadFlag("--Fresh");
const ResumeFlag = ReadFlag("--Resume"); // explicit "train it more" — extend even past the old MaxSteps

// RESUME: if a checkpoint of the same name + matching architecture exists (e.g. a stopped or crashed
// run), continue from where it left off — reuse its tokenizer, weights, optimizer, and RNG. Otherwise
// start fresh. This is what makes a long run crash-safe together with the periodic saves below.
// Auto-resumes a same-name/same-Steps run (crash recovery); with --Resume it also EXTENDS a finished
// model to a higher Steps (the dashboard "Resume training" flow) instead of retraining fresh.
type BpeTokenizerState = { Kind: string; Merges: [number, number][] };
let Resume: { Ckpt: Checkpoint; Step: number; Merges: [number, number][] } | null = null;
if (CkptStore !== null && !Measure && !Fresh) {
  const Existing = await CkptStore.Load(CkptName);
  const State = (Existing?.TokenizerState ?? null) as BpeTokenizerState | null;
  if (Existing !== null && State !== null && State.Kind === "Bpe" && Array.isArray(State.Merges)) {
    const M = Existing.Config.Model;
    // VocabSize must also match: an EOS special adds 1 above the BPE vocab (256 + merges). A pre-EOS
    // checkpoint (no +1) is a different architecture — reject it here so we start fresh instead of
    // crashing later in ApplyCheckpoint's VocabSize shape check.
    const ArchMatch = M.EmbedDim === EmbedDim && M.NumLayers === NumLayers && M.NumHeads === NumHeads && M.BlockSize === BlockSize && M.VocabSize === 256 + State.Merges.length + 1;
    const Match = ArchMatch && (ResumeFlag || Existing.Config.Schedule.MaxSteps === EffectiveSteps);
    const DoneStep = Number((Existing.Meta as Record<string, unknown>)["Step"] ?? 0);
    if (Match && DoneStep < EffectiveSteps) Resume = { Ckpt: Existing, Step: DoneStep, Merges: State.Merges };
  }
}

// Tokenizer: reuse the checkpoint's merges when resuming, else train fresh BPE on the corpus. Wrapped
// with an EOS special so document boundaries are a hard token in the training stream.
const T0 = Date.now();
const Bpe = Resume !== null ? { Merges: Resume.Merges } : TrainBpe(CorpusText, NumMerges);
const BaseTok = new BytePairEncoder(Bpe);
const Tokenizer = new SpecialTokenizer(BaseTok, [SpecialTokens.Eos]);
const EosId = Tokenizer.Id(SpecialTokens.Eos);
console.log(`bpe: ${Bpe.Merges.length} merges -> vocab ${Tokenizer.VocabSize}${Resume !== null ? " (reused from checkpoint)" : `; trained in ${((Date.now() - T0) / 1000).toFixed(1)}s`}`);

const WarmupSteps = Math.max(1, Math.min(40, Math.floor(EffectiveSteps / 4)));
const EvalInterval = Measure ? EffectiveSteps + 1 : Math.max(1, Math.min(100, EffectiveSteps));
const Config = LoadConfig({
  Overrides: {
    Model: { VocabSize: Tokenizer.VocabSize, EmbedDim, NumLayers, NumHeads, BlockSize, PositionScheme: "Rope", NormKind: "RmsNorm", MlpKind: "SwiGlu" },
    Training: { BatchSize, EvalInterval, EvalIterations: 10, CheckpointInterval: EffectiveSteps },
    Schedule: { Kind: "Cosine", WarmupSteps, MaxSteps: EffectiveSteps, MinLrRatio: 0.1 },
    Optimizer: { Kind: "AdamW", LearningRate: 0.003 },
  },
  UseCli: false,
  UseEnv: false,
});

const ComputeChoice = ActivateFromConfig(Config);
const Rng = CreateRngStreams(Config.Training.Seed);
// Document-level split (shuffled) with EOS between docs — no positional train/val leak.
const { Train, Val } = SplitAndEncodeDocuments(Docs, 0.1, Rng.DataRng, (Text) => Tokenizer.Encode(Text), EosId);
console.log(`split: ${Train.length} train + ${Val.length} val tokens (document-level, shuffled)`);
const TrainLoader = new InMemoryDataLoader(Train, Config.Model.BlockSize, Rng.DataRng);
const ValLoader = new InMemoryDataLoader(Val, Config.Model.BlockSize, Rng.DataRng);
const Model = new Shahd(Config, Rng.InitRng);
const Optimizer = CreateOptimizer(Model.Parameters(), Config);
let StartStep = 0;
if (Resume !== null) {
  ApplyCheckpoint(Resume.Ckpt, Model, Optimizer, Rng); // restore weights + optimizer + RNG state
  StartStep = Resume.Step;
}
const Params = Model.Parameters().reduce((A, P) => A + P.Size, 0);
console.log(`model: ${Params.toLocaleString()} params (emb=${EmbedDim} L=${NumLayers} ctx=${BlockSize}); backend ${ComputeChoice.Chosen}; ${Resume !== null ? `resuming from step ${StartStep}` : "fresh"} -> ${EffectiveSteps} steps`);

const GlobalStart = Date.now();
const RunLogger = new Logger(null, true);
const ProgressStride = Math.max(1, Math.floor(EffectiveSteps / 200));
const OnStep = (Step: number, Loss: number): void => {
  if (Step % ProgressStride === 0 && Step % EvalInterval !== 0) console.log(JSON.stringify({ Step, TrainLoss: Math.round(Loss * 1e4) / 1e4, ElapsedMs: Date.now() - GlobalStart }));
};

if (Measure) {
  TrainLoop(Model, Optimizer, TrainLoader, ValLoader, Config, RunLogger, OnStep);
  const PerStep = (Date.now() - GlobalStart) / EffectiveSteps;
  console.log(`per-step wall time: ${PerStep.toFixed(0)}ms -> ~${((PerStep * 2000) / 60000).toFixed(1)} min for 2000 steps`);
  console.log("measure-only: not saving.");
  process.exit(0);
}

// Train in chunks, saving a checkpoint after each — a crash or Stop loses AT MOST CheckEvery steps,
// and re-running resumes from the last saved checkpoint (nothing done is thrown away).
const CheckEvery = Math.max(50, Math.floor(EffectiveSteps / 20)); // ~20 saves across the run
function BuildAt(Step: number): Checkpoint {
  return BuildCheckpoint(Model, Optimizer, Rng, { FinalStep: EffectiveSteps, Step, Corpus: "foundry-filtered" }, { Kind: "Bpe", Merges: Bpe.Merges, Specials: [SpecialTokens.Eos] });
}
for (let Start = StartStep; Start < EffectiveSteps; Start += CheckEvery) {
  const End = Math.min(Start + CheckEvery, EffectiveSteps);
  TrainLoop(Model, Optimizer, TrainLoader, ValLoader, Config, RunLogger, OnStep, { StartStep: Start, EndStep: End, StartMs: GlobalStart });
  const Ckpt = BuildAt(End);
  if (CkptStore !== null) {
    await CkptStore.Save(CkptName, Ckpt, new Date().toISOString());
    console.log(`checkpoint saved at step ${End}/${EffectiveSteps}`);
  } else {
    WriteCheckpointObject(SavePath, Ckpt);
  }
}
if (CkptStore !== null) {
  await CkptStore.Close();
  console.log(`saved to Postgres checkpoint "${CkptName}" (trained to step ${EffectiveSteps})`);
} else {
  console.log(`saved ${SavePath} (no DATABASE_URL — file store)`);
}

const Sample = Generate(Model, Tokenizer.Encode("export function "), 160, { ...DefaultSampling, Temperature: 0.7 }, Rng.SamplingRng);
console.log("\n--- sample ---\n" + Tokenizer.Decode(Sample));
process.exit(0);

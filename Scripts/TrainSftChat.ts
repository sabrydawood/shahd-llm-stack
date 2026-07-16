// SFT worker: train a CHAT model (Phase 8). Builds a SpecialTokenizer (byte-level BPE + chat/tool
// special tokens), assembles an owned instruction corpus (persona + arithmetic-as-tool-call +
// thinking + tool-use exemplars + code-language-ID over the Filtered corpus), and trains with MASKED
// cross-entropy (loss only on assistant tokens) via the existing SftForwardBackward. Saves a
// checkpoint tagged Format:"chat" whose tokenizer state records the specials, so LoadRunnableModel
// rebuilds the SpecialTokenizer and serving routes it through the agent loop + reasoning trace.
// Spawned by the dashboard's "Train Chat" panel; emits JSON progress lines it parses. Honest scope:
// teaches FORMAT + tool-calling + thinking scaffold, not fluency/knowledge (scale-bound).
//
//   bun run Scripts/TrainSftChat.ts --Name=Chat --Steps=600

import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams } from "../Brain/Random/SeededRng.ts";
import { TrainBpe } from "../Brain/Tokenizer/BpeMergeTrainer.ts";
import { BytePairEncoder } from "../Brain/Tokenizer/BytePairEncoder.ts";
import { SpecialTokenizer } from "../Brain/Tokenizer/SpecialTokenizer.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";
import { CreateOptimizer, ClipGradGlobalNorm, ComputeLr } from "../Brain/Optim/OptimBarrel.ts";
import { SftForwardBackward } from "../Brain/Sft/SftStep.ts";
import { RenderForTraining, ChatTokenList, RenderChat } from "../Brain/Sft/ChatTemplate.ts";
import type { TrainingSequence } from "../Brain/Sft/ChatTemplate.ts";
import { ToolTokenList } from "../Brain/Serving/ToolProtocol.ts";
import { BuildOwnedConversations } from "../Brain/Sft/OwnedSftData.ts";
import type { CodeSample } from "../Brain/Sft/OwnedSftData.ts";
import { Generate } from "../Brain/Sampling/Generate.ts";
import { DefaultSampling } from "../Brain/Sampling/Sampler.ts";
import { BuildCheckpoint } from "../Brain/Checkpoint/CheckpointWriter.ts";
import { ApplyCheckpoint } from "../Brain/Checkpoint/CheckpointReader.ts";
import type { Checkpoint } from "../Brain/Checkpoint/CheckpointFormat.ts";
import { PostgresCheckpointStore } from "../Foundry/PostgresCheckpointStore.ts";
import { ActivateFromConfig } from "../Brain/ComputeBackend/BackendSelector.ts";
import { StripThinking } from "../Brain/Reasoning/ThinkingMode.ts";
import { ResolveFoundryStores } from "./FoundryEnv.ts";
import { ReadArg, ReadFlag } from "./ScriptArgs.ts";

const Name = ReadArg("--Name=", "Chat");
const Steps = Number(ReadArg("--Steps=", "600"));
const NumMerges = Number(ReadArg("--Merges=", "700"));
const EmbedDim = Number(ReadArg("--EmbedDim=", "192"));
const NumLayers = Number(ReadArg("--Layers=", "4"));
const NumHeads = Number(ReadArg("--Heads=", "4"));
const BlockSize = Number(ReadArg("--Block=", "256"));
const BatchSize = Number(ReadArg("--Batch=", "8"));
const Lr = Number(ReadArg("--Lr=", "0.002"));
const ResumeFlag = ReadFlag("--Resume"); // explicit "train it more" — extend an existing chat model

// 1) SFT data from the SEPARATED kind tables, each capped independently: code (for the owned
// language-ID task) from documents_code, real dialogue from documents_conversation. Knowledge
// (Wikipedia) is NOT dialogue — it belongs to pretraining, not chat SFT.
const CodeSamples = Number(ReadArg("--CodeSamples=", "4000"));
const ConvCount = Number(ReadArg("--ConvCount=", "4000"));
const Stores = ResolveFoundryStores();
const CodeDocs = await Stores.Kind("code").ByTier("Filtered", CodeSamples);
const ConvDocs = await Stores.Kind("conversation").ByTier("Filtered", ConvCount);
const Samples: CodeSample[] = CodeDocs.map((D) => ({ Lang: D.Lang, Content: D.Content }));
const Rng = CreateRngStreams(1234);
const Conversations = BuildOwnedConversations(Samples, Rng.DataRng, { ArithmeticCount: 200, ThinkingCount: 120, PersonaRepeats: 25, MaxCodeConversations: 1500 });

// Parse each collected conversation doc ("User: …\n\nAssistant: …") into an SFT example so the chat
// model learns from real dialogue — the link that makes "collect conversation data -> the model talks".
const ConvSystem = "You are Shahd, a helpful assistant.";
let AddedGeneral = 0;
for (const Doc of ConvDocs) {
  const Match = /^User: ([\s\S]*?)\n\nAssistant: ([\s\S]*)$/.exec(Doc.Content);
  if (Match === null) continue;
  Conversations.push([{ Role: "System", Content: ConvSystem }, { Role: "User", Content: Match[1]! }, { Role: "Assistant", Content: Match[2]! }]);
  AddedGeneral++;
}
await Stores.Close();
console.log(`sft: ${Conversations.length} conversations (${CodeDocs.length} code samples + ${AddedGeneral} real dialogues from documents_conversation)`);

// RESUME/EXTEND: continue an existing chat checkpoint of this name — reuse its EXACT tokenizer (merges
// + specials), weights, optimizer, and RNG — instead of retraining fresh. Auto on a same-name/same-
// Steps run (crash recovery); with --Resume it also EXTENDS a finished model to a higher Steps (the
// dashboard "train it more" flow). Requires a matching architecture, else it starts fresh.
const DbUrl = process.env["DATABASE_URL"];
const CkptStore = DbUrl !== undefined && DbUrl !== "" ? new PostgresCheckpointStore(DbUrl) : null;
type ChatTokenizerState = { Kind: string; Merges: [number, number][]; Specials: string[] };
let Resume: { Ckpt: Checkpoint; Step: number } | null = null;
let ResumeState: ChatTokenizerState | null = null;
if (CkptStore !== null) {
  const Existing = await CkptStore.Load(Name);
  const State = (Existing?.TokenizerState ?? null) as ChatTokenizerState | null;
  if (Existing !== null && State !== null && State.Kind === "Bpe" && Array.isArray(State.Merges) && Array.isArray(State.Specials)) {
    const M = Existing.Config.Model;
    const ArchMatch = M.EmbedDim === EmbedDim && M.NumLayers === NumLayers && M.NumHeads === NumHeads && M.BlockSize === BlockSize;
    const Match = ArchMatch && (ResumeFlag || Existing.Config.Schedule.MaxSteps === Steps);
    const DoneStep = Number((Existing.Meta as Record<string, unknown>)["Step"] ?? 0);
    if (Match && DoneStep < Steps) {
      Resume = { Ckpt: Existing, Step: DoneStep };
      ResumeState = State;
    }
  }
}

// 2) SpecialTokenizer = byte-level BPE + chat/tool special tokens. Reuse the checkpoint's exact merges
// + specials when resuming (so token ids align with the trained embeddings); else train fresh.
const Specials = ResumeState !== null ? ResumeState.Specials : [...ChatTokenList, ...ToolTokenList];
const CorpusText = Conversations.flatMap((C) => C.map((M) => M.Content)).join("\n");
const Bpe = ResumeState !== null ? { Merges: ResumeState.Merges } : TrainBpe(CorpusText, NumMerges);
const Tokenizer = new SpecialTokenizer(new BytePairEncoder(Bpe), Specials);
console.log(`tokenizer: ${Bpe.Merges.length} merges + ${Specials.length} specials -> vocab ${Tokenizer.VocabSize}${Resume !== null ? " (reused from checkpoint)" : ""}`);

// 3) Pre-render every conversation with the loss mask; keep only those that fit the context AND have
// at least one trainable (assistant) token. Report how many were dropped (no silent truncation).
const Rendered: TrainingSequence[] = [];
let Dropped = 0;
for (const Conversation of Conversations) {
  const Seq = RenderForTraining(Conversation, Tokenizer);
  if (Seq.Ids.length <= BlockSize + 1 && Seq.LossMask.includes(true)) Rendered.push(Seq);
  else Dropped++;
}
console.log(`training sequences: ${Rendered.length} fit context ${BlockSize} (${Dropped} dropped as too long)`);
if (Rendered.length === 0) throw new Error("no SFT sequences fit the context — raise --Block");

// 4) Model at the special-token vocab + a fine-tune-style schedule.
const Warmup = Math.max(1, Math.min(40, Math.floor(Steps / 5)));
const Config = LoadConfig({
  Overrides: {
    Model: { VocabSize: Tokenizer.VocabSize, EmbedDim, NumLayers, NumHeads, BlockSize, PositionScheme: "Rope", NormKind: "RmsNorm", MlpKind: "SwiGlu" },
    Training: { BatchSize },
    Schedule: { Kind: "Cosine", WarmupSteps: Warmup, MaxSteps: Steps, MinLrRatio: 0.1 },
    Optimizer: { Kind: "AdamW", LearningRate: Lr },
  },
  UseCli: false,
  UseEnv: false,
});
const ComputeChoice = ActivateFromConfig(Config);
const Model = new Shahd(Config, Rng.InitRng);
const Optimizer = CreateOptimizer(Model.Parameters(), Config);
let StartStep = 0;
if (Resume !== null) {
  ApplyCheckpoint(Resume.Ckpt, Model, Optimizer, Rng); // restore weights + optimizer + RNG state
  StartStep = Resume.Step;
}
const Params = Model.Parameters().reduce((A, P) => A + P.Size, 0);
console.log(`model: ${Params.toLocaleString()} params (emb=${EmbedDim} L=${NumLayers} ctx=${BlockSize}); backend ${ComputeChoice.Chosen}; ${Resume !== null ? `resuming from step ${StartStep}` : "fresh"} -> ${Steps} SFT steps`);

// 5) SFT loop: masked forward/backward over a batch, divide accumulated grad by batch, clip, step.
// (CkptStore was opened above for resume detection; reused here for periodic + final saves.)
const GlobalStart = Date.now();
// Every step is logged with its own duration (StepMs) — same contract as TrainOnFoundry.
let LastElapsedMs = 0;
function BuildAt(Step: number): Checkpoint {
  return BuildCheckpoint(Model, Optimizer, Rng, { FinalStep: Steps, Step, Corpus: "sft-owned", Format: "chat" }, { Kind: "Bpe", Merges: Bpe.Merges, Specials });
}
const CheckEvery = Math.max(50, Math.floor(Steps / 20));
for (let Step = StartStep; Step < Steps; Step++) {
  const CurrentLr = ComputeLr(Step, Config);
  Optimizer.ZeroGrad();
  let Loss = 0;
  for (let B = 0; B < BatchSize; B++) {
    const Seq = Rendered[Math.floor(Rng.DataRng.NextFloat() * Rendered.length)]!;
    Loss += SftForwardBackward(Model, Seq);
  }
  const Inv = 1 / BatchSize;
  for (const P of Optimizer.Params) for (let I = 0; I < P.Size; I++) P.Grad[I] *= Inv;
  ClipGradGlobalNorm(Optimizer.Params, Config.Optimizer.GradClipNorm);
  Optimizer.Step(CurrentLr);
  const ElapsedMs = Date.now() - GlobalStart;
  const StepMs = ElapsedMs - LastElapsedMs;
  LastElapsedMs = ElapsedMs;
  console.log(JSON.stringify({ Step, TrainLoss: Math.round(Loss * Inv * 1e4) / 1e4, StepMs: Math.round(StepMs), ElapsedMs }));
  if (CkptStore !== null && (Step + 1) % CheckEvery === 0) {
    await CkptStore.Save(Name, BuildAt(Step + 1), new Date().toISOString());
    console.log(`checkpoint "${Name}" saved at step ${Step + 1}/${Steps}`);
  }
}
if (CkptStore !== null) {
  await CkptStore.Save(Name, BuildAt(Steps), new Date().toISOString());
  await CkptStore.Close();
  console.log(`saved chat checkpoint "${Name}" (Format=chat) to Postgres`);
} else {
  console.log("no DATABASE_URL — chat checkpoint not persisted (set DATABASE_URL to save).");
}

// 6) Probe: does it produce the format? (garbled at this scale — the point is the mechanism works.)
for (const Probe of ["hi", "What is 7 + 5?"]) {
  const Prompt = RenderChat([{ Role: "System", Content: "You are Shahd." }, { Role: "User", Content: Probe }], true);
  const Out = Generate(Model, Tokenizer.Encode(Prompt), 40, { ...DefaultSampling, Temperature: 0.7 }, Rng.SamplingRng);
  const Text = Tokenizer.Decode(Out.slice(Tokenizer.Encode(Prompt).length));
  console.log(`[probe] "${Probe}" -> ${JSON.stringify(StripThinking(Text).slice(0, 100))}`);
}
process.exit(0);

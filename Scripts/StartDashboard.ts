// Serve the interactive Foundry control panel: a clear 3-stage pipeline — (1) COLLECT DATA into
// Postgres, (2) TRAIN the model on that data (a subprocess, so it never blocks the dashboard and can
// run alongside collection), (3) CHAT to test the trained model. Reads store + token from .env
// (Bun): Postgres when DATABASE_URL is set. The trained checkpoint lives in Postgres; after a Train
// run finishes it is hot-reloaded into the chat model with no restart.
//   bun run foundry:dashboard      # then open http://localhost:8090

import { StartDashboard, IngestDocuments, CreateGitHubRepoProvider, CreateLocalRepoProvider } from "../Foundry/FoundryBarrel.ts";
import type { LearnFn, WebProvider, RepoIngestInfo, LearnEvent, TrainFn, TrainSettings, TrainEvent, SourceInput } from "../Foundry/FoundryBarrel.ts";
import type { ChatStore, ChatMessage } from "../Foundry/ChatStore.ts";
import { PostgresChatStore } from "../Foundry/PostgresChatStore.ts";
import { InMemoryChatStore } from "../Foundry/InMemoryChatStore.ts";
import { ChatService } from "../Foundry/ChatService.ts";
import type { ChatStreamFn, ChatOpts } from "../Foundry/ChatService.ts";
import { DescribeModel } from "../Foundry/ModelInfo.ts";
import type { ModelInfo } from "../Foundry/ModelInfo.ts";
import { PostgresCheckpointStore } from "../Foundry/PostgresCheckpointStore.ts";
import type { CheckpointSummary } from "../Foundry/PostgresCheckpointStore.ts";
import { LoadRunnableModel, LoadRunnableModelFrom } from "../Brain/Checkpoint/LoadRunnableModel.ts";
import type { RunnableModel } from "../Brain/Checkpoint/LoadRunnableModel.ts";
import { GuardedGenerateStream } from "../Brain/Safety/GuardedGenerate.ts";
import { RenderMessages } from "../Brain/Serving/RenderChat.ts";
import { ChatSession } from "../Brain/Serving/ChatSession.ts";
import { RunAgent } from "../Brain/Serving/AgentLoop.ts";
import type { AgentStep } from "../Brain/Serving/AgentLoop.ts";
import { FormatTrace } from "../Brain/Serving/ReasoningTrace.ts";
import { BuildAgentTooling, RenderToolManifest } from "../Brain/Serving/Tools/ToolsBarrel.ts";
import { ChatTokens } from "../Brain/Sft/ChatTemplate.ts";
import { Generate } from "../Brain/Sampling/Generate.ts";
import { DefaultSampling } from "../Brain/Sampling/Sampler.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";
import { ResolveStore, GitHubToken } from "./FoundryEnv.ts";
import { ReadArg } from "./ScriptArgs.ts";
import { existsSync } from "node:fs";

const { Store, Kind } = ResolveStore();
const DbUrl = process.env["DATABASE_URL"];
const CkptName = process.env["FOUNDRY_CHECKPOINT_NAME"] ?? "Seed";

// Mutable model state — swapped in place when a training run finishes or the user picks a saved model.
type ModelState = { Runnable: RunnableModel | null; Info: ModelInfo | null; Source: string; Name: string };
const ModelHolder: ModelState = { Runnable: null, Info: null, Source: "none", Name: CkptName };

// Load a checkpoint (by name) as the live chat model: Postgres first (durable/synced), then a file
// fallback for the default name only.
async function ReloadModel(Name: string = ModelHolder.Name): Promise<void> {
  ModelHolder.Name = Name;
  if (DbUrl !== undefined && DbUrl !== "") {
    try {
      const CkptStore = new PostgresCheckpointStore(DbUrl);
      const Ckpt = await CkptStore.Load(Name);
      await CkptStore.Close();
      if (Ckpt !== null) {
        ModelHolder.Runnable = LoadRunnableModelFrom(Ckpt);
        ModelHolder.Info = DescribeModel(ModelHolder.Runnable.Model);
        ModelHolder.Source = `postgres:${Name}`;
        return;
      }
    } catch {
      // fall through to file
    }
  }
  const FilePath = process.env["FOUNDRY_CHECKPOINT"] ?? ReadArg("--Checkpoint=", "Checkpoints/Foundry.ckpt");
  if (existsSync(FilePath)) {
    ModelHolder.Runnable = LoadRunnableModel(FilePath);
    ModelHolder.Info = DescribeModel(ModelHolder.Runnable.Model);
    ModelHolder.Source = `file:${FilePath}`;
    return;
  }
  ModelHolder.Runnable = null;
  ModelHolder.Info = null;
  ModelHolder.Source = "none";
}

// List saved checkpoints (the chat-model picker).
async function ListCheckpoints(): Promise<CheckpointSummary[]> {
  if (DbUrl === undefined || DbUrl === "") return [];
  try {
    const CkptStore = new PostgresCheckpointStore(DbUrl);
    const List = await CkptStore.List();
    await CkptStore.Close();
    return List;
  } catch {
    return [];
  }
}

// Chat generation reads the CURRENT model from the holder, so a post-training reload takes effect
// immediately. Logs each inference to the console so testing the model in the dashboard is observable.
let Counter = 0;

// Chat-format (SFT) models serve through the AGENT LOOP: the model can emit tool calls and thinking,
// tools run against the config-gated registry, and the reasoning trace (thinking -> tool+result ->
// answer) is logged to the console — the development lens. Dormant until a chat checkpoint is trained
// (LoadRunnableModel sets Chat from the checkpoint's Meta.Format); base models keep the plain path.
async function ServeChatAgent(Loaded: RunnableModel, Messages: ChatMessage[], Opts: ChatOpts, OnDelta: (Delta: string) => void): Promise<string> {
  const { Model, Tokenizer, Config } = Loaded;
  const Tooling = BuildAgentTooling(Config);
  const Session = new ChatSession("You are Shahd.\n\n" + RenderToolManifest(Tooling.Registry.List()));
  for (const M of Messages) {
    if (M.Role === "assistant") Session.AddAssistant(M.Content);
    else Session.AddUser(M.Content);
  }
  Tooling.Context.Session = Session;
  const Rng = new SeededRng(Config.Training.Seed + Counter++);
  // One agent turn = the model continues the chat prompt up to <|endofturn|>, decoded to text.
  const Gen = (Prompt: string): string => {
    const Ids = Tokenizer.Encode(Prompt);
    const Out = Generate(Model, Ids, Opts.MaxTokens, { ...DefaultSampling, Temperature: Opts.Temperature }, Rng);
    const Text = Tokenizer.Decode(Out.slice(Ids.length));
    const End = Text.indexOf(ChatTokens.EndOfTurn);
    return End >= 0 ? Text.slice(0, End) : Text;
  };
  const Steps: AgentStep[] = [];
  const Result = await RunAgent(Session, Gen, Tooling.Registry, Tooling.MaxSteps, Tooling.Context, (Step) => {
    Steps.push(Step);
    console.log(`[chat] step ${Step.Index}${Step.Call ? ` tool=${Step.Call.Name}` : ""}${Step.Terminal ? " (final)" : ""}`);
  });
  console.log(`[chat] reasoning trace:\n${FormatTrace(Steps)}`);
  OnDelta(Result.FinalText);
  return Result.FinalText;
}

const Stream: ChatStreamFn = async (Messages, Opts, OnDelta) => {
  const Loaded = ModelHolder.Runnable;
  if (Loaded === null) throw new Error("no trained model yet — collect data, then press Train Model");
  if (Loaded.Chat) return ServeChatAgent(Loaded, Messages, Opts, OnDelta); // SFT/chat model -> agent + trace
  const { Model, Tokenizer, Config } = Loaded;
  const Prompt = RenderMessages(Messages.map((M) => ({ role: M.Role, content: M.Content })));
  const LastMsg = (Messages[Messages.length - 1]?.Content ?? "").slice(0, 70).replace(/\s+/g, " ");
  const T0 = Date.now();
  let Chars = 0;
  let Chunks = 0;
  console.log(`[chat] ▶ turns=${Messages.length} temp=${Opts.Temperature} maxTok=${Opts.MaxTokens} · "${LastMsg}"`);
  const Rng = new SeededRng(Config.Training.Seed + Counter++);
  const Reply = await GuardedGenerateStream(
    Model, Tokenizer, Prompt, Opts.MaxTokens, { ...DefaultSampling, Temperature: Opts.Temperature }, Rng, Config,
    (Delta) => {
      Chars += Delta.length;
      Chunks++;
      OnDelta(Delta);
    },
    Opts.ShouldStop,
  );
  const Ms = Date.now() - T0;
  console.log(`[chat] ✓ ${Chars} chars / ${Chunks} tokens in ${Ms}ms (${Ms > 0 ? Math.round(Chars / (Ms / 1000)) : 0} ch/s) · "${Reply.slice(0, 70).replace(/\s+/g, " ")}"`);
  return Reply;
};

const Chats: ChatStore = DbUrl !== undefined && DbUrl !== "" ? new PostgresChatStore(DbUrl) : new InMemoryChatStore();
const Chat = new ChatService(Chats, Stream);

// STAGE 1 — Collect data (the provider-backed Learn runner).
const Learn: LearnFn = async (Settings, OnEvent, Signal) => {
  const Learned = new Set(Settings.SkipLearned ? await Store.Sources() : []);
  const Skip = (Repo: string): boolean => Learned.has(Repo);
  const OnRepo = (Info: RepoIngestInfo): void => {
    const Event: LearnEvent = { kind: "repo", repo: Info.Repo, level: Info.Assessment.Level, files: Info.Assessment.FileCount, bytes: Info.Assessment.TotalBytes, ingested: Info.Ingested, reason: Info.Reason ?? null };
    OnEvent(Event);
  };
  // INCREMENTAL: store each repo the moment it is downloaded (before the next), so a crash keeps what
  // is already stored, memory stays low, and the log reflects real storage.
  const IngestedAt = new Date().toISOString();
  let TotalIngested = 0;
  const OnRepoReady = async (Source: string, Files: SourceInput[]): Promise<void> => {
    OnEvent({ kind: "scanning", label: "storing " + Source + " (" + Files.length + " files)…" });
    const Stride = Math.max(1, Math.floor(Files.length / 50)); // ~50 progress updates/repo
    const Stats = await IngestDocuments(Files, Store, IngestedAt, 256, (Done, Total) => {
      if (Done % Stride === 0 || Done === Total) OnEvent({ kind: "repo-progress", repo: Source, filesDone: Done, filesTotal: Total });
    });
    TotalIngested += Stats.Ingested;
  };
  // Fired before each (slow) download: show a "working" status, and stop cleanly at this boundary if
  // the user pressed Stop — already-stored repos stay (collect is incremental), so nothing is lost.
  const OnRepoStart = (Repo: string): void => {
    if (Signal?.aborted === true) throw new Error("stopped by user");
    OnEvent({ kind: "scanning", label: "downloading " + Repo + "…" });
  };
  const Providers: WebProvider[] = [];
  if (Settings.Source !== "local") {
    Providers.push(CreateGitHubRepoProvider({ Token: GitHubToken(), MinLevel: Settings.MinLevel, MaxFilesPerRepo: Settings.MaxFilesPerRepo, MaxBytesPerRepo: Settings.MaxBytesPerRepo, MaxContentBytesPerRepo: Settings.MaxContentBytes, SkipRepo: Skip, OnRepoStart, OnRepo, OnRepoReady }));
  }
  if (Settings.Source !== "github") {
    Providers.push(CreateLocalRepoProvider({ Roots: Settings.Repos, MinLevel: Settings.MinLevel, MaxFiles: Settings.MaxFilesPerRepo, MaxBytes: Settings.MaxBytesPerRepo, MaxContentBytes: Settings.MaxContentBytes, SkipRepo: Skip, OnRepoStart, OnRepo, OnRepoReady }));
  }
  OnEvent({ kind: "scanning", label: "searching for repos…" });
  console.log(`[learn] start: source=${Settings.Source} query="${Settings.Query}" maxRepos=${Settings.MaxRepos} minLevel=${Settings.MinLevel} skipLearned=${Settings.SkipLearned} (already know ${Learned.size} repos)`);
  for (const Provider of Providers) {
    try {
      await Provider.Fetch(Settings.Query, Settings.MaxRepos); // repos stream into OnRepoReady as they arrive
    } catch (Caught) {
      // A provider failure is non-fatal — what was already stored stays — but it is NEVER swallowed:
      // the user's Stop surfaces as a clean note, any real error is logged AND shown in the dashboard.
      const Message = (Caught as Error).message;
      if (Message.includes("stopped by user")) {
        console.log(`[learn] stopped by user during ${Provider.Name}`);
      } else {
        console.warn(`[learn] provider ${Provider.Name} failed: ${Message}`);
        OnEvent({ kind: "error", message: `${Provider.Name}: ${Message}` });
      }
    }
  }
  console.log(`[learn] done: ingested ${TotalIngested} files this run`);
  OnEvent({ kind: "done", ingested: TotalIngested });
};

// STAGE 2 — Train the model on the collected corpus (subprocess: never blocks the dashboard, can run
// alongside collection). Parses the trainer's stdout into progress/info events; saves to Postgres.
const ParseTrainLine = (Line: string, Settings: TrainSettings, OnEvent: (Event: TrainEvent) => void): void => {
  if (Line.startsWith("{")) {
    try {
      const J = JSON.parse(Line) as { Step?: number; TrainLoss?: number; ValLoss?: number; ElapsedMs?: number };
      if (typeof J.Step === "number") {
        OnEvent({
          kind: "train-progress",
          step: J.Step,
          steps: Settings.Steps,
          trainLoss: J.TrainLoss ?? 0,
          valLoss: typeof J.ValLoss === "number" ? J.ValLoss : undefined,
          elapsedMs: typeof J.ElapsedMs === "number" ? J.ElapsedMs : undefined,
        });
        return;
      }
    } catch {
      // not a progress line
    }
  }
  OnEvent({ kind: "train-info", text: Line });
};

const Train: TrainFn = async (Settings, OnEvent, Signal) => {
  // pretrain -> base model (TrainOnFoundry); chat -> SFT chat model (TrainSftChat). Both workers emit
  // the same {Step,TrainLoss,ElapsedMs} progress lines, so ParseTrainLine handles either identically.
  const Common = [
    `--Name=${Settings.Name}`, `--Steps=${Settings.Steps}`, `--Merges=${Settings.Merges}`,
    `--EmbedDim=${Settings.EmbedDim}`, `--Layers=${Settings.NumLayers}`, `--Heads=${Settings.NumHeads}`,
    `--Block=${Settings.BlockSize}`, `--Batch=${Settings.BatchSize}`,
  ];
  const Args = Settings.Kind === "chat"
    ? ["bun", "run", "Scripts/TrainSftChat.ts", ...Common]
    : ["bun", "run", "Scripts/TrainOnFoundry.ts", ...Common, `--CorpusMb=${Settings.CorpusMb}`];
  console.log(`[train] ${Settings.Kind} "${Settings.Name}": ${Settings.Steps} steps`);
  const Proc = Bun.spawn(Args, { stdout: "pipe", stderr: "pipe", env: { ...process.env } });
  const OnAbort = (): void => {
    try {
      Proc.kill();
    } catch {
      // already exited
    }
  };
  Signal?.addEventListener("abort", OnAbort);
  const Decoder = new TextDecoder();
  let Buf = "";
  const Reader = Proc.stdout.getReader();
  for (;;) {
    const { done: Done, value: Value } = await Reader.read();
    if (Done) break;
    if (Value !== undefined) {
      Buf += Decoder.decode(Value, { stream: true });
      let Idx = Buf.indexOf("\n");
      while (Idx >= 0) {
        const Line = Buf.slice(0, Idx).trim();
        Buf = Buf.slice(Idx + 1);
        if (Line !== "") ParseTrainLine(Line, Settings, OnEvent);
        Idx = Buf.indexOf("\n");
      }
    }
  }
  const Code = await Proc.exited;
  Signal?.removeEventListener("abort", OnAbort);
  if (Signal?.aborted === true) OnEvent({ kind: "train-error", message: "stopped by user — last saved checkpoint is kept; press Train to resume" });
  else if (Code === 0) OnEvent({ kind: "train-done", savedTo: `postgres:${CkptName}` });
  else OnEvent({ kind: "train-error", message: `trainer exited with code ${Code} (see server console)` });
};

await ReloadModel("Seed");

const Port = Number(ReadArg("--Port=", "8090"));
StartDashboard(Store, Port, Learn, {
  Chat,
  GetModel: () => ModelHolder.Info,
  GetModelName: () => ModelHolder.Name,
  Train,
  OnTrained: (Name: string) => ReloadModel(Name),
  Checkpoints: ListCheckpoints,
  LoadModel: (Name: string) => ReloadModel(Name),
});
console.log(`Foundry control panel: http://localhost:${Port}  (store=${Kind}, ${await Store.Count()} docs, GitHub token: ${GitHubToken() ? "yes" : "no"})`);
console.log(`  chat model: ${ModelHolder.Source} (memory: ${DbUrl ? "postgres" : "in-memory"})`);

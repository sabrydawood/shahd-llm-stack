// Serve the interactive Foundry control panel: a clear 3-stage pipeline — (1) COLLECT DATA into
// Postgres, (2) TRAIN the model on that data (a subprocess, so it never blocks the dashboard and can
// run alongside collection), (3) CHAT to test the trained model. Reads store + token from .env
// (Bun): Postgres when DATABASE_URL is set. The trained checkpoint lives in Postgres; after a Train
// run finishes it is hot-reloaded into the chat model with no restart.
//   bun run foundry:dashboard      # then open http://localhost:8090

import { StartDashboard, IngestDocuments, CreateGitHubRepoProvider, CreateLocalRepoProvider, CreateLocalFolderProvider, CreateOasstProvider, CreateWikipediaProvider, CreateGsmProvider, CreateHfParquetProvider, WikiDumpSource, StackExchangeSource, Oasst2Url, InMemoryDocumentStore, InMemoryCollectionStateStore, ComputeExhausted } from "../Foundry/FoundryBarrel.ts";
import type { CollectionStateStore, CollectionState } from "../Foundry/FoundryBarrel.ts";
import type { LearnFn, WebProvider, RepoIngestInfo, LearnEvent, TrainFn, TrainSettings, TrainEvent, SourceInput, DataKind, DocumentStore } from "../Foundry/FoundryBarrel.ts";
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
import { SpecialTokenizer } from "../Brain/Tokenizer/SpecialTokenizer.ts";
import { RunAgent } from "../Brain/Serving/AgentLoop.ts";
import type { AgentStep } from "../Brain/Serving/AgentLoop.ts";
import { FormatTrace, BuildTrace } from "../Brain/Serving/ReasoningTrace.ts";
import { BuildAgentTooling } from "../Brain/Serving/Tools/ToolsBarrel.ts";
import { OwnedSystemPrompt } from "../Brain/Sft/OwnedSftData.ts";
import { ChatTokens } from "../Brain/Sft/ChatTemplate.ts";
import { ToolTokens } from "../Brain/Serving/ToolProtocol.ts";
import { ExtractAnswer, NormalizeAnswer, MajorityVote } from "../Brain/Reasoning/ReasoningBarrel.ts";
import { GenerateAsync } from "../Brain/Sampling/Generate.ts";
import { DefaultSampling } from "../Brain/Sampling/Sampler.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";
import { ResolveFoundryStores, GitHubToken } from "./FoundryEnv.ts";
import { ReadArg } from "./ScriptArgs.ts";
import { existsSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const DbUrl = process.env["DATABASE_URL"];
// ONE shared checkpoint-store pool for the whole dashboard lifetime — reused by reload/list/delete
// instead of opening (and, on error paths, LEAKING) a fresh postgres pool on every call.
const SharedCkptStore = DbUrl !== undefined && DbUrl !== "" ? new PostgresCheckpointStore(DbUrl) : null;
// Per-kind stores (Postgres): collection routes each source to its own kind table, so data types stay
// separated. Without a database, fall back to one in-memory store (no kind separation).
const Stores = DbUrl !== undefined && DbUrl !== "" ? ResolveFoundryStores() : null;
const FallbackStore = new InMemoryDocumentStore();
const KindStore = (Kind: DataKind): DocumentStore => (Stores !== null ? Stores.Kind(Kind) : FallbackStore);
// Collection ledger (progress/exhausted per source) — on the shared Postgres pool, or in-memory without a DB.
const FallbackCollState = new InMemoryCollectionStateStore();
const CollState = (): CollectionStateStore => (Stores !== null ? Stores.CollectionState() : FallbackCollState);
const InspectStore = KindStore("code"); // the dashboard's inspection/stats view (code is the bulk)
const CkptName = process.env["FOUNDRY_CHECKPOINT_NAME"] ?? "Seed";

// Mutable model state — swapped in place when a training run finishes or the user picks a saved model.
type ModelState = { Runnable: RunnableModel | null; Info: ModelInfo | null; Source: string; Name: string };
const ModelHolder: ModelState = { Runnable: null, Info: null, Source: "none", Name: CkptName };

// Load a checkpoint (by name) as the live chat model: Postgres first (durable/synced), then a file
// fallback for the default name only.
async function ReloadModel(Name: string = ModelHolder.Name): Promise<void> {
  ModelHolder.Name = Name;
  if (SharedCkptStore !== null) {
    try {
      const Ckpt = await SharedCkptStore.Load(Name);
      if (Ckpt !== null) {
        ModelHolder.Runnable = LoadRunnableModelFrom(Ckpt);
        ModelHolder.Info = DescribeModel(ModelHolder.Runnable.Model);
        ModelHolder.Source = `postgres:${Name}`;
        return;
      }
    } catch (Caught) {
      console.warn(`[dashboard] checkpoint load failed for "${Name}": ${(Caught as Error).message} - falling back to file`);
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
  if (SharedCkptStore === null) return [];
  try {
    return await SharedCkptStore.List();
  } catch (Caught) {
    console.warn(`[dashboard] checkpoint list failed: ${(Caught as Error).message}`);
    return [];
  }
}

// Chat generation reads the CURRENT model from the holder, so a post-training reload takes effect
// immediately. Logs each inference to the console so testing the model in the dashboard is observable.
let Counter = 0;
// Self-consistency: with N>1 the chat model answers a turn N times (different sampling seeds) and the
// majority-voted answer wins — test-time compute that lifts reasoning accuracy with NO extra data or
// training. Off by default (N=1); set SHAHD_SELF_CONSISTENCY=5 to enable. Needs Temperature>0 to diverge.
const SelfConsistencySamples = Math.max(1, Math.floor(Number(process.env["SHAHD_SELF_CONSISTENCY"] ?? "1")) || 1);

// Chat-format (SFT) models serve through the AGENT LOOP: the model can emit tool calls and thinking,
// tools run against the config-gated registry, and the reasoning trace (thinking -> tool+result ->
// answer) is logged to the console — the development lens. Dormant until a chat checkpoint is trained
// (LoadRunnableModel sets Chat from the checkpoint's Meta.Format); base models keep the plain path.
async function ServeChatAgent(Loaded: RunnableModel, Messages: ChatMessage[], Opts: ChatOpts, OnDelta: (Delta: string) => void): Promise<string> {
  const { Model, Tokenizer, Config } = Loaded;
  const Tooling = BuildAgentTooling(Config);
  // TRAIN/SERVE PROMPT PARITY: the SFT tool/persona/arithmetic conversations were trained with the
  // short OwnedSystemPrompt and NO tool manifest — the tool-call FORMAT lives in the weights. The
  // previous prompt here (thinking preamble + a 20-tool rendered manifest) was hundreds of tokens the
  // model never saw in training and overflowed ctx=256 outright, so the model answered arithmetic
  // questions with language names (verified live). Tiny models do not survive prompt-distribution
  // drift; serve them EXACTLY what they were trained on.
  const SystemPrompt = OwnedSystemPrompt;

  // LIVE streaming (the request->response wall): with one run (the default), think text streams into
  // the trace as it is generated, each tool call surfaces the moment it executes, and the answer
  // streams into the bubble — the user watches the reasoning happen instead of waiting minutes for
  // one burst. With self-consistency N>1 the runs are candidates (losers are discarded), so nothing
  // can stream; those keep the buffered behavior.
  const Live = SelfConsistencySamples <= 1;
  let ThinkSeg = 0; // one segment per generation pass => the UI groups think spans per agent step

  // One full agent run over a FRESH session (so N self-consistency runs don't contaminate each other).
  const RunOnce = async (): Promise<{ Text: string; Steps: AgentStep[] }> => {
    const Session = new ChatSession(SystemPrompt);
    for (const M of Messages) {
      if (M.Role === "assistant") Session.AddAssistant(M.Content);
      else Session.AddUser(M.Content);
    }
    Tooling.Context.Session = Session;
    const Rng = new SeededRng(Config.Training.Seed + Counter++); // distinct seed per run => diverse samples
    // One agent turn = the model continues the chat prompt up to <|endofturn|>, decoded to text.
    // A chat model uses a SpecialTokenizer: render the prompt directly to ids (untrusted content
    // base-encoded) so a user/tool string can't smuggle a real control token. Only fall back to
    // string-encode for the unexpected non-special tokenizer case.
    //
    // ASYNC + EARLY-STOP, both load-bearing: the sync Generate here blocked the dashboard's whole
    // event loop for MaxTokens per agent step (x MaxSteps steps, x N self-consistency runs) — the UI
    // literally froze for the duration of every chat reply. GenerateAsync yields a macrotask per
    // token, honors the client-disconnect ShouldStop, and the <|endofturn|> token id ends the run
    // the moment the model closes its turn instead of always paying the full MaxTokens.
    const EndOfTurnId = Tokenizer instanceof SpecialTokenizer ? Tokenizer.Id(ChatTokens.EndOfTurn) : null;
    const Gen = async (Session2: ChatSession): Promise<string> => {
      const Ids = Tokenizer instanceof SpecialTokenizer ? Session2.RenderPromptIds(Tokenizer) : Tokenizer.Encode(Session2.RenderPrompt());
      let SawEndOfTurn = false;
      // Live split of the growing generation: <|think|>… streams to the trace, the text after
      // <|endthink|> streams to the answer bubble — but never a tool-call payload (that surfaces as a
      // structured tool line when it executes). Re-decoding the full slice each token keeps byte-BPE
      // UTF-8 assembly correct; a trailing replacement char (a split multi-byte char) is held back.
      const NewIds: number[] = [];
      const Seg = ThinkSeg++;
      let SentThink = 0;
      let SentAnswer = 0;
      const PushLive = (): void => {
        if (!Live) return;
        let T = Tokenizer.Decode(NewIds);
        const EndT = T.indexOf(ChatTokens.EndOfTurn);
        if (EndT >= 0) T = T.slice(0, EndT);
        if (T.endsWith("�")) T = T.slice(0, -1);
        let Think = "";
        let After = T;
        if (T.startsWith(ChatTokens.Think)) {
          const Close = T.indexOf(ChatTokens.EndThink);
          Think = Close >= 0 ? T.slice(ChatTokens.Think.length, Close) : T.slice(ChatTokens.Think.length);
          After = Close >= 0 ? T.slice(Close + ChatTokens.EndThink.length) : "";
        }
        const CallIdx = After.indexOf(ToolTokens.CallStart);
        const Visible = CallIdx >= 0 ? After.slice(0, CallIdx) : After;
        if (Think.length > SentThink) {
          Opts.OnThink?.(Think.slice(SentThink), Seg);
          SentThink = Think.length;
        }
        if (Visible.length > SentAnswer) {
          OnDelta(Visible.slice(SentAnswer));
          SentAnswer = Visible.length;
        }
      };
      const Out = await GenerateAsync(
        Model,
        Ids,
        Opts.MaxTokens,
        { ...DefaultSampling, Temperature: Opts.Temperature },
        Rng,
        (Id) => {
          if (EndOfTurnId !== null && Id === EndOfTurnId) SawEndOfTurn = true;
          else NewIds.push(Id);
          PushLive();
        },
        () => SawEndOfTurn || Opts.ShouldStop?.() === true,
      );
      const Text = Tokenizer.Decode(Out.slice(Ids.length));
      const End = Text.indexOf(ChatTokens.EndOfTurn);
      return End >= 0 ? Text.slice(0, End) : Text;
    };
    const Steps: AgentStep[] = [];
    const Result = await RunAgent(Session, Gen, Tooling.Registry, Tooling.MaxSteps, Tooling.Context, (Step) => {
      Steps.push(Step);
      // Surface each tool call/result the moment it happens — the live half of the trace; think and
      // answer text stream separately above. The final BuildTrace below remains authoritative.
      if (Live) for (const L of BuildTrace([Step])) if (L.Kind === "tool") Opts.OnToolStep?.(L);
    });
    return { Text: Result.FinalText, Steps };
  };

  let Best: { Text: string; Steps: AgentStep[] };
  if (SelfConsistencySamples <= 1) {
    Best = await RunOnce();
  } else {
    // Draw N runs, then majority-vote over their NORMALIZED final answers (so "42"/"42." count together);
    // the winning run's own text + trace is what we surface.
    const Runs: { Text: string; Steps: AgentStep[] }[] = [];
    for (let I = 0; I < SelfConsistencySamples; I++) Runs.push(await RunOnce());
    const Vote = MajorityVote(Runs.map((R) => ExtractAnswer(R.Text)), NormalizeAnswer);
    const WinnerKey = NormalizeAnswer(Vote.Winner);
    Best = Runs.find((R) => NormalizeAnswer(ExtractAnswer(R.Text)) === WinnerKey) ?? Runs[0]!;
    console.log(`[chat] self-consistency: ${SelfConsistencySamples} runs, winner "${Vote.Winner}" (${Vote.Count}/${Vote.Total}); tally ${JSON.stringify(Vote.Tally)}`);
  }

  console.log(`[chat] reasoning trace:\n${FormatTrace(Best.Steps)}`);
  Opts.OnTrace?.(BuildTrace(Best.Steps)); // surface the trace to the dashboard UI (the visible reasoning lens)
  if (!Live) OnDelta(Best.Text); // live runs already streamed the answer token-by-token
  return Best.Text;
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
// Parse a persisted parquet cursor ('{"Shard":N,"Offset":M}') into shard/offset, defaulting to the
// start on missing/garbage input so a corrupt ledger row can never crash a collect run.
function ParseShardCursor(Raw: string | undefined): { Shard: number; Offset: number } {
  try {
    const P = JSON.parse(Raw ?? "{}") as { Shard?: unknown; Offset?: unknown };
    const Shard = typeof P.Shard === "number" && P.Shard >= 0 ? Math.floor(P.Shard) : 0;
    const Offset = typeof P.Offset === "number" && P.Offset >= 0 ? Math.floor(P.Offset) : 0;
    return { Shard, Offset };
  } catch {
    return { Shard: 0, Offset: 0 };
  }
}

const Learn: LearnFn = async (Settings, OnEvent, Signal) => {
  // Route this run to the kind table for its source: oasst -> conversation, wikipedia -> knowledge,
  // gsm8k -> instruction (reasoning), github/local -> code. Everything collected this run lands in that
  // one physically-separate table.
  const SourceKind: DataKind =
    Settings.Source === "oasst" || Settings.Source === "oasst2" || Settings.Source === "stackexchange"
      ? "conversation"
      : Settings.Source === "wikipedia" || Settings.Source === "wikidump"
        ? "knowledge"
        : Settings.Source === "gsm8k"
          ? "instruction"
          : Settings.Source === "folder"
            ? Settings.Kind ?? "books"
            : "code";
  const CollectStore = KindStore(SourceKind);
  const Learned = new Set(Settings.SkipLearned ? await CollectStore.Sources() : []);
  const Skip = (Repo: string): boolean => Learned.has(Repo);
  // Collection ledger read (Phase 1/2): load this source's prior progress so a streaming parquet source
  // resumes from its persisted {Shard, Offset} cursor. Non-fatal — a read failure just starts fresh.
  const StateKey = `${Settings.Source}:${Settings.Query}`;
  let PrevState: CollectionState | null = null;
  try {
    PrevState = await CollState().Get(StateKey);
  } catch (Caught) {
    console.warn(`[learn] collection-state read failed (non-fatal): ${(Caught as Error).message}`);
  }
  let RunCursor = PrevState?.Cursor ?? "{}";
  const OnRepo = (Info: RepoIngestInfo): void => {
    const Event: LearnEvent = { kind: "repo", repo: Info.Repo, level: Info.Assessment.Level, files: Info.Assessment.FileCount, bytes: Info.Assessment.TotalBytes, ingested: Info.Ingested, reason: Info.Reason ?? null };
    OnEvent(Event);
  };
  // INCREMENTAL: store each repo the moment it is downloaded (before the next), so a crash keeps what
  // is already stored, memory stays low, and the log reflects real storage.
  const IngestedAt = new Date().toISOString();
  let TotalIngested = 0;
  let TotalNew = 0;
  let TotalDuplicate = 0;
  const OnRepoReady = async (Source: string, Files: SourceInput[]): Promise<void> => {
    OnEvent({ kind: "scanning", label: "storing " + Source + " (" + Files.length + " files)…" });
    const Stride = Math.max(1, Math.floor(Files.length / 50)); // ~50 progress updates/repo
    const Stats = await IngestDocuments(Files, CollectStore, IngestedAt, 256, (Done, Total) => {
      if (Done % Stride === 0 || Done === Total) OnEvent({ kind: "repo-progress", repo: Source, filesDone: Done, filesTotal: Total });
    });
    TotalIngested += Stats.Ingested;
    TotalNew += Stats.New;
    TotalDuplicate += Stats.Duplicate;
  };
  // Fired before each (slow) download: show a "working" status, and stop cleanly at this boundary if
  // the user pressed Stop — already-stored repos stay (collect is incremental), so nothing is lost.
  const OnRepoStart = (Repo: string): void => {
    if (Signal?.aborted === true) throw new Error("stopped by user");
    OnEvent({ kind: "scanning", label: "downloading " + Repo + "…" });
  };
  const Providers: WebProvider[] = [];
  // General/text sources: Query is the language filter, MaxRepos is the max items to pull.
  if (Settings.Source === "oasst") {
    Providers.push(CreateOasstProvider({ OnRepoStart, OnRepoReady }));
  } else if (Settings.Source === "oasst2") {
    Providers.push(CreateOasstProvider({ Url: Oasst2Url, OnRepoStart, OnRepoReady }));
  } else if (Settings.Source === "wikipedia") {
    Providers.push(CreateWikipediaProvider({ OnRepoStart, OnRepoReady }));
  } else if (Settings.Source === "gsm8k") {
    Providers.push(CreateGsmProvider({ OnRepoStart, OnRepoReady }));
  } else if (Settings.Source === "folder") {
    // Ingest every text file under the given folder(s) as the chosen kind — the on-disk path for a
    // downloaded corpus (e.g. all of Gutenberg). License defaults to public-domain (the books case).
    // NB: no MaxContentBytes override (the code cap is 512 KB — most books are bigger; the provider's
    // book-sized 20 MB default applies), and NO SkipRoot — per-folder skip would short-circuit the whole
    // folder on a second run (Source is the folder name), blocking the "add more books, re-collect"
    // workflow; content-hash dedup already skips books already stored, cheaply.
    Providers.push(
      CreateLocalFolderProvider({
        Roots: Settings.Repos,
        License: Settings.License,
        Lang: SourceKind === "books" ? "text" : `text-${SourceKind}`,
        OnRepoStart,
        OnRepoReady,
      }),
    );
  } else if (Settings.Source === "wikidump" || Settings.Source === "stackexchange") {
    // The two HF-parquet sources share the generic provider: resume from the persisted {Shard, Offset}
    // cursor and report the advanced cursor so it's saved for next run (the real use of the Phase-1 cursor).
    const Src = Settings.Source === "wikidump" ? WikiDumpSource : StackExchangeSource;
    const Cur = ParseShardCursor(PrevState?.Cursor);
    Providers.push(
      CreateHfParquetProvider(Src, {
        StartShard: Cur.Shard,
        StartOffset: Cur.Offset,
        MaxPerRun: Settings.MaxRepos,
        OnRepoStart,
        OnRepoReady,
        OnCursor: (Shard, Offset) => {
          RunCursor = JSON.stringify({ Shard, Offset });
        },
      }),
    );
  } else {
    if (Settings.Source !== "local") {
      Providers.push(CreateGitHubRepoProvider({ Token: GitHubToken(), MinLevel: Settings.MinLevel, MaxFilesPerRepo: Settings.MaxFilesPerRepo, MaxBytesPerRepo: Settings.MaxBytesPerRepo, MaxContentBytesPerRepo: Settings.MaxContentBytes, SkipRepo: Skip, OnRepoStart, OnRepo, OnRepoReady }));
    }
    if (Settings.Source !== "github") {
      Providers.push(CreateLocalRepoProvider({ Roots: Settings.Repos, MinLevel: Settings.MinLevel, MaxFiles: Settings.MaxFilesPerRepo, MaxBytes: Settings.MaxBytesPerRepo, MaxContentBytes: Settings.MaxContentBytes, SkipRepo: Skip, OnRepoStart, OnRepo, OnRepoReady }));
    }
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
  // Honest summary: distinguish NEW rows from content-hash duplicates. A re-collected bounded source
  // reads "0 new · N duplicate" (correct dedup) instead of a misleading "ingested N". Semantics tells
  // the user whether running again can grow the corpus (streaming) or not (bounded, already complete).
  const Semantics: "bounded" | "streaming" = Providers.some((P) => P.Semantics === "streaming") ? "streaming" : "bounded";

  // Collection ledger (Phase 1): record this source's lifetime progress durably. A bounded source that
  // produced 0 new but saw duplicates is confirmed fully collected -> Exhausted (the UI can say "complete"
  // instead of prompting another pointless run). Streaming sources are never exhausted. Non-fatal: a
  // ledger write failure must not fail the collect run.
  // Lifetime accumulates on the prior total (read once at the start). Cursor carries a streaming parquet
  // source's advanced {Shard, Offset} so the next run resumes; "{}" for sources without a cursor.
  const LifetimeCollected = (PrevState?.Collected ?? 0) + TotalNew;
  // Exhausted ONLY when a bounded source ran DRY (0 new, only dups) AND was not merely truncated by the
  // MaxRepos cap — see ComputeExhausted. Guards against the "you're stuck at this data" false signal.
  const Exhausted = ComputeExhausted(Semantics, TotalNew, TotalDuplicate, TotalIngested, Settings.MaxRepos);
  try {
    await CollState().Upsert({ SourceKey: StateKey, Kind: SourceKind, Cursor: RunCursor, Collected: LifetimeCollected, Exhausted, UpdatedAt: new Date().toISOString() });
  } catch (Caught) {
    console.warn(`[learn] collection-state write failed (non-fatal): ${(Caught as Error).message}`);
  }

  console.log(`[learn] done: ${TotalNew} new · ${TotalDuplicate} duplicate · ${TotalIngested} processed (${Semantics}${Exhausted ? ", exhausted" : ""}; lifetime ${LifetimeCollected})`);
  OnEvent({ kind: "done", ingested: TotalIngested, new: TotalNew, duplicate: TotalDuplicate, semantics: Semantics, collected: LifetimeCollected, exhausted: Exhausted });
};

// STAGE 2 — Train the model on the collected corpus (subprocess: never blocks the dashboard, can run
// alongside collection). Parses the trainer's stdout into progress/info events; saves to Postgres.
const ParseTrainLine = (Line: string, Settings: TrainSettings, OnEvent: (Event: TrainEvent) => void): void => {
  if (Line.startsWith("{")) {
    try {
      const J = JSON.parse(Line) as { Step?: number; TrainLoss?: number; ValLoss?: number; ElapsedMs?: number; StepMs?: number };
      if (typeof J.Step === "number") {
        OnEvent({
          kind: "train-progress",
          step: J.Step,
          steps: Settings.Steps,
          trainLoss: J.TrainLoss ?? 0,
          valLoss: typeof J.ValLoss === "number" ? J.ValLoss : undefined,
          elapsedMs: typeof J.ElapsedMs === "number" ? J.ElapsedMs : undefined,
          stepMs: typeof J.StepMs === "number" ? J.StepMs : undefined,
        });
        return;
      }
    } catch {
      // not a progress line
    }
  }
  OnEvent({ kind: "train-info", text: Line });
};

// Pump a byte stream into whole trimmed lines (blank lines dropped) — shared by the stdout and
// stderr readers below so both are consumed with identical buffering.
async function PumpLines(Stream: ReadableStream<Uint8Array<ArrayBuffer>>, OnLine: (Line: string) => void): Promise<void> {
  const Reader = Stream.getReader();
  const Decoder = new TextDecoder();
  let Buf = "";
  for (;;) {
    const { done: Done, value: Value } = await Reader.read();
    if (Done) break;
    if (Value !== undefined) {
      Buf += Decoder.decode(Value, { stream: true });
      let Idx = Buf.indexOf("\n");
      while (Idx >= 0) {
        const Line = Buf.slice(0, Idx).trim();
        Buf = Buf.slice(Idx + 1);
        if (Line !== "") OnLine(Line);
        Idx = Buf.indexOf("\n");
      }
    }
  }
}

const Train: TrainFn = async (Settings, OnEvent, Signal) => {
  // pretrain -> base model (TrainOnFoundry); chat -> SFT chat model (TrainSftChat). Both workers emit
  // the same {Step,TrainLoss,ElapsedMs} progress lines, so ParseTrainLine handles either identically.
  // Stop is GRACEFUL: the train loops are synchronous (stdin/IPC would starve), so the stop signal is
  // a file the child polls once per step — it saves a checkpoint at the EXACT current step and exits 0.
  const StopFile = join(tmpdir(), `shahd-train-stop-${process.pid}-${Date.now()}`);
  const Common = [
    `--Name=${Settings.Name}`, `--Steps=${Settings.Steps}`, `--Merges=${Settings.Merges}`,
    `--EmbedDim=${Settings.EmbedDim}`, `--Layers=${Settings.NumLayers}`, `--Heads=${Settings.NumHeads}`,
    `--Block=${Settings.BlockSize}`, `--Batch=${Settings.BatchSize}`, `--Workers=${Settings.Workers ?? 0}`,
    `--Precision=${Settings.Precision ?? "F64"}`, `--StopFile=${StopFile}`,
    ...(Settings.Resume ? ["--Resume"] : []), // continue/EXTEND an existing checkpoint of this name
  ];
  const Args = Settings.Kind === "chat"
    ? ["bun", "run", "Scripts/TrainSftChat.ts", ...Common, `--CodeSamples=${Settings.CodeSamples}`, `--ConvCount=${Settings.ConvCount}`, `--MultiTurn=${Settings.MultiTurn ?? 3000}`,
       // Warm start from a pretrained base (the pretrain→SFT bridge) — only when a base was picked.
       ...(Settings.From !== undefined && Settings.From !== "" ? [`--From=${Settings.From}`] : [])]
    : ["bun", "run", "Scripts/TrainOnFoundry.ts", ...Common, `--CorpusMb=${Settings.CorpusMb}`, `--KnowledgeMb=${Settings.KnowledgeMb}`];
  console.log(`[train] ${Settings.Kind} "${Settings.Name}": ${Settings.Steps} steps`);
  const Proc = Bun.spawn(Args, { stdout: "pipe", stderr: "pipe", env: { ...process.env } });
  let KillTimer: ReturnType<typeof setTimeout> | null = null;
  const OnAbort = (): void => {
    try {
      // Ask nicely first: the child sees the stop file at the next step boundary, saves the exact-step
      // checkpoint, and exits 0. Hard-kill only if it hasn't exited after the grace window (a save on
      // a big model takes a while — the periodic checkpoint still bounds the loss).
      writeFileSync(StopFile, "stop");
      OnEvent({ kind: "train-info", text: "stop requested — finishing the current step and saving a checkpoint at it…" });
      KillTimer = setTimeout(() => {
        try {
          Proc.kill();
        } catch {
          // already exited
        }
      }, 180_000);
    } catch {
      try {
        Proc.kill();
      } catch {
        // already exited
      }
    }
  };
  Signal?.addEventListener("abort", OnAbort);
  // Consume stdout (progress lines) and stderr (crash/error output) CONCURRENTLY — an unread pipe
  // fills its OS buffer and can stall the child, and a silently-dropped stderr hid the real error/
  // stack trace behind "see server console". Stderr lines are logged AND surfaced to the dashboard.
  await Promise.all([
    PumpLines(Proc.stdout, (Line) => ParseTrainLine(Line, Settings, OnEvent)),
    PumpLines(Proc.stderr, (Line) => {
      console.error(`[train:stderr] ${Line}`);
      OnEvent({ kind: "train-info", text: `[stderr] ${Line}` });
    }),
  ]);
  const Code = await Proc.exited;
  Signal?.removeEventListener("abort", OnAbort);
  if (KillTimer !== null) clearTimeout(KillTimer);
  rmSync(StopFile, { force: true });
  if (Signal?.aborted === true) {
    OnEvent(Code === 0
      ? { kind: "train-error", paused: true, message: "paused — checkpoint saved at the exact step you stopped at (and loaded into Chat); press Train (Resume) to continue from there" }
      : { kind: "train-error", message: "stopped — last periodic checkpoint is kept; press Train (Resume) to continue" });
  } else if (Code === 0) OnEvent({ kind: "train-done", savedTo: `postgres:${Settings.Name}` }); // the TRAINED model's name (CkptName is the startup-loaded one)
  else OnEvent({ kind: "train-error", message: `trainer exited with code ${Code} (see server console)` });
};

await ReloadModel("Seed");

const Port = Number(ReadArg("--Port=", "8090"));
StartDashboard(InspectStore, Port, Learn, {
  Chat,
  GetModel: () => ModelHolder.Info,
  GetModelName: () => ModelHolder.Name,
  Train,
  OnTrained: (Name: string) => ReloadModel(Name),
  Checkpoints: ListCheckpoints,
  LoadModel: (Name: string) => ReloadModel(Name),
  KindStats: () => (Stores !== null ? Stores.Stats() : Promise.resolve([])),
  Collection: () => CollState().All(),
  // Data browser + cleanup: kind-aware, so it reaches every documents_<kind> table (Postgres) or the
  // single fallback store (in-memory). Each op resolves the right store and delegates to it.
  Browse: (Kind, Filter, Offset, Limit) => KindStore(Kind).Query(Filter, Offset, Limit),
  Facets: (Kind) => KindStore(Kind).Stats(),
  DocContent: (Kind, Id) => KindStore(Kind).DocumentById(Id),
  DeleteDoc: (Kind, Id) => KindStore(Kind).DeleteById(Id),
  DeleteMatching: (Kind, Filter) => KindStore(Kind).DeleteMatching(Filter),
  DeleteCheckpoint: async (Name: string): Promise<void> => {
    if (SharedCkptStore === null) return;
    try {
      await SharedCkptStore.Delete(Name);
    } catch (Caught) {
      console.warn(`[dashboard] delete checkpoint "${Name}" failed: ${(Caught as Error).message}`);
    }
  },
});
console.log(`Foundry control panel: http://localhost:${Port}  (store=${Stores !== null ? "postgres/per-kind" : "memory"}, ${await InspectStore.Count()} code docs, GitHub token: ${GitHubToken() ? "yes" : "no"})`);
console.log(`  chat model: ${ModelHolder.Source} (memory: ${DbUrl ? "postgres" : "in-memory"})`);

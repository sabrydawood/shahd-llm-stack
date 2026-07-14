// Serve the interactive Foundry control panel with a live "Learn" runner + realtime WebSocket + a
// memory-backed chat. Reads the store + token from the environment (Bun loads .env): Postgres when
// DATABASE_URL is set. Press "Learn" in the UI to ingest whole repos (public GitHub and/or our own),
// skipping already-learned ones; open /chat to talk to the loaded checkpoint.
//   bun run foundry:dashboard      # then open http://localhost:8090

import { StartDashboard, IngestFromWeb, CreateGitHubRepoProvider, CreateLocalRepoProvider } from "../Foundry/FoundryBarrel.ts";
import type { LearnFn, WebProvider, RepoIngestInfo, LearnEvent } from "../Foundry/FoundryBarrel.ts";
import type { ChatStore } from "../Foundry/ChatStore.ts";
import { PostgresChatStore } from "../Foundry/PostgresChatStore.ts";
import { InMemoryChatStore } from "../Foundry/InMemoryChatStore.ts";
import { ChatService } from "../Foundry/ChatService.ts";
import type { ChatStreamFn } from "../Foundry/ChatService.ts";
import { DescribeModel } from "../Foundry/ModelInfo.ts";
import type { ModelInfo } from "../Foundry/ModelInfo.ts";
import { LoadRunnableModel } from "../Brain/Checkpoint/LoadRunnableModel.ts";
import { GuardedGenerateStream } from "../Brain/Safety/GuardedGenerate.ts";
import { RenderMessages } from "../Brain/Serving/RenderChat.ts";
import { DefaultSampling } from "../Brain/Sampling/Sampler.ts";
import { SeededRng } from "../Brain/Random/SeededRng.ts";
import { ResolveStore, GitHubToken } from "./FoundryEnv.ts";
import { ReadArg } from "./ScriptArgs.ts";
import { existsSync } from "node:fs";

const { Store, Kind } = ResolveStore();

// Best-effort: load a checkpoint so the /chat page works (streaming + persistent memory). If none
// exists (or loading fails), the dashboard still serves and chat reports "no model loaded".
function LoadChat(): { Chat?: ChatService; Model: ModelInfo | null; Note: string } {
  const Path = process.env["FOUNDRY_CHECKPOINT"] ?? ReadArg("--Checkpoint=", "Checkpoints/Corpus.ckpt");
  if (!existsSync(Path)) return { Model: null, Note: `no checkpoint at ${Path} (chat disabled)` };
  try {
    const { Model, Tokenizer, Config } = LoadRunnableModel(Path);
    let Counter = 0;
    const Stream: ChatStreamFn = (Messages, Opts, OnDelta) => {
      const Prompt = RenderMessages(Messages.map((M) => ({ role: M.Role, content: M.Content })));
      const Rng = new SeededRng(Config.Training.Seed + Counter++);
      return GuardedGenerateStream(Model, Tokenizer, Prompt, Opts.MaxTokens, { ...DefaultSampling, Temperature: Opts.Temperature }, Rng, Config, OnDelta, Opts.ShouldStop);
    };
    // Chat memory in Postgres (synced with the corpus, durable) when DATABASE_URL is set; else in-memory.
    const DbUrl = process.env["DATABASE_URL"];
    const Chats: ChatStore = DbUrl !== undefined && DbUrl !== "" ? new PostgresChatStore(DbUrl) : new InMemoryChatStore();
    return { Chat: new ChatService(Chats, Stream), Model: DescribeModel(Model), Note: `chat model: ${Path} (memory: ${DbUrl ? "postgres" : "in-memory"})` };
  } catch (Caught) {
    return { Model: null, Note: `chat disabled: ${(Caught as Error).message}` };
  }
}

// The real provider-backed Learn runner injected into the dashboard.
const Learn: LearnFn = async (Settings, OnEvent) => {
  const Learned = new Set(Settings.SkipLearned ? await Store.Sources() : []);
  const Skip = (Repo: string): boolean => Learned.has(Repo);
  const OnRepo = (Info: RepoIngestInfo): void => {
    const Event: LearnEvent = { kind: "repo", repo: Info.Repo, level: Info.Assessment.Level, files: Info.Assessment.FileCount, bytes: Info.Assessment.TotalBytes, ingested: Info.Ingested, reason: Info.Reason ?? null };
    OnEvent(Event);
  };
  const Providers: WebProvider[] = [];
  if (Settings.Source !== "local") {
    Providers.push(CreateGitHubRepoProvider({ Token: GitHubToken(), MinLevel: Settings.MinLevel, MaxFilesPerRepo: Settings.MaxFilesPerRepo, MaxBytesPerRepo: Settings.MaxBytesPerRepo, MaxContentBytesPerRepo: Settings.MaxContentBytes, SkipRepo: Skip, OnRepo }));
  }
  if (Settings.Source !== "github") {
    Providers.push(CreateLocalRepoProvider({ Roots: Settings.Repos, MinLevel: Settings.MinLevel, MaxFiles: Settings.MaxFilesPerRepo, MaxBytes: Settings.MaxBytesPerRepo, MaxContentBytes: Settings.MaxContentBytes, SkipRepo: Skip, OnRepo }));
  }
  // Throttle to ~50 updates per repo (one every Stride files, plus the final one) so a big repo does
  // not emit thousands of progress events.
  const OnProgress = (Repo: string, Done: number, Total: number): void => {
    const Stride = Math.max(1, Math.floor(Total / 50));
    if (Done % Stride === 0 || Done === Total) OnEvent({ kind: "repo-progress", repo: Repo, filesDone: Done, filesTotal: Total });
  };
  const Stats = await IngestFromWeb(Providers, [Settings.Query], Store, new Date().toISOString(), Settings.MaxRepos, 256, OnProgress);
  OnEvent({ kind: "done", ingested: Stats.Ingested });
};

const Port = Number(ReadArg("--Port=", "8090"));
const { Chat, Model, Note } = LoadChat();
StartDashboard(Store, Port, Learn, { Chat, Model });
console.log(`Foundry control panel: http://localhost:${Port}  (store=${Kind}, ${await Store.Count()} docs, GitHub token: ${GitHubToken() ? "yes" : "no"})`);
console.log(`  ${Note}`);

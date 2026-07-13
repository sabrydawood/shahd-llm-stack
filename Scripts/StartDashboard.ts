// Serve the interactive Foundry control panel with a live "Learn" runner. Reads the store + token
// from the environment (Bun loads .env): Postgres when DATABASE_URL is set. Press "Learn" in the UI
// to ingest whole repos (public GitHub and/or our own), skipping already-learned ones.
//   bun run foundry:dashboard      # then open http://localhost:8090

import { StartDashboard, IngestFromWeb, CreateGitHubRepoProvider, CreateLocalRepoProvider } from "../Foundry/FoundryBarrel.ts";
import type { LearnFn, WebProvider, RepoIngestInfo, LearnEvent } from "../Foundry/FoundryBarrel.ts";
import { ResolveStore, GitHubToken } from "./FoundryEnv.ts";
import { ReadArg } from "./ScriptArgs.ts";

const { Store, Kind } = ResolveStore();

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
    Providers.push(CreateGitHubRepoProvider({ Token: GitHubToken(), MinLevel: Settings.MinLevel, MaxFilesPerRepo: Settings.MaxFilesPerRepo, MaxBytesPerRepo: Settings.MaxBytesPerRepo, SkipRepo: Skip, OnRepo }));
  }
  if (Settings.Source !== "github") {
    Providers.push(CreateLocalRepoProvider({ Roots: Settings.Repos, MinLevel: Settings.MinLevel, MaxFiles: Settings.MaxFilesPerRepo, MaxBytes: Settings.MaxBytesPerRepo, SkipRepo: Skip, OnRepo }));
  }
  const Stats = await IngestFromWeb(Providers, [Settings.Query], Store, new Date().toISOString(), Settings.MaxRepos);
  OnEvent({ kind: "done", ingested: Stats.Ingested });
};

const Port = Number(ReadArg("--Port=", "8090"));
StartDashboard(Store, Port, Learn);
console.log(`Foundry control panel: http://localhost:${Port}  (store=${Kind}, ${await Store.Count()} docs, GitHub token: ${GitHubToken() ? "yes" : "no"})`);

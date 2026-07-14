// Types for the interactive Foundry dashboard (M9): the settings a "Learn" run accepts and the
// progress events it streams. LearnFn is injected into the dashboard handler so the handler stays
// testable (a mock in tests, the real provider-backed runner at serving time).

import type { RepoLevel } from "./RepoQuality.ts";

export type LearnSettings = {
  Source: "github" | "local" | "both";
  Query: string; // GitHub search query
  Repos: string[]; // local repo roots
  MinLevel: RepoLevel;
  MaxRepos: number; // how many repos to process this run
  MaxFilesPerRepo: number;
  MaxBytesPerRepo: number;
  MaxContentBytes: number; // per-file size cap (larger files are dropped)
  SkipLearned: boolean; // don't re-learn repos already in the store
};

export type LearnEvent =
  | { kind: "start"; query: string; source: string }
  | { kind: "repo"; repo: string; level: string; files: number; bytes: number; ingested: boolean; reason: string | null }
  | { kind: "done"; ingested: number }
  | { kind: "error"; message: string };

export type LearnFn = (Settings: LearnSettings, OnEvent: (Event: LearnEvent) => void) => Promise<void>;

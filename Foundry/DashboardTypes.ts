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
  | { kind: "start"; query: string; source: string; repos: number } // repos = MaxRepos, for the progress bar denominator
  | { kind: "scanning"; label: string } // a "working" status shown during silent gaps (searching / downloading a repo)
  | { kind: "repo"; repo: string; level: string; files: number; bytes: number; ingested: boolean; reason: string | null }
  | { kind: "repo-progress"; repo: string; filesDone: number; filesTotal: number } // per-repo file ingestion
  | { kind: "done"; ingested: number }
  | { kind: "error"; message: string };

export type LearnFn = (Settings: LearnSettings, OnEvent: (Event: LearnEvent) => void, Signal?: AbortSignal) => Promise<void>;

// Model TRAINING (distinct from Learn/data-collection): turn the collected Postgres corpus into a
// trained model checkpoint. Runs as a subprocess so it never blocks the dashboard event loop.
export type TrainSettings = {
  Name: string; // checkpoint name — train/keep multiple models side by side
  Steps: number;
  CorpusMb: number;
  EmbedDim: number;
  NumLayers: number;
  NumHeads: number;
  BlockSize: number;
  Merges: number; // vocab = 256 + Merges
  BatchSize: number;
};

export type TrainEvent =
  | { kind: "train-start"; steps: number }
  | { kind: "train-info"; text: string } // corpus / bpe / model setup lines
  | { kind: "train-progress"; step: number; steps: number; trainLoss: number; valLoss?: number; elapsedMs?: number }
  | { kind: "train-done"; savedTo: string }
  | { kind: "train-error"; message: string };

export type TrainFn = (Settings: TrainSettings, OnEvent: (Event: TrainEvent) => void, Signal?: AbortSignal) => Promise<void>;

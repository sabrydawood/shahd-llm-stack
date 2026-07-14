// Ingest OUR OWN repositories from disk (M8). Each root directory is treated as a repo: walk it,
// keep substantive source (same CodeFileFilter as the web path), assess its level, and — if it
// qualifies — ingest EVERY file tagged Origin "owned" (trained on regardless of license, since it's
// our code). Skips node_modules/.git/dot-dirs. A WebProvider so it plugs into the same IngestFromWeb
// pipeline; the query/limit arguments are ignored.

import { readdirSync, readFileSync } from "node:fs";
import { join, relative, basename } from "node:path";
import type { WebProvider } from "./WebSource.ts";
import type { SourceInput } from "./Ingest.ts";
import type { RepoFile } from "./RepoArchive.ts";
import { DefaultRepoLimits } from "./RepoArchive.ts";
import { AssessRepo, LevelRank, EmptyAssessment } from "./RepoQuality.ts";
import type { RepoLevel, RepoIngestInfo } from "./RepoQuality.ts";
import { IsSubstantiveCodePath, IsSubstantiveCodeContent, LangForPath } from "./CodeFileFilter.ts";
import { StripLicenseHeader } from "./ContentNormalizer.ts";

function WalkRepo(Root: string, MaxFiles: number, MaxBytes: number, MaxContentBytes: number): RepoFile[] {
  const Out: RepoFile[] = [];
  let Bytes = 0;
  const Walk = (Dir: string): void => {
    if (Out.length >= MaxFiles || Bytes >= MaxBytes) return;
    let Entries: import("node:fs").Dirent[];
    try {
      Entries = readdirSync(Dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const Entry of Entries) {
      if (Out.length >= MaxFiles || Bytes >= MaxBytes) return;
      const Full = join(Dir, Entry.name);
      if (Entry.isDirectory()) {
        if (Entry.name === "node_modules" || Entry.name.startsWith(".")) continue;
        Walk(Full);
      } else if (Entry.isFile()) {
        const Rel = relative(Root, Full).split("\\").join("/"); // normalize Windows separators
        if (!IsSubstantiveCodePath(Rel)) continue;
        let Content: string;
        try {
          Content = StripLicenseHeader(readFileSync(Full, "utf8"));
        } catch {
          continue;
        }
        if (!IsSubstantiveCodeContent(Content, MaxContentBytes)) continue;
        Out.push({ Path: Rel, Content });
        Bytes += Content.length;
      }
    }
  };
  Walk(Root);
  return Out;
}

export type LocalRepoOptions = {
  Roots: string[]; // each directory is one repo
  License?: string; // metadata tag for our own code (default "OWNED")
  MinLevel?: RepoLevel;
  MaxFiles?: number;
  MaxBytes?: number;
  MaxContentBytes?: number;
  SkipRepo?: (Repo: string) => boolean;
  OnRepo?: (Info: RepoIngestInfo) => void;
};

export function CreateLocalRepoProvider(Options: LocalRepoOptions): WebProvider {
  const License = Options.License ?? "OWNED";
  const MinLevel = Options.MinLevel ?? "medium";
  const MaxFiles = Options.MaxFiles ?? DefaultRepoLimits.MaxFiles;
  const MaxBytes = Options.MaxBytes ?? DefaultRepoLimits.MaxBytes;
  const MaxContentBytes = Options.MaxContentBytes ?? DefaultRepoLimits.MaxContentBytes;
  return {
    Name: "local-repo",
    Fetch: async (): Promise<SourceInput[]> => {
      const Out: SourceInput[] = [];
      for (const Root of Options.Roots) {
        const Name = basename(Root);
        if (Options.SkipRepo?.(Name) === true) {
          Options.OnRepo?.({ Repo: Name, License, Assessment: EmptyAssessment, Ingested: false, Reason: "already learned" });
          continue;
        }
        const Files = WalkRepo(Root, MaxFiles, MaxBytes, MaxContentBytes);
        const Assessment = AssessRepo(Files);
        const Ingested = LevelRank[Assessment.Level] >= LevelRank[MinLevel];
        Options.OnRepo?.({ Repo: Name, License, Assessment, Ingested });
        if (!Ingested) continue;
        for (const File of Files) {
          Out.push({
            Source: Name,
            License,
            Lang: LangForPath(File.Path),
            Content: File.Content,
            Provenance: `${Root}/${File.Path}`,
            Origin: "owned",
          });
        }
      }
      return Out;
    },
  };
}

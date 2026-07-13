// Assess a repository's LEVEL before learning from it (M7). Sabry's intent: take a whole repo,
// understand its level, and learn from it — not random snippets. This aggregates the extracted files
// into a level so low-quality repos can be skipped and good ones ingested wholesale. Signals: how
// many substantive files, their average code-quality score, and whether the repo has real source
// structure (src/lib/packages) rather than a flat pile of scripts.

import { ScoreCodeQuality } from "../Brain/Data/QualityFilter.ts";
import type { RepoFile } from "./RepoArchive.ts";

export type RepoLevel = "high" | "medium" | "low";
export const LevelRank: Record<RepoLevel, number> = { high: 2, medium: 1, low: 0 };

export type RepoAssessment = {
  FileCount: number;
  TotalBytes: number;
  AvgQuality: number;
  HasStructure: boolean;
  Level: RepoLevel;
};

const StructureDir = /(^|\/)(src|lib|source|app|packages|internal|pkg|core)\//i;

export function AssessRepo(Files: RepoFile[]): RepoAssessment {
  let TotalBytes = 0;
  let QualitySum = 0;
  let HasStructure = false;
  for (const File of Files) {
    TotalBytes += Buffer.byteLength(File.Content, "utf8");
    QualitySum += ScoreCodeQuality(File.Content).Score;
    if (StructureDir.test(File.Path)) HasStructure = true;
  }
  const FileCount = Files.length;
  const AvgQuality = FileCount > 0 ? QualitySum / FileCount : 0;

  let Level: RepoLevel = "low";
  if (FileCount >= 5 && AvgQuality >= 0.8 && HasStructure) Level = "high";
  else if (FileCount >= 2 && AvgQuality >= 0.6) Level = "medium";

  return { FileCount, TotalBytes, AvgQuality, HasStructure, Level };
}

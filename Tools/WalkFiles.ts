// Shared filesystem walk for the CI gate tools. Single-source (DRY, rule #4):
// both CheckFileLength and CheckNamingConvention consume this instead of duplicating the walk.

import { readdirSync, existsSync } from "node:fs";
import { join, extname } from "node:path";

/** The source roots the CI gates police. */
export const SourceRoots: readonly string[] = ["Brain", "Scripts", "Tools", "Tests"];

/** Recursively collect every `.ts` file under `Dir`. Returns [] if `Dir` is absent. */
export function WalkTsFiles(Dir: string): string[] {
  const Collected: string[] = [];
  if (!existsSync(Dir)) return Collected; // root not created yet — not an error
  for (const Entry of readdirSync(Dir, { withFileTypes: true })) {
    const Full = join(Dir, Entry.name);
    if (Entry.isDirectory()) {
      Collected.push(...WalkTsFiles(Full));
    } else if (extname(Entry.name) === ".ts") {
      Collected.push(Full);
    }
  }
  return Collected;
}

/** Collect every `.ts` file across all SourceRoots. */
export function WalkAllSourceFiles(): string[] {
  const All: string[] = [];
  for (const Root of SourceRoots) All.push(...WalkTsFiles(Root));
  return All;
}

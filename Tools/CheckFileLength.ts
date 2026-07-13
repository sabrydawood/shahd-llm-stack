// CI gate — rule #3: no source file exceeds 600 lines. Fails the build (exit 1) on violation.

import { readFileSync } from "node:fs";
import { WalkAllSourceFiles } from "./WalkFiles.ts";

const MaxLines = 600;

const Violations: string[] = [];
for (const File of WalkAllSourceFiles()) {
  const LineCount = readFileSync(File, "utf8").split("\n").length;
  if (LineCount > MaxLines) {
    Violations.push(`  ${File}: ${LineCount} lines (max ${MaxLines})`);
  }
}

if (Violations.length > 0) {
  console.error(`CheckFileLength: ${Violations.length} file(s) exceed ${MaxLines} lines (rule #3):`);
  console.error(Violations.join("\n"));
  process.exit(1);
}

console.log("CheckFileLength: OK");

// CI gate — rules #1/#4: source filenames are PascalCase, and grab-bag names are banned
// (Utils/Helpers/Common/Misc hide duplication; Index enforces "no index.ts"). Fails on violation.

import { basename } from "node:path";
import { WalkAllSourceFiles } from "./WalkFiles.ts";

const PascalCase = /^[A-Z][A-Za-z0-9]*$/;
const BannedBaseNames: readonly string[] = ["Utils", "Helpers", "Common", "Misc", "Index"];
const TestSuffix = ".Test";

const Violations: string[] = [];
for (const File of WalkAllSourceFiles()) {
  const Base = basename(File, ".ts");
  // Allow the PascalCase test convention `Foo.Test.ts` — validate the `Foo` part.
  const NamePart = Base.endsWith(TestSuffix) ? Base.slice(0, -TestSuffix.length) : Base;

  if (!PascalCase.test(NamePart)) {
    Violations.push(`  ${File}: filename base "${NamePart}" is not PascalCase (rule #1)`);
  }
  if (BannedBaseNames.includes(NamePart)) {
    Violations.push(`  ${File}: grab-bag filename "${NamePart}" is banned (rule #4)`);
  }
}

if (Violations.length > 0) {
  console.error(`CheckNamingConvention: ${Violations.length} violation(s):`);
  console.error(Violations.join("\n"));
  process.exit(1);
}

console.log("CheckNamingConvention: OK");

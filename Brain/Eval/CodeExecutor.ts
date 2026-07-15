// Sandboxed code execution for eval + RL verifiable reward (Phase 5). Runs candidate code in a
// FRESH subprocess in a throwaway temp dir with a hard timeout, so model output can be checked
// against tests (pass/fail) without hanging the host on infinite loops.
//
// ⚠️ SECURITY (Sabry's absolute priority, dedicated place): this provides PROCESS ISOLATION + a HARD
// TIMEOUT CAP + a SECRET-SCRUBBED ENV (below). It still does NOT restrict filesystem or network access —
// executing untrusted model output at scale (real RL, or the `run_code` tool with ExecEnabled=true) MUST
// additionally run inside a container/VM/gVisor sandbox with no network and a read-only fs. Treat this
// as the controllable seam where that stronger isolation is plugged in. Do NOT enable ExecEnabled in a
// deployment reachable by untrusted input without that OS-level isolation.

import { writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

export type ExecResult = {
  Passed: boolean; // exit code 0 and not killed
  ExitCode: number | null;
  Stdout: string;
  Stderr: string;
  DurationMs: number;
};

// A hard ceiling on the (possibly model-supplied) timeout: a caller/model can shorten it but never make
// one execution block the host for an unbounded window. The whole call is synchronous, so an uncapped
// timeout on `while(true){}` would freeze the single-threaded process for that entire duration.
const HardMaxTimeoutMs = 10000;

// Run with a SECRET-SCRUBBED environment: model-authored code inherits the process env otherwise, so
// DB URLs / API tokens would be readable (and exfiltratable via network). Drop any var whose NAME looks
// sensitive; keep the rest (PATH, SystemRoot, TEMP, … stay so the runtime still starts on every OS).
function ScrubbedEnv(): Record<string, string> {
  const Sensitive = /token|secret|key|password|passwd|database|credential|auth|api/i;
  const Out: Record<string, string> = {};
  for (const [Name, Value] of Object.entries(process.env)) {
    if (Value !== undefined && !Sensitive.test(Name)) Out[Name] = Value;
  }
  return Out;
}

export function RunCode(Code: string, TimeoutMs = 5000): ExecResult {
  const Timeout = Math.min(Math.max(1, Math.floor(TimeoutMs)), HardMaxTimeoutMs);
  const Dir = mkdtempSync(join(tmpdir(), "shahd-exec-"));
  const File = join(Dir, "Candidate.ts");
  writeFileSync(File, Code);
  const Start = Date.now();
  try {
    const Proc = Bun.spawnSync(["bun", "run", File], {
      timeout: Timeout,
      stdout: "pipe",
      stderr: "pipe",
      env: ScrubbedEnv(),
    });
    return {
      Passed: Proc.exitCode === 0,
      ExitCode: Proc.exitCode,
      Stdout: Proc.stdout.toString(),
      Stderr: Proc.stderr.toString(),
      DurationMs: Date.now() - Start,
    };
  } finally {
    rmSync(Dir, { recursive: true, force: true });
  }
}

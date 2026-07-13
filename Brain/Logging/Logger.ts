// Structured JSONL logger. One JSON object per line, optionally mirrored to the console. Each
// run's log is a comparable artifact (step/loss/lr/tokens/gradnorm/wallclock tied to ConfigHash).

import { appendFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname } from "node:path";

export type LogEvent = Record<string, unknown>;

export class Logger {
  FilePath: string | null;
  Console: boolean;

  constructor(FilePath: string | null = null, Console = true) {
    this.FilePath = FilePath;
    this.Console = Console;
    if (FilePath !== null) {
      const Dir = dirname(FilePath);
      if (Dir !== "" && !existsSync(Dir)) mkdirSync(Dir, { recursive: true });
    }
  }

  Log(Event: LogEvent): void {
    const Line = JSON.stringify(Event);
    if (this.FilePath !== null) appendFileSync(this.FilePath, Line + "\n");
    if (this.Console) console.log(Line);
  }
}

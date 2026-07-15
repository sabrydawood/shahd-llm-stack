// The autograd "tape" switch. When Tape.On is false, ops skip recording parents/backward
// closures (used during sampling/eval so no graph is built). Mirrors nano-gpt.ts's TAPE_ON.
// An object (not a bare `let`) so importers can toggle it — ESM `let` bindings are read-only
// for consumers.

export const Tape = { On: true };

/** Run `Body` with the tape disabled, restoring the previous state afterwards. */
export function WithTapeOff<T>(Body: () => T): T {
  const Previous = Tape.On;
  Tape.On = false;
  try {
    const Result = Body();
    // Body must be synchronous: an async body would return before it actually finishes, so the
    // `finally` below restores Tape.On while the still-pending work keeps recording (or not
    // recording) against a tape state that has already moved on. Fail loudly instead.
    if (Result !== null && typeof Result === "object" && typeof (Result as { then?: unknown }).then === "function") {
      throw new Error("WithTapeOff: Body must be synchronous");
    }
    return Result;
  } finally {
    Tape.On = Previous;
  }
}

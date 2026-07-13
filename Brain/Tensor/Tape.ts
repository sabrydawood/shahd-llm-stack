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
    return Body();
  } finally {
    Tape.On = Previous;
  }
}

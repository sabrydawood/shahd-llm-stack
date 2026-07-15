// The dedicated, controllable safety boundary for every filesystem tool (mirrors GuardedGenerate's
// role for generation). A Workspace pins a Root directory and refuses any path that escapes it:
// user-supplied paths are resolved against Root and rejected if the relative result climbs out
// (starts with "..") or is absolute. This is the ONE place path-traversal is stopped, so file tools
// can never touch bytes outside the sanctioned root no matter what the model emits — INCLUDING via a
// symlink planted inside the root (CWE-59), which the lexical check alone would follow transparently.

import { resolve, relative, isAbsolute, sep, dirname } from "node:path";
import { realpathSync, existsSync } from "node:fs";

function Escapes(Rel: string): boolean {
  return Rel === ".." || Rel.startsWith(".." + sep) || isAbsolute(Rel);
}

export class Workspace {
  readonly Root: string;
  private readonly RealRoot: string;

  constructor(Root: string) {
    this.Root = resolve(Root);
    // Resolve the root's own symlinks once so containment is checked against the REAL directory.
    this.RealRoot = existsSync(this.Root) ? realpathSync(this.Root) : this.Root;
  }

  /** Resolve a workspace-relative path to an absolute one, or throw if it escapes Root. */
  Resolve(RelPath: string): string {
    const Absolute = resolve(this.Root, RelPath);
    const Rel = relative(this.Root, Absolute);
    if (Escapes(Rel)) throw new Error(`path escapes workspace root: ${RelPath}`);
    // Lexical check passed; now defeat SYMLINK escapes (only meaningful when the root exists, so real
    // components — a symlink among them — can exist): resolve the real path of the target (or its
    // nearest existing ancestor, for a not-yet-created file) and re-verify it is still under the REAL
    // root. A symlink inside Root pointing outside would otherwise be followed silently.
    if (existsSync(this.Root)) {
      const RealRel = relative(this.RealRoot, this.RealPathOfNearestExisting(Absolute));
      if (Escapes(RealRel)) throw new Error(`path escapes workspace root (symlink): ${RelPath}`);
    }
    return Absolute;
  }

  // realpath of the deepest ancestor of `Target` that actually exists (Target itself when it exists),
  // so a write to a not-yet-created file is still checked against real, symlink-resolved directories.
  private RealPathOfNearestExisting(Target: string): string {
    let Current = Target;
    for (;;) {
      if (existsSync(Current)) return realpathSync(Current);
      const Parent = dirname(Current);
      if (Parent === Current) return Current; // reached the fs root with nothing existing
      Current = Parent;
    }
  }

  /** The path shown back to the model — always relative to Root, never leaking the absolute prefix. */
  Display(Absolute: string): string {
    const Rel = relative(this.Root, Absolute);
    return Rel === "" ? "." : Rel;
  }
}

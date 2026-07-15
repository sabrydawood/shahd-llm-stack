// The dedicated, controllable safety boundary for every filesystem tool (mirrors GuardedGenerate's
// role for generation). A Workspace pins a Root directory and refuses any path that escapes it:
// user-supplied paths are resolved against Root and rejected if the relative result climbs out
// (starts with "..") or is absolute. This is the ONE place path-traversal is stopped, so file tools
// can never touch bytes outside the sanctioned root no matter what the model emits — INCLUDING via a
// symlink planted inside the root (CWE-59), which the lexical check alone would follow transparently.

import { resolve, relative, isAbsolute, sep, dirname, join, basename } from "node:path";
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

  /** Resolve a workspace-relative path to an absolute one, or throw if it escapes Root.
   *  When Root exists, the REAL (symlink-resolved) path is returned rather than the lexical one: this
   *  closes the TOCTOU window between this check and the caller's later statSync/readFileSync/
   *  writeFileSync, which would otherwise re-resolve a symlink that could have been swapped in between. */
  Resolve(RelPath: string): string {
    const Absolute = resolve(this.Root, RelPath);
    const Rel = relative(this.Root, Absolute);
    if (Escapes(Rel)) throw new Error(`path escapes workspace root: ${RelPath}`);
    if (!existsSync(this.Root)) return Absolute; // nothing real to resolve against yet
    // Lexical check passed; now defeat SYMLINK escapes: resolve the real path of the target (or its
    // nearest existing ancestor, for a not-yet-created file) and re-verify it is still under the REAL
    // root. A symlink inside Root pointing outside would otherwise be followed silently. RETURN this
    // real path (instead of re-deriving it later) so callers act on the already-validated location.
    const RealAbsolute = this.RealPathOf(Absolute);
    const RealRel = relative(this.RealRoot, RealAbsolute);
    if (Escapes(RealRel)) throw new Error(`path escapes workspace root (symlink): ${RelPath}`);
    return RealAbsolute;
  }

  // Real, symlink-resolved path for Target: if it exists, its own realpath; otherwise the realpath of
  // its nearest existing ancestor with the not-yet-existing suffix re-appended, so a write target can
  // be validated (and reused for the actual fs call) before the file itself exists.
  private RealPathOf(Target: string): string {
    const Suffix: string[] = [];
    let Current = Target;
    for (;;) {
      if (existsSync(Current)) return join(realpathSync(Current), ...Suffix);
      const Parent = dirname(Current);
      if (Parent === Current) return join(Current, ...Suffix); // reached the fs root with nothing existing
      Suffix.unshift(basename(Current));
      Current = Parent;
    }
  }

  /** The path shown back to the model — always relative to Root, never leaking the absolute prefix. */
  Display(Absolute: string): string {
    const Rel = relative(this.Root, Absolute);
    return Rel === "" ? "." : Rel;
  }
}

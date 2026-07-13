// Corpus provenance ledger + permissive-license filter (Phase 3). REVIEW.md flags licensing as
// a real business risk for a company that also ships client work: a small model memorizes and can
// regurgitate training code verbatim, so copyleft/unknown-license source must be kept out of the
// corpus and every slice must be traceable. Track provenance from the first ingested byte.

export const PermissiveLicenses: ReadonlySet<string> = new Set([
  "MIT",
  "Apache-2.0",
  "BSD-2-Clause",
  "BSD-3-Clause",
  "ISC",
  "0BSD",
  "Unlicense",
  "Zlib",
]);

export type ProvenanceEntry = {
  Source: string; // e.g. repo or dataset name
  License: string; // SPDX id
  Path: string; // file path within the source
  Bytes: number;
  IngestedAt: string; // ISO timestamp (supplied by the caller — keeps this pure/testable)
};

export function IsPermissive(License: string): boolean {
  return PermissiveLicenses.has(License);
}

export class LicenseManifest {
  Entries: ProvenanceEntry[] = [];

  Add(Entry: ProvenanceEntry): void {
    this.Entries.push(Entry);
  }

  /** Entries whose license is on the permissive allowlist (the only ones safe to train on). */
  PermissiveOnly(): ProvenanceEntry[] {
    return this.Entries.filter((E) => IsPermissive(E.License));
  }

  /** Total bytes and a per-license breakdown — the "what is in the training set" answer. */
  Summary(): { TotalBytes: number; ByLicense: Record<string, number> } {
    const ByLicense: Record<string, number> = {};
    let TotalBytes = 0;
    for (const E of this.Entries) {
      TotalBytes += E.Bytes;
      ByLicense[E.License] = (ByLicense[E.License] ?? 0) + E.Bytes;
    }
    return { TotalBytes, ByLicense };
  }

  ToJson(): string {
    return JSON.stringify(this.Entries);
  }

  static FromJson(Json: string): LicenseManifest {
    const Manifest = new LicenseManifest();
    Manifest.Entries = JSON.parse(Json) as ProvenanceEntry[];
    return Manifest;
  }
}

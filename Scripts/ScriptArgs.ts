// Shared CLI-arg helper for the entry scripts (single-source, rule #4).

export function ReadArg(Prefix: string, Fallback: string): string {
  const Found = process.argv.slice(2).find((A) => A.startsWith(Prefix));
  return Found ? Found.slice(Prefix.length) : Fallback;
}

/** True when a bare boolean flag (e.g. "--DryRun") is present in argv. */
export function ReadFlag(Flag: string): boolean {
  return process.argv.slice(2).includes(Flag);
}

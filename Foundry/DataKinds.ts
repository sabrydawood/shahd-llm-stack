// Data-kind taxonomy (Phase 9): every training document belongs to exactly ONE kind, and each kind
// lives in its OWN table (documents_<kind>) so data types stay physically separate. This lets models
// be trained in different directions — pure code, pure conversation, pure knowledge, or a MIX with a
// chosen size from each — instead of one blended blob. Extensible: add a kind here + a provider mapping
// and the store/migration/training pick it up. code/conversation/knowledge are in use; the rest are
// reserved (their tables exist and stay empty until a provider fills them).

export const DataKinds = ["code", "conversation", "knowledge", "books", "web", "instruction"] as const;
export type DataKind = (typeof DataKinds)[number];

export function IsDataKind(Value: string): Value is DataKind {
  return (DataKinds as readonly string[]).includes(Value);
}

/** The per-kind table name. One table per kind keeps the types cleanly separated. */
export function TableForKind(Kind: DataKind): string {
  return `documents_${Kind}`;
}

/** Which kind a collection provider produces. New providers slot in here (single source of truth). */
export function KindForProvider(ProviderName: string): DataKind {
  switch (ProviderName) {
    case "github-repo":
    case "github":
    case "local-repo":
      return "code";
    case "oasst":
      return "conversation";
    case "wikipedia":
      return "knowledge";
    case "gsm8k":
      return "instruction";
    case "web-search":
      return "web";
    default:
      return "code";
  }
}

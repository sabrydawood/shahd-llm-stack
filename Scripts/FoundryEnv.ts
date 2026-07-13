// Resolve Foundry run settings from the environment (Bun loads .env automatically) with CLI
// override. One home for this so every foundry script behaves the same: it defaults to Postgres
// whenever DATABASE_URL is set, reads GITHUB_TOKEN/BRAVE_API_KEY/queries/repos from env, and lets
// a --Flag override any of them.

import { InMemoryDocumentStore } from "../Foundry/FoundryBarrel.ts";
import type { DocumentStore } from "../Foundry/FoundryBarrel.ts";
import { PostgresDocumentStore } from "../Foundry/PostgresDocumentStore.ts";
import { ReadArg } from "./ScriptArgs.ts";

export function DatabaseUrl(): string {
  return process.env["DATABASE_URL"] ?? "postgres://postgres:postgres@localhost:5432/shahd";
}

export function GitHubToken(): string | undefined {
  return process.env["GITHUB_TOKEN"];
}

export function BraveKey(): string | undefined {
  return process.env["BRAVE_API_KEY"];
}

/** Store: --Store CLI > FOUNDRY_STORE env > (DATABASE_URL present ? postgres : memory). */
export function ResolveStore(): { Store: DocumentStore; Kind: string } {
  const Choice = ReadArg("--Store=", "") || process.env["FOUNDRY_STORE"] || (process.env["DATABASE_URL"] !== undefined ? "postgres" : "memory");
  if (Choice === "postgres") return { Store: new PostgresDocumentStore(DatabaseUrl()), Kind: "postgres" };
  return { Store: new InMemoryDocumentStore(), Kind: "memory" };
}

/** Web search/repo query: --Query CLI > FOUNDRY_QUERY env > the provided default. */
export function Query(Default: string): string {
  return ReadArg("--Query=", "") || process.env["FOUNDRY_QUERY"] || Default;
}

/** Own-repo roots: --Repos CLI > FOUNDRY_REPOS env > ".". Comma-separated. */
export function RepoRoots(): string[] {
  const Raw = ReadArg("--Repos=", "") || process.env["FOUNDRY_REPOS"] || ".";
  return Raw.split(",").map((S) => S.trim()).filter((S) => S.length > 0);
}

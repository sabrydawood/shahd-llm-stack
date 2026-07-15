// Filesystem tools — every one confined to Context.Workspace (path-traversal is refused there) and
// byte-capped by Context.MaxFileBytes. They are registered per Config.Tools.FileAccess: read/list/
// search need ReadOnly, write needs ReadWrite. Without a Workspace in the context they hard-error,
// so they are inert unless a host explicitly wired the sanctioned root.

import { readFileSync, writeFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import type { Tool, ToolContext } from "./ToolTypes.ts";
import { Err, RequireString } from "./ToolArgs.ts";

const MaxWalkEntries = 5000;
const MaxListEntries = 2000;

function RequireWorkspace(Context: ToolContext | undefined): NonNullable<ToolContext["Workspace"]> {
  if (Context?.Workspace === undefined) throw new Error("filesystem tools disabled (no workspace)");
  return Context.Workspace;
}

// Bounded recursive walk of relative paths under Root, capped so a huge tree cannot exhaust memory.
function WalkRelative(Root: string, Prefix: string, Out: string[]): void {
  if (Out.length >= MaxWalkEntries) return;
  let Entries: string[];
  try {
    Entries = readdirSync(join(Root, Prefix));
  } catch {
    return;
  }
  for (const Name of Entries) {
    if (Name === "node_modules" || Name === ".git") continue;
    const Rel = Prefix === "" ? Name : `${Prefix}/${Name}`;
    let IsDir = false;
    try {
      IsDir = statSync(join(Root, Rel)).isDirectory();
    } catch {
      continue;
    }
    if (IsDir) WalkRelative(Root, Rel, Out);
    else Out.push(Rel);
    if (Out.length >= MaxWalkEntries) return;
  }
}

// Read a UTF-8 file. Args: { path: string }.
export const FileReadTool: Tool = {
  Name: "file_read",
  Description: "Read a UTF-8 text file within the workspace.",
  Args: "{ path: string }",
  Execute: (Arguments, Context) => {
    const Workspace = RequireWorkspace(Context);
    const Cap = Context?.MaxFileBytes ?? 262144;
    const Absolute = Workspace.Resolve(RequireString(Arguments, "path"));
    const Bytes = statSync(Absolute).size;
    if (Bytes > Cap) return Err(`file exceeds cap (${Bytes} > ${Cap} bytes)`);
    return { path: Workspace.Display(Absolute), content: readFileSync(Absolute, "utf8") };
  },
};

// List a directory. Args: { path?: string }.
export const FileListTool: Tool = {
  Name: "file_list",
  Description: "List entries of a directory within the workspace.",
  Args: "{ path?: string }",
  Execute: (Arguments, Context) => {
    const Workspace = RequireWorkspace(Context);
    const Absolute = Workspace.Resolve(typeof Arguments["path"] === "string" ? String(Arguments["path"]) : ".");
    const All = readdirSync(Absolute);
    const Truncated = All.length > MaxListEntries;
    const Entries = All.slice(0, MaxListEntries).map((Name) => {
      const IsDir = statSync(join(Absolute, Name)).isDirectory();
      return { name: Name, kind: IsDir ? "dir" : "file" };
    });
    return { path: Workspace.Display(Absolute), entries: Entries, truncated: Truncated };
  },
};

// Search filenames by substring. Args: { query: string }.
export const FileSearchTool: Tool = {
  Name: "file_search",
  Description: "Find files whose path contains a substring (bounded, workspace-only).",
  Args: "{ query: string }",
  Execute: (Arguments, Context) => {
    const Workspace = RequireWorkspace(Context);
    const Query = RequireString(Arguments, "query").toLowerCase();
    const All: string[] = [];
    WalkRelative(Workspace.Root, "", All);
    const Matches = All.filter((Rel) => Rel.toLowerCase().includes(Query)).slice(0, 200);
    return { matches: Matches, count: Matches.length, truncated: All.length >= MaxWalkEntries };
  },
};

// Write a UTF-8 file. Args: { path: string, content: string }. Needs FileAccess=ReadWrite.
export const FileWriteTool: Tool = {
  Name: "file_write",
  Description: "Write a UTF-8 text file within the workspace (requires ReadWrite access).",
  Args: "{ path: string, content: string }",
  Execute: (Arguments, Context) => {
    const Workspace = RequireWorkspace(Context);
    const Cap = Context?.MaxFileBytes ?? 262144;
    const Content = RequireString(Arguments, "content");
    const ByteLength = Buffer.byteLength(Content, "utf8"); // writeFileSync writes UTF-8 bytes, not UTF-16 chars
    if (ByteLength > Cap) return Err(`content exceeds cap (${ByteLength} > ${Cap} bytes)`);
    const Absolute = Workspace.Resolve(RequireString(Arguments, "path"));
    writeFileSync(Absolute, Content);
    return { path: Workspace.Display(Absolute), bytesWritten: ByteLength };
  },
};

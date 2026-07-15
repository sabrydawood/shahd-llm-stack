// Assembles a ToolRegistry from a capability policy (derived from Config.Tools). Always-safe tools
// are registered unconditionally; the dangerous ones are gated: run_code needs ExecEnabled, file
// reads need FileAccess>=ReadOnly, file writes need ReadWrite. This is the enforcement half of the
// central Tools gate — the config decides, this function obeys.

import type { ResolvedConfig } from "../../Config/ConfigTypes.ts";
import { ToolRegistry } from "./ToolRegistry.ts";
import { CalculatorTool, StatsTool } from "./MathTools.ts";
import { JsonTool, RegexTool, TextTool } from "./TextTools.ts";
import { CurrentTimeTool, HashTool, UuidTool, RandomIntTool } from "./SystemTools.ts";
import { WebSearchTool, MemoryStoreTool, MemoryRecallTool } from "./KnowledgeTools.ts";
import { UserAskTool, ListToolsTool, PlanTool, CompactTool, FinishTool } from "./ControlTools.ts";
import { RunCodeTool } from "./CodeTools.ts";
import { FileReadTool, FileListTool, FileSearchTool, FileWriteTool } from "./FileTools.ts";

export type FileAccess = "Off" | "ReadOnly" | "ReadWrite";

export type ToolsPolicy = {
  FileAccess: FileAccess;
  ExecEnabled: boolean;
  WebSearchEnabled: boolean;
};

/** Read the capability policy straight off the validated config's Tools section. */
export function ToolsPolicyFromConfig(Config: ResolvedConfig): ToolsPolicy {
  return {
    FileAccess: Config.Tools.FileAccess,
    ExecEnabled: Config.Tools.ExecEnabled,
    WebSearchEnabled: Config.Tools.WebSearchEnabled,
  };
}

// Registered no matter what — none of these touch fs, network, or a subprocess.
const AlwaysSafe = [
  CalculatorTool, StatsTool,
  JsonTool, RegexTool, TextTool,
  CurrentTimeTool, HashTool, UuidTool, RandomIntTool,
  MemoryStoreTool, MemoryRecallTool,
  UserAskTool, ListToolsTool, PlanTool, CompactTool, FinishTool,
];

/** Build a registry honoring the capability policy. */
export function BuildToolRegistry(Policy: ToolsPolicy): ToolRegistry {
  const Registry = new ToolRegistry();
  for (const Tool of AlwaysSafe) Registry.Register(Tool);
  if (Policy.WebSearchEnabled) Registry.Register(WebSearchTool);
  if (Policy.ExecEnabled) Registry.Register(RunCodeTool);
  if (Policy.FileAccess !== "Off") {
    Registry.Register(FileReadTool);
    Registry.Register(FileListTool);
    Registry.Register(FileSearchTool);
  }
  if (Policy.FileAccess === "ReadWrite") Registry.Register(FileWriteTool);
  return Registry;
}

// The safe default policy for dev/tests: read-only files, no exec, offline web. Matches Constants.
export const DefaultToolsPolicy: ToolsPolicy = {
  FileAccess: "ReadOnly",
  ExecEnabled: false,
  WebSearchEnabled: false,
};

/** The default registry (safe policy). Backwards-compatible name for existing callers. */
export function DefaultToolRegistry(): ToolRegistry {
  return BuildToolRegistry(DefaultToolsPolicy);
}

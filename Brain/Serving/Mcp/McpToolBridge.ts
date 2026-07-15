// Bridge MCP server tools into Shahd's own tool system: each MCP tool becomes a Tool whose Execute
// calls the MCP server. Register these into a ToolRegistry and the agent can use external MCP tools
// exactly like the built-in ones. Names are prefixed (default "mcp_") to avoid collisions.

import type { Tool, ToolResult } from "../Tools/ToolTypes.ts";
import type { McpClient } from "./McpClient.ts";
import { ToolTokenList } from "../ToolProtocol.ts";

const MaxCleanLength = 2000;

function AsResult(Value: unknown): ToolResult {
  return typeof Value === "object" && Value !== null ? (Value as ToolResult) : { result: Value };
}

// An MCP server is an untrusted peer: its tool name/description/schema flow straight into our own
// tool manifest and conversation transcript, so a crafted server could otherwise smuggle a fake
// <|tool_call|>/<|tool_result|> sentinel or newline-based prompt structure. Strip our sentinel
// tokens, collapse newlines, and cap length before any of it is used.
function Clean(Text: string): string {
  let Out = Text;
  for (const Token of ToolTokenList) Out = Out.split(Token).join("");
  Out = Out.replace(/\s*[\r\n]+\s*/g, " ").trim();
  return Out.length > MaxCleanLength ? Out.slice(0, MaxCleanLength) : Out;
}

/** Fetch the MCP server's tools and wrap each as a Shahd Tool backed by the client. */
export async function McpToolsFromClient(Client: McpClient, Prefix = "mcp_"): Promise<Tool[]> {
  const Tools = await Client.ListTools();
  return Tools.map((Spec) => ({
    Name: Prefix + Clean(Spec.name),
    Description: Clean(Spec.description ?? `MCP tool ${Spec.name}`),
    Args: typeof Spec.inputSchema === "object" ? Clean(JSON.stringify(Spec.inputSchema)) : "{}",
    Execute: async (Arguments: Record<string, unknown>): Promise<ToolResult> => {
      return AsResult(await Client.CallTool(Spec.name, Arguments));
    },
  }));
}

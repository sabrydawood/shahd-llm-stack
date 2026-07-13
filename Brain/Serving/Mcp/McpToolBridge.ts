// Bridge MCP server tools into Shahd's own tool system: each MCP tool becomes a Tool whose Execute
// calls the MCP server. Register these into a ToolRegistry and the agent can use external MCP tools
// exactly like the built-in ones. Names are prefixed (default "mcp_") to avoid collisions.

import type { Tool, ToolResult } from "../Tools/ToolTypes.ts";
import type { McpClient } from "./McpClient.ts";

function AsResult(Value: unknown): ToolResult {
  return typeof Value === "object" && Value !== null ? (Value as ToolResult) : { result: Value };
}

/** Fetch the MCP server's tools and wrap each as a Shahd Tool backed by the client. */
export async function McpToolsFromClient(Client: McpClient, Prefix = "mcp_"): Promise<Tool[]> {
  const Tools = await Client.ListTools();
  return Tools.map((Spec) => ({
    Name: Prefix + Spec.name,
    Description: Spec.description ?? `MCP tool ${Spec.name}`,
    Args: typeof Spec.inputSchema === "object" ? JSON.stringify(Spec.inputSchema) : "{}",
    Execute: async (Arguments: Record<string, unknown>): Promise<ToolResult> => {
      return AsResult(await Client.CallTool(Spec.name, Arguments));
    },
  }));
}

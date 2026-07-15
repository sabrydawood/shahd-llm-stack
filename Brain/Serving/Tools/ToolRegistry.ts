// The tool registry: name -> Tool, plus a never-throwing async Run that executes a parsed tool call
// against an injected ToolContext. Run converts any thrown error (including ToolArgError) into an
// { error } result so the agent loop is never derailed by a bad call.

import type { ToolCall } from "../ToolProtocol.ts";
import type { Tool, ToolContext, ToolResult, ToolRegistryView } from "./ToolTypes.ts";
import { Err } from "./ToolArgs.ts";

// Hard ceiling on how long a single Execute may run. Without this a slow/hung tool (including a
// slow MCP-bridged one riding StdioTransport) could block RunAgent past MaxSteps indefinitely.
const ToolTimeoutMs = 15_000;

export class ToolRegistry implements ToolRegistryView {
  private Tools = new Map<string, Tool>();

  Register(Tool: Tool): void {
    this.Tools.set(Tool.Name, Tool);
  }

  Get(Name: string): Tool | undefined {
    return this.Tools.get(Name);
  }

  Has(Name: string): boolean {
    return this.Tools.has(Name);
  }

  List(): Tool[] {
    return [...this.Tools.values()];
  }

  /** Run a parsed tool call with the given context; returns the tool's result or an error object.
   *  Races Execute against ToolTimeoutMs so a hung tool can never stall the agent loop. */
  async Run(Call: ToolCall, Context?: ToolContext): Promise<ToolResult> {
    const Tool = this.Tools.get(Call.Name);
    if (Tool === undefined) return Err(`unknown tool: ${Call.Name}`);
    let TimeoutHandle: ReturnType<typeof setTimeout> | undefined;
    try {
      const Timeout = new Promise<never>((_Resolve, Reject) => {
        TimeoutHandle = setTimeout(() => Reject(new Error("tool timed out")), ToolTimeoutMs);
      });
      return await Promise.race([Promise.resolve(Tool.Execute(Call.Arguments, Context)), Timeout]);
    } catch (Error_) {
      return Err((Error_ as Error).message);
    } finally {
      if (TimeoutHandle !== undefined) clearTimeout(TimeoutHandle);
    }
  }
}

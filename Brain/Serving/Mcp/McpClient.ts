// MCP client over an injected transport. Does the initialize handshake, lists the server's tools,
// and calls them. Keeps no protocol state beyond what the transport needs, so it works the same over
// a mock (tests) or a real stdio server (StdioTransport).

import type { McpTransport, McpTool } from "./McpTypes.ts";

const ProtocolVersion = "2024-11-05";

export class McpClient {
  private Transport: McpTransport;
  private Initialized = false;

  constructor(Transport: McpTransport) {
    this.Transport = Transport;
  }

  /** Handshake: initialize, then send the initialized notification. */
  async Initialize(): Promise<void> {
    await this.Transport.Rpc("initialize", {
      protocolVersion: ProtocolVersion,
      capabilities: {},
      clientInfo: { name: "shahd", version: "1.0" },
    });
    this.Transport.Notify("notifications/initialized");
    this.Initialized = true;
  }

  /** List the tools the server exposes (initializes first if needed). */
  async ListTools(): Promise<McpTool[]> {
    if (!this.Initialized) await this.Initialize();
    const Result = (await this.Transport.Rpc("tools/list")) as { tools?: McpTool[] };
    return Result.tools ?? [];
  }

  /** Call a server tool by name; returns the raw MCP result. */
  async CallTool(Name: string, Arguments: Record<string, unknown>): Promise<unknown> {
    if (!this.Initialized) await this.Initialize();
    return this.Transport.Rpc("tools/call", { name: Name, arguments: Arguments });
  }

  Close(): void {
    this.Transport.Close();
  }
}

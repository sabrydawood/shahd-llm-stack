import { test, expect } from "bun:test";
import { McpClient, McpToolsFromClient } from "../Brain/Serving/Mcp/McpBarrel.ts";
import type { McpTransport } from "../Brain/Serving/Mcp/McpBarrel.ts";
import { ToolRegistry } from "../Brain/Serving/Tools/ToolsBarrel.ts";

// In-memory MCP server standing in for a real one over stdio.
class MockTransport implements McpTransport {
  Calls: string[] = [];
  Notifications: string[] = [];
  async Rpc(Method: string, Params?: unknown): Promise<unknown> {
    this.Calls.push(Method);
    if (Method === "initialize") return { protocolVersion: "2024-11-05", capabilities: {}, serverInfo: { name: "mock" } };
    if (Method === "tools/list") return { tools: [{ name: "echo", description: "Echo the input", inputSchema: { type: "object" } }] };
    if (Method === "tools/call") {
      const Args = (Params as { arguments?: Record<string, unknown> }).arguments ?? {};
      return { content: [{ type: "text", text: `echoed ${JSON.stringify(Args)}` }] };
    }
    return {};
  }
  Notify(Method: string): void {
    this.Notifications.push(Method);
  }
  Close(): void {}
}

test("MCP client initializes, lists, and calls tools", async () => {
  const Transport = new MockTransport();
  const Client = new McpClient(Transport);
  const Tools = await Client.ListTools(); // triggers initialize handshake first
  expect(Transport.Calls).toContain("initialize");
  expect(Transport.Notifications).toContain("notifications/initialized");
  expect(Tools[0].name).toBe("echo");
  const Result = (await Client.CallTool("echo", { msg: "hi" })) as { content: { text: string }[] };
  expect(Result.content[0].text).toContain("hi");
});

test("MCP tool bridge exposes server tools as agent tools in a registry", async () => {
  const Client = new McpClient(new MockTransport());
  const Registry = new ToolRegistry();
  for (const Tool of await McpToolsFromClient(Client)) Registry.Register(Tool);
  expect(Registry.Has("mcp_echo")).toBe(true);
  const Result = await Registry.Run({ Name: "mcp_echo", Arguments: { msg: "hello" } });
  expect(JSON.stringify(Result)).toContain("hello"); // the agent can call an external MCP tool
});

// Model Context Protocol (MCP) client types. MCP is JSON-RPC 2.0; a client connects to an MCP
// server, initializes, lists the server's tools, and calls them. The transport is an interface so
// the client is testable with an in-memory mock and can drive a real server over stdio in production.

export interface McpTransport {
  /** Send a JSON-RPC request and resolve with its `result` (or reject on error). */
  Rpc(Method: string, Params?: unknown): Promise<unknown>;
  /** Send a JSON-RPC notification (no response expected). */
  Notify(Method: string, Params?: unknown): void;
  Close(): void;
}

export type McpTool = {
  name: string;
  description?: string;
  inputSchema?: unknown;
};

export type JsonRpcRequest = { jsonrpc: "2.0"; id?: number; method: string; params?: unknown };
export type JsonRpcResponse = { jsonrpc: "2.0"; id: number; result?: unknown; error?: { code: number; message: string } };

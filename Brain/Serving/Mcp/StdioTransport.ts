// Real MCP transport: spawn a server process and speak newline-delimited JSON-RPC over its stdio.
// Integration glue — verified against a real MCP server (e.g. `bun run Scripts/McpDemo.ts` pointed
// at a server command); the client/bridge logic is unit-tested with a mock transport instead.
//
// Robustness: when the server stream ends (clean EOF, crash, or Close()), every in-flight request is
// REJECTED (never left hanging) and the pending map is cleared; a complete final message with no
// trailing newline is still flushed; and Close() cancels the reader instead of relying on process
// death alone.

import type { FileSink } from "bun";
import type { McpTransport, JsonRpcRequest, JsonRpcResponse } from "./McpTypes.ts";

type Pending = { Resolve: (Value: unknown) => void; Reject: (Reason: Error) => void };

// A malformed/malicious server that never emits a newline could otherwise grow `this.Buffer` without
// bound; cap it and fail the transport instead of exhausting memory.
const MaxBufferBytes = 8 * 1024 * 1024; // 8 MiB

export class StdioTransport implements McpTransport {
  private Proc: ReturnType<typeof Bun.spawn>;
  private Stdin: FileSink;
  private CancelReader: (() => void) | null = null; // Close() calls this to unblock the read loop
  private Pending = new Map<number, Pending>();
  private NextId = 1;
  private Buffer = "";
  private Closed = false;

  constructor(Command: string[]) {
    this.Proc = Bun.spawn(Command, { stdin: "pipe", stdout: "pipe", stderr: "inherit" });
    this.Stdin = this.Proc.stdin as FileSink;
    void this.Pump();
  }

  private async Pump(): Promise<void> {
    const Reader = (this.Proc.stdout as ReadableStream<Uint8Array>).getReader();
    this.CancelReader = () => void Reader.cancel().catch(() => {});
    const Decoder = new TextDecoder();
    try {
      for (;;) {
        const { done: Done, value: Value } = await Reader.read();
        if (Done) break;
        this.Buffer += Decoder.decode(Value, { stream: true });
        if (this.Buffer.length > MaxBufferBytes) {
          this.RejectAll(new Error(`MCP transport buffer exceeded max size (${MaxBufferBytes} bytes)`));
          this.Close();
          return;
        }
        let Newline = this.Buffer.indexOf("\n");
        while (Newline !== -1) {
          const Line = this.Buffer.slice(0, Newline).trim();
          this.Buffer = this.Buffer.slice(Newline + 1);
          if (Line.length > 0) this.Handle(Line);
          Newline = this.Buffer.indexOf("\n");
        }
      }
      const Last = this.Buffer.trim(); // flush a complete final message that lacked a trailing newline
      if (Last.length > 0) this.Handle(Last);
      this.RejectAll(new Error("MCP transport closed (server stream ended)"));
    } catch (Caught) {
      this.RejectAll(Caught instanceof Error ? Caught : new Error("MCP transport read error"));
    }
  }

  private Handle(Line: string): void {
    let Message: JsonRpcResponse;
    try {
      Message = JSON.parse(Line) as JsonRpcResponse;
    } catch {
      return; // ignore non-JSON server chatter
    }
    if (typeof Message.id !== "number") return; // server notification/log
    const Waiter = this.Pending.get(Message.id);
    if (Waiter === undefined) return;
    this.Pending.delete(Message.id);
    if (Message.error !== undefined) Waiter.Reject(new Error(`MCP ${Message.error.code}: ${Message.error.message}`));
    else Waiter.Resolve(Message.result);
  }

  private RejectAll(Reason: Error): void {
    for (const Waiter of this.Pending.values()) Waiter.Reject(Reason);
    this.Pending.clear();
  }

  Rpc(Method: string, Params?: unknown): Promise<unknown> {
    if (this.Closed) return Promise.reject(new Error("MCP transport is closed"));
    const Id = this.NextId++;
    const Request: JsonRpcRequest = { jsonrpc: "2.0", id: Id, method: Method, params: Params };
    const ResultPromise = new Promise<unknown>((Resolve, Reject) => this.Pending.set(Id, { Resolve, Reject }));
    this.Write(Request);
    return ResultPromise;
  }

  Notify(Method: string, Params?: unknown): void {
    if (this.Closed) return;
    this.Write({ jsonrpc: "2.0", method: Method, params: Params });
  }

  private Write(Message: JsonRpcRequest): void {
    this.Stdin.write(JSON.stringify(Message) + "\n");
    this.Stdin.flush();
  }

  Close(): void {
    this.Closed = true;
    if (this.CancelReader !== null) this.CancelReader();
    this.RejectAll(new Error("MCP transport closed"));
    this.Stdin.end();
    this.Proc.kill();
  }
}

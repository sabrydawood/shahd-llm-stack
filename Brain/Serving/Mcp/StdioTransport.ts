// Real MCP transport: spawn a server process and speak newline-delimited JSON-RPC over its stdio.
// Integration glue — verified against a real MCP server (e.g. `bun run Scripts/McpDemo.ts` pointed
// at a server command); the client/bridge logic is unit-tested with a mock transport instead.

import type { FileSink } from "bun";
import type { McpTransport, JsonRpcRequest, JsonRpcResponse } from "./McpTypes.ts";

type Pending = { Resolve: (Value: unknown) => void; Reject: (Error_: Error) => void };

export class StdioTransport implements McpTransport {
  private Proc: ReturnType<typeof Bun.spawn>;
  private Stdin: FileSink;
  private Pending = new Map<number, Pending>();
  private NextId = 1;
  private Buffer = "";

  constructor(Command: string[]) {
    this.Proc = Bun.spawn(Command, { stdin: "pipe", stdout: "pipe", stderr: "inherit" });
    this.Stdin = this.Proc.stdin as FileSink; // "pipe" => a writable FileSink
    void this.Pump();
  }

  private async Pump(): Promise<void> {
    const Reader = (this.Proc.stdout as ReadableStream<Uint8Array>).getReader();
    const Decoder = new TextDecoder();
    for (;;) {
      const { done: Done, value: Value } = await Reader.read();
      if (Done) break;
      this.Buffer += Decoder.decode(Value, { stream: true });
      let Newline = this.Buffer.indexOf("\n");
      while (Newline !== -1) {
        const Line = this.Buffer.slice(0, Newline).trim();
        this.Buffer = this.Buffer.slice(Newline + 1);
        if (Line.length > 0) this.Handle(Line);
        Newline = this.Buffer.indexOf("\n");
      }
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

  Rpc(Method: string, Params?: unknown): Promise<unknown> {
    const Id = this.NextId++;
    const Request: JsonRpcRequest = { jsonrpc: "2.0", id: Id, method: Method, params: Params };
    const ResultPromise = new Promise<unknown>((Resolve, Reject) => this.Pending.set(Id, { Resolve, Reject }));
    this.Write(Request);
    return ResultPromise;
  }

  Notify(Method: string, Params?: unknown): void {
    this.Write({ jsonrpc: "2.0", method: Method, params: Params });
  }

  private Write(Message: JsonRpcRequest): void {
    this.Stdin.write(JSON.stringify(Message) + "\n");
    this.Stdin.flush();
  }

  Close(): void {
    this.Stdin.end();
    this.Proc.kill();
  }
}

// In-process TS inference server (Phase 6, C4: serving stays Node/Bun). An OpenAI-compatible
// /v1/chat/completions endpoint over Bun.serve, generating through the SAFE GuardedGenerate path
// (prompt/output safety + resource Limits). Supports non-streaming JSON and SSE streaming.

import type { Shahd } from "../Nn/Shahd.ts";
import type { Tokenizer } from "../Tokenizer/TokenizerTypes.ts";
import type { ResolvedConfig } from "../Config/ConfigTypes.ts";
import { GuardedGenerate } from "../Safety/GuardedGenerate.ts";
import { DefaultSampling } from "../Sampling/Sampler.ts";
import { SeededRng } from "../Random/SeededRng.ts";

type ChatRequest = {
  messages?: { role: string; content: string }[];
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
};

function RenderMessages(Messages: { role: string; content: string }[]): string {
  return Messages.map((M) => `${M.role}: ${M.content}`).join("\n") + "\nassistant:";
}

export type ChatHandler = (Req: Request) => Promise<Response>;

export function CreateChatHandler(Model: Shahd, Tokenizer: Tokenizer, Config: ResolvedConfig): ChatHandler {
  let Counter = 0;
  return async (Req: Request): Promise<Response> => {
    const Url = new URL(Req.url);
    if (Url.pathname === "/health") return new Response("ok");
    if (Req.method !== "POST") return new Response("method not allowed", { status: 405 });

    let Body: ChatRequest;
    try {
      Body = (await Req.json()) as ChatRequest;
    } catch {
      return Response.json({ error: "invalid JSON body" }, { status: 400 });
    }
    const Prompt = RenderMessages(Body.messages ?? []);
    const MaxNew = Body.max_tokens ?? 128;
    const Temperature = Body.temperature ?? 0.8;
    const Rng = new SeededRng(Config.Training.Seed + Counter++);

    let Completion: string;
    try {
      const Full = GuardedGenerate(Model, Tokenizer, Prompt, MaxNew, { ...DefaultSampling, Temperature }, Rng, Config);
      Completion = Full.startsWith(Prompt) ? Full.slice(Prompt.length) : Full;
    } catch (Err) {
      return Response.json({ error: (Err as Error).message }, { status: 400 });
    }

    if (Body.stream === true) {
      const Stream = new ReadableStream<Uint8Array>({
        start: (Controller: ReadableStreamDefaultController<Uint8Array>): void => {
          const Encoder = new TextEncoder();
          const Chunk = { choices: [{ index: 0, delta: { role: "assistant", content: Completion } }] };
          Controller.enqueue(Encoder.encode(`data: ${JSON.stringify(Chunk)}\n\n`));
          Controller.enqueue(Encoder.encode("data: [DONE]\n\n"));
          Controller.close();
        },
      });
      return new Response(Stream, { headers: { "Content-Type": "text/event-stream" } });
    }

    return Response.json({
      object: "chat.completion",
      choices: [{ index: 0, message: { role: "assistant", content: Completion }, finish_reason: "stop" }],
    });
  };
}

export function StartServer(Model: Shahd, Tokenizer: Tokenizer, Config: ResolvedConfig, Port = 8080): ReturnType<typeof Bun.serve> {
  return Bun.serve({ port: Port, fetch: CreateChatHandler(Model, Tokenizer, Config) });
}

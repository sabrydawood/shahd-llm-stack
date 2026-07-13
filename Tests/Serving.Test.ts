import { test, expect } from "bun:test";
import { ParseToolCall, FormatToolResult, ToolTokens } from "../Brain/Serving/ToolProtocol.ts";
import { DefaultToolRegistry, CalculatorTool } from "../Brain/Serving/Tools.ts";
import { ChatSession } from "../Brain/Serving/ChatSession.ts";
import { RunAgent } from "../Brain/Serving/AgentLoop.ts";
import { CreateChatHandler } from "../Brain/Serving/InferenceServer.ts";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { CreateRngStreams } from "../Brain/Random/SeededRng.ts";
import { CharTokenizer } from "../Brain/Tokenizer/CharTokenizer.ts";
import { Shahd } from "../Brain/Nn/Shahd.ts";

test("ToolProtocol parses a tool call and ignores plain text", () => {
  const Text = `sure ${ToolTokens.CallStart}{"name":"calculator","arguments":{"a":2,"op":"+","b":3}}${ToolTokens.CallEnd}`;
  const Call = ParseToolCall(Text);
  expect(Call?.Name).toBe("calculator");
  expect(Call?.Arguments["a"]).toBe(2);
  expect(ParseToolCall("just a normal answer")).toBe(null);
  expect(FormatToolResult({ result: 5 })).toContain("5");
});

test("Tools execute safely (calculator + registry error handling)", () => {
  expect(CalculatorTool.Execute({ a: 2, op: "*", b: 4 })).toEqual({ result: 8 });
  expect(CalculatorTool.Execute({ a: 1, op: "/", b: 0 })).toEqual({ error: "division by zero" });
  const Registry = DefaultToolRegistry();
  expect(Registry.Run({ Name: "calculator", Arguments: { a: 10, op: "-", b: 3 } })).toEqual({ result: 7 });
  expect(Registry.Run({ Name: "nope", Arguments: {} })).toEqual({ error: "unknown tool: nope" });
});

test("ChatSession renders the running conversation", () => {
  const Session = new ChatSession("You are Shahd");
  Session.AddUser("hi");
  const Prompt = Session.RenderPrompt();
  expect(Prompt).toContain("You are Shahd");
  expect(Prompt).toContain("hi");
});

test("AgentLoop runs a tool call then returns the final answer", () => {
  const Session = new ChatSession("helper");
  Session.AddUser("what is 2 + 3?");
  const Registry = DefaultToolRegistry();
  let Calls = 0;
  const Generate = (): string => {
    Calls++;
    if (Calls === 1) {
      return `${ToolTokens.CallStart}{"name":"calculator","arguments":{"a":2,"op":"+","b":3}}${ToolTokens.CallEnd}`;
    }
    return "The answer is 5.";
  };
  const Result = RunAgent(Session, Generate, Registry);
  expect(Result.ToolCalls.length).toBe(1);
  expect(Result.ToolCalls[0].Name).toBe("calculator");
  expect(Result.FinalText).toContain("5");
  expect(Result.HitStepLimit).toBe(false);
});

test("InferenceServer handler returns an OpenAI-shaped response and /health", async () => {
  const Corpus = "user assistant system helper: hi hello what is the answer 2 + 3 = 5 \n abcdefghijklmnopqrstuvwxyz.,?";
  const Tokenizer = CharTokenizer.FromCorpus(Corpus);
  const Config = LoadConfig({
    Overrides: { Model: { VocabSize: Tokenizer.VocabSize, EmbedDim: 16, NumLayers: 1, NumHeads: 2, BlockSize: 64 } },
    UseCli: false,
    UseEnv: false,
  });
  const Model = new Shahd(Config, CreateRngStreams(Config.Training.Seed).InitRng);
  const Handler = CreateChatHandler(Model, Tokenizer, Config);

  const Req = new Request("http://localhost/v1/chat/completions", {
    method: "POST",
    body: JSON.stringify({ messages: [{ role: "user", content: "hi" }], max_tokens: 5 }),
  });
  const Res = await Handler(Req);
  const Json = (await Res.json()) as { choices: { message: { role: string; content: string } }[] };
  expect(Json.choices[0].message.role).toBe("assistant");
  expect(typeof Json.choices[0].message.content).toBe("string");

  const Health = await Handler(new Request("http://localhost/health"));
  expect(await Health.text()).toBe("ok");
}, 15000);

import { test, expect } from "bun:test";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { ChatSession } from "../Brain/Serving/ChatSession.ts";
import { RunAgent } from "../Brain/Serving/AgentLoop.ts";
import type { AgentStep } from "../Brain/Serving/AgentLoop.ts";
import { BuildTrace, FormatTrace } from "../Brain/Serving/ReasoningTrace.ts";
import { BuildAgentTooling } from "../Brain/Serving/Tools/ToolsBarrel.ts";
import { ChatTokens } from "../Brain/Sft/ChatTemplate.ts";
import { ToolTokens } from "../Brain/Serving/ToolProtocol.ts";

function Tooling() {
  return BuildAgentTooling(LoadConfig({ UseCli: false, UseEnv: false }));
}

test("RunAgent OnStep captures each reasoning step (think -> tool -> finish)", async () => {
  const T = Tooling();
  const Session = new ChatSession("You are Shahd.");
  Session.AddUser("compute 6*7 then finish");
  T.Context.Session = Session;
  const Script = [
    `${ChatTokens.Think}use the calculator${ChatTokens.EndThink}${ToolTokens.CallStart}{"name":"calculator","arguments":{"a":6,"op":"*","b":7}}${ToolTokens.CallEnd}`,
    `${ToolTokens.CallStart}{"name":"finish","arguments":{"answer":"6 * 7 = 42"}}${ToolTokens.CallEnd}`,
  ];
  let Turn = 0;
  const Steps: AgentStep[] = [];
  const Result = await RunAgent(Session, () => Script[Math.min(Turn++, Script.length - 1)]!, T.Registry, T.MaxSteps, T.Context, (S) => Steps.push(S));

  expect(Steps.length).toBe(2);
  expect(Steps[0]!.Call?.Name).toBe("calculator");
  expect(Steps[1]!.Call?.Name).toBe("finish");
  expect(Steps[1]!.Terminal).toBe(true);
  expect(Result.FinalText).toBe("6 * 7 = 42");

  const Lines = BuildTrace(Steps);
  const Kinds = Lines.map((L) => `${L.Kind}:${L.Text.slice(0, 10)}`);
  expect(Kinds.some((K) => K.startsWith("think"))).toBe(true); // reasoning surfaced
  expect(Lines.some((L) => L.Kind === "tool" && L.Text.includes("calculator"))).toBe(true);
  expect(FormatTrace(Steps)).toContain("42"); // calculator's result appears in the trace
});

test("BuildTrace renders a plain-answer turn as a single answer line", async () => {
  const T = Tooling();
  const Session = new ChatSession("You are Shahd.");
  Session.AddUser("say hi");
  const Steps: AgentStep[] = [];
  const Result = await RunAgent(Session, () => "Hello!", T.Registry, T.MaxSteps, T.Context, (S) => Steps.push(S));
  expect(Result.FinalText).toBe("Hello!");
  const Lines = BuildTrace(Steps);
  expect(Lines.length).toBe(1);
  expect(Lines[0]).toMatchObject({ Kind: "answer", Text: "Hello!" });
});

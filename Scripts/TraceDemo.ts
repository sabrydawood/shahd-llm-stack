// See the model's reasoning as a step-by-step trace (Phase 8). Runs the real agent loop with a
// SCRIPTED generator (stands in for a trained model until SFT can make the model emit tool calls
// itself) and prints exactly what it did each step: hidden thinking, tool call + result, final
// answer. The trace instrumentation (RunAgent OnStep -> ReasoningTrace) is identical to what a real
// model uses — swap the scripted generator for GuardedGenerate once an SFT/tool model exists.
//
//   bun run Scripts/TraceDemo.ts

import { LoadConfig } from "../Brain/Config/LoadConfig.ts";
import { ChatSession } from "../Brain/Serving/ChatSession.ts";
import { RunAgent } from "../Brain/Serving/AgentLoop.ts";
import type { AgentStep } from "../Brain/Serving/AgentLoop.ts";
import { FormatTrace } from "../Brain/Serving/ReasoningTrace.ts";
import { BuildAgentTooling, RenderToolManifest } from "../Brain/Serving/Tools/ToolsBarrel.ts";
import { ChatTokens } from "../Brain/Sft/ChatTemplate.ts";
import { ToolTokens } from "../Brain/Serving/ToolProtocol.ts";

const Config = LoadConfig({ UseCli: false, UseEnv: false });
const Tooling = BuildAgentTooling(Config);
const System = "You are Shahd.\n\n" + RenderToolManifest(Tooling.Registry.List());
const Session = new ChatSession(System);
Session.AddUser("What is 12 * 9? Think first, use the calculator, then finish.");
Tooling.Context.Session = Session;

// The scripted "model": reason -> call calculator -> read result -> finish. Real models produce this
// same shape once SFT'd; the trace below is exactly what serving will capture from a real generator.
const Script = [
  `${ChatTokens.Think}The user wants 12 * 9. I'll compute it with the calculator tool.${ChatTokens.EndThink}${ToolTokens.CallStart}{"name":"calculator","arguments":{"a":12,"op":"*","b":9}}${ToolTokens.CallEnd}`,
  `${ChatTokens.Think}The calculator returned 108, so that is the answer. I'll finish.${ChatTokens.EndThink}${ToolTokens.CallStart}{"name":"finish","arguments":{"answer":"12 * 9 = 108"}}${ToolTokens.CallEnd}`,
];
let Turn = 0;
const Steps: AgentStep[] = [];
const Result = await RunAgent(
  Session,
  () => Script[Math.min(Turn++, Script.length - 1)]!,
  Tooling.Registry,
  Tooling.MaxSteps,
  Tooling.Context,
  (Step) => Steps.push(Step),
);

console.log(`user: What is 12 * 9? Think first, use the calculator, then finish.\n`);
console.log("reasoning trace — the exact steps the model executed:");
console.log(FormatTrace(Steps));
console.log(`\nfinal answer: ${Result.FinalText}`);
console.log(`(${Result.Steps} steps; tools: ${Result.ToolCalls.map((C) => C.Name).join(" -> ")})`);

// Multi-step agent loop (Phase 6, extended in Phase 7): generate -> if the model emitted a tool
// call, run it against the injected ToolContext and feed the result back -> repeat, until the model
// produces a plain (non-tool) answer, calls the Terminal `finish` tool, or the step budget is hit.
// Async throughout so tools that await (user_ask, web_search) and async generators compose. The
// step budget + never-throwing registry are the guardrails; `Generate` is injected (GuardedGenerate
// at serving time, a mock in tests) so the loop stays model-agnostic.

import type { ChatSession } from "./ChatSession.ts";
import type { ToolRegistry } from "./Tools/ToolRegistry.ts";
import type { ToolContext } from "./Tools/ToolTypes.ts";
import type { ToolCall } from "./ToolProtocol.ts";
import { ParseToolCall, FormatToolResult } from "./ToolProtocol.ts";
import { StripThinking } from "../Reasoning/ThinkingMode.ts";

// The generator is handed the live ChatSession (not a pre-rendered string) so it renders the prompt
// however its tokenizer requires: a special-token (chat) model MUST render to ids via
// Session.RenderPromptIds (base-encoding untrusted content so it can't smuggle control tokens); tests
// can inspect Session.RenderPrompt()/Messages directly. This keeps the encode step where the tokenizer
// lives, and closes the control-token smuggling that a re-encoded rendered string allowed.
export type Generator = (Session: ChatSession) => string | Promise<string>;

// One observable step of the agent's reasoning — what it generated, whether that was a tool call,
// the tool's result, and whether the step ended the loop. This is the raw material of a thinking
// trace: it exposes exactly what the model did at each step so the process can be watched + improved.
export type AgentStep = {
  Index: number;
  Generated: string; // the raw text the model produced this step
  Call: ToolCall | null; // the parsed tool call, if the step was one
  Result: unknown | null; // the tool's result, if a call ran
  Terminal: boolean; // did this step end the loop (plain answer or a successful finish)
};

export type AgentResult = {
  FinalText: string;
  Steps: number;
  ToolCalls: ToolCall[];
  HitStepLimit: boolean;
};

export async function RunAgent(
  Session: ChatSession,
  Generate: Generator,
  Registry: ToolRegistry,
  MaxSteps = 6,
  Context?: ToolContext,
  OnStep?: (Step: AgentStep) => void, // fired once per loop iteration with the full step detail (trace hook)
): Promise<AgentResult> {
  const ToolCalls: ToolCall[] = [];
  for (let Step = 0; Step < MaxSteps; Step++) {
    const Generated = await Generate(Session);
    const Call = ParseToolCall(Generated);
    if (Call === null) {
      // Plain text is the canonical final answer. Strip the private <|think|> scratchpad before it
      // reaches the user or the conversation history — the raw text stays in the trace step (below) so
      // the reasoning is still observable, but it must never leak into the visible reply or context.
      const Answer = StripThinking(Generated);
      Session.AddAssistant(Answer);
      OnStep?.({ Index: Step, Generated, Call: null, Result: null, Terminal: true });
      return { FinalText: Answer, Steps: Step + 1, ToolCalls, HitStepLimit: false };
    }
    ToolCalls.push(Call);
    Session.AddAssistant(Generated); // record the tool-call turn
    const Result = await Registry.Run(Call, Context); // never throws
    Session.AddToolResult(FormatToolResult(Result));
    // The one other terminal path: a successful `finish` (Terminal) tool call ends the loop.
    const Tool = Registry.Get(Call.Name);
    const Terminal = Tool?.Terminal === true && !("error" in Result);
    OnStep?.({ Index: Step, Generated, Call, Result, Terminal });
    if (Terminal) {
      const Answer = typeof Result["answer"] === "string" ? Result["answer"] : StripThinking(Generated);
      return { FinalText: Answer, Steps: Step + 1, ToolCalls, HitStepLimit: false };
    }
  }
  return { FinalText: "(step budget exhausted)", Steps: MaxSteps, ToolCalls, HitStepLimit: true };
}

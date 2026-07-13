// Multi-step agent loop (Phase 6): generate -> if the model emitted a tool call, run it and feed
// the result back -> repeat, until the model produces a final (non-tool) answer or the step budget
// is hit. The step budget + never-throwing tool runner are the guardrails. `Generator` is injected
// (the real GuardedGenerate at serving time, a mock in tests) so the loop is model-agnostic.

import type { ChatSession } from "./ChatSession.ts";
import type { ToolRegistry } from "./Tools.ts";
import type { ToolCall } from "./ToolProtocol.ts";
import { ParseToolCall, FormatToolResult } from "./ToolProtocol.ts";

export type Generator = (Prompt: string) => string;

export type AgentResult = {
  FinalText: string;
  Steps: number;
  ToolCalls: ToolCall[];
  HitStepLimit: boolean;
};

export function RunAgent(
  Session: ChatSession,
  Generate: Generator,
  Registry: ToolRegistry,
  MaxSteps = 6,
): AgentResult {
  const ToolCalls: ToolCall[] = [];
  for (let Step = 0; Step < MaxSteps; Step++) {
    const Generated = Generate(Session.RenderPrompt());
    const Call = ParseToolCall(Generated);
    if (Call === null) {
      Session.AddAssistant(Generated);
      return { FinalText: Generated, Steps: Step + 1, ToolCalls, HitStepLimit: false };
    }
    ToolCalls.push(Call);
    Session.AddAssistant(Generated); // record the tool-call turn
    const Result = Registry.Run(Call); // never throws
    Session.AddToolResult(FormatToolResult(Result));
  }
  return { FinalText: "(step budget exhausted)", Steps: MaxSteps, ToolCalls, HitStepLimit: true };
}

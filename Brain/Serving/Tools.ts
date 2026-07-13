// Tool registry + built-in tools (Phase 6). Tools are the agent's hands: a name, a description,
// and a synchronous Execute. Built-ins here are safe by construction (no eval); the run_code tool
// routes through the sandboxed CodeExecutor.

import type { ToolCall } from "./ToolProtocol.ts";
import { RunCode } from "../Eval/CodeExecutor.ts";

export type Tool = {
  Name: string;
  Description: string;
  Execute: (Arguments: Record<string, unknown>) => unknown;
};

export class ToolRegistry {
  private Tools = new Map<string, Tool>();

  Register(Tool: Tool): void {
    this.Tools.set(Tool.Name, Tool);
  }

  Get(Name: string): Tool | undefined {
    return this.Tools.get(Name);
  }

  List(): Tool[] {
    return [...this.Tools.values()];
  }

  /** Run a parsed tool call; returns the tool's result or an error object (never throws). */
  Run(Call: ToolCall): unknown {
    const Tool = this.Tools.get(Call.Name);
    if (Tool === undefined) return { error: `unknown tool: ${Call.Name}` };
    try {
      return Tool.Execute(Call.Arguments);
    } catch (Err) {
      return { error: (Err as Error).message };
    }
  }
}

// A safe calculator (no eval): { a, op, b } arithmetic.
export const CalculatorTool: Tool = {
  Name: "calculator",
  Description: "Arithmetic on two numbers. Args: { a: number, op: '+'|'-'|'*'|'/', b: number }",
  Execute: (Arguments) => {
    const A = Number(Arguments["a"]);
    const B = Number(Arguments["b"]);
    const Op = String(Arguments["op"]);
    if (!Number.isFinite(A) || !Number.isFinite(B)) return { error: "a and b must be numbers" };
    switch (Op) {
      case "+": return { result: A + B };
      case "-": return { result: A - B };
      case "*": return { result: A * B };
      case "/": return B === 0 ? { error: "division by zero" } : { result: A / B };
      default: return { error: `unknown op: ${Op}` };
    }
  },
};

// Run code in the sandbox. Args: { code: string, timeoutMs?: number }.
export const RunCodeTool: Tool = {
  Name: "run_code",
  Description: "Execute code in a sandbox. Args: { code: string, timeoutMs?: number }",
  Execute: (Arguments) => {
    const Code = String(Arguments["code"] ?? "");
    const TimeoutMs = Number(Arguments["timeoutMs"] ?? 5000);
    const Result = RunCode(Code, Number.isFinite(TimeoutMs) ? TimeoutMs : 5000);
    return { passed: Result.Passed, exitCode: Result.ExitCode, stdout: Result.Stdout, stderr: Result.Stderr };
  },
};

export function DefaultToolRegistry(): ToolRegistry {
  const Registry = new ToolRegistry();
  Registry.Register(CalculatorTool);
  Registry.Register(RunCodeTool);
  return Registry;
}

// Tool-calling protocol (Phase 6). The model signals a tool call by emitting a JSON object between
// sentinel tokens; the host parses it, runs the tool, and feeds the result back as another sentinel
// block. This is a MECHANISM that works at any model scale — the model's competence at CHOOSING the
// right tool/args is the scale-dependent part (CAPABILITIES.md).

export const ToolTokens = {
  CallStart: "<|tool_call|>",
  CallEnd: "<|end_tool_call|>",
  ResultStart: "<|tool_result|>",
  ResultEnd: "<|end_tool_result|>",
} as const;

export const ToolTokenList: readonly string[] = Object.values(ToolTokens);

export type ToolCall = { Name: string; Arguments: Record<string, unknown> };

/** Parse the first tool call (JSON with a "name") from generated text, or null if none/invalid. */
export function ParseToolCall(Text: string): ToolCall | null {
  const Start = Text.indexOf(ToolTokens.CallStart);
  if (Start === -1) return null;
  const From = Start + ToolTokens.CallStart.length;
  const End = Text.indexOf(ToolTokens.CallEnd, From);
  const Json = Text.slice(From, End === -1 ? undefined : End).trim();
  try {
    const Parsed = JSON.parse(Json) as { name?: unknown; arguments?: unknown };
    if (typeof Parsed.name !== "string") return null;
    const Arguments = typeof Parsed.arguments === "object" && Parsed.arguments !== null
      ? (Parsed.arguments as Record<string, unknown>)
      : {};
    return { Name: Parsed.name, Arguments };
  } catch {
    return null;
  }
}

/** Format a tool result block to append to the conversation. */
export function FormatToolResult(Result: unknown): string {
  return `${ToolTokens.ResultStart}${JSON.stringify(Result)}${ToolTokens.ResultEnd}`;
}

// Fully-owned synthetic SFT conversations (Phase 8). Zero external data — every conversation is
// generated deterministically here, so it is license-clean and reproducible. These teach the CHAT
// FORMAT, tool-CALLING, and a thinking scaffold (mechanism, not knowledge — a tiny model can't hold
// much). Built from the REAL SFT infra (ChatMessage + BuildToolConversation + WrapThinking) so it
// feeds RenderForTraining/SftStep unchanged. External permissive text (OASST, Gutenberg…) is layered
// in via data collection later; this is the owned core that makes the model reply + call tools.

import type { ChatMessage } from "./ChatTemplate.ts";
import type { ToolExemplar } from "./ToolUseExamples.ts";
import { BuildToolConversation, ToolUseExemplars } from "./ToolUseExamples.ts";
import { WrapThinkingAnswer, DefaultThinkingSystemPrompt } from "../Reasoning/AnswerExtraction.ts";
import type { SeededRng } from "../Random/SeededRng.ts";

const System = "You are Shahd, a helpful coding assistant.";

// A fixed persona so the model has one consistent voice to imitate.
const Persona: readonly [string, string][] = [
  ["hi", "Hello! How can I help you with your code today?"],
  ["hello", "Hi there! What would you like help with?"],
  ["hey", "Hey! Ask me a question or give me a coding task."],
  ["who are you?", "I'm Shahd, a small coding assistant. Ask me a question or give me a task."],
  ["what can you do?", "I can answer questions, do calculations with my tools, and help with small coding tasks."],
  ["thanks", "You're welcome!"],
  ["thank you", "Happy to help!"],
  ["bye", "Goodbye! Come back anytime."],
  ["goodbye", "Goodbye! See you later."],
  ["quit", "Goodbye! See you later."],
  ["exit", "Goodbye! See you later."],
  ["stop", "Goodbye! See you later."],
];

const Ops: readonly [string, (A: number, B: number) => number][] = [
  ["+", (A, B): number => A + B],
  ["-", (A, B): number => A - B],
  ["*", (A, B): number => A * B],
  ["/", (A, B): number => B === 0 ? 0 : A / B],
  ["^", (A, B): number => A ** B],
  ["%", (A, B): number => B === 0 ? 0 : A % B],
];

function Simple(User: string, Assistant: string): ChatMessage[] {
  return [{ Role: "System", Content: System }, { Role: "User", Content: User }, { Role: "Assistant", Content: Assistant }];
}

// Arithmetic taught as a TOOL CALL (not a memorized answer) — the model learns to reach for the
// calculator, exactly the learned-tool behavior the agent loop expects.
function ArithmeticToolConversation(Rng: SeededRng): ChatMessage[] {
  const A = 1 + Math.floor(Rng.NextFloat() * 99);
  const B = 1 + Math.floor(Rng.NextFloat() * 99);
  const [Op, Fn] = Ops[Math.floor(Rng.NextFloat() * Ops.length)]!;
  const Result = Fn(A, B);
  const Exemplar: ToolExemplar = {
    User: `What is ${A} ${Op} ${B}?`,
    Call: { name: "calculator", arguments: { a: A, op: Op, b: B } },
    Result: { result: Result },
    Answer: `${A} ${Op} ${B} = ${Result}.`,
  };
  return BuildToolConversation(Exemplar, System);
}

// A visible think -> answer example in the CANONICAL <|think|>…<|endthink|><answer>…</answer> format
// the serving extractor parses (so training and inference agree). Varied across ALL operations and
// 2-digit operands — teaching the think-then-answer SCAFFOLD generally, not one memorized single-digit
// addition template. Uses DefaultThinkingSystemPrompt so the model is asked at serving to do exactly
// what it was taught here.
function ThinkingConversation(Rng: SeededRng): ChatMessage[] {
  const A = 2 + Math.floor(Rng.NextFloat() * 97);
  const B = 2 + Math.floor(Rng.NextFloat() * 97);
  const [Op, Fn] = Ops[Math.floor(Rng.NextFloat() * Ops.length)]!;
  const Result = Fn(A, B);
  const Reasoning = `The problem is ${A} ${Op} ${B}. Applying ${Op} to ${A} and ${B} gives ${Result}.`;
  const Assistant = WrapThinkingAnswer(Reasoning, String(Result));
  return [
    { Role: "System", Content: DefaultThinkingSystemPrompt },
    { Role: "User", Content: `What is ${A} ${Op} ${B}? Think first.` },
    { Role: "Assistant", Content: Assistant },
  ];
}

export type CodeSample = { Lang: string; Content: string };

// Language identification over a REAL snippet — a truthful, corpus-grounded task.
function CodeLangConversation(Sample: CodeSample): ChatMessage[] {
  const Snippet = Sample.Content.split("\n").slice(0, 12).join("\n").slice(0, 480).trimEnd();
  return Simple(`What programming language is this code written in?\n\n${Snippet}`, Sample.Lang);
}

export type OwnedSftOptions = { ArithmeticCount?: number; ThinkingCount?: number; PersonaRepeats?: number; MaxCodeConversations?: number };

/** Build the full owned SFT conversation set (each item is one system/user/assistant conversation).
 *  Deterministic given the same Rng + code samples. */
export function BuildOwnedConversations(CodeSamples: CodeSample[], Rng: SeededRng, Options: OwnedSftOptions = {}): ChatMessage[][] {
  const ArithmeticCount = Options.ArithmeticCount ?? 200;
  const ThinkingCount = Options.ThinkingCount ?? 100;
  const PersonaRepeats = Options.PersonaRepeats ?? 20;
  const MaxCodeConversations = Options.MaxCodeConversations ?? 1500;

  const Out: ChatMessage[][] = [];
  for (let R = 0; R < PersonaRepeats; R++) for (const [User, Assistant] of Persona) Out.push(Simple(User, Assistant));
  for (let I = 0; I < ArithmeticCount; I++) Out.push(ArithmeticToolConversation(Rng));
  for (let I = 0; I < ThinkingCount; I++) Out.push(ThinkingConversation(Rng));
  for (const Exemplar of ToolUseExemplars) for (let R = 0; R < 10; R++) Out.push(BuildToolConversation(Exemplar, System));
  let Added = 0;
  for (const Sample of CodeSamples) {
    if (Added >= MaxCodeConversations) break;
    if (Sample.Lang === "unknown" || Sample.Content.trim().length < 60) continue;
    Out.push(CodeLangConversation(Sample));
    Added++;
  }
  return Out;
}

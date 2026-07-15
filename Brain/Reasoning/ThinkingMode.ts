// Thinking mode (Phase 7): a model can emit private chain-of-thought between the <|think|> …
// <|endthink|> sentinels, then its visible answer. Reuses ChatTokens.Think/EndThink from
// ChatTemplate (rule #4: those sentinels have one home — do not redefine them). These helpers split
// a generation into hidden reasoning vs the user-facing answer so serving can hide the scratchpad.

import { ChatTokens } from "../Sft/ChatTemplate.ts";

export type SplitThought = { Thinking: string; Answer: string; HadThinking: boolean };

/** Separate a generation into its <|think|>…<|endthink|> block and the visible answer. */
export function SplitThinking(Text: string): SplitThought {
  const Start = Text.indexOf(ChatTokens.Think);
  if (Start === -1) {
    return { Thinking: "", Answer: Text.trim(), HadThinking: false };
  }
  const End = Text.indexOf(ChatTokens.EndThink, Start + ChatTokens.Think.length);
  if (End === -1) {
    // Unclosed thinking (generation cut off mid-scratchpad — common for a small model / small token
    // budget). Everything from <|think|> onward is incomplete reasoning: HIDE it, never leak it as the
    // answer. The visible answer is only whatever preceded the (now-dangling) think sentinel.
    const Thinking = Text.slice(Start + ChatTokens.Think.length).trim();
    const Answer = Text.slice(0, Start).trim();
    return { Thinking, Answer, HadThinking: true };
  }
  const Thinking = Text.slice(Start + ChatTokens.Think.length, End).trim();
  const Answer = (Text.slice(0, Start) + Text.slice(End + ChatTokens.EndThink.length)).trim();
  return { Thinking, Answer, HadThinking: true };
}

/** Just the user-facing answer (thinking removed). */
export function StripThinking(Text: string): string {
  return SplitThinking(Text).Answer;
}

/** Wrap reasoning + answer into the canonical think/answer format (for SFT targets or prompting). */
export function WrapThinking(Thinking: string, Answer: string): string {
  return `${ChatTokens.Think}${Thinking}${ChatTokens.EndThink}${Answer}`;
}

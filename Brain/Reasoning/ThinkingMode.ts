// Thinking mode (Phase 7): a model can emit private chain-of-thought between the <|think|> …
// <|endthink|> sentinels, then its visible answer. Reuses ChatTokens.Think/EndThink from
// ChatTemplate (rule #4: those sentinels have one home — do not redefine them). These helpers split
// a generation into hidden reasoning vs the user-facing answer so serving can hide the scratchpad.

import { ChatTokens } from "../Sft/ChatTemplate.ts";

export type SplitThought = { Thinking: string; Answer: string; HadThinking: boolean };

/** Separate a generation into its <|think|>…<|endthink|> block(s) and the visible answer. Walks the
 *  WHOLE string with a cursor so a second (or later) think block in the same generation is stripped
 *  too, not spliced verbatim into the answer. */
export function SplitThinking(Text: string): SplitThought {
  const ThinkingParts: string[] = [];
  const AnswerParts: string[] = [];
  let Cursor = 0;
  let HadThinking = false;
  while (true) {
    const Start = Text.indexOf(ChatTokens.Think, Cursor);
    if (Start === -1) {
      AnswerParts.push(Text.slice(Cursor));
      break;
    }
    HadThinking = true;
    AnswerParts.push(Text.slice(Cursor, Start));
    const End = Text.indexOf(ChatTokens.EndThink, Start + ChatTokens.Think.length);
    if (End === -1) {
      // Unclosed thinking (generation cut off mid-scratchpad — common for a small model / small token
      // budget). Everything from <|think|> onward is incomplete reasoning: HIDE it, never leak it as
      // the answer. Nothing after a dangling think sentinel can be a visible span.
      ThinkingParts.push(Text.slice(Start + ChatTokens.Think.length));
      Cursor = Text.length;
      break;
    }
    ThinkingParts.push(Text.slice(Start + ChatTokens.Think.length, End));
    Cursor = End + ChatTokens.EndThink.length;
  }
  return {
    Thinking: ThinkingParts.join("\n\n").trim(),
    Answer: AnswerParts.join("").trim(),
    HadThinking,
  };
}

/** Just the user-facing answer (thinking removed). */
export function StripThinking(Text: string): string {
  return SplitThinking(Text).Answer;
}

/** Wrap reasoning + answer into the canonical think/answer format (for SFT targets or prompting). */
export function WrapThinking(Thinking: string, Answer: string): string {
  return `${ChatTokens.Think}${Thinking}${ChatTokens.EndThink}${Answer}`;
}

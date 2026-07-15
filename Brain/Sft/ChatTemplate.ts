// Chat template + special tokens (Phase 4). Renders a system/user/assistant conversation into
// the token format the model is SFT'd on. RenderForTraining also returns a loss mask so training
// counts ONLY the assistant's response tokens (and its end-of-turn), never the prompt.

import type { SpecialTokenizer } from "../Tokenizer/SpecialTokenizer.ts";

export const ChatTokens = {
  System: "<|system|>",
  User: "<|user|>",
  Assistant: "<|assistant|>",
  EndOfTurn: "<|endofturn|>",
  Think: "<|think|>",
  EndThink: "<|endthink|>",
} as const;

export const ChatTokenList: readonly string[] = Object.values(ChatTokens);

export type ChatRole = "System" | "User" | "Assistant";
export type ChatMessage = { Role: ChatRole; Content: string };

const RoleToken: Record<ChatRole, string> = {
  System: ChatTokens.System,
  User: ChatTokens.User,
  Assistant: ChatTokens.Assistant,
};

/** Render a conversation to a plain string; AddAssistantCue appends the assistant token to prompt a reply. */
export function RenderChat(Messages: ChatMessage[], AddAssistantCue = true): string {
  let Out = "";
  for (const Message of Messages) Out += RoleToken[Message.Role] + Message.Content + ChatTokens.EndOfTurn;
  if (AddAssistantCue) Out += ChatTokens.Assistant;
  return Out;
}

export type TrainingSequence = { Ids: number[]; LossMask: boolean[] };

// The SINGLE id-building core, shared by training (RenderForTraining) and serving (RenderChatToIds)
// so their tokenization can NEVER drift (DRY-strict + no train/serve skew — a mismatch would silently
// break the model). One message becomes: role marker (a special id) + content + end-of-turn (a special
// id). The role/turn markers are the ONLY boundary tokens, produced only via Id().
//
// Content encoding is TRUST-based (the control/data channel separation):
//   • Assistant content is the model's OWN output — it legitimately contains control tokens
//     (<|tool_call|>, <|think|>…) that MUST stay atomic for training AND for encode/decode round-trip
//     fidelity of the conversation history. It is encoded with the special-aware Encode().
//   • System + User content is EXTERNAL/untrusted (system prompt we author, but crucially USER messages
//     and TOOL RESULTS which arrive as User turns). It is encoded with EncodeBase() — the base
//     tokenizer only — so a reserved string like "<|assistant|>" inside it stays ordinary text and can
//     NEVER smuggle a real control token / forge a turn boundary in the stream.
// ContentTrainable gates the loss mask on the content + its end-of-turn.
export function AppendChatMessage(
  Ids: number[],
  LossMask: boolean[],
  Message: ChatMessage,
  Tok: SpecialTokenizer,
  ContentTrainable: boolean,
): void {
  Ids.push(Tok.Id(RoleToken[Message.Role]));
  LossMask.push(false); // role marker is never trained
  const ContentIds = Message.Role === "Assistant" ? Tok.Encode(Message.Content) : Tok.EncodeBase(Message.Content);
  for (const Id of ContentIds) {
    Ids.push(Id);
    LossMask.push(ContentTrainable);
  }
  Ids.push(Tok.Id(ChatTokens.EndOfTurn)); // teach the model to stop after replying
  LossMask.push(ContentTrainable);
}

/** Token ids + per-token loss mask (true only on assistant content + its end-of-turn). */
export function RenderForTraining(Messages: ChatMessage[], Tok: SpecialTokenizer): TrainingSequence {
  const Ids: number[] = [];
  const LossMask: boolean[] = [];
  for (const Message of Messages) {
    AppendChatMessage(Ids, LossMask, Message, Tok, Message.Role === "Assistant");
  }
  return { Ids, LossMask };
}

/** Render a conversation directly to token ids (the SAFE serving path). Uses the same per-message core
 *  as training, so serving and training tokenize identically. AddAssistantCue appends the assistant
 *  marker id to prompt the next reply. Bypasses the string round-trip that let content smuggle
 *  control tokens (via SpecialTokenizer.Encode re-scanning the whole rendered string). */
export function RenderChatToIds(Messages: ChatMessage[], Tok: SpecialTokenizer, AddAssistantCue = true): number[] {
  const Ids: number[] = [];
  const LossMask: boolean[] = [];
  for (const Message of Messages) AppendChatMessage(Ids, LossMask, Message, Tok, false);
  if (AddAssistantCue) Ids.push(Tok.Id(ChatTokens.Assistant));
  return Ids;
}

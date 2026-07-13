// Multi-turn chat state (Phase 6). Holds the running conversation and renders the model prompt via
// the chat template. Tool results are appended as user-visible turns so the model can read them.

import type { ChatMessage } from "../Sft/ChatTemplate.ts";
import { RenderChat } from "../Sft/ChatTemplate.ts";

export class ChatSession {
  Messages: ChatMessage[] = [];

  constructor(SystemPrompt?: string) {
    if (SystemPrompt !== undefined) this.Messages.push({ Role: "System", Content: SystemPrompt });
  }

  AddUser(Content: string): void {
    this.Messages.push({ Role: "User", Content });
  }

  AddAssistant(Content: string): void {
    this.Messages.push({ Role: "Assistant", Content });
  }

  /** Feed a tool result back into the conversation (as a user turn the model reads next). */
  AddToolResult(Content: string): void {
    this.Messages.push({ Role: "User", Content });
  }

  /** Render the prompt string for the model, cueing it to produce the next assistant turn. */
  RenderPrompt(): string {
    return RenderChat(this.Messages, true);
  }
}

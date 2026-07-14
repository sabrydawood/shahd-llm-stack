// Render a chat message list into the flat prompt the base model continues. Shared by the
// OpenAI-compatible InferenceServer and the dashboard chat so the format lives in one place. The
// model is a base LM (no chat SFT yet), so this is a simple role-tagged transcript ending in
// "assistant:"; the serving cap trims the oldest tokens when the transcript exceeds the context.

export type RenderableMessage = { role: string; content: string };

export function RenderMessages(Messages: RenderableMessage[]): string {
  return Messages.map((M) => `${M.role}: ${M.content}`).join("\n") + "\nassistant:";
}

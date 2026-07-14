// Postgres-backed ChatStore: conversations + messages live in the SAME database as the corpus, so
// chat memory is durable, traceable, and synced (not just runtime state). Uses raw postgres-js with
// its own connection; tables are created on first use (idempotent). Same interface as the in-memory
// store, so the dashboard/tests are unaffected by which one is wired.

import postgres from "postgres";
import type { ChatStore, ConversationSummary, ChatMessage } from "./ChatStore.ts";

type ConvRow = { id: string; title: string; updated_at: string };
type MsgRow = { role: string; content: string };

export class PostgresChatStore implements ChatStore {
  private Sql: ReturnType<typeof postgres>;
  private Ready: Promise<void>;

  constructor(Url: string) {
    this.Sql = postgres(Url);
    // Swallow the rejection here so a DB blip at startup can't become an unhandled rejection (which
    // Bun turns into a process exit). The real failure still surfaces when a method awaits a query.
    this.Ready = this.Migrate().catch((Caught) => {
      console.warn(`PostgresChatStore: migration deferred: ${(Caught as Error).message}`);
    });
  }

  private async Migrate(): Promise<void> {
    await this.Sql`CREATE TABLE IF NOT EXISTS chat_conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)`;
    await this.Sql`CREATE TABLE IF NOT EXISTS chat_messages (id BIGSERIAL PRIMARY KEY, conv_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL)`;
    await this.Sql`CREATE INDEX IF NOT EXISTS chat_messages_conv ON chat_messages (conv_id, id)`;
  }

  async CreateConversation(Id: string, Title: string, At: string): Promise<void> {
    await this.Ready;
    await this.Sql`INSERT INTO chat_conversations (id, title, created_at, updated_at) VALUES (${Id}, ${Title}, ${At}, ${At}) ON CONFLICT (id) DO NOTHING`;
  }

  async ListConversations(): Promise<ConversationSummary[]> {
    await this.Ready;
    const Rows = (await this.Sql`SELECT id, title, updated_at FROM chat_conversations ORDER BY updated_at DESC`) as unknown as ConvRow[];
    return Rows.map((R) => ({ Id: R.id, Title: R.title, UpdatedAt: R.updated_at }));
  }

  async GetMessages(ConvId: string): Promise<ChatMessage[]> {
    await this.Ready;
    const Rows = (await this.Sql`SELECT role, content FROM chat_messages WHERE conv_id = ${ConvId} ORDER BY id`) as unknown as MsgRow[];
    return Rows.map((R) => ({ Role: R.role === "assistant" ? "assistant" : "user", Content: R.content }));
  }

  async AddMessage(ConvId: string, Role: "user" | "assistant", Content: string, At: string): Promise<void> {
    await this.Ready;
    await this.Sql`INSERT INTO chat_messages (conv_id, role, content, created_at) VALUES (${ConvId}, ${Role}, ${Content}, ${At})`;
    await this.Sql`UPDATE chat_conversations SET updated_at = ${At} WHERE id = ${ConvId}`;
  }

  async DeleteConversation(ConvId: string): Promise<void> {
    await this.Ready;
    await this.Sql`DELETE FROM chat_messages WHERE conv_id = ${ConvId}`;
    await this.Sql`DELETE FROM chat_conversations WHERE id = ${ConvId}`;
  }

  async Close(): Promise<void> {
    await this.Sql.end();
  }
}

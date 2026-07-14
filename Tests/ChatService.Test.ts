import { test, expect } from "bun:test";
import { ChatService, InMemoryChatStore } from "../Foundry/FoundryBarrel.ts";
import type { ChatMessage } from "../Foundry/FoundryBarrel.ts";

// Fix (adversarial review): concurrent turns on the SAME conversation must serialize, so the second
// turn sees the first's persisted messages and storage order is not interleaved.
test("ChatService serializes concurrent turns on one conversation; interleaves across conversations", async () => {
  const Store = new InMemoryChatStore();
  let Active = 0;
  let MaxActivePerConv = 0;
  const SeenContexts: string[][] = [];
  const Stream = async (Messages: ChatMessage[], _Opts: unknown, _OnDelta: (D: string) => void): Promise<string> => {
    Active++;
    MaxActivePerConv = Math.max(MaxActivePerConv, Active);
    SeenContexts.push(Messages.map((M) => M.Role + ":" + M.Content));
    await new Promise((R) => setTimeout(R, 8));
    Active--;
    return "reply-" + Messages[Messages.length - 1].Content;
  };
  const Svc = new ChatService(Store, Stream);

  // Two turns fired concurrently on the SAME conversation.
  await Promise.all([
    Svc.Turn("conv", "A", { Temperature: 0.5, MaxTokens: 4 }, () => undefined),
    Svc.Turn("conv", "B", { Temperature: 0.5, MaxTokens: 4 }, () => undefined),
  ]);

  expect(MaxActivePerConv).toBe(1); // never two generations at once for the same conversation
  const Msgs = await Store.GetMessages("conv");
  expect(Msgs.map((M) => M.Role)).toEqual(["user", "assistant", "user", "assistant"]); // paired, not interleaved
  // The second turn's context must include the first turn's completed user+assistant messages.
  const Second = SeenContexts.find((C) => C.length === 3);
  expect(Second).toBeDefined();
});

test("ChatService runs turns on DIFFERENT conversations concurrently", async () => {
  const Store = new InMemoryChatStore();
  let Active = 0;
  let MaxActive = 0;
  const Stream = async (_Messages: ChatMessage[], _Opts: unknown, _OnDelta: (D: string) => void): Promise<string> => {
    Active++;
    MaxActive = Math.max(MaxActive, Active);
    await new Promise((R) => setTimeout(R, 8));
    Active--;
    return "ok";
  };
  const Svc = new ChatService(Store, Stream);
  await Promise.all([
    Svc.Turn("c1", "hi", { Temperature: 0.5, MaxTokens: 4 }, () => undefined),
    Svc.Turn("c2", "hi", { Temperature: 0.5, MaxTokens: 4 }, () => undefined),
  ]);
  expect(MaxActive).toBe(2); // distinct conversations are not serialized against each other
});

import { test, expect } from "bun:test";
import { gzipSync } from "node:zlib";
import { CreateOasstProvider } from "../Foundry/OasstProvider.ts";
import type { SourceInput } from "../Foundry/FoundryBarrel.ts";
import { ScoreCodeQuality } from "../Brain/Data/QualityFilter.ts";
import { ClassifyDocument } from "../Foundry/FoundryBarrel.ts";

function MockGz(Trees: object[]): Uint8Array {
  return new Uint8Array(gzipSync(Buffer.from(Trees.map((T) => JSON.stringify(T)).join("\n"), "utf8")));
}

// One tree: an English exchange whose assistant reply has a nested Arabic follow-up exchange.
const Tree = {
  prompt: {
    text: "Hello", role: "prompter", lang: "en", message_id: "m1",
    replies: [{
      text: "Hi there!", role: "assistant", lang: "en", message_id: "a1",
      replies: [{
        text: "ما هي عاصمة مصر؟", role: "prompter", lang: "ar", message_id: "m2",
        replies: [{ text: "عاصمة مصر هي القاهرة.", role: "assistant", lang: "ar", message_id: "a2" }],
      }],
    }],
  },
};

test("OASST provider extracts prompter->assistant conversations as permissive documents", async () => {
  const Gz = MockGz([Tree]);
  const Batches: SourceInput[] = [];
  const Provider = CreateOasstProvider({ FetchBytes: async () => Gz, OnRepoReady: async (_S, Docs) => { Batches.push(...Docs); } });
  const Ret = await Provider.Fetch("all", 100);
  expect(Ret.length).toBe(0); // streaming mode returns [] (stored via OnRepoReady)
  expect(Batches.length).toBe(2); // en + ar exchanges
  const En = Batches.find((D) => D.Lang === "text-en")!;
  expect(En.Content).toBe("User: Hello\n\nAssistant: Hi there!");
  expect(En).toMatchObject({ License: "Apache-2.0", Origin: "web-permissive" });
  expect(Batches.find((D) => D.Lang === "text-ar")!.Content).toContain("القاهرة");
});

test("OASST language filter keeps only the requested language (batch mode returns docs)", async () => {
  const Provider = CreateOasstProvider({ FetchBytes: async () => MockGz([Tree]) }); // no OnRepoReady -> returns docs
  const Docs = await Provider.Fetch("ar", 100);
  expect(Docs.length).toBe(1);
  expect(Docs[0]!.Lang).toBe("text-ar");
});

test("quality filter no longer rejects Arabic prose as binary (Unicode-aware)", () => {
  const Arabic = "مرحبا بك في شهد. أنا مساعد برمجي صغير أتعلم أن أتحدث معك وأكتب الأكواد. كيف يمكنني مساعدتك اليوم؟\n".repeat(3);
  const Q = ScoreCodeQuality(Arabic);
  expect(Q.Passed).toBe(true); // was rejected before (non-ASCII counted as non-printable "binary")
  // and so an Apache-2.0 Arabic conversation is training-eligible (Filtered), not Rejected
  expect(ClassifyDocument("Apache-2.0", Arabic, "web-permissive").Tier).toBe("Filtered");
});

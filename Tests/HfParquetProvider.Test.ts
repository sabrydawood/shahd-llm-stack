import { test, expect } from "bun:test";
import { CreateHfParquetProvider, WikiDumpSource, StackExchangeSource } from "../Foundry/HfParquetProvider.ts";
import type { ParquetRow, SourceInput, AsyncBuffer } from "../Foundry/FoundryBarrel.ts";

// A fake shard row; ReadRows slices an array of these like hyparquet's rowStart/rowEnd would.
const Row = (I: number): ParquetRow => ({ id: I, title: `Title ${I}`, text: "A long enough article body sentence. ".repeat(10) });

test("WikiDumpSource.MapRow builds a knowledge doc and skips stubs", () => {
  const Doc = WikiDumpSource.MapRow({ id: 7, title: "Cairo", text: "Cairo is the capital of Egypt. ".repeat(10) }, "simple")!;
  expect(Doc).not.toBeNull();
  expect(Doc.Content).toContain("Cairo");
  expect(Doc.Lang).toBe("text-simple");
  expect(Doc.Provenance).toBe("wikipedia:simple:7");
  expect(WikiDumpSource.MapRow({ title: "Stub", text: "short" }, "simple")).toBeNull(); // below MinChars
});

test("StackExchangeSource.MapRow builds a Q&A conversation turn and skips trivial pairs", () => {
  const Doc = StackExchangeSource.MapRow({ INSTRUCTION: "How do I reverse a list in Python?", RESPONSE: "Use list.reverse() to reverse in place, or reversed(list) for an iterator.", SOURCE: "stackexchange-stackoverflow" }, "all")!;
  expect(Doc).not.toBeNull();
  expect(Doc.Content).toContain("User: How do I reverse");
  expect(Doc.Content).toContain("Assistant: Use list.reverse()");
  expect(Doc.Provenance).toBe("stackexchange:stackexchange-stackoverflow");
  expect(StackExchangeSource.MapRow({ INSTRUCTION: "hi", RESPONSE: "yo" }, "all")).toBeNull(); // too trivial
  expect(StackExchangeSource.Kind).toBe("conversation");
});

test("parquet provider streams a full shard and advances the cursor to the next shard", async () => {
  const Shards = [[Row(0), Row(1), Row(2), Row(3), Row(4)]]; // 5 rows, one shard
  const Cursors: { Shard: number; Offset: number }[] = [];
  const Batches: SourceInput[] = [];
  const Provider = CreateHfParquetProvider(WikiDumpSource, {
    ListShards: async () => ["s0"],
    OpenShard: async () => ({ byteLength: 8, slice: async () => new ArrayBuffer(8) }),
    ReadRows: async (_F, Start, End) => Shards[0]!.slice(Start, End),
    WindowRows: 2,
    OnCursor: (Shard, Offset) => Cursors.push({ Shard, Offset }),
    OnRepoReady: async (_S, Docs) => { Batches.push(...Docs); },
  });
  expect(Provider.Semantics).toBe("streaming");
  await Provider.Fetch("simple", 1000);
  expect(Batches.length).toBe(5); // all rows of the shard
  expect(Cursors.at(-1)).toEqual({ Shard: 1, Offset: 0 }); // shard finished -> next shard, offset reset
});

test("parquet provider resumes from a cursor and caps mid-shard without skipping rows", async () => {
  const Data = Array.from({ length: 10 }, (_, I) => Row(I));
  const Read = async (_F: AsyncBuffer, Start: number, End: number): Promise<ParquetRow[]> => Data.slice(Start, End);
  const Cursors: { Shard: number; Offset: number }[] = [];
  const Batches: SourceInput[] = [];
  const Provider = CreateHfParquetProvider(WikiDumpSource, {
    ListShards: async () => ["s0"],
    OpenShard: async () => ({ byteLength: 8, slice: async () => new ArrayBuffer(8) }),
    ReadRows: Read,
    WindowRows: 2,
    StartOffset: 4, // resume mid-shard
    MaxPerRun: 3, // cap after 3 docs
    OnCursor: (Shard, Offset) => Cursors.push({ Shard, Offset }),
    OnRepoReady: async (_S, Docs) => { Batches.push(...Docs); },
  });
  await Provider.Fetch("simple", 999);
  expect(Batches.length).toBe(3); // capped at MaxPerRun
  // Started at offset 4, consumed exactly 3 rows -> cursor stays on shard 0 at offset 7 (no skip, no dup).
  expect(Cursors.at(-1)).toEqual({ Shard: 0, Offset: 7 });
  expect(Batches.map((B) => B.Provenance)).toEqual(["wikipedia:simple:4", "wikipedia:simple:5", "wikipedia:simple:6"]);
});

test("parquet provider stops cleanly when the cursor is past the last shard", async () => {
  const Cursors: { Shard: number; Offset: number }[] = [];
  const Provider = CreateHfParquetProvider(WikiDumpSource, {
    ListShards: async () => ["s0", "s1"],
    OpenShard: async () => { throw new Error("should not download"); },
    ReadRows: async () => [],
    StartShard: 2, // past the end
    OnCursor: (Shard, Offset) => Cursors.push({ Shard, Offset }),
    OnRepoReady: async () => {},
  });
  const Ret = await Provider.Fetch("simple", 100);
  expect(Ret).toEqual([]); // nothing to do, no download attempted
});

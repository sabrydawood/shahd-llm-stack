import { test, expect } from "bun:test";
import { mkdtempSync, writeFileSync, rmSync, mkdirSync, symlinkSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  BuildToolRegistry,
  DefaultToolRegistry,
  Workspace,
  DefaultToolContext,
  RenderToolManifest,
  CalculatorTool,
  StatsTool,
  JsonTool,
  RegexTool,
  TextTool,
  UuidTool,
} from "../Brain/Serving/Tools/ToolsBarrel.ts";
import { BuildAgentTooling } from "../Brain/Serving/Tools/ToolsBarrel.ts";
import { RunAgent } from "../Brain/Serving/AgentLoop.ts";
import { ChatSession } from "../Brain/Serving/ChatSession.ts";
import { ToolTokens } from "../Brain/Serving/ToolProtocol.ts";
import { LoadConfig } from "../Brain/Config/LoadConfig.ts";

test("capability gate: exec + file access are off/read-only by the safe default policy", () => {
  const Safe = DefaultToolRegistry();
  expect(Safe.Has("run_code")).toBe(false); // exec off by default
  expect(Safe.Has("file_read")).toBe(true); // read-only by default
  expect(Safe.Has("file_write")).toBe(false); // no writes by default

  const Full = BuildToolRegistry({ FileAccess: "ReadWrite", ExecEnabled: true, WebSearchEnabled: false });
  expect(Full.Has("run_code")).toBe(true);
  expect(Full.Has("file_write")).toBe(true);

  const Locked = BuildToolRegistry({ FileAccess: "Off", ExecEnabled: false, WebSearchEnabled: false });
  expect(Locked.Has("file_read")).toBe(false);
  expect(Locked.Has("file_search")).toBe(false);
});

test("Workspace refuses path traversal outside the root", () => {
  const Ws = new Workspace(join(tmpdir(), "shahd-ws-root"));
  expect(() => Ws.Resolve("../secret.txt")).toThrow(/escapes workspace/);
  expect(() => Ws.Resolve("../../etc/passwd")).toThrow(/escapes workspace/);
  expect(Ws.Display(Ws.Resolve("sub/ok.txt"))).toBe(join("sub", "ok.txt"));
});

test("file tools read/write within a workspace and honor the byte cap", async () => {
  const Root = mkdtempSync(join(tmpdir(), "shahd-tools-"));
  try {
    writeFileSync(join(Root, "Hello.txt"), "hi there");
    const Registry = BuildToolRegistry({ FileAccess: "ReadWrite", ExecEnabled: false, WebSearchEnabled: false });
    const Context = DefaultToolContext({ Workspace: new Workspace(Root), MaxFileBytes: 1024 });

    const Read = await Registry.Run({ Name: "file_read", Arguments: { path: "Hello.txt" } }, Context);
    expect(Read["content"]).toBe("hi there");

    const Wrote = await Registry.Run({ Name: "file_write", Arguments: { path: "Out.txt", content: "written" } }, Context);
    expect(Wrote["bytesWritten"]).toBe(7);
    const Back = await Registry.Run({ Name: "file_read", Arguments: { path: "Out.txt" } }, Context);
    expect(Back["content"]).toBe("written");

    const Escape = await Registry.Run({ Name: "file_read", Arguments: { path: "../nope.txt" } }, Context);
    expect(String(Escape["error"])).toContain("escapes workspace");
  } finally {
    rmSync(Root, { recursive: true, force: true });
  }
});

test("pure tools compute the right values", () => {
  expect(CalculatorTool.Execute({ a: 2, op: "^", b: 10 })).toEqual({ result: 1024 });
  expect(StatsTool.Execute({ values: [1, 2, 3, 4], op: "mean" })).toEqual({ result: 2.5 });
  expect(JsonTool.Execute({ action: "parse", input: '{"a":1}' })).toEqual({ value: { a: 1 } });
  expect(TextTool.Execute({ action: "reverse", text: "abc" })).toEqual({ text: "cba" });
  expect(RegexTool.Execute({ pattern: "\\d+", text: "a12 b34", action: "match" })).toEqual({
    matches: ["12", "34"],
    count: 2,
  });
});

test("regex tool rejects catastrophic-backtracking patterns", () => {
  const Result = RegexTool.Execute({ pattern: "(a+)+$", text: "aaaaaaX" });
  expect(String((Result as { error: string }).error)).toContain("backtracking");
});

test("regex tool runs quantifier-bearing patterns through the isolated (timeout-fenced) sandbox", () => {
  // \d+ has a quantifier, so it takes the fenced subprocess path — a pattern that ReDoS-hangs there is
  // KILLED by the sandbox timeout instead of freezing the server (the heuristic screen is incomplete;
  // the fence is the real bound). Correct results still come back from the fenced path.
  expect(RegexTool.Execute({ pattern: "\\d+", text: "a12 b34", action: "match" })).toEqual({ matches: ["12", "34"], count: 2 });
}, 12000);

test("async knowledge tools: web_search stubs offline, memory round-trips", async () => {
  const Registry = DefaultToolRegistry();
  const Context = DefaultToolContext();
  const Search = await Registry.Run({ Name: "web_search", Arguments: { query: "x" } }, Context);
  expect(Search["stub"]).toBe(true);
  await Registry.Run({ Name: "memory_store", Arguments: { key: "k", value: "v" } }, Context);
  const Recall = await Registry.Run({ Name: "memory_recall", Arguments: { key: "k" } }, Context);
  expect(Recall).toEqual({ key: "k", value: "v" });
});

test("user_ask never blocks without a provider", async () => {
  const Registry = DefaultToolRegistry();
  const Result = await Registry.Run({ Name: "user_ask", Arguments: { question: "?" } }, DefaultToolContext());
  expect(String(Result["error"])).toContain("non-interactive");
});

test("DefaultToolContext is deterministic (fixed clock, seeded uuid)", () => {
  const A = UuidTool.Execute({}, DefaultToolContext({ Seed: 7 }));
  const B = UuidTool.Execute({}, DefaultToolContext({ Seed: 7 }));
  expect(A).toEqual(B);
  expect(String((A as { uuid: string }).uuid)).toMatch(/^[0-9a-f-]{36}$/);
});

test("agent loop: finish is terminal, compact shrinks the session", async () => {
  const Session = new ChatSession("sys");
  for (let I = 0; I < 6; I++) Session.AddUser(`turn ${I}`);
  expect(Session.Compact(2)).toBe(4); // drop 4 of 6 body turns
  expect(Session.Messages.length).toBe(4); // system + note + 2 recent

  const Registry = DefaultToolRegistry();
  const Result = await RunAgent(
    new ChatSession("sys"),
    () => `${ToolTokens.CallStart}{"name":"finish","arguments":{"answer":"done!"}}${ToolTokens.CallEnd}`,
    Registry,
  );
  expect(Result.FinalText).toBe("done!");
  expect(Result.HitStepLimit).toBe(false);
  expect(Result.ToolCalls.length).toBe(1);
});

test("Config.Tools actually governs behavior end-to-end (root, byte cap, exec, budget)", async () => {
  const Root = mkdtempSync(join(tmpdir(), "shahd-cfg-"));
  try {
    writeFileSync(join(Root, "In.txt"), "0123456789"); // 10 bytes
    const Config = LoadConfig({
      Overrides: { Tools: { FileAccess: "ReadWrite", ExecEnabled: true, WorkspaceRoot: Root, MaxFileBytes: 4, MaxToolSteps: 9 } },
      UseCli: false,
      UseEnv: false,
    });
    const Tooling = BuildAgentTooling(Config);
    // MaxToolSteps flows from config.
    expect(Tooling.MaxSteps).toBe(9);
    // ExecEnabled flows: run_code is registered.
    expect(Tooling.Registry.Has("run_code")).toBe(true);
    // WorkspaceRoot flows: reads resolve under the CONFIG root, and escaping it fails.
    const Escape = await Tooling.Registry.Run({ Name: "file_read", Arguments: { path: "../outside.txt" } }, Tooling.Context);
    expect(String(Escape["error"])).toContain("escapes workspace");
    // MaxFileBytes flows: the 10-byte file exceeds the config cap of 4.
    const Capped = await Tooling.Registry.Run({ Name: "file_read", Arguments: { path: "In.txt" } }, Tooling.Context);
    expect(String(Capped["error"])).toContain("cap");
  } finally {
    rmSync(Root, { recursive: true, force: true });
  }
});

test("Workspace refuses a symlink that escapes the root (CWE-59)", () => {
  const Root = mkdtempSync(join(tmpdir(), "shahd-ws-"));
  const Outside = mkdtempSync(join(tmpdir(), "shahd-out-"));
  try {
    writeFileSync(join(Outside, "secret.txt"), "TOP SECRET");
    mkdirSync(join(Root, "sub"));
    const Ws = new Workspace(Root);
    expect(() => Ws.Resolve("sub")).not.toThrow(); // a legit in-root path is fine
    let Linked = false;
    try {
      symlinkSync(Outside, join(Root, "link"), "junction"); // dir junction: no admin needed on Windows
      Linked = true;
    } catch {
      // symlink creation not permitted in this environment — skip the escape assertion, keep the rest
    }
    if (Linked) {
      expect(() => Ws.Resolve("link/secret.txt")).toThrow(/symlink/); // followed link would escape -> rejected
    }
  } finally {
    rmSync(Root, { recursive: true, force: true });
    rmSync(Outside, { recursive: true, force: true });
  }
});

test("list_tools + manifest expose the tool surface to the model", async () => {
  const Registry = DefaultToolRegistry();
  const Context = DefaultToolContext({ Registry });
  const Listed = await Registry.Run({ Name: "list_tools", Arguments: {} }, Context);
  const Names = (Listed["tools"] as { name: string }[]).map((T) => T.name);
  expect(Names).toContain("calculator");
  expect(Names).toContain("finish");

  const Manifest = RenderToolManifest(Registry.List());
  expect(Manifest).toContain(ToolTokens.CallStart);
  expect(Manifest).toContain("calculator");
});

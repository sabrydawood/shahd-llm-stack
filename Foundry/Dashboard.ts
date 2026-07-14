// Interactive Foundry control panel (M9). A Bun-served page + API over a DocumentStore:
//   GET  /                  the control-panel page
//   GET  /api/stats         aggregate counts (tiers/langs/licenses/bytes)
//   GET  /api/repos         per-repo rollup (accordion list)
//   GET  /api/documents     ?source=… files of one repo (accordion contents)
//   GET  /api/config        whether a Learn runner is wired
//   POST /api/learn         start a Learn run with the posted settings (one at a time)
//   GET  /api/learn/stream  Server-Sent Events streaming the run's progress
// The Learn runner is INJECTED (a mock in tests; the real provider-backed one at serving time), so
// the handler is testable and never imports the network/fs providers itself.

import type { DocumentStore } from "./DocumentStore.ts";
import type { LearnSettings, LearnEvent, LearnFn } from "./DashboardTypes.ts";
import type { RepoLevel } from "./RepoQuality.ts";
import { DashboardHtml } from "./DashboardHtml.ts";

export type DashboardHandler = (Req: Request) => Promise<Response>;

type JobState = { Running: boolean; Events: LearnEvent[]; Listeners: Set<(Event: LearnEvent) => void> };

function ToNum(Value: unknown, Default: number): number {
  if (Value === null || Value === undefined || Value === "") return Default; // Number(null) is 0, not the default
  const N = Number(Value);
  return Number.isFinite(N) ? N : Default;
}

function ParseSettings(Body: Record<string, unknown>): LearnSettings {
  const Source = Body["Source"];
  const Level = Body["MinLevel"];
  const Repos = Body["Repos"];
  return {
    Source: Source === "local" || Source === "both" ? Source : "github",
    Query: typeof Body["Query"] === "string" ? Body["Query"] : "language:typescript stars:>1000",
    Repos: Array.isArray(Repos) ? Repos.map(String) : ["."],
    MinLevel: (Level === "high" || Level === "low" ? Level : "medium") as RepoLevel,
    MaxRepos: ToNum(Body["MaxRepos"], 5),
    MaxFilesPerRepo: ToNum(Body["MaxFilesPerRepo"], 2000),
    MaxBytesPerRepo: ToNum(Body["MaxBytesPerRepo"], 32_000_000),
    SkipLearned: Body["SkipLearned"] !== false,
  };
}

function Json(Data: unknown, Status = 200): Response {
  return Response.json(Data, { status: Status });
}

export function CreateDashboardHandler(Store: DocumentStore, Learn?: LearnFn): DashboardHandler {
  const Job: JobState = { Running: false, Events: [], Listeners: new Set() };
  const Emit = (Event: LearnEvent): void => {
    Job.Events.push(Event);
    for (const Listener of Job.Listeners) Listener(Event);
    if (Event.kind === "done" || Event.kind === "error") Job.Running = false;
  };

  return async (Req: Request): Promise<Response> => {
    const Url = new URL(Req.url);
    const Path = Url.pathname;

    if (Path === "/" || Path === "/index.html") {
      return new Response(DashboardHtml, { headers: { "Content-Type": "text/html; charset=utf-8" } });
    }
    if (Path === "/api/stats") return Json(await Store.Stats());
    if (Path === "/api/repos") return Json(await Store.RepoSummaries());
    if (Path === "/api/config") return Json({ learnEnabled: Learn !== undefined, running: Job.Running });
    if (Path === "/api/documents") {
      const Source = Url.searchParams.get("source") ?? "";
      const Limit = ToNum(Url.searchParams.get("limit"), 500);
      const Docs = await Store.DocumentsBySource(Source, Limit);
      return Json(Docs.map((D) => ({ path: D.Provenance.split("/").slice(-3).join("/"), tier: D.Tier, lang: D.Lang, bytes: D.Bytes, provenance: D.Provenance })));
    }

    if (Path === "/api/learn" && Req.method === "POST") {
      if (Learn === undefined) return Json({ error: "read-only dashboard (no learn runner wired)" }, 501);
      if (Job.Running) return Json({ error: "a learn run is already in progress" }, 409);
      const Body = (await Req.json().catch(() => ({}))) as Record<string, unknown>;
      const Settings = ParseSettings(Body);
      Job.Running = true;
      Job.Events = [];
      Emit({ kind: "start", query: Settings.Query, source: Settings.Source });
      void Learn(Settings, Emit).catch((Caught) => Emit({ kind: "error", message: (Caught as Error).message }));
      return Json({ started: true }, 202);
    }

    if (Path === "/api/learn/stream") {
      let Listener: ((Event: LearnEvent) => void) | null = null;
      const Drop = (): void => {
        if (Listener !== null) {
          Job.Listeners.delete(Listener);
          Listener = null;
        }
      };
      const Stream = new ReadableStream<Uint8Array>({
        start: (Controller: ReadableStreamDefaultController<Uint8Array>): void => {
          const Encoder = new TextEncoder();
          let Closed = false;
          const End = (): void => {
            Closed = true;
            Drop();
            try {
              Controller.close();
            } catch {
              // already closed (idle timeout / client disconnect) — ignore
            }
          };
          const Send = (Event: LearnEvent): void => {
            if (Closed) return;
            try {
              Controller.enqueue(Encoder.encode(`data: ${JSON.stringify(Event)}\n\n`));
            } catch {
              End(); // the connection was closed underneath us — stop listening so Emit never throws
            }
          };
          for (const Event of Job.Events) Send(Event);
          if (!Job.Running) {
            End();
            return;
          }
          Listener = (Event: LearnEvent): void => {
            Send(Event);
            if (Event.kind === "done" || Event.kind === "error") End();
          };
          Job.Listeners.add(Listener);
        },
        cancel: Drop, // client disconnected — remove the listener
      });
      return new Response(Stream, { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" } });
    }

    return new Response("not found", { status: 404 });
  };
}

export function StartDashboard(Store: DocumentStore, Port = 8090, Learn?: LearnFn): ReturnType<typeof Bun.serve> {
  // idleTimeout 0 disables Bun's 10s request timeout, which would otherwise kill a long-lived SSE
  // stream mid-learn (a whole-repo ingest can take much longer than 10s).
  return Bun.serve({ port: Port, idleTimeout: 0, fetch: CreateDashboardHandler(Store, Learn) });
}

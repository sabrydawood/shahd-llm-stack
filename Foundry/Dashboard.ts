// Visual inspection dashboard for the Data Foundry (M4). A Bun-served page + JSON API over any
// DocumentStore (in-memory now, Postgres later) so data quality can be watched and cleaned: tier
// counts, the training-eligible byte total, a quality histogram, license/language breakdowns, and a
// browsable document list (with reject reasons) filtered by tier. Read-only — it never mutates data.

import type { DocumentStore } from "./DocumentStore.ts";
import { BuildReport } from "./QualityReport.ts";

export type DashboardHandler = (Req: Request) => Promise<Response>;

const Html = `<!doctype html><html><head><meta charset="utf-8"><title>Shahd Data Foundry</title>
<style>
 body{font:14px system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
 header{padding:16px 24px;background:#161a22;border-bottom:1px solid #262b36}
 h1{margin:0;font-size:18px} main{padding:24px;max-width:1100px;margin:0 auto}
 .cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}
 .card{background:#161a22;border:1px solid #262b36;border-radius:8px;padding:14px 18px;min-width:120px}
 .card b{display:block;font-size:22px} .muted{color:#8b93a1;font-size:12px}
 .bars{display:flex;gap:3px;align-items:flex-end;height:70px}
 .bar{flex:1;background:#3b82f6;min-height:2px;border-radius:2px 2px 0 0}
 table{width:100%;border-collapse:collapse;margin-top:8px} th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #262b36;vertical-align:top}
 select{background:#161a22;color:#e6e6e6;border:1px solid #262b36;border-radius:6px;padding:6px}
 pre{margin:0;white-space:pre-wrap;color:#a5b4c4;font-size:12px;max-height:120px;overflow:auto}
 .tier-Filtered{color:#22c55e} .tier-Rejected{color:#ef4444} .tier-Raw{color:#eab308}
 h2{font-size:14px;color:#8b93a1;text-transform:uppercase;letter-spacing:.05em;margin:24px 0 4px}
</style></head><body>
<header><h1>Shahd — Data Foundry</h1><span class="muted">tiered, inspectable training data</span></header>
<main>
 <div class="cards" id="cards"></div>
 <h2>Quality histogram (0.0 → 1.0)</h2><div class="bars" id="hist"></div>
 <h2>By license</h2><div id="licenses"></div>
 <h2>By language</h2><div id="langs"></div>
 <h2>Documents</h2>
 <select id="tier" onchange="loadDocs()"><option value="">all tiers</option><option>Filtered</option><option>Raw</option><option>Rejected</option></select>
 <table><thead><tr><th>tier</th><th>license</th><th>lang</th><th>quality</th><th>provenance / reject reason</th><th>preview</th></tr></thead><tbody id="docs"></tbody></table>
</main>
<script>
 const H=(s)=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
 async function loadReport(){
  const r=await (await fetch('/api/report')).json();
  document.getElementById('cards').innerHTML=
   '<div class="card"><b>'+r.Total+'</b><span class="muted">documents</span></div>'+
   '<div class="card"><b class="tier-Filtered">'+r.ByTier.Filtered+'</b><span class="muted">filtered</span></div>'+
   '<div class="card"><b class="tier-Raw">'+r.ByTier.Raw+'</b><span class="muted">raw</span></div>'+
   '<div class="card"><b class="tier-Rejected">'+r.ByTier.Rejected+'</b><span class="muted">rejected</span></div>'+
   '<div class="card"><b>'+r.FilteredBytes+'</b><span class="muted">trainable bytes</span></div>';
  const m=Math.max(1,...r.QualityHistogram);
  document.getElementById('hist').innerHTML=r.QualityHistogram.map(v=>'<div class="bar" style="height:'+(v/m*100)+'%" title="'+v+'"></div>').join('');
  const kv=(o)=>Object.entries(o).map(([k,v])=>H(k)+': '+v).join(' &nbsp; ');
  document.getElementById('licenses').innerHTML=kv(r.ByLicense);
  document.getElementById('langs').innerHTML=kv(r.ByLang);
 }
 async function loadDocs(){
  const t=document.getElementById('tier').value;
  const d=await (await fetch('/api/documents?limit=200'+(t?'&tier='+t:''))).json();
  document.getElementById('docs').innerHTML=d.map(x=>'<tr><td class="tier-'+x.tier+'">'+x.tier+'</td><td>'+H(x.license)+'</td><td>'+H(x.lang)+'</td><td>'+x.quality.toFixed(2)+'</td><td>'+H(x.provenance)+(x.rejectReason?'<br><span class="muted">'+H(x.rejectReason)+'</span>':'')+'</td><td><pre>'+H(x.preview)+'</pre></td></tr>').join('');
 }
 loadReport();loadDocs();
</script></body></html>`;

export function CreateDashboardHandler(Store: DocumentStore): DashboardHandler {
  return async (Req: Request): Promise<Response> => {
    const Url = new URL(Req.url);
    if (Url.pathname === "/" || Url.pathname === "/index.html") {
      return new Response(Html, { headers: { "Content-Type": "text/html; charset=utf-8" } });
    }
    if (Url.pathname === "/api/report") {
      return Response.json(BuildReport(await Store.All()));
    }
    if (Url.pathname === "/api/documents") {
      const Tier = Url.searchParams.get("tier");
      const Limit = Number(Url.searchParams.get("limit") ?? 100);
      let Docs = await Store.All();
      if (Tier !== null && Tier !== "") Docs = Docs.filter((D) => D.Tier === Tier);
      return Response.json(
        Docs.slice(0, Number.isFinite(Limit) ? Limit : 100).map((D) => ({
          id: D.Id,
          tier: D.Tier,
          license: D.License,
          lang: D.Lang,
          quality: D.QualityScore,
          provenance: D.Provenance,
          rejectReason: D.RejectReason,
          preview: D.Content.slice(0, 400),
        })),
      );
    }
    return new Response("not found", { status: 404 });
  };
}

export function StartDashboard(Store: DocumentStore, Port = 8090): ReturnType<typeof Bun.serve> {
  return Bun.serve({ port: Port, fetch: CreateDashboardHandler(Store) });
}

// The Foundry control-plane page (M14 rebuild) — a real sidebar+content app, not a single stuffed
// panel. Six views (Overview / Collect / Data browser / Train / Models / System) plus the dedicated
// /chat page, switched by a hash router. Manage everything: collect (per-kind, language-filtered),
// review + clean the corpus (paginated Data browser), train (new OR resume/extend a saved model),
// and delete models. Realtime over one WebSocket (/ws). The CSS + client JS live in their own files
// (DashboardStyles / DashboardScript) and are inlined here — the ONLY ${} interpolations in this
// template are those two trusted constants, so composing them re-introduces no fragility.

import { DashboardStyles } from "./DashboardStyles.ts";
import { DashboardScript } from "./DashboardScript.ts";

export const DashboardHtml = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Shahd — Control Plane</title><link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🧠</text></svg>">
<style>${DashboardStyles}</style></head><body>
<div class="app">
 <aside class="side" id="side">
  <div class="brand"><div class="logo">🧠</div><div><b>Shahd</b><span>control plane</span></div></div>
  <nav class="nav">
   <div class="sec">Workspace</div>
   <a data-v="overview" href="#overview" class="on"><span class="ic">◧</span> Overview</a>
   <a data-v="collect" href="#collect"><span class="ic">↓</span> Collect data</a>
   <a data-v="data" href="#data"><span class="ic">▦</span> Data browser</a>
   <a data-v="train" href="#train"><span class="ic">✳</span> Train</a>
   <a data-v="models" href="#models"><span class="ic">▤</span> Models <span class="tag" id="nav-models-tag">0</span></a>
   <a data-v="chat" href="#chat"><span class="ic">✦</span> Chat</a>
   <div class="sec">Machine</div>
   <a data-v="system" href="#system"><span class="ic">⚙</span> System</a>
  </nav>
  <div class="foot"><span class="dot" id="connfoot"></span> <span id="connfoot-t">connecting…</span></div>
 </aside>

 <main class="main">
  <header class="top">
   <button class="burger" onclick="document.getElementById('side').classList.toggle('open')">☰</button>
   <h1 id="vtitle">Overview</h1><span class="crumb" id="vcrumb">everything at a glance</span>
   <span class="spacer"></span>
   <span class="chip"><span class="dt" id="conndot"></span><span id="loadedchip">No model loaded</span></span>
   <button class="btn sm" id="themebtn" onclick="toggleTheme()" title="Toggle theme">☾</button>
  </header>
  <div class="content">

   <section class="view on" id="view-overview">
    <div class="grid cards4" style="margin-bottom:14px">
     <div class="card stat"><div class="k">Documents</div><div class="v tnum" id="ov-docs">—</div><div class="s" id="ov-docs-s">across active kinds</div></div>
     <div class="card stat"><div class="k">Training-eligible</div><div class="v tnum" id="ov-trainable" style="color:var(--good)">—</div><div class="s">Filtered tier</div></div>
     <div class="card stat"><div class="k">Saved models</div><div class="v tnum" id="ov-models">—</div><div class="s" id="ov-models-s">base · chat</div></div>
     <div class="card stat"><div class="k">Jobs</div><div class="v" id="ov-jobs" style="font-size:20px">Idle</div><div class="s" id="ov-jobs-s">no run in progress</div></div>
    </div>
    <div class="grid cards2">
     <div class="card"><h3>Data by kind <span class="r">physically separate tables</span></h3><div id="ov-kinds"><div class="empty">loading…</div></div></div>
     <div class="card"><h3>Loaded model</h3><div id="ov-model"><div class="empty">no model loaded</div></div></div>
    </div>
    <div class="card" style="margin-top:14px"><h3>Collection ledger <span class="r">lifetime progress per source</span></h3><div id="ov-ledger"><div class="empty">nothing collected yet</div></div></div>
   </section>

   <section class="view" id="view-collect">
    <h2 class="sec">Source — each lands in its own kind table</h2>
    <div class="src" id="srccards">
     <div class="s on" data-src="github" onclick="pickSource('github')"><b>GitHub repos <span class="pill code">code</span></b><small>permissive source, SPDX-filtered</small></div>
     <div class="s" data-src="local" onclick="pickSource('local')"><b>Local repos <span class="pill code">code</span></b><small>our own repositories on disk</small></div>
     <div class="s" data-src="oasst" onclick="pickSource('oasst')"><b>OASST <span class="pill conv">conversation</span></b><small>Apache dialogue · multilingual</small></div>
     <div class="s" data-src="oasst2" onclick="pickSource('oasst2')"><b>OASST2 <span class="pill conv">conversation</span></b><small>more Apache dialogue</small></div>
     <div class="s" data-src="wikipedia" onclick="pickSource('wikipedia')"><b>Wikipedia <span class="pill know">knowledge</span></b><small>CC-BY-SA · Arabic + English</small></div>
     <div class="s" data-src="gsm8k" onclick="pickSource('gsm8k')"><b>GSM8K <span class="pill" style="background:var(--mut)">instruction</span></b><small>MIT · math reasoning w/ steps</small></div>
     <div class="s" data-src="wikidump" onclick="pickSource('wikidump')"><b>Wikipedia dumps <span class="pill know">knowledge</span></b><small>CC-BY-SA · bulk parquet · resumes by shard</small></div>
     <div class="s" data-src="stackexchange" onclick="pickSource('stackexchange')"><b>Stack Exchange <span class="pill conv">conversation</span></b><small>CC-BY-SA · Q&amp;A pairs · parquet</small></div>
     <div class="s" data-src="folder" onclick="pickSource('folder')"><b>Local folder <span class="pill" style="background:var(--books)">any files</span></b><small>ingest a downloaded corpus (e.g. Gutenberg)</small></div>
    </div>
    <div class="card" style="margin-bottom:14px">
     <div class="row c3">
      <div id="c-genbox" style="display:none"><label class="f">Language to collect</label><select class="i" id="c-lang"></select></div>
      <div id="c-ghbox"><label class="f">GitHub query <span style="color:var(--faint);font-weight:400">— use ; for several (each grows the corpus)</span></label><input class="i" id="c-query" value="stars:>1000 language:typescript"></div>
      <div id="c-localbox" style="display:none"><label class="f">Local repo paths (comma-separated)</label><input class="i" id="c-repos" value="."></div>
      <div id="c-folderbox" style="display:none"><label class="f">Folder path(s) — comma-separated</label><input class="i" id="c-folderpath" value="" placeholder="D:\App\books"></div>
      <div><label class="f" id="c-maxlabel">Max repos</label><input class="i tnum" id="c-maxrepos" type="number" value="5"></div>
      <div id="c-levelbox"><label class="f">Min level</label><select class="i" id="c-minlevel"><option>medium</option><option>high</option><option>low</option></select></div>
     </div>
     <div class="row c3" id="c-capbox" style="margin-top:12px">
      <div><label class="f">Max files / repo</label><input class="i tnum" id="c-maxfiles" type="number" value="2000"></div>
      <div><label class="f">Max MB / repo</label><input class="i tnum" id="c-maxmb" type="number" value="32"></div>
      <div><label class="f">Max KB / file</label><input class="i tnum" id="c-maxkb" type="number" value="512"></div>
     </div>
     <div class="row c3" id="c-folderopts" style="display:none;margin-top:12px">
      <div><label class="f">Store as kind</label><select class="i" id="c-folderkind" onchange="syncCollectForm()"><option value="books">books</option><option value="knowledge">knowledge</option><option value="code">code</option><option value="instruction">instruction</option><option value="web">web</option></select></div>
      <div><label class="f">License</label><input class="i" id="c-folderlicense" value="public-domain"></div>
      <div></div>
     </div>
     <div style="display:flex;gap:14px;align-items:center;margin-top:14px;flex-wrap:wrap">
      <label class="chk"><input type="checkbox" id="c-skip" checked> Skip sources already collected</label>
      <span class="spacer"></span><span id="c-kindhint" style="font-size:12.5px;color:var(--mut)"></span>
     </div>
     <div style="display:flex;gap:9px;margin-top:14px;align-items:center"><button class="btn pri" id="c-start" onclick="cStart()">▶ Start collection</button><button class="btn danger" id="c-stop" style="display:none">■ Stop</button><span class="pill idle" id="c-badge">idle</span></div>
    </div>
    <div class="card"><h3>Live log</h3><div class="prog" id="c-prog"><i id="c-progi"></i></div><div class="crumb" id="c-plabel" style="margin-bottom:8px">idle</div><div class="log" id="c-log"><div>collect data into a kind table, then Train a model on it.</div></div></div>
   </section>

   <section class="view" id="view-data">
    <h2 class="sec">Browse, review, and clean the collected corpus</h2>
    <div class="filters">
     <div class="fg"><label class="f">Kind</label><select class="i" id="d-kind"><option value="code">code</option><option value="conversation">conversation</option><option value="knowledge">knowledge</option><option value="books">books</option><option value="web">web</option><option value="instruction">instruction</option></select></div>
     <div class="fg"><label class="f">Tier</label><select class="i" id="d-tier"><option value="">any tier</option><option value="Filtered">Filtered</option><option value="Raw">Raw</option><option value="Rejected">Rejected</option></select></div>
     <div class="fg"><label class="f">Language</label><select class="i" id="d-lang"><option value="">any language</option></select></div>
     <div class="fg"><label class="f">License</label><select class="i" id="d-license"><option value="">any license</option></select></div>
     <div class="fg grow"><label class="f">Search (path or content)</label><input class="i" id="d-q" placeholder="substring…" onkeydown="if(event.key==='Enter')dSearch()"></div>
     <div class="fg"><label class="f">&nbsp;</label><button class="btn pri" onclick="dSearch()">Search</button></div>
     <div class="fg"><label class="f">&nbsp;</label><button class="btn danger" id="d-bulk" onclick="dBulk()">Delete matching</button></div>
    </div>
    <div class="card" style="padding:6px 6px 0">
     <table class="t"><thead><tr><th>Provenance</th><th>Tier</th><th>Lang</th><th>License</th><th>Size</th><th></th></tr></thead><tbody id="d-tbody"><tr><td colspan="6" class="empty">loading…</td></tr></tbody></table>
    </div>
    <div class="pager"><span id="d-total">—</span><span class="spacer"></span><button class="btn sm" id="d-prev" onclick="dPrev()">← Prev</button><span id="d-pageinfo">page 1</span><button class="btn sm" id="d-next" onclick="dNext()">Next →</button></div>
   </section>

   <section class="view" id="view-train">
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;flex-wrap:wrap">
     <div class="seg"><button id="t-mode-pre" class="on" onclick="setMode('pretrain')">Pretrain (base)</button><button id="t-mode-chat" onclick="setMode('chat')">Chat / SFT</button></div>
     <select class="i" style="width:auto" id="t-resume" onchange="onResume()"><option value="">◇ New model</option></select>
     <span class="crumb">preset →</span>
     <select class="i" style="width:auto" id="t-preset" onchange="onPreset()"><option value="">Tier preset…</option><option>Seed</option><option>Nano</option><option>Micro</option><option>Mini</option><option>Small</option><option>Base</option><option>Large</option></select>
    </div>
    <div class="grid cards2">
     <div class="card">
      <h3>Architecture</h3>
      <div class="row c2" style="margin-bottom:11px"><div><label class="f">Model name</label><input class="i" id="t-name" value="foundry"></div><div><label class="f">Steps</label><input class="i tnum" id="t-steps" type="number" value="500"></div></div>
      <div class="row c4" style="margin-bottom:11px"><div><label class="f">Embed</label><input class="i tnum" id="t-embed" type="number" value="96"></div><div><label class="f">Layers</label><input class="i tnum" id="t-layers" type="number" value="3"></div><div><label class="f">Heads</label><input class="i tnum" id="t-heads" type="number" value="4"></div><div><label class="f">Context</label><input class="i tnum" id="t-ctx" type="number" value="96"></div></div>
      <div class="row c4"><div><label class="f">Vocab</label><input class="i tnum" id="t-vocab" type="number" value="512"></div><div><label class="f">Batch</label><input class="i tnum" id="t-batch" type="number" value="16"></div><div><label class="f" title="Sequence-parallel worker threads for pretrain AND chat/SFT (0 = sequential). 8 is the sweet spot on this 16GB machine; 16 needs several GB of free RAM.">Workers</label><input class="i tnum" id="t-workers" type="number" value="8"></div><div><label class="f" title="Storage precision. F32 halves weight/tape/pool memory and uses the 8-lane f32 SIMD kernels (~1.15x step today, the real win is memory for bigger models). F64 is exact. Resume keeps the checkpoint's precision.">Precision</label><select class="i" id="t-prec"><option value="F64">F64 (exact)</option><option value="F32">F32 (half memory)</option></select></div></div>
      <h3 style="margin-top:20px">Data mix — how much of each kind</h3>
      <div class="row c2" id="t-mix-pretrain"><div><label class="f"><span class="pill code">code</span> MB</label><input class="i tnum" id="t-corpus" type="number" step="0.5" value="1.5"></div><div><label class="f"><span class="pill know">knowledge</span> MB</label><input class="i tnum" id="t-know" type="number" value="0"></div></div>
      <div class="row c3" id="t-mix-chat" style="display:none"><div><label class="f"><span class="pill conv">conversation</span> examples</label><input class="i tnum" id="t-conv" type="number" value="4000"></div><div><label class="f"><span class="pill code">code</span> samples</label><input class="i tnum" id="t-code" type="number" value="4000"></div><div><label class="f" title="Warm start: seed the chat model with a pretrained base model's weights instead of random init. The base must have the SAME architecture (embed/layers/heads/context) — its tokenizer is reused verbatim.">From base (warm start)</label><select class="i" id="t-from"><option value="">◇ from scratch</option></select></div></div>
      <div class="note" id="t-resumenote" style="display:none"></div>
      <div class="note">Heads must divide Embed. Set a kind to 0 for a pure model — conversation drives "talks well"; code drives code. See Docs/MODEL-SCALING.md for tier presets.</div>
      <div style="display:flex;gap:9px;margin-top:16px"><button class="btn pri" id="t-start" onclick="tStart()">▶ Train model</button></div>
     </div>
     <div class="card">
      <h3>Live run <span class="r pill idle" id="t-badge">idle</span></h3>
      <div class="sparkwrap"><svg class="spark" id="t-spark" viewBox="0 0 300 74" preserveAspectRatio="none"></svg><div class="sparktip" id="t-tip"></div></div>
      <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--mut);margin:8px 0"><span>loss <b class="mono" id="t-loss" style="color:var(--text)">—</b></span><span>elapsed <b class="mono" id="t-elapsed" style="color:var(--text)">—</b></span><span id="t-eta"></span></div>
      <div class="log" id="t-log"><div>configure the model, then Train. Re-running a name (or picking it in Resume) continues it.</div></div>
     </div>
    </div>
   </section>

   <section class="view" id="view-models">
    <h2 class="sec">Saved models — load, add training, or delete</h2>
    <div class="card" style="padding:6px 6px 0">
     <table class="t"><thead><tr><th>Name</th><th>Type</th><th>Params</th><th>Architecture</th><th>Progress</th><th>Created</th><th></th></tr></thead><tbody id="m-tbody"><tr><td colspan="7" class="empty">loading…</td></tr></tbody></table>
    </div>
    <div class="note"><b>Resume training</b> loads a checkpoint and continues from its last step — raise Steps or change the data mix to keep improving a model you already trained.</div>
   </section>

   <section class="view" id="view-chat">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;flex-wrap:wrap">
     <label class="f" style="margin:0">Model</label>
     <select class="i" style="width:auto;min-width:220px" id="ch-model" onchange="chPick(this.value)"><option value="">(no model)</option></select>
     <label class="f" style="margin:0">Chat</label>
     <select class="i" style="width:auto;min-width:190px" id="ch-history" onchange="chOpen(this.value)" title="past conversations"><option value="">— history —</option></select>
     <span class="spacer"></span>
     <label class="f" style="margin:0">temp</label><input class="i tnum" style="width:62px" id="ch-temp" type="number" step="0.1" min="0" max="2" value="0.8">
     <label class="f" style="margin:0">max</label><input class="i tnum" style="width:80px" id="ch-max" type="number" min="1" max="4096" value="512">
     <button class="btn sm" onclick="chNew()">+ New</button>
     <button class="btn sm danger" onclick="chDelCur()" title="delete this conversation">Delete</button>
     <a class="btn sm" href="/chat" target="_blank" style="text-decoration:none">Full page ↗</a>
    </div>
    <div class="warnbar">⚠ Tiny from-scratch model — replies are experimental and often incoherent. The value here is the serving path + the visible reasoning trace, not fluency.</div>
    <div class="grid cards2">
     <div class="card"><h3>Conversation</h3><div class="bubbles" id="ch-bubbles"><div class="empty">say something to test the model</div></div>
      <div class="chatin"><textarea id="ch-box" placeholder="Type a message — Enter to send, Shift+Enter for newline"></textarea><button class="btn pri" id="ch-send" onclick="chSend()">Send</button></div>
      <div class="crumb" id="ch-stat" style="margin-top:6px"></div>
     </div>
     <div class="card"><h3>Reasoning trace <span class="r">how it answered</span></h3><div class="trace" id="ch-trace"><div class="empty">the model's steps (think → tool → answer) appear here after a reply</div></div>
      <div class="note">The trace shows the model's ACTUAL steps — the lens for spotting weaknesses and improving it. (Base models reply directly with no tool/think steps.)</div>
     </div>
    </div>
   </section>

   <section class="view" id="view-system"><div class="grid cards3" id="sys"><div class="empty">connecting…</div></div></section>

  </div>
 </main>
</div>

<div class="modal" id="modal" onclick="if(event.target===this)closeModal()">
 <div class="mcard"><div class="mhead"><b id="m-title"></b><button class="mx" onclick="closeModal()">✕</button></div><div class="mmeta" id="m-meta"></div><pre class="mbody" id="m-body"></pre></div>
</div>
<script>${DashboardScript}</script>
</body></html>`;

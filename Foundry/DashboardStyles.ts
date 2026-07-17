// Control-panel stylesheet (M14 rebuild). A token-based, dual-theme (dark default + light) design:
// sidebar + content shell, cards, per-kind accent colors, tables, pagination, the data-browser
// filters, the document viewer modal, and the training loss sparkline. Kept in its own file (a plain
// CSS string with NO backticks / no ${}) so DashboardHtml.ts can inline it and stay small. Themes:
// prefers-color-scheme is the default signal; the header toggle stamps :root[data-theme] which wins.
export const DashboardStyles = `
 :root{
  --bg:#0b0e14; --surface:#11151d; --panel:#151a24; --panel2:#1a2130; --line:#232b3a;
  --text:#e7edf5; --mut:#8b96a8; --faint:#5b6577;
  --accent:#4c8dff; --accent-soft:rgba(76,141,255,.14);
  --code:#4c8dff; --conv:#37b96b; --know:#a674f7; --books:#e0883c;
  --good:#37b96b; --warn:#e0a53c; --bad:#f0645a;
  --radius:11px; --radius-sm:8px;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px -12px rgba(0,0,0,.5);
 }
 @media (prefers-color-scheme: light){
  :root{
   --bg:#f4f6fa; --surface:#ffffff; --panel:#ffffff; --panel2:#f6f8fc; --line:#e2e7ef;
   --text:#171b22; --mut:#5a6472; --faint:#8b95a5;
   --accent:#2563eb; --accent-soft:rgba(37,99,235,.09);
   --code:#2563eb; --conv:#1a8f4c; --know:#8250df; --books:#c26a1e;
   --good:#1a8f4c; --warn:#b4791f; --bad:#d64a41;
   --shadow:0 1px 2px rgba(16,24,40,.05),0 10px 26px -14px rgba(16,24,40,.18);
  }
 }
 :root[data-theme="dark"]{
  --bg:#0b0e14; --surface:#11151d; --panel:#151a24; --panel2:#1a2130; --line:#232b3a;
  --text:#e7edf5; --mut:#8b96a8; --faint:#5b6577; --accent:#4c8dff; --accent-soft:rgba(76,141,255,.14);
  --code:#4c8dff; --conv:#37b96b; --know:#a674f7; --good:#37b96b; --warn:#e0a53c; --bad:#f0645a;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px -12px rgba(0,0,0,.5);
 }
 :root[data-theme="light"]{
  --bg:#f4f6fa; --surface:#ffffff; --panel:#ffffff; --panel2:#f6f8fc; --line:#e2e7ef;
  --text:#171b22; --mut:#5a6472; --faint:#8b95a5; --accent:#2563eb; --accent-soft:rgba(37,99,235,.09);
  --code:#2563eb; --conv:#1a8f4c; --know:#8250df; --good:#1a8f4c; --warn:#b4791f; --bad:#d64a41;
  --shadow:0 1px 2px rgba(16,24,40,.05),0 10px 26px -14px rgba(16,24,40,.18);
 }
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.55;-webkit-font-smoothing:antialiased}
 .tnum{font-variant-numeric:tabular-nums}
 .mono{font-family:var(--mono)}
 a{color:var(--accent)}
 .app{display:grid;grid-template-columns:236px 1fr;min-height:100vh}
 @media(max-width:820px){.app{grid-template-columns:1fr}.side{position:fixed;left:-260px;z-index:40;transition:left .2s}.side.open{left:0}}

 /* sidebar */
 .side{background:var(--surface);border-right:1px solid var(--line);display:flex;flex-direction:column;position:sticky;top:0;height:100vh}
 .brand{display:flex;align-items:center;gap:10px;padding:18px 18px 14px}
 .logo{width:30px;height:30px;border-radius:9px;background:linear-gradient(140deg,var(--accent),var(--know));display:grid;place-items:center;font-size:16px;box-shadow:0 3px 10px -3px var(--accent)}
 .brand b{font-size:15px;letter-spacing:-.01em} .brand span{font-size:11px;color:var(--mut);display:block;line-height:1}
 .nav{padding:6px 10px;display:flex;flex-direction:column;gap:2px;flex:1;overflow:auto}
 .nav .sec{font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--faint);padding:14px 10px 5px;font-weight:600}
 .nav a{display:flex;align-items:center;gap:11px;padding:8px 11px;border-radius:var(--radius-sm);color:var(--mut);text-decoration:none;font-weight:500;cursor:pointer;font-size:13.5px}
 .nav a .ic{width:17px;text-align:center;opacity:.9}
 .nav a:hover{background:var(--panel2);color:var(--text)}
 .nav a.on{background:var(--accent-soft);color:var(--accent)}
 .nav a .tag{margin-left:auto;font-size:10px;background:var(--panel2);color:var(--mut);padding:1px 7px;border-radius:20px;font-variant-numeric:tabular-nums}
 .side .foot{padding:12px 16px;border-top:1px solid var(--line);font-size:11.5px;color:var(--faint);display:flex;align-items:center;gap:8px}
 .side .foot .dot{width:7px;height:7px;border-radius:50%;background:var(--bad)} .side .foot .dot.on{background:var(--good)}

 /* main */
 .main{display:flex;flex-direction:column;min-width:0}
 .top{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--bg) 82%,transparent);backdrop-filter:blur(10px);border-bottom:1px solid var(--line);padding:13px 26px;display:flex;align-items:center;gap:14px}
 .top h1{margin:0;font-size:16px;font-weight:650;letter-spacing:-.01em}
 .top .crumb{color:var(--mut);font-size:12.5px}
 .burger{display:none;background:var(--panel2);border:1px solid var(--line);color:var(--text);border-radius:8px;padding:5px 10px;cursor:pointer;font-size:15px}
 @media(max-width:820px){.burger{display:inline-block}}
 .chip{display:inline-flex;align-items:center;gap:7px;background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:5px 12px;font-size:12.5px;font-weight:500}
 .chip .dt{width:7px;height:7px;border-radius:50%;background:var(--bad)} .chip .dt.on{background:var(--good)}
 .spacer{flex:1}
 .content{padding:24px 26px 60px;max-width:1180px;width:100%}
 .view{display:none} .view.on{display:block;animation:fade .25s ease}
 @keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
 @media (prefers-reduced-motion:reduce){.view.on{animation:none}}

 h2.sec{font-size:13px;text-transform:uppercase;letter-spacing:.07em;color:var(--mut);margin:0 0 12px;font-weight:600}
 .grid{display:grid;gap:14px}
 .cards4{grid-template-columns:repeat(4,1fr)} .cards3{grid-template-columns:repeat(3,1fr)} .cards2{grid-template-columns:1.4fr 1fr}
 @media(max-width:900px){.cards4{grid-template-columns:repeat(2,1fr)}.cards3,.cards2{grid-template-columns:1fr}}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px 17px;box-shadow:var(--shadow)}
 .stat .k{font-size:11.5px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;font-weight:600}
 .stat .v{font-size:26px;font-weight:680;letter-spacing:-.02em;margin-top:5px;font-family:var(--mono)}
 .stat .s{font-size:12px;color:var(--mut);margin-top:2px}
 .card h3{margin:0 0 14px;font-size:14.5px;font-weight:620;display:flex;align-items:center;gap:8px}
 .card h3 .r{margin-left:auto;font-size:12px;color:var(--mut);font-weight:500}

 .kind{display:flex;align-items:center;gap:12px;margin-bottom:13px} .kind:last-child{margin-bottom:0}
 .kind .nm{width:132px;display:flex;align-items:center;gap:8px;font-size:13px;font-weight:550}
 .kind .sw{width:9px;height:9px;border-radius:3px}
 .kind .track{flex:1;height:9px;background:var(--panel2);border-radius:6px;overflow:hidden}
 .kind .fill{height:100%;border-radius:6px}
 .kind .amt{width:160px;text-align:right;font-size:12.5px;color:var(--mut);font-family:var(--mono)}
 .kind.off{opacity:.5}

 .pill{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:600;padding:2px 9px;border-radius:20px;text-transform:uppercase;letter-spacing:.03em}
 .pill.code{background:color-mix(in srgb,var(--code) 15%,transparent);color:var(--code)}
 .pill.conv,.pill.conversation{background:color-mix(in srgb,var(--conv) 16%,transparent);color:var(--conv)}
 .pill.know,.pill.knowledge{background:color-mix(in srgb,var(--know) 16%,transparent);color:var(--know)}
 .pill.base,.pill.idle{background:var(--panel2);color:var(--mut)}
 .pill.run{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
 .pill.done{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good)}
 .pill.err{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}
 .pill.chat{background:color-mix(in srgb,var(--conv) 16%,transparent);color:var(--conv)}
 .tier-Filtered{color:var(--good)} .tier-Raw{color:var(--warn)} .tier-Rejected{color:var(--bad)}

 .btn{font:inherit;font-weight:600;font-size:13px;border:1px solid var(--line);background:var(--panel2);color:var(--text);border-radius:var(--radius-sm);padding:8px 14px;cursor:pointer;display:inline-flex;align-items:center;gap:7px}
 .btn:hover{border-color:var(--accent);color:var(--accent)}
 .btn.pri{background:var(--accent);border-color:var(--accent);color:#fff} .btn.pri:hover{filter:brightness(1.08);color:#fff}
 .btn.danger:hover{border-color:var(--bad);color:var(--bad)}
 .btn.sm{padding:5px 10px;font-size:12px}
 .btn:disabled{opacity:.45;cursor:not-allowed;border-color:var(--line);color:var(--mut)}

 label.f{display:block;font-size:12px;color:var(--mut);font-weight:600;margin:0 0 5px}
 input.i,select.i{width:100%;background:var(--panel2);border:1px solid var(--line);color:var(--text);border-radius:var(--radius-sm);padding:8px 10px;font:inherit;font-size:13px}
 input.i:focus,select.i:focus{outline:2px solid var(--accent-soft);border-color:var(--accent)}
 .row{display:grid;gap:11px} .row.c3{grid-template-columns:repeat(3,1fr)} .row.c2{grid-template-columns:repeat(2,1fr)} .row.c4{grid-template-columns:repeat(4,1fr)}
 @media(max-width:640px){.row.c3,.row.c4{grid-template-columns:repeat(2,1fr)}}
 .chk{display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--mut)} .chk input{width:auto}
 .sparkwrap{position:relative}
 .sparktip{position:absolute;top:4px;left:0;display:none;background:var(--panel2);border:1px solid var(--line);color:var(--text);border-radius:var(--radius-sm);padding:3px 9px;font-size:11.5px;white-space:nowrap;pointer-events:none;z-index:5}

 .seg{display:inline-flex;background:var(--panel2);border:1px solid var(--line);border-radius:var(--radius-sm);padding:3px;gap:3px}
 .seg button{font:inherit;font-weight:600;font-size:13px;border:0;background:transparent;color:var(--mut);padding:6px 15px;border-radius:6px;cursor:pointer}
 .seg button.on{background:var(--accent);color:#fff}

 .src{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
 .src .s{flex:1;min-width:150px;border:1px solid var(--line);border-radius:var(--radius);padding:13px;cursor:pointer;background:var(--panel)}
 .src .s.on{border-color:var(--accent);background:var(--accent-soft)}
 .src .s b{display:flex;align-items:center;gap:7px;font-size:13.5px;margin-bottom:5px} .src .s small{color:var(--mut);font-size:11.5px}

 table.t{width:100%;border-collapse:collapse;font-size:13px}
 table.t th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);font-weight:600;padding:0 12px 9px;border-bottom:1px solid var(--line)}
 table.t td{padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:middle}
 table.t tr:last-child td{border-bottom:0}
 table.t tbody tr{cursor:default} table.t tbody tr:hover td{background:var(--panel2)}
 .acts{display:flex;gap:6px;justify-content:flex-end;flex-wrap:wrap}
 .clip{max-width:520px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
 .empty{color:var(--mut);text-align:center;padding:34px 10px;font-size:13px}

 .filters{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end;margin-bottom:14px}
 .filters .fg{display:flex;flex-direction:column;gap:5px} .filters .fg.grow{flex:1;min-width:180px}
 .pager{display:flex;align-items:center;gap:12px;justify-content:flex-end;margin-top:13px;font-size:12.5px;color:var(--mut)}

 .log{background:var(--bg);border:1px solid var(--line);border-radius:var(--radius-sm);padding:12px 14px;font-family:var(--mono);font-size:12px;line-height:1.7;height:220px;overflow:auto;color:var(--mut)}
 .log div{white-space:pre-wrap;word-break:break-word} .log .ok{color:var(--good)} .log .w{color:var(--warn)} .log .a{color:var(--accent)} .log .e{color:var(--bad)}
 .prog{height:7px;background:var(--panel2);border-radius:5px;overflow:hidden;margin:12px 0} .prog i{display:block;height:100%;width:0;background:var(--accent);border-radius:5px;transition:width .2s}
 @keyframes pulse{0%,100%{opacity:.4}50%{opacity:1}} .prog.busy i{width:100%;background:var(--warn);animation:pulse 1.1s ease-in-out infinite;transition:none}
 .spark{width:100%;height:74px;display:block}
 .note{font-size:12.5px;color:var(--mut);background:var(--panel2);border:1px solid var(--line);border-left:3px solid var(--warn);border-radius:var(--radius-sm);padding:10px 13px;margin-top:14px}
 .warnbar{font-size:12px;color:var(--warn);background:color-mix(in srgb,var(--warn) 10%,transparent);border:1px solid color-mix(in srgb,var(--warn) 30%,transparent);border-radius:var(--radius-sm);padding:8px 12px;margin-bottom:14px}

 /* chat view */
 .bubbles{display:flex;flex-direction:column;gap:10px;height:360px;overflow:auto;padding-right:4px}
 .bub{border:1px solid var(--line);border-radius:10px;padding:9px 13px;font-size:13.5px;background:var(--panel2);white-space:pre-wrap;word-break:break-word}
 .bub.u{background:color-mix(in srgb,var(--accent) 9%,var(--panel2))} .bub.err{border-color:var(--bad);color:var(--bad)}
 .bub .who{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);margin-bottom:4px}
 .bub.cursor .who:after{content:' ▋';color:var(--mut);animation:blink 1s steps(2) infinite} @keyframes blink{50%{opacity:0}}
 .chatin{display:flex;gap:8px;margin-top:12px;align-items:flex-end}
 .chatin textarea{flex:1;background:var(--panel2);border:1px solid var(--line);color:var(--text);border-radius:var(--radius-sm);padding:9px 11px;font:inherit;font-size:13.5px;resize:vertical;min-height:44px;max-height:160px}
 .trace{display:flex;flex-direction:column;gap:9px;max-height:360px;overflow:auto}
 .step{display:flex;gap:10px;align-items:flex-start;font-size:12.5px}
 .step .b{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.03em;padding:2px 8px;border-radius:6px;flex:0 0 auto;margin-top:1px}
 .step .b.think{background:color-mix(in srgb,var(--know) 16%,transparent);color:var(--know)}
 .step .b.tool{background:color-mix(in srgb,var(--accent) 15%,transparent);color:var(--accent)}
 .step .b.answer{background:color-mix(in srgb,var(--good) 15%,transparent);color:var(--good)}
 .step .x{font-family:var(--mono);color:var(--text);word-break:break-word} .step .x .dt{color:var(--mut)}
 .trhist{margin-top:8px;border-top:1px dashed var(--line);padding-top:6px}
 .trhist summary{cursor:pointer;font-size:11.5px;color:var(--mut);user-select:none;list-style:none}
 .trhist summary::-webkit-details-marker{display:none}
 .trhist summary:hover{color:var(--text)}
 .trhist[open] summary{margin-bottom:7px}
 .trhist .trsteps{display:flex;flex-direction:column;gap:7px;white-space:normal} /* .bub is pre-wrap; trace steps must render like the side panel */

 .modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.62);align-items:center;justify-content:center;z-index:60;padding:20px}
 .modal.open{display:flex}
 .mcard{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);width:min(940px,95vw);max-height:88vh;display:flex;flex-direction:column;box-shadow:var(--shadow)}
 .mhead{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid var(--line)} .mhead b{word-break:break-all;font:13px var(--mono)}
 .mx{cursor:pointer;color:var(--mut);font-size:20px;line-height:1;background:none;border:0} .mx:hover{color:var(--text)}
 .mmeta{padding:8px 16px;color:var(--mut);font-size:11.5px;border-bottom:1px solid var(--line);word-break:break-all;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
 .mbody{margin:0;padding:14px 16px;overflow:auto;font:12px var(--mono);white-space:pre-wrap;word-break:break-word;tab-size:2}
`;

// The Foundry control-panel page (M9). Self-contained HTML+CSS+JS served at "/". Left column drives
// a "Learn" run (settings + live SSE progress); right column shows aggregate stats and a per-repo
// accordion whose contents load on demand. Kept in its own file so Dashboard.ts stays small.

export const DashboardHtml = `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Shahd — Data Foundry</title>
<style>
 :root{--bg:#0e1116;--panel:#161b22;--line:#262c36;--txt:#e6edf3;--mut:#8b949e;--blue:#388bfd;--green:#3fb950;--red:#f85149;--yellow:#d29922}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 system-ui,sans-serif}
 header{padding:14px 22px;background:var(--panel);border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:12px}
 header h1{margin:0;font-size:17px} header .m{color:var(--mut);font-size:12px}
 .wrap{display:grid;grid-template-columns:380px 1fr;gap:18px;padding:18px;max-width:1300px;margin:0 auto}
 @media(max-width:900px){.wrap{grid-template-columns:1fr}}
 .panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
 h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:0 0 12px}
 label{display:block;font-size:12px;color:var(--mut);margin:10px 0 3px}
 input,select{width:100%;background:#0d1117;color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:7px 9px;font:inherit}
 .row{display:flex;gap:8px} .row>div{flex:1}
 .chk{display:flex;align-items:center;gap:8px;margin-top:12px} .chk input{width:auto}
 button{margin-top:14px;width:100%;background:var(--blue);color:#fff;border:0;border-radius:7px;padding:10px;font:600 14px system-ui;cursor:pointer}
 button:disabled{opacity:.5;cursor:not-allowed}
 .cards{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px}
 .card{background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:11px 15px;min-width:96px}
 .card b{display:block;font-size:20px} .card span{color:var(--mut);font-size:11px}
 .chips{font-size:12px;color:var(--mut);margin:6px 0}
 .log{margin-top:12px;background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:10px;height:230px;overflow:auto;font:12px ui-monospace,monospace}
 .log div{padding:1px 0} .ok{color:var(--green)} .skip{color:var(--yellow)} .err{color:var(--red)}
 .acc{border:1px solid var(--line);border-radius:8px;margin-top:8px;overflow:hidden}
 .acc>.h{display:flex;justify-content:space-between;padding:9px 13px;cursor:pointer;background:#0d1117}
 .acc>.h:hover{background:#11161d} .acc .h .r{color:var(--mut);font-size:12px}
 .acc>.b{display:none;border-top:1px solid var(--line)} .acc.open>.b{display:block}
 .acc .b table{width:100%;border-collapse:collapse} .acc .b td{padding:4px 13px;border-top:1px solid var(--line);font:12px ui-monospace,monospace}
 .acc .b td.mut{color:var(--mut);text-align:right;white-space:nowrap}
 .lvl-high{color:var(--green)} .lvl-medium{color:var(--yellow)} .lvl-low{color:var(--red)}
 .tier-Filtered{color:var(--green)} .tier-Raw{color:var(--yellow)} .tier-Rejected{color:var(--red)}
</style></head><body>
<header><h1>Shahd — Data Foundry</h1><span class="m">learn from whole repos · tiered · inspectable</span></header>
<div class="wrap">
 <div>
  <div class="panel">
   <h2>Learn</h2>
   <label>Source</label>
   <select id="source"><option value="github">Public GitHub repos</option><option value="local">Our own repos (local)</option><option value="both">Both</option></select>
   <div id="ghbox"><label>GitHub query</label><input id="query" value="language:typescript stars:>1000"></div>
   <div id="localbox" style="display:none"><label>Local repo paths (comma-separated)</label><input id="repos" value="."></div>
   <div class="row"><div><label>Min level</label><select id="minlevel"><option>medium</option><option>high</option><option>low</option></select></div><div><label>Max repos</label><input id="maxrepos" type="number" value="5"></div></div>
   <div class="row"><div><label>Max files / repo</label><input id="maxfiles" type="number" value="2000"></div><div><label>Max MB / repo</label><input id="maxmb" type="number" value="32"></div></div>
   <div class="chk"><input type="checkbox" id="skip" checked><label style="margin:0">Skip repos already learned</label></div>
   <button id="go" onclick="learn()">▶ Learn</button>
  </div>
  <div class="panel" style="margin-top:16px"><h2>Progress</h2><div class="log" id="log"><div class="mut" style="color:var(--mut)">idle — configure and press Learn.</div></div></div>
 </div>
 <div>
  <div class="panel"><h2>Foundry stats</h2><div class="cards" id="cards"></div><div class="chips" id="langs"></div><div class="chips" id="lics"></div></div>
  <div class="panel" style="margin-top:16px"><h2>Learned repos</h2><div id="repos-list"></div></div>
 </div>
</div>
<script>
 const H=(s)=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
 const fmtB=(n)=>n>=1e6?(n/1e6).toFixed(1)+'MB':n>=1e3?(n/1e3).toFixed(0)+'KB':n+'B';
 document.getElementById('source').onchange=(e)=>{const v=e.target.value;document.getElementById('ghbox').style.display=v==='local'?'none':'';document.getElementById('localbox').style.display=v==='github'?'none':'';};
 async function loadStats(){
  const s=await (await fetch('/api/stats')).json();
  document.getElementById('cards').innerHTML=
   '<div class="card"><b>'+s.Total+'</b><span>documents</span></div>'+
   '<div class="card"><b class="tier-Filtered">'+s.ByTier.Filtered+'</b><span>trainable</span></div>'+
   '<div class="card"><b class="tier-Raw">'+s.ByTier.Raw+'</b><span>raw</span></div>'+
   '<div class="card"><b class="tier-Rejected">'+s.ByTier.Rejected+'</b><span>rejected</span></div>'+
   '<div class="card"><b>'+fmtB(s.FilteredBytes)+'</b><span>trainable bytes</span></div>';
  const kv=(o)=>Object.entries(o).sort((a,b)=>b[1]-a[1]).slice(0,14).map(([k,v])=>H(k)+':'+v).join(' · ');
  document.getElementById('langs').innerHTML='<b>langs</b> '+kv(s.ByLang);
  document.getElementById('lics').innerHTML='<b>licenses</b> '+kv(s.ByLicense);
 }
 async function loadRepos(){
  const r=await (await fetch('/api/repos')).json();
  document.getElementById('repos-list').innerHTML=r.length?r.map(x=>
   '<div class="acc"><div class="h" onclick="openRepo(this,'+JSON.stringify(H(x.Source)).replace(/"/g,'&quot;')+')"><span>'+H(x.Source)+'</span><span class="r">'+x.Files+' files · '+fmtB(x.Bytes)+'</span></div><div class="b"></div></div>'
  ).join(''):'<div style="color:var(--mut)">nothing learned yet.</div>';
 }
 async function openRepo(h,src){
  const acc=h.parentElement,body=acc.querySelector('.b');
  if(acc.classList.contains('open')){acc.classList.remove('open');return;}
  acc.classList.add('open');
  if(body.dataset.loaded)return; body.dataset.loaded='1';
  const d=await (await fetch('/api/documents?source='+encodeURIComponent(src)+'&limit=500')).json();
  body.innerHTML='<table>'+d.map(f=>'<tr><td class="tier-'+f.tier+'">'+H(f.path||f.provenance)+'</td><td class="mut">'+H(f.lang)+' · '+fmtB(f.bytes)+'</td></tr>').join('')+'</table>';
 }
 function learn(){
  const go=document.getElementById('go');go.disabled=true;
  const log=document.getElementById('log');log.innerHTML='';
  const line=(t,c)=>{const d=document.createElement('div');if(c)d.className=c;d.textContent=t;log.appendChild(d);log.scrollTop=log.scrollHeight;};
  const body={Source:document.getElementById('source').value,Query:document.getElementById('query').value,Repos:document.getElementById('repos').value.split(',').map(s=>s.trim()).filter(Boolean),MinLevel:document.getElementById('minlevel').value,MaxRepos:+document.getElementById('maxrepos').value,MaxFilesPerRepo:+document.getElementById('maxfiles').value,MaxBytesPerRepo:(+document.getElementById('maxmb').value)*1e6,SkipLearned:document.getElementById('skip').checked};
  fetch('/api/learn',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json()).then(res=>{
   if(res.error){line('error: '+res.error,'err');go.disabled=false;return;}
   const es=new EventSource('/api/learn/stream');
   es.onmessage=(m)=>{const e=JSON.parse(m.data);
    if(e.kind==='start')line('▶ learning from '+e.source+' ('+H(e.query||'own repos')+')');
    else if(e.kind==='repo')line((e.ingested?'✓ ':'· ')+e.repo+'  ['+e.level+', '+e.files+' files]'+(e.ingested?' INGESTED':' skipped'+(e.reason?' ('+e.reason+')':'')),e.ingested?'ok':'skip');
    else if(e.kind==='done'){line('done — '+e.ingested+' files ingested','ok');es.close();go.disabled=false;loadStats();loadRepos();}
    else if(e.kind==='error'){line('error: '+e.message,'err');es.close();go.disabled=false;}
   };
   es.onerror=()=>{es.close();go.disabled=false;};
  });
 }
 loadStats();loadRepos();
</script></body></html>`;

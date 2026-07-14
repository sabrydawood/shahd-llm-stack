// The Foundry control-panel page (M13). Realtime over a single WebSocket (/ws): the server pushes
// system + stats snapshots on a ticker, streams Learn progress (overall AND per-repo file-level),
// and the model panel shows the LOADED checkpoint's architecture + per-component parameter
// breakdown. File browsing/viewing stays on plain HTTP. No ${} template holes (this whole file is a
// template literal) — client JS uses string concatenation. Kept in its own file so Dashboard.ts is small.

export const DashboardHtml = `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Shahd — Data Foundry</title>
<style>
 :root{--bg:#0e1116;--panel:#161b22;--line:#262c36;--txt:#e6edf3;--mut:#8b949e;--blue:#388bfd;--green:#3fb950;--red:#f85149;--yellow:#d29922;--pur:#a371f7}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 system-ui,sans-serif}
 header{padding:14px 22px;background:var(--panel);border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px}
 header h1{margin:0;font-size:17px} header .m{color:var(--mut);font-size:12px}
 header .live{margin-left:auto;font-size:12px;display:flex;align-items:center;gap:6px}
 .dot{width:8px;height:8px;border-radius:50%;background:var(--red)} .dot.on{background:var(--green)}
 header a{color:var(--blue);text-decoration:none;font-size:13px}
 .wrap{display:grid;grid-template-columns:390px 1fr;gap:18px;padding:18px;max-width:1340px;margin:0 auto}
 @media(max-width:920px){.wrap{grid-template-columns:1fr}}
 .panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
 .panel+.panel{margin-top:16px}
 h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:0 0 12px;display:flex;justify-content:space-between}
 label{display:block;font-size:12px;color:var(--mut);margin:10px 0 3px}
 input,select{width:100%;background:#0d1117;color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:7px 9px;font:inherit}
 .row{display:flex;gap:8px} .row>div{flex:1}
 .chk{display:flex;align-items:center;gap:8px;margin-top:12px} .chk input{width:auto}
 button{margin-top:14px;width:100%;background:var(--blue);color:#fff;border:0;border-radius:7px;padding:10px;font:600 14px system-ui;cursor:pointer}
 button:disabled{opacity:.5;cursor:not-allowed}
 .cards{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px}
 .card{background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:11px 15px;min-width:92px;flex:1}
 .card b{display:block;font-size:20px} .card span{color:var(--mut);font-size:11px}
 .chips{font-size:12px;color:var(--mut);margin:6px 0;word-break:break-word}
 .srow{display:flex;justify-content:space-between;gap:12px;padding:3px 0;border-top:1px solid var(--line);font-size:12px} .srow:first-child{border-top:0} .srow span{color:var(--mut)} .srow b{font-weight:600;text-align:right;word-break:break-word}
 .pbar{height:9px;background:#0d1117;border:1px solid var(--line);border-radius:6px;overflow:hidden;margin:4px 0} .pfill{height:100%;width:0;background:var(--blue);transition:width .2s} .pfill.rp{background:var(--pur)}
 .plabel{display:flex;justify-content:space-between;font-size:11px;color:var(--mut)}
 .log{margin-top:10px;background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:10px;height:200px;overflow:auto;font:12px ui-monospace,monospace}
 .log div{padding:1px 0;white-space:pre-wrap} .log .t{color:var(--mut)} .ok{color:var(--green)} .skip{color:var(--yellow)} .err{color:var(--red)}
 .mgroup{margin:5px 0} .mgrow{display:flex;justify-content:space-between;font-size:12px;margin-bottom:2px} .mgrow span{color:var(--mut)}
 .mbar{height:7px;background:#0d1117;border:1px solid var(--line);border-radius:5px;overflow:hidden} .mbfill{height:100%;background:var(--pur)}
 .acc{border:1px solid var(--line);border-radius:8px;margin-top:8px;overflow:hidden}
 .acc>.h{display:flex;justify-content:space-between;padding:9px 13px;cursor:pointer;background:#0d1117}
 .acc>.h:hover{background:#11161d} .acc .h .r{color:var(--mut);font-size:12px}
 .acc>.b{display:none;border-top:1px solid var(--line);max-height:340px;overflow:auto} .acc.open>.b{display:block}
 .acc .b table{width:100%;border-collapse:collapse} .acc .b tr{cursor:pointer} .acc .b tr:hover td{background:#11161d}
 .acc .b td{padding:4px 13px;border-top:1px solid var(--line);font:12px ui-monospace,monospace}
 .acc .b td.mut{color:var(--mut);text-align:right;white-space:nowrap}
 .tier-Filtered{color:var(--green)} .tier-Raw{color:var(--yellow)} .tier-Rejected{color:var(--red)}
 .modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.62);align-items:center;justify-content:center;z-index:50;padding:20px}
 .mcard{background:var(--panel);border:1px solid var(--line);border-radius:10px;width:min(940px,94vw);max-height:88vh;display:flex;flex-direction:column}
 .mhead{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid var(--line)} .mhead b{word-break:break-all;font:13px ui-monospace,monospace}
 .mx{cursor:pointer;color:var(--mut);font-size:18px;line-height:1} .mx:hover{color:var(--txt)}
 .mmeta{padding:6px 16px;color:var(--mut);font-size:11px;border-bottom:1px solid var(--line);word-break:break-all}
 .mbody{margin:0;padding:14px 16px;overflow:auto;font:12px ui-monospace,monospace;white-space:pre;tab-size:2;color:var(--txt)}
</style></head><body>
<header><h1>Shahd — Data Foundry</h1><span class="m">learn whole repos · tiered · inspectable</span>
 <span class="live"><span class="dot" id="dot"></span><span id="livetxt">connecting…</span></span>
 <a href="/chat">Chat with the model →</a></header>
<div class="wrap">
 <div>
  <div class="panel">
   <h2>Learn</h2>
   <label>Source</label>
   <select id="source"><option value="github">Public GitHub repos</option><option value="local">Our own repos (local)</option><option value="both">Both</option></select>
   <div id="ghbox"><label>GitHub query</label><input id="query" value="language:typescript stars:>1000"></div>
   <div id="localbox" style="display:none"><label>Local repo paths (comma-separated)</label><input id="repos" value="."></div>
   <div class="row"><div><label>Min level</label><select id="minlevel"><option>medium</option><option>high</option><option>low</option></select></div><div><label>Max repos</label><input id="maxrepos" type="number" value="5"></div></div>
   <div class="row"><div><label>Max files/repo</label><input id="maxfiles" type="number" value="2000"></div><div><label>Max MB/repo</label><input id="maxmb" type="number" value="32"></div><div><label>Max KB/file</label><input id="maxkb" type="number" value="512"></div></div>
   <div class="chk"><input type="checkbox" id="skip" checked><label style="margin:0">Skip repos already learned</label></div>
   <button id="go" onclick="learn()">▶ Learn</button>
  </div>
  <div class="panel"><h2>Progress</h2>
   <div class="plabel"><span>Overall</span><span id="olab">idle</span></div>
   <div class="pbar"><div class="pfill" id="ofill"></div></div>
   <div class="plabel"><span id="rrepo">current repo</span><span id="rlab"></span></div>
   <div class="pbar"><div class="pfill rp" id="rfill"></div></div>
   <div class="log" id="log"></div>
  </div>
 </div>
 <div>
  <div class="panel"><h2>System <span id="systick" style="color:var(--mut)"></span></h2><div id="sys" class="chips">connecting…</div></div>
  <div class="panel"><h2>Model <span style="text-transform:none;color:var(--mut)">— loaded checkpoint</span></h2><div id="model">no model loaded</div></div>
  <div class="panel"><h2>Foundry stats</h2><div class="cards" id="cards"></div><div class="chips" id="langs"></div><div class="chips" id="lics"></div></div>
  <div class="panel"><h2>Learned repos <span style="text-transform:none;color:var(--mut)">— click a file to view it</span></h2><div id="repos-list"></div></div>
 </div>
</div>
<div class="modal" id="modal" onclick="if(event.target===this)closeModal()">
 <div class="mcard"><div class="mhead"><b id="mtitle"></b><span class="mx" onclick="closeModal()">✕</span></div><div class="mmeta" id="mmeta"></div><pre class="mbody" id="mbody"></pre></div>
</div>
<script>
 var H=function(s){return String(s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});};
 var fmtB=function(n){return n>=1e6?(n/1e6).toFixed(1)+'MB':n>=1e3?(n/1e3).toFixed(0)+'KB':n+'B';};
 var fmtN=function(n){return (n||0).toLocaleString();};
 var Q=function(id){return document.getElementById(id);};
 Q('source').onchange=function(e){var v=e.target.value;Q('ghbox').style.display=v==='local'?'none':'';Q('localbox').style.display=v==='github'?'none':'';};
 var FIELDS=['source','query','repos','minlevel','maxrepos','maxfiles','maxmb','maxkb','skip'];
 function saveSettings(){var o={};FIELDS.forEach(function(id){var el=Q(id);o[id]=el.type==='checkbox'?el.checked:el.value;});try{localStorage.setItem('shahd.learn',JSON.stringify(o));}catch(e){}}
 function restoreSettings(){try{var o=JSON.parse(localStorage.getItem('shahd.learn')||'{}');FIELDS.forEach(function(id){if(o[id]===undefined)return;var el=Q(id);if(el.type==='checkbox')el.checked=!!o[id];else el.value=o[id];});Q('source').dispatchEvent(new Event('change'));}catch(e){}}

 function renderSystem(s){
  var gpu=(s.gpu&&s.gpu!=='none detected')?s.gpu:'none';
  var r=function(k,v){return '<div class="srow"><span>'+k+'</span><b>'+v+'</b></div>';};
  Q('sys').innerHTML=
   r('Compute',s.gpuUsed?'GPU':'CPU · '+H(s.computeBackend))+
   r('CPU',H(s.cpuModel)+' × '+s.cpuCount)+
   r('Memory',s.memGb+' GB')+
   r('GPU',H(gpu)+(gpu==='none'||s.gpuUsed?'':' · detected, not used yet'))+
   r('Go FFI kernels',s.goFfiAvailable?'available':'TS fallback')+
   r('Runtime',H(s.runtime)+' · '+H(s.platform)+'/'+H(s.arch));
  Q('systick').textContent='live';
 }
 function renderModel(m){
  if(!m){Q('model').innerHTML='<div class="chips">no model loaded — train a checkpoint first</div>';return;}
  var a=function(k,v){return '<div class="srow"><span>'+k+'</span><b>'+v+'</b></div>';};
  var html=a('Total parameters',fmtN(m.TotalParams))+
   a('Architecture',m.EmbedDim+'d · '+m.NumLayers+'L · '+m.NumHeads+'h')+
   a('Context / vocab',m.BlockSize+' tok · '+fmtN(m.VocabSize))+
   a('Design',H(m.PositionScheme)+' · '+H(m.NormKind)+' · '+H(m.MlpKind)+(m.WeightTying?' · tied':''));
  html+='<div style="margin-top:10px">';
  m.Groups.forEach(function(g){
   html+='<div class="mgroup"><div class="mgrow"><span>'+H(g.Label)+'</span><b>'+fmtN(g.Params)+' · '+g.Pct.toFixed(1)+'%</b></div><div class="mbar"><div class="mbfill" style="width:'+g.Pct.toFixed(1)+'%"></div></div></div>';
  });
  html+='</div>';
  Q('model').innerHTML=html;
 }
 function renderStats(s){
  Q('cards').innerHTML=
   '<div class="card"><b>'+fmtN(s.Total)+'</b><span>documents</span></div>'+
   '<div class="card"><b class="tier-Filtered">'+fmtN(s.ByTier.Filtered)+'</b><span>trainable</span></div>'+
   '<div class="card"><b class="tier-Raw">'+fmtN(s.ByTier.Raw)+'</b><span>raw</span></div>'+
   '<div class="card"><b class="tier-Rejected">'+fmtN(s.ByTier.Rejected)+'</b><span>rejected</span></div>'+
   '<div class="card"><b>'+fmtB(s.FilteredBytes)+'</b><span>trainable bytes</span></div>';
  var kv=function(o){return Object.entries(o).sort(function(a,b){return b[1]-a[1];}).slice(0,14).map(function(e){return H(e[0])+':'+e[1];}).join(' · ');};
  Q('langs').innerHTML='<b>langs</b> '+kv(s.ByLang);
  Q('lics').innerHTML='<b>licenses</b> '+kv(s.ByLicense);
 }

 var maxRepos=5, seen=0, ingested=0, skipped=0, files=0, bytes=0, running=false;
 var ts=function(){return new Date().toTimeString().slice(0,8);};
 function line(t,c){var log=Q('log');var d=document.createElement('div');if(c)d.className=c;d.innerHTML='<span class="t">'+ts()+'</span>  '+H(t);log.appendChild(d);log.scrollTop=log.scrollHeight;}
 function setOverall(f,txt){Q('ofill').style.width=Math.max(0,Math.min(1,f))*100+'%';Q('olab').textContent=txt;}
 function setRepo(f,repo,txt){Q('rfill').style.width=Math.max(0,Math.min(1,f))*100+'%';Q('rrepo').textContent=repo;Q('rlab').textContent=txt;}
 function onLearn(e){
  if(e.kind==='start'){if(e.repos)maxRepos=e.repos;seen=0;ingested=0;skipped=0;files=0;bytes=0;running=true;Q('go').disabled=true;Q('log').innerHTML='';line('▶ learning from '+e.source+' ('+(e.query||'own repos')+')');setOverall(.02,'0 / '+maxRepos+' repos');setRepo(0,'current repo','');}
  else if(e.kind==='repo'){seen++;if(e.ingested){ingested++;files+=e.files;bytes+=e.bytes;}else{skipped++;}line((e.ingested?'✓ ':'· ')+e.repo+'  ['+e.level+', '+e.files+' files, '+fmtB(e.bytes)+']'+(e.ingested?' INGESTED':' skipped'+(e.reason?' ('+e.reason+')':'')),e.ingested?'ok':'skip');setOverall(seen/maxRepos,seen+' / '+maxRepos+' repos · '+ingested+' ingested · '+skipped+' skipped · '+files+' files');}
  else if(e.kind==='repo-progress'){setRepo(e.filesTotal?e.filesDone/e.filesTotal:0,'ingesting '+e.repo,e.filesDone+' / '+e.filesTotal+' files');}
  else if(e.kind==='done'){running=false;Q('go').disabled=false;setOverall(1,'done · '+ingested+' ingested · '+skipped+' skipped · '+e.ingested+' files');setRepo(1,'current repo','complete');line('done — '+e.ingested+' files ingested from '+ingested+' repos ('+skipped+' skipped — already learned)',e.ingested?'ok':'skip');if(e.ingested===0)line('⚠ 0 new files: every matching repo is already learned. Try a different query (other language/topic/star range) or uncheck "Skip repos already learned".','skip');loadRepos();}
  else if(e.kind==='error'){running=false;Q('go').disabled=false;setOverall(seen/maxRepos,'error');line('error: '+e.message,'err');}
 }
 function learn(){
  saveSettings();
  maxRepos=+Q('maxrepos').value||1;
  var settings={Source:Q('source').value,Query:Q('query').value,Repos:Q('repos').value.split(',').map(function(s){return s.trim();}).filter(Boolean),MinLevel:Q('minlevel').value,MaxRepos:maxRepos,MaxFilesPerRepo:+Q('maxfiles').value,MaxBytesPerRepo:(+Q('maxmb').value)*1e6,MaxContentBytes:(+Q('maxkb').value)*1e3,SkipLearned:Q('skip').checked};
  if(WS&&WS.readyState===1){WS.send(JSON.stringify({type:'learn',settings:settings}));}else{line('not connected — retrying…','err');}
 }

 async function loadRepos(){
  var r=await (await fetch('/api/repos')).json();
  Q('repos-list').innerHTML=r.length?r.map(function(x){return '<div class="acc"><div class="h" onclick="openRepo(this,'+JSON.stringify(H(x.Source)).replace(/"/g,'&quot;')+')"><span>'+H(x.Source)+'</span><span class="r">'+x.Files+' files · '+fmtB(x.Bytes)+'</span></div><div class="b"></div></div>';}).join(''):'<div style="color:var(--mut)">nothing learned yet.</div>';
 }
 async function openRepo(h,src){
  var acc=h.parentElement,body=acc.querySelector('.b');
  if(acc.classList.contains('open')){acc.classList.remove('open');return;}
  acc.classList.add('open');
  if(body.dataset.loaded)return; body.dataset.loaded='1';
  var d=await (await fetch('/api/documents?source='+encodeURIComponent(src)+'&limit=2000')).json();
  body.innerHTML='<table>'+d.map(function(f){return '<tr onclick="openFile('+JSON.stringify(f.id).replace(/"/g,'&quot;')+','+JSON.stringify(H(f.path||f.provenance)).replace(/"/g,'&quot;')+')"><td class="tier-'+f.tier+'">'+H(f.path||f.provenance)+'</td><td class="mut">'+H(f.lang)+' · '+fmtB(f.bytes)+'</td></tr>';}).join('')+'</table>';
 }
 async function openFile(id,path){
  var m=Q('modal');Q('mtitle').textContent=path;Q('mmeta').textContent='';Q('mbody').textContent='loading…';m.style.display='flex';
  try{var f=await (await fetch('/api/file?id='+encodeURIComponent(id))).json();
   if(f.error){Q('mbody').textContent='error: '+f.error;return;}
   Q('mmeta').textContent=[f.lang,f.tier,f.origin,f.license,fmtB(f.bytes),f.provenance].join(' · ');
   Q('mbody').textContent=f.content;
  }catch(e){Q('mbody').textContent='failed to load file';}
 }
 function closeModal(){Q('modal').style.display='none';}
 document.addEventListener('keydown',function(e){if(e.key==='Escape')closeModal();});

 var WS=null;
 function connect(){
  WS=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');
  WS.onopen=function(){Q('dot').className='dot on';Q('livetxt').textContent='live';};
  WS.onclose=function(){Q('dot').className='dot';Q('livetxt').textContent='reconnecting…';setTimeout(connect,2000);};
  WS.onmessage=function(ev){var m=JSON.parse(ev.data);
   if(m.type==='system')renderSystem(m.data);
   else if(m.type==='model')renderModel(m.data);
   else if(m.type==='stats')renderStats(m.data);
   else if(m.type==='learn')onLearn(m.event);
  };
 }
 restoreSettings();connect();loadRepos();
</script></body></html>`;

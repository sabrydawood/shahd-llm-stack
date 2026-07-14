// The Foundry control-panel page (M14) — a clear 3-stage pipeline: (1) COLLECT DATA and (2) TRAIN
// MODEL are two side-by-side panels, each with its own settings, progress, and LOG VIEWER, and each
// runs independently (you can collect and train at the same time); (3) CHAT is the linked page. A
// compact info strip (System / Model / Foundry stats) sits on top, and learned repos (with a file
// viewer) below. Realtime over one WebSocket (/ws). No ${} holes (this file is a template literal) —
// client JS uses string concatenation. Kept in its own file so Dashboard.ts stays small.

export const DashboardHtml = `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Shahd — Data Foundry</title>
<style>
 :root{--bg:#0e1116;--panel:#161b22;--line:#262c36;--txt:#e6edf3;--mut:#8b949e;--blue:#388bfd;--green:#3fb950;--red:#f85149;--yellow:#d29922;--pur:#a371f7}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 system-ui,sans-serif}
 header{padding:13px 22px;background:var(--panel);border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px}
 header h1{margin:0;font-size:17px} header .m{color:var(--mut);font-size:12px}
 header .live{margin-left:auto;font-size:12px;display:flex;align-items:center;gap:6px}
 .dot{width:8px;height:8px;border-radius:50%;background:var(--red)} .dot.on{background:var(--green)}
 header a{color:#fff;background:var(--blue);padding:7px 13px;border-radius:7px;text-decoration:none;font-size:13px;font-weight:600}
 .wrap{max-width:1400px;margin:0 auto;padding:16px}
 .info{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:16px}
 .pipes{display:grid;grid-template-columns:1fr 1fr;gap:16px}
 @media(max-width:1000px){.info{grid-template-columns:1fr}.pipes{grid-template-columns:1fr}}
 .panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:15px}
 .panel.stage{border-top:3px solid var(--blue)} .panel.stage.train{border-top-color:var(--pur)}
 h2{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);margin:0 0 11px;display:flex;justify-content:space-between;align-items:center}
 h2 .badge{font-size:10px;padding:2px 8px;border-radius:10px;background:#0d1117;border:1px solid var(--line);text-transform:none;letter-spacing:0}
 h2 .badge.run{background:#132033;border-color:var(--blue);color:var(--blue)} h2 .badge.run.tr{background:#1e1330;border-color:var(--pur);color:var(--pur)}
 h2 .badge.ok{border-color:var(--green);color:var(--green)} h2 .badge.err{border-color:var(--red);color:var(--red)}
 label{display:block;font-size:12px;color:var(--mut);margin:9px 0 3px}
 input,select{width:100%;background:#0d1117;color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:7px 9px;font:inherit}
 .row{display:flex;gap:8px} .row>div{flex:1}
 .chk{display:flex;align-items:center;gap:8px;margin-top:11px} .chk input{width:auto}
 button{margin-top:13px;width:100%;color:#fff;border:0;border-radius:7px;padding:10px;font:600 14px system-ui;cursor:pointer}
 button.collect{background:var(--blue)} button.train{background:var(--pur)} button.stop{background:var(--red)}
 button:disabled{opacity:.5;cursor:not-allowed}
 .hint{font-size:11px;color:var(--mut);margin-top:8px;line-height:1.5}
 .srow{display:flex;justify-content:space-between;gap:10px;padding:3px 0;border-top:1px solid var(--line);font-size:12px} .srow:first-child{border-top:0} .srow span{color:var(--mut)} .srow b{font-weight:600;text-align:right;word-break:break-word}
 .cards{display:flex;gap:8px;flex-wrap:wrap} .card{background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:9px 12px;flex:1;min-width:74px} .card b{display:block;font-size:18px} .card span{color:var(--mut);font-size:10px}
 .chips{font-size:11px;color:var(--mut);margin-top:7px;word-break:break-word}
 .mgroup{margin:4px 0} .mgrow{display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px} .mgrow span{color:var(--mut)}
 .mbar{height:6px;background:#0d1117;border:1px solid var(--line);border-radius:4px;overflow:hidden} .mbfill{height:100%;background:var(--pur)}
 .pbar{height:9px;background:#0d1117;border:1px solid var(--line);border-radius:6px;overflow:hidden;margin:4px 0} .pfill{height:100%;width:0;transition:width .2s;background:var(--blue)} .pfill.rp{background:#2d68b0} .pfill.tr{background:var(--pur)}
 @keyframes pulse{0%,100%{opacity:.35}50%{opacity:1}} .pfill.busy{width:100%;animation:pulse 1.1s ease-in-out infinite;background:var(--yellow);transition:none}
 .spin{display:inline-block;animation:spin 1s linear infinite} @keyframes spin{to{transform:rotate(360deg)}}
 .plabel{display:flex;justify-content:space-between;font-size:11px;color:var(--mut)}
 .log{margin-top:10px;background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:9px;height:190px;overflow:auto;font:11.5px ui-monospace,monospace}
 .log div{padding:1px 0;white-space:pre-wrap} .log .t{color:var(--mut)} .ok{color:var(--green)} .skip{color:var(--yellow)} .err{color:var(--red)} .info-l{color:var(--pur)}
 .acc{border:1px solid var(--line);border-radius:8px;margin-top:8px;overflow:hidden}
 .acc>.h{display:flex;justify-content:space-between;padding:9px 13px;cursor:pointer;background:#0d1117} .acc>.h:hover{background:#11161d} .acc .h .r{color:var(--mut);font-size:12px}
 .acc>.b{display:none;border-top:1px solid var(--line);max-height:320px;overflow:auto} .acc.open>.b{display:block}
 .acc .b table{width:100%;border-collapse:collapse} .acc .b tr{cursor:pointer} .acc .b tr:hover td{background:#11161d}
 .acc .b td{padding:4px 13px;border-top:1px solid var(--line);font:12px ui-monospace,monospace} .acc .b td.mut{color:var(--mut);text-align:right;white-space:nowrap}
 .tier-Filtered{color:var(--green)} .tier-Raw{color:var(--yellow)} .tier-Rejected{color:var(--red)}
 .modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.62);align-items:center;justify-content:center;z-index:50;padding:20px}
 .mcard{background:var(--panel);border:1px solid var(--line);border-radius:10px;width:min(940px,94vw);max-height:88vh;display:flex;flex-direction:column}
 .mhead{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid var(--line)} .mhead b{word-break:break-all;font:13px ui-monospace,monospace}
 .mx{cursor:pointer;color:var(--mut);font-size:18px} .mx:hover{color:var(--txt)}
 .mmeta{padding:6px 16px;color:var(--mut);font-size:11px;border-bottom:1px solid var(--line);word-break:break-all}
 .mbody{margin:0;padding:14px 16px;overflow:auto;font:12px ui-monospace,monospace;white-space:pre;tab-size:2}
</style></head><body>
<header><h1>Shahd — Data Foundry</h1><span class="m">collect → train → chat</span>
 <span class="live"><span class="dot" id="dot"></span><span id="livetxt">connecting…</span></span>
 <a href="/chat">③ Chat →</a></header>
<div class="wrap">
 <div class="info">
  <div class="panel"><h2>System <span id="systick" class="badge"></span></h2><div id="sys">connecting…</div></div>
  <div class="panel"><h2>Model <span style="text-transform:none;color:var(--mut)">loaded checkpoint</span></h2><div id="ckptsel"></div><div id="model">no model</div></div>
  <div class="panel"><h2>Foundry stats</h2><div class="cards" id="cards"></div><div class="chips" id="langs"></div></div>
 </div>
 <div class="pipes">
  <div class="panel stage">
   <h2>① Collect Data <span class="badge" id="cbadge">idle</span></h2>
   <label>Source</label>
   <select id="source"><option value="github">Public GitHub repos</option><option value="local">Our own repos (local)</option><option value="both">Both</option><option value="oasst">OASST conversations (general/chat — Apache-2.0)</option></select>
   <div class="hint">For OASST: put a language in <b>Query</b> (<code>all</code>, <code>en</code>, <code>ar</code>…) and the max conversations in <b>Max repos</b>.</div>
   <div id="ghbox"><label>GitHub query</label><input id="query" value="language:typescript stars:>1000"></div>
   <div id="localbox" style="display:none"><label>Local repo paths</label><input id="repos" value="."></div>
   <div class="row"><div><label>Min level</label><select id="minlevel"><option>medium</option><option>high</option><option>low</option></select></div><div><label>Max repos</label><input id="maxrepos" type="number" value="5"></div></div>
   <div class="row"><div><label>Max files/repo</label><input id="maxfiles" type="number" value="2000"></div><div><label>Max MB/repo</label><input id="maxmb" type="number" value="32"></div><div><label>Max KB/file</label><input id="maxkb" type="number" value="512"></div></div>
   <div class="chk"><input type="checkbox" id="skip" checked><label style="margin:0">Skip repos already collected</label></div>
   <button class="collect" id="cgo" onclick="cbtn()">▶ Collect Data</button>
   <div class="plabel" style="margin-top:11px"><span>Overall</span><span id="colab">idle</span></div>
   <div class="pbar"><div class="pfill" id="cofill"></div></div>
   <div class="plabel"><span id="crepo">current repo</span><span id="crlab"></span></div>
   <div class="pbar"><div class="pfill rp" id="crfill"></div></div>
   <div class="log" id="clog"></div>
  </div>
  <div class="panel stage train">
   <h2>② Train Model <span class="badge" id="tbadge">idle</span></h2>
   <label>Mode</label><select id="tkind"><option value="pretrain">Pretrain — base model (autocomplete)</option><option value="chat">Chat / SFT — replies + tools + thinking</option></select>
   <label>Model name <span style="color:var(--mut)">— train/keep several side by side</span></label><input id="tname" value="foundry">
   <div class="row"><div><label>Embed dim</label><input id="tembed" type="number" value="96"></div><div><label>Layers</label><input id="tlayers" type="number" value="3"></div><div><label>Heads</label><input id="theads" type="number" value="4"></div></div>
   <div class="row"><div><label>Context</label><input id="tctx" type="number" value="96"></div><div><label>Vocab</label><input id="tvocab" type="number" value="512"></div><div><label>Batch</label><input id="tbatch" type="number" value="16"></div></div>
   <div class="row"><div><label>Steps</label><input id="tsteps" type="number" value="500"></div><div><label>Corpus MB</label><input id="tcorpus" type="number" step="0.5" value="1.5"></div></div>
   <button class="train" id="tgo" onclick="tbtn()">▶ Train Model</button>
   <div class="hint">Trains a byte-level model on the collected corpus, then loads it into Chat. Heads must divide Embed. Re-running the same name resumes; a new name trains a separate model. See Docs/MODEL-SCALING.md for tier presets.</div>
   <div class="plabel" style="margin-top:11px"><span id="tstep">step</span><span id="tloss"></span></div>
   <div class="pbar"><div class="pfill tr" id="tfill"></div></div>
   <div class="log" id="tlog"></div>
  </div>
 </div>
 <div class="panel" style="margin-top:16px"><h2>Collected repos <span style="text-transform:none;color:var(--mut)">— click a file to view it</span></h2><div id="repos-list"></div></div>
</div>
<div class="modal" id="modal" onclick="if(event.target===this)closeModal()">
 <div class="mcard"><div class="mhead"><b id="mtitle"></b><span class="mx" onclick="closeModal()">✕</span></div><div class="mmeta" id="mmeta"></div><pre class="mbody" id="mbody"></pre></div>
</div>
<script>
 var H=function(s){return String(s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});};
 var fmtB=function(n){return n>=1e6?(n/1e6).toFixed(1)+'MB':n>=1e3?(n/1e3).toFixed(0)+'KB':n+'B';};
 var fmtN=function(n){return (n||0).toLocaleString();};
 var Q=function(id){return document.getElementById(id);};
 var badge=function(id,txt,cls){var b=Q(id);b.textContent=txt;b.className='badge'+(cls?' '+cls:'');};
 var logLine=function(el,t,c){var d=document.createElement('div');if(c)d.className=c;d.innerHTML='<span class="t">'+new Date().toTimeString().slice(0,8)+'</span>  '+H(t);el.appendChild(d);el.scrollTop=el.scrollHeight;};
 Q('source').onchange=function(e){var v=e.target.value;Q('ghbox').style.display=v==='local'?'none':'';Q('localbox').style.display=v==='github'?'none':'';};
 var FIELDS=['source','query','repos','minlevel','maxrepos','maxfiles','maxmb','maxkb','skip','tkind','tname','tsteps','tcorpus','tembed','tlayers','theads','tctx','tvocab','tbatch'];
 function saveSettings(){var o={};FIELDS.forEach(function(id){var el=Q(id);o[id]=el.type==='checkbox'?el.checked:el.value;});try{localStorage.setItem('shahd.cfg',JSON.stringify(o));}catch(e){}}
 function restoreSettings(){try{var o=JSON.parse(localStorage.getItem('shahd.cfg')||'{}');FIELDS.forEach(function(id){if(o[id]===undefined)return;var el=Q(id);if(el.type==='checkbox')el.checked=!!o[id];else el.value=o[id];});Q('source').dispatchEvent(new Event('change'));}catch(e){}}

 function renderSystem(s){
  var gpu=(s.gpu&&s.gpu!=='none detected')?s.gpu:'none';
  var r=function(k,v){return '<div class="srow"><span>'+k+'</span><b>'+v+'</b></div>';};
  Q('sys').innerHTML=r('Compute',s.gpuUsed?'GPU':'CPU · '+H(s.computeBackend))+r('CPU',H(s.cpuModel)+' × '+s.cpuCount)+r('Memory',s.memGb+' GB')+r('GPU',H(gpu))+r('Go FFI',s.goFfiAvailable?'available':'no')+r('Runtime',H(s.runtime));
  badge('systick','live','ok');
 }
 function renderModel(m){
  if(!m){Q('model').innerHTML='<div class="chips">no model yet — collect data then Train Model.</div>';return;}
  var a=function(k,v){return '<div class="srow"><span>'+k+'</span><b>'+v+'</b></div>';};
  var html=a('Parameters',fmtN(m.TotalParams))+a('Arch',m.EmbedDim+'d · '+m.NumLayers+'L · '+m.NumHeads+'h · ctx'+m.BlockSize)+a('Vocab',fmtN(m.VocabSize)+' ('+H(m.PositionScheme)+'/'+H(m.NormKind)+')');
  html+='<div style="margin-top:8px">';
  m.Groups.forEach(function(g){html+='<div class="mgroup"><div class="mgrow"><span>'+H(g.Label)+'</span><b>'+g.Pct.toFixed(0)+'%</b></div><div class="mbar"><div class="mbfill" style="width:'+g.Pct.toFixed(1)+'%"></div></div></div>';});
  Q('model').innerHTML=html+'</div>';
 }
 function renderStats(s){
  Q('cards').innerHTML='<div class="card"><b>'+fmtN(s.Total)+'</b><span>documents</span></div>'+'<div class="card"><b class="tier-Filtered">'+fmtN(s.ByTier.Filtered)+'</b><span>trainable</span></div>'+'<div class="card"><b class="tier-Rejected">'+fmtN(s.ByTier.Rejected)+'</b><span>rejected</span></div>'+'<div class="card"><b>'+fmtB(s.FilteredBytes)+'</b><span>train bytes</span></div>';
  var kv=function(o){return Object.entries(o).sort(function(a,b){return b[1]-a[1];}).slice(0,10).map(function(e){return H(e[0])+':'+e[1];}).join(' · ');};
  Q('langs').innerHTML='<b>langs</b> '+kv(s.ByLang);
 }

 // ── Start/Stop buttons ──
 var collecting=false,training=false;
 function setBtn(id,mode){var b=Q(id);var base=id==='cgo'?'collect':'train';var idle=id==='cgo'?'▶ Collect Data':'▶ Train Model';if(mode==='run'){b.disabled=false;b.textContent='■ Stop';b.className=base+' stop';}else if(mode==='stopping'){b.disabled=true;b.textContent='stopping…';}else{b.disabled=false;b.textContent=idle;b.className=base;}}
 function cbtn(){if(collecting){if(WS&&WS.readyState===1)WS.send(JSON.stringify({type:'learn-stop'}));setBtn('cgo','stopping');}else collect();}
 function tbtn(){if(training){if(WS&&WS.readyState===1)WS.send(JSON.stringify({type:'train-stop'}));setBtn('tgo','stopping');}else train();}

 // ── ① Collect ──
 var maxRepos=5,seen=0,ing=0,skp=0,files=0;
 function setCol(f,t){Q('cofill').style.width=Math.max(0,Math.min(1,f))*100+'%';Q('colab').textContent=t;}
 function setCrepo(f,r,t){Q('crfill').className='pfill rp';Q('crfill').style.width=Math.max(0,Math.min(1,f))*100+'%';Q('crepo').textContent=r;Q('crlab').textContent=t;}
 function onLearn(e){var log=Q('clog');
  if(e.kind==='start'){if(e.repos)maxRepos=e.repos;seen=0;ing=0;skp=0;files=0;collecting=true;setBtn('cgo','run');badge('cbadge','collecting…','run');Q('clog').innerHTML='';logLine(log,'▶ collecting from '+e.source+' ('+(e.query||'own repos')+')');setCol(.02,'0 / '+maxRepos);setCrepo(0,'current repo','');}
  else if(e.kind==='scanning'){Q('crepo').innerHTML='<span class="spin">⏳</span> '+H(e.label);Q('crlab').textContent='working…';Q('crfill').className='pfill rp busy';logLine(log,'⏳ '+e.label,'skip');}
  else if(e.kind==='repo'){seen++;if(e.ingested){ing++;files+=e.files;}else{skp++;}logLine(log,(e.ingested?'✓ ':'· ')+e.repo+' ['+e.level+', '+e.files+' files]'+(e.ingested?' INGESTED':' skipped'+(e.reason?' ('+e.reason+')':'')),e.ingested?'ok':'skip');setCol(seen/maxRepos,seen+' / '+maxRepos+' · '+ing+' new · '+skp+' skipped');}
  else if(e.kind==='repo-progress'){setCrepo(e.filesTotal?e.filesDone/e.filesTotal:0,'ingesting '+e.repo,e.filesDone+' / '+e.filesTotal+' files');}
  else if(e.kind==='done'){collecting=false;setBtn('cgo','idle');badge('cbadge',e.ingested?'done':'0 new','ok');setCol(1,ing+' new · '+skp+' skipped · '+e.ingested+' files');setCrepo(1,'current repo','complete');logLine(log,'done — '+e.ingested+' files from '+ing+' repos ('+skp+' skipped)',e.ingested?'ok':'skip');if(e.ingested===0)logLine(log,'⚠ 0 new: all matching repos already collected. Try a different query or uncheck Skip.','skip');loadRepos();}
  else if(e.kind==='error'){collecting=false;setBtn('cgo','idle');badge('cbadge','error','err');logLine(log,'error: '+e.message,'err');}
 }
 function collect(){saveSettings();maxRepos=+Q('maxrepos').value||1;var s={Source:Q('source').value,Query:Q('query').value,Repos:Q('repos').value.split(',').map(function(x){return x.trim();}).filter(Boolean),MinLevel:Q('minlevel').value,MaxRepos:maxRepos,MaxFilesPerRepo:+Q('maxfiles').value,MaxBytesPerRepo:(+Q('maxmb').value)*1e6,MaxContentBytes:(+Q('maxkb').value)*1e3,SkipLearned:Q('skip').checked};if(WS&&WS.readyState===1)WS.send(JSON.stringify({type:'learn',settings:s}));else logLine(Q('clog'),'not connected','err');}

 // ── ② Train ──
 var fmtDur=function(ms){var s=Math.round(ms/1000);if(s<90)return s+'s';var m=Math.round(s/60);if(m<90)return m+'m';var h=m/60;if(h<48)return h.toFixed(1)+'h';return (h/24).toFixed(1)+'d';};
 var lastVal='';
 function setTrain(f,step,loss){Q('tfill').style.width=Math.max(0,Math.min(1,f))*100+'%';if(step!==undefined)Q('tstep').textContent=step;if(loss!==undefined)Q('tloss').textContent=loss;}
 function onTrain(e){var log=Q('tlog');
  if(e.kind==='train-start'){lastVal='';training=true;setBtn('tgo','run');badge('tbadge','training…','run tr');Q('tlog').innerHTML='';logLine(log,'▶ training '+e.steps+' steps on the collected corpus…','info-l');setTrain(.01,'step 0 / '+e.steps,'');}
  else if(e.kind==='train-info'){logLine(log,e.text,'info-l');}
  else if(e.kind==='train-progress'){
   var frac=e.steps?e.step/e.steps:0;
   if(typeof e.valLoss==='number')lastVal=' · val '+e.valLoss.toFixed(3);
   var eta='';
   if(e.elapsedMs&&e.step>0){eta=' · ~'+fmtDur(e.elapsedMs/e.step*(e.steps-e.step))+' left';}
   setTrain(frac,'step '+e.step+' / '+e.steps,'loss '+e.trainLoss.toFixed(3)+lastVal+eta);
   if(typeof e.valLoss==='number')logLine(log,'step '+e.step+'/'+e.steps+'  loss '+e.trainLoss.toFixed(3)+' val '+e.valLoss.toFixed(3)+(eta?'  ('+eta.replace(' · ~','~').replace(' left','')+' left)':''));
  }
  else if(e.kind==='train-done'){training=false;setBtn('tgo','idle');badge('tbadge','done','ok');setTrain(1,'complete',Q('tloss').textContent);logLine(log,'✓ trained + saved to '+e.savedTo+' — model reloaded into Chat','ok');}
  else if(e.kind==='train-error'){training=false;setBtn('tgo','idle');var stopped=e.message.indexOf('stop')>=0;badge('tbadge',stopped?'stopped':'error',stopped?'skip':'err');logLine(log,(stopped?'■ ':'error: ')+e.message,stopped?'skip':'err');}
 }
 function train(){saveSettings();var s={Kind:Q('tkind').value,Name:Q('tname').value,Steps:+Q('tsteps').value,CorpusMb:+Q('tcorpus').value,EmbedDim:+Q('tembed').value,NumLayers:+Q('tlayers').value,NumHeads:+Q('theads').value,BlockSize:+Q('tctx').value,Merges:Math.max(0,(+Q('tvocab').value)-256),BatchSize:+Q('tbatch').value};if(WS&&WS.readyState===1)WS.send(JSON.stringify({type:'train',settings:s}));else logLine(Q('tlog'),'not connected','err');}
 // ── chat-model picker ──
 var loadedName='';
 function renderCheckpoints(list){if(!list||!list.length){Q('ckptsel').innerHTML='';return;}var opts=list.map(function(c){return '<option value="'+H(c.Name)+'"'+(c.Name===loadedName?' selected':'')+'>'+H(c.Name)+' — '+fmtN(c.Params)+'p · '+H(c.Arch)+'</option>';}).join('');Q('ckptsel').innerHTML='<label style="margin:0 0 3px">chat model ('+list.length+' saved — pick to switch)</label><select id="ckptdd" onchange="loadModel(this.value)" style="margin-bottom:9px">'+opts+'</select>';}
 function syncCkptSel(){var dd=Q('ckptdd');if(dd&&loadedName)dd.value=loadedName;}
 function loadModel(name){if(WS&&WS.readyState===1)WS.send(JSON.stringify({type:'load-model',name:name}));}

 // ── repos + file viewer ──
 async function loadRepos(){var r=await (await fetch('/api/repos')).json();Q('repos-list').innerHTML=r.length?r.map(function(x){return '<div class="acc"><div class="h" onclick="openRepo(this,'+JSON.stringify(H(x.Source)).replace(/"/g,'&quot;')+')"><span>'+H(x.Source)+'</span><span class="r">'+x.Files+' files · '+fmtB(x.Bytes)+'</span></div><div class="b"></div></div>';}).join(''):'<div style="color:var(--mut)">nothing collected yet.</div>';}
 async function openRepo(h,src){var acc=h.parentElement,body=acc.querySelector('.b');if(acc.classList.contains('open')){acc.classList.remove('open');return;}acc.classList.add('open');if(body.dataset.loaded)return;body.dataset.loaded='1';var d=await (await fetch('/api/documents?source='+encodeURIComponent(src)+'&limit=2000')).json();body.innerHTML='<table>'+d.map(function(f){return '<tr onclick="openFile('+JSON.stringify(f.id).replace(/"/g,'&quot;')+','+JSON.stringify(H(f.path||f.provenance)).replace(/"/g,'&quot;')+')"><td class="tier-'+f.tier+'">'+H(f.path||f.provenance)+'</td><td class="mut">'+H(f.lang)+' · '+fmtB(f.bytes)+'</td></tr>';}).join('')+'</table>';}
 async function openFile(id,path){var m=Q('modal');Q('mtitle').textContent=path;Q('mmeta').textContent='';Q('mbody').textContent='loading…';m.style.display='flex';try{var f=await (await fetch('/api/file?id='+encodeURIComponent(id))).json();if(f.error){Q('mbody').textContent='error: '+f.error;return;}Q('mmeta').textContent=[f.lang,f.tier,f.origin,f.license,fmtB(f.bytes),f.provenance].join(' · ')+(f.reason?'  —  '+f.reason:'');Q('mbody').textContent=f.content;}catch(e){Q('mbody').textContent='failed';}}
 function closeModal(){Q('modal').style.display='none';}
 document.addEventListener('keydown',function(e){if(e.key==='Escape')closeModal();});

 var WS=null;
 function connect(){
  WS=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');
  WS.onopen=function(){Q('dot').className='dot on';Q('livetxt').textContent='live';};
  WS.onclose=function(){Q('dot').className='dot';Q('livetxt').textContent='reconnecting…';setTimeout(connect,2000);};
  WS.onmessage=function(ev){var m=JSON.parse(ev.data);
   if(m.type==='system')renderSystem(m.data);
   else if(m.type==='model'){if(m.name)loadedName=m.name;renderModel(m.data);syncCkptSel();}
   else if(m.type==='checkpoints')renderCheckpoints(m.data);
   else if(m.type==='stats')renderStats(m.data);
   else if(m.type==='learn')onLearn(m.event);
   else if(m.type==='train')onTrain(m.event);
  };
 }
 restoreSettings();connect();loadRepos();
</script></body></html>`;

// The "chat with the model" page (M13). Real conversations with memory: a sidebar of persisted
// conversations, multi-turn context (the server feeds prior turns back), and token-by-token
// streaming over the shared /ws WebSocket. Deliberately HONEST in the banner: this is a tiny
// from-scratch model, so replies are experimental — chat verifies the serving path + shows model
// progress, it is not a usable assistant yet. No ${} holes (this file is a template literal).

export const ChatHtml = `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Shahd — Chat</title>
<style>
 :root{--bg:#0e1116;--panel:#161b22;--line:#262c36;--txt:#e6edf3;--mut:#8b949e;--blue:#388bfd;--green:#3fb950;--red:#f85149;--yellow:#d29922}
 *{box-sizing:border-box} html,body{height:100%} body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.6 system-ui,sans-serif}
 .app{display:grid;grid-template-columns:250px 1fr;height:100vh}
 @media(max-width:760px){.app{grid-template-columns:1fr}.side{display:none}}
 .side{background:var(--panel);border-right:1px solid var(--line);display:flex;flex-direction:column;overflow:hidden}
 .side .top{padding:12px}
 .side h1{margin:0 0 10px;font-size:15px} .side .newb{width:100%;background:var(--blue);color:#fff;border:0;border-radius:7px;padding:8px;font:600 13px system-ui;cursor:pointer}
 .side .list{flex:1;overflow:auto;padding:0 8px 8px}
 .conv{padding:8px 10px;border-radius:7px;cursor:pointer;font-size:13px;color:var(--txt);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:flex;justify-content:space-between;gap:6px;align-items:center}
 .conv:hover{background:#1c222b} .conv.active{background:#22303f}
 .conv .del{color:var(--mut);opacity:0;font-size:14px} .conv:hover .del{opacity:1}
 .side a{display:block;padding:12px;color:var(--blue);text-decoration:none;font-size:13px;border-top:1px solid var(--line)}
 .main{display:flex;flex-direction:column;height:100vh;overflow:hidden}
 .warn{background:#2d2410;border-bottom:1px solid #5a4611;color:var(--yellow);padding:9px 20px;font-size:12.5px}
 .chat{flex:1;overflow:auto;padding:20px;max-width:900px;width:100%;margin:0 auto}
 .msg{margin-bottom:14px;display:flex;gap:10px} .msg .who{flex:0 0 74px;color:var(--mut);font-size:12px;padding-top:3px;text-transform:uppercase;letter-spacing:.04em}
 .msg .txt{flex:1;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:9px 13px;white-space:pre-wrap;word-break:break-word;font:13px ui-monospace,monospace;min-height:20px}
 .msg.user .txt{background:#132033;border-color:#1d3350} .msg.err .txt{border-color:var(--red);color:var(--red)}
 .cursor:after{content:'▋';color:var(--mut);animation:bl 1s steps(2) infinite} @keyframes bl{50%{opacity:0}}
 .foot{border-top:1px solid var(--line);background:var(--panel);padding:12px 20px}
 .opts{max-width:900px;margin:0 auto 8px;display:flex;gap:16px;color:var(--mut);font-size:12px;align-items:center}
 .opts input{width:66px;background:#0d1117;color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:4px 7px;font:inherit}
 .in{max-width:900px;margin:0 auto;display:flex;gap:8px;align-items:flex-end}
 textarea{flex:1;background:#0d1117;color:var(--txt);border:1px solid var(--line);border-radius:8px;padding:9px 11px;font:13px ui-monospace,monospace;resize:vertical;min-height:44px;max-height:200px}
 .send{background:var(--blue);color:#fff;border:0;border-radius:8px;padding:11px 20px;font:600 14px system-ui;cursor:pointer} .send:disabled{opacity:.5;cursor:not-allowed}
</style></head><body>
<div class="app">
 <div class="side">
  <div class="top"><h1>Shahd — Chat</h1><button class="newb" onclick="newChat()">+ New conversation</button></div>
  <div class="list" id="convs"></div>
  <a href="/">← back to Data Foundry</a>
 </div>
 <div class="main">
  <div class="warn">⚠ Tiny from-scratch model — replies are <b>experimental and often incoherent</b>. It keeps conversation memory, but its context window is small; this verifies the serving path, not model quality.</div>
  <div class="chat" id="chat"></div>
  <div class="foot">
   <div class="opts"><label>temp <input id="temp" type="number" step="0.1" value="0.8"></label><label>max tokens <input id="max" type="number" value="160"></label><span id="stat"></span></div>
   <div class="in"><textarea id="box" placeholder="Type a message — Enter to send, Shift+Enter for newline"></textarea><button class="send" id="send" onclick="send()">Send</button></div>
  </div>
 </div>
</div>
<script>
 var H=function(s){return String(s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});};
 var Q=function(id){return document.getElementById(id);};
 var convId=null, streaming=false, curBubble=null, WS=null;

 function uuid(){ return (crypto&&crypto.randomUUID)?crypto.randomUUID():'c-'+Math.random().toString(16).slice(2)+Date.now().toString(16); }
 function addMsg(role,txt,cls){var d=document.createElement('div');d.className='msg '+(cls||role);d.innerHTML='<div class="who">'+H(role)+'</div><div class="txt"></div>';d.querySelector('.txt').textContent=txt;Q('chat').appendChild(d);Q('chat').scrollTop=Q('chat').scrollHeight;return d.querySelector('.txt');}
 function clearChat(){Q('chat').innerHTML='';}

 async function loadConvs(){
  var list=await (await fetch('/api/chat/conversations')).json();
  Q('convs').innerHTML=list.map(function(c){return '<div class="conv'+(c.Id===convId?' active':'')+'" onclick="openConv('+JSON.stringify(c.Id).replace(/"/g,'&quot;')+')"><span>'+H(c.Title)+'</span><span class="del" onclick="event.stopPropagation();delConv('+JSON.stringify(c.Id).replace(/"/g,'&quot;')+')">✕</span></div>';}).join('')||'<div style="color:var(--mut);font-size:12px;padding:8px 10px">no conversations yet</div>';
 }
 async function openConv(id){
  if(streaming)return; convId=id;
  var msgs=await (await fetch('/api/chat/conversation?id='+encodeURIComponent(id))).json();
  clearChat();
  if(!msgs.length){addMsg('system','Empty conversation — say something.');}
  msgs.forEach(function(m){addMsg(m.Role==='assistant'?'model':'you',m.Content,m.Role==='assistant'?'model':'user');});
  loadConvs();
 }
 function newChat(){ if(streaming)return; convId=uuid(); clearChat(); addMsg('system','New conversation. It will be saved once you send a message.'); loadConvs(); }
 async function delConv(id){ await fetch('/api/chat/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:id})}); if(id===convId)newChat(); else loadConvs(); }

 function send(){
  var box=Q('box'), text=box.value.trim();
  if(!text||streaming)return;
  if(!WS||WS.readyState!==1){Q('stat').textContent='not connected';return;}
  if(!convId)convId=uuid();
  box.value=''; addMsg('you',text,'user');
  curBubble=addMsg('model','',''); curBubble.parentElement.classList.add('cursor');
  streaming=true; Q('send').disabled=true; Q('stat').textContent='generating…';
  WS.send(JSON.stringify({type:'chat',convId:convId,message:text,temperature:+Q('temp').value,maxTokens:+Q('max').value}));
 }
 function endStream(){streaming=false;Q('send').disabled=false;if(curBubble)curBubble.parentElement.classList.remove('cursor');curBubble=null;Q('box').focus();}

 function connect(){
  WS=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');
  WS.onopen=function(){Q('stat').textContent='';};
  WS.onclose=function(){Q('stat').textContent='reconnecting…';if(streaming){if(curBubble){curBubble.parentElement.className='msg err';curBubble.textContent='connection lost — resend your message';}endStream();}setTimeout(connect,2000);};
  WS.onmessage=function(ev){var m=JSON.parse(ev.data);
   if(m.type==='chat-delta'&&m.convId===convId&&curBubble){curBubble.textContent+=m.delta;Q('chat').scrollTop=Q('chat').scrollHeight;}
   else if(m.type==='chat-done'&&m.convId===convId){Q('stat').textContent='';endStream();loadConvs();}
   else if(m.type==='chat-error'&&m.convId===convId){if(curBubble){curBubble.parentElement.className='msg err';curBubble.textContent='error: '+m.error;}Q('stat').textContent='';endStream();}
  };
 }
 Q('box').addEventListener('keydown',function(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
 newChat(); connect();
</script></body></html>`;

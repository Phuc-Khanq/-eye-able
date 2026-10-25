(() => {
  const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
  const clamp = (v,a,b) => Math.max(a,Math.min(b,v));
  const wait = ms => new Promise(r => setTimeout(r, ms));
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  /* ---------------- icons ---------------- */
  const ICON = {
    br:'<svg viewBox="0 0 24 24" fill="none" stroke="#2F6FD6" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/></svg>',
    ml:'<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 7l9 6 9-6"/></svg>',
    vt:'<svg viewBox="0 0 24 24"><path d="M9 7l8 5-8 5z" fill="#fff"/></svg>',
    fx:'<svg viewBox="0 0 24 24" fill="#fff"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" opacity=".95"/></svg>'
  };
  $$('.ai').forEach(el => { for (const k of ['br','ml','vt','fx']) if (el.classList.contains(k) && !el.innerHTML) el.innerHTML = ICON[k]; });

  /* ---------------- mascot ---------------- */
  const tpl = $('#mascot-tpl').innerHTML;
  $$('.mascot-slot').forEach((s,i) => { s.outerHTML = tpl.replaceAll('CLIPID','mclip'+i); });
  let mx = innerWidth/2, my = innerHeight/2;
  function lookAll(){
    $$('svg.mascot').forEach(m => {
      const r = m.getBoundingClientRect(); if (!r.width) return;
      const cx = r.left + r.width*0.5, cy = r.top + r.height*0.43;
      const dx = mx-cx, dy = my-cy, d = Math.hypot(dx,dy)||1, k = Math.min(1,d/300)*26;
      m.querySelector('.iris').style.transform = `translate(${dx/d*k}px,${dy/d*k}px)`;
    });
  }
  function mascotDo(cls){ $$('svg.mascot').forEach(m => { m.classList.remove(cls); void m.getBoundingClientRect(); m.classList.add(cls); setTimeout(()=>m.classList.remove(cls),750); }); }
  addEventListener('pointermove', e => { mx=e.clientX; my=e.clientY; lookAll(); labEye(e); }, {passive:true});

  /* ---------------- intro: eye lab ---------------- */




  /* ---------- Controls: expanding cards ---------- */
  (function ctlAcc(){
    const acc = $('#ctlAcc'); if (!acc) return;
    const cards = [...acc.querySelectorAll('.inp')];
    const open = i => { acc.classList.toggle('has-open', i >= 0); cards.forEach((c,k) => c.classList.toggle('open', k===i)); };
    cards.forEach((c,k) => { c.addEventListener('mouseenter', () => open(k)); c.addEventListener('focus', () => open(k)); c.addEventListener('click', () => open(k)); c.addEventListener('blur', () => { if (!acc.matches(':hover')) open(-1); }); });
    acc.addEventListener('mouseleave', () => open(-1));   // nothing hovered: all six cards equal again
  })();
  /* ---------- How it works: soft twinkling dots that stay on the page ---------- */
  (function howDots(){
    const cv = $('#howDots'), page = $('#pg-how'); if (!cv) return;
    const x = cv.getContext('2d'), reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const cols = ['220,228,204','169,184,146','195,143,62','185,133,58','150,150,150','120,128,107'];
    let W = 0, H = 0, D = [];
    function size(){ const dpr = Math.min(2, devicePixelRatio||1); W = innerWidth; H = innerHeight; cv.width = W*dpr; cv.height = H*dpr; x.setTransform(dpr,0,0,dpr,0,0);
      const n = Math.round(W*H/9000); D = Array.from({length:n}, () => ({ x:Math.random()*W, y:Math.random()*H, r:.6 + Math.random()*1.8, c:cols[(Math.random()*cols.length)|0],
        vx:(Math.random()-.5)*6, vy:(Math.random()-.5)*6, ph:Math.random()*6.283, sp:.6 + Math.random()*1.4 })); }
    size(); addEventListener('resize', size);
    let last = performance.now();
    (function frame(now){
      requestAnimationFrame(frame);
      const dt = Math.min(50, now - last)/1000; last = now;
      if (page.hidden || $('#intro').hidden) return;
      x.clearRect(0, 0, W, H); const t = now/1000;
      for (const d of D){
        if (!reduce){ d.x += d.vx*dt; d.y += d.vy*dt; if (d.x < -5) d.x = W+5; if (d.x > W+5) d.x = -5; if (d.y < -5) d.y = H+5; if (d.y > H+5) d.y = -5; }
        const s2 = Math.sin(t*d.sp + d.ph), a = 0.16 + 0.4*(s2*0.5 + 0.5);
        x.globalAlpha = a; x.fillStyle = `rgb(${d.c})`; x.beginPath(); x.arc(d.x, d.y, d.r, 0, Math.PI*2); x.fill();
        if (s2 > 0.95){ x.globalAlpha = 0.6*(s2 - 0.95)/0.05; x.shadowColor = `rgb(${d.c})`; x.shadowBlur = 8; x.beginPath(); x.arc(d.x, d.y, d.r*1.4, 0, Math.PI*2); x.fill(); x.shadowBlur = 0; }
      }
      x.globalAlpha = 1;
    })(last);
  })();
  /* ---------- How it works: circuit wires from eye panel to the 4 steps ---------- */
  (function howWires(){
    const svg = $('#howWires'), grid = $('#howGrid'), labP = $('#howLab');
    if (!svg || !grid || !labP) return;
    const NS = 'http://www.w3.org/2000/svg';
    const wires = [];
    const fr = [0.09, 0.26, 0.58, 0.85];          // where the 4 wires leave the eye panel (fraction of its height), as in the sketch
    function draw(){
      const g = grid.getBoundingClientRect(); if (!g.width || getComputedStyle(svg).display==='none') return;
      // layout box of an element, ignoring any entrance transform (translate / scale around its centre)
      const box = el => { const r = el.getBoundingClientRect(), m = new DOMMatrix(getComputedStyle(el).transform);
        const cx = (r.left + r.right)/2 - m.m41, cy = (r.top + r.bottom)/2 - m.m42, w = el.offsetWidth, h = el.offsetHeight;
        return { left:cx-w/2, right:cx+w/2, top:cy-h/2, bottom:cy+h/2, width:w, height:h }; };
      const L = box(labP), steps = $$('#howSteps .step'), H = L.height;
      svg.setAttribute('viewBox', `0 0 ${g.width} ${g.height}`); svg.innerHTML = '';
      steps.forEach((st, i) => { if (!i) return;            // vertical links joining the 4 step cards
        const a = box(steps[i-1]), b = box(st), x = (b.left + b.right)/2 - g.left;
        const p = document.createElementNS(NS,'path'); p.setAttribute('d', `M${x} ${a.bottom-g.top} V${b.top-g.top}`); p.setAttribute('class','w v'); svg.appendChild(p);
      });
      steps.forEach((st, i) => {
        const r = box(st);
        const sx = L.right - g.left, sy = L.top - g.top + H*fr[i];
        const ex = r.left - g.left, ey = r.top - g.top + r.height/2;
        const X = f => sx + (ex - sx)*f;
        // routes traced from the sketch: each one steps like a circuit trace
        const d = [
          `M${sx} ${sy} H${X(.46)} V${sy + (ey-sy)*.676} H${X(.68)} V${ey} H${ex}`,                         // 1: two steps up
          `M${sx} ${sy} H${X(.25)} V${sy + H*.06} H${X(.63)} V${ey} H${ex}`,                                 // 2: dip, then up
          `M${sx} ${sy} H${X(.34)} V${sy - H*.113} H${X(.55)} V${sy + H*.079} H${X(.84)} V${ey} H${ex}`,     // 3: up, down, up
          `M${sx} ${sy} H${X(.16)} V${sy + H*.176} H${X(.42)} V${ey} H${ex}`                                 // 4: down, then up
        ][i];
        for (const cls of ['w','p','c']){ const p = document.createElementNS(NS,'path'); p.setAttribute('d', d); p.setAttribute('class', cls); if (cls!=='w') p.style.animationDelay = (-i*0.8)+'s'; svg.appendChild(p); if (cls==='p') wires[i] = {p, len:p.getTotalLength(), st}; }
        for (const [x,yy] of [[sx,sy],[ex,ey]]){ const c = document.createElementNS(NS,'circle'); c.setAttribute('cx',x); c.setAttribute('cy',yy); c.setAttribute('r',7); c.setAttribute('class','d'); svg.appendChild(c); }
      });
    }
    // steps glow cyan; brighter when the cyan pulse reaches them
    (function lightLoop(){
      if (!$('#pg-how').hidden) wires.forEach(w => {
        if (!w || !w.p.isConnected) return;
        const an = w.p.getAnimations && w.p.getAnimations()[0]; if (!an || an.currentTime == null) return;
        const off = 300 * (1 - ((an.currentTime % 3200) + 3200) % 3200 / 3200);   // matches @keyframes wireFlow (300 → 0 over 3.2 s)
        const pos = w.len - 4, ph = ((pos + off) % 300 + 300) % 300;              // dash covers phase 0..34
        const now = performance.now(); if (ph < 34) w.t = now;

      });
      requestAnimationFrame(lightLoop);
    })();
    addEventListener('resize', draw);
    new MutationObserver(() => requestAnimationFrame(draw)).observe($('#pg-how'), {attributes:true, attributeFilter:['hidden','class']});
    if ('ResizeObserver' in window) new ResizeObserver(draw).observe(grid);
    draw();
  })();
  const lab = $('#labEye');
  /* 3D eye on the How it works page: iris sits on a sphere, foreshortens as it turns */
  const EYE3 = { cx:221, cy:112, R:150, yaw:0, pitch:0, ty:0, tp:0 };
  (function buildLabEye(){
    if (!lab) return; let sd = 11; const R2 = () => { sd = sd + 0x6D2B79F5 | 0; let t = Math.imul(sd ^ sd >>> 15, 1 | sd); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
    let f = '';
    for (let i=0;i<240;i++){ const a=R2()*Math.PI*2, r0=20+R2()*8, r1=32+R2()*22, pk=R2();
      const col = pk<.45 ? `rgba(232,222,186,${.25+R2()*.35})` : pk<.75 ? `rgba(200,158,90,${.25+R2()*.3})` : `rgba(40,40,30,${.25+R2()*.3})`;
      f += `<line x1="${(Math.cos(a)*r0).toFixed(1)}" y1="${(Math.sin(a)*r0).toFixed(1)}" x2="${(Math.cos(a)*r1).toFixed(1)}" y2="${(Math.sin(a)*r1).toFixed(1)}" stroke="${col}" stroke-width="${(.5+R2()*1.1).toFixed(2)}"/>`; }
    $('#labFibers').innerHTML = f;
    let c = ''; for (let j=0;j<=90;j++){ const a=j/90*Math.PI*2, rr=27+(j%2?3.2:-0.8)+R2()*1.6; c += (j?'L':'M')+(Math.cos(a)*rr).toFixed(1)+' '+(Math.sin(a)*rr).toFixed(1); }
    $('#labCollar').setAttribute('d', c+'Z');
    let l = ''; const P0=[44,112], P1=[220,-34], P2=[398,112];
    for (let i=0;i<17;i++){ const t = 0.16 + i*0.043;
      const x = (1-t)*(1-t)*P0[0]+2*(1-t)*t*P1[0]+t*t*P2[0], y = (1-t)*(1-t)*P0[1]+2*(1-t)*t*P1[1]+t*t*P2[1];
      const dx = 2*(1-t)*(P1[0]-P0[0])+2*t*(P2[0]-P1[0]), dy = 2*(1-t)*(P1[1]-P0[1])+2*t*(P2[1]-P1[1]), n = Math.hypot(dx,dy);
      const nx = dy/n, ny = -dx/n, len = 11 + Math.sin(t*Math.PI)*9;   // outward normal
      l += `<line x1="${(x+nx*6).toFixed(1)}" y1="${(y+ny*6).toFixed(1)}" x2="${(x+nx*(6+len)+dx/n*2).toFixed(1)}" y2="${(y+ny*(6+len)+dy/n*2).toFixed(1)}"/>`; }
    renderLabEye();
  })();
  function renderLabEye(){
    if (!lab) return; const E = EYE3;
    const sy = Math.sin(E.yaw), cyw = Math.cos(E.yaw), sp = Math.sin(E.pitch), cp = Math.cos(E.pitch);
    const ix = E.cx + E.R*sy*0.62, iy = E.cy + E.R*sp*0.5;
    $('#labIris').setAttribute('transform', `translate(${ix.toFixed(1)} ${iy.toFixed(1)}) scale(${(0.35+0.65*cyw).toFixed(3)} ${(0.45+0.55*cp).toFixed(3)})`);
    $('#labPupil').setAttribute('r', (17 + 4*Math.max(0, 1-Math.hypot(E.yaw,E.pitch)*1.6)).toFixed(1));
    $('#labGlint').setAttribute('cx', (ix + 17*(0.6+0.4*cyw)).toFixed(1)); $('#labGlint').setAttribute('cy', (iy - 17).toFixed(1));
    // the eyeball itself: a round sphere outline around the iris (the lids clip it into two curves, like the reference)
    const bx = E.cx + E.R*sy*0.3, by = E.cy + E.R*sp*0.22, BR = 92;
    let m = `<circle cx="${bx.toFixed(1)}" cy="${by.toFixed(1)}" r="${BR}"/><circle cx="${bx.toFixed(1)}" cy="${by.toFixed(1)}" r="${BR-7}" opacity=".35"/>`;
    $('#labMeridians').innerHTML = m;
  }
  (function labLoop(){ const E = EYE3; const k = 0.14;
    if (Math.abs(E.ty-E.yaw)+Math.abs(E.tp-E.pitch) > 0.0005){ E.yaw += (E.ty-E.yaw)*k; E.pitch += (E.tp-E.pitch)*k; renderLabEye(); }
    requestAnimationFrame(labLoop); })();
  function labEye(e){
    if ($('#intro').hidden) return;
    if (!lab) return;
    const r = lab.getBoundingClientRect(); if (!r.width) return;
    const nx = clamp((e.clientX - (r.left + r.width/2)) / (r.width*0.75), -1, 1);
    const ny = clamp((e.clientY - (r.top + r.height/2)) / (r.height*1.2), -1, 1);
    EYE3.ty = nx*0.85; EYE3.tp = ny*0.6;
    $('#roX').textContent = (nx>=0?'+':'') + (nx*0.32).toFixed(2);
    $('#roY').textContent = (ny>=0?'+':'') + (ny*0.18).toFixed(2);
    $('#roP').textContent = Math.round(960 + nx*960) + ', ' + Math.round(540 + ny*540);
  }

  /* ---------------- theme + navigation ---------------- */
  if ($('#themeBtn')) $('#themeBtn').onclick = () => { document.documentElement.classList.toggle('page-light'); };
  function go(view){
    $('#intro').hidden = view!=='intro'; $('#demo').hidden = view!=='demo'; document.body.classList.toggle('demo-on', view==='demo');
    $('#tab-intro').setAttribute('aria-selected', view==='intro');
    $('#tab-demo').setAttribute('aria-selected', view==='demo');
    try { history.replaceState(null,'', view==='demo' ? '#live' : location.pathname + location.search); } catch(e){}
    scrollTo({top:0});
    if (view==='intro' && typeof PEEP_READY !== 'undefined' && PEEP_READY) PEEP.assemble();
    if (view==='demo' && !started){ started = true; log('sys','Session started · 1 face found'); }
    requestAnimationFrame(lookAll);
  }
  let started = false;
  $$('[data-go]').forEach(b => b.addEventListener('click', e => { e.preventDefault(); go(b.dataset.go); }));
  $$('[data-scroll]').forEach(b => b.addEventListener('click', e => { e.preventDefault(); go('intro'); document.getElementById(b.dataset.scroll).scrollIntoView({behavior:'smooth'}); }));

  /* ---------------- toast / log / tip ---------------- */
  let tt; function toast(t){ const el=$('#toast'); el.textContent=t; el.classList.add('show'); clearTimeout(tt); tt=setTimeout(()=>el.classList.remove('show'),3200); }
  const t0 = Date.now();
  const stamp = () => { const s=Math.floor((Date.now()-t0)/1000); return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0'); };
  function log(src,msg){ const d=document.createElement('div'); d.innerHTML=`<time>${stamp()}</time><b class="src ${src}">${{eye:'EYES',head:'HEAD',voice:'VOICE',sys:'SYSTEM'}[src]}</b><span></span>`; d.lastChild.textContent=msg; $('#log').prepend(d); }
  $('#clearLog').onclick = () => $('#log').innerHTML='';
  const tip = t => $('#tip').textContent = t;

  /* ---------------- camera access ---------------- */
  const S = { stream:null, camState:'idle' };
  async function requestCam(){
    if (S.stream) return true;
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('unsupported');
      S.stream = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}, audio:false});
      S.camState = 'on';
      const v = $('#camVideo'); v.srcObject = S.stream; await v.play().catch(()=>{});
      setCamUI(); log('sys','Camera connected');
      return true;
    } catch(e){
      S.camState = 'blocked'; setCamUI();
      toast('Camera is blocked in this preview, so the demo uses a simulated face. Open the page file locally to use your webcam.');
      return false;
    }
  }
  function setCamUI(){
    const on = S.camState==='on', blocked = S.camState==='blocked';
    $('#allowCamTxt').textContent = on ? 'Camera allowed ✓' : blocked ? 'Camera blocked · using simulation' : 'Allow camera';
    $('#demoCamBtn').textContent = on ? 'Camera on' : 'Allow camera';
    $('#camSrc').textContent = on ? 'source: your webcam' : 'source: simulated face';
  }
  /* ---------------- Peep: particle eye that morphs into a face mesh ---------------- */
  const PEEP = (() => {
    const cv = $('#peepCanvas'), ctx = cv.getContext('2d'), slot = $('#heroEye');
    const band = { getBoundingClientRect: () => ({ left:0, top:0, width:innerWidth, height:innerHeight }) };   // the canvas covers the whole viewport
    const rng = seed => () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
    let W = 0, H = 0, dpr = 1, IR = 100, RS = 150, EX = 0, EY = 0, parts = [], links = [], colors = null, colT = 0;
    const st = { gx:0, gy:0, dil:1, hover:false, waiting:false, clickAt:-9999 };
    const T_DRIFT = 700, T_GATHER = 4400, T_HOLD = 7500, T_SCATTER = 2600, CYCLE = T_DRIFT + T_GATHER + T_HOLD + T_SCATTER;
    let aStart = performance.now();
    function phase(now){
      const ct = (now - aStart) % CYCLE, g0 = T_DRIFT, g1 = g0 + T_GATHER, h1 = g1 + T_HOLD;
      if (ct < g0) return { A:0, hold:-1 };
      if (ct < g1) return { A:(ct-g0)/T_GATHER, hold:-1 };
      if (ct < h1) return { A:1, hold:(ct-g1)/1000 };
      return { A:1 - (ct-h1)/T_SCATTER, hold:-1 };
    }
    function paintIris(r){
      const c = document.createElement('canvas'); c.width = c.height = Math.ceil(r*2); const x = c.getContext('2d'); x.translate(r,r);
      const R2 = rng(7);
      let gr = x.createRadialGradient(0,0,0,0,0,r);
      [[0,'#1a0f08'],[0.3,'#24150b'],[0.34,'#6e441a'],[0.41,'#9c6a27'],[0.48,'#b9853a'],[0.54,'#8e8054'],[0.63,'#78806b'],[0.8,'#66705d'],[0.92,'#4d5446'],[1,'#2a2e28']].forEach(([o,cl]) => gr.addColorStop(o,cl));
      x.fillStyle = gr; x.beginPath(); x.arc(0,0,r,0,Math.PI*2); x.fill();
      for (let i=0;i<700;i++){
        const ang = R2()*Math.PI*2, r0 = r*(0.36 + R2()*0.14), r1 = r*(0.58 + R2()*0.39), pick = R2();
        x.strokeStyle = pick<.45 ? `rgba(226,218,184,${.2+R2()*.35})` : pick<.75 ? `rgba(190,152,88,${.2+R2()*.3})` : `rgba(44,42,32,${.2+R2()*.3})`;
        x.lineWidth = 0.6 + R2()*1.6; const bend = (R2()-.5)*0.08, mr = (r0+r1)/2;
        x.beginPath(); x.moveTo(Math.cos(ang)*r0, Math.sin(ang)*r0); x.quadraticCurveTo(Math.cos(ang+bend)*mr, Math.sin(ang+bend)*mr, Math.cos(ang+bend*.5)*r1, Math.sin(ang+bend*.5)*r1); x.stroke();
      }
      x.beginPath();
      for (let j=0;j<=160;j++){ const ang=j/160*Math.PI*2, rr=r*(0.47+(j%2?0.055:-0.015)+R2()*0.03); j? x.lineTo(Math.cos(ang)*rr,Math.sin(ang)*rr) : x.moveTo(Math.cos(ang)*rr,Math.sin(ang)*rr); }
      x.closePath(); x.strokeStyle='rgba(205,145,55,.85)'; x.lineWidth=r*0.04; x.lineJoin='round'; x.stroke();
      for (let i=0;i<90;i++){ const ang=R2()*Math.PI*2, rr=r*(0.56+R2()*0.34); x.fillStyle=`rgba(40,38,30,${.2+R2()*.3})`; x.beginPath(); x.ellipse(Math.cos(ang)*rr,Math.sin(ang)*rr,1+R2()*2.4,0.6+R2()*1.2,ang,0,Math.PI*2); x.fill(); }
      x.strokeStyle='rgba(28,30,24,.9)'; x.lineWidth=r*0.07; x.beginPath(); x.arc(0,0,r*0.965,0,Math.PI*2); x.stroke();
      x.fillStyle='#0b0a0f'; x.beginPath(); x.arc(0,0,r*0.335,0,Math.PI*2); x.fill();
      x.fillStyle='#FFFFFF'; x.beginPath(); x.arc(-r*0.44,-r*0.17,r*0.14,0,Math.PI*2); x.fill();
      x.beginPath(); x.arc(-r*0.42,r*0.22,r*0.085,0,Math.PI*2); x.fill();
      return x.getImageData(0,0,c.width,c.height);
    }
    const THETA_I = 0.66;           // angular radius of the iris on the eyeball (≈ 38°)
    function build(){
      const br = band.getBoundingClientRect(), er = slot.getBoundingClientRect();
      EX = er.left - br.left + er.width/2; EY = er.top - br.top + er.height/2;
      IR = Math.max(70, Math.min(er.width, er.height) * 0.27); RS = IR*1.55;
      const img = paintIris(IR), w = img.width, d = img.data, step = Math.max(8, IR/13);
      const R4 = rng(99); parts = []; links = [];
      const key = new Map();
      // iris + pupil: a hex grid sampled from the painted iris, later wrapped onto the sphere
      for (let y=-IR, row=0; y<=IR; y+=step*0.87, row++){
        for (let x=-IR + (row%2)*step/2, col=0; x<=IR; x+=step, col++){
          const rr = Math.hypot(x,y); if (rr > IR*0.985) continue;
          const px = Math.round(x+IR), py = Math.round(y+IR), k = (clamp(py,0,w-1)*w + clamp(px,0,w-1))*4;
          const isHi = d[k]>235&&d[k+1]>235&&d[k+2]>235, inPupil = rr < IR*0.36;
          if (!isHi && !inPupil && R4() < 0.07) continue;                     // leave random gaps
          const jx = (R4()-.5)*step*0.45, jy = (R4()-.5)*step*0.45;            // break up the grid
          const xx = x + (inPupil ? jx*0.4 : jx), yy = y + (inPupil ? jy*0.4 : jy), r2 = Math.min(Math.hypot(xx,yy), IR*0.985);
          const big = R4();
          const sz = inPupil ? 0.9 + R4()*0.25 : big < 0.06 ? 1.35 + R4()*0.4 : big < 0.3 ? 0.55 + R4()*0.2 : 0.8 + R4()*0.35;
          key.set(row+','+col, parts.length);
          parts.push({ kind:'iris', ox:xx, oy:yy, r:r2, sz, row, col, rgb:[d[k],d[k+1],d[k+2]], hi:isHi,
            sx:R4()*W, sy:R4()*H, dl:R4(), c:(R4()<.5?-1:1)*(0.2+R4()*0.45), ph:R4()*6.283, step });
        }
      }
      parts.forEach((p,i) => {
        [[p.row, p.col+1], [p.row+1, p.col], [p.row+1, p.col + (p.row%2 ? 1 : -1)]].forEach(([r2,c2]) => { const j = key.get(r2+','+c2); if (j !== undefined && R4() < 0.75) links.push([i,j]); });
      });
      // sclera: sparse dots over the rest of the eyeball (Fibonacci sphere), linked to their nearest neighbours
      const sc0 = parts.length, NS = Math.round(4*Math.PI*RS*RS / (step*step*2.6));
      const ga = Math.PI*(3 - Math.sqrt(5));
      for (let i=0;i<NS;i++){
        const zc = 1 - 2*(i+0.5)/NS, th = Math.acos(zc), ph = i*ga;
        if (th < THETA_I + 0.06 || th > 1.95) continue;
        const bg2 = R4(); parts.push({ kind:'sclera', th: th + (R4()-.5)*0.06, phi: ph + (R4()-.5)*0.12, sz: bg2 < 0.06 ? 1.4 + R4()*0.4 : bg2 < 0.3 ? 0.5 + R4()*0.2 : 0.75 + R4()*0.4, rgb:[150,150,150], sx:R4()*W, sy:R4()*H, dl:R4(), c:(R4()<.5?-1:1)*(0.2+R4()*0.45), ph:R4()*6.283, step });
      }
      const loc = i => { const p = parts[i]; return [Math.sin(p.th)*Math.cos(p.phi), Math.sin(p.th)*Math.sin(p.phi), Math.cos(p.th)]; };
      for (let i=sc0;i<parts.length;i++){
        const a = loc(i), near = [];
        for (let j=sc0;j<parts.length;j++){ if (j===i) continue; const b2 = loc(j), dd = (a[0]-b2[0])**2+(a[1]-b2[1])**2+(a[2]-b2[2])**2; near.push([dd,j]); }
        near.sort((x,y)=>x[0]-y[0]); near.slice(0,3).forEach(([,j]) => { if (j > i) links.push([i,j]); });
      }
      // stitch the iris edge to the sclera
      parts.forEach((p,i) => { if (p.kind!=='iris' || p.r < IR*0.9) return; let best=-1, bd=9; const th = p.r/IR*THETA_I, ph = Math.atan2(p.oy,p.ox), a=[Math.sin(th)*Math.cos(ph),Math.sin(th)*Math.sin(ph),Math.cos(th)];
        for (let j=sc0;j<parts.length;j++){ const b2 = loc(j), dd=(a[0]-b2[0])**2+(a[1]-b2[1])**2+(a[2]-b2[2])**2; if (dd<bd){bd=dd;best=j;} } if (best>=0 && R4()<0.6) links.push([i,best]); });
    }
    function size(){ const r = band.getBoundingClientRect(); if (!r.width) return; dpr = Math.min(2, devicePixelRatio||1); W = r.width; H = r.height; cv.width = Math.round(W*dpr); cv.height = Math.round(H*dpr); build(); }
    const ro = new ResizeObserver(size); ro.observe(slot); addEventListener('resize', size); size();
    function col(now){ if (!colors || now-colT>1000){ const cs=getComputedStyle(document.documentElement), v=n=>cs.getPropertyValue(n).trim(); colors={iris:v('--iris'),muted:v('--muted'),ink:v('--ink')}; colT=now; } return colors; }
    const ease = t => t<.5 ? 4*t*t*t : 1-Math.pow(-2*t+2,3)/2;
    const overEye = () => { const r = band.getBoundingClientRect(); return Math.hypot(mx - (r.left+EX), my - (r.top+EY)) < RS; };
    addEventListener('pointermove', () => { st.hover = overEye(); }, {passive:true});
    slot.addEventListener('click', () => { if (overEye()) st.clickAt = performance.now(); });
    function assemble(){ aStart = performance.now(); }
    let PT = null;   // portal transition state (Home → How it works)
    function portal(o){ PT = Object.assign({ t0: performance.now(), x0: EX, y0: EY }, o); }
    function portalEnd(){ PT = null; ctx.setTransform(1,0,0,1,0,0); ctx.clearRect(0,0,cv.width,cv.height); assemble(); }
    const LX = -0.45, LY = -0.6, LZ = 0.66;   // light from the upper left, towards the viewer

    let X = new Float32Array(1), Y = new Float32Array(1), K = new Float32Array(1), V = new Float32Array(1), Z = new Float32Array(1);
    function frame(now){
      requestAnimationFrame(frame);
      if ((!PT && ($('#intro').hidden || $('#pg-home').hidden)) || !W) return;
      if (X.length < parts.length){ X = new Float32Array(parts.length); Y = new Float32Array(parts.length); K = new Float32Array(parts.length); V = new Float32Array(parts.length); Z = new Float32Array(parts.length); }
      const C = col(now), t = now/1000, PH = phase(now); let A = clamp(PH.A, 0, 1);
      if (PT){ const e = now - PT.t0;
        A = e < PT.burst ? A + (1 - A)*clamp(e/220, 0, 1) : 1 - Math.sin(clamp((e - PT.burst)/PT.scatter, 0, 1)*Math.PI/2); }
      ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,H);
      { const er = slot.getBoundingClientRect(), tc = slot.previousElementSibling.getBoundingClientRect(); EX = er.left + er.width/2; EY = tc.height ? tc.top + tc.height/2 : er.top + er.height/2; }
      if (PT){ const u = ease(clamp((now - PT.t0)/PT.move, 0, 1)); EX = PT.x0 + (PT.X - PT.x0)*u; EY = PT.y0 + (PT.Y - PT.y0)*u; }   // centre the eye on the text block   // follow the eye's slot (scroll, resize)
      const r = band.getBoundingClientRect();
      let mdx = clamp((mx - (r.left + EX)) / (innerWidth*0.35), -1, 1), mdy = clamp((my - (r.top + EY)) / (innerHeight*0.4), -1, 1);
      if (st.waiting){ mdx = Math.sin(t*1.3)*0.8; mdy = Math.sin(t*0.7)*0.35; }
      st.gx += (mdx - st.gx)*0.12; st.gy += (mdy - st.gy)*0.12;
      st.dil += ((st.hover ? 1.35 : 1) - st.dil)*0.08;
      const win = (x,a,d) => x>=a && x<a+d ? Math.sin((x-a)/d*Math.PI) : 0;
      const ct = PH.hold, cb = (now - st.clickAt)/1000;
      const blink = (A < 1 || PT) ? 0 : Math.max(win(ct,1.8,0.12), win(ct,5.2,0.12), Math.min(1, win(cb,0,0.3)*1.6));

      // rotation of the eyeball towards the pointer + a slow idle drift
      const yaw = st.gx*0.55 + Math.sin(t*0.4)*0.05, pitch = -st.gy*0.4 + Math.cos(t*0.33)*0.04;
      // upper eyelid: its edge comes down from above the ball and bows downward like a closed, sleepy eye
      const lidBase = -RS*1.1 + RS*1.05*blink, lidSag = RS*0.3*blink;
      const lidY = x => { const e = clamp((x - EX)/(RS*1.02), -1, 1); return EY + lidBase + lidSag*(1 - e*e); };
      const cy_ = Math.cos(yaw), sy_ = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const pr = IR*0.335, edge = IR*0.5, prD = pr*st.dil;

      for (let i=0;i<parts.length;i++){
        const p = parts[i];
        let lx, ly, lz;
        if (p.kind==='iris'){
          let rr = p.r;
          if (rr < edge && rr > 0.01) rr = rr < pr ? rr*st.dil : prD + (rr - pr)*(edge - prD)/(edge - pr);
          const th = rr/IR*THETA_I, ph = Math.atan2(p.oy, p.ox), bulge = 1 + 0.06*Math.cos(th/THETA_I*Math.PI/2); // cornea sits a little proud
          lx = Math.sin(th)*Math.cos(ph)*bulge; ly = Math.sin(th)*Math.sin(ph)*bulge; lz = Math.cos(th)*bulge;
        } else { lx = Math.sin(p.th)*Math.cos(p.phi); ly = Math.sin(p.th)*Math.sin(p.phi); lz = Math.cos(p.th); }
        // yaw (around y) then pitch (around x)
        let x1 = lx*cy_ + lz*sy_, z1 = -lx*sy_ + lz*cy_, y1 = ly*cp - z1*sp, z2 = ly*sp + z1*cp;
        const f = 3.2/(3.2 - z2), tx = EX + x1*RS*f, ty = EY + y1*RS*f;
        let vis = clamp((z2 + 0.25)/0.35, 0, 1);
        if (blink > 0.001){ if (ty < lidY(tx)) vis = 0; else vis *= 1 - 0.88*blink; }
        const k = (PT && now - PT.t0 > PT.burst) ? clamp(A*1.2 - p.dl*0.2, 0, 1) : ease(clamp(A*1.5 - p.dl*0.5, 0, 1));   // gentler, earlier departure during the transition
        let x = tx, y = ty;
        if (k < 1){
          const sx = p.sx + Math.sin(t*0.5 + p.ph)*14*(1-k), sy = p.sy + Math.cos(t*0.6 + p.ph)*10*(1-k);
          const dx = tx - sx, dy = ty - sy, dl = Math.hypot(dx,dy)||1, bend = Math.sin(k*Math.PI)*p.c*dl*0.4;
          x = sx + dx*k - dy/dl*bend; y = sy + dy*k + dx/dl*bend;
        }
        X[i]=x; Y[i]=y; K[i]=k; Z[i]=z2; V[i] = vis*k + (1-k)*0.9;
        p._sh = clamp(0.5 + 0.7*Math.max(0, x1*LX + y1*LY + z2*LZ), 0.45, 1.2);
      }
      // ring
      const Rr = RS*1.14, NRing = 120, spin = t*0.12, scan = (t*0.9) % (Math.PI*2);
      for (let i=0;i<NRing && !(PT && now - PT.t0 > PT.ring);i++){
        const a = i/NRing*Math.PI*2 + spin, d = Math.abs(((a - scan + Math.PI*3) % (Math.PI*2)) - Math.PI), hot = A < 1 && d < 0.35;
        ctx.fillStyle = hot ? C.iris : C.muted; ctx.globalAlpha = (hot ? 0.9 : 0.3) * (0.3 + 0.7*A);
        ctx.beginPath(); ctx.arc(EX + Math.cos(a)*Rr, EY + Math.sin(a)*Rr, hot?2.2:1.3, 0, Math.PI*2); ctx.fill();
      }
      // soft shading behind the eyeball so it reads as a sphere
      if (A > 0.6){
        const g = ctx.createRadialGradient(EX - RS*0.35, EY - RS*0.4, RS*0.1, EX, EY, RS*1.05);
        g.addColorStop(0, 'rgba(255,255,255,0.0)'); g.addColorStop(0.85, 'rgba(60,70,95,0.05)'); g.addColorStop(1, 'rgba(60,70,95,0.14)');
        ctx.globalAlpha = (A-0.6)/0.4; ctx.fillStyle = g; ctx.beginPath(); ctx.arc(EX, EY, RS*1.02, 0, Math.PI*2); ctx.fill();
      }
      // network / wireframe lines
      ctx.lineWidth = 1; ctx.strokeStyle = C.ink;
      for (const [i,j] of links){
        const v = Math.min(V[i], V[j]); if (v < 0.05) continue;
        const dx = X[i]-X[j], dy = Y[i]-Y[j], d2 = dx*dx + dy*dy, maxD = 130; if (d2 > maxD*maxD) continue;
        const kk = (K[i]+K[j])/2, settled = kk > 0.99;
        const depth = clamp(((Z[i]+Z[j])/2 + 0.3)/1.3, 0.15, 1);
        const a = (settled ? (parts[i].kind==='sclera' ? 0.16 : 0.24)*depth : 0.42*(1 - Math.sqrt(d2)/maxD)) * v;
        if (a < 0.02) continue;
        ctx.globalAlpha = a; ctx.beginPath(); ctx.moveTo(X[i],Y[i]); ctx.lineTo(X[j],Y[j]); ctx.stroke();
      }
      // dots, back to front
      const order = [...parts.keys()].sort((a,b)=>Z[a]-Z[b]);
      for (const i of order){
        const p = parts[i], v = V[i]; if (v < 0.03) continue;
        const k = K[i], depth = clamp((Z[i] + 0.3)/1.3, 0.2, 1);
        const sh = k > 0.5 ? p._sh : 1;
        const base = p.kind==='sclera' ? 0.24 : (p.r < pr ? 0.5*st.dil : 0.4);
        const rad = p.step*base*(0.65 + 0.5*depth)*(p.hi ? 1.25 : 1)*p.sz;
        const [cr,cg,cb] = p.rgb, m2 = p.hi ? 1 : sh;
        ctx.globalAlpha = v * (p.kind==='sclera' ? 0.75 : 1) * (0.55 + 0.45*k);
        let tw = 1;   // twinkle while the dots are flying out during the page transition
        if (PT && now - PT.t0 > PT.burst){ const s2 = Math.sin(t*9 + p.ph*7.3) * Math.sin(t*5.3 + p.dl*11); tw = 0.45 + 0.55*Math.abs(s2); if (s2 > 0.93) tw = 1.6; ctx.globalAlpha = Math.min(1, ctx.globalAlpha*tw); }
        ctx.fillStyle = `rgb(${Math.min(255,cr*m2)|0},${Math.min(255,cg*m2)|0},${Math.min(255,cb*m2)|0})`;
        ctx.beginPath(); ctx.arc(X[i], Y[i], rad, 0, Math.PI*2); ctx.fill();
        if (tw > 1.2 && rad > 1){ ctx.globalAlpha = 0.6; ctx.fillStyle = `rgb(${Math.min(255,cr*1.3)|0},${Math.min(255,cg*1.3)|0},${Math.min(255,cb*1.3)|0})`; ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = rad*3; ctx.beginPath(); ctx.arc(X[i], Y[i], rad*1.25, 0, Math.PI*2); ctx.fill(); ctx.shadowBlur = 0;   // a sparkle in the dot's own colour
          ctx.globalAlpha = 0.35; ctx.fillRect(X[i] - rad*2.4, Y[i] - 0.5, rad*4.8, 1); ctx.fillRect(X[i] - 0.5, Y[i] - rad*2.4, 1, rad*4.8); }
      }
      // the lid edge, lashes and crease (drawn only while blinking)
      if (blink > 0.02){
        const hw = Math.sqrt(Math.max(0, RS*RS - (lidBase + lidSag*0.6)**2)) * 1.02, x0 = EX - hw, x1 = EX + hw, N = 28;
        const pt = u => { const x = x0 + (x1-x0)*u; return [x, lidY(x)]; };
        ctx.globalAlpha = Math.min(1, blink*1.6); ctx.strokeStyle = C.ink; ctx.lineCap = 'round';
        // lid line: thicker in the middle
        for (let i=0;i<N;i++){ const [ax,ay] = pt(i/N), [bx,by] = pt((i+1)/N), u = (i+.5)/N; ctx.lineWidth = 1.2 + 2.6*Math.sin(u*Math.PI); ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.stroke(); }
        // lashes hanging down and outwards
        ctx.lineWidth = 1.4;
        for (let i=1;i<18;i++){ const u = i/18, [lx,ly] = pt(u), out = (u-0.5)*2, len = RS*(0.06 + 0.07*Math.sin(u*Math.PI))*blink;
          const ang = Math.PI/2 + out*0.9; ctx.beginPath(); ctx.moveTo(lx,ly); ctx.quadraticCurveTo(lx + Math.cos(ang)*len*0.5, ly + Math.sin(ang)*len*0.6, lx + Math.cos(ang + out*0.4)*len, ly + Math.sin(ang + out*0.4)*len); ctx.stroke(); }
        // crease above the lid
        ctx.globalAlpha = 0.35*blink; ctx.lineWidth = 1.3; ctx.beginPath();
        for (let i=0;i<=N;i++){ const u = 0.12 + 0.76*i/N, [x,y] = pt(u); const yy = y - RS*0.2 - RS*0.05*Math.sin(u*Math.PI); i ? ctx.lineTo(x,yy) : ctx.moveTo(x,yy); }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    requestAnimationFrame(frame);
    return { assemble, portal, portalEnd, ring: () => ({ x:EX, y:EY, r:RS*1.14 }), wait(){ st.waiting = true; }, result(){ st.waiting = S.camState!=='on'; } };
  })();

  /* ---------------- background: a wavy halftone dot field ---------------- */
  (() => {
    const cv = $('#dotField'), x = cv.getContext('2d');
    let W = 0, H = 0, dpr = 1, rgb = '255,255,255', last = 0;
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    function size(){ dpr = Math.min(2, devicePixelRatio || 1); W = innerWidth; H = innerHeight; cv.width = Math.round(W*dpr); cv.height = Math.round(H*dpr); }
    addEventListener('resize', size); size();
    function field(px, py, t){
      // a few slow, interfering waves: their sum makes the bright and dark bands
      const a = Math.sin(px*0.0062 + t*0.32) + Math.sin(py*0.0071 - t*0.27);
      const b = Math.sin((px + py)*0.0043 + t*0.21) + Math.sin((px - py*0.8)*0.0055 - t*0.18);
      const c = Math.sin(Math.hypot(px - W*0.3, py - H*0.6)*0.009 - t*0.4);
      return (a*0.9 + b*0.8 + c*0.7) / 4.8;   // roughly -1..1
    }
    function draw(now){
      requestAnimationFrame(draw);
      if (now - last < 33) return; last = now;          // ~30 fps is plenty for a backdrop
      const t = reduce ? 0 : now/1000;
      rgb = getComputedStyle(document.documentElement).getPropertyValue('--dot').trim() || '255,255,255';
      x.setTransform(dpr,0,0,dpr,0,0); x.clearRect(0,0,W,H);
      const gap = W < 600 ? 13 : 16;
      x.fillStyle = `rgb(${rgb})`;
      for (let gy = gap/2; gy < H + gap; gy += gap){
        for (let gx = gap/2; gx < W + gap; gx += gap){
          const v = field(gx, gy, t);
          const k = Math.max(0, Math.min(1, (v - 0.02) * 2.6));      // brightness of this dot (wide dark gaps between bands)
          if (k < 0.04) continue;
          // swirl the grid a little so rows bend like fabric
          const ox = Math.sin(gy*0.012 + t*0.5)*6 + Math.cos((gx+gy)*0.006 - t*0.3)*4;
          const oy = Math.cos(gx*0.011 - t*0.45)*6;
          x.globalAlpha = 0.1 + 0.7*k;
          x.beginPath(); x.arc(gx + ox, gy + oy, 0.5 + 1.35*k, 0, Math.PI*2); x.fill();
        }
      }
      x.globalAlpha = 1;
    }
  })();

  /* ---------------- intro pages (Home → How it works → Controls → Voice → Gestures) ---------------- */
  const IP = { cur:0, list: $$('.ipage'), names:['Home','How it works','Controls','Voice','Gestures'] };
  /* Home → How it works:
     1) Peep flies to the middle of the screen;  2) its ring opens out into the eye-tracking panel;
     3) Peep's own coloured dots scatter across the whole screen;  4) the panel slowly slides into place, then the steps build in. */
  let portalBusy = false;
  function portalToHow(){
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const slot = $('#heroEye'), r = slot && slot.getBoundingClientRect();
    if (reduce || !r || !r.width || portalBusy || typeof PEEP === 'undefined') return false;
    portalBusy = true;
    const home = $('#pg-home'), how = $('#pg-how'), CX = innerWidth/2, CY = innerHeight/2 + 30;
    const MOVE = 340, RING = 470, BURST = 360, SCATTER = 1900, SHOW = 350;
    home.classList.add('portal-out');
    PEEP.portal({ X:CX, Y:CY, move:MOVE, ring:RING, burst:BURST, scatter:SCATTER });
    setTimeout(() => {
      // page 2 goes live (everything on it still invisible); Home stays on top only to carry Peep's dots
      showPage(1, true); how.classList.add('portal-in');
      home.hidden = false; home.classList.add('portal-keep');
      const lab = $('#howLab'), lr = lab.getBoundingClientRect(), pr = PEEP.ring();
      const dx = CX - (lr.left + lr.width/2), dy = CY - (lr.top + lr.height/2);
      // the ring of dots becomes a glowing ring, which opens out into the panel's frame
      const ring = document.createElement('div'); ring.className = 'portal-ring'; document.body.appendChild(ring);
      ring.style.left = CX+'px'; ring.style.top = CY+'px';
      const d0 = pr.r*2, rad = getComputedStyle(lab).borderTopLeftRadius;
      const W1 = lr.width, H1 = lr.height;
      // one continuous, smoothly eased morph (sampled keyframes, no stops in between): circle → rounded square → frame
      const rEnd = parseFloat(rad) || 18, KF = [], N = 40, T = 1500, tm0 = 80/T, tm1 = 750/T;
      const sm = u => u*u*u*(u*(u*6 - 15) + 10);      // smootherstep: zero speed only at the very start and end
      for (let i = 0; i <= N; i++){
        const o = i/N, u = sm(clamp((o - tm0)/(tm1 - tm0), 0, 1));
        const w = d0 + (W1 - d0)*u, h = d0 + (H1 - d0)*u, r = (d0/2)*(1 - u) + rEnd*u;
        const op = o < tm0 ? 0.5 + 0.5*o/tm0 : o > 0.86 ? Math.max(0, 1 - (o - 0.86)/0.14) : 1;
        KF.push({ width:w+'px', height:h+'px', borderRadius:r+'px', opacity:op, offset:o });
      }
      ring.animate(KF, {duration:T, easing:'linear', fill:'forwards'}).finished.then(() => ring.remove());
      // the real panel takes over from the frame in the centre, waits, then glides slowly across
      lab.animate([
        {transform:`translate(${dx}px,${dy}px)`, opacity:0},
        {transform:`translate(${dx}px,${dy}px)`, opacity:0, offset:.48},
        {transform:`translate(${dx}px,${dy}px)`, opacity:1, offset:.6},
        {transform:`translate(${dx}px,${dy}px)`, opacity:1, offset:.7, easing:'cubic-bezier(.5,0,.12,1)'},
        {transform:'none', opacity:1}
      ], {duration:2150});
      // Peep's dots fade once they have spread out
      setTimeout(() => $('#peepCanvas').animate([{opacity:1},{opacity:0}], {duration:900, fill:'forwards'}), 1450);
      setTimeout(() => { PEEP.portalEnd(); home.classList.remove('portal-keep','portal-out'); home.hidden = true; $('#peepCanvas').getAnimations().forEach(a => a.cancel()); }, 2400);
      // wires draw out of the panel as it arrives; steps follow
      setTimeout(() => {
        $$('#howWires .w').forEach((w, k) => { const L = w.getTotalLength();
          w.animate([{strokeDasharray:`${L} ${L}`, strokeDashoffset:L},{strokeDasharray:`${L} ${L}`, strokeDashoffset:0}], {duration:400, delay:k*70, easing:'cubic-bezier(.5,0,.2,1)', fill:'backwards'}); });
        $$('#howWires .d').forEach((d, k) => d.animate([{opacity:0, transform:'scale(.2)'},{opacity:1, transform:'none'}], {duration:220, delay:(k%2 ? 280 : 0) + (k>>1)*70, fill:'backwards'}));
        $$('#howWires .p, #howWires .c').forEach(q => q.animate([{opacity:0},{opacity:0, offset:.75},{opacity:1}], {duration:900, fill:'backwards'}));
      }, 2020);
      setTimeout(() => { how.classList.remove('portal-in'); portalBusy = false; }, 3400);
    }, SHOW);
    return true;
  }

  /* How it works → Controls: the text slides in first, then the six cards slide in from the right, card 01 leading */
  function controlsIntro(){
    const pg = $('#pg-controls'), head = [pg.querySelector('.ipage-num'), ...pg.querySelectorAll('.sec-head > *')];
    const cards = [...pg.querySelectorAll('#ctlAcc .inp')], nav = pg.querySelector('.ctl-nav');
    const ease = 'cubic-bezier(.22,.9,.24,1)';
    head.forEach((el, k) => el && el.animate([{opacity:0, transform:'translateX(-70px)'},{opacity:1, transform:'none'}], {duration:620, delay:k*90, easing:ease, fill:'backwards'}));
    const right = innerWidth;
    const T0 = 520;                                   // cards start once the heading is mostly in
    cards.forEach((c, k) => {                          // 01 first, then 02 … 06, each sliding in from the right
      const r = c.getBoundingClientRect(), dx = right - r.left + 40;
      const D = 500, at = T0 + k*330;                  // next card leaves as this one is nearly home
      c.animate([{translate:`${dx}px 0`},{translate:'0 0'}], {duration:D, delay:at, easing:'cubic-bezier(.3,.7,.3,1)', fill:'backwards'});
      // motion trail: light smears out behind the card while it moves, then settles
      c.animate([
        {filter:'blur(2px) brightness(1.25) drop-shadow(36px 0 6px rgba(255,255,255,.45)) drop-shadow(90px 0 18px rgba(143,227,106,.35)) drop-shadow(170px 0 34px rgba(255,255,255,.15))', skew:'-6deg 0deg'},
        {filter:'blur(1px) brightness(1.1) drop-shadow(18px 0 6px rgba(255,255,255,.3)) drop-shadow(44px 0 14px rgba(143,227,106,.22)) drop-shadow(80px 0 24px rgba(255,255,255,.08))', skew:'-2deg 0deg', offset:.7},
        {filter:'blur(0px) brightness(1) drop-shadow(0 0 0 rgba(255,255,255,0)) drop-shadow(0 0 0 rgba(143,227,106,0)) drop-shadow(0 0 0 rgba(255,255,255,0))', skew:'0deg 0deg'}
      ], {duration:D, delay:at, easing:'cubic-bezier(.3,.7,.3,1)', fill:'backwards'});
      // ghost copies that lag behind and fade (afterimage)
      [0.12, 0.24].forEach((lag, g) => { const gh = document.createElement('div'); gh.className = 'ctl-ghost';
        const r0 = c.getBoundingClientRect(); Object.assign(gh.style, {left:r0.left+'px', top:r0.top+'px', width:r0.width+'px', height:r0.height+'px'}); document.body.appendChild(gh);
        gh.animate([{translate:`${dx}px 0`, opacity:.0},{opacity:g?.18:.3, offset:.15},{translate:'0 0', opacity:0}], {duration:D*(1+lag*2), delay:at + D*lag, easing:'cubic-bezier(.3,.7,.3,1)', fill:'both'}).finished.then(() => gh.remove()); });
    });
    nav && nav.animate([{opacity:0, transform:'translateY(14px)'},{opacity:1, transform:'none'}], {duration:400, delay:T0 + 5*330 + 460, fill:'backwards'});
  }
  function showPage(i, viaPortal){
    i = clamp(i, 0, IP.list.length-1);
    if (!viaPortal && i === 1 && IP.cur === 0 && !$('#intro').hidden && !$('#pg-home').hidden && portalToHow()) return;
    if ($('#intro').hidden) go('intro');
    const back = i < IP.cur;
    IP.list.forEach((p,k) => { p.hidden = k!==i; p.classList.remove('enter','back'); });
    const ctlIn = !viaPortal && i === 2 && IP.cur === 1 && !matchMedia('(prefers-reduced-motion: reduce)').matches && innerWidth > 1100;
    const p = IP.list[i]; void p.offsetWidth; if (!viaPortal && !ctlIn){ p.classList.add('enter'); if (back) p.classList.add('back'); }
    if (ctlIn) controlsIntro();
    IP.cur = i;
    $$('.pagelinks button').forEach((b,k)=>b.setAttribute('aria-current', k===i));
    scrollTo({top:0});
    requestAnimationFrame(lookAll);
  }
  $('#tab-intro').addEventListener('click', () => showPage(0));
  $$('[data-page-go]').forEach(b => b.addEventListener('click', () => { if ($('#intro').hidden) go('intro'); showPage(+b.dataset.pageGo); }));
  addEventListener('keydown', e => {
    if ($('#intro').hidden || e.target.closest('input,textarea')) return;
    if (e.key==='ArrowRight' || e.key==='PageDown'){ e.preventDefault(); if (IP.cur < IP.list.length-1) showPage(IP.cur+1); }
    if (e.key==='ArrowLeft' || e.key==='PageUp'){ e.preventDefault(); showPage(IP.cur-1); }
  });
  showPage(0);

  /* the intro hero fills exactly the space under the header */
  const setHdr = () => { const t = $('.top'), cs = getComputedStyle(t); document.documentElement.style.setProperty('--hdr', (t.offsetTop + t.offsetHeight + parseFloat(cs.marginBottom)) + 'px'); };
  requestAnimationFrame(setHdr);
  addEventListener('resize', setHdr); setHdr();

  var PEEP_READY = true;
  $('#allowCam').onclick = async () => { PEEP.wait(); await requestCam(); PEEP.result(); };
  $('#demoCamBtn').onclick = requestCam;

  /* ============================== DEMO WORLD ============================== */
  const W = {
    browser:{open:false,min:false,z:1,page:'home',hist:[],fwd:[],video:0,playing:true,liked:false,scroll:0},
    mail:{open:false,min:false,z:1,view:'inbox',read:null,scroll:0},
    top:null, zc:5, focus:null, startOpen:false,
    f:{gq:'',yq:'',to:'',subj:'',body:''},
    sent:[]
  };
  const VIDEOS = [
    ['Gentle morning stretches for seniors','Easy Movement Club','12:04','412K views',200],
    ['Relaxing piano music for reading','Quiet Keys','1:02:11','2.1M views',260],
    ['How to video call your family','Tech Made Simple','8:37','96K views',170],
    ['Easy vegetable soup in 20 minutes','Home Kitchen Diary','14:20','310K views',30],
    ['Balcony garden for beginners','Green Corner','10:51','188K views',120],
    ['Evening news summary','Daily Brief','6:15','54K views',220],
    ['Walking tour of an old riverside town','Slow Travel','42:08','870K views',190],
    ['Breathing exercises for better sleep','Calm Breath','9:12','640K views',280],
  ];
  const MAILS = [
    ['Minh Anh','Lunch on Sunday?','Hi! Are you free for lunch this Sunday? The new noodle place opened near the park.','9:12 AM',true],
    ['City Clinic','Appointment reminder · Tue 10:30','This is a reminder of your check-up on Tuesday at 10:30. Reply YES to confirm.','8:40 AM',true],
    ['Grandkids','Photos from the beach trip 🌊','We made you an album from last weekend. Look at the sandcastle!','Yesterday',false],
    ['Library','Your book is ready to collect','The large-print edition you reserved is ready at the front desk for 7 days.','Mon',false],
  ];
  const thumbBg = h => `linear-gradient(135deg,hsl(${h} 62% 58%),hsl(${(h+40)%360} 55% 38%))`;

  const field = (name, ph) => {
    const v = W.f[name], foc = W.focus===name;
    return `<div class="field${foc?' focus':''}" data-act="focus:${name}">${name==='gq'||name==='yq'?'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#5f6368" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M21 21l-5-5"/></svg>':''}${v?`<span>${esc(v)}</span>`:(foc?'':`<span class="ph">${ph}</span>`)}${foc?'<span class="caret"></span>':''}</div>`;
  };

  function renderBrowser(){
    const b = W.browser, P = $('#bPage');
    let html='', url='google.com', title='Google', fav='#9AA6BD';
    if (b.page==='home'){
      html = `<div class="g-home"><div class="g-word">Google</div>${field('gq','Search the web')}
        <div class="g-btns"><button class="g-btn" data-act="gsearch">Google Search</button><button class="g-btn" data-act="lucky">I'm Feeling Lucky</button></div>
        <div class="shortcuts">
          <button class="sc" data-act="nav:youtube"><span class="ai vt">${ICON.vt}</span>YouTube</button>
          <button class="sc" data-act="open:mail"><span class="ai ml">${ICON.ml}</span>Mail</button>
          <button class="sc" data-act="quick:weather today"><span class="ai fx">☀</span>Weather</button>
          <button class="sc" data-act="quick:news"><span class="ai br">${ICON.br}</span>News</button>
        </div></div>`;
    } else if (b.page==='gresults'){
      const q = W.f.gq; url='google.com/search?q='+encodeURIComponent(q); title=q+' - Google Search';
      const R = [
        ['news.example.com',`${q}: latest updates`,`Up-to-date information about ${q}, refreshed every hour.`,'res'],
        ['youtube.com',`${q} - Videos`,`Watch videos about ${q}. Open in YouTube.`,'yt'],
        ['learn.example.org',`What you should know about ${q}`,`A short, easy-to-read guide written in plain language.`,'res'],
        ['forum.example.net',`People also ask about ${q}`,`Answers from the community, sorted by most helpful.`,'res'],
      ];
      html = `<div class="g-res"><div class="g-res-top"><div class="g-word">Google</div>${field('gq','Search')}</div>
        <div class="res-list"><small style="color:#70757a;font-size:12.5px">About 1,240,000 results (0.31 seconds)</small>
        ${R.map(([d,t,s,k])=>`<button class="res" data-act="${k==='yt'?'ytsearch:'+esc(q):'toast:Opening '+d}"><small>${d}</small><b>${esc(t)}</b><p>${esc(s)}</p></button>`).join('')}</div></div>`;
    } else if (b.page==='youtube' || b.page==='yresults'){
      url = b.page==='youtube' ? 'youtube.com' : 'youtube.com/results?search_query='+encodeURIComponent(W.f.yq);
      title = b.page==='youtube' ? 'YouTube' : W.f.yq+' - YouTube'; fav='#E2434E';
      const top = `<div class="yt-top"><button class="yt-logo" data-act="nav:youtube"><i></i>YouTube</button>${field('yq','Search')}<button class="yt-sbtn" data-act="ysearch" aria-label="Search"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#111" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M21 21l-5-5"/></svg></button></div>`;
      if (b.page==='youtube'){
        html = `<div class="yt">${top}<div class="yt-chips"><span>All</span><span>Music</span><span>Cooking</span><span>Exercise</span><span>News</span><span>Travel</span></div>
          <div class="vgrid">${VIDEOS.map((v,i)=>`<button class="vcard" data-act="watch:${i}"><div class="thumb" style="background:${thumbBg(v[4])}">▶<span class="len">${v[2]}</span></div><b>${esc(v[0])}</b><small>${esc(v[1])} · ${v[3]}</small></button>`).join('')}</div></div>`;
      } else {
        const q = W.f.yq;
        const res = [0,1,2,3].map(i => { const base = VIDEOS[(i*3+1)%8]; return [i===0?cap(q):`${cap(q)} · ${['full playlist','for beginners','1 hour','live'][i]}`, base[1], base[2], base[3], (base[4]+i*37)%360, (i*3+1)%8]; });
        html = `<div class="yt">${top}<div class="vlist">${res.map((v,i)=>`<button class="vrow" data-act="watchq:${i}"><div class="thumb" style="background:${thumbBg(v[4])}">▶<span class="len">${v[2]}</span></div><div><b>${esc(v[0])}</b><small>${v[3]} · 2 weeks ago</small><small>${esc(v[1])}</small></div></button>`).join('')}</div></div>`;
      }
    } else if (b.page==='watch'){
      const v = b.watchData; url='youtube.com/watch?v=demo'+b.video; title=v[0]+' - YouTube'; fav='#E2434E';
      html = `<div class="yt"><div class="yt-top"><button class="yt-logo" data-act="nav:youtube"><i></i>YouTube</button>${field('yq','Search')}<button class="yt-sbtn" data-act="ysearch" aria-label="Search"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#111" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M21 21l-5-5"/></svg></button></div>
        <div class="watch"><div><div class="player${b.playing?'':' paused'}" style="background:${thumbBg(v[4])}"><div class="viz">${'<i></i>'.repeat(9)}</div>
          <div class="pbar"><button class="pbtn" data-act="play">${b.playing?'❚❚':'▶'}</button><div class="track"><i id="ptrack"></i></div><span id="ptime">0:00 / ${v[2]}</span></div></div>
          <div class="wmeta"><h4>${esc(v[0])}</h4><div class="row"><b style="font-size:13px">${esc(v[1])}</b><button class="pillbtn on" data-act="toast:Subscribed">Subscribe</button><span class="grow"></span><button class="pillbtn${b.liked?' on':''}" data-act="like">👍 ${b.liked?'Liked':'Like'}</button></div></div></div>
          <div class="upnext">${[2,4,7].map(i=>`<button class="vrow" data-act="watch:${i}"><div class="thumb" style="background:${thumbBg(VIDEOS[i][4])}"><span class="len">${VIDEOS[i][2]}</span></div><div><b>${esc(VIDEOS[i][0])}</b><small>${esc(VIDEOS[i][1])}</small></div></button>`).join('')}</div></div></div>`;
      // delay to stagger the in-video items (removed)
    }
    P.innerHTML = html;
    P.style.transform = `translateY(${-b.scroll}px)`;
    $('#omni').textContent = url; $('#tabTitle').textContent = title; $('#tabFav').style.background = fav;
  }
  const cap = s => s ? s[0].toUpperCase()+s.slice(1) : s;

  function renderMail(){
    const m = W.mail, P = $('#mPage');
    const unread = MAILS.filter(x=>x[4]).length;
    const side = `<div class="mside"><button class="newmail" data-act="compose">✎ New mail</button>
      <button class="mfold${m.view!=='sent'?' sel':''}" data-act="inbox"><span>Inbox</span><span>${unread}</span></button>
      <button class="mfold${m.view==='sent'?' sel':''}" data-act="sentbox"><span>Sent</span><span>${W.sent.length||''}</span></button>
      <button class="mfold" data-act="toast:Drafts is empty"><span>Drafts</span><span></span></button></div>`;
    let main='';
    if (m.view==='inbox'){
      main = `<div class="mhead">Inbox<span style="font-size:12px;color:#5f6470;font-weight:400">${MAILS.length} messages</span></div>${MAILS.map((x,i)=>`<button class="mrow${x[4]?' unread':''}" data-act="read:${i}"><b>${esc(x[0])}</b><span><b>${esc(x[1])}</b> · ${esc(x[2])}</span><time>${x[3]}</time></button>`).join('')}`;
    } else if (m.view==='sent'){
      main = `<div class="mhead">Sent</div>${W.sent.length?W.sent.map(x=>`<div class="mrow"><b>To: ${esc(x.to)}</b><span><b>${esc(x.subj||'(no subject)')}</b> · ${esc(x.body)}</span><time>${x.time}</time></div>`).join(''):'<p style="padding:16px;color:#5f6470;font-size:13px">Nothing sent yet.</p>'}`;
    } else if (m.view==='read'){
      const x = MAILS[m.read];
      main = `<div class="mhead"><span>${esc(x[1])}</span><button class="ghostbtn" data-act="reply">↩ Reply</button></div><div class="reading"><div class="from">From ${esc(x[0])} · ${x[3]}</div><p>${esc(x[2])}</p></div>`;
    } else if (m.view==='compose'){
      main = `<div class="mhead">New message</div><div class="compose">
        <div class="cfield"><span>To</span>${field('to','Recipient')}</div>
        <div class="cfield"><span>Subject</span>${field('subj','Add a subject')}</div>
        <div class="cfield body"><span>Message</span>${field('body','Click here, then speak or type')}</div>
        <div class="crow"><button class="ghostbtn" data-act="discard">Discard</button><button class="sendbtn" data-act="send">Send ➤</button></div></div>`;
    }
    P.innerHTML = `<div class="mailapp">${side}<div class="mmain">${main}</div></div>`;
  }

  const SUG = { gq:['weather today','news','easy recipes','bus times'], yq:['piano music','exercise for seniors','cooking','news'],
    to:['mai@example.com','son@example.com','clinic@example.com'], subj:['Dinner on Sunday','Thank you','Quick question'], body:['Hello,','Thank you so much','See you soon','Love, Grandma'] };
  function renderOSK(){
    const o = $('#osk');
    screen.classList.toggle('kb', !!W.focus);
    if (!W.focus){ o.hidden = true; return; }
    o.hidden = false;
    const rows = 'qwertyuiopasdfghjkl@zxcvbnm.,?';
    o.innerHTML = `<div class="osk-label"><span>On-screen keyboard · point at a key and click</span><span>typing into: ${{gq:'search',yq:'video search',to:'To',subj:'Subject',body:'Message'}[W.focus]}</span></div>
      <div class="osk-sug">${(SUG[W.focus]||[]).slice(0,4).map(s=>`<button data-act="sug:${esc(s)}">${esc(s)}</button>`).join('')}</div>
      <div class="osk-keys">${rows.split('').map(c=>`<button class="k" data-act="key:${c}">${c}</button>`).join('')}
      <button class="k w2" data-act="key:hide">⌄ Hide</button><button class="k w4" data-act="key:space">space</button><button class="k w2" data-act="key:bksp">⌫</button><button class="k w2 acc" data-act="key:enter">${W.focus==='body'?'↵':'Enter'}</button></div>`;
  }

  function renderAll(){
    const b=W.browser, m=W.mail;
    $('#winBrowser').hidden = !(b.open && !b.min); $('#winMail').hidden = !(m.open && !m.min);
    $('#winBrowser').style.zIndex = b.z; $('#winMail').style.zIndex = m.z;
    if (b.open) renderBrowser();
    if (m.open) renderMail();
    renderOSK();
    $('#startMenu').hidden = !W.startOpen;
    $('#tbBrowser').classList.toggle('open', b.open); $('#tbMail').classList.toggle('open', m.open);
    $('#tbBrowser').classList.toggle('active', b.open && W.top==='browser'); $('#tbMail').classList.toggle('active', m.open && W.top==='mail');
  }
  function bringTop(w){ W[w].z = ++W.zc; W.top = w; }
  function openWin(w){ const x=W[w]; if (!x.open){ x.open=true; log('sys', (w==='browser'?'Chrome':'Mail')+' opened'); } x.min=false; bringTop(w); W.startOpen=false; }
  function nav(page){ const b=W.browser; if (b.page!==page || page==='yresults' || page==='gresults'){ b.hist.push({page:b.page,video:b.video,watchData:b.watchData}); b.fwd=[]; } b.page=page; b.scroll=0; }
  function wtoast(title, body){ const s=$('#wtoastSlot'); s.innerHTML=`<div class="wtoast"><span class="ai ml" style="width:28px;height:28px;border-radius:7px;flex:none">${ICON.ml}</span><div><b>${esc(title)}</b>${esc(body)}</div></div>`; clearTimeout(wtoast.t); wtoast.t=setTimeout(()=>s.innerHTML='',3200); }

  function topScroller(){ return W.top==='mail' && W.mail.open ? 'mail' : (W.browser.open ? 'browser' : null); }
  function scrollBy(dir){
    const t = topScroller(); if (!t) return;
    if (t==='browser'){ const inner=$('#bPage'), max=Math.max(0, inner.scrollHeight - inner.parentElement.clientHeight); W.browser.scroll = clamp(W.browser.scroll + dir*180, 0, max); inner.style.transform=`translateY(${-W.browser.scroll}px)`; }
  }

  const ACTIONS = {};
  /* main action dispatcher: everything clickable has data-act */
  function run(a, src='eye'){
    if (!a) return;
    const [k, ...rest] = a.split(':'); const v = rest.join(':');
    const b = W.browser, m = W.mail;
    if (ACTIONS[k]){ ACTIONS[k](v, src); return; }   // actions that js/engine.js registers (the Diseyeble card's buttons)
    $('#ctx').hidden = true;
    switch(k){
      case 'open':
        if (v==='youtube'){ openWin('browser'); nav('youtube'); log(src,'Opened YouTube'); }
        else { openWin(v); if (v==='browser' && !b.hist.length && b.page!=='home') b.page='home'; }
        break;
      case 'close': W[v].open=false; if (W.focus && ((v==='mail' && ['to','subj','body'].includes(W.focus)) || (v==='browser' && ['gq','yq'].includes(W.focus)))) W.focus=null;
        if (v==='browser'){ Object.assign(b,{page:'home',hist:[],fwd:[]}); W.f.gq=''; W.f.yq=''; }
        W.top = W.browser.open ? 'browser' : (W.mail.open ? 'mail' : null); log(src,(v==='browser'?'Chrome':'Mail')+' closed'); break;
      case 'min': W[v].min=true; W.top=null; break;
      case 'start': W.startOpen=!W.startOpen; break;
      case 'nav': nav(v); break;
      case 'quick': W.f.gq=v; nav('gresults'); log(src,'Searched Google for "'+v+'"'); W.focus=null; break;
      case 'gsearch': if (W.f.gq.trim()){ nav('gresults'); W.focus=null; log(src,'Searched Google for "'+W.f.gq+'"'); } else { W.focus='gq'; } break;
      case 'lucky': W.f.gq = W.f.gq || 'piano music'; W.f.yq = W.f.gq; nav('yresults'); W.focus=null; break;
      case 'ysearch': if (W.f.yq.trim()){ nav('yresults'); W.focus=null; log(src,'Searched YouTube for "'+W.f.yq+'"'); } else { W.focus='yq'; } break;
      case 'ytsearch': W.f.yq=v; nav('yresults'); break;
      case 'watch': nav('watch'); b.video=+v; b.watchData=VIDEOS[+v]; b.playing=true; b.liked=false; b.t0=performance.now(); log(src,'Playing "'+VIDEOS[+v][0]+'"'); break;
      case 'watchq': { const base=VIDEOS[(+v*3+1)%8]; const t= +v===0?cap(W.f.yq):`${cap(W.f.yq)} · ${['full playlist','for beginners','1 hour','live'][+v]}`; nav('watch'); b.video=100+(+v); b.watchData=[t,base[1],base[2],base[3],(base[4]+(+v)*37)%360]; b.playing=true; b.liked=false; b.t0=performance.now(); log(src,'Playing "'+t+'"'); break; }
      case 'play': b.playing=!b.playing; log(src, b.playing?'Video playing':'Video paused'); break;
      case 'like': b.liked=!b.liked; break;
      case 'back': if (b.hist.length){ b.fwd.push({page:b.page,video:b.video,watchData:b.watchData}); Object.assign(b,b.hist.pop()); b.scroll=0; log(src,'Went back'); } break;
      case 'fwd': if (b.fwd.length){ b.hist.push({page:b.page,video:b.video,watchData:b.watchData}); Object.assign(b,b.fwd.pop()); } break;
      case 'reload': log(src,'Page reloaded'); break;
      case 'focus': W.focus=v; break;
      case 'key': typeKey(v); break;
      case 'sug': { const f=W.focus; if (f){ const cur=W.f[f], mm=cur.match(/(\S+)$/); if (mm && v.toLowerCase().startsWith(mm[1].toLowerCase())) W.f[f]=cur.slice(0,cur.length-mm[1].length)+v; else if (f==='body' && /^love/i.test(v)) W.f.body=(W.f.body.trim()+'\n\n'+v).slice(0,400); else insertText(v,true); } break; }
      case 'compose': m.view='compose'; W.f.to=W.f.to||''; W.focus='to'; break;
      case 'reply': { const x=MAILS[m.read]; m.view='compose'; W.f.to=x[0].toLowerCase().replace(/\s+/g,'.')+'@example.com'; W.f.subj='Re: '+x[1]; W.focus='body'; break; }
      case 'inbox': m.view='inbox'; break;
      case 'sentbox': m.view='sent'; break;
      case 'read': m.view='read'; m.read=+v; MAILS[+v][4]=false; break;
      case 'discard': W.f.to=W.f.subj=W.f.body=''; m.view='inbox'; W.focus=null; break;
      case 'send':
        if (!W.f.to.trim()){ W.focus='to'; toast('Add a recipient first'); break; }
        W.sent.unshift({to:W.f.to,subj:W.f.subj,body:W.f.body,time:new Date().toLocaleTimeString([], {hour:'numeric',minute:'2-digit'})});
        log(src,'Email sent to '+W.f.to); wtoast('Mail','Message sent to '+W.f.to);
        W.f.to=W.f.subj=W.f.body=''; m.view='sent'; W.focus=null; tip('Email sent! You did that with your eyes, your brows and your voice.'); break;
      case 'toast': toast(v); break;
      case 'ctx': ctxAction(v, src); return;
    }
    if (k==='open' || k==='nav' || k==='close') W.startOpen = false;
    // bring the window you're working in to the top
    const winOf = { compose:'mail', inbox:'mail', sentbox:'mail', read:'mail', send:'mail', discard:'mail', reply:'mail' };
    if (winOf[k]) bringTop('mail');
    renderAll();
  }
  function insertText(t, word){
    const f = W.focus; if (!f) return;
    let cur = W.f[f];
    if (word && cur && !/\s$/.test(cur)) cur += ' ';
    W.f[f] = (cur + t).slice(0, f==='body'?400:80);
  }
  function typeKey(v){
    const f = W.focus; if (!f) return;
    if (v==='hide'){ W.focus=null; return; }
    if (v==='bksp'){ W.f[f]=W.f[f].slice(0,-1); return; }
    if (v==='space'){ W.f[f]+=' '; return; }
    if (v==='enter'){
      if (f==='gq') return run('gsearch');
      if (f==='yq') return run('ysearch');
      if (f==='to') { W.focus='subj'; return; }
      if (f==='subj') { W.focus='body'; return; }
      if (f==='body') { W.f.body+='\n'; return; }
    }
    W.f[f] = (W.f[f] + v).slice(0, f==='body'?400:80);
  }

  /* ---------------- context menu (head shake) ---------------- */
  function openCtx(x,y){
    const c = $('#ctx');
    c.innerHTML = `<div class="ctx-tag">RIGHT CLICK · MOUTH OPEN</div>
      <button data-act="ctx:back">Back <small>Alt+←</small></button><button data-act="ctx:reload">Reload <small>Ctrl+R</small></button>
      <hr><button data-act="ctx:copy">Copy <small>Ctrl+C</small></button><button data-act="ctx:paste">Paste <small>Ctrl+V</small></button>
      <hr><button data-act="ctx:close">Close window</button><button data-act="ctx:cancel">Cancel</button>`;
    const sr = screen.getBoundingClientRect();
    c.style.left = clamp(x, 8, sr.width-220)+'px'; c.style.top = clamp(y, 8, sr.height-260)+'px'; c.hidden = false;
  }
  let clip = '';
  function ctxAction(v, src){
    $('#ctx').hidden = true;
    if (v==='back') run('back', src);
    else if (v==='reload') run('reload', src);
    else if (v==='copy'){ clip = W.focus ? W.f[W.focus] : (W.browser.watchData?.[0] || 'diseyeble'); toast('Copied'); }
    else if (v==='paste'){ if (W.focus){ insertText(clip,true); renderAll(); } else toast('Select a text box first'); }
    else if (v==='close'){ const t = W.top || (W.browser.open?'browser':W.mail.open?'mail':null); if (t) run('close:'+t, src); }
  }

  /* ---------------- gaze engine ---------------- */
  const screen = $('#screen'), gazeEl = $('#gaze'), prog = $('#prog');
  const G = { target:{x:380,y:260}, pos:{x:380,y:260}, steady:.7, jitter:9, paused:false, scripted:false, seen:false,
    brow:true, mouth:true, tilt:true, dwell:false, dwellMs:1000, calibrated:false, pending:null };
  let hot = null, hotSince = 0, dwellFired = false, dwellLock = null;
  screen.addEventListener('pointermove', e => {
    if (G.scripted) return;
    const r = screen.getBoundingClientRect(); G.target.x = e.clientX-r.left; G.target.y = e.clientY-r.top;
    if (!G.seen){ G.seen = true; tip('Look (move the mouse) roughly at Chrome on the taskbar and raise your brows (B or a left click): the click snaps onto it. You can also just say "open Chrome".'); }
  });
  screen.addEventListener('pointerdown', e => {
    if (G.scripted) return; e.preventDefault();
    if (e.button===2) gesture('mouth'); else gesture('brow');
  });
  screen.addEventListener('contextmenu', e => e.preventDefault());

  function hitTest(){
    const sr = screen.getBoundingClientRect();
    const el = document.elementFromPoint(sr.left+G.pos.x, sr.top+G.pos.y);
    if (!el || !screen.contains(el)) return null;
    return el.closest('[data-act]');
  }
  function ripple(kind, pos=G.pos){ const d=document.createElement('div'); d.className='ripple'+(kind==='shake'?' r':''); d.style.left=pos.x+'px'; d.style.top=pos.y+'px'; screen.append(d); setTimeout(()=>d.remove(),520); }
  function glabel(t){ const g=$('#glabel'); g.textContent=t; g.classList.add('show'); clearTimeout(glabel.t); glabel.t=setTimeout(()=>g.classList.remove('show'),900); }
  function flashSig(id){ const s=$(id); s.classList.add('fire'); clearTimeout(s._t); s._t=setTimeout(()=>s.classList.remove('fire'),700); }

  const FACE = { blinkStart:-9999, blinkEnd:-9999, nodAt:-9999, shakeAt:-9999, tiltAt:-9999, tiltDir:1, browAt:-9999, mouthAt:-9999, speakUntil:0, caption:null };
  const DUR = { natural:150, brow:450, mouth:700, nod:650, shake:800, tiltL:700, tiltR:700 };
  const CAPS = { brow:'BROW RAISE +0.31 → CLICK', mouth:'MOUTH OPEN 0.5 s → RIGHT CLICK', nod:'NOD 12° → YES', shake:'SHAKE ±15° → NO', tiltL:'HEAD UP → SCROLL UP', tiltR:'HEAD DOWN → SCROLL DOWN' };
  const SIGS = { brow:'#sigBlink', mouth:'#sigShake', nod:'#sigNod', shake:'#sigNod', tiltL:'#sigTilt', tiltR:'#sigTilt' };
  // snap click: the button under the pointer, or the nearest clickable thing within ~150 px
  function snapTarget(){
    const here = hitTest(); if (here) return here;
    const sr = screen.getBoundingClientRect(), px = sr.left + G.pos.x, py = sr.top + G.pos.y;
    let best = null, bd = 150 * (sr.width/1920*1.4 + .3);
    $$('[data-act]', screen).forEach(el => { if (!el.offsetParent) return; const r = el.getBoundingClientRect(); if (!r.width) return;
      const dx = Math.max(r.left - px, 0, px - r.right), dy = Math.max(r.top - py, 0, py - r.bottom), d = Math.hypot(dx, dy);
      const top = document.elementFromPoint(r.left + r.width/2, r.top + r.height/2); if (!top || !el.contains(top)) return;
      if (d < bd){ bd = d; best = el; } });
    return best;
  }
  function gesture(kind){
    if ($('#demo').hidden) return;
    const now = performance.now();
    if (kind==='natural'){ FACE.blinkStart = now; FACE.blinkEnd = now + DUR.natural; }
    if (kind==='brow'){ FACE.browAt = now; mascotDo('blinknow'); }
    if (kind==='mouth') FACE.mouthAt = now;
    if (kind==='nod'){ FACE.nodAt = now; mascotDo('nodnow'); }
    if (kind==='shake'){ FACE.shakeAt = now; mascotDo('shakenow'); }
    if (kind==='tiltL' || kind==='tiltR'){ FACE.tiltAt = now; FACE.tiltDir = kind==='tiltL' ? -1 : 1; mascotDo('nodnow'); }
    FACE.active = {kind, t0:now};
    const off = (kind==='brow' && !G.brow) || (kind==='mouth' && !G.mouth) || (kind.startsWith('tilt') && !G.tilt);
    const pos = {x:G.pos.x, y:G.pos.y};
    setTimeout(() => {
      const t = performance.now();
      FACE.active = null;
      if (kind==='natural'){ addEvent(t,'natural','natural blink 0.15 s · ignored'); setCaption('Quick blink (0.15 s) · ignored', 'muted'); log('eye','Natural blink 0.15 s · ignored'); return; }
      addEvent(t, kind, CAPS[kind]); flashSig(SIGS[kind]);
      if (G.paused){ setCaption(gcap(kind)+' seen · paused','muted'); return; }
      if (off){ setCaption(gcap(kind)+' seen · turned off in Settings','muted'); toast(gcap(kind)+' is turned off in Settings'); return; }
      // a question is waiting: brow / nod = yes, shake = no
      if (G.pending && (kind==='brow' || kind==='nod' || kind==='shake')){ setCaption('✓ ' + CAPS[kind].replace(/→.*/, kind==='shake' ? '→ NO' : '→ YES'), kind==='shake'?'amber':'iris'); answer(kind!=='shake', kind==='shake'?'Head shake':'Brow raise'); return; }
      setCaption('✓ ' + CAPS[kind], (kind==='mouth'||kind==='shake')?'amber':'iris');
      if (kind==='brow'){
        const el = snapTarget();
        if (el){ const sr=screen.getBoundingClientRect(), r=el.getBoundingClientRect(), x=r.left-sr.left+r.width/2, y=r.top-sr.top+r.height/2, moved = !hitTest() ;
          if (moved){ G.target.x = G.pos.x = x; G.target.y = G.pos.y = y; }
          ripple('blink', {x, y}); glabel(moved ? 'BROW RAISE · SNAP CLICK' : 'BROW RAISE · CLICK');
          log('eye', `Brow raise → ${moved?'snapped to':'click'} "${label(el)}"`); run(el.dataset.act, 'eye'); }
        else { ripple('blink', pos); glabel('BROW RAISE · CLICK'); log('eye','Brow raise → click (nothing clickable within 150 px)'); $('#ctx').hidden = true; if (W.startOpen){ W.startOpen=false; renderAll(); } }
        return;
      }
      if (kind==='mouth'){ ripple('shake', pos); glabel('MOUTH OPEN · RIGHT CLICK'); log('eye','Mouth open 0.5 s → right click'); openCtx(pos.x+6, pos.y+6); return; }
      if (kind==='tiltL' || kind==='tiltR'){ const d = kind==='tiltL' ? -1 : 1; glabel(d<0 ? 'HEAD UP · SCROLL UP' : 'HEAD DOWN · SCROLL DOWN'); log('head', `Head ${d<0?'up':'down'} (pitch held) → scroll ${d<0?'up':'down'}`); scrollBy(d); return; }
      if (kind==='shake'){ glabel('SHAKE · NO'); log('head','Head shake → no / cancel'); if (!$('#ctx').hidden){ $('#ctx').hidden = true; log('sys','Menu closed'); } return; }
      if (kind==='nod'){ glabel('NOD · YES'); log('head','Nod → yes (only answers a question; not a click)'); toast('A nod means "yes" when diseyeble asks. It is not a click.'); return; }
    }, DUR[kind]);
  }

  /* ---------- confirm: the engine asks before risky actions, 12 s to answer ---------- */
  let cqTimer = null;
  function ask(prompt, onYes){
    if (G.pending) answer(false, 'replaced');
    const t0 = performance.now(); G.pending = { prompt, onYes, t0 };
    $('#cqTxt').textContent = prompt; $('#confirmQ').hidden = false; log('sys', `Asking: "${prompt}"`); tip(prompt + ' Raise your brows or say "yes". Shake or say "no".');
    clearInterval(cqTimer); cqTimer = setInterval(() => { const left = 12 - (performance.now()-t0)/1000; if (left <= 0) answer(false, 'expired'); else $('#cqT').textContent = Math.ceil(left)+' s'; }, 200);
    $('#cqT').textContent = '12 s';
  }
  function answer(yes, how){
    const q = G.pending; if (!q) return; G.pending = null; clearInterval(cqTimer); $('#confirmQ').hidden = true;
    if (how==='replaced') return;
    if (how==='expired'){ log('sys', `"${q.prompt}" expired after 12 s · counts as no`); toast('No answer in 12 s, so nothing was done'); return; }
    log('sys', `"${q.prompt}" → ${yes?'yes':'no'} (${how})`);
    if (yes) q.onYes(); else toast('Cancelled');
  }
  $('#cqYes').onclick = () => answer(true, 'button'); $('#cqNo').onclick = () => answer(false, 'button');
  const gcap = k => ({brow:'Brow raise', mouth:'Mouth open', nod:'Nod', shake:'Head shake', tiltL:'Head up / down', tiltR:'Head up / down'})[k] || k;

  function setCaption(text, tone){ FACE.caption = {text, tone, until: performance.now()+1600}; }
  const label = el => (el.getAttribute('aria-label') || el.querySelector('b')?.textContent || el.textContent || '').replace(/\s+/g,' ').trim().slice(0,32) || el.dataset.act;

  /* natural blinks on the simulated face */
  (function natural(){ setTimeout(()=>{ if (!$('#demo').hidden && !FACE.active && !G.scripted && !(window.DZ && window.DZ.engineLive)) gesture('natural'); natural(); }, 3500+Math.random()*3500); })();

  /* keyboard shortcuts */
  let vHeld = false;
  addEventListener('keydown', e => {
    if ($('#demo').hidden || e.target.tagName==='INPUT' || e.target.tagName==='TEXTAREA' || e.repeat) return;
    if (G.scripted) return;
    else if (e.key==='b'||e.key==='B') gesture('brow');
    else if (e.key==='m'||e.key==='M') gesture('mouth');
    else if (e.key==='n'||e.key==='N') gesture('nod');
    else if (e.key==='h'||e.key==='H') gesture('shake');
    else if (e.key==='ArrowUp'){ e.preventDefault(); gesture('tiltL'); }
    else if (e.key==='ArrowDown'){ e.preventDefault(); gesture('tiltR'); }
    else if (e.key==='p'||e.key==='P') setPaused(!G.paused);
    else if ((e.key==='v'||e.key==='V') && !vHeld){ vHeld=true; micStart(); }
  });
  addEventListener('keyup', e => { if ((e.key==='v'||e.key==='V') && vHeld){ vHeld=false; micStop(); } });
  [['#sBrow','brow'],['#sMouth','mouth'],['#sNod','nod'],['#sShake','shake'],['#sNat','natural'],['#sTiltL','tiltL'],['#sTiltR','tiltR'],
   ['#bBrow','brow'],['#bMouth','mouth'],['#bTiltL','tiltL'],['#bTiltR','tiltR'],['#bShake','shake']].forEach(([id,k]) => { const b=$(id); if (b) b.onclick = () => gesture(k); });

  function setPaused(p){ G.paused=p; $('#trackPill').classList.toggle('paused',p); $('#trackTxt').textContent=p?'Paused':'Tracking'; $('#pauseBtn').firstChild.textContent=p?'Resume ':'Pause '; gazeEl.classList.toggle('hide',p); log('sys',p?'Tracking paused':'Tracking resumed'); }
  $('#pauseBtn').onclick = () => setPaused(!G.paused);

  /* settings */
  function sw(id,key,cb){ const b=$(id); b.onclick=()=>{ G[key]=!G[key]; b.setAttribute('aria-checked',G[key]); cb&&cb(); }; }
  sw('#swBrow','brow'); sw('#swMouth','mouth'); sw('#swTilt','tilt');
  sw('#swDwell','dwell',()=>tip(G.dwell?'Dwell is on: rest the pointer on a button for '+G.dwellMs+' ms and it clicks.':'Dwell is off (the default). Raise your brows to click.'));
  $('#dwellMs').oninput = e => { G.dwellMs=+e.target.value; $('#dwellOut').textContent=G.dwellMs+' ms'; };
  $('#steady').oninput = e => { G.steady=+e.target.value/100; $('#steadyOut').textContent=e.target.value+'%'; };

  /* ---------------- voice ---------------- */
  const VOCAB = ['open','chrome','browser','google','youtube','search','for','find','look','up','play','watch','the','first','one','video','mail','email','inbox','write','new','an','a','to','send','it','scroll','down','go','back','close','window','click','right','pause','stop','type','subject','message','music','piano','weather','today','news','dinner','on','sunday','hi','mai','son','see','you','at','six','thank','is','like','this','and','with','me','relaxing','exercise','cooking','seniors','hello','love','grandma','tomorrow','morning'];
  function lev(a,b){ const m=a.length,n=b.length; if(!m) return n; if(!n) return m; const d=Array.from({length:m+1},(_,i)=>[i]); for(let j=1;j<=n;j++) d[0][j]=j; for(let i=1;i<=m;i++) for(let j=1;j<=n;j++) d[i][j]=Math.min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(a[i-1]===b[j-1]?0:1)); return d[m][n]; }
  const simp = w => w.toLowerCase().replace(/[^a-z]/g,'').replace(/(.)\1+/g,'$1').replace(/ph/g,'f').replace(/ch/g,'tu').replace(/ee/g,'i');
  function best(w){ let bw=null, bs=1; const sw=simp(w); if(!sw) return [w,1]; for(const v of VOCAB){ const s = lev(sw, simp(v)) / Math.max(sw.length, simp(v).length); if (s<bs){ bs=s; bw=v; } } return [bw,bs]; }
  function normalize(raw){
    const toks = raw.toLowerCase().replace(/[…"“”]/g,' ').split(/[\s]+/).filter(Boolean);
    const out=[]; let costs=[];
    for (let i=0;i<toks.length;i++){
      const one = best(toks[i]);
      if (i+1<toks.length){ const two = best(toks[i]+toks[i+1]); if (two[1] < one[1] - .08 && two[1] < .45){ out.push(two[0]); costs.push(two[1]); i++; continue; } }
      if (one[1] < .5){ out.push(one[0]); costs.push(one[1]); } else { out.push(toks[i].replace(/[^a-z@.'-]/g,'')); costs.push(.6); }
    }
    const conf = costs.length ? clamp(Math.round((1 - costs.reduce((a,b)=>a+b,0)/costs.length*0.9)*100), 40, 99) : 0;
    return [out.join(' ').replace(/\s+/g,' ').trim(), conf];
  }
  function parseIntent(t){
    let m;
    if (/^(open |go to )?(the )?(chrome|browser|google)$/.test(t)) return {cmd:'open:browser', say:'Open Chrome'};
    if (/(open|go to|show)?\s*youtube/.test(t) && !/search|find|play/.test(t)) return {cmd:'open:youtube', say:'Open YouTube'};
    if ((m = t.match(/^(?:search|find|look up)(?: for)? (.+?)(?: on (youtube|google))?$/))){
      const q=m[1].trim(), where = m[2] || (W.browser.open && ['youtube','yresults','watch'].includes(W.browser.page) ? 'youtube' : 'google');
      return {cmd:where==='youtube'?'do:ysearch:'+q:'do:gsearch:'+q, say:`Search ${where==='youtube'?'YouTube':'Google'} for “${q}”`};
    }
    if ((m = t.match(/^(?:play|watch) (?:the )?first(?: one| video)?$/))) return {cmd:'do:first', say:'Play the first video'};
    if ((m = t.match(/^(?:play|watch) (.+)$/))) return {cmd:'do:playq:'+m[1], say:`Play “${m[1]}”`};
    if (/^(open|check|show)( my)? (mail|email|inbox)$/.test(t) || t==='mail' || t==='email') return {cmd:'open:mail', say:'Open Mail'};
    if ((m = t.match(/^(?:write|new|compose|send)(?: an| a)? (?:email|mail)(?: to (\w+))?$/))) return {cmd:'do:compose:'+(m[1]||''), say:'Write an email'+(m[1]?' to '+cap(m[1]):'')};
    if (/^(send( it| message)?)$/.test(t)) return {cmd:'do:sendask', say:'Send it · asks first'};
    if (/^(yes|yeah|yep|confirm|do it|go ahead|click it)$/.test(t)) return {cmd:'do:yes', say:'Yes'};
    if (/^(no|nope|cancel|never mind)$/.test(t)) return {cmd:'do:no', say:'No'};
    if (/^(resume|wake up|start listening)$/.test(t)) return {cmd:'do:resume', say:'Resume'};
    if (/^(calibrate)$/.test(t)) return {cmd:'do:calib', say:'Calibrate'};
    if (/^(scroll|scroll mode|start scrolling)$/.test(t)) return {cmd:'do:scrollmode', say:'Scroll mode'};
    if ((m = t.match(/^(?:click|open|select) (?!this$|that$|it$|here$|there$)(.+)$/)) && !/youtube|chrome|browser|google|mail|email|inbox/.test(m[1])) return {cmd:'do:find:'+m[1], say:`Click “${m[1]}” · asks first`};
    if (/scroll down|down$/.test(t)) return {cmd:'do:scroll:1', say:'Scroll down'};
    if (/scroll up/.test(t)) return {cmd:'do:scroll:-1', say:'Scroll up'};
    if (/^(go )?back$/.test(t)) return {cmd:'back', say:'Go back'};
    if (/^close( the)?( window| it| this)?$/.test(t)) return {cmd:'do:close', say:'Close window'};
    if (/^right click$/.test(t)) return {cmd:'do:right', say:'Right click'};
    if (/^click( this| that| here| there)?$/.test(t)) return {cmd:'do:click', say:'Click (snap)'};
    if (/^(pause|stop listening)$/.test(t)) return {cmd:'do:pause', say:'Pause'};
    if ((m = t.match(/^(?:type|write) (.+)$/))) return {cmd:'do:type:'+m[1], say:`Type “${m[1]}”`};
    return null;
  }
  function execIntent(cmd){
    if (!cmd.startsWith('do:')) return run(cmd,'voice');
    const [, k, ...r] = cmd.split(':'); const v = r.join(':');
    if (k==='gsearch'){ openWin('browser'); W.f.gq=v; nav('gresults'); W.focus=null; log('voice','Searched Google for "'+v+'"'); renderAll(); }
    else if (k==='ysearch'){ openWin('browser'); W.f.yq=v; nav('yresults'); W.focus=null; log('voice','Searched YouTube for "'+v+'"'); renderAll(); }
    else if (k==='first'){ if (W.browser.page==='yresults') run('watchq:0','voice'); else run('watch:0','voice'); }
    else if (k==='playq'){ openWin('browser'); W.f.yq=v; nav('yresults'); run('watchq:0','voice'); }
    else if (k==='compose'){ openWin('mail'); W.mail.view='compose'; if (v){ W.f.to = v+'@example.com'; W.focus='subj'; } else W.focus='to'; renderAll(); }
    else if (k==='scroll'){ scrollBy(+v); }
    else if (k==='close'){ const t=W.top||(W.browser.open?'browser':W.mail.open?'mail':null); if (t) run('close:'+t,'voice'); }
    else if (k==='right'){ openCtx(G.pos.x+6,G.pos.y+6); }
    else if (k==='click'){ if (G.pending) return answer(true,'voice'); const el=snapTarget(); if (el){ log('voice','"click" → snapped to "'+label(el)+'"'); run(el.dataset.act,'voice'); } }
    else if (k==='yes'){ if (G.pending) answer(true,'voice'); else toast('Nothing is waiting for a yes'); }
    else if (k==='no'){ if (G.pending) answer(false,'voice'); else if (!$('#ctx').hidden) $('#ctx').hidden=true; }
    else if (k==='resume'){ setPaused(false); }
    else if (k==='calib'){ $('#calibBtn').click(); }
    else if (k==='scrollmode'){ toast('Scroll mode: move your head down or up to scroll. Say "stop" to end.'); log('voice','Scroll mode on'); scrollBy(1); }
    else if (k==='sendask'){ const m=W.mail; if (m.open && m.view==='compose') ask('Send the email?', () => run('send','voice')); else toast('Nothing to send yet'); }
    else if (k==='find'){ const want=v.toLowerCase(); const el=[...$$('[data-act]', screen)].filter(e=>e.offsetParent).find(e=>label(e).toLowerCase().includes(want));
      if (!el){ toast('Could not find "'+v+'" on the screen'); log('voice','UI-Venus: nothing called "'+v+'"'); return; }
      const sr=screen.getBoundingClientRect(), r=el.getBoundingClientRect(); G.target.x=r.left-sr.left+r.width/2; G.target.y=r.top-sr.top+r.height/2;
      log('voice','UI-Venus found "'+label(el)+'"'); ask('Click '+label(el)+'?', () => run(el.dataset.act,'voice')); }
    else if (k==='pause'){ setPaused(true); }
    else if (k==='type'){ if (W.focus){ insertText(v,true); renderAll(); } else toast('Click a text box first (raise your brows on it)'); }
  }
  function processVoice(heard, meantOverride, confOverride){
    flashSig('#sigVoice'); FACE.speakUntil = performance.now()+900;
    let [meant, conf] = meantOverride ? [meantOverride, confOverride||88] : normalize(heard);
    $('#heardTxt').textContent = '“' + heard + '”';
    const intent = parseIntent(meant);
    if (intent){
      $('#meantTxt').innerHTML = esc(intent.say) + `<span class="conf">${conf}%</span>`;
      log('voice', `“${heard}” → ${intent.say} (${conf}%)`);
      execIntent(intent.cmd);
      return true;
    }
    if (W.focus){
      const text = meantOverride || meant;
      $('#meantTxt').innerHTML = 'Dictation: ' + esc(text) + `<span class="conf">${conf}%</span>`;
      log('voice', `Dictated “${text}”`);
      insertText(text, true); renderAll(); return true;
    }
    $('#meantTxt').innerHTML = `Not sure. Did you mean “open YouTube”? <span class="conf">${conf}%</span>`;
    log('voice', `“${heard}” not recognised`); tip('I did not catch that. Try one of the example phrases under the mic.');
    return false;
  }
  $('#vform').addEventListener('submit', e => { e.preventDefault(); const v=$('#vinput').value.trim(); if (!v) return; processVoice(v); $('#vinput').value=''; renderChips(); });

  /* contextual example phrases (slurred) */
  function contextPhrases(){
    const b=W.browser, m=W.mail;
    if (W.focus==='body') return [['hi my, see yoo sun-day', 'hi mai, see you sunday at six']];
    if (W.focus==='subj') return [['din-nah on sun-day','dinner on sunday']];
    if (m.open && W.top==='mail' && m.view==='compose') return [['seh-n it','send it']];
    if (!b.open && !m.open) return [['oh-peh chrohm','open chrome'],['oh-peh yoo-choob','open youtube'],['ri ee-meh to my','write an email to mai']];
    if (b.page==='home') return [['oh-peh yoo-choob','open youtube'],['seh fo wea-tha to-day','search for weather today']];
    if (b.page==='youtube') return [['seh fo pee-ah-no moo-sic','search for piano music'],['sco dow','scroll down']];
    if (b.page==='yresults') return [['pway da fuhs one','play the first one'],['go bah','go back']];
    if (b.page==='watch') return [['go bah','go back'],['ri ee-meh to my','write an email to mai']];
    return [['go bah','go back'],['oh-peh mehl','open mail']];
  }
  function renderChips(){ $('#chips').innerHTML = contextPhrases().map(([h])=>`<button type="button" data-say="${esc(h)}">“${esc(h)}”</button>`).join(''); }
  $('#chips').addEventListener('click', e => { const b=e.target.closest('[data-say]'); if (!b) return; const p=contextPhrases().find(x=>x[0]===b.dataset.say); speakSim(p[0], p[1]); });

  /* mic: real speech recognition when available, otherwise simulated */
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let rec=null, micLive=false, micT0=0, srGot=false;
  const mic = $('#mic');
  function micStart(){
    if (micLive || G.scripted) return; micLive=true; micT0=performance.now(); srGot=false; mic.classList.add('live'); flashSig('#sigVoice');
    $('#heardTxt').textContent='Listening…'; $('#meantTxt').textContent='–';
    if (SR){ try{ rec=new SR(); rec.lang='en-US'; rec.interimResults=false; rec.onresult=ev=>{ srGot=true; processVoice(ev.results[0][0].transcript); renderChips(); }; rec.onerror=()=>{ rec=null; }; rec.start(); }catch(e){ rec=null; } }
  }
  function micStop(){
    if (!micLive) return; micLive=false; mic.classList.remove('live');
    if (rec){ try{ rec.stop(); }catch(e){} setTimeout(()=>{ if(!srGot) fallback(); }, 1200); rec=null; } else fallback();
    function fallback(){ const p=contextPhrases()[0]; speakSim(p[0], p[1], true); }
  }
  mic.addEventListener('pointerdown', e => { e.preventDefault(); micStart(); });
  addEventListener('pointerup', () => { if (micLive && !vHeld) micStop(); });
  async function speakSim(heard, meant, already){
    micLive = true; mic.classList.add('live'); micT0=performance.now(); $('#heardTxt').textContent='Listening…';
    FACE.speakUntil = performance.now()+1400;
    await wait(already?500:1300); micLive=false; mic.classList.remove('live');
    const [, c] = normalize(heard);
    processVoice(heard, meant, clamp(c+8,70,96)); renderChips();
  }

  /* ---------------- calibration ---------------- */
  $('#calibBtn').onclick = async () => {
    if (G.scripted) return;
    const cal=$('#calib'), pt=$('#calibPt'), msg=$('#calibMsg');
    cal.hidden=false; pt.hidden=true; msg.hidden=false;
    msg.innerHTML='<h3>One-look calibration</h3><p>Look at the dot in the middle of the screen for about 2 seconds.</p>';
    log('sys','Calibration started (1 dot)'); await wait(1400); msg.hidden=true; pt.hidden=false;
    pt.classList.remove('go'); pt.style.left='50%'; pt.style.top='50%'; await wait(300); void pt.offsetWidth; pt.classList.add('go'); await wait(1500);
    pt.hidden=true; msg.hidden=false; G.jitter=3; G.calibrated=true;
    msg.innerHTML='<h3>Calibrated</h3><p>The pointer now follows your eyes more closely.</p>';
    log('sys','Calibrated: one look at the centre'); tip('Calibrated! Try a guided demo above the screen, or do it yourself.');
    await wait(1300); cal.hidden=true;
  };

  /* ---------------- guided demos ---------------- */
  let demoToken = 0;
  class Cancel extends Error{}
  async function step(txt){ $('#stepTxt').textContent = txt; tip(txt); }
  async function gazeTo(sel, ms=650, tok){
    const el = typeof sel==='string' ? $(sel) : sel;
    if (!el) throw new Error('missing '+sel);
    // small screens: if the target sits below the window's visible area, scroll the page to it first
    const inner = el.closest('#bPage');
    if (inner){ const vp = inner.parentElement.getBoundingClientRect(), r0 = el.getBoundingClientRect();
      if (r0.bottom > vp.bottom - 6 || r0.top < vp.top){ const max = Math.max(0, inner.scrollHeight - inner.parentElement.clientHeight);
        W.browser.scroll = clamp(W.browser.scroll + (r0.top + r0.height/2) - (vp.top + vp.height/2), 0, max); inner.style.transform = `translateY(${-W.browser.scroll}px)`; await wait(120); } }
    const sr=screen.getBoundingClientRect(), r=el.getBoundingClientRect();
    const tx=r.left-sr.left+r.width/2, ty=r.top-sr.top+r.height/2;
    const sx=G.target.x, sy=G.target.y, t0=performance.now();
    while (true){
      if (tok!==demoToken) throw new Cancel();
      const p = Math.min(1,(performance.now()-t0)/ms), e = p<.5?2*p*p:1-Math.pow(-2*p+2,2)/2;
      G.target.x = sx+(tx-sx)*e; G.target.y = sy+(ty-sy)*e;
      if (p>=1) break; await wait(16);
    }
    await wait(260); if (tok!==demoToken) throw new Cancel();
  }
  async function g(kind, tok){
    if (tok!==demoToken) throw new Cancel();
    // let the pointer settle on its target first (the head pointer is smoothed)
    for (let i=0; i<60 && Math.hypot(G.pos.x-G.target.x, G.pos.y-G.target.y) > 4; i++){ await wait(16); if (tok!==demoToken) throw new Cancel(); }
    gesture(kind); await wait(DUR[kind]+380);
  }
  async function say(heard, meant, tok){
    if (tok!==demoToken) throw new Cancel();
    micLive=true; mic.classList.add('live'); micT0=performance.now(); FACE.speakUntil=performance.now()+1300; $('#heardTxt').textContent='Listening…';
    await wait(1300); micLive=false; mic.classList.remove('live');
    if (tok!==demoToken) throw new Cancel();
    const [, c] = normalize(heard); processVoice(heard, meant, clamp(c+8,72,95)); await wait(700);
  }
  async function eyeType(text, tok){
    for (const ch of text){
      const sel = ch===' ' ? '[data-act="key:space"]' : `[data-act="key:${ch}"]`;
      await gazeTo(sel, 300, tok); await g('brow', tok);
    }
  }
  function resetWorld(){
    Object.assign(W.browser,{open:false,min:false,page:'home',hist:[],fwd:[],scroll:0}); Object.assign(W.mail,{open:false,min:false,view:'inbox'});
    W.focus=null; W.startOpen=false; W.f={gq:'',yq:'',to:'',subj:'',body:''}; $('#ctx').hidden=true; renderAll();
  }
  const DEMOS = {
    async youtube(tok){
      await step('Looking at Chrome on the taskbar…'); await gazeTo('#tbBrowser', 800, tok);
      await step('Raise your brows to click it'); await g('brow', tok);
      await step('Looking at the YouTube shortcut…'); await gazeTo('[data-act="nav:youtube"]', 700, tok);
      await step('Raise your brows to click'); await g('brow', tok);
      await step('Saying the search out loud (slurred speech)'); await say('seh fo pee-ah-no moo-sic','search for piano music', tok);
      await step('Lower your head to scroll down'); await g('tiltR', tok);
      await step('Looking roughly at the first video…'); await gazeTo('[data-act="watchq:0"]', 700, tok);
      await step('Raise your brows: the click snaps onto it'); await g('brow', tok);
      await wait(1200);
      await step('Open your mouth for the right-click menu'); await g('mouth', tok);
      await gazeTo('[data-act="ctx:back"]', 500, tok);
      await step('Raise your brows to choose "Back"'); await g('brow', tok);
      await step('Done: YouTube opened, searched and played with eyes, brows and voice.');
    },
    async mail(tok){
      await step('Saying "write an email to Mai"'); await say('ri ee-meh to my','write an email to mai', tok);
      await step('Dictating the subject'); await say('din-nah on sun-day','dinner on sunday', tok);
      if ($('[data-act="key:hide"]')){ await step('Hiding the keyboard'); await gazeTo('[data-act="key:hide"]', 600, tok); await g('brow', tok); }
      await step('Looking at the message box…'); await gazeTo('[data-act="focus:body"]', 700, tok);
      await step('Raise your brows to select it'); await g('brow', tok);
      await step('Dictating the message'); await say('hi my, see yoo sun-day at sick','hi mai, see you sunday at six', tok);
      await step('Picking a word suggestion'); await gazeTo('[data-act="sug:Love, Grandma"]', 600, tok); await g('brow', tok);
      await step('Saying "send it": diseyeble asks first'); await say('seh-n it','send it', tok);
      await step('Raise your brows to say yes'); await wait(500); await g('brow', tok);
      await step('Done: email written and sent with voice, a brow raise and a confirm.');
    },
    async google(tok){
      await step('Looking at the Start button…'); await gazeTo('#tbStart', 700, tok); await g('brow', tok);
      await step('Choosing Chrome'); await gazeTo('#startMenu [data-act="open:browser"]', 600, tok); await g('brow', tok);
      await step('Looking at the search box…'); await gazeTo('[data-act="focus:gq"]', 700, tok); await g('brow', tok);
      await step('Typing "bus" on the on-screen keys, one brow raise per key'); await eyeType('bus', tok);
      await step('Picking "bus times" from suggestions'); await gazeTo('[data-act="sug:bus times"]', 500, tok); await g('brow', tok);
      await gazeTo('[data-act="key:enter"]', 500, tok);
      await step('Raise your brows on Enter'); await g('brow', tok);
      await step('Lower your head to scroll down'); await g('tiltR', tok);
      await step('Saying "scroll down"'); await say('sco dow','scroll down', tok);
      await step('Done: searched Google with the eyes, brow clicks and a head-down scroll.');
    }
  };
  async function startDemo(name){
    if (G.scripted) return;
    if (G.paused) setPaused(false);
    const tok = ++demoToken; G.scripted = true; const prevDwell = G.dwell; G.dwell = false; if (G.pending) answer(false,'replaced');
    $('#demoBar').hidden = true; $('#runBar').hidden = false;
    resetWorld(); log('sys','Guided demo: '+name);
    try { await DEMOS[name](tok); await wait(1800); }
    catch(e){ if (!(e instanceof Cancel)) console.error(e); }
    finally { if (tok===demoToken){ G.scripted=false; G.dwell=prevDwell; dwellLock={x:G.pos.x,y:G.pos.y}; $('#demoBar').hidden=false; $('#runBar').hidden=true; renderChips(); } }
  }
  $$('[data-demo]').forEach(b => b.onclick = () => startDemo(b.dataset.demo));
  $('#stopDemo').onclick = () => { demoToken++; G.scripted=false; G.dwell=$('#swDwell').getAttribute('aria-checked')==='true'; dwellLock={x:G.pos.x,y:G.pos.y}; $('#demoBar').hidden=false; $('#runBar').hidden=true; log('sys','Demo stopped'); tip('Demo stopped. Your turn: move the mouse (your eyes) and press B to raise your brows.'); renderChips(); };

  /* ---------------- camera canvas ---------------- */
  const cam=$('#camCanvas'), cx=cam.getContext('2d'), video=$('#camVideo');
  function drawCam(now){
    const Wd=cam.width, H=cam.height, sr=screen.getBoundingClientRect();
    const gx = clamp(G.pos.x/(sr.width||1),0,1)*2-1, gy = clamp(G.pos.y/(sr.height||1),0,1)*2-1;
    const t=now/1000;
    const np = (now-FACE.nodAt)/600, sp = (now-FACE.shakeAt)/750;
    const np2 = (now-FACE.nodAt)/DUR.nod, sp2 = (now-FACE.shakeAt)/DUR.shake;
    const nod = np2>=0&&np2<1 ? Math.sin(np2*Math.PI)*24 : 0;                 // head dips down then comes back up
    const shake = sp2>=0&&sp2<1 ? Math.sin(sp2*Math.PI*3)*22*(1-sp2*.35) : 0; // left-right-left
    const tp = (now-FACE.tiltAt)/DUR.tiltL;
    const roll = 0, hpitch = tp>=0&&tp<1 ? FACE.tiltDir*Math.min(1, Math.sin(tp*Math.PI)*1.5)*14 : 0;   // held head up (−) / down (+)
    const open = eyeOpen(now), blinking = open < 0.25, speaking = now < FACE.speakUntil;
    const bp = (now-FACE.browAt)/DUR.brow, browUp = bp>=0&&bp<1 ? Math.sin(bp*Math.PI) : 0;
    const mp = (now-FACE.mouthAt)/DUR.mouth, mouthOpen = mp>=0&&mp<1 ? Math.min(1, Math.sin(mp*Math.PI)*1.6) : 0;
    const sig = { brow: 0.03 + 0.32*browUp + (Math.random()-.5)*.012, mouth: 0.03 + (speaking ? 0.12*Math.abs(Math.sin(t*9)) : 0) + 0.46*mouthOpen + (Math.random()-.5)*.012, ear: 0.04 + 0.26*open, pitch: gy*-6 - nod*0.75 - hpitch + (Math.random()-.5)*.4, yaw: gx*9 + shake*0.78 + (Math.random()-.5)*.4, roll: gx*1.5 + roll + (Math.random()-.5)*.4 };
    SIGNAL = sig; sample(now, sig);
    let fx = Wd/2 + gx*10 + shake + roll*1.6, fy = H*.52 + gy*5 + nod + hpitch*1.3, tilt = gx*.03 + shake*.004 + roll*Math.PI/180;
    if (S.stream && video.readyState>=2){
      cx.save(); cx.translate(Wd,0); cx.scale(-1,1); cx.drawImage(video,0,0,Wd,H); cx.restore();
      overlay(Wd*.32,H*.16,Wd*.36,H*.62,gx,gy,blinking,null,nod,shake,now);
      return;
    }
    const bg=cx.createLinearGradient(0,0,0,H); bg.addColorStop(0,'#2B2B2B'); bg.addColorStop(1,'#161616'); cx.fillStyle=bg; cx.fillRect(0,0,Wd,H);
    cx.fillStyle='rgba(255,255,255,.04)'; cx.fillRect(Wd*.68,H*.08,Wd*.24,H*.4); cx.strokeStyle='rgba(255,255,255,.07)'; cx.lineWidth=2; cx.strokeRect(Wd*.68,H*.08,Wd*.24,H*.4);
    cx.fillStyle='rgba(255,255,255,.035)'; cx.fillRect(Wd*.06,H*.12,Wd*.14,H*.3);
    cx.save(); cx.translate(fx, H*.52 + gy*4);
    cx.fillStyle='#6E6E6E'; cx.beginPath(); cx.ellipse(shake*.3,150,150,70,0,Math.PI,0); cx.fill();
    cx.fillStyle='#E9BE98'; cx.fillRect(-26,62,52,48);
    cx.restore();
    cx.save(); cx.translate(fx, fy); cx.rotate(tilt); cx.scale(1, 1 - (Math.abs(nod) + Math.abs(hpitch))/220);
    // hair back (gray, older person)
    cx.fillStyle='#C9CCD3'; cx.beginPath(); cx.ellipse(0,-30,96,92,0,Math.PI,0); cx.fill();
    cx.fillStyle='#F0C7A2';
    cx.beginPath(); cx.ellipse(0,0,84,102,0,0,Math.PI*2); cx.fill();
    cx.beginPath(); cx.ellipse(-84,8,13,22,0,0,Math.PI*2); cx.ellipse(84,8,13,22,0,0,Math.PI*2); cx.fill();
    cx.fillStyle='#D9DCE2'; cx.beginPath(); cx.moveTo(-90,-40); cx.quadraticCurveTo(-60,-110,0,-104); cx.quadraticCurveTo(60,-110,90,-40); cx.quadraticCurveTo(40,-78,0,-70); cx.quadraticCurveTo(-40,-78,-90,-40); cx.fill();
    // wrinkles
    cx.strokeStyle='rgba(150,90,60,.25)'; cx.lineWidth=2; cx.beginPath(); cx.moveTo(-30,-48); cx.quadraticCurveTo(0,-54,30,-48); cx.moveTo(-24,-40); cx.quadraticCurveTo(0,-44,24,-40); cx.stroke();
    const eyes=[[-33,-4],[33,-4]];
    eyes.forEach(([ex,ey])=>{
      cx.fillStyle='#fff'; cx.beginPath(); cx.ellipse(ex,ey,19,Math.max(1.2,11*open),0,0,Math.PI*2); cx.fill();
      if (open<1){ cx.strokeStyle='#B98563'; cx.lineWidth=2.5; cx.beginPath(); cx.ellipse(ex,ey,19,Math.max(1.2,11*open),0,Math.PI,0); cx.stroke(); }
      if (open>0.35){ const ix=ex+gx*8, iy=ey+gy*4.5; cx.fillStyle='#5B4030'; cx.beginPath(); cx.arc(ix,iy,7.5,0,Math.PI*2); cx.fill(); cx.fillStyle='#111'; cx.beginPath(); cx.arc(ix,iy,3.6,0,Math.PI*2); cx.fill(); cx.fillStyle='#fff'; cx.beginPath(); cx.arc(ix-2.5,iy-2.5,1.6,0,Math.PI*2); cx.fill(); }
      const bu = browUp*9; cx.strokeStyle='#9AA0AC'; cx.lineWidth=4; cx.lineCap='round'; cx.beginPath(); cx.moveTo(ex-17,ey-22-bu); cx.quadraticCurveTo(ex,ey-28-bu*1.3,ex+17,ey-22-bu); cx.stroke();
    });
    // glasses
    cx.strokeStyle='rgba(40,46,60,.8)'; cx.lineWidth=3; cx.beginPath(); cx.roundRect?cx.roundRect(-58,-20,50,34,10):cx.rect(-58,-20,50,34); cx.stroke(); cx.beginPath(); cx.roundRect?cx.roundRect(8,-20,50,34,10):cx.rect(8,-20,50,34); cx.stroke(); cx.beginPath(); cx.moveTo(-8,-6); cx.lineTo(8,-6); cx.stroke();
    cx.strokeStyle='rgba(120,70,50,.55)'; cx.lineWidth=3; cx.beginPath(); cx.moveTo(gx*4,12); cx.quadraticCurveTo(-8+gx*4,34,6+gx*4,36); cx.stroke();
    const mo = Math.max(speaking ? 4 + Math.abs(Math.sin(t*18))*9 : 0, mouthOpen*20);
    cx.fillStyle='#8E3F3A'; cx.strokeStyle='#A0524A'; cx.lineWidth=4;
    if (mo>1){ cx.beginPath(); cx.ellipse(0,60,16,mo,0,0,Math.PI*2); cx.fill(); } else { cx.beginPath(); cx.moveTo(-20,58); cx.quadraticCurveTo(0,66,20,58); cx.stroke(); }
    cx.restore();
    overlay(fx-100,fy-122,200,240,gx,gy,blinking,{fx,fy,tilt,eyes},nod,shake,now);
    cx.fillStyle='rgba(17,17,17,.06)'; cx.fillRect(0,(t*80)%H,Wd,14);
  }
  function overlay(x,y,w,h,gx,gy,blinking,f,nod,shake,now){
    const Wd=cam.width;
    cx.strokeStyle='#111111'; cx.lineWidth=3; const L=22;
    [[x,y,1,1],[x+w,y,-1,1],[x,y+h,1,-1],[x+w,y+h,-1,-1]].forEach(([px,py,sx,sy])=>{ cx.beginPath(); cx.moveTo(px+sx*L,py); cx.lineTo(px,py); cx.lineTo(px,py+sy*L); cx.stroke(); });
    cx.font='600 12px "JetBrains Mono", monospace'; cx.fillStyle='#111111'; cx.fillText('face 0.98', x, y-8);
    if (f){
      cx.save(); cx.translate(f.fx,f.fy); cx.rotate(f.tilt); cx.fillStyle='rgba(17,17,17,.75)';
      for (let i=0;i<46;i++){ const a=i/46*Math.PI*2; cx.fillRect(Math.cos(a)*80-1,Math.sin(a)*98-1,2.5,2.5); }
      for (let i=0;i<14;i++){ const a=i/14*Math.PI*2; f.eyes.forEach(([ex,ey])=>cx.fillRect(ex+Math.cos(a)*22-1,ey+Math.sin(a)*Math.max(2,13*eyeOpen(now))-1,2.5,2.5)); }
      if (!blinking){ cx.strokeStyle='#F0A94B'; cx.lineWidth=2.5; f.eyes.forEach(([ex,ey])=>{ cx.beginPath(); cx.moveTo(ex+gx*8,ey+gy*4.5); cx.lineTo(ex+gx*60,ey+gy*40); cx.stroke(); cx.beginPath(); cx.arc(ex+gx*60,ey+gy*40,3,0,Math.PI*2); cx.fillStyle='#F0A94B'; cx.fill(); }); }
      cx.restore();
    }
    cx.fillStyle='rgba(0,0,0,.5)'; cx.fillRect(Wd-158,8,150,100);
    cx.fillStyle='#F1DCA7'; cx.font='11px "JetBrains Mono", monospace';
    const sg = SIGNAL;
    cx.fillStyle = Math.abs(sg.yaw)>=15 ? '#F0A94B' : '#F1DCA7'; cx.fillText('yaw   '+sg.yaw.toFixed(1).padStart(6)+'°', Wd-150, 26);
    cx.fillStyle = sg.pitch<=-12 ? '#111111' : '#F1DCA7'; cx.fillText('pitch '+sg.pitch.toFixed(1).padStart(6)+'°', Wd-150, 44);
    cx.fillStyle = '#F1DCA7'; cx.fillText('roll  '+sg.roll.toFixed(1).padStart(6)+'°', Wd-150, 62);
    cx.fillStyle = sg.brow>=.25 ? '#F0A94B' : '#F1DCA7'; cx.fillText('brow  '+(sg.brow>=0?'+':'')+sg.brow.toFixed(2), Wd-150, 80);
    cx.fillStyle = sg.mouth>=.35 ? '#F0A94B' : '#F1DCA7'; cx.fillText('mouth '+sg.mouth.toFixed(2).padStart(5), Wd-150, 98);
    // live caption for the gesture in progress / just detected
    let cap2 = null, tone = 'iris';
    const a = FACE.active;
    if (a && a.kind==='brow') cap2 = `Brows up +${sg.brow.toFixed(2)} (needs 0.25 for 0.25 s)`;
    else if (a && a.kind==='mouth') cap2 = `Mouth open ${((now-a.t0)/1000).toFixed(2)} s (0.3 – 1.5 s, then close)`;
    else if (a && a.kind==='natural') cap2 = 'Quick blink …';
    else if (a && a.kind==='nod') cap2 = `Nodding · pitch ${sg.pitch.toFixed(0)}° (yes)`;
    else if (a && a.kind.startsWith('tilt')) cap2 = `Head ${a.kind==='tiltL'?'up':'down'} · pitch ${sg.pitch.toFixed(0)}° (hold to scroll)`;
    else if (a && a.kind==='shake') cap2 = `Shaking · yaw ${sg.yaw.toFixed(0)}° (no)`;
    else if (FACE.caption && now < FACE.caption.until){ cap2 = FACE.caption.text; tone = FACE.caption.tone; }
    if (cap2){
      cx.font='700 15px "JetBrains Mono", monospace';
      const tw = cx.measureText(cap2).width + 24, bx = (Wd-tw)/2, by = cam.height-70;
      cx.fillStyle = tone==='amber' ? 'rgba(240,169,75,.95)' : tone==='muted' ? 'rgba(60,68,92,.92)' : (a ? 'rgba(15,21,38,.85)' : 'rgba(17,17,17,.95)');
      cx.beginPath(); cx.roundRect ? cx.roundRect(bx,by,tw,30,8) : cx.rect(bx,by,tw,30); cx.fill();
      cx.fillStyle = (!a && tone!=='muted') ? '#2A1A10' : '#fff'; cx.fillText(cap2, bx+12, by+20);
    }
  }

  /* eyelid profile: 1 = open, 0 = closed */
  function eyeOpen(now){
    const s0 = FACE.blinkStart, s1 = FACE.blinkEnd;
    if (now < s0 || now > s1) return 1;
    const d = s1 - s0, close = Math.min(90, d*.3), t = now - s0;
    if (t < close) return 1 - t/close;
    if (t > d - close) return (t - (d - close))/close;
    return 0;
  }
  let SIGNAL = {ear:.3,brow:.03,mouth:.03,pitch:0,yaw:0,roll:0};
  const HIST = [], EVENTS = [], WINDOW_MS = 6000;
  function sample(now, s){ HIST.push({t:now, ...s}); while (HIST.length && now-HIST[0].t > WINDOW_MS+200) HIST.shift(); }
  function addEvent(t, kind, text, label){ EVENTS.push({t, kind, text, label}); while (EVENTS.length && t-EVENTS[0].t > WINDOW_MS+200) EVENTS.shift(); }
  const plot = $('#plot'), pctx = plot.getContext('2d');
  let PC = null, pcT = 0;
  function plotColors(now){
    if (!PC || now-pcT > 1000){ const cs = getComputedStyle(document.documentElement), v = n => cs.getPropertyValue(n).trim();
      PC = {ink:v('--ink'), muted:v('--muted'), line:v('--line'), band:v('--surface-2'), iris:v('--iris'), amber:v('--amber'), rose:v('--rose'), violet:v('--violet')}; pcT = now; }
    return PC;
  }
  function drawPlot(now){
    const Wd = plot.width, H = plot.height, c = plotColors(now), L = 112, Rp = 10, top = 8, axisH = 26;
    pctx.clearRect(0,0,Wd,H);
    const lanes = [
      {k:'brow',  name:'brow',  unit:'',  min:0,   max:.4,  thr:[.25], tl:'click > 0.25',  col:c.iris},
      {k:'mouth', name:'mouth', unit:'',  min:0,   max:.55, thr:[.35], tl:'right click > 0.35', col:c.rose},
      {k:'pitch', name:'pitch', unit:'°', min:-24, max:24,  thr:[10,-10], tl:'held up / down = scroll',   col:c.violet},
      {k:'yaw',   name:'yaw',   unit:'°', min:-24, max:24,  thr:[15,-15], tl:'shake = no', col:c.muted},
    ];
    const lh = (H - top - axisH) / lanes.length;
    const X = t => L + (1 - (now - t)/WINDOW_MS) * (Wd - L - Rp);
    pctx.font = '600 19px "JetBrains Mono", monospace'; pctx.textBaseline = 'middle';
    lanes.forEach((ln, i) => {
      const y0 = top + i*lh, Y = v => y0 + 6 + (1 - (clamp(v,ln.min,ln.max)-ln.min)/(ln.max-ln.min)) * (lh-12);
      pctx.fillStyle = c.band; pctx.fillRect(L, y0+2, Wd-L-Rp, lh-4);
      pctx.fillStyle = c.ink; pctx.fillText(ln.name, 8, y0 + lh/2 - 11);
      const cur = HIST.length ? HIST[HIST.length-1][ln.k] : 0;
      pctx.fillStyle = ln.col; pctx.fillText((ln.unit===''?cur.toFixed(2):cur.toFixed(0)+ln.unit), 8, y0 + lh/2 + 13);
      pctx.setLineDash([6,6]); pctx.strokeStyle = c.muted; pctx.lineWidth = 1.5;
      ln.thr.forEach(tv => { pctx.beginPath(); pctx.moveTo(L, Y(tv)); pctx.lineTo(Wd-Rp, Y(tv)); pctx.stroke(); });
      pctx.setLineDash([]);
      pctx.font = '15px "JetBrains Mono", monospace'; pctx.fillStyle = c.muted; pctx.textAlign = 'right';
      pctx.fillText(ln.tl, Wd-Rp-6, Y(ln.thr[0]) + (ln.unit==='' ? -12 : -12)); pctx.textAlign = 'left';
      pctx.font = '600 19px "JetBrains Mono", monospace';
      pctx.strokeStyle = ln.col; pctx.lineWidth = 3; pctx.lineJoin = 'round'; pctx.beginPath();
      HIST.forEach((h, j) => { const x = X(h.t); if (x < L || h[ln.k]===undefined) return; j && X(HIST[j-1].t) >= L ? pctx.lineTo(x, Y(h[ln.k])) : pctx.moveTo(x, Y(h[ln.k])); });
      pctx.stroke();
    });
    // detection markers
    EVENTS.forEach(ev => {
      const x = X(ev.t); if (x < L) return;
      const col = ev.kind==='natural' || ev.kind==='shake' ? c.muted : ev.kind==='mouth' ? c.rose : ev.kind==='nod' ? c.amber : ev.kind.startsWith('tilt') ? c.violet : c.iris;
      pctx.strokeStyle = col; pctx.lineWidth = 2; pctx.setLineDash(ev.kind==='natural'?[4,5]:[]);
      pctx.beginPath(); pctx.moveTo(x, top); pctx.lineTo(x, H-axisH); pctx.stroke(); pctx.setLineDash([]);
      const lane = {brow:0, natural:0, mouth:1, nod:2, shake:3, tiltL:2, tiltR:2}[ev.kind];
      const label = ev.label || {natural:'blink ignored', brow:'CLICK', mouth:'R-CLICK', nod:'YES', shake:'NO', tiltL:'SCROLL ↑', tiltR:'SCROLL ↓'}[ev.kind];
      pctx.font = '700 15px "JetBrains Mono", monospace';
      const tw = pctx.measureText(label).width + 12, lx = Math.min(x+4, Wd-Rp-tw), ly = top + lane*lh + 6;
      pctx.fillStyle = col; pctx.fillRect(lx, ly, tw, 22);
      pctx.fillStyle = ev.kind==='natural' ? '#fff' : '#fff'; pctx.fillText(label, lx+6, ly+11);
    });
    pctx.font = '15px "JetBrains Mono", monospace'; pctx.fillStyle = c.muted;
    pctx.fillText('−6 s', L, H-axisH/2); pctx.textAlign='center'; pctx.fillText('−3 s', L+(Wd-L-Rp)/2, H-axisH/2); pctx.textAlign='right'; pctx.fillText('now', Wd-Rp, H-axisH/2); pctx.textAlign='left';
  }

  /* waveform */
  const wave=$('#wave'), wx=wave.getContext('2d');
  function drawWave(now){
    const Wd=wave.width, H=wave.height; wx.clearRect(0,0,Wd,H);
    const col = getComputedStyle(document.documentElement).getPropertyValue('--violet').trim() || '#6A4BD6';
    const n=46, bw=Wd/n;
    for (let i=0;i<n;i++){
      const amp = micLive ? (0.25 + 0.75*Math.abs(Math.sin(now/120 + i*0.7) * Math.sin(now/310 + i*0.23))) : 0.06;
      const h = Math.max(4, amp*H*0.8);
      wx.fillStyle = col; wx.globalAlpha = micLive ? 0.9 : 0.35;
      wx.fillRect(i*bw+2, (H-h)/2, bw-4, h);
    }
    wx.globalAlpha=1;
  }

  /* ---------------- main loop ---------------- */
  let last = performance.now(), secT = 0;
  const C = 157.1;
  function loop(now){
    const dt = Math.min(50, now-last); last = now;
    if (!$('#demo').hidden){
      if (!G.paused){
        const jit = G.scripted ? 3 : G.jitter;
        const tx = G.target.x + (Math.random()-.5)*jit, ty = G.target.y + (Math.random()-.5)*jit;
        const steady = G.scripted ? .55 : G.steady;
        const k = clamp((1-Math.pow(steady, dt/16))*.9+.04, .04, 1);
        G.pos.x += (tx-G.pos.x)*k; G.pos.y += (ty-G.pos.y)*k;
        gazeEl.style.transform = `translate(${G.pos.x}px,${G.pos.y}px)`;
        const el = hitTest();
        if (el!==hot){ hot && hot.classList.remove('hot'); hot=el; hotSince=now; dwellFired=false; el && el.classList.add('hot'); }
        let p=0;
        if (dwellLock && Math.hypot(G.pos.x-dwellLock.x, G.pos.y-dwellLock.y) > 40) dwellLock = null;
        const locked = !!dwellLock;
        if (hot && G.dwell && !G.scripted && !locked){ p=clamp((now-hotSince)/G.dwellMs,0,1); if (p>=1 && !dwellFired){ dwellFired=true; dwellLock={act:hot.dataset.act,x:G.pos.x,y:G.pos.y}; ripple('blink'); glabel('DWELL · CLICK'); log('eye','Dwell 1.0 s → click "'+label(hot)+'"'); run(hot.dataset.act,'eye'); } }
        prog.style.strokeDashoffset = C*(1-p);
        if (now-secT>120){ secT=now; $('#gazeXY').textContent = Math.round(G.pos.x/ (screen.clientWidth||1)*1920)+', '+Math.round(G.pos.y/(screen.clientHeight||1)*1080); }
      }
      drawCam(now); drawWave(now); drawPlot(now);
      const b=W.browser;
      if (b.open && b.page==='watch' && b.playing){ const tr=$('#ptrack'); if (tr){ const el=(now-(b.t0||now))/1000; tr.style.width = Math.min(100, el/2.4)+'%'; const s=Math.floor(el); $('#ptime').firstChild && ($('#ptime').textContent = `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')} / ${b.watchData[2]}`); } }
      if (Math.floor(now/1000)!==Math.floor((now-dt)/1000)){ $('#clock').textContent = stamp(); const d=new Date(); $('#trayClock').innerHTML = d.toLocaleTimeString([], {hour:'numeric',minute:'2-digit'})+'<br>'+d.toLocaleDateString('en-GB'); }
    }
    requestAnimationFrame(loop);
  }

  /* ---------------- hooks for js/engine.js (the real engine over WebSocket) ---------------- */
  window.DZ = { $, esc, log, toast, tip, flashSig, glabel, mascotDo, setCaption, addEvent, FACE, DUR, G, screen, mic,
    W, run, openWin, nav, renderAll, insertText, typeKey, scrollBy, ctxAction, openCtx, ripple, hitTest, wtoast, ACTIONS, bringTop,
    get micLive(){ return micLive; }, set micLive(v){ micLive = v; mic.classList.toggle('live', v); } };

  /* ---------------- boot ---------------- */
  renderAll(); renderChips();
  go(location.hash==='#demo' || location.hash==='#live' ? 'demo' : 'intro');
  requestAnimationFrame(loop); lookAll();
})();

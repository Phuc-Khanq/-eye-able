/* js/engine.js: connects this page to the real Diseyeble engine (python -m engine.main) over WebSocket.
   Messages are the ones in engine/core/messages.py. With no engine running the page keeps its simulation.
   Needs window.DZ, which js/app.js exposes just before its boot block.
   Open the page with ?engine=ws://host:port to use another address (default ws://127.0.0.1:8765). */
(() => {
  const D = window.DZ; if (!D) return;
  const { $, log, toast, tip, flashSig, addEvent, mascotDo, FACE, G } = D;
  const URL_WS = new URLSearchParams(location.search).get('engine') || 'ws://127.0.0.1:8765';
  const RETRY_MS = 2000, SHAKY_CONF = 0.6;

  /* ---------- small UI pieces: status pill, readout, question panel ---------- */
  const st = document.createElement('style');
  st.textContent = `
    #engPill.off .dot{background:var(--muted)} #engRead{font:600 12px var(--f-mono);color:var(--muted);margin-left:8px}
    #engQ{display:flex;align-items:center;gap:12px;margin:0 0 10px;padding:10px 14px;border-radius:14px;background:var(--surface-2);border:1px solid var(--line)}
    #engQ[hidden]{display:none} #engQ b{flex:1;font-size:15px} #engQ small{font:600 12px var(--f-mono);color:var(--muted)}
    #engQ button,.eng-act button{padding:6px 14px;border-radius:999px;border:1px solid var(--line);background:var(--iris);color:var(--iris-ink);font:600 13px var(--f-body);cursor:pointer}
    #engQ button.no,.eng-act button.no{background:transparent;color:var(--ink)}
    .eng-act{display:inline-flex;gap:8px;margin-left:10px} .eng-w.shaky{color:#fbbf24;text-decoration:underline}
    .eng-live{opacity:.6}`;
  document.head.append(st);

  const bar = $('.statusbar');
  const pill = document.createElement('span');
  pill.className = 'pill off'; pill.id = 'engPill';
  pill.innerHTML = '<span class="dot"></span><span id="engTxt">Simulated</span>';
  const read = document.createElement('span'); read.id = 'engRead';
  bar.insertBefore(pill, bar.firstChild); bar.insertBefore(read, pill.nextSibling);
  const q = document.createElement('div'); q.id = 'engQ'; q.hidden = true;
  q.innerHTML = '<b></b><small></small><button class="yes">Yes</button><button class="no">No</button>';
  $('#demoBar').before(q);

  let starting = false;
  let ws = null, live = false, saved = null, qTimer = 0, qHide = 0, qId = null;
  const send = (type, fields) => { if (live && ws && ws.readyState === 1) ws.send(JSON.stringify({ type, ...fields })); };
  const setRead = t => { read.textContent = t || ''; };

  /* ---------- live / simulated switch ---------- */
  function setLive(on) {
    if (on === live) return;
    live = on; D.engineLive = on;   // app.js stops the fake natural blinks while live
    if (on) starting = false;
    $('#allowCamTxt').textContent = on ? 'Engine camera on ✓' : 'Allow camera'; $('#demoCamBtn').textContent = on ? 'Camera on' : 'Allow camera';
    pill.classList.toggle('off', !on);
    $('#engTxt').textContent = on ? 'Engine live' : (starting ? 'Starting engine…' : 'Simulated');
    if (on) {
      saved = { dwell: G.dwell, paused: G.paused };
      G.dwell = false; $('#swDwell').setAttribute('aria-checked', 'false');   // the engine decides about dwell clicks
      G.seen = true;
      $('#camSrc').textContent = 'source: simulated face · engine live';
      log('sys', 'Engine connected at ' + URL_WS); toast('Engine connected');
    } else {
      if (saved) { G.dwell = saved.dwell; $('#swDwell').setAttribute('aria-checked', String(saved.dwell)); setPausedUI(saved.paused); }
      $('#camSrc').textContent = 'source: simulated face';
      q.hidden = true; setRead(''); D.micLive = false;
      log('sys', 'Engine disconnected, back to the simulation');
    }
  }

  function connect() {
    try { ws = new WebSocket(URL_WS); } catch (e) { return setTimeout(connect, RETRY_MS); }
    ws.onopen = () => setLive(true);
    ws.onmessage = ev => { let m; try { m = JSON.parse(ev.data); } catch (e) { return; } if (m && m.type) handle(m); };
    ws.onclose = () => { setLive(false); setTimeout(connect, RETRY_MS); };
    ws.onerror = () => { try { ws.close(); } catch (e) {} };
  }

  /* ---------- engine -> page ---------- */
  const MODES = { pointer: 'Tracking', dictation: 'Dictation', scroll: 'Scroll', menu: 'Menu', paused: 'Paused' };
  function setPausedUI(p) {
    G.paused = p;
    $('#trackPill').classList.toggle('paused', p);
    $('#pauseBtn').firstChild.textContent = p ? 'Resume ' : 'Pause ';
    $('#gaze').classList.toggle('hide', p);
  }

  const GEST = {   // engine gesture -> [plot kind, signal tile, log source, plot label, log text]
    nod:        ['nod',   '#sigNod',   'head', 'NOD',    'Nod'],
    shake:      ['shake', '#sigShake', 'head', 'SHAKE',  'Head shake'],
    tilt_left:  ['tiltL', '#sigTilt',  'head', 'TILT L', 'Head tilt left'],
    tilt_right: ['tiltR', '#sigTilt',  'head', 'TILT R', 'Head tilt right'],
    brow_raise: ['blink', '#sigBlink', 'eye',  'RAISE',  'Eyebrow raise'],
    blink:      ['blink', '#sigBlink', 'eye',  'BLINK',  'Long blink'],
    mouth_open: ['blink', '#sigVoice', 'head', 'MOUTH',  'Mouth open'],
  };
  function gesture(name) {
    const g = GEST[name]; if (!g) return;
    const now = performance.now();
    if (name === 'nod') { FACE.nodAt = now; mascotDo('nodnow'); }
    else if (name === 'shake') { FACE.shakeAt = now; mascotDo('shakenow'); }
    else if (name === 'tilt_left' || name === 'tilt_right') { FACE.tiltAt = now; FACE.tiltDir = name === 'tilt_left' ? -1 : 1; mascotDo(name === 'tilt_left' ? 'tiltl' : 'tiltr'); }
    else if (name !== 'mouth_open') { FACE.blinkStart = now; FACE.blinkEnd = now + 520; mascotDo('blinknow'); }
    addEvent(now, g[0], g[4], g[3]);
    flashSig(g[1]);
    D.setCaption('✓ ' + g[4], 'iris');
    D.glabel(g[3]);
    log(g[2], g[4] + ' seen by the engine');
  }

  function onPreview(m) {
    const meant = $('#meantTxt'), heard = $('#heardTxt');
    meant.textContent = '';
    if (!m.words || !m.words.length) { meant.textContent = '–'; return; }
    heard.textContent = '“' + (m.original || m.words.map(w => w.text).join(' ')) + '”';
    meant.append((m.cleaned ? 'Will type (cleaned): ' : 'Will type: '));
    m.words.forEach((w, i) => {
      const s = document.createElement('span');
      s.className = 'eng-w' + ((w.conf !== undefined && w.conf < SHAKY_CONF) ? ' shaky' : '');
      s.textContent = (i ? ' ' : '') + w.text; meant.append(s);
    });
    const act = document.createElement('span'); act.className = 'eng-act';
    act.innerHTML = '<button class="ok">Type it</button><button class="no">Drop it</button>';
    act.querySelector('.ok').onclick = () => send('dictation', { action: 'insert' });
    act.querySelector('.no').onclick = () => send('dictation', { action: 'discard' });
    meant.append(act);
  }

  function onConfirm(m) {
    clearInterval(qTimer); clearTimeout(qHide);
    q.hidden = false; qId = m.id;
    q.querySelector('b').textContent = m.prompt;
    const small = q.querySelector('small');
    if (m.state === 'open') {
      q.classList.remove('eng-live'); q.querySelectorAll('button').forEach(b => b.hidden = false);
      let left = Math.round(m.expires_in_s || 0); small.textContent = left + ' s';
      qTimer = setInterval(() => { left = Math.max(0, left - 1); small.textContent = left + ' s'; if (!left) clearInterval(qTimer); }, 1000);
      tip('The engine asks: ' + m.prompt + ' Raise your eyebrows or say yes, or press Yes below.');
    } else {
      q.classList.add('eng-live'); q.querySelectorAll('button').forEach(b => b.hidden = true);
      small.textContent = { confirmed: 'done', cancelled: 'cancelled', expired: 'timed out' }[m.state] || m.state;
      log('sys', m.prompt + ' → ' + small.textContent);
      qHide = setTimeout(() => { q.hidden = true; }, 1800);
    }
  }
  q.querySelector('.yes').onclick = () => send('confirm_answer', { id: qId, yes: true });
  q.querySelector('.no').onclick = () => send('confirm_answer', { id: qId, yes: false });

  function handle(m) {
    switch (m.type) {
      case 'mode': {
        $('#trackTxt').textContent = MODES[m.mode] || m.mode;
        setPausedUI(m.mode === 'paused');
        log('sys', 'Mode: ' + (MODES[m.mode] || m.mode));
        break;
      }
      case 'gaze': {   // real screen pixels -> the page's fake screen
        if (G.paused) break;
        const sw = Math.round(screen.width * (devicePixelRatio || 1)) || 1920, sh = Math.round(screen.height * (devicePixelRatio || 1)) || 1080;
        const box = D.screen.getBoundingClientRect();
        G.target.x = Math.min(1, Math.max(0, m.x / sw)) * box.width;
        G.target.y = Math.min(1, Math.max(0, m.y / sh)) * box.height;
        break;
      }
      case 'gesture': gesture(m.gesture); break;
      case 'dwell': setRead(m.progress > 0 ? 'dwell ' + Math.round(m.progress * 100) + '%' : (m.ready ? 'dwell ready' : '')); break;
      case 'scroll': setRead(m.ready && m.rate ? 'scroll ' + (m.rate > 0 ? '↑' : '↓') + Math.abs(m.rate).toFixed(1) + '/s' : ''); break;
      case 'listening': D.micLive = !!m.speaking; if (m.speaking) FACE.speakUntil = performance.now() + 400; break;
      case 'dictation_live': $('#heardTxt').textContent = m.text ? '… ' + m.text : 'Listening…'; break;
      case 'dictation_preview': onPreview(m); break;
      case 'command': $('#meantTxt').textContent = 'Command: ' + m.label; flashSig('#sigVoice'); log('voice', 'Command: ' + m.label); break;
      case 'confirm': onConfirm(m); break;
      case 'target_searching': log('voice', m.snap ? 'UI-Venus: finding the button under the cursor…' : 'UI-Venus: looking for “' + m.label + '”…'); tip(m.snap ? 'Finding what to click…' : 'Looking for “' + m.label + '”…'); break;
      case 'target_found': log('voice', 'UI-Venus: found “' + m.label + '” at ' + m.x + ', ' + m.y); break;
      case 'target_none': log('voice', 'UI-Venus: could not find “' + m.label + '”'); toast('Could not find “' + m.label + '”'); break;
      case 'status': log('sys', (m.level === 'info' ? '' : m.level.toUpperCase() + ': ') + m.text); toast(m.text); tip(m.text); break;
      case 'calib_target': {
        const c = $('#calib'); c.hidden = false;
        $('#calibMsg').innerHTML = '<h3>Calibration</h3><p>Look at the dot on your real screen and keep still.</p>';
        $('#calibPt').hidden = true; log('sys', 'Calibrating: look at the dot at ' + m.x + ', ' + m.y);
        break;
      }
      case 'calib_result': {
        $('#calib').hidden = true;
        if (m.ok) { $('#accTxt').textContent = Math.round(m.error_px) + ' px'; log('sys', 'Calibrated, gaze shifted ' + Math.round(m.error_px) + ' px'); }
        else log('sys', 'Calibration cancelled');
        break;
      }
    }
  }

  /* ---------- page -> engine (only while live; otherwise app.js keeps simulating) ---------- */
  document.addEventListener('keydown', e => {   // capture: runs before app.js's own P shortcut
    if (!live || $('#demo').hidden || e.repeat || e.target.tagName === 'INPUT') return;
    if (e.key === 'p' || e.key === 'P') { e.stopPropagation(); send('set_mode', { mode: G.paused ? 'pointer' : 'paused' }); }
  }, true);
  document.addEventListener('click', e => {
    if (!live) return;
    if (e.target.closest('#pauseBtn')) { e.stopImmediatePropagation(); e.preventDefault(); send('set_mode', { mode: G.paused ? 'pointer' : 'paused' }); }
    else if (e.target.closest('#calibBtn')) { e.stopImmediatePropagation(); e.preventDefault(); send('calibrate', { action: 'start' }); }
  }, true);
  $('#dwellMs').addEventListener('input', e => send('settings', { dwell_ms: +e.target.value }));

  /* ---------- Allow camera = start the engine (needs serve_demo.py; the engine owns the webcam) ---------- */
  async function startEngine() {
    if (live) return toast('The engine is already running and using the webcam.');
    if (!location.protocol.startsWith('http')) return toast('To start the engine from here, run serve_demo.py and open http://localhost:8000');
    if (starting) return;
    starting = true; $('#engTxt').textContent = 'Starting engine…';
    log('sys', 'Starting the engine, a new window opens. The first start takes a minute.'); toast('Starting the engine…');
    try {
      const r = await fetch('/api/start-engine', { method: 'POST', headers: { 'X-Diseyeble': '1' } });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || 'refused');
    } catch (err) {
      starting = false; $('#engTxt').textContent = 'Simulated';
      return toast('Could not start the engine: ' + err.message + '. Is serve_demo.py running?');
    }
    setTimeout(() => { if (starting && !live) { starting = false; $('#engTxt').textContent = 'Simulated'; toast('The engine did not come up. Check its window for errors.'); } }, 120000);
  }
  // capture phase: the page's own camera code never runs, because one app at a time can hold a webcam on Windows
  document.addEventListener('click', e => {
    if (!e.target.closest('#allowCam, #demoCamBtn')) return;
    e.stopImmediatePropagation(); e.preventDefault(); startEngine();
  }, true);

  connect();
})();

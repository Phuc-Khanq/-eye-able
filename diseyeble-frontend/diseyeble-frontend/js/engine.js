/* js/engine.js: connects this page to the real Diseyeble engine (python -m engine.main) over WebSocket.
   Messages are the ones in engine/core/messages.py. With no engine running the page keeps its simulation.
   Needs window.DZ, which js/app.js exposes just before its boot block.
   Open the page with ?engine=ws://host:port to use another address (default ws://127.0.0.1:8765).

   While the engine is connected and this demo screen is showing, the page is in "page mode":
   - the pointer is the real mouse (the engine moves it with your head); a real click is a click on the simulated desktop
   - typing, keys, scrolling and "open ..." arrive as page_action messages and happen on the simulated desktop
   - the webcam picture arrives as debug_frame and shows in the camera box
   - the Diseyeble card (mode, what you are saying, what will be typed, questions) is drawn here on the simulated screen
   Every 2 s this page tells the engine "page_mode on"; if it stops, the engine goes back to the real computer. */
(() => {
  const D = window.DZ; if (!D) return;
  const { $, esc, log, toast, tip, flashSig, addEvent, mascotDo, FACE, G, W } = D;
  const URL_WS = new URLSearchParams(location.search).get('engine') || 'ws://127.0.0.1:8765';
  const RETRY_MS = 2000, SHAKY_CONF = 0.6, HEARTBEAT_MS = 1000, FRAME_GAP_MS = 3000;

  /* ---------- status pill + readout in the status bar ---------- */
  const st = document.createElement('style');
  st.textContent = `#engPill.off .dot{background:var(--muted)} #engRead{font:600 12px var(--f-mono);color:var(--muted);margin-left:8px}`;
  document.head.append(st);
  const bar = $('.statusbar');
  const pill = document.createElement('span');
  pill.className = 'pill off'; pill.id = 'engPill';
  pill.innerHTML = '<span class="dot"></span><span id="engTxt">Simulated</span>';
  const read = document.createElement('span'); read.id = 'engRead';
  bar.insertBefore(pill, bar.firstChild); bar.insertBefore(read, pill.nextSibling);
  const setRead = t => { read.textContent = t || ''; };

  /* ---------- the Diseyeble card, on the simulated screen ---------- */
  const MODES = { pointer: ['Diseyeble', ''], dictation: ['Dictation', 'say “dictate off” to stop'], scroll: ['Scroll', 'tilt your head · say “stop”'],
                  menu: ['Menu', ''], paused: ['Paused', 'say “resume”'] };
  const screenEl = D.screen;
  const card = document.createElement('div');
  card.className = 'dz-card'; card.hidden = true;
  card.innerHTML = `<div class="dz-pill"><i class="dz-dot"></i><span class="dz-pt">Diseyeble</span><small class="dz-ph"></small>
      <span class="dz-wave">${'<i></i>'.repeat(7)}</span></div>
    <div class="dz-panel" hidden><div class="dz-label"><span class="dz-lab"></span></div><span class="dz-chip" hidden></span>
      <div class="dz-body"></div><div class="dz-bar" hidden><i></i></div><div class="dz-actions"></div><div class="dz-hint"></div></div>`;
  screenEl.append(card);
  const badge = document.createElement('div');
  badge.className = 'sim-badge'; badge.textContent = 'SIMULATION'; badge.hidden = true;
  screenEl.append(badge);
  const $c = sel => card.querySelector(sel);

  const S = {                 // what the card shows
    mode: 'pointer', speaking: false, level: 0,
    flash: null, flashUntil: 0,               // a command's name in the pill
    liveText: '',                             // what Whisper hears so far
    preview: null,                            // {words:[{text,conf}], cleaned, original}
    q: null,                                  // {id, prompt, state, left, total, t0}
    done: null, doneUntil: 0,                 // "Typed" / "Dropped" / "Done" for a second
  };
  let lastTypeAt = 0, qTick = 0, waveT = 0;

  function say(text, ms = 1000) { S.done = text; S.doneUntil = performance.now() + ms; render(); setTimeout(render, ms + 30); }

  function render() {
    if (!live) { card.hidden = true; badge.hidden = true; return; }
    card.hidden = false; badge.hidden = false;
    const now = performance.now();
    const [name, hint] = MODES[S.mode] || [S.mode, ''];
    const flash = S.flash && now < S.flashUntil ? S.flash : null;
    card.className = 'dz-card ' + S.mode + (S.speaking ? ' hearing' : '');
    $c('.dz-pt').textContent = flash || name;
    $c('.dz-ph').textContent = flash ? '' : hint;
    const panel = $c('.dz-panel'), lab = $c('.dz-lab'), chip = $c('.dz-chip'), body = $c('.dz-body'),
          barEl = $c('.dz-bar'), acts = $c('.dz-actions'), hintEl = $c('.dz-hint');
    chip.hidden = true; barEl.hidden = true; acts.innerHTML = ''; hintEl.innerHTML = ''; body.innerHTML = '';
    const q = S.q && S.q.state === 'open' ? S.q : null;
    if (q) {                                                           // 1. a yes/no question
      panel.hidden = false; lab.textContent = 'Diseyeble asks';
      body.textContent = q.prompt;
      barEl.hidden = false; barEl.firstElementChild.style.width = Math.max(0, Math.min(100, 100 * q.left / q.total)) + '%';
      acts.innerHTML = '<button class="main" data-act="dz:yes">Yes</button><button data-act="dz:no">No</button>';
      hintEl.innerHTML = 'Blink or say <b>yes</b> · say <b>no</b> to cancel';
    } else if (S.preview && S.preview.words.length) {                  // 2. what will be typed
      panel.hidden = false; lab.textContent = S.preview.cleaned ? 'Will type (cleaned)' : 'Will type';
      if (S.preview.cleaned && S.preview.original) { chip.hidden = false; chip.textContent = S.preview.original; }
      body.innerHTML = S.preview.words.map(w => `<span class="${w.conf !== undefined && w.conf < SHAKY_CONF ? 'shaky' : ''}">${esc(w.text)}</span>`).join(' ')
        + (S.liveText ? ` <span class="live">${esc(S.liveText)}</span>` : '');
      acts.innerHTML = '<button class="main" data-act="dz:insert">Type it</button><button data-act="dz:discard">Drop it</button>'
        + (S.preview.cleaned ? '<button data-act="dz:original">Use original</button>' : '');
      hintEl.innerHTML = 'Blink or say <b>yes</b> · say <b>no</b> to drop it';
    } else if (S.liveText) {                                           // 3. hearing
      panel.hidden = false; lab.textContent = 'Hearing';
      body.innerHTML = `<span class="live">${esc(S.liveText)}</span>`;
    } else if (S.done && now < S.doneUntil) {                          // 4. a second of confirmation
      panel.hidden = false; lab.textContent = ''; body.textContent = S.done;
    } else {
      panel.hidden = true;
    }
    clearInterval(qTick);
    if (q) qTick = setInterval(() => {
      q.left = Math.max(0, q.total - (performance.now() - q.t0) / 1000);
      barEl.firstElementChild.style.width = (100 * q.left / q.total) + '%';
      if (!q.left) clearInterval(qTick);
    }, 150);
  }

  // the waveform in the pill moves while the mic hears you
  function wave() {
    const bars = card.querySelectorAll('.dz-wave i');
    bars.forEach((b, i) => { b.style.height = (S.speaking ? 4 + 12 * Math.abs(Math.sin(waveT + i * 0.9)) * (0.4 + S.level) : 4) + 'px'; });
    waveT += 0.45;
  }
  setInterval(() => { if (live && S.speaking) wave(); }, 80);

  /* ---------- connection ---------- */
  let starting = false, ws = null, live = false, saved = null, lastWant = null, frameAt = 0, camBackTimer = 0;
  const send = (type, fields) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type, ...fields })); };
  const camBox = $('#camBox'), camLive = $('#camLive');

  function setLive(on) {
    if (on === live) return;
    live = on; D.engineLive = on;       // app.js stops the fake natural blinks while live
    if (on) starting = false;
    $('#allowCamTxt').textContent = on ? 'Engine camera on ✓' : 'Allow camera'; $('#demoCamBtn').textContent = on ? 'Camera on' : 'Allow camera';
    pill.classList.toggle('off', !on);
    $('#engTxt').textContent = on ? 'Engine live' : (starting ? 'Starting engine…' : 'Simulated');
    screenEl.classList.toggle('dz-live', on);
    $('#accTxt').textContent = on ? 'head' : 'eyes';
    if (on) {
      saved = { dwell: G.dwell, jitter: G.jitter, steady: G.steady, paused: G.paused };
      G.dwell = false; G.jitter = 0; G.steady = 0;   // the pointer is the real mouse, which the engine moves: no fake wobble or lag
      G.seen = true;
      log('sys', 'Engine connected at ' + URL_WS); toast('Engine connected');
      tip('Move your head to move the pointer. A long blink clicks. Say “hi Dizzy” to dictate.');
    } else {
      if (saved) { G.dwell = saved.dwell; G.jitter = saved.jitter; G.steady = saved.steady; setPausedUI(saved.paused); }
      $('#camSrc').textContent = 'source: simulated face'; showWebcam(false);
      setRead(''); D.micLive = false; lastWant = null;
      S.q = null; S.preview = null; S.liveText = ''; S.speaking = false;
      log('sys', 'Engine disconnected, back to the simulation');
    }
    render();
  }

  function connect() {
    try { ws = new WebSocket(URL_WS); } catch (e) { return setTimeout(connect, RETRY_MS); }
    ws.onopen = () => { setLive(true); beat(true); };
    ws.onmessage = ev => { let m; try { m = JSON.parse(ev.data); } catch (e) { return; } if (m && m.type) handle(m); };
    ws.onclose = () => { setLive(false); setTimeout(connect, RETRY_MS); };
    ws.onerror = () => { try { ws.close(); } catch (e) {} };
  }

  /* page mode heartbeat: the engine acts inside this page only while this keeps arriving */
  function beat(force) {
    if (!live) return;
    const want = !$('#demo').hidden && document.visibilityState === 'visible';
    if (want || force || lastWant !== want) send('page_mode', { on: want });
    lastWant = want;
  }
  setInterval(beat, HEARTBEAT_MS);
  document.addEventListener('visibilitychange', () => beat(true));

  /* ---------- webcam picture ---------- */
  function showWebcam(on) {
    camBox.classList.toggle('live', on); camLive.hidden = !on;
    if (!on) camLive.removeAttribute('src');
  }
  function onFrame(m) {
    camLive.src = 'data:image/jpeg;base64,' + m.jpeg;
    if (camLive.hidden) { showWebcam(true); $('#camSrc').textContent = 'source: your webcam (engine)'; }
    $('#camRec').textContent = 'LIVE · ' + m.w + '×' + m.h;
    frameAt = performance.now();
    clearTimeout(camBackTimer);
    camBackTimer = setTimeout(() => { if (performance.now() - frameAt >= FRAME_GAP_MS) { showWebcam(false); $('#camSrc').textContent = 'source: simulated face'; $('#camRec').textContent = 'TRACKING · 640×480'; } }, FRAME_GAP_MS + 100);
  }

  /* ---------- engine -> page ---------- */
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
    brow_raise: ['brow',  '#sigBlink', 'eye',  'RAISE',  'Eyebrow raise'],
    blink:      ['brow',  '#sigBlink', 'eye',  'BLINK',  'Long blink'],
    mouth_open: ['mouth', '#sigVoice', 'head', 'MOUTH',  'Mouth open'],
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
    D.glabel(g[3]);
    log(g[2], g[4] + ' seen by the engine');
  }

  function handle(m) {
    switch (m.type) {
      case 'mode':
        S.mode = m.mode; $('#trackTxt').textContent = ({ pointer: 'Tracking', dictation: 'Dictation', scroll: 'Scroll', menu: 'Menu', paused: 'Paused' })[m.mode] || m.mode;
        setPausedUI(m.mode === 'paused'); log('sys', 'Mode: ' + m.mode); render(); break;
      case 'gaze': break;     // the engine moves the real mouse, which the screen already follows (pointermove)
      case 'gesture': gesture(m.gesture); break;
      case 'dwell': setRead(m.progress > 0 ? 'dwell ' + Math.round(m.progress * 100) + '%' : (m.ready ? 'dwell ready' : '')); break;
      case 'scroll': setRead(m.ready && m.rate ? 'scroll ' + (m.rate > 0 ? '↑' : '↓') + Math.abs(m.rate).toFixed(1) + '/s' : ''); break;
      case 'listening': S.speaking = !!m.speaking; S.level = m.level || 0; D.micLive = S.speaking; if (S.speaking) FACE.speakUntil = performance.now() + 400; render(); break;
      case 'dictation_live': S.liveText = m.text || ''; render(); break;
      case 'dictation_preview': {
        const had = S.preview && S.preview.words.length, has = m.words && m.words.length;
        S.preview = has ? { words: m.words, cleaned: !!m.cleaned, original: m.original || '' } : null;
        if (had && !has && performance.now() - lastTypeAt > 500) setTimeout(() => { if (performance.now() - lastTypeAt > 500) say('Dropped'); }, 450);
        render(); break;
      }
      case 'command': S.flash = m.label; S.flashUntil = performance.now() + 1400; flashSig('#sigVoice'); log('voice', 'Command: ' + m.label); render(); setTimeout(render, 1450); break;
      case 'confirm': onConfirm(m); break;
      case 'target_searching': log('voice', m.snap ? 'UI-Venus: finding the button under the cursor…' : 'UI-Venus: looking for “' + m.label + '”…'); tip(m.snap ? 'Finding what to click…' : 'Looking for “' + m.label + '”…'); break;
      case 'target_found': log('voice', 'UI-Venus: found “' + m.label + '” at ' + m.x + ', ' + m.y); break;
      case 'target_none': log('voice', 'UI-Venus: could not find “' + m.label + '”'); toast('Could not find “' + m.label + '”'); break;
      case 'status': log('sys', (m.level === 'info' ? '' : m.level.toUpperCase() + ': ') + m.text); toast(m.text); tip(m.text); break;
      case 'calib_target': {
        const c = $('#calib'); c.hidden = false;
        $('#calibMsg').innerHTML = '<h3>Calibration</h3><p>Look at the middle of the screen and keep still.</p>';
        $('#calibPt').hidden = true; log('sys', 'Calibrating: look at the middle of the screen'); break;
      }
      case 'calib_result': $('#calib').hidden = true; log('sys', m.ok ? 'Calibrated' : 'Calibration cancelled'); break;
      case 'debug_frame': onFrame(m); break;
      case 'page_action': onAction(m); break;
    }
  }

  function onConfirm(m) {
    if (m.state === 'open') {
      S.q = { id: m.id, prompt: m.prompt, state: 'open', total: m.expires_in_s || 12, left: m.expires_in_s || 12, t0: performance.now() };
      tip('Diseyeble asks: ' + m.prompt + ' Blink or say yes. Say no to cancel.');
      render();
    } else {
      S.q = null; clearInterval(qTick);
      say({ confirmed: 'Done', cancelled: 'Cancelled', expired: 'Timed out' }[m.state] || m.state);
      log('sys', m.prompt + ' → ' + m.state);
    }
  }

  /* ---------- the card's buttons (a real click on them is an ordinary click on the simulated screen) ---------- */
  D.ACTIONS.dz = v => {
    if (v === 'yes' || v === 'no') send('confirm_answer', { id: S.q && S.q.id, yes: v === 'yes' });
    else if (v === 'insert') send('dictation', { action: 'insert' });
    else if (v === 'discard') send('dictation', { action: 'discard' });
    else if (v === 'original') send('dictation', { action: 'use_original' });
  };

  /* ---------- real clicks land exactly where the pointer is (no snapping) ---------- */
  screenEl.addEventListener('pointerdown', e => {
    if (!live || e.button > 2) return;
    e.stopImmediatePropagation(); e.preventDefault();
    const r = screenEl.getBoundingClientRect(), pos = { x: e.clientX - r.left, y: e.clientY - r.top };
    G.target.x = G.pos.x = pos.x; G.target.y = G.pos.y = pos.y;
    if (G.paused) return;
    if (e.button === 2) { D.ripple('shake', pos); D.glabel('RIGHT CLICK'); D.openCtx(pos.x + 6, pos.y + 6); log('eye', 'Right click'); return; }
    const t = document.elementFromPoint(e.clientX, e.clientY), el = t && screenEl.contains(t) ? t.closest('[data-act]') : null;
    D.ripple('blink', pos);
    if (el) { D.glabel('CLICK'); log('eye', 'Click “' + (el.getAttribute('aria-label') || el.textContent || el.dataset.act).replace(/\s+/g, ' ').trim().slice(0, 30) + '”'); D.run(el.dataset.act, 'eye'); }
    else { D.glabel('CLICK'); $('#ctx').hidden = true; if (W.startOpen) { W.startOpen = false; D.renderAll(); } }
  }, true);

  /* the engine's head scrolling (scroll mode) is a real mouse wheel: it scrolls the simulated page */
  let wheelAcc = 0;
  screenEl.addEventListener('wheel', e => {
    if (!live) return;
    e.preventDefault(); wheelAcc += e.deltaY;
    while (Math.abs(wheelAcc) >= 60) { const d = Math.sign(wheelAcc); D.scrollBy(d); wheelAcc -= d * 60; }
  }, { passive: false });

  /* ---------- page_action: typing, keys, scroll, undo and "open ..." on the simulated desktop ---------- */
  const typed = [];            // {field, text} so "undo" can take back the last dictated chunk
  const FIELD_FOR = () => {    // where dictated words go when no text box has focus yet
    const b = W.browser, m = W.mail;
    if (W.top === 'mail' && m.open && m.view === 'compose') return ['to', 'subj', 'body'].find(f => !W.f[f]) || 'body';
    if (b.open && !b.min && (b.page === 'home' || b.page === 'gresults')) return 'gq';
    if (b.open && !b.min && ['youtube', 'yresults', 'watch'].includes(b.page)) return 'yq';
    return null;
  };
  function actType(text) {
    if (!text) return;
    lastTypeAt = performance.now();
    if (!W.focus) { const f = FIELD_FOR(); if (f) { W.focus = f; log('sys', 'No text box was selected, so typing into ' + f); } }
    const f = W.focus;
    if (!f) { toast('Click a text box first, then dictate'); log('sys', 'Typed “' + text.trim() + '” but no text box is open'); return; }
    const before = W.f[f].length;
    D.insertText(text, false);
    typed.push({ field: f, text: W.f[f].slice(before) });
    D.renderAll(); say('Typed'); log('voice', 'Typed “' + text.trim() + '”');
  }
  function topWin() { return W.top || (W.browser.open && !W.browser.min ? 'browser' : (W.mail.open && !W.mail.min ? 'mail' : null)); }
  function actKey(key) {
    const k = String(key).toLowerCase();
    const f = W.focus;
    if (k === 'enter') {
      if (f === 'body' && W.f.body.trim() && W.mail.view === 'compose') { D.run('send', 'voice'); return; }   // "send it": Enter sends the email
      if (f) { D.typeKey('enter'); D.renderAll(); } return;
    }
    if (k === 'shift+enter') { if (f === 'body') { W.f.body += '\n'; D.renderAll(); } return; }
    if (k === 'backspace') { if (f) { D.typeKey('bksp'); D.renderAll(); } return; }
    if (k === 'tab') { const o = ['to', 'subj', 'body']; if (o.includes(f)) { W.focus = o[(o.indexOf(f) + 1) % 3]; D.renderAll(); } return; }
    if (k === 'esc') { $('#ctx').hidden = true; W.startOpen = false; W.focus = null; D.renderAll(); return; }
    if (k === 'alt+left') return D.run('back', 'voice');
    if (k === 'alt+right') return D.run('fwd', 'voice');
    if (k === 'f5') { toast('Reloaded'); return D.run('reload', 'voice'); }
    if (k === 'ctrl+t' || k === 'ctrl+n' || k === 'ctrl+shift+t') { D.openWin('browser'); D.nav('home'); W.focus = 'gq'; D.renderAll(); return; }
    if (k === 'ctrl+w') { const t = topWin(); if (t) D.run('close:' + t, 'voice'); return; }
    if (k === 'alt+f4') { const t = topWin(); if (t) D.run('close:' + t, 'voice'); return; }
    if (k === 'alt+tab') { if (W.browser.open && W.mail.open) { const to = W.top === 'mail' ? 'browser' : 'mail'; W[to].min = false; D.bringTop(to); D.renderAll(); } return; }
    if (k === 'cmd+d') { W.browser.min = true; W.mail.min = true; W.top = null; D.renderAll(); return; }
    if (k === 'cmd+down') { const t = topWin(); if (t) { W[t].min = true; W.top = null; D.renderAll(); } return; }
    if (k === 'cmd+up') { toast('Already full size in this demo'); return; }
    if (k === 'page_down') { D.scrollBy(1); D.scrollBy(1); D.scrollBy(1); return; }
    if (k === 'page_up') { D.scrollBy(-1); D.scrollBy(-1); D.scrollBy(-1); return; }
    if (k === 'ctrl+end') { for (let i = 0; i < 30; i++) D.scrollBy(1); return; }
    if (k === 'ctrl+home') { for (let i = 0; i < 30; i++) D.scrollBy(-1); return; }
    if (k === 'ctrl+c') return D.ctxAction('copy', 'voice');
    if (k === 'ctrl+v') return D.ctxAction('paste', 'voice');
    if (k === 'ctrl+x') { D.ctxAction('copy', 'voice'); if (f) { W.f[f] = ''; D.renderAll(); } return; }
    if (k === 'ctrl+a') { toast('Selected all'); return; }
    if (k === 'ctrl+s') { toast('Saved'); return; }
    if (k === 'ctrl+f') { toast('Find: not in this demo'); return; }
    if (k === 'ctrl+=' || k === 'ctrl+-') { toast(k === 'ctrl+=' ? 'Zoom in' : 'Zoom out'); return; }
    if (k === 'ctrl+tab' || k === 'ctrl+shift+tab') { toast('One tab in this demo'); return; }
    log('sys', 'Key ' + k + ' (nothing to do in the demo)');
  }
  function actLaunch(m) {
    if (m.app) {
      const a = m.app;
      if (a === 'chrome' || a === 'edge') { D.run('open:browser', 'voice'); D.glabel('OPEN CHROME'); }
      else if (a === 'mail') { D.run('open:mail', 'voice'); D.glabel('OPEN MAIL'); }
      else toast('“' + a + '” opens on the real PC; this demo has Chrome, YouTube and Mail');
      return;
    }
    let u; try { u = new URL(m.url); } catch (e) { return; }
    const host = u.hostname.replace(/^www\./, ''), q = u.searchParams.get('q') || u.searchParams.get('search_query');
    if (host === 'youtube.com') {
      if (q) { D.openWin('browser'); W.f.yq = q; D.nav('yresults'); W.focus = null; D.renderAll(); log('voice', 'YouTube search “' + q + '”'); }
      else D.run('open:youtube', 'voice');
    } else if (host === 'google.com') {
      if (q) { D.openWin('browser'); W.f.gq = q; D.nav('gresults'); W.focus = null; D.renderAll(); log('voice', 'Google search “' + q + '”'); }
      else { D.openWin('browser'); D.nav('home'); D.renderAll(); }
    } else if (host === 'mail.google.com') D.run('open:mail', 'voice');
    else toast(host + ' is not part of this demo');
  }
  function onAction(m) {
    switch (m.action) {
      case 'type': actType(m.text); break;
      case 'key': actKey(m.key); break;
      case 'scroll': for (let i = 0; i < 2; i++) D.scrollBy(m.direction === 'up' ? -1 : 1); D.glabel('SCROLL ' + (m.direction === 'up' ? 'UP' : 'DOWN')); break;
      case 'undo': { const last = typed.pop(); if (last && W.f[last.field].endsWith(last.text)) { W.f[last.field] = W.f[last.field].slice(0, -last.text.length); D.renderAll(); log('voice', 'Undid the last typing'); } break; }
      case 'launch': actLaunch(m); break;
    }
  }

  /* ---------- page -> engine ---------- */
  document.addEventListener('keydown', e => {   // capture: runs before app.js's own P shortcut
    if (!live || $('#demo').hidden || e.repeat || e.target.tagName === 'INPUT') return;
    if (e.key === 'p' || e.key === 'P') { e.stopPropagation(); send('set_mode', { mode: G.paused ? 'pointer' : 'paused' }); }
  }, true);
  document.addEventListener('click', e => {
    if (!live) return;
    if (e.target.closest('#pauseBtn')) { e.stopImmediatePropagation(); e.preventDefault(); send('set_mode', { mode: G.paused ? 'pointer' : 'paused' }); }
    else if (e.target.closest('#calibBtn')) { e.stopImmediatePropagation(); e.preventDefault(); send('calibrate', { action: 'start' }); }
  }, true);

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

/* js/engine.js: connects this page to the real EYEable engine (python -m engine.main) over WebSocket.
   Messages are the ones in engine/core/messages.py. With no engine running the page keeps its simulation.
   Needs window.DZ, which js/app.js exposes just before its boot block.
   Open the page with ?engine=ws://host:port to use another address (default ws://127.0.0.1:8765).

   While the engine is connected and this demo screen is showing, the page is in "page mode":
   - the pointer is the real mouse (the engine moves it with your head); a real click is a click on the simulated desktop
   - typing, keys, scrolling and "open ..." arrive as page_action messages and happen on the simulated desktop
   - the webcam picture arrives as debug_frame and shows in the camera box
   - the EYEable card (mode, what you are saying, what will be typed, questions) is drawn here on the simulated screen
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

  /* ---------- the EYEable card, on the simulated screen ---------- */
  const MODES = { pointer: ['EYEable', ''], dictation: ['Dictation', 'say “dictate off” to stop'], scroll: ['Scroll', 'tilt your head · say “stop”'],
                  menu: ['Menu', ''], paused: ['Paused', 'say “resume”'] };
  const screenEl = D.screen;
  const card = document.createElement('div');
  card.className = 'dz-card'; card.hidden = true;
  card.innerHTML = `<div class="dz-pill"><i class="dz-dot"></i><span class="dz-pt">EYEable</span><small class="dz-ph"></small>
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
    if (!live && !liteCard) { card.hidden = true; badge.hidden = true; return; }
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
      panel.hidden = false; lab.textContent = 'EYEable asks';
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
  let liteSend = null, liteCard = false;   // lite mode's voice (js/lite-voice.js) plays the engine's part when there is no engine
  const send = (type, fields) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type, ...fields })); else if (liteSend) liteSend(type, fields); };
  const camBox = $('#camBox'), camLive = $('#camLive');

  function setLive(on) {
    if (on === live) return;
    live = on; D.engineLive = on;       // app.js stops the fake natural blinks while live
    if (on) starting = false;
    $('#allowCamTxt').textContent = on ? 'Engine camera on ✓' : 'Allow camera'; $('#demoCamBtn').textContent = on ? 'Camera on' : 'Allow camera';
    pill.classList.toggle('off', !on);
    $('#engTxt').textContent = on ? 'Engine live' : (starting ? 'Starting engine…' : 'Simulated');
    screenEl.classList.toggle('dz-live', on);
    $('#accTxt').textContent = on ? 'head' : 'simulated';
    if (on) {
      saved = { dwell: G.dwell, paused: G.paused };
      G.dwell = false;   // the engine decides about dwell clicks
      G.seen = true;
      log('sys', 'Engine connected at ' + URL_WS); toast('Engine connected');
      tip('Move your head to move the pointer. A long blink clicks. Say “hi Dizzy” to dictate.');
    } else {
      if (saved) { G.dwell = saved.dwell; setPausedUI(saved.paused); }
      $('#camSrc').textContent = 'source: simulated face'; showWebcam(false);
      setRead(''); D.micLive = false; lastWant = null;
      S.q = null; S.preview = null; S.liveText = ''; S.speaking = false;
      SYS.bright = 100; SYS.muted = false; applySys();
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

  /* page mode heartbeat: the engine acts inside this page only while this keeps arriving.
     "inside": the real cursor is over this page, so the engine sends its clicks here instead of through Windows. */
  let inside = false, offX = null, offY = null, lastX = 0, lastY = 0;
  function beat(force) {
    if (!live) return;
    const want = !$('#demo').hidden && document.visibilityState === 'visible';
    if (want || force || lastWant !== want) send('page_mode', { on: want, inside: want && inside && offX !== null });
    lastWant = want;
  }
  setInterval(beat, HEARTBEAT_MS);
  document.addEventListener('visibilitychange', () => beat(true));
  // where this page sits on the screen: screen px = (clientX + offX) * devicePixelRatio
  const track = e => {
    offX = e.screenX - e.clientX; offY = e.screenY - e.clientY; lastX = e.clientX; lastY = e.clientY;
    if (!inside) { inside = true; beat(true); }
  };
  document.addEventListener('pointermove', track, true);
  document.addEventListener('pointerdown', track, true);
  document.documentElement.addEventListener('mouseleave', () => { if (inside) { inside = false; beat(true); } });

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
      tip('EYEable asks: ' + m.prompt + ' Blink or say yes. Say no to cancel.');
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
    if (k === 'ctrl+=' || k === 'ctrl+-') { actZoom(k === 'ctrl+=' ? 1 : -1); return; }
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
  /* ---------- volume and brightness of the simulated desktop ("volume up", "mute", "brightness down", ...) ---------- */
  // Same steps as the real engine (engine/eyes/system_controls.py): 10% a command, brightness never under 10%.
  const SYS = { vol: 50, muted: false, bright: 100 };
  const dim = document.createElement('div'); dim.className = 'sim-dim'; screenEl.append(dim);
  const osd = document.createElement('div'); osd.className = 'sim-osd';
  osd.innerHTML = '<span class="ico"></span><span class="bar"><i></i></span><span class="num"></span>';
  screenEl.append(osd);
  const ICONS = {
    vol: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z"/><path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11"/></svg>',
    mute: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z"/><path d="M16 9.5l5 5M21 9.5l-5 5"/></svg>',
    sun: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4"/></svg>',
  };
  let osdTimer = 0, audio = null;
  // a short soft tone at the new volume, like a laptop's volume keys (browsers allow sound once the page was clicked)
  const wakeAudio = () => { try { audio = audio || new (window.AudioContext || window.webkitAudioContext)(); if (audio.state === 'suspended') audio.resume(); } catch (e) {} };
  document.addEventListener('pointerdown', wakeAudio, true); document.addEventListener('keydown', wakeAudio, true);
  function tone() {
    if (!audio || audio.state !== 'running' || SYS.muted || !SYS.vol) return;
    const o = audio.createOscillator(), g = audio.createGain(), t = audio.currentTime;
    o.type = 'sine'; o.frequency.value = 660;
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.25 * SYS.vol / 100 + 0.0001, t + 0.015);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.16);
    o.connect(g); g.connect(audio.destination); o.start(t); o.stop(t + 0.18);
  }
  function showOsd(kind, level, muted) {
    osd.querySelector('.ico').innerHTML = kind === 'bright' ? ICONS.sun : (muted ? ICONS.mute : ICONS.vol);
    osd.querySelector('.bar i').style.width = level + '%';
    osd.querySelector('.num').textContent = kind === 'vol' && muted ? 'Muted' : level;
    osd.classList.toggle('muted', kind === 'vol' && muted);
    osd.classList.add('show'); clearTimeout(osdTimer); osdTimer = setTimeout(() => osd.classList.remove('show'), 1800);
  }
  function applySys() {
    dim.style.opacity = ((100 - SYS.bright) / 100 * 0.75).toFixed(3);   // 10% brightness = 68% black over the screen
    const tv = $('#trayVol'); if (tv) tv.style.opacity = SYS.muted || !SYS.vol ? .3 : (SYS.vol < 34 ? .55 : SYS.vol < 67 ? .8 : 1);   // the tray volume glyph fades with the level
  }
  function actSystem(name) {
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    if (name === 'volume_up' || name === 'volume_down') {
      SYS.muted = false;                                     // like Windows: a volume key unmutes
      SYS.vol = clamp(SYS.vol + (name === 'volume_up' ? 10 : -10), 0, 100);
      showOsd('vol', SYS.vol, false); tone();
      S.flash = 'Volume ' + SYS.vol + '%'; log('voice', 'Volume ' + SYS.vol + '%');
    } else if (name === 'mute') {
      SYS.muted = !SYS.muted;
      showOsd('vol', SYS.vol, SYS.muted); if (!SYS.muted) tone();
      S.flash = SYS.muted ? 'Muted' : 'Sound on'; log('voice', SYS.muted ? 'Muted' : 'Sound on');
    } else if (name === 'brightness_up' || name === 'brightness_down') {
      SYS.bright = clamp(SYS.bright + (name === 'brightness_up' ? 10 : -10), 10, 100);
      showOsd('bright', SYS.bright, false);
      S.flash = 'Brightness ' + SYS.bright + '%'; log('voice', 'Brightness ' + SYS.bright + '%');
    } else return;
    applySys();
    S.flashUntil = performance.now() + 1400; render(); setTimeout(render, 1450);
  }
  // "zoom in" / "zoom out": the simulated browser page gets bigger or smaller, with the level shown like Chrome does
  let zoom = 1;
  function actZoom(dir) {
    zoom = Math.max(0.5, Math.min(2, Math.round((zoom + dir * 0.1) * 10) / 10));
    const p = $('#bPage'); if (p) p.style.zoom = zoom;
    let z = $('#simZoom');
    if (!z) { z = document.createElement('span'); z.id = 'simZoom'; z.className = 'sim-zoom'; const omni = $('.omni'); (omni || screenEl).append(z); }
    z.textContent = Math.round(zoom * 100) + '%'; z.hidden = zoom === 1;
    if (!W.browser.open) toast('Zoom ' + Math.round(zoom * 100) + '% (open Chrome to see it)');
    log('voice', 'Zoom ' + Math.round(zoom * 100) + '%');
  }

  // The engine's click (a long blink, "click", "click Send" after yes) while the cursor is over this page: click that spot
  function actClick(m) {
    if (offX === null) { log('sys', 'Click from the engine, but the page does not know where it is yet'); return; }
    const dpr = window.devicePixelRatio || 1;
    const cx = m.x / dpr - offX, cy = m.y / dpr - offY;
    const how = m.button === 'right' ? 'Right click' : 'Blink click';
    if (Math.hypot(cx - lastX, cy - lastY) > 40)   // a hint for debugging: the page's idea of the cursor and the engine's differ
      log('sys', `Engine click at ${Math.round(cx)}, ${Math.round(cy)}; the cursor was last seen at ${Math.round(lastX)}, ${Math.round(lastY)} (browser zoom not 100%?)`);
    const sr = screenEl.getBoundingClientRect();
    if (cx >= sr.left && cx <= sr.right && cy >= sr.top && cy <= sr.bottom) {
      const what = D.clickAt(cx, cy, m.button === 'right' ? 'right' : 'left', how);
      if (what === '') log('eye', how + ': nothing to click there');
      return;
    }
    // outside the simulated screen: an ordinary click on this page (Pause, the menu at the top, ...)
    const el = document.elementFromPoint(cx, cy), target = el && el.closest('button, a, [data-go], [data-page-go]');
    if (target) { log('eye', how + ' “' + (target.textContent || '').trim().slice(0, 30) + '”'); target.click(); }
  }

  function onAction(m) {
    switch (m.action) {
      case 'click': actClick(m); break;
      case 'system': actSystem(m.name); break;
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
  // capture phase: the page's own camera code never runs, because one app at a time can hold a webcam on Windows.
  // On the hosted website (GitHub Pages) there is no launcher: with no engine, the button shows the browser's own
  // webcam as before, and the page connects by itself whenever EYEable is running on this PC.
  const LOCAL = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
  document.addEventListener('click', e => {
    if (!e.target.closest('#allowCam, #demoCamBtn')) return;
    if (!LOCAL && !live) { toast('Showing your webcam. To control this page with your head, eyes and voice, start EYEable on your PC: the page connects by itself.'); return; }
    e.stopImmediatePropagation(); e.preventDefault(); startEngine();
  }, true);

  // lite mode (js/lite.js, js/lite-voice.js) drives the same card and page actions with its own in-browser voice
  D.bridge = { handle, onAction, setSend(fn) { liteSend = fn; }, setCard(on) { liteCard = !!on; render(); } };

  connect();
})();

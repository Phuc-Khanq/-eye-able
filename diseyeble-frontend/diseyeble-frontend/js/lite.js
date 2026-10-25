/* js/lite.js: "lite mode", hands-free control of the SIMULATED desktop with no install.
   The webcam stays in this browser tab. MediaPipe Face Landmarker (Google, Apache 2.0) runs here in WebAssembly;
   nothing is recorded or uploaded. It is NOT the Diseyeble engine: less accurate, no dysarthria-tuned dictation,
   no "click <name>" look-and-say, and it never touches the real computer, only the simulator on this page.

   What it does (the same gestures as the engine):
   - head turn / nod moves the pointer on the simulated screen (nose position relative to the eyes)
   - a long blink (0.4 to 1.2 s, both eyes) clicks where the pointer is; quick natural blinks are ignored
   - optional: mouth open for 0.6 s = right click (off by default: talking opens the mouth)
   - voice: the page's own microphone button (browser speech recognition) still works as before
   The engine wins: if Diseyeble is running on this PC, this mode switches itself off.

   The tracking maths lives in createCore() with no browser calls, so tests/lite_core.test.js can run it in node. */
(() => {
  'use strict';

  const CFG = {
    rangeX: 0.26,        // nose offset (in eye-distances) that reaches the screen edge at sensitivity 1, left-right
    rangeY: 0.15,        // same, up-down (nodding moves the nose less than turning does)
    blinkClose: 0.55, blinkOpen: 0.35,   // eyeBlink score (the smaller of the two eyes) to count as shut / open again
    blinkMinMs: 400, blinkMaxMs: 1200,   // a long blink; same window as the engine
    freezeAt: 0.30, freezeAfterMs: 250, backMs: 200,   // the pointer holds still while the eyelids move
    jawOpen: 0.6, jawClose: 0.3, jawHoldMs: 600,
    calibFrames: 20,
    slowAlpha: 0.25, fastAlpha: 0.6, fastDist: 0.04, deadZone: 0.003,
  };
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  /* sample: {nose:{x,y}, eyeA:{x,y}, eyeB:{x,y}, blinkL, blinkR, jaw, t(ms)} or null when no face.
     step() returns {x, y (0..1 on the screen), status, calib (0..1), blinkMs, events:[{type:'click'|'rightclick', ms}]} */
  function createCore(over = {}) {
    const cfg = { ...CFG, ...over };
    const st = { cx: null, cy: null, calib: [], sx: 0.5, sy: 0.5, hist: [],
                 closedSince: null, reopenAt: -1e9, jawSince: null, jawArmed: true,
                 sens: 1, flipX: false, flipY: false, mouthClick: false };

    function recenter() { st.cx = st.cy = null; st.calib = []; st.hist = []; }

    function step(s) {
      const out = { x: st.sx, y: st.sy, status: 'noface', calib: 0, blinkMs: 0, events: [] };
      if (!s || !s.nose || !s.eyeA || !s.eyeB) { st.closedSince = null; st.jawSince = null; return out; }
      const ed = Math.hypot(s.eyeA.x - s.eyeB.x, s.eyeA.y - s.eyeB.y);
      if (!(ed > 1e-4)) { st.closedSince = null; return out; }
      const dx = (s.nose.x - (s.eyeA.x + s.eyeB.x) / 2) / ed;
      const dy = (s.nose.y - (s.eyeA.y + s.eyeB.y) / 2) / ed;

      if (st.cx === null) {                         // look straight ahead for ~0.7 s: that is the centre
        st.calib.push({ dx, dy });
        if (st.calib.length >= cfg.calibFrames) {
          st.cx = st.calib.reduce((a, c) => a + c.dx, 0) / st.calib.length;
          st.cy = st.calib.reduce((a, c) => a + c.dy, 0) / st.calib.length;
          st.calib = []; st.sx = st.sy = 0.5;
        } else { out.status = 'calibrating'; out.calib = st.calib.length / cfg.calibFrames; return out; }
      }

      const t = s.t, mn = Math.min(s.blinkL || 0, s.blinkR || 0);
      // eyes: shut / open with hysteresis
      let closedNow = st.closedSince !== null ? mn > cfg.blinkOpen : mn >= cfg.blinkClose;
      if (closedNow && st.closedSince === null) {
        st.closedSince = t;
        const old = st.hist.find(h => h.t >= t - cfg.backMs);       // the pointer where it was before the lids moved
        if (old) { st.sx = old.x; st.sy = old.y; }
      } else if (!closedNow && st.closedSince !== null) {
        const dur = t - st.closedSince; st.closedSince = null; st.reopenAt = t;
        if (dur >= cfg.blinkMinMs && dur <= cfg.blinkMaxMs) out.events.push({ type: 'click', ms: dur });
      }
      if (st.closedSince !== null) out.blinkMs = t - st.closedSince;

      // pointer: hold still while the eyes are shut or just reopened
      const frozen = st.closedSince !== null || mn >= cfg.freezeAt || t - st.reopenAt < cfg.freezeAfterMs;
      if (!frozen) {
        const hx = cfg.rangeX / st.sens, hy = cfg.rangeY / st.sens;
        // the camera image is not mirrored: turning right moves the nose to the image's left, so subtract
        let tx = 0.5 + (st.flipX ? 1 : -1) * ((dx - st.cx) / hx) * 0.5;
        let ty = 0.5 + (st.flipY ? -1 : 1) * ((dy - st.cy) / hy) * 0.5;
        tx = clamp(tx, 0, 1); ty = clamp(ty, 0, 1);
        const d = Math.hypot(tx - st.sx, ty - st.sy);
        if (d > cfg.deadZone) {
          const a = d > cfg.fastDist ? cfg.fastAlpha : cfg.slowAlpha;
          st.sx += (tx - st.sx) * a; st.sy += (ty - st.sy) * a;
        }
        st.hist.push({ t, x: st.sx, y: st.sy });
        while (st.hist.length && st.hist[0].t < t - 1000) st.hist.shift();
      }

      // mouth open held = right click (if switched on)
      const jaw = s.jaw || 0;
      if (jaw >= cfg.jawOpen) {
        if (st.jawSince === null) st.jawSince = t;
        if (st.jawArmed && st.mouthClick && t - st.jawSince >= cfg.jawHoldMs) { st.jawArmed = false; out.events.push({ type: 'rightclick', ms: t - st.jawSince }); }
      } else if (jaw <= cfg.jawClose) { st.jawSince = null; st.jawArmed = true; }

      out.x = st.sx; out.y = st.sy;
      out.status = st.closedSince !== null ? 'closed' : 'tracking';
      return out;
    }
    const set = (k, v) => { st[k] = v; };
    return { step, recenter, set, cfg, state: st };
  }

  if (typeof window === 'undefined' || typeof document === 'undefined') {            // node (the unit tests)
    if (typeof module !== 'undefined') module.exports = { createCore, CFG };
    return;
  }
  const D = window.DZ; if (!D) return;
  const { $, log, toast, tip, G, screen } = D;

  /* ---------------- where MediaPipe comes from (pinned versions) ---------------- */
  const VERSION = '0.10.14';
  const BUNDLE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/vision_bundle.mjs`;
  const WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`;
  const LOCAL_MODEL = 'assets/lite/face_landmarker.task';     // optional: put the model here to avoid the Google download
  const REMOTE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
  const NOSE = 1, EYE_A = 33, EYE_B = 263;                     // landmark numbers: nose tip, outer eye corners

  async function realLandmarker() {
    const mod = await import(BUNDLE);
    const fileset = await mod.FilesetResolver.forVisionTasks(WASM);
    let model = REMOTE_MODEL;
    try { const r = await fetch(LOCAL_MODEL, { method: 'HEAD' }); if (r.ok) model = LOCAL_MODEL; } catch (e) { /* use the download */ }
    const make = delegate => mod.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: model, delegate }, runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true });
    let lm; try { lm = await make('GPU'); } catch (e) { lm = await make('CPU'); }
    return { detect: (v, t) => lm.detectForVideo(v, t), close: () => lm.close() };
  }

  /* ---------------- the panel under the camera ---------------- */
  const css = document.createElement('style');
  css.textContent = `
  .lite{margin-top:10px;display:grid;gap:8px;font-size:13px}
  .lite button{cursor:pointer}
  .lite-stat{font:600 12px var(--f-mono,monospace);color:var(--muted,#667);min-height:1.3em}
  .lite details{border:1px solid var(--line,#d8d8d8);border-radius:10px;padding:6px 10px}
  .lite summary{cursor:pointer;font-weight:600}
  .lite label{display:flex;align-items:center;gap:8px;margin:6px 0}
  .lite input[type=range]{flex:1}
  .lite p{margin:0;color:var(--muted,#667);font-size:12px;line-height:1.4}
  .lite-cur{position:absolute;left:0;top:0;width:30px;height:30px;margin:-15px 0 0 -15px;border-radius:50%;z-index:70;pointer-events:none;
    border:3px solid #fff;box-shadow:0 0 0 2px #111,0 0 14px rgba(0,0,0,.5);background:rgba(17,17,17,.25);transition:transform .08s,border-color .1s}
  .lite-cur::after{content:"";position:absolute;inset:10px;border-radius:50%;background:#111}
  .lite-cur.shut{border-color:#F0A94B}.lite-cur.ready{border-color:#4ADE80;transform:scale(1.25)}
  #liteCanvas{position:absolute;inset:0;width:100%;height:100%;display:block;z-index:3;border-radius:inherit}`;
  document.head.append(css);

  const camBox = $('#camBox');
  const panel = document.createElement('div');
  panel.className = 'lite';
  panel.innerHTML = `
    <button class="mini-btn" id="liteBtn" type="button">Try hands-free in your browser (lite)</button>
    <div class="lite-stat" id="liteStat" aria-live="polite"></div>
    <details id="liteSet"><summary>Lite settings</summary>
      <label>Sensitivity <input type="range" id="liteSens" min="0.5" max="2.5" step="0.1" value="1"><span id="liteSensV">1.0×</span></label>
      <label><input type="checkbox" id="liteFx"> Flip left-right</label>
      <label><input type="checkbox" id="liteFy"> Flip up-down</label>
      <label><input type="checkbox" id="liteMouth"> Mouth open 0.6 s = right click</label>
      <button class="mini-btn" id="liteRecenter" type="button">Recenter (look straight ahead)</button>
    </details>
    <p>Runs in this tab only: your camera is not recorded or sent anywhere. Lite mode is a simulation and less accurate than the Diseyeble app, which also types by voice with the tuned model.</p>`;
  camBox.parentElement.append(panel);
  const el = id => document.getElementById(id);
  const setStat = t => { el('liteStat').textContent = t; };

  /* ---------------- run / stop ---------------- */
  const core = createCore();
  let on = false, starting = false, lm = null, stream = null, video = null, cursor = null, canvas = null, badge = null,
      raf = 0, guard = 0, lastVideoT = -1, lastClickAt = 0;

  function sample(res, t) {
    const f = res && res.faceLandmarks && res.faceLandmarks[0];
    if (!f) return null;
    const bs = {};
    const cats = res.faceBlendshapes && res.faceBlendshapes[0] && res.faceBlendshapes[0].categories;
    if (cats) for (const c of cats) bs[c.categoryName] = c.score;
    return { nose: f[NOSE], eyeA: f[EYE_A], eyeB: f[EYE_B], blinkL: bs.eyeBlinkLeft, blinkR: bs.eyeBlinkRight, jaw: bs.jawOpen, t, f };
  }

  function loop() {
    if (!on) return;
    raf = requestAnimationFrame(loop);
    if (!video || video.readyState < 2 || video.currentTime === lastVideoT) return;
    lastVideoT = video.currentTime;
    const t = performance.now();
    let s = null;
    try { s = sample(lm.detect(video, t), t); } catch (e) { setStat('Tracker error: ' + e.message); return; }
    const o = core.step(s);
    draw(s, o);
    if (o.status === 'noface') { setStat('No face found. Move into view and face the screen.'); return; }
    if (o.status === 'calibrating') { setStat('Look at the middle of the screen and hold still… ' + Math.round(o.calib * 100) + '%'); return; }

    const r = screen.getBoundingClientRect();
    const px = o.x * r.width, py = o.y * r.height;
    if (!G.paused && !G.scripted) { G.target.x = G.pos.x = px; G.target.y = G.pos.y = py; }
    cursor.style.transform = `translate(${px}px,${py}px)`;
    cursor.className = 'lite-cur' + (o.status === 'closed' ? (o.blinkMs >= core.cfg.blinkMinMs ? ' ready' : ' shut') : '');
    setStat(o.status === 'closed'
      ? (o.blinkMs >= core.cfg.blinkMinMs ? 'Eyes shut ' + (o.blinkMs / 1000).toFixed(1) + ' s: open them to click' : 'Eyes shut… keep them closed a bit longer to click')
      : 'Tracking. Turn or nod to move, hold a long blink to click.');

    for (const ev of o.events) {
      if (ev.type === 'click') {
        if (G.pending) { log('eye', 'Lite: long blink → yes'); el('cqYes').click(); continue; }
        const what = D.clickAt(r.left + px, r.top + py, 'left', 'Lite: long blink → click');
        D.glabel('LONG BLINK · CLICK'); lastClickAt = t;
        if (what === '') log('eye', 'Lite: long blink → click (nothing to click there)');
      } else if (ev.type === 'rightclick') {
        D.clickAt(r.left + px, r.top + py, 'right', 'Lite: mouth open → right click');
      }
    }
  }

  function draw(s, o) {
    const c = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
    c.save(); c.translate(w, 0); c.scale(-1, 1);                     // mirrored, like a mirror
    c.drawImage(video, 0, 0, w, h);
    c.restore();
    if (s && s.f) {
      c.fillStyle = o.status === 'closed' ? '#F0A94B' : '#7DD3FC';
      for (const p of [s.nose, s.eyeA, s.eyeB]) { c.beginPath(); c.arc((1 - p.x) * w, p.y * h, 4, 0, 6.3); c.fill(); }
    }
  }

  async function start() {
    if (on || starting) return;
    if (D.engineLive) return toast('Diseyeble is running on this PC and already controls this page. Lite mode is not needed.');
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return toast('This browser can’t give the page a webcam. Use Chrome or Edge on https.');
    starting = true; el('liteBtn').disabled = true; setStat('Loading the face tracker (about 10 MB the first time)…');
    try {
      lm = await (window.__DZ_LITE_VISION ? window.__DZ_LITE_VISION() : realLandmarker());
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
    } catch (e) {
      starting = false; el('liteBtn').disabled = false; cleanup();
      setStat(e && e.name === 'NotAllowedError' ? 'The browser did not give this page the webcam.' : 'Could not start lite mode: ' + (e && e.message || e));
      return;
    }
    video = document.createElement('video'); video.playsInline = true; video.muted = true; video.srcObject = stream;
    await video.play().catch(() => {});
    canvas = document.createElement('canvas'); canvas.id = 'liteCanvas'; canvas.width = 480; canvas.height = 360; camBox.append(canvas);
    cursor = document.createElement('div'); cursor.className = 'lite-cur'; screen.append(cursor);
    badge = document.createElement('div'); badge.className = 'sim-badge'; badge.textContent = 'SIMULATION · LITE MODE'; screen.append(badge);
    $('#camSrc').textContent = 'source: your webcam (lite, in this tab)';
    core.recenter(); on = true; starting = false; D.engineLive = true;
    el('liteBtn').disabled = false; el('liteBtn').textContent = 'Stop lite mode';
    log('sys', 'Lite mode on: the head moves the pointer, a long blink clicks (simulation, runs in this tab)');
    tip('Lite mode: look at the middle of the screen for a second, then turn your head to move and hold a long blink to click.');
    guard = setInterval(() => { const p = document.getElementById('engPill'); if (p && !p.classList.contains('off')) stop('Diseyeble connected: using the real engine instead.', true); }, 1000);
    raf = requestAnimationFrame(loop);
  }

  function cleanup() {
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (lm && lm.close) { try { lm.close(); } catch (e) { /* ignore */ } }
    [cursor, canvas, badge].forEach(n => n && n.remove());
    stream = lm = video = cursor = canvas = badge = null;
  }
  function stop(why, engineTook) {
    if (!on) return;
    on = false; cancelAnimationFrame(raf); clearInterval(guard); cleanup();
    if (!engineTook) D.engineLive = false;
    el('liteBtn').textContent = 'Try hands-free in your browser (lite)';
    $('#camSrc').textContent = 'source: simulated face';
    setStat(why || 'Lite mode off.'); log('sys', why || 'Lite mode off');
  }

  el('liteBtn').addEventListener('click', () => (on ? stop() : start()));
  el('liteRecenter').addEventListener('click', () => { core.recenter(); setStat('Look at the middle of the screen and hold still…'); });
  el('liteSens').addEventListener('input', e => { core.set('sens', +e.target.value); el('liteSensV').textContent = (+e.target.value).toFixed(1) + '×'; });
  el('liteFx').addEventListener('change', e => core.set('flipX', e.target.checked));
  el('liteFy').addEventListener('change', e => core.set('flipY', e.target.checked));
  el('liteMouth').addEventListener('change', e => core.set('mouthClick', e.target.checked));
  window.DZLite = { start, stop, core };
})();

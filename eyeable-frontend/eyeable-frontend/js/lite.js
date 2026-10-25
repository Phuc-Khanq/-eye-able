/* js/lite.js: "lite mode", hands-free control of the SIMULATED desktop with no install.
   The webcam stays in this browser tab. MediaPipe Face Landmarker (Google, Apache 2.0) runs here in WebAssembly;
   nothing is recorded or uploaded. It is NOT the [eye]able engine: less accurate, no dysarthria-tuned dictation,
   no "click <name>" look-and-say, and it never touches the real computer, only the simulator on this page.

   What it does (the same gestures as the engine):
   - turning or nodding your head moves the pointer like a mouse (the engine's head pointer, ported: true head angles from
     MediaPipe's face matrix, acceleration, filtering); the eyes are not used to aim, only to click
   - a long blink (0.4 to 1.6 s, both eyes) clicks where the pointer is; quick natural blinks are ignored
   - optional: mouth open for 0.6 s = right click (off by default: talking opens the mouth)
   - scroll mode (say "scroll", or the Scroll button): the pointer holds still and tilting the chin down or up scrolls the page,
     faster the further you tilt, like the engine's engine/eyes/scroll.py; "stop" or a long blink ends it
   - voice: the page's own microphone button (browser speech recognition) still works as before
   The engine wins: if [eye]able is running on this PC, this mode switches itself off.

   The tracking maths lives in createCore() with no browser calls, so tests/lite_core.test.js can run it in node. */
(() => {
  'use strict';

  /* The pointer is the engine's "head" pointer (engine/eyes/pointer.py, HeadRelPointer), ported: the head is a mouse
     with acceleration. Turning slowly aims finely, turning fast throws the pointer far; it works from the CHANGE in
     head angle (so there is nothing to calibrate and no neck strain to reach the edges), the angles are filtered
     (One Euro), a resting head's wobble is gated out and a one-frame tracking glitch is skipped.
     Numbers are in the engine's 1920x1080 pixels; the page scales them to the simulated screen. */
  const REF_W = 1920, REF_H = 1080;
  const CFG = {
    gainSlow: 10, gainFast: 50, speedSlow: 8, speedFast: 60,   // px per degree, and the head speed (deg/s) between them
    maxStepPx: 90, vBoost: 1.4, maxStepDeg: 10, glitchFrames: 2,
    speedAttack: 0.5, speedRelease: 0.2, cutoff: 1.0, beta: 0.3, dCutoff: 1.0,
    restWindowMs: 150, restYaw: [0.08, 0.20], restPitch: [0.12, 0.30],
    frameS: 0.04, maxFrameScale: 4,
    blinkClose: 0.55, blinkOpen: 0.35, blinkMinMs: 400, blinkMaxMs: 1600,   // a long blink: same window as the engine (config.py)
    // edge help, ported from the engine's head-nose mode: the turn that reaches a side edge / the top or bottom is
    // 19 / 11 degrees; past 70% of it the pointer is carried on to the spot that angle points at (never backward)
    edgeYaw: 19, edgePitch: 11, edgeZone: 0.8, edgeRate: 3, edgeMaxPx: 25,   // lite: much weaker than the engine's (0.7 / 5 / 45): the nose pull made the pointer drift; it also only helps while the head is still moving outward
    absYaw: 15, absPitch: 10,          // 'face map' mode: this many degrees from straight ahead reaches a side edge / the top or bottom
    neutralFrames: 15,                 // straight ahead = the median of this many frames after start or recenter
    freezeAt: 0.30, freezeAfterMs: 350, backMs: 300,                        // the pointer holds still while the eyelids move
    jawOpen: 0.6, jawClose: 0.3, jawHoldMs: 600,
  };
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const ramp = (x, lo, hi) => { const k = clamp((x - lo) / (hi - lo), 0, 1); return k * k * (3 - 2 * k); };
  const alpha = (dt, cutoff) => 1 / (1 + 1 / (2 * Math.PI * cutoff) / dt);

  class OneEuro {
    constructor(minCutoff, beta, dCutoff) { this.mc = minCutoff; this.beta = beta; this.dc = dCutoff; this.reset(); }
    reset() { this.t = null; this.x = 0; this.dx = 0; }
    f(t, v) {
      if (this.t === null) { this.t = t; this.x = v; this.dx = 0; return v; }
      const dt = t - this.t; if (dt <= 0) return this.x;
      this.dx = alpha(dt, this.dc) * ((v - this.x) / dt) + (1 - alpha(dt, this.dc)) * this.dx;
      this.x += alpha(dt, this.mc + this.beta * Math.abs(this.dx)) * (v - this.x); this.t = t; return this.x;
    }
  }

  /* Head yaw and pitch (degrees) from MediaPipe's facial transformation matrix, the same maths as
     engine/eyes/head_pose.py: yaw > 0 = turned to your right, pitch > 0 = chin down. data is the 4x4 matrix as a flat
     array; MediaPipe packs it column by column, but the bottom row of zeros tells which way it came. */
  function anglesFromMatrix(data) {
    if (!data || data.length < 16) return null;
    const colMajor = Math.abs(data[3]) + Math.abs(data[7]) + Math.abs(data[11]) < 1e-3;
    const R = (i, j) => (colMajor ? data[j * 4 + i] : data[i * 4 + j]);
    const n = [0, 1, 2].map(j => Math.hypot(R(0, j), R(1, j), R(2, j)));
    if (n.some(v => !(v > 1e-9))) return null;
    const r = (i, j) => R(i, j) / n[j];
    const ry = Math.asin(clamp(-r(2, 0), -1, 1)), rx = Math.atan2(r(2, 1), r(2, 2));
    return { yaw: -ry * 180 / Math.PI, pitch: rx * 180 / Math.PI };
  }
  /* When the matrix is missing: rough angles from the nose's offset from the eyes (in eye-distances). */
  function anglesFromLandmarks(nose, eyeA, eyeB) {
    const ed = Math.hypot(eyeA.x - eyeB.x, eyeA.y - eyeB.y); if (!(ed > 1e-4)) return null;
    return { yaw: -((nose.x - (eyeA.x + eyeB.x) / 2) / ed) * 60, pitch: ((nose.y - (eyeA.y + eyeB.y) / 2) / ed) * 60 };
  }

  /* sample: {yaw, pitch (degrees), blinkL, blinkR, jaw (0..1), t (ms)} or null when no face.
     step() returns {x, y (0..1 on the screen), status, blinkMs, events:[{type:'click'|'rightclick', ms}]} */
  function createCore(over = {}) {
    const cfg = { ...CFG, ...over };
    const st = { nx: 0.5, ny: 0.5, fy: new OneEuro(cfg.cutoff, cfg.beta, cfg.dCutoff), fp: new OneEuro(cfg.cutoff, cfg.beta, cfg.dCutoff),
                 last: null, raw: null, wild: 0, speed: 0, recent: [], hist: [], needRebase: false, neutral: null, nbuf: [], stepX: 0, stepY: 0, mode: 'rel',
                 closedSince: null, reopenAt: -1e9, jawSince: null, jawArmed: true,
                 sens: 1, flipX: false, flipY: false, mouthClick: false };

    function rebase() { st.last = null; st.raw = null; st.wild = 0; st.speed = 0; st.recent = []; st.fy.reset(); st.fp.reset(); }
    function recenter() { st.nx = st.ny = 0.5; st.hist = []; st.neutral = null; st.nbuf = []; rebase(); }
    const gain = speed => { const k = ramp(speed, cfg.speedSlow, cfg.speedFast); return cfg.gainSlow + (cfg.gainFast - cfg.gainSlow) * k; };
    const frameScale = dt => clamp(dt / cfg.frameS, 1, cfg.maxFrameScale);

    /* head-nose edge help (engine/eyes/pointer.py _nose_assist): a relative pointer stops short of the edges as small
       errors pile up. In the outer part of a turn, carry it on to the spot the turn points at, and only forward. */
    function edgeHelp(dyaw, dpitch, dt, k) {
      const axes = [[dyaw * (st.flipX ? -1 : 1), cfg.edgeYaw, 'nx', st.stepX], [dpitch * (st.flipY ? -1 : 1), cfg.edgePitch, 'ny', st.stepY]];
      for (const [ang, edge, key, stepPx] of axes) {
        const a = ang / edge, m = Math.abs(a), dir = a > 0 ? 1 : -1;
        if (m <= cfg.edgeZone) continue;                       // not far enough into the turn
        if (stepPx * dir < 0.5) continue;                      // only while the head is still moving outward: stopped or coming back, no pull at all
        const target = clamp(0.5 + dir * Math.min(m, 1) * 0.5, 0, 1);
        const gap = (target - st[key]) * dir; if (gap <= 0) continue;   // already there, or past it
        const w = (1 - Math.exp(-cfg.edgeRate * dt)) * ramp(m, cfg.edgeZone, 1);
        const ref = key === 'nx' ? REF_W : REF_H;
        st[key] = clamp(st[key] + dir * Math.min(gap * w, cfg.edgeMaxPx * k / ref), 0, 1);
      }
    }

    function move(yaw, pitch, tMs) {
      if (st.raw) {
        const jump = Math.hypot(yaw - st.raw.yaw, pitch - st.raw.pitch);
        if (jump > cfg.maxStepDeg * frameScale((tMs - st.raw.t) / 1000)) {      // a tracking glitch, or a real fast turn
          st.wild++;
          if (st.wild < cfg.glitchFrames) return;                                // one wild frame: skip it
          rebase();                                                              // it stays there: start again from the new angle
        }
      }
      st.wild = 0; st.raw = { yaw, pitch, t: tMs };
      const t = tMs / 1000, fy = st.fy.f(t, yaw), fp = st.fp.f(t, pitch);
      if (!st.neutral) {                                   // straight ahead: the middle of the first frames after start
        st.nbuf.push({ y: yaw, p: pitch });
        if (st.nbuf.length >= cfg.neutralFrames) {
          const med = a => { const b = a.slice().sort((x, y) => x - y); return b[b.length >> 1]; };
          st.neutral = { yaw: med(st.nbuf.map(v => v.y)), pitch: med(st.nbuf.map(v => v.p)) }; st.nbuf = [];
        }
      }
      st.recent.push({ t: tMs, fy, fp });
      while (st.recent.length > 2 && tMs - st.recent[1].t >= cfg.restWindowMs) st.recent.shift();
      if (st.last) {
        const dt = Math.max((tMs - st.last.t) / 1000, 1e-3);
        st.stepX = st.stepY = 0;
        if (st.mode === 'abs') {
          if (st.neutral) {                                    // face map: the head angle IS the pointer position
            const sy = st.sens || 1;
            st.nx = clamp(0.5 + (st.flipX ? -1 : 1) * (fy - st.neutral.yaw) / (2 * cfg.absYaw / sy), 0, 1);
            st.ny = clamp(0.5 + (st.flipY ? -1 : 1) * (fp - st.neutral.pitch) / (2 * cfg.absPitch / sy), 0, 1);
          }
        } else if (dt < 0.5) {
          const dyaw = fy - st.last.fy, dpitch = fp - st.last.fp, k = frameScale(dt);
          const inst = Math.hypot(dyaw, dpitch) / dt;
          const rate = 1 - Math.pow(1 - (inst > st.speed ? cfg.speedAttack : cfg.speedRelease), k);
          st.speed += rate * (inst - st.speed);
          const g = gain(st.speed) * Math.max(st.sens, 0.1);
          const r0 = st.recent[0];
          const kx = ramp(Math.abs(fy - r0.fy), cfg.restYaw[0], cfg.restYaw[1]);   // a resting head's wobble does not move it
          const ky = ramp(Math.abs(fp - r0.fp), cfg.restPitch[0], cfg.restPitch[1]);
          let mx = dyaw * g * kx * (st.flipX ? -1 : 1), my = dpitch * g * cfg.vBoost * ky * (st.flipY ? -1 : 1);
          const step = Math.hypot(mx, my), limit = cfg.maxStepPx * k;
          if (step > limit) { mx *= limit / step; my *= limit / step; }
          st.nx = clamp(st.nx + mx / REF_W, 0, 1); st.ny = clamp(st.ny + my / REF_H, 0, 1);
          st.stepX = mx; st.stepY = my;
          if (st.neutral) edgeHelp(fy - st.neutral.yaw, fp - st.neutral.pitch, dt, k);
        }
      }
      st.last = { t: tMs, fy, fp };
    }

    function step(s) {
      const out = { x: st.nx, y: st.ny, status: 'noface', blinkMs: 0, events: [] };
      if (!s || !Number.isFinite(s.yaw) || !Number.isFinite(s.pitch)) { st.closedSince = null; st.jawSince = null; st.needRebase = true; return out; }
      const t = s.t, mn = Math.min(s.blinkL || 0, s.blinkR || 0);
      // eyes: shut / open, with hysteresis
      const closedNow = st.closedSince !== null ? mn > cfg.blinkOpen : mn >= cfg.blinkClose;
      if (closedNow && st.closedSince === null) {
        st.closedSince = t;
        const old = st.hist.find(h => h.t >= t - cfg.backMs);      // the pointer where it was before the lids moved
        if (old) { st.nx = old.x; st.ny = old.y; }
      } else if (!closedNow && st.closedSince !== null) {
        const dur = t - st.closedSince; st.closedSince = null; st.reopenAt = t;
        if (dur >= cfg.blinkMinMs && dur <= cfg.blinkMaxMs) out.events.push({ type: 'click', ms: dur });
      }
      if (st.closedSince !== null) out.blinkMs = t - st.closedSince;

      // the pointer holds still while the eyes are shut or just reopened, and picks up from the new angle afterwards
      const frozen = st.closedSince !== null || mn >= cfg.freezeAt || t - st.reopenAt < cfg.freezeAfterMs;
      if (frozen) st.needRebase = true;
      else {
        if (st.needRebase) { rebase(); st.needRebase = false; }
        move(s.yaw, s.pitch, t);
        st.hist.push({ t, x: st.nx, y: st.ny });
        while (st.hist.length && st.hist[0].t < t - 1000) st.hist.shift();
      }

      // mouth open held = right click (if switched on)
      const jaw = s.jaw || 0;
      if (jaw >= cfg.jawOpen) {
        if (st.jawSince === null) st.jawSince = t;
        if (st.jawArmed && st.mouthClick && t - st.jawSince >= cfg.jawHoldMs) { st.jawArmed = false; out.events.push({ type: 'rightclick', ms: t - st.jawSince }); }
      } else if (jaw <= cfg.jawClose) { st.jawSince = null; st.jawArmed = true; }

      out.x = st.nx; out.y = st.ny; out.status = st.closedSince !== null ? 'closed' : 'tracking';
      return out;
    }
    const set = (k, v) => { st[k] = v; };
    return { step, recenter, set, cfg, state: st };
  }

  /* Scrolling with the head (engine/eyes/scroll.py, ported). The head works like a joystick: the first 0.4 s sets "straight
     ahead" (the median pitch), then chin down scrolls down and chin up scrolls up: nothing inside a 2 degree dead zone, a gentle
     crawl just outside it for reading, faster the further you tilt, at most 25 notches a second at about 10 degrees. Speed, not
     position: hold a small tilt to keep going, come back to straight to stop. Rates are notches per second, positive = up. */
  const SCROLL = { dead: 2, gain: 1.5, power: 1.3, max: 25, settleMs: 400, maxStepDeg: 10, cutoff: 1, beta: 0.3, tauUp: 0.15, tauDown: 0.3 };
  function createHeadScroll(over = {}) {
    const c = { ...SCROLL, ...over }, f = new OneEuro(c.cutoff, c.beta, 1);
    let t0 = null, samples = [], raw = null, wild = 0, anchor = null;
    const me = { rate: 0, offset: 0,
      get settled() { return anchor !== null; },
      start() { f.reset(); t0 = null; samples = []; raw = null; wild = 0; anchor = null; me.rate = 0; me.offset = 0; },
      update(tMs, pitch) {                                                       // one camera frame -> the speed the head asks for
        if (!Number.isFinite(pitch)) return me.rate;
        if (raw !== null && Math.abs(pitch - raw) > c.maxStepDeg) { if (++wild < 2) return me.rate; }   // one wild frame: keep the last speed
        wild = 0; raw = pitch;
        const p = f.f(tMs / 1000, pitch);
        if (t0 === null) t0 = tMs;
        if (anchor === null) {
          samples.push(p);
          if (tMs - t0 < c.settleMs) return 0;
          const a = samples.slice().sort((x, y) => x - y), n = a.length; anchor = n % 2 ? a[n >> 1] : (a[n / 2 - 1] + a[n / 2]) / 2;
        }
        me.offset = p - anchor;
        const mag = Math.min(c.max, c.gain * Math.pow(Math.max(0, Math.abs(me.offset) - c.dead), c.power));
        me.rate = mag > 0 ? -Math.sign(me.offset) * mag : 0;
        return me.rate;
      } };
    me.start();
    return me;
  }
  /* The speed eases toward what the head asks for (0.15 s to speed up, 0.3 s to slow down), so it never jumps and it coasts to a
     stop when you leave the mode or lose the face. step(dtSeconds) returns how many notches to scroll in that time. */
  function createScroller(over = {}) {
    const c = { ...SCROLL, ...over }; let target = 0;
    const me = { rate: 0,
      set(r) { target = r; },
      step(dt) {
        const tau = Math.abs(target) > Math.abs(me.rate) ? c.tauUp : c.tauDown;
        me.rate += (target - me.rate) * (1 - Math.exp(-Math.max(dt, 0) / tau));
        if (Math.abs(me.rate) < 0.02 && target === 0) me.rate = 0;
        return me.rate * dt;
      } };
    return me;
  }

  if (typeof window === 'undefined' || typeof document === 'undefined') {            // node (the unit tests)
    if (typeof module !== 'undefined') module.exports = { createCore, createHeadScroll, createScroller, CFG, SCROLL, anglesFromMatrix, anglesFromLandmarks };
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
      baseOptions: { modelAssetPath: model, delegate }, runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true });
    let lm; try { lm = await make('GPU'); } catch (e) { lm = await make('CPU'); }
    return { detect: (v, t) => lm.detectForVideo(v, t), close: () => lm.close() };
  }

  /* ---------------- the panel under the camera ---------------- */
  const css = document.createElement('style');
  css.textContent = `
  .lite{margin-top:10px;display:grid;gap:8px;font-size:13px}
  .lite button{cursor:pointer}
  .lite [hidden]{display:none!important}
  .lite-main{font-weight:700}
  .lite-stat{font:600 12px var(--f-mono,monospace);color:var(--muted,#667);min-height:1.3em}
  .lite details{border:1px solid var(--line,#d8d8d8);border-radius:10px;padding:6px 10px}
  .lite summary{cursor:pointer;font-weight:600}
  .lite label{display:flex;align-items:center;gap:8px;margin:6px 0}
  .lite input[type=range]{flex:1}
  .lite select{flex:1;min-width:0;font:inherit;padding:4px 6px;border-radius:8px}
  .lite p{margin:0;color:var(--muted,#667);font-size:12px;line-height:1.4}
  .lite-cur{position:absolute;left:0;top:0;width:30px;height:30px;margin:-15px 0 0 -15px;border-radius:50%;z-index:70;pointer-events:none;
    border:3px solid #fff;box-shadow:0 0 0 2px #111,0 0 14px rgba(0,0,0,.5);background:rgba(17,17,17,.25);transition:transform .08s,border-color .1s}
  .lite-cur::after{content:"";position:absolute;inset:10px;border-radius:50%;background:#111}
  .lite-cur.shut{border-color:#F0A94B}.lite-cur.ready{border-color:#4ADE80;transform:scale(1.25)}
  .lite-cur.scrolling{border-style:dashed;border-color:#7aa7ff;opacity:.75}
  #liteCanvas{position:absolute;inset:0;width:100%;height:100%;display:block;z-index:3;border-radius:inherit}`;
  document.head.append(css);

  const camBox = $('#camBox');
  const panel = document.createElement('div');
  panel.className = 'lite';
  panel.innerHTML = `
    <button class="mini-btn lite-main" id="liteStartBtn" type="button">Start hands-free</button>
    <button class="mini-btn" id="liteBtn" type="button" hidden>Try hands-free in your browser (lite)</button>
    <button class="mini-btn" id="liteScrollBtn" type="button" hidden disabled title="Needs lite mode on. Chin down scrolls down, chin up scrolls up.">Scroll with my head</button>
    <div class="lite-stat" id="liteStat" aria-live="polite"></div>
    <details id="liteSet"><summary>Lite settings</summary>
      <label>Pointer <select id="liteMode"><option value="rel">Mouse-style, reaches the edges (best)</option><option value="abs">Face map: where you face is where it points</option></select></label>
      <label>Sensitivity <input type="range" id="liteSens" min="0.5" max="2.5" step="0.1" value="1"><span id="liteSensV">1.0×</span></label>
      <label><input type="checkbox" id="liteFx"> Flip left-right</label>
      <label><input type="checkbox" id="liteFy"> Flip up-down</label>
      <label><input type="checkbox" id="liteMouth"> Mouth open 0.6 s = right click</label>
      <button class="mini-btn" id="liteRecenter" type="button">Put the pointer in the middle</button>
    </details>
    <details id="liteInfo"><summary>Privacy and how it works</summary>
      <p>Runs in this tab only: your camera is not recorded or sent anywhere. Lite mode is a simulation and less accurate than the [eye]able app, which also types by voice with the tuned model.</p>
    </details>`;
  camBox.parentElement.append(panel);
  /* the privacy notes are open when the screen has room and fold away on short laptop screens, so the whole demo fits one screen;
     a person who opens or closes them is not overruled */
  { const info = panel.querySelector('#liteInfo'), roomy = matchMedia('(min-height:861px), (max-width:899px), (max-height:429px)'); let touched = false;
    info.addEventListener('toggle', () => { if (info.open !== roomy.matches) touched = true; });
    const sync = () => { if (!touched) info.open = roomy.matches; }; sync();
    (roomy.addEventListener ? roomy.addEventListener('change', sync) : roomy.addListener(sync)); }
  const el = id => document.getElementById(id);
  const setStat = t => { el('liteStat').textContent = t; };
  setStat('One button starts the camera and the voice. Then say “scroll”, “volume up” or “hi Dizzy”.');

  /* ---------------- run / stop ---------------- */
  const core = createCore();
  let on = false, starting = false, lm = null, stream = null, video = null, cursor = null, canvas = null, badge = null,
      raf = 0, guard = 0, lastVideoT = -1, lastClickAt = 0;

  /* ---------------- scroll mode: the head scrolls, the pointer stays ---------------- */
  const SCROLL_IDLE_MS = 10000;                                      // engine/config.py scroll_idle_s
  const hs = createHeadScroll(), scroller = createScroller();
  let scrolling = false, hold = null, lastFrameT = 0, lastActiveAt = 0, lastUiAt = 0;
  const scrollBtn = () => el('liteScrollBtn');
  function scrollMode(want) {                                        // returns true while head scrolling is on
    if (!want) { if (scrolling) endScroll(); return false; }
    if (!on) return false;
    if (scrolling) return true;
    scrolling = true; hs.start(); scroller.set(0); scroller.rate = 0;
    hold = { x: core.state.nx, y: core.state.ny }; lastFrameT = lastActiveAt = performance.now(); lastUiAt = 0;
    if (cursor) cursor.classList.add('scrolling');
    scrollBtn().textContent = 'Stop scrolling';
    log('sys', 'Lite: scroll mode on (chin down scrolls down, chin up scrolls up)');
    tip('Scroll mode: hold your head straight for a moment, then tilt your chin down to scroll down or up to scroll up (further = faster). Say “stop” or hold a long blink to finish.');
    return true;
  }
  function endScroll(why) {
    if (!scrolling) return;
    scrolling = false; scroller.set(0); scroller.rate = 0;
    if (hold) { core.state.nx = hold.x; core.state.ny = hold.y; core.state.needRebase = true; }       // the pointer picks up from where it stayed
    if (cursor) cursor.classList.remove('scrolling');
    scrollBtn().textContent = 'Scroll with my head';
    if (D.bridge) D.bridge.handle({ type: 'scroll', rate: 0, ready: false });
    if (why) { setStat(why); log('sys', why); }
    if (window.DZVoice) window.DZVoice.voice.leaveScroll();           // the card goes back to the pointer mode
  }
  function scrollFrame(s, o, t) {                                     // one camera frame in scroll mode
    scroller.set(o.status === 'noface' || !s ? 0 : hs.update(s.t, s.pitch));
    const dt = clamp((t - lastFrameT) / 1000, 0, 0.1); lastFrameT = t;
    const notches = scroller.step(dt);
    if (notches) { const ppn = clamp(screen.clientHeight * 0.1, 30, 90); D.scrollPx(-notches * ppn); }   // rate > 0 = up
    if (Math.abs(scroller.rate) > 0.5) lastActiveAt = t;
    if (t - lastActiveAt > SCROLL_IDLE_MS) { endScroll('Scroll mode ended: no scrolling for 10 seconds.'); return; }
    if (D.bridge && t - lastUiAt > 100) { lastUiAt = t; D.bridge.handle({ type: 'scroll', rate: +scroller.rate.toFixed(1), ready: hs.settled }); }
    const r = Math.abs(scroller.rate) > 0.5 ? (scroller.rate > 0 ? '↑ ' : '↓ ') + Math.abs(scroller.rate).toFixed(1) + ' notches/s' : 'still';
    setStat(o.status === 'noface' ? 'No face found. Scrolling has stopped.' : !hs.settled ? 'Scroll mode: hold your head straight for a moment…'
      : 'Scroll mode (' + r + '): chin down scrolls down, chin up scrolls up. Say “stop” or hold a long blink to finish.');
  }

  function sample(res, t) {
    const f = res && res.faceLandmarks && res.faceLandmarks[0];
    if (!f) return null;
    const bs = {};
    const cats = res.faceBlendshapes && res.faceBlendshapes[0] && res.faceBlendshapes[0].categories;
    if (cats) for (const c of cats) bs[c.categoryName] = c.score;
    const m = res.facialTransformationMatrixes && res.facialTransformationMatrixes[0];
    const ang = (m && anglesFromMatrix(m.data)) || anglesFromLandmarks(f[NOSE], f[EYE_A], f[EYE_B]);
    if (!ang) return null;
    return { yaw: ang.yaw, pitch: ang.pitch, blinkL: bs.eyeBlinkLeft, blinkR: bs.eyeBlinkRight, jaw: bs.jawOpen, t, f };
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
    if (scrolling) { o.x = hold.x; o.y = hold.y; core.state.nx = hold.x; core.state.ny = hold.y; }   // the pointer stays where it was
    if (o.status === 'noface') { if (scrolling) scrollFrame(s, o, t); else setStat('No face found. Move into view and face the screen.'); return; }

    const r = screen.getBoundingClientRect();
    const px = o.x * r.width, py = o.y * r.height;
    if (!G.paused && !G.scripted) { G.target.x = G.pos.x = px; G.target.y = G.pos.y = py; }
    cursor.style.transform = `translate(${px}px,${py}px)`;
    cursor.className = 'lite-cur' + (o.status === 'closed' ? (o.blinkMs >= core.cfg.blinkMinMs ? ' ready' : ' shut') : '');
    if (scrolling) {
      if (o.events.some(ev => ev.type === 'click')) { endScroll('Scroll mode ended: long blink.'); o.events.length = 0; }   // like the engine: a blink leaves scroll mode, it does not click
      else scrollFrame(s, o, t);
      o.events = o.events.filter(ev => ev.type !== 'rightclick' || !scrolling);
    }
    if (!scrolling) setStat(o.status === 'closed'
      ? (o.blinkMs >= core.cfg.blinkMinMs ? 'Eyes shut ' + (o.blinkMs / 1000).toFixed(1) + ' s: open them to click' : 'Eyes shut… keep them closed a bit longer to click')
      : 'Tracking. Turn your head to move (small turns aim, fast turns throw). Hold a long blink to click.');

    for (const ev of o.events) {
      if (ev.type === 'click') {
        if (window.DZVoice && window.DZVoice.waiting()) { log('eye', 'Lite: long blink → yes'); window.DZVoice.answer(true); continue; }   // a question or words are waiting: a blink is yes, like the engine
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
  }

  async function start() {
    if (on || starting) return;
    const pill = document.getElementById('engPill');                // the real engine is connected (D.engineLive is also set by lite mode itself)
    if (pill && !pill.classList.contains('off')) return toast('[eye]able is running on this PC and already controls this page. Lite mode is not needed.');
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
    el('liteBtn').disabled = false; el('liteBtn').textContent = 'Stop lite mode'; scrollBtn().disabled = false;
    log('sys', 'Lite mode on: the head moves the pointer, a long blink clicks (simulation, runs in this tab)');
    tip('Lite mode: turn your head to move the pointer like a mouse (small turns aim, quick turns throw it). Hold a long blink, about half a second, to click.');
    guard = setInterval(() => { const p = document.getElementById('engPill'); if (p && !p.classList.contains('off')) stop('[eye]able connected: using the real engine instead.', true); }, 1000);
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
    if (scrolling) endScroll();
    on = false; cancelAnimationFrame(raf); clearInterval(guard); cleanup(); scrollBtn().disabled = true;
    if (!engineTook) D.engineLive = false;
    el('liteBtn').textContent = 'Try hands-free in your browser (lite)';
    $('#camSrc').textContent = 'source: simulated face';
    setStat(why || 'Lite mode off.'); log('sys', why || 'Lite mode off');
  }

  el('liteBtn').addEventListener('click', () => (on ? stop() : start()));
  el('liteRecenter').addEventListener('click', () => { core.recenter(); setStat('Pointer in the middle.'); });
  el('liteMode').addEventListener('change', e => { core.set('mode', e.target.value); core.recenter(); setStat(e.target.value === 'abs' ? 'Face map: look at the middle of the screen, hold still a moment, then point with your face.' : 'Mouse-style pointer.'); });
  el('liteSens').addEventListener('input', e => { core.set('sens', +e.target.value); el('liteSensV').textContent = (+e.target.value).toFixed(1) + '×'; });
  el('liteFx').addEventListener('change', e => core.set('flipX', e.target.checked));
  el('liteFy').addEventListener('change', e => core.set('flipY', e.target.checked));
  el('liteMouth').addEventListener('change', e => core.set('mouthClick', e.target.checked));
  scrollBtn().addEventListener('click', () => {
    const V = window.DZVoice && window.DZVoice.voice;
    if (scrolling) { if (V) V.leaveScroll('Scroll off'); else endScroll(); } else if (V) V.enterScroll(); else scrollMode(true);
  });
  /* the one button: camera (head pointer, blink click), microphone (commands, dictation) and head scrolling ("scroll") start and stop together */
  const mainBtn = el('liteStartBtn');
  const voiceOn = () => !!(window.DZVoice && window.DZVoice.isOn && window.DZVoice.isOn());
  function syncMain() {
    const busy = starting && !on;
    mainBtn.textContent = busy ? 'Starting…' : (on || voiceOn()) ? 'Stop' : 'Start hands-free';
    mainBtn.disabled = busy; mainBtn.setAttribute('aria-pressed', String(on || voiceOn()));
  }
  mainBtn.addEventListener('click', () => {
    if (on || voiceOn()) { if (window.DZVoice) window.DZVoice.stop(); stop(); }
    else { if (window.DZVoice && window.DZVoice.start) window.DZVoice.start(); start(); }       // both inside this click, so the browser asks for camera and microphone together
    syncMain();
  });
  setInterval(syncMain, 250); syncMain();
  window.DZLite = { start, stop, core, scroll: scrollMode, isScrolling: () => scrolling, isOn: () => on };
})();

/* js/gesture_detect.js: turns face signals into gestures, in the browser. Pure logic, no DOM, so it can be tested in Node.
   Same ideas as the engine's detectors, with the same numbers where the engine has them:
     blink       eyes shut 0.4 to 1.2 s (a quick natural blink is ignored)
     brow_raise  eyebrow blendshape 0.25 above its resting level for 250 ms, 0.8 s cooldown
     nod         head pitch swings away from neutral and comes back, no sideways swing
     shake       head yaw crosses both sides of neutral
     tilt_left / tilt_right   head roll held past 15 degrees for 0.35 s
   Feed update() one sample per camera frame:
     { blink: 0..1 (both eyes), brow: 0..1, pitch, yaw, roll (degrees) }, or null when no face is seen.
   It returns the list of gestures that just happened. */
(function (root) {
  'use strict';
  const CFG = {
    blinkClosed: 0.55, blinkMinMs: 400, blinkMaxMs: 1200,
    browDelta: 0.25, browHoldMs: 250, browRelease: 0.7, browCooldownMs: 800, browRestMs: 2500,
    nodPeakDeg: 9, nodBackFrac: 0.4, nodWindowMs: 900, nodSideMaxDeg: 12,
    shakeSwingDeg: 9, shakeWindowMs: 1200, shakeVertMaxDeg: 12,
    tiltDeg: 15, tiltRearmDeg: 8, tiltHoldMs: 350, tiltSideMaxDeg: 15,
    headCooldownMs: 800, neutralTauMs: 2000, neutralMaxSpeed: 15,   // neutral follows the head only while it is still
    rollSign: 1,                                                     // flip to -1 if left and right tilts come out swapped
  };

  function median(a) {
    const s = a.slice().sort((x, y) => x - y), n = s.length;
    return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : 0;
  }

  class GestureDetector {
    constructor(cfg) { this.c = Object.assign({}, CFG, cfg); this.reset(); }

    reset() {
      this.closedSince = null;
      this.browHist = []; this.browSince = null; this.browArmed = true; this.browLast = -1e9;
      this.neutral = null; this.last = null; this.win = []; this.headLast = -1e9;
      this.tiltSince = null; this.tiltArmed = true;
      this.dev = { pitch: 0, yaw: 0, roll: 0 };
    }

    update(t, s) {
      if (!s) { this.reset(); return []; }
      const c = this.c, out = [];
      const closed = s.blink >= c.blinkClosed;

      // --- long blink: counted when the eyes open again
      if (closed && this.closedSince === null) this.closedSince = t;
      if (!closed && this.closedSince !== null) {
        const d = t - this.closedSince; this.closedSince = null;
        if (d >= c.blinkMinMs && d <= c.blinkMaxMs) out.push('blink');
      }

      // --- eyebrow raise: above its own resting level (median of the last few seconds, eyes open only)
      if (!closed) {
        this.browHist.push([t, s.brow]);
        while (this.browHist.length && t - this.browHist[0][0] > c.browRestMs) this.browHist.shift();
      }
      const rest = median(this.browHist.map(h => h[1])), up = s.brow - rest;
      if (!closed && this.browHist.length > 8) {
        if (up >= c.browDelta) {
          if (this.browSince === null) this.browSince = t;
          if (this.browArmed && t - this.browSince >= c.browHoldMs && t - this.browLast >= c.browCooldownMs) {
            out.push('brow_raise'); this.browLast = t; this.browArmed = false;
          }
        } else {
          this.browSince = null;
          if (up < c.browDelta * c.browRelease) this.browArmed = true;
        }
      }

      // --- head: motion away from a slowly following neutral
      if (this.neutral === null) this.neutral = { pitch: s.pitch, yaw: s.yaw, roll: s.roll };
      if (this.last) {
        const dt = Math.max(1, t - this.last.t);
        const speed = Math.max(Math.abs(s.pitch - this.last.pitch), Math.abs(s.yaw - this.last.yaw), Math.abs(s.roll - this.last.roll)) / dt * 1000;
        if (speed < c.neutralMaxSpeed) {
          const a = 1 - Math.exp(-dt / c.neutralTauMs);
          for (const k of ['pitch', 'yaw', 'roll']) this.neutral[k] += (s[k] - this.neutral[k]) * a;
        }
      }
      this.last = { t, pitch: s.pitch, yaw: s.yaw, roll: s.roll };
      const dev = this.dev = { pitch: s.pitch - this.neutral.pitch, yaw: s.yaw - this.neutral.yaw, roll: s.roll - this.neutral.roll };

      // tilt: held roll
      const rolled = Math.abs(dev.roll) >= c.tiltDeg && Math.abs(dev.yaw) < c.tiltSideMaxDeg;
      if (rolled) {
        if (this.tiltSince === null) this.tiltSince = t;
        if (this.tiltArmed && t - this.tiltSince >= c.tiltHoldMs) {
          out.push(dev.roll * c.rollSign < 0 ? 'tilt_left' : 'tilt_right'); this.tiltArmed = false; this.win = [];
        }
      } else {
        this.tiltSince = null;
        if (Math.abs(dev.roll) < c.tiltRearmDeg) this.tiltArmed = true;
      }

      // nod / shake: a short window of recent deviations
      this.win.push([t, dev.pitch, dev.yaw]);
      while (this.win.length && t - this.win[0][0] > Math.max(c.nodWindowMs, c.shakeWindowMs)) this.win.shift();
      if (!rolled && Math.abs(dev.roll) < c.tiltRearmDeg + 4 && t - this.headLast >= c.headCooldownMs) {
        const nodWin = this.win.filter(w => t - w[0] <= c.nodWindowMs);
        let peak = 0, tPeak = t, yMin = 1e9, yMax = -1e9;
        for (const w of nodWin) { if (Math.abs(w[1]) > peak) { peak = Math.abs(w[1]); tPeak = w[0]; } yMin = Math.min(yMin, w[2]); yMax = Math.max(yMax, w[2]); }
        const shakeWin = this.win;
        let pMin = 1e9, pMax = -1e9, yLo = 1e9, yHi = -1e9;
        for (const w of shakeWin) { pMin = Math.min(pMin, w[1]); pMax = Math.max(pMax, w[1]); yLo = Math.min(yLo, w[2]); yHi = Math.max(yHi, w[2]); }
        if (yHi >= c.shakeSwingDeg && yLo <= -c.shakeSwingDeg && pMax - pMin < c.shakeVertMaxDeg * 2 && Math.abs(dev.yaw) < c.shakeSwingDeg) {
          out.push('shake'); this.headLast = t; this.win = [];
        } else if (peak >= c.nodPeakDeg && Math.abs(dev.pitch) <= peak * c.nodBackFrac && t - tPeak >= 80 && yMax - yMin < c.nodSideMaxDeg * 2) {
          out.push('nod'); this.headLast = t; this.win = [];
        }
      }
      return out;
    }
  }

  /* head pose in degrees from MediaPipe's 4x4 facial transformation matrix (column-major, 16 numbers) */
  function poseFromMatrix(m) {
    const col = i => { const x = m[i * 4], y = m[i * 4 + 1], z = m[i * 4 + 2], n = Math.hypot(x, y, z) || 1; return [x / n, y / n, z / n]; };
    const c0 = col(0), c1 = col(1), c2 = col(2);
    const r00 = c0[0], r10 = c0[1], r20 = c0[2], r21 = c1[2], r22 = c2[2];
    const d = 180 / Math.PI;
    return { pitch: Math.atan2(r21, r22) * d, yaw: Math.atan2(-r20, Math.hypot(r00, r10)) * d, roll: Math.atan2(r10, r00) * d };
  }

  const api = { GestureDetector, poseFromMatrix, CFG };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.DiseyebleGestures = api;
})(typeof window !== 'undefined' ? window : globalThis);

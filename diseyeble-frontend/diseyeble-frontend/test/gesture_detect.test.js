// Run: node test/gesture_detect.test.js   (synthetic face signals through js/gesture_detect.js)
const assert = require('assert');
const { GestureDetector, poseFromMatrix } = require('../js/gesture_detect.js');

const DT = 33;                                   // about 30 frames a second
function run(seconds, fn, det = new GestureDetector(), t0 = 0) {
  const events = [];
  for (let t = 0; t <= seconds * 1000; t += DT) {
    const s = Object.assign({ blink: 0.1, brow: 0.2, pitch: 0, yaw: 0, roll: 0 }, fn(t));
    for (const e of det.update(t0 + t, s)) events.push([t0 + t, e]);
  }
  return events.map(e => e[1]);
}
const still = () => ({});
const settle = det => { run(3, still, det); return det; };
function after(fn, secs = 3) {                   // 3 s of rest, then the movement
  const det = new GestureDetector(), ev = [];
  for (let t = 0; t <= (3 + secs) * 1000; t += DT) {
    const s = Object.assign({ blink: 0.1, brow: 0.2, pitch: 0, yaw: 0, roll: 0 }, t >= 3000 ? fn(t - 3000) : {});
    for (const e of det.update(t, s)) ev.push(e);
  }
  return ev;
}
const tri = (t, a, b, h) => t < a || t > b ? 0 : h * Math.sin((t - a) / (b - a) * Math.PI);   // up and back

let n = 0; const ok = (name, got, want) => { assert.deepStrictEqual(got, want, `${name}: got ${JSON.stringify(got)}`); n++; console.log('ok  ', name); };

ok('rest gives nothing', after(still), []);
ok('natural blink (150 ms) ignored', after(t => ({ blink: t > 500 && t < 650 ? 0.9 : 0.1 })), []);
ok('long blink (600 ms)', after(t => ({ blink: t > 500 && t < 1100 ? 0.9 : 0.1 })), ['blink']);
ok('eyes shut 2 s is not a click', after(t => ({ blink: t > 500 && t < 2500 ? 0.9 : 0.1 }), 4), []);
ok('eyebrow raise', after(t => ({ brow: t > 500 && t < 1100 ? 0.65 : 0.2 })), ['brow_raise']);
ok('short brow twitch (150 ms) ignored', after(t => ({ brow: t > 500 && t < 650 ? 0.65 : 0.2 })), []);
ok('brow raised for 3 s fires once', after(t => ({ brow: t > 500 ? 0.65 : 0.2 }), 4), ['brow_raise']);
ok('nod', after(t => ({ pitch: tri(t, 300, 900, 14) })), ['nod']);
ok('nod the other way', after(t => ({ pitch: tri(t, 300, 900, -14) })), ['nod']);
ok('shake', after(t => ({ yaw: t < 300 ? 0 : t < 1100 ? 14 * Math.sin((t - 300) / 800 * Math.PI * 2) : 0 })), ['shake']);
ok('tilt held (roll +20)', after(t => ({ roll: t > 300 ? 20 : 0 }), 2), ['tilt_right']);
ok('tilt held (roll -20)', after(t => ({ roll: t > 300 ? -20 : 0 }), 2), ['tilt_left']);
ok('quick roll wobble ignored', after(t => ({ roll: tri(t, 300, 500, 20) })), []);
ok('slow turn to look at the screen edge ignored', after(t => ({ yaw: Math.min(20, t / 100) }), 4), []);
ok('small talking jitter ignored', after(t => ({ pitch: 2 * Math.sin(t / 90), yaw: 1.5 * Math.sin(t / 130), brow: 0.2 + 0.05 * Math.sin(t / 70) }), 5), []);
ok('face lost resets', (() => { const d = new GestureDetector(); settle(d); d.update(3100, null); return d.update(3200, { blink: 0.9, brow: 0.2, pitch: 0, yaw: 0, roll: 0 }); })(), []);

// pose: a pure rotation about each axis
const rot = (ax, deg) => { const a = deg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
  const R = ax === 'x' ? [1, 0, 0, 0, c, -s, 0, s, c] : ax === 'y' ? [c, 0, s, 0, 1, 0, -s, 0, c] : [c, -s, 0, s, c, 0, 0, 0, 1];
  return [R[0], R[3], R[6], 0, R[1], R[4], R[7], 0, R[2], R[5], R[8], 0, 0, 0, 0, 1].map(v => v * 1);   // column-major
};
const near = (a, b) => Math.abs(a - b) < 0.5;
let p = poseFromMatrix(rot('x', 20)); assert(near(p.pitch, 20) && near(p.yaw, 0) && near(p.roll, 0)); n++;
p = poseFromMatrix(rot('y', 25)); assert(near(p.yaw, 25) && near(p.pitch, 0)); n++;
p = poseFromMatrix(rot('z', -15)); assert(near(p.roll, -15)); n++;
const scaled = rot('z', 10).map((v, i) => (i % 4 < 3 && i < 12 ? v * 1.7 : v));
assert(near(poseFromMatrix(scaled).roll, 10)); n++;
console.log(`ok   pose from matrix (4 checks)\n${n} checks passed`);

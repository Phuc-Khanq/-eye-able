// Unit tests for the lite-mode head pointer and blink click (eyeable-frontend/eyeable-frontend/js/lite.js).
// Run: node tests/lite_core.test.js
const assert = require('assert');
const { createCore, anglesFromMatrix } = require('../eyeable-frontend/eyeable-frontend/js/lite.js');

let T = 0;
const face = (t, o = {}) => ({ yaw: o.yaw || 0, pitch: o.pitch || 0, blinkL: o.b || 0, blinkR: o.b || 0, jaw: o.jaw || 0, t });
const run = (c, n, o, dt = 33) => { let r; for (let i = 0; i < n; i++) { T += dt; r = c.step(face(T, o)); } return r; };
// sweep the head by `deg` in n frames (yaw and/or pitch), then hold
const sweep = (c, n, dyaw, dpitch, dt = 33) => { let r; const base = c.state.raw || { yaw: 0, pitch: 0 }; let y = base.yaw, p = base.pitch;
  for (let i = 0; i < n; i++) { T += dt; y += dyaw; p += dpitch; r = c.step(face(T, { yaw: y, pitch: p })); } return { r, yaw: y, pitch: p }; };
const ready = () => { const c = createCore(); run(c, 30, {}); return c; };
const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test('no face gives status noface', () => { assert.strictEqual(createCore().step(null).status, 'noface'); });
test('a still head leaves the pointer in the middle (wobble is gated out)', () => {
  const c = ready(); let r; for (let i = 0; i < 90; i++) { T += 33; r = c.step(face(T, { yaw: (i % 2 ? 0.05 : -0.05), pitch: (i % 3 ? 0.08 : -0.08) })); }
  assert(Math.abs(r.x - 0.5) < 0.01 && Math.abs(r.y - 0.5) < 0.01, `${r.x},${r.y}`);
});
test('a quick turn right throws the pointer right; a quick turn left throws it left', () => {
  const a = sweep(ready(), 8, 1.5, 0).r; assert(a.x > 0.6, 'x=' + a.x);
  const b = sweep(ready(), 8, -1.5, 0).r; assert(b.x < 0.4, 'x=' + b.x);
});
test('chin down moves the pointer down, chin up moves it up', () => {
  assert(sweep(ready(), 8, 0, 1.2).r.y > 0.55); assert(sweep(ready(), 8, 0, -1.2).r.y < 0.45);
});
test('slow turns aim finely: the same angle moves less when slow than fast', () => {
  const fast = sweep(ready(), 8, 1.5, 0).r.x - 0.5, slow = sweep(ready(), 60, 0.2, 0).r.x - 0.5;   // both 12 degrees
  assert(fast > slow * 2, `fast ${fast} slow ${slow}`); assert(slow > 0.01, 'slow still moves');
});
test('the pointer stays on the screen', () => { const r = sweep(ready(), 80, 1.5, 1.5).r; assert(r.x <= 1 && r.y <= 1 && r.x >= 0 && r.y >= 0); });
test('flip and sensitivity change the movement', () => {
  const c = ready(); c.set('flipX', true); assert(sweep(c, 8, 1.5, 0).r.x < 0.4);
  const a = ready(), b = ready(); b.set('sens', 2); assert(sweep(b, 8, 1, 0).r.x > sweep(a, 8, 1, 0).r.x);
});
test('a one-frame tracking glitch is skipped', () => {
  const c = ready(); T += 33; c.step(face(T, { yaw: 40 })); const r = run(c, 5, {}); assert(Math.abs(r.x - 0.5) < 0.02, 'x=' + r.x);
});
test('recenter puts the pointer back in the middle', () => { const c = ready(); sweep(c, 8, 1.5, 0); c.recenter(); assert(Math.abs(run(c, 3, {}).x - 0.5) < 0.01); });
test('a quick natural blink (150 ms) does not click', () => {
  const c = ready(); const ev = [...run(c, 5, { b: 0.9 }, 30).events, ...run(c, 10, {}).events]; assert.strictEqual(ev.length, 0);
});
test('a long blink (700 ms) clicks once, on reopening', () => {
  const c = ready(); const during = run(c, 21, { b: 0.9 }, 33); assert.strictEqual(during.events.length, 0); assert.strictEqual(during.status, 'closed');
  const after = run(c, 1, {}); assert.strictEqual(after.events.length, 1); assert.strictEqual(after.events[0].type, 'click');
  assert.strictEqual(run(c, 20, {}).events.length, 0);
});
test('eyes shut for 2 s (resting) does not click, and a wink does not either', () => {
  const c = ready(); run(c, 60, { b: 0.9 }, 33); assert.strictEqual(run(c, 1, {}).events.length, 0);
  const d = ready(); let n = 0; for (let i = 0; i < 25; i++) { T += 33; n += d.step({ ...face(T), blinkL: 0.9, blinkR: 0.05 }).events.length; } assert.strictEqual(n, 0);
});
test('blinking does not move the pointer: the click lands where it was, and the head angle change while the eyes are shut is ignored', () => {
  const c = ready(); const s = sweep(c, 8, 1.5, 0); const before = run(c, 12, { yaw: s.yaw, pitch: s.pitch }).x;
  // the eyelids drop and the estimated pitch/yaw jump by a few degrees while the eyes are shut and after
  const during = run(c, 3, { yaw: s.yaw + 2, pitch: s.pitch + 3, b: 0.9 }); assert(Math.abs(during.x - before) < 0.03, 'during');
  run(c, 20, { yaw: s.yaw + 2, pitch: s.pitch + 3, b: 0.9 });
  const open = run(c, 1, { yaw: s.yaw + 2, pitch: s.pitch + 3 }); assert.strictEqual(open.events.length, 1);
  const after = run(c, 25, { yaw: s.yaw + 2, pitch: s.pitch + 3 }); assert(Math.abs(after.x - before) < 0.03, `before ${before} after ${after.x}`);
});
test('edge help: a slow turn of about 19 degrees reaches the side edge, which the mouse-style pointer alone would not', () => {
  const a = ready(); const slow = sweep(a, 130, 0.15, 0).r;                 // 19.5 degrees at 4.5 deg/s
  assert(slow.x > 0.93, 'with help x=' + slow.x);
  const b = createCore({ edgeYaw: 1e9, edgePitch: 1e9 }); run(b, 30, {}); const plain = sweep(b, 130, 0.15, 0).r;
  assert(plain.x < 0.75, 'without help x=' + plain.x);
});
test('edge help does nothing in the middle of the range, and never pulls back toward the middle', () => {
  const c = ready(); const r = sweep(c, 40, 0.15, 0).r; assert(r.x < 0.6, 'x=' + r.x);   // 6 degrees
  const d = ready(); sweep(d, 130, 0.15, 0); const edge = d.state.nx; const back = sweep(d, 20, -0.15, 0).r; assert(back.x <= edge);
});
test('edge help also reaches the bottom and the top', () => {
  assert(sweep(ready(), 100, 0, 0.12).r.y > 0.93); assert(sweep(ready(), 100, 0, -0.12).r.y < 0.07);
});
test('face map mode: angle from straight ahead is the position (15 degrees = side edge, 10 = top or bottom)', () => {
  const c = createCore(); c.set('mode', 'abs'); run(c, 30, {});
  assert(Math.abs(run(c, 20, {}).x - 0.5) < 0.01);
  assert(run(c, 40, { yaw: 15 }).x > 0.97); assert(run(c, 40, { yaw: -15 }).x < 0.03);
  assert(Math.abs(run(c, 40, { yaw: 7.5 }).x - 0.75) < 0.03); assert(run(c, 40, { pitch: 10 }).y > 0.97);
});
test('face map mode still ignores the head angle during a blink and clicks where it was', () => {
  const c = createCore(); c.set('mode', 'abs'); run(c, 30, {}); const before = run(c, 40, { yaw: 7.5 }).x;
  const during = run(c, 3, { yaw: 10, b: 0.9 }); assert(Math.abs(during.x - before) < 0.03);
  run(c, 20, { yaw: 10, b: 0.9 }); assert.strictEqual(run(c, 1, { yaw: 10 }).events.length, 1);
});
test('a blink up to 1.6 s still clicks (the engine\'s window)', () => {
  const c = ready(); run(c, 45, { b: 0.9 }, 33); assert.strictEqual(run(c, 1, {}).events.length, 1);   // 1.5 s
});
test('mouth right click is off by default and works when switched on', () => {
  assert.strictEqual(run(ready(), 30, { jaw: 0.9 }).events.length, 0);
  const b = ready(); b.set('mouthClick', true); let n = 0; for (let i = 0; i < 30; i++) { T += 33; n += b.step(face(T, { jaw: 0.9 })).events.length; } assert.strictEqual(n, 1);
});
test('head angles from the face matrix: yaw > 0 turned right, pitch > 0 chin down (same maths as engine/eyes/head_pose.py)', () => {
  // expected values from scipy: Rotation.from_euler("xyz", [rx, ry, rz]) as extrinsic xyz, yaw = -ry, pitch = rx
  const rot = (rx, ry, rz) => { const [a, b, g] = [rx, ry, rz].map(d => d * Math.PI / 180);
    const Rx = [[1,0,0],[0,Math.cos(a),-Math.sin(a)],[0,Math.sin(a),Math.cos(a)]], Ry = [[Math.cos(b),0,Math.sin(b)],[0,1,0],[-Math.sin(b),0,Math.cos(b)]], Rz = [[Math.cos(g),-Math.sin(g),0],[Math.sin(g),Math.cos(g),0],[0,0,1]];
    const mul = (A, B) => A.map((r, i) => B[0].map((_, j) => r.reduce((s, _, k) => s + A[i][k] * B[k][j], 0)));
    return mul(Rz, mul(Ry, Rx)); };
  const pack = (R, colMajor, scale = 1) => { const M = [[R[0][0]*scale,R[0][1]*scale,R[0][2]*scale,1.5],[R[1][0]*scale,R[1][1]*scale,R[1][2]*scale,-2],[R[2][0]*scale,R[2][1]*scale,R[2][2]*scale,-40],[0,0,0,1]];
    const d = []; for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) d[colMajor ? j * 4 + i : i * 4 + j] = M[i][j]; return d; };
  for (const colMajor of [true, false]) for (const [rx, ry, rz] of [[10, -20, 5], [-15, 25, -8], [0, 0, 0]]) {
    const a = anglesFromMatrix(pack(rot(rx, ry, rz), colMajor, 1.3));
    assert(Math.abs(a.pitch - rx) < 1e-6 && Math.abs(a.yaw + ry) < 1e-6, `${colMajor} ${rx},${ry}: ${JSON.stringify(a)}`);
  }
  assert.strictEqual(anglesFromMatrix(null), null);
});

let failed = 0;
for (const [name, fn] of tests) { try { fn(); console.log('ok   ' + name); } catch (e) { failed++; console.log('FAIL ' + name + ' :: ' + e.message); } }
console.log(`${tests.length - failed}/${tests.length} passed`); process.exit(failed ? 1 : 0);

// Unit tests for the lite-mode tracking maths (diseyeble-frontend/diseyeble-frontend/js/lite.js). Run: node tests/lite_core.test.js
const assert = require('assert');
const { createCore } = require('../diseyeble-frontend/diseyeble-frontend/js/lite.js');

const face = (t, o = {}) => ({ nose: { x: 0.5 + (o.nx || 0), y: 0.5 + (o.ny || 0) }, eyeA: { x: 0.4, y: 0.4 }, eyeB: { x: 0.6, y: 0.4 },
  blinkL: o.b || 0, blinkR: o.b || 0, jaw: o.jaw || 0, t });
let T = 0;
const run = (core, n, o, dt = 33) => { let r; for (let i = 0; i < n; i++) { T += dt; r = core.step(face(T, o)); } return r; };
const ready = () => { const c = createCore(); run(c, 25, {}); return c; };
const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test('no face gives status noface', () => { assert.strictEqual(createCore().step(null).status, 'noface'); });
test('calibrates first, then tracks', () => {
  const c = createCore(); assert.strictEqual(run(c, 5, {}).status, 'calibrating');
  assert.strictEqual(run(c, 20, {}).status, 'tracking');
});
test('centred head puts the pointer in the middle', () => {
  const r = run(ready(), 30, {}); assert(Math.abs(r.x - 0.5) < 0.01 && Math.abs(r.y - 0.5) < 0.01);
});
test('turning right moves the pointer right (nose goes to the image left)', () => {
  const r = run(ready(), 60, { nx: -0.04 }); assert(r.x > 0.7, 'x=' + r.x);
});
test('turning left moves the pointer left', () => { const r = run(ready(), 60, { nx: 0.04 }); assert(r.x < 0.3, 'x=' + r.x); });
test('nodding down moves the pointer down, up moves it up', () => {
  assert(run(ready(), 60, { ny: 0.02 }).y > 0.7); assert(run(ready(), 60, { ny: -0.02 }).y < 0.3);
});
test('flip switches the direction', () => { const c = ready(); c.set('flipX', true); assert(run(c, 60, { nx: -0.04 }).x < 0.3); });
test('the pointer stays inside the screen', () => { const r = run(ready(), 80, { nx: -0.5, ny: 0.5 }); assert(r.x <= 1 && r.y <= 1 && r.x >= 0 && r.y >= 0); });
test('higher sensitivity reaches further', () => {
  const a = ready(), b = ready(); b.set('sens', 2);
  assert(run(b, 60, { nx: -0.02 }).x > run(a, 60, { nx: -0.02 }).x);
});
test('a quick natural blink (150 ms) does not click', () => {
  const c = ready(); const ev = [];
  [run(c, 5, { b: 0.9 }, 30), run(c, 10, {})].forEach(r => ev.push(...r.events)); assert.strictEqual(ev.length, 0);
});
test('a long blink (700 ms) clicks once, on reopening', () => {
  const c = ready(); const during = run(c, 21, { b: 0.9 }, 33); assert.strictEqual(during.events.length, 0); assert(during.status === 'closed');
  const after = run(c, 1, {}); assert.strictEqual(after.events.length, 1); assert.strictEqual(after.events[0].type, 'click');
  assert.strictEqual(run(c, 20, {}).events.length, 0);
});
test('eyes shut for 2 s (resting) does not click', () => {
  const c = ready(); run(c, 60, { b: 0.9 }, 33); assert.strictEqual(run(c, 1, {}).events.length, 0);
});
test('a wink (one eye) does not click', () => {
  const c = ready(); const f = t => ({ ...face(t), blinkL: 0.9, blinkR: 0.05 }); let n = 0;
  for (let i = 0; i < 25; i++) { T += 33; n += c.step(f(T)).events.length; } n += run(c, 3, {}).events.length; assert.strictEqual(n, 0);
});
test('the click lands where the pointer was before the blink, not where the lids dragged it', () => {
  const c = ready(); run(c, 40, { nx: -0.03 }); const before = run(c, 1, { nx: -0.03 });
  const c1 = run(c, 3, { nx: 0.02, b: 0.9 }); // the nose jitters while the eyes shut
  assert(Math.abs(c1.x - before.x) < 0.1, `before ${before.x} after ${c1.x}`);
  run(c, 20, { nx: 0.02, b: 0.9 }, 33); const r = run(c, 1, { nx: 0.02 });
  assert.strictEqual(r.events.length, 1); assert(Math.abs(r.x - before.x) < 0.1);
});
test('mouth right click is off by default and works when switched on', () => {
  const a = ready(); assert.strictEqual(run(a, 30, { jaw: 0.9 }).events.length + 0, 0);
  const b = ready(); b.set('mouthClick', true); let n = 0; for (let i = 0; i < 30; i++) { T += 33; n += b.step(face(T, { jaw: 0.9 })).events.length; }
  assert.strictEqual(n, 1);
  run(b, 5, {}); n = 0; for (let i = 0; i < 30; i++) { T += 33; n += b.step(face(T, { jaw: 0.9 })).events.length; } assert.strictEqual(n, 1);
});
test('recenter makes the current pose the middle', () => {
  const c = ready(); run(c, 40, { nx: -0.04 }); c.recenter(); run(c, 25, { nx: -0.04 });
  const r = run(c, 40, { nx: -0.04 }); assert(Math.abs(r.x - 0.5) < 0.05, 'x=' + r.x);
});

let failed = 0;
for (const [name, fn] of tests) { try { fn(); console.log('ok   ' + name); } catch (e) { failed++; console.log('FAIL ' + name + ' :: ' + e.message); } }
console.log(`${tests.length - failed}/${tests.length} passed`); process.exit(failed ? 1 : 0);

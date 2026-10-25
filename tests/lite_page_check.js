// Headless check of the lite-mode page glue with a stand-in tracker (no CDN needed). Run: node tests/lite_page_check.js
// Needs playwright (npm i playwright) and a Chromium. Serves eyeable-frontend/eyeable-frontend on localhost.
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const root = path.join(__dirname, '..', 'eyeable-frontend', 'eyeable-frontend');
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.jpg': 'image/jpeg', '.png': 'image/png' };
const srv = http.createServer((q, r) => { const f = path.join(root, decodeURIComponent(q.url.split('?')[0]).replace(/\/$/, '/index.html'));
  fs.readFile(f, (e, d) => { if (e) { r.writeHead(404); r.end(); } else { r.writeHead(200, { 'content-type': types[path.extname(f)] || 'application/octet-stream' }); r.end(d); } }); });
(async () => {
  await new Promise(r => srv.listen(8099, r));
  const b = await chromium.launch({ args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'] });
  const ctx = await b.newContext({ permissions: ['camera'], viewport: { width: 1500, height: 1000 } });
  const p = await ctx.newPage(); const errs = [];
  p.on('pageerror', e => errs.push(e.message)); p.on('console', m => { if (m.type() === 'error' && !/WebSocket|ERR_TUNNEL|ERR_CONNECTION/.test(m.text())) errs.push(m.text()); });  // no engine and no internet here are normal
  await p.addInitScript(() => {
    window.__face = { nx: 0, ny: 0, b: 0, jaw: 0, none: false };
    window.__DZ_LITE_VISION = async () => ({ close() {}, detect() {
      const s = window.__face; if (s.none) return { faceLandmarks: [] };
      const lm = []; lm[1] = { x: 0.5 + s.nx, y: 0.5 + s.ny }; lm[33] = { x: 0.4, y: 0.4 }; lm[263] = { x: 0.6, y: 0.4 };
      return { faceLandmarks: [lm], faceBlendshapes: [{ categories: [{ categoryName: 'eyeBlinkLeft', score: s.b }, { categoryName: 'eyeBlinkRight', score: s.b }, { categoryName: 'jawOpen', score: s.jaw }] }] };
    } });
  });
  await p.goto('http://localhost:8099/#demo'); await p.waitForSelector('#liteBtn');
  await p.evaluate(() => { window.__clicks = []; const o = window.DZ.clickAt; window.DZ.clickAt = (...a) => { window.__clicks.push(a.slice(2)); return o(...a); }; });
  await p.click('#liteBtn');
  await p.waitForFunction(() => document.querySelector('#liteStat').textContent.startsWith('Tracking'), null, { timeout: 15000 });
  const pos = () => p.evaluate(() => ({ x: window.DZ.G.pos.x, y: window.DZ.G.pos.y, w: window.DZ.screen.clientWidth, h: window.DZ.screen.clientHeight }));
  const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) process.exitCode = 1; };
  const c0 = await pos(); ok(Math.abs(c0.x / c0.w - 0.5) < 0.05, 'pointer starts in the middle');
  await p.evaluate(() => { const f = window.__face; let n = 0; const id = setInterval(() => { f.nx -= 0.012; if (++n >= 6) clearInterval(id); }, 33); }); await p.waitForTimeout(900);  // a quick turn to the right
  const r = await pos(); ok(r.x / r.w > 0.6, 'turning right moves the pointer right (' + (r.x / r.w).toFixed(2) + ')');
  ok(await p.evaluate(() => !!document.querySelector('.lite-cur') && !!document.querySelector('.sim-badge') && document.querySelector('#liteCanvas')), 'cursor, badge and camera preview are shown');
  await p.evaluate(() => { window.__face.b = 0.9; }); await p.waitForTimeout(800); await p.evaluate(() => { window.__face.b = 0; }); await p.waitForTimeout(300);
  const clicks = await p.evaluate(() => window.__clicks); ok(clicks.length === 1 && clicks[0][0] === 'left', 'a long blink made one left click: ' + JSON.stringify(clicks));
  await p.evaluate(() => { window.__face.b = 0.9; }); await p.waitForTimeout(150); await p.evaluate(() => { window.__face.b = 0; }); await p.waitForTimeout(300);
  ok((await p.evaluate(() => window.__clicks.length)) === 1, 'a quick blink did not click');
  await p.evaluate(() => { window.__face.none = true; }); await p.waitForTimeout(300);
  ok((await p.textContent('#liteStat')).startsWith('No face'), 'no face is reported');
  await p.click('#liteBtn'); await p.waitForTimeout(200);
  ok(await p.evaluate(() => !document.querySelector('.lite-cur') && !document.querySelector('#liteCanvas')), 'stopping removes the cursor and preview');
  ok(errs.length === 0, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
  await p.screenshot({ path: process.env.SHOT || '/tmp/lite.png' });
  await b.close(); srv.close();
})().catch(e => { console.error(e); process.exit(1); });

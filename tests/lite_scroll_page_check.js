// Headless check of head scrolling in lite mode with a stand-in tracker and speech recogniser.
// Run: node tests/lite_scroll_page_check.js   (W= and H= set the window size)
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const root = path.join(__dirname, '..', 'eyeable-frontend', 'eyeable-frontend');
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.jpg': 'image/jpeg', '.png': 'image/png', '.svg': 'image/svg+xml', '.webp': 'image/webp' };
const srv = http.createServer((q, r) => { const f = path.join(root, decodeURIComponent(q.url.split('?')[0]).replace(/\/$/, '/index.html'));
  fs.readFile(f, (e, d) => { if (e) { r.writeHead(404); r.end(); } else { r.writeHead(200, { 'content-type': types[path.extname(f)] || 'application/octet-stream' }); r.end(d); } }); });
(async () => {
  await new Promise(r => srv.listen(8091, r));
  const b = await chromium.launch({ args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'] });
  const p = await (await b.newContext({ permissions: ['camera'], viewport: { width: +(process.env.W || 1366), height: +(process.env.H || 650) } })).newPage();
  const errs = []; p.on('pageerror', e => errs.push(e.message)); p.on('console', m => { if (m.type() === 'error' && !/WebSocket|ERR_TUNNEL|ERR_CONNECTION|fonts/.test(m.text())) errs.push(m.text()); });
  await p.addInitScript(() => {
    window.__face = { nx: 0, ny: 0, b: 0, jaw: 0, none: false };
    window.__DZ_LITE_VISION = async () => ({ close() {}, detect() {
      const s = window.__face; if (s.none) return { faceLandmarks: [] };
      const lm = []; lm[1] = { x: 0.5 + s.nx, y: 0.5 + s.ny }; lm[33] = { x: 0.4, y: 0.4 }; lm[263] = { x: 0.6, y: 0.4 };
      return { faceLandmarks: [lm], faceBlendshapes: [{ categories: [{ categoryName: 'eyeBlinkLeft', score: s.b }, { categoryName: 'eyeBlinkRight', score: s.b }, { categoryName: 'jawOpen', score: s.jaw }] }] };
    } });
    window.__sr = null;
    window.SpeechRecognition = window.webkitSpeechRecognition = class { constructor() { window.__sr = this; } start() { setTimeout(() => this.onstart && this.onstart(), 0); } abort() {} stop() {} };
    window.__say = (text, final = true) => window.__sr.onresult({ resultIndex: 0, results: [{ 0: { transcript: text }, isFinal: final, length: 1 }] });
  });
  const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) process.exitCode = 1; };
  await p.goto('http://localhost:8091/#demo'); await p.waitForSelector('#liteBtn', { state: 'attached' });
  await p.evaluate(() => { window.__clicks = []; const o = window.DZ.clickAt; window.DZ.clickAt = (...a) => { window.__clicks.push(a.slice(2)); return o(...a); }; });
  ok(await p.evaluate(() => document.querySelector('#liteScrollBtn').disabled), 'the Scroll button waits for lite mode');
  await p.$eval('#liteBtn', e => e.click()); await p.waitForFunction(() => document.querySelector('#liteStat').textContent.startsWith('Tracking'), null, { timeout: 15000 });
  ok(await p.evaluate(() => !document.querySelector('#liteScrollBtn').disabled), 'it is ready once lite mode is on');
  const st = () => p.evaluate(() => ({ s: window.DZ.W.browser.scroll, m: document.querySelector('#mPage').scrollTop, pos: { x: Math.round(window.DZ.G.pos.x), y: Math.round(window.DZ.G.pos.y) }, mode: window.DZLite.isScrolling(), pill: (document.querySelector('.dz-pt') || {}).textContent }));
  const tilt = async deg => { await p.evaluate(d => { window.__face.ny = d / 60 * 0.2; }, deg); };   // 60 degrees per 1.0 of nose offset / eye distance (0.2)

  // a tall page in the simulated Chrome
  await p.evaluate(() => { const D = window.DZ; D.openWin('browser'); D.W.f.gq = 'cats and dogs'; D.run('gsearch', 'eye'); D.renderAll(); });
  await p.waitForTimeout(500);
  const max = await p.evaluate(() => { const i = document.querySelector('#bPage'); return i.scrollHeight - i.parentElement.clientHeight; });
  ok(max > 150, 'the results page is taller than the window (' + max + ' px to scroll)');

  // 1. start with the button; chin down scrolls down while the pointer stays
  await p.$eval('#liteScrollBtn', e => e.click()); await p.waitForTimeout(700);
  let a = await st(); ok(a.mode && /Stop scrolling/.test(await p.textContent('#liteScrollBtn')), 'the button starts scroll mode and turns into Stop scrolling');
  ok(/Scroll/.test(await p.textContent('#liteStat')), 'the status line explains scroll mode: ' + JSON.stringify((await p.textContent('#liteStat')).slice(0, 60)));
  await tilt(0); await p.waitForTimeout(300); a = await st(); ok(a.s === 0, 'a head held straight does not scroll');
  await tilt(6); await p.waitForTimeout(1200); const b1 = await st();
  ok(b1.s > 150, 'chin down scrolls the page down (' + b1.s.toFixed(0) + ' px)');
  ok(Math.abs(b1.pos.x - a.pos.x) <= 2 && Math.abs(b1.pos.y - a.pos.y) <= 2, 'the pointer stays where it was: ' + JSON.stringify(a.pos) + ' -> ' + JSON.stringify(b1.pos));
  await tilt(0); await p.waitForTimeout(900); const c1 = await st(); await p.waitForTimeout(500); const c2 = await st();
  ok(Math.abs(c2.s - c1.s) < 1, 'back to straight stops it');
  await tilt(-6); await p.waitForTimeout(900); const d1 = await st(); ok(d1.s < c2.s - 80, 'chin up scrolls back up (' + c2.s.toFixed(0) + ' -> ' + d1.s.toFixed(0) + ')');
  await tilt(0);

  // 2. a long blink leaves scroll mode and does not click
  const clicksBefore = (await p.evaluate(() => window.__clicks.length));
  await p.evaluate(() => { window.__face.b = 0.9; }); await p.waitForTimeout(800); await p.evaluate(() => { window.__face.b = 0; }); await p.waitForTimeout(400);
  a = await st(); ok(!a.mode && /Scroll with my head/.test(await p.textContent('#liteScrollBtn')), 'a long blink ends scroll mode');
  ok((await p.evaluate(() => window.__clicks.length)) === clicksBefore, 'and that blink was not a click');
  const e0 = await st(); await tilt(6); await p.waitForTimeout(600); const e1 = await st(); await tilt(0); ok(Math.abs(e1.s - e0.s) < 2, 'with scroll mode off, tilting the head does not scroll the page');

  // 3. voice: "scroll" starts it, "scroll down" still glides, "stop" ends it
  await p.$eval('#liteVoiceBtn', e => e.click()); await p.waitForTimeout(300);
  const say = async (t, ms = 250) => { await p.evaluate(t => window.__say(t, true), t); await p.waitForTimeout(ms); };
  await p.evaluate(() => { window.__face.ny = 0; }); await p.waitForTimeout(600);
  await say('scroll'); a = await st(); ok(a.mode && a.pill === 'Scroll', 'saying "scroll" starts it and the card says Scroll: ' + JSON.stringify(a.pill));
  await say('hello there my friend'); a = await st(); ok(a.mode, 'other speech is ignored while scrolling');
  await say('stop'); a = await st(); ok(!a.mode && a.pill === 'Scroll off', '"stop" ends it and the card confirms: ' + JSON.stringify(a.pill));

  // 4. Mail scrolls too (the inbox, a message and the form)
  await p.evaluate(() => { const D = window.DZ; D.openWin('mail'); D.renderAll(); }); await p.waitForTimeout(300);
  const moved = await p.evaluate(() => { const D = window.DZ; return D.scrollPx(60); });
  ok(moved > 0 || (await p.evaluate(() => { const m = document.querySelector('#mPage'); return m.scrollHeight <= m.clientHeight; })), 'scrolling moves the Mail window when it has more to show (' + moved + ' px)');

  // 5. turning lite mode off while scrolling ends scroll mode
  await p.evaluate(() => window.DZ.openWin('browser')); await p.$eval('#liteScrollBtn', e => e.click()); await p.waitForTimeout(300);
  await p.$eval('#liteBtn', e => e.click()); await p.waitForTimeout(300);
  a = await st(); ok(!a.mode && await p.evaluate(() => document.querySelector('#liteScrollBtn').disabled), 'stopping lite mode ends scrolling and disables the button');
  ok(errs.length === 0, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
  await b.close(); srv.close();
})().catch(e => { console.error(e); process.exit(1); });

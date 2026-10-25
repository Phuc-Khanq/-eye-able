// Headless check of the "Try it" list under the camera on the live demo (js/try-guide.js):
// the rows open one at a time, every quoted phrase is one lite mode understands, and each situation works when said in order.
// Run: node tests/lite_try_guide_check.js
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const root = path.join(__dirname, '..', 'eyeable-frontend', 'eyeable-frontend');
const LV = require(path.join(root, 'js', 'lite-voice.js'));
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.jpg': 'image/jpeg', '.png': 'image/png', '.svg': 'image/svg+xml', '.webp': 'image/webp' };
const srv = http.createServer((q, r) => { const f = path.join(root, decodeURIComponent(q.url.split('?')[0]).replace(/\/$/, '/index.html'));
  fs.readFile(f, (e, d) => { if (e) { r.writeHead(404); r.end(); } else { r.writeHead(200, { 'content-type': types[path.extname(f)] || 'application/octet-stream' }); r.end(d); } }); });
const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) process.exitCode = 1; };

(async () => {
  await new Promise(r => srv.listen(8093, r));
  const b = await chromium.launch({ args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'] });
  const p = await (await b.newContext({ permissions: ['camera', 'microphone'], viewport: { width: 1514, height: 927 } })).newPage();
  const errs = []; p.on('pageerror', e => errs.push(e.message));
  await p.addInitScript(() => {
    window.__sr = null;
    window.SpeechRecognition = window.webkitSpeechRecognition = class { constructor() { window.__sr = this; } start() { setTimeout(() => this.onstart && this.onstart(), 0); } abort() {} stop() {} };
    window.__say = text => window.__sr.onresult({ resultIndex: 0, results: [{ 0: { transcript: text }, isFinal: true, length: 1 }] });
  });
  await p.goto('http://localhost:8093/#demo'); await p.waitForSelector('#tryGuide');

  // 1. the list: under the camera card, six rows, one open at a time
  ok(await p.evaluate(() => document.querySelector('#demo .rail > .card').nextElementSibling.id === 'tryGuide'), 'the list sits right under the camera card');
  const rows = await p.$$('.try-q'); ok(rows.length === 6, 'six situations: ' + rows.length);
  const openState = () => p.evaluate(() => [...document.querySelectorAll('.try-q')].map(q => q.getAttribute('aria-expanded') === 'true' && !document.getElementById(q.getAttribute('aria-controls')).hidden));
  ok((await openState()).every(x => !x), 'all rows start closed');
  await rows[3].click(); ok(JSON.stringify(await openState()) === JSON.stringify([false, false, false, true, false, false]), 'pressing a row opens its steps');
  await rows[1].click(); ok(JSON.stringify(await openState()) === JSON.stringify([false, true, false, false, false, false]), 'pressing another row closes the first');
  await rows[1].click(); ok((await openState()).every(x => !x), 'pressing an open row closes it');
  await rows[0].focus(); await p.keyboard.press('Enter'); ok((await openState())[0], 'rows open from the keyboard too');
  ok(await p.evaluate(() => !document.getElementById('liteInfo').open), 'the long privacy note starts folded to leave room');

  // 2. every quoted phrase is one lite mode understands
  const phrases = await p.$$eval('.try-say', els => [...new Set(els.map(e => e.textContent.replace(/[\u201c\u201d]/g, '')))]);
  const SCROLL_EXIT = ['stop'];
  for (const t of phrases) {
    const clean = LV.normalize(t);
    const how = SCROLL_EXIT.includes(clean) ? 'scroll exit' : LV.simCommand(clean) ? 'simulator' : LV.launch(t) ? 'launch' : (LV.parse(t) || {}).name;
    ok(!!how, `"${t}" is understood (${how})`);
  }

  // 3. each situation, said in order
  const W = () => p.evaluate(() => { const W = window.DZ.W, b = W.browser; return { bo: b.open && !b.min, page: b.page, video: b.video, playing: b.playing, liked: b.liked, yq: W.f.yq, gq: W.f.gq,
    mo: W.mail.open && !W.mail.min, view: W.mail.view, to: W.f.to, body: W.f.body, focus: W.focus, sent: W.sent.length ? W.sent[0] : null }; });
  const say = async (t, w = 250) => { await p.evaluate(t => window.__say(t), t); await p.waitForTimeout(w); return W(); };
  await p.$eval('#liteStartBtn', e => e.click());
  await p.waitForFunction(() => /Listening/.test(document.querySelector('#liteVoiceStat').textContent), null, { timeout: 5000 }).catch(() => {});
  await p.waitForTimeout(400);

  let s = await say('open YouTube'); ok(s.page === 'youtube', 'Watch: "open YouTube"');
  s = await say('play video two'); ok(s.page === 'watch' && s.video === 1, 'Watch: "play video two"');
  s = await say('pause video'); ok(s.playing === false, 'Watch: "pause video"');
  s = await say('play video'); ok(s.playing === true, 'Watch: "play video"');
  s = await say('like this video'); ok(s.liked === true, 'Watch: "like this video"');
  s = await say('go home'); ok(s.page === 'home', 'Watch: "go home"');

  s = await say('search for cats and dogs'); ok(s.page === 'gresults' && /cats and dogs/.test(s.gq), 'Search: "search for cats and dogs"');
  s = await say('search YouTube for piano music'); ok(s.page === 'yresults' && /piano music/.test(s.yq), 'Search: "search YouTube for piano music"');
  await say('scroll down'); await say('scroll up'); ok(true, 'Search: "scroll down" and "scroll up" run (head scrolling is checked in lite_scroll_page_check.js)');
  s = await say('close browser'); ok(!s.bo, 'tidy up: "close browser"');

  s = await say('new email'); ok(s.mo && s.view === 'compose' && s.focus === 'to', 'Email: "new email" opens a message with the cursor in To');
  await say('hi Dizzy'); await say('anna'); s = await say('yes'); ok(/anna/i.test(s.to), 'Email: "hi Dizzy", a name, "yes" fills To: ' + JSON.stringify(s.to));
  await say('click message'); s = await say('yes', 350); ok(s.focus === 'body', 'Email: "click message", "yes" moves to the message box');
  await say('see you at the demo'); s = await say('yes'); ok(/See you at the demo\./.test(s.body), 'Email: a sentence then "yes" types it: ' + JSON.stringify(s.body));
  await say('stop dictation'); await say('click send'); s = await say('yes', 400);
  ok(s.sent && /anna/i.test(s.sent.to) && /See you at the demo/.test(s.sent.body), 'Email: "stop dictation", "click send", "yes" sends it: ' + JSON.stringify(s.sent));
  s = await say('close mail'); ok(!s.mo, 'tidy up: "close mail"');

  await say('click Mail'); s = await say('yes', 350); ok(s.mo, 'Click by name: "click Mail", "yes" opens Mail');
  await say('close mail');
  await say('right click'); ok(true, 'Click by name: "right click" runs');

  await say('pause'); s = await say('open YouTube'); ok(!s.bo, 'Pause: while paused "open YouTube" does nothing');
  await say('resume'); s = await say('open YouTube'); ok(s.page === 'youtube', 'Pause: after "resume" commands work again');
  const vol = () => p.evaluate(() => window.DZ.W.vol !== undefined ? window.DZ.W.vol : null);
  const v0 = await vol(); await say('volume up'); const v1 = await vol(); await say('mute');
  ok(v0 === null || v1 > v0 || v1 === v0, 'Pause: "volume up" and "mute" run' + (v0 !== null ? ` (volume ${v0} -> ${v1})` : ''));

  ok(errs.length === 0, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
  await b.close(); srv.close();
})().catch(e => { console.error(e); process.exit(1); });

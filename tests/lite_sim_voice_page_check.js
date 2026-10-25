// Headless check of the lite voice commands that work inside the simulated windows ("open the browser", "play video two", "new email" ...).
// Run: node tests/lite_sim_voice_page_check.js
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const root = path.join(__dirname, '..', 'eyeable-frontend', 'eyeable-frontend');
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.jpg': 'image/jpeg', '.png': 'image/png', '.svg': 'image/svg+xml', '.webp': 'image/webp' };
const srv = http.createServer((q, r) => { const f = path.join(root, decodeURIComponent(q.url.split('?')[0]).replace(/\/$/, '/index.html'));
  fs.readFile(f, (e, d) => { if (e) { r.writeHead(404); r.end(); } else { r.writeHead(200, { 'content-type': types[path.extname(f)] || 'application/octet-stream' }); r.end(d); } }); });
(async () => {
  await new Promise(r => srv.listen(8092, r));
  const b = await chromium.launch({ args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'] });
  const p = await (await b.newContext({ permissions: ['camera', 'microphone'], viewport: { width: 1366, height: 700 } })).newPage();
  const errs = []; p.on('pageerror', e => errs.push(e.message));
  await p.addInitScript(() => {
    window.__sr = null;
    window.SpeechRecognition = window.webkitSpeechRecognition = class { constructor() { window.__sr = this; } start() { setTimeout(() => this.onstart && this.onstart(), 0); } abort() {} stop() {} };
    window.__say = text => window.__sr.onresult({ resultIndex: 0, results: [{ 0: { transcript: text }, isFinal: true, length: 1 }] });
  });
  const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) process.exitCode = 1; };
  await p.goto('http://localhost:8092/#demo'); await p.waitForSelector('#liteStartBtn');
  const S = () => p.evaluate(() => { const W = window.DZ.W, b = W.browser; return { bo: b.open && !b.min, page: b.page, video: b.video, playing: b.playing, liked: b.liked, yq: W.f.yq, gq: W.f.gq, mo: W.mail.open && !W.mail.min, view: W.mail.view }; });
  const say = async t => { await p.evaluate(t => window.__say(t), t); await p.waitForTimeout(250); return S(); };
  await p.click('#liteStartBtn'); await p.waitForFunction(() => document.querySelector('#liteVoiceStat').textContent.startsWith('Starting') || /Listening/.test(document.querySelector('#liteVoiceStat').textContent), null, { timeout: 8000 });
  await p.waitForTimeout(500);
  let s = await say('open the browser'); ok(s.bo && s.page === 'home', '"open the browser" opens Chrome on its home page');
  s = await say('open YouTube'); ok(s.page === 'youtube', '"open YouTube" shows YouTube');
  s = await say('play video two'); ok(s.page === 'watch' && s.video === 1, '"play video two" plays the second video');
  s = await say('pause video'); ok(s.playing === false, '"pause video" pauses it');
  s = await say('play video'); ok(s.playing === true, '"play video" plays it again');
  s = await say('like this video'); ok(s.liked === true, '"like this video" likes it');
  s = await say('go home'); ok(s.page === 'home', '"go home" returns to the browser home page');
  s = await say('search YouTube for piano music'); ok(s.page === 'yresults' && /piano music/i.test(s.yq), '"search YouTube for piano music" shows results: ' + s.yq);
  s = await say('play the first video'); ok(s.page === 'watch', '"play the first video" plays a result');
  s = await say('search for cats and dogs'); ok(s.page === 'gresults' && /cats and dogs/.test(s.gq), '"search for cats and dogs" is a Google search');
  s = await say('close browser'); ok(!s.bo, '"close browser" closes Chrome');
  s = await say('new email'); ok(s.mo && s.view === 'compose', '"new email" opens a new message in Mail');
  s = await say('open the inbox'); ok(s.mo && s.view === 'inbox', '"open the inbox" shows the inbox');
  s = await say('close mail'); ok(!s.mo, '"close mail" closes Mail');
  s = await say('hi dizzy'); s = await say('go home'); ok(!s.bo, 'while dictating, "go home" is words to type, not a command');
  // the one button: Stop turns the voice off too, even when the camera could not start here (no tracker download in this test)
  ok(/Stop/.test(await p.textContent('#liteStartBtn')) && (await p.evaluate(() => ['liteBtn', 'liteVoiceBtn', 'liteScrollBtn'].every(i => document.getElementById(i).hidden))), 'one visible button, and it says Stop while anything is on');
  await p.click('#liteStartBtn'); await p.waitForTimeout(600);
  ok(/Voice off/.test(await p.textContent('#liteVoiceStat')) && /Start hands-free/.test(await p.textContent('#liteStartBtn')), 'pressing it again turns everything off and it says Start hands-free again');
  ok(errs.length === 0, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
  await b.close(); srv.close();
})().catch(e => { console.error(e); process.exit(1); });

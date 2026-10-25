// Headless check of the lite voice on the real page with a stand-in speech recogniser. Run: node tests/lite_voice_page_check.js
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const root = path.join(__dirname, '..', 'diseyeble-frontend', 'diseyeble-frontend');
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.jpg': 'image/jpeg' };
const srv = http.createServer((q, r) => { const f = path.join(root, decodeURIComponent(q.url.split('?')[0]).replace(/\/$/, '/index.html'));
  fs.readFile(f, (e, d) => { if (e) { r.writeHead(404); r.end(); } else { r.writeHead(200, { 'content-type': types[path.extname(f)] || 'application/octet-stream' }); r.end(d); } }); });
(async () => {
  await new Promise(r => srv.listen(8097, r));
  const b = await chromium.launch(); const p = await (await b.newContext({ viewport: { width: 1500, height: 1000 } })).newPage(); const errs = [];
  p.on('pageerror', e => errs.push(e.message)); p.on('console', m => { if (m.type() === 'error' && !/WebSocket|ERR_TUNNEL|ERR_CONNECTION/.test(m.text())) errs.push(m.text()); });
  await p.addInitScript(() => {
    window.__sr = null;
    window.SpeechRecognition = window.webkitSpeechRecognition = class { constructor() { window.__sr = this; this.started = false; }
      start() { this.started = true; } abort() { this.started = false; } stop() { this.started = false; } };
    window.__say = (text, final = true) => { const r = [{ 0: { transcript: text }, isFinal: final, length: 1 }]; window.__sr.onresult({ resultIndex: 0, results: r }); };
  });
  await p.goto('http://localhost:8097/#demo'); await p.waitForSelector('#liteVoiceBtn');
  const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) process.exitCode = 1; };
  const say = async (t, final = true) => { await p.evaluate(([t, f]) => window.__say(t, f), [t, final]); await p.waitForTimeout(120); };
  await p.click('#liteVoiceBtn'); await p.waitForTimeout(150);
  ok(await p.evaluate(() => window.__sr && window.__sr.started), 'the recogniser started');
  ok(await p.evaluate(() => !document.querySelector('.dz-card').hidden), 'the Diseyeble card is showing on the simulated screen');
  await say('volume up'); ok(await p.evaluate(() => document.querySelector('.sim-osd').classList.contains('show')), '"volume up" shows the volume bar');
  ok((await p.textContent('.sim-osd .num')) === '60', 'the volume went from 50 to 60');
  await say('brightness down'); ok((await p.textContent('.sim-osd .num')) === '90', '"brightness down" went to 90');
  await say('open chrome'); ok(await p.evaluate(() => window.DZ.W.browser.open), '"open chrome" opened Chrome');
  await say('open youtube'); ok(await p.evaluate(() => window.DZ.W.browser.page !== 'home'), '"open YouTube" went to YouTube');
  await say('go back'); await say('hi dizzy'); ok((await p.textContent('.dz-pt')).includes('Dictation'), '"hi Dizzy" starts dictation (pill says Dictation)');
  await say('hello', false); ok((await p.textContent('.dz-body')).includes('hello') && (await p.textContent('.dz-lab')).includes('Hearing'), 'while you speak the card says Hearing and shows the words');
  await say('hello world'); ok((await p.textContent('.dz-lab')).includes('Will type') && (await p.textContent('.dz-actions')).includes('Type it'), 'the finished words wait on the card with Type it / Drop it');
  await say('yes'); await p.waitForTimeout(200);
  ok(await p.evaluate(() => /hello world/.test(Object.values(window.DZ.W.f).join('|'))), '"yes" typed the words into a text box');
  await say('hi dizzy'); await say('drop me'); await p.click('.dz-actions button:nth-child(2)'); await p.waitForTimeout(150);
  ok((await p.textContent('.dz-body')) === '' || !(await p.textContent('.dz-body')).includes('drop me'), 'the Drop it button on the card drops the words');
  await say('bye'); await p.waitForTimeout(1200); ok(/Dictation off|Diseyeble/.test(await p.textContent('.dz-pt')) && (await p.evaluate(() => window.DZVoice.voice.state.mode)) === 'pointer', '"bye" ends dictation');
  await say('click banana'); await p.waitForTimeout(100);
  await say('click mail'); await p.waitForTimeout(150);
  ok(/Click/.test(await p.textContent('.dz-body')) , 'click <name> asks first on the card: ' + JSON.stringify(await p.textContent('.dz-body')));
  await say('yes'); await p.waitForTimeout(250); ok(await p.evaluate(() => window.DZ.W.mail.open), '"yes" then clicked Mail');
  await p.click('#liteVoiceBtn'); await p.waitForTimeout(200);
  ok(await p.evaluate(() => document.querySelector('.dz-card').hidden), 'turning voice off hides the card');
  ok(errs.length === 0, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
  await p.click('#liteVoiceBtn'); await say('hi dizzy'); await say('this is a test of the card');
  await p.screenshot({ path: process.env.SHOT || '/tmp/lite-voice.png' });
  await b.close(); srv.close();
})().catch(e => { console.error(e); process.exit(1); });

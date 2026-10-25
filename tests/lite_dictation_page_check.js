// Headless check of dictating on the hosted page's simulated desktop (stand-in speech recogniser).
// Run: node tests/lite_dictation_page_check.js   (SHOT=/path.png saves a screenshot of the finished state; W= and H= set the window size)
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const root = path.join(__dirname, '..', 'eyeable-frontend', 'eyeable-frontend');
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.jpg': 'image/jpeg', '.png': 'image/png', '.svg': 'image/svg+xml' };
const srv = http.createServer((q, r) => { const f = path.join(root, decodeURIComponent(q.url.split('?')[0]).replace(/\/$/, '/index.html'));
  fs.readFile(f, (e, d) => { if (e) { r.writeHead(404); r.end(); } else { r.writeHead(200, { 'content-type': types[path.extname(f)] || 'application/octet-stream' }); r.end(d); } }); });
(async () => {
  await new Promise(r => srv.listen(8099, r));
  const b = await chromium.launch({ args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'] });
  const p = await (await b.newContext({ viewport: { width: +(process.env.W || 1500), height: +(process.env.H || 1000) } })).newPage();
  const errs = []; p.on('pageerror', e => errs.push(e.message));
  await p.addInitScript(() => {
    window.__sr = null;
    window.SpeechRecognition = window.webkitSpeechRecognition = class { constructor() { window.__sr = this; } start() { setTimeout(() => this.onstart && this.onstart(), 0); } abort() {} stop() {} };
    window.__say = (text, final = true) => window.__sr.onresult({ resultIndex: 0, results: [{ 0: { transcript: text }, isFinal: final, length: 1 }] });
  });
  await p.goto('http://localhost:8099/#demo'); await p.waitForSelector('#liteVoiceBtn', { state: 'attached' });
  const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) process.exitCode = 1; };
  const say = async (t, final = true, w = 150) => { await p.evaluate(([t, f]) => window.__say(t, f), [t, final]); await p.waitForTimeout(w); };
  const F = () => p.evaluate(() => ({ ...window.DZ.W.f, focus: window.DZ.W.focus }));
  const card = () => p.textContent('.dz-body');
  const dictate = async text => { await say('hi dizzy'); await say(text); await say('yes', true, 250); };
  await p.evaluate(() => { window.DZ.engineLive = true; });
  await p.$eval('#liteVoiceBtn', e => e.click()); await p.waitForTimeout(400);

  // Chrome's search box: words are typed exactly as heard
  await say('open chrome', true, 300); await say('hi dizzy'); await say('cats and dogs');
  ok((await p.textContent('.dz-lab')).includes('into search'), 'the card says where the words will go: ' + JSON.stringify(await p.textContent('.dz-lab')));
  await say('yes', true, 250);
  ok((await F()).gq.trim() === 'cats and dogs', 'a search box gets the words as heard: ' + JSON.stringify((await F()).gq));
  await say('bye', true, 200);

  // a new message starts with the cursor in To: a sentence goes to the message instead
  await say('click mail'); await say('yes', true, 300); await say('click new mail'); await say('yes', true, 300);
  ok((await F()).focus === 'to', 'a new message starts in the To box');
  ok(!/^EYEable asks:/.test(await p.textContent('#tip')), 'the mascot no longer shows the answered question: ' + JSON.stringify(await p.textContent('#tip')));
  await say('hi dizzy'); await say('quarterly demo is ready', false);
  ok((await p.textContent('.dz-lab')).includes('Hearing') && (await card()).includes('quarterly demo is ready'), 'while you speak the card says Hearing');
  await say('quarterly demo is ready');
  ok((await card()) === 'Quarterly demo is ready.', 'the card shows the tidied words (capital and full stop): ' + JSON.stringify(await card()));
  await say('i will send the slides tonight', false);
  ok((await card()).includes('Quarterly demo is ready.') && (await card()).includes('i will send the slides tonight'), 'the grey live words continue after the waiting words');
  await say('i will send the slides tonight');
  ok((await card()) === 'Quarterly demo is ready. I will send the slides tonight.', 'two sentences wait together, tidied: ' + JSON.stringify(await card()));
  const snap = async n => { if (process.env.SHOTS) await p.screenshot({ path: path.join(process.env.SHOTS, n + '.png') }); };
  ok((await p.textContent('.dz-lab')).includes('into Message'), 'a sentence for a new message says "into Message": ' + JSON.stringify(await p.textContent('.dz-lab')));
  await snap('1-will-type');
  await say('yes', true, 250);
  await snap('2-typed');
  let f = await F();
  ok(f.body === 'Quarterly demo is ready. I will send the slides tonight. ' && f.to === '', 'yes typed them into the message, not the To box: ' + JSON.stringify({ to: f.to, body: f.body }));
  ok(f.focus === 'body', 'the message box now has the cursor');

  // choosing a box by voice
  await say('click subject'); ok(/Subject/.test(await card()), 'click subject asks first: ' + JSON.stringify(await card()));
  await say('yes', true, 300); ok((await F()).focus === 'subj', 'then the Subject box has the cursor');
  await dictate('dinner on sunday'); f = await F(); ok(f.subj.trim() === 'Dinner on sunday', 'a subject gets a capital and no full stop: ' + JSON.stringify(f.subj));
  await say('click recipient'); await say('yes', true, 300); ok((await F()).focus === 'to', 'click recipient puts the cursor in To');
  await dictate('anna at example dot com'); f = await F(); ok(f.to.trim() === 'anna at example dot com', 'an address stays as heard: ' + JSON.stringify(f.to));

  // a long passage: the card shows the last words with "N earlier words"
  await say('click message'); await say('yes', true, 300);
  const long = Array.from({ length: 40 }, (_, i) => 'word' + (i + 1)).join(' ');
  await say('hi dizzy'); await say(long);
  ok(/12 earlier words/.test(await p.textContent('.dz-body .more')) && (await card()).includes('word40'), 'a 40-word passage shows "… 12 earlier words" and its end');
  await p.screenshot({ path: process.env.SHOT || '/tmp/lite-dictation.png' });
  await say('no', true, 200);
  ok(errs.length === 0, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
  await b.close(); srv.close();
})().catch(e => { console.error(e); process.exit(1); });

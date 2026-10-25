// Unit tests for the in-browser voice (eyeable-frontend/eyeable-frontend/js/lite-voice.js). Run: node tests/lite_voice.test.js
const assert = require('assert');
const { createVoice, parse, launch } = require('../eyeable-frontend/eyeable-frontend/js/lite-voice.js');

function rig(findResult) {
  const out = { msgs: [], acts: [], clicks: [], timers: [], finds: [] };
  const hooks = {
    emit: m => out.msgs.push(m), act: m => out.acts.push(m), pointerClick: (b, s) => out.clicks.push([b, s]),
    find: l => { out.finds.push(l); return findResult === undefined ? null : findResult; }, recenter: () => out.msgs.push({ type: 'recenter' }), quit: () => out.msgs.push({ type: 'quit' }),
  };
  const v = createVoice(hooks, { setTimeout: (fn, ms) => { out.timers.push({ fn, ms }); return out.timers.length; }, clearTimeout: id => { if (out.timers[id - 1]) out.timers[id - 1].dead = true; } });
  out.v = v; out.last = type => [...out.msgs].reverse().find(m => m.type === type); out.of = type => out.msgs.filter(m => m.type === type);
  return out;
}
const tests = []; const test = (n, f) => tests.push([n, f]);

test('matching: fillers, case, fuzzy long phrases, exact short ones, whole utterance only', () => {
  assert.strictEqual(parse('Um, volume up please.').name, 'volume_up');
  assert.strictEqual(parse('scrol down').name, 'scroll_down');
  assert.strictEqual(parse('brightness dowm').name, 'brightness_down');
  assert.strictEqual(parse("I'll send it tomorrow"), null);
  assert.strictEqual(parse('no').name, 'no');                              // a short fixed command, exact
  assert.strictEqual(parse('go'), null);                                   // "go" never becomes "no"
  assert.deepStrictEqual(parse('Click Send.'), { name: 'target', label: 'send' });
  assert.strictEqual(parse('click on the send button').label, 'send button');
  assert.strictEqual(parse('click this').name, 'click_here');
});
test('open, go to and search: the engine\'s launch rules', () => {
  assert.deepStrictEqual(launch('Open Chrome.'), { app: 'chrome' });
  assert.deepStrictEqual(launch('open google chrome'), { app: 'chrome' });
  assert.deepStrictEqual(launch('open YouTube'), { url: 'https://youtube.com' });
  assert.deepStrictEqual(launch('go to YouTube dot com'), { url: 'https://youtube.com' });
  assert.deepStrictEqual(launch('go to youtube'), { url: 'https://youtube.com' });
  assert.deepStrictEqual(launch('search for cats'), { url: 'https://www.google.com/search?q=cats' });
  assert.deepStrictEqual(launch('search for how to bake bread'), { url: 'https://www.google.com/search?q=how+to+bake+bread' });
  assert.strictEqual(launch('go to sleep'), null); assert.strictEqual(launch('open compose'), null); assert.strictEqual(launch('open'), null);
});
test('commands become page actions', () => {
  const r = rig();
  for (const s of ['volume up', 'volume down', 'mute', 'brightness up', 'brightness down']) r.v.heard(s);
  assert.deepStrictEqual(r.acts.map(a => a.name), ['volume_up', 'volume_down', 'mute', 'brightness_up', 'brightness_down']);
  r.v.heard('scroll down'); r.v.heard('zoom in'); r.v.heard('go back'); r.v.heard('new tab');
  assert.deepStrictEqual(r.acts.slice(5), [{ action: 'scroll', direction: 'down' }, { action: 'key', key: 'ctrl+=' }, { action: 'key', key: 'alt+left' }, { action: 'key', key: 'ctrl+t' }]);
  r.v.heard('open YouTube'); r.v.heard('open chrome'); r.v.heard('search for cats');
  assert.deepStrictEqual(r.acts.slice(9).map(a => a.url || a.app), ['https://youtube.com', 'chrome', 'https://www.google.com/search?q=cats']);
  r.v.heard('click'); r.v.heard('right click'); assert.deepStrictEqual(r.clicks, [['left', true], ['right', false]]);
  r.v.heard('recenter'); assert(r.last('recenter'));
});
test('an utterance that is not a command is told so, not typed', () => {
  const r = rig(); r.v.heard('the weather is nice today'); assert(/Not a command/.test(r.last('status').text)); assert.strictEqual(r.acts.length, 0);
});
test('dictation: hi Dizzy, words wait on the card, yes types them', () => {
  const r = rig(); r.v.heard('hi dizzy'); assert.strictEqual(r.last('mode').mode, 'dictation');
  r.v.live('see you'); assert.strictEqual(r.last('dictation_live').text, 'see you');
  r.v.heard('see you at noon'); const p = r.last('dictation_preview'); assert.deepStrictEqual(p.words.map(w => w.text), ['see', 'you', 'at', 'noon']);
  assert.strictEqual(r.last('dictation_live').text, '');
  r.v.heard('yes'); assert.deepStrictEqual(r.acts.at(-1), { action: 'type', text: 'see you at noon ' }); assert.strictEqual(r.last('dictation_preview').words.length, 0);
});
test('dictation: more speech adds to the preview; "know" while words wait drops them; the card buttons work', () => {
  const r = rig(); r.v.heard('start dictation'); r.v.heard('hello mai'); r.v.heard('see you sunday');
  assert.strictEqual(r.last('dictation_preview').words.length, 5);
  r.v.heard('Know.'); assert.strictEqual(r.last('dictation_preview').words.length, 0); assert.strictEqual(r.acts.length, 0);
  r.v.heard('good morning'); r.v.insert(); assert.strictEqual(r.acts.at(-1).text, 'good morning ');
});
test('dictation: ordinary words that look like commands stay text, "click ..." and trigger words still work', () => {
  const r = rig(); r.v.heard('hi dizzy'); r.v.heard('open the door when you get home');
  assert.strictEqual(r.last('dictation_preview').words.length, 7); assert.strictEqual(r.acts.length, 0);
  r.v.heard('Venus, volume up'); assert.strictEqual(r.acts.at(-1).name, 'volume_up');
  const hit = { label: 'Send', click() { hit.clicked = true; } }; const r2 = rig(hit); r2.v.heard('hi dizzy'); r2.v.heard('click send'); assert.strictEqual(r2.last('confirm').prompt, 'Click Send?');
});
test('bye with words waiting asks first; bye with nothing waiting just stops', () => {
  const r = rig(); r.v.heard('hi dizzy'); r.v.heard('bye'); assert.strictEqual(r.last('mode').mode, 'pointer');
  const q = rig(); q.v.heard('hi dizzy'); q.v.heard('some words'); q.v.heard('bye'); assert.strictEqual(q.last('confirm').state, 'open'); assert.strictEqual(q.v.state.mode, 'dictation');
  q.v.heard('yes'); assert.strictEqual(q.v.state.mode, 'pointer'); assert.strictEqual(q.last('dictation_preview').words.length, 0);
});
test('send it asks, then types the words and presses Enter', () => {
  const r = rig(); r.v.heard('hi dizzy'); r.v.heard('on my way'); r.v.heard('send it'); assert(/send\?$/.test(r.last('confirm').prompt));
  r.v.heard('yes'); assert.deepStrictEqual(r.acts.slice(-2), [{ action: 'type', text: 'on my way ' }, { action: 'key', key: 'enter' }]);
});
test('a question times out after 12 s and counts as no', () => {
  const r = rig(); r.v.heard('close window'); assert.strictEqual(r.last('confirm').expires_in_s, 12); assert.strictEqual(r.timers.at(-1).ms, 12000);
  r.timers.at(-1).fn(); assert.strictEqual(r.last('confirm').state, 'expired'); assert.strictEqual(r.acts.length, 0);
});
test('click <name>: finds the button, asks, clicks on yes; says so when it cannot find it', () => {
  const hit = { label: 'Compose', click() { hit.clicked = true; } }; const r = rig(hit);
  r.v.heard('click compose'); assert.strictEqual(r.last('confirm').prompt, 'Click Compose?'); assert(!hit.clicked);
  r.v.heard('yes'); assert(hit.clicked);
  const n = rig(null); n.v.heard('click banana'); assert.strictEqual(n.last('target_none').label, 'banana');
});
test('the blink and the card buttons answer through answer()', () => {
  const r = rig(); r.v.heard('close window'); assert(r.v.pending()); assert.strictEqual(r.v.answer(true), true); assert.deepStrictEqual(r.acts.at(-1), { action: 'key', key: 'alt+f4' });
  assert.strictEqual(r.v.answer(true), false);
});
test('paused: only resume (and quit) are heard', () => {
  const r = rig(); r.v.heard('take a break'); assert.strictEqual(r.v.state.mode, 'paused');
  r.v.heard('volume up'); assert.strictEqual(r.acts.length, 0); r.v.heard('resume'); assert.strictEqual(r.v.state.mode, 'pointer'); r.v.heard('volume up'); assert.strictEqual(r.acts.length, 1);
});
test('undo drops waiting words, otherwise undoes the last typing', () => {
  const r = rig(); r.v.heard('hi dizzy'); r.v.heard('oops'); r.v.heard('scratch that'); assert.strictEqual(r.last('dictation_preview').words.length, 0);
  r.v.heard('undo'); assert.deepStrictEqual(r.acts.at(-1), { action: 'undo' });
});

let failed = 0;
for (const [n, f] of tests) { try { f(); console.log('ok   ' + n); } catch (e) { failed++; console.log('FAIL ' + n + ' :: ' + e.message); } }
console.log(`${tests.length - failed}/${tests.length} passed`); process.exit(failed ? 1 : 0);

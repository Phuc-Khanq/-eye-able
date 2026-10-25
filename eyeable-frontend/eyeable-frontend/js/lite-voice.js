/* js/lite-voice.js: voice for the hosted website, with no install: commands ("volume up", "open YouTube", "click Send",
   "scroll down", ...) and dictation ("hi Dizzy" ... "yes") on the SIMULATED desktop.

   It is NOT the EYEable engine's voice. Differences you should say out loud:
   - recognition is your browser's (Chrome or Edge): the engine uses Whisper large-v3-turbo on your own GPU, with the
     dysarthria fine-tune. In Chrome the browser's speech service may send audio to Google (we ask for on-device
     recognition first and tell you when it falls back).
   - no live "mouth moved" check, no clean-up model, no UI-Venus: "click <name>" finds the button by its label in
     the simulated screen.
   What it shares with the engine: the phrase table (js/lite-phrases.js is generated from engine/voice/commands.py),
   the matching rules (whole utterance, fuzzy for long phrases, exact for short ones), the answers, the dictation card
   (Hearing / Will type / Type it / Drop it), the 12 s questions and the on-screen volume and brightness.
   The card and the page actions are js/engine.js's, reached through window.DZ.bridge.

   createVoice() has no browser calls so tests/lite_voice.test.js can run it in node. */
(() => {
  'use strict';
  const P = (typeof window !== 'undefined' && window.DZ_PHRASES) || (typeof require !== 'undefined' ? require('./lite-phrases.js') : null);
  if (!P) return;

  /* ---------------- matching: the engine's commands.py, ported ---------------- */
  const FIXED = P.FIXED_COMMANDS, EXACT = new Set(P.EXACT_ONLY), FILLERS = new Set(P.FILLER_WORDS);
  const LEADING = new Set(P.LABEL_LEADING_WORDS), POINTING = new Set(P.POINTING_WORDS);
  const LONG = Object.keys(FIXED).filter(p => p.length >= P.EXACT_ONLY_BELOW && !EXACT.has(p));

  function normalize(text) {
    let t = String(text).toLowerCase().replace(/’/g, "'").replace(/[^\w\s']/g, ' ').replace(/\s+/g, ' ').trim();
    for (const pre of P.FILLER_PREFIXES) if (t.startsWith(pre)) t = t.slice(pre.length);
    return t.split(' ').filter(w => w && !FILLERS.has(w)).join(' ');
  }
  const collapse = c => { const out = []; for (const w of c.split(' ')) if (!out.length || out[out.length - 1] !== w) out.push(w); return out.join(' '); };
  function ratio(a, b) {                       // rapidfuzz fuzz.ratio: 100 * 2 * LCS / (len a + len b)
    if (!a.length && !b.length) return 100;
    const dp = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0));
    for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++)
      dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
    return 100 * 2 * dp[a.length][b.length] / (a.length + b.length);
  }
  function fixed(clean) {
    if (Object.prototype.hasOwnProperty.call(FIXED, clean)) return { name: FIXED[clean] };
    if (clean.length < P.EXACT_ONLY_BELOW || clean.split(' ').length > P.MAX_COMMAND_WORDS) return null;
    let best = null, bs = 0;
    for (const p of LONG) { const s = ratio(clean, p); if (s > bs) { bs = s; best = p; } }
    return best && bs >= P.FIXED_CUTOFF ? { name: FIXED[best], score: bs } : null;
  }
  function target(clean) {
    const i = clean.indexOf(' '); if (i < 0) return null;
    const first = clean.slice(0, i); let words = clean.slice(i + 1).split(' ');
    if (!P.TARGET_VERBS.some(v => ratio(first, v) >= P.VERB_CUTOFF)) return null;
    while (words.length && LEADING.has(words[0])) words.shift();
    if (!words.length || words.length > P.MAX_LABEL_WORDS) return null;
    if (POINTING.has(words[0])) return { name: 'click_here' };
    return { name: 'target', label: words.join(' ') };
  }
  function typed(clean) {
    const i = clean.indexOf(' '); if (i < 0) return null;
    const first = clean.slice(0, i), rest = clean.slice(i + 1);
    if (rest.split(' ')[0] === 'of') return null;
    return P.TYPE_VERBS.includes(first) && rest ? { name: 'type_text', label: rest } : null;
  }
  /* Command for a transcript, or null. allowTargets false = fixed commands (and "type ...") only. */
  function parse(text, allowTargets = true) {
    const clean = collapse(normalize(text)); if (!clean) return null;
    return fixed(clean) || (allowTargets ? target(clean) : null) || typed(clean);
  }
  const answerOf = text => P.ANSWER_MISHEARINGS[collapse(normalize(text))] || null;

  /* "open Chrome", "go to youtube.com", "search for cats": engine/voice/launch.py, ported */
  const LABEL = '[a-z0-9](?:[a-z0-9-]*[a-z0-9])?';
  const HOST = new RegExp(`^${LABEL}(?:\\.${LABEL})*(?:/[a-z0-9\\-._~/%?=&+#]*)?$`);
  function launchWords(text) {
    let t = ' ' + String(text).toLowerCase().replace(/’/g, "'").trim() + ' ';
    t = t.replace(/\s+dot\s+/g, '.').trim().replace(/[^\w\s'./+#-]/g, ' ');
    t = t.split(/\s+/).map(w => w.replace(/^[.'\-/]+|[.'\-/]+$/g, '')).filter(Boolean).join(' ');
    for (const pre of P.FILLER_PREFIXES) if (t.startsWith(pre)) t = t.slice(pre.length);
    return t.split(' ').filter(w => w && !FILLERS.has(w));
  }
  function launch(text) {
    const words = launchWords(text); if (words.length < 2) return null;
    const first = words[0]; let rest = words.slice(1);
    if (['open', 'start', 'launch'].includes(first)) {
      while (rest.length && rest[0] === 'the') rest = rest.slice(1);
      let name = rest.join(' '); name = P.launch_aliases[name] || name;
      if (Object.prototype.hasOwnProperty.call(P.launch_apps, name)) return { app: name };
      if (rest.length === 1) { const site = String(P.launch_sites[name] || ''); if (site && HOST.test(site) && site.includes('.')) return { url: 'https://' + site }; }
      return null;
    }
    if (first === 'go' && rest[0] === 'to') {
      rest = rest.slice(1); if (rest.length !== 1 || !HOST.test(rest[0])) return null;
      let [host, ...path] = rest[0].split('/'); path = path.join('/');
      if (!host.includes('.')) { host = String(P.launch_sites[host] || ''); if (!HOST.test(host) || !host.includes('.')) return null; }
      return { url: 'https://' + host + (path ? '/' + path : '') };
    }
    if (first === 'search') {
      if (rest[0] === 'for') rest = rest.slice(1); if (!rest.length) return null;
      return { url: P.launch_search_url.replace('{q}', encodeURIComponent(rest.join(' ')).replace(/%20/g, '+')) };
    }
    return null;
  }

  /* The browser's recognizer gives plain words. For the message body (kind 'body') and the subject ('subj') tidy them
     the way a reader expects: a capital to start, "I" for "i", a full stop at the end of a body sentence (not a subject).
     Anything else (a search box, an address) is left exactly as heard. This is NOT the engine's clean-up model: it never
     rewords, it only fixes capitals and the final stop. */
  function tidy(text, kind) {
    let t = String(text || '').replace(/\s+/g, ' ').trim();
    if (!t || (kind !== 'body' && kind !== 'subj')) return t;
    t = t.replace(/\bi\b(?!\.)/g, 'I').replace(/^[a-z]/, c => c.toUpperCase()).replace(/([.!?]\s+)([a-z])/g, (m, a, b) => a + b.toUpperCase());
    if (kind === 'body') { if (!/[.!?…:,;]["”')]?$/.test(t)) t += '.'; } else t = t.replace(/\.+$/, '');
    return t;
  }

  /* ---------------- the voice logic: modes, dictation preview, questions ---------------- */
  /* hooks: emit(msg) sends an engine-style message to the page (js/engine.js draws the card from them);
            act(msg) is a page_action; pointerClick(button, snap); find(label) -> {label, click()} | null;
            recenter(); quit().   timers: setTimeout/clearTimeout can be replaced in tests. */
  function createVoice(hooks, opt = {}) {
    const now = opt.now || (() => Date.now()), st = opt.setTimeout || setTimeout, ct = opt.clearTimeout || clearTimeout;
    const S = { mode: 'pointer', chunks: [], q: null, qTimer: 0, qId: 0 };
    const emit = m => hooks.emit(m);
    const status = (text, level = 'info') => emit({ type: 'status', level, text });
    const flash = label => emit({ type: 'command', label });
    const setMode = mode => { if (S.mode !== mode) { S.mode = mode; emit({ type: 'mode', mode }); } };
    /* hooks.kindFor(text) -> 'body' | 'subj' | 'plain': where the words will land (the page knows), so the card shows
       exactly what will be typed. No hook (the unit tests) = 'plain' = as heard. */
    const kindOf = () => { const raw = S.chunks.filter(c => c !== '\n').join(' '); try { return (hooks.kindFor && raw && hooks.kindFor(raw)) || 'plain'; } catch (e) { return 'plain'; } };
    const shown = () => { const kind = kindOf(); return S.chunks.map(c => (c === '\n' ? c : tidy(c, kind))); };
    const previewText = () => shown().join(' ').replace(/ ?\n ?/g, '\n').trim();
    const showPreview = () => emit({ type: 'dictation_preview', words: shown().filter(c => c !== '\n').join(' ').split(' ').filter(Boolean).map(text => ({ text, conf: 1 })), cleaned: false, original: '' });
    const live = text => emit({ type: 'dictation_live', text: text || '' });
    function dropPreview() { S.chunks = []; showPreview(); }
    function insert(extra) {
      const text = previewText(); if (!text) return false;
      S.chunks = []; showPreview(); hooks.act({ action: 'type', text: text + ' ' });
      if (extra === 'enter') hooks.act({ action: 'key', key: 'enter' });
      return true;
    }
    function closeQ(state) {
      if (!S.q) return null; const q = S.q; S.q = null; ct(S.qTimer);
      emit({ type: 'confirm', id: q.id, prompt: q.prompt, state }); return q;
    }
    function ask(prompt, onYes, onNo) {
      closeQ('cancelled');
      const id = ++S.qId; S.q = { id, prompt, onYes, onNo };
      emit({ type: 'confirm', id, prompt, state: 'open', expires_in_s: 12 });
      S.qTimer = st(() => { const q = closeQ('expired'); if (q && q.onNo) q.onNo(); }, 12000);
    }
    function answer(yes) {
      if (S.q) { const q = closeQ(yes ? 'confirmed' : 'cancelled'); if (yes) q.onYes(); else if (q.onNo) q.onNo(); return true; }
      if (S.chunks.length) { if (yes) insert(); else dropPreview(); return true; }
      return false;
    }

    /* Scroll mode (head scrolling), like the engine's: "scroll" starts it (never from dictation), the pointer stays and the head sets
       the speed; "stop", "done" or "stop scrolling" end it; "scroll up/down" still glide; any other command ends it and then runs;
       other speech is ignored. It needs lite mode's camera (hooks.scroll says yes or no). */
    function enterScroll() {
      if (S.mode === 'dictation') return false;
      if (S.mode === 'scroll') return true;
      if (!hooks.scroll || !hooks.scroll(true)) { status('Head scrolling needs lite mode: press “Try hands-free in your browser (lite)”. Say “scroll up” or “scroll down” for now.'); return false; }
      setMode('scroll'); flash('Scroll'); return true;
    }
    function leaveScroll(label) {
      if (S.mode !== 'scroll') return;
      if (hooks.scroll) hooks.scroll(false);
      setMode('pointer'); if (label) flash(label);
    }
    const SCROLL_EXIT = ['stop', 'done', 'stop scrolling'];

    function run(cmd, raw) {
      const n = cmd.name, say = l => flash(l);
      switch (n) {
        case 'start_dictation': setMode('dictation'); say('Dictation'); return;
        case 'stop_dictation':
          if (S.mode !== 'dictation') { status('Dictation is not on.'); return; }
          if (S.chunks.length) ask('Drop the waiting words and stop?', () => { dropPreview(); setMode('pointer'); });
          else { setMode('pointer'); say('Dictation off'); }
          return;
        case 'quit_app': say('Bye'); hooks.quit(); return;
        case 'send_it':
          if (!S.chunks.length) { status('Nothing to send yet.'); return; }
          ask(`Insert “${previewText().slice(0, 60)}” and send?`, () => insert('enter')); return;
        case 'undo': if (S.chunks.length) { dropPreview(); say('Dropped'); } else { hooks.act({ action: 'undo' }); say('Undo'); } return;
        case 'new_line': if (S.mode === 'dictation') { S.chunks.push('\n'); showPreview(); } return;
        case 'use_original': case 'clean_up': if (S.chunks.length) insert(); else status('Lite mode has no clean-up model; your own words are typed.'); return;
        case 'yes': if (!answer(true)) status('Nothing is waiting for a yes.'); return;
        case 'no': if (!answer(false)) status('Nothing is waiting for a no.'); return;
        case 'cancel': if (!answer(false)) hooks.act({ action: 'key', key: 'esc' }); say('Cancel'); return;
        case 'click_here': if (S.q) { answer(true); return; } hooks.pointerClick('left', true); say('Click'); return;
        case 'right_click': hooks.pointerClick('right', false); say('Right click'); return;
        case 'dwell_on': case 'dwell_off': status('Dwell clicking is not in lite mode.'); return;
        case 'scroll_up': hooks.act({ action: 'scroll', direction: 'up' }); say('Scroll up'); return;
        case 'scroll_down': hooks.act({ action: 'scroll', direction: 'down' }); say('Scroll down'); return;
        case 'scroll_mode': enterScroll(); return;
        case 'pause': setMode('paused'); say('Paused'); return;
        case 'resume': setMode('pointer'); say('Resume'); return;
        case 'menu': status('There is no menu screen yet.'); return;
        case 'calibrate': hooks.recenter(); say('Recentering'); return;
        case 'delete_voice_data': status('Lite mode saves no voice data.'); return;
        case 'volume_up': case 'volume_down': case 'mute': case 'brightness_up': case 'brightness_down':
          hooks.act({ action: 'system', name: n }); return;
        case 'close_window': ask('Close this window?', () => hooks.act({ action: 'key', key: 'alt+f4' })); return;
        case 'go_back': hooks.act({ action: 'key', key: 'alt+left' }); say('Back'); return;
        case 'escape': hooks.act({ action: 'key', key: 'esc' }); return;
        case 'type_text': S.chunks = [cmd.label]; showPreview(); return;
        case 'target': {
          const hit = hooks.find(cmd.label);
          if (!hit) { emit({ type: 'target_none', label: cmd.label }); return; }
          emit({ type: 'target_found', label: hit.label, x: 0, y: 0 });
          ask(`Click ${hit.label}?`, () => hit.click()); return;
        }
        default:
          if (P.KEY_SHORTCUTS[n]) { hooks.act({ action: 'key', key: P.KEY_SHORTCUTS[n] }); say(n.replace(/^key_/, '').replace(/_/g, ' ')); return; }
          status(`“${raw}” is not available in lite mode.`);
      }
    }

    /* a finished utterance */
    function heard(text) {
      live('');
      let t = String(text || '').trim(); if (!t) return;
      let forced = false;
      const m = /^\W*(venus|computer)\b\W*/i.exec(t);                    // "Venus, scroll down": always a command
      if (m) { t = t.slice(m[0].length); forced = true; }
      const clean = collapse(normalize(t)); if (!clean) return;
      if (S.mode === 'paused') {                                          // only resume and quit
        const c = parse(t, false); if (c && (c.name === 'resume' || c.name === 'quit_app')) run(c, t); return;
      }
      const waiting = S.q || S.chunks.length;
      const ans = waiting ? answerOf(t) : null;                           // "know" is a no, but only while something waits
      if (ans) { answer(ans === 'yes'); return; }
      if (S.mode === 'scroll') {
        if (SCROLL_EXIT.includes(clean)) { leaveScroll('Scroll off'); return; }
        const c = parse(t, false);
        if (c && (c.name === 'scroll_up' || c.name === 'scroll_down')) { run(c, t); return; }
        if (c && c.name !== 'type_text' && c.name !== 'scroll_mode') { leaveScroll(); run(c, t); return; }
        return;                                                           // anything else is not for us while scrolling
      }
      if (S.mode === 'dictation' && !forced) {
        if (P.DICTATION_EXIT.includes(clean)) { run({ name: 'stop_dictation' }, t); return; }
        const c = parse(t, false);                                        // fixed commands win; "click ..." works while dictating
        if (c && c.name !== 'type_text') { run(c, t); return; }
        const cl = clean.startsWith('click ') ? target(clean) : null;
        if (cl) { run(cl, t); return; }
        S.chunks.push(t.replace(/\s+/g, ' ')); showPreview(); return;      // everything else is words to type
      }
      let cmd = parse(t, false);                                          // pointer mode: fixed commands, then open/go to/search, then look-and-say
      if (cmd && cmd.name !== 'type_text') { run(cmd, t); return; }
      const l = launch(t);
      if (l) { hooks.act({ action: 'launch', ...l }); flash(l.app ? 'Open ' + l.app : 'Open ' + l.url.replace(/^https:\/\//, '')); return; }
      cmd = parse(t, true);
      if (cmd) { run(cmd, t); return; }
      status(`Heard “${t}”. Not a command: say “hi Dizzy” to dictate.`);
    }
    return { heard, answer, live, insert, dropPreview, state: S, parse, launch, ask, enterScroll, leaveScroll,
             pending: () => !!S.q, reset() { closeQ('cancelled'); S.chunks = []; if (S.mode === 'scroll' && hooks.scroll) hooks.scroll(false); S.mode = 'pointer'; } };
  }

  if (typeof window === 'undefined' || typeof document === 'undefined') {            // node (the unit tests)
    if (typeof module !== 'undefined') module.exports = { createVoice, parse, launch, normalize, answerOf, ratio, tidy };
    return;
  }
  const D = window.DZ; if (!D || !D.bridge) return;
  const { $, log, toast, screen } = D, B = D.bridge;

  /* ---------------- the browser side: microphone, buttons, finding things on the simulated screen ---------------- */
  const elLabel = el => (el.getAttribute('aria-label') || (el.querySelector('b') && el.querySelector('b').textContent) || el.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 40);
  /* Things you could really click right now: shown, on the screen, and not hidden under a window (on a small screen the desktop
     icons sit in several columns and an open window covers some of them; a click on a covered icon would hit the window). */
  function visibleActs() {
    return [...screen.querySelectorAll('[data-act]')].filter(e => {
      if (!e.offsetParent) return false;
      const r = e.getBoundingClientRect(); if (!r.width || !r.height) return false;
      if (e.closest('.dz-card')) return true;                                          // the card's own buttons
      const t = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
      return !!t && (t === e || e.contains(t));
    });
  }
  function clickThroughCard(x, y, button, how) {                                      // the question card must not catch the click meant for the screen
    const card = screen.querySelector('.dz-card'), was = card ? card.style.visibility : '';
    if (card) card.style.visibility = 'hidden';
    try { return D.clickAt(x, y, button, how); } finally { if (card) card.style.visibility = was; }
  }
  function centerOf(el) { const r = el.getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; }
  function find(label) {                                                              // look-and-say, by label, on the simulated screen
    const want = normalize(label); if (!want) return null;
    let best = null, bs = 0;
    for (const el of visibleActs()) {
      const l = normalize(elLabel(el)); if (!l) continue;
      const s = l === want ? 100 : (l.includes(want) || (l.length >= 3 && want.includes(l))) ? 90 : ratio(l, want);   // a one-letter key ("i") is not a match for "mail"
      if (s > bs) { bs = s; best = el; }
    }
    if (!best || bs < 70) return null;
    return { label: elLabel(best), click: () => { const c = centerOf(best); clickThroughCard(c.x, c.y, 'left', 'Voice: click “' + elLabel(best) + '”'); } };
  }
  function pointerClick(button, snap) {
    const r = screen.getBoundingClientRect(); let x = r.left + D.G.pos.x, y = r.top + D.G.pos.y;
    if (snap && !D.hitTest()) {                                                        // the nearest button within about 150 px, like the engine's snap click
      const lim = 150 * (r.width / 1920 * 1.4 + 0.3); let best = null, bd = lim;
      for (const el of visibleActs()) { const b = el.getBoundingClientRect(); const d = Math.hypot(Math.max(b.left - x, 0, x - b.right), Math.max(b.top - y, 0, y - b.bottom)); if (d < bd) { bd = d; best = el; } }
      if (best) { const c = centerOf(best); x = c.x; y = c.y; }
    }
    D.clickAt(x, y, button, button === 'right' ? 'Voice: right click' : 'Voice: click');
  }

  const voice = createVoice({
    emit: m => B.handle(m),
    act: m => B.onAction(m),
    pointerClick, find,
    kindFor: text => { const f = B.fieldFor && B.fieldFor(text); return f === 'body' ? 'body' : f === 'subj' ? 'subj' : 'plain'; },
    recenter: () => { if (window.DZLite) window.DZLite.core.recenter(); },
    scroll: on => !!(window.DZLite && window.DZLite.scroll && window.DZLite.scroll(on)),
    quit: () => { stop(); if (window.DZLite) window.DZLite.stop('Voice and lite mode are off.'); },
  });
  B.setSend((type, f) => {                                                    // the card's buttons: Yes / No / Type it / Drop it / Use original
    if (type === 'confirm_answer') voice.answer(!!f.yes);
    else if (type === 'dictation') { if (f.action === 'insert' || f.action === 'use_original') voice.insert(); else if (f.action === 'discard') voice.dropPreview(); }
  });
  window.DZVoice = { voice, start: () => start(), stop: () => stop(), pending: () => voice.pending(), waiting: () => !!(voice.state.q || voice.state.chunks.length), answer: y => voice.answer(y) };

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let rec = null, on = false, starting = false, local = false, restartT = 0, speaking = false, fails = 0, lastStart = 0, lastErr = '';
  const btn = document.createElement('button');
  btn.className = 'mini-btn'; btn.type = 'button'; btn.id = 'liteVoiceBtn'; btn.textContent = 'Turn on voice (commands and dictation)';
  const vstat = document.createElement('div');                                    // what the microphone is doing, in words (errors are never silent)
  vstat.className = 'lite-stat'; vstat.id = 'liteVoiceStat'; vstat.setAttribute('aria-live', 'polite');
  const note = document.createElement('p');
  note.textContent = 'Voice uses your browser’s speech recognition (Chrome or Edge); on Chrome it may send audio to Google unless on-device recognition is available. The EYEable app uses Whisper on your own PC. Try: “volume up”, “brightness down”, “open YouTube”, “search for cats”, “click Mail”, “hi Dizzy”, then speak, then “yes”.';
  const host = document.querySelector('.lite') || $('#camBox').parentElement;
  const anchor = host.querySelector('.lite-stat') || null;
  host.insertBefore(btn, anchor); host.insertBefore(vstat, anchor); (host.querySelector('#liteInfo') || host).append(note);
  const setV = t => { vstat.textContent = t; };
  if (!SR) { btn.disabled = true; btn.textContent = 'Voice needs Chrome or Edge'; setV('This browser has no speech recognition. Use Google Chrome or Microsoft Edge on a computer.'); }

  // the real engine connected (lite mode's head pointer also sets D.engineLive, so look at the engine pill instead)
  const engineUp = () => { const p = document.getElementById('engPill'); return !!(p && !p.classList.contains('off')); };
  const ERR = {
    'not-allowed': 'The microphone is blocked for this page. Click the microphone or lock icon in the address bar, allow the microphone, then turn voice on again.',
    'service-not-allowed': 'This browser does not allow its speech service here. Use Google Chrome or Microsoft Edge (Brave, Opera and Firefox do not support it).',
    'audio-capture': 'No microphone was found, or Windows is blocking it. Check Settings › Privacy › Microphone, and close other apps that use it (the EYEable app).',
    'network': 'The browser’s speech service could not be reached. Online recognition needs internet.',
  };
  const FATAL = new Set(['not-allowed', 'service-not-allowed', 'audio-capture']);

  async function micCheck() {                                                       // ask for the microphone once, so a blocked or missing mic says so
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return null;
    try { const s = await navigator.mediaDevices.getUserMedia({ audio: true }); s.getTracks().forEach(t => t.stop()); return null; }
    catch (e) { return (e.name === 'NotAllowedError' || e.name === 'SecurityError') ? 'not-allowed' : (e.name === 'NotFoundError' || e.name === 'NotReadableError') ? 'audio-capture' : null; }
  }
  async function onDeviceReady() {                                                  // use on-device speech only when the language pack is already installed
    try { if (SR.available) return (await SR.available({ langs: ['en-US'], processLocally: true })) === 'available'; } catch (e) { /* older browser */ }
    return false;
  }
  const listeningText = () => 'Listening' + (local ? ' (on this device)' : ' (browser’s online recognition)') + '. Say “volume up”, or “hi Dizzy” to dictate.';

  function setListening(v) { if (v !== speaking) { speaking = v; B.handle({ type: 'listening', speaking: v, level: 0.6 }); } }
  function begin() {
    rec = new SR(); rec.lang = 'en-US'; rec.continuous = true; rec.interimResults = true; rec.maxAlternatives = 1;
    if (local && 'processLocally' in rec) rec.processLocally = true;
    lastStart = performance.now();
    rec.onstart = () => { if (!fails) setV(listeningText()); };                  // keep an error message on screen while retrying
    rec.onspeechstart = () => setListening(true); rec.onspeechend = () => setListening(false);
    rec.onresult = ev => {
      let interim = '';
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const r = ev.results[i], t = r[0].transcript;
        if (r.isFinal) { fails = 0; setListening(false); setV('Heard: “' + t.trim() + '”'); log('voice', '“' + t.trim() + '”'); voice.heard(t); } else interim += t;
      }
      if (interim) { setV('Hearing: “' + interim.trim() + '”'); if (voice.state.mode === 'dictation') voice.live(interim); }
    };
    rec.onerror = e => {
      const code = e.error || 'unknown'; lastErr = code;
      if (code === 'no-speech' || code === 'aborted') return;                         // silence, or we stopped it: just listen again
      log('sys', 'Lite voice error: ' + code);
      if (code === 'language-not-supported' && local) { local = false; setV('On-device speech is not installed, so the browser’s online recognition is used (audio may go to Google).'); return; }
      if (FATAL.has(code)) { stop(ERR[code]); return; }
      setV(ERR[code] || ('Speech recognition error: ' + code + '. Retrying…'));
    };
    rec.onend = () => {                                                               // browsers end a session after a while: keep listening
      setListening(false); if (!on) return;
      const dur = performance.now() - lastStart;
      if (dur < 1500) fails++; else if (dur > 4000) fails = 0;                        // sessions that die at once are failures, long ones are normal
      if (fails >= 6) { stop('Speech recognition keeps stopping' + (lastErr ? ' (' + lastErr + ')' : '') + '. ' + (ERR[lastErr] || 'Check the microphone and use Chrome or Edge, then turn voice on again.')); return; }
      restartT = setTimeout(begin, fails ? Math.min(300 * fails, 2000) : 150);
    };
    try { rec.start(); } catch (e) { stop('Could not start speech recognition: ' + (e && e.message || e)); }
  }
  async function start() {
    if (on || starting || !SR) return;
    if (engineUp()) return toast('EYEable is running on this PC and already listens. Lite voice is not needed.');
    starting = true; btn.disabled = true; setV('Asking for the microphone…');
    const bad = await micCheck();
    if (bad) { starting = false; btn.disabled = false; setV(ERR[bad]); log('sys', 'Lite voice: microphone ' + bad); return; }
    local = await onDeviceReady();
    starting = false; btn.disabled = false;
    on = true; fails = 0; lastErr = ''; btn.textContent = 'Turn off voice'; B.setCard(true);
    setV(local ? 'Starting on-device speech…' : 'Starting the browser’s online speech recognition (audio may go to Google)…');
    begin();
    log('sys', 'Lite voice on (' + (local ? 'on-device' : 'online') + ' recognition): say “hi Dizzy” to dictate, or a command such as “volume up”');
  }
  function stop(why) {
    if (!on) return; on = false; clearTimeout(restartT);
    try { rec.onend = null; rec.abort(); } catch (e) { /* ignore */ }
    setListening(false); voice.reset(); B.handle({ type: 'dictation_live', text: '' }); B.handle({ type: 'mode', mode: 'pointer' });
    B.setCard(false); btn.textContent = 'Turn on voice (commands and dictation)';
    setV(why || 'Voice off.'); log('sys', why ? 'Lite voice off: ' + why : 'Lite voice off');
  }
  btn.addEventListener('click', () => (on ? stop() : start()));
  // the engine took over: this page is now the real thing
  setInterval(() => { if (on && engineUp()) { stop('EYEable connected: using the real engine instead.'); toast('EYEable connected: using the real engine instead.'); } }, 1000);
})();

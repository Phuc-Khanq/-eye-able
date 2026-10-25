/* js/desktop_actions.js: makes the virtual Windows screen react to what the engine (or the browser tracker) does.
   Gestures: long blink = left click at the pointer, mouth open = right click, head tilt = scroll.
   Voice: scroll, page up/down, back, forward, refresh, new/close tab, minimize, maximize, show desktop, switch window,
          zoom, copy/paste/select all, enter/tab/backspace, escape, volume and brightness, "click <name>".
   Dictation: the words you type with "Type it", or with an eyebrow raise, blink or nod while they wait, go into the focused box.
   The engine only tells the page what it hears and decides (messages.py), not what it types or clicks. So most of this is
   worked out from those messages. If the engine ever sends {"type": "os_action", ...} (action: click, type, key, scroll, launch,
   system), those exact actions are used instead and the guesses switch off. See the end of this file for the fields. */
(() => {
  const D = window.DZ; if (!D) return;
  const { $, log, toast, W, G } = D;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const screenEl = D.screen;

  const S = { mode: 'pointer', preview: null, question: false, exact: false, target: null, lastTyped: '', clip: '',
              vol: 50, bri: 60, muted: false, zoom: 1, scrollHold: 0 };

  /* ---------- small helpers ---------- */
  const windowsOpen = () => ['browser', 'mail'].filter(k => W[k].open && !W[k].min);
  const topWin = () => W.top && W[W.top].open ? W.top : (W.browser.open ? 'browser' : (W.mail.open ? 'mail' : null));
  const winEl = k => $(k === 'browser' ? '#winBrowser' : '#winMail');

  function osd(icon, text, pct) {
    let o = $('#osd');
    if (!o) { o = document.createElement('div'); o.id = 'osd'; o.className = 'osd'; o.hidden = true; screenEl.append(o); }
    o.innerHTML = `<span>${icon}</span><b>${text}</b>` + (pct == null ? '' : `<span class="bar"><i style="width:${pct}%"></i></span>`);
    o.hidden = false; clearTimeout(o._t); o._t = setTimeout(() => { o.hidden = true; }, 1600);
  }

  function rightClick() {
    const x = G.pos.x + 6, y = G.pos.y + 6;
    D.ripple('shake', G.pos); D.openCtx(x, y); log('head', 'Right click');
  }

  function typeIntoFocus(text) {
    if (!text) return;
    if (!W.focus) { toast('Click a text box first (a long blink on it), then dictate'); log('sys', 'Nothing to type into: click a text box first'); return; }
    D.insertText(text, true); S.lastTyped = text; D.renderAll(); log('voice', 'Typed “' + text.trim() + '”');
  }

  function undoTyped() {
    const f = W.focus; if (!f || !S.lastTyped) return toast('Nothing to undo');
    const cur = W.f[f], t = S.lastTyped.trim();
    if (cur.trimEnd().endsWith(t)) { W.f[f] = cur.slice(0, cur.lastIndexOf(t)).replace(/\s+$/, ' '); S.lastTyped = ''; D.renderAll(); log('voice', 'Undid the last typing'); }
  }

  function closeMenus() {
    $('#ctx').hidden = true; W.startOpen = false; if (W.focus && ['gq', 'yq', 'to', 'subj', 'body'].includes(W.focus)) W.focus = null; D.renderAll();
  }

  function scroll(dir, amt) { D.scrollBy(dir, amt); }

  function switchWindow() {
    const open = ['browser', 'mail'].filter(k => W[k].open);
    if (open.length < 2) return toast('Only one window is open');
    const other = open.find(k => k !== topWin()) || open[0];
    W[other].min = false; W[other].z = ++W.zc; W.top = other; D.renderAll();
  }

  function showDesktop() {
    const open = ['browser', 'mail'].filter(k => W[k].open);
    const allMin = open.length && open.every(k => W[k].min);
    open.forEach(k => { W[k].min = !allMin; }); W.top = allMin ? (open[0] || null) : null; D.renderAll();
  }

  function zoom(step) {
    S.zoom = clamp(+(S.zoom + step).toFixed(2), 0.6, 1.8);
    const inner = $('#bPage'); if (inner) inner.style.zoom = S.zoom; osd('🔍', Math.round(S.zoom * 100) + '%');
  }

  function system(action) {
    if (action === 'volume_up') { S.muted = false; S.vol = clamp(S.vol + 10, 0, 100); osd('🔊', 'Volume ' + S.vol, S.vol); }
    else if (action === 'volume_down') { S.vol = clamp(S.vol - 10, 0, 100); osd(S.vol ? '🔉' : '🔈', 'Volume ' + S.vol, S.vol); }
    else if (action === 'mute') { S.muted = !S.muted; osd(S.muted ? '🔇' : '🔊', S.muted ? 'Muted' : 'Volume ' + S.vol, S.muted ? 0 : S.vol); }
    else if (action === 'brightness_up') { S.bri = clamp(S.bri + 10, 0, 100); osd('☀', 'Brightness ' + S.bri, S.bri); }
    else if (action === 'brightness_down') { S.bri = clamp(S.bri - 10, 0, 100); osd('☀', 'Brightness ' + S.bri, S.bri); }
  }

  function launch(a) {
    const app = (a.app || '').toLowerCase(), url = (a.url || '').toLowerCase();
    if (/youtube/.test(app + url)) D.run('open:youtube', 'voice');
    else if (/mail|outlook|gmail/.test(app + url)) D.run('open:mail', 'voice');
    else if (/chrome|browser|edge|google/.test(app + url) || url) D.run('open:browser', 'voice');
    else toast('Opening ' + (a.app || a.url));
  }

  /* ---------- "click <name>": find the thing on the virtual screen ---------- */
  const norm = t => String(t || '').toLowerCase().replace(/[^a-z0-9 ]/g, ' ').replace(/\s+/g, ' ').trim();
  function findByName(name) {
    const q = norm(name); if (!q) return null;
    const words = q.split(' ');
    let best = null, bestScore = 0;
    for (const el of screenEl.querySelectorAll('[data-act]')) {
      if (!el.offsetParent) continue;
      const l = norm(D.label(el)); if (!l) continue;
      let score = l === q ? 4 : (l.includes(q) || q.includes(l)) ? 3 : words.some(w => w.length > 2 && l.split(' ').includes(w)) ? 1 : 0;
      if (score > bestScore) { best = el; bestScore = score; }
    }
    return best;
  }
  function hover(el) {
    const sr = screenEl.getBoundingClientRect(), r = el.getBoundingClientRect();
    G.target.x = r.left - sr.left + r.width / 2; G.target.y = r.top - sr.top + r.height / 2;
  }
  function clickTarget(el, src) { if (!el) return; D.ripple('nod', G.pos); D.run(el.dataset.act, src || 'voice'); }

  /* ---------- exact actions (os_action) and the guesses behind them ---------- */
  function perform(a) {
    switch (a.action) {
      case 'click': if (a.button === 'right') rightClick(); else D.click(); break;
      case 'type': typeIntoFocus(a.text); break;
      case 'key': {
        const k = a.key; if (k === 'enter') D.typeKey('enter'); else if (k === 'backspace') D.typeKey('bksp'); else if (k === 'tab') D.typeKey('space');
        D.renderAll(); break;
      }
      case 'scroll': scroll(a.direction === 'up' ? -1 : 1); break;
      case 'launch': launch(a); break;
      case 'system': system(a.action_name || a.system); break;
    }
  }

  const BY_LABEL = {
    'Scroll up': () => scroll(-1), 'Scroll down': () => scroll(1),
    'Page up': () => scroll(-1, 420), 'Page down': () => scroll(1, 420), 'Top': () => scroll(-1, 1e5), 'Bottom': () => scroll(1, 1e5),
    'Back': () => D.run('back', 'voice'), 'Forward': () => D.run('fwd', 'voice'), 'Refresh': () => D.run('reload', 'voice'),
    'New tab': () => { D.run('open:browser', 'voice'); D.run('nav:home', 'voice'); }, 'New window': () => D.run('open:browser', 'voice'),
    'Close tab': () => { if (W.browser.open) D.run('close:browser', 'voice'); }, 'Reopen tab': () => D.run('open:browser', 'voice'),
    'Minimize': () => { const t = topWin(); if (t) D.run('min:' + t, 'voice'); },
    'Maximize': () => { const t = topWin(); if (t) winEl(t).classList.toggle('max'); },
    'Show desktop': showDesktop, 'Switch window': switchWindow,
    'Next tab': switchWindow, 'Previous tab': switchWindow,
    'Zoom in': () => zoom(0.1), 'Zoom out': () => zoom(-0.1),
    'Copy': () => D.ctxAction('copy', 'voice'), 'Paste': () => D.ctxAction('paste', 'voice'),
    'Cut': () => { if (W.focus) { S.clip = W.f[W.focus]; W.f[W.focus] = ''; D.renderAll(); toast('Cut'); } },
    'Select all': () => toast('Selected all'), 'Save': () => toast('Saved (Ctrl+S)'), 'Find': () => toast('Find (Ctrl+F)'),
    'Enter': () => perform({ action: 'key', key: 'enter' }), 'Tab': () => perform({ action: 'key', key: 'tab' }),
    'Backspace': () => perform({ action: 'key', key: 'backspace' }), 'Undo': undoTyped, 'Escape': closeMenus,
    'Volume up': () => system('volume_up'), 'Volume down': () => system('volume_down'), 'Mute': () => system('mute'),
    'Brightness up': () => system('brightness_up'), 'Brightness down': () => system('brightness_down'),
  };

  function previewText() { return S.preview ? S.preview.map(w => w.text).join(' ') : ''; }

  function fromGesture(name) {   // engine gesture messages and the browser tracker both end up here
    if (S.exact) return;
    const waiting = !!(S.preview || S.question);
    if (S.mode === 'paused' || S.mode === 'menu') return;
    if (waiting) {   // a yes for the waiting preview or question: the engine types the preview
      if (['nod', 'brow_raise', 'blink'].includes(name) && S.preview && !S.question) { typeIntoFocus(previewText()); S.preview = null; }
      return;
    }
    if (S.mode === 'pointer' || S.mode === 'dictation') {
      if (name === 'blink') D.click();
    }
    if (S.mode === 'pointer') {
      if (name === 'mouth_open') rightClick();
      else if (name === 'tilt_left') scroll(-1, 120);
      else if (name === 'tilt_right') scroll(1, 120);
    }
  }

  function fromEngine(m) {
    switch (m.type) {
      case 'os_action': S.exact = true; perform(m); return;
      case 'mode': S.mode = m.mode; return;
      case 'gesture': fromGesture(m.gesture); return;
      case 'dictation_preview': S.preview = (m.words && m.words.length) ? m.words : null; return;
      case 'scroll':   // scroll mode: the head tilt sets a rate in notches per second, about 10 messages a second
        if (!S.exact && m.ready && m.rate) { S.scrollHold += Math.abs(m.rate) * 0.1; if (S.scrollHold >= 0.25) { scroll(m.rate > 0 ? -1 : 1, S.scrollHold * 90); S.scrollHold = 0; } }
        return;
      case 'command': if (!S.exact && BY_LABEL[m.label]) BY_LABEL[m.label](); return;
      case 'target_found':
        if (m.snap) { if (!S.exact) D.click(); return; }
        S.target = findByName(m.label); if (S.target) { hover(S.target); S.target.classList.add('hot'); } else toast('“' + m.label + '” isn’t on the virtual screen');
        return;
      case 'target_none': if (S.target) S.target.classList.remove('hot'); S.target = null; return;
      case 'confirm':
        S.question = m.state === 'open';
        if (m.state === 'confirmed') {
          if (S.target) { const t = S.target; S.target = null; t.classList.remove('hot'); clickTarget(t); }
          else if (/^send\??$/i.test(m.prompt) && W.mail.open && W.mail.view === 'compose') D.run('send', 'voice');
        } else if (m.state !== 'open' && S.target) { S.target.classList.remove('hot'); S.target = null; }
        return;
    }
  }

  /* the page's own "Type it" / "Drop it" buttons also type into the virtual box */
  function typedFromButton() { typeIntoFocus(previewText()); S.preview = null; }

  D.act = { fromEngine, fromGesture, perform, typedFromButton, state: S };
})();

/* Optional exact feed, if the engine ever relays its OS actions to the UI (one new message in engine/core/messages.py, sent
   from os_output's topics): {"type": "os_action", "action": "click", "button": "left"|"right"}, {"action": "type", "text": str},
   {"action": "key", "key": "enter"|"backspace"|...}, {"action": "scroll", "direction": "up"|"down"},
   {"action": "launch", "app": str} or {"url": str}, {"action": "system", "system": "volume_up"|...}. */

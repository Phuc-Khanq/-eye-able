/* js/browser_voice.js: voice commands and dictation inside the page, no Python. Uses the browser's own speech recognition
   (Chrome or Edge, needs internet; the engine's fine-tuned Whisper for hard-to-understand speech is a separate thing and needs the engine).
   Same ideas as the engine: always listening, commands anywhere, "dictate" starts dictation, words wait as a preview until you
   say yes, press Type it, or raise your eyebrows / blink / nod (js/desktop_actions.js handles the gestures).
   Starts with Allow camera. While the engine is connected it stops, because the engine listens and decides. */
(() => {
  const D = window.DZ, SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!D) return;
  const { $, log, toast } = D;
  const A = () => D.act, S = () => D.act.state;
  const SHAKY_CONF = 0.6;
  const statusEl = document.querySelector('.voice-card .card-h .label');

  let rec = null, on = false, mode = 'pointer', lastPreviewKey = '', failed = 0;

  const norm = t => String(t).toLowerCase().replace(/[.,!?;:"“”]/g, ' ').replace(/\b(please|um|uh|erm)\b/g, ' ').replace(/\s+/g, ' ').trim();
  const setStatus = t => { if (statusEl) statusEl.textContent = t; };

  function setMode(m) {
    mode = m; S().mode = m;
    const pill = $('#trackTxt'); if (pill && !D.engineLive) pill.textContent = { pointer: 'Tracking', dictation: 'Dictation' }[m] || m;
    log('sys', 'Mode: ' + ({ pointer: 'Tracking', dictation: 'Dictation' }[m] || m));
    if (m === 'pointer' && !S().preview) renderPreview();
  }

  /* ---------- the preview panel (same look as the engine's) ---------- */
  function renderPreview() {
    const words = S().preview, key = words ? words.map(w => w.text).join(' ') : '';
    if (key === lastPreviewKey) return; lastPreviewKey = key;
    const meant = $('#meantTxt'); meant.textContent = '';
    if (!words) { meant.textContent = '–'; return; }
    meant.append('Will type: ');
    words.forEach((w, i) => {
      const sp = document.createElement('span'); sp.className = 'eng-w' + (w.conf < SHAKY_CONF ? ' shaky' : ''); sp.textContent = (i ? ' ' : '') + w.text; meant.append(sp);
    });
    const act = document.createElement('span'); act.className = 'eng-act';
    act.innerHTML = '<button class="ok">Type it</button><button class="no">Drop it</button>';
    act.querySelector('.ok').onclick = () => A().typedFromButton();
    act.querySelector('.no').onclick = () => { S().preview = null; };
    meant.append(act);
  }
  setInterval(() => { if (on) renderPreview(); }, 250);   // the preview also clears from gestures and "yes"

  /* ---------- commands ---------- */
  const LABELS = [   // spoken phrase -> the label js/desktop_actions.js knows
    [/^scroll up$/, 'Scroll up'], [/^scroll down$/, 'Scroll down'], [/^page up$/, 'Page up'], [/^page down$/, 'Page down'],
    [/^(go to )?(the )?top$/, 'Top'], [/^(go to )?(the )?bottom$/, 'Bottom'],
    [/^(go )?back$/, 'Back'], [/^(go )?forward$/, 'Forward'], [/^(refresh|reload)( page)?$/, 'Refresh'],
    [/^new tab$/, 'New tab'], [/^new window$/, 'New window'], [/^close (the )?(tab|window)$/, 'Close tab'],
    [/^minimi[sz]e( window)?$/, 'Minimize'], [/^maximi[sz]e( window)?$/, 'Maximize'], [/^show (the )?desktop$/, 'Show desktop'],
    [/^switch (the )?window$/, 'Switch window'], [/^zoom in$/, 'Zoom in'], [/^zoom out$/, 'Zoom out'],
    [/^copy( that)?$/, 'Copy'], [/^paste( it| that)?$/, 'Paste'], [/^cut( that)?$/, 'Cut'], [/^select all$/, 'Select all'],
    [/^save( it)?$/, 'Save'], [/^find$/, 'Find'], [/^enter$/, 'Enter'], [/^tab$/, 'Tab'], [/^backspace$/, 'Backspace'],
    [/^escape$/, 'Escape'], [/^volume up$/, 'Volume up'], [/^volume down$/, 'Volume down'], [/^mute$/, 'Mute'],
    [/^brightness up$/, 'Brightness up'], [/^brightness down$/, 'Brightness down'],
  ];
  const YES = /^(yes|yeah|yep|yup|confirm|do it|go ahead|write it|write that|type it|type that|insert it|put it in|keep it)$/;
  const NO = /^(no|nope|nah|drop it|discard( it)?|scrap (it|that)|throw it away|not that|never mind|cancel)$/;

  function previewWords(text, conf) { return text.split(' ').filter(Boolean).map(w => ({ text: w, conf })); }
  function addPreview(text, conf) {
    const cur = S().preview || [];
    S().preview = cur.concat(previewWords(text, conf)); renderPreview();
  }

  function heard(raw) { $('#heardTxt').textContent = '“' + raw + '”'; }
  function understood(txt) { $('#meantTxt').textContent = txt; lastPreviewKey = '~'; }

  function handle(raw, conf) {
    const t = norm(raw); if (!t) return;
    heard(raw);
    const waiting = !!S().preview;

    // answers
    if (YES.test(t)) { if (waiting) { A().typedFromButton(); } else understood('Nothing is waiting for a yes'); return; }
    if (NO.test(t) && (waiting || t === 'cancel' || t === 'never mind')) { if (waiting) S().preview = null; else A().byLabel('Escape'); return; }
    if (/^(scratch that|undo( that)?)$/.test(t)) { if (waiting) S().preview = null; else A().byLabel('Undo'); return; }

    // modes
    if (/^(start )?dictat(e|ion)( on)?$|^start typing$|^hi dizz?(y|ie|i)$|^high dizz?y$/.test(t)) { setMode('dictation'); understood('Dictation on: say what to type'); return; }
    if (/^(stop|end|exit|close|turn off) (dictation|dictating|typing)$|^dictat(e|ion) (off|of)$|^done typing$|^bye( bye)?( dizz?(y|ie|i))?$/.test(t)) { setMode('pointer'); understood('Dictation off'); return; }
    let m;
    if ((m = t.match(/^(?:dictate|type) (.+)$/))) { addPreview(m[1], conf); return; }   // one-shot
    if (/^send( it| message)?$/.test(t)) {
      if (waiting) A().typedFromButton();
      if (D.W.mail.open && D.W.mail.view === 'compose') D.run('send', 'voice'); else A().perform({ action: 'key', key: 'enter' });
      return;
    }
    if (/^new line|next line$/.test(t)) { A().perform({ action: 'key', key: 'enter' }); return; }

    // virtual-screen commands (by label)
    for (const [re, label] of LABELS) if (re.test(t)) { log('voice', '“' + raw + '” → ' + label); understood(label); A().byLabel(label); return; }
    if (/^right click$/.test(t)) { understood('Right click'); A().perform({ action: 'click', button: 'right' }); return; }
    if (/^click( it| that| this| here| there)?$/.test(t)) { understood('Click'); A().perform({ action: 'click' }); return; }
    if ((m = t.match(/^(?:click|select|press|tap) (.+)$/))) {
      if (A().clickByName(m[1])) { log('voice', '“' + raw + '” → click ' + m[1]); understood('Click ' + m[1]); } else { understood('I can’t find “' + m[1] + '” on this screen'); toast('I can’t find “' + m[1] + '”'); }
      return;
    }
    if (/^(pause|stop listening|take a break)$/.test(t)) { $('#pauseBtn').click(); return; }

    // open apps, search, play, write an email (the page's own command grammar)
    if (D.tryIntent(raw)) return;

    // anything else: dictated words in dictation mode; in Pointer mode it is ignored, like the engine
    if (mode === 'dictation') { addPreview(t, conf); understood(''); renderPreview(); return; }
    understood('Not a command: say “dictate” to type');
  }

  /* ---------- speech recognition, always on ---------- */
  function start() {
    if (on || D.engineLive) return;
    if (!SR) { setStatus('not supported here'); return toast('Voice needs Chrome or Edge. Commands still work by typing them under the mic.'); }
    on = true; D.voiceOn = true; failed = 0; setStatus('listening'); S().mode = mode;
    log('sys', 'Voice on: say “dictate”, “scroll down”, “click Mail”… The browser may ask for the microphone.');
    listen();
  }
  function listen() {
    if (!on) return;
    rec = new SR(); rec.lang = 'en-US'; rec.continuous = true; rec.interimResults = true; rec.maxAlternatives = 1;
    rec.onspeechstart = () => { D.micLive = true; };
    rec.onspeechend = () => { D.micLive = false; };
    rec.onresult = ev => {
      let interim = '';
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const r = ev.results[i], alt = r[0];
        if (r.isFinal) { failed = 0; handle(alt.transcript.trim(), alt.confidence || 0.85); }
        else interim += alt.transcript;
      }
      if (interim) $('#heardTxt').textContent = '… ' + interim.trim();
    };
    rec.onerror = e => {
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') { on = false; D.voiceOn = false; setStatus('microphone blocked'); toast('Microphone is blocked for this page. Allow it in the address bar, or open the page through http://localhost.'); }
      else if (e.error === 'network') { failed++; setStatus('needs internet'); }
    };
    rec.onend = () => { D.micLive = false; if (on) setTimeout(listen, failed > 3 ? 3000 : 250); };   // Chrome ends the session after a while: start again
    try { rec.start(); } catch (e) { setTimeout(listen, 1000); }
  }
  function stop() {
    if (!on) return;
    on = false; D.voiceOn = false; D.micLive = false; setStatus('always listening');
    try { rec && rec.abort(); } catch (e) {} rec = null; S().preview = null; setMode('pointer');
  }

  D.stopVoice = stop;
  // Allow camera is the one click that starts everything: face tracking (js/browser_tracker.js) and voice
  document.addEventListener('click', e => { if (e.target.closest('#allowCam, #demoCamBtn') && !D.engineLive) start(); });
})();

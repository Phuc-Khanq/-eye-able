/* js/try-guide.js: the "Try it" list under the camera on the live demo page.
   Each row is a situation to try; pressing it opens the steps and the exact words to say, one row open at a time.
   Every phrase here is one lite mode really understands (tests/lite_try_guide_check.js says each one out loud).
   This file is ASCII only on purpose: quotes and dashes are HTML entities. */
(() => {
  'use strict';
  const rail = document.querySelector('#demo .rail');
  const camCard = rail && rail.querySelector(':scope > .card');
  if (!rail || !camCard || document.getElementById('tryGuide')) return;

  const say = t => '<span class="try-say">&ldquo;' + t + '&rdquo;</span>';
  const ITEMS = [
    { t: 'Get started', tag: 'head + blink', steps: [
      'Press <b>Start hands-free</b> above and allow the camera and microphone. Chrome or Edge work best.',
      'Sit about an arm&rsquo;s length from the screen, with light on your face.',
      'Turn your head gently to move the ring. Small turns aim, quick turns go further.',
      'To click, close your eyes for about half a second. Normal blinks never click.',
      'Ring drifted away? Say ' + say('recenter') + '.'] },
    { t: 'Watch a video', tag: 'voice', steps: [
      'Say ' + say('open YouTube') + '.',
      'Say ' + say('play video two') + ' (any number from one to eight).',
      'Say ' + say('pause video') + ', then ' + say('play video') + '.',
      'Say ' + say('like this video') + ', then ' + say('unlike the video') + ' to take it back.',
      'Say ' + say('go home') + ' to go back to the start page.'] },
    { t: 'Search and scroll', tag: 'voice + head', steps: [
      'Say ' + say('search for cats and dogs') + ' or ' + say('search YouTube for piano music') + '.',
      'Say ' + say('scroll') + '. The ring turns dashed and stays put.',
      'Lower your chin to scroll down, lift it to scroll up. Further means faster.',
      'Say ' + say('stop') + ' when you are done.'],
      tip: 'No camera? ' + say('scroll down') + ' and ' + say('scroll up') + ' move one step.' },
    { t: 'Send an email', tag: 'dictation', steps: [
      'Say ' + say('new email') + '. The cursor starts in the To box.',
      'Say ' + say('hi Dizzy') + ' to type by voice, then a name like &ldquo;Anna&rdquo;, then ' + say('yes') + '.',
      'Say ' + say('click message') + ', then ' + say('yes') + '.',
      'Speak your sentence. The card shows the words first: say ' + say('yes') + ' to type them, or ' + say('no') + ' to drop them.',
      'Say ' + say('stop dictation') + ', then ' + say('click send') + ' and ' + say('yes') + '.'] },
    { t: 'Click by name', tag: 'voice', steps: [
      'Say &ldquo;click&rdquo; and a name you can see, like ' + say('click Mail') + '.',
      '[eye]able asks first. Say ' + say('yes') + ' or ' + say('no') + '.',
      'Or point the ring at it and close your eyes for half a second.',
      'Say ' + say('right click') + ' for a right click where the ring is.'] },
    { t: 'Pause and volume', tag: 'voice', steps: [
      'Say ' + say('pause') + ': nothing is clicked or typed until you say ' + say('resume') + '.',
      'Say ' + say('volume up') + ', ' + say('volume down') + ' or ' + say('mute') + '.',
      'Press <b>Stop</b> above to turn the camera and microphone off.'] }
  ];

  const css = document.createElement('style');
  css.textContent = `
  #tryGuide{flex:1 1 auto;min-height:0;overflow-y:auto;scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.18) transparent;
    display:flex;flex-direction:column;gap:8px;padding:4px 2px 6px;container-type:inline-size}
  #tryGuide .try-list,#tryGuide .try-h{flex:none}
  #tryGuide .try-h{display:flex;align-items:baseline;justify-content:space-between;gap:8px;padding:0 4px}
  #tryGuide .try-h h3{margin:0;font-size:14px;font-weight:700;color:var(--ink)}
  #tryGuide .try-h span{font-family:var(--f-mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
  #tryGuide .try-list{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:6px}
  #tryGuide .try-q{width:100%;display:grid;grid-template-columns:20px minmax(0,1fr) auto;align-items:center;gap:8px;min-height:34px;padding:5px 10px 5px 12px;
    border-radius:12px;font:inherit;color:var(--ink);text-align:left;cursor:pointer;background:var(--surface);border:1px solid rgba(169,184,146,.45);
    box-shadow:0 0 8px rgba(169,184,146,.12);transition:border-color .2s,box-shadow .2s,background .2s}
  #tryGuide .try-q:hover{border-color:rgba(205,240,190,.8)}
  #tryGuide .try-q:focus-visible{outline:2px solid #8FE36A;outline-offset:2px}
  #tryGuide .try-q.on{background:linear-gradient(90deg,rgba(143,227,106,.1),rgba(255,255,255,.03));border-color:rgba(225,250,212,.95);
    box-shadow:0 0 0 1px rgba(143,227,106,.45),0 0 12px rgba(143,227,106,.35)}
  #tryGuide .try-n{font-family:var(--f-mono);font-size:12px;font-weight:600;color:#DCE4CC}
  #tryGuide .try-t{font-weight:700;font-size:13px;line-height:1.25;overflow-wrap:anywhere}
  #tryGuide .try-tag{font-family:var(--f-mono);font-size:10px;color:#C9CBC4;padding:2px 7px;border-radius:999px;border:1px solid rgba(255,255,255,.16);
    background:rgba(255,255,255,.04);white-space:nowrap}
  #tryGuide .try-q.on .try-tag{color:#E8F5DF;border-color:rgba(143,227,106,.5);background:rgba(143,227,106,.1)}
  #tryGuide .try-a{margin:6px 0 4px 11px;padding:2px 2px 2px 12px;border-left:1px solid rgba(143,227,106,.4)}
  #tryGuide .try-a[hidden]{display:none}
  #tryGuide .try-steps{margin:0;padding:0 0 0 18px;display:flex;flex-direction:column;gap:6px;font-size:12.5px;line-height:1.45;color:var(--ink)}
  #tryGuide .try-steps li::marker{font-family:var(--f-mono);font-size:11px;color:#8FE36A}
  #tryGuide .try-say{font-family:var(--f-mono);font-size:11.5px;color:#DDF6CF;background:rgba(143,227,106,.1);border:1px solid rgba(143,227,106,.32);
    border-radius:6px;padding:0 4px;-webkit-box-decoration-break:clone;box-decoration-break:clone}
  #tryGuide .try-tip{margin:8px 0 0;font-size:11.5px;line-height:1.45;color:var(--muted)}
  /* the camera keeps its full size; on short screens the list scrolls instead of squeezing the camera */
  @media (min-width:900px){#demo .rail > .card:first-child{flex:0 0 auto}#demo .rail .cam{flex:0 0 auto;height:auto;max-height:none;aspect-ratio:4/3}#demo .rail .cam.live{aspect-ratio:16/9}}
  @container (max-width:299px){#tryGuide .try-tag{display:none}#tryGuide .try-q{grid-template-columns:20px minmax(0,1fr)}}`;
  document.head.append(css);

  const box = document.createElement('section');
  box.id = 'tryGuide';
  box.setAttribute('aria-labelledby', 'tryGuideH');
  box.innerHTML = '<div class="try-h"><h3 id="tryGuideH">Try it</h3><span>pick one</span></div><ol class="try-list">' +
    ITEMS.map((it, i) => {
      const n = String(i + 1).padStart(2, '0'), id = 'tryA' + (i + 1);
      return '<li><button class="try-q" type="button" aria-expanded="false" aria-controls="' + id + '">' +
        '<span class="try-n">' + n + '</span><span class="try-t">' + it.t + '</span><span class="try-tag">' + it.tag + '</span></button>' +
        '<div class="try-a" id="' + id + '" hidden><ol class="try-steps">' + it.steps.map(s => '<li>' + s + '</li>').join('') + '</ol>' +
        (it.tip ? '<p class="try-tip">' + it.tip + '</p>' : '') + '</div></li>';
    }).join('') + '</ol>';
  camCard.after(box);

  const list = box.querySelector('.try-list');
  list.addEventListener('click', e => {
    const q = e.target.closest('.try-q'); if (!q) return;
    const open = q.getAttribute('aria-expanded') !== 'true';
    list.querySelectorAll('.try-q').forEach(b => {
      const on = b === q && open;
      b.setAttribute('aria-expanded', String(on)); b.classList.toggle('on', on);
      document.getElementById(b.getAttribute('aria-controls')).hidden = !on;
    });
    if (open) requestAnimationFrame(() => { try { q.parentElement.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); } catch (err) {} });
  });

  /* the list now does the "Try: ..." job, so the long privacy note starts folded to leave room (one click opens it) */
  const info = document.getElementById('liteInfo');
  if (info && info.open) info.open = false;
})();

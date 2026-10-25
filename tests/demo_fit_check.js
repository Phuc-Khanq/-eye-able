// The live demo page must fit one screen on every laptop: no page scroll, the camera rail and the
// activity row inside the window, and the mail form visible above the on-screen keyboard.
// Run: node tests/demo_fit_check.js     (SHOTS=/folder saves a picture of each size)
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const root = path.join(__dirname, '..', 'eyeable-frontend', 'eyeable-frontend');
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.jpg': 'image/jpeg', '.png': 'image/png', '.svg': 'image/svg+xml', '.webp': 'image/webp' };
const srv = http.createServer((q, r) => { const f = path.join(root, decodeURIComponent(q.url.split('?')[0]).replace(/\/$/, '/index.html'));
  fs.readFile(f, (e, d) => { if (e) { r.writeHead(404); r.end(); } else { r.writeHead(200, { 'content-type': types[path.extname(f)] || 'application/octet-stream' }); r.end(d); } }); });
// browser window sizes (the screen minus tabs, address bar and taskbar) of common laptops and displays
const SIZES = [[1024, 650], [1093, 614], [1280, 560], [1280, 610], [1280, 680], [1366, 650], [1440, 780], [1512, 860], [1536, 730], [1600, 790], [1680, 940], [1728, 1000], [1920, 960], [2048, 1040], [2560, 1300]];
(async () => {
  await new Promise(r => srv.listen(8094, r));
  const b = await chromium.launch();
  const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) process.exitCode = 1; };
  for (const [w, h] of SIZES) {
    const p = await (await b.newContext({ viewport: { width: w, height: h } })).newPage(); const errs = []; p.on('pageerror', e => errs.push(e.message));
    await p.goto('http://localhost:8094/#demo'); await p.waitForTimeout(700);
    const m = await p.evaluate(() => { const R = s => { const e = document.querySelector(s); if (!e) return null; const r = e.getBoundingClientRect(); return { t: r.top, b: r.bottom, l: r.left, r: r.right, h: r.height, w: r.width }; };
      const de = document.documentElement, rail = document.querySelector('#demo .rail');
      return { ih: innerHeight, iw: innerWidth, vs: de.scrollHeight - innerHeight, hs: de.scrollWidth - innerWidth, railOver: rail.scrollHeight - rail.clientHeight,
        logH: document.querySelector('#demo .log').getBoundingClientRect().height, buddyB: document.querySelector('#demo .buddy').getBoundingClientRect().bottom, top: R('header.top'), lower: R('#demo .lower'), rail: R('#demo .rail'), screen: R('#demo .screen'), cam: R('#demo .cam'), lite: R('#demo .lite') }; });
    const tag = `${w}x${h}`;
    ok(m.vs <= 0 && m.hs <= 0, `${tag}: the page does not scroll (${m.vs} px down, ${m.hs} px across)`);
    ok(m.lower.b <= m.ih && m.rail.b <= m.ih && m.top.t >= 0, `${tag}: header, camera rail and activity row are all inside the window`);
    ok(m.railOver <= 0, `${tag}: the camera rail needs no scrolling of its own (${m.railOver} px over)`);
    ok(m.screen.h >= 300 && m.screen.w >= 600, `${tag}: the simulated screen is usable (${Math.round(m.screen.w)}x${Math.round(m.screen.h)})`);
    ok(m.cam.h >= 90 && m.cam.b <= m.rail.b && m.lite.b <= m.rail.b + 1, `${tag}: camera picture (${Math.round(m.cam.h)} px tall) and the buttons under it are inside the rail`);
    if (w >= 1200) {   // wide laptops: the screen is in the middle, Activity on its left, the camera on its right
      ok(Math.abs((m.screen.l + m.screen.r) / 2 - m.iw / 2) <= 8, `${tag}: the simulated screen is centred on the window (${Math.round((m.screen.l + m.screen.r) / 2)} vs ${Math.round(m.iw / 2)})`);
      ok(m.lower.r <= m.screen.l && m.rail.l >= m.screen.r && m.lower.w >= 200 && m.rail.w >= 200, `${tag}: Activity (${Math.round(m.lower.w)} px) sits left of the screen and the camera (${Math.round(m.rail.w)} px) right of it`);
      ok(m.logH >= 60 && m.buddyB <= m.ih, `${tag}: the Activity list has room (${Math.round(m.logH)} px) and the helper is inside the window`);
    }
    if (h >= 600) {   // mail: new message with the keyboard up (it opens with the cursor in To)
      await p.click('.dicon[data-act="open:mail"]'); await p.waitForTimeout(400); await p.click('[data-act="compose"]'); await p.waitForTimeout(400);
      await p.click('.field[data-act="focus:body"]'); await p.waitForTimeout(500);
      const k = await p.evaluate(() => { const r = s => { const e = document.querySelector(s); if (!e || e.hidden) return null; const b = e.getBoundingClientRect(); return { t: b.top, b: b.bottom }; };
        return { win: r('#winMail'), body: r('#winMail .cfield.body .field'), send: r('#winMail .sendbtn'), osk: r('#osk') }; });
      ok(k.body && k.body.b <= k.win.b + 1 && k.body.t >= k.win.t, `${tag}: with the keyboard up the Message box is fully visible`);
      if (h >= 650) ok(k.send && k.send.b <= k.win.b + 1, `${tag}: and so is the Send button`);
    }
    if (process.env.SHOTS) { fs.mkdirSync(process.env.SHOTS, { recursive: true }); await p.screenshot({ path: path.join(process.env.SHOTS, tag + '.png') }); }
    ok(errs.length === 0, `${tag}: no page errors` + (errs.length ? ': ' + errs.join(' | ') : ''));
    await p.context().close();
  }
  await b.close(); srv.close();
})().catch(e => { console.error(e); process.exit(1); });

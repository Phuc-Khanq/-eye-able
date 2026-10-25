/* js/browser_tracker.js: face tracking inside the page, so Allow camera alone works (no Python, no install).
   Uses Google's MediaPipe Face Landmarker (the model the engine uses too), loaded from a CDN, so it needs internet.
   It moves the demo screen and shows gestures in Activity. It cannot control the real computer: that is the engine's job.
   Gesture rules live in js/gesture_detect.js. While the engine is connected, js/engine.js stops this and the engine owns the webcam. */
(() => {
  const D = window.DZ, GD = window.DiseyebleGestures; if (!D || !GD) return;
  const { $, log, toast, G } = D;
  const V = '0.10.14';
  const CDNS = ['https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@' + V, 'https://unpkg.com/@mediapipe/tasks-vision@' + V];
  const MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
  const PTR_GAIN = { x: 5, y: 5.5 };   // fraction of the demo screen per fraction of the camera frame the nose moves
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  const video = $('#camVideo');
  const det = new GD.GestureDetector();
  let landmarker = null, loading = null, running = false, raf = 0, lastTime = -1, nose0 = null, lastT = 0, faceSeen = false;

  async function load() {
    if (landmarker) return landmarker;
    if (loading) return loading;
    loading = (async () => {
      let lastErr;
      for (const base of CDNS) {
        try {
          const mod = await import(base + '/vision_bundle.mjs');
          const files = await mod.FilesetResolver.forVisionTasks(base + '/wasm');
          const opts = d => ({ baseOptions: { modelAssetPath: MODEL, delegate: d }, runningMode: 'VIDEO', numFaces: 1,
            outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true });
          try { landmarker = await mod.FaceLandmarker.createFromOptions(files, opts('GPU')); }
          catch (e) { landmarker = await mod.FaceLandmarker.createFromOptions(files, opts('CPU')); }
          return landmarker;
        } catch (e) { lastErr = e; }
      }
      throw lastErr || new Error('could not load the face tracker');
    })();
    try { return await loading; } finally { loading = null; }
  }

  async function start() {
    if (running || D.engineLive) return;
    log('sys', 'Loading the face tracker…'); toast('Loading the face tracker… (needs internet)');
    try { await load(); } catch (e) {
      log('sys', 'Face tracker failed to load: ' + (e && e.message || e));
      return toast('Could not load the face tracker. It needs internet. The demo keeps its simulated face.');
    }
    if (D.engineLive || !video.srcObject) return;
    running = true; D.tracking = true; det.reset(); nose0 = null; lastTime = -1; faceSeen = false;
    log('sys', 'Browser tracking on: long blinks, eyebrow raises, nods, shakes and tilts show here. It moves this demo screen only; start the engine to control the real computer.');
    toast('Tracking on. Look at the camera.');
    raf = requestAnimationFrame(tick);
  }

  function stop() {
    if (!running) return;
    running = false; D.tracking = false; cancelAnimationFrame(raf); D.signal = null; det.reset();
  }

  function tick() {
    if (!running) return;
    raf = requestAnimationFrame(tick);
    if (video.readyState < 2 || video.currentTime === lastTime) return;
    lastTime = video.currentTime;
    const now = performance.now();
    let res; try { res = landmarker.detectForVideo(video, now); } catch (e) { return; }
    const cats = res.faceBlendshapes && res.faceBlendshapes[0] && res.faceBlendshapes[0].categories;
    const mat = res.facialTransformationMatrixes && res.facialTransformationMatrixes[0];
    const lm = res.faceLandmarks && res.faceLandmarks[0];
    if (!cats || !mat || !lm) {
      if (faceSeen) { faceSeen = false; log('sys', 'Face lost: look at the camera'); }
      det.update(now, null); D.signal = null; return;
    }
    if (!faceSeen) { faceSeen = true; log('sys', 'Face found'); }
    const b = {}; for (const c of cats) b[c.categoryName] = c.score;
    const blink = ((b.eyeBlinkLeft || 0) + (b.eyeBlinkRight || 0)) / 2;
    const brow = ((b.browInnerUp || 0) + (b.browOuterUpLeft || 0) + (b.browOuterUpRight || 0)) / 3;
    const pose = GD.poseFromMatrix(mat.data);
    for (const g of det.update(now, { blink, brow, pitch: pose.pitch, yaw: pose.yaw, roll: pose.roll })) {
      if (D.showGesture) D.showGesture(g);
      if (g === 'blink') D.click();   // a long blink is the left click, as in the engine
    }
    // real values into the signal chart and the face readout
    const open = 1 - Math.min(1, blink / 0.6);
    D.signal = { ear: 0.04 + 0.26 * open, pitch: det.dev.pitch, yaw: det.dev.yaw, roll: det.dev.roll };

    // head pointer: the nose, mirrored like the picture on screen, relative to where it usually rests
    const nose = lm[1], dt = Math.min(200, now - (lastT || now)); lastT = now;
    if (!nose0) nose0 = { x: nose.x, y: nose.y };
    const a = 1 - Math.exp(-dt / 4000);
    nose0.x += (nose.x - nose0.x) * a * 0.3; nose0.y += (nose.y - nose0.y) * a * 0.3;
    if (!G.paused) {
      const box = D.screen.getBoundingClientRect();
      G.target.x = clamp(0.5 - (nose.x - nose0.x) * PTR_GAIN.x, 0, 1) * box.width;
      G.target.y = clamp(0.5 + (nose.y - nose0.y) * PTR_GAIN.y, 0, 1) * box.height;
      G.seen = true;
    }
  }

  D.stopTracking = stop;
  video.addEventListener('playing', start);
  video.addEventListener('emptied', stop);
})();

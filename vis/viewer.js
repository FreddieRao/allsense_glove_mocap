// index-finger calibration viewer. Everything is decided in build_viewer.py; this decodes, skins and draws.
// World is Motive's (Y up), mm, re-centred on the take's median marker. Ported from TableMocap's
// scripts/marker_assignment_mosca/viewer.js (sprites, orbit camera, timeline, controls).
(function () {
  "use strict";
  var D = window.HAND_VIEW, T = D.n_frames, FREE = 255, NAN16 = -32768;

  function dec(s, Type) {
    var bin = atob(s), u = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return new Type(u.buffer);
  }
  var ptr = dec(D.ptr, Int32Array), P = dec(D.p, Int16Array), CLS = dec(D.cls, Uint8Array);
  var frameIdx = dec(D.frame_idx, Int32Array), stats = dec(D.stats, Float32Array), held = dec(D.held, Uint8Array);
  var pose = {};
  ["fixture", "knuckle", "tip"].forEach(function (r) {
    pose[r] = { mosca: dec(D.poses[r].mosca, Float32Array), motive: dec(D.poses[r].motive, Float32Array) };
  });
  var M = D.mano, vRest = dec(M.v_rest, Float32Array), faces = dec(M.faces, Uint16Array);
  var wIdx = dec(M.w_idx, Uint8Array), wts = dec(M.w, Float32Array), A = dec(M.A, Float32Array);
  var chain = dec(M.chain, Int16Array), predTip = dec(D.pred.tip, Int16Array), predFl = dec(D.pred.float, Int16Array);
  var predKn = dec(D.pred.knuckle, Int16Array), NK = D.templates.knuckle.length, NS = 4;   // stats: tip, prox, mid, knuckle
  var NV = vRest.length / 3;
  var maxK = 0;
  for (var f = 0; f < T; f++) maxK = Math.max(maxK, ptr[f + 1] - ptr[f]);
  var clsCol = D.class_colours.map(function (c) { return new THREE.Color(c); });
  var FREEC = new THREE.Color("#b4bac4");

  // ------------------------------------------------------------------------------------- sprites
  function sprite(ring) {
    var c = document.createElement("canvas"); c.width = c.height = 64;
    var g = c.getContext("2d");
    g.beginPath(); g.arc(32, 32, ring ? 26 : 28, 0, 2 * Math.PI);
    if (ring) { g.lineWidth = 7; g.strokeStyle = "#fff"; g.stroke(); } else { g.fillStyle = "#fff"; g.fill(); }
    return new THREE.CanvasTexture(c);
  }
  var DOT = sprite(false), RINGTEX = sprite(true);
  function points(n, size, tex, opacity) {
    var g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    var m = new THREE.PointsMaterial({ size: size, map: tex, vertexColors: true, transparent: true,
                                       opacity: opacity, alphaTest: 0.3, sizeAttenuation: true, depthWrite: false });
    var p = new THREE.Points(g, m); p.frustumCulled = false; return p;
  }
  function lines(n, opacity) {
    var g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 6), 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 6), 3));
    var l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true,
                                                                    opacity: opacity, depthTest: false }));
    l.frustumCulled = false; l.renderOrder = 2; return l;
  }

  // -------------------------------------------------------------------------------------- scene
  var cv = document.getElementById("cv");
  var renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1); renderer.setClearColor(0x15171c);
  var scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 0.45));
  var sun = new THREE.DirectionalLight(0xffffff, 0.8); sun.position.set(300, 800, 400); scene.add(sun);
  var minY = 1e9;
  for (var i = 1; i < P.length; i += 3) if (P[i] !== NAN16) minY = Math.min(minY, P[i] / 10);
  var grid = new THREE.GridHelper(2000, 40, 0x2f3540, 0x232830); grid.position.y = minY - 20; scene.add(grid);
  var camera = new THREE.PerspectiveCamera(40, 1, 2, 20000);
  var target = new THREE.Vector3(0, 0, 0), az = 0.6, el = 0.5, dist = 450, follow = true;

  var dots = points(maxK, 7, DOT, 1.0), rings = points(7 + 2 * NK, 13, RINGTEX, 0.95), resid = lines(7 + 2 * NK, 0.9);
  var axes = lines(16, 1.0), motiveT = points(10, 4, DOT, 0.8), chainL = lines(4, 0.95);
  [dots, rings, resid, axes, motiveT, chainL].forEach(function (o) { scene.add(o); });
  var meshGeo = new THREE.BufferGeometry();
  meshGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(NV * 3), 3));
  meshGeo.setIndex(new THREE.BufferAttribute(new Uint16Array(faces), 1));
  var mesh = new THREE.Mesh(meshGeo, new THREE.MeshPhongMaterial({ color: 0xd9b8a0, transparent: true, opacity: 0.55,
                                                                   side: THREE.DoubleSide, shininess: 10, depthWrite: false }));
  mesh.frustumCulled = false; scene.add(mesh);
  // optional overlay: the same calibration tracked with the SMOOTHED WRIST (build_viewer.py)
  var OV = D.overlay || null, ovA = null, ovChain = null, ovKn = null, ovStats = null, ovMesh = null, ovChainL = null;
  if (OV) {
    ovA = dec(OV.A, Float32Array); ovChain = dec(OV.chain, Int16Array); ovKn = dec(OV.knuckle, Float32Array);
    ovStats = dec(OV.stats, Float32Array); var ovPredKn = dec(OV.pred_knuckle, Int16Array);
    var ovGeo = new THREE.BufferGeometry();
    ovGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(NV * 3), 3));
    ovGeo.setIndex(new THREE.BufferAttribute(new Uint16Array(faces), 1));
    ovMesh = new THREE.Mesh(ovGeo, new THREE.MeshPhongMaterial({ color: 0x6fc3ff, transparent: true, opacity: 0.45,
                                                                side: THREE.DoubleSide, shininess: 10, depthWrite: false }));
    ovMesh.frustumCulled = false; scene.add(ovMesh);
    ovChainL = lines(4, 0.95); scene.add(ovChainL);
    document.getElementById("ovl-label").textContent = OV.label;
  } else {
    document.getElementById("ovl-row").style.display = "none";
  }
  function skin(Aarr, o, geo) {
    var mp = geo.attributes.position.array;
    for (var v = 0; v < NV; v++) {
      var x = vRest[3 * v], y = vRest[3 * v + 1], z = vRest[3 * v + 2], X = 0, Y = 0, Z = 0;
      for (var k = 0; k < 4; k++) {
        var j = wIdx[4 * v + k], w = wts[4 * v + k], b = o + 12 * j;
        X += w * (Aarr[b] * x + Aarr[b + 1] * y + Aarr[b + 2] * z + Aarr[b + 3]);
        Y += w * (Aarr[b + 4] * x + Aarr[b + 5] * y + Aarr[b + 6] * z + Aarr[b + 7]);
        Z += w * (Aarr[b + 8] * x + Aarr[b + 9] * y + Aarr[b + 10] * z + Aarr[b + 11]);
      }
      mp[3 * v] = X; mp[3 * v + 1] = Y; mp[3 * v + 2] = Z;
    }
    geo.attributes.position.needsUpdate = true; geo.computeVertexNormals();
  }
  function drawChain(L, arr, col, show) {
    var pa = L.geometry.attributes.position, pc = L.geometry.attributes.color, n = 0;
    if (show) for (var s2 = 0; s2 < 4; s2++) {
      var p0 = v16(arr, (frame * 5 + s2) * 3), p1 = v16(arr, (frame * 5 + s2 + 1) * 3);
      if (!p0 || !p1) continue;
      setP(pa, 2 * n, p0); setP(pa, 2 * n + 1, p1); setC(pc, 2 * n, col); setC(pc, 2 * n + 1, col); n++;
    }
    L.geometry.setDrawRange(0, 2 * n); pa.needsUpdate = pc.needsUpdate = true;
  }

  function applyCam() {
    camera.position.set(target.x + dist * Math.cos(el) * Math.sin(az), target.y + dist * Math.sin(el),
                        target.z + dist * Math.cos(el) * Math.cos(az));
    camera.lookAt(target); render();
  }
  function render() { renderer.render(scene, camera); }
  function resize() {
    var w = cv.clientWidth, h = cv.clientHeight; renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix(); render();
  }

  // ----------------------------------------------------------------------------------- the frame
  var frame = 0, solo = null, opt = {};
  ["mesh", "chain", "pred", "knerr", "resid", "axes", "motive", "free", "fixture", "ovl"].forEach(function (k) {
    var e = document.getElementById("opt-" + k); opt[k] = e.checked;
    e.addEventListener("change", function () { opt[k] = e.checked; draw(); });
  });
  function q7(arr, f) { var o = 7 * f; return arr[o] !== arr[o] ? null :
    { q: new THREE.Quaternion(arr[o], arr[o + 1], arr[o + 2], arr[o + 3]), t: new THREE.Vector3(arr[o + 4], arr[o + 5], arr[o + 6]) }; }
  function v16(arr, o) { return arr[o] === NAN16 ? null : new THREE.Vector3(arr[o] / 10, arr[o + 1] / 10, arr[o + 2] / 10); }

  function setP(attr, j, v) { attr.array[3 * j] = v.x; attr.array[3 * j + 1] = v.y; attr.array[3 * j + 2] = v.z; }
  function setC(attr, j, c) { attr.array[3 * j] = c.r; attr.array[3 * j + 1] = c.g; attr.array[3 * j + 2] = c.b; }

  function draw() {
    // markers
    var pa = dots.geometry.attributes.position, ca = dots.geometry.attributes.color, n = 0;
    for (var i = ptr[frame]; i < ptr[frame + 1]; i++) {
      var c = CLS[i];
      if (c === FREE && !opt.free) continue;
      if (c === 0 && !opt.fixture) continue;
      if (solo !== null && c !== solo) continue;
      pa.array[3 * n] = P[3 * i] / 10; pa.array[3 * n + 1] = P[3 * i + 1] / 10; pa.array[3 * n + 2] = P[3 * i + 2] / 10;
      setC(ca, n, c === FREE ? FREEC : clsCol[c]); n++;
    }
    dots.geometry.setDrawRange(0, n); pa.needsUpdate = ca.needsUpdate = true;

    // predicted markers + residual lines (tip cap rows 0-4, floating 5-6)
    var ra = rings.geometry.attributes.position, rc = rings.geometry.attributes.color;
    var la = resid.geometry.attributes.position, lc = resid.geometry.attributes.color, nr = 0, nl = 0;
    var tipObs = [], knObs = [], flObs = [null, null];
    for (i = ptr[frame]; i < ptr[frame + 1]; i++) {
      var pv = new THREE.Vector3(P[3 * i] / 10, P[3 * i + 1] / 10, P[3 * i + 2] / 10);
      if (CLS[i] === 3 || CLS[i] === 4) flObs[CLS[i] - 3] = pv;
      if (CLS[i] === 2) tipObs.push(pv);
      if (CLS[i] === 1) knObs.push(pv);
    }
    // [predicted, colour, observed or null, pool to take the nearest observed from, line colour]
    var WHITE = new THREE.Color("#ffffff"), OVC = new THREE.Color("#6fc3ff"), preds = [];
    if (opt.pred) {
      for (var r = 0; r < 5; r++) preds.push([v16(predTip, (frame * 5 + r) * 3), clsCol[2], null, tipObs, WHITE]);
      for (r = 0; r < 2; r++) preds.push([v16(predFl, (frame * 2 + r) * 3), clsCol[3 + r], flObs[r], null, WHITE]);
    }
    if (opt.knerr) {                              // knuckle cap where the tracked MANO root puts it
      for (r = 0; r < NK; r++) preds.push([v16(predKn, (frame * NK + r) * 3), clsCol[1], null, knObs, WHITE]);
      if (OV && opt.ovl) for (r = 0; r < NK; r++) preds.push([v16(ovPredKn, (frame * NK + r) * 3), OVC, null, knObs, OVC]);
    }
    preds.forEach(function (pp) {
      if (!pp[0]) return;
      setP(ra, nr, pp[0]); setC(rc, nr, pp[1]); nr++;
      var obs = pp[2];
      if (pp[3] && pp[3].length) {               // a cap row: the nearest observed marker of that cap
        var best = null, bd = 1e9;
        pp[3].forEach(function (o) { var d = o.distanceTo(pp[0]); if (d < bd) { bd = d; best = o; } });
        obs = bd < 15 ? best : null;
      }
      if (obs && opt.resid) {
        setP(la, 2 * nl, pp[0]); setP(la, 2 * nl + 1, obs); setC(lc, 2 * nl, pp[4]); setC(lc, 2 * nl + 1, pp[4]); nl++;
      }
    });
    rings.geometry.setDrawRange(0, nr); ra.needsUpdate = rc.needsUpdate = true;
    resid.geometry.setDrawRange(0, 2 * nl); la.needsUpdate = lc.needsUpdate = true;

    // cap axes: MoSca (bright, 15 mm) and Motive (dim, 10 mm) + Motive's template placement
    var aa = axes.geometry.attributes.position, ac = axes.geometry.attributes.color, na = 0;
    var AX = [new THREE.Color("#ff5555"), new THREE.Color("#55ff77"), new THREE.Color("#5599ff")];
    var DIM = AX.map(function (c) { return c.clone().multiplyScalar(0.45); });
    var mta = motiveT.geometry.attributes.position, mtc = motiveT.geometry.attributes.color, nm = 0;
    ["knuckle", "tip"].forEach(function (role) {
      [["mosca", opt.axes, 15, AX], ["motive", opt.motive, 10, DIM]].forEach(function (s) {
        if (!s[1]) return;
        var ps = q7(pose[role][s[0]], frame); if (!ps) return;
        for (var a = 0; a < 3; a++) {
          var e = new THREE.Vector3(a === 0 ? s[2] : 0, a === 1 ? s[2] : 0, a === 2 ? s[2] : 0).applyQuaternion(ps.q).add(ps.t);
          setP(aa, 2 * na, ps.t); setP(aa, 2 * na + 1, e); setC(ac, 2 * na, s[3][a]); setC(ac, 2 * na + 1, s[3][a]); na++;
        }
        if (s[0] === "motive") D.templates[role].forEach(function (x) {
          var v = new THREE.Vector3(x[0], x[1], x[2]).applyQuaternion(ps.q).add(ps.t);
          setP(mta, nm, v); setC(mtc, nm, new THREE.Color("#c792ea")); nm++;
        });
      });
    });
    if (OV && opt.ovl) {                    // the knuckle cap implied by the smoothed wrist, light blue, 20 mm
      var ks = q7(ovKn, frame);
      if (ks) for (var a2 = 0; a2 < 3; a2++) {
        var e2 = new THREE.Vector3(a2 === 0 ? 20 : 0, a2 === 1 ? 20 : 0, a2 === 2 ? 20 : 0).applyQuaternion(ks.q).add(ks.t);
        var c2 = new THREE.Color("#6fc3ff");
        setP(aa, 2 * na, ks.t); setP(aa, 2 * na + 1, e2); setC(ac, 2 * na, c2); setC(ac, 2 * na + 1, AX[a2]); na++;
      }
    }
    axes.geometry.setDrawRange(0, 2 * na); aa.needsUpdate = ac.needsUpdate = true;
    motiveT.geometry.setDrawRange(0, nm); mta.needsUpdate = mtc.needsUpdate = true;

    // MANO: skin in the page from the world bone transforms
    var o = frame * 16 * 12, tracked = A[o] === A[o];
    mesh.visible = opt.mesh && tracked;
    if (mesh.visible) skin(A, o, meshGeo);
    drawChain(chainL, chain, new THREE.Color("#ffe08a"), opt.chain && tracked);
    var ovOn = OV && opt.ovl && ovA[o] === ovA[o];
    if (OV) {
      ovMesh.visible = ovOn;
      if (ovOn) skin(ovA, o, ovMesh.geometry);
      drawChain(ovChainL, ovChain, new THREE.Color("#6fc3ff"), ovOn && opt.chain);
    }

    if (follow) { var kp = q7(pose.knuckle.mosca, frame); if (kp) { target.copy(kp.t); } }
    applyCam();
    var st = stats.subarray(NS * frame, NS * frame + NS);
    var fmt = function (x) { return x < 0 ? "  -  " : x.toFixed(2); };
    var txt = "residual mm   tip cap " + fmt(st[0]) + "   float prox " + fmt(st[1]) + "   float mid " + fmt(st[2]) +
      "   knuckle cap " + fmt(st[3]);
    if (ovOn) {
      var so = ovStats.subarray(NS * frame, NS * frame + NS);
      txt += "\nsmoothed wrist   tip cap " + fmt(so[0]) + "   float prox " + fmt(so[1]) + "   float mid " + fmt(so[2]) +
        "   knuckle cap " + fmt(so[3]);
    }
    document.getElementById("count").textContent = txt + (held[frame] ? "   [held-out block]" : "") + (tracked ? "" : "   [not tracked]");
    document.getElementById("frame").textContent = "frame " + frameIdx[frame] + "  " + (frameIdx[frame] / (D.rate_hz * (frameIdx[1] || 1))).toFixed(2) + " s";
    slider.value = frame; drawTimeline();
  }
  function seek(f) { frame = Math.max(0, Math.min(T - 1, f | 0)); draw(); }

  // ------------------------------------------------------------------------------------- timeline
  var tl = document.getElementById("timeline"), tlBase = null, CAPS = [6.0, 6.0, 6.0, 1.0];   // full-row mm; knuckle is sub-mm
  function buildTimeline() {
    var w = tl.clientWidth, h = tl.height; tl.width = w;
    var off = document.createElement("canvas"); off.width = w; off.height = h;
    var g = off.getContext("2d"), cols = ["#e5484d", "#46c46a", "#f5a524", "#3e8ef7"];
    for (var x = 0; x < w; x++) {
      var f0 = Math.floor(x * T / w), f1 = Math.max(f0 + 1, Math.floor((x + 1) * T / w)), s = [0, 0, 0, 0], hb = 0;
      for (var ff = f0; ff < f1; ff++) { hb |= held[ff]; for (var k = 0; k < NS; k++) s[k] = Math.max(s[k], stats[NS * ff + k]); }
      if (hb) { g.fillStyle = "#3a3f4c"; g.fillRect(x, 0, 1, h); }
      for (k = 0; k < NS; k++) {
        var hh = Math.min(1, Math.max(0, s[k]) / CAPS[k]) * (h / NS - 1);
        g.fillStyle = cols[k]; g.fillRect(x, (k + 1) * h / NS - hh, 1, hh);
      }
    }
    tlBase = off;
  }
  function drawTimeline() {
    if (!tlBase) buildTimeline();
    var g = tl.getContext("2d"); g.clearRect(0, 0, tl.width, tl.height); g.drawImage(tlBase, 0, 0);
    var x = frame / T * tl.width; g.fillStyle = "#fff"; g.fillRect(x - 0.5, 0, 1.5, tl.height);
  }
  tl.addEventListener("mousedown", function (e) {
    var go = function (ev) { var r = tl.getBoundingClientRect(); seek((ev.clientX - r.left) / r.width * T); };
    go(e);
    var mv = function (ev) { go(ev); }, up = function () { window.removeEventListener("mousemove", mv); window.removeEventListener("mouseup", up); };
    window.addEventListener("mousemove", mv); window.addEventListener("mouseup", up);
  });

  // -------------------------------------------------------------------------------------- controls
  var slider = document.getElementById("slider"); slider.max = T - 1;
  slider.addEventListener("input", function () { seek(+slider.value); });
  var playing = false, last = 0, acc = 0, speed = 1;
  document.getElementById("speed").addEventListener("change", function (e) { speed = +e.target.value; });
  function tick(ts) {
    if (!playing) return;
    if (last) acc += (ts - last) / 1000 * D.rate_hz * speed;
    last = ts;
    if (acc >= 1) { var n = Math.floor(acc); acc -= n; seek((frame + n) % T); }
    requestAnimationFrame(tick);
  }
  function toggle() {
    playing = !playing; document.getElementById("play").innerHTML = playing ? "&#10074;&#10074;" : "&#9654;";
    last = 0; acc = 0; if (playing) requestAnimationFrame(tick);
  }
  document.getElementById("play").addEventListener("click", toggle);
  document.getElementById("prev").addEventListener("click", function () { seek(frame - 1); });
  document.getElementById("next").addEventListener("click", function () { seek(frame + 1); });
  var fb = document.getElementById("follow"); fb.classList.toggle("on", follow);
  fb.addEventListener("click", function () { follow = !follow; fb.classList.toggle("on", follow); draw(); });
  window.addEventListener("keydown", function (e) {
    if (e.target.tagName === "INPUT" && e.target.type !== "checkbox") return;
    if (e.key === " ") { toggle(); e.preventDefault(); }
    else if (e.key === "ArrowRight") seek(frame + (e.shiftKey ? 10 : 1));
    else if (e.key === "ArrowLeft") seek(frame - (e.shiftKey ? 10 : 1));
  });
  cv.addEventListener("contextmenu", function (e) { e.preventDefault(); });
  cv.addEventListener("mousedown", function (e) {
    var x0 = e.clientX, y0 = e.clientY, pan = e.button === 2 || e.shiftKey;
    var mv = function (ev) {
      var dx = ev.clientX - x0, dy = ev.clientY - y0; x0 = ev.clientX; y0 = ev.clientY;
      if (pan) {
        follow = false; fb.classList.remove("on");
        var s = dist * 0.0015, right = new THREE.Vector3(), up = new THREE.Vector3();
        camera.matrixWorld.extractBasis(right, up, new THREE.Vector3());
        target.addScaledVector(right, -dx * s).addScaledVector(up, dy * s);
      } else { az -= dx * 0.008; el = Math.max(-1.5, Math.min(1.5, el + dy * 0.008)); }
      applyCam();
    };
    var up = function () { window.removeEventListener("mousemove", mv); window.removeEventListener("mouseup", up); };
    window.addEventListener("mousemove", mv); window.addEventListener("mouseup", up);
  });
  cv.addEventListener("wheel", function (e) { dist *= Math.exp(e.deltaY * 0.0012); applyCam(); e.preventDefault(); },
                      { passive: false });

  // ---------------------------------------------------------------------------------- side panel
  var leg = document.getElementById("legend");
  D.class_names.concat(["unassigned"]).forEach(function (nm, k) {
    var d = document.createElement("div"), col = k < D.class_colours.length ? D.class_colours[k] : "#b4bac4";
    d.innerHTML = "<i style='background:" + col + "'></i>" + nm;
    var key = k < D.class_colours.length ? k : FREE;
    d.addEventListener("click", function () {
      solo = solo === key ? null : key;
      Array.prototype.forEach.call(leg.children, function (c, j) {
        var kj = j < D.class_colours.length ? j : FREE; c.classList.toggle("off", solo !== null && kj !== solo); });
      draw();
    });
    leg.appendChild(d);
  });
  var S = D.summary;
  document.getElementById("summary").textContent =
    "RMS over the take (mm)\n  tip cap      " + S.rms_mm.tip_cap.toFixed(2) + "\n  knuckle cap  " + S.rms_mm.knuckle_cap.toFixed(2) +
    "\n  float prox   " + S.rms_mm.float_proximal.toFixed(2) + "\n  float mid    " + S.rms_mm.float_middle.toFixed(2) +
    "\nfloat height above skin (mm)\n  proximal " + S.heights_mm.proximal + "  middle " + S.heights_mm.middle +
    "\nbetas\n  " + S.betas.slice(0, 5).join(" ") + "\n  " + S.betas.slice(5).join(" ");
  document.getElementById("title").textContent = "index-finger calibration  -  " + D.take + "  (" + D.rate_hz.toFixed(0) +
    " Hz shown, " + T + " frames)";
  document.getElementById("stats").textContent = "rings = predicted from MANO + calibration;  dots = observed (MoSca labels)";
  window.addEventListener("resize", function () { tlBase = null; resize(); draw(); });
  if (OV && /ovl/.test(location.hash)) { document.getElementById("opt-ovl").checked = true; opt.ovl = true; }
  var m = /f=(\d+)/.exec(location.hash);
  resize(); seek(m ? +m[1] : 0);
})();

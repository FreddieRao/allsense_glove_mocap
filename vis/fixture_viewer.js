// fixture registration inspector. Everything is computed in build_fixture_viewer.py; this decodes and draws.
// CAD frame, mm. Orbit camera ported from viewer.js.
(function () {
  "use strict";
  var D = window.FIX_VIEW;
  function dec(s, Type) {
    var bin = atob(s), u = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return new Type(u.buffer);
  }
  var cv = document.getElementById("cv");
  var renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1); renderer.setClearColor(0x15171c);
  var scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  var l1 = new THREE.DirectionalLight(0xffffff, 0.7); l1.position.set(200, 300, 500); scene.add(l1);
  var l2 = new THREE.DirectionalLight(0xffffff, 0.35); l2.position.set(-300, -200, -400); scene.add(l2);
  var camera = new THREE.PerspectiveCamera(40, 1, 1, 10000);
  var target = new THREE.Vector3(0, 110, 20), az = 0.5, el = 0.7, dist = 420;

  function meshFrom(m, color, opacity, verts) {
    var g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(verts || dec(m.v, Float32Array), 3));
    g.setIndex(new THREE.BufferAttribute(dec(m.f, Uint32Array), 1));
    g.computeVertexNormals();
    var mat = new THREE.MeshPhongMaterial({ color: color, transparent: opacity < 1, opacity: opacity,
                                            side: THREE.DoubleSide, shininess: 20, depthWrite: opacity >= 1 });
    var o = new THREE.Mesh(g, mat); scene.add(o); return o;
  }
  var cad = meshFrom(D.cad, 0x9aa3b0, 0.85);
  var designed = meshFrom(D.designed_hand, 0xd9b8a0, 0.45);
  var tip = meshFrom(D.tip_scan, 0xf5a524, 0.7); tip.visible = false;
  var fitted = meshFrom({ v: D.solutions[0].hand, f: D.mano_faces }, 0x3e8ef7, 0.45);

  var NM = D.marker_radius.length, sphere = new THREE.SphereGeometry(1, 20, 14);
  var markers = new THREE.InstancedMesh(sphere, new THREE.MeshPhongMaterial({ color: 0xffffff }), NM);
  scene.add(markers);
  var nc = D.n_caps[0] + D.n_caps[1] + D.n_caps[2];
  var caps = new THREE.InstancedMesh(sphere, new THREE.MeshPhongMaterial({ color: 0xffffff }), nc); scene.add(caps);
  var gapG = new THREE.BufferGeometry();
  gapG.setAttribute("position", new THREE.BufferAttribute(new Float32Array(NM * 6), 3));
  var gaps = new THREE.LineSegments(gapG, new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false }));
  gaps.renderOrder = 3; scene.add(gaps);

  function gapColour(g) {
    var t = Math.max(-1, Math.min(1, g / 3)), c = new THREE.Color(0xe8e8e8);
    return t < 0 ? c.lerp(new THREE.Color(0x3e8ef7), -t) : c.lerp(new THREE.Color(0xe5484d), t);
  }
  var sel = 0, opt = {};
  function show(k) {
    sel = k;
    var S = D.solutions[k], mk = dec(S.markers, Float32Array), sf = dec(S.surface, Float32Array), gp = dec(S.gap, Float32Array);
    var m4 = new THREE.Matrix4(), rows = "<tr><th>marker</th><th>radius</th><th>gap |d|-r</th></tr>";
    for (var i = 0; i < NM; i++) {
      var r = D.marker_radius[i];
      m4.makeScale(r, r, r).setPosition(mk[3 * i], mk[3 * i + 1], mk[3 * i + 2]);
      markers.setMatrixAt(i, m4); markers.setColorAt(i, gapColour(gp[i]));
      gapG.attributes.position.array.set([mk[3 * i], mk[3 * i + 1], mk[3 * i + 2], sf[3 * i], sf[3 * i + 1], sf[3 * i + 2]], 6 * i);
      rows += "<tr><td>" + (i + 1) + "</td><td>" + r.toFixed(2) + "</td><td style='color:#" + gapColour(gp[i]).getHexString() +
              "'>" + (gp[i] >= 0 ? "+" : "") + gp[i].toFixed(2) + "</td><td>" + (D.coplanar.indexOf(i) >= 0 ? "&#9679;" : "") + "</td></tr>";
    }
    markers.instanceMatrix.needsUpdate = true; markers.instanceColor.needsUpdate = true;
    gapG.attributes.position.needsUpdate = true;
    document.getElementById("markers").innerHTML = rows;
    var hv = dec(S.hand, Float32Array);
    fitted.geometry.attributes.position.array.set(hv); fitted.geometry.attributes.position.needsUpdate = true;
    fitted.geometry.computeVertexNormals();
    var cp = dec(S.caps, Float32Array), cols = [0x3e8ef7, 0xe5484d, 0x46c46a];
    for (i = 0; i < nc; i++) {
      var grp = i < D.n_caps[0] ? 0 : i < D.n_caps[0] + D.n_caps[1] ? 1 : 2;
      var rr = grp === 2 ? D.float_radius : 1.6;
      m4.makeScale(rr, rr, rr).setPosition(cp[3 * i], cp[3 * i + 1], cp[3 * i + 2]);
      caps.setMatrixAt(i, m4); caps.setColorAt(i, new THREE.Color(grp === 2 && i === nc - 1 ? 0xf5a524 : cols[grp]));
    }
    caps.instanceMatrix.needsUpdate = true; caps.instanceColor.needsUpdate = true;
    Array.prototype.forEach.call(document.querySelectorAll("#sols tr"), function (tr, j) { tr.classList.toggle("sel", j - 1 === k); });
    document.getElementById("count").textContent = S.name + "   rms gap " + S.rms_gap_mm.toFixed(2) + " mm   max |gap| " +
      S.max_abs_gap_mm.toFixed(2) + " mm   fitted hand vs designed (vertex p50) " + S.hand_vs_designed_p50_mm.toFixed(1) + " mm  (palm " +
      S.palm_vs_designed_p50_mm.toFixed(1) + ", index " + S.index_vs_designed_p50_mm.toFixed(1) + ")";
    render();
  }
  var t = "<table><tr><th>candidate</th><th>rms gap</th><th>max</th><th>&Delta;deg</th><th>&Delta;mm</th><th>hand&ndash;design</th></tr>";
  D.solutions.forEach(function (S) {
    t += "<tr><td>" + S.name + "</td><td>" + S.rms_gap_mm.toFixed(2) + "</td><td>" + S.max_abs_gap_mm.toFixed(1) + "</td><td>" +
         S.rot_from_first_deg.toFixed(1) + "</td><td>" + S.trans_from_first_mm.toFixed(1) + "</td><td>" + S.hand_vs_designed_p50_mm.toFixed(1) + "</td></tr>";
  });
  document.getElementById("sols").innerHTML = t + "</table><div style='color:#8a919c;font-size:11px;margin-top:3px'>" +
    "gap: marker-centre distance to CAD minus radius (mm); &Delta;: from the first row; " +
    "hand&ndash;design: fitted MANO vs hand_aligned_mm.obj, vertex p50 (mm)</div>";
  Array.prototype.forEach.call(document.querySelectorAll("#sols tr"), function (tr, j) {
    if (j > 0) tr.addEventListener("click", function () { show(j - 1); });
  });
  var objs = { cad: [cad], markers: [markers], gaps: [gaps], designed: [designed], fitted: [fitted], caps: [caps], tip: [tip] };
  Object.keys(objs).forEach(function (k) {
    var e = document.getElementById("opt-" + k);
    var f = function () { objs[k].forEach(function (o) { o.visible = e.checked; }); render(); };
    e.addEventListener("change", f); f();
  });

  function render() { renderer.render(scene, camera); }
  // orbit about CAD +z (up); el is the elevation above the CAD x-y plane
  function applyCam() {
    camera.position.set(target.x + dist * Math.cos(el) * Math.sin(az), target.y - dist * Math.cos(el) * Math.cos(az),
                        target.z + dist * Math.sin(el));
    camera.up.set(0, 0, 1); camera.lookAt(target); render();
  }
  function resize() {
    var w = cv.clientWidth, h = cv.clientHeight; renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix(); render();
  }
  cv.addEventListener("contextmenu", function (e) { e.preventDefault(); });
  cv.addEventListener("mousedown", function (e) {
    var x0 = e.clientX, y0 = e.clientY, pan = e.button === 2 || e.shiftKey;
    var mv = function (ev) {
      var dx = ev.clientX - x0, dy = ev.clientY - y0; x0 = ev.clientX; y0 = ev.clientY;
      if (pan) {
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
  window.addEventListener("resize", resize);
  resize(); show(0); applyCam();
})();

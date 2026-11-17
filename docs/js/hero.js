// Home page, part 1: why the model is more uncertain near the edges. The map is the model's
// uncertainty (posterior standard deviation), for one random set of measurements or averaged
// over many. A probe shows the characteristic distance over which a measurement informs a
// location; near an edge part of that neighbourhood lies outside the allowed range.
(function () {
  const root = document.getElementById("why");
  if (!root) return;
  const NG = 31, G = GP.grid(NG), M = NG * NG, NOISE = 0.0025, DESIGNS = 64;
  const q = s => root.querySelector(s);
  const map = q("#why-map"), cut = q("#why-cut"), more = q("#why-more");
  const nIn = q("#why-n"), ellIn = q("#why-ell"), nOut = q("#why-n-o"), ellOut = q("#why-ell-o");
  const shareOut = q("#why-share"), ratioOut = q("#why-ratio"), refOut = q("#why-ref"), say = q("#why-say");
  const viewSeg = q("#why-view"), posSeg = q("#why-pos");
  const PLACES = { centre: [0.5, 0.5], edge: [1, 0.5], corner: [1, 1] };
  let avg, one, X1, ell, probe = PLACES.edge, dragging = false, view = "avg", setSeed = 1;

  function oneSet() {
    const rand = GP.rng(100 + setSeed);
    X1 = Array.from({ length: +nIn.value }, () => [rand(), rand()]);
    one = Float64Array.from(GP.build(X1, null, ell, NOISE, [G])[1].v, Math.sqrt);
  }
  function compute() {
    const n = +nIn.value, rand = GP.rng(5), v = new Float64Array(M);
    ell = +ellIn.value;
    for (let s = 0; s < DESIGNS; s++) {
      const X = Array.from({ length: n }, () => [rand(), rand()]), set = GP.build(X, null, ell, NOISE, [G])[1];
      for (let j = 0; j < M; j++) v[j] += set.v[j];
    }
    // the square has eight symmetries; averaging over them removes most of the sampling noise
    avg = new Float64Array(M);
    for (let r = 0; r < NG; r++) for (let c = 0; c < NG; c++) {
      const a = NG - 1 - r, b = NG - 1 - c;
      const s = v[r * NG + c] + v[r * NG + b] + v[a * NG + c] + v[a * NG + b] + v[c * NG + r] + v[c * NG + a] + v[b * NG + r] + v[b * NG + a];
      avg[r * NG + c] = Math.sqrt(s / (8 * DESIGNS));
    }
    oneSet();
  }
  // share of the disk of radius ell around p that lies inside the square
  function shareInside(p) {
    let inside = 0, total = 0;
    for (let i = -20; i <= 20; i++) for (let j = -20; j <= 20; j++) {
      const dx = i / 20 * ell, dy = j / 20 * ell;
      if (dx * dx + dy * dy > ell * ell) continue;
      total++;
      const x = p[0] + dx, y = p[1] + dy;
      if (x >= 0 && x <= 1 && y >= 0 && y <= 1) inside++;
    }
    return inside / total;
  }
  // the square leaves a margin, so that the part of the neighbourhood outside it stays visible
  const square = () => GP.square(map, 0.2 * map.clientWidth);
  const draw = throttled(() => {
    nOut.textContent = nIn.value; ellOut.textContent = ell.toFixed(2);
    GP.seg(viewSeg, view);
    const place = Object.keys(PLACES).find(k => PLACES[k][0] === probe[0] && PLACES[k][1] === probe[1]);
    GP.seg(posSeg, place || "");
    const sd = view === "avg" ? avg : one, sq = square(), { ctx } = sq, centre = avg[(M - 1) / 2];
    let hi = 0, mean = 0; for (const v of sd) { hi = Math.max(hi, v); mean += v / M; }
    ctx.fillStyle = "#f1f3f5"; ctx.fillRect(0, 0, sq.w, sq.h);
    GP.heat(sq, sd, NG, view === "avg" ? { lo: Math.min(centre * 0.9, hi * 0.5), hi } : { lo: 0, hi });
    ctx.strokeStyle = "#1a1a1a"; ctx.lineWidth = 1.5; ctx.strokeRect(sq.x0, sq.y0, sq.s, sq.s);
    ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText("Outside the allowed settings", 10, 10);
    ctx.fillStyle = C.ink || "#1a1a1a"; ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText("Allowed experimental settings", sq.x0, sq.y0 + sq.s + 8);
    if (view === "one") GP.dots(sq, X1, { r: 3.5, fill: "#1a1a1a" });
    // the neighbourhood: tinted inside the allowed range, hatched outside
    const px = sq.px(probe[0]), py = sq.py(probe[1]), R = ell * sq.s;
    ctx.save();
    ctx.beginPath(); ctx.arc(px, py, R, 0, 2 * Math.PI); ctx.clip();
    ctx.fillStyle = "rgba(232, 89, 12, 0.16)"; ctx.fillRect(sq.x0, sq.y0, sq.s, sq.s);
    ctx.beginPath(); ctx.rect(0, 0, sq.w, sq.h); ctx.rect(sq.x0, sq.y0, sq.s, sq.s); ctx.clip("evenodd");
    ctx.fillStyle = "rgba(95, 103, 112, 0.16)"; ctx.fillRect(0, 0, sq.w, sq.h);
    ctx.restore();
    // solid outline inside the allowed settings, dashed outside
    ctx.strokeStyle = C.accent; ctx.lineWidth = 2;
    ctx.save(); ctx.beginPath(); ctx.rect(sq.x0, sq.y0, sq.s, sq.s); ctx.clip();
    ctx.beginPath(); ctx.arc(px, py, R, 0, 2 * Math.PI); ctx.stroke(); ctx.restore();
    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, sq.w, sq.h); ctx.rect(sq.x0, sq.y0, sq.s, sq.s); ctx.clip("evenodd");
    ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(px, py, R, 0, 2 * Math.PI); ctx.stroke(); ctx.restore();
    ctx.beginPath(); ctx.arc(px, py, 5.5, 0, 2 * Math.PI); ctx.fillStyle = C.accent; ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
    if (more.open) { ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(sq.x0, py); ctx.lineTo(sq.x0 + sq.s, py); ctx.strokeStyle = "rgba(26,26,26,0.45)"; ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]); }

    const ref = view === "avg" ? centre : mean, share = shareInside(probe);
    shareOut.textContent = `At the marker, ${Math.round(100 * share)} % of the circle lies inside the allowed settings.`;
    ratioOut.textContent = `${(GP.bilinear(sd, NG, probe[0], probe[1]) / ref).toFixed(2)} ×`;
    refOut.textContent = view === "avg" ? "the uncertainty at the centre" : "the average uncertainty of this set";
    const dx = Math.min(probe[0], 1 - probe[0]), dy = Math.min(probe[1], 1 - probe[1]);
    say.innerHTML = share > 0.97
      ? "<b>At the centre</b>, measurement locations are available in every direction."
      : Math.max(dx, dy) > 0.5 * ell
        ? "<b>Near an edge</b>, fewer nearby measurement locations are available. The model is therefore more uncertain here on average."
        : "<b>Near a corner</b>, the neighbourhood is cut off on two sides. This is where the model is most uncertain on average.";
    if (more.open) {
      const line = Array.from({ length: 61 }, (_, i) => GP.bilinear(sd, NG, i / 60, probe[1]) / ref);
      GP.lines(cut, [{ values: line, colour: C.navy, noDot: true }],
        { xStart: 0, xMax: 60, xTick: i => (i / 60).toFixed(2), mark: probe[0] * 60, xLabel: "position along the dashed line", yLabel: "relative uncertainty" });
    }
  });
  const clamp = v => Math.min(Math.max(v, 0), 1);
  function move(e) {
    const b = map.getBoundingClientRect(), [x, y] = square().inv(e.clientX - b.left, e.clientY - b.top);
    probe = [clamp(x), clamp(y)];
    draw();
  }
  map.addEventListener("pointerdown", e => { dragging = true; map.setPointerCapture(e.pointerId); move(e); });
  map.addEventListener("pointermove", e => { if (dragging) move(e); });
  map.addEventListener("pointerup", () => { dragging = false; });
  map.addEventListener("keydown", e => {
    const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[e.key];
    if (!d) return;
    e.preventDefault();
    probe = [clamp(probe[0] + 0.02 * d[0]), clamp(probe[1] + 0.02 * d[1])];
    draw();
  });
  posSeg.onclick = e => { if (e.target.dataset.v) { probe = PLACES[e.target.dataset.v]; draw(); } };
  viewSeg.onclick = e => { if (e.target.dataset.v) { view = e.target.dataset.v; draw(); } };
  q("#why-new").onclick = () => { setSeed++; oneSet(); view = "one"; draw(); };
  more.addEventListener("toggle", draw);
  nIn.oninput = ellIn.oninput = () => { compute(); draw(); };
  window.addEventListener("resize", draw);
  compute(); draw();
})();

// Home page: choosing points by the highest score, without and with a geometric weight.
// Both panels start from the same random points. No measured value is used; the picks depend
// only on where the earlier points are. The weight is the geometric weight of the
// Bayesian-optimization paper, applied to the posterior standard deviation.
// The sampler is used twice: as the opening visual (shown finished, with Replay) and as the
// full comparison with step controls.
function sampler(root, { finished = false, dot = 5 } = {}) {
  if (!root) return;
  const NB = FIELD_DATA.nb, ELL = FIELD_DATA.ell, INIT = 5, PICKS = 20, BAND = 0.05;
  const C0 = GP.grid(NB), geo = GP.field(FIELD_DATA, "geo"), q = c => root.querySelector(c);
  const cv = [q(".s-plain"), q(".s-geo")], out = [q(".s-plain-n"), q(".s-geo-n")], bar = [q(".s-plain-b"), q(".s-geo-b")];
  const play = q(".s-play"), count = q(".s-count");
  let runs, timer = null, seed = 0;

  function reset() {
    const X0 = GP.sobol(INIT, 40 + seed);
    runs = [0, 1].map(k => {
      const m = new GP.Model(ELL, 1e-8), set = m.track(C0);
      for (const p of X0) m.add(p);
      return { m, set, weighted: k === 1, score: null };
    });
    runs.forEach(score);
  }
  function score(r) {
    const g = r.weighted ? geo.grid(r.m.n) : null, s = new Float64Array(r.set.m);
    for (let j = 0; j < s.length; j++) s[j] = Math.sqrt(r.set.v[j]) * (g ? g[j] : 1);
    r.score = s;
  }
  const picked = () => runs[0].m.n - INIT, done = () => picked() >= PICKS;
  function advance() { for (const r of runs) { const j = GP.argmax(r.score); r.m.add([C0[2 * j], C0[2 * j + 1]]); score(r); } }
  function step() { advance(); if (done()) stop(); draw(); }
  const atEdge = X => X.slice(INIT).filter(p => Math.min(p[0], 1 - p[0], p[1], 1 - p[1]) < BAND).length;
  const draw = throttled(() => {
    runs.forEach((r, k) => {
      const sq = GP.square(cv[k], 10), { ctx } = sq, edge = atEdge(r.m.X);
      GP.heat(sq, r.score, NB, { lo: 0 });
      // the band that counts as "near the boundary"
      ctx.save(); ctx.beginPath(); ctx.rect(sq.x0, sq.y0, sq.s, sq.s);
      ctx.rect(sq.px(BAND), sq.py(1 - BAND), (1 - 2 * BAND) * sq.s, (1 - 2 * BAND) * sq.s); ctx.clip("evenodd");
      ctx.fillStyle = "rgba(26, 26, 26, 0.13)"; ctx.fillRect(sq.x0, sq.y0, sq.s, sq.s); ctx.restore();
      GP.frame(sq);
      GP.dots(sq, r.m.X, { to: INIT, fill: "#f2b134", stroke: "#1a1a1a", r: dot + 0.5 });
      GP.dots(sq, r.m.X, { from: INIT, r: dot, fill: "#1a1a1a" });
      if (picked() && !done()) GP.ring(sq, r.m.X[r.m.n - 1], C.accent, dot + 4);
      if (out[k]) out[k].textContent = `${edge} of ${picked()}`;
      if (bar[k]) bar[k].style.width = `${100 * edge / PICKS}%`;
    });
    if (count) count.textContent = `${picked()} / ${PICKS} chosen`;
    play.textContent = timer ? "Pause" : done() ? "Replay" : "Play";
  });
  function stop() { clearInterval(timer); timer = null; }
  play.onclick = () => {
    if (timer) stop();
    else { if (done()) reset(); timer = setInterval(step, 480); }
    draw();
  };
  const on = (c, f) => { const b = q(c); if (b) b.onclick = () => { stop(); f(); draw(); }; };
  on(".s-step", () => { if (done()) reset(); advance(); });
  on(".s-reset", reset);
  on(".s-seed", () => { seed++; reset(); });
  window.addEventListener("resize", draw);
  reset();
  if (finished) while (!done()) advance();
  draw();
}
sampler(document.getElementById("top-demo"), { finished: true, dot: 4 });

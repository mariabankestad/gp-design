// Experimental-design page, part 2: sequential design on a test function with three rules.
//   eig       posterior variance
//   eq        posterior variance times the geometric equalizer
//   eq_warp   the same, with the variance computed in coordinates warped by the posterior mean
// The steps follow the released code (acquisition.py, equalizer.py, warp.py); the candidates
// are a 41 x 41 grid and the warp is built from a 64 x 64 grid of probe points.
// The functions are standardized and share one noise variance, starting design and candidate
// grid. The first two rules never use a measured value, so their sequences are the same for
// every function. Switching the function replays all three rules to the same count.
(function () {
  const root = document.getElementById("warp");
  if (!root) return;
  const D = DESIGN_DATA, NG = D.ng, ELL = D.ell, NOISE = 0.0025, JIT = 1e-4, N0 = 10, ROUNDS = 40, START = 15;
  const NC = 41, Cg = GP.grid(NC), M = NC * NC, Tg = GP.grid(NG);
  const NP = 64, Ug = GP.grid(NP, true), MU = NP * NP;
  const NBINS = 16, NCOND = 4, REG = 0.05;
  const RULES = [
    { key: "eig", colour: "#1f4e79" },
    { key: "eq", colour: "#1b9e8a" },
    { key: "eq_warp", colour: "#e8590c" },
  ];
  const q = s => root.querySelector(s);
  const cvs = RULES.map(r => q(`#w-${r.key}`)), outs = RULES.map(r => q(`#w-${r.key}-o`)), states = RULES.map(r => q(`#w-${r.key}-s`));
  const rhoCv = q("#w-rho"), mapCv = q("#w-map"), errCv = q("#w-err");
  const fnSel = q("#w-fn"), play = q("#w-play"), count = q("#w-count"), viewSeg = q("#w-view"), scale = q("#w-scale");
  for (const [k, f] of Object.entries(D.functions)) fnSel.add(new Option(f.label, k));
  fnSel.value = "peaks";
  const prevBox = q("#w-prev");
  let Z, seed = 0, runs, timer = null, view = "points", prev = null, fnKey = fnSel.value;

  // ---- geometric equalizer: table on the sorted coordinates min(x, 1 - x) -----------------
  const LEV = D.levels, NL = LEV.length, KEYS = Object.keys(D.equalizer).map(Number).sort((a, b) => a - b);
  function equalizer(n) {
    let key = KEYS[0];
    for (const k of KEYS) if (k <= n) key = k;
    const G = D.equalizer[key], g = new Float64Array(M);
    const cell = z => { let hi = 1; while (hi < NL - 1 && LEV[hi] < z) hi++; return [hi - 1, Math.min(Math.max((z - LEV[hi - 1]) / (LEV[hi] - LEV[hi - 1]), 0), 1)]; };
    for (let j = 0; j < M; j++) {
      const a = Math.min(Cg[2 * j], 1 - Cg[2 * j]), b = Math.min(Cg[2 * j + 1], 1 - Cg[2 * j + 1]);
      const [i0, w0] = cell(Math.min(a, b)), [i1, w1] = cell(Math.max(a, b));
      g[j] = (1 - w0) * ((1 - w1) * G[i0 * NL + i1] + w1 * G[i0 * NL + i1 + 1]) + w0 * ((1 - w1) * G[(i0 + 1) * NL + i1] + w1 * G[(i0 + 1) * NL + i1 + 1]);
    }
    return g;
  }

  // ---- reconstruction warp -------------------------------------------------------------
  // Design density 1 + ||grad mu|| / c at the probe points (the exponent 2D/(D+2) is 1 in 2-D).
  function designDensity(mean) {
    const a = mean.weights(), X = mean.X, n = X.length, k5 = 5 / (3 * ELL * ELL), s5 = Math.sqrt(5) / ELL;
    const g = new Float64Array(MU);
    let s = 0, s2 = 0, g2 = 0;
    for (let u = 0; u < MU; u++) {
      const ux = Ug[2 * u], uy = Ug[2 * u + 1];
      let mu = 0, gx = 0, gy = 0;
      for (let i = 0; i < n; i++) {
        const dx = ux - X[i][0], dy = uy - X[i][1], z = s5 * Math.hypot(dx, dy), e = Math.exp(-z);
        mu += a[i] * (1 + z + z * z / 3) * e;
        const w = -a[i] * k5 * (1 + z) * e;
        gx += w * dx; gy += w * dy;
      }
      g[u] = Math.hypot(gx, gy); s += mu; s2 += mu * mu; g2 += g[u] * g[u];
    }
    const sd = Math.max(Math.sqrt(Math.max(s2 / MU - (s / MU) ** 2, 0) * MU / (MU - 1)), 1e-8);
    const c = Math.max(Math.sqrt(g2 / MU) / sd, 1e-8);
    for (let u = 0; u < MU; u++) g[u] = 1 + g[u] / sd / c;
    return g;
  }
  // Knothe-Rosenblatt map from the density to the uniform distribution: piecewise-linear
  // CDF along x, and along y given x in one of NCOND strips, interpolated between strips.
  function krMap(rho) {
    const bin = v => Math.min(Math.max(Math.ceil(v * NBINS) - 1, 0), NBINS - 1);
    function cdf(axis, strip) {
      const h = new Float64Array(NBINS);
      let tot = 0;
      for (let u = 0; u < MU; u++) {
        if (strip >= 0 && Math.min(Math.floor(Ug[2 * u] * NCOND), NCOND - 1) !== strip) continue;
        h[bin(Ug[2 * u + axis])] += rho[u]; tot += rho[u];
      }
      const c = new Float64Array(NBINS + 1), add = Math.max(tot / NBINS, 1e-8) * REG + 1e-10;
      for (let b = 0; b < NBINS; b++) c[b + 1] = c[b] + h[b] + add;
      return c.map(v => v / c[NBINS]);
    }
    const tx = cdf(0, -1), ty = Array.from({ length: NCOND }, (_, s) => cdf(1, s));
    const interp = (row, v) => { const i = bin(v), f = (v - i / NBINS) * NBINS; return row[i] + f * (row[i + 1] - row[i]); };
    const row = new Float64Array(NBINS + 1);
    return p => {
      const x = Math.min(Math.max(p[0], 1e-6), 1 - 1e-6), y = Math.min(Math.max(p[1], 1e-6), 1 - 1e-6);
      const pos = Math.min(Math.max(x * NCOND - 0.5, 0), NCOND - 1), i0 = Math.min(Math.floor(pos), NCOND - 2), w = Math.min(Math.max(pos - i0, 0), 1);
      for (let b = 0; b <= NBINS; b++) row[b] = (1 - w) * ty[i0][b] + w * ty[i0 + 1][b];
      return [interp(tx, x), interp(row, y)];
    };
  }

  // ---- runs --------------------------------------------------------------------------------
  const f = p => GP.bilinear(Z, NG, p[0], p[1]);
  function newRun(rule) {
    const r = { rule, rand: GP.rng(8000 + seed), X: [], y: [], err: [], d2: new Float64Array(NG * NG).fill(Infinity), nn: new Float64Array(NG * NG) };
    if (rule.key === "eq_warp") r.mean = new GP.Model(ELL, NOISE);
    else { r.m = new GP.Model(ELL, NOISE + JIT); r.set = r.m.track(Cg); }
    const r0 = GP.rng(1000 + seed);
    for (const p of D.x0[seed].slice(0, N0)) observe(r, p, f(p) + Math.sqrt(NOISE) * r0.normal());
    warp(r);
    r.err.push(r.rmse);
    return r;
  }
  function observe(r, p, y) {
    r.X.push(p); r.y.push(y);
    if (r.mean) r.mean.add(p, y); else r.m.add(p, y);
    const fp = f(p);
    let s = 0;
    for (let t = 0; t < NG * NG; t++) {
      const d = (Tg[2 * t] - p[0]) ** 2 + (Tg[2 * t + 1] - p[1]) ** 2;
      if (d < r.d2[t]) { r.d2[t] = d; r.nn[t] = fp; }
      s += (r.nn[t] - Z[t]) ** 2;
    }
    r.rmse = Math.sqrt(s / (NG * NG));
  }
  // the warp of the current round, from the measurements so far
  function warp(r) {
    if (!r.mean) return;
    r.rho = designDensity(r.mean);
    r.T = krMap(r.rho);
  }
  function pick(r) {
    const n = r.X.length;
    let v;
    if (r.T) {
      const WC = new Float64Array(2 * M);
      for (let j = 0; j < M; j++) { const w = r.T([Cg[2 * j], Cg[2 * j + 1]]); WC[2 * j] = w[0]; WC[2 * j + 1] = w[1]; }
      v = GP.build(r.X.map(r.T), null, ELL, NOISE + JIT, [WC])[1].v;
    } else v = r.set.v;
    if (r.rule.key === "eig") return GP.argmax(v);
    const g = equalizer(n), s = new Float64Array(M);
    for (let j = 0; j < M; j++) s[j] = v[j] * g[j];
    return GP.argmax(s);
  }
  function advance() {
    for (const r of runs) {
      const j = pick(r), p = [Cg[2 * j], Cg[2 * j + 1]];
      observe(r, p, f(p) + Math.sqrt(NOISE) * r.rand.normal());
      warp(r);
      r.err.push(r.rmse);
    }
  }
  const rounds = () => runs[0].err.length - 1, done = () => rounds() >= ROUNDS;
  function step() { advance(); if (done()) stop(); draw(); }
  function reset(n = 0) {
    fnKey = fnSel.value;
    Z = GP.decode(D.functions[fnKey].z);
    runs = RULES.map(newRun);
    for (let i = 0; i < n; i++) advance();
  }
  const draw = throttled(() => {
    const n0 = N0;
    let zlo = Infinity, zhi = -Infinity, ehi = 1e-9;
    for (const v of Z) { zlo = Math.min(zlo, v); zhi = Math.max(zhi, v); }
    const errs = runs.map(r => Float64Array.from(r.nn, (v, t) => Math.abs(v - Z[t])));
    for (const e of errs) for (const v of e) ehi = Math.max(ehi, v);
    GP.seg(viewSeg, view);
    scale.innerHTML = view === "error" ? 'no error <i class="warm"></i> large error' : 'low value <i class="grey"></i> high value';
    runs.forEach((r, k) => {
      const sq = GP.square(cvs[k], 10), n = r.X.length, small = view !== "points";
      if (view === "error") GP.heat(sq, errs[k], NG, { ramp: "warm", lo: 0, hi: ehi });
      else GP.heat(sq, view === "points" ? Z : r.nn, NG, { ramp: "grey", lo: zlo, hi: zhi });
      GP.frame(sq);
      // open circles: the points this rule chose on the previous function
      if (prev && prevBox.checked && !small) for (let i = n0; i < prev.X[k].length; i++) GP.ring(sq, prev.X[k][i], "rgba(26,26,26,0.55)", 6.5);
      GP.dots(sq, r.X, { to: n0, fill: "#1a1a1a", cross: true, r: small ? 2 : 3.5 });
      GP.dots(sq, r.X, { from: n0, fill: r.rule.colour, r: small ? 2 : 4 });
      outs[k].textContent = `error ${r.rmse.toFixed(3)}`;
      if (!prev) states[k].textContent = "";
      else {
        const m = Math.min(n, prev.X[k].length);
        let moved = 0;
        for (let i = n0; i < m; i++) if (r.X[i][0] !== prev.X[k][i][0] || r.X[i][1] !== prev.X[k][i][1]) moved++;
        states[k].innerHTML = moved === 0
          ? `<b>Sequence unchanged</b> from ${prev.label}: all ${m - n0} points are identical.`
          : `<b>Sequence changed</b> from ${prev.label}: ${moved} of ${m - n0} points differ.`;
      }
    });
    const w = runs[2], last = w.X[w.X.length - 1], newest = w.X.length > n0;
    let sq = GP.square(rhoCv, 10);
    GP.heat(sq, w.rho, NP, { ramp: "warm", lo: 1, nodes: false });
    const gridLines = (g, T) => {
      g.ctx.setLineDash([3, 3]); g.ctx.strokeStyle = "rgba(26,26,26,0.35)"; g.ctx.lineWidth = 1;
      for (let a = 1; a < 10; a++) for (const flip of [false, true]) {
        g.ctx.beginPath();
        for (let b = 0; b <= 48; b++) {
          const p = T(flip ? [b / 48, a / 10] : [a / 10, b / 48]), x = g.px(p[0]), y = g.py(p[1]);
          b ? g.ctx.lineTo(x, y) : g.ctx.moveTo(x, y);
        }
        g.ctx.stroke();
      }
      g.ctx.setLineDash([]);
    };
    gridLines(sq, p => p);
    GP.frame(sq);
    GP.dots(sq, w.X, { fill: "#1a1a1a", r: 3 });
    if (newest) GP.ring(sq, last, "#1a1a1a", 8);
    // the same points in the warped coordinates, with the image of a regular grid
    sq = GP.square(mapCv, 10);
    const { ctx } = sq;
    ctx.fillStyle = "#fff"; ctx.fillRect(sq.x0, sq.y0, sq.s, sq.s);
    gridLines(sq, w.T);
    GP.frame(sq);
    const WX = w.X.map(w.T);
    GP.dots(sq, WX, { fill: "#1a1a1a", r: 3 });
    if (newest) GP.ring(sq, WX[WX.length - 1], "#1a1a1a", 8);
    GP.lines(errCv, runs.map(r => ({ values: r.err, colour: r.rule.colour })),
      { xStart: n0, xMax: n0 + ROUNDS, xLabel: "number of measurements", yLabel: "reconstruction error" });
    count.textContent = `${runs[0].X.length} / ${n0 + ROUNDS} measurements`;
    play.textContent = timer ? "Pause" : done() ? "Play again" : "Play";
  });
  function stop() { clearInterval(timer); timer = null; }
  play.onclick = () => {
    if (timer) stop();
    else { if (done()) { prev = null; reset(); } timer = setInterval(step, 120); }
    draw();
  };
  q("#w-step").onclick = () => { stop(); if (!done()) step(); };
  q("#w-reset").onclick = () => { stop(); prev = null; reset(); draw(); };
  q("#w-seed").onclick = () => { stop(); prev = null; seed = (seed + 1) % D.x0.length; reset(); draw(); };
  prevBox.onchange = draw;
  viewSeg.onclick = e => { if (e.target.dataset.v) { view = e.target.dataset.v; draw(); } };
  // a new function: same starting points, same number of measurements, sequences recomputed
  fnSel.onchange = () => {
    stop();
    const n = rounds();
    prev = n ? { label: D.functions[fnKey].label, X: runs.map(r => r.X) } : null;
    reset(n);
    draw();
  };
  window.addEventListener("resize", draw);
  reset(START); draw();
})();

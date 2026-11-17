// Bayesian-optimization page: GP-LCB with a weight field on the exploration term,
//   alpha(x) = mu(x) + lambda * g(x) / c * sigma(x),
// for the standardized objective h (larger is better). As in the paper, the mean uses a
// lengthscale fitted each round, and the weighted rules use sigma at a fixed reference
// lengthscale. Here the next point is the best of a 61 x 61 grid.
(function () {
  const root = document.getElementById("bo");
  if (!root) return;
  const B = BO_DATA, NB = B.nb, Cg = GP.grid(NB), M = NB * NB, ELL = B.ell, NOISE = B.noise, BUDGET = 20;
  const RULES = [
    { key: "lcb", field: null, colour: "#2a78d6" },
    { key: "geo", field: GP.field(FIELD_DATA, "geo"), colour: "#eb6834" },
    { key: "uniform", field: GP.field(FIELD_DATA, "uniform"), colour: "#c98700" },
    { key: "eff", field: GP.field(FIELD_DATA, "eff"), colour: "#1baf7a" },
  ];
  const cvs = RULES.map(r => root.querySelector(`#bo-${r.key}`)), outs = RULES.map(r => root.querySelector(`#bo-${r.key}-o`));
  const regCv = root.querySelector("#bo-regret"), fnSel = root.querySelector("#bo-fn");
  const lamIn = root.querySelector("#bo-lam"), lamOut = root.querySelector("#bo-lam-o");
  const play = root.querySelector("#bo-play"), count = root.querySelector("#bo-count");
  for (const [k, f] of Object.entries(B.functions)) fnSel.add(new Option(f.label, k));
  let H, hstar, best, seed = 2, runs, timer = null, view = "landscape";
  const viewSeg = root.querySelector("#bo-view");
  // second panel: the two parts of the score for one rule
  const parts = document.getElementById("parts"), ruleSeg = parts.querySelector("#p-rule");
  const pCv = ["mu", "e", "a"].map(k => parts.querySelector(`#p-${k}`)), pLam = parts.querySelector("#p-lam"), pLamOut = parts.querySelector("#p-lam-o");
  const pSay = parts.querySelector("#p-say");
  let rule = 0;

  // MAP lengthscale under the prior log(ell) ~ N(sqrt(2) + log(D)/2, sqrt(3)), on a log grid.
  const ELLS = Array.from({ length: 36 }, (_, k) => 0.02 * (3 / 0.02) ** (k / 35));
  const MU0 = Math.SQRT2 + 0.5 * Math.log(2), VAR0 = 3;
  function fitLengthscale(X, y) {
    let bestEll = ELLS[0], bestV = Infinity;
    for (const ell of ELLS) {
      const v = GP.build(X, y, ell, NOISE)[0].nll() + Math.log(ell) + (Math.log(ell) - MU0) ** 2 / (2 * VAR0);
      if (v < bestV) { bestV = v; bestEll = ell; }
    }
    return bestEll;
  }
  const h = p => GP.bilinear(H, NB, p[0], p[1]);
  function newRun(rule) {
    const r = { rule, rand: GP.rng(8000 + seed), X: [], y: [], regret: [], top: -Infinity };
    if (rule.field) { r.ref = new GP.Model(ELL, 1e-8); r.set = r.ref.track(Cg); }
    const r0 = GP.rng(1000 + seed);
    for (const p of B.x0[seed]) observe(r, p, h(p) + Math.sqrt(NOISE) * r0.normal());
    return r;
  }
  function observe(r, p, y) {
    r.X.push(p); r.y.push(y);
    if (r.ref) r.ref.add(p);
    r.top = Math.max(r.top, h(p));
  }
  // The model of the current round: the predicted value mu and the exploration term e, so
  // that the next evaluation maximizes mu + lambda * e.
  function model(r) {
    const n = r.X.length;
    const [m, set] = GP.build(r.X, r.y, fitLengthscale(r.X, r.y), NOISE, [Cg]);
    r.mu = m.mean(set); r.e = new Float64Array(M);
    if (!r.rule.field) for (let j = 0; j < M; j++) r.e[j] = Math.sqrt(set.v[j]);
    else {
      const g = r.rule.field.grid(n);
      let c = 0;
      for (const p of r.X) c += GP.bilinear(g, NB, p[0], p[1]);
      c /= n;
      for (let j = 0; j < M; j++) r.e[j] = g[j] / c * Math.sqrt(r.set.v[j]);
    }
  }
  function pick(r) {
    const lam = +lamIn.value, a = new Float64Array(M);
    for (let j = 0; j < M; j++) a[j] = r.mu[j] + lam * r.e[j];
    r.a = a;
    return GP.argmax(a);
  }
  const at = j => [Cg[2 * j], Cg[2 * j + 1]];
  function step() {
    for (const r of runs) {
      const j = pick(r), p = [Cg[2 * j], Cg[2 * j + 1]];
      observe(r, p, h(p) + Math.sqrt(NOISE) * r.rand.normal());
      r.regret.push(Math.max(hstar - r.top, 0));
      model(r);
    }
    if (done()) stop();
    draw();
  }
  const done = () => runs[0].X.length >= BUDGET;
  function reset() {
    const fn = B.functions[fnSel.value];
    H = GP.decode(fn.z); hstar = fn.hstar;
    const j = GP.argmax(H); best = [Cg[2 * j], Cg[2 * j + 1]];
    runs = RULES.map(newRun);
    runs.forEach(r => { r.regret.push(Math.max(hstar - r.top, 0)); model(r); });
  }
  const draw = throttled(() => {
    const n0 = B.x0[seed].length;
    let hlo = Infinity, hhi = -Infinity;
    for (const v of H) { hlo = Math.min(hlo, v); hhi = Math.max(hhi, v); }
    GP.seg(viewSeg, view); GP.seg(ruleSeg, rule);
    runs.forEach((r, k) => {
      const sq = GP.square(cvs[k], 10), n = r.X.length;
      if (view === "landscape") GP.heat(sq, H, NB, { ramp: "grey" });
      else GP.heat(sq, r.mu, NB, { ramp: "grey", lo: hlo, hi: hhi });
      GP.frame(sq);
      GP.ring(sq, best, "#1a1a1a", 11);
      GP.dots(sq, r.X, { to: n0, fill: "#1a1a1a", cross: true, r: 4 });
      GP.dots(sq, r.X, { from: n0, fill: r.rule.colour, stroke: "#1a1a1a", r: 5 });
      r.next = pick(r);
      if (!done()) GP.dots(sq, [at(r.next)], { fill: C.accent, r: 5.5 });
      const wall = r.X.slice(n0).filter(p => Math.min(p[0], 1 - p[0], p[1], 1 - p[1]) < 0.05).length;
      outs[k].textContent = `regret ${r.regret[r.regret.length - 1].toFixed(2)} · at wall ${wall}`;
    });
    GP.lines(regCv, runs.map(r => ({ values: r.regret, colour: r.rule.colour })),
      { xStart: n0, xMax: BUDGET, xLabel: "number of evaluations", yLabel: "regret" });
    count.textContent = `${runs[0].X.length} / ${BUDGET} evaluations`;
    lamOut.textContent = pLamOut.textContent = (+lamIn.value).toFixed(1);
    // predicted value + lambda * exploration bonus = score, for the chosen rule
    const r = runs[rule], top = [GP.argmax(r.mu), GP.argmax(r.e), r.next];
    [[r.mu, "grey"], [r.e, "blue"], [r.a, "blue"]].forEach(([Zm, ramp], k) => {
      const sq = GP.square(pCv[k], 10);
      GP.heat(sq, Zm, NB, { ramp });
      GP.frame(sq);
      GP.dots(sq, r.X, { fill: "#1a1a1a", r: 3 });
      { const p = at(top[k]); sq.ctx.strokeStyle = "#1a1a1a"; sq.ctx.lineWidth = 2; sq.ctx.strokeRect(sq.px(p[0]) - 8, sq.py(p[1]) - 8, 16, 16); }
      GP.dots(sq, [at(r.next)], { fill: C.accent, r: 5.5 });
    });
    const d = (a, b) => Math.hypot(at(a)[0] - at(b)[0], at(a)[1] - at(b)[1]);
    pSay.textContent = d(top[2], top[0]) < 0.08 ? "Here the next evaluation is close to the best predicted value: the rule exploits."
      : d(top[2], top[1]) < 0.08 ? "Here the next evaluation is close to the largest exploration bonus: the rule explores."
      : "Here the next evaluation is a compromise. It is neither where the prediction is best nor where the exploration bonus is largest.";
    play.textContent = timer ? "Pause" : done() ? "Play again" : "Play";
  });
  function stop() { clearInterval(timer); timer = null; }
  play.onclick = () => {
    if (timer) stop();
    else { if (done()) reset(); timer = setInterval(step, 600); }
    draw();
  };
  root.querySelector("#bo-step").onclick = () => { stop(); if (!done()) step(); };
  root.querySelector("#bo-reset").onclick = () => { stop(); reset(); draw(); };
  root.querySelector("#bo-seed").onclick = () => { stop(); seed = (seed + 1) % B.x0.length; reset(); draw(); };
  viewSeg.onclick = e => { if (e.target.dataset.v) { view = e.target.dataset.v; draw(); } };
  fnSel.onchange = () => { stop(); reset(); draw(); };
  // changing lambda moves the next evaluation at once; nothing is evaluated
  lamIn.oninput = () => { pLam.value = lamIn.value; draw(); };
  pLam.oninput = () => { lamIn.value = pLam.value; draw(); };
  ruleSeg.onclick = e => { if (e.target.dataset.v) { rule = +e.target.dataset.v; draw(); } };
  parts.querySelector("#p-take").onclick = () => { stop(); if (!done()) step(); };
  window.addEventListener("resize", draw);
  reset(); draw();
})();

// The three weight fields for a chosen number of observations.
(function () {
  const root = document.getElementById("fields");
  if (!root) return;
  const kinds = ["geo", "uniform", "eff"], F = kinds.map(k => GP.field(FIELD_DATA, k));
  const cvs = kinds.map(k => root.querySelector(`#f-${k}`)), outs = kinds.map(k => root.querySelector(`#f-${k}-o`));
  const nIn = root.querySelector("#f-n"), nOut = root.querySelector("#f-n-o"), cutCv = root.querySelector("#f-cut");
  const COL = ["#eb6834", "#c98700", "#1baf7a"];
  nIn.min = FIELD_DATA.n_lo; nIn.max = FIELD_DATA.n_hi;
  const draw = throttled(() => {
    const N = +nIn.value;
    nOut.textContent = N;
    const nb = FIELD_DATA.nb, mid = (nb - 1) / 2, cuts = [];
    F.forEach((f, k) => {
      const g = f.grid(N), sq = GP.square(cvs[k], 10);
      cuts.push({ values: Array.from(g.subarray(mid * nb, (mid + 1) * nb)), colour: COL[k], noDot: true });
      let lo = Infinity, hi = -Infinity;
      for (const v of g) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      GP.heat(sq, g, FIELD_DATA.nb, { ramp: "viridis", lo: 0, hi: 1.5 });
      GP.frame(sq);
      outs[k].textContent = `smallest ${lo.toFixed(2)}`;
    });
    GP.lines(cutCv, cuts, { xStart: 0, xMax: nb - 1, xTick: i => (i / (nb - 1)).toFixed(2), xLabel: "position from the left wall to the right wall, through the center", yLabel: "weight" });
  });
  nIn.oninput = draw;
  window.addEventListener("resize", draw);
  draw();
})();

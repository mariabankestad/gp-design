// Experimental-design page, part 1: three acquisition functions on the unit square.
// Top row: the score of each for one set of points, computed here. Bottom row: where the
// highest score falls over thousands of random sets, computed in advance with the code of
// the paper (data/density.js). No function is measured anywhere on this panel.
(function () {
  const root = document.getElementById("boundary");
  if (!root) return;
  const DD = DENSITY_DATA, NC = DD.ncand, NT = DD.nint, NOISE = DD.noise, REG = NOISE + 1e-5;
  const lin = n => { const P = new Float64Array(2 * n * n); for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) { P[2 * (r * n + c)] = 0.005 + 0.99 * c / (n - 1); P[2 * (r * n + c) + 1] = 0.005 + 0.99 * r / (n - 1); } return P; };
  const Cg = lin(NC), Tg = lin(NT), M = NC * NC, MT = NT * NT;
  const NAMES = ["var", "nipv", "epig"];
  const top = NAMES.map(k => root.querySelector(`#b-${k}`)), bottom = NAMES.map(k => root.querySelector(`#b-${k}-d`));
  const edgeOut = NAMES.map(k => root.querySelector(`#b-${k}-e`));
  const nSeg = root.querySelector("#b-n"), ellSeg = root.querySelector("#b-ell");
  const rand = GP.rng(11), tables = {};
  let n = 15, ell = 0.2, kCT, X, scores, hist;

  function kernelTable() {
    if (!tables[ell]) {
      const k = new Float32Array(M * MT);
      for (let j = 0; j < M; j++) for (let t = 0; t < MT; t++)
        k[j * MT + t] = GP.m52(Math.hypot(Cg[2 * j] - Tg[2 * t], Cg[2 * j + 1] - Tg[2 * t + 1]), ell);
      tables[ell] = k;
    }
    kCT = tables[ell];
  }
  // Scores of all candidates for the design X: posterior variance, integrated variance
  // reduction (NIPV) and expected predictive information gain (EPIG).
  function acquisitions(X) {
    const [m, sc, st] = GP.build(X, null, ell, REG, [Cg, Tg]), k = m.n;
    const nipv = new Float64Array(M), epig = new Float64Array(M), cov = new Float64Array(MT);
    for (let j = 0; j < M; j++) {
      for (let t = 0; t < MT; t++) cov[t] = kCT[j * MT + t];
      for (let i = 0; i < k; i++) { const a = sc.V[i][j], Vt = st.V[i]; for (let t = 0; t < MT; t++) cov[t] -= a * Vt[t]; }
      const vy = sc.v[j] + NOISE;
      let s1 = 0, s2 = 0;
      for (let t = 0; t < MT; t++) {
        const c2 = cov[t] * cov[t];
        s1 += c2;
        s2 -= 0.5 * Math.log(1 - Math.min(c2 / (vy * (st.v[t] + NOISE)), 1 - 1e-6));
      }
      nipv[j] = s1 / vy / MT; epig[j] = s2 / MT;
    }
    return [Float64Array.from(sc.v), nipv, epig];
  }
  const randomDesign = k => Array.from({ length: k }, () => [rand(), rand()]);

  // counts over the precomputed designs, lightly smoothed for display
  function loadDensity() {
    const q = decodeU16(DD.hist[`${n}_${ell}`]);
    hist = NAMES.map((_, k) => {
      const a = Float64Array.from(q.subarray(k * M, (k + 1) * M)), b = new Float64Array(M);
      let edge = 0;
      for (let j = 0; j < M; j++) if (Math.min(Cg[2 * j], 1 - Cg[2 * j], Cg[2 * j + 1], 1 - Cg[2 * j + 1]) < 0.05) edge += a[j];
      for (let r = 0; r < NC; r++) for (let c = 0; c < NC; c++) {
        let s = 0, w = 0;
        for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
          const rr = r + dr, cc = c + dc;
          if (rr < 0 || cc < 0 || rr >= NC || cc >= NC) continue;
          const g = (dr ? 1 : 2) * (dc ? 1 : 2); s += g * a[rr * NC + cc]; w += g;
        }
        b[r * NC + c] = Math.sqrt(s / w);
      }
      return { map: b, edge: edge / DD.designs };
    });
  }
  function setup(newDesign) {
    GP.seg(nSeg, n);
    GP.seg(ellSeg, ell);
    kernelTable();
    if (newDesign || !X) X = randomDesign(n);
    scores = acquisitions(X);
    loadDensity();
    draw();
  }
  const draw = throttled(() => {
    NAMES.forEach((_, k) => {
      let sq = GP.square(top[k], 10);
      GP.heat(sq, scores[k], NC, { nodes: false });
      GP.frame(sq);
      GP.dots(sq, X, { r: 4.5, fill: "#1a1a1a" });
      const j = GP.argmax(scores[k]);
      GP.ring(sq, [Cg[2 * j], Cg[2 * j + 1]], C.accent, 9);
      sq = GP.square(bottom[k], 10);
      GP.heat(sq, hist[k].map, NC, { ramp: "warm", lo: 0, nodes: false });
      GP.frame(sq);
      edgeOut[k].textContent = `${Math.round(100 * hist[k].edge)} % at the edge`;
    });
  });
  root.querySelector("#b-new").onclick = () => { X = randomDesign(n); scores = acquisitions(X); draw(); };
  nSeg.onclick = e => { if (e.target.dataset.v) { n = +e.target.dataset.v; setup(true); } };
  ellSeg.onclick = e => { if (e.target.dataset.v) { ell = +e.target.dataset.v; setup(false); } };
  top.forEach(cv => cv.addEventListener("click", e => {
    const b = cv.getBoundingClientRect(), sq = GP.square(cv, 10), [x, y] = sq.inv(e.clientX - b.left, e.clientY - b.top);
    if (x < -0.03 || x > 1.03 || y < -0.03 || y > 1.03) return;
    X.push([Math.min(Math.max(x, 0), 1), Math.min(Math.max(y, 0), 1)]);
    scores = acquisitions(X);
    draw();
  }));
  window.addEventListener("resize", draw);
  setup(true);
})();

// Gaussian-process pieces shared by the pages: Matern-5/2 kernel on the unit square,
// a GP that is updated one observation at a time, grids, colours and drawing helpers.
const GP = (() => {
  const S5 = Math.sqrt(5);
  const m52 = (r, ell) => { const z = S5 * r / ell; return (1 + z + z * z / 3) * Math.exp(-z); };

  // n x n grid on [0,1]^2 as [x0, y0, x1, y1, ...]; index = row * n + col, row along y.
  function grid(n, centred = false) {
    const P = new Float64Array(2 * n * n);
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      P[2 * (r * n + c)] = centred ? (c + 0.5) / n : c / (n - 1);
      P[2 * (r * n + c) + 1] = centred ? (r + 0.5) / n : r / (n - 1);
    }
    return P;
  }

  // Zero-mean, unit-variance GP. Point sets registered with track() keep V = L^-1 k(X, P)
  // and the posterior variance v, so adding an observation costs O(n |P|).
  class Model {
    constructor(ell, reg) { this.ell = ell; this.reg = reg; this.X = []; this.y = []; this.L = []; this.sets = []; }
    get n() { return this.X.length; }
    track(P) {
      const m = P.length / 2, s = { P, m, V: [], v: new Float64Array(m).fill(1) };
      this.sets.push(s);
      return s;
    }
    add(x, y = 0) {
      const n = this.n, ell = this.ell, l = new Float64Array(n + 1);
      let ss = 0;
      for (let i = 0; i < n; i++) {
        const xi = this.X[i], Li = this.L[i];
        let s = m52(Math.hypot(x[0] - xi[0], x[1] - xi[1]), ell);
        for (let j = 0; j < i; j++) s -= Li[j] * l[j];
        l[i] = s / Li[i];
        ss += l[i] * l[i];
      }
      const d = Math.sqrt(Math.max(1 + this.reg - ss, 1e-10));
      l[n] = d;
      for (const s of this.sets) {
        const row = new Float64Array(s.m), P = s.P;
        for (let j = 0; j < s.m; j++) row[j] = m52(Math.hypot(x[0] - P[2 * j], x[1] - P[2 * j + 1]), ell);
        for (let i = 0; i < n; i++) {
          const li = l[i], Vi = s.V[i];
          if (li !== 0) for (let j = 0; j < s.m; j++) row[j] -= li * Vi[j];
        }
        for (let j = 0; j < s.m; j++) { row[j] /= d; s.v[j] = Math.max(s.v[j] - row[j] * row[j], 1e-8); }
        s.V.push(row);
      }
      this.X.push([x[0], x[1]]); this.y.push(y); this.L.push(l);
      return this;
    }
    // w = L^-1 y, so that mean(P) = V^T w
    whitened() {
      const n = this.n, w = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let s = this.y[i];
        for (let j = 0; j < i; j++) s -= this.L[i][j] * w[j];
        w[i] = s / this.L[i][i];
      }
      return w;
    }
    mean(set) {
      const w = this.whitened(), mu = new Float64Array(set.m);
      for (let i = 0; i < this.n; i++) { const wi = w[i], Vi = set.V[i]; for (let j = 0; j < set.m; j++) mu[j] += wi * Vi[j]; }
      return mu;
    }
    // a = K^-1 y
    weights() {
      const n = this.n, a = this.whitened();
      for (let i = n - 1; i >= 0; i--) {
        let s = a[i];
        for (let j = i + 1; j < n; j++) s -= this.L[j][i] * a[j];
        a[i] = s / this.L[i][i];
      }
      return a;
    }
    // negative log marginal likelihood, without the constant
    nll() {
      const w = this.whitened();
      let v = 0;
      for (let i = 0; i < this.n; i++) v += 0.5 * w[i] * w[i] + Math.log(this.L[i][i]);
      return v;
    }
  }

  function build(X, y, ell, reg, sets = []) {
    const m = new Model(ell, reg), out = sets.map(P => m.track(P));
    for (let i = 0; i < X.length; i++) m.add(X[i], y ? y[i] : 0);
    return [m, ...out];
  }

  // ---------------------------------------------------------------- random numbers
  function rng(seed) {
    let a = seed >>> 0;
    const u = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
    u.normal = () => Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u());
    return u;
  }

  // n points of the two-dimensional Sobol sequence with a random digital shift
  function sobol(n, seed) {
    const u = rng(seed), sx = (u() * 4294967296) >>> 0, sy = (u() * 4294967296) >>> 0, V = [0x80000000];
    for (let k = 1; k < 32; k++) V.push((V[k - 1] ^ (V[k - 1] >>> 1)) >>> 0);
    const P = [];
    let x = 0, y = 0;
    for (let i = 0; i < n; i++) {
      if (i) { let c = 0; while (!((i >>> c) & 1)) c++; x = (x ^ (0x80000000 >>> c)) >>> 0; y = (y ^ V[c]) >>> 0; }
      P.push([((x ^ sx) >>> 0) / 4294967296, ((y ^ sy) >>> 0) / 4294967296]);
    }
    return P;
  }

  // ---------------------------------------------------------------- lookups
  function decode(d) {
    const q = decodeU16(d.b64), out = new Float64Array(q.length), s = (d.hi - d.lo) / 65535;
    for (let i = 0; i < q.length; i++) out[i] = d.lo + s * q[i];
    return out;
  }
  // bilinear interpolation of an n x n grid (row along y) at (x, y)
  function bilinear(Z, n, x, y) {
    const fx = Math.min(Math.max(x, 0), 1) * (n - 1), fy = Math.min(Math.max(y, 0), 1) * (n - 1);
    const c = Math.min(Math.floor(fx), n - 2), r = Math.min(Math.floor(fy), n - 2), a = fx - c, b = fy - r;
    return (1 - b) * ((1 - a) * Z[r * n + c] + a * Z[r * n + c + 1]) + b * ((1 - a) * Z[(r + 1) * n + c] + a * Z[(r + 1) * n + c + 1]);
  }
  const argmax = a => { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i; return k; };

  // ---------------------------------------------------------------- drawing
  const RAMPS = {
    blue: [[255, 255, 255], [226, 235, 244], [158, 190, 222], [88, 140, 196], [44, 96, 150], [31, 78, 121]],
    grey: [[96, 102, 110], [160, 166, 172], [214, 218, 223], [248, 249, 250]],
    warm: [[255, 255, 255], [253, 231, 214], [246, 173, 120], [232, 89, 12], [150, 50, 5]],
  };
  function colour(ramp, t) {
    const R = RAMPS[ramp], x = Math.min(Math.max(t, 0), 1) * (R.length - 1), i = Math.min(Math.floor(x), R.length - 2), f = x - i;
    return [0, 1, 2].map(k => R[i][k] + f * (R[i + 1][k] - R[i][k]));
  }
  // The square plotting area of a canvas, with y pointing up.
  function square(canvas, pad = 10) {
    const { ctx, w, h } = fitCanvas(canvas), s = Math.min(w, h) - 2 * pad, x0 = (w - s) / 2, y0 = (h - s) / 2;
    return { ctx, w, h, s, x0, y0, px: x => x0 + x * s, py: y => y0 + (1 - y) * s,
             inv: (cx, cy) => [(cx - x0) / s, 1 - (cy - y0) / s] };
  }
  const scratch = document.createElement("canvas");
  // Heatmap of an n x n grid of values at the grid nodes, filling the square.
  function heat(sq, Z, n, { ramp = "blue", lo = null, hi = null, nodes = true } = {}) {
    if (lo === null) { lo = Infinity; for (const v of Z) if (v < lo) lo = v; }
    if (hi === null) { hi = -Infinity; for (const v of Z) if (v > hi) hi = v; }
    scratch.width = n; scratch.height = n;
    const sc = scratch.getContext("2d"), img = sc.createImageData(n, n), span = hi - lo || 1;
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const [R, G, B] = colour(ramp, (Z[r * n + c] - lo) / span), k = 4 * ((n - 1 - r) * n + c);
      img.data[k] = R; img.data[k + 1] = G; img.data[k + 2] = B; img.data[k + 3] = 255;
    }
    sc.putImageData(img, 0, 0);
    const { ctx, x0, y0, s } = sq;
    ctx.save();
    ctx.beginPath(); ctx.rect(x0, y0, s, s); ctx.clip();
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    const e = nodes ? s / (n - 1) / 2 : 0;       // node-centred pixels extend half a cell outside
    ctx.drawImage(scratch, x0 - e, y0 - e, s + 2 * e, s + 2 * e);
    ctx.restore();
  }
  function frame(sq) {
    const { ctx, x0, y0, s } = sq;
    ctx.strokeStyle = "#b8c0c8"; ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y0 + 0.5, s - 1, s - 1);
  }
  function dots(sq, X, { from = 0, to = X.length, fill = C.navy, stroke = "#fff", r = 4, cross = false } = {}) {
    const { ctx } = sq;
    for (let i = from; i < to; i++) {
      const x = sq.px(X[i][0]), y = sq.py(X[i][1]);
      ctx.beginPath();
      if (cross) {
        ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x - r, y + r); ctx.lineTo(x + r, y - r);
        ctx.strokeStyle = fill; ctx.lineWidth = 1.8; ctx.stroke();
      } else {
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke();
      }
    }
  }
  function ring(sq, p, colour = C.accent, r = 8) {
    const { ctx } = sq;
    ctx.beginPath(); ctx.arc(sq.px(p[0]), sq.py(p[1]), r, 0, 2 * Math.PI);
    ctx.strokeStyle = colour; ctx.lineWidth = 2; ctx.stroke();
  }
  // Line chart: series = [{ values, colour }], x from x0 in steps of 1.
  function lines(canvas, series, { xStart = 0, xMax, yLabel = "", xLabel = "", log = false, xTick = i => i, mark = null } = {}) {
    const { ctx, w, h } = fitCanvas(canvas), L = 58, R = 12, T = 10, B = 34;
    let lo = Infinity, hi = -Infinity;
    for (const s of series) for (const v of s.values) { const t = log ? Math.log10(Math.max(v, 1e-6)) : v; lo = Math.min(lo, t); hi = Math.max(hi, t); }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (!log) lo = Math.min(lo, 0);
    if (hi - lo < 1e-9) hi = lo + 1;
    const pad = 0.06 * (hi - lo); hi += pad; if (log) lo -= pad;
    const X = i => L + (i - xStart) / Math.max(xMax - xStart, 1) * (w - L - R), Y = v => T + (hi - (log ? Math.log10(Math.max(v, 1e-6)) : v)) / (hi - lo) * (h - T - B);
    ctx.strokeStyle = C.grid; ctx.fillStyle = C.muted; ctx.lineWidth = 1; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let k = 0; k <= 4; k++) {
      const t = lo + (hi - lo) * k / 4, y = T + (hi - t) / (hi - lo) * (h - T - B), val = log ? 10 ** t : t;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(w - R, y); ctx.stroke();
      ctx.fillText(val >= 10 ? val.toFixed(0) : val >= 1 ? val.toFixed(1) : val.toFixed(2), L - 6, y);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let k = 0; k <= 4; k++) { const i = Math.round(xStart + (xMax - xStart) * k / 4); ctx.fillText(xTick(i), X(i), h - B + 6); }
    ctx.fillText(xLabel, (L + w - R) / 2, h - 15);
    ctx.save(); ctx.translate(9, (T + h - B) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "middle"; ctx.fillText(yLabel, 0, 0); ctx.restore();
    if (mark !== null) { ctx.beginPath(); ctx.moveTo(X(mark), T); ctx.lineTo(X(mark), h - B); ctx.strokeStyle = C.accent; ctx.lineWidth = 1.5; ctx.stroke(); }
    for (const s of series) {
      ctx.beginPath();
      s.values.forEach((v, i) => i ? ctx.lineTo(X(xStart + i), Y(v)) : ctx.moveTo(X(xStart + i), Y(v)));
      ctx.strokeStyle = s.colour; ctx.lineWidth = 2; ctx.stroke();
      if (s.values.length && !s.noDot) { const i = s.values.length - 1; ctx.beginPath(); ctx.arc(X(xStart + i), Y(s.values[i]), 3, 0, 2 * Math.PI); ctx.fillStyle = s.colour; ctx.fill(); }
    }
  }

  // Weight field g(x; N) stored on one quadrant of an nb x nb grid for N = n_lo..n_hi.
  function field(data, kind) {
    const all = decode(data.fields[kind]), { nb, nq, n_lo, n_hi } = data, half = nb - 1;
    return {
      nb,
      grid(N) {
        const k = Math.min(Math.max(N, n_lo), n_hi) - n_lo, G = new Float64Array(nb * nb);
        for (let r = 0; r < nb; r++) for (let c = 0; c < nb; c++)
          G[r * nb + c] = all[k * nq * nq + Math.min(r, half - r) * nq + Math.min(c, half - c)];
        return G;
      },
    };
  }

  // Mark the button of a segmented control whose data-v equals the value.
  function seg(el, value) {
    for (const b of el.children) { const on = b.dataset.v === String(value); b.classList.toggle("on", on); b.setAttribute("aria-pressed", on); }
  }

  return { seg, sobol, m52, grid, Model, build, rng, decode, bilinear, argmax, colour, square, heat, frame, dots, ring, lines, field };
})();
if (typeof module !== "undefined") module.exports = GP;

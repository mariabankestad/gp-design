"""Export what the pages need: test functions on grids, the 2-D equalizer table,
the three BO weight fields, and Sobol initial designs.

    python pack_data.py <warp_gp release dir> <spatial-bo dir>
"""
import base64, itertools, json, math, os, sys
import numpy as np
import torch

REL, SBO = sys.argv[1], sys.argv[2]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "data")
torch.set_default_dtype(torch.float64)


def u16(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    lo, hi = float(a.min()), float(a.max())
    q = np.round((a - lo) / max(hi - lo, 1e-300) * 65535).astype("<u2")
    return {"lo": lo, "hi": hi, "b64": base64.b64encode(q.tobytes()).decode()}


def grid(n):
    g = torch.linspace(0, 1, n)
    return torch.stack(torch.meshgrid(g, g, indexing="xy"), -1).reshape(-1, 2)  # row = y, col = x


def write(name, var, obj):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        f.write(f"const {var} = " + json.dumps(obj, separators=(",", ":")) + ";\n")
    print(name, f"{os.path.getsize(p) / 1e3:.0f} kB")


# ---------------------------------------------------------------- experimental design
sys.path.insert(0, os.path.join(REL, "src"))
from warp_gp.functions import get_function, standardized
from warp_gp.loop import sobol, TABLE_NOISE
from warp_gp.gp import default_lengthscale
from warp_gp.equalizer import EqualizerTable

NG = 81
names = {"smooth_box_2d": "Smooth box", "peaks": "Peaks", "grlee08": "Gramacy-Lee",
         "zhou98": "Zhou", "levy2d": "Levy", "beale": "Beale"}
funcs = {}
for k, label in names.items():
    f, D = get_function(k)
    fn, _ = standardized(k)
    pilot = f(sobol(2, 4096, 0)).reshape(-1)
    sig = max(0.05 * float(pilot.std(unbiased=False)), 1e-12)
    fs = f(sobol(2, 8192, 999)).reshape(-1).std().clamp_min(1e-8)
    noise = max((sig / float(fs)) ** 2, 1e-4)
    funcs[k] = {"label": label, "noise": noise, "z": u16(fn(grid(NG)).numpy())}
tab = EqualizerTable(2, default_lengthscale(2), TABLE_NOISE)
eq = {}
for key in sorted(tab.cache, key=lambda s: int(s[1:])):
    tab._build(int(key[1:]))
    eq[key[1:]] = [round(float(v), 5) for v in tab.G]
write("design_data.js", "DESIGN_DATA", {
    "ng": NG, "ell": default_lengthscale(2), "functions": funcs,
    "levels": list(EqualizerTable.LEVELS), "equalizer": eq,
    "x0": [sobol(2, 20, s).numpy().round(5).tolist() for s in range(8)]})
for m in [m for m in sys.modules if m.startswith("warp_gp")]:
    del sys.modules[m]
sys.path.pop(0)

# ---------------------------------------------------------------- Bayesian optimization
sys.path.insert(0, SBO)
from spatialbo import FUNCS, load_field, reference_lengthscale, standardized as bstd
from spatialbo.functions import MB4_POS

NB, NQ, N_LO, N_HI = 61, 31, 4, 80
ell = reference_lengthscale(2)
Q = grid(NB).reshape(NB, NB, 2)[:NQ, :NQ].reshape(-1, 2)   # one quadrant; the fields are symmetric
fields = {}
for kind in ("geo", "uniform", "eff"):
    g = load_field(kind, 2)
    full = g(grid(NB), 20, ell).reshape(NB, NB)
    assert torch.allclose(full, full.flip(0), atol=1e-6) and torch.allclose(full, full.T, atol=1e-6)
    with torch.no_grad():
        fields[kind] = u16(torch.stack([g(Q, n, ell) for n in range(N_LO, N_HI + 1)]).numpy())
write("fields.js", "FIELD_DATA", {"nb": NB, "nq": NQ, "n_lo": N_LO, "n_hi": N_HI, "ell": ell, "fields": fields})

bnames = {"mb4_2d_interior": "Four basins, best one in the interior", "mb4_2d_face": "Four basins, best one at a wall",
          "mb4_2d_corner": "Four basins, best one in a corner", "branin": "Branin", "camel6": "Six-hump camel"}
bf = {}
for k, label in bnames.items():
    h, _ = bstd(k)
    hs = float(h(torch.quasirandom.SobolEngine(2, scramble=True, seed=3).draw(2 ** 17)).max())
    z = h(grid(NB))
    bf[k] = {"label": label, "hstar": max(hs, float(z.max())), "z": u16(z.numpy())}
write("bo_data.js", "BO_DATA", {
    "nb": NB, "ell": ell, "noise": 0.0025, "functions": bf, "basins": MB4_POS[2],
    "x0": [torch.quasirandom.SobolEngine(2, scramble=True, seed=100 + s).draw(4).clamp(1e-3, 1 - 1e-3)
           .numpy().round(5).tolist() for s in range(8)]})

"""Where the next point lands over many random designs, for variance maximization, NIPV and
EPIG, computed with the code of the boundary-bias paper (Matern-5/2, noise 0.003).

    python pack_density.py <gp-boundary-bias dir> [n_designs]
"""
import base64, json, os, sys
import numpy as np
import torch

REPO, NS = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4000
sys.path.insert(0, REPO)
from src.kernels import matern52_isotropic
from src.acquisitions import vm_nipv_epig

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "data", "density.js")
dt, NCAND, NINT, SIGMA2 = torch.float64, 64, 30, 0.003
NS_LIST, ELLS = [8, 15, 30], [0.1, 0.2, 0.4]


def grid(n):
    g = torch.linspace(0.005, 0.995, n, dtype=dt)
    yy, xx = torch.meshgrid(g, g, indexing="ij")          # row along y, column along x
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], -1)


C, T = grid(NCAND), grid(NINT)
out = {"ncand": NCAND, "nint": NINT, "noise": SIGMA2, "designs": NS, "n": NS_LIST, "ell": ELLS, "hist": {}}
for ell in ELLS:
    ls = torch.tensor(ell, dtype=dt)
    K = matern52_isotropic(T, C, ls)
    for n in NS_LIST:
        h = np.zeros((3, NCAND * NCAND), dtype="<u2")
        for s in range(NS):
            X = torch.rand(n, 2, dtype=dt, generator=torch.Generator().manual_seed(s))
            for k, a in enumerate(vm_nipv_epig(C, T, X, ls, matern52_isotropic, SIGMA2, K)):
                h[k, int(a.argmax())] += 1
        out["hist"][f"{n}_{ell}"] = base64.b64encode(h.tobytes()).decode()
        print(n, ell, flush=True)
with open(OUT, "w") as f:
    f.write("const DENSITY_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n")
print("wrote", OUT, os.path.getsize(OUT) // 1000, "kB")

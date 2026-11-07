"""Change of the albedo from normal incidence to 30 deg, all models.

Analytic models: results/models.npz (0 deg) and results/sza30/models.npz. Photontracer: the 1e5-ray, seed-1 runs at
both angles (results/benchmark/ and results/sza30/), so the same microstructure is compared (paired difference).
    python scripts/plot_incidence.py
Writes figures/albedo_change_sza30.png.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cases import CASES, case_name  # noqa: E402

RES = ROOT / "results"
RUNS = {0: (RES / "models.npz", RES / "benchmark"), 30: (RES / "sza30" / "models.npz", RES / "sza30")}
ANALYTIC = {"tartes": ("TARTES (default shape)", "#0072BD", "-"),
            "pysnicar": ("PySNICAR-ADv4 (spheres)", "#D95319", "-"),
            "mie_disort": ("Mie + DISORT (spheres)", "#77AC30", "--"),
            "malinka_random": ("Malinka, random shape", "#7E2F8E", "-."),
            "malinka_markov": ("Malinka, Markov mixture", "#EDB120", ":")}
TRACED = {"grs": ("photontracer, GRS", "#000000", "photontracer_grs/{case}.npz"),
          "grf": ("photontracer, GRF (seed 1)", "#E7298A", "photontracer_grf/{case}_seed1.npz")}

fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, (rho, ssa) in zip(axs, CASES):
    case = case_name(rho, ssa)
    z0, z30 = (np.load(RUNS[a][0]) for a in (0, 30))
    wl = z0["wavelength_m"] * 1e9
    for key, (label, colour, style) in ANALYTIC.items():
        ax.plot(wl, z30[f"{case}/{key}"] - z0[f"{case}/{key}"], color=colour, ls=style, lw=1.4, label=label)
    for key, (label, colour, path) in TRACED.items():
        a0, a30 = (np.load(RUNS[a][1] / path.format(case=case)) for a in (0, 30))
        d = a30["albedo"] - a0["albedo"]
        err = np.hypot(a0["albedo_error"], a30["albedo_error"])
        w = a0["wavelength_nm"]
        ax.fill_between(w, d - err, d + err, color=colour, alpha=0.15, lw=0)
        ax.plot(w, d, color=colour, lw=1.2, label=label + " (1e5 rays, ±1σ)")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set(xlabel="wavelength (nm)", title=f"{rho:g} kg m$^{{-3}}$, SSA {ssa:g} m$^2$ kg$^{{-1}}$")
    ax.grid(alpha=0.25)
axs[0].set_ylabel("albedo(30°) − albedo(0°)")
axs[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "albedo_change_sza30.png", dpi=160, bbox_inches="tight")

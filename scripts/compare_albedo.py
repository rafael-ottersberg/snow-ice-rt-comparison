"""Plot the spectral albedo of all models for the comparison cases (needs results/models.npz and
results/photontracer_grs/<case>.npz and results/photontracer_grf/<case>_seed*.npz; curves that are missing are left out).
    python scripts/compare_albedo.py
Writes figures/albedo_<case>.png, figures/mie_vs_pysnicar.png and figures/albedo_table.md (with SNOWRT_SZA != 0:
albedo_<case>_sza<angle>.png and albedo_table_sza<angle>.md; SNOWRT_RESULTS selects the run).
Two figures per case, both with the difference to photontracer GRS (grey band: its +-1 sigma statistical error):
albedo_<case>.png (TARTES, PySNICAR, Malinka Markov, photontracer GRS and GRF) and albedo_<case>_all.png (all models plus
TARTES with the `linear` shape option, g rising with absorption). The table has all models.
albedo_<case>_slide.png: the reduced figure for half of a 16:9 slide (two side by side), larger text, short labels and
the same difference axis for both cases.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import tartes

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cases import CASES, RESULTS, SZA_DEG, case_name  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
SUFFIX = "" if SZA_DEG == 0 else f"_sza{SZA_DEG:g}"
MODELS = {   # key: (label, colour, linestyle)
    "tartes": ("TARTES (default shape)", "#0072BD", "-"),
    "tartes_linear": ("TARTES (`linear` shape)", "#0072BD", (0, (2, 2))),
    "pysnicar": ("PySNICAR-ADv4 (spheres)", "#D95319", "-"),
    "mie_disort": ("Mie + DISORT (spheres)", "#77AC30", "--"),
    "malinka_random": ("Malinka, random shape", "#7E2F8E", "-."),
    "malinka_markov": ("Malinka, Markov mixture", "#EDB120", ":"),
    "photontracer_grs": ("photontracer, GRS ($\\sigma_r$=0.1)", "#000000", "-"),
    "photontracer_grf": ("photontracer, GRF microstructure", "#E7298A", "-"),
}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130,
                     "savefig.bbox": "tight"})


def load(case):
    z = np.load(RESULTS / "models.npz")
    wl = z["wavelength_m"] * 1e9
    curves = {k: (wl, z[f"{case}/{k}"], None) for k in MODELS if f"{case}/{k}" in z}
    rho, ssa = (float(x) for x in case.replace("rho", "").split("_ssa"))
    curves["tartes_linear"] = (wl, tartes.albedo(wl * 1e-9, ssa, rho, refrac_index="w2008", dir_frac=1.0, sza=SZA_DEG,
                                                  shape_parameterization="linear"), None)
    grs = RESULTS / "photontracer_grs" / f"{case}.npz"
    if grs.exists():
        r = np.load(grs)
        curves["photontracer_grs"] = (r["wavelength_nm"].astype(float), r["albedo"], r["albedo_error"])
    grf = [np.load(p) for p in sorted((RESULTS / "photontracer_grf").glob(f"{case}_seed*.npz"))]
    if grf:   # mean over seeds, spread as error
        a = np.array([r["albedo"] for r in grf])
        curves["photontracer_grf"] = (grf[0]["wavelength_nm"].astype(float), a.mean(0), a.std(0) if len(grf) > 1 else
                                      grf[0]["albedo_error"])
        curves["photontracer_grf_seeds"] = len(grf)
    return curves


VARIANTS = {"": ("tartes", "pysnicar", "malinka_markov", "photontracer_grs", "photontracer_grf"),   # file tag: curves
            "_all": tuple(MODELS)}
REFERENCE = "photontracer_grs"


def reference(curves, wl):
    r_wl, r, _ = curves[REFERENCE]
    return np.interp(wl, r_wl, r)


TABLE = [k for k in MODELS if k != "tartes_linear"]
rows = ["| case | λ (nm) | " + " | ".join(MODELS[k][0] for k in TABLE) + " |", "|---|---|" + "---|" * len(TABLE)]
for rho, ssa in CASES:
    case = case_name(rho, ssa)
    curves = load(case)
    seeds = curves.pop("photontracer_grf_seeds", 0)
    for tag, plotted in VARIANTS.items():
        fig, (ax, axd) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        for key in plotted:
            if key not in curves:
                continue
            wl, a, err = curves[key]
            label, colour, style = MODELS[key]
            if key == "photontracer_grf":
                label += f" ({seeds} seed{'s' if seeds > 1 else ''})"
            lw = 1.0 if key == "tartes_linear" else 1.2 if key.startswith("photontracer") else 1.6
            ax.plot(wl, a, color=colour, ls=style, lw=lw, label=label)
            if key == REFERENCE:
                axd.fill_between(wl, -err, err, color=colour, alpha=0.15, lw=0)
            else:
                axd.plot(wl, a - reference(curves, wl), color=colour, ls=style, lw=lw)
        ax.set(ylabel="albedo", ylim=(0, 1.02),
               title=f"Semi-infinite snow, density {rho:g} kg m$^{{-3}}$, SSA {ssa:g} m$^2$ kg$^{{-1}}$, "
                     f"direct beam {SZA_DEG:g}°")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        axd.axhline(0, color="0.4", lw=0.8)
        axd.set(xlabel="wavelength (nm)", ylabel="model − photontracer GRS")
        fig.tight_layout()
        fig.savefig(FIG / f"albedo_{case}{tag}{SUFFIX}.png", dpi=160)
        plt.close(fig)
    for lam in (500, 1030, 1300, 1500, 1800):
        cells = []
        for key in TABLE:
            if key in curves:
                wl, a, _ = curves[key]
                cells.append(f"{a[np.argmin(abs(wl - lam))]:.4f}")
            else:
                cells.append("–")
        rows.append(f"| {rho:g} / {ssa:g} | {lam} | " + " | ".join(cells) + " |")

# presentation version of the reduced figure: half a 16:9 slide each, same difference axis for both cases
SLIDE_LABELS = {"tartes": "TARTES", "pysnicar": "PySNICAR (spheres)", "malinka_markov": "Malinka, Markov",
                "photontracer_grs": "photontracer, GRS", "photontracer_grf": "photontracer, GRF"}
slide_curves = {case_name(r, s): load(case_name(r, s)) for r, s in CASES}
diffs = [c[k][1] - reference(c, c[k][0]) for c in slide_curves.values() for k in SLIDE_LABELS if k in c]
lo, hi = min(d.min() for d in diffs), max(d.max() for d in diffs)
pad = 0.08 * (hi - lo)
with plt.rc_context({"font.size": 14, "axes.titlesize": 15, "legend.fontsize": 12, "lines.linewidth": 2.2}):
    for (rho, ssa), (case, curves) in zip(CASES, slide_curves.items()):
        curves.pop("photontracer_grf_seeds", None)
        fig, (ax, axd) = plt.subplots(2, 1, figsize=(6.0, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.8, 1]})
        for key, label in SLIDE_LABELS.items():
            if key not in curves:
                continue
            wl, a, err = curves[key]
            _, colour, style = MODELS[key]
            ax.plot(wl, a, color=colour, ls=style, label=label, zorder=3 if key == REFERENCE else 2,
                    lw=1.6 if key == REFERENCE else None)
            if key == REFERENCE:
                axd.fill_between(wl, -err, err, color=colour, alpha=0.15, lw=0)
            else:
                axd.plot(wl, a - reference(curves, wl), color=colour, ls=style)
        ax.set(ylabel="albedo", ylim=(0, 1.02), xlim=(400, 2500),
               title=f"{rho:g} kg m$^{{-3}}$, SSA {ssa:g} m$^2$ kg$^{{-1}}$, {SZA_DEG:g}°")
        ax.legend(frameon=False, loc="upper right")
        axd.axhline(0, color="0.4", lw=1.0)
        axd.set(xlabel="wavelength (nm)", ylabel="Δ vs GRS", ylim=(lo - pad, hi + pad))
        fig.tight_layout(h_pad=0.4)
        fig.savefig(FIG / f"albedo_{case}_slide{SUFFIX}.png", dpi=200)
        plt.close(fig)

# Mie + DISORT against PySNICAR (both: independent ice spheres, same index and SSA)
(FIG / f"albedo_table{SUFFIX}.md").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
if SZA_DEG != 0:
    sys.exit()
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, (rho, ssa) in zip(axs, CASES):
    curves = load(case_name(rho, ssa))
    wl = curves["pysnicar"][0]
    ax.plot(wl, curves["mie_disort"][1] - curves["pysnicar"][1], color="#77AC30")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set(xlabel="wavelength (nm)", title=f"{rho:g} kg m$^{{-3}}$, SSA {ssa:g}")
axs[0].set_ylabel("Mie + DISORT − PySNICAR")
fig.tight_layout()
fig.savefig(FIG / "mie_vs_pysnicar.png", dpi=160)

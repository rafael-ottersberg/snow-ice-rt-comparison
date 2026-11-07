"""Plot the GRF convergence studies of grf_convergence.py: albedo and free-column fraction against cell size N, with the
scatter over seeds; and the other studies (depth, feature) if they have been run. Writes figures/grf_convergence_*.png
and a table figures/grf_convergence.md.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cases import CASES, RESULTS, case_name  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130, "savefig.bbox": "tight"})


def load(study, case):
    return [json.loads(p.read_text()) for p in sorted((RESULTS / "grf_convergence" / study).glob(f"{case}_*.json"))]


def plot_cell():
    fig, axs = plt.subplots(len(CASES), 2, figsize=(11, 4 * len(CASES)), squeeze=False)
    rows = ["| case | N | cell (mm) | free columns | " + " | ".join(f"{w} nm" for w in (900, 1030, 1300, 1800)) + " |",
            "|---|---|---|---|---|---|---|---|"]
    for (rho, ssa), (ax, axf) in zip(CASES, axs):
        runs = load("cell", case_name(rho, ssa))
        if not runs:
            continue
        wl = np.array(runs[0]["wavelength_nm"])
        ns = sorted({r["setting"]["n"] for r in runs})
        for w, colour in zip((900, 1030, 1300, 1800), ("#0072BD", "#D95319", "#77AC30", "#7E2F8E")):
            j = int(np.argmin(abs(wl - w)))
            mean = [np.mean([r["albedo"][j] for r in runs if r["setting"]["n"] == n]) for n in ns]
            std = [np.std([r["albedo"][j] for r in runs if r["setting"]["n"] == n]) for n in ns]
            ax.errorbar(ns, mean, yerr=std, marker="o", color=colour, capsize=3, label=f"{w} nm")
            for r in runs:
                ax.plot(r["setting"]["n"], r["albedo"][j], ".", color=colour, alpha=0.4)
        ax.set(xscale="log", xlabel="cell edge N (voxels)", ylabel="albedo",
               title=f"{rho:g} kg m$^{{-3}}$, SSA {ssa:g}: mean $\\pm$ std over seeds")
        ax.legend(frameon=False, fontsize=8)
        axf.plot([r["setting"]["n"] for r in runs], [r["free_column_fraction"] for r in runs], "o", color="0.3")
        axf.set_yscale("symlog", linthresh=1e-3)
        axf.set(xscale="log", xlabel="cell edge N (voxels)",
                ylabel="vertical lines without ice", title="straight channels through the repeated cell")
        for n in ns:
            sel = [r for r in runs if r["setting"]["n"] == n]
            cells = [f"{np.mean([r['albedo'][int(np.argmin(abs(wl - w)))] for r in sel]):.3f} "
                     f"± {np.std([r['albedo'][int(np.argmin(abs(wl - w)))] for r in sel]):.3f}" for w in (900, 1030, 1300, 1800)]
            rows.append(f"| {rho:g} / {ssa:g} | {n} | {sel[0]['cell_mm']:.2f} | "
                        f"{np.mean([r['free_column_fraction'] for r in sel]):.3f} | " + " | ".join(cells) + " |")
    fig.tight_layout()
    fig.savefig(FIG / "grf_convergence_cell.png", dpi=160)
    plt.close(fig)
    return rows


def plot_other(study, key, label):
    fig, axs = plt.subplots(1, len(CASES), figsize=(5.5 * len(CASES), 4), squeeze=False)
    drawn = False
    for (rho, ssa), ax in zip(CASES, axs[0]):
        runs = load(study, case_name(rho, ssa))
        for r in runs:
            drawn = True
            ax.plot(r["wavelength_nm"], r["albedo"], marker=".", label=key(r["setting"]))
        ax.set(xlabel="wavelength (nm)", ylabel="albedo", title=f"{rho:g} kg m$^{{-3}}$, SSA {ssa:g}")
        if runs:
            ax.legend(frameon=False, fontsize=7, title=label)
    if drawn:
        fig.tight_layout()
        fig.savefig(FIG / f"grf_convergence_{study}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    rows = plot_cell()
    plot_other("depth", lambda s: f"{s['d1'] * s['d2']} deep, {s['l1'] * s['l2']} wide", "cells")
    plot_other("feature", lambda s: f"{s['feature']:g} vox", "feature size")
    (FIG / "grf_convergence.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))

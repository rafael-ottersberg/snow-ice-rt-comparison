"""Spectral transmission of finite snow slabs, TARTES vs PySNICAR-ADv4, for the comparison cases.

A 10 cm slab over a black ground (soil albedo 0), direct beam at SNOWRT_SZA (default nadir). Transmission
is the downward irradiance at the bottom of the slab over the incident irradiance (TARTES: irradiance_profiles at
z = thickness; PySNICAR: net flux into the ground, equal to it over a black ground). TARTES uses its default shape
(robledano23: B = n^2, g = 0.82).
    python scripts/compare_transmission.py
Writes figures/transmission{SUFFIX}.png and prints the transmission at a few wavelengths.
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

import pysnicar  # noqa: E402
from cases import CASES, SZA_DEG, case_name, wavelengths_m  # noqa: E402

FIG = ROOT / "figures"
SUFFIX = "" if SZA_DEG == 0 else f"_sza{SZA_DEG:g}"
THICKNESS_CM = (10,)
COLOURS = ("#0072BD", "#D95319", "#77AC30", "#7E2F8E")
FLOOR = 1e-5                                    # ratio only where both are above this


def tartes_transmission(wl, ssa, rho, h):
    down, up = tartes.irradiance_profiles(wl, [0.0, h], ssa, rho, thickness=h, refrac_index="w2008", soilalbedo=0.0,
                                          dir_frac=1.0, sza=SZA_DEG)
    return down[:, 1] / down[:, 0]                 # (wavelength, depth)


wl = wavelengths_m()
fig, axs = plt.subplots(2, len(CASES), figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
rows = []
for col, (rho, ssa) in enumerate(CASES):
    ax, axd = axs[:, col]
    for h_cm, colour in zip(THICKNESS_CM, COLOURS):
        h = h_cm / 100
        t_tartes = tartes_transmission(wl, ssa, rho, h)
        t_snicar = pysnicar.transmission(wl, ssa, rho, h, refrac_index="w2008", shape=1, version="v4", dir_frac=1.0,
                                         sza=SZA_DEG)
        ax.semilogy(wl * 1e9, t_tartes, color=colour, lw=1.6, label=f"TARTES (default shape), {h_cm} cm")
        ax.semilogy(wl * 1e9, t_snicar, color="#D95319", lw=1.6, ls="--", label=f"PySNICAR-ADv4 (spheres), {h_cm} cm")
        ok = np.minimum(t_tartes, t_snicar) > FLOOR
        axd.semilogy(wl[ok] * 1e9, t_snicar[ok] / t_tartes[ok], color="#D95319", lw=1.4, ls="--")
        for lam in (500, 900, 1030, 1300):
            i = np.argmin(abs(wl * 1e9 - lam))
            rows.append(f"| {rho:g} / {ssa:g} | {h_cm} | {lam} | {t_tartes[i]:.4f} | {t_snicar[i]:.4f} | "
                        f"{100 * (t_snicar[i] / t_tartes[i] - 1):+.1f} |" if t_tartes[i] > FLOOR else
                        f"| {rho:g} / {ssa:g} | {h_cm} | {lam} | {t_tartes[i]:.1e} | {t_snicar[i]:.1e} | – |")
    ax.set(ylim=(1e-5, 1.2), ylabel="transmission",
           title=f"{rho:g} kg m$^{{-3}}$, SSA {ssa:g} m$^2$ kg$^{{-1}}$, direct beam {SZA_DEG:g}°, black ground")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=9)
    axd.axhline(1, color="0.4", lw=0.8)
    axd.set(xlabel="wavelength (nm)", ylabel="PySNICAR / TARTES", xlim=(400, 1500))
    axd.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(FIG / f"transmission{SUFFIX}.png", dpi=160)
print("| case | slab (cm) | λ (nm) | TARTES | PySNICAR | diff (%) |\n|---|---|---|---|---|---|")
print("\n".join(rows))

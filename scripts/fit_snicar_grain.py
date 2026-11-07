"""Fit the PySNICAR sphere size to the TARTES albedo, then compare the 10 cm slab transmission at the fitted size.

TARTES (default shape) stays at the case SSA. PySNICAR-ADv4 (spheres) is run over a sweep of radii (its optics library
has 1 um steps) for semi-infinite snow, direct beam at SNOWRT_SZA (default nadir); the best radius minimises the rms
albedo difference to TARTES for wavelengths below FIT_MAX_NM. With that radius, the 10 cm slab transmission over a black
ground is compared as in compare_transmission.py.
    python scripts/fit_snicar_grain.py
Writes results/snicar_grain_fit{SUFFIX}.json and figures/transmission_fitted{SUFFIX}.png.
"""
import json
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
from cases import CASES, RESULTS, RHO_ICE, SZA_DEG, case_name, wavelengths_m  # noqa: E402
from pysnicar.api import SEMI_INFINITE  # noqa: E402

FIG = ROOT / "figures"
SUFFIX = "" if SZA_DEG == 0 else f"_sza{SZA_DEG:g}"
FIT_MAX_NM = 1200
SLAB_M = 0.10
RADII_UM = np.arange(60, 301)            # sweep, about SSA 11-55 m2 kg-1
FLOOR = 1e-5                             # ratio only where both are above this


def tartes_transmission(wl, ssa, rho, h):   # as compare_transmission.py
    down, up = tartes.irradiance_profiles(wl, [0.0, h], ssa, rho, thickness=h, refrac_index="w2008", soilalbedo=0.0,
                                          dir_frac=1.0, sza=SZA_DEG)
    return down[:, 1] / down[:, 0]


def ssa_from_radius(r_um):
    return 3.0 / (RHO_ICE * r_um * 1e-6)


def snicar_albedo_sweep(wl, rho):
    cases = [dict(layers=[pysnicar.Layer(SEMI_INFINITE, rho, float(r))], solar_zenith=SZA_DEG, direct=True,
                  ground_albedo=0.0, ice_refractive_index="w2008") for r in RADII_UM]
    res = pysnicar.run_many(cases, version="v4")
    return np.array([np.interp(wl, r.wavelength, r.albedo) for r in res])


wl = wavelengths_m()
nm = wl * 1e9
fit = nm < FIT_MAX_NM
fig, axs = plt.subplots(3, len(CASES), figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.6, 1]})
summary = {}
for col, (rho, ssa) in enumerate(CASES):
    name = case_name(rho, ssa)
    a_tartes = tartes.albedo(wl, ssa, rho, refrac_index="w2008", dir_frac=1.0, sza=SZA_DEG)
    sweep = snicar_albedo_sweep(wl, rho)
    rms = np.sqrt(((sweep[:, fit] - a_tartes[fit]) ** 2).mean(axis=1))
    best = int(np.argmin(rms))
    r_fit, r_case = int(RADII_UM[best]), 3.0 / (RHO_ICE * ssa) * 1e6
    ssa_fit = ssa_from_radius(r_fit)
    i_case = int(np.argmin(abs(RADII_UM - r_case)))
    summary[name] = dict(ssa_case=ssa, radius_case_um=round(r_case, 1), radius_fit_um=r_fit, ssa_fit=round(ssa_fit, 2),
                         rms_fit=float(rms[best]), rms_case_radius=float(rms[i_case]),
                         rms_fit_full_range=float(np.sqrt(((sweep[best] - a_tartes) ** 2).mean())))
    print(name, summary[name], flush=True)

    t_tartes = tartes_transmission(wl, ssa, rho, SLAB_M)
    kw = dict(refrac_index="w2008", shape=1, version="v4", dir_frac=1.0, sza=SZA_DEG)
    t_case = pysnicar.transmission(wl, ssa, rho, SLAB_M, **kw)
    t_fit = pysnicar.transmission(wl, ssa_fit, rho, SLAB_M, **kw)
    summary[name]["transmission_10cm"] = {
        str(lam): dict(tartes=float(t_tartes[i]), snicar_case=float(t_case[i]), snicar_fit=float(t_fit[i]))
        for lam in (500, 700, 900, 1030) for i in [int(np.argmin(abs(nm - lam)))]}

    ax, axt, axr = axs[:, col]
    ax.axvspan(nm[0], FIT_MAX_NM, color="0.92", lw=0)
    ax.plot(nm, a_tartes, color="#0072BD", lw=1.6, label=f"TARTES (default shape), SSA {ssa:g}")
    ax.plot(nm, sweep[i_case], color="#D95319", lw=1.2, ls="--", alpha=0.6,
            label=f"PySNICAR, SSA {ssa:g} (r = {RADII_UM[i_case]} µm)")
    ax.plot(nm, sweep[best], color="#D95319", lw=1.6, label=f"PySNICAR, fitted SSA {ssa_fit:.1f} (r = {r_fit} µm)")
    ax.set(ylabel="albedo (semi-infinite)", ylim=(0, 1.02),
           title=f"{rho:g} kg m$^{{-3}}$, SSA {ssa:g} m$^2$ kg$^{{-1}}$, direct beam {SZA_DEG:g}°")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.text(FIT_MAX_NM - 20, 0.05, f"fit range\nrms {rms[best]:.4f}", ha="right", fontsize=8.5, color="0.35")
    axt.semilogy(nm, t_tartes, color="#0072BD", lw=1.6, label="TARTES")
    axt.semilogy(nm, t_case, color="#D95319", lw=1.2, ls="--", alpha=0.6, label="PySNICAR, case SSA")
    axt.semilogy(nm, t_fit, color="#D95319", lw=1.6, label="PySNICAR, fitted SSA")
    axt.set(ylim=(1e-5, 1.2), ylabel=f"transmission, {SLAB_M * 100:g} cm slab")
    axt.legend(frameon=False, fontsize=8.5)
    for t, style in ((t_case, dict(ls="--", alpha=0.6, lw=1.2)), (t_fit, dict(lw=1.6))):
        ok = np.minimum(t_tartes, t) > FLOOR
        axr.plot(nm[ok], t[ok] / t_tartes[ok], color="#D95319", **style)
    axr.axhline(1, color="0.4", lw=0.8)
    axr.set(xlabel="wavelength (nm)", ylabel="PySNICAR / TARTES", xlim=(400, 1500))
    for a in (ax, axt, axr):
        a.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(FIG / f"transmission_fitted{SUFFIX}.png", dpi=160)
(RESULTS / f"snicar_grain_fit{SUFFIX}.json").write_text(json.dumps(summary, indent=1))

"""Spectral albedo of the analytic / two-stream-type models for the comparison cases.

TARTES (default shape), PySNICAR-ADv4 (spheres), Malinka random-shape (Eq. 36) and Markov (Eq. 51) models,
and the DISORT Mie model (polydisperse Mie spheres, malinka.mie). Same ice index everywhere (Warren & Brandt 2008),
semi-infinite snowpack, direct beam at SZA_DEG. Result: results/models.npz.
    python scripts/run_models.py
    python scripts/run_models.py --only=malinka_markov     (recompute some models, update the existing models.npz)

Malinka Markov: the plane albedo is evaluated by reciprocity (isotropic illumination, radiance leaving at mu0;
malinka.snow.swapped_albedo). The direct-beam solution of the Markov system is not reciprocal and exceeds the
energy limit (0.999 instead of 0.990 at near-zero absorption, > 1 at 400 nm); the swapped value agrees with the
random-shape model there to 3e-4. At exactly mu0 = 1 it is extrapolated from the last quadrature node (~1e-3).

Speed: one thread per worker (BLAS threads x processes made the first version ~100 x slower), consecutive wavelengths
per worker (Malinka's interface operators are cached per rounded refractive index, ~15 s each), and a Mie sum with
16 radii instead of 40 and 17 Legendre moments (the DISORT run uses 17): differs from the full sum by < 1e-3 in albedo.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_var] = "1"          # before numpy is imported

import sys  # noqa: E402
from functools import partial  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pysnicar  # noqa: E402
import tartes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "rt-malinka"))

from bench import record, timed  # noqa: E402
from cases import CASES, RESULTS, RHO_ICE, SZA_DEG, case_name, wavelengths_m  # noqa: E402
from malinka import mie  # noqa: E402
from malinka.snow import plane_albedo as malinka_albedo  # noqa: E402
from malinka.snow import swapped_albedo  # noqa: E402

MU0 = np.cos(np.radians(SZA_DEG))
MIE_RADII, MIE_LMAX, MIE_ANGLES = 16, 16, (60, 120)


def r_eff(ssa):
    """Effective radius of the sphere distribution with this SSA: 3 / (rho_ice SSA) (= 3/4 of the mean chord)."""
    return 3.0 / (RHO_ICE * ssa)


def mie_disort_albedo(wl, ssa):
    """One wavelength: polydisperse Mie spheres (gamma distribution, r_eff) + DISORT, semi-infinite, beam at MU0."""
    m = mie.complex_index(wl)
    chi = mie.poly_mie(wl, m, r_eff(ssa), nr=MIE_RADII, theta=mie.angles(*MIE_ANGLES), lmax=MIE_LMAX)["chi"].copy()
    omega, g = mie.omega_g(wl, m, r_eff(ssa), nr=60)
    chi[1] = g
    return mie.plane_albedo(omega, chi, MU0)


def chunk_job(wl, only=None):
    """Malinka (random, Markov) and Mie + DISORT for consecutive wavelengths, all cases.
    Returns the albedos and the CPU-seconds spent per model (one process = one thread)."""
    warnings.simplefilter("ignore")
    out, cpu = {}, {}
    for rho, ssa in CASES:
        name = case_name(rho, ssa)
        jobs = {"malinka_random": lambda: malinka_albedo("random", wl, ssa, rho, MU0),
                "malinka_markov": lambda: swapped_albedo("markov", wl, ssa, rho, MU0),
                "mie_disort": lambda: np.array([mie_disort_albedo(w, ssa) for w in wl])}
        for key, job in jobs.items():
            if only and key not in only:
                continue
            t0 = time.perf_counter()
            out[f"{name}/{key}"] = job()
            cpu[f"{name}/{key}"] = time.perf_counter() - t0
    return out, cpu


if __name__ == "__main__":
    wl = wavelengths_m()
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--only=")), None)
    out = {"wavelength_m": wl}
    if only:
        out = dict(np.load(RESULTS / "models.npz"))
        assert np.allclose(out["wavelength_m"], wl)
    for rho, ssa in ([] if only else CASES):
        name = case_name(rho, ssa)
        with timed("tartes", case=name, wavelengths=len(wl)):
            out[f"{name}/tartes"] = tartes.albedo(wl, ssa, rho, refrac_index="w2008", dir_frac=1.0, sza=SZA_DEG)
        with timed("pysnicar", case=name, wavelengths=len(wl)):
            out[f"{name}/pysnicar"] = pysnicar.albedo(wl, ssa, rho, refrac_index="w2008", shape=1, version="v4",
                                                      dir_frac=1.0, sza=SZA_DEG)
    print("tartes, pysnicar done" if not only else f"recomputing {only}", flush=True)
    chunks = np.array_split(np.arange(len(wl)), min(os.cpu_count() or 1, 20))
    cpu_total, done = {}, set()
    t_pool = time.perf_counter()
    with Pool(len(chunks)) as pool:
        for part, cpu in pool.imap(partial(chunk_job, only=only), [wl[c] for c in chunks]):
            for key, value in part.items():
                out[key] = np.concatenate([out[key], value]) if key in done else value
                done.add(key)
            for key, value in cpu.items():
                cpu_total[key] = cpu_total.get(key, 0.0) + value
            print("chunk done", flush=True)
    if not only:
        record("pool_wall", time.perf_counter() - t_pool, workers=len(chunks), wavelengths=len(wl))
    for key, value in ({} if only else cpu_total).items():   # CPU-s summed over workers (cold caches in every worker)
        record("cpu_" + key.split("/")[1], value, case=key.split("/")[0], wavelengths=len(wl), workers=len(chunks))
    RESULTS.mkdir(exist_ok=True)
    np.savez(RESULTS / "models.npz", **out)
    for rho, ssa in CASES:
        j = int(np.argmin(abs(wl - 1.03e-6)))
        print(case_name(rho, ssa), {k.split("/")[1]: round(float(v[j]), 4) for k, v in out.items()
                                    if k.startswith(case_name(rho, ssa))})

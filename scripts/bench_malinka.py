"""Cost of the Malinka models without running the whole spectrum: cold vs warm interface-operator cache.

The operators (kernels.interface_operators) are cached per rounded refractive index, so the cost is
  (number of distinct rounded n) x (cold build) + (number of wavelengths) x (warm solve).
This times both on a few wavelengths in a fresh process and extrapolates to the 106-wavelength grid.
    python scripts/bench_malinka.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_var] = "1"

import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "rt-malinka"))

from bench import record  # noqa: E402
from cases import CASES, SZA_DEG, wavelengths_m  # noqa: E402
from malinka.kernels import interface_operators  # noqa: E402
from malinka.snow import _n_round, ice_index  # noqa: E402
from malinka.snow import plane_albedo as malinka_albedo  # noqa: E402

warnings.simplefilter("ignore")
MU0 = np.cos(np.radians(SZA_DEG))
wl_all = wavelengths_m()
n_all = np.array([_n_round(n) for n in ice_index(wl_all)[0]])
n_unique = len(np.unique(n_all))
print("distinct rounded n over the grid:", n_unique, "of", len(wl_all), "wavelengths")
rho, ssa = CASES[0]
# three wavelengths with distinct rounded n (cold), then the same again (warm)
pick, seen = [], set()
for i, n in enumerate(n_all):
    if n not in seen and len(pick) < 3:
        pick.append(i)
        seen.add(n)
wl = wl_all[pick]
for kind in ("random", "markov"):
    interface_operators.cache_clear()           # both kinds share the operators: time each from a cold cache
    cold = []
    for w in wl:
        t0 = time.perf_counter()
        malinka_albedo(kind, np.array([w]), ssa, rho, MU0)
        cold.append(time.perf_counter() - t0)
    warm = []
    for w in wl:
        t0 = time.perf_counter()
        malinka_albedo(kind, np.array([w]), ssa, rho, MU0)
        warm.append(time.perf_counter() - t0)
    cold_s, warm_s = float(np.mean(cold)), float(np.mean(warm))
    est = n_unique * (cold_s - warm_s) + len(wl_all) * warm_s
    print(f"{kind}: cold {cold_s:.1f} s, warm {warm_s:.2f} s per wavelength -> one case, 106 wavelengths ~ {est:.0f} s "
          f"single thread", flush=True)
    record(f"malinka_{kind}_estimate", est, cold_per_wavelength_s=round(cold_s, 2), warm_per_wavelength_s=round(warm_s, 3),
           distinct_n=n_unique, wavelengths=len(wl_all), note="extrapolated from 3 wavelengths, one thread")

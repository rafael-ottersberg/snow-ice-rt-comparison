"""The comparison cases and the grids shared by all models."""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("SNOWRT_RESULTS", ROOT / "results"))   # override to keep benchmark runs apart
SAMPLE = ROOT / "data" / "grs_sample"
RI_DIR = ROOT / "data" / "refractive_indices"

RHO_ICE = 917.0
CASES = [(120.0, 20.0), (300.0, 15.0)]          # density kg m-3, SSA m2 kg-1
WAVELENGTH_NM = (400, 2500, 20)                  # first, last (inclusive), step
SZA_DEG = float(os.environ.get("SNOWRT_SZA", 0.0))   # direct-beam incidence angle (0 = normal incidence, plane albedo)


def rays(default):
    """Rays per wavelength for the photontracer runs (SNOWRT_RAYS overrides)."""
    return int(float(os.environ.get("SNOWRT_RAYS", default)))


def wavelengths_m():
    return np.arange(WAVELENGTH_NM[0], WAVELENGTH_NM[1] + 1, WAVELENGTH_NM[2]) * 1e-9


def case_name(rho, ssa):
    return f"rho{rho:g}_ssa{ssa:g}"

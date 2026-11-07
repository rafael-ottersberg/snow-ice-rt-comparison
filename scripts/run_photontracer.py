"""Trace the GRS sample (10 000 particles, sigma_r = 0.1) for the comparison cases with photontracer.

Needs the packings made by scripts/pack_densities.py. The grain size is set by scaling the geometry:
SSA = 6 / (rho_ice d_opt) with d_opt = scale * (specific-surface diameter of the sample, sample.json).
Slab: 600 packed boxes deep (20 x 30 instances, as icegrain). Same tracing function and albedo definition as the GRF
runs (grf_trace.trace_geometry). Ice index: Warren & Brandt (2008) table.
Results: results/photontracer_grs/<case>.npz.
    python scripts/run_photontracer.py            (add --tiny for a quick smoke test)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import record
from cases import CASES, RESULTS, RHO_ICE, RI_DIR, SZA_DEG, rays, SAMPLE, WAVELENGTH_NM, case_name
from grf_trace import trace_geometry
from icegrain import meshes
from icegrain.raytracing import build_slab, load_packed_box

TINY = "--tiny" in sys.argv
RAYS = 20_000 if TINY else rays(1_000_000)
WAVELENGTHS = np.arange(400, 2501, 200 if TINY else WAVELENGTH_NM[2])
BEAM_RADIUS_UM = 2000.0            # before scaling, as in icegrain's reference setup
DEPTH_BOXES = (20, 30)

if __name__ == "__main__":
    n_types = len(json.loads((SAMPLE / "sample.json").read_text())["diameters_um"])
    d_opt_sample_um = json.loads((SAMPLE / "sample.json").read_text())["specific_surface_diameter_um"]
    particles = [meshes.read_obj(SAMPLE / "particles" / f"scaled_particle_{i}.obj") for i in range(1, n_types + 1)]
    out_dir = RESULTS / ("photontracer_grs_tiny" if TINY else "photontracer_grs")
    out_dir.mkdir(parents=True, exist_ok=True)
    for rho, ssa in CASES:
        vf = float(f"{rho / RHO_ICE:g}")
        scale = (6.0 / (RHO_ICE * ssa) * 1e6) / d_opt_sample_um
        packed = load_packed_box(SAMPLE / "packing" / f"packed_{vf:g}.gsd")
        t0 = time.time()
        geometry = build_slab(particles, packed, DEPTH_BOXES, scale)
        t_build = time.time() - t0
        res = trace_geometry(geometry, WAVELENGTHS, RI_DIR, rays=RAYS, beam_radius_um=BEAM_RADIUS_UM * scale,
                             incidence_deg=SZA_DEG, seed=1)
        if not TINY:
            info = dict(case=case_name(rho, ssa), sza_deg=SZA_DEG, rays=RAYS, wavelengths=len(WAVELENGTHS))
            record("grs_build_slab", t_build, **info)
            record("grs_trace_setup", res["setup_s"], **info)
            record("grs_trace_runs", res["run_s"].sum(), first_run_s=round(res["run_s"][0], 2),
                   median_run_s=round(float(np.median(res["run_s"])), 2), **info)
        np.savez(out_dir / f"{case_name(rho, ssa)}.npz", scale=scale, volume_fraction=vf, box_um=packed.box * scale, **res)
        print(case_name(rho, ssa), f"scale {scale * 100:.0f} %, {time.time() - t0:.0f} s, albedo at 1030/1300/1800 nm:",
              np.round([res["albedo"][np.argmin(abs(WAVELENGTHS - w))] for w in (1000, 1300, 1800)], 3),
              "free fraction max", np.round((res["free"] / RAYS).max(), 4), flush=True)

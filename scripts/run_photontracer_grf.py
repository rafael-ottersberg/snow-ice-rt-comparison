"""Trace the GRF microstructure (grr-microstructures, periodic level-cut Gaussian random field) for the comparison cases.

Cell size N and the number of seeds come from the convergence study (grf_convergence.py, plot_grf_convergence.py).
Slab: two instancing levels, 600 cells deep. Voxel size set per realisation so that the mesh has the target SSA.
Results: results/photontracer_grf/<case>_seed<k>.npz (existing files are skipped).
    python scripts/run_photontracer_grf.py            (add --tiny for a quick smoke test; --reverse goes through
    the realisations backwards, so that a second process can share the work)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import grf_slab  # noqa: E402
from bench import record  # noqa: E402
from cases import CASES, RESULTS, RHO_ICE, RI_DIR, SZA_DEG, rays, WAVELENGTH_NM, case_name  # noqa: E402
from grf_trace import instanced_slab, trace_geometry  # noqa: E402

TINY = "--tiny" in sys.argv
REVERSE = "--reverse" in sys.argv
N_CELL = 64 if TINY else 256            # cell edge in voxels (feature size 30 voxels)
SEEDS = (1,) if TINY else tuple(int(x) for x in os.environ.get("SNOWRT_SEEDS", "1,2,3").split(","))
RAYS = 20_000 if TINY else rays(1_000_000)
WAVELENGTHS = np.arange(400, 2501, 200 if TINY else WAVELENGTH_NM[2])
BEAM_RADIUS_UM = 1000.0
SLAB = dict(depth1=20, lateral1=21, depth2=30, lateral2=31)

if __name__ == "__main__":
    out_dir = RESULTS / ("photontracer_grf_tiny" if TINY else "photontracer_grf")
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(rho, ssa, seed) for rho, ssa in CASES for seed in SEEDS]
    for rho, ssa, seed in (jobs[::-1] if REVERSE else jobs):
        phi = float(f"{rho / RHO_ICE:g}")
        path = out_dir / f"{case_name(rho, ssa)}_seed{seed}.npz"
        if path.exists():
            continue
        t0 = time.time()
        field = grf_slab.make_field(N_CELL, seed)
        t1 = time.time()
        voxel = float(grf_slab.voxel_size_um(field, phi, ssa))
        cell, top = grf_slab.periodic_cell_mesh(field, phi, voxel), grf_slab.top_tile_mesh(field, phi, voxel)
        t2 = time.time()
        geometry = instanced_slab(cell, top, N_CELL * voxel, grf_slab.surface_height_um(N_CELL, voxel), **SLAB)
        t3 = time.time()
        res = trace_geometry(geometry, WAVELENGTHS, RI_DIR, rays=RAYS, beam_radius_um=BEAM_RADIUS_UM,
                             incidence_deg=SZA_DEG, seed=seed)
        if not TINY:
            info = dict(case=case_name(rho, ssa), sza_deg=SZA_DEG, seed=seed, rays=RAYS, wavelengths=len(WAVELENGTHS), n_cell=N_CELL)
            record("grf_field", t1 - t0, **info)
            record("grf_meshes", t2 - t1, **info)          # voxel size (marching cubes for the SSA) + cell + top tile
            record("grf_instanced_slab", t3 - t2, **info)
            record("grf_trace_setup", res["setup_s"], **info)
            record("grf_trace_runs", res["run_s"].sum(), first_run_s=round(res["run_s"][0], 2),
                   median_run_s=round(float(np.median(res["run_s"])), 2), **info)
        np.savez(path, voxel_um=voxel, cell_mm=N_CELL * voxel / 1e3, n_cell=N_CELL, seed=seed, **res)
        print(path.name, f"{time.time() - t0:.0f} s", flush=True)

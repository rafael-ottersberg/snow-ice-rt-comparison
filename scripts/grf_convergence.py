"""How big must the GRF cell and the slab be? Studies, each run for both cases (density, SSA):

  cell    : unit cell N varied (3 seeds), slab of 600 cells depth (20 x 30 instances, as icegrain)
            -> statistics of the cell, periodic-channel artefact (free rays), scatter over seeds
  depth   : fixed cell (N = 128), slab depth and width varied -> leakage through bottom / sides
  feature : fixed physical cell, feature size FEATURE_VOX varied (resolution of the structure in voxels)

    python scripts/grf_convergence.py cell|depth|feature
The slab is built from two instancing levels (grf_trace.instanced_slab). Results:
results/grf_convergence/<study>/<case>_<setting>.json (existing files are skipped); plot: plot_grf_convergence.py.
The voxel size of every realisation is set so that its mesh has exactly the target SSA.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import grf_slab  # noqa: E402
from cases import CASES, RESULTS, RHO_ICE, RI_DIR, case_name  # noqa: E402
from grf_trace import instanced_slab, trace_geometry  # noqa: E402

WAVELENGTHS_NM = [600, 800, 900, 1030, 1300, 1500, 1800, 2200]
RAYS = 300_000
BEAM_RADIUS_UM = 1000.0
SEEDS = (1, 2, 3)
SLAB = dict(d1=20, d2=30, l1=21, l2=31)            # depth / lateral instances: 600 cells deep, 651 wide

STUDIES = {   # name: list of dicts (n = cell edge in voxels, feature size in voxels, seed, slab)
    "cell": [dict(n=n, feature=30.0, seed=s, **SLAB) for n in (64, 96, 128, 192, 256, 384) for s in SEEDS],
    "depth": [dict(n=128, feature=30.0, seed=1, d1=d1, d2=d2, l1=l1, l2=l2) for d1, d2, l1, l2 in
              [(5, 2, 21, 31), (10, 3, 21, 31), (20, 5, 21, 31), (20, 15, 21, 31), (20, 30, 21, 31),
               (20, 30, 5, 3), (20, 30, 11, 5)]],
    # same physical cell: feature size f vox -> cell edge 128 f / 30 vox
    "feature": [dict(n=int(round(128 * f / 30)), feature=f, seed=1, **SLAB) for f in (20.0, 30.0, 45.0)],
}


def run(rho, ssa, setting):
    phi = float(f"{rho / RHO_ICE:g}")
    t0 = time.time()
    field = grf_slab.make_field(setting["n"], setting["seed"], setting["feature"])
    voxel = float(grf_slab.voxel_size_um(field, phi, ssa))
    cell = grf_slab.periodic_cell_mesh(field, phi, voxel)
    top = grf_slab.top_tile_mesh(field, phi, voxel, setting["feature"])
    edge = setting["n"] * voxel
    geometry = instanced_slab(cell, top, edge, grf_slab.surface_height_um(setting["n"], voxel), depth1=setting["d1"],
                              lateral1=setting["l1"], depth2=setting["d2"], lateral2=setting["l2"])
    t1 = time.time()
    res = trace_geometry(geometry, WAVELENGTHS_NM, RI_DIR, rays=RAYS, beam_radius_um=BEAM_RADIUS_UM, seed=1)
    return dict(setting=setting, rho=rho, ssa=ssa, voxel_um=voxel, cell_mm=edge / 1e3,
                depth_mm=setting["d1"] * setting["d2"] * edge / 1e3, width_mm=setting["l1"] * setting["l2"] * edge / 1e3,
                n_triangles=int(len(cell[1]) + len(top[1])), free_column_fraction=grf_slab.free_column_fraction(field, phi),
                seconds_mesh=t1 - t0, seconds_trace=time.time() - t1, **{k: v.tolist() for k, v in res.items()})


if __name__ == "__main__":
    study = sys.argv[1]
    out_dir = RESULTS / "grf_convergence" / study
    out_dir.mkdir(parents=True, exist_ok=True)
    for rho, ssa in CASES:
        for setting in STUDIES[study]:
            tag = "_".join(f"{k}{v:g}" for k, v in setting.items())
            path = out_dir / f"{case_name(rho, ssa)}_{tag}.json"
            if path.exists():
                continue
            result = run(rho, ssa, setting)
            path.write_text(json.dumps(result))
            print(path.name, f"{result['n_triangles'] / 1e6:.1f}M tris, free cols {result['free_column_fraction']:.3f}, "
                  f"mesh {result['seconds_mesh']:.0f}s trace {result['seconds_trace']:.0f}s, albedo",
                  np.round(result["albedo"], 3), flush=True)

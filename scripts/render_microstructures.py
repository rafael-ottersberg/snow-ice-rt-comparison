"""Render the GRS and GRF microstructures of the comparison cases side by side, same physical cube.

GRS: the packed particles (sigma_r = 0.1) whose centres lie in a cube of edge L, at the scale used for the traced slab.
GRF: the level-cut Gaussian random field (seed 1, N = 256 cell) as at the top of the traced slab: the top of the cube is
the graded free surface of the simulation (top_tile_mesh / grf_firn.graded_top: the threshold rises over 20 voxels, so
every grain ends in a smooth level-set cap); the sides and the bottom are cuts, their ice cross-sections drawn darker.
A second GRF view shows the same cube in oblique (cabinet-like) projection: the front face (-y) undistorted, the depth
drawn receding at 45 deg up and to the right, all edges parallel (the geometry is sheared, then viewed orthographically).
Particle meshes are decimated for the rendering only. Needs pyvista (off-screen).
    python scripts/render_microstructures.py
Writes figures/microstructure_<case>.png and figures/microstructure_grf_<case>_front.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from skimage import measure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import grf_slab  # noqa: E402
from cases import CASES, RHO_ICE, SAMPLE, case_name  # noqa: E402
from icegrain import meshes  # noqa: E402
from icegrain.raytracing import load_packed_box  # noqa: E402

FIG = ROOT / "figures"
EDGE_UM = 4800.0            # rendered cube edge (the GRF cell is 4.98 mm at 300 kg m-3)
N_CELL, SEED = 256, 1        # as run_photontracer_grf.py
PARTICLE_TRIANGLES = 1500    # per particle type after decimation
ICE = "#cfe3f2"
CAP = "#6f9bc4"
CAMERA = [(2.2 * EDGE_UM, -1.85 * EDGE_UM, 1.6 * EDGE_UM), (0, 0, 0), (0, 0, 1)]
OBLIQUE = 0.4                  # depth scale of the oblique view (cabinet projection: 0.5)
SHEAR = np.array([[1, OBLIQUE / np.sqrt(2), 0, 0], [0, 1, 0, 0], [0, OBLIQUE / np.sqrt(2), 1, 0], [0, 0, 0, 1]])
FRONT_CAMERA = [(0, -3.3 * EDGE_UM, 0), (0, 0, 0), (0, 0, 1)]
pv.OFF_SCREEN = True


def polydata(verts, faces):
    return pv.PolyData(np.asarray(verts, float), np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64))


def grs_mesh(rho, ssa):
    sample = json.loads((SAMPLE / "sample.json").read_text())
    scale = (6.0 / (RHO_ICE * ssa) * 1e6) / sample["specific_surface_diameter_um"]
    packed = load_packed_box(SAMPLE / "packing" / f"packed_{float(f'{rho / RHO_ICE:g}'):g}.gsd")
    types = []
    for i in range(1, len(sample["diameters_um"]) + 1):
        v, f = meshes.read_obj(SAMPLE / "particles" / f"scaled_particle_{i}.obj")
        m = polydata(v, f)
        types.append(m.decimate(1.0 - PARTICLE_TRIANGLES / m.n_cells))
    centres = packed.transforms[:, :, 3] * scale
    inside = np.all(np.abs(centres) <= EDGE_UM / 2, axis=1)
    parts = []
    for t, tid in zip(packed.transforms[inside], packed.type_ids[inside]):
        m = types[tid].copy()
        m.points = (m.points @ t[:, :3].T + t[:, 3]) * scale
        parts.append(m)
    return pv.merge(parts), int(inside.sum())


def grf_mesh(rho, ssa):
    """Ice surface of the top block of the slab (graded top, open at the cut faces) and the caps of the cuts, in um."""
    phi = float(f"{rho / RHO_ICE:g}")
    field = grf_slab.make_field(N_CELL, SEED)
    voxel = float(grf_slab.voxel_size_um(field, phi, ssa))
    n = min(N_CELL, int(round(EDGE_UM / voxel)))
    z_surf, width = int(grf_slab.SURFACE_VOX * N_CELL), grf_slab.WIDTH_VOX       # as top_tile_mesh (feature 30 voxels)
    z = np.arange(z_surf + int(width) + 2 - n, z_surf + int(width) + 2)           # the cube ends just above the surface
    ramp = np.clip((z - (z_surf - width)) / width, 0.0, None)
    g = np.take(field[:n, :n], z, axis=2, mode="wrap") - grf_slab.threshold(field, phi) - 3.0 * ramp**2
    grid = pv.ImageData(dimensions=(n, n, n), spacing=(voxel,) * 3, origin=(-(n - 1) * voxel / 2,) * 3)
    grid.point_data["ice"] = g.ravel(order="F")
    surface = grid.contour([0.0], scalars="ice").decimate_pro(0.6, preserve_topology=True,
                                                               boundary_vertex_deletion=False)
    # caps: the parts of the cut faces inside the ice, cut along the same (linear) iso-line as the surface
    faces = grid.extract_surface().triangulate()
    caps = faces.clip_scalar(scalars="ice", value=0.0, invert=False).extract_surface()
    return surface, caps, voxel


if __name__ == "__main__":
    for rho, ssa in CASES:
        grs, count = grs_mesh(rho, ssa)
        grf, caps, voxel = grf_mesh(rho, ssa)
        p = pv.Plotter(shape=(1, 2), window_size=(1800, 900), off_screen=True, border=False)
        for col, (mesh, title) in enumerate(((grs, f"GRS, sigma_r = 0.1 ({count} particles)"),
                                             (grf, f"GRF, seed {SEED}, graded top (voxel {voxel:.1f} um)"))):
            p.subplot(0, col)
            p.set_background("white")
            p.add_mesh(mesh, color=ICE, smooth_shading=True, specular=0.3, ambient=0.15)
            if col == 1:
                p.add_mesh(caps, color=CAP, ambient=0.25)
            p.add_mesh(pv.Box(bounds=[-EDGE_UM / 2, EDGE_UM / 2] * 3).outline(), color="#4d4d4d", line_width=1.5)
            p.add_text(title, font_size=12, color="black")
            p.camera_position = CAMERA
        p.add_text(f"density {rho:g} kg/m3, SSA {ssa:g} m2/kg, cube edge {EDGE_UM / 1000:g} mm",
                   position="lower_left", font_size=11, color="black")
        p.screenshot(FIG / f"microstructure_{case_name(rho, ssa)}.png")
        p.close()
        # GRF alone, oblique view: the -y face to the viewer, depth (+y) sheared up and to the right
        p = pv.Plotter(window_size=(900, 900), off_screen=True, border=False)
        p.set_background("white")
        p.add_mesh(grf.transform(SHEAR, inplace=False), color=ICE, smooth_shading=True, specular=0.3, ambient=0.15)
        p.add_mesh(caps.transform(SHEAR, inplace=False), color=CAP, ambient=0.25)
        p.add_mesh(pv.Box(bounds=[-EDGE_UM / 2, EDGE_UM / 2] * 3).outline().transform(SHEAR, inplace=False),
                   color="#4d4d4d", line_width=1.5)
        p.add_text(f"GRF, seed {SEED}, density {rho:g} kg/m3, SSA {ssa:g} m2/kg, cube edge {EDGE_UM / 1000:g} mm",
                   font_size=10, color="black")
        p.camera_position = FRONT_CAMERA
        p.enable_parallel_projection()
        p.camera.parallel_scale = 0.56 * EDGE_UM * (1 + OBLIQUE / np.sqrt(2))
        p.screenshot(FIG / f"microstructure_grf_{case_name(rho, ssa)}_front.png")
        p.close()
        print(case_name(rho, ssa), f"GRS {count} particles, {grs.n_cells} triangles; GRF {grf.n_cells} + {caps.n_cells} cap triangles",
              flush=True)

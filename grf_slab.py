"""A GRF snow slab as ONE closed triangle mesh.

The periodic bulk tile (grr-microstructures, ``grf_firn.grf``) is repeated nx x ny x nz times, a free-surface tile
(``graded_top`` recipe of the same repository, written as a continuous field) is put on top, the volume is padded with
vacuum on all sides and the ice surface (level set of the field) is extracted with marching cubes. The mesh is closed
and outward-oriented; no instancing is needed. Lengths: voxels until ``voxel_um`` is applied.
"""
import sys
from pathlib import Path

import numpy as np
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parent / "external" / "grr-microstructures" / "code"))
from grf_firn import grf  # noqa: E402

RHO_ICE = 917.0
FEATURE_VOX = 30.0       # 2 pi / k0: feature size of the field in voxels (grr-microstructures default)
SURFACE_VOX = 0.7        # height of the surface in the surface tile, fraction of the tile edge (grr default)
WIDTH_VOX = 20.0         # width of the graded top in voxels (grr default)


def make_field(n, seed, feature_vox=FEATURE_VOX, anis=1.0):
    """Periodic unit-variance GRF on an n^3 grid (float32)."""
    k0 = 2 * np.pi / feature_vox
    return grf((n, n, n), k0, 0.35 * k0, anis=anis, seed=seed).astype(np.float32)


def threshold(field, phi):
    """Level of the field above which there is ice (exact volume fraction phi)."""
    return float(np.quantile(field, 1.0 - phi))


def cell_surface_per_volume(field, phi):
    """Ice surface per ice volume [1/voxel] of the periodic cell, from the marching-cubes mesh (one plane of the
    cell repeated at the end of every axis, so the periodic surface is exact)."""
    n = field.shape[0]
    padded = np.pad(field, ((0, 1),) * 3, mode="wrap")
    verts, faces, _, _ = measure.marching_cubes(padded, threshold(field, phi))
    return measure.mesh_surface_area(verts, faces) / (phi * n**3)


def voxel_size_um(field, phi, ssa):
    """Voxel edge [um] for which the cell has the specific surface area ``ssa`` [m2 kg-1]."""
    return cell_surface_per_volume(field, phi) / (RHO_ICE * ssa) * 1e6


def slab_volume(field, phi, nx, ny, nz):
    """Scalar volume (ice where > 0) of the repeated bulk plus the graded-top tile, closed by vacuum padding.
    Returns (volume, k_surface): index of the nominal surface along the last axis."""
    n = field.shape[0]
    t = threshold(field, phi)
    z_surf = int(SURFACE_VOX * n)
    z = np.arange(z_surf + int(WIDTH_VOX) + 2)
    ramp = np.clip((z - (z_surf - WIDTH_VOX)) / WIDTH_VOX, 0.0, None)
    top = field[:, :, :len(z)] - t - 3.0 * ramp**2                  # graded_top: ice = field > t + 3 ramp^2
    volume = np.concatenate([np.tile(field - t, (nx, ny, nz)), np.tile(top, (nx, ny, 1))], axis=2)
    return np.pad(volume, 1, constant_values=-1.0), nz * n + z_surf + 1


def mesh_volume(volume, chunk=128):
    """Marching cubes (level 0) in chunks along z with one shared plane; returns float32 vertices [voxel] and faces."""
    all_v, all_f, offset = [], [], 0
    for z0 in range(0, volume.shape[2] - 1, chunk):
        part = volume[:, :, z0:z0 + chunk + 1]
        if part.shape[2] < 2 or not (part > 0).any():
            continue
        v, f, _, _ = measure.marching_cubes(part, 0.0)
        v[:, 2] += z0
        all_v.append(v.astype(np.float32))
        all_f.append((f[:, ::-1] + offset).astype(np.uint32))     # skimage winds inward; flip to outward
        offset += len(v)
    return np.concatenate(all_v), np.concatenate(all_f)


def slab_mesh(field, phi, nx, ny, nz, voxel_um):
    """Closed slab mesh in um: x, y centred on 0, z = 0 at the nominal surface (ice up to ~WIDTH_VOX above)."""
    volume, k_surf = slab_volume(field, phi, nx, ny, nz)
    verts, faces = mesh_volume(volume)
    verts -= np.array([volume.shape[0] / 2, volume.shape[1] / 2, k_surf], dtype=np.float32)
    return verts * np.float32(voxel_um), faces


def signed_volume(verts, faces):
    """Enclosed volume of a closed mesh (positive for outward-oriented faces)."""
    a, b, c = (verts[faces[:, i]].astype(np.float64) for i in range(3))
    return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)


# ---- periodic tiles for instancing (open meshes whose faces match exactly when repeated)
def periodic_cell_mesh(field, phi, voxel_um):
    """Ice surface of the periodic cell [um], centred on 0, spanning one cell edge in x, y, z. Marching cubes on the
    cell plus one wrapped plane per axis, so the vertices on opposite faces coincide and copies join seamlessly.
    Open at the cell faces (ice continues into the neighbour); outward-oriented."""
    n = field.shape[0]
    verts, faces, _, _ = measure.marching_cubes(np.pad(field, ((0, 1),) * 3, mode="wrap"), threshold(field, phi))
    verts = (verts.astype(np.float32) - n / 2) * np.float32(voxel_um)
    return verts, np.ascontiguousarray(faces[:, ::-1]).astype(np.uint32)


def top_tile_mesh(field, phi, voxel_um, feature_vox=FEATURE_VOX):
    """Free-surface tile [um]: same field as the cell, graded top (see slab_volume; the grading width scales with the
    feature size), periodic in x, y and z (the field is wrapped where the tile is taller than the cell), closed at the
    top, open at the bottom where it joins the cell below. x, y centred on 0, z = 0 at its bottom plane; the nominal
    surface is at ``surface_height_um``."""
    n = field.shape[0]
    t = threshold(field, phi)
    width = WIDTH_VOX * feature_vox / FEATURE_VOX
    z_surf = int(SURFACE_VOX * n)
    z = np.arange(z_surf + int(width) + 2)
    ramp = np.clip((z - (z_surf - width)) / width, 0.0, None)
    g = np.take(field, z, axis=2, mode="wrap") - t - 3.0 * ramp**2
    g = np.pad(g, ((0, 1), (0, 1), (0, 0)), mode="wrap")
    g = np.pad(g, ((0, 0), (0, 0), (0, 1)), constant_values=-1.0)
    verts, faces, _, _ = measure.marching_cubes(g, 0.0)
    verts = verts.astype(np.float32) - np.array([n / 2, n / 2, 0.0], dtype=np.float32)
    return verts * np.float32(voxel_um), np.ascontiguousarray(faces[:, ::-1]).astype(np.uint32)


def surface_height_um(n, voxel_um):
    """Height of the nominal surface above the bottom plane of the top tile."""
    return int(SURFACE_VOX * n) * voxel_um


def free_column_fraction(field, phi):
    """Fraction of the vertical lines through the periodic cell that never touch ice. In the repeated cell such a line
    is an endless channel: rays at normal incidence can fly through the whole slab."""
    return float((field <= threshold(field, phi)).all(axis=2).mean())

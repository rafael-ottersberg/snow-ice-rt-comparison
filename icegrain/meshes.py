"""Triangle meshes as plain numpy arrays: OBJ files, volume, surface, scaling to a target size.

A mesh is a pair ``(vertices, faces)``: float array (n, 3) and integer array (m, 3), zero-based.
"""
from pathlib import Path

import numpy as np

from .sizedist import sphere_volume


def read_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                indices = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
                if len(indices) != 3:
                    raise ValueError(f"{path}: only triangular faces are supported")
                faces.append(indices)
    return np.array(vertices), np.array(faces, dtype=np.int64)


def write_obj(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("# OBJ\n")
        for v in vertices:
            f.write("v {:.10g} {:.10g} {:.10g}\n".format(*v))
        for tri in faces + 1:
            f.write("f {} {} {}\n".format(*tri))


def _triangle_corners(vertices: np.ndarray, faces: np.ndarray):
    return (vertices[faces[:, i]] for i in range(3))


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Volume by the divergence theorem (positive for outward-facing normals)."""
    a, b, c = _triangle_corners(vertices, faces)
    return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0)


def mesh_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    a, b, c = _triangle_corners(vertices, faces)
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def mesh_centroid(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Centre of volume."""
    a, b, c = _triangle_corners(vertices, faces)
    tetra_volume = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    tetra_centroid = (a + b + c) / 4.0                # fourth corner of each tetrahedron is the origin
    return (tetra_centroid * tetra_volume[:, None]).sum(axis=0) / tetra_volume.sum()


def scale_to_equivalent_diameter(vertices: np.ndarray, faces: np.ndarray, diameter_um: float) -> np.ndarray:
    """Centre the mesh on its centre of volume and scale it to the volume of a sphere of ``diameter_um``.

    (The old pipeline centred on the minimal bounding sphere; the origin only fixes the centre of the
    particle's rotations and has no physical effect.)
    """
    centred = vertices - mesh_centroid(vertices, faces)
    volume = mesh_volume(centred, faces)
    if volume <= 0:
        raise ValueError("mesh has non-positive volume (inverted normals?)")
    return centred * (float(sphere_volume(diameter_um)) / volume) ** (1.0 / 3.0)


def export_scaled_particles(mesh_paths: list[Path], diameters_um, out_dir: str | Path) -> list[Path]:
    """Write ``scaled_particle_<i>.obj`` (i from 1), each at the volume of its size bin."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, (path, diameter) in enumerate(zip(mesh_paths, diameters_um), start=1):
        vertices, faces = read_obj(path)
        target = out_dir / f"scaled_particle_{i}.obj"
        write_obj(target, scale_to_equivalent_diameter(vertices, faces, diameter), faces)
        written.append(target)
    return written


def specific_surface_diameter(meshes: list[tuple[np.ndarray, np.ndarray]], frequencies) -> float:
    """Diameter of the sphere with the same total volume / total surface as the sample,
    ``6 * sum(f_i V_i) / sum(f_i A_i)`` over particle types with number fractions ``f_i``.

    This is the "optical" grain size of the look-up table. For GRS particles it is smaller than the
    Sauter diameter of equal-volume spheres by their surface excess (about 4 % at rel_std 0.1), so it
    depends on the shape statistics of each sample.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    volume = sum(f * mesh_volume(v, t) for f, (v, t) in zip(frequencies, meshes))
    area = sum(f * mesh_surface_area(v, t) for f, (v, t) in zip(frequencies, meshes))
    return float(6.0 * volume / area)

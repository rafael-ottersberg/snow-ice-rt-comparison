"""Ray tracing with PhotonTracer: packed GSD -> tiled slab -> spectrum.

``run_trace`` computes one spectrum (all wavelengths of one ``TraceConfig``) on one device and
checkpoints a single result file after every wavelength; a killed run resumes where it stopped.
There is no scheduler code: to use several GPUs or a cluster, start one process per config.

Geometry (as in the old ``rt_grid_gsd_mesh.py``)::

    particle meshes -> packed box (instances of particles)
                    -> level 1: boxes tiled ``depth_boxes[0]`` deep (and depth+1 laterally)
                    -> level 2: level-1 blocks tiled ``depth_boxes[1]`` deep (and depth+1 laterally)
                    -> the whole slab scaled by ``scale_percent / 100``

The top surface of the slab is the plane z = 0 and the light comes from +z.
"""
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import meshes
from . import refractive_index as ri
from .angles import emission_angles_deg, incident_direction, phase_angles_deg
from .config import TraceConfig
from .provenance import software_versions
from .results import ResultWriter, SpectrumResult

log = logging.getLogger(__name__)

VACUUM, ICE = 0, 1            # material ids used in the scene


# -- geometry helpers (pure numpy, no GPU needed) -----------------------------

def quaternions_to_matrices(quaternions: np.ndarray) -> np.ndarray:
    """Rotation matrices (n, 3, 3) from quaternions (n, 4) in (w, x, y, z) order, as HOOMD/GSD store them."""
    q = np.asarray(quaternions, dtype=np.float64)
    w, x, y, z = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
    matrices = np.empty((len(q), 3, 3))
    matrices[:, 0] = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=1)
    matrices[:, 1] = np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=1)
    matrices[:, 2] = np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1)
    return matrices


@dataclass
class PackedBox:
    """One periodic box of particles, ready to be used as instances."""
    transforms: np.ndarray    # (n, 3, 4) float32, [rotation | position] of every particle
    type_ids: np.ndarray      # (n,) uint32, index of the particle mesh
    box: np.ndarray           # (3,) edge lengths in um


def load_packed_box(gsd_path: str | Path) -> PackedBox:
    import gsd.hoomd
    with gsd.hoomd.open(str(gsd_path), "r") as trajectory:
        frame = trajectory[len(trajectory) - 1]
    rotation = quaternions_to_matrices(frame.particles.orientation)
    transforms = np.concatenate([rotation, frame.particles.position[:, :, None]], axis=2)
    return PackedBox(transforms.astype(np.float32), frame.particles.typeid.astype(np.uint32),
                     np.array(frame.configuration.box[:3], dtype=float))


def tile_transforms(cell: np.ndarray, depth: int, top_face_at_zero: bool = False) -> np.ndarray:
    """Translations (m, 3, 4) that place copies of a cell of size ``cell`` on a regular grid.

    Laterally the indices run from ``-depth//2`` to ``depth//2`` (centred; ``depth + 1`` sites for even
    depth), downwards there are ``depth`` layers starting at z = 0. With ``top_face_at_zero`` the first
    layer's top face (not its centre) sits at z = 0.
    """
    lateral = np.arange(-depth // 2, depth // 2 + 1)
    ix, iy, iz = (a.ravel() for a in np.meshgrid(lateral, lateral, np.arange(depth), indexing="ij"))
    transforms = np.zeros((ix.size, 3, 4), dtype=np.float32)
    transforms[:, [0, 1, 2], [0, 1, 2]] = 1.0                       # identity rotation
    transforms[:, 0, 3] = ix * cell[0]
    transforms[:, 1, 3] = iy * cell[1]
    transforms[:, 2, 3] = -iz * cell[2] - (cell[2] / 2.0 if top_face_at_zero else 0.0)
    return transforms


def beam_origin(cfg: TraceConfig) -> np.ndarray:
    """Centre of the source disk: at height ``beam_distance_um`` above the surface, placed so that the
    central ray hits the surface at (0, 0, 0)."""
    direction = incident_direction(cfg.incidence_deg, cfg.azimuth_deg)
    return -direction * (cfg.beam_distance_um / np.cos(np.radians(cfg.incidence_deg)))


def wavelengths_nm(cfg: TraceConfig) -> np.ndarray:
    return ri.wavelength_grid_nm(*cfg.wavelength_nm)


# -- scene (needs photontracer) ----------------------------------------------

def build_slab(particle_meshes, packed_box: PackedBox, depth_boxes=(20, 30), scale: float = 1.0):
    """Nested instance geometry of the semi-infinite slab (see module docstring)."""
    from photontracer import InstanceGeometry, MeshGeometry

    def instances(sub_geometries, transforms, type_ids, material_ids=None):
        if material_ids is None:
            material_ids = np.zeros(len(transforms), np.uint32)
        return InstanceGeometry(sub_geometries, transforms, type_ids, material_ids)

    particle_geometries = [MeshGeometry(np.asarray(v, np.float32), np.asarray(f, np.uint32))
                           for v, f in particle_meshes]
    box = instances(particle_geometries, packed_box.transforms, packed_box.type_ids,
                    np.full(len(packed_box.type_ids), ICE, np.uint32))

    depth1, depth2 = depth_boxes
    level1_transforms = tile_transforms(packed_box.box, depth1, top_face_at_zero=True)
    level1 = instances([box], level1_transforms, np.zeros(len(level1_transforms), np.uint32))

    level2_transforms = tile_transforms(packed_box.box * depth1, depth2) * np.float32(scale)
    return instances([level1], level2_transforms, np.zeros(len(level2_transforms), np.uint32))


def make_simulation(cfg: TraceConfig, geometry, gpu_id: int = 0, cpu_threads: int = 0):
    from photontracer import Backend, LengthUnit, Simulation
    if cfg.backend == "embree":
        sim = Simulation(backend=Backend.EMBREE, cpu_threads=cpu_threads)
    else:
        sim = Simulation(gpu_id, backend=Backend.OPTIX)
    sim.length_unit = LengthUnit.MICRO_METER
    sim.max_scattering_count = int(cfg.max_scattering_count)
    sim.max_nested_geometry_levels = 4
    if cfg.seed is not None:
        sim.seed = int(cfg.seed)
    sim.geometry = geometry
    return sim


def beam_offset_um(cfg: TraceConfig, box_edges_um) -> np.ndarray:
    """Lateral shift (x, y, 0) of the beam centre in um: ``beam_offset_cell`` times the edge of the packed box,
    scaled with the geometry. The slab is periodic, so any shift within +-0.5 box edges is a valid other patch."""
    shift = np.asarray(cfg.beam_offset_cell, dtype=float) * np.asarray(box_edges_um[:2], dtype=float)
    return np.array([shift[0], shift[1], 0.0]) * (cfg.scale_percent / 100.0)


def set_beam(sim, cfg: TraceConfig, box_edges_um=(0.0, 0.0, 0.0)) -> None:
    """Parallel beam from a disk lying *parallel to the surface* (``disk_normal`` = +z).

    The illuminated patch is then a circle of the same radius at every incidence angle and no ray can
    start inside the slab (a disk perpendicular to a tilted beam reaches below z = 0 at large angles).
    Needs photontracer with ``ParallelRayGenerator(disk_normal=...)`` (public ``feat/parallel-disk-normal``).
    ``box_edges_um`` (the packed box) turns ``cfg.beam_offset_cell`` into a lateral shift of the beam centre.
    """
    from photontracer import OutputType, ParallelRayGenerator
    try:
        sim.ray_generator = ParallelRayGenerator(
            number_of_rays=int(cfg.number_of_rays),
            origin=beam_origin(cfg) + beam_offset_um(cfg, box_edges_um),
            direction=incident_direction(cfg.incidence_deg, cfg.azimuth_deg),
            offset_radius=cfg.beam_radius_um * cfg.scale_percent / 100.0,
            disk_normal=(0.0, 0.0, 1.0))
    except TypeError as error:
        raise RuntimeError("the installed photontracer has no ParallelRayGenerator(disk_normal=...); install the "
                           "public repo's feat/parallel-disk-normal branch (or a release that contains it)") from error
    sim.outputs = [OutputType.RAY_STATE, OutputType.LAST_DIRECTION]


# -- running ------------------------------------------------------------------

def _trace_one_wavelength(sim, cfg: TraceConfig, wavelength_nm: int, n: float, k: float, writer: ResultWriter) -> float:
    """Run the simulation for one wavelength, record the counts in ``writer`` and return the albedo."""
    from photontracer import Material, MaterialType, OutputType
    sim.materials = [Material(MaterialType.REFRACTIVE, 1 + 0j), Material(MaterialType.REFRACTIVE, complex(n, k))]
    sim.wavelength_um = wavelength_nm / 1000.0
    sim.run()

    state = sim.get_output_buffer(OutputType.RAY_STATE)           # 0 escaped, 1 absorbed, other: lost/error
    escaped = state == 0
    reflected, absorbed = int(escaped.sum()), int((state == 1).sum())
    directions = sim.get_output_buffer(OutputType.LAST_DIRECTION)[escaped]
    incident = incident_direction(cfg.incidence_deg, cfg.azimuth_deg)
    writer.add(
        wavelength_nm, reflected=reflected, absorbed=absorbed, unfinished=len(state) - reflected - absorbed,
        emission_histogram=np.histogram(emission_angles_deg(directions), bins=writer.emission_bins)[0],
        phase_histogram=np.histogram(phase_angles_deg(directions, incident), bins=writer.phase_bins)[0])
    writer.flush()
    return reflected / max(reflected + absorbed, 1)


def run_trace(cfg: TraceConfig, particle_files, gsd_path: str | Path, out_path: str | Path, ri_dir: str | Path,
              gpu_id: int = 0, cpu_threads: int = 0) -> SpectrumResult:
    """Compute (or finish) the spectrum of ``cfg`` and write it to ``out_path``.

    ``ri_dir`` is the folder with the refractive-index tables."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = wavelengths_nm(cfg)
    writer = ResultWriter(out_path, {
        "software": software_versions(), "config": cfg.to_dict(),
        "particle_files": [str(p) for p in particle_files], "gsd": str(gsd_path)})
    todo = [int(w) for w in grid if int(w) not in writer.done_nm]
    if not todo:
        log.info("%s is already complete", out_path)
        return writer.to_result()

    table = ri.load(cfg.ri_source, Path(ri_dir), cfg.species, int(cfg.temperature_k))
    index = table.resample(grid, cfg.ri_bandwidth_um)
    n_and_k = {int(w): (n, k) for w, n, k in zip(grid, index.n, index.k)}

    t0 = time.time()
    packed_box = load_packed_box(gsd_path)
    slab = build_slab([meshes.read_obj(p) for p in particle_files], packed_box, cfg.depth_boxes,
                      cfg.scale_percent / 100.0)
    sim = make_simulation(cfg, slab, gpu_id, cpu_threads)
    set_beam(sim, cfg, packed_box.box)
    log.info("scene set up in %.1f s, %d wavelengths to do", time.time() - t0, len(todo))

    for i, wavelength in enumerate(todo):
        t1 = time.time()
        albedo = _trace_one_wavelength(sim, cfg, wavelength, *n_and_k[wavelength], writer)
        if i % 25 == 0 or i == len(todo) - 1:
            log.info("%d/%d  %d nm  albedo %.4f  (%.2f s)", i + 1, len(todo), wavelength, albedo, time.time() - t1)
    return writer.to_result()


def trace_sample(sample_folder: str | Path, cfg: TraceConfig, ri_dir: str | Path, out_root: str | Path | None = None,
                 **kwargs) -> Path:
    """Trace ``cfg`` on a sample made by ``sample.build_sample``.

    Result: ``<sample>/rt/<config id>/spectrum.npz`` next to a ``config.json``. Returns the path of the spectrum.
    """
    sample_folder = Path(sample_folder)
    n_types = len(json.loads((sample_folder / "sample.json").read_text())["diameters_um"])
    particle_files = [sample_folder / "particles" / f"scaled_particle_{i}.obj" for i in range(1, n_types + 1)]
    gsd_path = sample_folder / "packing" / f"packed_{cfg.volume_fraction:g}.gsd"
    if not gsd_path.exists():
        available = sorted(p.name for p in (sample_folder / "packing").glob("packed_*.gsd"))
        raise FileNotFoundError(f"no packing with volume fraction {cfg.volume_fraction:g} in {sample_folder}; "
                                f"packed: {available}")

    out_dir = Path(out_root or sample_folder / "rt") / cfg.folder_name()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
    run_trace(cfg, particle_files, gsd_path, out_dir / "spectrum.npz", ri_dir, **kwargs)
    return out_dir / "spectrum.npz"

"""Hard-particle Monte Carlo packing of polydisperse (concave) particles with HOOMD-blue.

HOOMD (GPU build, conda-forge) and trimesh/vhacdx are imported lazily, so the rest of the
modules work without them. Differences to the old ``packing_helper.PackingSimulation``:

* everything is seeded (type draw, initial lattice shuffle, HOOMD seed);
* the compression step budget is per call (the old ``max_steps`` was compared to the *global*
  timestep, so later targets of a multi-target run could be silently cut short);
* no overlaps is asserted after every compression, targets that are not reached raise;
* no output is overwritten, files are written atomically.
"""
import itertools
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sizedist import SizeDistribution, sphere_volume

log = logging.getLogger(__name__)
PROGRESS_INTERVAL_S = 60       # seconds between progress lines during a long compression


# -- convex decomposition ----------------------------------------------------

def is_convex(vertices: np.ndarray, faces: np.ndarray) -> bool:
    import trimesh
    return bool(trimesh.Trimesh(vertices, faces, process=False).is_convex)


def decompose_convex(mesh: tuple[np.ndarray, np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    """Approximate convex decomposition (V-HACD). Returns the vertices of each convex part,
    centred on the part's centre of mass, and the (n_parts, 3) centre positions."""
    import trimesh
    vertices, faces = mesh
    parts = trimesh.Trimesh(vertices, faces, process=False).convex_decomposition()
    if not isinstance(parts, list):
        parts = [parts]
    centers = np.array([p.center_mass for p in parts])
    return [np.asarray(p.vertices) - c for p, c in zip(parts, centers)], centers


def decompose_all(meshes: list[tuple[np.ndarray, np.ndarray]], max_workers: int | None = None):
    """Decompose in parallel. Uses spawned worker processes, so a calling *script* must be guarded by
    ``if __name__ == "__main__":`` (as for any multiprocessing with spawn)."""
    # spawn, not fork: HOOMD/CUDA may already be running threads in this process
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as pool:
        return list(pool.map(decompose_convex, meshes))


# -- initial state -----------------------------------------------------------

def lattice_positions(n: int, spacing_um: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """``n`` random sites of a cubic lattice with the given spacing. Returns positions
    (centred on the origin) and the box edge lengths."""
    per_axis = int(np.ceil(n ** (1.0 / 3.0)))
    box = np.full(3, per_axis * spacing_um)
    axis = (np.arange(per_axis) - (per_axis - 1) / 2.0) * spacing_um
    sites = np.array(list(itertools.product(axis, axis, axis)))
    rng.shuffle(sites)
    return sites[:n], box


def bounding_radius(vertices: np.ndarray) -> float:
    return float(np.linalg.norm(vertices, axis=1).max())


def is_sphere(vertices: np.ndarray, tolerance: float = 1e-3) -> bool:
    """True if all vertices lie on one sphere around the origin (relative spread below ``tolerance``).
    The GRS meshes with sigma_r = 0 are such spheres; HOOMD cannot pack their 6400 vertices as a polyhedron
    ("Too many vertices in polyhedron"), so they are packed as exact spheres instead."""
    radii = np.linalg.norm(vertices, axis=1)
    return bool((radii.max() - radii.min()) / radii.mean() < tolerance)


# -- simulation --------------------------------------------------------------

def make_device(prefer_gpu: bool = True):
    import hoomd
    if prefer_gpu and getattr(hoomd.version, "gpu_enabled", False):
        try:
            return hoomd.device.GPU()
        except RuntimeError as e:   # no usable GPU
            log.warning("GPU unavailable (%s), falling back to CPU", e)
    return hoomd.device.CPU()


@dataclass
class CompressionResult:
    target_volume_fraction: float
    volume_fraction: float
    steps: int
    seconds: float
    acceptance: float


class Packer:
    """Polydisperse HPMC packing.

    ``meshes[i]`` is the (vertices, faces) of the particle of size bin ``i``, already scaled
    to its equal-volume diameter ``dist.diameters[i]`` (see ``meshes.export_scaled_particles``).

    Three ways to pack, chosen from the meshes: exact spheres (all meshes round), a convex polyhedron (convex
    meshes), or a union of convex parts from a convex decomposition (concave meshes, the GRS particles).
    """

    def __init__(self, meshes: list[tuple[np.ndarray, np.ndarray]], dist: SizeDistribution,
                 n_particles: int, seed: int = 42, use_convex_hull: bool = False,
                 initial_spacing: float = 5.0, max_workers: int | None = None):
        if len(meshes) != len(dist.diameters):
            raise ValueError(f"{len(meshes)} meshes for {len(dist.diameters)} size bins")
        self.meshes, self.dist = meshes, dist
        self.n_particles, self.seed = n_particles, seed
        self.use_convex_hull = use_convex_hull
        self.type_names = [f"t{i}" for i in range(len(meshes))]
        self.type_volumes = sphere_volume(dist.diameters)       # equal-volume spheres, by construction
        self.spheres = all(is_sphere(v) for v, _ in meshes)
        self.concave = not self.spheres and any(not is_convex(*m) for m in meshes)

        rng = np.random.default_rng(seed)
        self.type_ids = dist.sample_type_ids(n_particles, rng)
        r_max = max(bounding_radius(v) for v, _ in meshes)
        self.positions, self.box = lattice_positions(n_particles, initial_spacing * r_max, rng)
        # exact spheres do not rotate in the simulation: give every particle a random orientation up front so the
        # facets of the (traced) meshes are not all aligned
        self.orientations = None
        if self.spheres:
            quaternions = rng.normal(size=(n_particles, 4))
            self.orientations = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)

        self.decomposed = None
        if self.concave and not use_convex_hull:
            t0 = time.time()
            self.decomposed = decompose_all(meshes, max_workers)
            log.info("convex decomposition: %.1f s", time.time() - t0)

        self.simulation = None
        self.integrator = None

    # -- setup --
    def _make_integrator(self):
        import hoomd
        max_translation = 0.1 * self.dist.mean_diameter
        max_rotation = 0.2
        if self.spheres:
            integ = hoomd.hpmc.integrate.Sphere()
            for name, (vertices, _) in zip(self.type_names, self.meshes):
                integ.shape[name] = dict(diameter=2.0 * bounding_radius(vertices))      # the circumsphere: no overlaps
                integ.d[name] = max_translation
        elif self.decomposed is not None:
            integ = hoomd.hpmc.integrate.ConvexSpheropolyhedronUnion()
            for name, (parts, centers) in zip(self.type_names, self.decomposed):
                integ.shape[name] = {"shapes": [{"vertices": p.tolist()} for p in parts],
                                     "positions": centers}
                integ.d[name], integ.a[name] = max_translation, max_rotation
        else:
            integ = hoomd.hpmc.integrate.ConvexPolyhedron()
            for name, (vertices, faces) in zip(self.type_names, self.meshes):
                if self.use_convex_hull:
                    import trimesh
                    vertices = trimesh.Trimesh(vertices, faces, process=False).convex_hull.vertices
                integ.shape[name] = dict(vertices=vertices)
                integ.d[name], integ.a[name] = max_translation, max_rotation
        integ.nselect = 2
        return integ

    def setup(self, device) -> None:
        import hoomd
        snap = hoomd.Snapshot()
        snap.particles.N = self.n_particles
        snap.configuration.box = [*self.box, 0, 0, 0]
        snap.particles.types = self.type_names
        snap.particles.position[:] = self.positions
        snap.particles.typeid[:] = self.type_ids
        if self.orientations is not None:
            snap.particles.orientation[:] = self.orientations
        self.integrator = self._make_integrator()
        sim = hoomd.Simulation(device=device, seed=self.seed)
        sim.operations.integrator = self.integrator
        sim.create_state_from_snapshot(snap)
        self.simulation = sim

    # -- dynamics --
    @property
    def total_particle_volume(self) -> float:
        return float(self.type_volumes[self.type_ids].sum())

    @property
    def volume_fraction(self) -> float:
        return self.total_particle_volume / self.simulation.state.box.volume

    def randomize(self, steps: int = 10_000) -> None:
        t0 = time.time()
        self.simulation.run(steps)
        log.info("randomize: %d steps, %.1f s", steps, time.time() - t0)

    def compress(self, target_volume_fraction: float, max_steps: int = 1_000_000,
                 chunk: int = 1000) -> CompressionResult:
        """Quick-compress to the target volume fraction. Raises if it is not reached in ``max_steps``
        additional steps."""
        import hoomd
        sim = self.simulation
        box = hoomd.Box.from_box(sim.state.box)
        box.volume = self.total_particle_volume / target_volume_fraction
        updater = hoomd.hpmc.update.QuickCompress(trigger=hoomd.trigger.Periodic(10), target_box=box)
        sim.operations.updaters.append(updater)
        start_step, t0 = sim.timestep, time.time()
        last_report = t0
        complete = False
        try:
            while sim.timestep - start_step < max_steps:
                sim.run(chunk)
                complete = bool(updater.complete)   # only valid while the updater is attached
                if complete:
                    break
                if time.time() - last_report > PROGRESS_INTERVAL_S:
                    log.info("compressing to vf=%g: now %.4f, %d steps, %.0f s", target_volume_fraction,
                             self.volume_fraction, sim.timestep - start_step, time.time() - t0)
                    last_report = time.time()
        finally:
            sim.operations.updaters.remove(updater)
        if not complete:
            raise RuntimeError(f"compression to {target_volume_fraction} not reached in {max_steps} steps "
                               f"(at {self.volume_fraction:.4f})")
        if self.integrator.overlaps != 0:
            raise RuntimeError(f"{self.integrator.overlaps} overlaps after compression")
        moves = self.integrator.translate_moves
        result = CompressionResult(target_volume_fraction, self.volume_fraction, sim.timestep - start_step,
                                   time.time() - t0, moves[0] / max(sum(moves), 1))
        log.info("compressed to vf=%.4f in %d steps, %.1f s (acceptance %.2f)", result.volume_fraction,
                 result.steps, result.seconds, result.acceptance)
        return result

    def write(self, path: str | Path, overwrite: bool = False) -> None:
        """Write the current state as GSD (atomically)."""
        import hoomd
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.unlink(missing_ok=True)
        hoomd.write.GSD.write(state=self.simulation.state, filename=str(tmp), mode="xb")
        os.replace(tmp, path)


def pack_sample(meshes, dist, n_particles: int, volume_fractions, out_dir: str | Path, device,
                seed: int = 42, randomize_steps: int = 10_000, use_convex_hull: bool = False,
                overwrite: bool = False) -> list[CompressionResult]:
    """Randomise, then compress through the (ascending) ``volume_fractions`` writing
    ``packed_<vf>.gsd`` for each."""
    vfs = sorted(volume_fractions)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [out_dir / f"packed_{vf:g}.gsd" for vf in vfs]
    if not overwrite and all(t.exists() for t in targets):
        log.info("all packings exist in %s", out_dir)
        return []
    packer = Packer(meshes, dist, n_particles, seed=seed, use_convex_hull=use_convex_hull)
    packer.setup(device)
    packer.randomize(randomize_steps)
    results = []
    for vf, target in zip(vfs, targets):
        results.append(packer.compress(vf))
        if overwrite or not target.exists():
            packer.write(target, overwrite=overwrite)
    return results

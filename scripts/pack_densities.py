"""Pack the GRS sample (10 000 particles, sigma_r = 0.1) to the volume fractions of the two comparison cases.

Uses the particle meshes in data/grs_sample (no GS needed). Run with a Python that has HOOMD (GPU or CPU):
    PYTHONPATH=. <hoomd python> scripts/pack_densities.py
Writes data/grs_sample/packing/packed_<vf>.gsd.
"""
import json
import logging
from pathlib import Path

from icegrain import meshes, packing
from icegrain.config import SizeDistConfig
from icegrain.sizedist import gaussian_mass_distribution

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "grs_sample"
RHO_ICE = 917.0
DENSITIES = (120.0, 300.0)                       # kg m-3
N_PARTICLES, SEED, RANDOMIZE_STEPS = 10_000, 42, 10_000   # as in the sample's config.json

if __name__ == "__main__":      # required: the convex decomposition starts worker processes
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    sizes_cfg = SizeDistConfig(**json.loads((SAMPLE / "config.json").read_text())["config"]["sizes"])
    sizes = gaussian_mass_distribution(sizes_cfg)
    files = [SAMPLE / "particles" / f"scaled_particle_{i}.obj" for i in range(1, sizes_cfg.n_bins + 1)]
    particle_meshes = [meshes.read_obj(f) for f in files]
    volume_fractions = [float(f"{rho / RHO_ICE:g}") for rho in DENSITIES]   # rounded as in the file names
    packing.pack_sample(particle_meshes, sizes, N_PARTICLES, volume_fractions, SAMPLE / "packing",
                        packing.make_device(), seed=SEED, randomize_steps=RANDOMIZE_STEPS)

"""Particle size distributions used for packing."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm

from .config import SizeDistConfig


def sphere_volume(diameter):
    return np.pi / 6.0 * np.asarray(diameter, dtype=float) ** 3


@dataclass(frozen=True)
class SizeDistribution:
    """Discrete distribution: ``frequencies[i]`` is the *number* fraction of
    particles with equivalent-sphere diameter ``diameters[i]`` (um)."""
    diameters: np.ndarray
    frequencies: np.ndarray

    def __post_init__(self):
        if self.diameters.shape != self.frequencies.shape:
            raise ValueError("diameters and frequencies must have the same shape")
        if not np.isclose(self.frequencies.sum(), 1.0):
            raise ValueError("frequencies must sum to 1")

    @property
    def mean_diameter(self) -> float:
        """Number-weighted mean diameter d10."""
        return float(np.sum(self.diameters * self.frequencies))

    @property
    def mean_volume(self) -> float:
        return float(np.sum(sphere_volume(self.diameters) * self.frequencies))

    @property
    def mean_volume_diameter(self) -> float:
        """Diameter of the sphere with the mean particle volume (d30)."""
        return float((6.0 * self.mean_volume / np.pi) ** (1.0 / 3.0))

    @property
    def sauter_diameter(self) -> float:
        """Volume-to-surface mean diameter d32 (what Hapke/Mie-type models effectively see)."""
        d, f = self.diameters, self.frequencies
        return float(np.sum(f * d ** 3) / np.sum(f * d ** 2))

    def sample_type_ids(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(len(self.diameters), size=n, p=self.frequencies)


def gaussian_mass_distribution(cfg: SizeDistConfig) -> SizeDistribution:
    """Gaussian pdf in diameter, interpreted as a *mass* distribution and
    converted to number fractions (pdf / particle volume)."""
    lo = cfg.mean_um - cfg.sigma_multiplier * cfg.std_um
    hi = cfg.mean_um + cfg.sigma_multiplier * cfg.std_um
    edges = np.linspace(lo, hi, cfg.n_bins + 1)
    diameters = 0.5 * (edges[:-1] + edges[1:])
    mass = norm.pdf(diameters, loc=cfg.mean_um, scale=cfg.std_um)
    number = mass / sphere_volume(diameters)
    return SizeDistribution(diameters, number / number.sum())


def monodisperse(diameter_um: float, n_types: int = 1) -> SizeDistribution:
    return SizeDistribution(np.full(n_types, float(diameter_um)), np.full(n_types, 1.0 / n_types))


def from_histogram_npz(path: str | Path) -> SizeDistribution:
    """Measured histogram as saved by the old notebooks (``bin_centers[_diameter]``,
    ``normalized_counts``)."""
    data = np.load(path)
    key = "bin_centers_diameter" if "bin_centers_diameter" in data else "bin_centers"
    freq = np.asarray(data["normalized_counts"], dtype=float)
    return SizeDistribution(np.asarray(data[key], dtype=float), freq / freq.sum())

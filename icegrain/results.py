"""The result of one ray-tracing run: one spectrum in one compressed file.

(The old layout was one ``.npz`` per wavelength: ~1900 files of 167 kB each, mostly an unused
10000-bin depth histogram.) We keep what the band-depth analysis needs: ray counts and the
emission-angle histogram for every wavelength.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .angles import DEFAULT_EMISSION_BINS, DEFAULT_PHASE_BINS


@dataclass
class SpectrumResult:
    wavelength_nm: np.ndarray             # (n,) integer, ascending
    reflected: np.ndarray                 # (n,) rays that left through the top surface
    absorbed: np.ndarray                  # (n,) rays absorbed in the medium
    unfinished: np.ndarray                # (n,) rays that neither escaped nor were absorbed (errors, step limit)
    hist_emission_angles: np.ndarray      # (n, n_emission_bins) escaped rays per emission-angle bin
    bins_emission_angles: np.ndarray      # (n_emission_bins + 1,) bin edges in degrees
    hist_phase_angles: np.ndarray         # (n, n_phase_bins) escaped rays per phase-angle bin
    bins_phase_angles: np.ndarray         # (n_phase_bins + 1,) bin edges in degrees
    meta: dict = field(default_factory=dict)

    @property
    def number_of_rays(self) -> np.ndarray:
        return self.reflected + self.absorbed

    @property
    def albedo(self) -> np.ndarray:
        """Hemispherical reflectance: reflected / (reflected + absorbed)."""
        total = self.number_of_rays
        return self.reflected / np.where(total == 0, np.nan, total)

    def albedo_error(self) -> np.ndarray:
        """Binomial standard error of the albedo."""
        albedo = self.albedo
        return np.sqrt(albedo * (1.0 - albedo) / self.number_of_rays)

    def save(self, path: str | Path) -> None:
        """Write atomically: a killed job never leaves a half-written file."""
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:      # a file object, so numpy does not append '.npz'
            np.savez_compressed(
                f, wavelength_nm=self.wavelength_nm, reflected=self.reflected, absorbed=self.absorbed,
                unfinished=self.unfinished, hist_emission_angles=self.hist_emission_angles,
                bins_emission_angles=self.bins_emission_angles, hist_phase_angles=self.hist_phase_angles,
                bins_phase_angles=self.bins_phase_angles, meta=np.array(json.dumps(self.meta, sort_keys=True)))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> "SpectrumResult":
        with np.load(path) as z:
            if "hist_phase_angles" not in z:
                raise ValueError(f"{path} was written before phase-angle histograms existed; recompute it")
            return cls(z["wavelength_nm"], z["reflected"], z["absorbed"], z["unfinished"],
                       z["hist_emission_angles"], z["bins_emission_angles"],
                       z["hist_phase_angles"], z["bins_phase_angles"], json.loads(str(z["meta"])))


class ResultWriter:
    """Collects wavelengths as they finish and checkpoints the single result file.

    To resume a killed run, create a ``ResultWriter`` for the same path and skip everything in ``done_nm``.
    """

    def __init__(self, path: str | Path, meta: dict | None = None,
                 emission_bins: np.ndarray = DEFAULT_EMISSION_BINS, phase_bins: np.ndarray = DEFAULT_PHASE_BINS):
        self.path = Path(path)
        self.meta = dict(meta or {})
        self.emission_bins = np.asarray(emission_bins, dtype=float)
        self.phase_bins = np.asarray(phase_bins, dtype=float)
        self._rows: dict[int, dict] = {}                    # wavelength -> counts
        if self.path.exists():
            self._load_existing()

    def _load_existing(self) -> None:
        old = SpectrumResult.load(self.path)
        if not (np.array_equal(old.bins_emission_angles, self.emission_bins)
                and np.array_equal(old.bins_phase_angles, self.phase_bins)):
            raise ValueError(f"{self.path}: existing file uses different angle bins")
        for i, wavelength in enumerate(old.wavelength_nm):
            self._rows[int(wavelength)] = dict(
                reflected=old.reflected[i], absorbed=old.absorbed[i], unfinished=old.unfinished[i],
                emission=old.hist_emission_angles[i], phase=old.hist_phase_angles[i])

    @property
    def done_nm(self) -> set[int]:
        return set(self._rows)

    def add(self, wavelength_nm: int, *, reflected: int, absorbed: int, unfinished: int,
            emission_histogram, phase_histogram) -> None:
        emission = np.asarray(emission_histogram, dtype=np.int64)
        phase = np.asarray(phase_histogram, dtype=np.int64)
        if emission.shape != (len(self.emission_bins) - 1,) or phase.shape != (len(self.phase_bins) - 1,):
            raise ValueError(f"expected {len(self.emission_bins) - 1} emission and {len(self.phase_bins) - 1} "
                             f"phase bins, got {emission.shape} and {phase.shape}")
        self._rows[int(wavelength_nm)] = dict(reflected=reflected, absorbed=absorbed, unfinished=unfinished,
                                              emission=emission, phase=phase)

    def to_result(self) -> SpectrumResult:
        wavelengths = sorted(self._rows)
        count = lambda key: np.array([self._rows[w][key] for w in wavelengths], dtype=np.int64)
        histogram = lambda key, bins: count(key).reshape(len(wavelengths), len(bins) - 1)
        return SpectrumResult(
            np.array(wavelengths, dtype=np.int64), count("reflected"), count("absorbed"), count("unfinished"),
            histogram("emission", self.emission_bins), self.emission_bins,
            histogram("phase", self.phase_bins), self.phase_bins, self.meta)

    def flush(self) -> None:
        self.to_result().save(self.path)

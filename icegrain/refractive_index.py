"""Complex refractive index of water ice from the tables used in the project.

All sources return a :class:`RefractiveIndex` (wavelength in um, ascending);
``resample`` puts it on the integer-nm grid used by the ray-tracing runs.

The loaders take the folder that holds the tables (``root``) explicitly.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = ("grundy_schmitt", "mastrapa", "warren", "picard")


def wavelength_grid_nm(first: int, last: int, step: int) -> np.ndarray:
    """Integer-nm grid, both ends inclusive (avoids the float-rounding file names of ``np.arange`` in um)."""
    return np.arange(first, last + 1, step, dtype=np.int64)


@dataclass(frozen=True)
class RefractiveIndex:
    wavelength_um: np.ndarray
    n: np.ndarray
    k: np.ndarray
    label: str = ""

    def __post_init__(self):
        if np.any(np.diff(self.wavelength_um) <= 0):
            raise ValueError("wavelength must be strictly ascending")

    def at(self, wavelength_um) -> tuple[np.ndarray, np.ndarray]:
        """Linear interpolation of n and k."""
        wl = np.asarray(wavelength_um, dtype=float)
        return (np.interp(wl, self.wavelength_um, self.n),
                np.interp(wl, self.wavelength_um, self.k))

    def resample(self, grid_nm, bandwidth_um: float | None = None) -> "RefractiveIndex":
        """Values on ``grid_nm``.

        Without ``bandwidth_um``: linear interpolation. With it: at every grid point, the Gaussian-weighted
        mean (sigma = bandwidth, window +-bandwidth) of the table points around it, which mimics an
        instrument bandwidth (the old ``*_bandwidth`` readers). Where the window holds no table point the
        value is interpolated.
        """
        wavelength = np.asarray(grid_nm, dtype=float) / 1000.0
        if bandwidth_um is None:
            n, k = self.at(wavelength)
        else:
            n, k = self._smoothed(wavelength, bandwidth_um)
        return RefractiveIndex(wavelength, n, k, self.label)

    def _smoothed(self, wavelength: np.ndarray, bandwidth_um: float):
        n, k = np.empty_like(wavelength), np.empty_like(wavelength)
        # a small tolerance keeps the choice of points in the window independent of float rounding
        reach = bandwidth_um * (1 + 1e-9)
        for i, center in enumerate(wavelength):
            first = np.searchsorted(self.wavelength_um, center - reach, side="left")
            last = np.searchsorted(self.wavelength_um, center + reach, side="right")
            if last == first:
                n[i], k[i] = self.at(center)
                continue
            weights = np.exp(-0.5 * ((self.wavelength_um[first:last] - center) / bandwidth_um) ** 2)
            weights /= weights.sum()
            n[i], k[i] = weights @ self.n[first:last], weights @ self.k[first:last]
        return n, k


def _from_frame(df: pd.DataFrame, label: str) -> RefractiveIndex:
    df = df.sort_values("Wavelength").drop_duplicates("Wavelength")
    return RefractiveIndex(df["Wavelength"].to_numpy(float), df["n"].to_numpy(float),
                           df["k"].to_numpy(float), label)


def load_grundy_schmitt(root: Path, species: str = "crystalline", temperature: int = 80) -> RefractiveIndex:
    """Grundy & Schmitt (1998) NIR ice constants (tables per temperature, in wavenumbers)."""
    abbrev = {"crystalline": "Ih", "liquid": "liq"}[species.lower()]
    name = f"optcte-NIR-H2O-{abbrev}-{int(temperature)}K"
    path = Path(root) / "EXPERIMENT_BS_20120924_011" / name / f"{name}.data.txt"
    df = pd.read_table(path, skiprows=35, names=["wavenumber", "n", "k"], sep=r"\s+")
    df["Wavelength"] = 1e4 / df["wavenumber"]
    return _from_frame(df, f"Grundy&Schmitt {species} {int(temperature)}K")


def load_mastrapa(root: Path, species: str = "crystalline", temperature: int = 80) -> RefractiveIndex:
    """Mastrapa et al. (2008) crystalline/amorphous ice, 20-250 K."""
    df = pd.read_table(Path(root) / "mastrapa_2008.txt", skiprows=16,
                       names=["Type", "Temperature", "Wavelength", "n", "k"], sep=r"\s+")
    sel = df[(df["Type"].str.lower() == species.lower()) & (df["Temperature"] == temperature)]
    if sel.empty:
        avail = sorted(set(map(tuple, df[["Type", "Temperature"]].drop_duplicates().to_numpy())))
        raise ValueError(f"no Mastrapa table for {species} {temperature} K; available: {avail}")
    return _from_frame(sel, f"Mastrapa2008 {species} {int(temperature)}K")


def load_warren(root: Path) -> RefractiveIndex:
    """Warren & Brandt (2008) / Warren (2009) compilation, ~266 K, UV to microwave."""
    df = pd.read_table(Path(root) / "warren_2009.txt", skiprows=16,
                       names=["Wavelength", "n", "k"], sep=r"\s+")
    return _from_frame(df, "Warren2009")


def load_picard() -> RefractiveIndex:
    """Picard et al. (2016) via the optional ``snowoptics`` package."""
    import snowoptics  # noqa: PLC0415 - optional dependency
    wl = np.asarray(snowoptics.refractive_index.wavelengths2016, dtype=float)
    n, k = snowoptics.refractive_index.refice2016(wl * 1e-9)
    return _from_frame(pd.DataFrame({"Wavelength": wl / 1000.0, "n": n, "k": k}), "Picard2016")


def load(source: str, root: Path, species: str = "crystalline", temperature: int = 80) -> RefractiveIndex:
    """One of ``SOURCES``; ``root`` is the folder with the tables (not used by "picard")."""
    if source == "grundy_schmitt":
        return load_grundy_schmitt(root, species, temperature)
    if source == "mastrapa":
        return load_mastrapa(root, species, temperature)
    if source == "warren":
        return load_warren(root)
    if source == "picard":
        return load_picard()
    raise ValueError(f"unknown refractive-index source {source!r}; choose from {SOURCES}")

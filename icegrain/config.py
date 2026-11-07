"""Parameter sets of the pipeline stages.

Every stage takes one frozen dataclass. ``config_id()`` is a short stable hash of its values, so output
folders are named after *what was computed* instead of hand-built strings.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field


def _short_count(n: int) -> str:
    """1_000_000 -> '1e6', 2_500_000 -> '2.5e6', 12345 -> '12345'."""
    if n >= 1000:
        exponent = len(str(n)) - 1
        mantissa = n / 10 ** exponent
        if round(mantissa, 3) == mantissa:
            return f"{mantissa:g}e{exponent}"
    return str(n)


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class _Hashable:
    def to_dict(self) -> dict:
        return asdict(self)  # type: ignore[call-overload]

    @classmethod
    def from_dict(cls, values: dict):
        """Inverse of ``to_dict`` (JSON turns tuples into lists, so lists become tuples again)."""
        return cls(**{key: tuple(v) if isinstance(v, list) else v for key, v in values.items()})

    def config_id(self) -> str:
        return hashlib.sha1(_canonical_json(self.to_dict()).encode()).hexdigest()[:10]


@dataclass(frozen=True)
class GRSConfig(_Hashable):
    """Gaussian-random-sphere generator (siris4 ``GS``).

    ``rel_std`` is the relative standard deviation of the radial distance (the
    "std dev of the radius" of the GRS particles). One distinct random shape
    is generated per size bin.
    """
    n_shapes: int = 31
    seed: int = 1               # must be > 0: GS draws a random seed for <= 0 (not reproducible)
    rel_std: float = 0.1
    power_law: float = 3.0      # correlation power-law index (larger = more irregular)
    l_min: int = 2
    l_max: int = 7
    rows_per_octant: int = 40   # mesh resolution, 8*rows^2 triangles

    def __post_init__(self):
        if self.seed <= 0:
            raise ValueError("seed must be positive so that shapes are reproducible")


@dataclass(frozen=True)
class SizeDistConfig(_Hashable):
    """Gaussian *mass* distribution of the equivalent-sphere diameter (um)."""
    mean_um: float = 100.0
    std_um: float = 20.0
    n_bins: int = 31
    sigma_multiplier: float = 2.5   # distribution truncated at mean +- k*std


@dataclass(frozen=True)
class PackingConfig(_Hashable):
    n_particles: int = 100_000
    volume_fractions: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5)
    seed: int = 42
    randomize_steps: int = 10_000


@dataclass(frozen=True)
class SampleConfig(_Hashable):
    """Everything that defines a packed sample (independent of the optics)."""
    grs: GRSConfig = field(default_factory=GRSConfig)
    sizes: SizeDistConfig = field(default_factory=SizeDistConfig)
    packing: PackingConfig = field(default_factory=PackingConfig)

    def __post_init__(self):
        if self.grs.n_shapes != self.sizes.n_bins:
            raise ValueError("one GRS shape per size bin is required: "
                             f"n_shapes={self.grs.n_shapes} != n_bins={self.sizes.n_bins}")

    @classmethod
    def from_dict(cls, values: dict) -> "SampleConfig":
        return cls(grs=GRSConfig.from_dict(values["grs"]), sizes=SizeDistConfig.from_dict(values["sizes"]),
                   packing=PackingConfig.from_dict(values["packing"]))

    def label(self) -> str:
        """Human-readable summary of the main settings, for folder names, e.g. ``grs0.1_d100sd20_b31_N100000``
        (GRS std, mean and std of the size distribution, number of size bins, number of particles).
        Only a hint for people: ``config_id()`` is what makes a folder unique."""
        return (f"grs{self.grs.rel_std:g}_d{self.sizes.mean_um:g}sd{self.sizes.std_um:g}"
                f"_b{self.sizes.n_bins}_N{self.packing.n_particles}")


@dataclass(frozen=True)
class TraceConfig(_Hashable):
    """One ray-tracing run = one (sample, volume fraction, scale, optics, geometry of illumination)."""
    sample_id: str = ""
    volume_fraction: float = 0.5
    scale_percent: float = 100.0         # geometry scale relative to the sample sizes
    ri_source: str = "grundy_schmitt"    # see refractive_index.py
    species: str = "crystalline"
    temperature_k: float = 80.0
    ri_bandwidth_um: float | None = 0.002   # Gaussian smoothing of the tables; None = interpolate
    wavelength_nm: tuple[int, int, int] = (1200, 2500, 2)   # first, last (inclusive), step
    incidence_deg: float = 0.0           # angle from the surface normal (0 = normal incidence)
    azimuth_deg: float = 0.0             # direction of the beam's horizontal component
    number_of_rays: int = 10_000_000
    beam_radius_um: float = 2000.0       # radius of the illuminated disc (before the scale factor)
    beam_offset_cell: tuple[float, float] = (0.0, 0.0)   # beam centre shift, fractions of the box edge
    beam_distance_um: float = 10_000.0   # height of the source disk above the surface (must clear the particles)
    depth_boxes: tuple[int, int] = (20, 30)   # slab depth in packed boxes = product (two instancing levels)
    max_scattering_count: int = 40_000_000
    seed: int | None = None
    backend: str = "optix"               # or "embree" (CPU)

    def __post_init__(self):
        if not 0.0 <= self.incidence_deg < 90.0:
            raise ValueError("incidence_deg must be in [0, 90)")
        if self.backend not in ("optix", "embree"):
            raise ValueError("backend must be 'optix' or 'embree'")
        if not all(-0.5 <= x <= 0.5 for x in self.beam_offset_cell):
            raise ValueError("beam_offset_cell must be within +-0.5 (a fraction of the periodic box edge)")

    def label(self) -> str:
        """Human-readable summary for folder names, e.g.
        ``scale10_vf0.1_grundy_schmitt-crystalline-80K_inc0_wl1360-1830-2_beam+0.33-0.33_1e6rays_seed2``.
        Only a hint for people: ``config_id()`` is what makes a folder unique."""
        first, last, step = self.wavelength_nm
        azimuth = f"az{self.azimuth_deg:g}" if self.azimuth_deg else ""
        offset = f"{self.beam_offset_cell[0]:+.2f}{self.beam_offset_cell[1]:+.2f}"
        seed = "none" if self.seed is None else self.seed
        return (f"scale{self.scale_percent:g}_vf{self.volume_fraction:g}_{self.ri_source}-{self.species}-"
                f"{self.temperature_k:g}K_inc{self.incidence_deg:g}{azimuth}_wl{first}-{last}-{step}_"
                f"beam{offset}_{_short_count(self.number_of_rays)}rays_seed{seed}")

    def folder_name(self) -> str:
        """Name of the result folder: the readable label and the hash."""
        return f"{self.label()}_{self.config_id()}"

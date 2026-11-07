"""Directions and angles of the light: incidence, emission (zenith) and phase angle.

Coordinates: the surface is the plane z = 0, +z points out of the surface towards the light and the
observer. A ray *travelling* along ``direction`` has ``direction[2] < 0`` when it goes into the surface.

    incidence angle i   angle of the incoming light from the surface normal (0 = straight down)
    emission angle e    angle of an outgoing ray from the surface normal, ``arccos(d_out[2])``
    phase angle g       angle between the direction to the light source and the outgoing ray:
                        ``cos g = d_out . (-d_in)``.  g = 0 is retro-reflection (opposition), g = 2i the
                        specular direction, and for normal incidence g = e.
"""
import numpy as np

# Zenith-angle bins of the old exports (deg), for comparability with the old results.
DEFAULT_EMISSION_BINS = np.array([0, *np.arange(5, 86, 10), 90], dtype=float)
# The same bins mirrored about 90 deg, so the first 10 bins of a phase histogram are the emission bins.
DEFAULT_PHASE_BINS = np.concatenate([DEFAULT_EMISSION_BINS, 180.0 - DEFAULT_EMISSION_BINS[-2::-1]])


def incident_direction(incidence_deg: float, azimuth_deg: float = 0.0) -> np.ndarray:
    """Unit vector of the beam travelling towards the slab; the incidence angle is from the surface normal."""
    theta, phi = np.radians(incidence_deg), np.radians(azimuth_deg)
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), -np.cos(theta)])


def emission_angles_deg(directions: np.ndarray) -> np.ndarray:
    """Emission (zenith) angle of outgoing rays, ``directions`` of shape (n, 3)."""
    return np.degrees(np.arccos(np.clip(directions[:, 2], -1.0, 1.0)))


def phase_angles_deg(directions: np.ndarray, incident: np.ndarray) -> np.ndarray:
    """Phase angle of outgoing rays for a beam travelling along ``incident``."""
    return np.degrees(np.arccos(np.clip(directions @ -np.asarray(incident, dtype=float), -1.0, 1.0)))


def lambertian_fractions(incidence_deg: float, bin_edges_deg: np.ndarray, kind: str = "phase",
                         n_theta: int = 1800, n_phi: int = 720) -> np.ndarray:
    """Fraction of the light that an ideal Lambertian surface sends into each angle bin.

    This is the reference for reflectance factors: ``counts / counts_total / lambertian_fractions`` is the
    reflectance relative to a Lambertian surface (1 = Lambertian). ``kind`` is "phase" or "emission"; the
    emission fractions do not depend on the incidence angle. Numerical integration of the cosine-weighted
    upper hemisphere (midpoint rule), accurate to about 1e-4.
    """
    if kind not in ("phase", "emission"):
        raise ValueError("kind must be 'phase' or 'emission'")
    theta = (np.arange(n_theta) + 0.5) * (np.pi / 2 / n_theta)
    phi = (np.arange(n_phi) + 0.5) * (2 * np.pi / n_phi)
    weights = np.cos(theta) * np.sin(theta) * (np.pi / 2 / n_theta) * (2 * np.pi / n_phi) / np.pi
    weights = np.broadcast_to(weights[:, None], (n_theta, n_phi))
    if kind == "emission":
        angles = np.broadcast_to(np.degrees(theta)[:, None], (n_theta, n_phi))
    else:
        incident = incident_direction(incidence_deg)
        out = np.stack([np.sin(theta)[:, None] * np.cos(phi)[None], np.sin(theta)[:, None] * np.sin(phi)[None],
                        np.broadcast_to(np.cos(theta)[:, None], (n_theta, n_phi))], axis=-1)
        angles = phase_angles_deg(out.reshape(-1, 3), incident).reshape(n_theta, n_phi)
    return np.histogram(angles.ravel(), bins=bin_edges_deg, weights=weights.ravel())[0]

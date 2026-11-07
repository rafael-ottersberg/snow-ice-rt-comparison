# snow-rt-comparison

Spectral albedo of semi-infinite snow from six radiative-transfer models, for the same density and specific surface area (SSA)
and the same ice refractive index (Warren & Brandt 2008):

| model | what | where |
|---|---|---|
| TARTES (default shape) | two-stream, analytic | `external/tartes` (upstream, PyPI `tartes`) |
| PySNICAR-ADv4 (spheres) | SNICAR via Octave | `external/pysnicar` (private) |
| Mie + DISORT (spheres) | polydisperse Mie spheres, discrete ordinates | `external/rt-malinka` (`malinka.mie`, private) |
| Malinka, random shape / Markov mixture | stereological model, Malinka (2023) | `external/rt-malinka` (private) |
| photontracer, GRS | Monte Carlo on 10 000 packed Gaussian random spheres (sigma_r = 0.1, d = 100 um, 31 bins) | `data/grs_sample`, `icegrain/` |
| photontracer, GRF | Monte Carlo on a periodic level-cut Gaussian random field | `external/grr-microstructures`, `grf_slab.py` |

Cases (`cases.py`): density 120 kg m-3 with SSA 20 m2 kg-1, and density 300 kg m-3 with SSA 15 m2 kg-1; direct beam at
normal incidence (plane albedo), 400-2500 nm in 20 nm steps.

## Setup
```bash
git submodule update --init --recursive
mamba env create -f environment.yml && mamba activate snow-rt     # photontracer 1.1.0 from PyPI
pip install -e external/pysnicar
pysnicar config --set $PWD/data/pysnicar && pysnicar fetch v4 --variant limited   # ~0.9 GB, Octave needed
```

## Run order
```bash
PYTHONPATH=. <python with HOOMD> scripts/pack_densities.py   # GRS packings at the two densities (GPU, ~1 h)
python scripts/run_models.py                                  # TARTES, PySNICAR, Malinka, Mie + DISORT  (~10 min)
python scripts/run_photontracer.py                            # GRS slab, 1e6 rays x 106 wavelengths     (~30 min per case)
python scripts/grf_convergence.py cell|depth|feature          # how big must the GRF cell / slab be
python scripts/plot_grf_convergence.py
python scripts/run_photontracer_grf.py                        # GRF slab, 3 seeds per case
python scripts/compare_albedo.py                              # figures/albedo_<case>.png, albedo_table.md, mie_vs_pysnicar.png
SNOWRT_SZA=30 SNOWRT_RESULTS=$PWD/results/sza30 python scripts/compare_albedo.py   # the 30 deg run
python scripts/plot_incidence.py                              # figures/albedo_change_sza30.png
python scripts/render_microstructures.py                       # figures/microstructure_<case>.png
```
Results go to `results/` (not tracked), figures to `figures/`.

## Layout
* `cases.py` - cases, wavelength grid, paths.
* `grf_slab.py` - GRF cell, free-surface tile and slab meshes (marching cubes). `grf_trace.py` - photontracer scene
  (two instancing levels, as in icegrain) and the tracing loop shared by GRS and GRF.
* `icegrain/` - subset of the modules of `~/code/icegrain` (packing, tracing, refractive index); `data/grs_sample/` the
  particle meshes and packings (made with icegrain: GRS sigma_r = 0.1, 10 000 particles); `data/refractive_indices/`
  the Warren table.

## Photontracer: definitions and caveats
* **Albedo** = rays leaving upwards / rays that interacted at least once. A periodic cell repeated exactly has straight
  channels without ice (a ray at normal incidence can fly through all 600 layers). An infinite medium has none, so these rays
  (`free`) are removed from the normalisation; they are 0.01 % of the rays for GRS and up to 8 % for a GRF cell of
  N = 128 voxels at 120 kg m-3. Rays that leave through the sides or bottom are not reflected.
* **Slab** = 600 cells deep and 651 wide (20 x 30 and 21 x 31 instances), surface at z = 0. Depth/width tests
  (`grf_convergence.py depth`): the albedo is converged for lambda >= 800 nm from 100 cells deep; at 600 nm it needs >= 300 deep and the
  full width.
* **GRF cell size** (`grf_convergence.py cell`, 3 seeds): the albedo no longer changes with N from N = 192 (120 kg m-3) or
  N = 96 (300 kg m-3) within the scatter over seeds (+-0.003-0.007); N = 256 is used. Smaller cells are biased by the
  vertical channels (N = 64: +0.02 in albedo at 120 kg m-3). Feature size (resolution): 20, 30, 45 voxels agree within 0.003-0.007.
* The voxel size of every GRF realisation is set so that its mesh has exactly the target SSA; the GRS size by scaling the
  geometry (SSA = 6 / (rho_ice d_opt), d_opt from `data/grs_sample/sample.json`).
* The free surface of the GRF slab is the graded top of `grf_firn.graded_top` (width 20 voxels); floating ice is not removed (at
  these densities the ice network is not necessarily connected).
* A small fraction of the rays in the instanced GRF slabs reports geometry warnings (`NUMBER_OF_WARNINGS`, probably the seams
  between open tiles): 0.03 % in the smoke test, up to 0.2 % of the rays at some wavelengths in the final runs. Not investigated further.
* TARTES: the code default is g0 = 0.82 and B = n^2 (`shape_parameterization="robledano23"`); the docstring says 0.86.
* The depth/width tests were done at 600-2200 nm; the slab size was not tested below 600 nm.

## Model overview
| | TARTES | PySNICAR-ADv4 | Mie + DISORT | Malinka random shape | Malinka Markov | photontracer GRS | photontracer GRF |
|---|---|---|---|---|---|---|---|
| **RT solver** | delta-Eddington two-stream, closed form per layer (banded solve for several layers) | delta-Eddington layers combined by adding-doubling | DISORT, 16 streams, delta-M (full phase function) | discrete ordinates (N = 16), one-phase RTE, Malinka 2023 Eq. 36 | discrete ordinates (N = 16), coupled air + ice RTE, Eq. 51 | Monte Carlo ray tracing, 1e6 rays | Monte Carlo ray tracing, 1e6 rays |
| **Angular accuracy** | two-stream; about +0.003 to +0.011 high vs exact DISORT for the same omega, g (HG) | same two-stream physics; direct-beam semi-infinite albedo equals TARTES's formula | converged in streams (not tested here) | converged to 1e-4 for N >= 16 (repo) | as random shape | exact for the mesh geometry (statistical error) | as GRS |
| **Shape** | no geometry: B (absorption enhancement), g; default B = n^2, g = 0.82 constant | spheres, one effective radius 3/(rho_ice SSA) | polydisperse spheres (gamma distribution in rt-malinka) | chord statistics (a1, a0), random convex grains | chord statistics, Markov mixture (exponential chords) | 10 000 packed Gaussian random spheres (sigma_r = 0.1, d = 100 um) | level cut of a Gaussian random field (N = 256 voxel cell, feature 30 voxels) |
| **Optical input** | omega, g from SSA, B, g, n, kappa (Kokhanovsky-Zege asymptotics) | Mie tables (sphere) | Mie theory | geometric optics (Fresnel, Beer-Lambert) averaged over chords | as random shape | geometric optics on the mesh | geometric optics on the mesh |
| **Diffraction** | included implicitly (asymptotic parameters) | included (Mie) | included (Mie) | not included (Q_ext = 1); no effect on semi-infinite albedo | as random shape | not included; same remark | same |
| **Scattering assumption** | independent | independent | independent | independent grains (sparse limit a1 -> 0) | correlated two-phase medium, any volume fraction | explicit packing, dependent scattering included | explicit microstructure, dependent scattering included |
| **Surface** | none (no Fresnel layer) | optional Fresnel layer for ice (not checked for our run) | none | rough (J = J0) | rough/flat/mixed; the direct-beam albedo can exceed 1 (we use the swapped evaluation) | real mesh surface (graded top for GRF) | graded top surface |
| **Applicability** | fast, any layering and impurities; relies on shape parameters (B, g) | layered snow, ice, aerosols; sphere optics | any size distribution of spheres; semi-infinite or layered | needs only SSA, density, n | as random shape, dense media | needs a particle and packing set; cost about 30 min per case on a GPU | needs a microstructure; seeds and periodic cell size matter |
| **Known issues here** | too high at strong absorption (> 1400 nm): fixed g = 0.82; `linear` shape (g rises with absorption) cuts the rms from 0.011 to 0.003 | lowest of the six near 1000 to 1300 nm; sphere B and g | agrees with PySNICAR to about 0.001 in the visible, to 0.005 to 0.017 in the NIR | 7 to 14 % above TARTES in the asymptotic q (cause not found; measured before the Markov switch, random shape unchanged) | direct-beam solution not reciprocal (0.997 at 500 nm, > 1 at 400 nm); now evaluated by reciprocity (swapped), which agrees with random shape at near-zero absorption to 3e-4; then 0.004-0.010 (120 kg m-3) and up to 0.018 (300 kg m-3) below random shape in the NIR | geometry warnings up to 0.5-0.7 % of the rays (at 400-420 nm, long paths); SSA set by scaling | up to 1.4 % free rays excluded; warnings up to 0.2 % |

Notes: the Malinka (2014) single-scattering model (Eqs. 56 to 60) is not implemented here; for a semi-infinite pack it should equal the random-shape curve. The Malinka and Mie + DISORT entries are a summary of `external/rt-malinka`, not checked against the papers line by line.

## Cost (benchmarks)
Measured on 24 CPU cores and one RTX 3090 (`bench.py` writes `RESULTS/benchmarks.jsonl` from every run script;
`scripts/bench_malinka.py` estimates the Malinka cost from a few wavelengths). One case = 106 wavelengths, semi-infinite,
direct beam. Environment overrides: `SNOWRT_RESULTS` (output dir), `SNOWRT_SZA` (incidence, deg), `SNOWRT_RAYS`, `SNOWRT_SEEDS`.

| model | one case (106 wavelengths) | notes |
|---|---|---|
| TARTES | 0.01 s | 0.2 ms for one wavelength; 4 layers 0.4 ms |
| PySNICAR | 0.4 s | Octave session; SNICAR always computes its 480 bands; 4 layers 0.6 s |
| Mie + DISORT | ~1300 CPU-s (~12 s per wavelength) | 16 radii; the most expensive analytic model |
| Malinka random / Markov, cold | ~620 / ~440 s, one thread | 69 distinct rounded n on the grid, 6-14 s per operator build (slower when 20 workers build at once) |
| Malinka random / Markov, warm | 5 / 10 s | 0.05 / 0.09 s per wavelength; operators depend only on (N, n, M, npsi), not on density, SSA, absorption or incidence |
| photontracer GRS, 1e6 rays | 1218 s (rho 120) | the visible dominates (first wavelength 162 s, median 0.05 s) |
| photontracer GRS, 1e5 rays | 236 / 152 s (rho 120 / 300) | ~125 s per case is fixed overhead, so 1e5 is only ~5x cheaper than 1e6 |
| photontracer GRF, 1e5 rays, one seed | 198 / 294 s | field + meshes + instancing ~4 s, the rest is tracing |

The 30 deg runs cost the same. 1e5 rays give an albedo error of <= 0.0015 (1e5 vs 1e6 for GRS: max difference 0.0034,
<= 2.3 sigma). Estimated per-wavelength cost of compiled implementations (operation counts, not measured): TARTES ~0.1 us,
SNICAR-AD ~1 us per layer, Malinka random ~0.1-0.3 ms and Markov ~1-3 ms with cached operators (the beam columns, recomputed
on every call now, would add ~10-40 ms), cold operator build ~0.3-1 s per n.

## Incidence 30 deg (results/sza30, 1e5 rays, seed 1)
`figures/albedo_change_sza30.png` (albedo 30 deg minus 0 deg; photontracer paired on the same microstructure),
`figures/albedo_<case>_sza30.png`. Two regimes:
* 1000-1400 nm: all analytic models rise by +0.024 to +0.032 (similar to each other); GRS (+0.016 to +0.017) and GRF
  (+0.011 to +0.013) clearly less (rho 120; paired errors ~0.002; rho 300 similar, GRS closer to the analytic models).
* > 1450 nm (strong absorption): the models split. TARTES rises most (+0.021 at 1800 nm), PySNICAR and Malinka less,
  Mie + DISORT only +0.005 to +0.009 - close to photontracer (+0.004). So here the large angle dependence looks like a
  property of the two-stream solvers (and Malinka) rather than of the plane-parallel geometry; in 1000-1400 nm it is not.
Not investigated further; a rough real surface (spread of local incidence angles) remains a candidate for the 1000-1400 nm
difference. No free rays at 30 deg.

PySNICAR at 30 deg has a spurious bump at 1440-1450 nm (native bands: 0.082, 0.099, 0.082, 0.025, against a smooth
0.064, 0.039, 0.025, 0.016 at 0 deg); probably the delta-Eddington singularity at lambda * mu0 = 1 (SNICAR regularises the
denominator 1 - lm^2 mu0^2 only with epsilon = 1e-5). Not checked further.

## Figures
* `albedo_<case>.png`, `albedo_<case>_sza30.png`: TARTES, PySNICAR, Malinka Markov, photontracer GRS and GRF, and the
  difference to photontracer GRS; `albedo_<case>_all*.png`: the same with all models and TARTES `linear` (the table
  `albedo_table*.md` has all models); `albedo_<case>_slide*.png`: the reduced figure for half of a 16:9 slide
  (6 x 6.4 in, larger text, same difference axis for both cases).
* `albedo_change_sza30.png`: albedo change from 0 to 30 deg (`scripts/plot_incidence.py`).
* `microstructure_<case>.png`: GRS (particles with centres in the cube) and GRF (seed 1; top: the graded free surface
  of the simulation, grains ending in smooth caps; sides and bottom cut, cross-sections in darker blue) in the same 4.8 mm cube (`scripts/render_microstructures.py`, pyvista off-screen; particle meshes decimated for the rendering only).
* `mie_vs_pysnicar.png`, `grf_convergence_*.png`, `albedo_table*.md`.

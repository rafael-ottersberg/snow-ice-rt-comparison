"""Trace a closed slab mesh (grf_slab) with photontracer: albedo of the top surface for direct illumination."""
import numpy as np

from icegrain import refractive_index as ri
from icegrain.angles import incident_direction
from icegrain.config import TraceConfig
from icegrain.raytracing import ICE, make_simulation, set_beam


def closed_geometry(verts, faces):
    """One closed mesh as the whole scene (ice inside)."""
    from photontracer import InstanceGeometry, MeshGeometry
    identity = np.eye(3, 4, dtype=np.float32)[None]
    return InstanceGeometry([MeshGeometry(np.asarray(verts, np.float32), np.asarray(faces, np.uint32))], identity,
                            np.zeros(1, np.uint32), np.full(1, ICE, np.uint32))


def _grid(centres, spacing):
    """Centred grid positions along one axis: ``centres`` sites, ``spacing`` apart."""
    return (np.arange(centres) - (centres - 1) / 2.0) * spacing


def instanced_slab(cell, top, cell_edge_um, surface_um, depth1=20, lateral1=21, depth2=30, lateral2=31):
    """Semi-infinite slab from two instancing levels (as icegrain's ``build_slab``): the periodic ``cell`` mesh and the
    free-surface ``top`` tile (both from grf_slab) repeated ``lateral1`` x ``lateral1`` x ``depth1`` times in a block;
    the top block (with the surface tile on top) is repeated ``lateral2`` x ``lateral2`` times and ``depth2 - 1`` bulk
    blocks are put below it. Total: lateral1*lateral2 cells wide, depth1*depth2 cells deep, surface at z = 0."""
    from photontracer import InstanceGeometry, MeshGeometry
    mesh = lambda m: MeshGeometry(np.asarray(m[0], np.float32), np.asarray(m[1], np.uint32))
    identity = np.eye(3, 4, dtype=np.float32)[None]
    leaf = lambda m: InstanceGeometry([mesh(m)], identity, np.zeros(1, np.uint32), np.full(1, ICE, np.uint32))
    bulk, tile = leaf(cell), leaf(top)

    def transforms(positions):
        t = np.zeros((len(positions), 3, 4), np.float32)
        t[:, [0, 1, 2], [0, 1, 2]] = 1.0
        t[:, :, 3] = positions
        return t

    def instances(subs, positions, types):
        return InstanceGeometry(subs, transforms(np.asarray(positions)), np.asarray(types, np.uint32),
                                np.zeros(len(positions), np.uint32))

    ix, iy = (a.ravel() for a in np.meshgrid(_grid(lateral1, cell_edge_um), _grid(lateral1, cell_edge_um),
                                             indexing="ij"))
    z_join = -surface_um                                      # bottom plane of the top tile = top plane of the cell layers
    layer_z = z_join - (np.arange(depth1) + 0.5) * cell_edge_um
    # block with the surface tile: tile + depth1 cell layers below it (origin at z = 0, the nominal surface)
    lat = np.stack([ix, iy], axis=1)
    pos_top = np.concatenate([np.c_[lat, np.full(len(lat), z_join)]] +
                             [np.c_[lat, np.full(len(lat), z)] for z in layer_z])
    types_top = np.r_[np.zeros(len(lat)), np.ones(len(lat) * depth1)]
    top_block = instances([tile, bulk], pos_top, types_top)
    # block of cell layers only (origin at its top plane)
    pos_bulk = np.concatenate([np.c_[lat, np.full(len(lat), -(k + 0.5) * cell_edge_um)] for k in range(depth1)])
    bulk_block = instances([bulk], pos_bulk, np.zeros(len(pos_bulk)))
    # level 2: blocks side by side, top block at the surface, bulk blocks below
    block_edge = lateral1 * cell_edge_um
    bx, by = (a.ravel() for a in np.meshgrid(_grid(lateral2, block_edge), _grid(lateral2, block_edge), indexing="ij"))
    blat = np.stack([bx, by], axis=1)
    pos = [np.c_[blat, np.zeros(len(blat))]]
    types = [np.zeros(len(blat))]
    for j in range(depth2 - 1):
        z0 = z_join - depth1 * cell_edge_um * (j + 1)
        pos.append(np.c_[blat, np.full(len(blat), z0)])
        types.append(np.ones(len(blat)))
    return instances([top_block, bulk_block], np.concatenate(pos), np.concatenate(types))


def trace_mesh(verts, faces, wavelength_nm, ri_dir, **kwargs):
    """``trace_geometry`` of one closed mesh."""
    return trace_geometry(closed_geometry(verts, faces), wavelength_nm, ri_dir, **kwargs)


def trace_geometry(geometry, wavelength_nm, ri_dir, rays=200_000, beam_radius_um=1000.0, incidence_deg=0.0, seed=1,
                   backend="optix", gpu_id=0, ri_source="warren"):
    """Returns a dict of arrays over ``wavelength_nm``.

    albedo: rays that leave upwards / all rays that interacted at least once. Rays that fly through the periodic
    structure without touching ice (straight channels through the repeated cell; ``free``) are left out: an infinite
    medium has no such channel. Others: ``up`` (left upwards), ``escaped`` (left in any direction), ``absorbed``,
    ``unfinished`` (step limit / error), ``warnings`` (rays with geometry inconsistencies) and the binomial error of
    albedo. Rays that leave through the sides or the bottom do not count as reflected."""
    import time
    from photontracer import Material, MaterialType, OutputType
    t_start = time.perf_counter()
    cfg = TraceConfig(ri_source=ri_source, ri_bandwidth_um=None, incidence_deg=incidence_deg, number_of_rays=rays,
                      beam_radius_um=beam_radius_um, scale_percent=100.0, seed=seed, backend=backend)
    sim = make_simulation(cfg, geometry, gpu_id, 0)
    set_beam(sim, cfg)
    sim.outputs = [OutputType.RAY_STATE, OutputType.LAST_DIRECTION, OutputType.SCATTERING_COUNT,
                   OutputType.NUMBER_OF_WARNINGS]
    grid = np.asarray(wavelength_nm, dtype=int)
    index = ri.load(ri_source, ri_dir).resample(grid, None)
    out = {k: np.zeros(len(grid)) for k in ("up", "escaped", "absorbed", "unfinished", "free", "warnings")}
    out["run_s"] = np.zeros(len(grid))          # wall seconds of sim.run() per wavelength
    out["setup_s"] = time.perf_counter() - t_start   # scene upload / acceleration structure
    for i, (n, k) in enumerate(zip(index.n, index.k)):
        sim.materials = [Material(MaterialType.REFRACTIVE, 1 + 0j), Material(MaterialType.REFRACTIVE, complex(n, k))]
        sim.wavelength_um = grid[i] / 1000.0
        t_run = time.perf_counter()
        sim.run()
        out["run_s"][i] = time.perf_counter() - t_run
        state = sim.get_output_buffer(OutputType.RAY_STATE)
        escaped = state == 0
        direction = sim.get_output_buffer(OutputType.LAST_DIRECTION)[escaped]
        out["escaped"][i], out["absorbed"][i] = escaped.sum(), (state == 1).sum()
        out["up"][i] = (direction[:, 2] > 0).sum()
        touched = sim.get_output_buffer(OutputType.SCATTERING_COUNT) > 0
        out["free"][i] = (escaped & ~touched).sum()
        out["warnings"][i] = (sim.get_output_buffer(OutputType.NUMBER_OF_WARNINGS) > 0).sum()
        out["unfinished"][i] = len(state) - out["escaped"][i] - out["absorbed"][i]
    finished = out["escaped"] + out["absorbed"] - out["free"]
    out["albedo"] = out["up"] / finished
    out["albedo_error"] = np.sqrt(out["albedo"] * (1 - out["albedo"]) / finished)
    out["wavelength_nm"] = grid
    return out

"""
Face transmissibility cache for black-oil simulation.

Compute upwinding, gravity potential, and phase/total transmissibility for every interior connection.
"""

import typing

import numba
import numpy as np
import numpy.typing as npt

from bores.blackoil.caches.mobility import MobilityCache
from bores.blackoil.caches.pvt import PVTCache
from bores.constants import c, get_conversion_factors
from bores.precision import get_dtype
from bores.reservoir.model import Reservoir
from bores.types import CellArray, IntCellArray, UnitSystem

__all__ = ["TransmissibilityCache", "compute_transmissibility_cache"]


class TransmissibilityCache(typing.NamedTuple):
    """
    Per-interior-connection upwinding, gravity, and transmissibility.

    Every field is shape `(n_interior_faces,)`, aligned with
    `grid.interior_face_indices`/`reservoir.transmissibilities.interior`.

    That is, Position `p` here is the same connection as position `p` there, not a
    global face index.

    `*_upstream_cell` is the winning cell's *global* cell index
    (whichever of the connection's owner/neighbour has higher gravity-adjusted
    potential for that phase).

    `dT*_dP`/ `dT*_dsw` etc. give the transmissibility's sensitivity to the
    *upstream* cell's primary variables only. This is because by construction
    of upstream weighting, the downstream cell's mobility never enters a TPFA
    phase transmissibility at all, so its derivative is exactly zero and isn't
    stored. Which cell "upstream" refers to for a given connection is `*_upstream_cell`.
    """

    water_upstream_cell: IntCellArray
    """Global cell index of the upstream cell for the water phase."""
    oil_upstream_cell: IntCellArray
    """Global cell index of the upstream cell for the oil phase."""
    gas_upstream_cell: IntCellArray
    """Global cell index of the upstream cell for the gas phase."""

    water_potential_difference: CellArray
    """Owner-minus-neighbour gravity-adjusted potential difference, water phase."""
    oil_potential_difference: CellArray
    """Owner-minus-neighbour gravity-adjusted potential difference, oil phase."""
    gas_potential_difference: CellArray
    """Owner-minus-neighbour gravity-adjusted potential difference, gas phase."""

    water_face_density: CellArray
    """Upstream-cell density for the water phase."""
    oil_face_density: CellArray
    """Upstream-cell density for the oil phase."""
    gas_face_density: CellArray
    """Upstream-cell density for the gas phase."""

    water_face_viscosity: CellArray
    """Upstream-cell viscosity for the water phase."""
    oil_face_viscosity: CellArray
    """Upstream-cell viscosity for the oil phase."""
    gas_face_viscosity: CellArray
    """Upstream-cell viscosity for the gas phase."""

    water_transmissibility: CellArray
    """`geometric_transmissibility * water_mobility[water_upstream_cell]`."""
    oil_transmissibility: CellArray
    """`geometric_transmissibility * oil_mobility[oil_upstream_cell]`."""
    gas_transmissibility: CellArray
    """`geometric_transmissibility * gas_mobility[gas_upstream_cell]`."""
    total_transmissibility: CellArray
    """Sum of the three phase transmissibilities."""

    dTw_dP: CellArray
    """Pressure derivative of the water transmissibility."""
    dTo_dP: CellArray
    """Pressure derivative of the oil transmissibility."""
    dTg_dP: CellArray
    """Pressure derivative of the gas transmissibility."""

    dTw_dsw: CellArray
    """Water-transmissibility derivative w.r.t. water saturation."""
    dTo_dsw: CellArray
    """Oil-transmissibility derivative w.r.t. water saturation."""
    dTg_dsw: CellArray
    """Gas-transmissibility derivative w.r.t. water saturation."""
    dTw_dso: CellArray
    """Water-transmissibility derivative w.r.t. oil saturation."""
    dTo_dso: CellArray
    """Oil-transmissibility derivative w.r.t. oil saturation."""
    dTg_dso: CellArray
    """Gas-transmissibility derivative w.r.t. oil saturation."""
    dTw_dsg: CellArray
    """Water-transmissibility derivative w.r.t. gas saturation."""
    dTo_dsg: CellArray
    """Oil-transmissibility derivative w.r.t. gas saturation."""
    dTg_dsg: CellArray
    """Gas-transmissibility derivative w.r.t. gas saturation."""


CACHE_INT_FIELDS = ("water_upstream_cell", "oil_upstream_cell", "gas_upstream_cell")
CACHE_FLOAT_FIELDS = tuple(
    name for name in TransmissibilityCache._fields if name not in CACHE_INT_FIELDS
)


def make_new_cache(n_interior_faces: int, dtype: npt.DTypeLike) -> TransmissibilityCache:
    float_fields = {name: np.zeros(n_interior_faces, dtype=dtype) for name in CACHE_FLOAT_FIELDS}
    int_fields = {name: np.full(n_interior_faces, -1, dtype=np.int32) for name in CACHE_INT_FIELDS}
    return TransmissibilityCache(**float_fields, **int_fields)  # type: ignore[arg-type]


def get_gravity_acceleration(unit_system: UnitSystem) -> float:
    """
    Get gravity acceleration in `unit_system`'s length/time units.

    :param unit_system: Target unit system.
    :return: Gravity acceleration, e.g. ft/day^2 for FIELD.
    """
    if unit_system == UnitSystem.FIELD:
        return c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_DAY_SQUARE
    if unit_system == UnitSystem.SI:
        return c.ACCELERATION_DUE_TO_GRAVITY_METER_PER_SECONDS_SQUARE
    base = c.ACCELERATION_DUE_TO_GRAVITY_METER_PER_SECONDS_SQUARE
    factors = get_conversion_factors(UnitSystem.SI, unit_system)
    return base * factors["length"] / (factors["time"] ** 2)


@numba.njit(cache=True, parallel=True)
def update_transmissibility_cache(
    owner_indices: IntCellArray,
    neighbour_indices: IntCellArray,
    geometric_transmissibility: CellArray,
    cell_depth: CellArray,
    gravity_acceleration: float,
    pvt_cache: PVTCache,
    mobility_cache: MobilityCache,
    water_pressure: CellArray,
    oil_pressure: CellArray,
    gas_pressure: CellArray,
    out: TransmissibilityCache,
) -> None:
    """
    Fill `out` in place, one interior connection at a time.

    :param owner_indices: Shape `(n_interior_faces,)` - `grid.
        face_cell_indices[grid.interior_face_indices, 0]`, gathered by the
        caller.
    :param neighbour_indices: Same, `face_cell_indices[..., 1]`.
    :param geometric_transmissibility: `reservoir.transmissibilities.interior`.
    :param cell_depth: Shape `(n_cells,)` - `grid.cell_centroids[:, 2]`
        (positive-down, this codebase's convention throughout).
    :param gravity_acceleration: From `get_gravity_acceleration`.
    :param pvt_cache: Current `PVTCache`.
    :param mobility_cache: Current `MobilityCache`.
    :param water_pressure: Shape `(n_cells,)` - `oil_pressure - oil_water_capillary_pressure`.
    :param oil_pressure: Shape `(n_cells,)` reference/oil pressure.
    :param gas_pressure: Shape `(n_cells,)` - `oil_pressure + gas_oil_capillary_pressure`.
    :param out: Pre-allocated `TransmissibilityCache`, written in place.
    """
    water_density = pvt_cache.water_density
    oil_density = pvt_cache.oil_density
    gas_density = pvt_cache.gas_density
    water_viscosity = pvt_cache.water_viscosity
    oil_viscosity = pvt_cache.oil_viscosity
    gas_viscosity = pvt_cache.gas_viscosity

    water_mobility = mobility_cache.water_mobility
    oil_mobility = mobility_cache.oil_mobility
    gas_mobility = mobility_cache.gas_mobility
    dʎw_dP = mobility_cache.dʎw_dP
    dʎo_dP = mobility_cache.dʎo_dP
    dʎg_dP = mobility_cache.dʎg_dP
    dʎw_dsw = mobility_cache.dʎw_dsw
    dʎo_dsw = mobility_cache.dʎo_dsw
    dʎg_dsw = mobility_cache.dʎg_dsw
    dʎw_dso = mobility_cache.dʎw_dso
    dʎo_dso = mobility_cache.dʎo_dso
    dʎg_dso = mobility_cache.dʎg_dso
    dʎw_dsg = mobility_cache.dʎw_dsg
    dʎo_dsg = mobility_cache.dʎo_dsg
    dʎg_dsg = mobility_cache.dʎg_dsg

    out_water_upstream_cell = out.water_upstream_cell
    out_oil_upstream_cell = out.oil_upstream_cell
    out_gas_upstream_cell = out.gas_upstream_cell
    out_water_potential_difference = out.water_potential_difference
    out_oil_potential_difference = out.oil_potential_difference
    out_gas_potential_difference = out.gas_potential_difference
    out_water_face_density = out.water_face_density
    out_oil_face_density = out.oil_face_density
    out_gas_face_density = out.gas_face_density
    out_water_face_viscosity = out.water_face_viscosity
    out_oil_face_viscosity = out.oil_face_viscosity
    out_gas_face_viscosity = out.gas_face_viscosity
    out_water_transmissibility = out.water_transmissibility
    out_oil_transmissibility = out.oil_transmissibility
    out_gas_transmissibility = out.gas_transmissibility
    out_total_transmissibility = out.total_transmissibility
    out_dTw_dP = out.dTw_dP
    out_dTo_dP = out.dTo_dP
    out_dTg_dP = out.dTg_dP
    out_dTw_dsw = out.dTw_dsw
    out_dTo_dsw = out.dTo_dsw
    out_dTg_dsw = out.dTg_dsw
    out_dTw_dso = out.dTw_dso
    out_dTo_dso = out.dTo_dso
    out_dTg_dso = out.dTg_dso
    out_dTw_dsg = out.dTw_dsg
    out_dTo_dsg = out.dTo_dsg
    out_dTg_dsg = out.dTg_dsg

    n_connections = owner_indices.shape[0]
    for i in numba.prange(n_connections):
        owner = owner_indices[i]
        neighbour = neighbour_indices[i]
        geometric_t = geometric_transmissibility[i]
        depth_difference = cell_depth[owner] - cell_depth[neighbour]

        # Water
        average_water_density = 0.5 * (water_density[owner] + water_density[neighbour])
        water_potential_difference = (
            water_pressure[owner] - water_pressure[neighbour]
        ) - average_water_density * gravity_acceleration * depth_difference
        water_upstream = owner if water_potential_difference >= 0.0 else neighbour
        out_water_upstream_cell[i] = water_upstream
        out_water_potential_difference[i] = water_potential_difference
        out_water_face_density[i] = water_density[water_upstream]
        out_water_face_viscosity[i] = water_viscosity[water_upstream]
        water_transmissibility = geometric_t * water_mobility[water_upstream]
        out_water_transmissibility[i] = water_transmissibility
        out_dTw_dP[i] = geometric_t * dʎw_dP[water_upstream]
        out_dTw_dsw[i] = geometric_t * dʎw_dsw[water_upstream]
        out_dTw_dso[i] = geometric_t * dʎw_dso[water_upstream]
        out_dTw_dsg[i] = geometric_t * dʎw_dsg[water_upstream]

        # Oil
        average_oil_density = 0.5 * (oil_density[owner] + oil_density[neighbour])
        oil_potential_difference = (
            oil_pressure[owner] - oil_pressure[neighbour]
        ) - average_oil_density * gravity_acceleration * depth_difference
        oil_upstream = owner if oil_potential_difference >= 0.0 else neighbour
        out_oil_upstream_cell[i] = oil_upstream
        out_oil_potential_difference[i] = oil_potential_difference
        out_oil_face_density[i] = oil_density[oil_upstream]
        out_oil_face_viscosity[i] = oil_viscosity[oil_upstream]
        oil_transmissibility = geometric_t * oil_mobility[oil_upstream]
        out_oil_transmissibility[i] = oil_transmissibility
        out_dTo_dP[i] = geometric_t * dʎo_dP[oil_upstream]
        out_dTo_dsw[i] = geometric_t * dʎo_dsw[oil_upstream]
        out_dTo_dso[i] = geometric_t * dʎo_dso[oil_upstream]
        out_dTo_dsg[i] = geometric_t * dʎo_dsg[oil_upstream]

        # Gas
        average_gas_density = 0.5 * (gas_density[owner] + gas_density[neighbour])
        gas_potential_difference = (
            gas_pressure[owner] - gas_pressure[neighbour]
        ) - average_gas_density * gravity_acceleration * depth_difference
        gas_upstream = owner if gas_potential_difference >= 0.0 else neighbour
        out_gas_upstream_cell[i] = gas_upstream
        out_gas_potential_difference[i] = gas_potential_difference
        out_gas_face_density[i] = gas_density[gas_upstream]
        out_gas_face_viscosity[i] = gas_viscosity[gas_upstream]
        gas_transmissibility = geometric_t * gas_mobility[gas_upstream]
        out_gas_transmissibility[i] = gas_transmissibility
        out_dTg_dP[i] = geometric_t * dʎg_dP[gas_upstream]
        out_dTg_dsw[i] = geometric_t * dʎg_dsw[gas_upstream]
        out_dTg_dso[i] = geometric_t * dʎg_dso[gas_upstream]
        out_dTg_dsg[i] = geometric_t * dʎg_dsg[gas_upstream]

        out_total_transmissibility[i] = (
            water_transmissibility + oil_transmissibility + gas_transmissibility
        )


def compute_transmissibility_cache(
    reservoir: Reservoir,
    pvt_cache: PVTCache,
    mobility_cache: MobilityCache,
    oil_pressure: CellArray,
    oil_water_capillary_pressure: CellArray,
    gas_oil_capillary_pressure: CellArray,
    out: TransmissibilityCache | None = None,
    dtype: npt.DTypeLike = None,
) -> TransmissibilityCache:
    """
    Build (or refresh, in place) a `TransmissibilityCache` for every interior
    connection in `reservoir.grid`.

    Water/gas pressures are derived here from `oil_pressure` and the
    capillary pressures already sitting in a `SatFuncCache`
    (`Pw = Po - Pcow`, `Pg = Po + Pcgo`) - pass
    `saturation_function_cache.oil_water_capillary_pressure`/
    `.gas_oil_capillary_pressure` directly rather than recomputing them.

    **Reuse**: pass a previous call's returned `TransmissibilityCache` as
    `out` to overwrite its arrays in place; the return value is `out` itself
    in that case.

    :param reservoir: Provides `grid` (face/cell connectivity, cell depths)
        and `transmissibilities.interior` (geometric TPFA transmissibility -
        not recomputed here).
    :param pvt_cache: Current `PVTCache`, same cell count/ordering as `reservoir.grid`.
    :param mobility_cache: Current `MobilityCache`, same cell count/ordering.
    :param oil_pressure: Shape `(n_cells,)` reference/oil pressure - the same
        array passed as `pressure` to `caches.pvt.compute_pvt_cache`.
    :param oil_water_capillary_pressure: Shape `(n_cells,)` - `Pcow`, e.g.
        `SatFuncCache.oil_water_capillary_pressure`.
    :param gas_oil_capillary_pressure: Shape `(n_cells,)` - `Pcgo`, e.g.
        `SatFuncCache.gas_oil_capillary_pressure`.
    :param out: Previous `TransmissibilityCache` to overwrite in place, or
        `None` to allocate a new one.
    :return: The populated `TransmissibilityCache` - `out` itself if given,
        otherwise a newly allocated one. Always returned, never `None`.
    :raises ValueError: If `out` is given but sized for a different
        interior-connection count than `reservoir.grid`.
    """
    grid = reservoir.grid
    n_interior_faces = len(grid.interior_face_indices)

    if out is not None:
        if len(out.total_transmissibility) != n_interior_faces:
            raise ValueError(
                f"`out` was allocated for {len(out.total_transmissibility)} "
                f"interior connections; `reservoir.grid` has {n_interior_faces}."
            )
        cache = out
    else:
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        cache = make_new_cache(n_interior_faces, dtype=dtype)

    owner_indices = grid.face_cell_indices[grid.interior_face_indices, 0]
    neighbour_indices = grid.face_cell_indices[grid.interior_face_indices, 1]
    assert grid.cell_centroids is not None
    cell_depth = typing.cast(CellArray, grid.cell_centroids[:, 2])
    gravity_acceleration = get_gravity_acceleration(reservoir.unit_system)

    water_pressure = typing.cast(CellArray, oil_pressure - oil_water_capillary_pressure)
    gas_pressure = typing.cast(CellArray, oil_pressure + gas_oil_capillary_pressure)

    update_transmissibility_cache(
        owner_indices=owner_indices,
        neighbour_indices=neighbour_indices,
        geometric_transmissibility=reservoir.transmissibilities.interior,
        cell_depth=cell_depth,
        gravity_acceleration=gravity_acceleration,
        pvt_cache=pvt_cache,
        mobility_cache=mobility_cache,
        water_pressure=water_pressure,
        oil_pressure=oil_pressure,
        gas_pressure=gas_pressure,
        out=cache,
    )
    return cache

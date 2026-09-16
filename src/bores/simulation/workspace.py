"""Reusable workspace for one simulation run."""

import typing

import numpy as np
import numpy.typing as npt

from bores.blackoil.caches.physics import PhysicsCache, compute_physics_cache
from bores.blackoil.caches.transmissibility import (
    TransmissibilityCache,
    compute_transmissibility_cache,
)
from bores.blackoil.fluids.model import BlackOil
from bores.precision import get_dtype
from bores.reservoir.model import Reservoir
from bores.reservoir.regions import Regions
from bores.reservoir.state import ReservoirState
from bores.types import Integer, NumberArray, OneDimension
from bores.wells.resolution.compile import WellsWorkspace, build_wells_workspace

__all__ = [
    "ReservoirWorkspace",
    "SimulationWorkspace",
    "build_reservoir_workspace",
    "build_simulation_workspace",
]


class ReservoirWorkspace(typing.NamedTuple):
    """
    Per-cell reservoir primary unknowns, one row per cell, for the whole run.
    The reservoir state (`ReservoirState`) is decompiled from whatever this holds at a
    reported timestep, mirroring `WellsWorkspace`/`WellsStates`.
    """

    pressure: NumberArray[OneDimension]
    """Shape `(n_cells,)`. Oil-phase reference pressure."""

    oil_saturation: NumberArray[OneDimension]
    """Shape `(n_cells,)`."""

    water_saturation: NumberArray[OneDimension]
    """Shape `(n_cells,)`."""

    gas_saturation: NumberArray[OneDimension]
    """Shape `(n_cells,)`."""

    solution_gor: NumberArray[OneDimension]
    """Shape `(n_cells,)`. Rs."""

    oil_bubble_point_pressure: NumberArray[OneDimension]
    """Shape `(n_cells,)`."""

    vaporized_oil_to_gas_ratio: NumberArray[OneDimension]
    """Shape `(n_cells,)`. Rv."""

    gas_dew_point_pressure: NumberArray[OneDimension]
    """Shape `(n_cells,)`."""

    gas_solubility_in_water: NumberArray[OneDimension]
    """Shape `(n_cells,)`. Rsw."""

    water_bubble_point_pressure: NumberArray[OneDimension]
    """Shape `(n_cells,)`."""


def build_reservoir_workspace(
    *, state: ReservoirState, dtype: npt.DTypeLike = None
) -> ReservoirWorkspace:
    """
    Builds a `ReservoirWorkspace` from a reservoir state's own primary unknowns.

    :param state: The reservoir state to seed the workspace from.
    :param n_cells: Number of grid cells.
    :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
    :returns: `ReservoirWorkspace` seeded from `state`.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    return ReservoirWorkspace(
        pressure=state.pressure.astype(dtype, copy=True),
        oil_saturation=state.oil_saturation.astype(dtype, copy=True),
        water_saturation=state.water_saturation.astype(dtype, copy=True),
        gas_saturation=state.gas_saturation.astype(dtype, copy=True),
        solution_gor=state.solution_gor.astype(dtype, copy=True),
        oil_bubble_point_pressure=state.oil_bubble_point_pressure.astype(dtype, copy=True),
        vaporized_oil_to_gas_ratio=state.vaporized_oil_to_gas_ratio.astype(dtype, copy=True),
        gas_dew_point_pressure=state.gas_dew_point_pressure.astype(dtype, copy=True),
        gas_solubility_in_water=state.gas_solubility_in_water.astype(dtype, copy=True),
        water_bubble_point_pressure=state.water_bubble_point_pressure.astype(dtype, copy=True),
    )


class SimulationWorkspace(typing.NamedTuple):
    """
    Reusable scratch buffer and ocomputed cache for one simulation run.

    Allocated once per `SimulationCase`, mutated in place at every time step.
    """

    reservoir: ReservoirWorkspace
    """This run's reservoir primary unknowns."""

    physics: PhysicsCache
    """
    PVT/satfunc/mobility at the current cell state. Refreshed in place
    (`compute_physics_cache(..., out=...)`) every timestep.
    """

    transmissibilities: TransmissibilityCache
    """
    Per-interior-connection upwinding, gravity, and transmissibility
    at the current cell state. 
    
    Refreshed in place (`compute_transmissibility_cache(..., out=...)`) every timestep.
    """

    wells: WellsWorkspace
    """This run's well control-resolution results."""


def build_simulation_workspace(
    *,
    reservoir: Reservoir,
    regions: Regions,
    fluid: BlackOil,
    initial_state: ReservoirState,
    n_wells: Integer,
    n_connections: Integer,
    dtype: npt.DTypeLike = None,
) -> SimulationWorkspace:
    """
    Builds a `SimulationWorkspace` from a case's own reservoir, fluid,
    and initial state. Call once at the start of a run.

    :param reservoir: The run's reservoir (grid + rock + regions).
    :param regions: The run's `PVTNUM`/`SATNUM` region arrays.
    :param fluid: The run's fluid model (PVT + saturation functions).
    :param initial_state: The reservoir's state at the start of the run.
    :param n_wells: Number of wells.
    :param n_connections: Total active connections across every well.
    :param resolver_spec: Well-control resolution tunables to compile.
        `ControlResolverSpec`'s own defaults if not given.
    :param dtype: Output array dtype for every buffer. `bores.precision.get_dtype()` if not given.
    :returns: The assembled `SimulationWorkspace`.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    n_cells = reservoir.grid.n_cells
    pvt_region = (
        regions.pvt_region if regions.pvt_region is not None else np.ones(n_cells, dtype=np.int32)
    )
    saturation_region = (
        regions.saturation_region
        if regions.saturation_region is not None
        else np.ones(n_cells, dtype=np.int32)
    )
    physics = compute_physics_cache(
        pressure=initial_state.pressure,
        temperature=initial_state.temperature,
        solution_gas_oil_ratio=initial_state.solution_gor,
        water_saturation=initial_state.water_saturation,
        oil_saturation=initial_state.oil_saturation,
        gas_saturation=initial_state.gas_saturation,
        pvt_region=pvt_region,
        saturation_region=saturation_region,
        fluid=fluid,
        dtype=dtype,
    )
    transmissibilities = compute_transmissibility_cache(
        reservoir=reservoir,
        pvt_cache=physics.pvt,
        mobility_cache=physics.mobility,
        oil_pressure=initial_state.pressure,
        oil_water_capillary_pressure=physics.satfunc.oil_water_capillary_pressure,
        gas_oil_capillary_pressure=physics.satfunc.gas_oil_capillary_pressure,
        dtype=dtype,
    )
    return SimulationWorkspace(
        physics=physics,
        transmissibilities=transmissibilities,
        wells=build_wells_workspace(n_wells=n_wells, n_connections=n_connections, dtype=dtype),
        reservoir=build_reservoir_workspace(state=initial_state, dtype=dtype),
    )

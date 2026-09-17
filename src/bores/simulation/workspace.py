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
from bores.reservoir.state import Hysteresis, ReservoirState
from bores.reservoir.workspace import (
    HysteresisWorkspace,
    ReservoirWorkspace,
)
from bores.simulation.spec import RunSpec
from bores.types import CellArray, Integer
from bores.wells.workspace import WellsWorkspace, build_wells_workspace

__all__ = [
    "SimulationWorkspace",
    "build_simulation_workspace",
]


class SimulationWorkspace(typing.NamedTuple):
    """
    Reusable scratch buffer and ocomputed cache for one simulation run.

    Allocated once per `SimulationCase`, mutated in place at every time step.
    """

    reservoir: ReservoirWorkspace
    """This run's reservoir primary unknowns. Refreshed in place every timestep."""

    physics: PhysicsCache
    """
    PVT/satfunc/mobility at the current cell state. Refreshed in place 
    every timestep.
    """

    transmissibilities: TransmissibilityCache
    """
    Per-interior-connection upwinding, gravity, and transmissibility
    at the current cell state. 
    
    Refreshed in place every timestep.
    """

    wells: WellsWorkspace
    """This run's well control-resolution workspace. Refreshed in place every timestep."""

    hysteresis: HysteresisWorkspace | None
    """Optional mutable hysteresis history for the run."""

    salinity: CellArray | None
    """Optional cell-wise salinity field for salinity-dependent calculations."""


def build_simulation_workspace(
    *,
    reservoir: Reservoir,
    regions: Regions,
    fluid: BlackOil,
    initial_state: ReservoirState,
    n_wells: Integer,
    n_connections: Integer,
    runspec: RunSpec,
    hysteresis: Hysteresis | HysteresisWorkspace | None = None,
    salinity: CellArray | None = None,
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
    :param runspec: Simulation run settings controlling enabled physics terms such as gravity and capillary effects.
    :param dtype: Output array dtype for every buffer. `bores.precision.get_dtype()` if not given.
    :returns: The assembled `SimulationWorkspace`.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    if isinstance(hysteresis, Hysteresis):
        hysteresis = HysteresisWorkspace.from_state(hysteresis, dtype=dtype)

    n_cells = reservoir.grid.n_cells
    pvt_region = (
        regions.pvt_region if regions.pvt_region is not None else np.ones(n_cells, dtype=np.int32)
    )
    saturation_region = (
        regions.saturation_region
        if regions.saturation_region is not None
        else np.ones(n_cells, dtype=np.int32)
    )
    rock = reservoir.rock
    physics_cache = compute_physics_cache(
        pressure=initial_state.pressure,
        temperature=initial_state.temperature,
        solution_gas_oil_ratio=initial_state.solution_gor,
        water_saturation=initial_state.water_saturation,
        oil_saturation=initial_state.oil_saturation,
        gas_saturation=initial_state.gas_saturation,
        pvt_region=pvt_region,
        saturation_region=saturation_region,
        fluid=fluid,
        hysteresis=hysteresis,
        salinity=salinity,
        irreducible_water_saturation=rock.irreducible_water_saturation,
        residual_gas_saturation=rock.residual_gas_saturation,
        residual_oil_saturation_water=rock.residual_oil_saturation_water,
        residual_oil_saturation_gas=rock.residual_oil_saturation_gas,
        dtype=dtype,
    )
    transmissibility_cache = compute_transmissibility_cache(
        reservoir=reservoir,
        pvt_cache=physics_cache.pvt,
        mobility_cache=physics_cache.mobility,
        oil_pressure=initial_state.pressure,
        oil_water_capillary_pressure=physics_cache.satfunc.oil_water_capillary_pressure,
        gas_oil_capillary_pressure=physics_cache.satfunc.gas_oil_capillary_pressure,
        gravity_enabled=runspec.gravity_enabled,
        capillary_effects_enabled=runspec.capillary_effects_enabled,
        dtype=dtype,
    )
    reservoir_workspace = ReservoirWorkspace.from_state(initial_state, dtype=dtype)
    wells_workspace = build_wells_workspace(
        n_wells=n_wells, n_connections=n_connections, dtype=dtype
    )
    return SimulationWorkspace(
        physics=physics_cache,
        transmissibilities=transmissibility_cache,
        wells=wells_workspace,
        reservoir=reservoir_workspace,
        hysteresis=hysteresis,
        salinity=salinity,
    )

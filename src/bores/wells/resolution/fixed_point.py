"""
The fixed-point iteration between a well's connection pressures and its connection rates, compiled.

Each solver is a single compiled call. They share one structure: set an initial pressure profile
from zero flow, then repeat: update the connection rates from the profile, update the profile from
the rates, and stop when the total rate stops changing. They differ only in how the profile is
updated, which is why each refers directly to its own kernel. Referring to it directly keeps the
solver cacheable between runs.
"""

import math
import typing

import numba
import numpy as np

from bores.types import Number
from bores.wells.hydraulics.beggs_and_brill import (
    BeggsAndBrillWellbore,
)
from bores.wells.hydraulics.beggs_and_brill import (
    walk_perforation_pressures as walk_beggs_and_brill,
)
from bores.wells.hydraulics.gray import GrayWellbore
from bores.wells.hydraulics.gray import walk_perforation_pressures as walk_gray
from bores.wells.hydraulics.hagedorn_brown import HagedornBrownWellbore
from bores.wells.hydraulics.hagedorn_brown import (
    walk_perforation_pressures as walk_hagedorn_brown,
)
from bores.wells.hydraulics.homogeneous import HomogeneousWellbore
from bores.wells.hydraulics.homogeneous import (
    walk_perforation_pressures as walk_homogeneous,
)
from bores.wells.hydraulics.woldesemayat_ghajar import WoldesemayatGhajarWellbore
from bores.wells.hydraulics.woldesemayat_ghajar import (
    walk_perforation_pressures as walk_woldesemayat_ghajar,
)
from bores.wells.resolution.hydrostatic import walk_hydrostatic_pressures
from bores.wells.workspace import (
    PerforationWorkspace,
    accumulate_phase_rates,
    correct_well_indices_for_non_darcy,
)

__all__ = [
    "FIXED_POINT_SOLVERS",
    "FixedPointResult",
    "solve_beggs_and_brill",
    "solve_gray",
    "solve_hagedorn_brown",
    "solve_homogeneous",
    "solve_hydrostatic",
    "solve_uniform",
    "solve_woldesemayat_ghajar",
]


class FixedPointResult(typing.NamedTuple):
    """What a fixed-point solve ended with."""

    converged: bool
    """Whether the total rate stopped changing within the iteration limit."""

    previous_total_rate: Number
    """Total rate of the iteration before the last one."""

    total_rate: Number
    """Total rate of the last iteration."""

    oil_rate: Number
    """Oil rate at reservoir conditions, zero if oil isn't one of the phases counted."""

    water_rate: Number
    """Water rate at reservoir conditions, zero if water isn't one of the phases counted."""

    gas_rate: Number
    """Gas rate at reservoir conditions, zero if gas isn't one of the phases counted."""

    surface_oil_rate: Number
    """Oil rate at surface conditions, zero if oil isn't one of the phases counted."""

    surface_water_rate: Number
    """Water rate at surface conditions, zero if water isn't one of the phases counted."""

    surface_gas_rate: Number
    """Gas rate at surface conditions, zero if gas isn't one of the phases counted."""


@numba.njit(cache=True)
def accumulate_rates(
    workspace: PerforationWorkspace,
    relevant_oil: bool,
    relevant_water: bool,
    relevant_gas: bool,
    is_injector: bool,
    radial_flow_constant: Number,
) -> tuple[Number, Number, Number, Number, Number, Number]:
    """
    Updates a well's connection rates from the pressure profile now in its workspace.

    A producer's connection rates always include every phase, since all of them flow. The returned
    totals are only those of the phases counted toward the controlled quantity.

    :param workspace: The well's workspace. Its well indices and connection rates are updated.
    :param relevant_oil: Whether oil counts toward the controlled quantity.
    :param relevant_water: Whether water counts toward the controlled quantity.
    :param relevant_gas: Whether gas counts toward the controlled quantity.
    :param is_injector: Whether the well is an injector.
    :param radial_flow_constant: The constant in the radial flow equation, in the well's unit
        system.
    :returns: The well's oil, water and gas rates at reservoir conditions, then at surface
        conditions, each zero if the phase isn't counted.
    """
    count_oil = relevant_oil or not is_injector
    count_water = relevant_water or not is_injector
    count_gas = relevant_gas or not is_injector
    correct_well_indices_for_non_darcy(
        static_well_indices=workspace.static_well_indices,
        connection_conductivities=workspace.connection_conductivities,
        connection_gas_rates=workspace.connection_gas_rates,
        d_factor=workspace.d_factor,
        out_well_indices=workspace.well_indices,
    )
    (
        oil_rate,
        water_rate,
        gas_rate,
        surface_oil_rate,
        surface_water_rate,
        surface_gas_rate,
    ) = accumulate_phase_rates(
        connection_pressures=workspace.connection_pressures,
        well_indices=workspace.well_indices,
        reservoir_pressures=workspace.reservoir_pressures,
        oil_mobilities=workspace.oil_mobilities,
        water_mobilities=workspace.water_mobilities,
        gas_mobilities=workspace.gas_mobilities,
        oil_formation_volume_factors=workspace.oil_formation_volume_factors,
        water_formation_volume_factors=workspace.water_formation_volume_factors,
        gas_formation_volume_factors=workspace.gas_formation_volume_factors,
        relevant_oil=count_oil,
        relevant_water=count_water,
        relevant_gas=count_gas,
        is_injector=is_injector,
        radial_flow_constant=radial_flow_constant,
        out_connection_oil_rates=workspace.connection_oil_rates,
        out_connection_water_rates=workspace.connection_water_rates,
        out_connection_gas_rates=workspace.connection_gas_rates,
    )
    return (
        oil_rate if relevant_oil else 0.0,
        water_rate if relevant_water else 0.0,
        gas_rate if relevant_gas else 0.0,
        surface_oil_rate if relevant_oil else 0.0,
        surface_water_rate if relevant_water else 0.0,
        surface_gas_rate if relevant_gas else 0.0,
    )


@numba.njit(cache=True)
def has_settled(new_total_rate: Number, total_rate: Number, tolerance: Number) -> bool:
    """
    Whether a total rate has stopped changing.

    :param new_total_rate: The total rate of this iteration.
    :param total_rate: The total rate of the iteration before.
    :param tolerance: Relative tolerance, against the larger of the new rate and one.
    :returns: `True` if the new total rate is finite and the change is within the tolerance.
    """
    return math.isfinite(new_total_rate) and (
        abs(new_total_rate - total_rate) <= tolerance * max(abs(new_total_rate), 1.0)
    )


@numba.njit(cache=True)
def solve_homogeneous(
    model: HomogeneousWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    workspace: PerforationWorkspace,
    relevant_oil: bool,
    relevant_water: bool,
    relevant_gas: bool,
    is_injector: bool,
    hydrostatic_divisor: Number,
    radial_flow_constant: Number,
    max_iterations: int,
    tolerance: Number,
) -> FixedPointResult:
    """
    Solves a well's connection pressures and rates together, with the pressure profile from the
    homogeneous correlation.

    The profile starts from zero flow and is updated from the rates each iteration.

    :param model: The well's `HomogeneousWellbore`.
    :param hydrostatic_divisor: The model's unit system's gravitational factor times its
        hydrostatic area factor, as `get_hydrostatic_divisor` gives it.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param workspace: The well's workspace. Its connection pressures, well indices and connection
        rates are updated.
    :param relevant_oil: Whether oil counts toward the controlled quantity.
    :param relevant_water: Whether water counts toward the controlled quantity.
    :param relevant_gas: Whether gas counts toward the controlled quantity.
    :param is_injector: Whether the well is an injector.
    :param radial_flow_constant: The constant in the radial flow equation, in the well's unit
        system.
    :param max_iterations: The most iterations to run.
    :param tolerance: Relative tolerance on the total rate.
    :returns: What the solve ended with.
    """
    walk_homogeneous(
        model,
        reference_depth,
        workspace.reference_measured_depth,
        reference_pressure,
        np.zeros_like(workspace.connection_oil_rates),
        np.zeros_like(workspace.connection_water_rates),
        np.zeros_like(workspace.connection_gas_rates),
        workspace.representative_depths,
        workspace.representative_measured_depths,
        workspace.inclinations_from_vertical,
        workspace.connection_fluids,
        is_injector,
        hydrostatic_divisor,
        workspace.connection_pressures,
    )

    total_rate = 0.0
    previous_total_rate = 0.0
    oil_rate = 0.0
    water_rate = 0.0
    gas_rate = 0.0
    surface_oil_rate = 0.0
    surface_water_rate = 0.0
    surface_gas_rate = 0.0
    converged = False
    for _ in range(max_iterations):
        (
            oil_rate,
            water_rate,
            gas_rate,
            surface_oil_rate,
            surface_water_rate,
            surface_gas_rate,
        ) = accumulate_rates(
            workspace,
            relevant_oil,
            relevant_water,
            relevant_gas,
            is_injector,
            radial_flow_constant,
        )
        new_total_rate = oil_rate + water_rate + gas_rate
        walk_homogeneous(
            model,
            reference_depth,
            workspace.reference_measured_depth,
            reference_pressure,
            workspace.connection_oil_rates,
            workspace.connection_water_rates,
            workspace.connection_gas_rates,
            workspace.representative_depths,
            workspace.representative_measured_depths,
            workspace.inclinations_from_vertical,
            workspace.connection_fluids,
            is_injector,
            hydrostatic_divisor,
            workspace.connection_pressures,
        )
        if has_settled(new_total_rate, total_rate, tolerance):
            total_rate = new_total_rate
            converged = True
            break
        previous_total_rate = total_rate
        total_rate = new_total_rate

    return FixedPointResult(
        converged,
        previous_total_rate,
        total_rate,
        oil_rate,
        water_rate,
        gas_rate,
        surface_oil_rate,
        surface_water_rate,
        surface_gas_rate,
    )


@numba.njit(cache=True)
def solve_beggs_and_brill(
    model: BeggsAndBrillWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    workspace: PerforationWorkspace,
    relevant_oil: bool,
    relevant_water: bool,
    relevant_gas: bool,
    is_injector: bool,
    hydrostatic_divisor: Number,
    radial_flow_constant: Number,
    max_iterations: int,
    tolerance: Number,
) -> FixedPointResult:
    """
    Solves a well's connection pressures and rates together, with the pressure profile from the
    Beggs and Brill correlation.

    The profile starts from zero flow and is updated from the rates each iteration.

    :param model: The well's `BeggsAndBrillWellbore`.
    :param hydrostatic_divisor: The model's unit system's gravitational factor times its
        hydrostatic area factor, as `get_hydrostatic_divisor` gives it.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param workspace: The well's workspace. Its connection pressures, well indices and connection
        rates are updated.
    :param relevant_oil: Whether oil counts toward the controlled quantity.
    :param relevant_water: Whether water counts toward the controlled quantity.
    :param relevant_gas: Whether gas counts toward the controlled quantity.
    :param is_injector: Whether the well is an injector.
    :param radial_flow_constant: The constant in the radial flow equation, in the well's unit
        system.
    :param max_iterations: The most iterations to run.
    :param tolerance: Relative tolerance on the total rate.
    :returns: What the solve ended with.
    """
    walk_beggs_and_brill(
        model,
        reference_depth,
        workspace.reference_measured_depth,
        reference_pressure,
        np.zeros_like(workspace.connection_oil_rates),
        np.zeros_like(workspace.connection_water_rates),
        np.zeros_like(workspace.connection_gas_rates),
        workspace.representative_depths,
        workspace.representative_measured_depths,
        workspace.inclinations_from_vertical,
        workspace.connection_fluids,
        is_injector,
        hydrostatic_divisor,
        workspace.connection_pressures,
    )

    total_rate = 0.0
    previous_total_rate = 0.0
    oil_rate = 0.0
    water_rate = 0.0
    gas_rate = 0.0
    surface_oil_rate = 0.0
    surface_water_rate = 0.0
    surface_gas_rate = 0.0
    converged = False
    for _ in range(max_iterations):
        (
            oil_rate,
            water_rate,
            gas_rate,
            surface_oil_rate,
            surface_water_rate,
            surface_gas_rate,
        ) = accumulate_rates(
            workspace,
            relevant_oil,
            relevant_water,
            relevant_gas,
            is_injector,
            radial_flow_constant,
        )
        new_total_rate = oil_rate + water_rate + gas_rate
        walk_beggs_and_brill(
            model,
            reference_depth,
            workspace.reference_measured_depth,
            reference_pressure,
            workspace.connection_oil_rates,
            workspace.connection_water_rates,
            workspace.connection_gas_rates,
            workspace.representative_depths,
            workspace.representative_measured_depths,
            workspace.inclinations_from_vertical,
            workspace.connection_fluids,
            is_injector,
            hydrostatic_divisor,
            workspace.connection_pressures,
        )
        if has_settled(new_total_rate, total_rate, tolerance):
            total_rate = new_total_rate
            converged = True
            break
        previous_total_rate = total_rate
        total_rate = new_total_rate

    return FixedPointResult(
        converged,
        previous_total_rate,
        total_rate,
        oil_rate,
        water_rate,
        gas_rate,
        surface_oil_rate,
        surface_water_rate,
        surface_gas_rate,
    )


@numba.njit(cache=True)
def solve_hagedorn_brown(
    model: HagedornBrownWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    workspace: PerforationWorkspace,
    relevant_oil: bool,
    relevant_water: bool,
    relevant_gas: bool,
    is_injector: bool,
    hydrostatic_divisor: Number,
    radial_flow_constant: Number,
    max_iterations: int,
    tolerance: Number,
) -> FixedPointResult:
    """
    Solves a well's connection pressures and rates together, with the pressure profile from the
    Hagedorn and Brown correlation.

    The profile starts from zero flow and is updated from the rates each iteration.

    :param model: The well's `HagedornBrownWellbore`.
    :param hydrostatic_divisor: The model's unit system's gravitational factor times its
        hydrostatic area factor, as `get_hydrostatic_divisor` gives it.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param workspace: The well's workspace. Its connection pressures, well indices and connection
        rates are updated.
    :param relevant_oil: Whether oil counts toward the controlled quantity.
    :param relevant_water: Whether water counts toward the controlled quantity.
    :param relevant_gas: Whether gas counts toward the controlled quantity.
    :param is_injector: Whether the well is an injector.
    :param radial_flow_constant: The constant in the radial flow equation, in the well's unit
        system.
    :param max_iterations: The most iterations to run.
    :param tolerance: Relative tolerance on the total rate.
    :returns: What the solve ended with.
    """
    walk_hagedorn_brown(
        model,
        reference_depth,
        workspace.reference_measured_depth,
        reference_pressure,
        np.zeros_like(workspace.connection_oil_rates),
        np.zeros_like(workspace.connection_water_rates),
        np.zeros_like(workspace.connection_gas_rates),
        workspace.representative_depths,
        workspace.representative_measured_depths,
        workspace.inclinations_from_vertical,
        workspace.connection_fluids,
        is_injector,
        hydrostatic_divisor,
        workspace.connection_pressures,
    )

    total_rate = 0.0
    previous_total_rate = 0.0
    oil_rate = 0.0
    water_rate = 0.0
    gas_rate = 0.0
    surface_oil_rate = 0.0
    surface_water_rate = 0.0
    surface_gas_rate = 0.0
    converged = False
    for _ in range(max_iterations):
        (
            oil_rate,
            water_rate,
            gas_rate,
            surface_oil_rate,
            surface_water_rate,
            surface_gas_rate,
        ) = accumulate_rates(
            workspace,
            relevant_oil,
            relevant_water,
            relevant_gas,
            is_injector,
            radial_flow_constant,
        )
        new_total_rate = oil_rate + water_rate + gas_rate
        walk_hagedorn_brown(
            model,
            reference_depth,
            workspace.reference_measured_depth,
            reference_pressure,
            workspace.connection_oil_rates,
            workspace.connection_water_rates,
            workspace.connection_gas_rates,
            workspace.representative_depths,
            workspace.representative_measured_depths,
            workspace.inclinations_from_vertical,
            workspace.connection_fluids,
            is_injector,
            hydrostatic_divisor,
            workspace.connection_pressures,
        )
        if has_settled(new_total_rate, total_rate, tolerance):
            total_rate = new_total_rate
            converged = True
            break
        previous_total_rate = total_rate
        total_rate = new_total_rate

    return FixedPointResult(
        converged,
        previous_total_rate,
        total_rate,
        oil_rate,
        water_rate,
        gas_rate,
        surface_oil_rate,
        surface_water_rate,
        surface_gas_rate,
    )


@numba.njit(cache=True)
def solve_gray(
    model: GrayWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    workspace: PerforationWorkspace,
    relevant_oil: bool,
    relevant_water: bool,
    relevant_gas: bool,
    is_injector: bool,
    hydrostatic_divisor: Number,
    radial_flow_constant: Number,
    max_iterations: int,
    tolerance: Number,
) -> FixedPointResult:
    """
    Solves a well's connection pressures and rates together, with the pressure profile from the
    Gray correlation.

    The profile starts from zero flow and is updated from the rates each iteration.

    :param model: The well's `GrayWellbore`.
    :param hydrostatic_divisor: The model's unit system's gravitational factor times its
        hydrostatic area factor, as `get_hydrostatic_divisor` gives it.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param workspace: The well's workspace. Its connection pressures, well indices and connection
        rates are updated.
    :param relevant_oil: Whether oil counts toward the controlled quantity.
    :param relevant_water: Whether water counts toward the controlled quantity.
    :param relevant_gas: Whether gas counts toward the controlled quantity.
    :param is_injector: Whether the well is an injector.
    :param radial_flow_constant: The constant in the radial flow equation, in the well's unit
        system.
    :param max_iterations: The most iterations to run.
    :param tolerance: Relative tolerance on the total rate.
    :returns: What the solve ended with.
    """
    walk_gray(
        model,
        reference_depth,
        workspace.reference_measured_depth,
        reference_pressure,
        np.zeros_like(workspace.connection_oil_rates),
        np.zeros_like(workspace.connection_water_rates),
        np.zeros_like(workspace.connection_gas_rates),
        workspace.representative_depths,
        workspace.representative_measured_depths,
        workspace.inclinations_from_vertical,
        workspace.connection_fluids,
        is_injector,
        hydrostatic_divisor,
        workspace.connection_pressures,
    )

    total_rate = 0.0
    previous_total_rate = 0.0
    oil_rate = 0.0
    water_rate = 0.0
    gas_rate = 0.0
    surface_oil_rate = 0.0
    surface_water_rate = 0.0
    surface_gas_rate = 0.0
    converged = False
    for _ in range(max_iterations):
        (
            oil_rate,
            water_rate,
            gas_rate,
            surface_oil_rate,
            surface_water_rate,
            surface_gas_rate,
        ) = accumulate_rates(
            workspace,
            relevant_oil,
            relevant_water,
            relevant_gas,
            is_injector,
            radial_flow_constant,
        )
        new_total_rate = oil_rate + water_rate + gas_rate
        walk_gray(
            model,
            reference_depth,
            workspace.reference_measured_depth,
            reference_pressure,
            workspace.connection_oil_rates,
            workspace.connection_water_rates,
            workspace.connection_gas_rates,
            workspace.representative_depths,
            workspace.representative_measured_depths,
            workspace.inclinations_from_vertical,
            workspace.connection_fluids,
            is_injector,
            hydrostatic_divisor,
            workspace.connection_pressures,
        )
        if has_settled(new_total_rate, total_rate, tolerance):
            total_rate = new_total_rate
            converged = True
            break
        previous_total_rate = total_rate
        total_rate = new_total_rate

    return FixedPointResult(
        converged,
        previous_total_rate,
        total_rate,
        oil_rate,
        water_rate,
        gas_rate,
        surface_oil_rate,
        surface_water_rate,
        surface_gas_rate,
    )


@numba.njit(cache=True)
def solve_woldesemayat_ghajar(
    model: WoldesemayatGhajarWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    workspace: PerforationWorkspace,
    relevant_oil: bool,
    relevant_water: bool,
    relevant_gas: bool,
    is_injector: bool,
    hydrostatic_divisor: Number,
    radial_flow_constant: Number,
    max_iterations: int,
    tolerance: Number,
) -> FixedPointResult:
    """
    Solves a well's connection pressures and rates together, with the pressure profile from the
    Woldesemayat and Ghajar correlation.

    The profile starts from zero flow and is updated from the rates each iteration.

    :param model: The well's `WoldesemayatGhajarWellbore`.
    :param hydrostatic_divisor: The model's unit system's gravitational factor times its
        hydrostatic area factor, as `get_hydrostatic_divisor` gives it.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param workspace: The well's workspace. Its connection pressures, well indices and connection
        rates are updated.
    :param relevant_oil: Whether oil counts toward the controlled quantity.
    :param relevant_water: Whether water counts toward the controlled quantity.
    :param relevant_gas: Whether gas counts toward the controlled quantity.
    :param is_injector: Whether the well is an injector.
    :param radial_flow_constant: The constant in the radial flow equation, in the well's unit
        system.
    :param max_iterations: The most iterations to run.
    :param tolerance: Relative tolerance on the total rate.
    :returns: What the solve ended with.
    """
    walk_woldesemayat_ghajar(
        model,
        reference_depth,
        workspace.reference_measured_depth,
        reference_pressure,
        np.zeros_like(workspace.connection_oil_rates),
        np.zeros_like(workspace.connection_water_rates),
        np.zeros_like(workspace.connection_gas_rates),
        workspace.representative_depths,
        workspace.representative_measured_depths,
        workspace.inclinations_from_vertical,
        workspace.connection_fluids,
        is_injector,
        hydrostatic_divisor,
        workspace.connection_pressures,
    )

    total_rate = 0.0
    previous_total_rate = 0.0
    oil_rate = 0.0
    water_rate = 0.0
    gas_rate = 0.0
    surface_oil_rate = 0.0
    surface_water_rate = 0.0
    surface_gas_rate = 0.0
    converged = False
    for _ in range(max_iterations):
        (
            oil_rate,
            water_rate,
            gas_rate,
            surface_oil_rate,
            surface_water_rate,
            surface_gas_rate,
        ) = accumulate_rates(
            workspace,
            relevant_oil,
            relevant_water,
            relevant_gas,
            is_injector,
            radial_flow_constant,
        )
        new_total_rate = oil_rate + water_rate + gas_rate
        walk_woldesemayat_ghajar(
            model,
            reference_depth,
            workspace.reference_measured_depth,
            reference_pressure,
            workspace.connection_oil_rates,
            workspace.connection_water_rates,
            workspace.connection_gas_rates,
            workspace.representative_depths,
            workspace.representative_measured_depths,
            workspace.inclinations_from_vertical,
            workspace.connection_fluids,
            is_injector,
            hydrostatic_divisor,
            workspace.connection_pressures,
        )
        if has_settled(new_total_rate, total_rate, tolerance):
            total_rate = new_total_rate
            converged = True
            break
        previous_total_rate = total_rate
        total_rate = new_total_rate

    return FixedPointResult(
        converged,
        previous_total_rate,
        total_rate,
        oil_rate,
        water_rate,
        gas_rate,
        surface_oil_rate,
        surface_water_rate,
        surface_gas_rate,
    )


@numba.njit(cache=True)
def solve_hydrostatic(
    reference_depth: Number,
    reference_pressure: Number,
    workspace: PerforationWorkspace,
    relevant_oil: bool,
    relevant_water: bool,
    relevant_gas: bool,
    is_injector: bool,
    pressure_per_density_and_depth: Number,
    radial_flow_constant: Number,
    max_iterations: int,
    tolerance: Number,
) -> FixedPointResult:
    """
    Solves a well's connection pressures and rates together, with the pressure profile from the
    weight of the fluid between connections.

    The profile starts from zero flow and is updated from the rates each iteration.

    :param pressure_per_density_and_depth: As `get_pressure_per_density_and_depth` gives it.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param reference_pressure: Pressure at `reference_depth`.
    :param workspace: The well's workspace. Its connection pressures, well indices and connection
        rates are updated.
    :param relevant_oil: Whether oil counts toward the controlled quantity.
    :param relevant_water: Whether water counts toward the controlled quantity.
    :param relevant_gas: Whether gas counts toward the controlled quantity.
    :param is_injector: Whether the well is an injector.
    :param radial_flow_constant: The constant in the radial flow equation, in the well's unit
        system.
    :param max_iterations: The most iterations to run.
    :param tolerance: Relative tolerance on the total rate.
    :returns: What the solve ended with.
    """
    walk_hydrostatic_pressures(
        reference_depth,
        reference_pressure,
        workspace.representative_depths,
        workspace.connection_fluids,
        np.zeros_like(workspace.connection_oil_rates),
        np.zeros_like(workspace.connection_water_rates),
        np.zeros_like(workspace.connection_gas_rates),
        pressure_per_density_and_depth,
        workspace.connection_pressures,
    )

    total_rate = 0.0
    previous_total_rate = 0.0
    oil_rate = 0.0
    water_rate = 0.0
    gas_rate = 0.0
    surface_oil_rate = 0.0
    surface_water_rate = 0.0
    surface_gas_rate = 0.0
    converged = False
    for _ in range(max_iterations):
        (
            oil_rate,
            water_rate,
            gas_rate,
            surface_oil_rate,
            surface_water_rate,
            surface_gas_rate,
        ) = accumulate_rates(
            workspace,
            relevant_oil,
            relevant_water,
            relevant_gas,
            is_injector,
            radial_flow_constant,
        )
        new_total_rate = oil_rate + water_rate + gas_rate
        walk_hydrostatic_pressures(
            reference_depth,
            reference_pressure,
            workspace.representative_depths,
            workspace.connection_fluids,
            workspace.connection_oil_rates,
            workspace.connection_water_rates,
            workspace.connection_gas_rates,
            pressure_per_density_and_depth,
            workspace.connection_pressures,
        )
        if has_settled(new_total_rate, total_rate, tolerance):
            total_rate = new_total_rate
            converged = True
            break
        previous_total_rate = total_rate
        total_rate = new_total_rate

    return FixedPointResult(
        converged,
        previous_total_rate,
        total_rate,
        oil_rate,
        water_rate,
        gas_rate,
        surface_oil_rate,
        surface_water_rate,
        surface_gas_rate,
    )


@numba.njit(cache=True)
def solve_uniform(
    reference_pressure: Number,
    workspace: PerforationWorkspace,
    relevant_oil: bool,
    relevant_water: bool,
    relevant_gas: bool,
    is_injector: bool,
    radial_flow_constant: Number,
    max_iterations: int,
    tolerance: Number,
) -> FixedPointResult:
    """
    Solves a well's connection rates with the reference pressure applied unchanged at every
    connection.

    :param reference_pressure: Pressure at every connection.
    :param workspace: The well's workspace. Its connection pressures, well indices and connection
        rates are updated.
    :param relevant_oil: Whether oil counts toward the controlled quantity.
    :param relevant_water: Whether water counts toward the controlled quantity.
    :param relevant_gas: Whether gas counts toward the controlled quantity.
    :param is_injector: Whether the well is an injector.
    :param radial_flow_constant: The constant in the radial flow equation, in the well's unit
        system.
    :param max_iterations: The most iterations to run.
    :param tolerance: Relative tolerance on the total rate.
    :returns: What the solve ended with.
    """
    workspace.connection_pressures[:] = reference_pressure

    total_rate = 0.0
    previous_total_rate = 0.0
    oil_rate = 0.0
    water_rate = 0.0
    gas_rate = 0.0
    surface_oil_rate = 0.0
    surface_water_rate = 0.0
    surface_gas_rate = 0.0
    converged = False
    for _ in range(max_iterations):
        (
            oil_rate,
            water_rate,
            gas_rate,
            surface_oil_rate,
            surface_water_rate,
            surface_gas_rate,
        ) = accumulate_rates(
            workspace,
            relevant_oil,
            relevant_water,
            relevant_gas,
            is_injector,
            radial_flow_constant,
        )
        new_total_rate = oil_rate + water_rate + gas_rate
        if has_settled(new_total_rate, total_rate, tolerance):
            total_rate = new_total_rate
            converged = True
            break
        previous_total_rate = total_rate
        total_rate = new_total_rate

    return FixedPointResult(
        converged,
        previous_total_rate,
        total_rate,
        oil_rate,
        water_rate,
        gas_rate,
        surface_oil_rate,
        surface_water_rate,
        surface_gas_rate,
    )


FIXED_POINT_SOLVERS = {
    "homogeneous": solve_homogeneous,
    "beggs_brill": solve_beggs_and_brill,
    "beggs_and_brill": solve_beggs_and_brill,
    "hagedorn_brown": solve_hagedorn_brown,
    "gray": solve_gray,
    "woldesemayat_ghajar": solve_woldesemayat_ghajar,
}
"""Fixed-point solver for each hydraulics correlation, by the name of its `WellBoreModel`."""

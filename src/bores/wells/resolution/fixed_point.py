"""
Fixed-point iteration between a well's connection pressures and its connection rates.
"""

import math
import typing

import numba
import numpy as np

from bores.types import Boolean, Integer, Number, NumberArray, OneDimension
from bores.wells.hydraulics.beggs_and_brill import (
    BeggsAndBrillWellbore,
)
from bores.wells.hydraulics.beggs_and_brill import (
    walk_perforation_pressures as walk_beggs_and_brill,
)
from bores.wells.hydraulics.beggs_and_brill import (
    walk_perforation_pressures_from_table as walk_perforation_pressures_from_table_beggs_and_brill,
)
from bores.wells.hydraulics.gray import GrayWellbore
from bores.wells.hydraulics.gray import (
    walk_perforation_pressures as walk_gray,
)
from bores.wells.hydraulics.gray import (
    walk_perforation_pressures_from_table as walk_perforation_pressures_from_table_gray,
)
from bores.wells.hydraulics.hagedorn_brown import HagedornBrownWellbore
from bores.wells.hydraulics.hagedorn_brown import (
    walk_perforation_pressures as walk_hagedorn_brown,
)
from bores.wells.hydraulics.hagedorn_brown import (
    walk_perforation_pressures_from_table as walk_perforation_pressures_from_table_hagedorn_brown,
)
from bores.wells.hydraulics.homogeneous import HomogeneousWellbore
from bores.wells.hydraulics.homogeneous import (
    walk_perforation_pressures as walk_homogeneous,
)
from bores.wells.hydraulics.homogeneous import (
    walk_perforation_pressures_from_table as walk_perforation_pressures_from_table_homogeneous,
)
from bores.wells.hydraulics.hydrostatic import walk_hydrostatic_pressures
from bores.wells.hydraulics.tubing_fluids import TubingFluidTable
from bores.wells.hydraulics.woldesemayat_ghajar import WoldesemayatGhajarWellbore
from bores.wells.hydraulics.woldesemayat_ghajar import (
    walk_perforation_pressures as walk_woldesemayat_ghajar,
)
from bores.wells.hydraulics.woldesemayat_ghajar import (
    walk_perforation_pressures_from_table as walk_perforation_pressures_from_table_woldesemayat_ghajar,
)
from bores.wells.workspace import (
    PerforationWorkspace,
    apply_non_darcy_correction_to_well_indices,
)
from bores.wells.workspace import (
    accumulate_rates as accumulate_phase_rates,
)

__all__ = [
    "FIXED_POINT_SOLVERS",
    "RATE_BISECTORS",
    "BisectionResult",
    "FixedPointResult",
    "bisect_beggs_and_brill",
    "bisect_gray",
    "bisect_hagedorn_brown",
    "bisect_homogeneous",
    "bisect_hydrostatic",
    "bisect_uniform",
    "bisect_woldesemayat_ghajar",
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

    converged: Boolean
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
    relevant_oil: Boolean,
    relevant_water: Boolean,
    relevant_gas: Boolean,
    is_injector: Boolean,
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
    apply_non_darcy_correction_to_well_indices(
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
        reservoir_pressures=workspace.connection_fluids.pressures,
        oil_mobilities=workspace.connection_fluids.oil_mobilities,
        water_mobilities=workspace.connection_fluids.water_mobilities,
        gas_mobilities=workspace.connection_fluids.gas_mobilities,
        oil_formation_volume_factors=workspace.connection_fluids.oil_formation_volume_factors,
        water_formation_volume_factors=workspace.connection_fluids.water_formation_volume_factors,
        gas_formation_volume_factors=workspace.connection_fluids.gas_formation_volume_factors,
        dissolved_gas_ratios=workspace.dissolved_gas_ratios,
        vaporized_oil_ratios=workspace.vaporized_oil_ratios,
        stb_to_volume_factor=workspace.stb_to_volume_factor,
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
def adjust_relaxation(
    relaxation: Number, signed_change: Number, previous_signed_change: Number
) -> Number:
    """
    Lowers the relaxation of the connection pressures when the total rate swings back and forth
    without settling.

    A well flowing at a very low pressure has gas that expands so much that the pressure it
    needs and the rate it flows at push each other past their balance on every iteration. Taking
    only part of each new pressure damps that, as it would in any fixed-point iteration.

    :param relaxation: Share of each new pressure that is taken, from `0.05` to one.
    :param signed_change: Change of the total rate in this iteration.
    :param previous_signed_change: Change of the total rate in the iteration before.
    :returns: The relaxation to use in the next iteration.
    """
    swung_back = signed_change * previous_signed_change < 0.0
    if swung_back and abs(signed_change) >= 0.5 * abs(previous_signed_change):
        return max(0.05, 0.5 * relaxation)
    return relaxation


@numba.njit(cache=True)
def relax_pressures(
    previous: NumberArray[OneDimension],
    pressures: NumberArray[OneDimension],
    relaxation: Number,
) -> None:
    """
    Replaces `pressures` with the share `relaxation` of the way from `previous` to it.

    :param previous: Pressures before this iteration.
    :param pressures: Pressures after this iteration, replaced in place.
    :param relaxation: Share of the new pressures to take.
    """
    for idx in range(pressures.shape[0]):
        pressures[idx] = previous[idx] + relaxation * (pressures[idx] - previous[idx])


@numba.njit(cache=True)
def has_settled(new_total_rate: Number, total_rate: Number, tolerance: Number) -> Boolean:
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
def get_seed_connection_rates(
    workspace: PerforationWorkspace, warm_start: Boolean
) -> tuple[NumberArray[OneDimension], NumberArray[OneDimension], NumberArray[OneDimension]]:
    """
    The connection rates a solve builds its first pressure profile from.

    :param workspace: The well's workspace.
    :param warm_start: Whether to use the connection rates already in the workspace, as a
        previous solve at a nearby reference pressure left them. Otherwise the profile starts
        from zero flow.
    :returns: The oil, water and gas rate at each connection. These are the workspace's own
        arrays when `warm_start` is set, so callers must only read them.
    """
    if warm_start:
        return (
            workspace.connection_oil_rates,
            workspace.connection_water_rates,
            workspace.connection_gas_rates,
        )
    return (
        np.zeros_like(workspace.connection_oil_rates),
        np.zeros_like(workspace.connection_water_rates),
        np.zeros_like(workspace.connection_gas_rates),
    )


@numba.njit(cache=True)
def solve_homogeneous(
    model: HomogeneousWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    workspace: PerforationWorkspace,
    relevant_oil: Boolean,
    relevant_water: Boolean,
    relevant_gas: Boolean,
    is_injector: Boolean,
    hydrostatic_divisor: Number,
    radial_flow_constant: Number,
    max_iterations: Integer,
    tolerance: Number,
    tubing_fluid_table: TubingFluidTable | None = None,
    warm_start: Boolean = False,
) -> FixedPointResult:
    """
    Solves a well's connection pressures and rates together, with the pressure profile from the
    homogeneous correlation.

    The profile starts from zero flow, or from the rates already in the workspace when
    `warm_start` is set, and is updated from the rates each iteration.

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
    :param warm_start: Whether the first pressure profile is built from the connection rates
        already in the workspace instead of from zero flow.
    :returns: What the solve ended with.
    """
    seed_oil_rates, seed_water_rates, seed_gas_rates = get_seed_connection_rates(
        workspace=workspace, warm_start=warm_start
    )
    walk_homogeneous(
        model=model,
        reference_depth=reference_depth,
        reference_measured_depth=workspace.reference_measured_depth,
        reference_pressure=reference_pressure,
        connection_oil_rates=seed_oil_rates,
        connection_water_rates=seed_water_rates,
        connection_gas_rates=seed_gas_rates,
        representative_depths=workspace.representative_depths,
        representative_measured_depths=workspace.representative_measured_depths,
        inclinations_from_vertical=workspace.inclinations_from_vertical,
        connection_fluids=workspace.connection_fluids,
        is_injector=is_injector,
        hydrostatic_divisor=hydrostatic_divisor,
        pressures=workspace.connection_pressures,
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
    relaxation = 1.0
    previous_signed_change = 0.0
    for _ in range(max_iterations):
        (
            oil_rate,
            water_rate,
            gas_rate,
            surface_oil_rate,
            surface_water_rate,
            surface_gas_rate,
        ) = accumulate_rates(
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            radial_flow_constant=radial_flow_constant,
        )
        new_total_rate = oil_rate + water_rate + gas_rate
        previous_pressures = workspace.connection_pressures
        if relaxation < 1.0:
            previous_pressures = workspace.connection_pressures.copy()
        if tubing_fluid_table is None:
            walk_homogeneous(
                model=model,
                reference_depth=reference_depth,
                reference_measured_depth=workspace.reference_measured_depth,
                reference_pressure=reference_pressure,
                connection_oil_rates=workspace.connection_oil_rates,
                connection_water_rates=workspace.connection_water_rates,
                connection_gas_rates=workspace.connection_gas_rates,
                representative_depths=workspace.representative_depths,
                representative_measured_depths=workspace.representative_measured_depths,
                inclinations_from_vertical=workspace.inclinations_from_vertical,
                connection_fluids=workspace.connection_fluids,
                is_injector=is_injector,
                hydrostatic_divisor=hydrostatic_divisor,
                pressures=workspace.connection_pressures,
            )
        else:
            walk_perforation_pressures_from_table_homogeneous(
                model=model,
                reference_depth=reference_depth,
                reference_measured_depth=workspace.reference_measured_depth,
                reference_pressure=reference_pressure,
                connection_oil_rates=workspace.connection_oil_rates,
                connection_water_rates=workspace.connection_water_rates,
                connection_gas_rates=workspace.connection_gas_rates,
                representative_depths=workspace.representative_depths,
                representative_measured_depths=workspace.representative_measured_depths,
                inclinations_from_vertical=workspace.inclinations_from_vertical,
                connection_fluids=workspace.connection_fluids,
                is_injector=is_injector,
                pressures=workspace.connection_pressures,
                tubing_fluid_table=tubing_fluid_table,
            )
        if relaxation < 1.0:
            relax_pressures(
                previous=previous_pressures,
                pressures=workspace.connection_pressures,
                relaxation=relaxation,
            )
        signed_change = new_total_rate - total_rate
        if has_settled(new_total_rate=new_total_rate, total_rate=total_rate, tolerance=tolerance):
            total_rate = new_total_rate
            converged = True
            break
        relaxation = adjust_relaxation(
            relaxation=relaxation,
            signed_change=signed_change,
            previous_signed_change=previous_signed_change,
        )
        previous_signed_change = signed_change
        previous_total_rate = total_rate
        total_rate = new_total_rate

    return FixedPointResult(
        converged=converged,
        previous_total_rate=previous_total_rate,
        total_rate=total_rate,
        oil_rate=oil_rate,
        water_rate=water_rate,
        gas_rate=gas_rate,
        surface_oil_rate=surface_oil_rate,
        surface_water_rate=surface_water_rate,
        surface_gas_rate=surface_gas_rate,
    )


@numba.njit(cache=True)
def solve_beggs_and_brill(
    model: BeggsAndBrillWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    workspace: PerforationWorkspace,
    relevant_oil: Boolean,
    relevant_water: Boolean,
    relevant_gas: Boolean,
    is_injector: Boolean,
    hydrostatic_divisor: Number,
    radial_flow_constant: Number,
    max_iterations: Integer,
    tolerance: Number,
    tubing_fluid_table: TubingFluidTable | None = None,
    warm_start: Boolean = False,
) -> FixedPointResult:
    """
    Solves a well's connection pressures and rates together, with the pressure profile from the
    Beggs and Brill correlation.

    The profile starts from zero flow, or from the rates already in the workspace when
    `warm_start` is set, and is updated from the rates each iteration.

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
    :param warm_start: Whether the first pressure profile is built from the connection rates
        already in the workspace instead of from zero flow.
    :returns: What the solve ended with.
    """
    seed_oil_rates, seed_water_rates, seed_gas_rates = get_seed_connection_rates(
        workspace=workspace, warm_start=warm_start
    )
    walk_beggs_and_brill(
        model=model,
        reference_depth=reference_depth,
        reference_measured_depth=workspace.reference_measured_depth,
        reference_pressure=reference_pressure,
        connection_oil_rates=seed_oil_rates,
        connection_water_rates=seed_water_rates,
        connection_gas_rates=seed_gas_rates,
        representative_depths=workspace.representative_depths,
        representative_measured_depths=workspace.representative_measured_depths,
        inclinations_from_vertical=workspace.inclinations_from_vertical,
        connection_fluids=workspace.connection_fluids,
        is_injector=is_injector,
        hydrostatic_divisor=hydrostatic_divisor,
        pressures=workspace.connection_pressures,
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
    relaxation = 1.0
    previous_signed_change = 0.0
    for _ in range(max_iterations):
        (
            oil_rate,
            water_rate,
            gas_rate,
            surface_oil_rate,
            surface_water_rate,
            surface_gas_rate,
        ) = accumulate_rates(
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            radial_flow_constant=radial_flow_constant,
        )
        new_total_rate = oil_rate + water_rate + gas_rate
        previous_pressures = workspace.connection_pressures
        if relaxation < 1.0:
            previous_pressures = workspace.connection_pressures.copy()
        if tubing_fluid_table is None:
            walk_beggs_and_brill(
                model=model,
                reference_depth=reference_depth,
                reference_measured_depth=workspace.reference_measured_depth,
                reference_pressure=reference_pressure,
                connection_oil_rates=workspace.connection_oil_rates,
                connection_water_rates=workspace.connection_water_rates,
                connection_gas_rates=workspace.connection_gas_rates,
                representative_depths=workspace.representative_depths,
                representative_measured_depths=workspace.representative_measured_depths,
                inclinations_from_vertical=workspace.inclinations_from_vertical,
                connection_fluids=workspace.connection_fluids,
                is_injector=is_injector,
                hydrostatic_divisor=hydrostatic_divisor,
                pressures=workspace.connection_pressures,
            )
        else:
            walk_perforation_pressures_from_table_beggs_and_brill(
                model=model,
                reference_depth=reference_depth,
                reference_measured_depth=workspace.reference_measured_depth,
                reference_pressure=reference_pressure,
                connection_oil_rates=workspace.connection_oil_rates,
                connection_water_rates=workspace.connection_water_rates,
                connection_gas_rates=workspace.connection_gas_rates,
                representative_depths=workspace.representative_depths,
                representative_measured_depths=workspace.representative_measured_depths,
                inclinations_from_vertical=workspace.inclinations_from_vertical,
                connection_fluids=workspace.connection_fluids,
                is_injector=is_injector,
                pressures=workspace.connection_pressures,
                tubing_fluid_table=tubing_fluid_table,
            )
        if relaxation < 1.0:
            relax_pressures(
                previous=previous_pressures,
                pressures=workspace.connection_pressures,
                relaxation=relaxation,
            )
        signed_change = new_total_rate - total_rate
        if has_settled(new_total_rate=new_total_rate, total_rate=total_rate, tolerance=tolerance):
            total_rate = new_total_rate
            converged = True
            break
        relaxation = adjust_relaxation(
            relaxation=relaxation,
            signed_change=signed_change,
            previous_signed_change=previous_signed_change,
        )
        previous_signed_change = signed_change
        previous_total_rate = total_rate
        total_rate = new_total_rate

    return FixedPointResult(
        converged=converged,
        previous_total_rate=previous_total_rate,
        total_rate=total_rate,
        oil_rate=oil_rate,
        water_rate=water_rate,
        gas_rate=gas_rate,
        surface_oil_rate=surface_oil_rate,
        surface_water_rate=surface_water_rate,
        surface_gas_rate=surface_gas_rate,
    )


@numba.njit(cache=True)
def solve_hagedorn_brown(
    model: HagedornBrownWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    workspace: PerforationWorkspace,
    relevant_oil: Boolean,
    relevant_water: Boolean,
    relevant_gas: Boolean,
    is_injector: Boolean,
    hydrostatic_divisor: Number,
    radial_flow_constant: Number,
    max_iterations: Integer,
    tolerance: Number,
    tubing_fluid_table: TubingFluidTable | None = None,
    warm_start: Boolean = False,
) -> FixedPointResult:
    """
    Solves a well's connection pressures and rates together, with the pressure profile from the
    Hagedorn and Brown correlation.

    The profile starts from zero flow, or from the rates already in the workspace when
    `warm_start` is set, and is updated from the rates each iteration.

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
    :param warm_start: Whether the first pressure profile is built from the connection rates
        already in the workspace instead of from zero flow.
    :returns: What the solve ended with.
    """
    seed_oil_rates, seed_water_rates, seed_gas_rates = get_seed_connection_rates(
        workspace=workspace, warm_start=warm_start
    )
    walk_hagedorn_brown(
        model=model,
        reference_depth=reference_depth,
        reference_measured_depth=workspace.reference_measured_depth,
        reference_pressure=reference_pressure,
        connection_oil_rates=seed_oil_rates,
        connection_water_rates=seed_water_rates,
        connection_gas_rates=seed_gas_rates,
        representative_depths=workspace.representative_depths,
        representative_measured_depths=workspace.representative_measured_depths,
        inclinations_from_vertical=workspace.inclinations_from_vertical,
        connection_fluids=workspace.connection_fluids,
        is_injector=is_injector,
        hydrostatic_divisor=hydrostatic_divisor,
        pressures=workspace.connection_pressures,
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
    relaxation = 1.0
    previous_signed_change = 0.0
    for _ in range(max_iterations):
        (
            oil_rate,
            water_rate,
            gas_rate,
            surface_oil_rate,
            surface_water_rate,
            surface_gas_rate,
        ) = accumulate_rates(
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            radial_flow_constant=radial_flow_constant,
        )
        new_total_rate = oil_rate + water_rate + gas_rate
        previous_pressures = workspace.connection_pressures
        if relaxation < 1.0:
            previous_pressures = workspace.connection_pressures.copy()
        if tubing_fluid_table is None:
            walk_hagedorn_brown(
                model=model,
                reference_depth=reference_depth,
                reference_measured_depth=workspace.reference_measured_depth,
                reference_pressure=reference_pressure,
                connection_oil_rates=workspace.connection_oil_rates,
                connection_water_rates=workspace.connection_water_rates,
                connection_gas_rates=workspace.connection_gas_rates,
                representative_depths=workspace.representative_depths,
                representative_measured_depths=workspace.representative_measured_depths,
                inclinations_from_vertical=workspace.inclinations_from_vertical,
                connection_fluids=workspace.connection_fluids,
                is_injector=is_injector,
                hydrostatic_divisor=hydrostatic_divisor,
                pressures=workspace.connection_pressures,
            )
        else:
            walk_perforation_pressures_from_table_hagedorn_brown(
                model=model,
                reference_depth=reference_depth,
                reference_measured_depth=workspace.reference_measured_depth,
                reference_pressure=reference_pressure,
                connection_oil_rates=workspace.connection_oil_rates,
                connection_water_rates=workspace.connection_water_rates,
                connection_gas_rates=workspace.connection_gas_rates,
                representative_depths=workspace.representative_depths,
                representative_measured_depths=workspace.representative_measured_depths,
                inclinations_from_vertical=workspace.inclinations_from_vertical,
                connection_fluids=workspace.connection_fluids,
                is_injector=is_injector,
                pressures=workspace.connection_pressures,
                tubing_fluid_table=tubing_fluid_table,
            )
        if relaxation < 1.0:
            relax_pressures(
                previous=previous_pressures,
                pressures=workspace.connection_pressures,
                relaxation=relaxation,
            )
        signed_change = new_total_rate - total_rate
        if has_settled(new_total_rate=new_total_rate, total_rate=total_rate, tolerance=tolerance):
            total_rate = new_total_rate
            converged = True
            break
        relaxation = adjust_relaxation(
            relaxation=relaxation,
            signed_change=signed_change,
            previous_signed_change=previous_signed_change,
        )
        previous_signed_change = signed_change
        previous_total_rate = total_rate
        total_rate = new_total_rate

    return FixedPointResult(
        converged=converged,
        previous_total_rate=previous_total_rate,
        total_rate=total_rate,
        oil_rate=oil_rate,
        water_rate=water_rate,
        gas_rate=gas_rate,
        surface_oil_rate=surface_oil_rate,
        surface_water_rate=surface_water_rate,
        surface_gas_rate=surface_gas_rate,
    )


@numba.njit(cache=True)
def solve_gray(
    model: GrayWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    workspace: PerforationWorkspace,
    relevant_oil: Boolean,
    relevant_water: Boolean,
    relevant_gas: Boolean,
    is_injector: Boolean,
    hydrostatic_divisor: Number,
    radial_flow_constant: Number,
    max_iterations: Integer,
    tolerance: Number,
    tubing_fluid_table: TubingFluidTable | None = None,
    warm_start: Boolean = False,
) -> FixedPointResult:
    """
    Solves a well's connection pressures and rates together, with the pressure profile from the
    Gray correlation.

    The profile starts from zero flow, or from the rates already in the workspace when
    `warm_start` is set, and is updated from the rates each iteration.

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
    :param warm_start: Whether the first pressure profile is built from the connection rates
        already in the workspace instead of from zero flow.
    :returns: What the solve ended with.
    """
    seed_oil_rates, seed_water_rates, seed_gas_rates = get_seed_connection_rates(
        workspace=workspace, warm_start=warm_start
    )
    walk_gray(
        model=model,
        reference_depth=reference_depth,
        reference_measured_depth=workspace.reference_measured_depth,
        reference_pressure=reference_pressure,
        connection_oil_rates=seed_oil_rates,
        connection_water_rates=seed_water_rates,
        connection_gas_rates=seed_gas_rates,
        representative_depths=workspace.representative_depths,
        representative_measured_depths=workspace.representative_measured_depths,
        inclinations_from_vertical=workspace.inclinations_from_vertical,
        connection_fluids=workspace.connection_fluids,
        is_injector=is_injector,
        hydrostatic_divisor=hydrostatic_divisor,
        pressures=workspace.connection_pressures,
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
    relaxation = 1.0
    previous_signed_change = 0.0
    for _ in range(max_iterations):
        (
            oil_rate,
            water_rate,
            gas_rate,
            surface_oil_rate,
            surface_water_rate,
            surface_gas_rate,
        ) = accumulate_rates(
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            radial_flow_constant=radial_flow_constant,
        )
        new_total_rate = oil_rate + water_rate + gas_rate
        previous_pressures = workspace.connection_pressures
        if relaxation < 1.0:
            previous_pressures = workspace.connection_pressures.copy()
        if tubing_fluid_table is None:
            walk_gray(
                model=model,
                reference_depth=reference_depth,
                reference_measured_depth=workspace.reference_measured_depth,
                reference_pressure=reference_pressure,
                connection_oil_rates=workspace.connection_oil_rates,
                connection_water_rates=workspace.connection_water_rates,
                connection_gas_rates=workspace.connection_gas_rates,
                representative_depths=workspace.representative_depths,
                representative_measured_depths=workspace.representative_measured_depths,
                inclinations_from_vertical=workspace.inclinations_from_vertical,
                connection_fluids=workspace.connection_fluids,
                is_injector=is_injector,
                hydrostatic_divisor=hydrostatic_divisor,
                pressures=workspace.connection_pressures,
            )
        else:
            walk_perforation_pressures_from_table_gray(
                model=model,
                reference_depth=reference_depth,
                reference_measured_depth=workspace.reference_measured_depth,
                reference_pressure=reference_pressure,
                connection_oil_rates=workspace.connection_oil_rates,
                connection_water_rates=workspace.connection_water_rates,
                connection_gas_rates=workspace.connection_gas_rates,
                representative_depths=workspace.representative_depths,
                representative_measured_depths=workspace.representative_measured_depths,
                inclinations_from_vertical=workspace.inclinations_from_vertical,
                connection_fluids=workspace.connection_fluids,
                is_injector=is_injector,
                pressures=workspace.connection_pressures,
                tubing_fluid_table=tubing_fluid_table,
            )
        if relaxation < 1.0:
            relax_pressures(
                previous=previous_pressures,
                pressures=workspace.connection_pressures,
                relaxation=relaxation,
            )
        signed_change = new_total_rate - total_rate
        if has_settled(new_total_rate=new_total_rate, total_rate=total_rate, tolerance=tolerance):
            total_rate = new_total_rate
            converged = True
            break
        relaxation = adjust_relaxation(
            relaxation=relaxation,
            signed_change=signed_change,
            previous_signed_change=previous_signed_change,
        )
        previous_signed_change = signed_change
        previous_total_rate = total_rate
        total_rate = new_total_rate

    return FixedPointResult(
        converged=converged,
        previous_total_rate=previous_total_rate,
        total_rate=total_rate,
        oil_rate=oil_rate,
        water_rate=water_rate,
        gas_rate=gas_rate,
        surface_oil_rate=surface_oil_rate,
        surface_water_rate=surface_water_rate,
        surface_gas_rate=surface_gas_rate,
    )


@numba.njit(cache=True)
def solve_woldesemayat_ghajar(
    model: WoldesemayatGhajarWellbore,
    reference_depth: Number,
    reference_pressure: Number,
    workspace: PerforationWorkspace,
    relevant_oil: Boolean,
    relevant_water: Boolean,
    relevant_gas: Boolean,
    is_injector: Boolean,
    hydrostatic_divisor: Number,
    radial_flow_constant: Number,
    max_iterations: Integer,
    tolerance: Number,
    tubing_fluid_table: TubingFluidTable | None = None,
    warm_start: Boolean = False,
) -> FixedPointResult:
    """
    Solves a well's connection pressures and rates together, with the pressure profile from the
    Woldesemayat and Ghajar correlation.

    The profile starts from zero flow, or from the rates already in the workspace when
    `warm_start` is set, and is updated from the rates each iteration.

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
    :param warm_start: Whether the first pressure profile is built from the connection rates
        already in the workspace instead of from zero flow.
    :returns: What the solve ended with.
    """
    seed_oil_rates, seed_water_rates, seed_gas_rates = get_seed_connection_rates(
        workspace=workspace, warm_start=warm_start
    )
    walk_woldesemayat_ghajar(
        model=model,
        reference_depth=reference_depth,
        reference_measured_depth=workspace.reference_measured_depth,
        reference_pressure=reference_pressure,
        connection_oil_rates=seed_oil_rates,
        connection_water_rates=seed_water_rates,
        connection_gas_rates=seed_gas_rates,
        representative_depths=workspace.representative_depths,
        representative_measured_depths=workspace.representative_measured_depths,
        inclinations_from_vertical=workspace.inclinations_from_vertical,
        connection_fluids=workspace.connection_fluids,
        is_injector=is_injector,
        hydrostatic_divisor=hydrostatic_divisor,
        pressures=workspace.connection_pressures,
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
    relaxation = 1.0
    previous_signed_change = 0.0
    for _ in range(max_iterations):
        (
            oil_rate,
            water_rate,
            gas_rate,
            surface_oil_rate,
            surface_water_rate,
            surface_gas_rate,
        ) = accumulate_rates(
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            radial_flow_constant=radial_flow_constant,
        )
        new_total_rate = oil_rate + water_rate + gas_rate
        previous_pressures = workspace.connection_pressures
        if relaxation < 1.0:
            previous_pressures = workspace.connection_pressures.copy()
        if tubing_fluid_table is None:
            walk_woldesemayat_ghajar(
                model=model,
                reference_depth=reference_depth,
                reference_measured_depth=workspace.reference_measured_depth,
                reference_pressure=reference_pressure,
                connection_oil_rates=workspace.connection_oil_rates,
                connection_water_rates=workspace.connection_water_rates,
                connection_gas_rates=workspace.connection_gas_rates,
                representative_depths=workspace.representative_depths,
                representative_measured_depths=workspace.representative_measured_depths,
                inclinations_from_vertical=workspace.inclinations_from_vertical,
                connection_fluids=workspace.connection_fluids,
                is_injector=is_injector,
                hydrostatic_divisor=hydrostatic_divisor,
                pressures=workspace.connection_pressures,
            )
        else:
            walk_perforation_pressures_from_table_woldesemayat_ghajar(
                model=model,
                reference_depth=reference_depth,
                reference_measured_depth=workspace.reference_measured_depth,
                reference_pressure=reference_pressure,
                connection_oil_rates=workspace.connection_oil_rates,
                connection_water_rates=workspace.connection_water_rates,
                connection_gas_rates=workspace.connection_gas_rates,
                representative_depths=workspace.representative_depths,
                representative_measured_depths=workspace.representative_measured_depths,
                inclinations_from_vertical=workspace.inclinations_from_vertical,
                connection_fluids=workspace.connection_fluids,
                is_injector=is_injector,
                pressures=workspace.connection_pressures,
                tubing_fluid_table=tubing_fluid_table,
            )
        if relaxation < 1.0:
            relax_pressures(
                previous=previous_pressures,
                pressures=workspace.connection_pressures,
                relaxation=relaxation,
            )
        signed_change = new_total_rate - total_rate
        if has_settled(new_total_rate=new_total_rate, total_rate=total_rate, tolerance=tolerance):
            total_rate = new_total_rate
            converged = True
            break
        relaxation = adjust_relaxation(
            relaxation=relaxation,
            signed_change=signed_change,
            previous_signed_change=previous_signed_change,
        )
        previous_signed_change = signed_change
        previous_total_rate = total_rate
        total_rate = new_total_rate

    return FixedPointResult(
        converged=converged,
        previous_total_rate=previous_total_rate,
        total_rate=total_rate,
        oil_rate=oil_rate,
        water_rate=water_rate,
        gas_rate=gas_rate,
        surface_oil_rate=surface_oil_rate,
        surface_water_rate=surface_water_rate,
        surface_gas_rate=surface_gas_rate,
    )


@numba.njit(cache=True)
def solve_hydrostatic(
    reference_depth: Number,
    reference_pressure: Number,
    workspace: PerforationWorkspace,
    relevant_oil: Boolean,
    relevant_water: Boolean,
    relevant_gas: Boolean,
    is_injector: Boolean,
    pressure_per_density_and_depth: Number,
    radial_flow_constant: Number,
    max_iterations: Integer,
    tolerance: Number,
    warm_start: Boolean = False,
) -> FixedPointResult:
    """
    Solves a well's connection pressures and rates together, with the pressure profile from the
    weight of the fluid between connections.

    The profile starts from zero flow, or from the rates already in the workspace when
    `warm_start` is set, and is updated from the rates each iteration.

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
    :param warm_start: Whether the first pressure profile is built from the connection rates
        already in the workspace instead of from zero flow.
    :returns: What the solve ended with.
    """
    seed_oil_rates, seed_water_rates, seed_gas_rates = get_seed_connection_rates(
        workspace=workspace, warm_start=warm_start
    )
    walk_hydrostatic_pressures(
        reference_depth=reference_depth,
        reference_pressure=reference_pressure,
        representative_depths=workspace.representative_depths,
        connection_fluids=workspace.connection_fluids,
        connection_oil_rates=seed_oil_rates,
        connection_water_rates=seed_water_rates,
        connection_gas_rates=seed_gas_rates,
        pressure_per_density_and_depth=pressure_per_density_and_depth,
        pressures=workspace.connection_pressures,
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
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            radial_flow_constant=radial_flow_constant,
        )
        new_total_rate = oil_rate + water_rate + gas_rate
        walk_hydrostatic_pressures(
            reference_depth=reference_depth,
            reference_pressure=reference_pressure,
            representative_depths=workspace.representative_depths,
            connection_fluids=workspace.connection_fluids,
            connection_oil_rates=workspace.connection_oil_rates,
            connection_water_rates=workspace.connection_water_rates,
            connection_gas_rates=workspace.connection_gas_rates,
            pressure_per_density_and_depth=pressure_per_density_and_depth,
            pressures=workspace.connection_pressures,
        )
        if has_settled(new_total_rate=new_total_rate, total_rate=total_rate, tolerance=tolerance):
            total_rate = new_total_rate
            converged = True
            break
        previous_total_rate = total_rate
        total_rate = new_total_rate

    return FixedPointResult(
        converged=converged,
        previous_total_rate=previous_total_rate,
        total_rate=total_rate,
        oil_rate=oil_rate,
        water_rate=water_rate,
        gas_rate=gas_rate,
        surface_oil_rate=surface_oil_rate,
        surface_water_rate=surface_water_rate,
        surface_gas_rate=surface_gas_rate,
    )


@numba.njit(cache=True)
def solve_uniform(
    reference_pressure: Number,
    workspace: PerforationWorkspace,
    relevant_oil: Boolean,
    relevant_water: Boolean,
    relevant_gas: Boolean,
    is_injector: Boolean,
    radial_flow_constant: Number,
    max_iterations: Integer,
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
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            radial_flow_constant=radial_flow_constant,
        )
        new_total_rate = oil_rate + water_rate + gas_rate
        if has_settled(new_total_rate=new_total_rate, total_rate=total_rate, tolerance=tolerance):
            total_rate = new_total_rate
            converged = True
            break
        previous_total_rate = total_rate
        total_rate = new_total_rate

    return FixedPointResult(
        converged=converged,
        previous_total_rate=previous_total_rate,
        total_rate=total_rate,
        oil_rate=oil_rate,
        water_rate=water_rate,
        gas_rate=gas_rate,
        surface_oil_rate=surface_oil_rate,
        surface_water_rate=surface_water_rate,
        surface_gas_rate=surface_gas_rate,
    )


class BisectionResult(typing.NamedTuple):
    """What a bisection of a well's reference pressure ended with."""

    bhp: Number
    """The reference pressure of the last solve."""

    result: FixedPointResult
    """The last solve. If it didn't converge, the bisection stopped there."""


@numba.njit(cache=True)
def bisect_homogeneous(
    model: HomogeneousWellbore,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    relevant_oil: Boolean,
    relevant_water: Boolean,
    relevant_gas: Boolean,
    is_injector: Boolean,
    hydrostatic_divisor: Number,
    radial_flow_constant: Number,
    max_iterations: Integer,
    tolerance: Number,
    target: Number,
    low: Number,
    high: Number,
    reservoir_condition: Boolean,
    max_bisections: Integer,
    rate_tolerance: Number,
    tubing_fluid_table: TubingFluidTable | None = None,
) -> BisectionResult:
    """
    Finds the reference pressure at which a well's total rate matches a target, with the pressure
    profile from the homogeneous correlation.

    The total rate falls as the reference pressure rises for a producer and rises with it for an
    injector. Each step bisects the bounds and solves the connection pressures and rates there. If a
    solve doesn't converge the bisection stops and returns it, for the caller to report.

    :param model: The well's `HomogeneousWellbore`.
    :param hydrostatic_divisor: The model's unit system's gravitational factor times its
        hydrostatic area factor, as `get_hydrostatic_divisor` gives it.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: The well's workspace. Its connection pressures, well indices and connection
        rates are updated.
    :param relevant_oil: Whether oil counts toward the controlled quantity.
    :param relevant_water: Whether water counts toward the controlled quantity.
    :param relevant_gas: Whether gas counts toward the controlled quantity.
    :param is_injector: Whether the well is an injector.
    :param radial_flow_constant: The constant in the radial flow equation, in the well's unit
        system.
    :param max_iterations: The most fixed-point iterations of each solve.
    :param tolerance: Relative tolerance on the total rate of each solve.
    :param target: The total rate to match.
    :param low: Lower bound of the reference pressure.
    :param high: Upper bound of the reference pressure.
    :param reservoir_condition: Whether `target` is a reservoir-condition rate, instead of a
        surface-condition one.
    :param max_bisections: The most bisection steps to take.
    :param rate_tolerance: Relative tolerance on the match to `target`.
    :returns: The reference pressure of the last solve, and that solve.
    """
    average = 0.5 * (low + high)
    result = FixedPointResult(
        converged=False,
        previous_total_rate=0.0,
        total_rate=0.0,
        oil_rate=0.0,
        water_rate=0.0,
        gas_rate=0.0,
        surface_oil_rate=0.0,
        surface_water_rate=0.0,
        surface_gas_rate=0.0,
    )
    for step in range(max_bisections):
        average = 0.5 * (low + high)
        result = solve_homogeneous(
            model=model,
            reference_depth=reference_depth,
            reference_pressure=average,
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            hydrostatic_divisor=hydrostatic_divisor,
            radial_flow_constant=radial_flow_constant,
            max_iterations=max_iterations,
            tolerance=tolerance,
            tubing_fluid_table=tubing_fluid_table,
            warm_start=step > 0,
        )
        if not result.converged:
            return BisectionResult(bhp=average, result=result)
        if reservoir_condition:
            value = result.oil_rate + result.water_rate + result.gas_rate
        else:
            value = result.surface_oil_rate + result.surface_water_rate + result.surface_gas_rate
        if abs(value - target) <= rate_tolerance * max(abs(target), 1.0):
            break
        if (value < target) == is_injector:
            low = average
        else:
            high = average
    return BisectionResult(bhp=average, result=result)


@numba.njit(cache=True)
def bisect_beggs_and_brill(
    model: BeggsAndBrillWellbore,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    relevant_oil: Boolean,
    relevant_water: Boolean,
    relevant_gas: Boolean,
    is_injector: Boolean,
    hydrostatic_divisor: Number,
    radial_flow_constant: Number,
    max_iterations: Integer,
    tolerance: Number,
    target: Number,
    low: Number,
    high: Number,
    reservoir_condition: Boolean,
    max_bisections: Integer,
    rate_tolerance: Number,
    tubing_fluid_table: TubingFluidTable | None = None,
) -> BisectionResult:
    """
    Finds the reference pressure at which a well's total rate matches a target, with the pressure
    profile from the Beggs and Brill correlation.

    The total rate falls as the reference pressure rises for a producer and rises with it for an
    injector. Each step bisects the bounds and solves the connection pressures and rates there. If a
    solve doesn't converge the bisection stops and returns it, for the caller to report.

    :param model: The well's `BeggsAndBrillWellbore`.
    :param hydrostatic_divisor: The model's unit system's gravitational factor times its
        hydrostatic area factor, as `get_hydrostatic_divisor` gives it.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: The well's workspace. Its connection pressures, well indices and connection
        rates are updated.
    :param relevant_oil: Whether oil counts toward the controlled quantity.
    :param relevant_water: Whether water counts toward the controlled quantity.
    :param relevant_gas: Whether gas counts toward the controlled quantity.
    :param is_injector: Whether the well is an injector.
    :param radial_flow_constant: The constant in the radial flow equation, in the well's unit
        system.
    :param max_iterations: The most fixed-point iterations of each solve.
    :param tolerance: Relative tolerance on the total rate of each solve.
    :param target: The total rate to match.
    :param low: Lower bound of the reference pressure.
    :param high: Upper bound of the reference pressure.
    :param reservoir_condition: Whether `target` is a reservoir-condition rate, instead of a
        surface-condition one.
    :param max_bisections: The most bisection steps to take.
    :param rate_tolerance: Relative tolerance on the match to `target`.
    :returns: The reference pressure of the last solve, and that solve.
    """
    average = 0.5 * (low + high)
    result = FixedPointResult(
        converged=False,
        previous_total_rate=0.0,
        total_rate=0.0,
        oil_rate=0.0,
        water_rate=0.0,
        gas_rate=0.0,
        surface_oil_rate=0.0,
        surface_water_rate=0.0,
        surface_gas_rate=0.0,
    )
    for step in range(max_bisections):
        average = 0.5 * (low + high)
        result = solve_beggs_and_brill(
            model=model,
            reference_depth=reference_depth,
            reference_pressure=average,
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            hydrostatic_divisor=hydrostatic_divisor,
            radial_flow_constant=radial_flow_constant,
            max_iterations=max_iterations,
            tolerance=tolerance,
            tubing_fluid_table=tubing_fluid_table,
            warm_start=step > 0,
        )
        if not result.converged:
            return BisectionResult(bhp=average, result=result)
        if reservoir_condition:
            value = result.oil_rate + result.water_rate + result.gas_rate
        else:
            value = result.surface_oil_rate + result.surface_water_rate + result.surface_gas_rate
        if abs(value - target) <= rate_tolerance * max(abs(target), 1.0):
            break
        if (value < target) == is_injector:
            low = average
        else:
            high = average
    return BisectionResult(bhp=average, result=result)


@numba.njit(cache=True)
def bisect_hagedorn_brown(
    model: HagedornBrownWellbore,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    relevant_oil: Boolean,
    relevant_water: Boolean,
    relevant_gas: Boolean,
    is_injector: Boolean,
    hydrostatic_divisor: Number,
    radial_flow_constant: Number,
    max_iterations: Integer,
    tolerance: Number,
    target: Number,
    low: Number,
    high: Number,
    reservoir_condition: Boolean,
    max_bisections: Integer,
    rate_tolerance: Number,
    tubing_fluid_table: TubingFluidTable | None = None,
) -> BisectionResult:
    """
    Finds the reference pressure at which a well's total rate matches a target, with the pressure
    profile from the Hagedorn and Brown correlation.

    The total rate falls as the reference pressure rises for a producer and rises with it for an
    injector. Each step bisects the bounds and solves the connection pressures and rates there. If a
    solve doesn't converge the bisection stops and returns it, for the caller to report.

    :param model: The well's `HagedornBrownWellbore`.
    :param hydrostatic_divisor: The model's unit system's gravitational factor times its
        hydrostatic area factor, as `get_hydrostatic_divisor` gives it.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: The well's workspace. Its connection pressures, well indices and connection
        rates are updated.
    :param relevant_oil: Whether oil counts toward the controlled quantity.
    :param relevant_water: Whether water counts toward the controlled quantity.
    :param relevant_gas: Whether gas counts toward the controlled quantity.
    :param is_injector: Whether the well is an injector.
    :param radial_flow_constant: The constant in the radial flow equation, in the well's unit
        system.
    :param max_iterations: The most fixed-point iterations of each solve.
    :param tolerance: Relative tolerance on the total rate of each solve.
    :param target: The total rate to match.
    :param low: Lower bound of the reference pressure.
    :param high: Upper bound of the reference pressure.
    :param reservoir_condition: Whether `target` is a reservoir-condition rate, instead of a
        surface-condition one.
    :param max_bisections: The most bisection steps to take.
    :param rate_tolerance: Relative tolerance on the match to `target`.
    :returns: The reference pressure of the last solve, and that solve.
    """
    average = 0.5 * (low + high)
    result = FixedPointResult(
        converged=False,
        previous_total_rate=0.0,
        total_rate=0.0,
        oil_rate=0.0,
        water_rate=0.0,
        gas_rate=0.0,
        surface_oil_rate=0.0,
        surface_water_rate=0.0,
        surface_gas_rate=0.0,
    )
    for step in range(max_bisections):
        average = 0.5 * (low + high)
        result = solve_hagedorn_brown(
            model=model,
            reference_depth=reference_depth,
            reference_pressure=average,
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            hydrostatic_divisor=hydrostatic_divisor,
            radial_flow_constant=radial_flow_constant,
            max_iterations=max_iterations,
            tolerance=tolerance,
            tubing_fluid_table=tubing_fluid_table,
            warm_start=step > 0,
        )
        if not result.converged:
            return BisectionResult(bhp=average, result=result)
        if reservoir_condition:
            value = result.oil_rate + result.water_rate + result.gas_rate
        else:
            value = result.surface_oil_rate + result.surface_water_rate + result.surface_gas_rate
        if abs(value - target) <= rate_tolerance * max(abs(target), 1.0):
            break
        if (value < target) == is_injector:
            low = average
        else:
            high = average
    return BisectionResult(bhp=average, result=result)


@numba.njit(cache=True)
def bisect_gray(
    model: GrayWellbore,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    relevant_oil: Boolean,
    relevant_water: Boolean,
    relevant_gas: Boolean,
    is_injector: Boolean,
    hydrostatic_divisor: Number,
    radial_flow_constant: Number,
    max_iterations: Integer,
    tolerance: Number,
    target: Number,
    low: Number,
    high: Number,
    reservoir_condition: Boolean,
    max_bisections: Integer,
    rate_tolerance: Number,
    tubing_fluid_table: TubingFluidTable | None = None,
) -> BisectionResult:
    """
    Finds the reference pressure at which a well's total rate matches a target, with the pressure
    profile from the Gray correlation.

    The total rate falls as the reference pressure rises for a producer and rises with it for an
    injector. Each step bisects the bounds and solves the connection pressures and rates there. If a
    solve doesn't converge the bisection stops and returns it, for the caller to report.

    :param model: The well's `GrayWellbore`.
    :param hydrostatic_divisor: The model's unit system's gravitational factor times its
        hydrostatic area factor, as `get_hydrostatic_divisor` gives it.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: The well's workspace. Its connection pressures, well indices and connection
        rates are updated.
    :param relevant_oil: Whether oil counts toward the controlled quantity.
    :param relevant_water: Whether water counts toward the controlled quantity.
    :param relevant_gas: Whether gas counts toward the controlled quantity.
    :param is_injector: Whether the well is an injector.
    :param radial_flow_constant: The constant in the radial flow equation, in the well's unit
        system.
    :param max_iterations: The most fixed-point iterations of each solve.
    :param tolerance: Relative tolerance on the total rate of each solve.
    :param target: The total rate to match.
    :param low: Lower bound of the reference pressure.
    :param high: Upper bound of the reference pressure.
    :param reservoir_condition: Whether `target` is a reservoir-condition rate, instead of a
        surface-condition one.
    :param max_bisections: The most bisection steps to take.
    :param rate_tolerance: Relative tolerance on the match to `target`.
    :returns: The reference pressure of the last solve, and that solve.
    """
    average = 0.5 * (low + high)
    result = FixedPointResult(
        converged=False,
        previous_total_rate=0.0,
        total_rate=0.0,
        oil_rate=0.0,
        water_rate=0.0,
        gas_rate=0.0,
        surface_oil_rate=0.0,
        surface_water_rate=0.0,
        surface_gas_rate=0.0,
    )
    for step in range(max_bisections):
        average = 0.5 * (low + high)
        result = solve_gray(
            model=model,
            reference_depth=reference_depth,
            reference_pressure=average,
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            hydrostatic_divisor=hydrostatic_divisor,
            radial_flow_constant=radial_flow_constant,
            max_iterations=max_iterations,
            tolerance=tolerance,
            tubing_fluid_table=tubing_fluid_table,
            warm_start=step > 0,
        )
        if not result.converged:
            return BisectionResult(bhp=average, result=result)
        if reservoir_condition:
            value = result.oil_rate + result.water_rate + result.gas_rate
        else:
            value = result.surface_oil_rate + result.surface_water_rate + result.surface_gas_rate
        if abs(value - target) <= rate_tolerance * max(abs(target), 1.0):
            break
        if (value < target) == is_injector:
            low = average
        else:
            high = average
    return BisectionResult(bhp=average, result=result)


@numba.njit(cache=True)
def bisect_woldesemayat_ghajar(
    model: WoldesemayatGhajarWellbore,
    reference_depth: Number,
    workspace: PerforationWorkspace,
    relevant_oil: Boolean,
    relevant_water: Boolean,
    relevant_gas: Boolean,
    is_injector: Boolean,
    hydrostatic_divisor: Number,
    radial_flow_constant: Number,
    max_iterations: Integer,
    tolerance: Number,
    target: Number,
    low: Number,
    high: Number,
    reservoir_condition: Boolean,
    max_bisections: Integer,
    rate_tolerance: Number,
    tubing_fluid_table: TubingFluidTable | None = None,
) -> BisectionResult:
    """
    Finds the reference pressure at which a well's total rate matches a target, with the pressure
    profile from the Woldesemayat and Ghajar correlation.

    The total rate falls as the reference pressure rises for a producer and rises with it for an
    injector. Each step bisects the bounds and solves the connection pressures and rates there. If a
    solve doesn't converge the bisection stops and returns it, for the caller to report.

    :param model: The well's `WoldesemayatGhajarWellbore`.
    :param hydrostatic_divisor: The model's unit system's gravitational factor times its
        hydrostatic area factor, as `get_hydrostatic_divisor` gives it.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: The well's workspace. Its connection pressures, well indices and connection
        rates are updated.
    :param relevant_oil: Whether oil counts toward the controlled quantity.
    :param relevant_water: Whether water counts toward the controlled quantity.
    :param relevant_gas: Whether gas counts toward the controlled quantity.
    :param is_injector: Whether the well is an injector.
    :param radial_flow_constant: The constant in the radial flow equation, in the well's unit
        system.
    :param max_iterations: The most fixed-point iterations of each solve.
    :param tolerance: Relative tolerance on the total rate of each solve.
    :param target: The total rate to match.
    :param low: Lower bound of the reference pressure.
    :param high: Upper bound of the reference pressure.
    :param reservoir_condition: Whether `target` is a reservoir-condition rate, instead of a
        surface-condition one.
    :param max_bisections: The most bisection steps to take.
    :param rate_tolerance: Relative tolerance on the match to `target`.
    :returns: The reference pressure of the last solve, and that solve.
    """
    average = 0.5 * (low + high)
    result = FixedPointResult(
        converged=False,
        previous_total_rate=0.0,
        total_rate=0.0,
        oil_rate=0.0,
        water_rate=0.0,
        gas_rate=0.0,
        surface_oil_rate=0.0,
        surface_water_rate=0.0,
        surface_gas_rate=0.0,
    )
    for step in range(max_bisections):
        average = 0.5 * (low + high)
        result = solve_woldesemayat_ghajar(
            model=model,
            reference_depth=reference_depth,
            reference_pressure=average,
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            hydrostatic_divisor=hydrostatic_divisor,
            radial_flow_constant=radial_flow_constant,
            max_iterations=max_iterations,
            tolerance=tolerance,
            tubing_fluid_table=tubing_fluid_table,
            warm_start=step > 0,
        )
        if not result.converged:
            return BisectionResult(bhp=average, result=result)
        if reservoir_condition:
            value = result.oil_rate + result.water_rate + result.gas_rate
        else:
            value = result.surface_oil_rate + result.surface_water_rate + result.surface_gas_rate
        if abs(value - target) <= rate_tolerance * max(abs(target), 1.0):
            break
        if (value < target) == is_injector:
            low = average
        else:
            high = average
    return BisectionResult(bhp=average, result=result)


@numba.njit(cache=True)
def bisect_hydrostatic(
    reference_depth: Number,
    workspace: PerforationWorkspace,
    relevant_oil: Boolean,
    relevant_water: Boolean,
    relevant_gas: Boolean,
    is_injector: Boolean,
    pressure_per_density_and_depth: Number,
    radial_flow_constant: Number,
    max_iterations: Integer,
    tolerance: Number,
    target: Number,
    low: Number,
    high: Number,
    reservoir_condition: Boolean,
    max_bisections: Integer,
    rate_tolerance: Number,
) -> BisectionResult:
    """
    Finds the reference pressure at which a well's total rate matches a target, with the pressure
    profile from the weight of the fluid between connections.

    See `bisect_homogeneous` for how the bisection proceeds.

    :param pressure_per_density_and_depth: As `get_pressure_per_density_and_depth` gives it.
    :param reference_depth: The well's BHP/THP reporting datum.
    :param workspace: The well's workspace. Its connection pressures, well indices and connection
        rates are updated.
    :param relevant_oil: Whether oil counts toward the controlled quantity.
    :param relevant_water: Whether water counts toward the controlled quantity.
    :param relevant_gas: Whether gas counts toward the controlled quantity.
    :param is_injector: Whether the well is an injector.
    :param radial_flow_constant: The constant in the radial flow equation, in the well's unit
        system.
    :param max_iterations: The most fixed-point iterations of each solve.
    :param tolerance: Relative tolerance on the total rate of each solve.
    :param target: The total rate to match.
    :param low: Lower bound of the reference pressure.
    :param high: Upper bound of the reference pressure.
    :param reservoir_condition: Whether `target` is a reservoir-condition rate, instead of a
        surface-condition one.
    :param max_bisections: The most bisection steps to take.
    :param rate_tolerance: Relative tolerance on the match to `target`.
    :returns: The reference pressure of the last solve, and that solve.
    """
    average = 0.5 * (low + high)
    result = FixedPointResult(
        converged=False,
        previous_total_rate=0.0,
        total_rate=0.0,
        oil_rate=0.0,
        water_rate=0.0,
        gas_rate=0.0,
        surface_oil_rate=0.0,
        surface_water_rate=0.0,
        surface_gas_rate=0.0,
    )
    for step in range(max_bisections):
        average = 0.5 * (low + high)
        result = solve_hydrostatic(
            reference_depth=reference_depth,
            reference_pressure=average,
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            pressure_per_density_and_depth=pressure_per_density_and_depth,
            radial_flow_constant=radial_flow_constant,
            max_iterations=max_iterations,
            tolerance=tolerance,
            warm_start=step > 0,
        )
        if not result.converged:
            return BisectionResult(bhp=average, result=result)
        if reservoir_condition:
            value = result.oil_rate + result.water_rate + result.gas_rate
        else:
            value = result.surface_oil_rate + result.surface_water_rate + result.surface_gas_rate
        if abs(value - target) <= rate_tolerance * max(abs(target), 1.0):
            break
        if (value < target) == is_injector:
            low = average
        else:
            high = average
    return BisectionResult(bhp=average, result=result)


@numba.njit(cache=True)
def bisect_uniform(
    workspace: PerforationWorkspace,
    relevant_oil: Boolean,
    relevant_water: Boolean,
    relevant_gas: Boolean,
    is_injector: Boolean,
    radial_flow_constant: Number,
    max_iterations: Integer,
    tolerance: Number,
    target: Number,
    low: Number,
    high: Number,
    reservoir_condition: Boolean,
    max_bisections: Integer,
    rate_tolerance: Number,
) -> BisectionResult:
    """
    Finds the reference pressure at which a well's total rate matches a target, with that
    pressure applied unchanged at every connection.

    See `bisect_homogeneous` for how the bisection proceeds.

    :param workspace: The well's workspace. Its connection pressures, well indices and connection
        rates are updated.
    :param relevant_oil: Whether oil counts toward the controlled quantity.
    :param relevant_water: Whether water counts toward the controlled quantity.
    :param relevant_gas: Whether gas counts toward the controlled quantity.
    :param is_injector: Whether the well is an injector.
    :param radial_flow_constant: The constant in the radial flow equation, in the well's unit
        system.
    :param max_iterations: The most fixed-point iterations of each solve.
    :param tolerance: Relative tolerance on the total rate of each solve.
    :param target: The total rate to match.
    :param low: Lower bound of the reference pressure.
    :param high: Upper bound of the reference pressure.
    :param reservoir_condition: Whether `target` is a reservoir-condition rate, instead of a
        surface-condition one.
    :param max_bisections: The most bisection steps to take.
    :param rate_tolerance: Relative tolerance on the match to `target`.
    :returns: The reference pressure of the last solve, and that solve.
    """
    average = 0.5 * (low + high)
    result = FixedPointResult(
        converged=False,
        previous_total_rate=0.0,
        total_rate=0.0,
        oil_rate=0.0,
        water_rate=0.0,
        gas_rate=0.0,
        surface_oil_rate=0.0,
        surface_water_rate=0.0,
        surface_gas_rate=0.0,
    )
    for _ in range(max_bisections):
        average = 0.5 * (low + high)
        result = solve_uniform(
            reference_pressure=average,
            workspace=workspace,
            relevant_oil=relevant_oil,
            relevant_water=relevant_water,
            relevant_gas=relevant_gas,
            is_injector=is_injector,
            radial_flow_constant=radial_flow_constant,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        if not result.converged:
            return BisectionResult(bhp=average, result=result)
        if reservoir_condition:
            value = result.oil_rate + result.water_rate + result.gas_rate
        else:
            value = result.surface_oil_rate + result.surface_water_rate + result.surface_gas_rate
        if abs(value - target) <= rate_tolerance * max(abs(target), 1.0):
            break
        if (value < target) == is_injector:
            low = average
        else:
            high = average
    return BisectionResult(bhp=average, result=result)


class HydraulicSolver(typing.Protocol):
    """A fixed-point solver for a well's connection pressures and rates."""

    def __call__(
        self,
        model: typing.Any,
        reference_depth: Number,
        reference_pressure: Number,
        workspace: PerforationWorkspace,
        relevant_oil: Boolean,
        relevant_water: Boolean,
        relevant_gas: Boolean,
        is_injector: Boolean,
        hydrostatic_divisor: Number,
        radial_flow_constant: Number,
        max_iterations: Integer,
        tolerance: Number,
        tubing_fluid_table: TubingFluidTable | None,
    ) -> FixedPointResult: ...


class HydraulicBisector(typing.Protocol):
    """A bisection of a well's reference pressure to match a target total rate."""

    def __call__(
        self,
        model: typing.Any,
        reference_depth: Number,
        workspace: PerforationWorkspace,
        relevant_oil: Boolean,
        relevant_water: Boolean,
        relevant_gas: Boolean,
        is_injector: Boolean,
        hydrostatic_divisor: Number,
        radial_flow_constant: Number,
        max_iterations: Integer,
        tolerance: Number,
        target: Number,
        low: Number,
        high: Number,
        reservoir_condition: Boolean,
        max_bisections: Integer,
        rate_tolerance: Number,
        tubing_fluid_table: TubingFluidTable | None,
    ) -> BisectionResult: ...


class HydrostaticSolver(typing.Protocol):
    """
    A fixed-point solver for a well's connection pressures and rates, with the pressure profile
    from the weight of the fluid between connections.
    """

    def __call__(
        self,
        reference_depth: Number,
        reference_pressure: Number,
        workspace: PerforationWorkspace,
        relevant_oil: Boolean,
        relevant_water: Boolean,
        relevant_gas: Boolean,
        is_injector: Boolean,
        pressure_per_density_and_depth: Number,
        radial_flow_constant: Number,
        max_iterations: Integer,
        tolerance: Number,
    ) -> FixedPointResult: ...


class HydrostaticBisector(typing.Protocol):
    """
    A bisection of a well's reference pressure to match a target total rate, with the pressure profile
    from the weight of the fluid between connections.
    """

    def __call__(
        self,
        reference_depth: Number,
        workspace: PerforationWorkspace,
        relevant_oil: Boolean,
        relevant_water: Boolean,
        relevant_gas: Boolean,
        is_injector: Boolean,
        pressure_per_density_and_depth: Number,
        radial_flow_constant: Number,
        max_iterations: Integer,
        tolerance: Number,
        target: Number,
        low: Number,
        high: Number,
        reservoir_condition: Boolean,
        max_bisections: Integer,
        rate_tolerance: Number,
    ) -> BisectionResult: ...


class UniformSolver(typing.Protocol):
    """
    A fixed-point solver for a well's connection pressures and rates, assuming a uniform pressure profile.
    """

    def __call__(
        self,
        reference_pressure: Number,
        workspace: PerforationWorkspace,
        relevant_oil: Boolean,
        relevant_water: Boolean,
        relevant_gas: Boolean,
        is_injector: Boolean,
        radial_flow_constant: Number,
        max_iterations: Integer,
        tolerance: Number,
    ) -> FixedPointResult: ...


class UniformBisector(typing.Protocol):
    """
    A bisection of a well's reference pressure to match a target total rate, with the pressure profile
    from the weight of the fluid between connections.
    """

    def __call__(
        self,
        workspace: PerforationWorkspace,
        relevant_oil: Boolean,
        relevant_water: Boolean,
        relevant_gas: Boolean,
        is_injector: Boolean,
        radial_flow_constant: Number,
        max_iterations: Integer,
        tolerance: Number,
        target: Number,
        low: Number,
        high: Number,
        reservoir_condition: Boolean,
        max_bisections: Integer,
        rate_tolerance: Number,
    ) -> BisectionResult: ...


FixedPointSolver = HydraulicSolver | HydrostaticSolver | UniformSolver
FixedPointBisector = HydraulicBisector | HydrostaticBisector | UniformBisector


FIXED_POINT_SOLVERS: dict[str, FixedPointSolver] = {
    "homogeneous": solve_homogeneous,
    "beggs_brill": solve_beggs_and_brill,
    "beggs_and_brill": solve_beggs_and_brill,
    "hagedorn_brown": solve_hagedorn_brown,
    "gray": solve_gray,
    "woldesemayat_ghajar": solve_woldesemayat_ghajar,
}
"""Fixed-point solver for each hydraulics correlation, by the name of its `WellBoreModel`."""


RATE_BISECTORS: dict[str, FixedPointBisector] = {
    "homogeneous": bisect_homogeneous,
    "beggs_brill": bisect_beggs_and_brill,
    "beggs_and_brill": bisect_beggs_and_brill,
    "hagedorn_brown": bisect_hagedorn_brown,
    "gray": bisect_gray,
    "woldesemayat_ghajar": bisect_woldesemayat_ghajar,
}
"""Rate bisection for each hydraulics correlation, by the name of its `WellBoreModel`."""

"""
Derivatives of a resolved well's rates and tubing head pressure with respect to the unknowns a
fully implicit solver iterates on: the pressure, water saturation and gas saturation of each
connected cell, and the well's own bottom-hole pressure.

A connection's rate depends on the pressure in the wellbore beside it, and that pressure depends
on the rates of the connections it is fed by, through the weight and friction of what they carry.
So every rate depends on every other, and the derivatives are those of the whole set of
equations, found by implicit differentiation of the system the fixed-point solve settles on. Only
the pressure change that a change in rate makes in the wellbore is found numerically, by
differencing the wellbore correlation itself, so that every correlation is covered. Everything
else is analytic.

The computation is compiled end to end and works in the buffers of `DerivativeScratch`, so it
allocates nothing per well beyond what the linear solve itself needs.
"""

import math
import typing

import numba
import numpy as np

from bores.errors import ValidationError
from bores.types import (
    AnyDimension,
    Boolean,
    Integer,
    Number,
    NumberArray,
    OneDimension,
    PhaseValues,
)
from bores.wells.compile import CompiledWellSystem, WellKind
from bores.wells.hydraulics.base import (
    ConnectionDerivatives,
    ConnectionFluids,
    SurfaceSample,
    WellBoreModel,
)
from bores.wells.hydraulics.beggs_and_brill import (
    walk_perforation_pressures as walk_beggs_and_brill,
)
from bores.wells.hydraulics.beggs_and_brill import (
    walk_perforation_pressures_from_table as walk_beggs_and_brill_from_table,
)
from bores.wells.hydraulics.gray import walk_perforation_pressures as walk_gray
from bores.wells.hydraulics.gray import (
    walk_perforation_pressures_from_table as walk_gray_from_table,
)
from bores.wells.hydraulics.hagedorn_brown import (
    walk_perforation_pressures as walk_hagedorn_brown,
)
from bores.wells.hydraulics.hagedorn_brown import (
    walk_perforation_pressures_from_table as walk_hagedorn_brown_from_table,
)
from bores.wells.hydraulics.homogeneous import walk_perforation_pressures as walk_homogeneous
from bores.wells.hydraulics.homogeneous import (
    walk_perforation_pressures_from_table as walk_homogeneous_from_table,
)
from bores.wells.hydraulics.hydrostatic import walk_hydrostatic_pressures
from bores.wells.hydraulics.tubing_fluids import TubingFluidTable
from bores.wells.hydraulics.vfp import VFPTables
from bores.wells.hydraulics.woldesemayat_ghajar import (
    walk_perforation_pressures as walk_woldesemayat_ghajar,
)
from bores.wells.hydraulics.woldesemayat_ghajar import (
    walk_perforation_pressures_from_table as walk_woldesemayat_ghajar_from_table,
)
from bores.wells.resolution.engine import build_well_perforation_workspace
from bores.wells.resolution.solvers import ALL_PHASES, get_fixed_point_setup, phase_mask
from bores.wells.resolution.spec import WellControlSpec
from bores.wells.resolution.tubing import (
    attach_flowing_fluid,
    compute_tubing_head_pressure,
    resolve_well_tubing,
)
from bores.wells.workspace import (
    ConnectionRates,
    PerforationWorkspace,
    WellsWorkspace,
    apply_non_darcy_correction_to_well_indices,
)

__all__ = [
    "WalkSetup",
    "WellDerivatives",
    "compute_well_derivatives",
    "get_well_derivatives",
]

WALKS: dict[str, typing.Callable[..., None]] = {
    "homogeneous": walk_homogeneous,
    "beggs_brill": walk_beggs_and_brill,
    "beggs_and_brill": walk_beggs_and_brill,
    "hagedorn_brown": walk_hagedorn_brown,
    "gray": walk_gray,
    "woldesemayat_ghajar": walk_woldesemayat_ghajar,
}
"""Pressure walk of each hydraulics correlation, by the name of its `WellBoreModel`."""

TABLE_WALKS: dict[str, typing.Callable[..., None]] = {
    "homogeneous": walk_homogeneous_from_table,
    "beggs_brill": walk_beggs_and_brill_from_table,
    "beggs_and_brill": walk_beggs_and_brill_from_table,
    "hagedorn_brown": walk_hagedorn_brown_from_table,
    "gray": walk_gray_from_table,
    "woldesemayat_ghajar": walk_woldesemayat_ghajar_from_table,
}
"""As `WALKS`, for a well whose connection pressures are read from its tubing fluid table."""

RELATIVE_STEP = 1e-5
"""Relative size of the step finite differences take through the wellbore correlation."""


class WalkSetup(typing.NamedTuple):
    """What a pressure walk needs beyond the connection rates and the reference pressure."""

    model: typing.Any
    """The correlation's options. `None` for a hydrostatic or uniform walk."""

    unit_scale: Number | None
    """The hydrostatic divisor of a correlation, or the pressure of a unit column. `None` if uniform."""

    tubing_fluid_table: TubingFluidTable | None
    """The well's tubing fluid table, for a walk that reads the fluids from it."""

    reference_depth: Number
    """Depth the reference pressure refers to."""

    is_injector: Boolean
    """Whether the well is an injector."""


@numba.njit(cache=True)
def evaluate_hydraulic_walk(
    walk,
    setup: WalkSetup,
    workspace: PerforationWorkspace,
    oil_rates: NumberArray[OneDimension],
    water_rates: NumberArray[OneDimension],
    gas_rates: NumberArray[OneDimension],
    reference_pressure: Number,
    pressures: NumberArray[OneDimension],
) -> None:
    """
    Writes the pressure at every connection into `pressures` with a hydraulics correlation's walk.

    :param walk: The correlation's `walk_perforation_pressures`.
    :param setup: How the walk is set up.
    :param workspace: The well's perforation workspace.
    :param oil_rates: Oil rate at each connection.
    :param water_rates: Water rate at each connection.
    :param gas_rates: Gas rate at each connection.
    :param reference_pressure: Pressure at the reference depth.
    :param pressures: Output, the pressure at each connection.
    """
    walk(
        model=setup.model,
        reference_depth=setup.reference_depth,
        reference_measured_depth=workspace.reference_measured_depth,
        reference_pressure=reference_pressure,
        connection_oil_rates=oil_rates,
        connection_water_rates=water_rates,
        connection_gas_rates=gas_rates,
        representative_depths=workspace.representative_depths,
        representative_measured_depths=workspace.representative_measured_depths,
        inclinations_from_vertical=workspace.inclinations_from_vertical,
        connection_fluids=workspace.connection_fluids,
        is_injector=setup.is_injector,
        hydrostatic_divisor=setup.unit_scale,
        pressures=pressures,
    )


@numba.njit(cache=True)
def evaluate_table_walk(
    walk,
    setup: WalkSetup,
    workspace: PerforationWorkspace,
    oil_rates: NumberArray[OneDimension],
    water_rates: NumberArray[OneDimension],
    gas_rates: NumberArray[OneDimension],
    reference_pressure: Number,
    pressures: NumberArray[OneDimension],
) -> None:
    """
    As `evaluate_hydraulic_walk`, with the fluids read from the well's tubing fluid table.

    :param walk: The correlation's `walk_perforation_pressures_from_table`.
    :param setup: How the walk is set up.
    :param workspace: The well's perforation workspace.
    :param oil_rates: Oil rate at each connection.
    :param water_rates: Water rate at each connection.
    :param gas_rates: Gas rate at each connection.
    :param reference_pressure: Pressure at the reference depth.
    :param pressures: Output, the pressure at each connection.
    """
    walk(
        model=setup.model,
        reference_depth=setup.reference_depth,
        reference_measured_depth=workspace.reference_measured_depth,
        reference_pressure=reference_pressure,
        connection_oil_rates=oil_rates,
        connection_water_rates=water_rates,
        connection_gas_rates=gas_rates,
        representative_depths=workspace.representative_depths,
        representative_measured_depths=workspace.representative_measured_depths,
        inclinations_from_vertical=workspace.inclinations_from_vertical,
        connection_fluids=workspace.connection_fluids,
        tubing_fluid_table=setup.tubing_fluid_table,
        is_injector=setup.is_injector,
        pressures=pressures,
    )


@numba.njit(cache=True)
def evaluate_hydrostatic_walk(
    walk,
    setup: WalkSetup,
    workspace: PerforationWorkspace,
    oil_rates: NumberArray[OneDimension],
    water_rates: NumberArray[OneDimension],
    gas_rates: NumberArray[OneDimension],
    reference_pressure: Number,
    pressures: NumberArray[OneDimension],
) -> None:
    """
    As `evaluate_hydraulic_walk`, for the weight of the fluid between connections alone.

    :param walk: `walk_hydrostatic_pressures`.
    :param setup: How the walk is set up.
    :param workspace: The well's perforation workspace.
    :param oil_rates: Oil rate at each connection.
    :param water_rates: Water rate at each connection.
    :param gas_rates: Gas rate at each connection.
    :param reference_pressure: Pressure at the reference depth.
    :param pressures: Output, the pressure at each connection.
    """
    walk(
        reference_depth=setup.reference_depth,
        reference_pressure=reference_pressure,
        representative_depths=workspace.representative_depths,
        connection_fluids=workspace.connection_fluids,
        connection_oil_rates=oil_rates,
        connection_water_rates=water_rates,
        connection_gas_rates=gas_rates,
        pressure_per_density_and_depth=setup.unit_scale,
        pressures=pressures,
    )


@numba.njit(cache=True)
def evaluate_uniform_walk(
    walk,
    setup: WalkSetup,
    workspace: PerforationWorkspace,
    oil_rates: NumberArray[OneDimension],
    water_rates: NumberArray[OneDimension],
    gas_rates: NumberArray[OneDimension],
    reference_pressure: Number,
    pressures: NumberArray[OneDimension],
) -> None:
    """
    As `evaluate_hydraulic_walk`, for the reference pressure applied unchanged at every connection.

    :param walk: Not used.
    :param setup: Not used.
    :param workspace: Not used.
    :param oil_rates: Not used.
    :param water_rates: Not used.
    :param gas_rates: Not used.
    :param reference_pressure: Pressure at every connection.
    :param pressures: Output, the pressure at each connection.
    """
    pressures[:] = reference_pressure


@numba.njit(cache=True)
def get_step(value: Number, scale: Number) -> Number:
    """
    The size of a finite-difference step for a quantity.

    :param value: The quantity's value.
    :param scale: A typical size of the quantity, so a quantity of zero still gets a step.
    :returns: The step.
    """
    return RELATIVE_STEP * max(abs(value), 1e-3 * scale, 1e-12)


@numba.njit(cache=True)
def differentiate_pressures(
    evaluate,
    walk,
    setup: WalkSetup,
    workspace: PerforationWorkspace,
    count: Integer,
    reference_pressure: Number,
    oil_rates: NumberArray[OneDimension],
    water_rates: NumberArray[OneDimension],
    gas_rates: NumberArray[OneDimension],
    shifted_oil_rates: NumberArray[OneDimension],
    shifted_water_rates: NumberArray[OneDimension],
    shifted_gas_rates: NumberArray[OneDimension],
    forward_pressures: NumberArray[OneDimension],
    backward_pressures: NumberArray[OneDimension],
    dpc_dq_buffer: NumberArray[OneDimension],
    dpc_dbhp: NumberArray[OneDimension],
) -> None:
    """
    Finds how the pressure at every connection changes with every connection rate and with the
    reference pressure, by differencing the pressure walk.

    Each rate is differenced centrally, except one too small to step down from without changing
    direction, which is differenced forward. The walk takes the nested `ConnectionFluids` of the
    workspace, which numba cannot hand to the threads of a parallel loop, so the rates are
    differenced one after another.

    :param evaluate: The adapter for the kind of walk, `evaluate_*_walk`.
    :param walk: The walk function the adapter calls.
    :param setup: How the walk is set up.
    :param workspace: The well's perforation workspace.
    :param count: Number of flowing connections.
    :param reference_pressure: The reference pressure the well was solved at.
    :param oil_rates: Buffer holding the oil rate at each connection in its first `count` entries.
    :param water_rates: As `oil_rates`, for water.
    :param gas_rates: As `oil_rates`, for gas.
    :param shifted_oil_rates: Buffer of at least `3 * count * count` entries to difference in.
    :param shifted_water_rates: As `shifted_oil_rates`.
    :param shifted_gas_rates: As `shifted_oil_rates`.
    :param forward_pressures: As `shifted_oil_rates`.
    :param backward_pressures: As `shifted_oil_rates`.
    :param dpc_dq_buffer: Output buffer, filled as shape `(count, 3 * count)`. Entry
        `(i, 3 * k + b)` is the change of pressure at connection `i` with phase `b`'s rate at
        connection `k`.
    :param dpc_dbhp: Output buffer of `count` entries: the change of pressure at
        each connection with the reference pressure.
    """
    size = 3 * count
    base_oil = oil_rates[:count]
    base_water = water_rates[:count]
    base_gas = gas_rates[:count]
    shifted_oil = shifted_oil_rates[: size * count].reshape(size, count)
    shifted_water = shifted_water_rates[: size * count].reshape(size, count)
    shifted_gas = shifted_gas_rates[: size * count].reshape(size, count)
    forward = forward_pressures[: size * count].reshape(size, count)
    backward = backward_pressures[: size * count].reshape(size, count)
    dpc_dq = dpc_dq_buffer[: count * size].reshape(count, size)

    scale = 1.0
    for index in range(count):
        scale = max(scale, abs(base_oil[index]), abs(base_water[index]), abs(base_gas[index]))

    for task in range(size):
        connection = task // 3
        phase = task % 3
        row_oil = shifted_oil[task]
        row_water = shifted_water[task]
        row_gas = shifted_gas[task]
        row_oil[:] = base_oil
        row_water[:] = base_water
        row_gas[:] = base_gas
        if phase == 0:
            value = base_oil[connection]
        elif phase == 1:
            value = base_water[connection]
        else:
            value = base_gas[connection]
        step = get_step(value=value, scale=scale)
        lower = -step if abs(value) > 2.0 * step else 0.0

        if phase == 0:
            row_oil[connection] = value + step
        elif phase == 1:
            row_water[connection] = value + step
        else:
            row_gas[connection] = value + step

        evaluate(
            walk=walk,
            setup=setup,
            workspace=workspace,
            oil_rates=row_oil,
            water_rates=row_water,
            gas_rates=row_gas,
            reference_pressure=reference_pressure,
            pressures=forward[task],
        )

        if phase == 0:
            row_oil[connection] = value + lower
        elif phase == 1:
            row_water[connection] = value + lower
        else:
            row_gas[connection] = value + lower

        evaluate(
            walk=walk,
            setup=setup,
            workspace=workspace,
            oil_rates=row_oil,
            water_rates=row_water,
            gas_rates=row_gas,
            reference_pressure=reference_pressure,
            pressures=backward[task],
        )

        denominator = step - lower
        for index in range(count):
            dpc_dq[index, task] = (forward[task, index] - backward[task, index]) / denominator

    reference_step = get_step(value=reference_pressure, scale=1.0)
    evaluate(
        walk=walk,
        setup=setup,
        workspace=workspace,
        oil_rates=base_oil,
        water_rates=base_water,
        gas_rates=base_gas,
        reference_pressure=reference_pressure + reference_step,
        pressures=forward[0],
    )
    evaluate(
        walk=walk,
        setup=setup,
        workspace=workspace,
        oil_rates=base_oil,
        water_rates=base_water,
        gas_rates=base_gas,
        reference_pressure=reference_pressure - reference_step,
        pressures=backward[0],
    )
    for index in range(count):
        dpc_dbhp[index] = (forward[0, index] - backward[0, index]) / (2.0 * reference_step)


@numba.njit(cache=True, parallel=True)
def compute_connection_partials(
    workspace: PerforationWorkspace,
    derivatives: ConnectionDerivatives,
    count: Integer,
    radial_flow_constant: Number,
    is_injector: Boolean,
    relevant_oil: Number,
    relevant_water: Number,
    relevant_gas: Number,
    oil_rates: NumberArray[OneDimension],
    water_rates: NumberArray[OneDimension],
    gas_rates: NumberArray[OneDimension],
    connection_pressures: NumberArray[OneDimension],
    well_indices: NumberArray[OneDimension],
    dq_dpc_buffer: NumberArray[OneDimension],
    dq_dx_direct_buffer: NumberArray[OneDimension],
    dq_dqg_buffer: NumberArray[OneDimension],
    dq_dq_composition_buffer: NumberArray[OneDimension],
) -> None:
    """
    Differentiates the rate each connection has directly, for the status it has in the final
    solution.

    A connection that flows with the well has `rate = k * WI * mobility * drawdown`. One that flows
    against it, on a well that allows that, has the same with the drawdown reversed, for an
    injector, and takes the produced fluid in its proportions, for a producer. One with no drawdown
    has no rate and no derivative. A well index that depends on the connection's own gas rate,
    through the non-Darcy correction, adds that dependence.

    The outputs are filled as: `dq_dpc` `(count, 3)` against the pressure in
    the wellbore beside the connection; `dq_dx_direct` `(count, 3, 3)` against the cell's
    pressure, water saturation and gas saturation; `dq_dqg` `(count, 3)` against the
    connection's own gas rate; `dq_dq_composition` `(3 * count, 3 * count)`, where entry
    `(3 * i + a, 3 * k + b)` is the change of reversed producer connection `i`'s phase `a` with the
    phase `b` rate of connection `k`.

    :param workspace: The well's perforation workspace, in double precision.
    :param derivatives: How the fluid properties at each connection change with the cell's unknowns.
    :param count: Number of flowing connections.
    :param radial_flow_constant: The unit system's radial flow constant.
    :param is_injector: Whether the well is an injector.
    :param relevant_oil: Positive if a forward connection flows oil.
    :param relevant_water: Positive if a forward connection flows water.
    :param relevant_gas: Positive if a forward connection flows gas.
    :param oil_rates: Final oil rate at each connection.
    :param water_rates: Final water rate at each connection.
    :param gas_rates: Final gas rate at each connection.
    :param connection_pressures: Final pressure at each connection.
    :param well_indices: Each connection's well index after the non-Darcy correction.
    :param dq_dpc_buffer: Output buffer.
    :param dq_dx_direct_buffer: Output buffer.
    :param dq_dqg_buffer: Output buffer.
    :param dq_dq_composition_buffer: Output buffer.
    """
    fluids = workspace.connection_fluids
    cell_pressures = fluids.pressures
    oil_mobilities = fluids.oil_mobilities
    water_mobilities = fluids.water_mobilities
    gas_mobilities = fluids.gas_mobilities
    conductivities = workspace.connection_conductivities
    d_factor = workspace.d_factor
    allows_crossflow = workspace.allows_crossflow
    dʎo_dp = derivatives.dʎo_dp
    dʎo_dsw = derivatives.dʎo_dsw
    dʎo_dsg = derivatives.dʎo_dsg
    dʎw_dp = derivatives.dʎw_dp
    dʎw_dsw = derivatives.dʎw_dsw
    dʎw_dsg = derivatives.dʎw_dsg
    dʎg_dp = derivatives.dʎg_dp
    dʎg_dsw = derivatives.dʎg_dsw
    dʎg_dsg = derivatives.dʎg_dsg
    size = 3 * count
    sign = -1.0 if is_injector else 1.0
    relevant = (relevant_oil > 0.0, relevant_water > 0.0, relevant_gas > 0.0)
    has_non_darcy = not math.isnan(d_factor)

    dq_dpc_buffer[: count * 3] = 0.0
    dq_dx_direct_buffer[: count * 9] = 0.0
    dq_dqg_buffer[: count * 3] = 0.0
    dq_dq_composition_buffer[: size * size] = 0.0
    dq_dpc = dq_dpc_buffer[: count * 3].reshape(count, 3)
    dq_dx_direct = dq_dx_direct_buffer[: count * 9].reshape(count, 3, 3)
    dq_dqg = dq_dqg_buffer[: count * 3].reshape(count, 3)
    dq_dq_composition = dq_dq_composition_buffer[: size * size].reshape(size, size)

    forward_oil = 0.0
    forward_water = 0.0
    forward_gas = 0.0
    for index in range(count):
        if sign * (cell_pressures[index] - connection_pressures[index]) > 0.0:
            forward_oil += oil_rates[index]
            forward_water += water_rates[index]
            forward_gas += gas_rates[index]

    forward_total = forward_oil + forward_water + forward_gas
    forward_fractions = (
        (forward_oil / forward_total, forward_water / forward_total, forward_gas / forward_total)
        if forward_total > 0.0
        else (0.0, 0.0, 0.0)
    )

    for index in numba.prange(count):
        signed = sign * (cell_pressures[index] - connection_pressures[index])
        if signed == 0:
            continue
        mobilities = (
            oil_mobilities[index],
            water_mobilities[index],
            gas_mobilities[index],
        )
        dm = (
            dʎo_dp[index],
            dʎo_dsw[index],
            dʎo_dsg[index],
            dʎw_dp[index],
            dʎw_dsw[index],
            dʎw_dsg[index],
            dʎg_dp[index],
            dʎg_dsw[index],
            dʎg_dsg[index],
        )
        factor = radial_flow_constant * well_indices[index]
        dwi_dqg = 0.0
        conductivity = conductivities[index]
        if has_non_darcy and not math.isnan(conductivity):
            gas_direction = (
                1.0 if gas_rates[index] > 0.0 else (-1.0 if gas_rates[index] < 0.0 else 0.0)
            )
            dwi_dqg = -(well_indices[index] ** 2) * d_factor * gas_direction / conductivity

        if signed > 0.0:
            for phase in range(3):
                if not relevant[phase]:
                    continue
                mobility = mobilities[phase]
                dq_dpc[index, phase] = -factor * mobility * sign
                dq_dx_direct[index, phase, 0] = factor * (
                    dm[3 * phase + 0] * signed + mobility * sign
                )
                dq_dx_direct[index, phase, 1] = factor * dm[3 * phase + 1] * signed
                dq_dx_direct[index, phase, 2] = factor * dm[3 * phase + 2] * signed
                dq_dqg[index, phase] = radial_flow_constant * mobility * signed * dwi_dqg
        elif allows_crossflow:
            reverse_drawdown = -signed
            if is_injector:
                for phase in range(3):
                    mobility = mobilities[phase]
                    dq_dpc[index, phase] = factor * mobility
                    dq_dx_direct[index, phase, 0] = -factor * (
                        dm[3 * phase + 0] * reverse_drawdown + mobility
                    )
                    dq_dx_direct[index, phase, 1] = -factor * dm[3 * phase + 1] * reverse_drawdown
                    dq_dx_direct[index, phase, 2] = -factor * dm[3 * phase + 2] * reverse_drawdown
                    dq_dqg[index, phase] = (
                        -radial_flow_constant * mobility * reverse_drawdown * dwi_dqg
                    )
            elif forward_total > 0.0:
                total_mobility = mobilities[0] + mobilities[1] + mobilities[2]
                total_pressure_derivative = dm[0] + dm[3] + dm[6]
                total_water_derivative = dm[1] + dm[4] + dm[7]
                total_gas_derivative = dm[2] + dm[5] + dm[8]
                reverse_rate = factor * total_mobility * reverse_drawdown
                for phase in range(3):
                    fraction = forward_fractions[phase]
                    dq_dpc[index, phase] = -factor * total_mobility * fraction
                    dq_dx_direct[index, phase, 0] = (
                        -factor
                        * (total_pressure_derivative * reverse_drawdown - total_mobility)
                        * fraction
                    )
                    dq_dx_direct[index, phase, 1] = (
                        -factor * total_water_derivative * reverse_drawdown * fraction
                    )
                    dq_dx_direct[index, phase, 2] = (
                        -factor * total_gas_derivative * reverse_drawdown * fraction
                    )
                    dq_dqg[index, phase] = (
                        -radial_flow_constant
                        * total_mobility
                        * reverse_drawdown
                        * dwi_dqg
                        * fraction
                    )
                    for source in range(count):
                        if sign * (cell_pressures[source] - connection_pressures[source]) > 0.0:
                            for source_phase in range(3):
                                dq_dq_composition[3 * index + phase, 3 * source + source_phase] = (
                                    -reverse_rate
                                    * ((1.0 if phase == source_phase else 0.0) - fraction)
                                    / forward_total
                                )


@numba.njit(cache=True)
def solve_linear_system(
    system: NumberArray[AnyDimension], right_hand_side: NumberArray[AnyDimension]
) -> NumberArray[AnyDimension]:
    """
    Solves `system @ x = right_hand_side`, in the least-squares sense if `system` is singular.

    :param system: Square matrix.
    :param right_hand_side: One column per right-hand side.
    :returns: The solution, all `NaN` if either input has a value that is not finite.
    """
    if not (np.isfinite(system).all() and np.isfinite(right_hand_side).all()):
        return np.full(right_hand_side.shape, np.nan)
    try:
        solution = np.linalg.solve(system, right_hand_side)
    except Exception:
        solution = np.linalg.lstsq(system, right_hand_side)[0]
    return solution


@numba.njit(cache=True, parallel=True)
def solve_rate_derivatives(
    *,
    count: Integer,
    dq_dpc: NumberArray[OneDimension],
    dq_dx_direct_buffer: NumberArray[OneDimension],
    dq_dqg: NumberArray[OneDimension],
    dq_dq_composition_buffer: NumberArray[OneDimension],
    dpc_dq_buffer: NumberArray[OneDimension],
    dpc_dbhp: NumberArray[OneDimension],
    system_buffer: NumberArray[OneDimension],
    right_hand_side_buffer: NumberArray[OneDimension],
    dq_dx_buffer: NumberArray[OneDimension],
    dq_dbhp: NumberArray[OneDimension],
) -> None:
    """
    Solves for the total derivatives of the connection rates, by implicit differentiation of
    `rates = F(rates, cell unknowns, bottom-hole pressure)`.

    The rates enter `F` directly through the non-Darcy correction and the reversed producer
    connections' composition, and through the pressure in the wellbore, which the rates set. The
    derivatives are those of `(I - dF/drates) * d(rates) = dF/d(cell unknowns) +
    dF/d(bottom-hole pressure)`.

    :param count: Number of flowing connections.
    :param dq_dpc: As `compute_connection_partials` fills it.
    :param dq_dx_direct_buffer: As `compute_connection_partials` fills it.
    :param dq_dqg: As `compute_connection_partials` fills it.
    :param dq_dq_composition_buffer: As `compute_connection_partials` fills it.
    :param dpc_dq_buffer: As `differentiate_pressures` fills it.
    :param dpc_dbhp: As `differentiate_pressures` fills it.
    :param system_buffer: Buffer of at least `(3 * count) ** 2` entries.
    :param right_hand_side_buffer: Buffer of at least `3 * count * (3 * count + 1)` entries.
    :param dq_dx_buffer: Output buffer, filled as shape `(3 * count, 3 * count)`: the
        connection rates against the cell unknowns.
    :param dq_dbhp: Output buffer of `3 * count` entries: the connection rates
        against the bottom-hole pressure.
    """
    size = 3 * count
    dpc_dq = dpc_dq_buffer[: count * size].reshape(count, size)
    dq_dq_composition = dq_dq_composition_buffer[: size * size].reshape(size, size)
    dq_dx_direct = dq_dx_direct_buffer[: 3 * size]
    system = system_buffer[: size * size].reshape(size, size)
    right_hand_side = right_hand_side_buffer[: size * (size + 1)].reshape(size, size + 1)

    for row in numba.prange(size):
        index = row // 3
        dq_dpc_row = dq_dpc[row]
        for column in range(size):
            system[row, column] = (
                -dq_dq_composition[row, column] - dq_dpc_row * dpc_dq[index, column]
            )
            right_hand_side[row, column] = 0.0
        system[row, row] += 1.0
        system[row, 3 * index + 2] -= dq_dqg[row]
        right_hand_side[row, size] = dq_dpc_row * dpc_dbhp[index]
        for unknown in range(3):
            right_hand_side[row, 3 * index + unknown] = dq_dx_direct[3 * row + unknown]

    solution = solve_linear_system(system=system, right_hand_side=right_hand_side)

    dq_dx = dq_dx_buffer[: size * size].reshape(size, size)
    for row in numba.prange(size):
        for column in range(size):
            dq_dx[row, column] = solution[row, column]
        dq_dbhp[row] = solution[row, size]


@numba.njit(cache=True, parallel=True)
def convert_to_surface(
    *,
    workspace: PerforationWorkspace,
    derivatives: ConnectionDerivatives,
    count: Integer,
    is_injector: Boolean,
    oil_rates: NumberArray[OneDimension],
    water_rates: NumberArray[OneDimension],
    gas_rates: NumberArray[OneDimension],
    dq_dx_buffer: NumberArray[OneDimension],
    dq_dbhp: NumberArray[OneDimension],
    dqs_dx_buffer: NumberArray[OneDimension],
    dqs_dbhp: NumberArray[OneDimension],
) -> None:
    """
    Turns the derivatives of the reservoir-condition connection rates into those of the
    surface-condition component rates: stock-tank oil, stock-tank water and standard gas.

    A producing connection brings the gas its oil holds in solution and the oil its gas carries as
    vapor to the surface. So does an injector's connection that flows against the well. The
    formation volume factors, `Rs` and `Rv` change with the cell's pressure, which adds to the
    derivative against that pressure.

    :param workspace: The well's perforation workspace, in double precision.
    :param derivatives: How the fluid properties at each connection change with the cell's unknowns.
    :param count: Number of flowing connections.
    :param is_injector: Whether the well is an injector.
    :param oil_rates: Final oil rate at each connection.
    :param water_rates: Final water rate at each connection.
    :param gas_rates: Final gas rate at each connection.
    :param dq_dx_buffer: The connection rates against the cell unknowns, as `(3n, 3n)`.
    :param dq_dbhp: The connection rates against the bottom-hole pressure.
    :param dqs_dx_buffer: Output buffer, filled as `(3n, 3n)`.
    :param dqs_dbhp: Output buffer of `3n` entries.
    """
    fluids = workspace.connection_fluids
    oil_volume_factors = fluids.oil_formation_volume_factors
    water_volume_factors = fluids.water_formation_volume_factors
    gas_volume_factors = fluids.gas_formation_volume_factors
    solution_ratios = workspace.dissolved_gas_ratios
    vaporized_ratios = workspace.vaporized_oil_ratios
    oil_volume_factor_slopes = derivatives.dbo_dp
    water_volume_factor_slopes = derivatives.dbw_dp
    gas_volume_factor_slopes = derivatives.dbg_dp
    solution_ratio_slopes = derivatives.drs_dp
    vaporized_ratio_slopes = derivatives.drv_dp
    size = 3 * count
    factor = workspace.stb_to_volume_factor
    dq_dx = dq_dx_buffer[: size * size].reshape(size, size)
    dqs_dx = dqs_dx_buffer[: size * size].reshape(size, size)

    for index in numba.prange(count):
        oil_rate = oil_rates[index]
        water_rate = water_rates[index]
        gas_rate = gas_rates[index]
        oil_volume_factor = oil_volume_factors[index]
        water_volume_factor = water_volume_factors[index]
        gas_volume_factor = gas_volume_factors[index]
        solution_ratio = solution_ratios[index]
        vaporized_ratio = vaporized_ratios[index]
        carries = (
            1.0
            if (not is_injector) or oil_rate < 0.0 or water_rate < 0.0 or gas_rate < 0.0
            else 0.0
        )

        oil_to_oil = 1.0 / (oil_volume_factor * factor)
        gas_to_oil = carries * vaporized_ratio / gas_volume_factor
        water_to_water = 1.0 / (water_volume_factor * factor)
        oil_to_gas = carries * solution_ratio / (oil_volume_factor * factor)
        gas_to_gas = 1.0 / gas_volume_factor

        surface_oil_base = oil_rate * oil_to_oil
        free_gas_base = gas_rate * gas_to_gas
        dqo_base_dp = -oil_rate * oil_volume_factor_slopes[index] / (oil_volume_factor**2 * factor)
        dqg_base_dp = -gas_rate * gas_volume_factor_slopes[index] / gas_volume_factor**2
        dqs_o_dp_explicit = dqo_base_dp + carries * (
            vaporized_ratio_slopes[index] * free_gas_base + vaporized_ratio * dqg_base_dp
        )
        dqs_w_dp_explicit = (
            -water_rate * water_volume_factor_slopes[index] / (water_volume_factor**2 * factor)
        )
        dqs_g_dp_explicit = dqg_base_dp + carries * (
            solution_ratio_slopes[index] * surface_oil_base + solution_ratio * dqo_base_dp
        )

        oil_row = 3 * index
        for column in range(size):
            dqs_dx[oil_row, column] = (
                oil_to_oil * dq_dx[oil_row, column] + gas_to_oil * dq_dx[oil_row + 2, column]
            )
            dqs_dx[oil_row + 1, column] = water_to_water * dq_dx[oil_row + 1, column]
            dqs_dx[oil_row + 2, column] = (
                oil_to_gas * dq_dx[oil_row, column] + gas_to_gas * dq_dx[oil_row + 2, column]
            )
        dqs_dx[oil_row, oil_row] += dqs_o_dp_explicit
        dqs_dx[oil_row + 1, oil_row] += dqs_w_dp_explicit
        dqs_dx[oil_row + 2, oil_row] += dqs_g_dp_explicit
        dqs_dbhp[oil_row] = oil_to_oil * dq_dbhp[oil_row] + gas_to_oil * dq_dbhp[oil_row + 2]
        dqs_dbhp[oil_row + 1] = water_to_water * dq_dbhp[oil_row + 1]
        dqs_dbhp[oil_row + 2] = oil_to_gas * dq_dbhp[oil_row] + gas_to_gas * dq_dbhp[oil_row + 2]


@numba.njit(cache=True, parallel=True)
def write_well_derivatives(
    *,
    count: Integer,
    active_rows: NumberArray[OneDimension],
    start: Integer,
    n_compiled: Integer,
    well_row: Integer,
    block_start: Integer,
    dq_dx_buffer: NumberArray[OneDimension],
    dq_dbhp: NumberArray[OneDimension],
    dqs_dx_buffer: NumberArray[OneDimension],
    dqs_dbhp: NumberArray[OneDimension],
    dq_dx_connection: NumberArray[OneDimension],
    dqs_dx_connection: NumberArray[OneDimension],
    dq_dbhp_connection: NumberArray[AnyDimension],
    dqs_dbhp_connection: NumberArray[AnyDimension],
    dq_dx_well: NumberArray[AnyDimension],
    dqs_dx_well: NumberArray[AnyDimension],
    dq_dbhp_well: NumberArray[AnyDimension],
    dqs_dbhp_well: NumberArray[AnyDimension],
) -> None:
    """
    Writes a well's derivatives into its place in the workspace arrays, and sums them into the
    derivatives of the well's own total rates.

    The well's compiled connections that are not flowing get zero rows and columns.

    :param count: Number of flowing connections.
    :param active_rows: The flowing connections' rows in the compiled system, in order.
    :param start: The well's first connection row in the compiled system.
    :param n_compiled: The number of connections the well has, flowing or not.
    :param well_row: The well's row.
    :param block_start: Where the well's block starts in the flat block arrays.
    :param dq_dx_buffer: Scratch `(3 * count, 3 * count)`: connection rates against cells.
    :param dq_dbhp: Scratch `(3 * count,)`: connection rates against the BHP.
    :param dqs_dx_buffer: Scratch `(3 * count, 3 * count)`: surface rates against cells.
    :param dqs_dbhp: Scratch `(3 * count,)`: surface rates against the BHP.
    :param dq_dx_connection: Workspace array written.
    :param dqs_dx_connection: Workspace array written.
    :param dq_dbhp_connection: Workspace array written.
    :param dqs_dbhp_connection: Workspace array written.
    :param dq_dx_well: Workspace array written.
    :param dqs_dx_well: Workspace array written.
    :param dq_dbhp_well: Workspace array written.
    :param dqs_dbhp_well: Workspace array written.
    """
    size = 3 * count
    full_size = 3 * n_compiled
    dq_dx = dq_dx_buffer[: size * size].reshape(size, size)
    dqs_dx = dqs_dx_buffer[: size * size].reshape(size, size)
    block_end = block_start + full_size * full_size

    dq_dx_connection[block_start:block_end] = 0.0
    dqs_dx_connection[block_start:block_end] = 0.0
    dq_dbhp_connection[start : start + n_compiled] = 0.0
    dqs_dbhp_connection[start : start + n_compiled] = 0.0
    dq_dx_well[start : start + n_compiled] = 0.0
    dqs_dx_well[start : start + n_compiled] = 0.0

    for index in numba.prange(count):
        local_row = active_rows[index] - start
        for phase in range(3):
            row = 3 * index + phase
            for other in range(count):
                local_column = active_rows[other] - start
                for unknown in range(3):
                    offset = block_start + (3 * local_row + phase) * full_size + 3 * local_column
                    dq_dx_connection[offset + unknown] = dq_dx[row, 3 * other + unknown]
                    dqs_dx_connection[offset + unknown] = dqs_dx[row, 3 * other + unknown]
            dq_dbhp_connection[active_rows[index], phase] = dq_dbhp[row]
            dqs_dbhp_connection[active_rows[index], phase] = dqs_dbhp[row]

    for other in numba.prange(count):
        for phase in range(3):
            for unknown in range(3):
                total = 0.0
                surface_total = 0.0
                for index in range(count):
                    total += dq_dx[3 * index + phase, 3 * other + unknown]
                    surface_total += dqs_dx[3 * index + phase, 3 * other + unknown]
                dq_dx_well[active_rows[other], phase, unknown] = total
                dqs_dx_well[active_rows[other], phase, unknown] = surface_total

    for phase in range(3):
        total = 0.0
        surface_total = 0.0
        for index in range(count):
            total += dq_dbhp[3 * index + phase]
            surface_total += dqs_dbhp[3 * index + phase]
        dq_dbhp_well[well_row, phase] = total
        dqs_dbhp_well[well_row, phase] = surface_total


class WellDerivatives(typing.NamedTuple):
    """
    One well's derivatives, as views of its place in a `WellsWorkspace`.

    Phases are ordered oil, water, gas, and a cell's unknowns pressure, water saturation, gas
    saturation. The connections are all of the well's compiled connections, in order, and a
    connection that is not flowing has zero rows and columns.
    """

    dq_dx: NumberArray[AnyDimension]
    """Shape `(3 * n, 3 * n)`. Reservoir-condition connection rates against every cell's unknowns."""

    dqs_dx: NumberArray[AnyDimension]
    """Shape `(3 * n, 3 * n)`. Surface-condition component rates against every cell's unknowns."""

    dq_dbhp: NumberArray[AnyDimension]
    """Shape `(n, 3)`. Reservoir-condition connection rates against the bottom-hole pressure."""

    dqs_dbhp: NumberArray[AnyDimension]
    """Shape `(n, 3)`. Surface-condition component rates against the bottom-hole pressure."""

    dq_dx_well: NumberArray[AnyDimension]
    """Shape `(n, 3, 3)`. The well's total reservoir-condition rates against each cell's unknowns."""

    dqs_dx_well: NumberArray[AnyDimension]
    """Shape `(n, 3, 3)`. The well's total surface-condition rates against each cell's unknowns."""

    dq_dbhp_well: NumberArray[AnyDimension]
    """Shape `(3,)`. The well's total reservoir-condition rates against its bottom-hole pressure."""

    dqs_dbhp_well: NumberArray[AnyDimension]
    """Shape `(3,)`. The well's total surface-condition rates against its bottom-hole pressure."""

    dthp_dbhp: Number
    """The well's tubing head pressure against its bottom-hole pressure. `NaN` if it has none."""

    dthp_dx: NumberArray[AnyDimension]
    """Shape `(n, 3)`. The well's tubing head pressure against each cell's unknowns."""


def get_well_derivatives(
    *, compiled_system: CompiledWellSystem, workspace: WellsWorkspace, well_row: Integer
) -> WellDerivatives:
    """
    Reads one well's derivatives from a `WellsWorkspace`.

    :param compiled_system: The compiled well system `workspace` was resolved against.
    :param workspace: A workspace whose derivatives have been computed.
    :param well_row: The well's row in `compiled_system`.
    :returns: Views of the well's derivatives, which change if the workspace is solved again.
    :raises ValidationError: If the workspace was built without room for derivatives.
    """
    perforations = compiled_system.perforations
    start, end = perforations.well_offsets[well_row], perforations.well_offsets[well_row + 1]
    count = int(end - start)
    block_start = workspace.derivative_offsets[well_row]
    block_end = workspace.derivative_offsets[well_row + 1]
    if block_end - block_start != (3 * count) ** 2:
        raise ValidationError(
            "This `WellsWorkspace` has no room for well derivatives. Build it with `well_offsets`."
        )
    size = 3 * count
    return WellDerivatives(
        dq_dx=workspace.dq_dx_connection[block_start:block_end].reshape(size, size),
        dqs_dx=workspace.dqs_dx_connection[block_start:block_end].reshape(size, size),
        dq_dbhp=workspace.dq_dbhp_connection[start:end],
        dqs_dbhp=workspace.dqs_dbhp_connection[start:end],
        dq_dx_well=workspace.dq_dx_well[start:end],
        dqs_dx_well=workspace.dqs_dx_well[start:end],
        dq_dbhp_well=workspace.dq_dbhp_well[well_row],
        dqs_dbhp_well=workspace.dqs_dbhp_well[well_row],
        dthp_dbhp=workspace.dthp_dbhp[well_row],
        dthp_dx=workspace.dthp_dx[start:end],
    )


def differentiate_tubing_head_pressure(
    *,
    tubing: typing.Any,
    reference_depth: Number,
    reference_pressure: Number,
    relevant_phases: PhaseValues,
    is_injector: Boolean,
    surface_sample: SurfaceSample | None,
    rates: tuple[Number, Number, Number],
    surface_rates: tuple[Number, Number, Number],
    connection_rates: ConnectionRates,
) -> tuple[Number, list[Number], list[Number], list[Number]]:
    """
    Differentiates the tubing head pressure against what it is computed from: the bottom-hole
    pressure, the well's reservoir-condition and surface-condition rates, and, for a VFP table
    whose datum depth differs from the reference depth, the rate at each connection.

    The tubing head pressure is found through VFP tables and tubing fluid tables, which are not
    compiled, so this is evaluated directly. It takes seven evaluations, and with a VFP table
    three more per connection.

    :param tubing: The well's `WellTubing`.
    :param reference_depth: Depth the bottom-hole pressure refers to.
    :param reference_pressure: The bottom-hole pressure.
    :param relevant_phases: The phases the tubing head pressure is computed with the rates of:
        all for a producer, the injected phase for an injector.
    :param is_injector: Whether the well is an injector.
    :param surface_sample: Surface fluid properties, where the tubing head pressure needs them.
    :param rates: The well's reservoir-condition oil, water and gas rates.
    :param surface_rates: The well's surface-condition oil, water and gas rates.
    :param connection_rates: The reservoir-condition rates at each connection. Perturbed and
        restored in place for a VFP table.
    :returns: `(dthp_dbhp, dthp_dq, dthp_dqs_rates, d_connection_rates)`. The connection derivatives
        are flat, ordered connection by connection, oil, water, gas.
    """
    mask = (relevant_phases.oil > 0.0, relevant_phases.water > 0.0, relevant_phases.gas > 0.0)
    reservoir = list(rates)
    surface = list(surface_rates)
    arrays = (connection_rates.oil, connection_rates.water, connection_rates.gas)

    def evaluate(*, pressure: Number) -> Number:
        return compute_tubing_head_pressure(
            tubing=tubing,
            reference_depth=reference_depth,
            reference_pressure=pressure,
            rates=PhaseValues(*(reservoir[phase] if mask[phase] else 0.0 for phase in range(3))),
            surface_rates=PhaseValues(
                *(surface[phase] if mask[phase] else 0.0 for phase in range(3))
            ),
            surface_sample=surface_sample,
            is_injector=is_injector,
            connection_rates=connection_rates,
        )

    def difference(
        target: list[Number] | NumberArray[OneDimension], position: int, scale: Number
    ) -> Number:
        value = target[position]
        step = get_step(value=value, scale=scale)
        if abs(value) > 2.0 * step:
            target[position] = value + step
            upper = evaluate(pressure=reference_pressure)
            target[position] = value - step
            lower = evaluate(pressure=reference_pressure)
            target[position] = value
            return (upper - lower) / (2.0 * step)
        target[position] = value + step
        upper = evaluate(pressure=reference_pressure)
        target[position] = value
        return (upper - evaluate(pressure=reference_pressure)) / step

    bhp_step = get_step(value=reference_pressure, scale=1.0)
    dthp_dbhp = (
        evaluate(pressure=reference_pressure + bhp_step)
        - evaluate(pressure=reference_pressure - bhp_step)
    ) / (2.0 * bhp_step)
    rate_scale = max(max(abs(value) for value in rates), 1.0)
    surface_scale = max(max(abs(value) for value in surface_rates), 1.0)
    dthp_dq = [
        difference(reservoir, phase, rate_scale) if mask[phase] else 0.0 for phase in range(3)
    ]
    dthp_dqs = [
        difference(surface, phase, surface_scale) if mask[phase] else 0.0 for phase in range(3)
    ]
    dthp_dq_connection: list[Number] = []
    if tubing.vfp_table is not None:
        for connection in range(arrays[0].shape[0]):
            for phase in range(3):
                dthp_dq_connection.append(difference(arrays[phase], connection, rate_scale))
    return dthp_dbhp, dthp_dq, dthp_dqs, dthp_dq_connection


def compute_well_derivatives(
    *,
    compiled_system: CompiledWellSystem,
    well_row: Integer,
    tubing_performance: WellBoreModel | VFPTables | None,
    connection_fluids: ConnectionFluids,
    connection_derivatives: ConnectionDerivatives,
    workspace: WellsWorkspace,
    spec: WellControlSpec,
    surface_sample: SurfaceSample | None = None,
) -> None:
    """
    Computes one resolved well's derivatives and writes them into `workspace`, in the well's place
    in the arrays that `WellsWorkspace` describes.

    Nothing here assumes a constant mobility, a constant well index or a pressure profile that does
    not respond to the rates. A change in the pressure of any connection's cell, or in the well's
    bottom-hole pressure, changes the rate of every connection, because the pressure in the
    wellbore beside each depends on the rates of the others. Those derivatives are solved for
    together. A well that is shut, shut in by an economic limit, or that has no flowing
    connection has zero derivatives, and one that is not yet in effect is left as it is.

    The well must have been resolved into `workspace` already. The derivatives are those of the
    solution it holds, at the pressure and saturations the connection fluids describe.

    :param compiled_system: The compiled well system.
    :param well_row: Which well, by row index into `compiled_system`.
    :param tubing_performance: What relates the well's BHP to its THP, as given to
        `resolve_well_control`.
    :param connection_fluids: The reservoir conditions the well was resolved with.
    :param connection_derivatives: How those conditions change with the cells' unknowns, for the
        same connections in the same order.
    :param workspace: The workspace the well was resolved into. Its derivative arrays are written.
    :param spec: The tunables the well was resolved with.
    :param surface_sample: The surface fluid properties the well was resolved with, if it needed
        them.
    :raises ValidationError: If the workspace was built without room for derivatives, or
        `connection_derivatives` does not have one entry per flowing connection.
    """
    if compiled_system.schedule_statuses[well_row] == 0:
        return

    perforations = compiled_system.perforations
    start, end = (
        int(perforations.well_offsets[well_row]),
        int(perforations.well_offsets[well_row + 1]),
    )
    n_compiled = end - start
    block_start = int(workspace.derivative_offsets[well_row])
    block_end = int(workspace.derivative_offsets[well_row + 1])
    if block_end - block_start != (3 * n_compiled) ** 2:
        raise ValidationError(
            "This `WellsWorkspace` has no room for well derivatives. Build it with `well_offsets`."
        )

    bhp = float(workspace.bhps[well_row])
    active_open = compiled_system.get_flowing_connection_rows(well_row=well_row)
    count = len(active_open)
    if count == 0 or math.isnan(bhp) or workspace.economic_shutins[well_row]:
        workspace.dq_dx_connection[block_start:block_end] = 0.0
        workspace.dqs_dx_connection[block_start:block_end] = 0.0
        workspace.dq_dbhp_connection[start:end] = 0.0
        workspace.dqs_dbhp_connection[start:end] = 0.0
        workspace.dq_dx_well[start:end] = 0.0
        workspace.dqs_dx_well[start:end] = 0.0
        workspace.dq_dbhp_well[well_row] = 0.0
        workspace.dqs_dbhp_well[well_row] = 0.0
        workspace.dthp_dbhp[well_row] = math.nan
        workspace.dthp_dx[start:end] = math.nan
        return
    if len(connection_derivatives.dʎo_dp) != count:
        raise ValidationError(
            f"`connection_derivatives` has {len(connection_derivatives.dʎo_dp)} "
            f"connections but the well has {count} flowing connections. They must match 1:1."
        )

    controls = compiled_system.controls
    is_injector = bool(controls.well_kinds[well_row] == WellKind.INJECTOR)
    relevant_phases = phase_mask(controls.injected_phases[well_row]) if is_injector else ALL_PHASES
    reference_depth = compiled_system.reference_depths[well_row]

    perforation_workspace = build_well_perforation_workspace(
        compiled_system=compiled_system,
        well_row=well_row,
        active_open=active_open,
        connection_fluids=connection_fluids,
        dtype=np.float64,
    )
    derivatives = ConnectionDerivatives(
        *(
            typing.cast("NumberArray[OneDimension]", np.asarray(array, dtype=np.float64))
            for array in connection_derivatives
        )
    )
    tubing = attach_flowing_fluid(
        tubing=resolve_well_tubing(
            tubing_performance=tubing_performance,
            compiled_system=compiled_system,
            well_row=well_row,
        ),
        workspace=perforation_workspace,
        relevant_phases=relevant_phases,
    )
    fixed_point = get_fixed_point_setup(tubing=tubing, spec=spec)
    wellbore_name = tubing.wellbore.name if tubing.wellbore is not None else ""
    if fixed_point.kind == "uniform":
        evaluate, walk = evaluate_uniform_walk, walk_hydrostatic_pressures
    elif fixed_point.kind == "hydrostatic":
        evaluate, walk = evaluate_hydrostatic_walk, walk_hydrostatic_pressures
    elif fixed_point.tubing_fluid_table is None:
        evaluate, walk = evaluate_hydraulic_walk, WALKS[wellbore_name]
    else:
        evaluate, walk = evaluate_table_walk, TABLE_WALKS[wellbore_name]

    setup = WalkSetup(
        model=fixed_point.model,
        unit_scale=fixed_point.unit_scale,
        tubing_fluid_table=fixed_point.tubing_fluid_table,
        reference_depth=reference_depth,
        is_injector=is_injector,
    )

    scratch = workspace.derivative_scratch
    np.take(
        workspace.connection_oil_rates, active_open, out=scratch.oil_rates[:count], mode="clip"
    )
    np.take(
        workspace.connection_water_rates, active_open, out=scratch.water_rates[:count], mode="clip"
    )
    np.take(
        workspace.connection_gas_rates, active_open, out=scratch.gas_rates[:count], mode="clip"
    )
    np.take(
        workspace.connection_pressures,
        active_open,
        out=scratch.connection_pressures[:count],
        mode="clip",
    )
    apply_non_darcy_correction_to_well_indices(
        static_well_indices=perforation_workspace.static_well_indices,
        connection_conductivities=perforation_workspace.connection_conductivities,
        connection_gas_rates=typing.cast("NumberArray[OneDimension]", scratch.gas_rates[:count]),
        d_factor=perforation_workspace.d_factor,
        out_well_indices=typing.cast("NumberArray[OneDimension]", scratch.well_indices[:count]),
    )

    differentiate_pressures(
        evaluate=evaluate,
        walk=walk,
        setup=setup,
        workspace=perforation_workspace,
        count=count,
        reference_pressure=bhp,
        oil_rates=scratch.oil_rates,
        water_rates=scratch.water_rates,
        gas_rates=scratch.gas_rates,
        shifted_oil_rates=scratch.shifted_oil_rates,
        shifted_water_rates=scratch.shifted_water_rates,
        shifted_gas_rates=scratch.shifted_gas_rates,
        forward_pressures=scratch.forward_pressures,
        backward_pressures=scratch.backward_pressures,
        dpc_dq_buffer=scratch.dpc_dq,
        dpc_dbhp=scratch.dpc_dbhp,
    )
    compute_connection_partials(
        workspace=perforation_workspace,
        derivatives=derivatives,
        count=count,
        radial_flow_constant=spec.radial_flow_constant,
        is_injector=is_injector,
        relevant_oil=relevant_phases.oil,
        relevant_water=relevant_phases.water,
        relevant_gas=relevant_phases.gas,
        oil_rates=scratch.oil_rates,
        water_rates=scratch.water_rates,
        gas_rates=scratch.gas_rates,
        connection_pressures=scratch.connection_pressures,
        well_indices=scratch.well_indices,
        dq_dpc_buffer=scratch.dq_dpc,
        dq_dx_direct_buffer=scratch.dq_dx_direct,
        dq_dqg_buffer=scratch.dq_dqg,
        dq_dq_composition_buffer=scratch.dq_dq_composition,
    )
    solve_rate_derivatives(
        count=count,
        dq_dpc=scratch.dq_dpc,
        dq_dx_direct_buffer=scratch.dq_dx_direct,
        dq_dqg=scratch.dq_dqg,
        dq_dq_composition_buffer=scratch.dq_dq_composition,
        dpc_dq_buffer=scratch.dpc_dq,
        dpc_dbhp=scratch.dpc_dbhp,
        system_buffer=scratch.system,
        right_hand_side_buffer=scratch.right_hand_side,
        dq_dx_buffer=scratch.dq_dx,
        dq_dbhp=scratch.dq_dbhp,
    )
    convert_to_surface(
        workspace=perforation_workspace,
        derivatives=derivatives,
        count=count,
        is_injector=is_injector,
        oil_rates=scratch.oil_rates,
        water_rates=scratch.water_rates,
        gas_rates=scratch.gas_rates,
        dq_dx_buffer=scratch.dq_dx,
        dq_dbhp=scratch.dq_dbhp,
        dqs_dx_buffer=scratch.dqs_dx,
        dqs_dbhp=scratch.dqs_dbhp,
    )
    write_well_derivatives(
        count=count,
        active_rows=active_open,
        start=start,
        n_compiled=n_compiled,
        well_row=well_row,
        block_start=block_start,
        dq_dx_buffer=scratch.dq_dx,
        dq_dbhp=scratch.dq_dbhp,
        dqs_dx_buffer=scratch.dqs_dx,
        dqs_dbhp=scratch.dqs_dbhp,
        dq_dx_connection=workspace.dq_dx_connection,
        dqs_dx_connection=workspace.dqs_dx_connection,
        dq_dbhp_connection=workspace.dq_dbhp_connection,
        dqs_dbhp_connection=workspace.dqs_dbhp_connection,
        dq_dx_well=workspace.dq_dx_well,
        dqs_dx_well=workspace.dqs_dx_well,
        dq_dbhp_well=workspace.dq_dbhp_well,
        dqs_dbhp_well=workspace.dqs_dbhp_well,
    )

    workspace.dthp_dbhp[well_row] = math.nan
    workspace.dthp_dx[start:end] = math.nan
    if math.isnan(workspace.thps[well_row]):
        return

    size = 3 * count
    dq_dx = scratch.dq_dx[: size * size].reshape(size, size)
    dq_dbhp = scratch.dq_dbhp[:size]
    well_rates = (
        workspace.oil_rates[well_row],
        workspace.water_rates[well_row],
        workspace.gas_rates[well_row],
    )
    well_surface_rates = (
        workspace.surface_oil_rates[well_row],
        workspace.surface_water_rates[well_row],
        workspace.surface_gas_rates[well_row],
    )
    dthp_dbhp_direct, dthp_dq, dthp_dqs, dthp_dq_connection = differentiate_tubing_head_pressure(
        tubing=tubing,
        reference_depth=reference_depth,
        reference_pressure=bhp,
        relevant_phases=relevant_phases,
        is_injector=is_injector,
        surface_sample=surface_sample,
        rates=well_rates,
        surface_rates=well_surface_rates,
        connection_rates=ConnectionRates(
            oil=typing.cast("NumberArray[OneDimension]", scratch.oil_rates[:count]),
            water=typing.cast("NumberArray[OneDimension]", scratch.water_rates[:count]),
            gas=typing.cast("NumberArray[OneDimension]", scratch.gas_rates[:count]),
        ),
    )
    dq_dx_well = workspace.dq_dx_well[active_open]
    dqs_dx_well = workspace.dqs_dx_well[active_open]
    dthp_dbhp = (
        dthp_dbhp_direct
        + np.dot(dthp_dq, workspace.dq_dbhp_well[well_row])  # type: ignore[arg-type]
        + np.dot(dthp_dqs, workspace.dqs_dbhp_well[well_row])  # type: ignore[arg-type]
    )
    dthp_dx = np.einsum("a,kav->kv", dthp_dq, dq_dx_well) + np.einsum(  # type: ignore
        "a,kav->kv",
        dthp_dqs,  # type: ignore
        dqs_dx_well,
    )
    if dthp_dq_connection:
        gradient = np.asarray(dthp_dq_connection)
        dthp_dbhp += float(gradient @ dq_dbhp)
        dthp_dx += (gradient @ dq_dx).reshape(count, 3)
    workspace.dthp_dbhp[well_row] = dthp_dbhp
    workspace.dthp_dx[active_open] = dthp_dx

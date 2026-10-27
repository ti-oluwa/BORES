"""Compiled (buffer-reuse) structures for the well-control resolution."""

import math
import typing

import numba
import numpy as np
import numpy.typing as npt

from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.types import Boolean, IntArray, Integer, Number, NumberArray, OneDimension
from bores.utils import none_if_nan
from bores.wells.base import Wells
from bores.wells.compile import UNSET_INT, CompiledWellSystem
from bores.wells.decompile import decompile_limit, decompile_perforations, decompile_well_control
from bores.wells.hydraulics.base import ConnectionFluids
from bores.wells.state import (
    PerforationState,
    PhaseValues,
    WellState,
    WellStates,
)

__all__ = [
    "ConnectionRates",
    "PerforationWorkspace",
    "WellsWorkspace",
    "accumulate_rates",
    "accumulate_well_volumes",
    "apply_non_darcy_correction_to_well_indices",
    "build_perforation_workspace",
    "build_wells_workspace",
    "compute_perforation_drawdown",
    "get_connection_rates",
    "load_wells_states",
    "restore_connection_rates",
]


class PerforationWorkspace(typing.NamedTuple):
    """
    One well's active-connection data for the resolution hot path.

    Built once per well per control resolution call, and reused across every
    fixed-point and bisection iteration within that call.

    The per-connection values never change mid-resolution, and `connection_pressures`
    is a scratch buffer callers overwrite in place on every iteration.
    The other arrays are read-only for the duration of a control resolution call.
    """

    well_indices: NumberArray[OneDimension]
    """
    This well's active connections' connection factors, at the current
    fixed-point iteration. Equal to `static_well_indices` unless a
    non-Darcy correction is active, in which case this is overwritten in
    place every iteration from `static_well_indices` and the previous
    iteration's gas rate.
    """

    static_well_indices: NumberArray[OneDimension]
    """
    This well's active connections' connection factors, at zero rate.
    Never mutated during resolution. The non-Darcy correction recomputes
    `well_indices` from this array every iteration rather than
    compounding a previous correction.
    """

    connection_conductivities: NumberArray[OneDimension]
    """
    This well's active connections' Peaceman numerator, isolated from
    skin/geometry. `NaN` at a connection with an overridden well index,
    where there's no such decomposition. Feeds the non-Darcy correction;
    a `NaN` entry leaves that connection's well index at its static value.
    """

    d_factor: Number
    """This well's non-Darcy flow coefficient. `NaN` disables the correction."""

    connection_fluids: ConnectionFluids
    """Matching reservoir conditions at each connection."""

    representative_depths: NumberArray[OneDimension]
    """Representative depth of each connection, for hydrostatic pressure correction."""

    representative_measured_depths: NumberArray[OneDimension]
    """Representative measured depth of each connection along the wellbore, for friction."""

    reference_measured_depth: Number
    """Measured depth along the wellbore of this well's reference depth."""

    inclinations_from_vertical: NumberArray[OneDimension]
    """Inclination of each connection from vertical, for hydrostatic pressure correction."""

    connection_pressures: NumberArray[OneDimension]
    """Buffer for the flowing pressure at each connection, at this well's current BHP. Overwritten on every call."""

    connection_oil_rates: NumberArray[OneDimension]
    """
    Buffer for the reservoir-condition oil rate at each connection, 
    at this well's current BHP. Overwritten on every call.
    """

    connection_water_rates: NumberArray[OneDimension]
    """
    Buffer for the reservoir-condition water rate at each connection, 
    at this well's current BHP. Overwritten on every call.
    """

    connection_gas_rates: NumberArray[OneDimension]
    """
    Buffer for the reservoir-condition gas rate at each connection, 
    at this well's current BHP. Overwritten on every call.
    """

    dissolved_gas_ratios: NumberArray[OneDimension]
    """
    Gas each connection's oil holds in solution, per volume of oil at surface conditions, in the
    volume unit that surface oil rates are in. Multiplying a connection's surface oil rate by it
    gives the gas that oil brings to the surface in solution.
    """


def build_perforation_workspace(
    well_indices: NumberArray[OneDimension],
    representative_depths: NumberArray[OneDimension],
    representative_measured_depths: NumberArray[OneDimension],
    inclinations_from_vertical: NumberArray[OneDimension],
    connection_fluids: ConnectionFluids,
    reference_measured_depth: Number,
    connection_conductivities: NumberArray[OneDimension] | None = None,
    d_factor: Number = math.nan,
    stb_to_volume_factor: Number = 1.0,
    dtype: npt.DTypeLike = None,
) -> PerforationWorkspace:
    """
    Builds a `PerforationWorkspace` for one well.

    :param well_indices: This well's active connections' connection
        factors, at zero rate. Becomes `static_well_indices`.
    :param representative_depths: Matching depths, same order as `well_indices`.
    :param representative_measured_depths: Matching measured depths along the wellbore, same
        order as `well_indices`.
    :param inclinations_from_vertical: Matching inclinations, same order as `well_indices`.
    :param connection_fluids: Matching reservoir conditions, same order as `well_indices`.
    :param reference_measured_depth: Measured depth along the wellbore of this well's reference
        depth.
    :param connection_conductivities: Matching Peaceman numerators, same
        order as `well_indices`, `NaN` at an overridden connection. All
        `NaN` (the default) if this well has no non-Darcy correction to apply.
    :param d_factor: This well's non-Darcy flow coefficient. `NaN`
        (the default) disables the correction.
    :param stb_to_volume_factor: Volume of one stock-tank barrel in the volume unit of surface
        rates, which turns solution gas-oil ratios into gas per volume of surface oil. One
        (the default) if the ratios are already per volume of surface oil.
    :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
    :returns: `PerforationWorkspace` for this well.
    """
    resolved_dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    n = len(well_indices)
    resolved_fluids = ConnectionFluids(
        *(np.asarray(array, dtype=resolved_dtype) for array in connection_fluids)  # type: ignore[arg-type]
    )
    if len(resolved_fluids.pressures) != n:
        raise ValidationError(
            f"`connection_fluids` has {len(resolved_fluids.pressures)} connections but "
            f"`well_indices` has {n}; they must match 1:1."
        )

    static_well_indices = typing.cast(
        NumberArray[OneDimension], np.asarray(well_indices, dtype=resolved_dtype)
    )
    resolved_connection_conductivities = (
        np.full(n, np.nan, dtype=resolved_dtype)
        if connection_conductivities is None
        else np.asarray(connection_conductivities, dtype=resolved_dtype)
    )
    return PerforationWorkspace(
        well_indices=static_well_indices.copy(),
        static_well_indices=static_well_indices,
        connection_conductivities=typing.cast(
            NumberArray[OneDimension], resolved_connection_conductivities
        ),
        d_factor=d_factor,
        connection_fluids=resolved_fluids,
        representative_depths=typing.cast(
            NumberArray[OneDimension],
            np.asarray(representative_depths, dtype=resolved_dtype),
        ),
        representative_measured_depths=typing.cast(
            NumberArray[OneDimension],
            np.asarray(representative_measured_depths, dtype=resolved_dtype),
        ),
        reference_measured_depth=reference_measured_depth,
        inclinations_from_vertical=typing.cast(
            NumberArray[OneDimension],
            np.asarray(inclinations_from_vertical, dtype=resolved_dtype),
        ),
        connection_pressures=typing.cast(
            NumberArray[OneDimension], np.empty(n, dtype=resolved_dtype)
        ),
        connection_oil_rates=typing.cast(
            NumberArray[OneDimension], np.empty(n, dtype=resolved_dtype)
        ),
        connection_water_rates=typing.cast(
            NumberArray[OneDimension], np.empty(n, dtype=resolved_dtype)
        ),
        connection_gas_rates=typing.cast(
            NumberArray[OneDimension], np.zeros(n, dtype=resolved_dtype)
        ),
        dissolved_gas_ratios=typing.cast(
            NumberArray[OneDimension],
            (resolved_fluids.solution_gas_oil_ratios / stb_to_volume_factor).astype(
                resolved_dtype, copy=False
            ),
        ),
    )


class WellsWorkspace(typing.NamedTuple):
    """Every well's control-resolution result, one row per well."""

    bhps: NumberArray[OneDimension]
    """Shape `(n_wells,)`. `NaN` for a well not yet resolved this pass."""

    oil_rates: NumberArray[OneDimension]
    """Shape `(n_wells,)`. Reservoir-condition oil phase rate."""

    water_rates: NumberArray[OneDimension]
    """Shape `(n_wells,)`. Reservoir-condition water phase rate."""

    gas_rates: NumberArray[OneDimension]
    """Shape `(n_wells,)`. Reservoir-condition gas phase rate."""

    surface_oil_rates: NumberArray[OneDimension]
    """Shape `(n_wells,)`. Surface-condition oil phase rate."""

    surface_water_rates: NumberArray[OneDimension]
    """Shape `(n_wells,)`. Surface-condition water phase rate."""

    surface_gas_rates: NumberArray[OneDimension]
    """Shape `(n_wells,)`. Surface-condition gas phase rate."""

    thps: NumberArray[OneDimension]
    """Shape `(n_wells,)`. `NaN` where not computed."""

    active_limit_rows: IntArray[OneDimension]
    """
    Shape `(n_wells,)`. Row index into that well's slice of
    `CompiledLimits` identifying the currently-binding limit; `UNSET_INT`
    if none is binding.
    """

    economic_shutins: IntArray[OneDimension]
    """
    Shape `(n_wells,)`. `1` if an `EconomicLimit` shut this well in
    this pass, `0` otherwise.
    """

    connection_pressures: NumberArray[OneDimension]
    """
    Shape `(n_connections,)`, CSR-indexed by well the same way
    `CompiledPerforations`' own arrays are (use the same `well_offsets`
    to slice out one well's rows). Flowing pressure at each active
    connection, at that well's final governing BHP. `NaN` for a
    connection whose well hasn't been resolved this pass.

    Not zeroed for an economically shut-in well, as a shut well still has a
    real wellbore pressure profile at zero flow, only its rates are zero.
    """

    connection_oil_rates: NumberArray[OneDimension]
    """
    Shape `(n_connections,)` each, same CSR indexing as `connection_pressures`.
    Each active connection's own reservoir-condition oil phase rate, at that
    well's final governing BHP. Zeroed (not `NaN`) for a connection
    belonging to an economically shut-in well, matching `oil_rates`'s own zeroing. 
    `NaN` means "not resolved this pass", not "resolved to zero".
    """

    connection_water_rates: NumberArray[OneDimension]
    """
    Shape `(n_connections,)` each, same CSR indexing as `connection_pressures`.
    Each active connection's own reservoir-condition water phase rate, at that
    well's final governing BHP. Zeroed (not `NaN`) for a connection
    belonging to an economically shut-in well, matching `water_rates`'s own zeroing. 
    `NaN` means "not resolved this pass", not "resolved to zero".
    """

    connection_gas_rates: NumberArray[OneDimension]
    """
    Shape `(n_connections,)` each, same CSR indexing as `connection_pressures`.
    Each active connection's own reservoir-condition gas phase rate, at that
    well's final governing BHP. Zeroed (not `NaN`) for a connection
    belonging to an economically shut-in well, matching `gas_rates`'s own zeroing. 
    `NaN` means "not resolved this pass", not "resolved to zero".
    """

    cumulative_oil_volumes: NumberArray[OneDimension]
    """
    Shape `(n_wells,)`. Surface-condition oil volume each well has produced
    or injected since the start of the run. Never `NaN`. Always double
    precision, whatever the run's working precision, so a small step's volume is
    not lost against a large running total. Advanced by `accumulate_well_volumes`.
    """

    cumulative_water_volumes: NumberArray[OneDimension]
    """Shape `(n_wells,)`. Same as `cumulative_oil_volumes`, for water."""

    cumulative_gas_volumes: NumberArray[OneDimension]
    """Shape `(n_wells,)`. Same as `cumulative_oil_volumes`, for gas."""


def build_wells_workspace(
    *, n_wells: Integer, n_connections: Integer, dtype: npt.DTypeLike = None
) -> WellsWorkspace:
    """
    Builds an empty `WellsWorkspace` for a system of `n_wells` wells with
    `n_connections` active connections in total. To be called once at the start
    of a run. Every `resolve_control` call across every timestep updates rows in
    this same object in place.

    :param n_wells: Number of wells.
    :param n_connections: Total active connections across every well,
        matching `CompiledPerforations`' own row count. Sizes `connection_pressures`.
    :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
    :returns: `WellsWorkspace` with every row `NaN`/`UNSET_INT`/`0`.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    return WellsWorkspace(
        bhps=np.full(n_wells, np.nan, dtype=dtype),
        oil_rates=np.full(n_wells, np.nan, dtype=dtype),
        water_rates=np.full(n_wells, np.nan, dtype=dtype),
        gas_rates=np.full(n_wells, np.nan, dtype=dtype),
        surface_oil_rates=np.full(n_wells, np.nan, dtype=dtype),
        surface_water_rates=np.full(n_wells, np.nan, dtype=dtype),
        surface_gas_rates=np.full(n_wells, np.nan, dtype=dtype),
        thps=np.full(n_wells, np.nan, dtype=dtype),
        active_limit_rows=np.full(n_wells, UNSET_INT, dtype=np.int32),
        economic_shutins=np.zeros(n_wells, dtype=np.int32),
        connection_pressures=np.full(n_connections, np.nan, dtype=dtype),
        connection_oil_rates=np.full(n_connections, np.nan, dtype=dtype),
        connection_water_rates=np.full(n_connections, np.nan, dtype=dtype),
        connection_gas_rates=np.full(n_connections, np.nan, dtype=dtype),
        cumulative_oil_volumes=np.zeros(n_wells, dtype=np.float64),
        cumulative_water_volumes=np.zeros(n_wells, dtype=np.float64),
        cumulative_gas_volumes=np.zeros(n_wells, dtype=np.float64),
    )


class ConnectionRates(typing.NamedTuple):
    """Reservoir-condition rate of each phase at every one of a well's connections."""

    oil: NumberArray[OneDimension]
    """Oil rate at each connection."""

    water: NumberArray[OneDimension]
    """Water rate at each connection."""

    gas: NumberArray[OneDimension]
    """Gas rate at each connection."""


def get_connection_rates(
    *, workspace: PerforationWorkspace, copy: bool = False
) -> ConnectionRates:
    """
    The per-connection rates the most recent solve left in a well's `PerforationWorkspace`.

    The workspace's rate arrays are overwritten by every solve, so the values only stay valid
    until the next one. Take a copy to keep them past it.

    :param workspace: The well's perforation workspace.
    :param copy: Whether to copy the arrays instead of returning the workspace's own.
    :returns: The connection rates.
    """
    rates = ConnectionRates(
        oil=workspace.connection_oil_rates,
        water=workspace.connection_water_rates,
        gas=workspace.connection_gas_rates,
    )
    if copy:
        return ConnectionRates(
            oil=rates.oil.copy(), water=rates.water.copy(), gas=rates.gas.copy()
        )
    return rates


def restore_connection_rates(*, workspace: PerforationWorkspace, rates: ConnectionRates) -> None:
    """
    Writes saved per-connection rates back into a well's `PerforationWorkspace`.

    :param workspace: The well's perforation workspace.
    :param rates: Rates saved with `get_connection_rates(..., copy=True)`.
    """
    workspace.connection_oil_rates[:] = rates.oil
    workspace.connection_water_rates[:] = rates.water
    workspace.connection_gas_rates[:] = rates.gas


def accumulate_well_volumes(*, workspace: WellsWorkspace, step_size: Number) -> None:
    """
    Adds one accepted time step's surface volumes to every well's running totals, in place.

    Each well's rate from the step just resolved is taken as constant over
    the whole step, which is exactly how an implicit step applies it. Call
    once per accepted step, after well control resolution and never for a
    rejected attempt, so totals stay exact no matter how rarely summaries
    are recorded. A well with a `NaN` rate (not yet resolved) adds nothing.

    :param workspace: The run's wells workspace.
    :param step_size: The accepted step's size, in the run's time unit.
    """
    for rates, volumes in (
        (workspace.surface_oil_rates, workspace.cumulative_oil_volumes),
        (workspace.surface_water_rates, workspace.cumulative_water_volumes),
        (workspace.surface_gas_rates, workspace.cumulative_gas_volumes),
    ):
        np.add(volumes, rates * step_size, out=volumes, where=~np.isnan(rates))


@numba.njit(cache=True)
def compute_perforation_drawdown(
    reservoir_pressure: Number, connection_pressure: Number, is_injector: Boolean
) -> Number:
    """
    Computes one connection's drawdown: the pressure difference driving
    flow at that connection.

    :param reservoir_pressure: Reservoir pressure at the connection.
    :param connection_pressure: Flowing wellbore pressure at the connection.
    :param is_injector: Whether this well is an injector.
    :returns: The driving pressure difference, clipped to `>= 0` so a
        connection with reversed drawdown contributes zero flow rather
        than crossflow.
    """
    if is_injector:
        drawdown = connection_pressure - reservoir_pressure
    else:
        drawdown = reservoir_pressure - connection_pressure
    return max(drawdown, 0.0)


@numba.njit(cache=True, parallel=True)
def apply_non_darcy_correction_to_well_indices(
    static_well_indices: NumberArray[OneDimension],
    connection_conductivities: NumberArray[OneDimension],
    connection_gas_rates: NumberArray[OneDimension],
    d_factor: Number,
    out_well_indices: NumberArray[OneDimension],
) -> None:
    """
    Corrects each connection's well index in place for non-Darcy (rate-dependent) skin.

    Recomputes from `static_well_indices` every call rather than
    compounding a previous correction, using `connection_gas_rates` from
    the previous fixed-point iteration (lagged, avoiding an implicit
    solve). A connection with no conductivity decomposition (`NaN`) is
    left at its static well index, as is every connection when `d_factor`
    itself is `NaN`.

    :param static_well_indices: Each connection's well index at zero rate.
    :param connection_conductivities: Each connection's Peaceman
        numerator, `NaN` at an overridden connection.
    :param connection_gas_rates: Each connection's own reservoir-condition
        gas rate from the previous iteration.
    :param d_factor: This well's non-Darcy flow coefficient.
    :param out_well_indices: Buffer to write the corrected well index
        into, same shape as `static_well_indices`. May be
        `static_well_indices` itself only when the caller no longer needs
        the uncorrected value.
    """
    if math.isnan(d_factor):
        out_well_indices[:] = static_well_indices
        return

    for idx in numba.prange(static_well_indices.shape[0]):
        conductivity = connection_conductivities[idx]
        if math.isnan(conductivity):
            out_well_indices[idx] = static_well_indices[idx]
            continue

        out_well_indices[idx] = 1.0 / (
            1.0 / static_well_indices[idx]
            + d_factor * abs(connection_gas_rates[idx]) / conductivity
        )


@numba.njit(cache=True)
def accumulate_rates(
    connection_pressures: NumberArray[OneDimension],
    well_indices: NumberArray[OneDimension],
    reservoir_pressures: NumberArray[OneDimension],
    oil_mobilities: NumberArray[OneDimension],
    water_mobilities: NumberArray[OneDimension],
    gas_mobilities: NumberArray[OneDimension],
    oil_formation_volume_factors: NumberArray[OneDimension],
    water_formation_volume_factors: NumberArray[OneDimension],
    gas_formation_volume_factors: NumberArray[OneDimension],
    dissolved_gas_ratios: NumberArray[OneDimension],
    relevant_oil: Boolean,
    relevant_water: Boolean,
    relevant_gas: Boolean,
    is_injector: Boolean,
    radial_flow_constant: Number,
    out_connection_oil_rates: NumberArray[OneDimension],
    out_connection_water_rates: NumberArray[OneDimension],
    out_connection_gas_rates: NumberArray[OneDimension],
) -> tuple[Number, Number, Number, Number, Number, Number]:
    """
    Sums each relevant phase's reservoir-condition and surface-condition
    rate across every connection in a `PerforationWorkspace`, at a given
    set of connection pressures.

    Also writes each connection's own reservoir-condition contribution
    into the `out_connection_*` buffers, in the same pass.
    These feed the segmented hydraulics walk's `connection_phase_rates`,
    which needs each connection's individual rate rather than only the well total.

    :param connection_pressures: Flowing pressure at each connection.
    :param well_indices: `PerforationWorkspace.well_indices`.
    :param reservoir_pressures: `PerforationWorkspace.reservoir_pressures`.
    :param oil_mobilities: `PerforationWorkspace.oil_mobilities`.
    :param water_mobilities: `PerforationWorkspace.water_mobilities`.
    :param gas_mobilities: `PerforationWorkspace.gas_mobilities`.
    :param oil_formation_volume_factors: `PerforationWorkspace.oil_formation_volume_factors`.
    :param water_formation_volume_factors: `PerforationWorkspace.water_formation_volume_factors`.
    :param gas_formation_volume_factors: `PerforationWorkspace.gas_formation_volume_factors`.
    :param dissolved_gas_ratios: `PerforationWorkspace.dissolved_gas_ratios`.
    :param relevant_oil: Whether oil counts toward the primary target.
    :param relevant_water: Whether water counts toward the primary target.
    :param relevant_gas: Whether gas counts toward the primary target.
    :param is_injector: Whether this well is an injector.
    :param radial_flow_constant: `WellControlSpec.radial_flow_constant` -
        converts `well_index * mobility * drawdown` (`compute_peaceman_well_index`'s
        own units, which have no 2*pi built in) into a real reservoir-condition
        rate in this simulation's own unit system.
    :param out_connection_oil_rates: Written in place with each
        connection's own reservoir-condition oil rate (`0.0` if
        `relevant_oil` is `False`). `PerforationWorkspace.connection_oil_rates`.
    :param out_connection_water_rates: Water analogue of `out_connection_oil_rates`.
    :param out_connection_gas_rates: Gas analogue of `out_connection_oil_rates`.
    :returns: `(oil_rate, water_rate, gas_rate, surface_oil_rate,
        surface_water_rate, surface_gas_rate)` - reservoir- and
        surface-condition well totals. A non-relevant phase's rate is `0.0`. The surface gas rate
        of a producer is the total at the surface: the gas that flows free at the connections plus
        the gas that its oil carries in solution. The reservoir-condition gas rate is the free
        gas only.
    """
    # TODO: We can actually use prange here. Will it benefit us. If it will do it.
    # We may need to change the way we accumulate total rates. May be switch to using an array?
    oil_rate = 0.0
    water_rate = 0.0
    gas_rate = 0.0
    surface_oil_rate = 0.0
    surface_water_rate = 0.0
    surface_gas_rate = 0.0

    for idx in range(well_indices.shape[0]):
        drawdown = compute_perforation_drawdown(
            reservoir_pressure=reservoir_pressures[idx],
            connection_pressure=connection_pressures[idx],
            is_injector=is_injector,
        )
        well_index = well_indices[idx]

        if relevant_oil:
            contribution = radial_flow_constant * well_index * oil_mobilities[idx] * drawdown
            oil_rate += contribution
            surface_oil_contribution = contribution / oil_formation_volume_factors[idx]
            surface_oil_rate += surface_oil_contribution
            if not is_injector:
                surface_gas_rate += surface_oil_contribution * dissolved_gas_ratios[idx]
            out_connection_oil_rates[idx] = contribution
        else:
            out_connection_oil_rates[idx] = 0.0

        if relevant_water:
            contribution = radial_flow_constant * well_index * water_mobilities[idx] * drawdown
            water_rate += contribution
            surface_water_rate += contribution / water_formation_volume_factors[idx]
            out_connection_water_rates[idx] = contribution
        else:
            out_connection_water_rates[idx] = 0.0

        if relevant_gas:
            contribution = radial_flow_constant * well_index * gas_mobilities[idx] * drawdown
            gas_rate += contribution
            surface_gas_rate += contribution / gas_formation_volume_factors[idx]
            out_connection_gas_rates[idx] = contribution
        else:
            out_connection_gas_rates[idx] = 0.0

    return (
        oil_rate,
        water_rate,
        gas_rate,
        surface_oil_rate,
        surface_water_rate,
        surface_gas_rate,
    )


def load_wells_states(
    wells: Wells, compiled_system: CompiledWellSystem, workspace: WellsWorkspace
) -> WellStates:
    """
    Load `WellStates` from a resolved `WellsWorkspace`.

    Only covers wells actually resolved this pass/step. A well whose
    `WellStatus` is still `PENDING` (its BHP is left `NaN` by
    `resolve_control`) is skipped rather than reported with meaningless
    values. A well that was resolved as shut, because it is shut or has no open
    connection in effect, is reported with `is_open=False`, no perforation states, zero
    rates, and a `NaN` bottom-hole pressure, since it has no flowing pressure.

    :param wells: The original rich `Wells` this system was compiled
        from. This supplies each `PerforationState.perforation`, which the
        compiled layer doesn't retain a reference to.
    :param compiled_system: The system `workspace` was resolved against.
    :param workspace: A `WellsWorkspace` from a completed resolve pass.
    :returns: One `WellState` per resolved well, keyed by well name.
    """
    controls = compiled_system.controls
    perforations = compiled_system.perforations
    unit_system = compiled_system.unit_system

    states: dict[str, WellState] = {}
    for well_row, well_name in enumerate(compiled_system.names):
        bhp = workspace.bhps[well_row]
        # A well resolved as shut flowed nothing, which `record_shut_well` writes as zero rates
        # with no flowing pressure. A well never resolved has no rates either.
        is_shut = math.isnan(bhp) and not math.isnan(workspace.oil_rates[well_row])
        if math.isnan(bhp) and not is_shut:
            continue  # Not resolved this pass (PENDING, or UNSET control)

        row_start = perforations.well_offsets[well_row]
        rich_perforations = decompile_perforations(wells, well_name, perforations, well_row)

        perforation_states = []
        for row in range(row_start, perforations.well_offsets[well_row + 1]):
            pressure = workspace.connection_pressures[row]
            if math.isnan(pressure):
                continue  # this connection wasn't active this pass (shut or pending)

            perforation_states.append(
                PerforationState(
                    perforation=rich_perforations[row - row_start],
                    cell_index=perforations.cell_indices[row],
                    flowing_pressure=pressure,
                    rates=PhaseValues(
                        oil=workspace.connection_oil_rates[row],
                        water=workspace.connection_water_rates[row],
                        gas=workspace.connection_gas_rates[row],
                    ),
                    unit_system=unit_system,
                )
            )

        active_limit_row = workspace.active_limit_rows[well_row]
        active_limit = (
            None
            if active_limit_row == UNSET_INT
            else decompile_limit(controls, active_limit_row, unit_system)
        )

        states[well_name] = WellState(
            well_name=well_name,
            is_open=not is_shut and not workspace.economic_shutins[well_row],
            active_control=decompile_well_control(controls, well_row, unit_system),
            bhp=bhp,
            perforation_states=tuple(perforation_states),
            rates=PhaseValues(
                oil=workspace.oil_rates[well_row],
                water=workspace.water_rates[well_row],
                gas=workspace.gas_rates[well_row],
            ),
            surface_rates=PhaseValues(
                oil=workspace.surface_oil_rates[well_row],
                water=workspace.surface_water_rates[well_row],
                gas=workspace.surface_gas_rates[well_row],
            ),
            active_limit=active_limit,
            thp=none_if_nan(workspace.thps[well_row]),
            unit_system=unit_system,
        )

    return WellStates(states=states, unit_system=unit_system)

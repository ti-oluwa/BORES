"""Compiled (buffer-reuse) structures for the well-control resolution hot path."""

import math
import typing

import numba
import numpy as np
import numpy.typing as npt

from bores.precision import get_dtype
from bores.types import Boolean, IntArray, Integer, Number, NumberArray, OneDimension
from bores.utils import none_if_nan
from bores.wells.base import Wells
from bores.wells.compile import UNSET_INT, CompiledWellSystem
from bores.wells.decompile import decompile_limit, decompile_perforations, decompile_well_control
from bores.wells.states import (
    ConnectionSample,
    PerforationState,
    PhaseValues,
    WellsStates,
    WellState,
)

__all__ = [
    "PerforationWorkspace",
    "WellsWorkspace",
    "accumulate_phase_rates",
    "build_connection_phase_rates",
    "build_perforation_workspace",
    "build_wells_workspace",
    "compute_perforation_drawdown",
    "correct_well_indices_for_non_darcy",
    "load_wells_states",
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

    reservoir_pressures: NumberArray[OneDimension]
    """Matching reservoir pressure at each connection."""

    oil_mobilities: NumberArray[OneDimension]
    """Matching reservoir-condition oil mobility at each connection."""

    water_mobilities: NumberArray[OneDimension]
    """Matching reservoir-condition water mobility at each connection."""

    gas_mobilities: NumberArray[OneDimension]
    """Matching reservoir-condition gas mobility at each connection."""

    oil_formation_volume_factors: NumberArray[OneDimension]
    """Matching oil formation volume factor at each connection."""

    water_formation_volume_factors: NumberArray[OneDimension]
    """Matching water formation volume factor at each connection."""

    gas_formation_volume_factors: NumberArray[OneDimension]
    """Matching gas formation volume factor at each connection."""

    representative_depths: NumberArray[OneDimension]
    """Representative depth of each connection, for hydrostatic pressure correction."""

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


def build_perforation_workspace(
    well_indices: NumberArray[OneDimension],
    representative_depths: NumberArray[OneDimension],
    inclinations_from_vertical: NumberArray[OneDimension],
    connection_samples: typing.Sequence[ConnectionSample],
    connection_conductivities: NumberArray[OneDimension] | None = None,
    d_factor: Number = math.nan,
    dtype: npt.DTypeLike = None,
) -> PerforationWorkspace:
    """
    Builds a `PerforationWorkspace` for one well.

    :param well_indices: This well's active connections' connection
        factors, at zero rate. Becomes `static_well_indices`.
    :param representative_depths: Matching depths, same order as `well_indices`.
    :param inclinations_from_vertical: Matching inclinations, same order as `well_indices`.
    :param connection_samples: Matching reservoir conditions, same order as `well_indices`.
    :param connection_conductivities: Matching Peaceman numerators, same
        order as `well_indices`, `NaN` at an overridden connection. All
        `NaN` (the default) if this well has no non-Darcy correction to apply.
    :param d_factor: This well's non-Darcy flow coefficient. `NaN`
        (the default) disables the correction.
    :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
    :returns: `PerforationWorkspace` for this well.
    """
    resolved_dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    n = len(well_indices)
    reservoir_pressures = np.empty(n, dtype=resolved_dtype)
    oil_mobilities = np.empty(n, dtype=resolved_dtype)
    water_mobilities = np.empty(n, dtype=resolved_dtype)
    gas_mobilities = np.empty(n, dtype=resolved_dtype)
    oil_fvf = np.empty(n, dtype=resolved_dtype)
    water_fvf = np.empty(n, dtype=resolved_dtype)
    gas_fvf = np.empty(n, dtype=resolved_dtype)

    for i, sample in enumerate(connection_samples):
        reservoir_pressures[i] = sample.pressure
        oil_mobilities[i] = sample.phase_mobilities.oil
        water_mobilities[i] = sample.phase_mobilities.water
        gas_mobilities[i] = sample.phase_mobilities.gas
        oil_fvf[i] = sample.phase_fvfs.oil
        water_fvf[i] = sample.phase_fvfs.water
        gas_fvf[i] = sample.phase_fvfs.gas

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
        reservoir_pressures=reservoir_pressures,
        oil_mobilities=oil_mobilities,
        water_mobilities=water_mobilities,
        gas_mobilities=gas_mobilities,
        oil_formation_volume_factors=oil_fvf,
        water_formation_volume_factors=water_fvf,
        gas_formation_volume_factors=gas_fvf,
        representative_depths=typing.cast(
            NumberArray[OneDimension],
            np.asarray(representative_depths, dtype=resolved_dtype),
        ),
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
    )


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


@numba.njit(cache=True)
def correct_well_indices_for_non_darcy(
    static_well_indices: NumberArray[OneDimension],
    connection_conductivities: NumberArray[OneDimension],
    connection_gas_rates: NumberArray[OneDimension],
    d_factor: Number,
    out_well_indices: NumberArray[OneDimension],
) -> None:
    """
    Corrects each connection's well index in place for non-Darcy
    (rate-dependent) skin.

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
    for i in range(static_well_indices.shape[0]):
        conductivity = connection_conductivities[i]
        if math.isnan(conductivity):
            out_well_indices[i] = static_well_indices[i]
            continue
        out_well_indices[i] = 1.0 / (
            1.0 / static_well_indices[i]
            + d_factor * abs(connection_gas_rates[i]) / conductivity
        )


@numba.njit(cache=True)
def accumulate_phase_rates(
    connection_pressures: NumberArray[OneDimension],
    well_indices: NumberArray[OneDimension],
    reservoir_pressures: NumberArray[OneDimension],
    oil_mobilities: NumberArray[OneDimension],
    water_mobilities: NumberArray[OneDimension],
    gas_mobilities: NumberArray[OneDimension],
    oil_formation_volume_factors: NumberArray[OneDimension],
    water_formation_volume_factors: NumberArray[OneDimension],
    gas_formation_volume_factors: NumberArray[OneDimension],
    relevant_oil: Boolean,
    relevant_water: Boolean,
    relevant_gas: Boolean,
    is_injector: Boolean,
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
    :param relevant_oil: Whether oil counts toward the primary target.
    :param relevant_water: Whether water counts toward the primary target.
    :param relevant_gas: Whether gas counts toward the primary target.
    :param is_injector: Whether this well is an injector.
    :param out_connection_oil_rates: Written in place with each
        connection's own reservoir-condition oil rate (`0.0` if
        `relevant_oil` is `False`). `PerforationWorkspace.connection_oil_rates`.
    :param out_connection_water_rates: Water analogue of `out_connection_oil_rates`.
    :param out_connection_gas_rates: Gas analogue of `out_connection_oil_rates`.
    :returns: `(oil_rate, water_rate, gas_rate, surface_oil_rate,
        surface_water_rate, surface_gas_rate)` - reservoir- and
        surface-condition well totals. A non-relevant phase's rate is `0.0`.
    """
    oil_rate = 0.0
    water_rate = 0.0
    gas_rate = 0.0
    surface_oil_rate = 0.0
    surface_water_rate = 0.0
    surface_gas_rate = 0.0

    for i in range(well_indices.shape[0]):
        drawdown = compute_perforation_drawdown(
            reservoir_pressure=reservoir_pressures[i],
            connection_pressure=connection_pressures[i],
            is_injector=is_injector,
        )
        well_index = well_indices[i]

        if relevant_oil:
            contribution = well_index * oil_mobilities[i] * drawdown
            oil_rate += contribution
            surface_oil_rate += contribution / oil_formation_volume_factors[i]
            out_connection_oil_rates[i] = contribution
        else:
            out_connection_oil_rates[i] = 0.0

        if relevant_water:
            contribution = well_index * water_mobilities[i] * drawdown
            water_rate += contribution
            surface_water_rate += contribution / water_formation_volume_factors[i]
            out_connection_water_rates[i] = contribution
        else:
            out_connection_water_rates[i] = 0.0

        if relevant_gas:
            contribution = well_index * gas_mobilities[i] * drawdown
            gas_rate += contribution
            surface_gas_rate += contribution / gas_formation_volume_factors[i]
            out_connection_gas_rates[i] = contribution
        else:
            out_connection_gas_rates[i] = 0.0

    return (
        oil_rate,
        water_rate,
        gas_rate,
        surface_oil_rate,
        surface_water_rate,
        surface_gas_rate,
    )


def build_connection_phase_rates(
    connection_oil_rates: NumberArray[OneDimension],
    connection_water_rates: NumberArray[OneDimension],
    connection_gas_rates: NumberArray[OneDimension],
) -> list[PhaseValues]:
    """
    Builds the per-connection `PhaseValues` sequence the segmented
    hydraulics walk (`compute_perforation_pressures`'
    `connection_phase_rates` parameter) needs, from `accumulate_phase_rates`'
    per-connection output buffers.

    :param connection_oil_rates: `PerforationWorkspace.connection_oil_rates`,
        already populated by `accumulate_phase_rates`.
    :param connection_water_rates: Water analogue of `connection_oil_rates`.
    :param connection_gas_rates: Gas analogue of `connection_oil_rates`.
    :returns: One `PhaseValues` per connection, same order as the inputs.
    """
    return [
        PhaseValues(oil=oil, water=water, gas=gas)
        for oil, water, gas in zip(
            connection_oil_rates, connection_water_rates, connection_gas_rates, strict=False
        )
    ]


def load_wells_states(
    wells: Wells, compiled_system: CompiledWellSystem, workspace: WellsWorkspace
) -> WellsStates:
    """
    Load `WellsStates` from a resolved `WellsWorkspace`.

    Only covers wells actually resolved this pass/step. A well whose
    `WellStatus` is still `PENDING` (its BHP is left `NaN` by
    `resolve_control`) is skipped rather than reported with meaningless
    values.

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
        if math.isnan(bhp):
            continue  # not resolved this pass (PENDING, or UNSET control)

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
                    cell_index=int(perforations.cell_indices[row]),
                    flowing_pressure=pressure,
                    phase_rates=PhaseValues(
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
            is_open=not bool(workspace.economic_shutins[well_row]),
            active_control=decompile_well_control(controls, well_row, unit_system),
            bhp=bhp,
            perforation_states=tuple(perforation_states),
            phase_rates=PhaseValues(
                oil=workspace.oil_rates[well_row],
                water=workspace.water_rates[well_row],
                gas=workspace.gas_rates[well_row],
            ),
            surface_phase_rates=PhaseValues(
                oil=workspace.surface_oil_rates[well_row],
                water=workspace.surface_water_rates[well_row],
                gas=workspace.surface_gas_rates[well_row],
            ),
            active_limit=active_limit,
            thp=none_if_nan(workspace.thps[well_row]),
            unit_system=unit_system,
        )

    return WellsStates(states=states, unit_system=unit_system)

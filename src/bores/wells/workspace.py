"""Compiled (buffer-reuse) structures for the well-control resolution."""

import math
import typing

import numba
import numpy as np
import numpy.typing as npt

from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.types import (
    Boolean,
    IntArray,
    Integer,
    Number,
    NumberArray,
    OneDimension,
    PhaseValues,
    ThreeDimensions,
    TwoDimensions,
)
from bores.utils import none_if_nan
from bores.wells.base import Wells
from bores.wells.compile import UNSET_INT, CompiledWellSystem
from bores.wells.decompile import decompile_limit, decompile_perforations, decompile_well_control
from bores.wells.hydraulics.base import ConnectionFluids
from bores.wells.state import (
    PerforationState,
    WellState,
    WellStates,
)

__all__ = [
    "ConnectionRates",
    "DerivativeScratch",
    "PerforationWorkspace",
    "WellsWorkspace",
    "accumulate_rates",
    "accumulate_well_volumes",
    "apply_non_darcy_correction_to_well_indices",
    "build_derivative_scratch",
    "build_perforation_workspace",
    "build_wells_workspace",
    "compute_perforation_drawdown",
    "get_connection_rates",
    "load_wells_states",
    "restore_connection_rates",
    "sum_connection_rates",
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

    allows_crossflow: Boolean
    """
    Whether this well allows its connections to flow in the opposite direction to the well. A
    connection with reversed drawdown is given zero flow when this is `False`.
    """

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
    Gas each connection's oil holds in solution, per stock-tank barrel of oil. Multiplying a
    connection's surface oil rate by it gives the gas that oil brings to the surface in solution.
    """

    vaporized_oil_ratios: NumberArray[OneDimension]
    """
    Oil each connection's gas carries as vapor, in stock-tank barrels per standard cubic foot of
    gas. Multiplying a connection's free gas rate at the surface by it gives the oil that gas
    brings to the surface as condensate.
    """

    stb_to_volume_factor: Number
    """
    Volume of one stock-tank barrel in the volume unit of reservoir rates. Reservoir rates are
    in volume per day, while oil and water rates at the surface are in stock-tank barrels per day.
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
    allows_crossflow: Boolean = True,
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
    :param allows_crossflow: Whether this well allows crossflow. `True` (the default) matches
        a deck that does not say.
    :param stb_to_volume_factor: Volume of one stock-tank barrel in the volume unit of reservoir
        rates. One (the default) in the unit systems where a stock-tank volume is a plain volume.
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
        "NumberArray[OneDimension]", np.asarray(well_indices, dtype=resolved_dtype)
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
            "NumberArray[OneDimension]", resolved_connection_conductivities
        ),
        d_factor=d_factor,
        allows_crossflow=allows_crossflow,
        connection_fluids=resolved_fluids,
        representative_depths=typing.cast(
            "NumberArray[OneDimension]",
            np.asarray(representative_depths, dtype=resolved_dtype),
        ),
        representative_measured_depths=typing.cast(
            "NumberArray[OneDimension]",
            np.asarray(representative_measured_depths, dtype=resolved_dtype),
        ),
        reference_measured_depth=reference_measured_depth,
        inclinations_from_vertical=typing.cast(
            "NumberArray[OneDimension]",
            np.asarray(inclinations_from_vertical, dtype=resolved_dtype),
        ),
        connection_pressures=typing.cast(
            "NumberArray[OneDimension]", np.empty(n, dtype=resolved_dtype)
        ),
        connection_oil_rates=typing.cast(
            "NumberArray[OneDimension]", np.empty(n, dtype=resolved_dtype)
        ),
        connection_water_rates=typing.cast(
            "NumberArray[OneDimension]", np.empty(n, dtype=resolved_dtype)
        ),
        connection_gas_rates=typing.cast(
            "NumberArray[OneDimension]", np.zeros(n, dtype=resolved_dtype)
        ),
        dissolved_gas_ratios=resolved_fluids.solution_gas_oil_ratios,
        vaporized_oil_ratios=resolved_fluids.vaporized_oil_gas_ratios,
        stb_to_volume_factor=stb_to_volume_factor,
    )


class DerivativeScratch(typing.NamedTuple):
    """
    Double-precision buffers the well derivative computation works in, sized once for the
    well with the most connections, so that computing a well's derivatives allocates none.

    For a well with `n` connections, `n <= max_connections`, each buffer is used through its
    first `n` (or `3 * n`, `3 * n * n`...) entries and reshaped there. The names give each
    buffer's logical shape, with `size` standing for `3 * n`.
    """

    oil_rates: NumberArray[OneDimension]
    """Shape `(n,)`. The final oil rate at each flowing connection."""

    water_rates: NumberArray[OneDimension]
    """Shape `(n,)`. The final water rate at each flowing connection."""

    gas_rates: NumberArray[OneDimension]
    """Shape `(n,)`. The final gas rate at each flowing connection."""

    connection_pressures: NumberArray[OneDimension]
    """Shape `(n,)`. The final pressure at each flowing connection."""

    well_indices: NumberArray[OneDimension]
    """Shape `(n,)`. Each flowing connection's well index after the non-Darcy correction."""

    shifted_oil_rates: NumberArray[OneDimension]
    """Shape `(size, n)`. Row `t` is the oil rates with the rate `t` is differenced by changed."""

    shifted_water_rates: NumberArray[OneDimension]
    """Shape `(size, n)`. As `shifted_oil_rates`, for water."""

    shifted_gas_rates: NumberArray[OneDimension]
    """Shape `(size, n)`. As `shifted_oil_rates`, for gas."""

    forward_pressures: NumberArray[OneDimension]
    """Shape `(size, n)`. Row `t` is the connection pressures with rate `t` stepped one way."""

    backward_pressures: NumberArray[OneDimension]
    """Shape `(size, n)`. Row `t` is the connection pressures with rate `t` stepped the other way."""

    pressure_rate_sensitivities: NumberArray[OneDimension]
    """Shape `(n, size)`. How each connection's pressure changes with every connection rate."""

    pressure_reference_sensitivities: NumberArray[OneDimension]
    """Shape `(n,)`. How each connection's pressure changes with the reference pressure."""

    wellbore_pressure_partials: NumberArray[OneDimension]
    """Shape `(n, 3)`. Each connection's rates against the pressure in the wellbore beside it."""

    cell_partials: NumberArray[OneDimension]
    """Shape `(n, 3, 3)`. Each connection's rates against its cell's pressure and saturations."""

    gas_rate_partials: NumberArray[OneDimension]
    """Shape `(n, 3)`. Each connection's rates against its own gas rate, through non-Darcy flow."""

    composition_partials: NumberArray[OneDimension]
    """Shape `(size, size)`. A reversed producer connection's rates against the producing ones."""

    system: NumberArray[OneDimension]
    """Shape `(size, size)`. The matrix the total derivatives are solved with."""

    right_hand_side: NumberArray[OneDimension]
    """Shape `(size, size + 1)`. The cell unknowns' columns, then the bottom-hole pressure's."""

    rate_cell_derivatives: NumberArray[OneDimension]
    """Shape `(size, size)`. The connection rates against the cell unknowns."""

    rate_bhp_derivatives: NumberArray[OneDimension]
    """Shape `(size,)`. The connection rates against the bottom-hole pressure."""

    surface_cell_derivatives: NumberArray[OneDimension]
    """Shape `(size, size)`. The surface component rates against the cell unknowns."""

    surface_bhp_derivatives: NumberArray[OneDimension]
    """Shape `(size,)`. The surface component rates against the bottom-hole pressure."""


def build_derivative_scratch(*, max_connections: Integer) -> DerivativeScratch:
    """
    Allocates the buffers of `DerivativeScratch`.

    :param max_connections: The most connections any one well has.
    :returns: Buffers, all `NaN`, in double precision.
    """
    count = int(max_connections)
    size = 3 * count

    def buffer(entries: Integer) -> NumberArray[OneDimension]:
        return typing.cast(
            "NumberArray[OneDimension]", np.full(int(entries), np.nan, dtype=np.float64)
        )

    return DerivativeScratch(
        oil_rates=buffer(count),
        water_rates=buffer(count),
        gas_rates=buffer(count),
        connection_pressures=buffer(count),
        well_indices=buffer(count),
        shifted_oil_rates=buffer(size * count),
        shifted_water_rates=buffer(size * count),
        shifted_gas_rates=buffer(size * count),
        forward_pressures=buffer(size * count),
        backward_pressures=buffer(size * count),
        pressure_rate_sensitivities=buffer(count * size),
        pressure_reference_sensitivities=buffer(count),
        wellbore_pressure_partials=buffer(count * 3),
        cell_partials=buffer(count * 9),
        gas_rate_partials=buffer(count * 3),
        composition_partials=buffer(size * size),
        system=buffer(size * size),
        right_hand_side=buffer(size * (size + 1)),
        rate_cell_derivatives=buffer(size * size),
        rate_bhp_derivatives=buffer(size),
        surface_cell_derivatives=buffer(size * size),
        surface_bhp_derivatives=buffer(size),
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

    derivative_offsets: IntArray[OneDimension]
    """
    Shape `(n_wells + 1,)`. Where each well's block starts in `connection_rate_cell_derivatives`
    and `connection_surface_rate_cell_derivatives`. A well with `n` connections has a block of
    `(3 * n) ** 2` entries, so its block is `derivative_offsets[w]:derivative_offsets[w + 1]`.
    All zero when the workspace was built without `well_offsets`, and then no derivatives are
    stored.
    """

    connection_rate_cell_derivatives: NumberArray[OneDimension]
    """
    Each well's block, flattened row by row, of how its connections' reservoir-condition phase
    rates change with the primary unknowns of every one of its connections' cells. A well's
    block reshapes to `(3 * n, 3 * n)`. Row `3 * i + a` is phase `a` (oil, water, gas) at
    connection `i`. Column `3 * k + v` is unknown `v` (pressure, water saturation, gas
    saturation) of connection `k`'s cell. These are total derivatives: they include the change
    in the pressure at every connection that the change in rate makes through the wellbore, the
    non-Darcy correction, and the reversed connections' dependence on the producing ones, with
    the well's bottom-hole pressure held fixed. A connection that is not flowing has zero
    rows and columns. `get_well_derivatives` reads a well's block.
    """

    connection_surface_rate_cell_derivatives: NumberArray[OneDimension]
    """
    As `connection_rate_cell_derivatives`, for the surface-condition rate of each component at
    each connection: stock-tank oil, stock-tank water and standard gas, in that order. The gas
    that a producer's oil carries in solution and the oil its gas carries as vapor are included,
    so these are the rates at which the reservoir's components leave or enter through the well.
    """

    connection_rate_bhp_derivatives: NumberArray[TwoDimensions]
    """
    Shape `(n_connections, 3)`. How each connection's reservoir-condition oil, water and gas rates
    change with the well's bottom-hole pressure, indexed like `connection_pressures`. A total
    derivative, with every cell's unknowns held fixed.
    """

    connection_surface_rate_bhp_derivatives: NumberArray[TwoDimensions]
    """
    Shape `(n_connections, 3)`. As `connection_rate_bhp_derivatives`, for the surface-condition
    oil, water and gas component rates.
    """

    well_rate_cell_derivatives: NumberArray[ThreeDimensions]
    """
    Shape `(n_connections, 3, 3)`. For the well each connection belongs to, how that well's
    total reservoir-condition oil, water and gas rates (axis 1) change with the pressure, water
    saturation and gas saturation (axis 2) of the cell at that connection, indexed like
    `connection_pressures`. These are the sums of the well's rows of `connection_rate_cell_derivatives`
    and give the Jacobian row of a reservoir-volume-rate control.
    """

    well_surface_rate_cell_derivatives: NumberArray[ThreeDimensions]
    """
    Shape `(n_connections, 3, 3)`. As `well_rate_cell_derivatives`, for the well's total
    surface-condition oil, water and gas rates. A surface rate control's Jacobian row is the
    sum of these over the phases it counts.
    """

    well_rate_bhp_derivatives: NumberArray[TwoDimensions]
    """
    Shape `(n_wells, 3)`. How each well's total reservoir-condition oil, water and gas rates
    change with its own bottom-hole pressure.
    """

    well_surface_rate_bhp_derivatives: NumberArray[TwoDimensions]
    """
    Shape `(n_wells, 3)`. As `well_rate_bhp_derivatives`, for the surface-condition oil, water and
    gas rates.
    """

    thp_bhp_derivatives: NumberArray[OneDimension]
    """
    Shape `(n_wells,)`. How each well's tubing head pressure changes with its own bottom-hole
    pressure, including the change in the rates that the bottom-hole pressure makes. `NaN` for
    a well with no tubing head pressure.
    """

    thp_cell_derivatives: NumberArray[TwoDimensions]
    """
    Shape `(n_connections, 3)`. For the well each connection belongs to, how that well's tubing
    head pressure changes with the pressure, water saturation and gas saturation of the cell at
    that connection, through the rates, with the bottom-hole pressure held fixed. `NaN` for a
    well with no tubing head pressure.
    """

    derivative_scratch: DerivativeScratch
    """Buffers the well derivatives are computed in. Empty when `derivative_offsets` is all zero."""

    group_control_modes: IntArray[OneDimension]
    """
    Shape `(n_wells,)`. For a well configured for group control, the concrete control mode tag
    (`ProducerControlModeTag` or `InjectorControlModeTag`) its group's allocation gave it this
    pass; `UNSET_INT` for every other well and for a group-controlled well not yet allocated.
    The well's own configured control mode stays group control throughout.
    """

    group_target_rates: NumberArray[OneDimension]
    """
    Shape `(n_wells,)`. The rate target in `group_control_modes`' quantity that the well's group
    allocation gave it this pass, after any redistribution of a group's shortfall. `NaN` where
    `group_control_modes` is `UNSET_INT`.
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
    *,
    n_wells: Integer,
    n_connections: Integer,
    well_offsets: IntArray[OneDimension] | None = None,
    dtype: npt.DTypeLike = None,
) -> WellsWorkspace:
    """
    Builds an empty `WellsWorkspace` for a system of `n_wells` wells with
    `n_connections` active connections in total. To be called once at the start
    of a run. Every `resolve_control` call across every timestep updates rows in
    this same object in place.

    :param n_wells: Number of wells.
    :param n_connections: Total active connections across every well,
        matching `CompiledPerforations`' own row count. Sizes `connection_pressures`.
    :param well_offsets: Shape `(n_wells + 1,)`. Where each well's connections start, as
        `CompiledPerforations.well_offsets` gives it. Sizes the stored well derivatives. Without
        it no derivatives can be stored.
    :param dtype: Output array dtype. `bores.precision.get_dtype()` if not given.
    :returns: `WellsWorkspace` with every row `NaN`/`UNSET_INT`/`0`.
    :raises ValidationError: If `well_offsets` does not match `n_wells` and `n_connections`.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    if well_offsets is None:
        derivative_offsets = np.zeros(n_wells + 1, dtype=np.int64)
    else:
        offsets = np.asarray(well_offsets, dtype=np.int64)
        if offsets.shape != (n_wells + 1,) or offsets[-1] != n_connections:
            raise ValidationError(
                f"`well_offsets` must have {n_wells + 1} entries ending at {n_connections}; "
                f"got {offsets.tolist()}."
            )
        derivative_offsets = np.concatenate(([0], np.cumsum((3 * np.diff(offsets)) ** 2)))
    n_block_entries = int(derivative_offsets[-1])
    max_connections = (
        0 if well_offsets is None else int(np.diff(np.asarray(well_offsets)).max(initial=0))
    )
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
        derivative_offsets=typing.cast("IntArray[OneDimension]", derivative_offsets),
        connection_rate_cell_derivatives=np.full(n_block_entries, np.nan, dtype=dtype),
        connection_surface_rate_cell_derivatives=np.full(n_block_entries, np.nan, dtype=dtype),
        connection_rate_bhp_derivatives=typing.cast(
            "NumberArray[TwoDimensions]", np.full((n_connections, 3), np.nan, dtype=dtype)
        ),
        connection_surface_rate_bhp_derivatives=typing.cast(
            "NumberArray[TwoDimensions]", np.full((n_connections, 3), np.nan, dtype=dtype)
        ),
        well_rate_cell_derivatives=typing.cast(
            "NumberArray[ThreeDimensions]", np.full((n_connections, 3, 3), np.nan, dtype=dtype)
        ),
        well_surface_rate_cell_derivatives=typing.cast(
            "NumberArray[ThreeDimensions]", np.full((n_connections, 3, 3), np.nan, dtype=dtype)
        ),
        well_rate_bhp_derivatives=typing.cast(
            "NumberArray[TwoDimensions]", np.full((n_wells, 3), np.nan, dtype=dtype)
        ),
        well_surface_rate_bhp_derivatives=typing.cast(
            "NumberArray[TwoDimensions]", np.full((n_wells, 3), np.nan, dtype=dtype)
        ),
        thp_bhp_derivatives=np.full(n_wells, np.nan, dtype=dtype),
        thp_cell_derivatives=typing.cast(
            "NumberArray[TwoDimensions]", np.full((n_connections, 3), np.nan, dtype=dtype)
        ),
        derivative_scratch=build_derivative_scratch(max_connections=max_connections),
        group_control_modes=np.full(n_wells, UNSET_INT, dtype=np.int32),
        group_target_rates=np.full(n_wells, np.nan, dtype=dtype),
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


def sum_connection_rates(
    *, workspace: PerforationWorkspace, is_injector: Boolean
) -> tuple[PhaseValues, PhaseValues]:
    """
    Sums a well's per-connection rates, as the most recent solve left them in its
    `PerforationWorkspace`, into well totals of every phase.

    An injector's connections that flow against the well produce the formation's own fluid, so
    the well total of a phase it does not inject is negative when any of them does. The solve's
    own totals count only the injected phase, since those are what an injection target is
    matched against. A connection that produces, whether it belongs to a producer or flows
    against an injector, brings the gas its oil holds in solution and the oil its gas carries as
    vapor to the surface with it.

    :param workspace: The well's perforation workspace.
    :param is_injector: Whether the well is an injector.
    :returns: `(rates, surface_rates)`, the reservoir- and surface-condition totals.
    """
    fluids = workspace.connection_fluids
    oil = workspace.connection_oil_rates
    water = workspace.connection_water_rates
    gas = workspace.connection_gas_rates
    if is_injector:
        carries = (oil < 0.0) | (water < 0.0) | (gas < 0.0)
    else:
        carries = np.ones(oil.shape[0], dtype=np.bool_)
    factor = workspace.stb_to_volume_factor
    surface_oil = oil / (fluids.oil_formation_volume_factors * factor)
    surface_free_gas = gas / fluids.gas_formation_volume_factors
    rates = PhaseValues(oil=oil.sum(), water=water.sum(), gas=gas.sum())
    surface_rates = PhaseValues(
        oil=(
            surface_oil + np.where(carries, surface_free_gas * workspace.vaporized_oil_ratios, 0.0)
        ).sum(),
        water=(water / (fluids.water_formation_volume_factors * factor)).sum(),
        gas=(
            surface_free_gas + np.where(carries, surface_oil * workspace.dissolved_gas_ratios, 0.0)
        ).sum(),
    )
    return rates, surface_rates


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
    :returns: The driving pressure difference, clipped to `>= 0` so a connection with reversed
        drawdown contributes no forward flow. Reverse flow, where a well allows it, is added
        separately by `accumulate_rates`.
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
    vaporized_oil_ratios: NumberArray[OneDimension],
    stb_to_volume_factor: Number,
    relevant_oil: Boolean,
    relevant_water: Boolean,
    relevant_gas: Boolean,
    is_injector: Boolean,
    radial_flow_constant: Number,
    allows_crossflow: Boolean,
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
    :param vaporized_oil_ratios: `PerforationWorkspace.vaporized_oil_ratios`.
    :param stb_to_volume_factor: `PerforationWorkspace.stb_to_volume_factor`.
    :param relevant_oil: Whether oil counts toward the primary target.
    :param relevant_water: Whether water counts toward the primary target.
    :param relevant_gas: Whether gas counts toward the primary target.
    :param is_injector: Whether this well is an injector.
    :param radial_flow_constant: `WellControlSpec.radial_flow_constant` -
        converts `well_index * mobility * drawdown` (`compute_peaceman_well_index`'s
        own units, which have no 2*pi built in) into a real reservoir-condition
        rate in this simulation's own unit system.
    :param allows_crossflow: Whether a connection whose drawdown is reversed flows the other
        way instead of giving no flow. A producer's reversed connection takes fluid from the
        wellbore into the formation, in the proportions of the fluid the well's other
        connections produce. An injector's reversed connection produces the formation's own fluid
        into the wellbore, all three phases, whatever the injected phase. Both use the
        connection's own formation volume factors and gas ratios to find surface rates, and are
        negative in the connection and well rates. A producer with nothing flowing the normal way
        has no fluid to give a reversed connection, so those get no flow.
    :param out_connection_oil_rates: Written in place with each
        connection's own reservoir-condition oil rate (`0.0` if
        `relevant_oil` is `False`). `PerforationWorkspace.connection_oil_rates`.
    :param out_connection_water_rates: Water analogue of `out_connection_oil_rates`.
    :param out_connection_gas_rates: Gas analogue of `out_connection_oil_rates`.
    :returns: `(oil_rate, water_rate, gas_rate, surface_oil_rate,
        surface_water_rate, surface_gas_rate)` - reservoir- and surface-condition well totals. A
        non-relevant phase's rate is `0.0`. Surface oil and water rates are in stock-tank barrels
        per day and the surface gas rate in standard cubic feet per day. A producer's surface gas
        rate is the total, the gas that flows free at the connections plus the gas its oil carries
        in solution, and its surface oil rate is the total as well, the oil that flows as liquid
        plus the condensate its gas carries. The reservoir-condition rates are those of the
        phases that flow at the connections.
    """
    oil_rate = 0.0
    water_rate = 0.0
    gas_rate = 0.0
    surface_oil_rate = 0.0
    surface_water_rate = 0.0
    surface_gas_rate = 0.0

    carries_dissolved_gas = relevant_gas and not is_injector
    carries_condensate = relevant_oil and not is_injector
    for idx in range(well_indices.shape[0]):
        drawdown = compute_perforation_drawdown(
            reservoir_pressure=reservoir_pressures[idx],
            connection_pressure=connection_pressures[idx],
            is_injector=is_injector,
        )
        well_index = well_indices[idx]

        out_connection_oil_rates[idx] = 0.0
        if relevant_oil or carries_dissolved_gas:
            contribution = radial_flow_constant * well_index * oil_mobilities[idx] * drawdown
            surface_oil_contribution = contribution / (
                oil_formation_volume_factors[idx] * stb_to_volume_factor
            )
            if relevant_oil:
                oil_rate += contribution
                surface_oil_rate += surface_oil_contribution
                out_connection_oil_rates[idx] = contribution
            if carries_dissolved_gas:
                surface_gas_rate += surface_oil_contribution * dissolved_gas_ratios[idx]

        if relevant_water:
            contribution = radial_flow_constant * well_index * water_mobilities[idx] * drawdown
            water_rate += contribution
            surface_water_rate += contribution / (
                water_formation_volume_factors[idx] * stb_to_volume_factor
            )
            out_connection_water_rates[idx] = contribution
        else:
            out_connection_water_rates[idx] = 0.0

        out_connection_gas_rates[idx] = 0.0
        if relevant_gas or carries_condensate:
            contribution = radial_flow_constant * well_index * gas_mobilities[idx] * drawdown
            surface_free_gas_contribution = contribution / gas_formation_volume_factors[idx]
            if relevant_gas:
                gas_rate += contribution
                surface_gas_rate += surface_free_gas_contribution
                out_connection_gas_rates[idx] = contribution
            if carries_condensate:
                surface_oil_rate += surface_free_gas_contribution * vaporized_oil_ratios[idx]

    if allows_crossflow:
        forward_oil_rate = oil_rate
        forward_water_rate = water_rate
        forward_gas_rate = gas_rate
        forward_total_rate = forward_oil_rate + forward_water_rate + forward_gas_rate
        for idx in range(well_indices.shape[0]):
            if is_injector:
                signed_drawdown = connection_pressures[idx] - reservoir_pressures[idx]
            else:
                signed_drawdown = reservoir_pressures[idx] - connection_pressures[idx]
            if signed_drawdown >= 0.0:
                continue
            reverse_drawdown = -signed_drawdown
            well_index = well_indices[idx]
            if is_injector:
                reverse_oil_rate = (
                    radial_flow_constant * well_index * oil_mobilities[idx] * reverse_drawdown
                )
                reverse_water_rate = (
                    radial_flow_constant * well_index * water_mobilities[idx] * reverse_drawdown
                )
                reverse_gas_rate = (
                    radial_flow_constant * well_index * gas_mobilities[idx] * reverse_drawdown
                )
            else:
                if forward_total_rate <= 0.0:
                    continue
                total_reverse_rate = (
                    radial_flow_constant
                    * well_index
                    * (oil_mobilities[idx] + water_mobilities[idx] + gas_mobilities[idx])
                    * reverse_drawdown
                )
                reverse_oil_rate = total_reverse_rate * forward_oil_rate / forward_total_rate
                reverse_water_rate = total_reverse_rate * forward_water_rate / forward_total_rate
                reverse_gas_rate = total_reverse_rate * forward_gas_rate / forward_total_rate

            out_connection_oil_rates[idx] = -reverse_oil_rate
            out_connection_water_rates[idx] = -reverse_water_rate
            out_connection_gas_rates[idx] = -reverse_gas_rate
            oil_rate -= reverse_oil_rate
            water_rate -= reverse_water_rate
            gas_rate -= reverse_gas_rate

            surface_oil_contribution = reverse_oil_rate / (
                oil_formation_volume_factors[idx] * stb_to_volume_factor
            )
            surface_oil_rate -= surface_oil_contribution
            surface_gas_rate -= surface_oil_contribution * dissolved_gas_ratios[idx]
            surface_water_rate -= reverse_water_rate / (
                water_formation_volume_factors[idx] * stb_to_volume_factor
            )
            surface_free_gas_contribution = reverse_gas_rate / gas_formation_volume_factors[idx]
            surface_gas_rate -= surface_free_gas_contribution
            surface_oil_rate -= surface_free_gas_contribution * vaporized_oil_ratios[idx]

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
    values. A group-controlled well is reported with the control its group allocated it this
    pass. A well that was resolved as shut, because it is shut or has no open
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
    allocated = workspace.group_control_modes != UNSET_INT
    if allocated.any():
        control_modes = controls.control_modes.copy()
        target_rates = controls.target_rates.copy()
        control_modes[allocated] = workspace.group_control_modes[allocated]
        target_rates[allocated] = workspace.group_target_rates[allocated]
        controls = controls._replace(control_modes=control_modes, target_rates=target_rates)
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

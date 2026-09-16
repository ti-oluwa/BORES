"""Compiled (buffer-reuse) structures for the well-control resolution hot path."""

import typing

import numba
import numpy as np
import numpy.typing as npt

from bores.precision import get_dtype
from bores.types import Boolean, IntArray, Integer, Number, NumberArray, OneDimension
from bores.wells.compile import UNSET_INT
from bores.wells.states import ConnectionSample, PhaseValues

__all__ = [
    "PerforationWorkspace",
    "WellsWorkspace",
    "accumulate_phase_rates",
    "build_connection_phase_rates",
    "build_perforation_workspace",
    "build_wells_workspace",
    "compute_perforation_drawdown",
]


class PerforationWorkspace(typing.NamedTuple):
    """
    One well's active-connection data for the resolution hot path.

    Built once per well per `resolve_control` call, and reused across every
    fixed-point and bisection iteration within that call.

    The per-connection values never change mid-resolution, and `connection_pressures`
    is a scratch buffer callers overwrite in place (via
    `compute_perforation_pressures(..., out=workspace.connection_pressures)`)
    rather than allocating a fresh array on every iteration.
    """

    well_indices: NumberArray[OneDimension]
    """This well's active connections' connection factors."""

    reservoir_pressures: NumberArray[OneDimension]
    """Matching reservoir pressure at each connection."""

    oil_mobilities: NumberArray[OneDimension]
    water_mobilities: NumberArray[OneDimension]
    gas_mobilities: NumberArray[OneDimension]
    oil_formation_volume_factors: NumberArray[OneDimension]
    water_formation_volume_factors: NumberArray[OneDimension]
    gas_formation_volume_factors: NumberArray[OneDimension]

    representative_depths: NumberArray[OneDimension]
    inclinations_from_vertical: NumberArray[OneDimension]

    connection_pressures: NumberArray[OneDimension]
    """
    Scratch buffer for `compute_perforation_pressures`' `out` parameter.
    Overwritten on every call and has no meaningful contents before the first one.
    """

    connection_oil_rates: NumberArray[OneDimension]
    connection_water_rates: NumberArray[OneDimension]
    connection_gas_rates: NumberArray[OneDimension]
    """
    Scratch buffers for `accumulate_phase_rates`' per-connection `out_*`
    parameters. Each connection's own reservoir-condition phase rate, `0.0`
    for a phase not relevant to the current solve. Overwritten on every
    call; feeds the segmented hydraulics walk's `connection_phase_rates`.
    """


def build_perforation_workspace(
    well_indices: NumberArray[OneDimension],
    representative_depths: NumberArray[OneDimension],
    inclinations_from_vertical: NumberArray[OneDimension],
    connection_samples: typing.Sequence[ConnectionSample],
    dtype: npt.DTypeLike = None,
) -> PerforationWorkspace:
    """
    Builds a `PerforationWorkspace` for one well.

    :param well_indices: This well's active connections' connection factors.
    :param representative_depths: Matching depths, same order as `well_indices`.
    :param inclinations_from_vertical: Matching inclinations, same order as `well_indices`.
    :param connection_samples: Matching reservoir conditions, same order as `well_indices`.
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

    return PerforationWorkspace(
        well_indices=typing.cast(
            NumberArray[OneDimension], np.asarray(well_indices, dtype=resolved_dtype)
        ),
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
            NumberArray[OneDimension], np.empty(n, dtype=resolved_dtype)
        ),
    )


class WellsWorkspace(typing.NamedTuple):
    """
    Every well's control-resolution result, one row per well.

    Built once for the whole run (`build_wells_workspace`), not once per
    timestep - every `resolve_control` call updates rows in place. A
    reported timestep's `WellsStates` is decompiled from whatever this
    holds at that point; nothing here is cleared between timesteps
    before the next resolve pass overwrites each row it touches.
    """

    bhps: NumberArray[OneDimension]
    """Shape `(n_wells,)`. `NaN` for a well not yet resolved this pass."""

    oil_rates: NumberArray[OneDimension]
    water_rates: NumberArray[OneDimension]
    gas_rates: NumberArray[OneDimension]
    """Shape `(n_wells,)` each. Reservoir-condition phase rate."""

    surface_oil_rates: NumberArray[OneDimension]
    surface_water_rates: NumberArray[OneDimension]
    surface_gas_rates: NumberArray[OneDimension]
    """Shape `(n_wells,)` each. Surface-condition phase rate."""

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
    connection_water_rates: NumberArray[OneDimension]
    connection_gas_rates: NumberArray[OneDimension]
    """
    Shape `(n_connections,)` each, same CSR indexing as `connection_pressures`.
    Each active connection's own reservoir-condition phase rate, at that
    well's final governing BHP. Zeroed (not `NaN`) for a connection
    belonging to an economically shut-in well, matching `oil_rates`/
    `water_rates`/`gas_rates`'s own zeroing. `NaN` still means "not
    resolved this pass", not "resolved to zero".
    """


def build_wells_workspace(
    *, n_wells: Integer, n_connections: Integer, dtype: npt.DTypeLike = None
) -> WellsWorkspace:
    """
    Builds an empty `WellsWorkspace` for a system of `n_wells` wells with
    `n_connections` active connections in total. Call once at the start
    of a run. Every `resolve_control` call across every timestep
    updates rows in this same object in place.

    :param n_wells: Number of wells.
    :param n_connections: Total active connections across every well,
        matching `CompiledPerforations`' own row count - sizes
        `connection_pressures`.
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

    Not `numba.njit` - `PhaseValues` is a plain `NamedTuple` consumed by
    the still-Python `compute_perforation_pressures` orchestration, not
    the jitted primitives. Callers on the hot path should expect this to
    allocate one list and `n` tuples per call; folding the hydraulics
    walk itself into the compiled layer (Step 5/8) is what removes this.

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

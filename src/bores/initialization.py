"""
Reservoir equilibrium state initialization from EQUIL, explicit deck arrays, or user-supplied arrays.

Precedence per field is:

```
RESTART (not yet supported but planned) -> explicit array/keyword -> EQUIL
```
"""

import functools
import typing
import warnings

import numba
import numpy as np
import numpy.typing as npt

from bores.blackoil.pvt.regions import PVT
from bores.blackoil.pvt.tables import PVTTable
from bores.blackoil.satfunc.capillary_pressure.tables import CapillaryPressureTable
from bores.blackoil.satfunc.regions import SatFunc
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.precision import get_dtype
from bores.reservoir.equilibrium import DepthTable, Equilibrium, EquilibriumRegion
from bores.reservoir.mass import compute_masses
from bores.reservoir.model import Reservoir
from bores.reservoir.state import Hysteresis, ReservoirState
from bores.reservoir.temperature import Temperature, TemperatureGradient, TemperatureTable
from bores.types import (
    Boolean,
    CellArray,
    IntArray,
    IntCellArray,
    Integer,
    Number,
    NumberArray,
    OneDimension,
    TwoDimensions,
    UnitSystem,
)
from bores.utils import get_hydrostatic_gradient_factor, iterate_regions

__all__ = [
    "EquilibriumArrays",
    "compute_water_phase_pressure",
    "initialize_equilibrium_arrays",
    "initialize_reservoir_state",
    "initialize_water_gas_state",
    "invert_capillary_pressure_curves",
]

N_SATURATION_SAMPLES = 100
"""
Default sample count for the capillary-pressure inversion grid.
"""

CAPILLARY_INVERSION_BATCH_SIZE = 2_000_000
"""
Largest number of capillary pressure curve samples evaluated in one pass, which bounds the
memory used by the inversion on large grids.
"""


class EquilibriumArrays(typing.NamedTuple):
    """Per-cell arrays produced by an equilibration algorithm for one region."""

    pressure: CellArray
    water_saturation: CellArray
    gas_saturation: CellArray
    solution_gor: CellArray
    oil_bubble_point_pressure: CellArray
    vaporized_oil_gas_ratio: CellArray
    gas_dew_point_pressure: CellArray


def _rk4_march(
    density_fn: typing.Callable[[Number, Number], Number],
    reference_depth: Number,
    reference_pressure: Number,
    endpoint_depth: Number,
    gradient_factor: Number,
    step: Number = 1.0,
    dtype: npt.DTypeLike = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Integrate `dP/dz = gradient_factor * density_fn(pressure, depth)` from `reference_depth` to
    `endpoint_depth` via fixed-step RK4.

    :returns: `(depths, pressures)`, both ascending in depth and spanning
        `[min(reference_depth, endpoint_depth), max(reference_depth, endpoint_depth)]`.
    """
    if np.isclose(reference_depth, endpoint_depth):
        return np.array([reference_depth], dtype=dtype), np.array(
            [reference_pressure], dtype=dtype
        )

    direction = 1.0 if endpoint_depth > reference_depth else -1.0
    n_steps = max(1, int(np.ceil(abs(endpoint_depth - reference_depth) / step)))
    h = direction * abs(endpoint_depth - reference_depth) / n_steps

    depths = np.empty(n_steps + 1, dtype=dtype)
    pressures = np.empty(n_steps + 1, dtype=dtype)
    depths[0], pressures[0] = reference_depth, reference_pressure
    z, p = reference_depth, reference_pressure
    for i in range(n_steps):
        k1 = gradient_factor * density_fn(p, z)
        k2 = gradient_factor * density_fn(p + 0.5 * h * k1, z + 0.5 * h)
        k3 = gradient_factor * density_fn(p + 0.5 * h * k2, z + 0.5 * h)
        k4 = gradient_factor * density_fn(p + h * k3, z + h)
        p = p + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        z = z + h
        depths[i + 1], pressures[i + 1] = z, p

    if direction < 0:
        depths = depths[::-1]
        pressures = pressures[::-1]
    return depths, pressures


COLLOCATION_MAX_ITERATIONS = 60
"""
Most passes `integrate_hydrostatic_pressure` makes over a column before it gives up on the
vectorized solve and integrates the column step by step instead.
"""


def integrate_hydrostatic_pressure(
    density_fn: typing.Callable[[typing.Any, typing.Any], typing.Any],
    reference_depth: Number,
    reference_pressure: Number,
    endpoint_depth: Number,
    gradient_factor: Number,
    step: Number = 1.0,
    dtype: npt.DTypeLike = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Integrate `dP/dz = gradient_factor * density_fn(pressure, depth)` from `reference_depth`
    to `endpoint_depth`, at the same depths a fixed-step RK4 march with `step` would visit.

    The pressure at every depth is solved for together, so `density_fn` is called a handful of
    times on whole columns instead of four times per step on single values, which is what
    makes the cost of the PVT table lookups bearable on tall columns. Each pass evaluates the
    density along the current pressure profile and integrates it with Simpson's rule, and
    the passes stop once the profile no longer changes. The result has the same fourth-order
    accuracy as RK4. If the passes do not settle (an extreme column), the march falls back to
    `_rk4_march`.

    :param density_fn: Fluid density at the given pressure and depth. It must accept arrays.
    :returns: `(depths, pressures)`, both ascending in depth and spanning
        `[min(reference_depth, endpoint_depth), max(reference_depth, endpoint_depth)]`.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    fallback = functools.partial(
        _rk4_march,
        density_fn=density_fn,
        reference_depth=reference_depth,
        reference_pressure=reference_pressure,
        endpoint_depth=endpoint_depth,
        gradient_factor=gradient_factor,
        step=step,
        dtype=dtype,
    )
    if np.isclose(reference_depth, endpoint_depth):
        return fallback()

    direction = 1.0 if endpoint_depth > reference_depth else -1.0
    n_steps = max(1, int(np.ceil(abs(endpoint_depth - reference_depth) / step)))
    h = direction * abs(endpoint_depth - reference_depth) / n_steps
    node_depths = reference_depth + h * np.arange(n_steps + 1, dtype=np.float64)
    midpoint_depths = node_depths[:-1] + 0.5 * h

    pressures = np.full(n_steps + 1, reference_pressure, dtype=np.float64)
    reference_gradient = (
        gradient_factor
        * np.ravel(density_fn(np.array([reference_pressure]), np.array([reference_depth])))[0]
    )
    pressures += reference_gradient * (node_depths - reference_depth)

    converged = False
    with np.errstate(all="ignore"):
        for _ in range(COLLOCATION_MAX_ITERATIONS):
            node_gradient = gradient_factor * np.asarray(
                density_fn(pressures, node_depths), dtype=np.float64
            )
            # Cubic Hermite estimate of the pressure at the middle of every step
            midpoint_pressures = 0.5 * (pressures[:-1] + pressures[1:]) + (h / 8.0) * (
                node_gradient[:-1] - node_gradient[1:]
            )
            midpoint_gradient = gradient_factor * np.asarray(
                density_fn(midpoint_pressures, midpoint_depths), dtype=np.float64
            )
            increments = (h / 6.0) * (
                node_gradient[:-1] + 4.0 * midpoint_gradient + node_gradient[1:]
            )
            updated = np.empty_like(pressures)
            updated[0] = reference_pressure
            np.cumsum(increments, out=updated[1:])
            updated[1:] += reference_pressure
            if not np.all(np.isfinite(updated)):
                break
            # The tolerance sits above the rounding noise of the density in the working dtype
            tolerance = (
                16.0
                * np.finfo(dtype).eps  # type: ignore
                * max(  # type: ignore
                    np.ptp(updated), 1e-3 * abs(reference_pressure)
                )
            )
            change = np.max(np.abs(updated - pressures))
            pressures = updated
            if change <= tolerance:
                converged = True
                break

    if not converged:
        return fallback()

    depths_out = node_depths.astype(dtype)
    pressures_out = pressures.astype(dtype)
    if direction < 0:
        depths_out = depths_out[::-1]
        pressures_out = pressures_out[::-1]
    return depths_out, pressures_out


def _march_full_range(
    density_fn: typing.Callable[[Number, Number], Number],
    reference_depth: Number,
    reference_pressure: Number,
    min_depth: Number,
    max_depth: Number,
    gradient_factor: Number,
    step: Number,
    dtype: npt.DTypeLike,
) -> tuple[NumberArray[OneDimension], NumberArray[OneDimension]]:
    """
    March `density_fn` outward from `reference_depth` in both directions, covering
    `[min_depth, max_depth]` in one sorted, concatenated profile.
    """
    depths_up, pressures_up = integrate_hydrostatic_pressure(
        density_fn=density_fn,
        reference_depth=reference_depth,
        reference_pressure=reference_pressure,
        endpoint_depth=min_depth,
        gradient_factor=gradient_factor,
        step=step,
        dtype=dtype,
    )
    depths_down, pressures_down = integrate_hydrostatic_pressure(
        density_fn=density_fn,
        reference_depth=reference_depth,
        reference_pressure=reference_pressure,
        endpoint_depth=max_depth,
        gradient_factor=gradient_factor,
        step=step,
        dtype=dtype,
    )
    depths = np.concatenate([depths_up[:-1], depths_down], dtype=dtype)
    pressures = np.concatenate([pressures_up[:-1], pressures_down], dtype=dtype)
    return typing.cast(NumberArray[OneDimension], depths), typing.cast(
        NumberArray[OneDimension], pressures
    )


class CapillarySaturations(typing.NamedTuple):
    """Saturations of a cell set recovered from capillary-gravity equilibrium."""

    water_saturation: CellArray
    """Water saturation of each cell."""

    gas_saturation: CellArray
    """Gas saturation of each cell."""

    oil_water_capillary_pressure: CellArray
    """
    Oil-water capillary pressure each cell actually sits at, which is the equilibrium value
    limited to the range the cell's capillary pressure curve can reach.
    """

    gas_oil_capillary_pressure: CellArray
    """
    Gas-oil capillary pressure each cell actually sits at, which is the equilibrium value
    limited to the range the cell's capillary pressure curve can reach.
    """


@numba.njit(cache=True)
def invert_curves(
    saturation_grid: NumberArray[TwoDimensions],
    capillary_pressure_grid: NumberArray[TwoDimensions],
    curve_index: IntArray[OneDimension],
    target_capillary_pressure: NumberArray[OneDimension],
    pressure_falls_when_flat: Boolean,
    prefer_highest_pressure_end: Boolean,
    saturation_out: NumberArray[OneDimension],
    capillary_pressure_out: NumberArray[OneDimension],
) -> None:
    """
    Finds, for every target, the saturation at which its curve reaches that target.

    Curve `curve_index[cell]` of the two grids is the curve of `cell`. The rows are read in
    place, in the direction in which their pressure rises, so no oriented copy of a row is made.
    The results are written to `saturation_out` and `capillary_pressure_out`.
    """
    sample_count = capillary_pressure_grid.shape[1]
    last = sample_count - 1
    for cell in range(target_capillary_pressure.shape[0]):
        row = curve_index[cell]
        target = target_capillary_pressure[cell]

        pressure_change = capillary_pressure_grid[row, last] - capillary_pressure_grid[row, 0]
        if pressure_change == 0:
            descending = pressure_falls_when_flat
        else:
            descending = pressure_change < 0
        # Sample positions of the lowest- and highest-pressure end of the curve
        if descending:
            lowest_position = last
            highest_position = 0
        else:
            lowest_position = 0
            highest_position = last
        lowest_pressure = capillary_pressure_grid[row, lowest_position]
        highest_pressure = capillary_pressure_grid[row, highest_position]

        samples_below_target = 0
        for sample in range(sample_count):
            if capillary_pressure_grid[row, sample] < target:
                samples_below_target += 1
        upper = min(max(samples_below_target, 1), last)
        lower = upper - 1
        if descending:
            lower_position = last - lower
            upper_position = last - upper
        else:
            lower_position = lower
            upper_position = upper

        pressure_lower = capillary_pressure_grid[row, lower_position]
        pressure_span = capillary_pressure_grid[row, upper_position] - pressure_lower
        if pressure_span > 0.0:
            fraction = min(max((target - pressure_lower) / pressure_span, 0.0), 1.0)
        else:
            fraction = 1.0
        saturation_lower = saturation_grid[row, lower_position]
        saturation = saturation_lower + fraction * (
            saturation_grid[row, upper_position] - saturation_lower
        )

        at_lowest_end = target <= lowest_pressure
        at_highest_end = target >= highest_pressure
        lowest_end_saturation = saturation_grid[row, lowest_position]
        highest_end_saturation = saturation_grid[row, highest_position]
        if prefer_highest_pressure_end:
            if at_lowest_end:
                saturation = lowest_end_saturation
            if at_highest_end:
                saturation = highest_end_saturation
        else:
            if at_highest_end:
                saturation = highest_end_saturation
            if at_lowest_end:
                saturation = lowest_end_saturation

        saturation_out[cell] = saturation
        capillary_pressure_out[cell] = min(max(target, lowest_pressure), highest_pressure)


def invert_capillary_pressure_curves(
    *,
    saturation_grid: NumberArray[typing.Any],
    capillary_pressure_grid: NumberArray[typing.Any],
    target_capillary_pressure: NumberArray[OneDimension],
    pressure_falls_with_saturation: Boolean,
    prefer_highest_pressure_end: Boolean,
    curve_index: IntArray[OneDimension] | None = None,
) -> tuple[NumberArray[OneDimension], NumberArray[OneDimension]]:
    """
    Find the saturation at which each capillary pressure curve reaches its target pressure.

    Every row of the two grids is one monotonic `Pc(saturation)` curve, sampled at the same
    points. A target above the curve's highest pressure gives the saturation at the
    highest-pressure end, and a target below its lowest pressure gives the saturation at the
    lowest-pressure end, so a curve that cannot reach the target is clamped, never extrapolated.
    A flat curve (for example a table of zeros) has the same pressure at both ends, so which
    end is the highest-pressure one comes from `pressure_falls_with_saturation`, and a target
    exactly equal to its pressure is resolved by `prefer_highest_pressure_end`.

    :param saturation_grid: Saturations the curves are sampled at, shape `(n_curves, n_samples)`.
    :param capillary_pressure_grid: Capillary pressure at each sample, same shape.
    :param target_capillary_pressure: Pressure to invert at, one per target, shape `(n_targets,)`.
    :param pressure_falls_with_saturation: Direction a curve is taken to run in when it is flat,
        which for the oil-water curve is `True` (`Pcow` falls as `Sw` rises) and for the
        gas-oil curve is `False` (`Pcgo` rises with `Sg`). A curve that is not flat runs in
        whichever direction its own values do.
    :param prefer_highest_pressure_end: Which end a target that equals both the lowest and the
        highest pressure of a flat curve resolves to. `True` picks the saturation at the
        highest-pressure end.
    :param curve_index: Row of the grids each target is inverted on, shape `(n_targets,)`, which
        lets targets that share a curve share its row. Target `i` is inverted on row `i` when
        not given.
    :returns: `(saturation, capillary_pressure)`, each of shape `(n_targets,)`. The second is the
        target limited to the range of its curve.
    """
    target = np.asarray(target_capillary_pressure)
    if curve_index is None:
        curve_index = np.arange(target.shape[0])
    result_dtype = np.result_type(saturation_grid, capillary_pressure_grid, target)
    saturation = np.empty(target.shape[0], dtype=result_dtype)
    capillary_pressure = np.empty(target.shape[0], dtype=result_dtype)
    invert_curves(
        saturation_grid=saturation_grid,
        capillary_pressure_grid=capillary_pressure_grid,
        curve_index=curve_index,
        target_capillary_pressure=typing.cast(NumberArray[OneDimension], target),
        pressure_falls_when_flat=bool(pressure_falls_with_saturation),
        prefer_highest_pressure_end=bool(prefer_highest_pressure_end),
        saturation_out=saturation,
        capillary_pressure_out=capillary_pressure,
    )
    return typing.cast(
        tuple[NumberArray[OneDimension], NumberArray[OneDimension]],
        (saturation, capillary_pressure),
    )


def index_unique_endpoints(
    *endpoints: NumberArray[OneDimension],
) -> tuple[IntArray[OneDimension], tuple[NumberArray[OneDimension], ...]]:
    """
    Groups cells whose endpoint saturations are all the same, since such cells share one
    capillary pressure curve.

    :param endpoints: One array per endpoint saturation, each of shape `(n_cells,)`.
    :returns: `(curve_ids, unique_endpoints)`. `curve_ids[cell]` is the position of the cell's
        endpoints in each of the `unique_endpoints` arrays.
    """
    n_cells = endpoints[0].shape[0]
    if n_cells == 0 or all(np.all(values == values[0]) for values in endpoints):
        return (
            typing.cast(IntArray[OneDimension], np.zeros(n_cells, dtype=np.intp)),
            typing.cast(
                tuple[NumberArray[OneDimension], ...], tuple(values[:1] for values in endpoints)
            ),
        )
    unique_rows, curve_ids = np.unique(np.stack(endpoints, axis=1), axis=0, return_inverse=True)
    return (
        typing.cast(IntArray[OneDimension], curve_ids.reshape(-1)),
        typing.cast(
            tuple[NumberArray[OneDimension], ...],
            tuple(unique_rows[:, column] for column in range(len(endpoints))),
        ),
    )


def invert_curves_per_cell(
    *,
    curve_ids: IntArray[OneDimension],
    unique_endpoints: tuple[NumberArray[OneDimension], ...],
    target_capillary_pressure: NumberArray[OneDimension],
    build_curves: typing.Callable[
        [tuple[NumberArray[OneDimension], ...]],
        tuple[NumberArray[TwoDimensions], NumberArray[TwoDimensions]],
    ],
    pressure_falls_with_saturation: Boolean,
    prefer_highest_pressure_end: Boolean,
    saturation_samples: int,
    dtype: npt.DTypeLike,
) -> tuple[NumberArray[OneDimension], NumberArray[OneDimension]]:
    """
    Invert a capillary pressure curve for every cell, evaluating each distinct curve once.

    :param curve_ids: Distinct curve of each cell, see `index_unique_endpoints`.
    :param unique_endpoints: Endpoint saturations of every distinct curve.
    :param target_capillary_pressure: Pressure to invert at for each cell.
    :param build_curves: Takes the endpoints of some distinct curves and returns their
        saturation and capillary pressure grids, each of shape `(n_curves, saturation_samples)`.
    :param saturation_samples: Number of samples on each curve.
    :returns: `(saturation, capillary_pressure)` of every cell, see `invert_capillary_pressure_curves`.
    """
    n_cells = target_capillary_pressure.shape[0]
    n_curves = unique_endpoints[0].shape[0]
    saturation = np.empty(n_cells, dtype=dtype)
    capillary_pressure = np.empty(n_cells, dtype=dtype)
    # The grids of all the distinct curves are built in one go when they fit the batch size
    chunk_size = (
        n_cells
        if n_curves * saturation_samples <= CAPILLARY_INVERSION_BATCH_SIZE
        else max(1, CAPILLARY_INVERSION_BATCH_SIZE // saturation_samples)
    )
    for start in range(0, n_cells, max(chunk_size, 1)):
        cells = slice(start, min(start + chunk_size, n_cells))
        if chunk_size >= n_cells:
            chunk_endpoints = unique_endpoints
            local_curve_ids = curve_ids
        else:
            used_curves, local_curve_ids = np.unique(curve_ids[cells], return_inverse=True)
            chunk_endpoints = tuple(values[used_curves] for values in unique_endpoints)
        saturation_grid, capillary_pressure_grid = build_curves(chunk_endpoints)  # type: ignore[arg-type]
        saturation[cells], capillary_pressure[cells] = invert_capillary_pressure_curves(
            saturation_grid=saturation_grid,
            capillary_pressure_grid=capillary_pressure_grid,
            target_capillary_pressure=target_capillary_pressure[cells],  # type: ignore[arg-type]
            pressure_falls_with_saturation=pressure_falls_with_saturation,
            prefer_highest_pressure_end=prefer_highest_pressure_end,
            curve_index=local_curve_ids.reshape(-1),
        )
    return (
        typing.cast(NumberArray[OneDimension], saturation),
        typing.cast(NumberArray[OneDimension], capillary_pressure),
    )


def get_saturations_from_capillary_pressure(
    *,
    depths: CellArray,
    oil_pressure: CellArray,
    connate_water_saturation: CellArray,
    residual_oil_saturation_water: CellArray,
    residual_oil_saturation_gas: CellArray,
    residual_gas_saturation: CellArray,
    capillary_pressure: CapillaryPressureTable,
    water_profile: tuple[npt.NDArray, npt.NDArray] | None,
    gas_profile: tuple[npt.NDArray, npt.NDArray] | None,
    dtype: npt.DTypeLike,
    saturation_samples: int = N_SATURATION_SAMPLES,
) -> CapillarySaturations:
    """
    Recover `Sw(z)` and `Sg(z)` from the capillary pressure the fluids sit at, which gives a
    smooth transition zone instead of a sharp step at each contact.

    The oil-water capillary pressure of a cell is `Po(z) - Pw(z)` and the gas-oil capillary
    pressure is `Pg(z) - Po(z)`, where `Po(z)` is the oil-phase pressure of a continuous oil
    column through the whole reservoir, including the part of it that lies in the gas cap
    and in the aquifer. Each region's `Pcow`/`Pcgo` curves are then inverted at those values.
    Water saturation ranges from connate to 1, and gas saturation from 0 to `1 - Swc`, so a
    cell far enough below the water-oil contact is fully water-filled and a cell far enough above
    the gas-oil contact holds only gas and connate water.

    The two inversions are independent, so a cell can end up with `Sw + Sg > 1`, which has
    no oil to hold the excess. This happens in a gas cap whose `Pcow` curve still reaches a
    water saturation above connate, and in a thin oil column between two close contacts.
    Such a cell holds no oil, and its water saturation is taken from the gas-water capillary
    pressure `Pg - Pw = Pcow(Sw) + Pcgo(1 - Sw)` instead, with `Sg = 1 - Sw`.

    :param depths: Depth of each cell.
    :param oil_pressure: Oil-phase pressure of each cell along a continuous oil column.
    :param water_profile: Full-column `(depth, pressure)` water pressure profile, or `None`
        when the region has no water-oil contact.
    :param gas_profile: Full-column `(depth, pressure)` gas pressure profile, or `None` when
        the region has no gas-oil contact.
    :param saturation_samples: Number of samples on each cell's capillary pressure curve.
        Higher values give a more accurate inversion at proportionally higher cost.
        Defaults to `N_SATURATION_SAMPLES`.
    :returns: The saturations, and the capillary pressures the cells sit at.
    """
    n = len(depths)
    swc = np.asarray(connate_water_saturation, dtype=dtype)
    sorw = np.asarray(residual_oil_saturation_water, dtype=dtype)
    sorg = np.asarray(residual_oil_saturation_gas, dtype=dtype)
    sgr = np.asarray(residual_gas_saturation, dtype=dtype)

    water_saturation = swc.copy()
    gas_saturation = np.zeros(n, dtype=dtype)
    oil_water_capillary_pressure = np.zeros(n, dtype=dtype)
    gas_oil_capillary_pressure = np.zeros(n, dtype=dtype)

    fractions = np.linspace(0.0, 1.0, saturation_samples, dtype=dtype)
    # Cells with the same endpoint saturations sit on the same curves, so each distinct curve
    # is evaluated once however many cells share it
    curve_ids, unique_endpoints = index_unique_endpoints(swc, sorw, sorg, sgr)  # type: ignore[arg-type]
    target_pcow: NumberArray[OneDimension] | None = None
    target_pcgo: NumberArray[OneDimension] | None = None

    def endpoint_arguments(
        endpoints: tuple[NumberArray[OneDimension], ...],
    ) -> dict[str, NumberArray[OneDimension]]:
        return typing.cast(
            dict[str, NumberArray[OneDimension]],
            dict(
                irreducible_water_saturation=np.repeat(endpoints[0], saturation_samples),
                residual_oil_saturation_water=np.repeat(endpoints[1], saturation_samples),
                residual_oil_saturation_gas=np.repeat(endpoints[2], saturation_samples),
                residual_gas_saturation=np.repeat(endpoints[3], saturation_samples),
            ),
        )

    def build_water_curves(
        endpoints: tuple[NumberArray[OneDimension], ...],
    ) -> tuple[NumberArray[TwoDimensions], NumberArray[TwoDimensions]]:
        connate = endpoints[0]
        saturation_grid = connate[:, None] + fractions[None, :] * (1.0 - connate)[:, None]
        flat_saturation = saturation_grid.ravel()
        pcow_grid = capillary_pressure.evaluate(
            flat_saturation,
            1.0 - flat_saturation,
            np.zeros_like(flat_saturation),
            **endpoint_arguments(endpoints),
        )["oil_water"]
        return typing.cast(
            tuple[NumberArray[TwoDimensions], NumberArray[TwoDimensions]],
            (saturation_grid, np.reshape(pcow_grid, saturation_grid.shape)),
        )

    def build_gas_curves(
        endpoints: tuple[NumberArray[OneDimension], ...],
    ) -> tuple[NumberArray[TwoDimensions], NumberArray[TwoDimensions]]:
        connate = endpoints[0]
        saturation_grid = fractions[None, :] * (1.0 - connate)[:, None]
        flat_saturation = saturation_grid.ravel()
        flat_connate = np.repeat(connate, saturation_samples)
        pcgo_grid = capillary_pressure.evaluate(
            flat_connate,  # type: ignore[arg-type]
            np.clip(1.0 - flat_connate - flat_saturation, 0.0, 1.0),
            flat_saturation,
            **endpoint_arguments(endpoints),
        )["gas_oil"]
        return typing.cast(
            tuple[NumberArray[TwoDimensions], NumberArray[TwoDimensions]],
            (saturation_grid, np.reshape(pcgo_grid, saturation_grid.shape)),
        )

    def build_gas_water_curves(
        endpoints: tuple[NumberArray[OneDimension], ...],
    ) -> tuple[NumberArray[TwoDimensions], NumberArray[TwoDimensions]]:
        connate = endpoints[0]
        saturation_grid = connate[:, None] + fractions[None, :] * (1.0 - connate)[:, None]
        flat_water = saturation_grid.ravel()
        flat_zero = np.zeros_like(flat_water)
        arguments = endpoint_arguments(endpoints)
        pcow_grid = capillary_pressure.evaluate(
            flat_water, 1.0 - flat_water, flat_zero, **arguments
        )["oil_water"]
        pcgo_grid = capillary_pressure.evaluate(
            flat_water, flat_zero, np.clip(1.0 - flat_water, 0.0, 1.0), **arguments
        )["gas_oil"]
        return typing.cast(
            tuple[NumberArray[TwoDimensions], NumberArray[TwoDimensions]],
            (saturation_grid, np.reshape(pcow_grid + pcgo_grid, saturation_grid.shape)),
        )

    if water_profile is not None:
        water_pressure_at_depth = np.interp(depths, water_profile[0], water_profile[1])
        target_pcow = typing.cast(
            NumberArray[OneDimension], oil_pressure - water_pressure_at_depth
        )  # Pcow = Po - Pw
        water_saturation, oil_water_capillary_pressure = invert_curves_per_cell(
            curve_ids=curve_ids,
            unique_endpoints=unique_endpoints,
            target_capillary_pressure=target_pcow,
            build_curves=build_water_curves,
            pressure_falls_with_saturation=True,
            prefer_highest_pressure_end=True,  # A cell exactly on the contact is oil-bearing
            saturation_samples=saturation_samples,
            dtype=dtype,
        )

    if gas_profile is not None:
        gas_pressure_at_depth = np.interp(depths, gas_profile[0], gas_profile[1])
        target_pcgo = typing.cast(
            NumberArray[OneDimension], gas_pressure_at_depth - oil_pressure
        )  # Pcgo = Pg - Po
        gas_saturation, gas_oil_capillary_pressure = invert_curves_per_cell(
            curve_ids=curve_ids,
            unique_endpoints=unique_endpoints,
            target_capillary_pressure=target_pcgo,
            build_curves=build_gas_curves,
            pressure_falls_with_saturation=False,
            prefer_highest_pressure_end=False,  # A cell exactly on the contact is oil-bearing
            saturation_samples=saturation_samples,
            dtype=dtype,
        )

    if target_pcow is not None and target_pcgo is not None:
        # Where the two curves each claim the same pore space there is no oil: the cell holds
        # only water and gas, whose split is set by the gas-water capillary pressure, which is
        # `Pcow(Sw) + Pcgo(1 - Sw)` and equals `Pg - Pw = Pcow + Pcgo` at the cell's depth.
        # This is what a gas cap above a contact with a high-reaching `Pcow` curve, or a thin
        # oil column between two contacts, needs.
        overlapping = np.flatnonzero(water_saturation + gas_saturation > 1.0 + 1e-6)
        if overlapping.size > 0:
            overlapping_curve_ids = curve_ids[overlapping]
            water_in_gas, _ = invert_curves_per_cell(
                curve_ids=typing.cast(IntArray[OneDimension], overlapping_curve_ids),
                unique_endpoints=unique_endpoints,
                target_capillary_pressure=typing.cast(
                    NumberArray[OneDimension],
                    target_pcow[overlapping] + target_pcgo[overlapping],
                ),
                build_curves=build_gas_water_curves,
                pressure_falls_with_saturation=True,
                prefer_highest_pressure_end=True,
                saturation_samples=saturation_samples,
                dtype=dtype,
            )
            water_saturation[overlapping] = water_in_gas
            gas_saturation[overlapping] = 1.0 - water_in_gas

            # The pressures the cell sits at are those of its final saturations
            final_water = water_saturation[overlapping]
            final = capillary_pressure.evaluate(
                final_water,
                np.zeros_like(final_water),
                gas_saturation[overlapping],
                irreducible_water_saturation=swc[overlapping],
                residual_oil_saturation_water=sorw[overlapping],
                residual_oil_saturation_gas=sorg[overlapping],
                residual_gas_saturation=sgr[overlapping],
            )
            oil_water_capillary_pressure[overlapping] = final["oil_water"]
            gas_oil_capillary_pressure[overlapping] = final["gas_oil"]

    return CapillarySaturations(
        water_saturation=typing.cast(CellArray, water_saturation),
        gas_saturation=typing.cast(CellArray, gas_saturation),
        oil_water_capillary_pressure=typing.cast(CellArray, oil_water_capillary_pressure),
        gas_oil_capillary_pressure=typing.cast(CellArray, gas_oil_capillary_pressure),
    )


def initialize_center_point_equilibrium(
    *,
    region: EquilibriumRegion,
    depths: CellArray,
    connate_water_saturation: CellArray,
    oil_table: PVTTable,
    gas_table: PVTTable | None,
    water_table: PVTTable | None,
    rsvd_table: DepthTable | None,
    rvvd_table: DepthTable | None,
    temperature: CellArray,
    unit_system: UnitSystem,
    capillary_pressure: CapillaryPressureTable | None = None,
    residual_oil_saturation_water: CellArray | None = None,
    residual_oil_saturation_gas: CellArray | None = None,
    residual_gas_saturation: CellArray | None = None,
    depth_step: Number = 1.0,
    dtype: npt.DTypeLike = None,
    saturation_samples: int = N_SATURATION_SAMPLES,
    **kwargs: typing.Any,
) -> EquilibriumArrays:
    """
    Center-point `EQUIL` initialization: evaluate pressure and saturation once
    at each cell's centroid depth (`region.accuracy_flag == 0`).

    :param saturation_samples: Forwarded to `get_saturations_from_capillary_pressure`
        when `capillary_pressure` is supplied; ignored for sharp-contact initialization.
    :raises ValidationError: If `datum_depth` is outside the oil zone, or a
        contact is present without its corresponding PVT table.

    For a wet-gas (`PVTG`-based) gas table the gas cap's `Rv(depth)` comes from `rvvd_table` when
    one is supplied, and otherwise the gas is assumed saturated (`Rv = Rv_sat(P, T)`,
    `Pdew = P`), the same default `EQUIL` applies to a defaulted `RVVD`.
    """
    is_wet_gas = gas_table is not None and gas_table.has("vaporized_oil_gas_ratio")

    gradient_factor = get_hydrostatic_gradient_factor(unit_system)
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()

    goc_depth = region.goc_depth
    woc_depth = region.woc_depth
    oil_zone_low = goc_depth if goc_depth is not None else min(depths.min(), region.datum_depth)
    oil_zone_high = woc_depth if woc_depth is not None else max(depths.max(), region.datum_depth)
    if not (oil_zone_low <= region.datum_depth <= oil_zone_high):
        raise ValidationError(
            f"`EQUIL` `datum_depth` ({region.datum_depth}) must lie within the oil "
            f"zone [{oil_zone_low}, {oil_zone_high}] for center-point "
            "initialization; datum points in the gas cap or aquifer are not "
            "yet supported."
        )

    order = np.argsort(depths)
    sorted_depths = depths[order]
    sorted_temperature = temperature[order]

    def get_temperature_at(depth: Number) -> Number:
        return np.interp(depth, sorted_depths, sorted_temperature)

    def compute_oil_density(pressure: Number, depth: Number) -> Number:
        temperature_at_depth = get_temperature_at(depth)
        if rsvd_table is not None:
            # `Rs(depth)` from `RSVD`: oil is undersaturated wherever `P` exceeds the
            # bubble point of that Rs
            density = oil_table.density(
                pressure,
                temperature_at_depth,
                solution_gor=rsvd_table.at_depth(depth),  # type: ignore[arg-type]
            )
        else:
            # No `RSVD`: assume saturated oil (`Pb = P`) everywhere in the oil zone, the same
            # assumption used for `Rs` below. `bubble_point_pressure=pressure` selects the
            # saturated branch, which reads the density straight off the table at `P`.
            density = oil_table.density(
                pressure, temperature_at_depth, bubble_point_pressure=pressure
            )
        return density.astype(dtype, copy=False)  # type: ignore[union-attr, return-value]

    def compute_gas_density(pressure: Number, depth: Number) -> Number:
        assert gas_table is not None
        # Wet gas: the gas's own `Rv(depth)` when `RVVD` is given, otherwise saturated gas
        density = gas_table.density(
            pressure,
            get_temperature_at(depth),
            vaporized_oil_ratio=(
                rvvd_table.at_depth(depth)  # type: ignore[arg-type]
                if rvvd_table is not None
                else None
            ),
        )
        return density.astype(dtype, copy=False)  # type: ignore[union-attr, return-value]

    def compute_water_density(pressure: Number, depth: Number) -> Number:
        assert water_table is not None
        return water_table.density(pressure, get_temperature_at(depth)).astype(  # type: ignore[union-attr, return-value]
            dtype, copy=False
        )

    min_depth = min(depths.min(), region.datum_depth, oil_zone_low)
    max_depth = max(depths.max(), region.datum_depth, oil_zone_high)

    # Oil zone: integrate from datum outward to the zone edges.
    oil_depths_up, oil_pressures_up = integrate_hydrostatic_pressure(
        density_fn=compute_oil_density,
        reference_depth=region.datum_depth,
        reference_pressure=region.datum_pressure,
        endpoint_depth=oil_zone_low,
        gradient_factor=gradient_factor,
        step=depth_step,
        dtype=dtype,
    )
    oil_depths_down, oil_pressures_down = integrate_hydrostatic_pressure(
        density_fn=compute_oil_density,
        reference_depth=region.datum_depth,
        reference_pressure=region.datum_pressure,
        endpoint_depth=oil_zone_high,
        gradient_factor=gradient_factor,
        step=depth_step,
        dtype=dtype,
    )
    all_depths = [oil_depths_up[:-1], oil_depths_down]
    all_pressures = [oil_pressures_up[:-1], oil_pressures_down]

    # Capillary pressures are measured against the oil-phase pressure of a continuous oil
    # column, so the oil column is also carried through the gas cap and the aquifer.
    oil_column_depths: list[npt.NDArray] = [oil_depths_up[:-1], oil_depths_down]
    oil_column_pressures: list[npt.NDArray] = [oil_pressures_up[:-1], oil_pressures_down]
    if capillary_pressure is not None:
        if goc_depth is not None and min_depth < oil_zone_low:
            cap_depths, cap_pressures = integrate_hydrostatic_pressure(
                density_fn=compute_oil_density,
                reference_depth=oil_zone_low,
                reference_pressure=oil_pressures_up[0],
                endpoint_depth=min_depth,
                gradient_factor=gradient_factor,
                step=depth_step,
                dtype=dtype,
            )
            oil_column_depths.insert(0, cap_depths[:-1])
            oil_column_pressures.insert(0, cap_pressures[:-1])

        if woc_depth is not None and max_depth > oil_zone_high:
            base_depths, base_pressures = integrate_hydrostatic_pressure(
                density_fn=compute_oil_density,
                reference_depth=oil_zone_high,
                reference_pressure=oil_pressures_down[-1],
                endpoint_depth=max_depth,
                gradient_factor=gradient_factor,
                step=depth_step,
                dtype=dtype,
            )
            oil_column_depths.append(base_depths[1:])
            oil_column_pressures.append(base_pressures[1:])

    # Gas cap: full-range Pg(z) march from GOC, re-referenced to Po via pcgo_at_goc.
    # Full range (not just the gas-cap zone) so Pcgo(z) is defined everywhere,
    # which the capillary-pressure saturation path needs.
    gas_profile: tuple[npt.NDArray, npt.NDArray] | None = None
    if goc_depth is not None:
        if gas_table is None:
            raise ValidationError(
                "Region has a gas-oil contact but no gas PVT table was supplied."
            )
        gas_pressure_at_goc = (
            np.interp(goc_depth, oil_depths_up, oil_pressures_up) + region.pcgo_at_goc
        )
        gas_depths_full, gas_pressures_full = _march_full_range(
            density_fn=compute_gas_density,
            reference_depth=goc_depth,
            reference_pressure=gas_pressure_at_goc,
            min_depth=min_depth,
            max_depth=max_depth,
            gradient_factor=gradient_factor,
            step=depth_step,
            dtype=dtype,
        )
        gas_profile = (gas_depths_full, gas_pressures_full)
        in_gas_zone = gas_depths_full < goc_depth
        if min_depth < goc_depth:
            all_depths.append(gas_depths_full[in_gas_zone])
            all_pressures.append(gas_pressures_full[in_gas_zone] - region.pcgo_at_goc)

    # Aquifer: full-range Pw(z) march from WOC, re-referenced to Po via pcow_at_woc.
    water_profile: tuple[npt.NDArray, npt.NDArray] | None = None
    if woc_depth is not None:
        if water_table is None:
            raise ValidationError(
                "Region has a water-oil contact but no water PVT table was supplied."
            )
        water_pressure_at_woc = (
            np.interp(woc_depth, oil_depths_down, oil_pressures_down) - region.pcow_at_woc
        )
        water_depths_full, water_pressures_full = _march_full_range(
            density_fn=compute_water_density,
            reference_depth=woc_depth,
            reference_pressure=water_pressure_at_woc,
            min_depth=min_depth,
            max_depth=max_depth,
            gradient_factor=gradient_factor,
            step=depth_step,
            dtype=dtype,
        )
        water_profile = (water_depths_full, water_pressures_full)
        in_aquifer = water_depths_full > woc_depth
        if max_depth > woc_depth:
            all_depths.append(water_depths_full[in_aquifer])
            all_pressures.append(water_pressures_full[in_aquifer] + region.pcow_at_woc)

    grid_depths = np.concatenate(all_depths, dtype=dtype)
    grid_pressures = np.concatenate(all_pressures, dtype=dtype)
    grid_order = np.argsort(grid_depths)
    grid_depths = grid_depths[grid_order]
    grid_pressures = grid_pressures[grid_order]

    pressure = typing.cast(
        CellArray,
        np.interp(depths, grid_depths, grid_pressures).astype(dtype, copy=False),
    )

    n = len(depths)
    if capillary_pressure is not None:
        oil_pressure = np.interp(
            depths,
            np.concatenate(oil_column_depths, dtype=dtype),
            np.concatenate(oil_column_pressures, dtype=dtype),
        ).astype(dtype, copy=False)
        capillary_saturations = get_saturations_from_capillary_pressure(
            depths=depths,
            oil_pressure=typing.cast(CellArray, oil_pressure),
            connate_water_saturation=connate_water_saturation,
            residual_oil_saturation_water=(
                residual_oil_saturation_water
                if residual_oil_saturation_water is not None
                else np.zeros(n, dtype=dtype)
            ),
            residual_oil_saturation_gas=(
                residual_oil_saturation_gas
                if residual_oil_saturation_gas is not None
                else np.zeros(n, dtype=dtype)
            ),
            residual_gas_saturation=(
                residual_gas_saturation
                if residual_gas_saturation is not None
                else np.zeros(n, dtype=dtype)
            ),
            capillary_pressure=capillary_pressure,
            water_profile=water_profile,
            gas_profile=gas_profile,
            dtype=dtype,
            saturation_samples=saturation_samples,
        )
        water_saturation = capillary_saturations.water_saturation
        gas_saturation = capillary_saturations.gas_saturation

        # Outside the oil zone the oil phase is not continuous, so the pressure reported for a
        # cell is the pressure of the phase that fills it, less the capillary pressure it sits at.
        if gas_profile is not None and goc_depth is not None:
            in_gas_cap = depths < goc_depth
            pressure[in_gas_cap] = (
                np.interp(depths[in_gas_cap], gas_profile[0], gas_profile[1])
                - capillary_saturations.gas_oil_capillary_pressure[in_gas_cap]
            )
        if water_profile is not None and woc_depth is not None:
            in_aquifer = depths > woc_depth
            pressure[in_aquifer] = (
                np.interp(depths[in_aquifer], water_profile[0], water_profile[1])
                + capillary_saturations.oil_water_capillary_pressure[in_aquifer]
            )
    else:
        # Sharp contact: Sw = 1 below WOC, Sw = swc in the oil zone,
        # Sg = 1 - swc above GOC. No smooth transition zone.
        water_saturation = connate_water_saturation.astype(dtype, copy=False).copy()
        gas_saturation = np.zeros(n, dtype=dtype)
        if goc_depth is not None:
            in_gas_cap = depths < goc_depth
            gas_saturation[in_gas_cap] = 1.0 - connate_water_saturation[in_gas_cap]
            water_saturation[in_gas_cap] = connate_water_saturation[in_gas_cap]
        if woc_depth is not None:
            in_aquifer = depths > woc_depth
            water_saturation[in_aquifer] = 1.0
            gas_saturation[in_aquifer] = 0.0

    # Solution GOR / bubble-point pressure. Every cell is given the oil properties of its depth
    # and pressure, including gas cap and aquifer cells that hold no oil yet, so that oil
    # arriving in one of them later starts from a defined, saturated state.
    if rsvd_table is not None:
        solution_gor = rsvd_table.at_depth(depths).astype(  # type: ignore[arg-type]
            dtype, copy=False
        )
        oil_bubble_point_pressure = oil_table.bubble_point_pressure(
            temperature=temperature, solution_gor=solution_gor
        ).astype(dtype, copy=False)  # type: ignore[union-attr]
    else:
        # No RSVD: assume saturated oil (Pbub = P). Forcing `bubble_point_pressure=pressure`
        # makes the saturated branch always trigger, reading Rs straight off the table at P.
        solution_gor = oil_table.solution_gas_to_oil_ratio(
            pressure=pressure,
            temperature=temperature,
            bubble_point_pressure=pressure,
        ).astype(dtype, copy=False)  # type: ignore[union-attr]
        oil_bubble_point_pressure = pressure.copy()

    # Vaporized oil ratio / dew-point pressure (dry gas: Rv = 0 unless RVVD). As for the oil,
    # every cell is given the gas properties of its depth and pressure.
    vaporized_oil_gas_ratio = np.zeros(n, dtype=dtype)
    gas_dew_point_pressure = np.zeros(n, dtype=dtype)
    if rvvd_table is not None:
        vaporized_oil_gas_ratio = rvvd_table.at_depth(depths).astype(  # type: ignore[arg-type, union-attr]
            dtype, copy=False
        )
        if gas_table is not None:
            # A 2-D `Pdew(Rv, T)` table needs the gas's Rv
            dew_point = gas_table.dew_point_pressure(
                temperature=temperature,
                vaporized_oil_ratio=vaporized_oil_gas_ratio,
            )
            if dew_point is not None:
                gas_dew_point_pressure = dew_point.astype(dtype, copy=False)  # type: ignore[union-attr]
    elif is_wet_gas and gas_table is not None:
        # No `RVVD`: assume saturated gas (`Pdew = P`), so `Rv = Rv_sat(P, T)`
        saturated_rv = gas_table.vaporized_oil_gas_ratio(pressure, temperature)
        if saturated_rv is not None:
            vaporized_oil_gas_ratio = saturated_rv.astype(dtype, copy=False)  # type: ignore[union-attr]
            gas_dew_point_pressure = pressure.copy()

    return EquilibriumArrays(
        pressure=typing.cast(CellArray, pressure),
        water_saturation=typing.cast(CellArray, water_saturation),
        gas_saturation=typing.cast(CellArray, gas_saturation),
        solution_gor=typing.cast(CellArray, solution_gor),
        oil_bubble_point_pressure=typing.cast(CellArray, oil_bubble_point_pressure),
        vaporized_oil_gas_ratio=typing.cast(CellArray, vaporized_oil_gas_ratio),
        gas_dew_point_pressure=typing.cast(CellArray, gas_dew_point_pressure),
    )


SAMPLE_WEIGHTS = 0
"""Averages a field with the plain sample weights."""

OIL_CONTENT_WEIGHTS = 1
"""Averages a field with the sample weights times each sample's oil content."""

GAS_CONTENT_WEIGHTS = 2
"""Averages a field with the sample weights times each sample's gas content."""


@numba.njit(cache=True)
def average_samples(
    values: NumberArray[TwoDimensions],
    water_saturation: NumberArray[TwoDimensions],
    gas_saturation: NumberArray[TwoDimensions],
    sample_weights: NumberArray[TwoDimensions],
    weighting: Integer,
    averages: NumberArray[OneDimension],
) -> None:
    """
    Averages the samples of every cell in one pass, weighted as `weighting` says.

    A cell whose weights add up to nothing, such as one wholly in the gas cap that has no oil
    for the oil content weights, is averaged with the plain sample weights instead, and a cell
    whose plain weights add up to nothing averages to zero.
    """
    n_cells, n_samples = values.shape
    for cell in range(n_cells):
        total_weight = 0.0
        weighted_sum = 0.0
        for use_plain_weights in range(2):
            total_weight = 0.0
            weighted_sum = 0.0
            for sample in range(n_samples):
                weight = float(sample_weights[cell, sample])
                if use_plain_weights == 0:
                    if weighting == OIL_CONTENT_WEIGHTS:
                        oil = 1.0 - water_saturation[cell, sample] - gas_saturation[cell, sample]
                        weight *= min(max(oil, 0.0), 1.0)
                    elif weighting == GAS_CONTENT_WEIGHTS:
                        weight *= gas_saturation[cell, sample]
                total_weight += weight
                weighted_sum += values[cell, sample] * weight
            if use_plain_weights == 1 or weighting == SAMPLE_WEIGHTS or total_weight > 0.0:
                break
        averages[cell] = weighted_sum / total_weight if total_weight > 0.0 else 0.0


def average_equilibrium_samples(
    sample_arrays: EquilibriumArrays,
    *,
    n_cells: int,
    sample_weights: NumberArray[TwoDimensions],
) -> EquilibriumArrays:
    """
    Combine the samples taken through the thickness of each cell into one value per cell.

    Pressure and saturations are averaged with the given sample weights. The gas dissolved in
    oil (`solution_gor` and `oil_bubble_point_pressure`) only exists where there is oil, so those
    are averaged weighted by each sample's oil content, and the oil dissolved in gas
    (`vaporized_oil_gas_ratio` and `gas_dew_point_pressure`) is weighted by each sample's gas
    content. A cell that straddles a contact therefore keeps the dissolved ratio of the
    fluid that is actually in it, instead of one diluted by samples that hold none of that fluid.
    A cell with none of the fluid in any sample takes the plain weighted average.

    :param sample_arrays: Arrays for every sample of every cell, cell-major, so that the samples of
        a cell are consecutive.
    :param n_cells: Number of cells.
    :param sample_weights: Weight of each sample, shape `(n_cells, n_samples)`.
    :returns: One value per cell for every array.
    """

    water_saturation = typing.cast(
        NumberArray[TwoDimensions], np.reshape(sample_arrays.water_saturation, (n_cells, -1))
    )
    gas_saturation = typing.cast(
        NumberArray[TwoDimensions], np.reshape(sample_arrays.gas_saturation, (n_cells, -1))
    )
    weighting_by_field = {
        "solution_gor": OIL_CONTENT_WEIGHTS,
        "oil_bubble_point_pressure": OIL_CONTENT_WEIGHTS,
        "vaporized_oil_gas_ratio": GAS_CONTENT_WEIGHTS,
        "gas_dew_point_pressure": GAS_CONTENT_WEIGHTS,
    }

    averages = {}
    for name in EquilibriumArrays._fields:
        values = np.reshape(getattr(sample_arrays, name), (n_cells, -1))
        averages[name] = np.empty(n_cells, dtype=values.dtype)
        average_samples(
            values=values,
            water_saturation=water_saturation,
            gas_saturation=gas_saturation,
            sample_weights=sample_weights,
            weighting=weighting_by_field.get(name, SAMPLE_WEIGHTS),
            averages=averages[name],
        )
    return EquilibriumArrays(**averages)


def initialize_horizontal_subdivision_equilibrium(
    *,
    region: EquilibriumRegion,
    cell_top: CellArray,
    cell_bottom: CellArray,
    n_subdivisions: int | None = None,
    **kwargs: typing.Any,
) -> EquilibriumArrays:
    """
    Vertical-equilibrium averaging via `N` horizontal sub-divisions per cell
    (`region.accuracy_flag > 0`).

    Samples `N` depths between each cell's top/bottom (from the grid's
    per-cell AABB), evaluates `initialize_center_point_equilibrium` at all
    of them, and averages back down to one value per cell. Arithmetic mean
    assumes uniform cross-sectional area across depth and doesn't account for
    cell dip (see the tilted variant).
    """
    n_sub = n_subdivisions or abs(region.accuracy_flag)
    n_cells = len(cell_top)
    fractions = (np.arange(n_sub, dtype=cell_top.dtype) + 0.5) / n_sub
    sub_depth = cell_top[:, None] + fractions[None, :] * (cell_bottom - cell_top)[:, None]

    repeated_kwargs = {
        key: np.repeat(value, n_sub)
        if isinstance(value, np.ndarray) and value.shape == (n_cells,)
        else value
        for key, value in kwargs.items()
    }
    sub_arrays = initialize_center_point_equilibrium(
        region=region,
        depths=sub_depth.ravel(),
        **repeated_kwargs,  # type: ignore[arg-type]
    )

    return average_equilibrium_samples(
        sub_arrays,
        n_cells=n_cells,
        sample_weights=typing.cast(
            NumberArray[TwoDimensions], np.ones((n_cells, n_sub), dtype=cell_top.dtype)
        ),
    )


def initialize_tilted_subdivision_equilibrium(
    *,
    region: EquilibriumRegion,
    grid: Grid,
    cell_indices: IntArray[OneDimension],
    n_subdivisions: int | None = None,
    **kwargs: typing.Any,
) -> EquilibriumArrays:
    """
    Tilted-cell variant of horizontal subdivision, additionally accounting
    for cell dip (`region.accuracy_flag < 0`).

    Uses each cell's actual top/bottom face geometry (see
    `Grid.get_cell_top_bottom_faces`) rather than its AABB to get a
    dip-aware depth range, then weights the `N` sub-samples by a linear
    taper between the top and bottom face areas instead of averaging them
    with equal weight. This is a volume-weighted average under a
    linear-taper (frustum-like) cross-section model, which is exact for a
    cell whose cross-sectional area truly varies linearly from top to
    bottom and a first-order improvement otherwise; it is not a full
    polyhedron/plane-intersection volume average; genuinely irregular
    (e.g. non-monotonic) cross-sections along the cell's vertical axis are
    still only approximated.
    """
    n_sub = n_subdivisions or abs(region.accuracy_flag)
    n_cells = len(cell_indices)

    top_depth, bottom_depth, top_area, bottom_area = grid.get_cell_top_bottom_faces(
        cell_indices=cell_indices
    )
    dtype = kwargs["dtype"]
    top_depth = top_depth.astype(dtype, copy=False)
    bottom_depth = bottom_depth.astype(dtype, copy=False)
    top_area = top_area.astype(dtype, copy=False)
    bottom_area = bottom_area.astype(dtype, copy=False)

    fractions = (np.arange(n_sub, dtype=dtype) + 0.5) / n_sub
    sub_depth = top_depth[:, None] + fractions[None, :] * (bottom_depth - top_depth)[:, None]
    area_weights = top_area[:, None] + fractions[None, :] * (bottom_area - top_area)[:, None]
    # Guards against a degenerate zero/negative taper (e.g. a pinched-out
    # face) collapsing the weighted average to NaN.
    area_weights = np.clip(area_weights, np.finfo(dtype).tiny, None)

    repeated_kwargs = {
        key: np.repeat(value, n_sub)
        if isinstance(value, np.ndarray) and value.shape == (n_cells,)
        else value
        for key, value in kwargs.items()
    }
    sub_arrays = initialize_center_point_equilibrium(
        region=region,
        depths=sub_depth.ravel(),
        **repeated_kwargs,  # type: ignore[arg-type]
    )
    return average_equilibrium_samples(sub_arrays, n_cells=n_cells, sample_weights=area_weights)


def initialize_equilibrium_arrays(
    reservoir: Reservoir,
    pvt: PVT,
    equilibrium: Equilibrium,
    temperature: CellArray,
    *,
    satfunc: SatFunc | None = None,
    depth_step: Number = 1.0,
    dtype: npt.DTypeLike = None,
    saturation_samples: int = N_SATURATION_SAMPLES,
) -> EquilibriumArrays:
    """
    Compute `EQUIL`-derived pressure/saturation/Rs/Rv arrays for every cell.

    Dispatches each `EQLNUM` region to the algorithm selected by that
    region's `accuracy_flag` and assembles the full-grid arrays.

    :param satfunc: Optional `SatFunc`; when given, saturations
        are computed via capillary-pressure inversion instead of a sharp
        contact (see `get_saturations_from_capillary_pressure`). Doing so
        requires a `SATNUM` region per cell to select the right
        capillary-pressure curve; if `reservoir.regions.saturation_region`
        is unavailable, every cell defaults to saturation region 1 and a
        `UserWarning` is raised (see `Warns`).
    :param saturation_samples: Sample count for the capillary-pressure
        inversion grid, forwarded to
        `get_saturations_from_capillary_pressure` whenever `satfunc` is
        supplied; ignored for sharp-contact initialization. Defaults to
        `N_SATURATION_SAMPLES`.
    :raises ValidationError: If `reservoir`, `pvt`, `equilibrium`, or
        `satfunc` (when supplied) don't all share the same unit system;
        if a cell's `EQLNUM` has no matching `EquilibriumRegion`, an
        `EQLNUM` region spans more than one `PVTNUM`/`SATNUM` region, or a
        required PVT table is missing.
    :warns UserWarning: If `satfunc` is supplied but
        `reservoir.regions.saturation_region` (`SATNUM`) is unavailable;
        every cell is assigned to saturation region 1 in that case. Supply
        `reservoir.regions.saturation_region` explicitly (e.g.
        `np.ones(n_cells, dtype=np.int32)` for a single-region reservoir)
        to make the assignment explicit and silence the warning.
    """
    n_cells = reservoir.n_cells
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    if pvt.unit_system != reservoir.unit_system:
        raise ValidationError(
            f"`reservoir.unit_system` ({reservoir.unit_system.value!r}) does not "
            f"match `pvt.unit_system` ({pvt.unit_system.value!r})."
        )
    if equilibrium.unit_system != reservoir.unit_system:
        raise ValidationError(
            f"`reservoir.unit_system` ({reservoir.unit_system.value!r}) does not "
            f"match `equilibrium.unit_system` ({equilibrium.unit_system.value!r})."
        )
    if (
        satfunc is not None
        and satfunc.unit_system is not None
        and satfunc.unit_system != reservoir.unit_system
    ):
        raise ValidationError(
            f"`reservoir.unit_system` ({reservoir.unit_system.value!r}) does not "
            f"match `satfunc.unit_system` ({satfunc.unit_system.value!r})."
        )

    equilibrium_region_index: IntCellArray = (
        reservoir.regions.equilibrium_region
        if reservoir.regions is not None and reservoir.regions.equilibrium_region is not None
        else typing.cast(IntCellArray, np.ones(n_cells, dtype=np.int32))
    )
    pvt_region_index: IntCellArray = (
        reservoir.regions.pvt_region
        if reservoir.regions is not None and reservoir.regions.pvt_region is not None
        else typing.cast(IntCellArray, np.ones(n_cells, dtype=np.int32))
    )
    saturation_region_index: IntCellArray | None = (
        reservoir.regions.saturation_region if reservoir.regions is not None else None
    )
    if satfunc is not None and saturation_region_index is None:
        warnings.warn(
            "`satfunc` was supplied but `reservoir.regions.saturation_region` "
            "(`SATNUM`) is unavailable; defaulting every cell to saturation region 1. "
            "Set `reservoir.regions.saturation_region` explicitly (e.g. "
            "`np.ones(reservoir.n_cells, dtype=np.int32)` for a single-region "
            "reservoir) to make this assignment explicit and silence this warning.",
            UserWarning,
            stacklevel=2,
        )
        saturation_region_index = typing.cast(IntCellArray, np.ones(n_cells, dtype=np.int32))

    depth = reservoir.depth
    connate_water_saturation = reservoir.rock.connate_water_saturation
    residual_oil_saturation_water = reservoir.rock.residual_oil_saturation_water
    residual_oil_saturation_gas = reservoir.rock.residual_oil_saturation_gas
    residual_gas_saturation = reservoir.rock.residual_gas_saturation

    eqlnums = np.unique(equilibrium_region_index)
    single_region = len(eqlnums) == 1
    # Every cell belongs to exactly one region, so each array is fully written below
    assembled = (
        None
        if single_region
        else EquilibriumArrays(
            *(np.empty(n_cells, dtype=dtype) for _ in EquilibriumArrays._fields)
        )
    )

    for eqlnum in eqlnums:
        # With one region the cells are all of the grid, which needs no mask and no copies
        mask = slice(None) if single_region else equilibrium_region_index == eqlnum
        equilibrium_region = equilibrium.region(int(eqlnum))

        region_pvtnum = np.unique(pvt_region_index[mask])
        if len(region_pvtnum) != 1:
            raise ValidationError(
                f"`EQLNUM` {eqlnum} cells span multiple `PVTNUM` regions "
                f"({sorted(region_pvtnum.tolist())}); each equilibration "
                "region must map to exactly one PVT region for "
                "`initialize_equilibrium_arrays`."
            )
        pvt_region = pvt.region(region_pvtnum[0])

        capillary_pressure: CapillaryPressureTable | None = None
        if satfunc is not None:
            # Guaranteed non-`None` here: defaulted to region 1 (with a
            # warning) above whenever `satfunc` is supplied.
            assert saturation_region_index is not None
            region_satnum = np.unique(saturation_region_index[mask])
            if len(region_satnum) != 1:
                raise ValidationError(
                    f"`EQLNUM` {eqlnum} cells span multiple `SATNUM` regions "
                    f"({sorted(region_satnum.tolist())}); each equilibration "
                    "region must map to exactly one saturation-function "
                    "region for capillary-pressure-based initialization."
                )
            capillary_pressure = satfunc.region(region_satnum[0]).capillary_pressure

        rsvd_table = (
            equilibrium.rsvd_tables.get(equilibrium_region.rsvd_table)
            if equilibrium_region.uses_rsvd and equilibrium.rsvd_tables
            else None
        )
        rvvd_table = (
            equilibrium.rvvd_tables.get(equilibrium_region.rvvd_table)
            if equilibrium_region.uses_rvvd and equilibrium.rvvd_tables
            else None
        )

        initialization_kwargs: dict[str, typing.Any] = dict(
            region=equilibrium_region,
            connate_water_saturation=connate_water_saturation[mask],
            oil_table=pvt_region.tables.oil,
            gas_table=pvt_region.tables.gas,
            water_table=pvt_region.tables.water,
            rsvd_table=rsvd_table,
            rvvd_table=rvvd_table,
            temperature=temperature[mask],
            unit_system=reservoir.unit_system,
            capillary_pressure=capillary_pressure,
            residual_oil_saturation_water=residual_oil_saturation_water[mask],
            residual_oil_saturation_gas=residual_oil_saturation_gas[mask],
            residual_gas_saturation=residual_gas_saturation[mask],
            depth_step=depth_step,
            dtype=dtype,
            saturation_samples=saturation_samples,
        )

        if equilibrium_region.accuracy_flag == 0:
            region_arrays = initialize_center_point_equilibrium(
                depths=depth[mask],  # type: ignore[arg-type]
                **initialization_kwargs,
            )
        elif equilibrium_region.accuracy_flag > 0:
            region_arrays = initialize_horizontal_subdivision_equilibrium(
                cell_top=reservoir.grid.cell_min_xyz[mask, 2],  # type: ignore[arg-type]
                cell_bottom=reservoir.grid.cell_max_xyz[mask, 2],  # type: ignore[arg-type]
                **initialization_kwargs,
            )
        else:
            region_arrays = initialize_tilted_subdivision_equilibrium(
                grid=reservoir.grid,
                cell_indices=np.arange(n_cells) if single_region else np.nonzero(mask)[0],  # type: ignore[arg-type]
                **initialization_kwargs,
            )

        if assembled is None:
            return EquilibriumArrays(
                *(np.asarray(array, dtype=dtype) for array in region_arrays)  # type: ignore[arg-type]
            )
        for assembled_array, region_array in zip(assembled, region_arrays, strict=True):
            assembled_array[mask] = region_array

    assert assembled is not None
    return assembled


@numba.njit(cache=True)
def all_finite(values: NumberArray[OneDimension]) -> Boolean:
    """Whether every value is finite, stopping at the first that is not."""
    for index in range(values.shape[0]):
        if not np.isfinite(values[index]):
            return False
    return True


@numba.njit(cache=True)
def derive_oil_saturation(
    water_saturation: NumberArray[OneDimension],
    gas_saturation: NumberArray[OneDimension],
    oil_saturation: NumberArray[OneDimension],
) -> tuple[Number, Number, Number, Number, Number]:
    """
    Fill `oil_saturation` with what the water and gas leave, limited to `[0, 1]`, and return the
    ranges the validation needs, all from one pass over the cells.

    :returns: `(lowest water, highest water, lowest gas, highest gas, largest amount by which
        water plus gas exceeds 1)`.
    """
    lowest_water = np.inf
    highest_water = -np.inf
    lowest_gas = np.inf
    highest_gas = -np.inf
    largest_excess = -np.inf
    for index in range(water_saturation.shape[0]):
        water = float(water_saturation[index])
        gas = float(gas_saturation[index])
        lowest_water = min(lowest_water, water)
        highest_water = max(highest_water, water)
        lowest_gas = min(lowest_gas, gas)
        highest_gas = max(highest_gas, gas)
        largest_excess = max(largest_excess, water + gas - 1.0)
        oil_saturation[index] = min(max(1.0 - water - gas, 0.0), 1.0)
    return lowest_water, highest_water, lowest_gas, highest_gas, largest_excess


def get_temperature_array_from_regions(
    temperature: Temperature,
    pvt_region_index: IntCellArray,
    depth: CellArray,
    dtype: npt.DTypeLike = None,
) -> CellArray:
    """
    Evaluate a `Temperature` at every cell's depth, grouped by `PVTNUM` (the
    same grain `PVT.from_deck` resolves `Temperature` against, so PVT
    tables and this array stay consistent).

    Duck-types each region's spec. A bare number is broadcast, anything
    with an `at_depth` method (e.g. a depth-dependent table) is evaluated
    per cell.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    result = np.empty(len(pvt_region_index), dtype=dtype)
    for pvtnum, mask in iterate_regions(pvt_region_index):
        spec = temperature.region(pvtnum)
        if isinstance(spec, (TemperatureGradient, TemperatureTable)):
            result[mask] = spec.at_depth(depth[mask]).astype(dtype, copy=False)  # type: ignore[union-attr]
        else:
            result[mask] = spec  # type: ignore[arg-type]
    return typing.cast(CellArray, result)


def resolve_temperature(
    reservoir: Reservoir,
    deck_file: DeckFile | None,
    temperature: Temperature | Number | None,
    dtype: npt.DTypeLike = None,
    unit_system: UnitSystem | None = None,
) -> CellArray:
    """
    Resolve per-cell temperature. Explicit `temperature` kwarg > deck
    `RTEMP`/`TEMPVD` > error. (Restart temperature is not yet supported.)
    """
    n_cells = reservoir.n_cells
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()

    source: Temperature | None = None
    if isinstance(temperature, Temperature):
        source = temperature
    elif temperature is not None:
        return typing.cast(CellArray, np.full(n_cells, temperature, dtype=dtype))
    elif deck_file is not None and (
        deck_file.has("RTEMP") or deck_file.has("TEMPVD") or deck_file.has("RTEMPVD")
    ):
        # Ensure that the load `Temperture`'s unit syatem is consistent with what we need
        unit_system = unit_system if unit_system is not None else reservoir.unit_system
        source = Temperature.from_deck(deck_file, dtype=dtype).convert(unit_system)
    else:
        raise ValidationError(
            "No temperature source available. Pass `temperature=` explicitly "
            "(a constant, or a `Temperature`), or supply a `deck_file` "
            "with `RTEMP`/`TEMPVD`."
        )

    pvt_region_index: IntCellArray = (
        reservoir.regions.pvt_region
        if reservoir.regions is not None and reservoir.regions.pvt_region is not None
        else typing.cast(IntCellArray, np.ones(n_cells, dtype=np.int32))
    )
    return get_temperature_array_from_regions(
        source,
        pvt_region_index=pvt_region_index,
        depth=reservoir.depth,
        dtype=dtype,
    )


def compute_water_phase_pressure(
    reservoir: Reservoir,
    state: ReservoirState,
    satfunc: SatFunc | None = None,
) -> CellArray:
    """
    Initial water-phase pressure of every cell, which is the oil-phase pressure less the
    oil-water capillary pressure at the cell's saturations (`Pw = Po - Pcow`).

    :param reservoir: Reservoir geometry and rock properties, which give each cell's
        endpoint saturations and saturation region.
    :param state: The reservoir state, whose `pressure` is the oil-phase pressure.
    :param satfunc: The saturation functions. Without them, or for a region that has no
        capillary pressure, `Pcow` is taken as zero.
    :returns: Water-phase pressure of every cell, in the same unit system as `state`.
    """
    water_pressure = np.asarray(state.pressure).copy()
    if satfunc is None:
        return typing.cast(CellArray, water_pressure)

    saturation_region = (
        reservoir.regions.saturation_region if reservoir.regions is not None else None
    )
    if saturation_region is None:
        saturation_region = typing.cast(
            IntCellArray, np.ones(water_pressure.shape[0], dtype=np.int32)
        )

    rock = reservoir.rock
    for satnum, mask in iterate_regions(saturation_region):
        capillary_pressure = satfunc.region(satnum).capillary_pressure
        if capillary_pressure is None:
            continue
        oil_water = capillary_pressure.evaluate(
            state.water_saturation[mask],  # type: ignore[arg-type]
            state.oil_saturation[mask],  # type: ignore[arg-type]
            state.gas_saturation[mask],  # type: ignore[arg-type]
            irreducible_water_saturation=rock.connate_water_saturation[mask],  # type: ignore[arg-type]
            residual_oil_saturation_water=rock.residual_oil_saturation_water[mask],  # type: ignore[arg-type]
            residual_oil_saturation_gas=rock.residual_oil_saturation_gas[mask],  # type: ignore[arg-type]
            residual_gas_saturation=rock.residual_gas_saturation[mask],  # type: ignore[arg-type]
        )["oil_water"]
        water_pressure[mask] -= np.asarray(oil_water, dtype=water_pressure.dtype)  # type: ignore
    return typing.cast(CellArray, water_pressure)


WATER_BUBBLE_POINT_ITERATIONS = 60
"""Bisection steps used to find the pressure at which saturated water holds a given `Rsw`."""


def initialize_water_gas_state(
    *,
    pvt: PVT,
    pvt_region_index: IntCellArray,
    pressure: CellArray,
    temperature: CellArray,
    gas_solubility_in_water: CellArray | None = None,
    salinity: CellArray | None = None,
    iterations: Integer = WATER_BUBBLE_POINT_ITERATIONS,
    dtype: npt.DTypeLike = None,
) -> tuple[CellArray, CellArray]:
    """
    Initial gas dissolved in water (`Rsw`) and the water bubble-point pressure of every cell.

    Each `PVTNUM` region's own tables are used, with `Rsw` read from its gas table:

    - Without `gas_solubility_in_water`, the water is taken as saturated with gas, `Rsw` is the
      saturated value at the cell's pressure and temperature, and the bubble-point pressure is the
      water table's value at that pressure. This is what `solution_gor` and
      `oil_bubble_point_pressure` do for oil when no `RS` is supplied.
    - With `gas_solubility_in_water`, the bubble-point pressure is the pressure at which saturated
      water holds that `Rsw`, capped at the cell's pressure since water cannot hold more gas than
      saturation. An `Rsw` below what the table holds at the lowest pressure searched (0.1% of the
      cell's pressure) gets that lowest pressure.

    Looking up `Rsw` needs a salinity, so a region whose gas table holds `Rsw` data and has no
    default salinity needs `salinity` to be given.

    A region whose gas table has no `Rsw` data gets zero `Rsw` and zero bubble-point pressure when
    `Rsw` is not supplied. When it is supplied there, the bubble-point pressure falls back to the
    cell's pressure (saturated water).

    :param pvt: PVT tables for all regions.
    :param pvt_region_index: `PVTNUM` per cell.
    :param pressure: Shape `(n_cells,)` cell pressure.
    :param temperature: Shape `(n_cells,)` cell temperature.
    :param gas_solubility_in_water: Optional shape `(n_cells,)` explicit `Rsw`.
    :param salinity: Optional shape `(n_cells,)` water salinity (ppm NaCl). Each table's own
        default salinity is used where `None`.
    :param iterations: Bisection steps used to find the bubble-point pressure of an explicit `Rsw`.
    :param dtype: Output dtype. `bores.precision.get_dtype()` if not given.
    :returns: The `Rsw` array and the water bubble-point pressure array.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    n_cells = len(pressure)
    solubility = np.zeros(n_cells, dtype=dtype)
    bubble_point = np.zeros(n_cells, dtype=dtype)

    for pvtnum, mask in iterate_regions(pvt_region_index):
        tables = pvt.region(pvtnum).tables
        p = pressure[mask]  # type: ignore
        t = temperature[mask]  # type: ignore
        s = salinity[mask] if salinity is not None else None  # type: ignore
        gas = tables.gas
        water = tables.water

        saturated = (
            gas.solubility_in_water(p, t, salinity=s)
            if gas is not None and gas.has_solubility_in_water
            else None
        )

        if gas_solubility_in_water is None:
            if saturated is None:
                continue
            solubility[mask] = saturated  # type: ignore
            region_bubble_point = (
                water.bubble_point_pressure(t, pressure=p, salinity=s)
                if water is not None
                else None
            )
            bubble_point[mask] = region_bubble_point if region_bubble_point is not None else p  # type: ignore
            continue

        rsw = np.asarray(gas_solubility_in_water[mask], dtype=dtype)  # type: ignore
        solubility[mask] = rsw  # type: ignore
        if saturated is None:
            bubble_point[mask] = p  # type: ignore
            continue

        # Saturated `Rsw` rises with pressure, so the bubble point is found by bisection between
        # a pressure low enough to hold less than `rsw` and the cell's own pressure.
        saturated = np.broadcast_to(saturated, p.shape)
        search_dtype = np.result_type(p.dtype, dtype)
        low = np.asarray(p * 1e-3, dtype=search_dtype)
        high = np.array(p, dtype=search_dtype, copy=True)
        middle = np.empty_like(high)
        above = np.empty(p.shape, dtype=np.bool_)
        for _ in range(iterations):
            np.add(low, high, out=middle)
            middle *= 0.5
            holds = gas.solubility_in_water(middle, t, salinity=s)  # type: ignore[union-attr]
            below = np.broadcast_to(holds, p.shape) < rsw  # type: ignore[arg-type]
            np.logical_not(below, out=above)
            np.copyto(low, middle, where=below)
            np.copyto(high, middle, where=above)

        region_bubble_point = 0.5 * (low + high)
        # At or above the saturated value the water is saturated, so its bubble point is `p`.
        bubble_point[mask] = np.where(rsw >= saturated, p, region_bubble_point)  # type: ignore
    return typing.cast(CellArray, solubility), typing.cast(CellArray, bubble_point)


def initialize_reservoir_state(
    reservoir: Reservoir,
    pvt: PVT,
    *,
    deck_file: DeckFile | None = None,
    equilibrium: Equilibrium | None = None,
    satfunc: SatFunc | None = None,
    temperature: Temperature | Number | None = None,
    pressure: CellArray | None = None,
    water_saturation: CellArray | None = None,
    gas_saturation: CellArray | None = None,
    solution_gor: CellArray | None = None,
    vaporized_oil_gas_ratio: CellArray | None = None,
    gas_solubility_in_water: CellArray | None = None,
    water_salinity: CellArray | None = None,
    depth_step: Number = 1.0,
    hysteresis_enabled: bool = False,
    dtype: npt.DTypeLike = None,
    saturation_samples: int = N_SATURATION_SAMPLES,
) -> ReservoirState:
    """
    Build a complete `ReservoirState` from equilibration data (e.g `EQUIL`) and/or explicit arrays.

    "Explicit array/keyword" means either the corresponding kwarg here, or
    (if the kwarg is `None`) the matching `PRESSURE`/`SWAT`/`SGAS`/`RS`/`RV`
    keyword in `deck_file`. Any field left uncovered by both is filled in
    from equilibration, which therefore only runs at all if at least one
    field needs it. `oil_saturation` is always derived as
    `1 - water_saturation - gas_saturation`, never taken from an explicit array.

    `oil_bubble_point_pressure` and `gas_dew_point_pressure` are taken from
    equilibration when it derives the matching `solution_gor` and `vaporized_oil_gas_ratio`.
    Wherever one of those is given explicitly, or no equilibration is involved, each
    `PVTNUM` region's own PVT tables are queried per-cell -
    `oil_bubble_point_pressure` from `PVTTable.bubble_point_pressure(temperature,
    solution_gor)` on the oil table, `gas_dew_point_pressure` from
    `PVTTable.dew_point_pressure(temperature, vaporized_oil_ratio)` on the gas table - and only
    fall back to an assumption where a table has no such data: saturated oil
    (`oil_bubble_point_pressure = pressure`) and no free gas ever
    (`gas_dew_point_pressure = 0`).

    Gas dissolved in water (`gas_solubility_in_water`) is never derived by equilibration. It is
    taken from the `gas_solubility_in_water` kwarg, or the `RSW` keyword in `deck_file`, and
    otherwise each `PVTNUM` region's gas table gives the value for water saturated with gas at the
    cell's pressure and temperature, which is zero where that table has no `Rsw` data. The
    `water_bubble_point_pressure` follows from it, see `initialize_water_gas_state`.

    Note:
    Ensure the unit system of all inputs (`reservoir`, `pvt`, ..., and any explicit arrays) is consistent.
    The function will raise a `ValidationError` if there is a detected mismatch.

    :param reservoir: Reservoir geometry and rock properties.
    :param pvt: PVT tables for all regions.
    :param deck_file: Optional `DeckFile` to read explicit arrays from.
    :param equilibrium: Optional `Equilibrium` for any fields not covered by an
        explicit array/keyword.
    :param satfunc: Optional `SatFunc` for capillary-pressure-based
        saturations instead of a sharp contact. If supplied and
        `reservoir.regions.saturation_region` (SATNUM) is unavailable, every
        cell defaults to saturation region 1 and a `UserWarning` is raised
        (see `initialize_equilibrium_arrays`).
    :param temperature: Optional per-cell temperature (constant or `Temperature`)
        for equilibration. If `None`, will be read from `deck_file` if available.
    :param pressure: Optional explicit pressure array (overrides `PRESSURE` keyword).
    :param water_saturation: Optional explicit water saturation array (overrides `SWAT` keyword).
    :param gas_saturation: Optional explicit gas saturation array (overrides `SGAS` keyword).
    :param solution_gor: Optional explicit solution GOR array (overrides `RS` keyword).
    :param vaporized_oil_gas_ratio: Optional explicit vaporized oil ratio array (overrides `RV` keyword).
    :param gas_solubility_in_water: Optional explicit gas solubility in water (`Rsw`) array, in the
        same units as `solution_gor` (overrides `RSW` keyword).
    :param water_salinity: Optional per-cell water salinity (ppm NaCl) used to look up `Rsw` and the
        water bubble-point pressure. Each PVT table's own default salinity is used if `None`.
    :param depth_step: Step size for hydrostatic integration in equilibration.
    :param hysteresis_enabled: If `True`, initialize hysteresis state from the
        resolved saturations (`Hysteresis.from_saturation`) for
        later scanning-curve tracking. Works regardless of whether
        saturations came from a sharp contact or capillary-pressure
        inversion; `satfunc` is not itself required.
    :param dtype: Preferred dtype for the returned `ReservoirState`. Defaults to `get_dtype()`
    :param saturation_samples: Sample count for the capillary-pressure
        inversion grid, forwarded to `initialize_equilibrium_arrays` when
        `satfunc` is supplied. Defaults to `N_SATURATION_SAMPLES`.
    :raises NotImplementedError: If `deck_file` has a `RESTART` keyword
        (not yet supported).
    :raises ValidationError: If some field is covered by neither an
        explicit array/keyword nor `equilibrium`/`deck_file`, if `reservoir`
        and `pvt` unit systems disagree, or if `equilibrium`/`satfunc`
        (when supplied) don't share that same unit system, or if
        saturations are physically inconsistent.
    :warns UserWarning: If `satfunc` is supplied but
        `reservoir.regions.saturation_region` (`SATNUM`) is unavailable; see
        `initialize_equilibrium_arrays`. Also if exactly one of
        `pressure`/`solution_gor` (or `pressure`/`vaporized_oil_gas_ratio`) is
        explicit while the other is equilibration-derived, since `EQUIL`
        derives `solution_gor`/`vaporized_oil_gas_ratio` from its own
        internally-integrated hydrostatic pressure profile, not from an
        explicit `pressure` override as the two may end up off the oil
        bubble-point / gas dew-point curve together.
    """
    if equilibrium is None and deck_file is not None and deck_file.has("RESTART"):
        raise NotImplementedError(
            "Restart-file state initialization is not yet supported. "
            "Supply `equilibrium` and/or explicit arrays instead."
        )

    n_cells = reservoir.n_cells
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    unit_system = reservoir.unit_system
    if pvt.unit_system != unit_system:
        raise ValidationError(
            f"`reservoir.unit_system` ({unit_system.value!r}) does not "
            f"match `pvt.unit_system` ({pvt.unit_system.value!r})."
        )
    if equilibrium is not None and equilibrium.unit_system != unit_system:
        raise ValidationError(
            f"`reservoir.unit_system` ({unit_system.value!r}) does not "
            f"match `equilibrium.unit_system` ({equilibrium.unit_system.value!r})."
        )
    if satfunc is not None and satfunc.unit_system != unit_system:
        raise ValidationError(
            f"`reservoir.unit_system` ({unit_system.value!r}) does not "
            f"match `satfunc.unit_system` ({satfunc.unit_system.value!r})."
        )
    if isinstance(temperature, Temperature) and temperature.unit_system != unit_system:
        raise ValidationError(
            f"`reservoir.unit_system` ({unit_system.value!r}) does not "
            f"match `temperature.unit_system` ({temperature.unit_system.value!r})."
        )

    temperature_array = resolve_temperature(
        reservoir=reservoir,
        deck_file=deck_file,
        temperature=temperature,
        dtype=dtype,
        unit_system=unit_system,
    )
    explicit: dict[str, CellArray | None] = {
        "pressure": pressure,
        "water_saturation": water_saturation,
        "gas_saturation": gas_saturation,
        "solution_gor": solution_gor,
        "vaporized_oil_gas_ratio": vaporized_oil_gas_ratio,
    }
    field_keywords = {
        "pressure": "PRESSURE",
        "water_saturation": "SWAT",
        "gas_saturation": "SGAS",
        "solution_gor": "RS",
        "vaporized_oil_gas_ratio": "RV",
    }
    if deck_file is not None:
        for field, keyword in field_keywords.items():
            if explicit[field] is None:
                deck_array = deck_file.get(keyword)
                if deck_array is not None:
                    explicit[field] = typing.cast(CellArray, deck_array)

    equilibrium_arrays: EquilibriumArrays | None = None
    if any(value is None for value in explicit.values()):
        if equilibrium is None:
            if deck_file is not None and deck_file.get("EQUIL"):
                equilibrium = Equilibrium.from_deck(deck_file).convert(unit_system)
            else:
                missing = [field for field, value in explicit.items() if value is None]
                raise ValidationError(
                    f"No explicit array/keyword and no `EQUIL` data available "
                    f"for: {missing}. Supply `equilibrium`, a "
                    "`deck_file` with an `EQUIL` keyword, or explicit arrays "
                    "for these fields."
                )
        equilibrium_arrays = initialize_equilibrium_arrays(
            reservoir=reservoir,
            pvt=pvt,
            equilibrium=equilibrium,
            temperature=temperature_array,
            satfunc=satfunc,
            depth_step=depth_step,
            dtype=dtype,
            saturation_samples=saturation_samples,
        )

        # `pressure` and `solution_gor`/`vaporized_oil_gas_ratio` are physically
        # coupled through the oil bubble-point / gas dew-point curve. `EQUIL`
        # derives Rs/Rv from its own internally-integrated hydrostatic
        # pressure profile, not from an explicit `pressure` override, so
        # taking exactly one of the pair from `EQUIL` and the other from an
        # explicit array/keyword can silently produce a state that isn't on
        # that curve (e.g. an Rs inconsistent with the stored pressure).
        # Supplying both explicitly, or neither, keeps them consistent by
        # construction; this only warns (rather than raising) since a
        # deliberately mismatched pair, e.g. a measured `pressure` history-
        # matched against an `EQUIL`-only Rs/Rv profile, is a legitimate,
        # if unusual, use of the per-field override contract above.
        pressure_is_explicit = explicit["pressure"] is not None
        for gas_ratio_field in ("solution_gor", "vaporized_oil_gas_ratio"):
            if pressure_is_explicit != (explicit[gas_ratio_field] is not None):
                warnings.warn(
                    f"`pressure` is explicit but `{gas_ratio_field}` is "
                    "`EQUIL`-derived (or vice versa); the two may be "
                    "physically inconsistent since `EQUIL` derives "
                    f"`{gas_ratio_field}` from its own internally-integrated "
                    "hydrostatic pressure profile, not from the explicit "
                    "`pressure` array. Supply both explicitly, or neither, "
                    "to guarantee a state consistent with the oil "
                    "bubble-point / gas dew-point curve.",
                    UserWarning,
                    stacklevel=2,
                )

    def resolve(field: str, equilibrium_value: CellArray | None) -> CellArray:
        value = explicit[field]
        if value is not None:
            array = np.asarray(value, dtype=dtype)
            if array.shape != (n_cells,):
                raise ValidationError(
                    f"Explicit `{field}` has shape {array.shape}; expected ({n_cells},)."
                )
            return typing.cast(CellArray, array)
        assert equilibrium_value is not None  # Guaranteed by the branch above
        return equilibrium_value

    pressure_array = resolve(
        "pressure",
        equilibrium_arrays.pressure if equilibrium_arrays is not None else None,
    )
    water_saturation_array = resolve(
        "water_saturation",
        equilibrium_arrays.water_saturation if equilibrium_arrays is not None else None,
    )
    gas_saturation_array = resolve(
        "gas_saturation",
        equilibrium_arrays.gas_saturation if equilibrium_arrays is not None else None,
    )
    solution_gor_array = resolve(
        "solution_gor",
        equilibrium_arrays.solution_gor if equilibrium_arrays is not None else None,
    )
    vaporized_oil_ratio_array = resolve(
        "vaporized_oil_gas_ratio",
        equilibrium_arrays.vaporized_oil_gas_ratio if equilibrium_arrays is not None else None,
    )
    if equilibrium_arrays is not None:
        oil_bubble_point_pressure_array = equilibrium_arrays.oil_bubble_point_pressure
        gas_dew_point_pressure_array = equilibrium_arrays.gas_dew_point_pressure
    else:
        # Filled in per-`PVTNUM` below from each region's own PVT tables.
        # `oil_bubble_point_pressure_array` falls back to `pressure_array` (assume
        # saturated oil) and `gas_dew_point_pressure_array` falls back to zero
        # (assume no free gas) only where a region's table has no
        # bubble-/dew-point data of its own.
        oil_bubble_point_pressure_array = pressure_array.copy()
        gas_dew_point_pressure_array = np.zeros(n_cells, dtype=dtype)

    for field_name, field_array in (
        ("pressure", pressure_array),
        ("water_saturation", water_saturation_array),
        ("gas_saturation", gas_saturation_array),
        ("solution_gor", solution_gor_array),
        ("vaporized_oil_gas_ratio", vaporized_oil_ratio_array),
    ):
        if not all_finite(values=field_array):
            raise ValidationError(f"`{field_name}` contains NaN or infinite values.")

    if n_cells > 0 and pressure_array.min() <= 0.0:
        raise ValidationError("`pressure` must be positive in every cell.")
    oil_saturation_array = np.empty(n_cells, dtype=dtype)
    (
        lowest_water,
        highest_water,
        lowest_gas,
        highest_gas,
        largest_saturation_excess,
    ) = derive_oil_saturation(
        water_saturation=water_saturation_array,
        gas_saturation=gas_saturation_array,
        oil_saturation=oil_saturation_array,
    )
    for field_name, lowest, highest in (
        ("water_saturation", lowest_water, highest_water),
        ("gas_saturation", lowest_gas, highest_gas),
    ):
        if lowest < -1e-6 or highest > 1.0 + 1e-6:
            raise ValidationError(
                f"`{field_name}` must be between 0 and 1 in every cell; got values from "
                f"{lowest:.4g} to {highest:.4g}."
            )
    if largest_saturation_excess > 1e-6:
        raise ValidationError(
            "water_saturation + gas_saturation exceeds 1.0 in at least one "
            f"cell (max excess {largest_saturation_excess:.3e})."
        )

    pore_volumes = reservoir.compute_pore_volumes(pressure_array)
    pvt_region_index: IntCellArray = (
        reservoir.regions.pvt_region
        if reservoir.regions is not None and reservoir.regions.pvt_region is not None
        else typing.cast(IntCellArray, np.ones(n_cells, dtype=np.int32))
    )

    # A bubble/dew point equilibration returns belongs to the `Rs`/`Rv` it derived, so
    # an explicit `Rs`/`Rv` has to get its own from the PVT tables instead.
    bubble_point_from_tables = equilibrium_arrays is None or explicit["solution_gor"] is not None
    dew_point_from_tables = (
        equilibrium_arrays is None or explicit["vaporized_oil_gas_ratio"] is not None
    )
    if bubble_point_from_tables or dew_point_from_tables:
        for pvtnum, mask in iterate_regions(pvt_region_index):
            pvt_region = pvt.region(pvtnum)
            t = temperature_array[mask]  # type: ignore

            if bubble_point_from_tables:
                oil_bubble_point = pvt_region.tables.oil.bubble_point_pressure(  # type: ignore[union-attr]
                    temperature=t,
                    solution_gor=solution_gor_array[mask],  # type: ignore
                )
                if oil_bubble_point is not None:
                    oil_bubble_point_pressure_array[mask] = oil_bubble_point  # type: ignore

            if dew_point_from_tables and pvt_region.tables.gas is not None:
                gas_dew_point = pvt_region.tables.gas.dew_point_pressure(
                    temperature=t,
                    vaporized_oil_ratio=vaporized_oil_ratio_array[mask],  # type: ignore
                )
                if gas_dew_point is not None:
                    gas_dew_point_pressure_array[mask] = gas_dew_point  # type: ignore

    solubility_in_water_input = gas_solubility_in_water
    if solubility_in_water_input is None and deck_file is not None:
        deck_solubility = deck_file.get("RSW")
        if deck_solubility is not None:
            solubility_in_water_input = typing.cast(CellArray, deck_solubility)
    if solubility_in_water_input is not None:
        solubility_in_water_input = np.asarray(solubility_in_water_input, dtype=dtype)
        if solubility_in_water_input.shape != (n_cells,):
            raise ValidationError(
                f"Explicit `gas_solubility_in_water` has shape {solubility_in_water_input.shape}; "
                f"expected ({n_cells},)."
            )
        if not all_finite(values=solubility_in_water_input) or (  # type: ignore[arg-type]
            n_cells > 0 and solubility_in_water_input.min() < 0.0
        ):
            raise ValidationError(
                "`gas_solubility_in_water` must be finite and non-negative in every cell."
            )
    if water_salinity is not None:
        water_salinity = np.asarray(water_salinity, dtype=dtype)  # type: ignore
        if water_salinity.shape != (n_cells,):  # type: ignore[attr-defined]
            raise ValidationError(
                f"`water_salinity` has shape {water_salinity.shape}; expected ({n_cells},)."  # type: ignore[attr-defined]
            )
    gas_solubility_in_water_array, water_bubble_point_pressure_array = initialize_water_gas_state(
        pvt=pvt,
        pvt_region_index=pvt_region_index,
        pressure=pressure_array,
        temperature=temperature_array,
        gas_solubility_in_water=typing.cast(CellArray | None, solubility_in_water_input),
        salinity=water_salinity,
        dtype=dtype,
    )

    masses = compute_masses(
        pressure=pressure_array,
        temperature=temperature_array,
        oil_saturation=oil_saturation_array,
        water_saturation=water_saturation_array,
        gas_saturation=gas_saturation_array,
        solution_gor=solution_gor_array,
        vaporized_oil_gas_ratio=vaporized_oil_ratio_array,
        pore_volumes=pore_volumes,
        pvt=pvt,
        pvt_region_index=pvt_region_index,
        gas_solubility_in_water=gas_solubility_in_water_array,
        unit_system=unit_system,
        dtype=dtype,
    )

    hysteresis = (
        Hysteresis.from_saturation(
            water_saturation=water_saturation_array,
            gas_saturation=gas_saturation_array,
            dtype=dtype,
        )
        if hysteresis_enabled
        else None
    )
    return ReservoirState(
        pressure=pressure_array,
        temperature=temperature_array,
        oil_saturation=oil_saturation_array,  # type: ignore[arg-type]
        water_saturation=water_saturation_array,
        gas_saturation=gas_saturation_array,
        solution_gor=solution_gor_array,
        oil_bubble_point_pressure=typing.cast(CellArray, oil_bubble_point_pressure_array),
        vaporized_oil_gas_ratio=vaporized_oil_ratio_array,
        gas_dew_point_pressure=typing.cast(CellArray, gas_dew_point_pressure_array),
        gas_solubility_in_water=gas_solubility_in_water_array,
        water_bubble_point_pressure=water_bubble_point_pressure_array,
        oil_mass=masses.oil_mass,
        water_mass=masses.water_mass,
        free_gas_mass=masses.free_gas_mass,
        dissolved_gas_mass_in_oil=masses.dissolved_gas_mass_in_oil,
        dissolved_gas_mass_in_water=masses.dissolved_gas_mass_in_water,
        vaporized_oil_mass_in_gas=masses.vaporized_oil_mass_in_gas,
        hysteresis=hysteresis,
        unit_system=unit_system,
    )

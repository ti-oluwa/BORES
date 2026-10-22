"""
Utilities for reservoir equilibrium state initialization from EQUIL, explicit deck arrays, or user-supplied
arrays.

Precedence per field is:

```
RESTART (not yet supported but planned) -> explicit array/keyword -> EQUIL
```
"""

import typing
import warnings

import numpy as np
import numpy.typing as npt

from bores.blackoil.pvt.regions import PVT
from bores.blackoil.pvt.tables import PVTTable
from bores.blackoil.satfunc.capillary_pressure.tables import (
    CapillaryPressureTable,
)
from bores.blackoil.satfunc.regions import SatFunc
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.precision import get_dtype
from bores.reservoir.equilibrium import (
    DepthTable,
    Equilibrium,
    EquilibriumRegion,
)
from bores.reservoir.mass import compute_masses
from bores.reservoir.model import Reservoir
from bores.reservoir.state import Hysteresis, ReservoirState
from bores.reservoir.temperature import (
    Temperature,
    TemperatureGradient,
    TemperatureTable,
)
from bores.types import (
    Boolean,
    CellArray,
    IntArray,
    IntCellArray,
    Number,
    NumberArray,
    OneDimension,
    TwoDimensions,
    UnitSystem,
)
from bores.utils import get_hydrostatic_gradient_factor

__all__ = [
    "EquilibriumArrays",
    "compute_water_phase_pressure",
    "initialize_equilibrium_arrays",
    "initialize_reservoir_state",
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
    depths_up, pressures_up = _rk4_march(
        density_fn=density_fn,
        reference_depth=reference_depth,
        reference_pressure=reference_pressure,
        endpoint_depth=min_depth,
        gradient_factor=gradient_factor,
        step=step,
        dtype=dtype,
    )
    depths_down, pressures_down = _rk4_march(
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


def invert_capillary_pressure_curves(
    *,
    saturation_grid: NumberArray[typing.Any],
    capillary_pressure_grid: NumberArray[typing.Any],
    target_capillary_pressure: NumberArray[OneDimension],
    pressure_falls_with_saturation: Boolean,
    prefer_highest_pressure_end: Boolean,
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
    :param target_capillary_pressure: Pressure to invert each curve at, shape `(n_curves,)`.
    :param pressure_falls_with_saturation: Direction a curve is taken to run in when it is flat,
        which for the oil-water curve is `True` (`Pcow` falls as `Sw` rises) and for the
        gas-oil curve is `False` (`Pcgo` rises with `Sg`). A curve that is not flat runs in
        whichever direction its own values do.
    :param prefer_highest_pressure_end: Which end a target that equals both the lowest and the
        highest pressure of a flat curve resolves to. `True` picks the saturation at the
        highest-pressure end.
    :returns: `(saturation, capillary_pressure)`, each of shape `(n_curves,)`. The second is the
        target limited to the range of its curve.
    """
    sample_count = capillary_pressure_grid.shape[1]
    # Orient every curve so that pressure never decreases along the sample axis
    pressure_change = capillary_pressure_grid[:, -1] - capillary_pressure_grid[:, 0]
    descending = np.where(
        pressure_change == 0, pressure_falls_with_saturation, pressure_change < 0
    )
    if np.any(descending):
        flip = descending[:, None]
        saturation_grid = np.where(flip, saturation_grid[:, ::-1], saturation_grid)
        capillary_pressure_grid = np.where(
            flip, capillary_pressure_grid[:, ::-1], capillary_pressure_grid
        )

    rows = np.arange(capillary_pressure_grid.shape[0])
    lowest_pressure = capillary_pressure_grid[:, 0]
    highest_pressure = capillary_pressure_grid[:, -1]

    samples_below_target = np.count_nonzero(
        capillary_pressure_grid < target_capillary_pressure[:, None], axis=1
    )
    upper = np.clip(samples_below_target, 1, sample_count - 1)
    lower = upper - 1
    pressure_lower = capillary_pressure_grid[rows, lower]
    pressure_span = capillary_pressure_grid[rows, upper] - pressure_lower
    has_span = pressure_span > 0.0
    fraction = np.clip(
        np.where(
            has_span,
            (target_capillary_pressure - pressure_lower) / np.where(has_span, pressure_span, 1.0),
            1.0,
        ),
        0.0,
        1.0,
    )
    saturation_lower = saturation_grid[rows, lower]
    saturation = saturation_lower + fraction * (saturation_grid[rows, upper] - saturation_lower)

    at_lowest_end = target_capillary_pressure <= lowest_pressure
    at_highest_end = target_capillary_pressure >= highest_pressure
    lowest_end_saturation = saturation_grid[:, 0]
    highest_end_saturation = saturation_grid[:, -1]
    if prefer_highest_pressure_end:
        saturation = np.where(at_lowest_end, lowest_end_saturation, saturation)
        saturation = np.where(at_highest_end, highest_end_saturation, saturation)
    else:
        saturation = np.where(at_highest_end, highest_end_saturation, saturation)
        saturation = np.where(at_lowest_end, lowest_end_saturation, saturation)

    capillary_pressure = np.clip(target_capillary_pressure, lowest_pressure, highest_pressure)
    return typing.cast(
        tuple[NumberArray[OneDimension], NumberArray[OneDimension]],
        (saturation, capillary_pressure),
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
    # Bounds the size of the `(cells, samples)` arrays built per pass
    chunk_size = max(1, CAPILLARY_INVERSION_BATCH_SIZE // saturation_samples)

    if water_profile is not None:
        water_pressure_at_depth = np.interp(depths, water_profile[0], water_profile[1])
        target_pcow = oil_pressure - water_pressure_at_depth  # Pcow = Po - Pw
        for start in range(0, n, chunk_size):
            cells = slice(start, min(start + chunk_size, n))
            swc_chunk = swc[cells]
            saturation_grid = swc_chunk[:, None] + fractions[None, :] * (1.0 - swc_chunk)[:, None]
            flat_saturation = saturation_grid.ravel()
            pcow_grid = capillary_pressure.evaluate(
                flat_saturation,
                1.0 - flat_saturation,
                np.zeros_like(flat_saturation),
                irreducible_water_saturation=np.repeat(swc_chunk, saturation_samples),
                residual_oil_saturation_water=np.repeat(sorw[cells], saturation_samples),
                residual_oil_saturation_gas=np.repeat(sorg[cells], saturation_samples),
                residual_gas_saturation=np.repeat(sgr[cells], saturation_samples),
            )["oil_water"]
            water_saturation[cells], oil_water_capillary_pressure[cells] = (
                invert_capillary_pressure_curves(
                    saturation_grid=saturation_grid,
                    capillary_pressure_grid=np.reshape(pcow_grid, saturation_grid.shape),
                    target_capillary_pressure=typing.cast(
                        NumberArray[OneDimension], target_pcow[cells]
                    ),
                    pressure_falls_with_saturation=True,
                    prefer_highest_pressure_end=True,  # A cell exactly on the contact is oil-bearing
                )
            )

    if gas_profile is not None:
        gas_pressure_at_depth = np.interp(depths, gas_profile[0], gas_profile[1])
        target_pcgo = gas_pressure_at_depth - oil_pressure  # Pcgo = Pg - Po
        for start in range(0, n, chunk_size):
            cells = slice(start, min(start + chunk_size, n))
            swc_chunk = swc[cells]
            saturation_grid = fractions[None, :] * (1.0 - swc_chunk)[:, None]
            flat_saturation = saturation_grid.ravel()
            flat_connate = np.repeat(swc_chunk, saturation_samples)
            pcgo_grid = capillary_pressure.evaluate(
                flat_connate,
                np.clip(1.0 - flat_connate - flat_saturation, 0.0, 1.0),
                flat_saturation,
                irreducible_water_saturation=flat_connate,
                residual_oil_saturation_water=np.repeat(sorw[cells], saturation_samples),
                residual_oil_saturation_gas=np.repeat(sorg[cells], saturation_samples),
                residual_gas_saturation=np.repeat(sgr[cells], saturation_samples),
            )["gas_oil"]
            gas_saturation[cells], gas_oil_capillary_pressure[cells] = (
                invert_capillary_pressure_curves(
                    saturation_grid=saturation_grid,
                    capillary_pressure_grid=np.reshape(pcgo_grid, saturation_grid.shape),
                    target_capillary_pressure=typing.cast(
                        NumberArray[OneDimension], target_pcgo[cells]
                    ),
                    pressure_falls_with_saturation=False,
                    prefer_highest_pressure_end=False,  # A cell exactly on the contact is oil-bearing
                )
            )

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
    oil_depths_up, oil_pressures_up = _rk4_march(
        density_fn=compute_oil_density,
        reference_depth=region.datum_depth,
        reference_pressure=region.datum_pressure,
        endpoint_depth=oil_zone_low,
        gradient_factor=gradient_factor,
        step=depth_step,
        dtype=dtype,
    )
    oil_depths_down, oil_pressures_down = _rk4_march(
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
            cap_depths, cap_pressures = _rk4_march(
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
            base_depths, base_pressures = _rk4_march(
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

    has_oil = (1.0 - water_saturation - gas_saturation) > 0.0

    # Solution GOR / bubble-point pressure
    solution_gor = np.zeros(n, dtype=dtype)
    oil_bubble_point_pressure = np.zeros(n, dtype=dtype)
    if np.any(has_oil):
        oil_pressure = pressure[has_oil]
        oil_temperature = temperature[has_oil]
        if rsvd_table is not None:
            gor = rsvd_table.at_depth(depths[has_oil]).astype(  # type: ignore[arg-type]
                dtype, copy=False
            )
            bubble_point_pressure = oil_table.bubble_point_pressure(
                temperature=oil_temperature, solution_gor=gor
            ).astype(dtype, copy=False)  # type: ignore[union-attr]
        else:
            # No RSVD: assume saturated oil (Pbub = P) everywhere in the oil
            # zone. Forcing `bubble_point_pressure=oil_pressure` makes the saturated
            # branch always trigger, reading Rs straight off the table at P.
            gor = oil_table.solution_gas_to_oil_ratio(
                pressure=oil_pressure,
                temperature=oil_temperature,
                bubble_point_pressure=oil_pressure,
            ).astype(dtype, copy=False)  # type: ignore[union-attr]
            bubble_point_pressure = oil_pressure.copy()

        solution_gor[has_oil] = gor
        oil_bubble_point_pressure[has_oil] = bubble_point_pressure

    # Vaporized oil ratio / dew-point pressure (dry gas: Rv = 0 unless RVVD)
    vaporized_oil_gas_ratio = np.zeros(n, dtype=dtype)
    gas_dew_point_pressure = np.zeros(n, dtype=dtype)
    has_free_gas = gas_saturation > 0.0
    if rvvd_table is not None and np.any(has_free_gas):
        vaporized_oil_gas_ratio[has_free_gas] = rvvd_table.at_depth(
            depths[has_free_gas]  # type: ignore[arg-type]
        ).astype(dtype, copy=False)  # type: ignore[union-attr]

        if gas_table is not None:
            # A 2-D `Pdew(Rv, T)` table needs the gas's Rv
            dew_point = gas_table.dew_point_pressure(
                temperature=temperature[has_free_gas],
                vaporized_oil_ratio=vaporized_oil_gas_ratio[has_free_gas],
            )
            if dew_point is not None:
                gas_dew_point_pressure[has_free_gas] = dew_point.astype(  # type: ignore[union-attr]
                    dtype, copy=False
                )
    elif is_wet_gas and gas_table is not None and np.any(has_free_gas):
        # No `RVVD`: assume saturated gas (`Pdew = P`), so `Rv = Rv_sat(P, T)`
        free_gas_pressure = pressure[has_free_gas]
        saturated_rv = gas_table.vaporized_oil_gas_ratio(
            free_gas_pressure, temperature[has_free_gas]
        )
        if saturated_rv is not None:
            vaporized_oil_gas_ratio[has_free_gas] = saturated_rv.astype(  # type: ignore[union-attr]
                dtype, copy=False
            )
            gas_dew_point_pressure[has_free_gas] = free_gas_pressure

    return EquilibriumArrays(
        pressure=typing.cast(CellArray, pressure),
        water_saturation=typing.cast(CellArray, water_saturation),
        gas_saturation=typing.cast(CellArray, gas_saturation),
        solution_gor=typing.cast(CellArray, solution_gor),
        oil_bubble_point_pressure=typing.cast(CellArray, oil_bubble_point_pressure),
        vaporized_oil_gas_ratio=typing.cast(CellArray, vaporized_oil_gas_ratio),
        gas_dew_point_pressure=typing.cast(CellArray, gas_dew_point_pressure),
    )


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

    :param sample_arrays: Arrays for every sample of every cell, cell-major, so that the samples of
        a cell are consecutive.
    :param n_cells: Number of cells.
    :param sample_weights: Weight of each sample, shape `(n_cells, n_samples)`.
    :returns: One value per cell for every array.
    """

    def to_samples(field: CellArray) -> NumberArray[TwoDimensions]:
        return typing.cast(NumberArray[TwoDimensions], np.reshape(field, (n_cells, -1)))

    oil_content = (
        1.0 - to_samples(sample_arrays.water_saturation) - to_samples(sample_arrays.gas_saturation)
    ).clip(0.0, 1.0) * sample_weights
    gas_content = to_samples(sample_arrays.gas_saturation) * sample_weights
    weights_by_field = {
        "solution_gor": oil_content,
        "oil_bubble_point_pressure": oil_content,
        "vaporized_oil_gas_ratio": gas_content,
        "gas_dew_point_pressure": gas_content,
    }

    averages = {}
    for name in EquilibriumArrays._fields:
        weights = weights_by_field.get(name, sample_weights)
        total_weight = weights.sum(axis=1)
        weighted_sum = (to_samples(getattr(sample_arrays, name)) * weights).sum(axis=1)
        # No oil (or gas) in any sample means the field is zero in every sample
        averages[name] = np.divide(
            weighted_sum,
            total_weight,
            out=np.zeros_like(weighted_sum),
            where=total_weight > 0.0,
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


def get_dip_aware_top_bottom_faces(
    grid: Grid, cell_indices: IntArray[OneDimension]
) -> tuple[
    NumberArray[OneDimension],
    NumberArray[OneDimension],
    NumberArray[OneDimension],
    NumberArray[OneDimension],
]:
    """
    For each cell, find its true top/bottom face depths and areas from grid
    geometry, rather than the cell's axis-aligned bounding box.

    Each cell's own faces are inspected (via `cell_face_offsets`/
    `cell_face_indices`) and classified as top/bottom candidates by their
    outward-normal z-component (`Grid.get_face_normal_for_cell`): a face
    whose normal is within 45 degrees of vertical is a candidate, since
    `z` is positive downward, an upward-pointing (negative-z) normal marks a
    top face and a downward-pointing (positive-z) normal marks a bottom
    face. Candidates are split into a shallow group and a deep group at
    their midpoint depth, and each group's area-weighted centroid depth and
    total area become that cell's top/bottom depth/area. This resolves
    correctly for a single quad top/bottom face (structured/corner-point
    hexahedra) as well as a top/bottom face triangulated into several
    polygons (general unstructured cells).

    A dipping cell's true top-to-bottom depth span (and area taper) along
    its own face-normal direction can differ materially from its AABB span,
    which is inflated by the lateral spread of all of its vertices; using
    the actual faces is what makes this dip-aware.

    Falls back to the cell's AABB z-range with equal top/bottom areas
    (i.e. uniform-weighting downstream) for any cell with fewer than two
    vertical-ish faces - e.g. a degenerate sliver cell, or a 2-D grid with
    no k-direction faces at all.

    :returns: `(top_depth, bottom_depth, top_area, bottom_area)`, each shape
        `(len(cell_indices),)`, `float64`.
    """
    n = len(cell_indices)
    # Use float64 here since the grid arrays are mostly in float64
    # also the logic here is precision sensitive. We can cast to
    # the preferred dtype in the caller
    top_depth = np.empty(n, dtype=np.float64)
    bottom_depth = np.empty(n, dtype=np.float64)
    top_area = np.empty(n, dtype=np.float64)
    bottom_area = np.empty(n, dtype=np.float64)

    min_abs_normal_z = np.cos(np.deg2rad(45.0))

    for i in range(n):
        cell_index = int(cell_indices[i])
        start = grid.cell_face_offsets[cell_index]
        end = grid.cell_face_offsets[cell_index + 1]
        faces = grid.cell_face_indices[start:end]

        candidate_depths = []
        candidate_areas = []
        for face_index in faces:
            face_index = int(face_index)
            normal = grid.get_face_normal_for_cell(face_index, cell_index)
            if abs(normal[2]) >= min_abs_normal_z:
                candidate_depths.append(grid.face_centroids[face_index, 2])
                candidate_areas.append(grid.face_areas[face_index])

        if len(candidate_depths) >= 2:
            depths_array = np.asarray(candidate_depths, dtype=np.float64)
            areas_array = np.asarray(candidate_areas, dtype=np.float64)
            midpoint = 0.5 * (depths_array.min() + depths_array.max())
            is_top = depths_array <= midpoint
            if not np.any(is_top) or np.all(is_top):
                # All candidates landed on one side of the midpoint (e.g.
                # a single degenerate face), so we split by the extremes instead.
                is_top = depths_array == depths_array.min()
            is_bottom = ~is_top

            top_depth[i] = np.average(depths_array[is_top], weights=areas_array[is_top])
            bottom_depth[i] = np.average(depths_array[is_bottom], weights=areas_array[is_bottom])
            top_area[i] = areas_array[is_top].sum()
            bottom_area[i] = areas_array[is_bottom].sum()
        else:
            top_depth[i] = grid.cell_min_xyz[cell_index, 2]
            bottom_depth[i] = grid.cell_max_xyz[cell_index, 2]
            top_area[i] = bottom_area[i] = 1.0

    return (
        typing.cast(NumberArray[OneDimension], top_depth),
        typing.cast(NumberArray[OneDimension], bottom_depth),
        typing.cast(NumberArray[OneDimension], top_area),
        typing.cast(NumberArray[OneDimension], bottom_area),
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
    `get_dip_aware_top_bottom_faces`) rather than its AABB to get a
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

    top_depth, bottom_depth, top_area, bottom_area = get_dip_aware_top_bottom_faces(
        grid=grid, cell_indices=cell_indices
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

    pressure = np.zeros(n_cells, dtype=dtype)
    water_saturation = np.zeros(n_cells, dtype=dtype)
    gas_saturation = np.zeros(n_cells, dtype=dtype)
    solution_gor = np.zeros(n_cells, dtype=dtype)
    oil_bubble_point_pressure = np.zeros(n_cells, dtype=dtype)
    vaporized_oil_gas_ratio = np.zeros(n_cells, dtype=dtype)
    gas_dew_point_pressure = np.zeros(n_cells, dtype=dtype)

    for eqlnum in np.unique(equilibrium_region_index):
        mask = equilibrium_region_index == eqlnum
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
                cell_indices=np.nonzero(mask)[0],  # type: ignore[arg-type]
                **initialization_kwargs,
            )

        pressure[mask] = region_arrays.pressure
        water_saturation[mask] = region_arrays.water_saturation
        gas_saturation[mask] = region_arrays.gas_saturation
        solution_gor[mask] = region_arrays.solution_gor
        oil_bubble_point_pressure[mask] = region_arrays.oil_bubble_point_pressure
        vaporized_oil_gas_ratio[mask] = region_arrays.vaporized_oil_gas_ratio
        gas_dew_point_pressure[mask] = region_arrays.gas_dew_point_pressure

    return EquilibriumArrays(
        pressure=typing.cast(CellArray, pressure),
        water_saturation=typing.cast(CellArray, water_saturation),
        gas_saturation=typing.cast(CellArray, gas_saturation),
        solution_gor=typing.cast(CellArray, solution_gor),
        oil_bubble_point_pressure=typing.cast(CellArray, oil_bubble_point_pressure),
        vaporized_oil_gas_ratio=typing.cast(CellArray, vaporized_oil_gas_ratio),
        gas_dew_point_pressure=typing.cast(CellArray, gas_dew_point_pressure),
    )


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
    for pvtnum in np.unique(pvt_region_index):
        mask = pvt_region_index == pvtnum
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
    for satnum in np.unique(saturation_region):
        capillary_pressure = satfunc.region(satnum).capillary_pressure
        if capillary_pressure is None:
            continue
        mask = saturation_region == satnum
        oil_water = capillary_pressure.evaluate(
            state.water_saturation[mask],
            state.oil_saturation[mask],
            state.gas_saturation[mask],
            irreducible_water_saturation=rock.connate_water_saturation[mask],
            residual_oil_saturation_water=rock.residual_oil_saturation_water[mask],
            residual_oil_saturation_gas=rock.residual_oil_saturation_gas[mask],
            residual_gas_saturation=rock.residual_gas_saturation[mask],
        )["oil_water"]
        water_pressure[mask] -= np.asarray(oil_water, dtype=water_pressure.dtype)
    return typing.cast(CellArray, water_pressure)


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
        if not np.all(np.isfinite(field_array)):
            raise ValidationError(f"`{field_name}` contains NaN or infinite values.")

    if np.any(pressure_array <= 0.0):
        raise ValidationError("`pressure` must be positive in every cell.")
    for field_name, field_array in (
        ("water_saturation", water_saturation_array),
        ("gas_saturation", gas_saturation_array),
    ):
        if np.any(field_array < -1e-6) or np.any(field_array > 1.0 + 1e-6):
            raise ValidationError(
                f"`{field_name}` must be between 0 and 1 in every cell; got values from "
                f"{float(field_array.min()):.4g} to {float(field_array.max()):.4g}."
            )

    saturation_sum_excess = (water_saturation_array + gas_saturation_array) - 1.0
    if np.any(saturation_sum_excess > 1e-6):
        raise ValidationError(
            "water_saturation + gas_saturation exceeds 1.0 in at least one "
            f"cell (max excess {float(saturation_sum_excess.max()):.3e})."
        )
    oil_saturation_array = np.clip(
        1.0 - water_saturation_array - gas_saturation_array, 0.0, 1.0, dtype=dtype
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
        for pvtnum in np.unique(pvt_region_index):
            mask = pvt_region_index == pvtnum
            pvt_region = pvt.region(pvtnum)
            t = temperature_array[mask]

            if bubble_point_from_tables:
                oil_bubble_point = pvt_region.tables.oil.bubble_point_pressure(  # type: ignore[union-attr]
                    temperature=t, solution_gor=solution_gor_array[mask]
                )
                if oil_bubble_point is not None:
                    oil_bubble_point_pressure_array[mask] = oil_bubble_point

            if dew_point_from_tables and pvt_region.tables.gas is not None:
                gas_dew_point = pvt_region.tables.gas.dew_point_pressure(
                    temperature=t, vaporized_oil_ratio=vaporized_oil_ratio_array[mask]
                )
                if gas_dew_point is not None:
                    gas_dew_point_pressure_array[mask] = gas_dew_point

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
        unit_system=unit_system,
        dtype=dtype,
    )

    zeros = np.zeros(n_cells, dtype=dtype)
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
        gas_solubility_in_water=zeros,
        water_bubble_point_pressure=zeros,
        oil_mass=masses.oil_mass,
        water_mass=masses.water_mass,
        free_gas_mass=masses.free_gas_mass,
        dissolved_gas_mass_in_oil=masses.dissolved_gas_mass_in_oil,
        dissolved_gas_mass_in_water=masses.dissolved_gas_mass_in_water,
        vaporized_oil_mass_in_gas=masses.vaporized_oil_mass_in_gas,
        hysteresis=hysteresis,
        unit_system=unit_system,
    )

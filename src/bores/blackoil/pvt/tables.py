import logging
import typing
import warnings
from os import PathLike

import attrs
import numpy as np
import numpy.typing as npt
from scipy.interpolate import (  # type: ignore[import-untyped]
    PchipInterpolator,
    RectBivariateSpline,
    RegularGridInterpolator,
    interp1d,
)
from typing_extensions import Self

from bores.blackoil.pvt.data import PVTData, PVTDataSet
from bores.blackoil.pvt.static import StaticPVT
from bores.constants import UnitConversionTable, get_conversion_factors
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.reservoir.temperature import Temperature
from bores.serde.stores import StoreSerializable
from bores.types import (
    Boolean,
    BooleanArray,
    FluidPhase,
    InterpolationMethod,
    NDimension,
    Number,
    NumberArray,
    OneDimension,
    TableQuery,
    TableResult,
    TwoDimensions,
    UnitSystem,
)

logger = logging.getLogger(__name__)

__all__ = ["PVTTable", "PVTTables"]

INTERPOLATION_DEGREES: dict[str, int] = {"linear": 1, "cubic": 3}


def build_pchip_2d_interpolator(
    pressures: NumberArray[NDimension],
    temperatures: NumberArray[NDimension],
    table: NumberArray[NDimension],
    dtype: npt.DTypeLike,
) -> typing.Callable[
    [NumberArray[NDimension], NumberArray[NDimension]], NumberArray[OneDimension]
]:
    """
    Build a two-stage PCHIP 2-D interpolator for a property table.

    Fits independent PCHIP interpolants along the pressure axis at each
    temperature knot, then re-interpolates along temperature for each query
    point. Preserves monotonicity along each axis independently.

    The returned callable has the same interface as
    `RectBivariateSpline.ev`: `(pressure_points, temperature_points) -> values`.

    :param pressures: 1-D array of pressure knots, strictly increasing.
    :param temperatures: 1-D array of temperature knots, strictly increasing.
    :param table: 2-D array of shape `(n_p, n_t)` containing property values.
    :returns: Callable with signature `(p, t) -> values`.
    """
    p_interps: list[PchipInterpolator] = [
        PchipInterpolator(pressures, table[:, j], extrapolate=True)
        for j in range(len(temperatures))
    ]

    def interpolator(
        p: NumberArray[NDimension], t: NumberArray[NDimension]
    ) -> NumberArray[OneDimension]:
        p = p.astype(dtype, copy=False).ravel()  # type: ignore
        t = t.astype(dtype, copy=False).ravel()  # type: ignore
        n = len(p)

        values = np.empty((len(temperatures), n), dtype=dtype)
        for j, interp in enumerate(p_interps):
            values[j] = interp(p)

        result = np.empty(n, dtype=dtype)
        for i in range(n):
            result[i] = PchipInterpolator(temperatures, values[:, i], extrapolate=True)(t[i])
        return result

    return interpolator


def build_pchip_2d_derivative_interpolator(
    pressures: NumberArray[NDimension],
    temperatures: NumberArray[NDimension],
    table: NumberArray[NDimension],
    dtype: npt.DTypeLike,
) -> typing.Callable[
    [NumberArray[NDimension], NumberArray[NDimension]], NumberArray[OneDimension]
]:
    """
    Build a two-stage PCHIP interpolator for `∂table/∂P`.

    For each temperature column, builds a PCHIP and takes its analytical
    first derivative (via `PchipInterpolator.derivative`), then
    re-interpolates along temperature exactly as the primal interpolator does.

    :param pressures: 1-D array of pressure knots, strictly increasing.
    :param temperatures: 1-D array of temperature knots, strictly increasing.
    :param table: 2-D array of shape `(n_p, n_t)`.
    :returns: Callable `(p, t) -> ∂table/∂P`.
    """
    dp_interps: list[PchipInterpolator] = [
        PchipInterpolator(pressures, table[:, j], extrapolate=True).derivative(1)
        for j in range(len(temperatures))
    ]

    def interpolator(
        p: NumberArray[NDimension], t: NumberArray[NDimension]
    ) -> NumberArray[OneDimension]:
        p = p.astype(dtype, copy=False).ravel()  # type: ignore
        t = t.astype(dtype, copy=False).ravel()  # type: ignore
        n = len(p)

        values = np.empty((len(temperatures), n), dtype=dtype)
        for j, d_interp in enumerate(dp_interps):
            values[j] = d_interp(p)

        result = np.empty(n, dtype=dtype)
        for i in range(n):
            result[i] = PchipInterpolator(temperatures, values[:, i], extrapolate=True)(t[i])
        return result

    return interpolator


def build_bilinear_2d_derivative_interpolator(
    pressures: NumberArray[NDimension],
    temperatures: NumberArray[NDimension],
    table: NumberArray[NDimension],
    dtype: npt.DTypeLike,
) -> typing.Callable[
    [NumberArray[NDimension], NumberArray[NDimension]], NumberArray[OneDimension]
]:
    """
    Build the exact ∂table/∂P of a *bilinear* (kx=ky=1) surface.

    `RectBivariateSpline.partial_derivative(1, 0)` requires `dx < kx`, which
    is never true for `kx=1` ("linear" interpolation) - scipy has no way to
    hand back a derivative object for a degree-1 spline. This computes it
    directly instead: within a pressure cell, `∂z/∂P` is the linear-in-y
    blend of the two bracketing columns' slopes, `(z[i+1,j] - z[i,j]) /
    (P[i+1] - P[i])`.
    """
    # Per-column slopes between consecutive pressure knots: shape (n_p-1, n_t)
    column_slopes = np.diff(table, axis=0) / np.diff(pressures)[:, np.newaxis]
    # One linear interpolator per pressure cell, blending slopes over temperature
    cell_interps = [
        interp1d(
            temperatures,
            column_slopes[i, :],
            kind="linear",
            bounds_error=False,
            fill_value=(column_slopes[i, 0], column_slopes[i, -1]),
        )
        for i in range(column_slopes.shape[0])
    ]

    def interpolator(
        p: NumberArray[NDimension], t: NumberArray[NDimension]
    ) -> NumberArray[OneDimension]:
        p = p.astype(dtype, copy=False).ravel()  # type: ignore
        t = t.astype(dtype, copy=False).ravel()  # type: ignore
        # Clip to the last cell at/above the top knot (flat extrapolation,
        # matching the value interpolant's own boundary behavior).
        cell_idx = np.clip(
            np.searchsorted(pressures, p, side="right") - 1,
            0,
            len(cell_interps) - 1,
        )
        result = np.empty(len(p), dtype=dtype)
        for idx in np.unique(cell_idx):
            mask = cell_idx == idx
            result[mask] = cell_interps[idx](t[mask])
        return result

    return interpolator


def build_pchip_3d_derivative_interpolator(
    pressures: NumberArray[NDimension],
    temperatures: NumberArray[NDimension],
    salinities: NumberArray[NDimension],
    table: NumberArray[NDimension],
    dtype: npt.DTypeLike,
) -> typing.Callable[[NumberArray[NDimension]], NumberArray[OneDimension]]:
    """
    Build a three-stage PCHIP interpolator for `∂table/∂P` over
    `(pressure, temperature, salinity)`.

    At every `(temperature, salinity)` grid node, builds a PCHIP along
    pressure and takes its analytical first derivative, exactly as
    `build_pchip_2d_derivative_interpolator` does per temperature column.
    The resulting derivative surface is then re-interpolated over salinity
    and temperature for each query point, same two-stage blend as the 2-D
    case with one extra axis.

    :param pressures: 1-D array of pressure knots, strictly increasing.
    :param temperatures: 1-D array of temperature knots, strictly increasing.
    :param salinities: 1-D array of salinity knots, strictly increasing.
    :param table: 3-D array of shape `(n_p, n_t, n_s)`.
    :returns: Callable `(points) -> ∂table/∂P`, where `points` is an
        `(n, 3)` array with columns `(pressure, temperature, salinity)`.
    """
    n_t = len(temperatures)
    n_s = len(salinities)
    dp_interps = [
        [
            PchipInterpolator(pressures, table[:, j, k], extrapolate=True).derivative(1)
            for k in range(n_s)
        ]
        for j in range(n_t)
    ]

    def interpolator(points: NumberArray[NDimension]) -> NumberArray[OneDimension]:
        points = typing.cast(NumberArray[TwoDimensions], np.atleast_2d(points))  # type: ignore
        p = points[:, 0].astype(dtype, copy=False)
        t = points[:, 1].astype(dtype, copy=False)
        s = points[:, 2].astype(dtype, copy=False)
        n = len(p)

        # ∂/∂P at every (temperature, salinity) grid node, at each query
        # point's own pressure.
        values = np.empty((n_t, n_s, n), dtype=dtype)
        for j in range(n_t):
            for k in range(n_s):
                values[j, k] = dp_interps[j][k](p)

        result = np.empty(n, dtype=dtype)
        for i in range(n):
            over_salinity = np.empty(n_t, dtype=dtype)
            for j in range(n_t):
                over_salinity[j] = PchipInterpolator(
                    salinities, values[j, :, i], extrapolate=True
                )(s[i])
            result[i] = PchipInterpolator(temperatures, over_salinity, extrapolate=True)(t[i])
        return result

    return interpolator


def build_bilinear_3d_derivative_interpolator(
    pressures: NumberArray[NDimension],
    temperatures: NumberArray[NDimension],
    salinities: NumberArray[NDimension],
    table: NumberArray[NDimension],
    dtype: npt.DTypeLike,
) -> typing.Callable[[NumberArray[NDimension]], NumberArray[OneDimension]]:
    """
    Build the exact `∂table/∂P` of a trilinear (`kx=ky=kz=1`) surface.

    Same reasoning as `build_bilinear_2d_derivative_interpolator`: within a
    pressure cell, `∂z/∂P` is the bracketing pressure columns' slope,
    `(z[i+1,j,k] - z[i,j,k]) / (P[i+1] - P[i])`, bilinearly blended over
    `(temperature, salinity)` for the query point.

    :param pressures: 1-D array of pressure knots, strictly increasing.
    :param temperatures: 1-D array of temperature knots, strictly increasing.
    :param salinities: 1-D array of salinity knots, strictly increasing.
    :param table: 3-D array of shape `(n_p, n_t, n_s)`.
    :returns: Callable `(points) -> ∂table/∂P`, where `points` is an
        `(n, 3)` array with columns `(pressure, temperature, salinity)`.
    """
    # Per-column slopes between consecutive pressure knots: shape (n_p-1, n_t, n_s)
    column_slopes = np.diff(table, axis=0) / np.diff(pressures)[:, np.newaxis, np.newaxis]
    slope_interps = [
        RegularGridInterpolator(
            points=(temperatures, salinities),
            values=column_slopes[i],
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        for i in range(column_slopes.shape[0])
    ]

    def interpolator(points: NumberArray[NDimension]) -> NumberArray[OneDimension]:
        points = typing.cast(NumberArray[TwoDimensions], np.atleast_2d(points))  # type: ignore
        p = points[:, 0].astype(dtype, copy=False)
        t = points[:, 1].astype(dtype, copy=False)
        s = points[:, 2].astype(dtype, copy=False)
        # Clip to the last cell at/above the top knot (flat extrapolation,
        # matching the value interpolant's own boundary behavior).
        cell_idx = np.clip(
            np.searchsorted(pressures, p, side="right") - 1,
            0,
            len(slope_interps) - 1,
        )
        result = np.empty(len(p), dtype=dtype)
        for idx in np.unique(cell_idx):
            mask = cell_idx == idx
            result[mask] = slope_interps[idx](np.column_stack([t[mask], s[mask]]))
        return result

    return interpolator


def clip_compressibility(
    values: NumberArray[NDimension],
    *,
    dtype: npt.DTypeLike,
    max_value: Number = 1e-1,
    context: str = "compressibility",
) -> NumberArray[NDimension]:
    """
    Clip a raw `-(1/B)·(dB/dP)` array to the physically valid range `[0, max_value]`.

    Negative values are expected on saturated-branch (Rs/Rv-bracketed) tables -
    `PVTO`/`PVTG` - where this formula implicitly assumes constant Rs/Rv, which
    doesn't hold on the saturated envelope. That's a known modeling artifact, not
    noise, so it's floored to 0 quietly (debug log only).

    Values above *max_value* are a different story: nothing physically
    reasonable should ever hit `1e-1` psi⁻¹ (typical oil/water compressibility
    is `~1e-6`-`1e-5`), so an excess almost always means a noisy or sparsely
    tabulated PVT table producing a PCHIP-derivative blow-up. Those are warned
    on loudly instead of silently absorbed.

    :param values: Raw compressibility array before clipping.
    :param dtype: Output dtype.
    :param max_value: Upper clip bound (1/psi or 1/bar, matching *values*' unit system).
    :param context: Label used in the warning/log message (e.g. `"PVTO oil compressibility"`).
    :returns: Clipped array, dtype *dtype*.
    """
    n_negative = int(np.count_nonzero(values < 0.0))
    if n_negative:
        logger.debug(
            "%s: %d value(s) were negative (min %.4g), clipped to 0. Expected on "
            "the saturated branch, where -(1/B)*(dB/dP) is not a true "
            "constant-composition compressibility.",
            context,
            n_negative,
            float(np.min(values)),
        )

    n_excess = int(np.count_nonzero(values > max_value))
    if n_excess:
        warnings.warn(
            f"{context}: {n_excess} value(s) exceeded the {max_value:g} ceiling "
            f"(max {float(np.max(values)):.4g}) and were clipped. This usually "
            "indicates a noisy or sparsely-tabulated PVT table rather than "
            "physical compressibility. Consider checking the source table.",
            UserWarning,
            stacklevel=3,
        )
    return np.clip(values, 0.0, max_value, dtype=dtype, out=values)


TWO_DIMENSIONAL_TABLES = (
    "viscosity_table",
    "density_table",
    "formation_volume_factor_table",
    "compressibility_table",
    "solution_gor_table",
    "compressibility_factor_table",
    "gas_free_water_fvf_table",
    "vaporized_oil_ratio_table",
)
COMMON_THREE_DIMENSIONAL_TABLES = (
    "solubility_in_water_table",
    "bubble_point_pressure_table",
)
SHARED_THREE_DIMENSIONAL_TABLES = (
    "solubility_in_water_table",
    "bubble_point_pressure_table",
)
THREE_DIMENSIONAL_TABLES = COMMON_THREE_DIMENSIONAL_TABLES + SHARED_THREE_DIMENSIONAL_TABLES


def build_derived_tables(data: PVTData, pvt: StaticPVT, dtype: npt.DTypeLike = None) -> PVTData:
    """
    Build `density_table` and `compressibility_table` if absent.

    Uses the standard reservoir-engineering formulas so that at
    simulation time every property is a single interpolator call.

    :param data: Source `PVTData`.
    :param pvt: Static PVT properties.
    :returns: `PVTData` with derived tables filled in (may be the same
        object if nothing was missing).
    """
    phase = typing.cast(FluidPhase, data.phase)
    pressures = data.pressures
    temperatures = data.temperatures
    n_p = len(pressures)
    n_t = len(temperatures)
    pressure_table, _ = np.meshgrid(pressures, temperatures, indexing="ij")

    updates: dict[str, typing.Any] = {}
    stock_tank_oil_density = pvt.stock_tank_oil_density
    stock_tank_gas_density = pvt.stock_tank_gas_density
    stock_tank_water_density = pvt.stock_tank_water_density

    # Oil Phase
    if phase == FluidPhase.OIL:
        oil_fvf_table = data.formation_volume_factor_table
        solution_gor_table = data.solution_gor_table

        # Density: ρo = (ρo,SC + Rs·ρg,SC) / Bo
        if (
            data.density_table is None
            and oil_fvf_table is not None
            and solution_gor_table is not None
            and stock_tank_oil_density is not None
            and stock_tank_gas_density is not None
        ):
            density_table = (
                stock_tank_oil_density + solution_gor_table * stock_tank_gas_density
            ) / oil_fvf_table
            updates["density_table"] = density_table.astype(dtype, copy=False)

        # Compressibility: co = -(1/Bo)·(∂Bo/∂P)
        if data.compressibility_table is None and oil_fvf_table is not None:
            # Build dBo/dP via PCHIP derivative at each temperature column
            dbo_dp = np.empty((n_p, n_t), dtype=dtype)
            for j in range(n_t):
                d = PchipInterpolator(pressures, oil_fvf_table[:, j]).derivative(1)
                dbo_dp[:, j] = d(pressures)

            oil_compressibility_table = -(1.0 / oil_fvf_table) * dbo_dp
            clip_compressibility(
                oil_compressibility_table,
                dtype=dtype,
                context="PVT table derived oil compressibility",
            )
            updates["compressibility_table"] = oil_compressibility_table.astype(dtype, copy=False)

    # Gas Phase
    elif phase == FluidPhase.GAS:
        gas_fvf_table = data.formation_volume_factor_table
        vaporized_oil_ratio_table = data.vaporized_oil_ratio_table
        compressibility_factor_table = data.compressibility_factor_table

        # Density: ρg = (ρg,SC + Rv·ρo,SC) / Bg  [wet] or ρg,SC / Bg [dry]
        if (
            data.density_table is None
            and gas_fvf_table is not None
            and stock_tank_oil_density is not None
            and stock_tank_gas_density is not None
        ):
            if vaporized_oil_ratio_table is not None:
                density = (
                    stock_tank_gas_density + vaporized_oil_ratio_table * stock_tank_oil_density
                ) / gas_fvf_table
            else:
                density = stock_tank_gas_density / gas_fvf_table
            updates["density_table"] = density.astype(dtype, copy=False)

        # Compressibility: cg = 1/P - (1/z)·(∂z/∂P)
        if data.compressibility_table is None:
            if compressibility_factor_table is not None:
                dz_dp = np.empty((n_p, n_t), dtype=dtype)
                for j in range(n_t):
                    d = PchipInterpolator(
                        pressures, compressibility_factor_table[:, j]
                    ).derivative(1)
                    dz_dp[:, j] = d(pressures)

                gas_compressibility_table = (
                    1.0 / pressure_table - (1.0 / compressibility_factor_table) * dz_dp
                )
                clip_compressibility(
                    gas_compressibility_table,
                    dtype=dtype,
                    context="PVT table derived gas compressibility",
                )
                updates["compressibility_table"] = gas_compressibility_table.astype(
                    dtype, copy=False
                )

            elif gas_fvf_table is not None:
                # Fallback: cg ≈ -(1/Bg)·(∂Bg/∂P)
                dbg_dp = np.empty((n_p, n_t), dtype=dtype)
                for j in range(n_t):
                    d = PchipInterpolator(pressures, gas_fvf_table[:, j]).derivative(1)
                    dbg_dp[:, j] = d(pressures)

                gas_compressibility_table = -(1.0 / gas_fvf_table) * dbg_dp
                clip_compressibility(
                    gas_compressibility_table,
                    dtype=dtype,
                    context="PVT table derived gas compressibility",
                )
                updates["compressibility_table"] = gas_compressibility_table.astype(
                    dtype, copy=False
                )

    # Water Phase
    elif phase == FluidPhase.WATER:
        gas_free_water_fvf_table = data.gas_free_water_fvf_table

        # Density: ρw = (ρw,SC + Rsw · ρg,SC) / Bw
        if (
            data.density_table is None
            and gas_free_water_fvf_table is not None
            and stock_tank_water_density is not None
            and stock_tank_gas_density is not None
        ):
            salinities = data.salinities
            if salinities is not None:
                n_s = len(salinities)
                density_3d_table = np.empty((n_p, n_t, n_s), dtype=dtype)
                for s_idx in range(n_s):
                    # Use gas solubility in water (Rsw) if available,
                    # otherwise fall back to gas-free approximation (Rsw = 0)
                    if (
                        data.solubility_in_water_table is not None
                        and data.solubility_in_water_table.shape[2] > s_idx
                    ):
                        rsw_slice = data.solubility_in_water_table[:, :, s_idx]
                        density_3d_table[:, :, s_idx] = (
                            stock_tank_water_density + rsw_slice * stock_tank_gas_density
                        ) / gas_free_water_fvf_table
                    else:
                        density_3d_table[:, :, s_idx] = (
                            stock_tank_water_density / gas_free_water_fvf_table
                        )
                updates["density_table"] = density_3d_table.astype(dtype, copy=False)

        # Compressibility: cw = -(1/Bw_gf)·(∂Bw_gf/∂P) [undersaturated]
        if data.compressibility_table is None and gas_free_water_fvf_table is not None:
            dbw_dp = np.empty((n_p, n_t), dtype=dtype)
            for j in range(n_t):
                d = PchipInterpolator(pressures, gas_free_water_fvf_table[:, j]).derivative(1)
                dbw_dp[:, j] = d(pressures)

            water_compressibility_2d_table = -(1.0 / gas_free_water_fvf_table) * dbw_dp
            clip_compressibility(
                water_compressibility_2d_table,
                dtype=dtype,
                context="PVT table derived water compressibility",
            )
            if data.salinities is not None:
                n_s = len(data.salinities)
                water_compressibility_3d_table = np.broadcast_to(
                    water_compressibility_2d_table[:, :, np.newaxis],
                    (n_p, n_t, n_s),
                ).copy()
                updates["compressibility_table"] = water_compressibility_3d_table.astype(
                    dtype, copy=False
                )

    if not updates:
        return data
    return attrs.evolve(data, **updates)


def validate_pvt_data(data: PVTData) -> None:
    """Validate grid monotonicity, shapes, and phase-specific field/table rules."""
    pressures = data.pressures
    temperatures = data.temperatures
    salinities = data.salinities
    n_p = len(pressures)
    n_t = len(temperatures)
    n_s = len(salinities) if salinities is not None else None

    if pressures.ndim != 1:
        raise ValidationError("`pressures` must be 1-dimensional.")
    if temperatures.ndim != 1:
        raise ValidationError("`temperatures` must be 1-dimensional.")
    if not np.all(np.diff(pressures) > 0):
        raise ValidationError("`pressures` must be strictly monotonically increasing.")
    if not np.all(np.diff(temperatures) > 0):
        raise ValidationError("`temperatures` must be strictly monotonically increasing.")
    if salinities is not None:
        if salinities.ndim != 1:
            raise ValidationError("`salinities` must be 1-dimensional.")
        if not np.all(np.diff(salinities) > 0):
            raise ValidationError("`salinities` must be strictly monotonically increasing.")

    bubble_point_array = data.bubble_point_pressures
    if bubble_point_array is not None:
        if bubble_point_array.ndim == 1:
            if len(bubble_point_array) != n_t:
                raise ValidationError(
                    f"`bubble_point_pressures` 1-D length {len(bubble_point_array)} must "
                    f"match n_temperatures={n_t}."
                )

        elif bubble_point_array.ndim == 2:
            solution_gors = data.solution_gas_to_oil_ratios
            if solution_gors is None:
                raise ValidationError(
                    "2-D `bubble_point_pressures` requires `solution_gas_to_oil_ratios`."
                )
            if not np.all(np.diff(solution_gors) > 0):
                raise ValidationError(
                    "`solution_gas_to_oil_ratios` must be strictly monotonically increasing."
                )
            if bubble_point_array.shape != (len(solution_gors), n_t):
                raise ValidationError(
                    f"`bubble_point_pressures` shape {bubble_point_array.shape} must be "
                    f"({len(solution_gors)}, {n_t})."
                )
        else:
            raise ValidationError("`bubble_point_pressures` must be 1-D or 2-D.")

    for table_name in TWO_DIMENSIONAL_TABLES:
        array = getattr(data, table_name, None)
        if array is not None and array.ndim == 2 and array.shape != (n_p, n_t):
            raise ValidationError(
                f"`{table_name}` shape {array.shape} must be "
                f"(n_pressures={n_p}, n_temperatures={n_t})."
            )

    phase = typing.cast(FluidPhase, data.phase)
    shared_3d_tables = SHARED_THREE_DIMENSIONAL_TABLES if phase == FluidPhase.WATER else ()
    for table_name in COMMON_THREE_DIMENSIONAL_TABLES + shared_3d_tables:
        array = getattr(data, table_name, None)
        if array is None:
            continue
        if array.ndim == 3:
            if n_s is None:
                raise ValidationError(f"`{table_name}` is 3-D but `salinities` was not provided.")
            if array.shape != (n_p, n_t, n_s):
                raise ValidationError(
                    f"`{table_name}` shape {array.shape} must be "
                    f"(n_p={n_p}, n_t={n_t}, n_s={n_s})."
                )

    if data.solubility_in_water_table is not None and salinities is None:
        raise ValidationError("`solubility_in_water_table` is 3-D and requires `salinities`.")


def check_physical_consistency(data: PVTData) -> None:
    """Phase-aware physical sanity checks on table values."""
    phase = typing.cast(FluidPhase, data.phase)
    if data.viscosity_table is not None and np.any(data.viscosity_table <= 0):
        raise ValidationError(f"{phase.value.upper()} viscosity must be positive everywhere.")
    if data.density_table is not None and np.any(data.density_table <= 0):
        raise ValidationError(f"{phase.value.upper()} density must be positive everywhere.")
    if data.formation_volume_factor_table is not None and np.any(
        data.formation_volume_factor_table <= 0
    ):
        raise ValidationError(f"{phase.value.upper()} FVF must be positive everywhere.")
    if (
        phase == FluidPhase.GAS
        and data.compressibility_factor_table is not None
        and np.any(data.compressibility_factor_table <= 0)
    ):
        raise ValidationError("Gas z-factor must be positive everywhere.")

    max_gas_density_field_unit = 50  # 50 lbm/ft³
    factors = get_conversion_factors(UnitSystem.FIELD, data.unit_system)
    max_gas_density = max_gas_density_field_unit * factors["density"]
    if (
        phase == FluidPhase.GAS
        and data.density_table is not None
        and np.any(data.density_table >= max_gas_density)
    ):
        warnings.warn(
            f"Gas density table contains values >= {max_gas_density} density units (e.g, lbm/ft³ for FIELD). "
            "Verify that the table's values are in right unit system.",
            UserWarning,
            stacklevel=3,
        )


class PVTTable(StoreSerializable):
    """
    Phase-aware PVT property lookup with pre-built interpolators.

    Wraps a `PVTData` instance and builds SciPy / PCHIP interpolators for
    every table column, including derived tables (density, compressibility) if
    they were not supplied in the data.

    **Construction-time derivation**

    When `PVT` reference densities are supplied and the data
    does not already contain a `density_table` or `compressibility_table`,
    this constructor builds those tables now from the primary columns
    (Bo / Rs / Bg / Rv / Bw) so that all subsequent simulation lookups are a
    single interpolator call.

    **Derivative methods**

    Every primary property `prop(p, t)` has a matching `prop_dp(p, t)`
    that returns `∂prop/∂P` using the analytical derivative of the same PCHIP
    or the partial-derivative of the `RectBivariateSpline`. No finite
    differences are used.

    **Interpolation modes**

    - `"linear"`: `RectBivariateSpline` (kx=ky=1) for 2-D; `interp1d` for 1-D.
    - `"cubic"`: two-stage PCHIP-along-axes for 2-D; monotone cubic for 1-D.

    The cubic path preserves monotonicity independently along each axis,
    avoiding oscillation on noisy PVT data.

    **Unit-system agnostic**

    All dimensional quantities (pressure, temperature, viscosity, density, FVF, compressibility, etc.)
    follow the unit system specified in the input `PVTData`. Query results are in
    the same unit system as the underlying data and the optional `StaticPVT`. Units do not affect
    the interpolation algorithms, which are purely numerical.

    **Water phase**

    When only one salinity value is present the 3-D tables are pre-sliced to
    2-D at construction time and the fast 2-D interpolators are used.

    **Extrapolation**

    Outside table bounds the interpolators extrapolate monotonically. Set
    `warn_on_extrapolation=True` to log warnings.
    """

    __abstract_serializable__ = True

    def __init__(
        self,
        data: PVTData | PathLike[str] | str,
        interpolation_method: InterpolationMethod = "linear",
        validate: bool = True,
        warn_on_extrapolation: bool = False,
        pvt: StaticPVT | None = None,
        dtype: npt.DTypeLike = None,
    ) -> None:
        """
        Build a `PVTTable` from raw tabulated data.

        :param data: Phase-tagged `PVTData` or path to a serialised file.
        :param interpolation_method: `"linear"` (default) or `"cubic"`.
        :param validate: Run physical-consistency checks on `data`.
        :param warn_on_extrapolation: Log a warning when queries fall outside
            the table pressure / temperature / salinity bounds.
        :param pvt: Optional `StaticPVT` carrying stock-tank reference densities
            (`stock_tank_oil_density`, `stock_tank_gas_density`, `stock_tank_water_density`).
            When provided, missing `density_table` and `compressibility_table` entries are
            built automatically. Units must match the `data.unit_system`.
        """
        if interpolation_method not in INTERPOLATION_DEGREES:
            raise ValidationError(
                f"Invalid interpolation_method {interpolation_method!r}. "
                f"Must be one of: {list(INTERPOLATION_DEGREES.keys())}"
            )
        if not isinstance(data, PVTData):
            pvt_data = PVTData.read(data)  # type: ignore[assignment]
            if pvt_data is None:
                raise ValidationError(f"Invalid PVT data file {data!s}")
            data = pvt_data

        self._phase = data.phase
        self.interpolation_method: InterpolationMethod = interpolation_method
        self.validate = validate
        self.warn_on_extrapolation = warn_on_extrapolation

        if interpolation_method != "linear":
            if len(data.pressures) < 4:
                raise ValidationError(
                    f"At least 4 pressure points required for {interpolation_method!r} "
                    f"interpolation, got {len(data.pressures)}."
                )
            if len(data.temperatures) < 4:
                raise ValidationError(
                    f"At least 4 temperature points required for "
                    f"{interpolation_method!r} interpolation, "
                    f"got {len(data.temperatures)}."
                )

        validate_pvt_data(data)
        if validate:
            check_physical_consistency(data)

        self._extrapolation_bounds: dict[str, tuple[Number, Number]] = {
            "pressure": (data.pressures[0], data.pressures[-1]),
            "temperature": (data.temperatures[0], data.temperatures[-1]),
        }
        if data.salinities is not None:
            self._extrapolation_bounds["salinity"] = (
                data.salinities[0],
                data.salinities[-1],
            )

        self._bubble_point_ndim: int | None = None
        if data.phase == FluidPhase.OIL and data.bubble_point_pressures is not None:
            self._bubble_point_ndim = data.bubble_point_pressures.ndim

        self.default_salinity: Number | None = (
            data.salinities[0] if data.salinities is not None else None
        )
        self._water_constant_salinity: bool = (
            data.phase == FluidPhase.WATER
            and data.salinities is not None
            and len(data.salinities) == 1
        )

        if dtype is None:
            if data.dtype is not None:
                self.dtype = data.dtype
            else:
                self.dtype = get_dtype()
                data.ensure_dtype(self.dtype, force=True)
        else:
            self.dtype = np.dtype(dtype)
            data.ensure_dtype(self.dtype, force=False)

        # Potentially augment data with derived tables before building interpolators
        self._stock_tank_oil_density: Number | None = None
        self._stock_tank_gas_density: Number | None = None
        if pvt is not None:
            if pvt.unit_system != data.unit_system:
                pvt = pvt.convert(self.unit_system)
            self._stock_tank_oil_density = pvt.stock_tank_oil_density
            self._stock_tank_gas_density = pvt.stock_tank_gas_density
            data = build_derived_tables(data, pvt, dtype)

        self._data = data
        self._interpolatants: dict[str, typing.Any] = {}
        self._derivative_interpolatants: dict[str, typing.Any] = {}
        self.build_interpolants(data)

        logger.debug(
            "%s init: phase=%s, n_p=%d, n_t=%d, method=%r, n_interp=%d, n_deriv=%d",
            self.__class__.__name__,
            data.phase.value.upper(),  # type: ignore
            len(data.pressures),
            len(data.temperatures),
            interpolation_method,
            len(self._interpolatants),
            len(self._derivative_interpolatants),
        )

    def build_interpolants(self, data: PVTData) -> None:
        """Build scipy / PCHIP interpolators and their pressure-derivatives."""
        phase = typing.cast(FluidPhase, data.phase)
        pressures = data.pressures
        temperatures = data.temperatures
        salinities = data.salinities
        k = INTERPOLATION_DEGREES[self.interpolation_method]
        use_pchip = self.interpolation_method == "cubic"

        def register_2d(name: str, table: npt.NDArray | None) -> None:
            """Register a 2-D property interpolator and its ∂/∂P derivative."""
            if table is None:
                return

            if use_pchip:
                self._interpolatants[name] = build_pchip_2d_interpolator(
                    pressures, temperatures, table, dtype=self.dtype
                )
                self._derivative_interpolatants[name] = build_pchip_2d_derivative_interpolator(
                    pressures, temperatures, table, dtype=self.dtype
                )
            else:
                spline = RectBivariateSpline(x=pressures, y=temperatures, z=table, kx=k, ky=k)
                self._interpolatants[name] = spline
                # RectBivariateSpline.partial_derivative(dx, dy) requires
                # dx < kx; for "linear" (kx=1) that's never satisfiable, so
                # build the exact bilinear derivative directly instead.
                self._derivative_interpolatants[name] = build_bilinear_2d_derivative_interpolator(
                    pressures, temperatures, table, dtype=self.dtype
                )

        def register_3d(name: str, table: npt.NDArray | None) -> None:
            """
            Register a 3-D property interpolator.

            For single-salinity tables the 3-D array is pre-sliced to 2-D and
            a fast 2-D interpolator is registered instead.
            """
            if table is None or salinities is None:
                return

            if self._water_constant_salinity:
                register_2d(name, table[:, :, 0])
            else:
                self._interpolatants[name] = RegularGridInterpolator(
                    points=(pressures, temperatures, salinities),
                    values=table,
                    method=self.interpolation_method,
                    bounds_error=False,
                    fill_value=None,
                )
                if use_pchip:
                    self._derivative_interpolatants[name] = build_pchip_3d_derivative_interpolator(
                        pressures, temperatures, salinities, table, dtype=self.dtype
                    )
                else:
                    self._derivative_interpolatants[name] = (
                        build_bilinear_3d_derivative_interpolator(
                            pressures, temperatures, salinities, table, dtype=self.dtype
                        )
                    )

        # Shared properties
        if phase == FluidPhase.WATER:
            register_3d("viscosity", data.viscosity_table)
            register_3d("density", data.density_table)
            register_3d("formation_volume_factor", data.formation_volume_factor_table)
            register_3d("compressibility", data.compressibility_table)
        else:
            register_2d("viscosity", data.viscosity_table)
            register_2d("density", data.density_table)
            register_2d("formation_volume_factor", data.formation_volume_factor_table)
            register_2d("compressibility", data.compressibility_table)

        # Phase-specific
        if phase == FluidPhase.OIL:
            register_2d("solution_gor", data.solution_gor_table)
            register_2d("vaporized_oil_to_gas_ratio", None)  # not applicable for oil

            bubble_point_array = data.bubble_point_pressures
            if bubble_point_array is not None:
                if bubble_point_array.ndim == 1:
                    self._interpolatants["bubble_point_pressure"] = interp1d(
                        x=temperatures,
                        y=bubble_point_array,
                        kind=self.interpolation_method,
                        bounds_error=False,
                        fill_value="extrapolate",  # type: ignore[arg-type]
                    )
                else:
                    solution_gors = data.solution_gas_to_oil_ratios
                    assert solution_gors is not None
                    if use_pchip:
                        self._interpolatants["bubble_point_pressure"] = (
                            build_pchip_2d_interpolator(
                                solution_gors,
                                temperatures,
                                bubble_point_array,  # type: ignore
                                dtype=self.dtype,
                            )
                        )
                        self._derivative_interpolatants["dpb_drs"] = (
                            build_pchip_2d_derivative_interpolator(
                                solution_gors,
                                temperatures,
                                bubble_point_array,  # type: ignore
                                dtype=self.dtype,
                            )
                        )
                    else:
                        spline = RectBivariateSpline(
                            x=solution_gors,
                            y=temperatures,
                            z=bubble_point_array,
                            kx=k,
                            ky=k,
                        )
                        self._interpolatants["bubble_point_pressure"] = spline
                        self._derivative_interpolatants["dpb_drs"] = (
                            build_bilinear_2d_derivative_interpolator(
                                solution_gors,
                                temperatures,
                                bubble_point_array,
                                dtype=self.dtype,
                            )
                        )

        if phase == FluidPhase.GAS:
            register_2d("compressibility_factor", data.compressibility_factor_table)
            register_2d("vaporized_oil_to_gas_ratio", data.vaporized_oil_ratio_table)
            register_3d("solubility_in_water", data.solubility_in_water_table)

            dp = data.dew_point_pressures
            if dp is not None:
                self._interpolatants["dew_point_pressure"] = interp1d(
                    x=temperatures,
                    y=dp,
                    kind=self.interpolation_method,
                    bounds_error=False,
                    fill_value="extrapolate",  # type: ignore[arg-type]
                )

        if phase == FluidPhase.WATER:
            register_3d("bubble_point_pressure", data.bubble_point_pressure_table)
            register_2d("gas_free_fvf", data.gas_free_water_fvf_table)

    def __dump__(self) -> dict[str, typing.Any]:
        return {
            "data": self._data.dump(),
            "interpolation_method": self.interpolation_method,
            "validate": self.validate,
            "warn_on_extrapolation": self.warn_on_extrapolation,
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> "PVTTable":
        pvt_data = PVTData.load(data["data"])
        return cls(
            data=pvt_data,
            interpolation_method=data.get("interpolation_method", "linear"),
            validate=data.get("validate", True),
            warn_on_extrapolation=data.get("warn_on_extrapolation", False),
        )

    @property
    def phase(self) -> FluidPhase:
        """Fluid phase this table describes."""
        return typing.cast(FluidPhase, self._phase)

    @property
    def unit_system(self) -> UnitSystem:
        """Unit system of the underlying table data."""
        return typing.cast(UnitSystem, self._data.unit_system)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `PVTTable` with all dimensional quantities rescaled to *target*.

        Dimensionless properties (specific gravity, compressibility factor,
        vaporized oil ratio, solution GOR ratio) and multiplier-type quantities
        are copied unchanged. Pressure axes, densities, FVFs, and viscosities
        are rescaled using `get_conversion_factors`.

        :param target: Target `UnitSystem`.
        :returns `PVTTable`: New `PVTTable` in *target* units.
        """
        if target == self.unit_system:
            return self

        return self.__class__(
            data=self._data.convert(target, table=table),
            interpolation_method=self.interpolation_method,
            validate=False,  # already validated at construction
            warn_on_extrapolation=self.warn_on_extrapolation,
            dtype=self.dtype,
        )

    def has(self, name: str) -> bool:
        """Return `True` if an interpolator for *name* was built."""
        return name in self._interpolatants

    def __contains__(self, name: str, /) -> bool:
        return self.has(name)

    def _warn_extrapolation(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: TableQuery[NDimension] | None = None,
    ) -> None:
        if not self.warn_on_extrapolation:
            return

        pressure_array = np.atleast_1d(pressure)
        temperature_array = np.atleast_1d(temperature)
        min_pressure, max_pressure = self._extrapolation_bounds["pressure"]
        min_temperature, max_temperature = self._extrapolation_bounds["temperature"]
        if np.any(pressure_array < min_pressure) or np.any(pressure_array > max_pressure):
            logger.warning(
                "Pressure extrapolation: queried P ∈ [%.1f, %.1f] psi, "
                "table range [%.1f, %.1f] psi",
                float(pressure_array.min()),
                float(pressure_array.max()),
                min_pressure,
                max_pressure,
            )

        if np.any(temperature_array < min_temperature) or np.any(
            temperature_array > max_temperature
        ):
            logger.warning(
                "Temperature extrapolation: queried T ∈ [%.1f, %.1f] °F, "
                "table range [%.1f, %.1f] °F",
                float(temperature_array.min()),
                float(temperature_array.max()),
                min_temperature,
                max_temperature,
            )

        if salinity is not None and "salinity" in self._extrapolation_bounds:
            salinity_array = np.atleast_1d(salinity)
            min_salinity, max_salinity = self._extrapolation_bounds["salinity"]
            if np.any(salinity_array < min_salinity) or np.any(salinity_array > max_salinity):
                logger.warning(
                    "Salinity extrapolation: queried S ∈ [%.0f, %.0f] ppm, "
                    "table range [%.0f, %.0f] ppm",
                    float(salinity_array.min()),
                    float(salinity_array.max()),
                    min_salinity,
                    max_salinity,
                )

    def query(
        self,
        name: str,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        *,
        derivative: bool = False,
    ) -> TableResult[NDimension] | None:
        """
        Evaluate a 2-D interpolator at `(pressure, temperature)` pairs.

        :param name: Key in `_interpolatants` / `_derivative_interpolatants`.
        :param pressure: Pressure.
        :param temperature: Temperature.
        :param derivative: If `True`, return `∂/∂P` instead of the value.
        :returns: Interpolated value (or derivative), or `None` when the
            table was not built.
        """
        store = self._derivative_interpolatants if derivative else self._interpolatants
        interp = store.get(name)
        if interp is None:
            return None
        if not derivative:
            self._warn_extrapolation(pressure, temperature)

        is_scalar = np.isscalar(pressure) and np.isscalar(temperature)
        dtype = self.dtype
        if is_scalar:
            raw = (
                interp.ev(pressure, temperature)
                if hasattr(interp, "ev")
                else interp(np.atleast_1d(pressure), np.atleast_1d(temperature))[0]
            )
            result = dtype.type(raw)  # type: ignore[attr-defined]
        else:
            pressure_array = np.atleast_1d(pressure)
            temperature_array = np.atleast_1d(temperature)
            if pressure_array.shape != temperature_array.shape:
                if pressure_array.size == 1:
                    pressure_array = np.full_like(
                        temperature_array, pressure_array[0], dtype=dtype
                    )
                elif temperature_array.size == 1:
                    temperature_array = np.full_like(
                        pressure_array, temperature_array[0], dtype=dtype
                    )
                else:
                    raise ValidationError(
                        f"Incompatible shapes: pressure {pressure_array.shape}, temperature {temperature_array.shape}"
                    )
            result = (
                interp.ev(pressure_array, temperature_array)
                if hasattr(interp, "ev")
                else interp(pressure_array, temperature_array)
            )

        if (is_array := isinstance(result, np.ndarray)) and result.ndim == 0:
            return typing.cast(Number, dtype.type(result))  # type: ignore[attr-defined]
        elif is_array and result.size == 1:
            return typing.cast(Number, dtype.type(result.item()))  # type: ignore[attr-defined]
        return typing.cast(NumberArray[NDimension], result.astype(dtype, copy=False))

    def squery(
        self,
        name: str,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: TableQuery[NDimension],
        *,
        derivative: bool = False,
    ) -> TableResult[NDimension] | None:
        """
        Evaluate a 3-D interpolator at `(pressure, temperature, salinity)` points.

        When `_water_constant_salinity` is `True` the registered
        interpolator is actually 2-D (the salinity slice was taken at
        construction) and `query` is called directly.

        :param name: Interpolator key.
        :param pressure: Pressure.
        :param temperature: Temperature.
        :param salinity: Salinity (ppm NaCl).
        :param derivative: Return `∂/∂P` when `True`.
        :returns: Interpolated value or `None`.
        """
        if self._water_constant_salinity:
            return self.query(name, pressure, temperature, derivative=derivative)

        store = self._derivative_interpolatants if derivative else self._interpolatants
        interp = store.get(name)
        if interp is None:
            return None
        if not derivative:
            self._warn_extrapolation(pressure, temperature, salinity)

        dtype = self.dtype
        pressure_array = np.atleast_1d(pressure)
        temperature_array = np.atleast_1d(temperature)
        salinity_array = np.atleast_1d(salinity)
        pressure_array, temperature_array, salinity_array = np.broadcast_arrays(
            pressure_array, temperature_array, salinity_array
        )
        points = np.column_stack([
            pressure_array.ravel(),
            temperature_array.ravel(),
            salinity_array.ravel(),
        ])
        result = interp(points).reshape(pressure_array.shape).astype(dtype, copy=False)

        if result.ndim == 0:
            return typing.cast(Number, dtype.type(result))  # type: ignore[attr-defined]
        if result.size == 1:
            return typing.cast(Number, dtype.type(result.item()))  # type: ignore[attr-defined]
        return typing.cast(NumberArray[NDimension], result.astype(dtype, copy=False))

    def resolve_salinity(self, salinity: TableQuery[NDimension] | None) -> TableQuery[NDimension]:
        if salinity is not None:
            return salinity
        if self.default_salinity is not None:
            return self.default_salinity
        raise ValidationError(
            "Salinity is required for water-phase PVT lookup but none was provided "
            "and no default salinity is set."
        )

    def formation_volume_factor(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: TableQuery[NDimension] | None = None,
        solution_gor: TableQuery[NDimension] | None = None,
        bubble_point_pressure: TableQuery[NDimension] | None = None,
    ) -> TableResult[NDimension] | None:
        """
        Get formation volume factor `B`.

        Units depend on `unit_system` and phase:
        - Oil/water: bbl/STB (FIELD), m³/Sm³ (METRIC/SI), cc/scc (LAB)
        - Gas: ft³/SCF (FIELD), m³/Sm³ (METRIC/SI), cc/scc (LAB)

        **Oil phase** - saturated / undersaturated switching:

        - Saturated (P ≤ bubble_point_array): interpolated directly from the Bo table.
        - Undersaturated (P > bubble_point_array): applies the McCain exponential correction
          `Bo = Bob · exp(-co · (P - bubble_point_array))`, where `compressibility` comes from the
          compressibility table. Falls back to `Bob` if unavailable.

        **Gas and water** - direct table interpolation; no saturation switching.

        :param pressure: Pressure.
        :param temperature: Temperature.
        :param salinity: Salinity (ppm NaCl). Water phase only.
        :param solution_gor: Solution GOR. Oil, for 2-D `bubble_point_array` table only.
        :param bubble_point_pressure: Pre-computed `bubble_point_array`. Oil only; skips
            internal `bubble_point_array` lookup when supplied.
        :returns: FVF or `None` if the table is not present.
        """
        if self._phase == FluidPhase.WATER:
            return self.squery(
                "formation_volume_factor",
                pressure,
                temperature,
                self.resolve_salinity(salinity),
            )
        if self._phase == FluidPhase.GAS:
            return self.query("formation_volume_factor", pressure, temperature)

        if not self.has("formation_volume_factor"):
            return None

        bubble_point_array = (
            bubble_point_pressure
            if bubble_point_pressure is not None
            else self.bubble_point_pressure(temperature=temperature, solution_gor=solution_gor)
        )
        if bubble_point_array is None:
            return self.query("formation_volume_factor", pressure, temperature)

        dtype = self.dtype
        pressure_array = np.atleast_1d(pressure)
        temperature_array = np.atleast_1d(temperature)
        bubble_point_array = np.atleast_1d(bubble_point_array)
        pressure_array, temperature_array, bubble_point_array = np.broadcast_arrays(
            pressure_array, temperature_array, bubble_point_array
        )

        result = np.zeros_like(pressure_array, dtype=dtype)
        saturated = pressure_array <= bubble_point_array
        undersaturated = ~saturated

        if np.any(saturated):
            result[saturated] = self.query(  # type: ignore[index]
                "formation_volume_factor",
                pressure_array[saturated],
                temperature_array[saturated],
            )
        if np.any(undersaturated):
            fvf_at_bubble_point = self.query(
                "formation_volume_factor",
                bubble_point_array[undersaturated],
                temperature_array[undersaturated],
            )
            if "compressibility" in self._interpolatants:
                compressibility_at_bubble_point = self.query(
                    "compressibility",
                    bubble_point_array[undersaturated],
                    temperature_array[undersaturated],
                )
                compressibility_at_pressure = self.query(
                    "compressibility",
                    pressure_array[undersaturated],
                    temperature_array[undersaturated],
                )
                avg_compressibility_arr = 0.5 * (
                    np.asarray(compressibility_at_bubble_point)
                    + np.asarray(compressibility_at_pressure)
                )
                result[undersaturated] = np.asarray(fvf_at_bubble_point) * np.exp(
                    -avg_compressibility_arr
                    * (pressure_array[undersaturated] - bubble_point_array[undersaturated])
                )
            else:
                result[undersaturated] = np.asarray(fvf_at_bubble_point)

        return typing.cast(
            TableResult[NDimension],
            dtype.type(result.item())  # type: ignore[attr-defined]
            if result.size == 1
            else result.astype(dtype, copy=False),
        )

    fvf = B = b = formation_volume_factor

    def db_dp(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: TableQuery[NDimension] | None = None,
        solution_gor: TableQuery[NDimension] | None = None,
        bubble_point_pressure: TableQuery[NDimension] | None = None,
    ) -> TableResult[NDimension] | None:
        """
        Return `∂B/∂P`. Units depend on `unit_system`.

        **For oil/water**: (psi⁻¹ · bbl/STB) in FIELD, (bar⁻¹ · m³/Sm³) in METRIC, (Pa⁻¹ · m³/Sm³) in SI,
        (atm⁻¹ · cc/scc) in LAB

        **For gas**: (psi⁻¹ · ft³/SCF) in FIELD, (bar⁻¹ · m³/Sm³) in METRIC, (Pa⁻¹ · m³/Sm³) in SI,
        (atm⁻¹ · cc/scc) in LAB

        **Oil phase** - saturated/undersaturated switching, matching
        `formation_volume_factor`'s own regime split exactly (previously this
        always returned the raw saturated-curve derivative regardless of
        regime, which is wrong above the bubble point):

        - Saturated (`P <= bubble_point_array`): the raw table interpolant's
          own derivative - correct, since interpolation is valid in this
          regime.
        - Undersaturated (`P > bubble_point_array`): analytically
          differentiated from the *exact* McCain correction formula
          `formation_volume_factor` itself evaluates
          (`Bo = Bob * exp(-cbar(P) * (P - Pb))`,
          `cbar(P) = 0.5*(c(Pb) + c(P))`), including `cbar`'s own
          P-dependence via the compressibility table's own derivative - not
          a finite-difference or other approximation. Validated against
          finite-difference differentiation of the value formula directly
          (~1e-9 relative error on a representative case) before writing
          this.

        **Gas and water** - unchanged, direct table derivative; no switching.

        :param pressure: Pressure.
        :param temperature: Temperature.
        :param salinity: Salinity (ppm NaCl). Water phase only.
        :param solution_gor: Solution GOR. Oil, for 2-D `bubble_point_array`
            table only.
        :param bubble_point_pressure: Pre-computed `bubble_point_array`. Oil only.
        :returns: `∂B/∂P` or `None` if table is absent.
        """
        if self._phase == FluidPhase.WATER:
            return self.squery(
                "formation_volume_factor",
                pressure,
                temperature,
                self.resolve_salinity(salinity),
                derivative=True,
            )
        if self._phase == FluidPhase.GAS:
            return self.query("formation_volume_factor", pressure, temperature, derivative=True)

        # Mirror `formation_volume_factor`'s own saturated/undersaturated
        # split rather than using the raw (saturated-only) derivative.
        if not self.has("formation_volume_factor"):
            return None

        bubble_point_array = (
            bubble_point_pressure
            if bubble_point_pressure is not None
            else self.bubble_point_pressure(temperature=temperature, solution_gor=solution_gor)
        )
        if bubble_point_array is None:
            return self.query("formation_volume_factor", pressure, temperature, derivative=True)

        dtype = self.dtype
        pressure_array = np.atleast_1d(pressure)
        temperature_array = np.atleast_1d(temperature)
        bubble_point_array = np.atleast_1d(bubble_point_array)
        pressure_array, temperature_array, bubble_point_array = np.broadcast_arrays(
            pressure_array, temperature_array, bubble_point_array
        )

        result = np.zeros_like(pressure_array, dtype=dtype)
        saturated = pressure_array <= bubble_point_array
        undersaturated = ~saturated

        if np.any(saturated):
            result[saturated] = self.query(  # type: ignore[index]
                "formation_volume_factor",
                pressure_array[saturated],
                temperature_array[saturated],
                derivative=True,
            )

        if np.any(undersaturated) and "compressibility" in self._interpolatants:
            p_undersaturated = pressure_array[undersaturated]
            t_undersaturated = temperature_array[undersaturated]
            pb_undersaturated = bubble_point_array[undersaturated]

            bo_at_pressure = np.asarray(
                self.formation_volume_factor(
                    p_undersaturated, t_undersaturated, bubble_point_pressure=pb_undersaturated
                )
            )
            c_at_bubble_point = np.asarray(
                self.query("compressibility", pb_undersaturated, t_undersaturated)
            )
            c_at_pressure = np.asarray(
                self.query("compressibility", p_undersaturated, t_undersaturated)
            )
            dc_dp_at_pressure = self.query(
                "compressibility", p_undersaturated, t_undersaturated, derivative=True
            )
            dc_dp_at_pressure = (
                np.zeros_like(p_undersaturated)
                if dc_dp_at_pressure is None
                else np.asarray(dc_dp_at_pressure)
            )
            average_compressibility = 0.5 * (c_at_bubble_point + c_at_pressure)
            result[undersaturated] = bo_at_pressure * (
                -0.5 * dc_dp_at_pressure * (p_undersaturated - pb_undersaturated)
                - average_compressibility
            )
        elif np.any(undersaturated):
            # No compressibility table so `formation_volume_factor` falls back
            # to a flat Bob in this regime, so its derivative is zero here.
            result[undersaturated] = 0.0

        return typing.cast(
            TableResult[NDimension],
            dtype.type(result.item())  # type: ignore[attr-defined]
            if result.size == 1
            else result.astype(dtype, copy=False),
        )

    def viscosity(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: TableQuery[NDimension] | None = None,
        solution_gor: TableQuery[NDimension] | None = None,
        bubble_point_pressure: TableQuery[NDimension] | None = None,
    ) -> TableResult[NDimension] | None:
        """
        Get fluid viscosity `μ`. Units depend on `unit_system` (cP in FIELD/METRIC/LAB, Pa·s in SI).

        **Oil phase** - saturated / undersaturated switching using the
        Beggs-Robinson undersaturated correction above bubble point.

        **Water / gas** - direct table interpolation.

        :param pressure: Pressure.
        :param temperature: Temperature.
        :param salinity: Salinity (ppm NaCl). Water phase only.
        :param solution_gor: Solution GOR. Oil, 2-D bubble_point_array table only.
        :param bubble_point_pressure: Pre-computed bubble_point_array. Oil only.
        :returns: Viscosity in cP, or `None` if table is absent.
        """
        if self._phase == FluidPhase.WATER:
            return self.squery(
                "viscosity",
                pressure,
                temperature,
                self.resolve_salinity(salinity),
            )
        if self._phase == FluidPhase.GAS:
            return self.query("viscosity", pressure, temperature)

        if not self.has("viscosity"):
            return None

        bubble_point_array = (
            bubble_point_pressure
            if bubble_point_pressure is not None
            else self.bubble_point_pressure(temperature=temperature, solution_gor=solution_gor)
        )
        if bubble_point_array is None:
            return self.query("viscosity", pressure, temperature)

        dtype = self.dtype
        pressure_array = np.atleast_1d(pressure)
        temperature_array = np.atleast_1d(temperature)
        bubble_point_array = np.atleast_1d(bubble_point_array)
        pressure_array, temperature_array, bubble_point_array = np.broadcast_arrays(
            pressure_array, temperature_array, bubble_point_array
        )

        result = np.zeros_like(pressure_array, dtype=dtype)
        saturated = pressure_array <= bubble_point_array
        undersaturated = ~saturated

        if np.any(saturated):
            result[saturated] = self.query(
                "viscosity", pressure_array[saturated], temperature_array[saturated]
            )  # type: ignore[index]

        if np.any(undersaturated):
            mu_ob = np.asarray(
                self.query(
                    "viscosity",
                    bubble_point_array[undersaturated],
                    temperature_array[undersaturated],
                )
            )
            p_undersaturated = pressure_array[undersaturated]
            pb_undersaturated = bubble_point_array[undersaturated]
            X = 2.6 * (p_undersaturated**1.187) * np.exp(-11.513 - 8.98e-5 * p_undersaturated)
            X = np.clip(X, 0.0, 5.0, dtype=dtype)
            ratio = np.clip(p_undersaturated / pb_undersaturated, 1.0, None, dtype=dtype)
            result[undersaturated] = np.clip(mu_ob * (ratio**X), mu_ob, mu_ob * 100.0, dtype=dtype)

        return typing.cast(
            TableResult[NDimension],
            dtype.type(result.item())  # type: ignore[attr-defined]
            if result.size == 1
            else result.astype(dtype, copy=False),
        )

    μ = mu = viscosity

    def dμ_dp(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: TableQuery[NDimension] | None = None,
        solution_gor: TableQuery[NDimension] | None = None,
        bubble_point_pressure: TableQuery[NDimension] | None = None,
    ) -> TableResult[NDimension] | None:
        """
        Return `∂μ/∂P` (viscosity-unit / pressure-unit, unit_system-dependent - cP/psi in FIELD).

        **Oil phase** - saturated/undersaturated switching, matching
        `viscosity`'s own regime split (previously this always returned the
        raw saturated-curve derivative regardless of regime, which is wrong
        above the bubble point):

        - Saturated: the raw table interpolant's own derivative.
        - Undersaturated: central finite-difference of `viscosity` itself,
          not a hand-differentiated closed form. The Beggs-Robinson
          correction `viscosity` evaluates there has two separately clipped
          terms (the exponent and the final ratio), and differentiating
          that piecewise by hand risks getting a kink wrong silently;
          central-differencing the already-correct value function sidesteps
          that entirely and stays exactly consistent with what `viscosity`
          actually returns, at the cost of ordinary finite-difference
          truncation error rather than an exact analytical derivative.

        **Gas and water** - unchanged, direct table derivative; no switching.

        :param pressure: Pressure.
        :param temperature: Temperature.
        :param salinity: Salinity (ppm NaCl). Water phase only.
        :param solution_gor: Solution GOR. Oil, for 2-D `bubble_point_array`
            table only.
        :param bubble_point_pressure: Pre-computed `bubble_point_array`. Oil only.
        :returns: `∂μ/∂P` or `None`.
        """
        if self._phase == FluidPhase.WATER:
            return self.squery(
                "viscosity",
                pressure,
                temperature,
                self.resolve_salinity(salinity),
                derivative=True,
            )
        if self._phase == FluidPhase.GAS:
            return self.query("viscosity", pressure, temperature, derivative=True)

        # Mirror viscosity's own saturated/undersaturated split rather
        # than using the raw (saturated-only) derivative.
        if not self.has("viscosity"):
            return None

        bubble_point_array = (
            bubble_point_pressure
            if bubble_point_pressure is not None
            else self.bubble_point_pressure(temperature=temperature, solution_gor=solution_gor)
        )
        if bubble_point_array is None:
            return self.query("viscosity", pressure, temperature, derivative=True)

        dtype = self.dtype
        pressure_array = np.atleast_1d(pressure)
        temperature_array = np.atleast_1d(temperature)
        bubble_point_array = np.atleast_1d(bubble_point_array)
        pressure_array, temperature_array, bubble_point_array = np.broadcast_arrays(
            pressure_array, temperature_array, bubble_point_array
        )

        result = np.zeros_like(pressure_array, dtype=dtype)
        saturated = pressure_array <= bubble_point_array
        undersaturated = ~saturated

        if np.any(saturated):
            result[saturated] = self.query(  # type: ignore[index]
                "viscosity",
                pressure_array[saturated],
                temperature_array[saturated],
                derivative=True,
            )

        if np.any(undersaturated):
            p_undersaturated = pressure_array[undersaturated]
            t_undersaturated = temperature_array[undersaturated]
            pb_undersaturated = bubble_point_array[undersaturated]
            step = np.maximum(np.abs(p_undersaturated) * 1e-6, 1e-6).astype(dtype)
            mu_plus = np.asarray(
                self.viscosity(
                    p_undersaturated + step,
                    t_undersaturated,
                    bubble_point_pressure=pb_undersaturated,
                )
            )
            mu_minus = np.asarray(
                self.viscosity(
                    p_undersaturated - step,
                    t_undersaturated,
                    bubble_point_pressure=pb_undersaturated,
                )
            )
            result[undersaturated] = (mu_plus - mu_minus) / (2 * step)

        return typing.cast(
            TableResult[NDimension],
            dtype.type(result.item())  # type: ignore[attr-defined]
            if result.size == 1
            else result.astype(dtype, copy=False),
        )

    dmu_dp = dμ_dp

    def density(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: TableQuery[NDimension] | None = None,
        solution_gor: TableQuery[NDimension] | None = None,
        bubble_point_pressure: TableQuery[NDimension] | None = None,
    ) -> TableResult[NDimension] | None:
        """
        Get fluid density `ρ`. Units depend on `unit_system` (lbm/ft³ in FIELD, kg/m³ in METRIC/SI, g/cm³ in LAB).

        Returned from the pre-built density table which was constructed using
        the standard formula at `PVTTable` initialisation time:

        - Oil:   `ρo = (ρo,SC + Rs·ρg,SC) / Bo`
        - Gas:   `ρg = (ρg,SC + Rv·ρo,SC) / Bg`  [wet gas]
                 `ρg = ρg,SC / Bg`                 [dry gas]
        - Water: `ρw = ρw,SC / Bw`

        **Oil phase - saturated/undersaturated switching**: the pre-built
        table's pressure axis is the bubble curve, so it's only exactly
        correct where `pressure` equals the cell's own bubble point.
        Above it (`pressure > bubble_point_array`), `Rs` no longer changes
        with pressure - the raw table would return `ρo` for whatever *other*
        state has its bubble point at this `pressure`, not this cell's own
        density. This branches the same way `formation_volume_factor`
        does, using its own already-correct `Bo` for the undersaturated
        recomputation, when the stock-tank densities needed for that
        recomputation were supplied at construction (`pvt=`); falls back to
        the raw table (potentially wrong above the bubble point) otherwise.

        :param pressure: Pressure. Units depend on `unit_system`.
        :param temperature: Temperature. Units depend on `unit_system`.
        :param salinity: Salinity (ppm NaCl). Water phase only.
        :param solution_gor: Solution GOR. Oil, for 2-D `bubble_point_array`
            table only - also enables the undersaturated correction.
        :param bubble_point_pressure: Pre-computed `bubble_point_array`. Oil
            only; skips the internal `bubble_point_array` lookup when supplied.
        :returns: Density, or `None` if table not present.
        """
        if self._phase == FluidPhase.WATER:
            return self.squery(
                "density",
                pressure,
                temperature,
                self.resolve_salinity(salinity),
            )
        if self._phase == FluidPhase.GAS:
            return self.query("density", pressure, temperature)

        # Mirror `formation_volume_factor`'s own saturated/undersaturated
        # split, recomputing live above the bubble point when possible.
        if not self.has("density"):
            return None
        if self._stock_tank_oil_density is None or self._stock_tank_gas_density is None:
            return self.query("density", pressure, temperature)

        bubble_point_array = (
            bubble_point_pressure
            if bubble_point_pressure is not None
            else self.bubble_point_pressure(temperature=temperature, solution_gor=solution_gor)
        )
        if bubble_point_array is None:
            return self.query("density", pressure, temperature)

        dtype = self.dtype
        pressure_array = np.atleast_1d(pressure)
        temperature_array = np.atleast_1d(temperature)
        bubble_point_array = np.atleast_1d(bubble_point_array)
        pressure_array, temperature_array, bubble_point_array = np.broadcast_arrays(
            pressure_array, temperature_array, bubble_point_array
        )

        result = np.zeros_like(pressure_array, dtype=dtype)
        saturated = pressure_array <= bubble_point_array
        undersaturated = ~saturated

        if np.any(saturated):
            result[saturated] = self.query(  # type: ignore[index]
                "density", pressure_array[saturated], temperature_array[saturated]
            )
        if np.any(undersaturated):
            # Rs is fixed at its bubble-point value above Pb - by
            # construction, that value is exactly the `solution_gor` this
            # cell's own bubble point was computed from.
            solution_gor_array = np.broadcast_to(np.atleast_1d(solution_gor), pressure_array.shape)  # type: ignore[arg-type]
            undersaturated_fvf = self.formation_volume_factor(
                pressure_array[undersaturated],
                temperature_array[undersaturated],
                bubble_point_pressure=bubble_point_array[undersaturated],
            )
            result[undersaturated] = (
                self._stock_tank_oil_density
                + solution_gor_array[undersaturated] * self._stock_tank_gas_density
            ) / np.asarray(undersaturated_fvf)

        return typing.cast(
            TableResult[NDimension],
            dtype.type(result.item())  # type: ignore[attr-defined]
            if result.size == 1
            else result.astype(dtype, copy=False),
        )

    ρ = rho = density

    def dρ_dp(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: TableQuery[NDimension] | None = None,
        solution_gor: TableQuery[NDimension] | None = None,
        bubble_point_pressure: TableQuery[NDimension] | None = None,
    ) -> TableResult[NDimension] | None:
        """
        Return `∂ρ/∂P` (density-unit / pressure-unit, unit_system-dependent - lbm/ft³/psi in FIELD).

        **Oil phase - saturated/undersaturated switching**, matching
        `density`'s own regime split:

        - Saturated: the raw table interpolant's own derivative.
        - Undersaturated: analytically differentiated from `density`'s own
          undersaturated formula (`ρ = (ρo,SC + Rs·ρg,SC) / Bo`, `Rs` fixed
          above `Pb`), which reduces to `∂ρ/∂P = -ρ · (∂Bo/∂P) / Bo` - reusing
          `db_dp`'s already-correct undersaturated derivative rather than
          re-deriving it.

        Falls back to the raw table derivative throughout (potentially
        wrong above the bubble point) when the stock-tank densities needed
        for the undersaturated recomputation weren't supplied at
        construction (`pvt=`).

        :param pressure: Pressure.
        :param temperature: Temperature.
        :param salinity: Salinity (ppm NaCl). Water phase only.
        :param solution_gor: Solution GOR. Oil, for 2-D `bubble_point_array`
            table only - also enables the undersaturated correction.
        :param bubble_point_pressure: Pre-computed `bubble_point_array`. Oil only.
        :returns: `∂ρ/∂P` or `None`.
        """
        if self._phase == FluidPhase.WATER:
            return self.squery(
                "density",
                pressure,
                temperature,
                self.resolve_salinity(salinity),
                derivative=True,
            )
        if self._phase == FluidPhase.GAS:
            return self.query("density", pressure, temperature, derivative=True)

        if not self.has("density"):
            return None
        if self._stock_tank_oil_density is None or self._stock_tank_gas_density is None:
            return self.query("density", pressure, temperature, derivative=True)

        bubble_point_array = (
            bubble_point_pressure
            if bubble_point_pressure is not None
            else self.bubble_point_pressure(temperature=temperature, solution_gor=solution_gor)
        )
        if bubble_point_array is None:
            return self.query("density", pressure, temperature, derivative=True)

        dtype = self.dtype
        pressure_array = np.atleast_1d(pressure)
        temperature_array = np.atleast_1d(temperature)
        bubble_point_array = np.atleast_1d(bubble_point_array)
        pressure_array, temperature_array, bubble_point_array = np.broadcast_arrays(
            pressure_array, temperature_array, bubble_point_array
        )

        result = np.zeros_like(pressure_array, dtype=dtype)
        saturated = pressure_array <= bubble_point_array
        undersaturated = ~saturated

        if np.any(saturated):
            result[saturated] = self.query(  # type: ignore[index]
                "density",
                pressure_array[saturated],
                temperature_array[saturated],
                derivative=True,
            )
        if np.any(undersaturated):
            p_undersaturated = pressure_array[undersaturated]
            t_undersaturated = temperature_array[undersaturated]
            pb_undersaturated = bubble_point_array[undersaturated]
            rho_undersaturated = self.density(
                p_undersaturated,
                t_undersaturated,
                solution_gor=solution_gor,
                bubble_point_pressure=pb_undersaturated,
            )
            bo_undersaturated = self.formation_volume_factor(
                p_undersaturated, t_undersaturated, bubble_point_pressure=pb_undersaturated
            )
            dbo_dp_undersaturated = self.db_dp(
                p_undersaturated, t_undersaturated, bubble_point_pressure=pb_undersaturated
            )
            result[undersaturated] = (
                -np.asarray(rho_undersaturated)
                * np.asarray(dbo_dp_undersaturated)
                / np.asarray(bo_undersaturated)
            )

        return typing.cast(
            TableResult[NDimension],
            dtype.type(result.item())  # type: ignore[attr-defined]
            if result.size == 1
            else result.astype(dtype, copy=False),
        )

    drho_dp = dρ_dp

    def compressibility(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: TableQuery[NDimension] | None = None,
    ) -> TableResult[NDimension] | None:
        """
        Get fluid compressibility `c`. Units depend on `unit_system` (1/psi in FIELD, 1/bar in METRIC, 1/atm in LAB, 1/Pa in SI).

        Returned from the pre-built compressibility table constructed at
        `PVTTable` initialisation time using:

        - Oil / water: `c = -(1/B) · (∂B/∂P)`
        - Gas: `cg = 1/P - (1/z) · (∂z/∂P)`

        :param pressure: Pressure. Units depend on `unit_system`.
        :param temperature: Temperature. Units depend on `unit_system`.
        :param salinity: Salinity (ppm NaCl). Water phase only.
        :returns: Compressibility. Units depend on `unit_system`, or `None`.
        """
        if self._phase == FluidPhase.WATER:
            return self.squery(
                "compressibility",
                pressure,
                temperature,
                self.resolve_salinity(salinity),
            )
        return self.query("compressibility", pressure, temperature)

    c = cf = compressibility

    def dc_dp(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: TableQuery[NDimension] | None = None,
    ) -> TableResult[NDimension] | None:
        """
        Return `∂c/∂P` (1 / pressure-unit-squared, unit_system-dependent - psi⁻² in FIELD).

        :param pressure: Pressure.
        :param temperature: Temperature.
        :param salinity: Salinity (ppm NaCl). Water phase only.
        :returns: `∂c/∂P` or `None`.
        """
        if self._phase == FluidPhase.WATER:
            return self.squery(
                "compressibility",
                pressure,
                temperature,
                self.resolve_salinity(salinity),
                derivative=True,
            )
        return self.query("compressibility", pressure, temperature, derivative=True)

    dcf_dp = dc_dp

    def bubble_point_pressure(
        self,
        temperature: TableQuery[NDimension],
        solution_gor: TableQuery[NDimension] | None = None,
        pressure: TableQuery[NDimension] | None = None,
        salinity: TableQuery[NDimension] | None = None,
    ) -> TableResult[NDimension] | None:
        """
        Get bubble-point pressure `bubble_point_array`. Units depend on `unit_system`.

        **Oil phase:**

        - 1-D table bubble_point_array(T): pass `temperature` only.
        - 2-D table bubble_point_array(Rs, T): both `temperature` and `solution_gor` required.

        **Water phase:** Pbw(P, T, S) - `pressure` required; uses
        `default_salinity` when `salinity` is `None`.

        **Gas phase:** returns `None`.

        :param temperature: Temperature.
        :param solution_gor: Solution GOR. Required for 2-D bubble_point_array table.
        :param pressure: Pressure. Required for water `bubble_point_array` table.
        :param salinity: Salinity (ppm NaCl). Water phase only.
        :returns: Bubble-point pressure, or `None`.
        """
        if self._phase == FluidPhase.GAS:
            return None

        if self._phase == FluidPhase.WATER:
            if pressure is None:
                raise ValidationError(
                    "Water bubble-point pressure requires the `pressure` argument."
                )
            return self.squery(
                "bubble_point_pressure",
                pressure,
                temperature,
                self.resolve_salinity(salinity),
            )

        interp = self._interpolatants.get("bubble_point_pressure")
        if interp is None:
            return None

        dtype = self.dtype
        if self._bubble_point_ndim == 1:
            result = interp(temperature)
            return (
                dtype.type(result)  # type: ignore[attr-defined]
                if np.isscalar(temperature)
                else result.astype(dtype, copy=False)  # type: ignore[attr-defined]
            )

        if solution_gor is None:
            raise ValidationError("2-D bubble-point table requires the `solution_gor` argument.")

        solution_gor_array = np.atleast_1d(solution_gor)
        temperature_array = np.atleast_1d(temperature)
        if solution_gor_array.shape != temperature_array.shape:
            if solution_gor_array.size == 1:
                solution_gor_array = np.full_like(temperature_array, solution_gor_array[0])
            elif temperature_array.size == 1:
                temperature_array = np.full_like(solution_gor_array, temperature_array[0])
            else:
                raise ValidationError(
                    "`solution_gor` and `temperature` must have compatible shapes."
                )
        result = (
            interp.ev(solution_gor_array, temperature_array)
            if hasattr(interp, "ev")
            else interp(solution_gor_array, temperature_array)
        )
        return typing.cast(
            TableResult[NDimension],
            dtype.type(result.item())  # type: ignore[attr-defined]
            if result.size == 1
            else result.astype(dtype, copy=False),
        )

    pb = pbub = p_bub = bubble_point_pressure

    def dpb_drs(
        self,
        solution_gor: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
    ) -> TableResult[NDimension] | None:
        """
        Return `∂Pb/∂Rs` (pressure-unit / GOR-unit, unit_system-dependent - psi · STB/SCF in FIELD).

        Only available when the bubble-point table is 2-D (bubble_point_array(Rs, T)).
        Returns `None` for 1-D bubble_point_array(T) tables or gas / water phases.

        :param solution_gor: Solution GOR.
        :param temperature: Temperature.
        :returns: `∂Pb/∂Rs` or `None`.
        """
        if self._phase != FluidPhase.OIL or self._bubble_point_ndim != 2:
            return None

        interp = self._derivative_interpolatants.get("dpb_drs")
        if interp is None:
            return None

        dtype = self.dtype
        solution_gor_array = np.atleast_1d(solution_gor)
        temperature_array = np.atleast_1d(temperature)
        result = (
            interp.ev(solution_gor_array, temperature_array)
            if hasattr(interp, "ev")
            else interp(solution_gor_array, temperature_array)
        )
        return typing.cast(
            TableResult[NDimension],
            dtype.type(result.item())  # type: ignore[attr-defined]
            if result.size == 1
            else result.astype(dtype, copy=False),
        )

    dpbub_drs = dpb_drs

    def solution_gas_to_oil_ratio(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        solution_gor: TableQuery[NDimension] | None = None,
        bubble_point_pressure: TableQuery[NDimension] | None = None,
    ) -> TableResult[NDimension] | None:
        """
        Get solution gas-to-oil ratio `Rs`. Units: SCF/STB (FIELD), Sm³/Sm³ (METRIC/SI), scc/scc (LAB).

        **Oil phase only.**  Rs is frozen at Rsb above bubble point.

        :param pressure: Pressure. Units depend on `unit_system`.
        :param temperature: Temperature. Units depend on `unit_system`.
        :param solution_gor: Solution GOR (dimensionless ratio). Required for 2-D `bubble_point_array` table.
        :param bubble_point_pressure: Pre-computed `bubble_point_array`. Units depend on `unit_system`.
        :returns: Rs. Units depend on `unit_system`, or `None`.
        """
        if self._phase != FluidPhase.OIL:
            return None
        if not self.has("solution_gor"):
            return None

        dtype = self.dtype
        bubble_point_array = (
            bubble_point_pressure
            if bubble_point_pressure is not None
            else self.bubble_point_pressure(temperature=temperature, solution_gor=solution_gor)
        )
        pressure_array = np.atleast_1d(pressure)
        temperature_array = np.atleast_1d(temperature)

        if bubble_point_array is None:
            return self.query("solution_gor", pressure_array, temperature_array)

        bubble_point_array = np.atleast_1d(bubble_point_array)
        pressure_array, temperature_array, bubble_point_array = np.broadcast_arrays(
            pressure_array, temperature_array, bubble_point_array
        )

        result = np.zeros_like(pressure_array)
        saturated = pressure_array <= bubble_point_array
        undersaturated = ~saturated

        if np.any(saturated):
            result[saturated] = self.query(
                "solution_gor", pressure_array[saturated], temperature_array[saturated]
            )  # type: ignore[index]
        if np.any(undersaturated):
            # Rs is constant at Rsb above bubble point
            result[undersaturated] = self.query(  # type: ignore[index]
                "solution_gor",
                bubble_point_array[undersaturated],
                temperature_array[undersaturated],
            )

        return typing.cast(
            TableResult[NDimension],
            dtype.type(result.item())  # type: ignore[attr-defined]
            if result.size == 1
            else result.astype(dtype, copy=False),
        )

    rs = Rs = solution_gor = solution_gas_to_oil_ratio

    def drs_dp(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
    ) -> TableResult[NDimension] | None:
        """
        Return `∂Rs/∂P` (GOR-unit / pressure-unit, unit_system-dependent - SCF/STB/psi in FIELD).
        Oil phase only.

        :param pressure: Pressure.
        :param temperature: Temperature.
        :returns: `∂Rs/∂P` or `None`.
        """
        if self._phase != FluidPhase.OIL:
            return None
        return self.query("solution_gor", pressure, temperature, derivative=True)

    dRs_dP = drs_dp

    def is_saturated(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        solution_gor: TableQuery[NDimension] | None = None,
    ) -> Boolean | BooleanArray[NDimension] | None:
        """
        Determine whether conditions are saturated (P ≤ bubble_point_array). Oil phase only.

        :param pressure: Pressure.
        :param temperature: Temperature.
        :param solution_gor: Solution GOR. Required for 2-D bubble_point_array table.
        :returns: Boolean mask (True = saturated), or `None` for gas.
        """
        if self._phase == FluidPhase.GAS:
            return None

        bubble_point_array = self.bubble_point_pressure(
            temperature=temperature, solution_gor=solution_gor
        )
        if bubble_point_array is None:
            return None

        pressure_array = np.atleast_1d(pressure)
        bubble_point_array = np.atleast_1d(bubble_point_array)
        result = pressure_array <= bubble_point_array
        return typing.cast(
            Boolean | BooleanArray[NDimension],
            bool(result.item()) if result.size == 1 else result,
        )

    def compressibility_factor(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
    ) -> TableResult[NDimension] | None:
        """
        Get gas z-factor `z` (dimensionless). Gas phase only.

        :param pressure: Pressure.
        :param temperature: Temperature.
        :returns: z-factor or `None`.
        """
        if self._phase != FluidPhase.GAS:
            return None
        return self.query("compressibility_factor", pressure, temperature)

    z = Z = compressibility_factor

    def dz_dp(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
    ) -> TableResult[NDimension] | None:
        """
        Return `∂z/∂P` (1 / pressure-unit, unit_system-dependent - psi⁻¹ in FIELD).
        Gas phase only.

        :param pressure: Pressure.
        :param temperature: Temperature.
        :returns: `∂z/∂P` or `None`.
        """
        if self._phase != FluidPhase.GAS:
            return None
        return self.query("compressibility_factor", pressure, temperature, derivative=True)

    dZ_dP = dz_dp

    def vaporized_oil_to_gas_ratio(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        dew_point_pressure: TableQuery[NDimension] | None = None,
    ) -> TableResult[NDimension] | None:
        """
        Get vaporized oil ratio `Rv` (unit_system-dependent - STB/scf in FIELD).
        Gas / condensate phase only.

        Rv is capped at Rv_sat (the value at dew-point pressure) above the dew
        point, analogous to Rs being capped at Rsb above bubble point for oil.

        :param pressure: Pressure.
        :param temperature: Temperature.
        :param dew_point_pressure: Pre-computed dew-point pressure. When
            provided, skips the internal dew-point lookup.
        :returns: Rv in STB/scf, or `None` if table is absent.
        """
        if self._phase != FluidPhase.GAS:
            return None
        if not self.has("vaporized_oil_to_gas_ratio"):
            return None

        dew_point_pressure = (
            dew_point_pressure
            if dew_point_pressure is not None
            else self.dew_point_pressure(temperature=temperature)
        )

        dtype = self.dtype
        pressure_array = np.atleast_1d(pressure)
        temperature_array = np.atleast_1d(temperature)

        if dew_point_pressure is None:
            return self.query("vaporized_oil_to_gas_ratio", pressure_array, temperature_array)

        dew_point_array = np.atleast_1d(dew_point_pressure)
        pressure_array, temperature_array, dew_point_array = np.broadcast_arrays(
            pressure_array, temperature_array, dew_point_array
        )

        result = np.zeros_like(pressure_array, dtype=dtype)
        above = pressure_array >= dew_point_array  # above dew point: Rv = Rv_sat (frozen)
        below = ~above

        if np.any(below):
            result[below] = self.query(  # type: ignore[index]
                "vaporized_oil_to_gas_ratio", pressure_array[below], temperature_array[below]
            )
        if np.any(above):
            result[above] = self.query(  # type: ignore[index]
                "vaporized_oil_to_gas_ratio", dew_point_array[above], temperature_array[above]
            )
        return typing.cast(
            TableResult[NDimension],
            dtype.type(result.item())  # type: ignore[attr-defined]
            if result.size == 1
            else result.astype(dtype, copy=False),
        )

    rv = Rv = vaporized_ogr = vaporized_oil_to_gas_ratio

    def drv_dp(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
    ) -> TableResult[NDimension] | None:
        """
        Return `∂Rv/∂P` (vaporized-oil-ratio-unit / pressure-unit, unit_system-dependent - STB/Mscf/psi in FIELD).
        Gas / condensate phase only.

        :param pressure: Pressure.
        :param temperature: Temperature.
        :returns: `∂Rv/∂P` or `None`.
        """
        if self._phase != FluidPhase.GAS:
            return None
        return self.query("vaporized_oil_to_gas_ratio", pressure, temperature, derivative=True)

    def dew_point_pressure(
        self,
        temperature: TableQuery[NDimension],
    ) -> TableResult[NDimension] | None:
        """
        Get gas dew-point pressure `Pdew(T)`. Gas phase only.

        :param temperature: Temperature.
        :returns: Dew-point pressure, or `None` if table is absent.
        """
        if self._phase != FluidPhase.GAS:
            return None

        interp = self._interpolatants.get("dew_point_pressure")
        if interp is None:
            return None

        dtype = self.dtype
        result = interp(temperature)
        return (
            dtype.type(result)  # type: ignore[attr-defined]
            if np.isscalar(temperature)
            else result.astype(dtype, copy=False)
        )

    pd = pdew = p_dew = dew_point_pressure

    def solubility_in_water(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: TableQuery[NDimension] | None = None,
    ) -> TableResult[NDimension] | None:
        """
        Get gas solubility in water `Rsw` (SCF/STB). Gas phase only.

        :param pressure: Pressure.
        :param temperature: Temperature.
        :param salinity: Salinity (ppm NaCl); uses default if `None`.
        :returns: Rsw in SCF/STB, or `None`.
        """
        if self._phase != FluidPhase.GAS:
            return None
        return self.squery(
            "solubility_in_water",
            pressure,
            temperature,
            self.resolve_salinity(salinity),
        )

    rsw = Rsw = solubility_in_water

    def drsw_dp(
        self,
        pressure: TableQuery[NDimension],
        temperature: TableQuery[NDimension],
        salinity: TableQuery[NDimension] | None = None,
    ) -> TableResult[NDimension] | None:
        """
        Return `∂Rsw/∂P` (solubility-unit / pressure-unit, unit_system-dependent - SCF/STB/psi in FIELD).
        Gas phase only.

        :param pressure: Pressure.
        :param temperature: Temperature.
        :param salinity: Salinity (ppm NaCl).
        :returns: `∂Rsw/∂P` or `None`.
        """
        if self._phase != FluidPhase.GAS:
            return None
        return self.squery(
            "solubility_in_water",
            pressure,
            temperature,
            self.resolve_salinity(salinity),
            derivative=True,
        )

    dRsw_dP = drsw_dp


@attrs.frozen(slots=True)
class PVTTables(StoreSerializable):
    """
    Bundle of phase-specific `PVTTable` instances for a single PVT region.

    Access phase tables via attributes:

    ```python
    tables.oil.viscosity(p, t)
    tables.gas.formation_volume_factor(p, t)
    tables.water.density(p, t, salinity=35_000)
    ```

    Build from a `PVTDataSet` or from individual serialised files:

    ```python
    tables = PVTTables.from_dataset(dataset, interpolation_method="cubic")
    tables = PVTTables.read_files(oil="oil.h5", gas="gas.h5")
    tables = PVTTables.from_deck(deck_file, temperature=200.0)
    ```
    """

    oil: PVTTable | None = None
    """PVT table for the oil phase."""

    gas: PVTTable | None = None
    """PVT table for the gas phase."""

    water: PVTTable | None = None
    """PVT table for the water phase."""

    unit_system: UnitSystem = attrs.field(init=False, repr=False, eq=False, hash=False)
    """Unit system of the underlying tables."""

    def __attrs_post_init__(self) -> None:
        if self.oil is None and self.gas is None and self.water is None:
            raise ValidationError("At least one of `oil`, `gas`, or `water` must be provided.")

        # Check that the phase tables are consistent with their declared phases
        for phase, table in (
            (FluidPhase.OIL, self.oil),
            (FluidPhase.GAS, self.gas),
            (FluidPhase.WATER, self.water),
        ):
            if table is not None and table.phase != phase:
                raise ValidationError(
                    f"Phase table mismatch: {phase.value} table has phase {table.phase.value}."
                )

        # All unit systems must be the same across phases
        unit_systems = {
            table.unit_system for table in (self.oil, self.gas, self.water) if table is not None
        }
        if len(unit_systems) > 1:
            raise ValidationError("All phase tables must have the same unit system.")

        unit_system = unit_systems.pop()
        object.__setattr__(self, "unit_system", unit_system)

    @classmethod
    def from_dataset(
        cls,
        dataset: PVTDataSet,
        interpolation_method: InterpolationMethod = "linear",
        validate: bool = True,
        warn_on_extrapolation: bool = False,
        pvt: StaticPVT | None = None,
        dtype: npt.DTypeLike = None,
    ) -> Self:
        """
        Build a `PVTTables` bundle from a `PVTDataSet`.

        :param dataset: Source `PVTDataSet`.
        :param interpolation_method: `"linear"` or `"cubic"`.
        :param validate: Run physical-consistency checks on each phase.
        :param warn_on_extrapolation: Log warnings when queries exceed table bounds.
        :param pvt: Reference densities for derived table construction
            (`stock_tank_oil_density`, `stock_tank_gas_density`, `stock_tank_water_density`).
        :returns: `PVTTables` ready for simulation.
        """
        base_kwargs: dict[str, typing.Any] = dict(
            interpolation_method=interpolation_method,
            validate=validate,
            warn_on_extrapolation=warn_on_extrapolation,
            pvt=pvt,
            dtype=dtype,
        )
        oil_table = PVTTable(dataset.oil, **base_kwargs) if dataset.oil is not None else None
        gas_table = PVTTable(dataset.gas, **base_kwargs) if dataset.gas is not None else None
        water_table = PVTTable(dataset.water, **base_kwargs) if dataset.water is not None else None
        return cls(oil=oil_table, gas=gas_table, water=water_table)

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        temperature: Temperature | Number,
        pvtnum: int = 1,
        *,
        interpolation_method: InterpolationMethod = "linear",
        validate: bool = True,
        warn_on_extrapolation: bool = False,
        dtype: npt.DTypeLike = None,
    ) -> Self:
        """
        Build `PVTTables` for a single PVT region from a parsed `DeckFile`.

        Convenience wrapper around `PVT.from_deck` for the
        common single-region case.

        :param deck_file: Parsed `DeckFile` containing PROPS-section keywords.
        :param temperature: Reservoir temperature - used as the single
            temperature value for deck-loaded isothermal tables, or a reservoir
            regional `Temperature` instance.
        :param pvtnum: 1-based PVT region index to extract (default 1).
        :param interpolation_method: `"linear"` or `"cubic"`.
        :param validate: Run physical-consistency checks.
        :param warn_on_extrapolation: Log warnings on extrapolation.
        :param pvt: Reference densities for derived table derivation.
        :returns: `PVTTables` for the specified region.
        """
        from bores.blackoil.pvt.regions import PVT

        regions = PVT.from_deck(
            deck_file=deck_file,
            temperature=temperature,
            interpolation_method=interpolation_method,
            validate=validate,
            warn_on_extrapolation=warn_on_extrapolation,
            dtype=dtype,
        )
        return typing.cast(Self, regions.region(pvtnum))

    @property
    def dataset(self) -> PVTDataSet:
        """
        Return the underlying raw data as a `PVTDataSet`.

        Useful when you want to persist the raw tabulated data independently
        of interpolator settings.

        :returns: `PVTDataSet` with each non-`None` phase's raw data.
        """
        return PVTDataSet(
            oil=self.oil._data if self.oil is not None else None,
            gas=self.gas._data if self.gas is not None else None,
            water=self.water._data if self.water is not None else None,
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `PVTTables` with all phase tables converted to *target*.

        :param target: Target `UnitSystem`.
        :returns: New `PVTTables` in *target* units.
        """
        return attrs.evolve(
            self,
            oil=self.oil.convert(target, table=table) if self.oil is not None else None,
            gas=self.gas.convert(target, table=table) if self.gas is not None else None,
            water=self.water.convert(target, table=table) if self.water is not None else None,
        )

import logging
import typing
import warnings

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RectBivariateSpline  # type: ignore[import-untyped]

from bores.blackoil.fluids.simple import Fluid
from bores.blackoil.pvt.data import PVTData, PVTDataSet
from bores.blackoil.pvt.tables import PVTTable
from bores.constants import c
from bores.correlations import arrays, scalars
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.types import (
    FluidPhase,
    NDimension,
    Number,
    NumberArray,
    OneDimension,
    ThreeDimensions,
    TwoDimensions,
    UnitSystem,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_gas_pvt_data",
    "build_oil_pvt_data",
    "build_pvt_dataset",
    "build_water_pvt_data",
]


def _resolve_gas(
    gas: typing.Any | None,
    gas_gravity: Number | None,
) -> tuple[str, Number | None, PVTTable | None]:
    """
    Resolve the *gas* argument used across all three correlation builders.

    Returns `(gas_name, gas_gravity, pvt_table)` where:

    - `gas_name` - string name for correlation lookup.
    - `gas_gravity` - scalar gravity (may be `None` if taken from `pvt_table`).
    - `pvt_table` - `PVTTable` to read gas properties from (may be `None`).

    When `gas_gravity` is absent and a gas `PVTTable` is available, gravity is
    derived from the table's own density lookup (`ρ_gas / ρ_air` at the same
    average P/T) rather than from a `specific_gravity` accessor - `PVTTable`
    only exposes `density`, `formation_volume_factor`, `viscosity`, etc., it
    has no `specific_gravity` method.

    :param gas: Gas identity. Either a string name (e.g. `"Methane"`) or a
        `Fluid` instance.
    :param gas_gravity: Override gas gravity (air=1).
    :returns: Resolved 3-tuple.
    """
    pvt_table: PVTTable | None = None
    gas_name: str = c.RESERVOIR_GAS

    if gas is None:
        pass
    elif isinstance(gas, str):
        gas_name = gas
    elif isinstance(gas, Fluid):
        gas_name = gas.name
        if gas.pvt is not None:
            pvt_table = gas.pvt

    if gas_gravity is None and pvt_table is not None:
        avg_pressure = np.mean(pvt_table._extrapolation_bounds["pressure"])  # type: ignore[arg-type]
        avg_temperature = np.mean(pvt_table._extrapolation_bounds["temperature"])  # type: ignore[arg-type]
        avg_density = typing.cast(
            Number | None,
            pvt_table.density(avg_pressure, avg_temperature),
        )
        if avg_density is not None:
            gas_gravity = typing.cast(
                Number,
                scalars.compute_gas_gravity_from_density(
                    pressure=avg_pressure,
                    temperature=avg_temperature,
                    density=avg_density,
                ),
            )

    if gas_gravity is None:
        gas_gravity = typing.cast(Number, scalars.compute_gas_gravity(gas=gas_name))

    return gas_name, gas_gravity, pvt_table


def get_gas_tables_from_pvt_table(
    pvt_table: PVTTable,
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    dtype: npt.DTypeLike = None,
) -> dict[str, NumberArray[NDimension] | None]:
    """
    Evaluate a gas `PVTTable` on a `(P, T)` meshgrid.

    :param pvt_table: Source gas `PVTTable`.
    :param pressure: 2-D pressure grid, shape `(n_p, n_t)`.
    :param temperature: 2-D temperature grid, shape `(n_p, n_t)`.
    :returns: Dict of property name -> 2-D array.
    """
    flat_pressure = pressure.ravel()
    flat_temperature = temperature.ravel()
    shape = pressure.shape

    def _eval(method_name: str) -> NumberArray[NDimension] | None:
        method = getattr(pvt_table, method_name, None)
        if method is None:
            return None

        result = method(flat_pressure, flat_temperature)
        if result is None:
            return None
        return typing.cast(
            NumberArray[NDimension],
            np.asarray(result).reshape(shape).astype(dtype, copy=False),
        )

    return {
        "viscosity_table": _eval("viscosity"),
        "density_table": _eval("density"),
        "formation_volume_factor_table": _eval("formation_volume_factor"),
        "compressibility_table": _eval("compressibility"),
        "compressibility_factor_table": _eval("compressibility_factor"),
    }


def build_oil_pvt_data(
    pressures: NumberArray[OneDimension],
    temperatures: NumberArray[OneDimension],
    oil_specific_gravity: Number = 0.85,
    gas_gravity: Number | None = None,
    estimatedsolution_gor: Number | None = None,
    bubble_point_pressures: NumberArray[OneDimension] | NumberArray[TwoDimensions] | None = None,
    solution_gas_to_oil_ratios: NumberArray[OneDimension] | None = None,
    gas: str | typing.Any | None = None,
    viscosity_table: NumberArray[TwoDimensions] | None = None,
    density_table: NumberArray[TwoDimensions] | None = None,
    formation_volume_factor_table: NumberArray[TwoDimensions] | None = None,
    compressibility_table: NumberArray[TwoDimensions] | None = None,
    solution_gas_to_oil_ratio_table: NumberArray[TwoDimensions] | None = None,
    stock_tank_oil_density: Number | None = None,
    stock_tank_gas_density: Number | None = None,
    dtype: npt.DTypeLike = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
    **kwargs: typing.Any,
) -> PVTData:
    """
    Build a `PVTData` for the oil phase using empirical correlations.

    Missing primary tables (Bo, μo, Rs) are computed from correlations
    (Standing, Vazquez-Beggs, Beggs-Robinson). Missing derived tables
    (density, compressibility) are built from the primary tables and the
    stock-tank reference densities if supplied.

    :param pressures: 1-D array of pressures (psi), strictly increasing.
    :param temperatures: 1-D array of temperatures (°F), strictly increasing.
    :param oil_specific_gravity: Oil specific gravity (dimensionless, water=1).
    :param gas_gravity: Gas specific gravity (air=1). Derived from `gas` if absent.
    :param estimatedsolution_gor: Estimated Rs (SCF/STB) for the 1-D Pb
        correlation. Estimated from API if absent.
    :param bubble_point_pressures: Pre-computed Pb array. 1-D Pb(T) or 2-D Pb(Rs, T).
    :param solution_gas_to_oil_ratios: Rs axis for the 2-D Pb table.
    :param gas: Gas identity (string name or `Fluid`).
    :param viscosity_table: Pre-computed μo(P, T) (n_p, n_t) in cP.
    :param density_table: Pre-computed ρo(P, T) (n_p, n_t) in lbm/ft³.
    :param formation_volume_factor_table: Pre-computed Bo(P, T) (n_p, n_t) in bbl/STB.
    :param compressibility_table: Pre-computed co(P, T) (n_p, n_t) in psi⁻¹.
    :param solution_gas_to_oil_ratio_table: Pre-computed Rs(P, T) (n_p, n_t) in SCF/STB.
    :param stock_tank_oil_density: Stock-tank oil density (lbm/ft³). Used to derive
        `density_table` when absent.
    :param stock_tank_gas_density: Stock-tank gas density (lbm/ft³). Used to derive
        `density_table` when absent.
    :returns: `PVTData` with `phase=OIL`.
    """
    if pressures.ndim != 1 or not np.all(np.diff(pressures) > 0):
        raise ValidationError("`pressures` must be a strictly increasing 1-D array.")
    if temperatures.ndim != 1 or not np.all(np.diff(temperatures) > 0):
        raise ValidationError("`temperatures` must be a strictly increasing 1-D array.")

    n_p = len(pressures)
    n_t = len(temperatures)
    dtype = np.dtype(dtype if dtype is not None else get_dtype())

    _, gas_gravity, pvt_table = _resolve_gas(gas, gas_gravity)
    assert gas_gravity is not None

    pressure_table, temperature_table = np.meshgrid(pressures, temperatures, indexing="ij")

    # Gas FVF grid for compressibility calculation
    gas_fvf_table: NumberArray[TwoDimensions] | None = None
    if pvt_table is not None:
        gas_tables = get_gas_tables_from_pvt_table(
            pvt_table, pressure_table, temperature_table, dtype=dtype
        )
        gas_fvf_table = typing.cast(
            NumberArray[TwoDimensions] | None,
            gas_tables.get("formation_volume_factor_table"),
        )

    if gas_fvf_table is None:
        z_factor_table = typing.cast(
            NumberArray[TwoDimensions],
            arrays.compute_gas_compressibility_factor(
                pressure=pressure_table,
                temperature=temperature_table,
                gas_gravity=np.full((n_p, n_t), gas_gravity, dtype=dtype),
            ),
        )
        gas_fvf_table = typing.cast(
            NumberArray[TwoDimensions],
            arrays.compute_gas_formation_volume_factor(
                pressure=pressure_table,
                temperature=temperature_table,
                gas_compressibility_factor=z_factor_table,
            ),
        )

    oil_api_gravity = typing.cast(
        Number,
        scalars.compute_oil_api_gravity(oil_specific_gravity=oil_specific_gravity),
    )
    oil_api_gravity_table = np.full((n_p, n_t), oil_api_gravity, dtype=dtype)
    specific_gravity_table = np.full((n_p, n_t), oil_specific_gravity, dtype=dtype)

    if bubble_point_pressures is None:
        gas_gravity_1d = np.full(n_t, gas_gravity, dtype=dtype)
        oil_api_gravity_1d = typing.cast(
            NumberArray[OneDimension],
            arrays.compute_oil_api_gravity(np.full(n_t, oil_specific_gravity, dtype=dtype)),
        )
        if solution_gas_to_oil_ratios is not None:
            n_rs = len(solution_gas_to_oil_ratios)
            bubble_point_pressures = np.zeros((n_rs, n_t), dtype=dtype)
            for i, solution_gor in enumerate(solution_gas_to_oil_ratios):
                bubble_point_pressures[i, :] = arrays.compute_oil_bubble_point_pressure(
                    gas_gravity=gas_gravity_1d,
                    oil_api_gravity=oil_api_gravity_1d,
                    temperature=temperatures,
                    gas_to_oil_ratio=np.full(n_t, solution_gor, dtype=dtype),
                )
        else:
            oil_api_gravity = typing.cast(
                Number, scalars.compute_oil_api_gravity(oil_specific_gravity)
            )
            if estimatedsolution_gor is None:
                estimatedsolution_gor = np.clip(
                    10 ** (0.0125 * oil_api_gravity) * 50.0, 50.0, 2000.0
                )
                warnings.warn(
                    f"No `estimatedsolution_gor` provided. Estimating Rs = "
                    f"{estimatedsolution_gor:.1f} SCF/STB from API = {oil_api_gravity:.1f}°. "
                    "Pass `estimatedsolution_gor` for best results.",
                    UserWarning,
                    stacklevel=2,
                )
            bubble_point_pressures = typing.cast(
                NumberArray[OneDimension],
                arrays.compute_oil_bubble_point_pressure(
                    gas_gravity=gas_gravity_1d,
                    oil_api_gravity=oil_api_gravity_1d,
                    temperature=temperatures,
                    gas_to_oil_ratio=np.full(n_t, estimatedsolution_gor, dtype=dtype),
                ),
            )

    # 2-D Pb grid for property table construction
    if bubble_point_pressures.ndim == 1:
        bubble_point_pressure_table = np.broadcast_to(
            bubble_point_pressures[np.newaxis, :], (n_p, n_t)
        ).copy()
    else:
        if solution_gas_to_oil_ratio_table is None:
            solution_gas_to_oil_ratio_table = typing.cast(
                NumberArray[TwoDimensions],
                arrays.estimate_solution_gor(
                    pressure=pressure_table,
                    temperature=temperature_table,
                    oil_api_gravity=oil_api_gravity_table,
                    gas_gravity=np.full((n_p, n_t), gas_gravity, dtype=dtype),
                    maximum_iterations=20,
                    tolerance=1e-4,
                ),
            )
        assert solution_gas_to_oil_ratios is not None
        pb_interp = RectBivariateSpline(
            x=solution_gas_to_oil_ratios,
            y=temperatures,
            z=bubble_point_pressures,
            kx=1,
            ky=1,
        )
        temperatures_flat = np.broadcast_to(temperatures, (n_p, n_t))
        assert solution_gas_to_oil_ratio_table is not None
        bubble_point_pressure_table = (
            pb_interp
            .ev(solution_gas_to_oil_ratio_table.ravel(), temperatures_flat.ravel())
            .reshape(n_p, n_t)
            .astype(dtype)
        )

    gas_gravity_table = np.full((n_p, n_t), gas_gravity, dtype=dtype)

    if solution_gas_to_oil_ratio_table is None:
        solution_gas_to_oil_ratio_table = typing.cast(
            NumberArray[TwoDimensions],
            arrays.compute_gas_to_oil_ratio(
                pressure=pressure_table,
                temperature=temperature_table,
                bubble_point_pressure=bubble_point_pressure_table,  # type: ignore[arg-type]
                gas_gravity=gas_gravity_table,
                oil_api_gravity=oil_api_gravity_table,
            ),
        )

    needs_fvf = formation_volume_factor_table is None
    needs_co = compressibility_table is None

    gor_at_bubble_point_table = typing.cast(
        NumberArray[TwoDimensions],
        arrays.compute_gas_to_oil_ratio(
            pressure=bubble_point_pressure_table,  # type: ignore[arg-type]
            temperature=temperature_table,
            bubble_point_pressure=bubble_point_pressure_table,  # type: ignore[arg-type]
            gas_gravity=gas_gravity_table,
            oil_api_gravity=oil_api_gravity_table,
        ),
    )

    if needs_fvf and needs_co:
        estimated_oil_compressibility = np.full((n_p, n_t), 1e-5, dtype=dtype)
        max_delta = 0.0
        for _ in range(10):
            formation_volume_factor_table = typing.cast(
                NumberArray[TwoDimensions],
                arrays.compute_oil_formation_volume_factor(
                    pressure=pressure_table,
                    temperature=temperature_table,
                    bubble_point_pressure=bubble_point_pressure_table,  # type: ignore[arg-type]
                    oil_specific_gravity=specific_gravity_table,
                    gas_gravity=gas_gravity_table,
                    gas_to_oil_ratio=solution_gas_to_oil_ratio_table,
                    oil_compressibility=estimated_oil_compressibility,
                ),
            )
            new_oil_compressibility = typing.cast(
                NumberArray[TwoDimensions],
                arrays.compute_oil_compressibility(
                    pressure=pressure_table,
                    temperature=temperature_table,
                    bubble_point_pressure=bubble_point_pressure_table,  # type: ignore[arg-type]
                    oil_api_gravity=oil_api_gravity_table,
                    gas_gravity=gas_gravity_table,
                    gor_at_bubble_point_pressure=gor_at_bubble_point_table,
                    gas_formation_volume_factor=gas_fvf_table,
                    oil_formation_volume_factor=formation_volume_factor_table,
                ),
            )
            max_delta = np.max(np.abs(new_oil_compressibility - estimated_oil_compressibility))
            estimated_oil_compressibility = new_oil_compressibility
            if max_delta < 1e-7:
                break
        else:
            warnings.warn(
                f"Bo/co bootstrap did not converge (final max Δco = {max_delta:.3e} psi⁻¹).",
                UserWarning,
                stacklevel=2,
            )
        compressibility_table = estimated_oil_compressibility

    elif needs_fvf:
        assert compressibility_table is not None
        formation_volume_factor_table = typing.cast(
            NumberArray[TwoDimensions],
            arrays.compute_oil_formation_volume_factor(
                pressure=pressure_table,
                temperature=temperature_table,
                bubble_point_pressure=bubble_point_pressure_table,  # type: ignore[arg-type]
                oil_specific_gravity=specific_gravity_table,
                gas_gravity=gas_gravity_table,
                gas_to_oil_ratio=solution_gas_to_oil_ratio_table,
                oil_compressibility=compressibility_table,
            ),
        )

    elif needs_co:
        assert formation_volume_factor_table is not None
        # NOTE: previously called with `oil_api_gravity_grid=`/`gas_gravity_grid=`,
        # which are not parameters of `compute_oil_compressibility` (it takes
        # `oil_api_gravity`/`gas_gravity`) - this branch raised `TypeError` any
        # time a caller supplied `formation_volume_factor_table` without also
        # supplying `compressibility_table`. Fixed to use the real parameter names.
        compressibility_table = typing.cast(
            NumberArray[TwoDimensions],
            arrays.compute_oil_compressibility(
                pressure=pressure_table,
                temperature=temperature_table,
                bubble_point_pressure=bubble_point_pressure_table,  # type: ignore[arg-type]
                oil_api_gravity=oil_api_gravity_table,
                gas_gravity=gas_gravity_table,
                gor_at_bubble_point_pressure=gor_at_bubble_point_table,
                gas_formation_volume_factor=gas_fvf_table,
                oil_formation_volume_factor=formation_volume_factor_table,
            ),
        )

    # Derived density table using the formula ρo = (ρo,SC + Rs·ρg,SC) / Bo
    if density_table is None:
        if stock_tank_oil_density is not None and stock_tank_gas_density is not None:
            density_table = (
                (stock_tank_oil_density + solution_gas_to_oil_ratio_table * stock_tank_gas_density)
                / formation_volume_factor_table  # type: ignore
            ).astype(dtype)
        else:
            assert formation_volume_factor_table is not None
            density_table = typing.cast(
                NumberArray[TwoDimensions],
                arrays.compute_live_oil_density(
                    api_gravity=oil_api_gravity_table,
                    gas_gravity=gas_gravity_table,
                    gas_to_oil_ratio=solution_gas_to_oil_ratio_table,
                    formation_volume_factor=formation_volume_factor_table,
                ),
            )

    if viscosity_table is None:
        viscosity_table = typing.cast(
            NumberArray[TwoDimensions],
            arrays.compute_oil_viscosity(
                pressure=pressure_table,
                temperature=temperature_table,
                bubble_point_pressure=bubble_point_pressure_table,  # type: ignore[arg-type]
                oil_specific_gravity=specific_gravity_table,
                gas_to_oil_ratio=solution_gas_to_oil_ratio_table,
                gor_at_bubble_point_pressure=gor_at_bubble_point_table,
            ),
        )
    return PVTData(
        phase=FluidPhase.OIL,
        pressures=pressures,
        temperatures=temperatures,
        bubble_point_pressures=bubble_point_pressures,
        solution_gas_to_oil_ratios=solution_gas_to_oil_ratios,
        viscosity_table=viscosity_table,
        density_table=density_table,
        formation_volume_factor_table=formation_volume_factor_table,
        compressibility_table=compressibility_table,
        solution_gor_table=solution_gas_to_oil_ratio_table,
        dtype=dtype,
        unit_system=unit_system,
    )


def build_gas_pvt_data(
    pressures: NumberArray[OneDimension],
    temperatures: NumberArray[OneDimension],
    gas_gravity: Number | None = None,
    molecular_weight: Number | None = None,
    gas: str | typing.Any | None = None,
    water_salinities: NumberArray[OneDimension] | None = None,
    viscosity_table: NumberArray[TwoDimensions] | None = None,
    density_table: NumberArray[TwoDimensions] | None = None,
    formation_volume_factor_table: NumberArray[TwoDimensions] | None = None,
    compressibility_table: NumberArray[TwoDimensions] | None = None,
    compressibility_factor_table: NumberArray[TwoDimensions] | None = None,
    molecular_weight_table: NumberArray[TwoDimensions] | None = None,
    solubility_in_water_table: NumberArray[ThreeDimensions] | None = None,
    stock_tank_gas_density: Number | None = None,
    stock_tank_oil_density: Number | None = None,
    vaporized_oil_ratio_table: NumberArray[TwoDimensions] | None = None,
    dtype: npt.DTypeLike = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
    **kwargs: typing.Any,
) -> PVTData:
    """
    Build a `PVTData` for the gas phase using empirical correlations.

    Computes missing tables from correlations (DAK z-factor, Lee-Gonzalez-Eakin
    viscosity, etc.).  Density is derived from `ρg = ρg,SC / Bg` (dry gas)
    or `ρg = (ρg,SC + Rv·ρo,SC) / Bg` (wet gas) when reference densities
    are provided; otherwise falls back to the correlation-based grid builder.

    :param pressures: 1-D array of pressures (psi), strictly increasing.
    :param temperatures: 1-D array of temperatures (°F), strictly increasing.
    :param gas_gravity: Gas specific gravity (air=1). Derived from `gas` if absent.
    :param gas: Gas identity (string or `Fluid`).
    :param water_salinities: 1-D salinity array (ppm) for Rsw(P, T, S) table.
    :param viscosity_table: Pre-computed μg(P, T) (n_p, n_t) in cP.
    :param density_table: Pre-computed ρg(P, T) (n_p, n_t) in lbm/ft³.
    :param formation_volume_factor_table: Pre-computed Bg(P, T) (n_p, n_t) in ft³/SCF.
    :param compressibility_table: Pre-computed cg(P, T) (n_p, n_t) in psi⁻¹.
    :param compressibility_factor_table: Pre-computed z(P, T) (n_p, n_t).
    :param molecular_weight: Pre-computed Mg in lbm/lb-mol. NOTE: takes
        precedence over `molecular_weight_table`. If you're passing a
        pre-computed table instead of a scalar, this function currently
        still recomputes it from `gas_gravity` unless `molecular_weight`
        (the scalar) is also given.
    :param solubility_in_water_table: Pre-computed Rsw(P, T, S) (n_p, n_t, n_s).
    :param stock_tank_gas_density: Stock-tank gas density (lbm/ft³) for density derivation.
    :param stock_tank_oil_density: Stock-tank oil density (lbm/ft³) for wet-gas density
        derivation when `vaporized_oil_ratio_table` is provided.
    :param vaporized_oil_ratio_table: Pre-computed Rv(P, T) (n_p, n_t) in STB/Mscf.
    :returns: `PVTData` with `phase=GAS`.
    """
    if pressures.ndim != 1 or not np.all(np.diff(pressures) > 0):
        raise ValidationError("`pressures` must be a strictly increasing 1-D array.")
    if temperatures.ndim != 1 or not np.all(np.diff(temperatures) > 0):
        raise ValidationError("`temperatures` must be a strictly increasing 1-D array.")

    gas_name, gas_gravity, pvt_table = _resolve_gas(gas, gas_gravity)
    assert gas_gravity is not None

    dtype = np.dtype(dtype if dtype is not None else get_dtype())
    n_p = len(pressures)
    n_t = len(temperatures)

    pressure_table, temperature_table = np.meshgrid(pressures, temperatures, indexing="ij")

    if pvt_table is not None:
        gas_tables = get_gas_tables_from_pvt_table(
            pvt_table, pressure_table, temperature_table, dtype=dtype
        )

        if viscosity_table is None:
            viscosity_table = typing.cast(
                NumberArray[TwoDimensions] | None,
                gas_tables.get("viscosity_table"),
            )
        if density_table is None:
            density_table = typing.cast(
                NumberArray[TwoDimensions] | None,
                gas_tables.get("density_table"),
            )
        if formation_volume_factor_table is None:
            formation_volume_factor_table = typing.cast(
                NumberArray[TwoDimensions] | None,
                gas_tables.get("formation_volume_factor_table"),
            )
        if compressibility_table is None:
            compressibility_table = typing.cast(
                NumberArray[TwoDimensions] | None,
                gas_tables.get("compressibility_table"),
            )
        if compressibility_factor_table is None:
            compressibility_factor_table = typing.cast(
                NumberArray[TwoDimensions] | None,
                gas_tables.get("compressibility_factor_table"),
            )

    gas_gravity_table = np.full((n_p, n_t), gas_gravity, dtype=dtype)

    if molecular_weight is None:
        molecular_weight_table = typing.cast(
            NumberArray[TwoDimensions],
            arrays.compute_gas_molecular_weight(gas_gravity=gas_gravity_table),
        )
    else:
        molecular_weight_table = np.full((n_p, n_t), molecular_weight, dtype=dtype)

    if compressibility_factor_table is None:
        compressibility_factor_table = typing.cast(
            NumberArray[TwoDimensions],
            arrays.compute_gas_compressibility_factor(
                pressure=pressure_table,
                temperature=temperature_table,
                gas_gravity=gas_gravity_table,
            ),
        )

    if formation_volume_factor_table is None:
        formation_volume_factor_table = typing.cast(
            NumberArray[TwoDimensions],
            arrays.compute_gas_formation_volume_factor(
                pressure=pressure_table,
                temperature=temperature_table,
                gas_compressibility_factor=compressibility_factor_table,
            ),
        )

    if compressibility_table is None:
        compressibility_table = typing.cast(
            NumberArray[TwoDimensions],
            arrays.compute_gas_compressibility(
                pressure=pressure_table,
                temperature=temperature_table,
                gas_gravity=gas_gravity_table,
                gas_compressibility_factor=compressibility_factor_table,
            ),
        )

    # Derived density: formula-first, then correlation fallback
    if density_table is None:
        if stock_tank_gas_density is not None:
            if vaporized_oil_ratio_table is not None and stock_tank_oil_density is not None:
                density_table = (  # type: ignore[assignment]
                    (stock_tank_gas_density + vaporized_oil_ratio_table * stock_tank_oil_density)
                    / formation_volume_factor_table
                ).astype(dtype, copy=False)
            else:
                density_table = typing.cast(
                    NumberArray[TwoDimensions],
                    (stock_tank_gas_density / formation_volume_factor_table).astype(
                        dtype, copy=False
                    ),
                )
        else:
            density_table = typing.cast(
                NumberArray[TwoDimensions],
                arrays.compute_gas_density(
                    pressure=pressure_table,
                    temperature=temperature_table,
                    gas_gravity=gas_gravity_table,
                    gas_compressibility_factor=compressibility_factor_table,
                ),
            )

    if viscosity_table is None:
        assert density_table is not None
        viscosity_table = typing.cast(
            NumberArray[TwoDimensions],
            arrays.compute_gas_viscosity(
                temperature=temperature_table,
                gas_density=density_table,
                gas_molecular_weight=molecular_weight_table,
            ),
        )

    if solubility_in_water_table is None and water_salinities is not None:
        pressure_table_3d, temperature_table_3d, salinity_table_3d = np.meshgrid(
            pressures, temperatures, water_salinities, indexing="ij"
        )
        solubility_in_water_table = typing.cast(
            NumberArray[ThreeDimensions],
            arrays.compute_gas_solubility_in_water(
                pressure=pressure_table_3d,
                temperature=temperature_table_3d,
                salinity=salinity_table_3d,
                gas=gas_name,
            ),
        )

    return PVTData(
        phase=FluidPhase.GAS,
        pressures=pressures,
        temperatures=temperatures,
        salinities=water_salinities,
        viscosity_table=viscosity_table,
        density_table=density_table,
        formation_volume_factor_table=formation_volume_factor_table,
        compressibility_table=compressibility_table,
        compressibility_factor_table=compressibility_factor_table,
        solubility_in_water_table=solubility_in_water_table,
        vaporized_oil_ratio_table=vaporized_oil_ratio_table,
        dtype=dtype,
        unit_system=unit_system,
    )


def build_water_pvt_data(
    pressures: NumberArray[OneDimension],
    temperatures: NumberArray[OneDimension],
    salinities: NumberArray[OneDimension] | None = None,
    water_salinity: Number | None = None,
    gas_gravity: Number | None = None,
    gas: str | typing.Any | None = None,
    viscosity_table: NumberArray[ThreeDimensions] | None = None,
    density_table: NumberArray[ThreeDimensions] | None = None,
    formation_volume_factor_table: NumberArray[ThreeDimensions] | None = None,
    compressibility_table: NumberArray[ThreeDimensions] | None = None,
    bubble_point_pressure_table: NumberArray[ThreeDimensions] | None = None,
    gas_free_water_fvf_table: NumberArray[TwoDimensions] | None = None,
    stock_tank_water_density: Number | None = None,
    dtype: npt.DTypeLike = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
    **kwargs: typing.Any,
) -> PVTData:
    """
    Build a `PVTData` for the water phase using empirical correlations.

    All property tables are 3-D with shape `(n_pressures, n_temperatures,
    n_salinities)`. Density is derived from `ρw = ρw,SC / Bw` when
    `stock_tank_water_density` is provided; otherwise falls back to the
    McCain-based grid builder.

    :param pressures: 1-D array of pressures (psi), strictly increasing.
    :param temperatures: 1-D array of temperatures (°F), strictly increasing.
    :param salinities: 1-D array of salinities (ppm NaCl), strictly increasing.
        Falls back to `[water_salinity]` or library default when absent.
    :param water_salinity: Single salinity scalar (ppm).  Used when `salinities`
        is absent.
    :param gas_gravity: Gas specific gravity for compressibility liberation correction.
    :param gas: Gas identity (string or `Fluid`).
    :param viscosity_table: Pre-computed μw(P, T, S) (n_p, n_t, n_s) in cP.
    :param density_table: Pre-computed ρw(P, T, S) (n_p, n_t, n_s) in lbm/ft³.
    :param formation_volume_factor_table: Pre-computed Bw(P, T, S) (n_p, n_t, n_s).
    :param compressibility_table: Pre-computed cw(P, T, S) (n_p, n_t, n_s) in psi⁻¹.
    :param bubble_point_pressure_table: Pre-computed Pbw(P, T, S) (n_p, n_t, n_s).
    :param gas_free_water_fvf_table: Pre-computed Bw_gf(P, T) (n_p, n_t) in bbl/STB.
    :param stock_tank_water_density: Stock-tank water density (lbm/ft³) for density derivation.
    :returns: `PVTData` with `phase=WATER`.
    """
    if pressures.ndim != 1 or not np.all(np.diff(pressures) > 0):
        raise ValidationError("`pressures` must be a strictly increasing 1-D array.")
    if temperatures.ndim != 1 or not np.all(np.diff(temperatures) > 0):
        raise ValidationError("`temperatures` must be a strictly increasing 1-D array.")

    dtype = np.dtype(dtype if dtype is not None else get_dtype())
    n_p = len(pressures)
    n_t = len(temperatures)

    if salinities is None:
        salinity = water_salinity if water_salinity is not None else c.DEFAULT_WATER_SALINITY_PPM
        salinities = typing.cast(NumberArray[OneDimension], np.array([salinity], dtype=dtype))

    gas, gas_gravity, pvt_table = _resolve_gas(gas, gas_gravity)
    assert gas_gravity is not None

    n_s = len(salinities)

    pressure_table_2d, temperature_table_2d = np.meshgrid(pressures, temperatures, indexing="ij")
    pressure_table_3d, temperature_table_3d, salinity_table_3d = np.meshgrid(
        pressures, temperatures, salinities, indexing="ij"
    )

    # Gas Bg (2-D) for water compressibility liberation correction
    if pvt_table is not None:
        gas_tables = get_gas_tables_from_pvt_table(
            pvt_table, pressure_table_2d, temperature_table_2d, dtype=dtype
        )
        gas_fvf_table_2d = typing.cast(
            NumberArray[TwoDimensions] | None,
            gas_tables.get("formation_volume_factor_table"),
        )
        if gas_fvf_table_2d is not None:
            gas_fvf_table_3d = np.broadcast_to(
                gas_fvf_table_2d[:, :, np.newaxis], (n_p, n_t, n_s)
            ).copy()
        else:
            gas_fvf_table_3d = np.ones((n_p, n_t, n_s), dtype=dtype)
    else:
        z_factor_table_2d = typing.cast(
            NumberArray[TwoDimensions],
            arrays.compute_gas_compressibility_factor(
                pressure=pressure_table_2d,
                temperature=temperature_table_2d,
                gas_gravity=np.full((n_p, n_t), gas_gravity, dtype=dtype),
            ),
        )
        gas_fvf_table_2d = typing.cast(
            NumberArray[TwoDimensions],
            arrays.compute_gas_formation_volume_factor(
                pressure=pressure_table_2d,
                temperature=temperature_table_2d,
                gas_compressibility_factor=z_factor_table_2d,
            ),
        )
        gas_fvf_table_3d = np.broadcast_to(
            gas_fvf_table_2d[:, :, np.newaxis], (n_p, n_t, n_s)
        ).copy()

    gas_gravity_table_3d = np.full((n_p, n_t, n_s), gas_gravity, dtype=dtype)

    if gas_free_water_fvf_table is None:
        gas_free_water_fvf_table = typing.cast(
            NumberArray[TwoDimensions],
            arrays.compute_gas_free_water_formation_volume_factor(
                pressure=pressure_table_2d,
                temperature=temperature_table_2d,
            ),
        )

    gas_solubility_table_3d = typing.cast(
        NumberArray[ThreeDimensions],
        arrays.compute_gas_solubility_in_water(
            pressure=pressure_table_3d,
            temperature=temperature_table_3d,
            salinity=salinity_table_3d,
            gas=gas,
        ),
    )

    if bubble_point_pressure_table is None:
        bubble_point_pressure_table = typing.cast(
            NumberArray[ThreeDimensions],
            arrays.compute_water_bubble_point_pressure(
                temperature=temperature_table_3d,
                gas_solubility_in_water=gas_solubility_table_3d,
                salinity=salinity_table_3d,
                gas=gas,
            ),
        )

    gas_free_fvf_table_3d = np.broadcast_to(
        gas_free_water_fvf_table[:, :, np.newaxis], (n_p, n_t, n_s)
    ).copy()

    if compressibility_table is None:
        compressibility_table = typing.cast(
            NumberArray[ThreeDimensions],
            arrays.compute_water_compressibility(
                pressure=pressure_table_3d,
                temperature=temperature_table_3d,
                bubble_point_pressure=bubble_point_pressure_table,
                gas_formation_volume_factor=gas_fvf_table_3d,  # type: ignore[arg-type]
                gas_solubility_in_water=gas_solubility_table_3d,
                gas_free_water_formation_volume_factor=gas_free_fvf_table_3d,  # type: ignore[arg-type]
                salinity=salinity_table_3d,
            ),
        )

    # Derived density: formula first
    if density_table is None:
        if stock_tank_water_density is not None:
            density_table = (stock_tank_water_density / gas_free_fvf_table_3d).astype(  # type: ignore[assignment]
                dtype
            )
        else:
            density_table = typing.cast(
                NumberArray[ThreeDimensions],
                arrays.compute_water_density(
                    pressure=pressure_table_3d,
                    temperature=temperature_table_3d,
                    gas_gravity=gas_gravity_table_3d,
                    salinity=salinity_table_3d,
                    gas_solubility_in_water=gas_solubility_table_3d,
                    gas_free_water_formation_volume_factor=gas_free_fvf_table_3d,  # type: ignore[arg-type]
                ),
            )

    if formation_volume_factor_table is None:
        assert density_table is not None
        formation_volume_factor_table = typing.cast(
            NumberArray[ThreeDimensions],
            arrays.compute_water_formation_volume_factor(
                water_density=density_table,
                salinity=salinity_table_3d,
            ),
        )

    if viscosity_table is None:
        viscosity_table = typing.cast(
            NumberArray[ThreeDimensions],
            arrays.compute_water_viscosity(
                temperature=temperature_table_3d,
                salinity=salinity_table_3d,
                pressure=pressure_table_3d,
            ),
        )

    return PVTData(
        phase=FluidPhase.WATER,
        pressures=pressures,
        temperatures=temperatures,
        salinities=salinities,
        viscosity_table=viscosity_table,
        density_table=density_table,
        formation_volume_factor_table=formation_volume_factor_table,
        compressibility_table=compressibility_table,
        bubble_point_pressure_table=bubble_point_pressure_table,
        gas_free_water_fvf_table=gas_free_water_fvf_table,
        dtype=dtype,
        unit_system=unit_system,
    )


def build_pvt_dataset(
    pressures: NumberArray[OneDimension],
    temperatures: NumberArray[OneDimension],
    oil_specific_gravity: Number = 0.85,
    gas_gravity: Number | None = None,
    water_salinity: Number | None = None,
    salinities: NumberArray[OneDimension] | None = None,
    estimatedsolution_gor: Number | None = None,
    bubble_point_pressures: NumberArray[OneDimension] | NumberArray[TwoDimensions] | None = None,
    solution_gas_to_oil_ratios: NumberArray[OneDimension] | None = None,
    gas: str | typing.Any | None = None,
    build_oil_data: bool = True,
    build_gas_data: bool = True,
    build_water_data: bool = True,
    stock_tank_oil_density: Number | None = None,
    stock_tank_gas_density: Number | None = None,
    stock_tank_water_density: Number | None = None,
    dtype: npt.DTypeLike = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
    **kwargs: typing.Any,
) -> PVTDataSet:
    """
    Build water, oil, and gas `PVTData` and return a `PVTDataSet`.

    Keyword arguments prefixed with `oil_`, `gas_`, or `water_` are
    forwarded to the appropriate phase builder after stripping the prefix.
    All other `kwargs` are forwarded to all three builders.

    :param pressures: 1-D array of pressures (psi).
    :param temperatures: 1-D array of temperatures (°F).
    :param oil_specific_gravity: Oil specific gravity (water=1).
    :param gas_gravity: Gas specific gravity (air=1).
    :param water_salinity: Single salinity scalar (ppm).
    :param salinities: 1-D salinity array (ppm).
    :param estimatedsolution_gor: Estimated Rs (SCF/STB) for bubble-point correlation.
    :param bubble_point_pressures: Pre-computed Pb array.
    :param solution_gas_to_oil_ratios: Rs axis for 2-D Pb table.
    :param gas: Gas identity (string or `Fluid`).
    :param build_oil_data: Skip oil builder when `False`.
    :param build_gas_data: Skip gas builder when `False`.
    :param build_water_data: Skip water builder when `False`.
    :param stock_tank_oil_density: Stock-tank oil density (lbm/ft³).
    :param stock_tank_gas_density: Stock-tank gas density (lbm/ft³).
    :param stock_tank_water_density: Stock-tank water density (lbm/ft³).
    :returns: `PVTDataSet`.
    """
    oil_kwargs: dict[str, typing.Any] = {}
    gas_kwargs: dict[str, typing.Any] = {}
    water_kwargs: dict[str, typing.Any] = {}
    shared_kwargs: dict[str, typing.Any] = {
        "dtype": dtype,
        "unit_system": unit_system,
    }

    for key, val in kwargs.items():
        if key.startswith("oil_"):
            oil_kwargs[key[4:]] = val
        elif key.startswith("gas_"):
            gas_kwargs[key[4:]] = val
        elif key.startswith("water_"):
            water_kwargs[key[6:]] = val
        else:
            shared_kwargs[key] = val

    oil_data = (
        build_oil_pvt_data(
            pressures=pressures,
            temperatures=temperatures,
            oil_specific_gravity=oil_specific_gravity,
            gas_gravity=gas_gravity,
            estimatedsolution_gor=estimatedsolution_gor,
            bubble_point_pressures=bubble_point_pressures,
            solution_gas_to_oil_ratios=solution_gas_to_oil_ratios,
            gas=gas,
            stock_tank_oil_density=stock_tank_oil_density,
            stock_tank_gas_density=stock_tank_gas_density,
            **{**shared_kwargs, **oil_kwargs},
        )
        if build_oil_data
        else None
    )

    gas_data = (
        build_gas_pvt_data(
            pressures=pressures,
            temperatures=temperatures,
            gas_gravity=gas_gravity,
            gas=gas,
            water_salinities=salinities,
            stock_tank_gas_density=stock_tank_gas_density,
            stock_tank_oil_density=stock_tank_oil_density,
            **{**shared_kwargs, **gas_kwargs},
        )
        if build_gas_data
        else None
    )

    water_data = (
        build_water_pvt_data(
            pressures=pressures,
            temperatures=temperatures,
            salinities=salinities,
            water_salinity=water_salinity,
            gas_gravity=gas_gravity,
            gas=gas,
            stock_tank_water_density=stock_tank_water_density,
            **{**shared_kwargs, **water_kwargs},
        )
        if build_water_data
        else None
    )
    return PVTDataSet(oil=oil_data, gas=gas_data, water=water_data)

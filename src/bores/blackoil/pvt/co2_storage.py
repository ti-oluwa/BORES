"""
PVT of a CO2 storage model with CO2 as the gas and brine as the water.

A CO2 storage deck has no PVT tables. The properties of the CO2 and of the brine are computed from
pressure, temperature and salinity, and tabulated here once on a grid so that the simulation reads
them like the tables of any other deck.

CO2 density, viscosity and compressibility come from an equation of state (CoolProp), whose
density is the Span and Wagner (1996) equation. The brine density is the Batzle and Wang (1992)
correlation and its viscosity the Batzle and Wang (1992) correlation. The solubility of CO2 in the
brine, which is only tabulated when the deck lets gas dissolve in water, is the Duan and Sun (2003)
model, and the volume the dissolved CO2 adds to the brine follows from its apparent molar volume of
Garcia (2001).
"""

import typing
import warnings

import numpy as np
import numpy.typing as npt

from bores.blackoil.pvt.data import PVTData, PVTDataSet
from bores.blackoil.pvt.static import StaticPVT
from bores.blackoil.pvt.tables import PVTTables
from bores.constants import c
from bores.correlations import (
    PropsSI,
    compute_fluid_compressibility,
    compute_fluid_compressibility_factor,
    compute_fluid_density,
    compute_fluid_viscosity,
    compute_gas_solubility_in_water_duan_sun_co2,
    compute_water_density_batzle,
    compute_water_viscosity_batzle,
)
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.types import (
    Boolean,
    CellArray,
    FloatArray,
    FluidPhase,
    Integer,
    InterpolationMethod,
    Number,
    OneDimension,
    ThreeDimensions,
    TwoDimensions,
    UnitSystem,
)

__all__ = [
    "CO2_FLUID",
    "build_co2_pressure_axis",
    "build_co2_salinity_axis",
    "build_co2_storage_tables",
    "compute_co2_dissolved_volume",
]

CO2_FLUID = "CO2"
"""The CoolProp name of the gas of a CO2 storage model."""

CRITICAL_REGION_LOWER_FRACTION = 0.7
"""Start of the denser part of the pressure axis, as a fraction of the CO2 critical pressure."""

CRITICAL_REGION_UPPER_FRACTION = 1.6
"""End of the denser part of the pressure axis, as a fraction of the CO2 critical pressure."""

WATER_COMPRESSIBILITY_PRESSURE_STEP = 1.0

CO2_MOLAR_MASS = 44.0095e-3
"""Molar mass of CO2 in kg/mol."""

CUBIC_CENTIMETERS_TO_CUBIC_METERS = 1.0e-6

POUND_PER_CUBIC_FOOT_TO_KILOGRAM_PER_CUBIC_METER = 16.01846337396014
"""Half-width in psi of the pressure interval the brine compressibility is differenced over."""


def build_co2_pressure_axis(
    pressure_range: tuple[Number, Number],
    *,
    n_base_points: Integer = 100,
    n_critical_points: Integer = 600,
    dtype: npt.DTypeLike = None,  # type: ignore[assignment]
) -> FloatArray[OneDimension]:
    """
    Returns the pressures the CO2 storage tables are built at.

    The density of CO2 changes sharply with pressure near its critical pressure when the
    temperature is close to its critical temperature, as in many storage reservoirs. The axis is
    therefore evenly spaced over the whole range and has extra points through the pressures
    around the critical pressure.

    :param pressure_range: The lowest and highest pressure, in psi.
    :param n_base_points: Points spread evenly over the whole range.
    :param n_critical_points: Extra points spread evenly around the critical pressure.
    :param dtype: Dtype of the axis. `bores.precision.get_dtype()` if not given.
    :returns: Strictly increasing pressures in psi.
    :raises ValidationError: If the range is empty or not positive.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    lowest, highest = pressure_range[0], pressure_range[1]
    if not 0.0 < lowest < highest:
        raise ValidationError(
            f"The pressure range of a CO2 storage model must satisfy 0 < lowest < highest, "
            f"but it is ({lowest:g}, {highest:g}) psi."
        )

    critical_pressure = PropsSI("PCRIT", CO2_FLUID) / c.PSI_TO_PASCAL
    critical_lowest = max(lowest, CRITICAL_REGION_LOWER_FRACTION * critical_pressure)
    critical_highest = min(highest, CRITICAL_REGION_UPPER_FRACTION * critical_pressure)
    axes = [np.linspace(lowest, highest, n_base_points)]
    if critical_lowest < critical_highest:
        axes.append(np.linspace(critical_lowest, critical_highest, n_critical_points))
    return np.unique(np.concatenate(axes)).astype(dtype, copy=False)  # type: ignore[return-value]


def build_co2_salinity_axis(
    salinity: Number | CellArray,
    *,
    spacing: Number = 10000.0,
    maximum_points: Integer = 12,
    dtype: npt.DTypeLike = None,  # type: ignore[assignment]
) -> FloatArray[OneDimension]:
    """
    Returns the salinities the brine tables are built at, covering every salinity in `salinity`.

    A single salinity, or cells that all have the same one, gives an axis of one point. Otherwise
    the axis spans the lowest to the highest salinity in steps of at most `spacing`, which keeps
    the interpolation error of the brine density below a thousandth of a percent.

    :param salinity: A salinity, or the salinity of each cell, in ppm by weight of salt.
    :param spacing: The largest step of the axis, in ppm.
    :param maximum_points: The most points the axis may have.
    :param dtype: Dtype of the axis. `bores.precision.get_dtype()` if not given.
    :returns: Strictly increasing salinities in ppm.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    values = np.asarray(salinity, dtype=np.float64).ravel()
    lowest, highest = values.min(), values.max()
    if highest - lowest < 1.0:
        return np.array([values.mean()], dtype=dtype)  # type: ignore[return-value]
    count = min(max(int(np.ceil((highest - lowest) / spacing)) + 1, 2), maximum_points)
    return np.linspace(lowest, highest, count, dtype=dtype)  # type: ignore[return-value]


def compute_co2_dissolved_volume(
    temperatures: FloatArray[OneDimension],
    *,
    co2_standard_density: Number,
) -> FloatArray[OneDimension]:
    """
    Returns the volume that dissolved CO2 adds to brine, per unit standard volume of the CO2.

    Dissolved CO2 occupies its apparent molar volume `Vφ` in the brine, from Garcia (2001),

        Vφ = 37.51 - 9.585e-2 T + 8.740e-4 T² - 5.044e-7 T³   [cm³/mol, T in °C]

    which is the volume it takes at standard conditions. A standard volume of CO2 holds `ρ / M` moles.

    :param temperatures: Temperatures in F.
    :param co2_standard_density: CO2 density at standard conditions in lbm/ft³.
    :returns: The added volume for each temperature. It has no unit.
    """
    celsius = (np.asarray(temperatures, dtype=np.float64) - 32.0) * 5.0 / 9.0
    apparent_molar_volume = (
        37.51 + celsius * (-9.585e-2 + celsius * (8.74e-4 - celsius * 5.044e-7))
    ) * CUBIC_CENTIMETERS_TO_CUBIC_METERS
    moles_per_standard_volume = (
        co2_standard_density * POUND_PER_CUBIC_FOOT_TO_KILOGRAM_PER_CUBIC_METER / CO2_MOLAR_MASS
    )
    return moles_per_standard_volume * apparent_molar_volume  # type: ignore[return-value]


def build_co2_storage_tables(
    *,
    pressures: FloatArray[OneDimension],
    temperatures: FloatArray[OneDimension],
    salinities: FloatArray[OneDimension],
    reference_salinity: Number,
    dissolved_gas: Boolean = False,
    interpolation_method: InterpolationMethod = "linear",
    validate: Boolean = True,
    warn_on_extrapolation: Boolean = False,
    dtype: npt.DTypeLike = None,  # type: ignore[assignment]
) -> tuple[StaticPVT, PVTTables]:
    """
    Returns the static properties and tables of one PVT region of a CO2 storage model, in field units.

    The gas is CO2 and the water is brine. Formation volume factors are relative to the standard
    conditions of 60 F and 1 atm. The brine is measured at one standard density, that of brine of
    the `reference_salinity`, whatever its own salinity. The formation volume factor of brine of
    another salinity is then the ratio of that density to its own, so the mass of brine in a cell
    is exact for any salinity.

    The solubility of CO2 in the brine is tabulated only if `dissolved_gas` is set, and then the
    brine also swells with the CO2 it holds, by the volume the CO2 takes up in solution.

    :param pressures: Pressure axis in psi, strictly increasing.
    :param temperatures: Temperature axis in F, strictly increasing.
    :param salinities: Salinity axis in ppm by weight of salt, strictly increasing.
    :param reference_salinity: Salinity of the brine whose density is the standard density, in ppm.
    :param dissolved_gas: Whether CO2 can dissolve in the brine. The solubility model holds from
        88 F (31 C), so the temperature axis must not go below that.
    :param interpolation_method: Interpolation method of the tables.
    :param validate: Whether the tables check their own data.
    :param warn_on_extrapolation: Whether the tables log a warning for a query outside their axes.
    :param dtype: Dtype of the tables. `bores.precision.get_dtype()` if not given.
    :returns: The static properties and the tables, in `UnitSystem.FIELD`, with a gas table, a
        water table and no oil table.
    :raises ValidationError: If `dissolved_gas` is set and a temperature is below 88 F.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    pressure_column = pressures.astype(np.float64)[:, np.newaxis]
    temperature_row = temperatures.astype(np.float64)[np.newaxis, :]
    pressure_grid = pressures.astype(np.float64)[:, np.newaxis, np.newaxis]
    temperature_grid = temperatures.astype(np.float64)[np.newaxis, :, np.newaxis]
    salinity_grid = salinities.astype(np.float64)[np.newaxis, np.newaxis, :]
    standard_pressure = c.STANDARD_PRESSURE_PSI
    standard_temperature = c.STANDARD_TEMPERATURE_FAHRENHEIT

    # CO2
    co2_standard_density = float(
        compute_fluid_density(standard_pressure, standard_temperature, CO2_FLUID)
    )
    co2_density = np.asarray(compute_fluid_density(pressure_column, temperature_row, CO2_FLUID))
    co2_formation_volume_factor = co2_standard_density / co2_density
    co2_viscosity = np.asarray(
        compute_fluid_viscosity(pressure_column, temperature_row, CO2_FLUID)
    )
    co2_compressibility_factor = np.asarray(
        compute_fluid_compressibility_factor(pressure_column, temperature_row, CO2_FLUID)
    )
    co2_compressibility = np.asarray(
        compute_fluid_compressibility(pressure_column, temperature_row, CO2_FLUID)
    )

    # Brine, over pressure, temperature and salinity
    brine_standard_density = float(
        compute_water_density_batzle(standard_pressure, standard_temperature, reference_salinity)
    )
    brine_density = np.asarray(
        compute_water_density_batzle(pressure_grid, temperature_grid, salinity_grid)
    )
    brine_formation_volume_factor = brine_standard_density / brine_density
    brine_viscosity = np.broadcast_to(
        np.asarray(compute_water_viscosity_batzle(temperature_grid, salinity_grid)),
        brine_density.shape,
    )
    step = WATER_COMPRESSIBILITY_PRESSURE_STEP
    brine_compressibility = (
        compute_water_density_batzle(pressure_grid + step, temperature_grid, salinity_grid)
        - compute_water_density_batzle(pressure_grid - step, temperature_grid, salinity_grid)
    ) / (2.0 * step * brine_density)
    reference_brine_density = np.asarray(
        compute_water_density_batzle(pressure_column, temperature_row, reference_salinity)
    )

    def cast_2d(values: typing.Any) -> FloatArray[TwoDimensions]:
        return typing.cast("FloatArray[TwoDimensions]", np.array(values, dtype=dtype))

    def cast_3d(values: typing.Any) -> FloatArray[ThreeDimensions]:
        return typing.cast("FloatArray[ThreeDimensions]", np.array(values, dtype=dtype))

    solubility_in_water_table: FloatArray[ThreeDimensions] | None = None
    dissolved_gas_volume_table: FloatArray[TwoDimensions] | None = None
    if dissolved_gas:
        solubility_in_water_table = cast_3d(
            compute_gas_solubility_in_water_duan_sun_co2(
                pressure_grid, temperature_grid, salinity_grid
            )
        )
        dissolved_gas_volume_table = cast_2d(
            np.broadcast_to(
                compute_co2_dissolved_volume(
                    temperatures.astype(np.float64), co2_standard_density=co2_standard_density
                )[np.newaxis, :],
                co2_density.shape,
            )
        )

    pressure_axis = typing.cast("FloatArray[OneDimension]", pressures.astype(dtype))
    temperature_axis = typing.cast("FloatArray[OneDimension]", temperatures.astype(dtype))
    salinity_axis = typing.cast("FloatArray[OneDimension]", salinities.astype(dtype))
    gas_data = PVTData(
        phase=FluidPhase.GAS,
        formation_volume_factor_table=cast_2d(co2_formation_volume_factor),
        viscosity_table=cast_2d(co2_viscosity),
        density_table=cast_2d(co2_density),
        compressibility_table=cast_2d(co2_compressibility),
        compressibility_factor_table=cast_2d(co2_compressibility_factor),
        solubility_in_water_table=solubility_in_water_table,
        salinities=salinity_axis if dissolved_gas else None,
        dtype=dtype,
        unit_system=UnitSystem.FIELD,
        pressures=pressure_axis,
        temperatures=temperature_axis,
    )
    water_data = PVTData(
        phase=FluidPhase.WATER,
        salinities=salinity_axis,
        formation_volume_factor_table=cast_3d(brine_formation_volume_factor),
        viscosity_table=cast_3d(brine_viscosity),
        density_table=cast_3d(brine_density),
        compressibility_table=cast_3d(brine_compressibility),
        gas_free_water_fvf_table=cast_2d(brine_standard_density / reference_brine_density),
        dissolved_gas_volume_table=dissolved_gas_volume_table,
        dtype=dtype,
        unit_system=UnitSystem.FIELD,
        pressures=pressure_axis,
        temperatures=temperature_axis,
    )

    reference_row = len(pressures) // 2
    reference_column = len(temperatures) // 2
    reference_salinity_index = int(np.argmin(np.abs(salinities - reference_salinity)))
    reference_index = (reference_row, reference_column, reference_salinity_index)
    static = StaticPVT(
        stock_tank_water_density=brine_standard_density,
        stock_tank_gas_density=co2_standard_density,
        water_reference_pressure=pressures[reference_row],
        water_reference_fvf=brine_formation_volume_factor[reference_index],
        water_reference_viscosity=brine_viscosity[reference_index],
        water_reference_compressibility=brine_compressibility[reference_index],
        water_salinity=reference_salinity,
        unit_system=UnitSystem.FIELD,
    )
    with warnings.catch_warnings():
        # CO2 at storage pressures is denser than any hydrocarbon gas, which is what this
        # warning guards against, so it does not apply here.
        warnings.filterwarnings("ignore", message="Gas density table contains values")
        tables = PVTTables.from_dataset(
            PVTDataSet(gas=gas_data, water=water_data),
            pvt=static,
            interpolation_method=interpolation_method,
            validate=bool(validate),
            warn_on_extrapolation=bool(warn_on_extrapolation),
            dtype=dtype,
        )
    return static, tables

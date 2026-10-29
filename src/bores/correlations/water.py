import logging
import typing

import numba  # type: ignore[import-untyped]
import numpy as np

from bores.constants import c
from bores.correlations.core import (
    HENRY_CONSTANTS,
    HENRY_REFERENCE_TEMPERATURE,
    SETSCHENOW_CONSTANTS,
    fahrenheit_to_celsius,
    fahrenheit_to_kelvin,
    get_gas_symbol,
)
from bores.correlations.elementwise import elementwise, resolve_result_dtype
from bores.correlations.gas import compute_co2_fugacity_coefficient
from bores.errors import ComputationError, ValidationError
from bores.types import AnyDimension, Number, NumberOrArray

logger = logging.getLogger(__name__)

__all__ = [
    "compute_gas_free_water_formation_volume_factor",
    "compute_gas_solubility_in_water",
    "compute_gas_solubility_in_water_duan_sun_co2",
    "compute_gas_solubility_in_water_henry_law",
    "compute_gas_solubility_in_water_mccain_methane",
    "compute_gas_solubility_in_water_pressure_derivative",
    "compute_salt_molality",
    "compute_water_bubble_point_pressure",
    "compute_water_compressibility",
    "compute_water_density",
    "compute_water_density_batzle",
    "compute_water_density_mccain",
    "compute_water_formation_volume_factor",
    "compute_water_mass_per_stock_tank_barrel",
    "compute_water_standard_density",
    "compute_water_vapor_pressure",
    "compute_water_viscosity",
]

# Physical constants, read from `bores.constants` when this module is imported.
BARRELS_TO_CUBIC_FEET = c.BARRELS_TO_CUBIC_FEET
BARRELS_TO_CUBIC_METER = c.BARRELS_TO_CUBIC_METER
FAHRENHEIT_TO_RANKINE_OFFSET = c.FAHRENHEIT_TO_RANKINE_OFFSET
KILOGRAM_PER_CUBIC_METER_TO_POUNDS_PER_CUBIC_FEET = (
    c.KILOGRAM_PER_CUBIC_METER_TO_POUNDS_PER_CUBIC_FEET
)
KILOGRAMS_TO_GRAMS = c.KILOGRAMS_TO_GRAMS
MOLECULAR_WEIGHT_AIR = c.MOLECULAR_WEIGHT_AIR
MOLECULAR_WEIGHT_NACL = c.MOLECULAR_WEIGHT_NACL
PASCAL_TO_BAR = c.PASCAL_TO_BAR
PASCAL_TO_MEGAPASCAL = c.PASCAL_TO_MEGAPASCAL
POUNDS_PER_CUBIC_FEET_TO_GRAMS_PER_CUBIC_CENTIMETER = (
    c.POUNDS_PER_CUBIC_FEET_TO_GRAMS_PER_CUBIC_CENTIMETER
)
POUNDS_TO_GRAMS = c.POUNDS_TO_GRAMS
PPM_TO_WEIGHT_FRACTION = c.PPM_TO_WEIGHT_FRACTION
PPM_TO_WEIGHT_PERCENT = c.PPM_TO_WEIGHT_PERCENT
PSI_TO_PASCAL = c.PSI_TO_PASCAL
SCF_PER_POUND_MOLE = c.SCF_PER_POUND_MOLE

METHANE_MCCAIN_TEMPERATURE_RANGE = (100.0, 400.0)
"""Temperature range in °F where the McCain methane solubility correlation is used."""
DUAN_SUN_TEMPERATURE_RANGE = (88.0, 500.0)
"""Temperature range in °F of the Duan and Sun CO₂ solubility model."""
DUAN_SUN_MAXIMUM_PRESSURE = 29000.0
"""Highest pressure in psi (about 2000 bar) of the Duan and Sun CO₂ solubility model."""
DUAN_SUN_MAXIMUM_SALINITY = 200000.0
"""Highest salinity in ppm (about 4.3 mol/kg NaCl) of the Duan and Sun CO₂ solubility model."""
DUAN_SUN_CHEMICAL_POTENTIAL_COEFFICIENTS = (
    28.9447706,
    -0.0354581768,
    -4770.67077,
    1.02782768e-5,
    33.8126098,
    9.04037140e-3,
    -1.14934031e-3,
    -0.307405726,
    -0.0907301486,
    9.32713393e-4,
    0.0,
)
"""Coefficients of the standard chemical potential of CO₂ in water divided by `RT`."""
DUAN_SUN_LAMBDA_COEFFICIENTS = (
    -0.411370585,
    6.07632013e-4,
    97.5347708,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.0237622469,
    0.0170656236,
    0.0,
    1.41335834e-5,
)
"""Coefficients of the CO₂-Na interaction parameter."""
DUAN_SUN_ZETA_COEFFICIENTS = (
    3.36389723e-4,
    -1.98298980e-5,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    2.12220830e-3,
    -5.24873303e-3,
    0.0,
    0.0,
)
"""Coefficients of the CO₂-Na-Cl interaction parameter."""
BUBBLE_POINT_BISECTION_ITERATIONS = 80


@elementwise(non_negative=("salinity",))
def compute_water_standard_density(
    salinity: NumberOrArray[AnyDimension],
) -> NumberOrArray[AnyDimension]:
    """
    Computes the density of brine at standard conditions (60 °F, 14.696 psi) from its
    salinity (McCain).

        ρ = 62.368 + 0.438603 * S + 1.60074e-3 * S²

    where S is the salinity in weight percent of dissolved solids.

    :param salinity: Water salinity (ppm by weight of total dissolved solids)
    :return: Brine density at standard conditions (lbm/ft³)
    """
    weight_percent = salinity * PPM_TO_WEIGHT_PERCENT
    return 62.368 + 0.438603 * weight_percent + 1.60074e-3 * weight_percent**2


@elementwise(non_negative=("salinity", "pressure"), positive=("temperature",))
def compute_water_viscosity(
    temperature: NumberOrArray[AnyDimension],
    salinity: NumberOrArray[AnyDimension],
    pressure: NumberOrArray[AnyDimension],
) -> NumberOrArray[AnyDimension]:
    """
    Computes the brine viscosity with the McCain correlation.

        μ_w1 = A * T^B
        μ_w = μ_w1 * (0.9994 + 4.0295e-5 * P + 3.1062e-9 * P²)

    where
        A = 109.574 - 0.840564 * S + 0.313314 * S² + 8.72213e-3 * S³
        B = -1.12166 + 0.0263951 * S - 6.79461e-4 * S² - 5.47119e-5 * S³ + 1.55586e-6 * S⁴

    and S is the salinity (weight percent of dissolved solids), T the temperature (°F)
    and P the pressure (psi). `μ_w1` is the viscosity at atmospheric pressure.

    Typical range: 86-390 °F, up to 26% salinity, up to 15000 psi.

    :param temperature: Temperature (°F)
    :param salinity: Water salinity (ppm by weight of total dissolved solids)
    :param pressure: Pressure (psi)
    :return: Water viscosity (cP)
    """
    s = salinity * PPM_TO_WEIGHT_PERCENT
    a = 109.574 - 0.840564 * s + 0.313314 * s**2 + 8.72213e-3 * s**3
    b = -1.12166 + 2.63951e-2 * s - 6.79461e-4 * s**2 - 5.47119e-5 * s**3 + 1.55586e-6 * s**4
    atmospheric_viscosity = a * temperature**b
    return atmospheric_viscosity * (0.9994 + 4.0295e-5 * pressure + 3.1062e-9 * pressure**2)


@elementwise(
    non_negative=("pressure",), bounds={"temperature": (-FAHRENHEIT_TO_RANKINE_OFFSET, np.inf)}
)
def compute_gas_free_water_formation_volume_factor(
    pressure: NumberOrArray[AnyDimension],
    temperature: NumberOrArray[AnyDimension],
) -> NumberOrArray[AnyDimension]:
    """
    Computes the water formation volume factor with the McCain correlation. The effect
    of salinity and of dissolved gas on the factor is small and is not modeled.

        Bw = (1 + ΔVwp) * (1 + ΔVwT)

    where
        ΔVwT = -0.010001 + 1.33391e-4 * T + 5.50654e-7 * T²
        ΔVwp = -1.95301e-9 * P * T - 1.72834e-13 * P² * T - 3.58922e-7 * P - 2.25341e-10 * P²

    with the temperature T in °F and the pressure P in psi.

    Typical range: 70-260 °F, 200-5000 psi.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :return: Water formation volume factor (bbl/STB)
    """
    delta_v_temperature = -0.010001 + 1.33391e-4 * temperature + 5.50654e-7 * temperature**2
    delta_v_pressure = (
        -1.95301e-9 * pressure * temperature
        - 1.72834e-13 * pressure**2 * temperature
        - 3.58922e-7 * pressure
        - 2.25341e-10 * pressure**2
    )
    return (1.0 + delta_v_pressure) * (1.0 + delta_v_temperature)


@elementwise(non_negative=("pressure", "salinity"), positive=("temperature",))
def compute_gas_solubility_in_water_mccain_methane(
    pressure: NumberOrArray[AnyDimension],
    temperature: NumberOrArray[AnyDimension],
    salinity: NumberOrArray[AnyDimension] = 0.0,
) -> NumberOrArray[AnyDimension]:
    """
    Computes the solubility of methane in brine with the McCain correlation.

        Rsw = (A + B * P + C * P²) * 10^(-0.0840655 * S * T^-0.285854)

    where A, B and C are cubic and quartic polynomials of the temperature T (°F),

        A = 8.15839 - 6.12265e-2 * T + 1.91663e-4 * T² - 2.1654e-7 * T³
        B = 1.01021e-2 - 7.44241e-5 * T + 3.05553e-7 * T² - 2.94883e-10 * T³
        C = (-9.02505 + 0.130237 * T - 8.53425e-4 * T² + 2.34122e-6 * T³
             - 2.37049e-9 * T⁴) * 1e-7

    and S is the salinity (weight percent of dissolved solids).

    Typical range: 100-340 °F, up to 10000 psi.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param salinity: Water salinity (ppm by weight of total dissolved solids)
    :return: Solubility of methane in water (scf/STB)
    """
    s = salinity * PPM_TO_WEIGHT_PERCENT
    a = (
        8.15839
        - 6.12265e-2 * temperature
        + 1.91663e-4 * temperature**2
        - 2.1654e-7 * temperature**3
    )
    b = (
        1.01021e-2
        - 7.44241e-5 * temperature
        + 3.05553e-7 * temperature**2
        - 2.94883e-10 * temperature**3
    )
    c_ = (
        -9.02505
        + 0.130237 * temperature
        - 8.53425e-4 * temperature**2
        + 2.34122e-6 * temperature**3
        - 2.37049e-9 * temperature**4
    ) * 1e-7
    pure_water_solubility = a + b * pressure + c_ * pressure**2
    salting_out = 10.0 ** (-0.0840655 * s * temperature**-0.285854)
    return max(pure_water_solubility * salting_out, 0.0)


@elementwise(non_negative=("salinity",))
def compute_salt_molality(salinity: NumberOrArray[AnyDimension]) -> NumberOrArray[AnyDimension]:
    """
    Computes the molality of NaCl in brine.

        m = w / (1 - w) / M

    where `w` is the salt mass fraction and `M` the molar mass of NaCl in kg/mol.

    :param salinity: Water salinity (ppm by weight of total dissolved solids)
    :return: Salt molality (mol of NaCl per kg of water)
    """
    salt_mass_fraction = salinity * PPM_TO_WEIGHT_FRACTION
    return (
        salt_mass_fraction
        / (1.0 - salt_mass_fraction)
        * KILOGRAMS_TO_GRAMS
        / MOLECULAR_WEIGHT_NACL
    )


@elementwise(non_negative=("salinity",))
def compute_water_mass_per_stock_tank_barrel(
    salinity: NumberOrArray[AnyDimension],
) -> NumberOrArray[AnyDimension]:
    """
    Computes the mass of water in one stock tank barrel of brine, which is the mass of
    the brine less the dissolved salt.

    :param salinity: Water salinity (ppm by weight of total dissolved solids)
    :return: Mass of water (kg)
    """
    brine_density = (
        compute_water_standard_density(salinity)
        / KILOGRAM_PER_CUBIC_METER_TO_POUNDS_PER_CUBIC_FEET
    )
    return BARRELS_TO_CUBIC_METER * brine_density * (1.0 - salinity * PPM_TO_WEIGHT_FRACTION)


@elementwise(bounds={"temperature": (32.0, 700.0)})
def compute_water_vapor_pressure(
    temperature: NumberOrArray[AnyDimension],
) -> NumberOrArray[AnyDimension]:
    """
    Computes the vapor pressure of water with the Wagner and Pruss (1993) equation.

        ln(Ps / Pc) = (Tc / T) * (a1 τ + a2 τ^1.5 + a3 τ³ + a4 τ^3.5 + a5 τ⁴ + a6 τ^7.5)

    where `τ = 1 - T / Tc`, `Tc = 647.096 K` and `Pc = 220.64 bar`.

    :param temperature: Temperature (°F)
    :return: Vapor pressure (psi)
    """
    critical_temperature = 647.096
    critical_pressure_bar = 220.64
    temperature_kelvin = fahrenheit_to_kelvin(temperature)
    tau = 1.0 - temperature_kelvin / critical_temperature
    exponent = (
        -7.85951783 * tau
        + 1.84408259 * tau**1.5
        - 11.7866497 * tau**3
        + 22.6807411 * tau**3.5
        - 15.9618719 * tau**4
        + 1.80122502 * tau**7.5
    )
    pressure_bar = critical_pressure_bar * np.exp(
        critical_temperature / temperature_kelvin * exponent
    )
    return pressure_bar / (PSI_TO_PASCAL * PASCAL_TO_BAR)


@numba.njit(cache=True)
def compute_duan_sun_parameter(
    coefficients: tuple[Number, ...],
    temperature_kelvin: Number,
    pressure_bar: Number,
) -> Number:
    """
    Evaluates a temperature and pressure dependent parameter of the Duan and Sun model:

        c1 + c2 T + c3/T + c4 T² + c5/(630 - T) + c6 P + c7 P ln T + c8 P/T
           + c9 P/(630 - T) + c10 P²/(630 - T)² + c11 T ln P

    :param coefficients: The eleven coefficients c1 to c11.
    :param temperature_kelvin: Temperature (K)
    :param pressure_bar: Pressure (bar)
    :return: The value of the parameter.
    """
    t = temperature_kelvin
    p = pressure_bar
    return (
        coefficients[0]
        + coefficients[1] * t
        + coefficients[2] / t
        + coefficients[3] * t**2
        + coefficients[4] / (630.0 - t)
        + coefficients[5] * p
        + coefficients[6] * p * np.log(t)
        + coefficients[7] * p / t
        + coefficients[8] * p / (630.0 - t)
        + coefficients[9] * p**2 / (630.0 - t) ** 2
        + coefficients[10] * t * np.log(p)
    )


@elementwise(
    non_negative=("salinity",),
    bounds={
        "temperature": DUAN_SUN_TEMPERATURE_RANGE,
        "pressure": (0.0, DUAN_SUN_MAXIMUM_PRESSURE),
        "salinity": (0.0, DUAN_SUN_MAXIMUM_SALINITY),
    },
)
def compute_gas_solubility_in_water_duan_sun_co2(
    pressure: NumberOrArray[AnyDimension],
    temperature: NumberOrArray[AnyDimension],
    salinity: NumberOrArray[AnyDimension] = 0.0,
) -> NumberOrArray[AnyDimension]:
    """
    Computes the solubility of CO₂ in brine with the Duan and Sun (2003) model.

    The solubility `m` in mol of CO₂ per kg of water follows from equating the
    chemical potential of CO₂ in the vapor and in the brine:

        ln m = ln(y * φ * P) - μ / RT - 2 λ m_Na - ζ m_Na m_Cl

    where `y = (P - P_H2O) / P` is the CO₂ mole fraction of the vapor, `φ` the CO₂
    fugacity coefficient (`compute_co2_fugacity_coefficient`), `μ / RT` the standard
    chemical potential, and `λ` and `ζ` the CO₂-Na and CO₂-Na-Cl interaction
    parameters. For NaCl brine `m_Na = m_Cl` is the salt molality. The solubility is
    zero at or below the vapor pressure of water.

    The solubility per stock tank barrel is `m` times the mass of water in a stock
    tank barrel (`compute_water_mass_per_stock_tank_barrel`), converted to standard
    cubic feet.

    Valid for 88 to 500 °F, up to 29000 psi (2000 bar) and up to 200000 ppm NaCl.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param salinity: Water salinity (ppm by weight of NaCl)
    :return: Solubility of CO₂ in water (scf/STB)
    """
    temperature_kelvin = fahrenheit_to_kelvin(temperature)
    pressure_bar = pressure * PSI_TO_PASCAL * PASCAL_TO_BAR
    vapor_pressure_bar = compute_water_vapor_pressure(temperature) * PSI_TO_PASCAL * PASCAL_TO_BAR
    if pressure_bar <= vapor_pressure_bar:
        return 0.0
    vapor_fraction = (pressure_bar - vapor_pressure_bar) / pressure_bar
    fugacity_coefficient = compute_co2_fugacity_coefficient(pressure, temperature)
    salt_molality = compute_salt_molality(salinity)
    ln_molality = (
        np.log(vapor_fraction * fugacity_coefficient * pressure_bar)
        - compute_duan_sun_parameter(
            DUAN_SUN_CHEMICAL_POTENTIAL_COEFFICIENTS,
            temperature_kelvin,  # type: ignore[arg-type]
            pressure_bar,
        )
        - 2.0
        * compute_duan_sun_parameter(
            DUAN_SUN_LAMBDA_COEFFICIENTS,
            temperature_kelvin,  # type: ignore[arg-type]
            pressure_bar,
        )
        * salt_molality
        - compute_duan_sun_parameter(DUAN_SUN_ZETA_COEFFICIENTS, temperature_kelvin, pressure_bar)  # type: ignore[arg-type]
        * salt_molality**2
    )
    standard_cubic_feet_per_mole = SCF_PER_POUND_MOLE / POUNDS_TO_GRAMS
    return (
        np.exp(ln_molality)
        * compute_water_mass_per_stock_tank_barrel(salinity)
        * standard_cubic_feet_per_mole
    )


@elementwise(
    non_negative=("pressure", "salinity"),
    positive=("henry_constant",),
    bounds={"temperature": (-FAHRENHEIT_TO_RANKINE_OFFSET, np.inf)},
)
def compute_gas_solubility_in_water_henry_law(
    pressure: NumberOrArray[AnyDimension],
    temperature: NumberOrArray[AnyDimension],
    salinity: NumberOrArray[AnyDimension],
    henry_constant: NumberOrArray[AnyDimension],
    henry_temperature_coefficient: NumberOrArray[AnyDimension],
    setschenow_constant: NumberOrArray[AnyDimension],
) -> NumberOrArray[AnyDimension]:
    """
    Computes the solubility of a gas in brine with Henry's law and a Setschenow
    salting-out correction.

        kH(T) = kH * exp[C * (1 / T - 1 / T_ref)]
        c = kH(T) * P                       (mol of gas per m³ of water)
        m = c / ρ_w                         (mol of gas per kg of water)
        m_brine = m * 10^(-k_s * m_salt)

    The solubility per stock tank barrel of water is `m_brine` times the mass of water
    in a stock tank barrel, converted to standard cubic feet of gas.

    The gas is treated as ideal, so the solubility is overestimated at high pressure
    and never saturates.
    Use `compute_gas_solubility_in_water` to look up the constants of a gas.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param salinity: Water salinity (ppm by weight of total dissolved solids)
    :param henry_constant: Solubility `kH` at 298.15 K, the reference temperature `T_ref` (mol/(m³·Pa))
    :param henry_temperature_coefficient: `d(ln kH) / d(1 / T)` (K)
    :param setschenow_constant: Setschenow constant of the gas for NaCl (base 10, per mol/kg)
    :return: Solubility of the gas in water (scf/STB)
    """
    temperature_kelvin = fahrenheit_to_kelvin(temperature)
    henry_at_temperature = henry_constant * np.exp(
        henry_temperature_coefficient
        * (1.0 / temperature_kelvin - 1.0 / HENRY_REFERENCE_TEMPERATURE)
    )
    concentration = henry_at_temperature * pressure * PSI_TO_PASCAL
    water_density = compute_water_density_batzle(pressure, temperature, 0.0)
    water_density_si = water_density / KILOGRAM_PER_CUBIC_METER_TO_POUNDS_PER_CUBIC_FEET
    molality = concentration / water_density_si
    salt_molality = compute_salt_molality(salinity)
    molality_in_brine = molality * 10.0 ** (-setschenow_constant * salt_molality)
    standard_cubic_feet_per_mole = SCF_PER_POUND_MOLE / POUNDS_TO_GRAMS
    return (
        molality_in_brine
        * compute_water_mass_per_stock_tank_barrel(salinity)
        * standard_cubic_feet_per_mole
    )


def compute_gas_solubility_in_water(
    pressure: NumberOrArray[AnyDimension],
    temperature: NumberOrArray[AnyDimension],
    salinity: NumberOrArray[AnyDimension] = 0.0,
    gas: str = "methane",
) -> NumberOrArray[AnyDimension]:
    """
    Computes the solubility of a gas in brine.

    Methane between 100 and 400 °F uses the McCain correlation
    (`compute_gas_solubility_in_water_mccain_methane`). CO₂ between 88 and 500 °F, up
    to 29000 psi and 200000 ppm uses the Duan and Sun model
    (`compute_gas_solubility_in_water_duan_sun_co2`). Every other case uses Henry's
    law (`compute_gas_solubility_in_water_henry_law`), for the gases CO₂, CH₄, N₂, O₂,
    Ar, He and H₂. Henry's law treats the gas as ideal, so it overestimates the
    solubility at high pressure.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param salinity: Water salinity (ppm by weight of total dissolved solids)
    :param gas: Gas name or formula (e.g., "methane", "CH4", "CO2")
    :return: Solubility of the gas in water (scf/STB)
    :raises ValidationError: If the gas is not supported.
    """
    symbol = get_gas_symbol(gas)
    if symbol not in HENRY_CONSTANTS:
        raise ValidationError(
            f"Solubility in water is not available for gas '{gas}'. "
            f"Supported gases: {sorted(HENRY_CONSTANTS)}."
        )
    henry_constant, temperature_coefficient = HENRY_CONSTANTS[symbol]
    henry = compute_gas_solubility_in_water_henry_law(
        pressure=pressure,
        temperature=temperature,
        salinity=salinity,
        henry_constant=henry_constant,
        henry_temperature_coefficient=temperature_coefficient,
        setschenow_constant=SETSCHENOW_CONSTANTS[symbol],
    )
    if symbol == "co2":
        lower, upper = DUAN_SUN_TEMPERATURE_RANGE
        in_range = (
            (np.asarray(temperature) >= lower)
            & (np.asarray(temperature) <= upper)
            & (np.asarray(pressure) <= DUAN_SUN_MAXIMUM_PRESSURE)
            & (np.asarray(salinity) <= DUAN_SUN_MAXIMUM_SALINITY)
        )
        duan_sun = compute_gas_solubility_in_water_duan_sun_co2(
            pressure=np.clip(pressure, 0.0, DUAN_SUN_MAXIMUM_PRESSURE),
            temperature=np.clip(temperature, lower, upper),
            salinity=np.clip(salinity, 0.0, DUAN_SUN_MAXIMUM_SALINITY),
        )
        result = np.where(in_range, duan_sun, henry)
        return typing.cast(NumberOrArray[AnyDimension], result if result.ndim else result[()])

    if symbol != "ch4":
        return henry
    mccain = compute_gas_solubility_in_water_mccain_methane(
        pressure=pressure, temperature=temperature, salinity=salinity
    )
    lower, upper = METHANE_MCCAIN_TEMPERATURE_RANGE
    in_range = (np.asarray(temperature) >= lower) & (np.asarray(temperature) <= upper)
    result = np.where(in_range, mccain, henry)
    return result if result.ndim else result[()]


def compute_gas_solubility_in_water_pressure_derivative(
    pressure: NumberOrArray[AnyDimension],
    temperature: NumberOrArray[AnyDimension],
    salinity: NumberOrArray[AnyDimension] = 0.0,
    gas: str = "methane",
) -> NumberOrArray[AnyDimension]:
    """
    Computes `dRsw/dP`, the derivative of the solubility of a gas in brine with respect
    to pressure, with a central difference of `compute_gas_solubility_in_water`.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param salinity: Water salinity (ppm by weight of total dissolved solids)
    :param gas: Gas name or formula (e.g., "methane", "CH4", "CO2")
    :return: Derivative of the solubility with respect to pressure (scf/STB/psi)
    """
    pressure = np.asarray(pressure)  # type: ignore[assignment]
    step = np.maximum(0.01, 1e-4 * pressure)
    upper = compute_gas_solubility_in_water(
        pressure=pressure + step, temperature=temperature, salinity=salinity, gas=gas
    )
    lower_pressure = np.maximum(pressure - step, 0.0)
    lower = compute_gas_solubility_in_water(
        pressure=lower_pressure, temperature=temperature, salinity=salinity, gas=gas
    )
    result = (upper - lower) / (pressure + step - lower_pressure)
    return typing.cast(NumberOrArray[AnyDimension], result if np.ndim(result) else result[()])


def compute_water_bubble_point_pressure(
    temperature: NumberOrArray[AnyDimension],
    gas_solubility_in_water: NumberOrArray[AnyDimension],
    salinity: NumberOrArray[AnyDimension] = 0.0,
    gas: str = "methane",
) -> NumberOrArray[AnyDimension]:
    """
    Computes the bubble point pressure of water, the pressure at which the water holds
    a given amount of dissolved gas, by inverting `compute_gas_solubility_in_water`
    with bisection.

    :param temperature: Temperature (°F)
    :param gas_solubility_in_water: Dissolved gas in the water (scf/STB)
    :param salinity: Water salinity (ppm by weight of total dissolved solids)
    :param gas: Gas name or formula (e.g., "methane", "CH4", "CO2")
    :return: Bubble point pressure (psi)
    :raises ComputationError: If the dissolved gas is outside the solubility range
        between 1 psi and the maximum valid pressure.
    """
    target, temperature_array, salinity_array = np.broadcast_arrays(
        np.asarray(gas_solubility_in_water, dtype=np.float64),
        np.asarray(temperature, dtype=np.float64),
        np.asarray(salinity, dtype=np.float64),
    )
    lower = np.full(target.shape, 1.0)
    upper = np.full(target.shape, float(c.MAXIMUM_VALID_PRESSURE))

    def solubility(pressure: np.ndarray) -> np.ndarray:
        return np.asarray(
            compute_gas_solubility_in_water(
                pressure=pressure,
                temperature=temperature_array,
                salinity=salinity_array,
                gas=gas,
            )
        )

    if np.any(target < solubility(lower)) or np.any(target > solubility(upper)):
        raise ComputationError(
            "The dissolved gas is outside the solubility range of the gas between "
            f"1 psi and {c.MAXIMUM_VALID_PRESSURE} psi, so no bubble point pressure exists."
        )
    for _ in range(BUBBLE_POINT_BISECTION_ITERATIONS):
        middle = 0.5 * (lower + upper)
        below = solubility(middle) < target
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)

    result = 0.5 * (lower + upper)
    result = result.astype(
        resolve_result_dtype(temperature, gas_solubility_in_water, salinity), copy=False
    )
    return result if result.ndim else result[()]


@elementwise(
    non_negative=("pressure", "gas_solubility_in_water", "salinity"),
    positive=("gas_formation_volume_factor",),
    bounds={"temperature": (-FAHRENHEIT_TO_RANKINE_OFFSET, np.inf)},
)
def compute_water_compressibility(
    pressure: NumberOrArray[AnyDimension],
    temperature: NumberOrArray[AnyDimension],
    bubble_point_pressure: NumberOrArray[AnyDimension],
    gas_formation_volume_factor: NumberOrArray[AnyDimension],
    gas_solubility_in_water: NumberOrArray[AnyDimension],
    gas_solubility_pressure_derivative: NumberOrArray[AnyDimension],
    salinity: NumberOrArray[AnyDimension] = 0.0,
) -> NumberOrArray[AnyDimension]:
    """
    Computes the isothermal water compressibility.

    The compressibility of the liquid (McCain), with the effect of dissolved gas
    (Dodson and Standing) and of salinity:

        cw = -(1 / Bw) * ∂Bw/∂P * (1 + 8.9e-3 * Rsw) * (1 + S * f(T))
        f(T) = -0.052 + 2.7e-4 * T - 1.14e-6 * T² + 1.121e-9 * T³

    where `Bw` is the McCain water formation volume factor
    (`compute_gas_free_water_formation_volume_factor`), `Rsw` the dissolved gas
    (scf/STB), S the salinity (weight percent) and T the temperature (°F).

    Where the water is saturated, at or below the bubble point pressure, the gas
    released as pressure falls adds a term:

        cw = cw + (Bg / Bw) * dRsw/dP / 5.615

    The water counts as undersaturated only when the pressure exceeds the bubble point
    pressure by more than a relative 1e-6.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param bubble_point_pressure: Bubble point pressure of the water (psi)
    :param gas_formation_volume_factor: Gas formation volume factor (ft³/scf)
    :param gas_solubility_in_water: Dissolved gas in the water (scf/STB)
    :param gas_solubility_pressure_derivative: `dRsw/dP` (scf/STB/psi), see
        `compute_gas_solubility_in_water_pressure_derivative`
    :param salinity: Water salinity (ppm by weight of total dissolved solids)
    :return: Water compressibility (psi⁻¹)
    """
    delta_v_temperature = -0.010001 + 1.33391e-4 * temperature + 5.50654e-7 * temperature**2
    delta_v_pressure = (
        -1.95301e-9 * pressure * temperature
        - 1.72834e-13 * pressure**2 * temperature
        - 3.58922e-7 * pressure
        - 2.25341e-10 * pressure**2
    )
    d_delta_v_pressure = (
        -1.95301e-9 * temperature
        - 2.0 * 1.72834e-13 * pressure * temperature
        - 3.58922e-7
        - 2.0 * 2.25341e-10 * pressure
    )
    water_fvf = (1.0 + delta_v_pressure) * (1.0 + delta_v_temperature)
    compressibility = -(1.0 + delta_v_temperature) * d_delta_v_pressure / water_fvf
    compressibility *= 1.0 + 8.9e-3 * gas_solubility_in_water
    weight_percent = salinity * PPM_TO_WEIGHT_PERCENT
    salinity_factor = (
        -0.052 + 2.7e-4 * temperature - 1.14e-6 * temperature**2 + 1.121e-9 * temperature**3
    )
    compressibility *= 1.0 + weight_percent * salinity_factor

    undersaturated = pressure > bubble_point_pressure * (1.0 + 1e-6)
    if not undersaturated:
        compressibility += (
            gas_formation_volume_factor
            / water_fvf
            * gas_solubility_pressure_derivative
            / BARRELS_TO_CUBIC_FEET
        )
    return max(compressibility, 0.0)


@elementwise(
    bounds={"temperature": (-FAHRENHEIT_TO_RANKINE_OFFSET, np.inf)},
    non_negative=("pressure", "salinity"),
)
def compute_water_density_mccain(
    pressure: NumberOrArray[AnyDimension],
    temperature: NumberOrArray[AnyDimension],
    salinity: NumberOrArray[AnyDimension] = 0.0,
) -> NumberOrArray[AnyDimension]:
    """
    Computes the density of dissolved-gas-free brine with the McCain method: the
    density at standard conditions divided by the formation volume factor.

        ρ_w = ρ_w,sc / Bw

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param salinity: Water salinity (ppm by weight of total dissolved solids)
    :return: Brine density (lbm/ft³)
    """
    return compute_water_standard_density(salinity) / (
        compute_gas_free_water_formation_volume_factor(pressure, temperature)
    )


@elementwise(
    non_negative=("pressure", "salinity"),
    bounds={"temperature": (-FAHRENHEIT_TO_RANKINE_OFFSET, np.inf)},
)
def compute_water_density_batzle(
    pressure: NumberOrArray[AnyDimension],
    temperature: NumberOrArray[AnyDimension],
    salinity: NumberOrArray[AnyDimension] = 0.0,
) -> NumberOrArray[AnyDimension]:
    """
    Computes the density of dissolved-gas-free brine with the Batzle and Wang (1992)
    correlation.

        ρ_w = 1 + 1e-6 * (-80 T - 3.3 T² + 0.00175 T³ + 489 P - 2 T P + 0.016 T² P
                          - 1.3e-5 T³ P - 0.333 P² - 0.002 T P²)
        ρ_b = ρ_w + S * (0.668 + 0.44 S + 1e-6 * (300 P - 2400 P S
                          + T * (80 + 3 T - 3300 S - 13 P + 47 P S)))

    in g/cm³, with the temperature T in °C, the pressure P in MPa and the salinity S
    as a weight fraction.

    Typical range: up to 350 °C, up to 100 MPa, up to 32% salinity.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param salinity: Water salinity (ppm by weight of total dissolved solids)
    :return: Brine density (lbm/ft³)
    """
    t = fahrenheit_to_celsius(temperature)
    p = pressure * PSI_TO_PASCAL * PASCAL_TO_MEGAPASCAL
    s = salinity * PPM_TO_WEIGHT_FRACTION
    pure_water = 1.0 + 1e-6 * (
        -80.0 * t
        - 3.3 * t**2
        + 0.00175 * t**3
        + 489.0 * p
        - 2.0 * t * p
        + 0.016 * t**2 * p
        - 1.3e-5 * t**3 * p
        - 0.333 * p**2
        - 0.002 * t * p**2
    )
    brine = pure_water + s * (
        0.668
        + 0.44 * s
        + 1e-6
        * (
            300.0 * p
            - 2400.0 * p * s
            + t * (80.0 + 3.0 * t - 3300.0 * s - 13.0 * p + 47.0 * p * s)
        )
    )
    return brine / POUNDS_PER_CUBIC_FEET_TO_GRAMS_PER_CUBIC_CENTIMETER


@elementwise(
    non_negative=("gas_solubility_in_water", "salinity"),
    positive=("gas_gravity", "gas_free_water_formation_volume_factor"),
)
def compute_water_density(
    gas_gravity: NumberOrArray[AnyDimension],
    salinity: NumberOrArray[AnyDimension],
    gas_solubility_in_water: NumberOrArray[AnyDimension],
    gas_free_water_formation_volume_factor: NumberOrArray[AnyDimension],
) -> NumberOrArray[AnyDimension]:
    """
    Computes the density of water holding dissolved gas, from a mass balance on one
    stock tank barrel.

        ρ_w = (m_water + m_gas) / (Bw * 5.615)

    where `m_water` is the mass of the stock tank brine
    (`compute_water_standard_density`) and `m_gas = Rsw * γ_g * 28.96 / 379.49` the mass
    of the dissolved gas, both per stock tank barrel. The volume is that of the
    dissolved-gas-free brine.

    :param gas_gravity: Gas specific gravity (air = 1.0)
    :param salinity: Water salinity (ppm by weight of total dissolved solids)
    :param gas_solubility_in_water: Dissolved gas in the water (scf/STB)
    :param gas_free_water_formation_volume_factor: Water formation volume factor (bbl/STB)
    :return: Water density (lbm/ft³)
    """
    mass_water = compute_water_standard_density(salinity) * BARRELS_TO_CUBIC_FEET
    gas_mass_per_scf = gas_gravity * MOLECULAR_WEIGHT_AIR / SCF_PER_POUND_MOLE
    mass_gas = gas_solubility_in_water * gas_mass_per_scf
    volume = gas_free_water_formation_volume_factor * BARRELS_TO_CUBIC_FEET
    return (mass_water + mass_gas) / volume


@elementwise(
    positive=("water_density",),
    non_negative=("salinity",),
)
def compute_water_formation_volume_factor(
    water_density: NumberOrArray[AnyDimension],
    salinity: NumberOrArray[AnyDimension],
    standard_density: NumberOrArray[AnyDimension] = np.nan,
) -> NumberOrArray[AnyDimension]:
    """
    Computes the water formation volume factor from the density of dissolved-gas-free
    water at reservoir conditions.

        Bw = ρ_sc / ρ_w

    The result is exact for dissolved-gas-free water. For water holding dissolved gas
    it is too small by the dissolved gas mass fraction, about 0.2%.

    :param water_density: Water density at reservoir conditions (lbm/ft³)
    :param salinity: Water salinity (ppm by weight of total dissolved solids)
    :param standard_density: Density of the water at standard conditions (lbm/ft³).
        Computed from the salinity (`compute_water_standard_density`) when omitted (NaN).
    :return: Water formation volume factor (bbl/STB)
    """
    if np.isnan(standard_density):
        standard_density = compute_water_standard_density(salinity)
    return standard_density / water_density

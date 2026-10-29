import logging
import typing
from collections.abc import Mapping

import numba  # type: ignore[import-untyped]
import numpy as np

from bores.constants import c
from bores.correlations.core import (
    compute_fluid_density,
    fahrenheit_to_kelvin,
    is_CoolProp_fluid,
)
from bores.correlations.elementwise import elementwise
from bores.errors import ValidationError
from bores.types import AnyDimension, Number, NumberOrArray

logger = logging.getLogger(__name__)

__all__ = [
    "compute_co2_fugacity_coefficient",
    "compute_gas_compressibility",
    "compute_gas_compressibility_factor",
    "compute_gas_compressibility_factor_dranchuk_abou_kassem",
    "compute_gas_compressibility_factor_hall_yarborough",
    "compute_gas_compressibility_factor_papay",
    "compute_gas_density",
    "compute_gas_formation_volume_factor",
    "compute_gas_gravity",
    "compute_gas_gravity_from_density",
    "compute_gas_molecular_weight",
    "compute_gas_pseudocritical_pressure",
    "compute_gas_pseudocritical_properties",
    "compute_gas_pseudocritical_temperature",
    "compute_gas_viscosity",
]

# Physical constants, read from `bores.constants` when this module is imported.
FAHRENHEIT_TO_RANKINE_OFFSET = c.FAHRENHEIT_TO_RANKINE_OFFSET
IDEAL_GAS_CONSTANT_IMPERIAL = c.IDEAL_GAS_CONSTANT_IMPERIAL
MOLECULAR_WEIGHT_AIR = c.MOLECULAR_WEIGHT_AIR
PASCAL_TO_BAR = c.PASCAL_TO_BAR
POUNDS_PER_CUBIC_FEET_TO_GRAMS_PER_CUBIC_METER = c.POUNDS_PER_CUBIC_FEET_TO_GRAMS_PER_CUBIC_METER
PSI_TO_PASCAL = c.PSI_TO_PASCAL
STANDARD_PRESSURE_IMPERIAL = c.STANDARD_PRESSURE_IMPERIAL
STANDARD_TEMPERATURE_RANKINE = c.STANDARD_TEMPERATURE_RANKINE


CO2_CRITICAL_TEMPERATURE = 304.1282
"""Critical temperature of CO₂ in K, as used by the Duan et al. (1992) equation of state."""
CO2_CRITICAL_PRESSURE = 73.773
"""Critical pressure of CO₂ in bar, as used by the Duan et al. (1992) equation of state."""
CO2_EQUATION_OF_STATE_COEFFICIENTS = (
    8.99288497e-2,
    -4.94783127e-1,
    4.77922245e-2,
    1.03808883e-2,
    -2.82516861e-2,
    9.49887563e-2,
    5.20600880e-4,
    -2.93540971e-4,
    -1.77265112e-3,
    -2.51101973e-5,
    8.93353441e-5,
    7.88998563e-5,
    -1.66727022e-2,
    1.398,
    2.96e-2,
)
"""Coefficients a1 to a15 of the Duan et al. (1992) equation of state for CO₂."""
CO2_FUGACITY_MINIMUM_TEMPERATURE = 88.0
"""Lowest temperature in °F of `compute_co2_fugacity_coefficient`, just above the critical temperature."""
Z_FACTOR_MINIMUM = 0.2
Z_FACTOR_MAXIMUM = 3.0
MAXIMUM_ITERATIONS = 100


def compute_gas_gravity(gas: str) -> Number:
    """
    Computes the specific gravity of a gas relative to air, both at standard conditions
    (60 °F, 14.696 psi), from an equation of state using CoolProp.

    :param gas: Gas name supported by CoolProp (e.g., "Methane", "CO2")
    :return: Gas specific gravity (air = 1.0)
    :raises ValidationError: If CoolProp does not support the gas.
    """
    if not is_CoolProp_fluid(gas):
        raise ValidationError(f"Gas '{gas}' is not supported by CoolProp.")

    gas_density = compute_fluid_density(
        c.STANDARD_PRESSURE_IMPERIAL, c.STANDARD_TEMPERATURE_IMPERIAL, fluid=gas
    )
    air_density = compute_fluid_density(
        c.STANDARD_PRESSURE_IMPERIAL, c.STANDARD_TEMPERATURE_IMPERIAL, fluid="Air"
    )
    return typing.cast(Number, gas_density / air_density)


def compute_gas_gravity_from_density(
    pressure: NumberOrArray[AnyDimension],
    temperature: NumberOrArray[AnyDimension],
    density: NumberOrArray[AnyDimension],
) -> NumberOrArray[AnyDimension]:
    """
    Computes the specific gravity of a gas from its density, relative to air at the
    same pressure and temperature (from an equation of state, using CoolProp).

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param density: Gas density at that pressure and temperature (lbm/ft³)
    :return: Gas specific gravity (air = 1.0)
    """
    air_density = compute_fluid_density(pressure, temperature, fluid="Air")
    return typing.cast(NumberOrArray[AnyDimension], density / air_density)


@elementwise(
    positive=("pressure", "gas_compressibility_factor"),
    bounds={"temperature": (-FAHRENHEIT_TO_RANKINE_OFFSET, np.inf)},
)
def compute_gas_formation_volume_factor(
    pressure: NumberOrArray[AnyDimension],
    temperature: NumberOrArray[AnyDimension],
    gas_compressibility_factor: NumberOrArray[AnyDimension],
) -> NumberOrArray[AnyDimension]:
    """
    Computes the gas formation volume factor.

        Bg = Z * T * P_sc / (P * T_sc)

    where Z is the compressibility factor, T the temperature (°R), P the pressure (psi)
    and `P_sc`, `T_sc` the standard pressure and temperature.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param gas_compressibility_factor: Gas compressibility factor Z (dimensionless)
    :return: Gas formation volume factor (ft³/scf)
    """
    temperature_rankine = temperature + FAHRENHEIT_TO_RANKINE_OFFSET
    return (
        gas_compressibility_factor
        * temperature_rankine
        * STANDARD_PRESSURE_IMPERIAL
        / (pressure * STANDARD_TEMPERATURE_RANKINE)
    )


@elementwise(positive=("gas_gravity",))
def compute_gas_molecular_weight(
    gas_gravity: NumberOrArray[AnyDimension],
) -> NumberOrArray[AnyDimension]:
    """
    Computes the apparent molecular weight of a gas from its specific gravity.

        M = γ_g * 28.96

    :param gas_gravity: Gas specific gravity (air = 1.0)
    :return: Molecular weight (lbm/lbmol)
    """
    return gas_gravity * MOLECULAR_WEIGHT_AIR


@elementwise(
    positive=("pressure", "gas_compressibility_factor", "gas_gravity"),
    bounds={"temperature": (-FAHRENHEIT_TO_RANKINE_OFFSET, np.inf)},
)
def compute_gas_density(
    pressure: NumberOrArray[AnyDimension],
    temperature: NumberOrArray[AnyDimension],
    gas_gravity: NumberOrArray[AnyDimension],
    gas_compressibility_factor: NumberOrArray[AnyDimension],
) -> NumberOrArray[AnyDimension]:
    """
    Computes the gas density from the real gas law.

        ρ_g = P * M / (Z * R * T)

    where M is the apparent molecular weight, R the gas constant (10.732
    psi·ft³/(lbmol·°R)) and T the temperature (°R).

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param gas_gravity: Gas specific gravity (air = 1.0)
    :param gas_compressibility_factor: Gas compressibility factor Z (dimensionless)
    :return: Gas density (lbm/ft³)
    """
    temperature_rankine = temperature + FAHRENHEIT_TO_RANKINE_OFFSET
    molecular_weight = gas_gravity * MOLECULAR_WEIGHT_AIR
    return (
        pressure
        * molecular_weight
        / (gas_compressibility_factor * IDEAL_GAS_CONSTANT_IMPERIAL * temperature_rankine)
    )


@elementwise(
    positive=("gas_density", "gas_molecular_weight"),
    bounds={"temperature": (-FAHRENHEIT_TO_RANKINE_OFFSET, np.inf)},
)
def compute_gas_viscosity(
    temperature: NumberOrArray[AnyDimension],
    gas_density: NumberOrArray[AnyDimension],
    gas_molecular_weight: NumberOrArray[AnyDimension],
) -> NumberOrArray[AnyDimension]:
    """
    Computes the gas viscosity with the Lee, Gonzalez and Eakin (1966) correlation.

        μ_g = 1e-4 * K * exp(X * ρ^Y)
        K = (9.4 + 0.02 * M) * T^1.5 / (209 + 19 * M + T)
        X = 3.5 + 986 / T + 0.01 * M
        Y = 2.4 - 0.2 * X

    where T is the temperature (°R), M the molecular weight (lbm/lbmol) and ρ the
    gas density (g/cm³).

    :param temperature: Temperature (°F)
    :param gas_density: Gas density (lbm/ft³)
    :param gas_molecular_weight: Gas molecular weight (lbm/lbmol)
    :return: Gas viscosity (cP)
    """
    temperature_rankine = temperature + FAHRENHEIT_TO_RANKINE_OFFSET
    density_in_grams_per_cm3 = gas_density * POUNDS_PER_CUBIC_FEET_TO_GRAMS_PER_CUBIC_METER
    k = (
        (9.4 + 0.02 * gas_molecular_weight)
        * temperature_rankine**1.5
        / (209.0 + 19.0 * gas_molecular_weight + temperature_rankine)
    )
    x = 3.5 + 986.0 / temperature_rankine + 0.01 * gas_molecular_weight
    y = 2.4 - 0.2 * x
    exponent = min(max(x * density_in_grams_per_cm3**y, -700.0), 700.0)
    return 1e-4 * k * np.exp(exponent)


def validate_contaminant_fractions(values: Mapping[str, NumberOrArray[AnyDimension]]) -> None:
    """
    Validates the mole fractions of H₂S, CO₂ and N₂ of a gas.

    :param values: Arguments of the correlation, by name.
    :raises ValidationError: If a fraction is out of range or the acid gas fraction is too large.
    """
    h2s = values["h2s_mole_fraction"]
    co2 = values["co2_mole_fraction"]
    if np.max(h2s + co2) > 0.4:
        raise ValidationError(
            "The combined H₂S and CO₂ mole fraction must not exceed 0.4 for the "
            "Wichert-Aziz correction."
        )


@numba.njit(cache=True)
def compute_gas_pseudocritical_properties(
    gas_gravity: Number,
    h2s_mole_fraction: Number = 0.0,
    co2_mole_fraction: Number = 0.0,
    n2_mole_fraction: Number = 0.0,
) -> tuple[Number, Number]:
    """
    Computes the pseudocritical pressure and temperature of a gas, for one gas.

    Sutton (1985) for the hydrocarbon gas:

        Ppc = 756.8 - 131.0 * γ_g - 3.6 * γ_g²
        Tpc = 169.2 + 349.5 * γ_g - 74.0 * γ_g²

    Wichert and Aziz (1972) for H₂S and CO₂:

        ε = 120 * (A^0.9 - A^1.6) + 15 * (B^0.5 - B^4)
        Tpc' = Tpc - ε
        Ppc' = Ppc * Tpc' / (Tpc + B * (1 - B) * ε)

    where A is the combined H₂S and CO₂ mole fraction and B the H₂S mole fraction.

    Carr, Kobayashi and Burrows (1954) for N₂, applied after the acid gas correction:

        Tpc'' = Tpc' - 250 * y_N2
        Ppc'' = Ppc' - 170 * y_N2

    :param gas_gravity: Gas specific gravity (air = 1.0). For a contaminated gas, the
        gravity of the whole mixture.
    :param h2s_mole_fraction: Mole fraction of H₂S (0 to 1)
    :param co2_mole_fraction: Mole fraction of CO₂ (0 to 1)
    :param n2_mole_fraction: Mole fraction of N₂ (0 to 1)
    :return: The pseudocritical pressure (psi) and temperature (°R).
    """
    pressure = 756.8 - 131.0 * gas_gravity - 3.6 * gas_gravity**2
    temperature = 169.2 + 349.5 * gas_gravity - 74.0 * gas_gravity**2

    acid_fraction = h2s_mole_fraction + co2_mole_fraction
    if acid_fraction > 0.0:
        epsilon = 120.0 * (acid_fraction**0.9 - acid_fraction**1.6) + 15.0 * (
            h2s_mole_fraction**0.5 - h2s_mole_fraction**4
        )
        corrected_temperature = temperature - epsilon
        pressure = (
            pressure
            * corrected_temperature
            / (temperature + h2s_mole_fraction * (1.0 - h2s_mole_fraction) * epsilon)
        )
        temperature = corrected_temperature

    pressure = pressure - 170.0 * n2_mole_fraction
    temperature = temperature - 250.0 * n2_mole_fraction
    return pressure, temperature


@elementwise(
    positive=("gas_gravity",),
    bounds={"h2s_mole_fraction": (0.0, 0.25), "co2_mole_fraction": (0.0, 0.4)},
    non_negative=("n2_mole_fraction",),
    validators=(validate_contaminant_fractions,),
)
def compute_gas_pseudocritical_pressure(
    gas_gravity: NumberOrArray[AnyDimension],
    h2s_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
    co2_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
    n2_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
) -> NumberOrArray[AnyDimension]:
    """
    Computes the pseudocritical pressure of a gas (see `compute_gas_pseudocritical_properties`).

    :param gas_gravity: Gas specific gravity (air = 1.0)
    :param h2s_mole_fraction: Mole fraction of H₂S (0 to 0.25)
    :param co2_mole_fraction: Mole fraction of CO₂ (0 to 0.4)
    :param n2_mole_fraction: Mole fraction of N₂ (0 to 1)
    :return: Pseudocritical pressure (psi)
    """
    pressure, _ = compute_gas_pseudocritical_properties(
        gas_gravity,  # type: ignore[arg-type]
        h2s_mole_fraction,  # type: ignore[arg-type]
        co2_mole_fraction,  # type: ignore[arg-type]
        n2_mole_fraction,  # type: ignore[arg-type]
    )
    return pressure


@elementwise(
    positive=("gas_gravity",),
    bounds={"h2s_mole_fraction": (0.0, 0.25), "co2_mole_fraction": (0.0, 0.4)},
    non_negative=("n2_mole_fraction",),
    validators=(validate_contaminant_fractions,),
)
def compute_gas_pseudocritical_temperature(
    gas_gravity: NumberOrArray[AnyDimension],
    h2s_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
    co2_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
    n2_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
) -> NumberOrArray[AnyDimension]:
    """
    Computes the pseudocritical temperature of a gas (see `compute_gas_pseudocritical_properties`).

    :param gas_gravity: Gas specific gravity (air = 1.0)
    :param h2s_mole_fraction: Mole fraction of H₂S (0 to 0.25)
    :param co2_mole_fraction: Mole fraction of CO₂ (0 to 0.4)
    :param n2_mole_fraction: Mole fraction of N₂ (0 to 1)
    :return: Pseudocritical temperature (°R)
    """
    _, temperature = compute_gas_pseudocritical_properties(
        gas_gravity,  # type: ignore[arg-type]
        h2s_mole_fraction,  # type: ignore[arg-type]
        co2_mole_fraction,  # type: ignore[arg-type]
        n2_mole_fraction,  # type: ignore[arg-type]
    )
    return temperature


@elementwise(
    positive=("gas_gravity",),
    non_negative=("pressure", "n2_mole_fraction"),
    bounds={
        "temperature": (-FAHRENHEIT_TO_RANKINE_OFFSET, np.inf),
        "h2s_mole_fraction": (0.0, 0.25),
        "co2_mole_fraction": (0.0, 0.4),
    },
    validators=(validate_contaminant_fractions,),
)
def compute_gas_compressibility_factor_papay(
    pressure: NumberOrArray[AnyDimension],
    temperature: NumberOrArray[AnyDimension],
    gas_gravity: NumberOrArray[AnyDimension],
    h2s_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
    co2_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
    n2_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
) -> NumberOrArray[AnyDimension]:
    """
    Computes the gas compressibility factor with the Papay correlation.

        Z = 1 - 3.52 * Pr / 10^(0.9813 * Tr) + 0.274 * Pr² / 10^(0.8157 * Tr)

    where Pr and Tr are the pseudoreduced pressure and temperature. It is explicit and
    fast, and within about 6% of the Standing-Katz chart for Pr up to 8 and
    1.2 ≤ Tr ≤ 2.4. Beyond that it drifts, so prefer the Dranchuk and Abou-Kassem
    correlation. The result is limited to 0.2-3.0.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param gas_gravity: Gas specific gravity (air = 1.0)
    :param h2s_mole_fraction: Mole fraction of H₂S (0 to 0.25)
    :param co2_mole_fraction: Mole fraction of CO₂ (0 to 0.4)
    :param n2_mole_fraction: Mole fraction of N₂ (0 to 1)
    :return: Gas compressibility factor Z (dimensionless)
    """
    pseudocritical_pressure, pseudocritical_temperature = compute_gas_pseudocritical_properties(
        gas_gravity,  # type: ignore[arg-type]
        h2s_mole_fraction,  # type: ignore[arg-type]
        co2_mole_fraction,  # type: ignore[arg-type]
        n2_mole_fraction,  # type: ignore[arg-type]
    )
    reduced_pressure = pressure / pseudocritical_pressure
    reduced_temperature = (temperature + FAHRENHEIT_TO_RANKINE_OFFSET) / pseudocritical_temperature
    z = (
        1.0
        - 3.52 * reduced_pressure / 10.0 ** (0.9813 * reduced_temperature)
        + 0.274 * reduced_pressure**2 / 10.0 ** (0.8157 * reduced_temperature)
    )
    return min(max(z, Z_FACTOR_MINIMUM), Z_FACTOR_MAXIMUM)


@elementwise(
    positive=("gas_gravity",),
    non_negative=("pressure", "n2_mole_fraction"),
    bounds={
        "temperature": (-FAHRENHEIT_TO_RANKINE_OFFSET, np.inf),
        "h2s_mole_fraction": (0.0, 0.25),
        "co2_mole_fraction": (0.0, 0.4),
    },
    validators=(validate_contaminant_fractions,),
)
def compute_gas_compressibility_factor_hall_yarborough(
    pressure: NumberOrArray[AnyDimension],
    temperature: NumberOrArray[AnyDimension],
    gas_gravity: NumberOrArray[AnyDimension],
    h2s_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
    co2_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
    n2_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
) -> NumberOrArray[AnyDimension]:
    """
    Computes the gas compressibility factor with the Hall and Yarborough (1973)
    equation of state fit to the Standing-Katz chart.

    With `t = 1 / Tr`, the reduced density `y` solves

        -A * Pr + (y + y² + y³ - y⁴) / (1 - y)³ - B * y² + C * y^D = 0

    and `Z = A * Pr / y`, where

        A = 0.06125 * t * exp(-1.2 * (1 - t)²)
        B = t * (14.76 - 9.76 * t + 4.58 * t²)
        C = t * (90.7 - 242.2 * t + 42.4 * t²)
        D = 2.18 + 2.82 * t

    The root is found with Newton's method safeguarded by bisection inside (0, 1), so
    it always converges. Valid for 1.2 ≤ Tr ≤ 3.0 and Pr ≤ 24. The result is limited
    to 0.2-3.0.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param gas_gravity: Gas specific gravity (air = 1.0)
    :param h2s_mole_fraction: Mole fraction of H₂S (0 to 0.25)
    :param co2_mole_fraction: Mole fraction of CO₂ (0 to 0.4)
    :param n2_mole_fraction: Mole fraction of N₂ (0 to 1)
    :return: Gas compressibility factor Z (dimensionless)
    """
    pseudocritical_pressure, pseudocritical_temperature = compute_gas_pseudocritical_properties(
        gas_gravity,  # type: ignore[arg-type]
        h2s_mole_fraction,  # type: ignore[arg-type]
        co2_mole_fraction,  # type: ignore[arg-type]
        n2_mole_fraction,  # type: ignore[arg-type]
    )
    reduced_pressure = pressure / pseudocritical_pressure
    if reduced_pressure < 1e-2:
        return 1.0
    t = pseudocritical_temperature / (temperature + FAHRENHEIT_TO_RANKINE_OFFSET)
    a = 0.06125 * t * np.exp(-1.2 * (1.0 - t) ** 2)
    b = t * (14.76 - 9.76 * t + 4.58 * t**2)
    c_ = t * (90.7 - 242.2 * t + 42.4 * t**2)
    d = 2.18 + 2.82 * t

    lower = 0.0
    upper = 1.0 - 1e-9
    y = min(max(a * reduced_pressure, 1e-9), 0.9)
    for _ in range(MAXIMUM_ITERATIONS):
        one_minus_y = 1.0 - y
        f = (
            -a * reduced_pressure
            + (y + y**2 + y**3 - y**4) / one_minus_y**3
            - b * y**2
            + c_ * y**d
        )
        derivative = (
            (1.0 + 4.0 * y + 4.0 * y**2 - 4.0 * y**3 + y**4) / one_minus_y**4
            - 2.0 * b * y
            + c_ * d * y ** (d - 1.0)
        )
        if f > 0.0:
            upper = y
        else:
            lower = y
        y_next = y - f / derivative if derivative > 1e-30 else 0.5 * (lower + upper)
        if y_next <= lower or y_next >= upper:
            y_next = 0.5 * (lower + upper)
        converged = abs(y_next - y) <= 1e-13 * max(1.0, abs(y_next))
        y = y_next
        if converged:
            break
    z = a * reduced_pressure / y
    return min(max(z, Z_FACTOR_MINIMUM), Z_FACTOR_MAXIMUM)


@elementwise(
    positive=("gas_gravity",),
    non_negative=("pressure", "n2_mole_fraction"),
    bounds={
        "temperature": (-FAHRENHEIT_TO_RANKINE_OFFSET, np.inf),
        "h2s_mole_fraction": (0.0, 0.25),
        "co2_mole_fraction": (0.0, 0.4),
    },
    validators=(validate_contaminant_fractions,),
)
def compute_gas_compressibility_factor_dranchuk_abou_kassem(
    pressure: NumberOrArray[AnyDimension],
    temperature: NumberOrArray[AnyDimension],
    gas_gravity: NumberOrArray[AnyDimension],
    h2s_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
    co2_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
    n2_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
) -> NumberOrArray[AnyDimension]:
    """
    Computes the gas compressibility factor with the Dranchuk and Abou-Kassem (1975)
    equation of state fit to the Standing-Katz chart.

    With the reduced density `ρr = 0.27 * Pr / (Z * Tr)`, the equation of state is

        Z = 1 + (A1 + A2/Tr + A3/Tr³ + A4/Tr⁴ + A5/Tr⁵) * ρr
              + (A6 + A7/Tr + A8/Tr²) * ρr²
              - A9 * (A7/Tr + A8/Tr²) * ρr⁵
              + A10 * (1 + A11 * ρr²) * (ρr² / Tr³) * exp(-A11 * ρr²)

    Substituting `Z` gives a single equation in `ρr`, which is solved with Newton's
    method safeguarded by bisection, so it always converges. Valid for 1.0 < Tr ≤ 3.0
    and 0.2 ≤ Pr < 30. The result is limited to 0.2-3.0.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param gas_gravity: Gas specific gravity (air = 1.0)
    :param h2s_mole_fraction: Mole fraction of H₂S (0 to 0.25)
    :param co2_mole_fraction: Mole fraction of CO₂ (0 to 0.4)
    :param n2_mole_fraction: Mole fraction of N₂ (0 to 1)
    :return: Gas compressibility factor Z (dimensionless)
    """
    pseudocritical_pressure, pseudocritical_temperature = compute_gas_pseudocritical_properties(
        gas_gravity,  # type: ignore[arg-type]
        h2s_mole_fraction,  # type: ignore[arg-type]
        co2_mole_fraction,  # type: ignore[arg-type]
        n2_mole_fraction,  # type: ignore[arg-type]
    )
    reduced_pressure = pressure / pseudocritical_pressure
    if reduced_pressure < 1e-2:
        return 1.0
    reduced_temperature = (temperature + FAHRENHEIT_TO_RANKINE_OFFSET) / pseudocritical_temperature
    t = 1.0 / reduced_temperature

    a1 = 0.3265
    a2 = -1.0700
    a3 = -0.5339
    a4 = 0.01569
    a5 = -0.05165
    a6 = 0.5475
    a7 = -0.7361
    a8 = 0.1844
    a9 = 0.1056
    a10 = 0.6134
    a11 = 0.7210

    linear = a1 + a2 * t + a3 * t**3 + a4 * t**4 + a5 * t**5
    quadratic = a6 + a7 * t + a8 * t**2
    quintic = a9 * (a7 * t + a8 * t**2)
    exponential_coefficient = a10 * t**3
    target = 0.27 * reduced_pressure * t

    lower = 0.0
    upper = 5.0
    for _ in range(8):
        r2 = upper * upper
        value = (
            upper
            + linear * r2
            + quadratic * r2 * upper
            - quintic * r2**3
            + exponential_coefficient * r2 * upper * (1.0 + a11 * r2) * np.exp(-a11 * r2)
            - target
        )
        if value > 0.0:
            break
        upper *= 2.0

    density = min(max(target, 1e-9), 0.5 * upper)
    for _ in range(MAXIMUM_ITERATIONS):
        r2 = density * density
        exponential = np.exp(-a11 * r2)
        f = (
            density
            + linear * r2
            + quadratic * r2 * density
            - quintic * r2**3
            + exponential_coefficient * r2 * density * (1.0 + a11 * r2) * exponential
            - target
        )
        derivative = (
            1.0
            + 2.0 * linear * density
            + 3.0 * quadratic * r2
            - 6.0 * quintic * r2**2 * density
            + exponential_coefficient
            * exponential
            * (3.0 * r2 + 3.0 * a11 * r2**2 - 2.0 * a11**2 * r2**3)
        )
        if f > 0.0:
            upper = density
        else:
            lower = density
        density_next = density - f / derivative if derivative > 1e-30 else 0.5 * (lower + upper)
        if density_next <= lower or density_next >= upper:
            density_next = 0.5 * (lower + upper)
        converged = abs(density_next - density) <= 1e-13 * max(1.0, abs(density_next))
        density = density_next
        if converged:
            break
    z = target / density
    return min(max(z, Z_FACTOR_MINIMUM), Z_FACTOR_MAXIMUM)


def compute_gas_compressibility_factor(
    pressure: NumberOrArray[AnyDimension],
    temperature: NumberOrArray[AnyDimension],
    gas_gravity: NumberOrArray[AnyDimension],
    h2s_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
    co2_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
    n2_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
    method: typing.Literal["dak", "hall-yarborough", "papay"] = "dak",
) -> NumberOrArray[AnyDimension]:
    """
    Computes the gas compressibility factor with the chosen correlation.

    - "dak": Dranchuk and Abou-Kassem (1975), the default and most accurate.
    - "hall-yarborough": Hall and Yarborough (1973).
    - "papay": Papay, explicit and fastest, least accurate.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param gas_gravity: Gas specific gravity (air = 1.0)
    :param h2s_mole_fraction: Mole fraction of H₂S (0 to 0.25)
    :param co2_mole_fraction: Mole fraction of CO₂ (0 to 0.4)
    :param n2_mole_fraction: Mole fraction of N₂ (0 to 1)
    :param method: Correlation to use.
    :return: Gas compressibility factor Z (dimensionless)
    :raises ValidationError: If the method is not recognized.
    """
    if method == "dak":
        correlation = compute_gas_compressibility_factor_dranchuk_abou_kassem
    elif method == "hall-yarborough":
        correlation = compute_gas_compressibility_factor_hall_yarborough
    elif method == "papay":
        correlation = compute_gas_compressibility_factor_papay
    else:
        raise ValidationError(
            f"Unknown method {method!r}. Use 'dak', 'hall-yarborough' or 'papay'."
        )
    return correlation(
        pressure=pressure,
        temperature=temperature,
        gas_gravity=gas_gravity,
        h2s_mole_fraction=h2s_mole_fraction,
        co2_mole_fraction=co2_mole_fraction,
        n2_mole_fraction=n2_mole_fraction,
    )


@elementwise(
    positive=("pressure", "gas_gravity"),
    bounds={
        "temperature": (-FAHRENHEIT_TO_RANKINE_OFFSET, np.inf),
        "h2s_mole_fraction": (0.0, 0.25),
        "co2_mole_fraction": (0.0, 0.4),
    },
    non_negative=("n2_mole_fraction",),
    validators=(validate_contaminant_fractions,),
)
def compute_gas_compressibility(
    pressure: NumberOrArray[AnyDimension],
    temperature: NumberOrArray[AnyDimension],
    gas_gravity: NumberOrArray[AnyDimension],
    gas_compressibility_factor: NumberOrArray[AnyDimension] = np.nan,
    h2s_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
    co2_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
    n2_mole_fraction: NumberOrArray[AnyDimension] = 0.0,
) -> NumberOrArray[AnyDimension]:
    """
    Computes the isothermal gas compressibility.

        cg = 1 / P - (1 / Z) * dZ/dP

    where `dZ/dP` is a central difference of the Dranchuk and Abou-Kassem
    compressibility factor over a small pressure step.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param gas_gravity: Gas specific gravity (air = 1.0)
    :param gas_compressibility_factor: Gas compressibility factor Z at the pressure.
        Computed with the Dranchuk and Abou-Kassem correlation when omitted (NaN).
    :param h2s_mole_fraction: Mole fraction of H₂S (0 to 0.25)
    :param co2_mole_fraction: Mole fraction of CO₂ (0 to 0.4)
    :param n2_mole_fraction: Mole fraction of N₂ (0 to 1)
    :return: Gas compressibility (psi⁻¹), not less than zero
    """
    z = gas_compressibility_factor
    if np.isnan(z):
        z = compute_gas_compressibility_factor_dranchuk_abou_kassem(
            pressure,
            temperature,
            gas_gravity,
            h2s_mole_fraction,
            co2_mole_fraction,
            n2_mole_fraction,
        )
    delta_p = max(0.01, 1e-4 * pressure)
    z_plus = compute_gas_compressibility_factor_dranchuk_abou_kassem(
        pressure + delta_p,
        temperature,
        gas_gravity,
        h2s_mole_fraction,
        co2_mole_fraction,
        n2_mole_fraction,
    )
    if pressure - delta_p > 0.0:
        z_minus = compute_gas_compressibility_factor_dranchuk_abou_kassem(
            pressure - delta_p,
            temperature,
            gas_gravity,
            h2s_mole_fraction,
            co2_mole_fraction,
            n2_mole_fraction,
        )
        dz_dp = (z_plus - z_minus) / (2.0 * delta_p)
    else:
        dz_dp = (z_plus - z) / delta_p
    return max(1.0 / pressure - dz_dp / z, 0.0)  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_co2_equation_of_state_factor(
    reduced_volume: Number,
    reduced_temperature: Number,
) -> tuple[Number, Number, Number, Number, Number]:
    """
    Evaluates the Duan et al. (1992) equation of state for CO₂ at a reduced volume.

        Z = 1 + B/Vr + C/Vr² + D/Vr⁴ + E/Vr⁵ + F/Vr² * (β + γ/Vr²) * exp(-γ/Vr²)

    where B, C, D and E are `a + b/Tr² + c/Tr³` in the coefficients a1 to a12,
    `F = a13 / Tr³`, `β = a14` and `γ = a15`.

    :param reduced_volume: Volume divided by the critical volume.
    :param reduced_temperature: Temperature divided by the critical temperature.
    :return: The compressibility factor Z and the coefficients B, C, D and E.
    """
    a = CO2_EQUATION_OF_STATE_COEFFICIENTS
    tr2 = reduced_temperature**2
    tr3 = reduced_temperature**3
    b = a[0] + a[1] / tr2 + a[2] / tr3
    c_ = a[3] + a[4] / tr2 + a[5] / tr3
    d = a[6] + a[7] / tr2 + a[8] / tr3
    e = a[9] + a[10] / tr2 + a[11] / tr3
    f = a[12] / tr3
    v2 = reduced_volume**2
    z = (
        1.0
        + b / reduced_volume
        + c_ / v2
        + d / v2**2
        + e / (v2**2 * reduced_volume)
        + f / v2 * (a[13] + a[14] / v2) * np.exp(-a[14] / v2)
    )
    return z, b, c_, d, e


@elementwise(
    non_negative=("pressure",),
    bounds={"temperature": (CO2_FUGACITY_MINIMUM_TEMPERATURE, np.inf)},
)
def compute_co2_fugacity_coefficient(
    pressure: NumberOrArray[AnyDimension], temperature: NumberOrArray
) -> NumberOrArray[AnyDimension]:
    """
    Computes the fugacity coefficient of pure CO₂ with the Duan et al. (1992) equation of
    state, valid up to 8000 bar.

    The reduced volume is solved from `Z = Pr * Vr / Tr` by bisection, which is
    unique above the critical temperature, and the fugacity coefficient follows from
    the equation of state:

        ln φ = Z - 1 - ln Z + B/Vr + C/(2 Vr²) + D/(4 Vr⁴) + E/(5 Vr⁵)
               + F / (2γ) * [β + 1 - (β + 1 + γ/Vr²) * exp(-γ/Vr²)]

    The result approaches 1 as the pressure approaches zero.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F). Must be above the critical temperature of CO₂.
    :return: Fugacity coefficient (dimensionless)
    """
    if pressure <= 0.0:
        return 1.0
    a = CO2_EQUATION_OF_STATE_COEFFICIENTS
    reduced_temperature = (
        fahrenheit_to_kelvin(temperature) / CO2_CRITICAL_TEMPERATURE  # type: ignore[arg-type]
    )
    reduced_pressure = pressure * PSI_TO_PASCAL * PASCAL_TO_BAR / CO2_CRITICAL_PRESSURE

    lower = 0.05
    for _ in range(20):
        z_lower, _, _, _, _ = compute_co2_equation_of_state_factor(lower, reduced_temperature)  # type: ignore[arg-type]
        if reduced_pressure * lower / reduced_temperature - z_lower < 0.0:
            break
        lower *= 0.5
    upper = 200.0 * reduced_temperature / reduced_pressure
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        z_middle, _, _, _, _ = compute_co2_equation_of_state_factor(middle, reduced_temperature)  # type: ignore[arg-type]
        if reduced_pressure * middle / reduced_temperature - z_middle < 0.0:
            lower = middle
        else:
            upper = middle
    volume = 0.5 * (lower + upper)

    z, b, c_, d, e = compute_co2_equation_of_state_factor(volume, reduced_temperature)  # type: ignore[arg-type]
    f = a[12] / reduced_temperature**3
    beta = a[13]
    gamma = a[14]
    v2 = volume**2
    ln_phi = (
        z
        - 1.0
        - np.log(z)
        + b / volume
        + c_ / (2.0 * v2)
        + d / (4.0 * v2**2)
        + e / (5.0 * v2**2 * volume)
        + f / (2.0 * gamma) * (beta + 1.0 - (beta + 1.0 + gamma / v2) * np.exp(-gamma / v2))
    )
    return np.exp(ln_phi)

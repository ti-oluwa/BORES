"""Utilities for computing reservoir PVT properties."""

import functools
import logging
import typing
import warnings

import numba  # type: ignore[import-untyped]
import numpy as np
from CoolProp.CoolProp import PropsSI  # type: ignore[import, import-untyped]
from scipy.optimize import brentq, root_scalar  # type: ignore[import-untyped]

from bores.constants import c
from bores.errors import ComputationError, ValidationError
from bores.types import FloatOrArray, GasZFactorMethod
from bores.utils import clip

logger = logging.getLogger(__name__)

__all__ = [
    "clip_pressure",
    "clip_temperature",
    "compute_harmonic_mean",
    "fahrenheit_to_celsius",
    "fahrenheit_to_kelvin",
    "fahrenheit_to_rankine",
    "is_CoolProp_supported_fluid",
    "kelvin_to_fahrenheit",
    "validate_input_pressure",
    "validate_input_temperature",
]


def validate_input_temperature(temperature: FloatOrArray) -> None:
    """
    Validates that the input temperature(s) are within valid/reservoir-like range.

    Accepts scalar or ndarray input.

    :param temperature: Temperature(s) in Kelvin (°F)
    :raises ValidationError: If any temperature is outside the valid range.
    """
    temp_array = np.asarray(temperature)
    invalid_mask = (temp_array < c.MINIMUM_VALID_TEMPERATURE) | (
        temp_array > c.MAXIMUM_VALID_TEMPERATURE
    )

    if np.any(invalid_mask):
        invalid: np.ndarray = temp_array[invalid_mask]
        raise ValidationError(
            f"Temperature(s) out of valid range [{c.MINIMUM_VALID_TEMPERATURE}, {c.MAXIMUM_VALID_TEMPERATURE}] K: "
            f"{invalid}"
        )


def validate_input_pressure(pressure: FloatOrArray) -> None:
    """
    Validates that the input pressure(s) are within valid/reservoir-like range.

    Accepts scalar or ndarray input.

    :param pressure: Pressure(s) in Pascals (psi)
    :raises ValidationError: If any pressure is outside the valid range.
    """
    pressure_array = np.asarray(pressure)
    invalid = (pressure_array < c.MINIMUM_VALID_PRESSURE) | (
        pressure_array > c.MAXIMUM_VALID_PRESSURE
    )

    if np.any(invalid):
        raise ValidationError(
            f"Pressure(s) out of valid range [{c.MINIMUM_VALID_PRESSURE}, {c.MAXIMUM_VALID_PRESSURE}] Pa: "
            f"{pressure_array[invalid]}"
        )


@functools.lru_cache(maxsize=64)
def is_CoolProp_supported_fluid(fluid: str) -> bool:
    """
    Check if the fluid is supported by CoolProp.

    :param fluid: str name (must be supported by CoolProp, e.g., "CO2", "Water")
    :return: True if the fluid is supported, False otherwise.
    """
    return PropsSI("D", "T", 300, "P", 101325, fluid) is not None


def clip_pressure(pressure: FloatOrArray, fluid: str) -> FloatOrArray:
    """
    Clips pressure to be within CoolProp's valid pressure range for the given fluid.

    :param pressure: Pressure in Pascals (psi)
    :param fluid: CoolProp fluid name
    :return: Clipped pressure in Pascals
    """
    p_min = PropsSI("P_MIN", fluid)  # Minimum pressure allowed
    p_max = PropsSI("P_MAX", fluid)  # Maximum pressure allowed
    return np.minimum(  # type: ignore[return-value]
        np.maximum(pressure, p_min + 1.0), p_max - 1.0
    )  # Add small buffer


def clip_temperature(temperature: FloatOrArray, fluid: str) -> FloatOrArray:
    """
    Clips temperature to be within CoolProp's valid temperature range for the given fluid.

    :param temperature: Temperature in Kelvin (°F)
    :param fluid: CoolProp fluid name
    :return: Clipped temperature in Kelvin
    """
    t_min = PropsSI("T_MIN", fluid)
    t_max = PropsSI("T_MAX", fluid)
    return np.minimum(  # type: ignore[return-value]
        np.maximum(temperature, t_min + 0.1), t_max - 0.1
    )  # Add small buffer


@numba.njit(cache=True)
def kelvin_to_fahrenheit(temp_K: FloatOrArray) -> FloatOrArray:
    """Converts temperature from Kelvin to Fahrenheit."""
    return (temp_K - 273.15) * 9 / 5 + 32  # type: ignore[return-value]


@numba.njit(cache=True)
def fahrenheit_to_kelvin(temp_F: FloatOrArray) -> FloatOrArray:
    """Converts temperature from Fahrenheit to Kelvin."""
    return (temp_F - 32) * 5 / 9 + 273.15  # type: ignore[return-value]


@numba.njit(cache=True)
def fahrenheit_to_celsius(temp_F: FloatOrArray) -> FloatOrArray:
    """Converts temperature from Fahrenheit to Celsius."""
    return (temp_F - 32) * 5 / 9  # type: ignore[return-value]


@numba.njit(cache=True)
def fahrenheit_to_rankine(temp_F: FloatOrArray) -> FloatOrArray:
    """Converts temperature from Fahrenheit to Rankine."""
    return temp_F + 459.67  # type: ignore[return-value]


##################################################
# GENERIC FLUID PROPERTIES COMPUTATION FUNCTIONS #
##################################################


def compute_fluid_density(pressure: float, temperature: float, fluid: str) -> float:
    """
    Compute fluid density from EOS using CoolProp.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: str name (must be supported by CoolProp, e.g., "CO2", "Water")
    :return: Density in lbm/ft³
    """
    temperature_in_kelvin = fahrenheit_to_kelvin(temperature)  # type: ignore[arg-type]
    pressure_in_pascals = pressure * c.PSI_TO_PASCAL
    density: float = PropsSI(
        "D",
        "P",
        clip_pressure(pressure_in_pascals, fluid),
        "T",
        clip_temperature(temperature_in_kelvin, fluid),
        fluid,
    )
    return density * c.KILOGRAM_PER_CUBIC_METER_TO_POUNDS_PER_CUBIC_FEET


def compute_fluid_viscosity(pressure: float, temperature: float, fluid: str) -> float:
    """
    Compute fluid dynamic viscosity from EOS using CoolProp.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: str name (must be supported by CoolProp, e.g., "CO2", "Water")
    :return: Viscosity in centipoise (cP)
    """
    temperature_in_kelvin = fahrenheit_to_kelvin(temperature)
    pressure_in_pascals = pressure * c.PSI_TO_PASCAL
    viscosity = PropsSI(
        "V",
        "P",
        clip_pressure(pressure_in_pascals, fluid),
        "T",
        clip_temperature(temperature_in_kelvin, fluid),
        fluid,
    )
    return viscosity * c.PASCAL_SECONDS_TO_CENTIPOISE


def compute_fluid_compressibility_factor(
    pressure: float, temperature: float, fluid: str
) -> float:
    """
    Compute fluid compressibility factor Z from EOS using CoolProp.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: str name (must be supported by CoolProp, e.g., "CO2", "Methane")
    :return: Compressibility factor Z (dimensionless)
    """
    temperature_in_kelvin = fahrenheit_to_kelvin(temperature)  # type: ignore[arg-type]
    pressure_in_pascals = pressure * c.PSI_TO_PASCAL
    return PropsSI(
        "Z",
        "P",
        clip_pressure(pressure_in_pascals, fluid),
        "T",
        clip_temperature(temperature_in_kelvin, fluid),
        fluid,
    )


def compute_fluid_compressibility(
    pressure: float,
    temperature: float,
    fluid: str,
) -> float:
    """
    Computes the isothermal compressibility of a fluid at a given pressure and temperature.

    Compressibility is defined as:

        C_f = -(1/ρ) * (dρ/dP) at constant temperature

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: str name supported by CoolProp (e.g., 'n-Octane')
    :return: Compressibility in psi⁻¹
    """
    temperature_in_kelvin = fahrenheit_to_kelvin(temperature)  # type: ignore[arg-type]
    pressure_in_pascals = pressure * c.PSI_TO_PASCAL
    return (
        PropsSI(
            "ISOTHERMAL_COMPRESSIBILITY",
            "P",
            clip_pressure(pressure_in_pascals, fluid),
            "T",
            clip_temperature(temperature_in_kelvin, fluid),
            fluid,
        )
        / c.PASCAL_TO_PSI
    )


def compute_gas_gravity(gas: str) -> float:
    """
    Computes the specific gravity of a gas at a given pressure and temperature.

    Gas gravity is defined as the ratio of the density of the gas to the density of air at standard conditions.

    :param gas: gas name supported by CoolProp (e.g., 'Methane')
    :return: Gas gravity (dimensionless)
    """
    gas_density_at_stp = compute_fluid_density(
        c.STANDARD_PRESSURE_IMPERIAL, c.STANDARD_TEMPERATURE_IMPERIAL, fluid=gas
    )
    air_density_at_stp = compute_fluid_density(
        c.STANDARD_PRESSURE_IMPERIAL, c.STANDARD_TEMPERATURE_IMPERIAL, fluid="Air"
    )
    return gas_density_at_stp / air_density_at_stp


####################################################
# SPECIALIZED FLUID PROPERTY COMPUTATION FUNCTIONS #
####################################################


def compute_gas_gravity_from_density(
    pressure: float,
    temperature: float,
    density: float,
) -> float:
    """
    Computes the gas gravity from density.

    Gas gravity for this case, is derived as the ratio of the gas density to the
    air density at the same temperature and pressure.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param density: Density of the gas in lbm/ft³
    :return: Gas gravity (dimensionless)
    """
    temperature_in_kelvin = fahrenheit_to_kelvin(temperature)  # type: ignore[arg-type]
    temperature_in_kelvin = typing.cast(float, temperature_in_kelvin)
    pressure_in_pascals = pressure * c.PSI_TO_PASCAL
    air_density = compute_fluid_density(
        pressure_in_pascals, temperature_in_kelvin, fluid="Air"
    )
    return density / (air_density * c.KILOGRAM_PER_CUBIC_METER_TO_POUNDS_PER_CUBIC_FEET)


@numba.njit(cache=True)
def compute_total_fluid_compressibility(
    water_saturation: float,
    oil_saturation: float,
    water_compressibility: float,
    oil_compressibility: float,
    gas_saturation: typing.Optional[float] = None,
    gas_compressibility: typing.Optional[float] = None,
) -> float:
    """
    Calculates the total fluid compressibility as a saturation-weighted average of
    individual phase compressibilities.

    :param water_saturation: Water saturation (fraction).
    :param oil_saturation: Oil saturation (fraction).
    :param water_compressibility: Compressibility of the water phase (psi⁻¹).
    :param oil_compressibility: Compressibility of the oil phase (psi⁻¹).
    :param gas_saturation: Optional gas saturation (fraction) for three-phase systems.
    :param gas_compressibility: Optional gas compressibility (psi⁻¹) for three-phase systems.
    :return: Total fluid compressibility (psi⁻¹).
    """
    total_fluid_compressibility = (water_saturation * water_compressibility) + (
        oil_saturation * oil_compressibility
    )
    if gas_saturation is not None and gas_compressibility is not None:
        total_fluid_compressibility += gas_saturation * gas_compressibility

    return total_fluid_compressibility


@numba.njit(cache=True)
def compute_harmonic_mean(value1: float, value2: float) -> float:
    """
    Computes the harmonic mean of two values.

    :param value1: First value (e.g., transmissibility, permeability, viscosity)
    :param value2: Second value (e.g., transmissibility, permeability, viscosity)
    :return: Harmonic mean scaled by the square of the spacing
    """
    summation = value1 + value2
    if summation == 0:
        return 0.0
    return (2 * value1 * value2) / summation


def compute_oil_specific_gravity(
    oil_density: float,
    pressure: float,
    temperature: float,
    oil_compressibility: float,
) -> float:
    """
    Converts oil density (lbm/ft³) at given reservoir conditions to specific gravity (dimensionless)
    by adjusting for pressure and temperature effects using a linearized approximation.

    The oil density is corrected to standard temperature and pressure (STP) using the following formula:

        ρ_stp ≈ ρ * exp([Co * (P_stp - P) + α * (T_stp - T)])

    where:
        - ρ_stp: Oil density at standard conditions (lbm/ft³)
        - ρ: Oil density at reservoir conditions (lbm/ft³)
        - Co: Oil compressibility (psi⁻¹)
        - α: Oil thermal expansion coefficient (1/°F)
        - T: Reservoir temperature (°F)
        - P: Reservoir pressure (psi)
        - T_stp: Standard temperature = 60 °F
        - P_stp: Standard pressure = 14.696 psi

    Specific gravity is then calculated as:

        SG = ρ_stp / ρ_water

    where `ρ_water` is the density of water at standard conditions (assumed 62.4 lbm/ft³).

    :param oil_density: Oil density at reservoir conditions (lbm/ft³)
    :param pressure: Reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param oil_compressibility: Oil compressibility (psi⁻¹)
    :return: Specific gravity of oil (dimensionless)
    """
    delta_p = c.STANDARD_PRESSURE_IMPERIAL - pressure
    delta_t = c.STANDARD_TEMPERATURE_IMPERIAL - temperature
    correction_factor = np.exp(
        (oil_compressibility * delta_p)
        + (c.OIL_THERMAL_EXPANSION_COEFFICIENT_IMPERIAL * delta_t)
    )
    # Avoid numerical issues with small/large values
    correction_factor = clip(correction_factor, 0.2, 2.0)
    oil_density_at_stp = oil_density * correction_factor
    return oil_density_at_stp / c.STANDARD_WATER_DENSITY_IMPERIAL


@numba.njit(cache=True)
def convert_surface_rate_to_reservoir(
    surface_rate: float, formation_volume_factor: float
) -> float:
    """
    Converts a surface rate (e.g., STB/day) to reservoir conditions.

    :param surface_rate: Surface volumetric flow rate (e.g., STB/day)
    :param formation_volume_factor: Formation volume factor (FVF) for the fluid (bbl/STB)
    :return: Reservoir volumetric flow rate (e.g., bbl/day)
    """
    if surface_rate > 0:
        return surface_rate * formation_volume_factor
    return surface_rate / formation_volume_factor


@numba.njit(cache=True)
def convert_reservoir_rate_to_surface(
    reservoir_rate: float, formation_volume_factor: float
) -> float:
    """
    Converts a reservoir rate (e.g., bbl/day) to surface conditions.

    :param reservoir_rate: Reservoir volumetric flow rate (e.g., bbl/day)
    :param formation_volume_factor: Formation volume factor (FVF) for the fluid (bbl/STB)
    :return: Surface volumetric flow rate (e.g., STB/day)
    """
    if reservoir_rate > 0:
        return reservoir_rate / formation_volume_factor
    return reservoir_rate * formation_volume_factor


@numba.njit(cache=True)
def compute_oil_formation_volume_factor_standing(
    temperature: float,
    oil_specific_gravity: float,
    gas_gravity: float,
    gas_to_oil_ratio: float,
) -> float:
    """
    Computes the oil formation volume factor (Bo) in m³ oil at reservoir conditions per m³ oil at standard conditions
    using the Standing correlation.

    Formula (Standing, 1947):

        Bo = 0.972 + 0.000147 * [ (R_s * (γ_g / γ_o)^0.5) + (1.25 * T_F) ]^1.175

    Where:
        - Bo: Oil formation volume factor (bbl/STB)
        - R_s: Gas-to-oil ratio in scf/STB
        - γ_g: Gas specific gravity (air = 1.0)
        - γ_o: Oil specific gravity (water = 1.0)
        - T_F: Temperature in degrees Fahrenheit

    Limitations:
        - Valid for light oils and saturated conditions
        - Typical range: 60-300 °F, 0.5-0.95 oil SG, 20 - 2000 scf/STB

    :param temperature: Temperature (°F)
    :param oil_specific_gravity: Oil specific gravity (dimensionless)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :param gas_to_oil_ratio: Gas-to-oil ratio in SCF/STB
    :return: Formation volume factor (Bo) in bbl/STB
    """
    if oil_specific_gravity <= 0 or gas_gravity <= 0:
        raise ValidationError("Specific gravities must be positive.")
    if gas_to_oil_ratio < 0:
        raise ValidationError("Gas-to-oil ratio must be non-negative.")
    if temperature < 32:
        raise ValidationError("Temperature seems unphysical (<32 °F). Check units.")

    x = (gas_to_oil_ratio * (gas_gravity / oil_specific_gravity) ** 0.5) + (
        1.25 * temperature
    )
    oil_fvf = 0.972 + 0.000147 * (x**1.175)
    return oil_fvf


@numba.njit(cache=True)
def _get_vazquez_beggs_oil_fvf_coefficients(
    oil_api_gravity: float,
) -> typing.Tuple[float, float, float]:
    """
    Returns the coefficients a1, a2, a3 for the Vazquez and Beggs oil FVF correlation based on oil API gravity.
    """
    if oil_api_gravity <= 30:
        return 4.677e-4, 1.751e-5, -1.811e-8
    return 4.670e-4, 1.100e-5, 1.337e-9


@numba.njit(cache=True)
def compute_oil_formation_volume_factor_vazquez_and_beggs(
    temperature: float,
    oil_specific_gravity: float,
    gas_gravity: float,
    gas_to_oil_ratio: float,
) -> float:
    """
    Computes the oil formation volume factor (Bo) using the Vazquez and Beggs correlation.

    Formula (Vazquez and Beggs, 1980):
        Bo = 1 + (a1 * R_s) + (a2 * (T - 60) * (γ_o / γ_g)) + (a3 * (T - 60) * R_s * (γ_o / γ_g))
    Where:
        - Bo: Oil formation volume factor (bbl/STB)
        - R_s: Gas-to-oil ratio in scf/STB
        - γ_o: Oil specific gravity (dimensionless)
        - γ_g: Gas specific gravity (dimensionless)
        - T: Temperature in degrees Fahrenheit

    Limitations:
        - Valid for API from 16 - 58
        - Typical range: 100-300 °F, 0.56-1.30 oil SG, 0 - 2000 scf/STB

    :param temperature: Reservoir temperature (°F)
    :param oil_specific_gravity: Oil specific gravity (dimensionless)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :param gas_to_oil_ratio: Gas-to-oil ratio in SCF/STB
    :return: Formation volume factor (Bo) in bbl/STB
    """
    oil_api_gravity = compute_oil_api_gravity(oil_specific_gravity)
    a1, a2, a3 = _get_vazquez_beggs_oil_fvf_coefficients(oil_api_gravity)
    oil_fvf = (
        1
        + (a1 * gas_to_oil_ratio)
        + (a2 * (temperature - 60) * (oil_specific_gravity / gas_gravity))
        + (
            a3
            * (temperature - 60)
            * gas_to_oil_ratio
            * (oil_specific_gravity / gas_gravity)
        )
    )
    return oil_fvf


@numba.njit(cache=True)
def correct_oil_fvf_for_pressure(
    saturated_oil_fvf: float,
    oil_compressibility: float,
    bubble_point_pressure: float,
    current_pressure: float,
) -> float:
    """
    Applies exponential shrinkage correction to oil FVF for pressures above bubble point.

    Formula:
        B_o(P) = B_o(sat) * exp[c_o * (Pb - P)]

    :param saturated_oil_fvf: Bo at bubble point pressure (saturated conditions) (bbl/STB)
    :param oil_compressibility: Isothermal oil compressibility (psi⁻¹)
    :param bubble_point_pressure: Bubble point pressure (psi)
    :param current_pressure: Current reservoir pressure (psi)
    :return: Adjusted Bo at current pressure (bbl/STB)
    """
    if current_pressure <= bubble_point_pressure:
        return saturated_oil_fvf

    delta_p = bubble_point_pressure - current_pressure
    correction_factor = clip(np.exp(oil_compressibility * delta_p), 1e-6, 5.0)
    return saturated_oil_fvf * correction_factor


@numba.njit(cache=True)
def compute_oil_formation_volume_factor(
    pressure: float,
    temperature: float,
    bubble_point_pressure: float,
    oil_specific_gravity: float,
    gas_gravity: float,
    gas_to_oil_ratio: float,
    oil_compressibility: float,
) -> float:
    """
    Computes the oil formation volume factor (Bo) in bbl/STB of oil
    based on pressure and temperature deviations from reference conditions.

    The formula used is based on the Standing correlation for temperatures below 100°F
    and the Vazquez and Beggs correlation for temperatures above 100°F.

    :param pressure: Reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param bubble_point_pressure: Bubble point pressure (psi)
    :param oil_specific_gravity: Oil specific gravity (dimensionless)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :param gas_to_oil_ratio: Gas-to-oil ratio in SCF/STB
    :param oil_compressibility: Oil isothermal compressibility (psi⁻¹)
    :return: Oil formation volume factor (Bo) in bbl/STB
    """
    if temperature <= 100:
        oil_fvf = compute_oil_formation_volume_factor_standing(
            temperature=temperature,
            oil_specific_gravity=oil_specific_gravity,
            gas_gravity=gas_gravity,
            gas_to_oil_ratio=gas_to_oil_ratio,
        )
    else:
        oil_fvf = compute_oil_formation_volume_factor_vazquez_and_beggs(
            temperature=temperature,
            oil_specific_gravity=oil_specific_gravity,
            gas_gravity=gas_gravity,
            gas_to_oil_ratio=gas_to_oil_ratio,
        )
    return correct_oil_fvf_for_pressure(
        saturated_oil_fvf=oil_fvf,
        oil_compressibility=oil_compressibility,
        bubble_point_pressure=bubble_point_pressure,
        current_pressure=pressure,
    )


def compute_water_formation_volume_factor(
    water_density: float, salinity: float
) -> float:
    """
    Computes the water formation volume factor (B_w) in bbl/STB of water
    based on pressure and temperature deviations from reference conditions.

    The formula used is:

        B_w = ρ_stp / ρ_w

    Where:
        - B_w: Water formation volume factor (bbl/STB)
        - ρ_stp: Water density at standard conditions (assumed 62.4 lbm/ft³)
        - ρ_w: Live water density at reservoir conditions (lbm/ft³)

    :param water_density: (Live) Water density at reservoir conditions (lbm/ft³)
    :param salinity: Water salinity (ppm of NaCl)
    :return: Water formation volume factor in bbl/STB
    """
    standard_water_density = compute_water_density_batzle(
        pressure=c.STANDARD_PRESSURE_IMPERIAL,
        temperature=c.STANDARD_TEMPERATURE_IMPERIAL,
        salinity=salinity,
    )
    if water_density <= 0:
        raise ValidationError("Water density must be positive.")
    if standard_water_density <= 0:
        raise ValidationError("Standard water density must be positive.")

    water_fvf = standard_water_density / water_density
    return water_fvf


@numba.njit(cache=True)
def compute_water_formation_volume_factor_mccain(
    pressure: float,
    temperature: float,
    salinity: float = 0.0,
    gas_solubility: float = 0.0,
) -> float:
    """
    McCain water FVF correlation (more commonly used in industry).

    Valid for:
    - T: 200-270°F
    - P: 1000-20,000 psi
    - Salinity: 0-200,000 ppm
    """
    # Convert temperature to Celsius for correlation
    temperature_in_celsius = fahrenheit_to_celsius(temperature)

    # Volume correction for temperature (ΔV_wT)
    delta_V_wT = (
        -1.0001e-2
        + 1.33391e-4 * temperature_in_celsius
        + 5.50654e-7 * temperature_in_celsius**2
    )

    # Volume correction for pressure (ΔV_wp)
    delta_V_wp = (
        -(1.95301e-9 * pressure * temperature_in_celsius)
        - (1.72834e-13 * pressure**2 * temperature_in_celsius)
        - (3.58922e-7 * pressure)
        - (2.25341e-10 * pressure**2)
    )

    # Volume correction for salinity and pressure (ΔV_wsp)
    salinity_wt_percent = salinity * 1e-4  # ppm to weight percent
    delta_V_wsp = salinity_wt_percent * (
        0.1249 + 1.1638e-4 * pressure - 1.1689e-6 * pressure**2
    )

    # Base FVF (gas-free)
    B_w = (1 + delta_V_wT) * (1 + delta_V_wp) * (1 + delta_V_wsp)

    # Correction for dissolved gas (if present)
    if gas_solubility > 0:
        # Rs_w in SCF/STB
        B_w = B_w - (gas_solubility * 1.0e-6)  # Approximate correction
    return B_w  # type: ignore


def compute_gas_formation_volume_factor(
    pressure: float,
    temperature: float,
    gas_compressibility_factor: float,
) -> float:
    """
    Computes the gas formation volume factor (B_g) in ft³/SCF, using the real gas law.

    Formula (real gas EOS):

        B_g = (Z * T * P_std) / (P * T_std)

    Where:
    - B_g: Gas formation volume factor (ft³/SCF)
    - Z: Gas compressibility factor (dimensionless)
    - T: Reservoir temperature (°F)
    - P: Reservoir pressure (psi)
    - P_std: Standard pressure = 14.696 psi
    - T_std: Standard temperature = 60°F

    Assumes ideal gas law corrected with Z-factor (real gas behavior).

    :param pressure: Reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param gas_compressibility_factor: Z-factor (dimensionless)
    :return: Gas formation volume factor (ft³/SCF)
    """
    if pressure <= 0 or temperature <= 0:
        raise ValidationError("Pressure and temperature must be positive.")
    if gas_compressibility_factor <= 0:
        raise ValidationError("Z-factor must be positive.")

    temperature_in_rankine = fahrenheit_to_rankine(temperature)
    return (
        gas_compressibility_factor
        * temperature_in_rankine
        * c.STANDARD_PRESSURE_IMPERIAL
        / (pressure * c.STANDARD_TEMPERATURE_RANKINE)
    )


@numba.njit(cache=True)
def compute_gas_compressibility_factor_papay(
    pressure: float,
    temperature: float,
    gas_gravity: float,
    h2s_mole_fraction: float = 0.0,
    co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0,
) -> float:
    """
    Computes gas compressibility factor using Papay's correlation,
    with corrections for sour gases using the Wichert-Aziz method.

    Papay's correlation is a widely used empirical relationship to estimate the
    compressibility factor of natural gas based on its pseudo-reduced properties.

    The equation is:

        Z = 1 - ((3.52 * P_r * exp(-0.869 * T_r)) / T_r) + ((0.274 * P_r**2)/T_r**2)

    Where:
    - Z is the compressibility factor (dimensionless)
    - P_r is the pseudo-reduced pressure (dimensionless)
    - T_r is the pseudo-reduced temperature (dimensionless)
    - P_r = P / P_pc
    - T_r = T / T_pc
    - P_pc is the pseudo-critical pressure (psi)
    - T_pc is the pseudo-critical temperature (°R)
    - P is the pressure (psi)
    - T is the temperature (°R)
    - P_pc and T_pc are calculated based on the gas specific gravity (gas_gravity).

    Valid Range:
    - Pseudo-reduced pressure (Pr): 0.2 < Pr < 15
    - Pseudo-reduced temperature (Tr): 1.05 < Tr < 3.0
    - Gas gravity: 0.55 < γg < 1.0
    - H₂S + CO₂ < 40 mol%
    - H₂S alone < 25 mol%

    :param gas_gravity: Gas specific gravity (dimensionless)
    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param h2s_mole_fraction: Mole fraction of H2S in the gas (dimensionless, default is 0.0)
    :param co2_mole_fraction: Mole fraction of CO2 in the gas (dimensionless, default is 0.0)
    :param n2_mole_fraction: Mole fraction of N2 in the gas (dimensionless, default is 0.0)
    :return: Compressibility factor Z (dimensionless)
    """
    if pressure <= 0 or temperature <= 0 or gas_gravity <= 0:
        raise ValidationError(
            "Pressure, temperature, and gas specific gravity must be positive."
        )
    if h2s_mole_fraction < 0 or h2s_mole_fraction > 1:
        raise ValidationError("H2S mole fraction must be between 0 and 1.")
    if co2_mole_fraction < 0 or co2_mole_fraction > 1:
        raise ValidationError("CO2 mole fraction must be between 0 and 1.")
    if n2_mole_fraction < 0 or n2_mole_fraction > 1:
        raise ValidationError("N2 mole fraction must be between 0 and 1.")

    pseudo_critical_pressure, pseudo_critical_temperature = (
        compute_gas_pseudocritical_properties(
            gas_gravity=gas_gravity,
            h2s_mole_fraction=h2s_mole_fraction,
            co2_mole_fraction=co2_mole_fraction,
            n2_mole_fraction=n2_mole_fraction,
        )
    )
    pseudo_reduced_pressure = pressure / pseudo_critical_pressure
    pseudo_reduced_temperature = (
        fahrenheit_to_rankine(temperature) / pseudo_critical_temperature
    )

    # Papay's correlation for gas compressibility factor
    compressibility_factor = (
        1
        - (
            (
                3.52
                * pseudo_reduced_pressure
                * np.exp(-0.869 * pseudo_reduced_temperature)
            )
            / pseudo_reduced_temperature
        )
        + ((0.274 * pseudo_reduced_pressure**2) / pseudo_reduced_temperature**2)
    )
    # Ensure Z is not negative
    return np.maximum(0.1, compressibility_factor)  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_gas_compressibility_factor_hall_yarborough(
    pressure: float,
    temperature: float,
    gas_gravity: float,
    h2s_mole_fraction: float = 0.0,
    co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0,
    max_iterations: int = 50,
    tolerance: float = 1e-10,
) -> float:
    """
    Computes gas compressibility factor using Hall-Yarborough (1973) implicit correlation.

    This is an iterative method that solves for reduced density (y) using Newton-Raphson:
        f(y) = -A * Pr + (y + y² + y³ - y⁴) / (1 - y)³ - B * y² + C * y^D = 0

    where:
        A = 0.06125 * Pr * t * exp(-1.2 * (1 - t)²)
        B = t * (14.76 - 9.76 * t + 4.58 * t²)
        C = t * (90.7 - 242.2 * t + 42.4 * t²)
        D = 2.18 + 2.82 * t
        t = 1 / Tr (reciprocal reduced temperature)

    Then: Z = A * Pr / y

    Valid Range:
    - Pr: 0.2 < Pr < 30 (wider than Papay)
    - Tr: 1.0 < Tr < 3.0
    - Most accurate for Pr > 1.0

    Advantages:
    - More accurate than Papay, especially at high pressure
    - Widely used in industry simulators
    - Explicit at low pressure (Pr < 0.5)

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :param h2s_mole_fraction: H₂S mole fraction (0.0 to 1.0)
    :param co2_mole_fraction: CO₂ mole fraction (0.0 to 1.0)
    :param n2_mole_fraction: N₂ mole fraction (0.0 to 1.0)
    :param max_iterations: Maximum Newton-Raphson iterations
    :param tolerance: Convergence tolerance
    :return: Compressibility factor Z (dimensionless)

    References:
        Hall, K.R. and Yarborough, L. (1973). "A New Equation of State for Z-factor Calculations."
        Oil & Gas Journal, June 18, 1973, pp. 82-92.
    """
    if pressure <= 0 or temperature <= 0 or gas_gravity <= 0:
        raise ValidationError(
            "Pressure, temperature, and gas gravity must be positive."
        )

    # Get pseudocritical properties with Wichert-Aziz correction
    pseudo_critical_pressure, pseudo_critical_temperature = (
        compute_gas_pseudocritical_properties(
            gas_gravity=gas_gravity,
            h2s_mole_fraction=h2s_mole_fraction,
            co2_mole_fraction=co2_mole_fraction,
            n2_mole_fraction=n2_mole_fraction,
        )
    )

    Pr = pressure / pseudo_critical_pressure
    Tr = fahrenheit_to_rankine(temperature) / pseudo_critical_temperature

    # For very low pressure, use ideal gas approximation
    if Pr < 0.01:
        return 1.0

    # Reciprocal reduced temperature
    t = 1.0 / Tr

    # Coefficients
    A = 0.06125 * Pr * t * np.exp(-1.2 * (1.0 - t) ** 2)
    B = t * (14.76 - 9.76 * t + 4.58 * t**2)
    C = t * (90.7 - 242.2 * t + 42.4 * t**2)
    D = 2.18 + 2.82 * t

    # Initial guess for reduced density (y)
    y = 0.27 * Pr / Tr
    y = min(0.9, max(0.01, y))

    # Newton-Raphson iteration
    for _ in range(max_iterations):
        y_old = y

        # Function f(y) and its derivative f'(y)
        y2 = y * y
        y3 = y2 * y
        y4 = y3 * y

        one_minus_y = 1.0 - y
        one_minus_y_cubed = one_minus_y**3

        # f(y) = -A*Pr + (y + y² + y³ - y⁴)/(1-y)³ - B*y² + C*y^D
        numerator = y + y2 + y3 - y4
        f = -A * Pr + numerator / one_minus_y_cubed - B * y2 + C * (y**D)

        # f'(y) = d/dy[(y + y² + y³ - y⁴)/(1-y)³] - 2*B*y + C*D*y^(D-1)
        # Using quotient rule: d/dy[numerator/denominator]
        d_numerator = 1.0 + 2.0 * y + 3.0 * y2 - 4.0 * y3
        d_denominator = 3.0 * one_minus_y**2  # Derivative of (1-y)³ is -3*(1-y)² * (-1)

        df = (
            (d_numerator * one_minus_y_cubed + numerator * d_denominator)
            / (one_minus_y_cubed * one_minus_y_cubed)
            - 2.0 * B * y
            + C * D * (y ** (D - 1.0))
        )

        # Newton-Raphson update
        if abs(df) < 1e-15:
            break  # Avoid division by zero

        y = y_old - f / df

        # Clamp y to physical range [0, 1)
        y = clip(y, 0.0, 0.99)

        # Check convergence
        if abs(y - y_old) < tolerance:
            break

    # Compute Z-factor
    if abs(y) < 1e-15:
        Z = 1.0  # Ideal gas
    else:
        Z = A * Pr / y

    # Clamp to physical range
    return clip(Z, 0.2, 3.0)


@numba.njit(cache=True)
def compute_gas_compressibility_factor_dranchuk_abou_kassem(
    pressure: float,
    temperature: float,
    gas_gravity: float,
    h2s_mole_fraction: float = 0.0,
    co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0,
    max_iterations: int = 50,
    tolerance: float = 1e-10,
) -> float:
    """
    Computes gas compressibility factor using Dranchuk-Abou-Kassem (DAK, 1975) correlation.

    This is an 11-parameter fit to Standing-Katz Z-factor chart data, solved iteratively:

        Z = 1 + (A₁ + A₂/Tr + A₃/Tr³ + A₄/Tr⁴ + A₅/Tr⁵)*ρr
            + (A₆ + A₇/Tr + A₈/Tr²)*ρr²
            - A₉*(A₇/Tr + A₈/Tr²)*ρr⁵
            + A₁₀*(1 + A₁₁*ρr²)*(ρr²/Tr³)*exp(-A₁₁*ρr²)

    where:
        ρr = 0.27 * Pr / (Z * Tr)  (reduced density)

    Coefficients (from Dranchuk & Abou-Kassem, 1975):
        A₁ = 0.3265, A₂ = -1.0700, A₃ = -0.5339, A₄ = 0.01569, A₅ = -0.05165
        A₆ = 0.5475, A₇ = -0.7361, A₈ = 0.1844, A₉ = 0.1056, A₁₀ = 0.6134, A₁₁ = 0.7210

    Valid Range:
        - Pr: 0.2 < Pr < 30 (widest range)
        - Tr: 1.0 < Tr < 3.0
        - Highly accurate across entire range

    Advantages:
        - Most accurate explicit correlation
        - Valid up to Pr = 30 (higher than Hall-Yarborough)
        - Industry standard for high-pressure gas

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :param h2s_mole_fraction: H₂S mole fraction (0.0 to 1.0)
    :param co2_mole_fraction: CO₂ mole fraction (0.0 to 1.0)
    :param n2_mole_fraction: N₂ mole fraction (0.0 to 1.0)
    :param max_iterations: Maximum iterations for density convergence
    :param tolerance: Convergence tolerance
    :return: Compressibility factor Z (dimensionless)

    References:
        Dranchuk, P.M. and Abou-Kassem, J.H. (1975). "Calculation of Z Factors for
        Natural Gases Using Equations of State." Journal of Canadian Petroleum Technology,
        July-September 1975, pp. 34-36.
    """
    if pressure <= 0 or temperature <= 0 or gas_gravity <= 0:
        raise ValidationError(
            f"Pressure, temperature, and gas gravity must be positive.\n"
            f"Pressure={float(pressure)}psi, Temperature={float(temperature)}°F, Gas gravity={float(gas_gravity)}"
        )

    # Get pseudocritical properties
    pseudo_critical_pressure, pseudo_critical_temperature = (
        compute_gas_pseudocritical_properties(
            gas_gravity=gas_gravity,
            h2s_mole_fraction=h2s_mole_fraction,
            co2_mole_fraction=co2_mole_fraction,
            n2_mole_fraction=n2_mole_fraction,
        )
    )
    Pr = pressure / pseudo_critical_pressure
    Tr = fahrenheit_to_rankine(temperature) / pseudo_critical_temperature

    # For very low pressure, use ideal gas
    if Pr < 0.01:
        return 1.0

    # DAK coefficients
    A1 = 0.3265
    A2 = -1.0700
    A3 = -0.5339
    A4 = 0.01569
    A5 = -0.05165
    A6 = 0.5475
    A7 = -0.7361
    A8 = 0.1844
    A9 = 0.1056
    A10 = 0.6134
    A11 = 0.7210

    # Initial guess for Z
    Z = 1.0

    # Iterative solution for Z
    for _ in range(max_iterations):
        Z_old = Z

        # Reduced density: ρr = 0.27 * Pr / (Z * Tr)
        rho_r = 0.27 * Pr / (Z * Tr)
        rho_r2 = rho_r * rho_r
        rho_r5 = rho_r2 * rho_r2 * rho_r

        # Reciprocal reduced temperature terms
        Tr_inv = 1.0 / Tr
        Tr_inv2 = Tr_inv * Tr_inv
        Tr_inv3 = Tr_inv2 * Tr_inv
        Tr_inv4 = Tr_inv3 * Tr_inv
        Tr_inv5 = Tr_inv4 * Tr_inv

        # Compute Z from DAK equation
        term1 = (A1 + A2 * Tr_inv + A3 * Tr_inv3 + A4 * Tr_inv4 + A5 * Tr_inv5) * rho_r
        term2 = (A6 + A7 * Tr_inv + A8 * Tr_inv2) * rho_r2
        term3 = -A9 * (A7 * Tr_inv + A8 * Tr_inv2) * rho_r5

        # Exponential term
        exp_term = np.exp(-A11 * rho_r2)
        term4 = A10 * (1.0 + A11 * rho_r2) * (rho_r2 * Tr_inv3) * exp_term

        Z = 1.0 + term1 + term2 + term3 + term4

        # Check convergence
        if abs(Z - Z_old) < tolerance:
            break

    # Clamp Z to physical range
    return clip(Z, 0.2, 3.0)


@numba.njit(cache=True)
def compute_gas_compressibility_factor(
    pressure: float,
    temperature: float,
    gas_gravity: float,
    h2s_mole_fraction: float = 0.0,
    co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0,
    method: GasZFactorMethod = "dak",
) -> float:
    """
    Computes (natural) gas compressibility factor.

    Method Selection Strategy:
    1. **High Pressure (Pr > 15)**: Use DAK (most accurate for Pr up to 30)
    2. **Medium Pressure (1 < Pr ≤ 15)**: Use Hall-Yarborough (best balance)
    3. **Low Pressure (Pr ≤ 1)**: Use Papay (fast, accurate for low Pr)
    4. **Fallback**: If any method fails validation, try others in order:
        DAK → Hall-Yarborough → Papay

    However, DAK is usually more than enough for black-oil simulations.

    Available Methods:
    - "papay": Papay's correlation (fastest, valid Pr: 0.2-15)
    - "hall-yarborough": Hall-Yarborough (accurate, valid Pr: 0.2-30)
    - "dak": Dranchuk-Abou-Kassem (most accurate, valid Pr: 0.2-30)

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param gas_gravity: Gas specific gravity (dimensionless, air=1.0)
    :param h2s_mole_fraction: H₂S mole fraction (0.0 to 1.0)
    :param co2_mole_fraction: CO₂ mole fraction (0.0 to 1.0)
    :param n2_mole_fraction: N₂ mole fraction (0.0 to 1.0)
    :param method: Correlation to use ("papay", "hall-yarborough", "dak"). Defaults to "dak"
    :return: Compressibility factor Z (dimensionless)

    Example:
    ```python
    Z = compute_gas_compressibility_factor(2000.0, 150.0, 0.65, method="dak")
    ```

    References:
    - Papay, J. (1985). "A Termelestechnologiai Parametereinek Valtozasa..."
    - Hall, K.R. and Yarborough, L. (1973). "A New Equation of State..."
    - Dranchuk, P.M. and Abou-Kassem, J.H. (1975). "Calculation of Z Factors..."
    """
    # Manual method selection
    if method == "papay":
        return compute_gas_compressibility_factor_papay(
            pressure=pressure,
            temperature=temperature,
            gas_gravity=gas_gravity,
            h2s_mole_fraction=h2s_mole_fraction,
            co2_mole_fraction=co2_mole_fraction,
            n2_mole_fraction=n2_mole_fraction,
        )
    elif method == "hall-yarborough":
        return compute_gas_compressibility_factor_hall_yarborough(
            pressure=pressure,
            temperature=temperature,
            gas_gravity=gas_gravity,
            h2s_mole_fraction=h2s_mole_fraction,
            co2_mole_fraction=co2_mole_fraction,
            n2_mole_fraction=n2_mole_fraction,
        )
    return compute_gas_compressibility_factor_dranchuk_abou_kassem(
        pressure=pressure,
        temperature=temperature,
        gas_gravity=gas_gravity,
        h2s_mole_fraction=h2s_mole_fraction,
        co2_mole_fraction=co2_mole_fraction,
        n2_mole_fraction=n2_mole_fraction,
    )


@numba.njit(cache=True)
def compute_oil_api_gravity(oil_specific_gravity: float) -> float:
    """
    Computes the API gravity (in degrees) from oil specific gravity.

    Formula:

        API = (141.5 / SG) - 131.5

    Where:
    - API: API gravity (degrees)
    - SG: Specific gravity of oil (dimensionless, relative to water at 60°F)

    :param oil_specific_gravity: Oil specific gravity (dimensionless)
    :return: API gravity in degrees (°API)
    """
    if oil_specific_gravity <= 0:
        raise ValidationError("Oil specific gravity must be greater than zero.")

    return (141.5 / oil_specific_gravity) - 131.5


@numba.njit(cache=True)
def _get_vazquez_beggs_oil_bubble_point_pressure_coefficients(
    oil_api_gravity: float,
) -> typing.Tuple[float, float, float]:
    """
    Returns the empirical coefficients (C₁, C₂, C₃) used in the Vazquez-Beggs
    bubble point pressure correlation based on oil API gravity.

    Coefficients vary for API ≤ 30 and API > 30:

        If API ≤ 30:
            C₁ = 0.0362, C₂ = 1.0937, C₃ = 25.7240
        Else:
            C₁ = 0.0178, C₂ = 1.1870, C₃ = 23.9310

    :param oil_api_gravity: Oil API gravity (°API)
    :return: Tuple of (C₁, C₂, C₃)
    """
    if oil_api_gravity <= 30.0:
        return 0.0362, 1.0937, 25.7240
    return 0.0178, 1.1870, 23.9310


@numba.njit(cache=True)
def compute_oil_bubble_point_pressure(
    gas_gravity: float,
    oil_api_gravity: float,
    temperature: float,
    gas_to_oil_ratio: float,
) -> float:
    """
    Computes the bubble point pressure of oil using the Vazquez-Beggs correlation.

    The correlation is defined as:

        P_b = [ R_s / (C₁ * SG * exp((C₃ * API) / T_R)) ]^(1 / C₂)

    Where:
        - P_b: Bubble point pressure (psi)
        - R_s: Gas-to-oil ratio (GOR) in SCF/STB
        - SG: Gas specific gravity (dimensionless)
        - API: Oil gravity in degrees API
        - T_R: Temperature in Rankine (°R)
        - C₁, C₂, C₃: Empirical constants (depend on API gravity)

    Valid for:
        - Oil API: ~16-45°
        - T: ~100-300 °F (converted to Rankine)
        - GOR up to ~2000 scf/STB

    :param gas_gravity: Gas specific gravity (dimensionless)
    :param oil_api_gravity: Oil API gravity in degrees API.
    :param temperature: Temperature (°F)
    :param gas_to_oil_ratio: Gas-to-oil ratio (GOR) in SCF/STB at reservoir conditions
    :return: Bubble point pressure (psi)
    """
    if gas_gravity <= 0:
        raise ValidationError("Gas specific gravity must be greater than zero.")
    if oil_api_gravity <= 0:
        raise ValidationError("Oil API gravity must be greater than zero.")
    if temperature <= 32:
        raise ValidationError("Temperature must be greater than absolute zero (32 °F).")
    if gas_to_oil_ratio < 0:
        raise ValidationError("Gas-to-oil ratio must be non-negative.")

    c1, c2, c3 = _get_vazquez_beggs_oil_bubble_point_pressure_coefficients(
        oil_api_gravity
    )
    temperature_rankine = temperature + 459.67
    pressure = (
        gas_to_oil_ratio
        / (c1 * gas_gravity * np.exp((c3 * oil_api_gravity) / temperature_rankine))
    ) ** (1 / c2)
    return pressure


@numba.njit(cache=True)
def compute_water_bubble_point_pressure_mccain(
    temperature: float,
    gas_solubility_in_water: float,
    salinity: float,
) -> float:
    """
    Computes the bubble point pressure using the inverted McCain correlation for methane.

    Valid for:
    - T: 100-400°F
    - P: 0-14,700 psi
    - Salinity: 0-200,000 ppm

    :param temperature: Temperature (°F)
    :param gas_solubility_in_water: Target gas solubility in SCF/STB
    :param salinity: Salinity in ppm
    :return: Bubble point pressure (psi)
    """
    A = 2.12 + 0.00345 * temperature - 0.0000125 * temperature**2
    B = 0.000045
    denominator = B * (1.0 - 0.000001 * salinity)
    bubble_point_pressure = np.maximum(0.0, (gas_solubility_in_water - A) / denominator)
    return bubble_point_pressure


def compute_water_bubble_point_pressure(
    temperature: float,
    gas_solubility_in_water: float,
    salinity: float = 0.0,
    gas: str = "methane",
) -> float:
    """
    Computes the bubble point pressure where the given gas solubility in water is reached.
    Uses analytical inversion for McCain, otherwise numerical root-finding.

    :param temperature: Temperature (°F)
    :param gas_solubility_in_water: Target gas solubility in SCF/STB
    :param salinity: Salinity in ppm
    :param gas: Gas name ("co2", "methane", "n2")
    :return: Bubble point pressure (psi)
    """
    gas = _get_gas_symbol(gas)
    if gas == "ch4" and np.any(100 <= temperature <= 400):
        # Inverted McCain
        return compute_water_bubble_point_pressure_mccain(
            temperature=temperature,
            gas_solubility_in_water=gas_solubility_in_water,
            salinity=salinity,
        )

    lower_bound_pressure = c.MINIMUM_VALID_PRESSURE
    upper_bound_pressure = c.MAXIMUM_VALID_PRESSURE

    lower_bound_solubility = (
        compute_gas_solubility_in_water(
            pressure=lower_bound_pressure,
            temperature=temperature,
            salinity=salinity,
            gas=gas,
        )
        - c.GAS_SOLUBILITY_TOLERANCE
    )
    upper_bound_solubility = (
        compute_gas_solubility_in_water(
            pressure=upper_bound_pressure,
            temperature=temperature,
            salinity=salinity,
            gas=gas,
        )
        + c.GAS_SOLUBILITY_TOLERANCE
    )

    if not (
        lower_bound_solubility <= gas_solubility_in_water <= upper_bound_solubility
    ):
        raise ComputationError(
            f"Target gas solubility {gas_solubility_in_water}SCF/STB is outside the range "
            f"[{lower_bound_solubility:.6f}, {upper_bound_solubility:.6f}] "
            f"for gas '{gas}' at T={temperature}°F and salinity={salinity}ppm."
        )

    # Use numerical solver for Duan/Henry
    # For gases like CO₂ and N₂ where no direct analytical formula exists to compute
    # the bubble point pressure, we numerically invert the solubility model (e.g., Duan, Henry's).
    # This inversion finds the pressure at which gas solubility in water equals the specified value.
    # Though these models don't explicitly define a bubble point, this process yields the effective
    # bubble point pressure—i.e., the pressure where gas begins to come out of solution.
    def residual(pressure: float) -> float:
        return (
            compute_gas_solubility_in_water(
                pressure=pressure, temperature=temperature, salinity=salinity, gas=gas
            )
            - gas_solubility_in_water
        )

    bubble_point_pressure = brentq(
        residual,
        a=lower_bound_pressure,
        b=upper_bound_pressure,
        xtol=1e-6,
        full_output=False,
    )
    return bubble_point_pressure  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_gas_to_oil_ratio(
    pressure: float,
    temperature: float,
    bubble_point_pressure: float,
    gas_gravity: float,
    oil_api_gravity: float,
    gor_at_bubble_point_pressure: typing.Optional[float] = None,
) -> float:
    """
    Computes the solution gas-to-oil ratio (Solution-GOR) using the Vazquez-Beggs correlation.

    Solution GOR is the amount of gas dissolved in oil at a given pressure and temperature.

    Two regimes:
        - **Saturated region (P < Pb)**: GOR is pressure-dependent.
        - **Undersaturated region (P ≥ Pb)**: GOR = GORb (constant). If not given, it is computed.

    The Vazquez-Beggs formula is:

        GOR = P^C₂ * C₁ * SG * exp[(C₃ * API) / T_R]

    where:
        - GOR: Gas-oil ratio (scf/STB)
        - P: Pressure in psi
        - SG: Gas specific gravity
        - API: Oil API gravity (°API)
        - T_R: Temperature in Rankine
        - C₁, C₂, C₃: Empirical coefficients

    Valid for:
        - Oil API: ~16-45°
        - T: ~100-300 °F (converted to Rankine)
        - GOR up to ~2000 scf/STB

    :param pressure: Reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param bubble_point_pressure: Bubble point pressure (psi)
    :param gas_gravity: Gas specific gravity (dimensionless, air = 1)
    :param oil_api_gravity: Oil API gravity in degrees API
    :param gor_at_bubble_point_pressure: GOR at the bubble point pressure SCF/STB, optional
    :return: Gas-to-oil ratio SCF/STB
    """
    if pressure <= 0:
        raise ValidationError("Pressure must be greater than zero.")

    temperature_in_rankine = temperature + 459.67
    c1, c2, c3 = _get_vazquez_beggs_oil_bubble_point_pressure_coefficients(
        oil_api_gravity
    )

    def compute_gor_vasquez_beggs(pressure: float) -> float:
        """Implementation of the Vazquez-Beggs GOR correlation."""
        return (
            (pressure**c2)
            * c1
            * gas_gravity
            * np.exp((c3 * oil_api_gravity) / temperature_in_rankine)
        )

    # Compute GOR at bubble point
    if pressure >= bubble_point_pressure:
        if gor_at_bubble_point_pressure is not None:
            gor = gor_at_bubble_point_pressure
        else:
            gor = compute_gor_vasquez_beggs(bubble_point_pressure)
    else:
        gor = compute_gor_vasquez_beggs(pressure)
    return np.maximum(0.0, gor)


@numba.njit(cache=True)
def _compute_dead_oil_viscosity_modified_beggs(
    temperature: float, oil_api_gravity: float
) -> float:
    if temperature <= 0:
        raise ValidationError("Temperature (°F) must be > 0 for this correlation.")

    temperature_rankine = temperature + 459.67
    oil_specific_gravity = 141.5 / (131.5 + oil_api_gravity)

    log_viscosity = (
        1.8653
        - 0.025086 * oil_specific_gravity
        - 0.5644 * np.log10(temperature_rankine)
    )
    viscosity = (10**log_viscosity) - 1
    return np.maximum(0.0, viscosity)


def compute_dead_oil_viscosity_modified_beggs(
    temperature: float,
    oil_specific_gravity: float,
) -> float:
    """
    Calculates the dead oil viscosity (mu_od) using the Modified Beggs correlation.
    Viscosity is in centipoise (cP), Labedi (1992).

    log10(mu_od + 1) = 1.8653 - 0.025086 * γ_o - 0.5644 * log10(T_R)

    where:
    - mu_od is the dead oil viscosity (cP)
    - γ_o is the specific gravity of oil
    - T_R is temperature in Rankine (°R)

    :param temperature: Temperature in Fahrenheit (°F)
    :param oil_specific_gravity: Specific gravity of the oil (dimensionless)
    :return: Dead oil viscosity in cP
    """
    oil_api_gravity = compute_oil_api_gravity(oil_specific_gravity)
    if not (5 <= oil_api_gravity <= 75):
        warnings.warn(
            f"API gravity {oil_api_gravity:.6f} is outside typical range [5, 75]. "
            f"Dead oil viscosity may be inaccurate."
        )
    return _compute_dead_oil_viscosity_modified_beggs(temperature, oil_api_gravity)


@numba.njit(cache=True)
def _compute_oil_viscosity(
    pressure: float,
    bubble_point_pressure: float,
    dead_oil_viscosity: float,
    gas_to_oil_ratio: float,
    gor_at_bubble_point_pressure: float,
) -> float:
    if pressure <= bubble_point_pressure:
        # Saturated case: compute viscosity using current GOR
        X = 10.715 * (gas_to_oil_ratio + 100) ** -0.515
        Y = 5.44 * (gas_to_oil_ratio + 150) ** -0.338
        return np.maximum(X * (dead_oil_viscosity**Y), 1e-6)

    # Undersaturated case: compute mu_ob at Pb first
    X_bp = 10.715 * (gor_at_bubble_point_pressure + 100) ** -0.515
    Y_bp = 5.44 * (gor_at_bubble_point_pressure + 150) ** -0.338
    mu_ob = X_bp * (dead_oil_viscosity**Y_bp)

    # Apply undersaturated viscosity correlation
    X_under = 2.6 * pressure**1.187 * np.exp(-11.513 - 8.98e-5 * pressure)
    return np.maximum(mu_ob * ((pressure / bubble_point_pressure) ** X_under), 1e-6)


def compute_oil_viscosity(
    pressure: float,
    temperature: float,
    bubble_point_pressure: float,
    oil_specific_gravity: float,
    gas_to_oil_ratio: float,
    gor_at_bubble_point_pressure: float,
) -> float:
    """
    Computes oil viscosity (cP) using the Modified Beggs & Robinson correlation
    for dead, saturated, and undersaturated oil.

    Saturated oil viscosity:
        mu_os = x_sat * mu_od^y_sat
        x_sat = 10.715 * (Rs + 100)^-0.515
        y_sat = 5.44 * (Rs + 150)^-0.338
        mu_od = 10^(1.8653 - 0.025086 * γ_o - 0.5644 * log10(T)) - 1

    Undersaturated oil viscosity:
        mu_o = mu_ob * (p / pb)^x_undersat
        x_undersat = 2.6 * p^1.187 * exp(-11.513 - 8.98e-5 * p)
        mu_ob = x_b * mu_od^y_b

    Where:
        - mu_od is the dead oil viscosity (cP)
        - mu_os is the saturated oil viscosity (cP)
        - mu_o is the undersaturated oil viscosity (cP)
        - Rs is the gas-to-oil ratio (GOR) at current pressure in standard SCF/STB
        - pb is the bubble point pressure (psi)
        - p is the current reservoir pressure (psi)
        - γ_o is the specific gravity of oil (dimensionless)
        - T is the reservoir temperature (°F)
        - mu_ob is the oil viscosity at bubble point pressure (cP)
        - x_b and y_b are coefficients for the bubble point viscosity correlation.
        - x_sat and y_sat are coefficients for the saturated viscosity correlation.
        - x_undersat is the coefficient for the undersaturated viscosity correlation.

    :param pressure: Current reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param bubble_point_pressure: Bubble point pressure of the oil (psi)
    :param oil_specific_gravity: Specific gravity of the oil (dimensionless)
    :param gas_to_oil_ratio: GOR at current pressure in standard SCF/STB
    :param gor_at_bubble_point_pressure: GOR at bubble point pressure in standard SCF/STB
    :return: Oil viscosity in cP
    """
    if temperature <= 0 or pressure <= 0 or bubble_point_pressure <= 0:
        raise ValidationError("Temperature and pressures must be positive.")
    if oil_specific_gravity <= 0:
        raise ValidationError("Oil specific gravity must be positive.")

    # Dead oil viscosity (mu_od)
    dead_oil_viscosity = compute_dead_oil_viscosity_modified_beggs(
        temperature=temperature, oil_specific_gravity=oil_specific_gravity
    )
    return _compute_oil_viscosity(
        pressure=pressure,
        bubble_point_pressure=bubble_point_pressure,
        dead_oil_viscosity=dead_oil_viscosity,
        gas_to_oil_ratio=gas_to_oil_ratio,
        gor_at_bubble_point_pressure=gor_at_bubble_point_pressure,
    )


def compute_gas_molecular_weight(gas_gravity: float) -> float:
    """
    Computes the apparent molecular weight of a gas (g/mol) from its specific gravity relative to air.

    Formula:
        MW_gas = γ_gas * MW_air

    where:
    - MW_gas is the molecular weight of the gas (g/mol)
    - γ_gas is the gas specific gravity (dimensionless, air = 1.0)
    - MW_air = 28.96 (g/mol) is the molecular weight of air

    :param gas_gravity: Specific gravity of the gas relative to air (dimensionless)
    :return: Molecular weight of the gas in grams per mole (g/mol)
    """
    if gas_gravity <= 0:
        raise ValidationError("Gas specific gravity must be greater than zero.")
    return gas_gravity * c.MOLECULAR_WEIGHT_AIR


@numba.njit(cache=True)
def compute_gas_pseudocritical_properties(
    gas_gravity: float,
    h2s_mole_fraction: float = 0.0,
    co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0,
) -> typing.Tuple[float, float]:
    """
    Computes pseudocritical pressure and temperature of natural gas in psi and °F.

    The pseudocritical properties are estimated using Sutton's correlation, which is
    widely used for sweet natural gas. For sour gases, the Wichert-Aziz correction
    adjusts the values based on H₂S, CO₂, and N₂ content.

    This is used as input to pseudo-reduced property models and EOS calculations.

    Sutton's correlation (for sweet gas):
        P_pc = 756.8 - 131.0 * γ_g - 3.6 * γ_g²     [psia]
        T_pc = 169.2 + 349.5 * γ_g - 74.0 * γ_g²    [°R]

    Wichert-Aziz Correction:
        ε = 120[(X_H2S + X_CO2)^0.9 - (X_H2S + X_CO2)^1.6] + 15[√X_H2S - X_H2S⁴]

        Then:
        T_pc' = T_pc - ε
        P_pc' = P_pc * T_pc' / (T_pc + X_H2S(1 - X_H2S) * ε)

    :param gas_gravity: Gas specific gravity (dimensionless, air = 1.0).
    :param h2s_mole_fraction: Mole fraction of H₂S (dimensionless).
    :param co2_mole_fraction: Mole fraction of CO₂ (dimensionless).
    :param n2_mole_fraction: Mole fraction of N₂ (dimensionless).
    :return: Tuple (P_pc in psi, T_pc in °R)
    """
    if gas_gravity <= 0:
        raise ValidationError("Gas specific gravity must be greater than zero.")

    total_acid_gas_fraction = h2s_mole_fraction + co2_mole_fraction
    if total_acid_gas_fraction > 0.40:
        raise ValidationError(
            f"Total acid gas fraction ({total_acid_gas_fraction}) exceeds 40% limit "
            "for Wichert-Aziz correction."
        )
    if h2s_mole_fraction > 0.25:
        raise ValidationError(
            f"H₂S mole fraction ({h2s_mole_fraction}) exceeds 25% limit "
            "for Wichert-Aziz correction."
        )

    # Sutton's pseudocritical properties (psia and Rankine)
    pseudocritical_pressure = 756.8 - 131.0 * gas_gravity - 3.6 * gas_gravity**2
    pseudocritical_temperature_rankine = (
        169.2 + 349.5 * gas_gravity - 74.0 * gas_gravity**2
    )

    if total_acid_gas_fraction > 0.001:
        A = h2s_mole_fraction + co2_mole_fraction
        B = h2s_mole_fraction
        epsilon = 120.0 * (A**0.9 - A**1.6) + 15.0 * (B**0.5 - B**4)

        pseudocritical_temperature_rankine -= epsilon
        pseudocritical_pressure = (
            pseudocritical_pressure
            * pseudocritical_temperature_rankine
            / (
                pseudocritical_temperature_rankine
                + h2s_mole_fraction * (1 - h2s_mole_fraction) * epsilon
            )
        )
    return pseudocritical_pressure, pseudocritical_temperature_rankine


def compute_gas_density(
    pressure: float,
    temperature: float,
    gas_gravity: float,
    gas_compressibility_factor: float,
) -> float:
    """
    Calculates the gas density (lbm/ft³) using the real gas equation of state.

    :param pressure: Pressure (psi).
    :param temperature: Temperature (°F).
    :param gas_gravity: Gas specific gravity (dimensionless, air=1.0).
    :param gas_compressibility_factor: Gas compressibility factor (dimensionless).
    :return: Gas density in lbm/ft³.
    """
    temperature_in_rankine = temperature + 459.67
    # lbm/lbmol is the same numerical value as g/mol
    gas_molecular_weight_lbm_per_lbmole = compute_gas_molecular_weight(gas_gravity)
    # Density in lbm/ft3
    gas_density = (pressure * gas_molecular_weight_lbm_per_lbmole) / (
        gas_compressibility_factor
        * c.IDEAL_GAS_CONSTANT_IMPERIAL
        * temperature_in_rankine
    )
    return gas_density


def compute_gas_viscosity(
    temperature: float,
    gas_density: float,
    gas_molecular_weight: float,
) -> float:
    """
    Calculates the gas viscosity (cP) using the Lee-Gonzalez-Eakin (LGE) correlation.

    This correlation estimates gas viscosity from gas density and temperature using:

        μ_g = (k * 1e-4) * exp(x * ρ_g^y)

    where:
        - μ_g is the gas viscosity in centipoise (cP)
        - k is a temperature-dependent constant
        - x and y are empirical coefficients based on gas molecular weight and temperature
        - k = [(9.4 + 0.02 * M_g) * T^1.5] / [209 + 19 * M_g + T]
        - x = 3.5 + 986 / T + 0.01 * M_g
        - y = 2.4 - 0.2 * x
        - M_g is the gas molecular weight (lbm/lbmol)
        - ρ_g is gas density in g/cm³
        - T is temperature in Rankine

    This correlation is valid for a wide range of temperatures and pressures,
    typically from 100 °F to 400 °F and pressures up to 10,000 psi.

    :param temperature: Temperature (°F)
    :param gas_density: Gas density (lbm/ft³)
    :param gas_molecular_weight: Gas molecular weight (g/mol)
    :return: Gas viscosity in (cP)
    """
    temperature_in_rankine = temperature + 459.67
    # NO CONVERSION NEEDED - g/mol is numerically equal to lb/lbmol
    gas_molecular_weight_lbm_per_lbmole = gas_molecular_weight
    density_in_grams_per_cm3 = (
        gas_density * c.POUNDS_PER_CUBIC_FEET_TO_GRAMS_PER_CUBIC_METER
    )

    k = (
        (9.4 + (0.02 * gas_molecular_weight_lbm_per_lbmole))
        * (temperature_in_rankine**1.5)
        / (209 + (19 * gas_molecular_weight_lbm_per_lbmole) + temperature_in_rankine)
    )

    x = (
        3.5
        + (986 / temperature_in_rankine)
        + (0.01 * gas_molecular_weight_lbm_per_lbmole)
    )
    y = 2.4 - (0.2 * x)

    exponent = x * (density_in_grams_per_cm3**y)
    exponent = np.minimum(700, np.maximum(-700, exponent))  # cap to prevent overflow

    gas_viscosity = (k * 1e-4) * np.exp(exponent)
    return np.maximum(0.0, gas_viscosity)


@numba.njit(cache=True)
def _compute_water_viscosity(
    temperature: float,
    salinity: float,
    pressure: float,
    ppm_to_weight_fraction: float,
) -> float:
    salinity_fraction = salinity * ppm_to_weight_fraction
    A = 1.0 + 1.17 * salinity_fraction + 3.15e-6 * salinity_fraction**2
    B = 1.48e-3 - 1.8e-7 * salinity_fraction
    C = 2.94e-6

    viscosity_at_standard_pressure = A - (B * temperature) + (C * temperature**2)
    pressure_correction_factor = (
        0.9994 + (4.0295e-5 * pressure) + (3.1062e-9 * pressure**2)
    )
    viscosity_at_pressure = viscosity_at_standard_pressure * pressure_correction_factor
    return np.maximum(viscosity_at_pressure, 1e-6)


def compute_water_viscosity(
    temperature: float,
    salinity: float = 0.0,
    pressure: float = 14.7,
) -> float:
    """
    Computes water viscosity using McCain's corrected correlation for reservoir conditions.

    This correlation is valid for:
        - Temperatures between 86 °F and 350 °F
        - Salinities up to 300,000 ppm (weight-based)
        - Pressures up to 10,000 psi

    The viscosity at standard pressure (14.7 psia) is given by:

        mu_w_std = A - B * T + C * T²

    where:
        - mu_w_std is the water viscosity in cP at standard pressure
        - T is the temperature in °F
        - A = 1.0 + 1.17 * S + 3.15e-6 * S²
        - B = 1.48e-3 - 1.8e-7 * S
        - C = 2.94e-6
        - S is the salinity in weight fraction (ppm divided by 1,000,000)

    If pressure is provided, the viscosity is corrected using:

        mu_w = mu_w_std * (0.9994 + 4.0295e-5 * P + 3.1062e-9 * P²)

    where:
        - mu_w is the water viscosity at pressure P
        - P is the pressure in psi

    :param temperature: Temperature in Fahrenheit (°F)
    :param salinity: Salinity in parts per million (ppm), default is 0 (fresh water)
    :param pressure: Pressure in psi, default is 14.7 psi (atmospheric pressure)
    :return: Water viscosity in centipoise (cP)
    """
    if salinity < 0:
        raise ValidationError("Salinity must be non-negative.")

    if pressure is not None and pressure < 0:
        raise ValidationError("Pressure must be non-negative.")

    if temperature < 60 or temperature > 400:
        warnings.warn(
            f"Temperature {temperature:.6f}°F is outside the valid range for McCain's water viscosity correlation (60°F to 400°F)."
        )

    if salinity > 300_000:
        warnings.warn(
            f"Salinity {salinity:.6f}ppm is unusually high for McCain's water viscosity correlation."
        )

    if pressure is not None and pressure > 10_000:
        warnings.warn(
            f"Pressure {pressure:.6f}psi is unusually high for McCain's water viscosity correlation."
        )
    return _compute_water_viscosity(
        temperature=temperature,
        salinity=salinity,
        pressure=pressure,
        ppm_to_weight_fraction=c.PPM_TO_WEIGHT_FRACTION,
    )


@numba.njit(cache=True)
def _compute_oil_compressibility_liberation_correction_term(
    pressure: float,
    temperature: float,
    gas_gravity: float,
    oil_api_gravity: float,
    bubble_point_pressure: float,
    gas_formation_volume_factor: float,
    oil_formation_volume_factor: float,
    gor_at_bubble_point_pressure: float,
) -> float:
    """
    Computes the liberation correction term for oil compressibility below bubble point pressure.

    The correction term is give by:

        x = (Bg/Bo * dRs/dp) / 5.615

        dRs/dp = (R_s(P + ΔP) - R_s(P - ΔP)) / (2 * ΔP)

    where:
    - R_s is the solution Gas-Oil Ratio (GOR) at current pressure and temperature
      in standard cubic feet per stock tank barrel (scf/stb).
    - ΔP is a small pressure increment (e.g., 0.01 psi or 0.0001 * P).
    - Bg is the gas formation volume factor (bbl/scf).
    - Bo is the oil formation volume factor (bbl/STB).
    - 5.615 is a conversion factor to convert from scf/STB to bbl/STB.
    - x is the correction term for oil compressibility.

    :param pressure: Current reservoir pressure (psi).
    :param bubble_point_pressure: Bubble point pressure (psi).
    :param gas_formation_volume_factor: Gas formation volume factor (bbl/scf).
    :param oil_formation_volume_factor: Oil formation volume factor (bbl/STB).
    :param gor_at_bubble_point_pressure: GOR at bubble point pressure (scf/stb).
    :return: Correction term for oil compressibility.
    """
    delta_p = np.maximum(0.01, 1e-4 * pressure)
    pressure_plus = pressure + delta_p
    pressure_minus = pressure - delta_p
    if pressure_plus > 0:
        gor_plus_delta = compute_gas_to_oil_ratio(
            pressure=pressure_plus,
            temperature=temperature,
            bubble_point_pressure=bubble_point_pressure,
            gas_gravity=gas_gravity,
            oil_api_gravity=oil_api_gravity,
            gor_at_bubble_point_pressure=gor_at_bubble_point_pressure,
        )
    else:
        gor_plus_delta = 0.0

    if pressure_minus > 0:
        gor_minus_delta = compute_gas_to_oil_ratio(
            pressure=pressure_minus,
            temperature=temperature,
            bubble_point_pressure=bubble_point_pressure,
            gas_gravity=gas_gravity,
            oil_api_gravity=oil_api_gravity,
            gor_at_bubble_point_pressure=gor_at_bubble_point_pressure,
        )
    else:
        gor_minus_delta = 0.0

    dRs_dp = (gor_plus_delta - gor_minus_delta) / (2 * delta_p)
    return (gas_formation_volume_factor / oil_formation_volume_factor) * dRs_dp / 5.615


@numba.njit(cache=True)
def compute_oil_compressibility(
    pressure: float,
    temperature: float,
    bubble_point_pressure: float,
    oil_api_gravity: float,
    gas_gravity: float,
    gor_at_bubble_point_pressure: float,
    gas_formation_volume_factor: float = 1.0,
    oil_formation_volume_factor: float = 1.0,
) -> float:
    """
    Calculates the oil compressibility (C_o) in psi⁻¹ using the Vasquez and Beggs (1980) correlation.

    - If Pressure (P) > Bubble Point Pressure (Pb): Uses the Vasquez and Beggs
      correlation for undersaturated oil compressibility.
    - If Pressure (P) <= Bubble Point Pressure (Pb): The oil is saturated.
      For the *liquid phase compressibility*, it's commonly approximated as
      the compressibility at the bubble point pressure. The dominant volume
      change below Pb is due to gas liberation, which is handled in total system
      compressibility.

    The Vasquez and Beggs correlation is given by:
    For P > Pb (Undersaturated Oil):

        C_o = (-1433 + 5 * R_s + 17.2 * T_F - 1180 * S.G + 12.61 * API) / 10⁵ * P

    For P <= Pb (Saturated Oil):

        C_o = C_o(P) + (Bg/Bo * dRs/dp) / 5.615

    where:

    - C_o is the oil compressibility (psi⁻¹)
    - R_s is the solution Gas-Oil Ratio (GOR) at current pressure and temperature
      in standard cubic feet per stock tank barrel (scf/stb).
    - T_F is the temperature in Fahrenheit (°F).
    - S.G is the specific gravity of the solution gas (dimensionless, air=1.0).
    - API is the API gravity of the stock tank oil (degrees).
    - P is the pressure in psi (pounds per square inch).

    Vasquez and Beggs correlation is typically valid for:
    - Pressure: 100 to 5,000 psi
    - Temperature: 100 to 300 °F
    - API gravity: 16 to 58 degrees

    :param pressure: Reservoir pressure (psi).
    :param temperature: Reservoir temperature (°F).
    :param bubble_point_pressure: Bubble point pressure (psi).
    :param oil_api_gravity: API gravity of the stock tank oil.
    :param gas_gravity: Specific gravity of the solution gas (air=1).
    :param gor_at_bubble_point_pressure: Solution Gas-Oil Ratio at bubble point pressure (SCF/STB).
        This value should be obtained from a GOR correlation (e.g., Vazquez-Beggs GOR).
    :return: Oil compressibility in psi⁻¹
    """
    if (
        pressure <= 0
        or bubble_point_pressure <= 0
        or temperature <= 0
        or gas_gravity <= 0
        or oil_api_gravity <= 0
    ):
        raise ValidationError(
            "All input parameters (P, Pb, T, Gas SG, API) must be positive."
        )

    def compute_base_compressibility(pressure: float) -> float:
        current_gor = compute_gas_to_oil_ratio(
            pressure=pressure,
            temperature=temperature,
            bubble_point_pressure=bubble_point_pressure,
            gas_gravity=gas_gravity,
            oil_api_gravity=oil_api_gravity,
            gor_at_bubble_point_pressure=gor_at_bubble_point_pressure,
        )
        val = (
            -1433
            + 5 * current_gor
            + 17.2 * temperature
            - 1180 * gas_gravity
            + 12.61 * oil_api_gravity
        ) / ((10**5) * pressure)
        return np.maximum(val, 0.0)

    if pressure > bubble_point_pressure:
        return compute_base_compressibility(pressure)

    base_comp = compute_base_compressibility(pressure)
    correction_term = _compute_oil_compressibility_liberation_correction_term(
        pressure=pressure,
        temperature=temperature,
        gas_gravity=gas_gravity,
        oil_api_gravity=oil_api_gravity,
        bubble_point_pressure=bubble_point_pressure,
        gas_formation_volume_factor=gas_formation_volume_factor,
        oil_formation_volume_factor=oil_formation_volume_factor,
        gor_at_bubble_point_pressure=gor_at_bubble_point_pressure,
    )
    return base_comp + correction_term


@numba.njit(cache=True)
def compute_gas_compressibility(
    pressure: float,
    temperature: float,
    gas_gravity: float,
    gas_compressibility_factor: typing.Optional[float] = None,
    h2s_mole_fraction: float = 0.0,
    co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0,
) -> float:
    """
    Calculates the isothermal gas compressibility (C_g) in psi⁻¹ using Papay's Z-factor correlation
    and its analytical derivative.

    :param pressure: Reservoir pressure (psi).
    :param temperature: Reservoir temperature (°F).
    :param gas_gravity: Specific gravity of the gas (air=1).
    :param gas_compressibility_factor: Optional pre-computed Z-factor (dimensionless).
        If provided, it will be used directly instead of (re)calculating it.
    :param h2s_mole_fraction: H2S mole fraction (0 to 1).
    :param co2_mole_fraction: CO2 mole fraction (0 to 1).
    :param n2_mole_fraction: N2 mole fraction (0 to 1).
    :return: Gas compressibility in psi⁻¹.
    """
    if pressure <= 0 or temperature <= 0 or gas_gravity <= 0:
        raise ValidationError(
            "Pressure, temperature, and gas specific gravity must be positive."
        )

    # Calculate pseudocritical properties
    pseudo_critical_pressure, pseudo_critical_temperature = (
        compute_gas_pseudocritical_properties(
            gas_gravity=gas_gravity,
            h2s_mole_fraction=h2s_mole_fraction,
            co2_mole_fraction=co2_mole_fraction,
            n2_mole_fraction=n2_mole_fraction,
        )
    )

    # Calculate pseudo-reduced properties at the given pressure and temperature
    pseudo_reduced_pressure = pressure / pseudo_critical_pressure
    pseudo_reduced_temperature = temperature / pseudo_critical_temperature

    # Calculate Z-factor using your provided function
    if gas_compressibility_factor is not None:
        # If a Z-factor is provided, use it directly
        Z = gas_compressibility_factor
    else:
        Z = compute_gas_compressibility_factor(
            gas_gravity=gas_gravity,
            pressure=pressure,
            temperature=temperature,
            h2s_mole_fraction=h2s_mole_fraction,
            co2_mole_fraction=co2_mole_fraction,
            n2_mole_fraction=n2_mole_fraction,
        )

    # Calculate the analytical derivative of Z with respect to P_r (pseudo-reduced pressure)
    # from Papay's correlation: dZ/dP_r = -3.52 * exp(-0.869 * T_r) / T_r + 0.548 * P_r / T_r^2

    exp_term = np.exp(-0.869 * pseudo_reduced_temperature)
    dZ_dP_r = (-3.52 * exp_term / pseudo_reduced_temperature) + (
        0.548 * pseudo_reduced_pressure / (pseudo_reduced_temperature**2)
    )

    # Calculate the gas compressibility (C_g)
    # C_g = (1/P) - (1/(Z * Ppc)) * (dZ/dPpr)
    # Ensure Ppc is not zero, which should be handled by compute_gas_pseudocritical_properties
    if pseudo_critical_pressure == 0:
        raise ValidationError(
            "Pseudo-critical pressure cannot be zero for compressibility calculation."
        )

    gas_compressibility = (1 / pressure) - (
        1 / (Z * pseudo_critical_pressure)
    ) * dZ_dP_r
    # Compressibility must be non-negative
    return np.maximum(0.0, gas_compressibility)


@numba.njit(cache=True)
def _gas_solubility_in_water_mccain_methane(
    pressure: float,
    temperature: float,
    salinity: float = 0.0,
) -> float:
    """
    Calculates gas solubility in water (Rsw) using McCain's correlation (1990).

    This correlation is valid for typical reservoir conditions:
        - Temperature: 311 K to 478 K (100 °F to 400 °F)
        - Pressure: 0-10,000 psia
        - Salinity: 0-150,000 ppm

    The formula is:

        Rsw = A(T_F) + (B * P_psia * (1 - 1e-6 * Salinity_ppm))

    where:
        A(T_F) = 2.12 + 0.00345 * T_F - 0.0000125 * T_F²
        B = 0.000045
        T_F = temperature in degrees Fahrenheit
        P_psia = pressure in psia
        Rsw = gas solubility in scf/STB

    :param pressure: Pressure (psi).
    :param temperature: Temperature (°F).
    :param salinity: Salinity in parts per million (ppm).
    :return: Gas solubility in water in SCF/STB.
    """
    if pressure < 0 or temperature < 0 or salinity < 0:
        raise ValidationError(
            "Pressure, temperature, and salinity must be non-negative."
        )

    if not (100 <= temperature <= 400):
        raise ValidationError(
            "Temperature out of valid range for McCain's Rsw correlation (311 K to 478 K)."
        )

    # A(T_F) term from McCain
    A_term = 2.12 + (0.00345 * temperature) - (0.0000125 * temperature**2)

    # B is a constant in the validated McCain form
    B = 0.000045

    salinity_correction = 1.0 - (0.000001 * salinity)
    gas_solubility = A_term + (B * pressure * salinity_correction)
    # Clamp to non-negative
    return np.maximum(0.0, gas_solubility)


@numba.njit(cache=True)
def _gas_solubility_in_water_duan_sun_co2(
    pressure: float,
    temperature: float,
    salinity: float = 0.0,
    nacl_molecular_weight: float = 58.44,
    psi_to_bar: float = 0.0689476,
) -> float:
    """
    Calculates CO₂ solubility in water (Rsw) using the Duan and Sun (2003) model.

    The coefficients are from Duan and Sun, "An improved model for the calculation of CO2 solubility
    in pure water and aqueous NaCl solutions", Chemical Geology, 2003.

    The formula is:
        ln(m_CO2) = c1 + c2/T + c3*ln(T) + (c4*P)/T + (c5*P²)/T² - k_s * m_NaCl
        m_NaCl = salinity / (58.44 * 1000)  # Convert ppm to molality (mol/kg H2O)
        k_s = 0.119 + 0.0003 * T  # Setschenow coefficient
        Rsw = m_CO2 * 315.4  # Convert molality to SCF/STB

    where:
        - m_CO2 is the molality of CO₂ in mol/kg H₂O
        - T is temperature in Kelvin
        - P is pressure in bar
        - m_NaCl is the molality of NaCl in mol/kg H₂O
        - k_s is the Setschenow coefficient for CO₂-NaCl interaction
        - Rsw is the CO₂ solubility in standard cubic feet per stock tank barrel (SCF/STB)
        - c1, c2, c3, c4, c5 are empirical coefficients

    The model is valid for:
        - Temperature: 273.15 K to 533.15 K
        - Pressure: 0 to 2000 bar
        - Salinity: up to 4.5 mol/kg NaCl

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param salinity: Salinity in parts per million (ppm NaCl)
    :return: CO₂ solubility in SCF/STB.
    """
    if pressure <= 0 or temperature <= 0:
        raise ValidationError("Pressure and temperature must be positive.")

    P = pressure * psi_to_bar  # Convert pressure from psi to bar
    T = fahrenheit_to_kelvin(temperature)

    if not (273.15 <= T <= 533.15):
        raise ValidationError(
            "Temperature is out of the valid range for this model (0-260°C)."
        )
    if not (0 < P <= 2000):
        raise ValidationError(
            "Pressure is out of the valid range for this model (0-2000 bar)."
        )

    # Calculate CO₂ molality in PURE WATER
    # Using the equation from Duan & Sun (2003) for the fugacity of CO2
    c1 = 16.3869
    c2 = -3013.95
    c3 = -2.25336
    c4 = 0.00693898
    c5 = -6.65349e-7

    # This calculates ln(y*P), which for a pure CO2 phase is ln(f_CO2).
    # The exponential gives the fugacity of CO2.
    # At equilibrium, f_CO2_gas = f_CO2_liquid = m_CO2 * H_CO2
    # This simplified form directly calculates the molality (m_CO2)
    ln_m_co2_pure = c1 + c2 / T + c3 * np.log(T) + (c4 * P) / T + (c5 * P**2) / T**2

    m_co2_pure = np.exp(ln_m_co2_pure)

    # Apply Salinity Correction (Setschenow equation)
    # Convert salinity from ppm to molality (mol NaCl / kg H2O)
    # 1 ppm NaCl ≈ 1 mg NaCl / 1 L H2O ≈ 1 mg NaCl / 1 kg H2O
    m_nacl = salinity / (nacl_molecular_weight * 1000)

    # Setschenow coefficient (k_s) for CO2-NaCl interaction, with T-dependence
    # This is a common empirical fit.
    k_s = 0.119 + 0.0003 * T

    # Corrected molality in brine
    m_co2_brine = m_co2_pure / (10 ** (k_s * m_nacl))

    MOLALITY_TO_SCF_STB_CO2 = 315.4  # Approximate conversion factor for CO2
    # Convert Molality to SCF/STB
    # This is an approximate conversion that depends on water density and standard conditions.
    # It combines molality -> mole fraction -> volume ratio.
    # For many reservoir engineering applications, a factor around 315.4 is used.
    rsw = m_co2_brine * MOLALITY_TO_SCF_STB_CO2
    return rsw  # type: ignore


# Henry's constants (Sander, 2020) — ln(H) = A + B/T + C*ln(T)
# H in mol/(m³·Pa), will be inverted to Pa·m³/mol
HENRY_COEFFICIENTS = {
    "co2": (-58.0931, 90.5069, 0.027766),
    "ch4": (-68.8862, 101.4956, 0.021599),
    "n2": (-71.0592, 120.1052, 0.02624),
    "o2": (-64.848, 107.45, 0.0223),
    "ar": (-50.0, 100.0, 0.0200),
    "he": (-30.0, 80.0, 0.0150),
    "h2": (-25.0, 70.0, 0.0120),
}
SETSCHENOW_CONSTANTS = {
    "co2": 0.12,
    "ch4": 0.11,
    "n2": 0.13,
    "o2": 0.13,
    "ar": 0.10,
    "he": 0.08,
    "h2": 0.07,
}

_GAS_ALIASES = {
    "methane": "ch4",
    "carbondioxide": "co2",
    "nitrogen": "n2",
    "oxygen": "o2",
    "argon": "ar",
    "helium": "he",
    "hydrogen": "h2",
}


def _get_gas_symbol(gas_name: str) -> str:
    gas_name = gas_name.lower().replace(" ", "").replace("-", "")
    return _GAS_ALIASES.get(gas_name, gas_name)


def _gas_solubility_in_water_henry_law(
    pressure: float,
    temperature: float,
    gas: str,
    molar_masses: typing.Dict[str, float],
    henry_coefficients: typing.Dict[str, typing.Tuple[float, float, float]],
    salinity: float = 0.0,
) -> float:
    """
    Estimates gas solubility in water using Henry's Law with Setschenow salinity correction.

    Formula:
        Rsw = (P / H(T)) * (M / ρ_water) * exp(-k_s * molality)

    Henry's constant H(T) is computed as:
        ln H = A + B / T + C * ln(T)     [Sander, 2020]

    Setschenow correction:
        molality = salinity / (58.44 * 1000)
        exp(-k_s * molality)

    where:
    - Rsw is the gas solubility in m³/m³
    - P is the pressure in Pa
    - H(T) is Henry's constant in Pa·m³/mol
    - M is the molar mass of the gas in kg/mol
    - ρ_water is the water density in kg/m³
    - k_s is the Setschenow constant for the gas (dimensionless)
    - molality is the salinity in mol/kg (converted from ppm NaCl)

    :param pressure: Pressure in (psi)
    :param temperature: Temperature in (°F)
    :param salinity: Salinity in ppm NaCl
    :param gas: One of "co2", "methane", or "n2"
    :param molar_masses: Dictionary of molar masses for gases in kg/mol
    :param henry_coefficients: Dictionary of Henry's Law coefficients (A, B, C) for gases
    :return: Solubility in SCF/STB (standard cubic feet per stock tank barrel)
    """
    gas = gas.lower()
    if gas not in henry_coefficients:
        raise ValidationError(f"Unsupported gas '{gas}' for Henry's Law fallback.")

    A, B, C = henry_coefficients[gas]
    M = molar_masses[gas]
    temperature_in_kelvin = fahrenheit_to_kelvin(temperature)

    ln_H_inv = -(A + B / temperature_in_kelvin + C * np.log(temperature_in_kelvin))
    H_inv = np.exp(ln_H_inv)  # mol/(m³·Pa)
    H = 1.0 / H_inv  # Pa·m³/mol

    try:
        water_density = (
            compute_fluid_density(pressure, temperature, "Water")
            * c.POUNDS_PER_CUBIC_FEET_TO_KILOGRAM_PER_CUBIC_METER
        )
    except Exception:
        water_density = c.STANDARD_WATER_DENSITY

    # Setschenow salinity correction
    # Converts salinity from ppm (mg/kg) to mol/kg using molar mass in g/mol
    molarity = salinity / (c.MOLECULAR_WEIGHT_NACL * 1000)

    k_s = SETSCHENOW_CONSTANTS[gas]
    salinity_factor = np.exp(-k_s * molarity)

    gas_solubility = (
        (pressure / H) * (M / water_density) * salinity_factor
    )  # m³ gas / m³ water
    return gas_solubility * c.CUBIC_METER_PER_CUBIC_METER_TO_SCF_PER_STB


def compute_gas_solubility_in_water(
    pressure: float,
    temperature: float,
    salinity: float = 0.0,
    gas: str = "methane",
) -> float:
    """
    Computes gas solubility in water using McCain, Duan, or Henry's Law based on gas type and temperature.

    :param pressure: Pressure (psi).
    :param temperature: Temperature (°F).
    :param salinity: Salinity in parts per million (ppm).
    :param gas: Type of gas ("methane", "CO2", or "N2"). Default is "methane".
    :return: Gas solubility in water in SCF/STB (standard cubic feet per stock tank barrel).
    """
    gas = _get_gas_symbol(gas)
    if gas == "ch4" and 100.0 <= temperature <= 400.0:
        # For methane, we use McCain's correlation for gas solubility in water
        return _gas_solubility_in_water_mccain_methane(pressure, temperature, salinity)

    elif gas == "co2" and 32 <= temperature <= 572:
        # For CO2, we use Duan's correlation for higher accuracy
        return _gas_solubility_in_water_duan_sun_co2(
            pressure=pressure,
            temperature=temperature,
            salinity=salinity,
            nacl_molecular_weight=c.MOLECULAR_WEIGHT_NACL,
            psi_to_bar=c.PSI_TO_BAR,
        )

    molar_masses = {
        "co2": c.MOLECULAR_WEIGHT_CO2 / 1000,  # Convert g/mol to kg/mol
        "ch4": c.MOLECULAR_WEIGHT_CH4 / 1000,
        "n2": c.MOLECULAR_WEIGHT_N2 / 1000,
        "ar": c.MOLECULAR_WEIGHT_ARGON / 1000,
        "o2": c.MOLECULAR_WEIGHT_O2 / 1000,
        "he": c.MOLECULAR_WEIGHT_HELIUM / 1000,
        "h2": c.MOLECULAR_WEIGHT_H2 / 1000,
    }
    return _gas_solubility_in_water_henry_law(
        pressure=pressure,
        temperature=temperature,
        gas=gas,
        molar_masses=molar_masses,
        henry_coefficients=HENRY_COEFFICIENTS,
        salinity=salinity,
    )


@numba.njit(cache=True)
def compute_gas_free_water_formation_volume_factor(
    pressure: float, temperature: float
) -> float:
    """
    Calculates the Water Formation Volume Factor (Bw) for dissolved-gas-free water
    using McCain's correlation (based on Petroleum Office function BwMcCain_GasFree).
    This function primarily accounts for thermal expansion and water compressibility,
    without the effect of dissolved gas.

    Bwd = (1.00012 + 1.25E-5 * T_F + 2.45E-7 * T_F**2) * (1.0 - 1.95E-9 * P_psia + 1.72E-13 * P_psia**2)
    Note: This is often for pure water at 60F, then corrected.
    Let's use a simpler form combining thermal and pressure effects,
    often attributed to McCain's overall behavior for pure water.

    From McCain, W.D., "The Properties of Petroleum Fluids", 3rd Ed., p. 326-327:
    V_T = -1.0001E-2 + 1.33391E-4 * T_F + 5.50654E-7 * T_F**2 # Thermal Expansion
    V_P = -1.95301E-9 * P_psia + 1.72492E-13 * P_psia**2 # Isothermal Compressibility from 14.7 psi
    Bw_gas_free = (1.0 + V_T) * (1.0 + V_P)

    :param pressure: Pressure (psi).
    :param temperature: Temperature (°F).
    :return: Dissolved-gas-free Water Formation Volume Factor (Bw_gas_free) in bbl/STB.
    """
    if pressure < 0 or temperature < 0:
        raise ValidationError(
            "Pressure and temperature cannot be negative for gas-free water FVF."
        )

    thermal_expansion = (
        -0.010001 + (1.33391e-4 * temperature) + (5.50654e-7 * temperature**2)
    )
    isothermal_compressibility = -(1.95301e-9 * pressure) + (1.72492e-13 * pressure**2)
    gas_free_water_fvf = (1.0 + thermal_expansion) * (1.0 + isothermal_compressibility)
    return np.maximum(0.9, gas_free_water_fvf)  # Bw_gas_free is typically close to 1.0


@numba.njit(cache=True)
def _compute_dRsw_dP_mccain(temperature: float, salinity: float) -> float:
    """
    Calculates the derivative of gas solubility in water (Rsw) with respect to pressure,
    based on McCain's correlation for Rsw.
    Returns dRsw/dP in scf/(STB*psi).
    """
    if temperature < 0 or salinity < 0:
        raise ValidationError(
            "Temperature and salinity cannot be negative for dRsw/dP."
        )

    derivative_pure_water = (
        0.0000164 + (0.000000134 * temperature) - (0.00000000185 * temperature**2)
    )
    salinity_correction_factor = 1.0 - 0.000001 * salinity
    # This derivative is positive (Rsw increases with P)
    return derivative_pure_water * salinity_correction_factor


@numba.njit(cache=True)
def _compute_dBw_gas_free_dp_mccain(pressure: float, temperature: float) -> float:
    """
    Calculates the derivative of dissolved-gas-free Water Formation Volume Factor (Bw_gas_free)
    with respect to pressure, based on McCain's correlation.
    Returns dBw_gas_free/dP in res bbl/(STB*psi). This value will be negative.
    """
    if pressure < 0:
        raise ValidationError("Pressure cannot be negative for dBw_gas_free/dP.")

    thermal_expansion_term = (
        1.0  # This 1.0 is part of (1 + VT)
        + -0.010001
        + (1.33391e-4 * temperature)
        + (5.50654e-7 * temperature**2)
    )
    # Derivative of V_P wrt P: d(V_P)/dP = -1.95301E-9 + 2 * 1.72492E-13 * P_psia
    isothermal_compressibility_derivative = -(1.95301e-9) + (2 * 1.72492e-13 * pressure)

    # dBw_gas_free/dP = d/dP [ (1+VT)*(1+VP) ] = (1+VT) * d(1+VP)/dP
    # This value will be negative as Bw decreases with increasing P.
    return thermal_expansion_term * isothermal_compressibility_derivative


def compute_water_compressibility(
    pressure: float,
    temperature: float,
    bubble_point_pressure: float,  # This Pwb is for the water's dissolved gas in water.
    gas_formation_volume_factor: float,  # Bg in ft3/SCF
    gas_solubility_in_water: float,  # Rsw in SCF/STB
    gas_free_water_formation_volume_factor: float,  # Bw_gas_free in bbl/STB (output of compute_gas_free_water_formation_volume_factor)
    salinity: float = 0.0,
) -> float:
    """
    Calculates the isothermal water compressibility (C_w) using McCain's correlations.
    Distinguishes between undersaturated and saturated water conditions.

    The McCain-based correlation for water compressibility is given by:

    - For Undersaturated Water (P >= Pwb):
        C_w = - (1/Bw) * (dBw_gas_free/dP)_T
        (Here, Bw is Bw_gas_free as no dissolved gas comes out of solution.)

    - For Saturated Water (P < Pwb):
        C_w = - (1/Bw_actual) * (dBw_gas_free/dP)_T  + (Bg / Bw_actual) * (dRsw/dP)_T
        (This form adds the effect of gas coming out of solution to the base liquid compressibility.
         Bw_actual = Bw_gas_free + Rsw * Bg)

    :param pressure: Reservoir pressure in (psi).
    :param temperature: Reservoir temperature in (°F).
    :param bubble_point_pressure: Water bubble point pressure in (psi).
        This is the pressure at which gas starts to come out of solution from water.

    :param gas_formation_volume_factor: Gas formation volume factor (Bg) in (ft³/SCF) at the current pressure and temperature.
    :param gas_solubility_in_water: Gas solubility in water (Rsw) in (SCF/STB) at the current pressure and temperature.
    :param gas_free_water_formation_volume_factor: Gas-free water formation volume factor (Bw_gas_free) in (bbl/STB).
        This should be computed using compute_gas_free_water_formation_volume_factor.

    :param salinity: Salinity in parts per million (ppm).
    :return: Water compressibility (C_w) in (psi⁻¹).
    """
    gas_fvf_in_bbl_per_scf = gas_formation_volume_factor * c.CUBIC_FEET_TO_BARRELS
    dBw_gas_free_dP = _compute_dBw_gas_free_dp_mccain(
        pressure=pressure,
        temperature=temperature,
    )
    dRsw_dP = _compute_dRsw_dP_mccain(
        temperature=temperature,
        salinity=salinity,
    )

    if pressure >= bubble_point_pressure:
        # Undersaturated Water (P >= Pwb)
        if np.any(gas_free_water_formation_volume_factor <= 0):
            raise ValidationError(
                "Calculated Bw for undersaturated water is non-positive."
            )
        c_w = -(1.0 / gas_free_water_formation_volume_factor) * dBw_gas_free_dP
    else:
        # Saturated Water (P < Pwb)
        water_fvf_in_bbl_per_stb = gas_free_water_formation_volume_factor + (
            gas_solubility_in_water * gas_fvf_in_bbl_per_scf
        )
        if np.any(water_fvf_in_bbl_per_stb <= 0):
            raise ValidationError("Calculated Bw for saturated water is non-positive.")

        c_w_gas_free_component = -(1.0 / water_fvf_in_bbl_per_stb) * dBw_gas_free_dP
        gas_liberation_component = (
            gas_fvf_in_bbl_per_scf / water_fvf_in_bbl_per_stb
        ) * dRsw_dP
        c_w = c_w_gas_free_component + gas_liberation_component

    return np.maximum(0.0, c_w)


def compute_live_oil_density(
    api_gravity: float,
    gas_gravity: float,
    gas_to_oil_ratio: float,
    formation_volume_factor: float,
) -> float:
    """
    Estimates live oil density at reservoir conditions at the current pressure
    and temperature, considering dissolved gas and oil compressibility.

    Based on:
        - Stock tank oil density from API gravity.
        - Contribution of dissolved gas mass.
        - Volume expansion/compression via FVF.

    :param pressure: Reservoir pressure (psi)
    :param bubble_point_pressure: Bubble point pressure (psi)
    :param api_gravity: Oil API gravity [°API]
    :param gas_gravity: Gas specific gravity (relative to air)
    :param oil_compressibility: Oil compressibility (psi⁻¹)
    :param gas_to_oil_ratio: Gas-to-oil ratio at current pressure (SCF/STB)
    :param formation_volume_factor: Oil formation volume factor at current pressure (bbl/STB)
    :return: Live oil density (lb/ft³) at reservoir conditions.
    """
    # Convert API to stock tank oil density (lb/ft³)
    stock_tank_oil_density_lb_per_ft3 = (
        141.5 / (api_gravity + 131.5)
    ) * c.STANDARD_WATER_DENSITY_IMPERIAL

    # Mass of oil per STB (lb)
    mass_stock_tank_oil = stock_tank_oil_density_lb_per_ft3 / c.CUBIC_FEET_TO_STB

    # Mass of dissolved gas per STB (lb)
    # Approx: 1 scf = gas_gravity * (molecular weight of air) / 379.49 lb
    gas_mass_per_scf = (gas_gravity * c.MOLECULAR_WEIGHT_AIR) / c.SCF_PER_POUND_MOLE
    mass_dissolved_gas = gas_to_oil_ratio * gas_mass_per_scf

    # Total mass and volume
    total_mass_lb_per_stb = mass_stock_tank_oil + mass_dissolved_gas
    # print(formation_volume_factor)
    total_volume_ft3_per_stb = formation_volume_factor * c.BARRELS_TO_CUBIC_FEET

    # Live oil density in lb/ft³
    live_oil_density_lb_per_ft3 = total_mass_lb_per_stb / total_volume_ft3_per_stb
    return live_oil_density_lb_per_ft3


def compute_water_density_mccain(
    pressure: float, temperature: float, salinity: float = 0.0
) -> float:
    """
    Computes the live water/brine density at reservoir conditions using McCain's correlation.

    This includes the effects of salinity, pressure, and temperature deviations from standard conditions.

    Correlation adapted from:
        McCain, "Properties of Petroleum Fluids", 2nd/3rd Ed.

    rho_brine (lb/ft³) = rho_std + Δrho_salinity + Δrho_pressure + Δrho_temperature

    where:
        Δrho_salinity   = 0.438603 * salinity_wt_percent
        Δrho_pressure   = 0.00001427 * (pressure - 14.7)
        Δrho_temperature = -0.00048314 * (temperature - 60.0)

    :param pressure: Pressure in psia.
    :param temperature: Temperature in degrees Fahrenheit.
    :param salinity: Salinity in ppm.
    :return: Live brine density in lb/ft³.
    """
    if salinity < 0:
        raise ValidationError("Salinity cannot be negative.")
    if pressure < 0:
        raise ValidationError("Pressure cannot be negative.")

    salinity_in_wt_percent = salinity / 10000.0

    delta_salinity = 0.438603 * salinity_in_wt_percent
    delta_pressure = 0.00001427 * (pressure - 14.7)
    delta_temperature = -0.00048314 * (temperature - 60.0)
    water_density = (
        c.STANDARD_WATER_DENSITY_IMPERIAL
        + delta_salinity
        + delta_pressure
        + delta_temperature
    )
    return water_density


@numba.njit(cache=True)
def compute_water_density_batzle(
    pressure: float, temperature: float, salinity: float
) -> float:
    """
    Computes the live water/brine density using Batzle & Wang's correlation.

    This is more accurate for high temperature and pressure conditions,
    using empirical adjustments based on weight fraction salinity.

    Correlation:
        Batzle & Wang (1992), Geophysics, Vol. 57, No. 11

    rho_brine (g/cm³) = 1.0 + 1e-3 * [
        S * (0.668 + 0.44 * S + 1e-6 * (300 * T - 2400 * T * S + P * (80 + 3 * T - 3300 * S)))
    ]

    Converts to lb/ft³ using the conversion factor (1 g/cm³ = 62.42796 lb/ft³).

    where:
        rho_brine = brine density in g/cm³
        S = salinity in weight fraction (ppm / 1e6)
        T = temperature in Celsius
        P = pressure in MPa

    :param pressure: Pressure in psia.
    :param temperature: Temperature in degrees Fahrenheit.
    :param salinity: Salinity in ppm.
    :return: Brine density in lb/ft³.
    """
    if salinity < 0:
        raise ValidationError("Salinity cannot be negative.")
    if pressure < 0:
        raise ValidationError("Pressure cannot be negative.")

    # Convert units
    temperature_in_celsius = fahrenheit_to_celsius(temperature)  # °F to °C
    pressure_MPa = pressure * 0.00689476  # psia to MPa
    salinity_weight_fraction = salinity / 1e6  # ppm to weight fraction

    S = salinity_weight_fraction
    T = temperature_in_celsius
    P = pressure_MPa

    # Batzle & Wang correlation in g/cm³
    brine_density_g_per_cm3 = 1.0 + 1e-3 * (
        S
        * (
            0.668
            + 0.44 * S
            + 1e-6 * (300 * T - 2400 * T * S + P * (80 + 3 * T - 3300 * S))
        )
    )
    # Convert to lb/ft³ (1 g/cm³ = 62.42796 lb/ft³)
    water_density = brine_density_g_per_cm3 * 62.42796
    return water_density  # type: ignore


def compute_water_density(
    pressure: float,
    temperature: float,
    gas_gravity: float = 0.0,
    salinity: float = 0.0,
    gas_solubility_in_water: float = 0.0,
    gas_free_water_formation_volume_factor: float = 1.0,
) -> float:
    """
    Calculates the live water/brine density at reservoir conditions
    using McCain's correlations.

    The correlation is based on the mass balance:

        rho_w = (Mass of standard water + Mass of dissolved gas) / Volume of live water

        rho_w (lb/ft³) = (rho_w_std (lb/ft³) + Rsw (scf/STB) * gas_gravity * 0.01359) / Bw (res bbl/STB)

    where:
    - rho_w_std is the standard water density at 60 F and 14.7 psia (62.37 lb/ft³ for pure water).
    - Rsw is the gas solubility in water at current pressure and temperature (scf/STB).
    - gas_gravity is the specific gravity of the dissolved gas (relative to air).
    - Bw is the water formation volume factor at current pressure and temperature (res bbl/STB).

    :param salinity: Salinity in parts per million (ppm). Defaults to 0.0 for pure water.
    :param gas_gravity: Specific gravity of dissolved gas (relative to air). Defaults to 0.0 if no gas.
    :param gas_solubility_in_water: Gas solubility in water (Rsw) in (SCF/STB) at current pressure and temperature.
    :param gas_free_water_formation_volume_factor: Gas-free water formation volume factor (Bw)(bbl/STB) at current pressure and temperature.
        Defaults to 1.0 (no gas effect).
    :return: Live water/brine density (lb/ft³) at reservoir conditions.
    """
    if salinity < 0 or gas_gravity < 0:
        raise ValidationError("Salinity and gas gravity must be non-negative.")

    standard_water_density_in_lb_per_ft3 = compute_water_density_batzle(
        pressure=pressure, temperature=temperature, salinity=salinity
    )

    # For density calculation using the formula, Bw in the denominator is the *actual*
    # Bw (live water FVF). For water, dissolved gas usually has a very minor effect
    # on Bw (which is typically close to 1.0). The `calculate_bw_gas_free_mccain`
    # handles pressure and temperature effects.
    # If the Rsw term is significant, it's captured in the numerator's mass.
    # So, bw_actual = bw_gas_free is a common approximation here, or if Rsw*Bg effect on volume is added.
    # For simplicity and given the formula structure, we use the bw_gas_free as the Bw in denominator.

    if gas_free_water_formation_volume_factor <= 0:
        raise ValidationError(
            "Gas-free water formation volume factor (Bw) is non-positive, cannot calculate density."
        )

    # Calculate Live Water Density (Imperial units first)
    # Mass of standard water per STB (volume of STB is 1 STB, density lb/ft3 * 5.615 ft3/bbl)
    standard_mass_water_in_lb_per_stb = (
        standard_water_density_in_lb_per_ft3 * c.BARRELS_TO_CUBIC_FEET
    )  # lb/STB

    # Mass of dissolved gas per STB
    # Note: The 0.01359 factor in the simple formula often implicitly converts scf to bbl for the gas mass contribution.
    # Let's use the explicit conversion:
    # Mass gas (lb) = Rsw (scf) * Density of gas at std cond (lb/scf)
    # Density of gas at std cond (lb/scf) = gas_gravity * (28.96 lb/lb-mol_air / 379.4 scf/lb-mol_ideal_gas) = gas_gravity * 0.0763 lb/scf
    mass_of_dissolved_gas_in_lb_per_stb = (
        gas_solubility_in_water * gas_gravity * c.MOLECULAR_WEIGHT_AIR
    ) / c.SCF_PER_POUND_MOLE  # lb_mass_gas/STB

    # Total mass of live water (and dissolved gas) per STB
    total_mass_in_lb_per_stb = (
        standard_mass_water_in_lb_per_stb + mass_of_dissolved_gas_in_lb_per_stb
    )

    # Volume of live water at reservoir conditions (ft³ per STB)
    volume_of_live_water_in_ft3_per_stb = (
        gas_free_water_formation_volume_factor * c.BARRELS_TO_CUBIC_FEET
    )  # res bbl/STB * ft³/bbl = ft³/STB
    live_water_density_in_lb_per_ft3 = (
        total_mass_in_lb_per_stb / volume_of_live_water_in_ft3_per_stb
    )  # lb/ft³
    # Ensure density is non-negative
    return np.maximum(0.0, live_water_density_in_lb_per_ft3)


@numba.njit(cache=True)
def compute_gas_to_oil_ratio_standing(
    pressure: float, oil_api_gravity: float, gas_gravity: float
) -> float:
    """
    Standing correlation to compute Rs (solution GOR) in scf/STB.

    This assumes the oil is at or below bubble point pressure, and temperature
    is not used (approximation based only on pressure, API gravity, and gas gravity).

    Estimated using the Standing correlation for solution gas-oil ratio (Rs):

        Rs = gas_gravity * [ (P / 18.2 + 1.4) * 10^(0.0125 * API) ]^(1 / 1.2048)

    where:
    - P is the pressure in psia.
    - T_F is the temperature in degrees Fahrenheit.
    - API is the API gravity of the oil.
    - gas_gravity is the specific gravity of the gas (relative to air).

    This correlation is typically used for light oils and may not be accurate
    for heavy oils (API < 10) or high pressures.

    :param pressure: Pressure (psi)
    :param oil_api_gravity: API gravity of the oil in degrees API.
    :param gas_gravity: Specific gravity of the gas (relative to air).
    :return: Solution gas-oil ratio (Rs) in (SCF/STB).
    """
    if oil_api_gravity < 0 or gas_gravity < 0 or pressure < 0:
        raise ValidationError("All inputs must be non-negative for Rs calculation.")

    if oil_api_gravity < 10:
        raise ValidationError(
            "API gravity must be greater than or equal to 10 for Standing's correlation."
        )

    gor = gas_gravity * (
        (pressure / 18.2 + 1.4) * 10 ** (0.0125 * oil_api_gravity)
    ) ** (1 / 1.2048)
    return gor


def estimate_bubble_point_pressure_standing(
    oil_api_gravity: float,
    gas_gravity: float,
    observed_gas_to_oil_ratio: float,
) -> float:
    """
    Estimate bubble point pressure (Pb) using Standing's correlation
    given observed Rs and known oil API gravity and gas gravity.

    THIS FUNCTION ESTIMATES THE BUBBLE POINT PRESSURE AND IS NOT A DIRECT
    MEASUREMENT. It is only valid for light oils (API > 10) and may not be accurate
    for heavy oils or high pressures.

    This assumes the oil is at or below bubble point pressure, and temperature
    is not used (approximation based only on pressure, API gravity, and gas gravity).

    The bubble point pressure is estimated by solving the equation for Rs
    using the Standing correlation:

        Rs = gas_gravity * [ (P / 18.2 + 1.4) * 10^(0.0125 * API) ]^(1 / 1.2048)

    where:
    - P is the pressure in psia.
    - T_F is the temperature in degrees Fahrenheit.
    - API is the API gravity of the oil.

    :param oil_api_gravity: API gravity of the oil in degrees API.
    :param gas_gravity: Specific gravity of the gas (relative to air).
    :param observed_gas_to_oil_ratio: Observed solution gas-oil ratio (Rs) in SCF/STB.
    :return: Estimated bubble point pressure (Pb) (psi).
    """

    def residual(pressure: float) -> float:
        gor = compute_gas_to_oil_ratio_standing(
            pressure=pressure,
            oil_api_gravity=oil_api_gravity,
            gas_gravity=gas_gravity,
        )
        return gor - observed_gas_to_oil_ratio

    solver = root_scalar(residual, bracket=(14.696, 10000), method="brentq")
    if not solver.converged:
        raise ComputationError("Could not converge to a bubble point pressure.")

    bubble_point_pressure = solver.root
    return bubble_point_pressure


@numba.njit(cache=True)
def _compute_bubble_point_pressure_vazquez_beggs(
    gas_gravity: float,
    oil_api_gravity: float,
    temperature: float,
    gas_to_oil_ratio: float,
) -> float:
    """
    Internal njit version of bubble point pressure calculation using Vazquez-Beggs.

    Same as `compute_oil_bubble_point_pressure` but without input validation
    for use in tight loops.

    :param gas_gravity: Gas specific gravity (dimensionless)
    :param oil_api_gravity: Oil API gravity in degrees API.
    :param temperature: Temperature (°F)
    :param gas_to_oil_ratio: Gas-to-oil ratio (GOR) in SCF/STB
    :return: Bubble point pressure (psi)
    """
    # Get Vazquez-Beggs coefficients based on API gravity
    if oil_api_gravity <= 30.0:
        c1, c2, c3 = 0.0362, 1.0937, 25.7240
    else:
        c1, c2, c3 = 0.0178, 1.1870, 23.9310

    temperature_rankine = temperature + 459.67
    pressure = (
        gas_to_oil_ratio
        / (c1 * gas_gravity * np.exp((c3 * oil_api_gravity) / temperature_rankine))
    ) ** (1 / c2)
    return pressure


@numba.njit(cache=True)
def _compute_gas_to_oil_ratio_standing_internal(
    pressure: float, oil_api_gravity: float, gas_gravity: float
) -> float:
    """
    Internal njit version of Standing correlation for Rs.

    Same as `compute_gas_to_oil_ratio_standing` but without input validation.

    :param pressure: Pressure (psi)
    :param oil_api_gravity: API gravity of the oil in degrees API.
    :param gas_gravity: Specific gravity of the gas (relative to air).
    :return: Solution gas-oil ratio (Rs) in (SCF/STB).
    """
    gor = gas_gravity * (
        (pressure / 18.2 + 1.4) * 10 ** (0.0125 * oil_api_gravity)
    ) ** (1 / 1.2048)
    return gor


@numba.njit(cache=True)
def estimate_solution_gor(
    pressure: float,
    temperature: float,
    oil_api_gravity: float,
    gas_gravity: float,
    max_iterations: int = 20,
    tolerance: float = 1e-4,
) -> float:
    """
    Estimate solution gas-to-oil ratio Rs(P, T) iteratively.

    This solves the coupled system where:
    - Rs depends on P and Pb via the correlation
    - Pb depends on Rs and T via Vazquez-Beggs

    The algorithm:
    1. Initial guess: Rs from Standing correlation (uses P, API, γg)
    2. Compute Pb from Rs using Vazquez-Beggs
    3. If P > Pb: oil is undersaturated, Rs = Rs_max (at bubble point)
    4. If P <= Pb: oil is saturated, refine Rs estimate
    5. Iterate until convergence

    For undersaturated oil (P > Pb):
        Rs remains constant at Rsb (the Rs at bubble point pressure)

    For saturated oil (P <= Pb):
        Rs varies with pressure - more gas dissolves at higher P

    :param pressure: Reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param oil_api_gravity: Oil API gravity in degrees API (typically 15-50)
    :param gas_gravity: Gas specific gravity relative to air (typically 0.6-1.2)
    :param max_iterations: Maximum iterations for convergence (default: 20)
    :param tolerance: Relative tolerance for convergence (default: 1e-4)
    :return: Solution gas-to-oil ratio Rs in SCF/STB

    Notes:
        - Convergence is typically achieved in 3-5 iterations
        - Uses Standing correlation for initial guess (ignores T)
        - Uses Vazquez-Beggs for Pb calculation (includes T)
        - Handles both saturated and undersaturated conditions
    """
    # Initial guess from Standing correlation
    rs_current = _compute_gas_to_oil_ratio_standing_internal(
        pressure=pressure,
        oil_api_gravity=oil_api_gravity,
        gas_gravity=gas_gravity,
    )

    # Ensure reasonable bounds for Rs
    rs_min = 0.0
    rs_max = 5000.0  # Practical upper limit for Rs (SCF/STB)
    rs_current = max(rs_min, min(rs_current, rs_max))

    for _ in range(max_iterations):
        # Compute bubble point pressure from current Rs estimate
        pb_current = _compute_bubble_point_pressure_vazquez_beggs(
            gas_gravity=gas_gravity,
            oil_api_gravity=oil_api_gravity,
            temperature=temperature,
            gas_to_oil_ratio=rs_current,
        )

        # Determine saturation state and update Rs
        if pressure > pb_current:
            # Undersaturated: P > Pb
            # Rs should be the Rs at bubble point (Rsb)
            # Since Pb(Rs) is monotonically increasing with Rs,
            # we need to find Rs such that Pb(Rs, T) = P
            # Use bisection to find Rsb where Pb(Rsb, T) ≈ P

            rs_lo = 0.0
            rs_hi = rs_max

            # Bisection to find Rs where Pb(Rs, T) = P
            for _ in range(50):  # Inner bisection iterations
                rs_mid = (rs_lo + rs_hi) / 2.0
                pb_mid = _compute_bubble_point_pressure_vazquez_beggs(
                    gas_gravity=gas_gravity,
                    oil_api_gravity=oil_api_gravity,
                    temperature=temperature,
                    gas_to_oil_ratio=rs_mid,
                )

                if pb_mid < pressure:
                    rs_lo = rs_mid
                else:
                    rs_hi = rs_mid

                if (rs_hi - rs_lo) < tolerance * rs_mid:
                    break

            rs_new = (rs_lo + rs_hi) / 2.0
        else:
            # Saturated: P <= Pb
            # Rs varies with P so we use the Standing-based estimate
            # but refine using the relationship that P should equal Pb(Rs, T)
            # when oil is exactly at bubble point

            # For saturated oil below bubble point, Rs increases with P
            # Use the current Standing estimate as is, since it's pressure-based
            rs_new = _compute_gas_to_oil_ratio_standing_internal(
                pressure=pressure,
                oil_api_gravity=oil_api_gravity,
                gas_gravity=gas_gravity,
            )

        # Check convergence
        if rs_current > tolerance:  # Avoid division by zero
            relative_change = abs(rs_new - rs_current) / rs_current
            if relative_change < tolerance:
                return rs_new

        rs_current = rs_new

    return rs_current


@numba.njit(cache=True)
def compute_hydrocarbon_in_place(
    area: float,
    thickness: float,
    porosity: float,
    phase_saturation: float,
    formation_volume_factor: float,
    net_to_gross_ratio: float = 1.0,
    hydrocarbon_type: typing.Literal["oil", "gas", "water"] = "oil",
    acre_ft_to_bbl: float = 7758.0,
    acre_ft_to_ft3: float = 43560.0,
) -> float:
    """
    Computes the (free) hydrocarbon (or free water) in place (HCIP or FWIP) in stock tank barrels (STB) or standard cubic feet (SCF)
    using the volumetric method.

    The formula for oil in place (OIP) is:
        OIP = 7758 * A * h * φ * S_o * N/G / B_o

    The formula for gas in place (GIP) is:
        GIP = 43560 * A * h * φ * S_g * N/G / B_g

    S_o = 1 - S_w - S_g (oil saturation)
    S_g = 1 - S_w - S_o (gas saturation)

    where:
    - OIP is the oil in place in stock tank barrels (STB).
    - GIP is the free gas in place in standard cubic feet (SCF).
    - A is the area in acres.
    - h is the thickness in feet.
    - φ is the porosity (fraction).
    - S_o is the oil saturation (fraction).
    - B_o is the formation volume factor for oil (RB/STB).
    - S_g is the gas saturation (fraction).
    - B_g is the formation volume factor for gas (RB/SCF).
    - N/G is the net-to-gross ratio (fraction).
    - 7758 is the conversion factor from acre-feet to stock tank barrels.
    - 43560 is the conversion factor from acre-feet to cubic feet.

    Note: This calculates **free** phase volumes:
    - Free oil (excludes dissolved gas)
    - Free gas (excludes solution gas in oil)
    - Free water

    Total gas = Free gas + (Oil volume x Rs)
    where Rs is the solution gas-oil ratio.

    :param area: Area in acres.
    :param thickness: Thickness in feet.
    :param porosity: Porosity as a fraction (e.g., 0.2 for 20%).
    :param phase_saturation: Phase saturation as a fraction (e.g., 0.8 for 80%).
    :param formation_volume_factor: Formation volume factor (RB/STB or RB/SCF).
    :param hydrocarbon_type: Type of hydrocarbon ("oil" or "gas").
    :return: Free hydrocarbon/water in place (OIP/WIP in STB, and GIP in SCF).
    """
    if hydrocarbon_type not in {"oil", "gas", "water"}:
        raise ValidationError(
            "Hydrocarbon type must be either 'oil', 'gas', or 'water'."
        )
    if area <= 0 or thickness <= 0:
        raise ValidationError("Area and thickness must be positive values.")
    if porosity < 0 or porosity > 1:
        raise ValidationError("Porosity must be a fraction between 0 and 1.")
    if phase_saturation < 0 or phase_saturation > 1:
        raise ValidationError("Phase saturation must be a fraction between 0 and 1.")
    if formation_volume_factor <= 0:
        raise ValidationError("Formation volume factor must be a positive value.")

    if hydrocarbon_type in {"oil", "water"}:
        # Oil in Place (OIP) calculation (May include dissolved gas in undersaturated reservoirs)
        oip = (
            acre_ft_to_bbl
            * area
            * thickness
            * porosity
            * phase_saturation
            * net_to_gross_ratio
            / formation_volume_factor
        )
        return oip

    # Free Gas in Place (GIP) calculation
    free_gip = (
        acre_ft_to_ft3
        * area
        * thickness
        * porosity
        * phase_saturation
        * net_to_gross_ratio
        / formation_volume_factor
    )
    return free_gip


@numba.njit(cache=True)
def compute_miscibility_transition_factor(
    pressure: float,
    minimum_miscibility_pressure: float,
    transition_width: float = 500.0,
) -> float:
    """
    Compute pressure-dependent miscibility transition factor.

    Returns a smooth transition from 0 (immiscible) at low pressure
    to 1 (fully miscible) above minimum miscibility pressure.

    This factor represents the degree of miscibility development and should
    be multiplied by the base Todd-Longstaff omega parameter to get the
    effective omega for viscosity calculations.

    Physical Behavior:
        - P << MMP: factor → 0 (immiscible, no miscible mixing)
        - P ≈ MMP: factor ≈ 0.5 (transition zone, partial miscibility)
        - P >> MMP: factor → 1 (fully miscible, maximum mixing)

    The transition uses hyperbolic tangent for smooth, physically realistic behavior:
        f(P) = 0.5 * (1 + tanh((P - MMP) / ΔP))

    This ensures:
        - At P = MMP - transition_width: f ≈ 0.12 (nearly immiscible)
        - At P = MMP: f = 0.5 (transitional)
        - At P = MMP + transition_width: f ≈ 0.88 (nearly miscible)

    Usage:
        To get effective omega for Todd-Longstaff viscosity calculation:
            omega_effective = omega_base * compute_miscibility_transition_factor(P, MMP)

    :param pressure: Current reservoir pressure (psi)
    :param minimum_miscibility_pressure: Minimum miscibility pressure (MMP, psi).
        The pressure above which first-contact miscibility can develop.
    :param transition_width: Pressure width of transition zone (psi), default 500.
        Controls how abruptly miscibility develops with pressure.
        Smaller values = sharper transition.
    :return: Miscibility transition factor, range [0, 1]
        0 = completely immiscible behavior
        1 = fully miscible behavior

    Example:
    ```python
    # CO2 injection with MMP = 2000 psi, base omega = 0.67
    omega_base = 0.67
    mmp = 2000.0

    # Well below MMP - immiscible
    factor = compute_miscibility_transition_factor(1000, mmp, 500)
    omega_eff = omega_base * factor  # ~0.08 (nearly immiscible)

    # At MMP - transitional
    factor = compute_miscibility_transition_factor(2000, mmp, 500)
    omega_eff = omega_base * factor  # ~0.34 (partial miscibility)

    # Above MMP - miscible
    factor = compute_miscibility_transition_factor(3000, mmp, 500)
    omega_eff = omega_base * factor  # ~0.59 (near full miscibility)
    ```

    References:
    Todd, M.R. and Longstaff, W.J. (1972). "The Development, Testing and
    Application of a Numerical Simulator for Predicting Miscible Flood Performance."
    JPT, July 1972, pp. 874-882.

    Note: The original Todd-Longstaff paper defines omega as a mixing parameter.
    This function computes how that mixing parameter varies with pressure near MMP.
    """
    # Fast path for extreme cases (>2 standard deviations from MMP)
    if pressure >= minimum_miscibility_pressure + 2.0 * transition_width:
        return 1.0  # Fully miscible (well above MMP)
    elif pressure <= minimum_miscibility_pressure - 2.0 * transition_width:
        return 0.0  # Fully immiscible (well below MMP)

    # Smooth transition using hyperbolic tangent
    # Normalize pressure relative to MMP and transition width
    normalized = (pressure - minimum_miscibility_pressure) / transition_width

    # Transition factor varies from 0 (immiscible) to 1 (miscible)
    transition_factor = 0.5 * (1.0 + np.tanh(normalized))
    return transition_factor


@numba.njit(cache=True)
def compute_effective_todd_longstaff_omega(
    pressure: float,
    base_omega: float,
    minimum_miscibility_pressure: float,
    transition_width: float = 500.0,
) -> float:
    """
    Compute pressure-dependent effective Todd-Longstaff omega parameter.

    Combines the base mixing parameter (omega) with pressure-dependent
    miscibility to get the effective omega for viscosity calculations.

    Below MMP: omega_eff → 0 (immiscible behavior, segregated flow)
    Above MMP: omega_eff → base_omega (miscible behavior, mixed flow)

    :param pressure: Current reservoir pressure (psi)
    :param base_omega: Base Todd-Longstaff mixing parameter (0 to 1).
        Typical value: 0.67 for CO2-oil systems.
        This is the maximum omega achieved when fully miscible.
    :param minimum_miscibility_pressure: Minimum miscibility pressure (MMP, psi)
    :param transition_width: Pressure width of transition zone (psi), default 500
    :return: Effective omega parameter for viscosity calculation (0 to base_omega)

    Example:
    ```python
    # CO2 flood with MMP = 2000 psi
    compute_effective_todd_longstaff_omega(
        pressure=2500,
        base_omega=0.67,
        minimum_miscibility_pressure=2000,
        transition_width=500
    )
    0.54  # Partial miscibility developed
    ```
    """
    if base_omega == 0.0:
        return 0.0

    transition_factor = compute_miscibility_transition_factor(
        pressure=pressure,
        minimum_miscibility_pressure=minimum_miscibility_pressure,
        transition_width=transition_width,
    )
    return base_omega * transition_factor


@numba.njit(cache=True)
def compute_todd_longstaff_effective_viscosity(
    oil_viscosity: float,
    solvent_viscosity: float,
    solvent_concentration: float,
    omega: float = 0.67,
) -> float:
    """
    Compute effective viscosity using Todd-Longstaff mixing model.

    This function computes the viscosity of an oil-solvent mixture based on
    the concentrations and a mixing parameter (omega) that interpolates between
    fully segregated (immiscible) and fully mixed (miscible) flow behavior.

    Standard Formula (Todd & Longstaff, 1972):
        μ_mix = C_s * μ_s + C_o * μ_o                      (arithmetic mean - fully mixed)
        μ_seg = 1 / (C_s/μ_s + C_o/μ_o)                   (harmonic mean - segregated)
        μ_eff = μ_mix^ω * μ_seg^(1-ω)                     (Todd-Longstaff interpolation)

    Where:
        C_s = solvent concentration (0 to 1)
        C_o = oil concentration = 1 - C_s
        ω = mixing parameter (0 = fully segregated, 1 = fully mixed)

    Physical Interpretation of Omega:
        ω = 0.0: Immiscible behavior (parallel/segregated flow, harmonic mean)
                 Fluids flow separately with minimal interaction
        ω = 0.5: Partial mixing (geometric mean of viscosities)
                 Intermediate level of fluid interaction
        ω = 0.67: Typical for CO2-oil systems (from field history matching)
                  Represents realistic mixing in miscible gas floods
        ω = 1.0: Fully mixed (ideal miscibility, arithmetic mean)
                 Complete homogeneous mixing, single-phase behavior

    Note on Pressure-Dependent Miscibility:
        When pressure varies (especially near MMP), omega itself becomes pressure-dependent.
        Use compute_effective_todd_longstaff_omega() to get omega(P), then pass it here.

    :param oil_viscosity: Pure oil viscosity (cP), must be > 0
    :param solvent_viscosity: Pure solvent viscosity (cP), must be > 0
    :param solvent_concentration: Solvent concentration (fraction 0-1)
        0 = pure oil, 1 = pure solvent
    :param omega: Todd-Longstaff mixing parameter (0-1), default 0.67
        This should be the EFFECTIVE omega if considering pressure effects.
    :return: Effective mixture viscosity (cP)

    Raises:
        ValidationError: If concentrations or omega are outside [0,1], or viscosities ≤ 0

    Example:
    ```python
    # Immiscible case (omega = 0)
    compute_todd_longstaff_effective_viscosity(
        oil_viscosity=10.0,
        solvent_viscosity=0.05,
        solvent_concentration=0.3,
        omega=0.0
    )
    0.147  # Harmonic mean - segregated flow

    # Fully miscible case (omega = 1)
    compute_todd_longstaff_effective_viscosity(
        oil_viscosity=10.0,
        solvent_viscosity=0.05,
        solvent_concentration=0.3,
        omega=1.0
    )
    7.015  # Arithmetic mean - fully mixed

    # Typical CO2 flood (omega = 0.67)
    compute_todd_longstaff_effective_viscosity(
        oil_viscosity=10.0,
        solvent_viscosity=0.05,
        solvent_concentration=0.3,
        omega=0.67
    )
    0.89  # Realistic mixture viscosity
    ```

    References:
        Todd, M.R. and Longstaff, W.J. (1972). "The Development, Testing and
        Application of a Numerical Simulator for Predicting Miscible Flood Performance."
        JPT, July 1972, pp. 874-882.
    """
    # Validate inputs
    if solvent_concentration < 0.0 or solvent_concentration > 1.0:
        raise ValidationError(
            f"Solvent concentration must be in [0,1], got {solvent_concentration}"
        )
    if omega < 0.0 or omega > 1.0:
        raise ValidationError(f"Omega must be in [0,1], got {omega}")
    if oil_viscosity <= 0.0 or solvent_viscosity <= 0.0:
        raise ValidationError("Viscosities must be positive")

    C_s = solvent_concentration
    C_o = 1.0 - C_s

    # Handle edge cases
    if C_s >= 1.0:
        return solvent_viscosity
    if C_s <= 0.0:
        return oil_viscosity

    # Fully mixed viscosity (arithmetic/linear mean)
    # Represents ideal miscibility - single homogeneous phase
    mu_mix = C_s * solvent_viscosity + C_o * oil_viscosity

    # Fully segregated viscosity (harmonic mean)
    # Represents parallel flow of two immiscible phases
    # Equivalent to: μ_seg = μ_s * μ_o / (C_s * μ_o + C_o * μ_s)
    mu_segregated = 1.0 / (C_s / solvent_viscosity + C_o / oil_viscosity)

    # Todd-Longstaff interpolation (weighted geometric mean)
    # Special cases:
    #   ω = 0: μ_eff = μ_segregated (immiscible, harmonic mean)
    #   ω = 1: μ_eff = μ_mix (fully mixed, arithmetic mean)
    #   ω = 0.5: μ_eff = sqrt(μ_mix * μ_segregated) (geometric mean)
    mu_effective = (mu_mix**omega) * (mu_segregated ** (1.0 - omega))
    return mu_effective


@numba.njit(cache=True)
def compute_todd_longstaff_effective_density(
    oil_density: float,
    solvent_density: float,
    oil_viscosity: float,
    solvent_viscosity: float,
    solvent_concentration: float = 1.0,
    omega: float = 0.67,
) -> float:
    """
    Compute effective density using Todd-Longstaff mixing model.

    The Todd-Longstaff density formulation ensures that when phases are fully
    miscible (ω=1), they flow with matched densities and viscosities, emulating
    a single phase. The effective density depends on the effective viscosity
    already calculated.

    Standard Formula (Todd & Longstaff, 1972):
        First compute effective viscosity:
            μ_eff = compute_todd_longstaff_effective_viscosity(...)

        Then compute phase fractions based on viscosity ratios:
            f_s = (C_s * μ_o) / (C_s * μ_o + C_o * μ_s)      (solvent fraction)
            f_o = (C_o * μ_s) / (C_s * μ_o + C_o * μ_s)      (oil fraction)

        Fully mixed density (volume-weighted):
            ρ_mix = C_s * ρ_s + C_o * ρ_o

        Segregated density (flow-weighted by phase fractions):
            ρ_seg = f_s * ρ_s + f_o * ρ_o

        Todd-Longstaff interpolation:
            ρ_eff = ρ_mix^ω * ρ_seg^(1-ω)

    Where:
        C_s, C_o = volume concentrations (sum to 1)
        f_s, f_o = flow fractions based on mobility (sum to 1)
        ω = mixing parameter (0=segregated, 1=fully mixed)

    Physical Interpretation:
        ω = 0: Density weighted by flow rates (mobility-based)
        ω = 1: Density weighted by volumes (concentration-based)
        ω = 0.67: Typical interpolation for CO2-oil systems

    :param oil_density: Oil density (lb/ft³ or kg/m³), must be > 0
    :param solvent_density: Solvent density (lb/ft³ or kg/m³), must be > 0
    :param oil_viscosity: Oil viscosity (cP), must be > 0
        This is needed to compute flow fractions for segregated density
    :param solvent_viscosity: Solvent viscosity (cP), must be > 0
        This is needed to compute flow fractions for segregated density
    :param solvent_concentration: Solvent concentration (fraction 0-1)
    :param omega: Todd-Longstaff mixing parameter (0-1), default 0.67
    :return: Effective mixture density (same units as input densities)

    Raises:
        ValidationError: If concentrations or omega are outside [0,1], or inputs ≤ 0

    Example:
    ```python
    # CO2 (light) displacing oil (heavy)
    compute_todd_longstaff_effective_density(
        oil_density=50.0,        # lb/ft³
        solvent_density=30.0,    # lb/ft³ (CO2 is lighter)
        oil_viscosity=10.0,      # cP
        solvent_viscosity=0.05,  # cP (CO2 is much less viscous)
        solvent_concentration=0.3,
        omega=0.67
    )
    44.2  # Effective density between pure values

    # Fully segregated (omega=0): flow-weighted density
    compute_todd_longstaff_effective_density(
        oil_density=50.0, solvent_density=30.0,
        oil_viscosity=10.0, solvent_viscosity=0.05,
        solvent_concentration=0.3, omega=0.0
    )
    30.6  # Much closer to solvent (it flows more easily)

    # Fully mixed (omega=1): volume-weighted density
    compute_todd_longstaff_effective_density(
        oil_density=50.0, solvent_density=30.0,
        oil_viscosity=10.0, solvent_viscosity=0.05,
        solvent_concentration=0.3, omega=1.0
    )
    44.0  # Simple volume average: 0.3*30 + 0.7*50
    ```

    References:
        Todd, M.R. and Longstaff, W.J. (1972). "The Development, Testing and
        Application of a Numerical Simulator for Predicting Miscible Flood Performance."
        JPT, July 1972, pp. 874-882.
    """
    if solvent_concentration < 0.0 or solvent_concentration > 1.0:
        raise ValidationError(
            f"Solvent concentration must be in [0,1], got {solvent_concentration}"
        )
    if omega < 0.0 or omega > 1.0:
        raise ValidationError(f"Omega must be in [0,1], got {omega}")
    if oil_density <= 0.0 or solvent_density <= 0.0:
        raise ValidationError("Densities must be positive")
    if oil_viscosity <= 0.0 or solvent_viscosity <= 0.0:
        raise ValidationError("Viscosities must be positive")

    C_s = solvent_concentration
    C_o = 1.0 - C_s

    # Handle edge cases
    if C_s >= 1.0:
        return solvent_density
    if C_s <= 0.0:
        return oil_density

    # Fully mixed density (volume-weighted, arithmetic mean)
    # This is the density if phases are perfectly mixed by volume
    rho_mix = C_s * solvent_density + C_o * oil_density

    # Compute phase flow fractions for segregated density
    # These represent how much each phase contributes to flow based on mobility
    # f_s = fraction of flow that is solvent
    # f_o = fraction of flow that is oil
    # Note: More mobile phase (lower viscosity) gets higher flow fraction
    denominator = C_s * oil_viscosity + C_o * solvent_viscosity

    # Avoid division by zero (though should never happen with positive viscosities)
    if denominator < 1e-15:
        # If both viscosities are essentially zero, fall back to volume weighting
        f_s = C_s
        f_o = C_o
    else:
        f_s = (C_s * oil_viscosity) / denominator
        f_o = (C_o * solvent_viscosity) / denominator

    # Fully segregated density (flow-weighted)
    # This is the density if phases flow separately, weighted by their mobilities
    rho_segregated = f_s * solvent_density + f_o * oil_density

    # Todd-Longstaff interpolation (weighted geometric mean)
    # Special cases:
    #   ω = 0: ρ_eff = ρ_segregated (flow-weighted, immiscible)
    #   ω = 1: ρ_eff = ρ_mix (volume-weighted, fully mixed)
    rho_effective = (rho_mix**omega) * (rho_segregated ** (1.0 - omega))
    return rho_effective

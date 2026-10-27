import functools
import logging
import typing

import numba  # type: ignore[import-untyped]
import numpy as np

from bores.constants import c
from bores.errors import ValidationError
from bores.types import NDimension, Number, NumberOrArray

logger = logging.getLogger(__name__)

__all__ = [
    "GAS_ALIASES",
    "HENRY_CONSTANTS",
    "SETSCHENOW_CONSTANTS",
    "PropsSI",
    "clip_pressure",
    "clip_temperature",
    "compute_fluid_compressibility",
    "compute_fluid_compressibility_factor",
    "compute_fluid_density",
    "compute_fluid_in_place",
    "compute_fluid_viscosity",
    "compute_total_fluid_compressibility",
    "evaluate_coolprop",
    "fahrenheit_to_celsius",
    "fahrenheit_to_kelvin",
    "fahrenheit_to_rankine",
    "get_gas_symbol",
    "is_CoolProp_fluid",
    "kelvin_to_fahrenheit",
    "validate_input_pressure",
    "validate_input_temperature",
]


def PropsSI(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """
    Wrapper for `CoolProp.CoolProp.PropsSI`.

    This helper lazily imports CoolProp and forwards all positional and
    keyword arguments to `CoolProp.CoolProp.PropsSI`.

    :raises ImportError: If CoolProp is not installed. In that case, users should
        install `bores-framework[coolprop]`.
    """
    try:
        from CoolProp.CoolProp import (  # type: ignore[import, import-untyped]
            PropsSI as CoolPropPropsSI,
        )
    except ImportError as exc:
        raise ImportError(
            "CoolProp is required for this operation. Install "
            "`bores-framework[coolprop]` to enable CoolProp support. "
            "Run `uv add 'bores-framework[coolprop]'` or "
            "`pip install 'bores-framework[coolprop]'` to install."
        ) from exc
    return CoolPropPropsSI(*args, **kwargs)


def validate_input_temperature(temperature: NumberOrArray[NDimension]) -> None:
    """
    Validates that the input temperature(s) are within the valid/reservoir-like range.

    :param temperature: Temperature(s) in °F. Scalar or array.
    :raises ValidationError: If any temperature is outside the valid range.
    """
    temperature_array = np.asarray(temperature)
    invalid_mask = (temperature_array < c.MINIMUM_VALID_TEMPERATURE) | (
        temperature_array > c.MAXIMUM_VALID_TEMPERATURE
    )
    if np.any(invalid_mask):
        raise ValidationError(
            f"Temperature(s) out of valid range [{c.MINIMUM_VALID_TEMPERATURE}, "
            f"{c.MAXIMUM_VALID_TEMPERATURE}] °F: {temperature_array[invalid_mask]}"
        )


def validate_input_pressure(pressure: NumberOrArray[NDimension]) -> None:
    """
    Validates that the input pressure(s) are within the valid/reservoir-like range.

    :param pressure: Pressure(s) in psi. Scalar or array.
    :raises ValidationError: If any pressure is outside the valid range.
    """
    pressure_array = np.asarray(pressure)
    invalid_mask = (pressure_array < c.MINIMUM_VALID_PRESSURE) | (
        pressure_array > c.MAXIMUM_VALID_PRESSURE
    )
    if np.any(invalid_mask):
        raise ValidationError(
            f"Pressure(s) out of valid range [{c.MINIMUM_VALID_PRESSURE}, "
            f"{c.MAXIMUM_VALID_PRESSURE}] psi: {pressure_array[invalid_mask]}"
        )


@functools.lru_cache(maxsize=64)
def is_CoolProp_fluid(fluid: str) -> bool:
    """
    Checks whether CoolProp supports a fluid.

    :param fluid: Fluid name (e.g., "CO2", "Water")
    :return: True if the fluid is supported, False otherwise.
    """
    try:
        return PropsSI("D", "T", 300, "P", 101325, fluid) is not None
    except ValueError:
        return False


def clip_pressure(pressure: NumberOrArray[NDimension], fluid: str) -> NumberOrArray[NDimension]:
    """
    Clips pressure to CoolProp's valid pressure range for a fluid.

    :param pressure: Pressure in Pascals
    :param fluid: CoolProp fluid name
    :return: Clipped pressure in Pascals
    """
    p_min = PropsSI("P_MIN", fluid)
    p_max = PropsSI("P_MAX", fluid)
    return np.minimum(np.maximum(pressure, p_min + 1.0), p_max - 1.0)  # type: ignore[return-value]


def clip_temperature(
    temperature: NumberOrArray[NDimension], fluid: str
) -> NumberOrArray[NDimension]:
    """
    Clips temperature to CoolProp's valid temperature range for a fluid.

    :param temperature: Temperature in Kelvin
    :param fluid: CoolProp fluid name
    :return: Clipped temperature in Kelvin
    """
    t_min = PropsSI("T_MIN", fluid)
    t_max = PropsSI("T_MAX", fluid)
    return np.minimum(np.maximum(temperature, t_min + 0.1), t_max - 0.1)  # type: ignore[return-value]


@numba.njit(cache=True)
def kelvin_to_fahrenheit(
    temp_K: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
    """Converts temperature from Kelvin to Fahrenheit."""
    return (temp_K - 273.15) * 9 / 5 + 32  # type: ignore[return-value]


@numba.njit(cache=True)
def fahrenheit_to_kelvin(
    temp_F: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
    """Converts temperature from Fahrenheit to Kelvin."""
    return (temp_F - 32) * 5 / 9 + 273.15  # type: ignore[return-value]


@numba.njit(cache=True)
def fahrenheit_to_celsius(
    temp_F: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
    """Converts temperature from Fahrenheit to Celsius."""
    return (temp_F - 32) * 5 / 9  # type: ignore[return-value]


@numba.njit(cache=True)
def fahrenheit_to_rankine(
    temp_F: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
    """Converts temperature from Fahrenheit to Rankine."""
    return temp_F + 459.67  # type: ignore[return-value]


HENRY_CONSTANTS: dict[str, tuple[float, float]] = {
    "co2": (3.3e-4, 2400.0),
    "ch4": (1.4e-5, 1600.0),
    "n2": (6.4e-6, 1600.0),
    "o2": (1.3e-5, 1500.0),
    "ar": (1.4e-5, 1700.0),
    "he": (3.8e-6, 230.0),
    "h2": (7.8e-6, 530.0),
}
"""
Henry's law solubility constants in pure water (Sander, 2015). Each gas maps to
`(kH, C)`, where `kH` is the solubility at 298.15 K in mol/(m³·Pa) and `C` is
`d(ln kH)/d(1/T)` in K, so that `kH(T) = kH * exp(C * (1/T - 1/298.15))`.
"""

SETSCHENOW_CONSTANTS: dict[str, float] = {
    "co2": 0.12,
    "ch4": 0.11,
    "n2": 0.13,
    "o2": 0.13,
    "ar": 0.10,
    "he": 0.08,
    "h2": 0.07,
}
"""
Approximate Setschenow salting-out constants for NaCl, per mol/kg of salt, in
base 10: `log10(S_pure / S_brine) = constant * salt molality`.
"""

GAS_ALIASES: dict[str, str] = {
    "methane": "ch4",
    "carbondioxide": "co2",
    "nitrogen": "n2",
    "oxygen": "o2",
    "argon": "ar",
    "helium": "he",
    "hydrogen": "h2",
}


def get_gas_symbol(gas_name: str) -> str:
    """
    Normalizes a gas name or formula to the lowercase symbol used by the solubility models.

    :param gas_name: Gas name or formula (e.g., "Methane", "CO2", "carbon dioxide").
    :return: Lowercase gas symbol (e.g., "ch4", "co2").
    """
    gas_name = gas_name.lower().replace(" ", "").replace("-", "")
    return GAS_ALIASES.get(gas_name, gas_name)


def evaluate_coolprop(
    output: str,
    pressure: NumberOrArray[NDimension],
    temperature: NumberOrArray[NDimension],
    fluid: str,
) -> NumberOrArray[NDimension]:
    """
    Evaluates a CoolProp property at pressures (psi) and temperatures (°F).

    Pressure and temperature broadcast against each other. Temperatures are clipped
    to the fluid's valid range and pressures to its maximum pressure. Pressures are
    not raised to CoolProp's minimum pressure, which is the triple point pressure and
    for gases such as CO₂ lies far above atmospheric pressure, where the gas phase
    is still well defined.

    :param output: CoolProp output key (e.g., "D", "V", "Z").
    :param pressure: Pressure in psi. Scalar or array.
    :param temperature: Temperature in °F. Scalar or array.
    :param fluid: CoolProp fluid name.
    :return: The property in CoolProp's SI units, with the broadcast shape of the inputs.
    """
    pressure_in_pascals, temperature_in_kelvin = np.broadcast_arrays(
        np.asarray(pressure, dtype=np.float64) * c.PSI_TO_PASCAL,
        fahrenheit_to_kelvin(np.asarray(temperature, dtype=np.float64)),
    )
    shape = pressure_in_pascals.shape
    values = PropsSI(
        output,
        "P",
        np.clip(pressure_in_pascals.ravel(), 1.0, PropsSI("P_MAX", fluid) - 1.0),
        "T",
        clip_temperature(temperature_in_kelvin.ravel(), fluid),
        fluid,
    )
    values = np.asarray(values, dtype=np.float64).reshape(shape)
    if isinstance(pressure, np.ndarray) and pressure.dtype == np.float32:
        values = values.astype(np.float32, copy=False)
    return typing.cast(NumberOrArray[NDimension], values if values.ndim else values[()])


def compute_fluid_density(
    pressure: NumberOrArray[NDimension], temperature: NumberOrArray[NDimension], fluid: str
) -> NumberOrArray[NDimension]:
    """
    Computes fluid density from an equation of state using CoolProp.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: Fluid name supported by CoolProp (e.g., "CO2", "Water")
    :return: Density in lbm/ft³
    """
    density = evaluate_coolprop("D", pressure, temperature, fluid)
    return density * c.KILOGRAM_PER_CUBIC_METER_TO_POUNDS_PER_CUBIC_FEET


def compute_fluid_viscosity(
    pressure: NumberOrArray[NDimension], temperature: NumberOrArray[NDimension], fluid: str
) -> NumberOrArray[NDimension]:
    """
    Computes fluid dynamic viscosity from an equation of state using CoolProp.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: Fluid name supported by CoolProp (e.g., "CO2", "Water")
    :return: Viscosity in centipoise (cP)
    """
    viscosity = evaluate_coolprop("V", pressure, temperature, fluid)
    return viscosity * c.PASCAL_SECONDS_TO_CENTIPOISE


def compute_fluid_compressibility_factor(
    pressure: NumberOrArray[NDimension], temperature: NumberOrArray[NDimension], fluid: str
) -> NumberOrArray[NDimension]:
    """
    Computes the compressibility factor Z of a fluid from an equation of state using CoolProp.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: Fluid name supported by CoolProp (e.g., "CO2", "Methane")
    :return: Compressibility factor Z (dimensionless)
    """
    return evaluate_coolprop("Z", pressure, temperature, fluid)


def compute_fluid_compressibility(
    pressure: NumberOrArray[NDimension], temperature: NumberOrArray[NDimension], fluid: str
) -> NumberOrArray[NDimension]:
    """
    Computes the isothermal compressibility of a fluid, `-(1/V) * dV/dP` at constant
    temperature, from an equation of state using CoolProp.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: Fluid name supported by CoolProp (e.g., "n-Octane")
    :return: Compressibility in psi⁻¹
    """
    compressibility = evaluate_coolprop("ISOTHERMAL_COMPRESSIBILITY", pressure, temperature, fluid)
    return compressibility * c.PSI_TO_PASCAL


def compute_total_fluid_compressibility(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    water_compressibility: NumberOrArray[NDimension],
    oil_compressibility: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension] | None = None,
    gas_compressibility: NumberOrArray[NDimension] | None = None,
) -> NumberOrArray[NDimension]:
    """
    Computes the total fluid compressibility as the saturation-weighted sum of the
    phase compressibilities.

    :param water_saturation: Water saturation (fraction).
    :param oil_saturation: Oil saturation (fraction).
    :param water_compressibility: Compressibility of the water phase (psi⁻¹).
    :param oil_compressibility: Compressibility of the oil phase (psi⁻¹).
    :param gas_saturation: Gas saturation (fraction), for three-phase systems.
    :param gas_compressibility: Gas compressibility (psi⁻¹), for three-phase systems.
    :return: Total fluid compressibility (psi⁻¹).
    """
    total = water_saturation * water_compressibility + oil_saturation * oil_compressibility
    if gas_saturation is not None and gas_compressibility is not None:
        total = total + gas_saturation * gas_compressibility
    return typing.cast(NumberOrArray[NDimension], total)


def compute_fluid_in_place(
    area: NumberOrArray[NDimension],
    thickness: NumberOrArray[NDimension],
    porosity: NumberOrArray[NDimension],
    phase_saturation: NumberOrArray[NDimension],
    formation_volume_factor: NumberOrArray[NDimension],
    net_to_gross_ratio: NumberOrArray[NDimension] = 1.0,
    hydrocarbon_type: typing.Literal["oil", "gas", "water"] = "oil",
    acre_ft_to_bbl: Number = 7758.0,
    acre_ft_to_ft3: Number = 43560.0,
) -> NumberOrArray[NDimension]:
    """
    Computes the free oil, free gas or free water in place with the volumetric method.

    Oil and water in place, in stock tank barrels:

        N = 7758 * A * h * φ * S * N/G / B

    Free gas in place, in standard cubic feet:

        G = 43560 * A * h * φ * S * N/G / Bg

    where A is the area (acres), h the thickness (ft), φ the porosity, S the phase
    saturation, N/G the net-to-gross ratio, B the oil or water formation volume
    factor (RB/STB) and Bg the gas formation volume factor (ft³/scf).

    Only free phase volumes are computed. Gas dissolved in the oil is not included;
    it is the oil in place multiplied by the solution gas-oil ratio.

    :param area: Area in acres.
    :param thickness: Thickness in feet.
    :param porosity: Porosity (fraction).
    :param phase_saturation: Saturation of the phase (fraction).
    :param formation_volume_factor: Formation volume factor of the phase (RB/STB for oil
        and water, ft³/scf for gas).
    :param net_to_gross_ratio: Net-to-gross ratio (fraction).
    :param hydrocarbon_type: Phase to compute, "oil", "gas" or "water".
    :param acre_ft_to_bbl: Barrels per acre-foot.
    :param acre_ft_to_ft3: Cubic feet per acre-foot.
    :return: Oil or water in place (STB), or free gas in place (SCF).
    :raises ValidationError: If an input is outside its valid range.
    """
    if hydrocarbon_type not in {"oil", "gas", "water"}:
        raise ValidationError("Hydrocarbon type must be either 'oil', 'gas', or 'water'.")
    if np.min(area) <= 0 or np.min(thickness) <= 0:
        raise ValidationError("Area and thickness must be positive values.")
    if np.min(porosity) < 0 or np.max(porosity) > 1:
        raise ValidationError("Porosity must be a fraction between 0 and 1.")
    if np.min(phase_saturation) < 0 or np.max(phase_saturation) > 1:
        raise ValidationError("Phase saturation must be a fraction between 0 and 1.")
    if np.min(net_to_gross_ratio) < 0 or np.max(net_to_gross_ratio) > 1:
        raise ValidationError("Net-to-gross ratio must be a fraction between 0 and 1.")
    if np.min(formation_volume_factor) <= 0:
        raise ValidationError("Formation volume factor must be a positive value.")

    unit_factor = acre_ft_to_ft3 if hydrocarbon_type == "gas" else acre_ft_to_bbl
    return typing.cast(
        NumberOrArray[NDimension],
        (
            unit_factor
            * area
            * thickness
            * porosity
            * phase_saturation
            * net_to_gross_ratio
            / formation_volume_factor
        ),
    )

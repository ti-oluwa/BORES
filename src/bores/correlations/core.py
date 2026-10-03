import functools
import logging
import typing

import numba  # type: ignore[import-untyped]
import numpy as np

from bores.constants import c
from bores.errors import ValidationError
from bores.types import NDimension, NumberOrArray

logger = logging.getLogger(__name__)

__all__ = [
    "fahrenheit_to_celsius",
    "fahrenheit_to_kelvin",
    "fahrenheit_to_rankine",
    "kelvin_to_fahrenheit",
]


def PropsSI(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """
    Wrapper for `CoolProp.CoolProp.PropsSI`.

    This helper lazily imports CoolProp and forwards all positional and
    keyword arguments to `CoolProp.CoolProp.PropsSI`.

    Raises:
        ImportError: If CoolProp is not installed. In that case, users should
            install `bores-framework[coolprop]`.
    """
    try:
        from CoolProp.CoolProp import (  # type: ignore[import, import-untyped]
            PropsSI as CoolPropPropsSI,
        )
    except ImportError as exc:
        raise ImportError(
            "CoolProp is required for this operation. Install "
            "`bores-framework[coolprop]` to enable CoolProp support."
            "Run `uv add 'bores-framework[coolprop]' or `pip install 'bores-framework[coolprop]' to install."
        ) from exc
    return CoolPropPropsSI(*args, **kwargs)


def validate_input_temperature(temperature: NumberOrArray[NDimension]) -> None:
    """
    Validates that the input temperature(s) are within valid/reservoir-like range.

    Accepts scalar or ndarray input.

    :param temperature: Temperature(s) in Kelvin (°F)
    :raises ValidationError: If any temperature is outside the valid range.
    """
    temperature_arr = np.asarray(temperature)
    invalid_mask = (temperature_arr < c.MINIMUM_VALID_TEMPERATURE) | (
        temperature_arr > c.MAXIMUM_VALID_TEMPERATURE
    )

    if np.any(invalid_mask):
        invalid: np.ndarray = temperature_arr[invalid_mask]
        raise ValidationError(
            f"Temperature(s) out of valid range [{c.MINIMUM_VALID_TEMPERATURE}, {c.MAXIMUM_VALID_TEMPERATURE}] K: "
            f"{invalid}"
        )


def validate_input_pressure(pressure: NumberOrArray[NDimension]) -> None:
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
def is_CoolProp_fluid(fluid: str) -> bool:
    """
    Check if the fluid is supported by CoolProp.

    :param fluid: str name (must be supported by CoolProp, e.g., "CO2", "Water")
    :return: True if the fluid is supported, False otherwise.
    """
    return PropsSI("D", "T", 300, "P", 101325, fluid) is not None


def clip_pressure(pressure: NumberOrArray[NDimension], fluid: str) -> NumberOrArray[NDimension]:
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


def clip_temperature(
    temperature: NumberOrArray[NDimension], fluid: str
) -> NumberOrArray[NDimension]:
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


# Henry's constants (Sander, 2020) - ln(H) = A + B/T + C*ln(T)
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

GAS_ALIASES = {
    "methane": "ch4",
    "carbondioxide": "co2",
    "nitrogen": "n2",
    "oxygen": "o2",
    "argon": "ar",
    "helium": "he",
    "hydrogen": "h2",
}


def get_gas_symbol(gas_name: str) -> str:
    gas_name = gas_name.lower().replace(" ", "").replace("-", "")
    return GAS_ALIASES.get(gas_name, gas_name)

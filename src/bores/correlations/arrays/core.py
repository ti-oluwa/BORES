import logging
import typing

import numba
import numpy as np

from bores.constants import c
from bores.correlations.core import (
    PropsSI,
    clip_pressure,
    clip_temperature,
    fahrenheit_to_kelvin,
)
from bores.errors import ValidationError
from bores.types import NDimension, Number, NumberArray, NumberOrArray
from bores.utils import max_, min_

logger = logging.getLogger(__name__)

__all__ = [
    "compute_fluid_compressibility",
    "compute_fluid_compressibility_factor",
    "compute_fluid_density",
    "compute_fluid_viscosity",
    "compute_hydrocarbon_in_place",
    "compute_total_fluid_compressibility",
]


def compute_fluid_density(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    fluid: str,
) -> NumberArray[NDimension]:
    """
    Compute fluid density from EOS using CoolProp.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: str name (must be supported by CoolProp, e.g., "CO2", "Water")
    :return: Density in lbm/ft³
    """
    dtype = pressure.dtype
    temperature_array = fahrenheit_to_kelvin(temperature)  # type: ignore[arg-type]
    pressure_array = np.multiply(pressure, c.PSI_TO_PASCAL, dtype=dtype)

    def _compute_density(pressure_in_pascals: Number, temperature_in_kelvin: Number, fluid: str):
        density: Number = PropsSI(
            "D",
            "P",
            clip_pressure(pressure_in_pascals, fluid),
            "T",
            clip_temperature(temperature_in_kelvin, fluid),
            fluid,
        )
        return density * c.KILOGRAM_PER_CUBIC_METER_TO_POUNDS_PER_CUBIC_FEET

    density_array = np.empty_like(pressure_array)
    for idx in np.ndindex(pressure_array.shape):
        density_array[idx] = _compute_density(
            pressure_in_pascals=pressure_array[idx],  # type: ignore
            temperature_in_kelvin=temperature_array[idx],  # type: ignore
            fluid=fluid,
        )
    return density_array  # type: ignore[return-value]


def compute_fluid_viscosity(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    fluid: str,
) -> NumberArray[NDimension]:
    """
    Compute fluid dynamic viscosity from EOS using CoolProp.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: str name (must be supported by CoolProp, e.g., "CO2", "Water")
    :return: Viscosity in centipoise (cP)
    """
    dtype = pressure.dtype
    temperature_array = fahrenheit_to_kelvin(temperature)  # type: ignore[arg-type]
    pressure_array = np.multiply(pressure, c.PSI_TO_PASCAL, dtype=dtype)

    def compute_viscosity(pressure_in_pascals, temperature_in_kelvin, fluid: str):
        viscosity = PropsSI(
            "V",
            "P",
            clip_pressure(pressure_in_pascals, fluid),
            "T",
            clip_temperature(temperature_in_kelvin, fluid),
            fluid,
        )
        return viscosity * c.PASCAL_SECONDS_TO_CENTIPOISE

    viscosity_array = np.empty_like(pressure_array)
    for idx in np.ndindex(pressure_array.shape):
        viscosity_array[idx] = compute_viscosity(
            pressure_in_pascals=pressure_array[idx],  # type: ignore
            temperature_in_kelvin=temperature_array[idx],  # type: ignore
            fluid=fluid,
        )
    return viscosity_array  # type: ignore[return-value]


def compute_fluid_compressibility_factor(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    fluid: str,
) -> NumberArray[NDimension]:
    """
    Compute fluid compressibility factor Z from EOS using CoolProp.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: str name (must be supported by CoolProp, e.g., "CO2", "Methane")
    :return: Compressibility factor Z (dimensionless)
    """
    dtype = pressure.dtype
    temperature_array = fahrenheit_to_kelvin(temperature)  # type: ignore[arg-type]
    pressure_array = np.multiply(pressure, c.PSI_TO_PASCAL, dtype=dtype)

    def compute_z_factor(pressure_in_pascals, temperature_in_kelvin, fluid: str):
        return PropsSI(
            "Z",
            "P",
            clip_pressure(pressure_in_pascals, fluid),
            "T",
            clip_temperature(temperature_in_kelvin, fluid),
            fluid,
        )

    z_array = np.empty_like(pressure_array)
    for idx in np.ndindex(pressure_array.shape):
        z_array[idx] = compute_z_factor(
            pressure_in_pascals=pressure_array[idx],
            temperature_in_kelvin=temperature_array[idx],  # type: ignore
            fluid=fluid,
        )
    return z_array  # type: ignore[return-value]


def compute_fluid_compressibility(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    fluid: str,
) -> NumberArray[NDimension]:
    """
    Computes the isothermal compressibility of a fluid at a given pressure and temperature.

    Compressibility is defined as:

        C_f = -(1/ρ) * (dρ/dP) at constant temperature

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: str name supported by CoolProp (e.g., 'n-Octane')
    :return: Compressibility in psi⁻¹
    """
    dtype = pressure.dtype
    temperature_array = fahrenheit_to_kelvin(temperature)  # type: ignore[arg-type]
    pressure_array = np.multiply(pressure, c.PSI_TO_PASCAL, dtype=dtype)

    def compute_compressibility(pressure_in_pascals, temperature_in_kelvin, fluid: str):
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

    compressibility_array = np.empty_like(pressure_array)
    for idx in np.ndindex(pressure_array.shape):
        compressibility_array[idx] = compute_compressibility(
            pressure_in_pascals=pressure_array[idx],
            temperature_in_kelvin=temperature_array[idx],  # type: ignore
            fluid=fluid,
        )
    return compressibility_array  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_total_fluid_compressibility(
    water_saturation: NumberArray[NDimension],
    oil_saturation: NumberArray[NDimension],
    water_compressibility: NumberArray[NDimension],
    oil_compressibility: NumberArray[NDimension],
    gas_saturation: NumberArray[NDimension] | None = None,
    gas_compressibility: NumberArray[NDimension] | None = None,
) -> NumberArray[NDimension]:
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
    return total_fluid_compressibility  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_hydrocarbon_in_place(
    area: NumberArray[NDimension],
    thickness: NumberArray[NDimension],
    porosity: NumberArray[NDimension],
    phase_saturation: NumberArray[NDimension],
    formation_volume_factor: NumberArray[NDimension],
    net_to_gross_ratio: NumberOrArray[NDimension] = 1.0,
    hydrocarbon_type: typing.Literal["oil", "gas", "water"] = "oil",
    acre_ft_to_bbl: NumberOrArray[NDimension] = 7758.0,
    acre_ft_to_ft3: NumberOrArray[NDimension] = 43560.0,
) -> NumberArray[NDimension]:
    """
    Computes the (free) hydrocarbon (or free water) in place (HCIP or FWIP) in stock tank barrels (STB) or standard cubic feet (SCF)
    using the volumetric method.

    The formula for oil in place (OIP) is:
        OIP = 7758 * A * h * φ * S_o * N/G / B_o

    The formula for gas in place (GIP) is:
        GIP = 43560 * A * h * φ * S_g * N/G / B_g

    S_o = 1 - S_w - S_g (oil saturation)
    S_g = 1 - S_w - S_o (gas saturation)

    Where:
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
        raise ValidationError("Hydrocarbon type must be either 'oil', 'gas', or 'water'.")
    if min_(area) <= 0 or min_(thickness) <= 0:
        raise ValidationError("Area and thickness must be positive values.")
    if min_(porosity) < 0 or max_(porosity) > 1:
        raise ValidationError("Porosity must be a fraction between 0 and 1.")
    if min_(phase_saturation) < 0 or max_(phase_saturation) > 1:
        raise ValidationError("Phase saturation must be a fraction between 0 and 1.")
    if min_(formation_volume_factor) <= 0:
        raise ValidationError("Formation volume factor must be a positive value.")

    dtype = area.dtype
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
        return oip.astype(dtype)  # type: ignore[return-value]

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
    return free_gip.astype(dtype)  # type: ignore[return-value]

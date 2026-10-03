import logging
import warnings

import numba  # type: ignore[import-untyped]
import numpy as np
from scipy.optimize import brentq  # type: ignore[import-untyped]

from bores.constants import c
from bores.correlations.scalars import oil as soil
from bores.errors import ValidationError
from bores.types import (
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
)
from bores.utils import apply_mask, clip, get_mask, max_, min_

logger = logging.getLogger(__name__)

__all__ = [
    "compute_base_compressibility",
    "compute_dead_oil_viscosity_modified_beggs",
    "compute_effective_todd_longstaff_omega",
    "compute_gas_to_oil_ratio",
    "compute_gas_to_oil_ratio_standing",
    "compute_live_oil_density",
    "compute_miscibility_transition_factor",
    "compute_oil_api_gravity",
    "compute_oil_bubble_point_pressure",
    "compute_oil_compressibility",
    "compute_oil_formation_volume_factor",
    "compute_oil_formation_volume_factor_standing",
    "compute_oil_formation_volume_factor_vazquez_and_beggs",
    "compute_oil_specific_gravity",
    "compute_oil_viscosity",
    "compute_todd_longstaff_effective_density",
    "compute_todd_longstaff_effective_viscosity",
    "correct_oil_fvf_for_pressure",
    "estimate_bubble_point_pressure_standing",
    "estimate_solution_gor",
]


def compute_oil_specific_gravity(
    oil_density: NumberArray[NDimension],
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    oil_compressibility: NumberArray[NDimension],
) -> NumberArray[NDimension]:
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

    where ρ_water is the density of water at standard conditions (assumed 62.4 lbm/ft³).

    :param oil_density: Oil density at reservoir conditions (lbm/ft³)
    :param pressure: Reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param oil_compressibility: Oil compressibility (psi⁻¹)
    :return: Specific gravity of oil (dimensionless)
    """
    dtype = pressure.dtype
    delta_p = np.subtract(c.STANDARD_PRESSURE_IMPERIAL, pressure, dtype=dtype)
    delta_t = np.subtract(c.STANDARD_TEMPERATURE_IMPERIAL, temperature, dtype=dtype)
    correction_factor = np.exp(
        (oil_compressibility * delta_p)
        + np.multiply(c.OIL_THERMAL_EXPANSION_COEFFICIENT_IMPERIAL, delta_t, dtype=dtype),
        dtype=dtype,
    )
    correction_factor = clip(
        correction_factor, 0.2, 2.0
    )  # Avoid numerical issues with small/large values
    oil_density_at_stp = np.multiply(oil_density, correction_factor, dtype=dtype)
    return np.divide(oil_density_at_stp, c.STANDARD_WATER_DENSITY_IMPERIAL, dtype=dtype)  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_oil_formation_volume_factor_standing(
    temperature: NumberArray[NDimension],
    oil_specific_gravity: NumberArray[NDimension],
    gas_gravity: NumberArray[NDimension],
    gas_to_oil_ratio: NumberArray[NDimension],
) -> NumberArray[NDimension]:
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
        - Valid for light soil and saturated conditions
        - Typical range: 60-300 °F, 0.5-0.95 oil SG, 20 - 2000 scf/STB

    :param temperature: Temperature (°F)
    :param oil_specific_gravity: Oil specific gravity (dimensionless)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :param gas_to_oil_ratio: Gas-to-oil ratio in SCF/STB
    :return: Formation volume factor (Bo) in bbl/STB
    """
    if min_(oil_specific_gravity) <= 0 or min_(gas_gravity) <= 0:
        raise ValidationError("Specific gravities must be positive.")
    if min_(gas_to_oil_ratio) < 0:
        raise ValidationError("Gas-to-oil ratio must be non-negative.")
    if min_(temperature) < 32:
        raise ValidationError("Temperature seems unphysical (<32 °F). Check units.")

    x = (gas_to_oil_ratio * (gas_gravity / oil_specific_gravity) ** 0.5) + (1.25 * temperature)
    oil_fvf = 0.972 + 0.000147 * (x**1.175)
    dtype = temperature.dtype
    return oil_fvf.astype(dtype)  # type: ignore[return-value]


@numba.njit(cache=True)
def get_vazquez_beggs_oil_fvf_coefficients(
    oil_api_gravity: NumberArray[NDimension],
) -> tuple[
    NumberArray[NDimension],
    NumberArray[NDimension],
    NumberArray[NDimension],
]:
    """
    Returns the coefficients a1, a2, a3 for the Vazquez and Beggs oil FVF correlation based on oil API gravity.
    """
    input_type = oil_api_gravity.dtype
    less_equal_30 = oil_api_gravity <= 30
    a1 = np.where(less_equal_30, 4.677e-4, 4.670e-4)
    a2 = np.where(less_equal_30, 1.751e-5, 1.100e-5)
    a3 = np.where(less_equal_30, -1.811e-8, 1.337e-9)
    return a1.astype(input_type), a2.astype(input_type), a3.astype(input_type)  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_oil_formation_volume_factor_vazquez_and_beggs(
    temperature: NumberArray[NDimension],
    oil_specific_gravity: NumberArray[NDimension],
    gas_gravity: NumberArray[NDimension],
    gas_to_oil_ratio: NumberArray[NDimension],
) -> NumberArray[NDimension]:
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
    dtype = oil_specific_gravity.dtype
    oil_api_gravity = compute_oil_api_gravity(oil_specific_gravity)
    a1, a2, a3 = get_vazquez_beggs_oil_fvf_coefficients(oil_api_gravity)
    oil_fvf = (
        1
        + (a1 * gas_to_oil_ratio)
        + (a2 * (temperature - 60) * (oil_specific_gravity / gas_gravity))
        + (a3 * (temperature - 60) * gas_to_oil_ratio * (oil_specific_gravity / gas_gravity))
    )
    return oil_fvf.astype(dtype)  # type: ignore[return-value]


@numba.njit(cache=True)
def correct_oil_fvf_for_pressure(
    saturated_oil_fvf: NumberArray[NDimension],
    oil_compressibility: NumberArray[NDimension],
    bubble_point_pressure: NumberArray[NDimension],
    current_pressure: NumberArray[NDimension],
) -> NumberArray[NDimension]:
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
    result = np.empty_like(current_pressure)
    saturated_mask = current_pressure <= bubble_point_pressure
    undersaturated_mask = np.invert(saturated_mask)

    # Saturated: just use saturated_oil_fvf
    saturated_fvf = get_mask(saturated_oil_fvf, saturated_mask)
    apply_mask(result, saturated_mask, saturated_fvf)

    # Undersaturated: compute correction only where needed
    if np.any(undersaturated_mask):
        undersaturated_pressure = get_mask(current_pressure, undersaturated_mask)
        undersaturated_fvf = get_mask(saturated_oil_fvf, undersaturated_mask)
        delta_p = bubble_point_pressure - undersaturated_pressure
        correction_factor = clip(np.exp(oil_compressibility * delta_p), 1e-6, 5.0)
        apply_mask(result, undersaturated_mask, undersaturated_fvf * correction_factor)

    return result


@numba.njit(cache=True)
def compute_oil_formation_volume_factor(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    bubble_point_pressure: NumberArray[NDimension],
    oil_specific_gravity: NumberArray[NDimension],
    gas_gravity: NumberArray[NDimension],
    gas_to_oil_ratio: NumberArray[NDimension],
    oil_compressibility: NumberArray[NDimension],
) -> NumberArray[NDimension]:
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
    oil_fvf = np.empty_like(temperature)
    standing_mask = temperature <= 100
    vazquez_mask = np.invert(standing_mask)

    # Compute Standing FVF only where T <= 100
    if np.any(standing_mask):
        standing_result = compute_oil_formation_volume_factor_standing(
            temperature=get_mask(temperature, standing_mask),
            oil_specific_gravity=get_mask(oil_specific_gravity, standing_mask),
            gas_gravity=get_mask(gas_gravity, standing_mask),
            gas_to_oil_ratio=get_mask(gas_to_oil_ratio, standing_mask),
        )
        apply_mask(oil_fvf, standing_mask, standing_result)

    # Compute Vazquez-Beggs FVF only where T > 100
    if np.any(vazquez_mask):
        vazquez_result = compute_oil_formation_volume_factor_vazquez_and_beggs(
            temperature=get_mask(temperature, vazquez_mask),
            oil_specific_gravity=get_mask(oil_specific_gravity, vazquez_mask),
            gas_gravity=get_mask(gas_gravity, vazquez_mask),
            gas_to_oil_ratio=get_mask(gas_to_oil_ratio, vazquez_mask),
        )
        apply_mask(oil_fvf, vazquez_mask, vazquez_result)

    return correct_oil_fvf_for_pressure(
        saturated_oil_fvf=oil_fvf,
        oil_compressibility=oil_compressibility,
        bubble_point_pressure=bubble_point_pressure,
        current_pressure=pressure,
    )


@numba.njit(cache=True)
def compute_oil_api_gravity(
    oil_specific_gravity: NumberArray[NDimension],
) -> NumberArray[NDimension]:
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
    if np.any(oil_specific_gravity <= 0):
        raise ValidationError("Oil specific gravity must be greater than zero.")

    dtype = oil_specific_gravity.dtype
    return ((141.5 / oil_specific_gravity) - 131.5).astype(dtype)  # type: ignore[return-value]


@numba.njit(cache=True)
def get_vazquez_beggs_oil_bubble_point_pressure_coefficients(
    oil_api_gravity: NumberArray[NDimension],
) -> tuple[
    NumberArray[NDimension],
    NumberArray[NDimension],
    NumberArray[NDimension],
]:
    """
    Returns the empirical coefficients (C₁, C₂, C₃) used in the Vazquez-Beggs
    bubble point pressure correlation based on oil API gravity.

    Coefficients vary for API ≤ 30 and API > 30:

        If API ≤ 30:
            C₁ = 0.0362, C₂ = 1.0937, C₃ = 25.7240
        Else:
            C₁ = 0.0178, C₂ = 1.1870, C₃ = 23.9310

    :param oil_api_gravity: Oil API gravity (°API)
    :return: Tuple (C₁, C₂, C₃)
    """
    less_equal_30 = oil_api_gravity <= 30
    dtype = oil_api_gravity.dtype
    c1 = np.where(less_equal_30, 0.0362, 0.0178)
    c2 = np.where(less_equal_30, 1.0937, 1.1870)
    c3 = np.where(less_equal_30, 25.7240, 23.9310)
    return c1.astype(dtype), c2.astype(dtype), c3.astype(dtype)  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_oil_bubble_point_pressure(
    gas_gravity: NumberArray[NDimension],
    oil_api_gravity: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    gas_to_oil_ratio: NumberArray[NDimension],
) -> NumberArray[NDimension]:
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
    if min_(gas_gravity) <= 0:
        raise ValidationError("Gas specific gravity must be greater than zero.")
    if min_(oil_api_gravity) <= 0:
        raise ValidationError("Oil API gravity must be greater than zero.")
    if min_(temperature) <= 32:
        raise ValidationError("Temperature must be greater than absolute zero (32 °F).")
    if min_(gas_to_oil_ratio) < 0:
        raise ValidationError("Gas-to-oil ratio must be non-negative.")

    c1, c2, c3 = get_vazquez_beggs_oil_bubble_point_pressure_coefficients(oil_api_gravity)
    temperature_rankine = temperature + 459.67
    dtype = gas_to_oil_ratio.dtype
    pressure = (
        gas_to_oil_ratio
        / (c1 * gas_gravity * np.exp((c3 * oil_api_gravity) / temperature_rankine))
    ) ** (1 / c2)
    return pressure.astype(dtype)  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_gor_vasquez_beggs(
    pressure: NumberArray[NDimension],
    gas_gravity: NumberArray[NDimension],
    oil_api_gravity: NumberArray[NDimension],
    temperature_in_rankine: NumberArray[NDimension],
) -> NumberArray[NDimension]:
    """Implementation of the Vazquez-Beggs GOR correlation."""
    c1, c2, c3 = get_vazquez_beggs_oil_bubble_point_pressure_coefficients(oil_api_gravity)
    dtype = pressure.dtype
    return (  # type: ignore[return-value]
        (pressure**c2) * c1 * gas_gravity * np.exp((c3 * oil_api_gravity) / temperature_in_rankine)
    ).astype(dtype)


@numba.njit(cache=True)
def compute_gas_to_oil_ratio(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    bubble_point_pressure: NumberArray[NDimension],
    gas_gravity: NumberArray[NDimension],
    oil_api_gravity: NumberArray[NDimension],
    gor_at_bubble_point_pressure: NumberArray[NDimension] | None = None,
) -> NumberArray[NDimension]:
    """
    Computes the solution gas-to-oil ratio (Solution-GOR) using the Vazquez-Beggs correlation.

    Solution GOR is the amount of gas dissolved in oil at a given pressure and temperature.

    Two regimes:
    - **Saturated region (P < Pb)**: GOR is pressure-dependent.
    - **Undersaturated region (P >= Pb)**: GOR = GORb (constant). If not given, it is computed.

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
    if min_(pressure) <= 0:
        raise ValidationError("Pressure must be greater than zero.")

    dtype = pressure.dtype
    temperature_in_rankine = temperature + dtype.type(459.67)

    # Compute GOR at bubble point
    if gor_at_bubble_point_pressure is not None:
        gor_at_bp = gor_at_bubble_point_pressure.astype(dtype)
    else:
        gor_at_bp = compute_gor_vasquez_beggs(
            pressure=bubble_point_pressure,
            gas_gravity=gas_gravity,
            oil_api_gravity=oil_api_gravity,
            temperature_in_rankine=temperature_in_rankine,
        )

    gor = np.empty_like(pressure, dtype=dtype)
    saturated_mask = pressure < bubble_point_pressure
    undersaturated_mask = np.invert(saturated_mask)

    # Undersaturated: use GOR at bubble point
    undersaturated_gor = get_mask(gor_at_bp, undersaturated_mask)
    apply_mask(gor, undersaturated_mask, undersaturated_gor)

    # Saturated: compute GOR at current pressure
    if np.any(saturated_mask):
        saturated_pressure = get_mask(pressure, saturated_mask)
        saturated_gor = compute_gor_vasquez_beggs(
            pressure=saturated_pressure,
            gas_gravity=gas_gravity,
            oil_api_gravity=oil_api_gravity,
            temperature_in_rankine=temperature_in_rankine,
        )
        apply_mask(gor, saturated_mask, saturated_gor)

    return np.maximum(0.0, gor).astype(dtype)  # type: ignore[return-value]


@numba.njit(cache=True)
def _compute_dead_oil_viscosity_modified_beggs(
    temperature: NumberArray[NDimension],
    oil_api_gravity: NumberArray[NDimension],
) -> NumberArray[NDimension]:
    if np.any(temperature <= 0):
        raise ValidationError("Temperature (°F) must be > 0 for this correlation.")

    temperature_rankine = temperature + 459.67
    oil_specific_gravity = 141.5 / (131.5 + oil_api_gravity)

    log_viscosity = (
        1.8653 - 0.025086 * oil_specific_gravity - 0.5644 * np.log10(temperature_rankine)
    )
    viscosity = (10**log_viscosity) - 1
    dtype = temperature.dtype
    return np.maximum(0.0, viscosity).astype(dtype)  # type: ignore[return-value]


def compute_dead_oil_viscosity_modified_beggs(
    temperature: NumberArray[NDimension],
    oil_specific_gravity: NumberArray[NDimension],
) -> NumberArray[NDimension]:
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
    if min_(oil_api_gravity) < 5 or max_(oil_api_gravity) > 75:
        warnings.warn(
            f"API gravity min={min_(oil_api_gravity):.6f}, max={max_(oil_api_gravity):.6f} is outside typical range [5, 75]. "
            f"Dead oil viscosity may be inaccurate.",
            stacklevel=2,
        )
    return _compute_dead_oil_viscosity_modified_beggs(temperature, oil_api_gravity)


@numba.njit(cache=True)
def _compute_oil_viscosity(
    pressure: NumberArray[NDimension],
    bubble_point_pressure: NumberArray[NDimension],
    dead_oil_viscosity: NumberArray[NDimension],
    gas_to_oil_ratio: NumberArray[NDimension],
    gor_at_bubble_point_pressure: NumberArray[NDimension],
) -> NumberArray[NDimension]:
    result = np.empty_like(pressure)
    saturated_mask = pressure <= bubble_point_pressure
    undersaturated_mask = np.invert(saturated_mask)

    # Saturated case: compute viscosity using current GOR
    if np.any(saturated_mask):
        gas_to_oil_ratio_saturated = get_mask(gas_to_oil_ratio, saturated_mask)
        dead_oil_viscosity_saturated = get_mask(dead_oil_viscosity, saturated_mask)

        X_saturated = 10.715 * (gas_to_oil_ratio_saturated + 100) ** -0.515
        Y_saturated = 5.44 * (gas_to_oil_ratio_saturated + 150) ** -0.338
        saturated_viscosity = X_saturated * (dead_oil_viscosity_saturated**Y_saturated)
        apply_mask(result, saturated_mask, saturated_viscosity)

    # Undersaturated case: compute mu_ob at Pb first
    if np.any(undersaturated_mask):
        pressure_undersaturated = get_mask(pressure, undersaturated_mask)
        bubble_point_pressure_undersaturated = get_mask(bubble_point_pressure, undersaturated_mask)
        dead_oil_viscosity_undersaturated = get_mask(dead_oil_viscosity, undersaturated_mask)
        gor_at_bubble_point_pressure_undersaturated = get_mask(
            gor_at_bubble_point_pressure, undersaturated_mask
        )

        X_bubble_point = (
            10.715 * (gor_at_bubble_point_pressure_undersaturated + 100) ** -0.515  # type: ignore
        )
        Y_bubble_point = (
            5.44 * (gor_at_bubble_point_pressure_undersaturated + 150) ** -0.338  # type: ignore
        )
        dead_oil_viscosity_at_bubble_point = X_bubble_point * (
            dead_oil_viscosity_undersaturated**Y_bubble_point
        )

        # Apply undersaturated viscosity correlation
        X_undersaturated = (
            2.6
            * pressure_undersaturated**1.187
            * np.exp(-11.513 - 8.98e-5 * pressure_undersaturated)
        )
        undersaturated_viscosity = dead_oil_viscosity_at_bubble_point * (
            (pressure_undersaturated / bubble_point_pressure_undersaturated) ** X_undersaturated
        )
        apply_mask(result, undersaturated_mask, undersaturated_viscosity)

    dtype = pressure.dtype
    return np.maximum(result, 1e-6).astype(dtype)  # type: ignore[return-value]


def compute_oil_viscosity(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    bubble_point_pressure: NumberArray[NDimension],
    oil_specific_gravity: NumberArray[NDimension],
    gas_to_oil_ratio: NumberArray[NDimension],
    gor_at_bubble_point_pressure: NumberArray[NDimension],
) -> NumberArray[NDimension]:
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
    if min_(temperature) <= 0 or min_(pressure) <= 0 or min_(bubble_point_pressure) <= 0:
        raise ValidationError("Temperature and pressures must be positive.")
    if min_(oil_specific_gravity) <= 0:
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


@numba.njit(cache=True)
def compute_oil_compressibility_liberation_correction_term(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    bubble_point_pressure: NumberArray[NDimension],
    gor_at_bubble_point_pressure: NumberArray[NDimension],
    gas_gravity: NumberArray[NDimension],
    oil_api_gravity: NumberArray[NDimension],
    gas_formation_volume_factor: NumberOrArray[NDimension],
    oil_formation_volume_factor: NumberOrArray[NDimension],
) -> NumberArray[NDimension]:
    """
    Computes the liberation correction term for oil compressibility below bubble point pressure.

    The correction term is give by:

        x = (Bg/Bo * dRs/dp) / 5.615

        dRs/dp = (Rs(P + ΔP) - Rs(P - ΔP)) / (2 * ΔP)

    where:
    - Rs is the solution Gas-Oil Ratio (GOR) at current pressure and temperature
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
    dtype = pressure.dtype
    delta_p = np.maximum(0.01, 1e-4 * pressure).astype(dtype)
    pressure_plus = pressure + delta_p
    pressure_minus = pressure - delta_p
    gor_plus_delta = compute_gas_to_oil_ratio(
        pressure=pressure_plus,  # type: ignore[arg-type]
        temperature=temperature,
        bubble_point_pressure=bubble_point_pressure,
        gas_gravity=gas_gravity,
        oil_api_gravity=oil_api_gravity,
        gor_at_bubble_point_pressure=gor_at_bubble_point_pressure,
    )
    gor_minus_delta = compute_gas_to_oil_ratio(
        pressure=pressure_minus,  # type: ignore[arg-type]
        temperature=temperature,
        bubble_point_pressure=bubble_point_pressure,
        gas_gravity=gas_gravity,
        oil_api_gravity=oil_api_gravity,
        gor_at_bubble_point_pressure=gor_at_bubble_point_pressure,
    )

    dRs_dp = (gor_plus_delta - gor_minus_delta) / (2 * delta_p)
    return (gas_formation_volume_factor / oil_formation_volume_factor) * dRs_dp / 5.615  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_base_compressibility(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    bubble_point_pressure: NumberArray[NDimension],
    oil_api_gravity: NumberArray[NDimension],
    gas_gravity: NumberArray[NDimension],
    gor_at_bubble_point_pressure: NumberArray[NDimension],
) -> NumberArray[NDimension]:
    current_gor = compute_gas_to_oil_ratio(
        pressure=pressure,
        temperature=temperature,
        bubble_point_pressure=bubble_point_pressure,
        gas_gravity=gas_gravity,
        oil_api_gravity=oil_api_gravity,
        gor_at_bubble_point_pressure=gor_at_bubble_point_pressure,
    )
    val = (
        -1433 + 5 * current_gor + 17.2 * temperature - 1180 * gas_gravity + 12.61 * oil_api_gravity
    ) / ((10**5) * pressure)
    return np.maximum(val, 0.0).astype(pressure.dtype)  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_oil_compressibility(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    bubble_point_pressure: NumberArray[NDimension],
    oil_api_gravity: NumberArray[NDimension],
    gas_gravity: NumberArray[NDimension],
    gor_at_bubble_point_pressure: NumberArray[NDimension],
    gas_formation_volume_factor: NumberOrArray[NDimension] = 1.0,
    oil_formation_volume_factor: NumberOrArray[NDimension] = 1.0,
) -> NumberArray[NDimension]:
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
        min_(pressure) <= 0
        or min_(bubble_point_pressure) <= 0
        or min_(temperature) <= 0
        or min_(gas_gravity) <= 0
        or min_(oil_api_gravity) <= 0
    ):
        raise ValidationError("All input parameters (P, Pb, T, Gas SG, API) must be positive.")

    result = np.empty_like(pressure)
    undersaturated_mask = pressure > bubble_point_pressure
    saturated_mask = np.invert(undersaturated_mask)

    # Use atmospheric pressure as fill value instead of np.nan (default fill) to avoid issues
    # With `compute_base_compressibility` complaining about NaNs or zero pressure.
    # Undersaturated: just base compressibility
    if np.any(undersaturated_mask):
        undersaturated_pressure = get_mask(pressure, undersaturated_mask, fill_value=14.7)
        undersaturated_compressibility = compute_base_compressibility(
            pressure=undersaturated_pressure,
            temperature=temperature,
            bubble_point_pressure=bubble_point_pressure,
            oil_api_gravity=oil_api_gravity,
            gas_gravity=gas_gravity,
            gor_at_bubble_point_pressure=gor_at_bubble_point_pressure,
        )
        apply_mask(result, undersaturated_mask, undersaturated_compressibility)

    # Saturated: base compressibility + correction term
    if np.any(saturated_mask):
        pressure_saturated = get_mask(pressure, saturated_mask, fill_value=14.7)
        base_compressibility_saturated = compute_base_compressibility(
            pressure=pressure_saturated,
            temperature=temperature,
            bubble_point_pressure=bubble_point_pressure,
            oil_api_gravity=oil_api_gravity,
            gas_gravity=gas_gravity,
            gor_at_bubble_point_pressure=gor_at_bubble_point_pressure,
        )

        correction_term = compute_oil_compressibility_liberation_correction_term(
            pressure=pressure_saturated,
            temperature=temperature,
            gas_gravity=gas_gravity,
            oil_api_gravity=oil_api_gravity,
            bubble_point_pressure=bubble_point_pressure,
            gas_formation_volume_factor=gas_formation_volume_factor,
            oil_formation_volume_factor=oil_formation_volume_factor,
            gor_at_bubble_point_pressure=gor_at_bubble_point_pressure,
        )
        saturated_compressibility = base_compressibility_saturated + correction_term
        apply_mask(result, saturated_mask, saturated_compressibility)

    return result


def compute_live_oil_density(
    api_gravity: NumberArray[NDimension],
    gas_gravity: NumberArray[NDimension],
    gas_to_oil_ratio: NumberArray[NDimension],
    formation_volume_factor: NumberArray[NDimension],
) -> NumberArray[NDimension]:
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
    dtype = api_gravity.dtype
    stock_tank_oil_density_lb_per_ft3 = np.multiply(
        np.divide(141.5, np.add(api_gravity, 131.5, dtype=dtype), dtype=dtype),
        c.STANDARD_WATER_DENSITY_IMPERIAL,
        dtype=dtype,
    )

    # Mass of oil per STB (lb)
    mass_stock_tank_oil = np.divide(
        stock_tank_oil_density_lb_per_ft3, c.CUBIC_FEET_TO_STB, dtype=dtype
    )

    # Mass of dissolved gas per STB (lb)
    # Approx: 1 scf = gas_gravity * (molecular weight of air) / 379.49 lb
    gas_mass_per_scf = np.divide(
        (gas_gravity * c.MOLECULAR_WEIGHT_AIR), c.SCF_PER_POUND_MOLE, dtype=dtype
    )
    mass_dissolved_gas = np.multiply(gas_to_oil_ratio, gas_mass_per_scf, dtype=dtype)

    # Total mass and volume
    total_mass_lb_per_stb = mass_stock_tank_oil + mass_dissolved_gas
    # print(formation_volume_factor)
    total_volume_ft3_per_stb = np.multiply(
        formation_volume_factor, c.BARRELS_TO_CUBIC_FEET, dtype=dtype
    )

    # Live oil density in lb/ft³
    live_oil_density_lb_per_ft3 = np.divide(
        total_mass_lb_per_stb, total_volume_ft3_per_stb, dtype=dtype
    )
    return live_oil_density_lb_per_ft3  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_gas_to_oil_ratio_standing(
    pressure: NumberArray[NDimension],
    oil_api_gravity: NumberArray[NDimension],
    gas_gravity: NumberArray[NDimension],
) -> NumberArray[NDimension]:
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

    This correlation is typically used for light soil and may not be accurate
    for heavy soil (API < 10) or high pressures.

    :param pressure: Pressure (psi)
    :param oil_api_gravity: API gravity of the oil in degrees API.
    :param gas_gravity: Specific gravity of the gas (relative to air).
    :return: Solution gas-oil ratio (Rs) in (SCF/STB).
    """
    if min_(oil_api_gravity) < 0 or min_(gas_gravity) < 0 or min_(pressure) < 0:
        raise ValidationError("All inputs must be non-negative for Rs calculation.")

    if min_(oil_api_gravity) < 10:
        raise ValidationError(
            "API gravity must be greater than or equal to 10 for Standing's correlation."
        )

    gor = gas_gravity * ((pressure / 18.2 + 1.4) * 10 ** (0.0125 * oil_api_gravity)) ** (
        1 / 1.2048
    )
    dtype = pressure.dtype
    return gor.astype(dtype)  # type: ignore[return-value]


@numba.njit(cache=True)
def estimate_solution_gor(
    pressure: NumberArray[NDimension],
    temperature: NumberArray[NDimension],
    oil_api_gravity: NumberArray[NDimension],
    gas_gravity: NumberArray[NDimension],
    maximum_iterations: int = 20,
    tolerance: float = 1e-4,
) -> NumberArray[NDimension]:
    """
    Estimate solution gas-to-oil ratio Rs(P, T) iteratively for arrays.

    This solves the coupled system where:
    - Rs depends on P and Pb via the correlation
    - Pb depends on Rs and T via Vazquez-Beggs

    The algorithm at each point:
    1. Initial guess: Rs from Standing correlation (uses P, API, γg)
    2. Compute Pb from Rs using Vazquez-Beggs
    3. If P > Pb: oil is undersaturated, find Rs where Pb(Rs,T) = P
    4. If P <= Pb: oil is saturated, Rs from Standing
    5. Iterate until convergence

    For undersaturated oil (P > Pb):
        Rs remains constant at Rsb (the Rs at bubble point pressure)

    For saturated oil (P <= Pb):
        Rs varies with pressure - more gas dissolves at higher P

    :param pressure: Reservoir pressure array (psi)
    :param temperature: Reservoir temperature array (°F)
    :param oil_api_gravity: Oil API gravity array (°API, typically 15-50)
    :param gas_gravity: Gas specific gravity array (dimensionless, typically 0.6-1.2)
    :param maximum_iterations: Maximum iterations for convergence (default: 20)
    :param tolerance: Relative tolerance for convergence (default: 1e-4)
    :return: Solution gas-to-oil ratio Rs array (SCF/STB)

    Notes:
    - Convergence is typically achieved in 3-5 iterations
    - Uses Standing correlation for initial guess (ignores T)
    - Uses Vazquez-Beggs for Pb calculation (includes T)
    - Handles both saturated and undersaturated conditions
    - Parallelized for performance on large arrays
    """
    flat_size = pressure.size
    dtype = pressure.dtype
    result = np.empty(flat_size, dtype=dtype)

    pressure_flat = pressure.ravel()
    temperature_flat = temperature.ravel()
    api_gravity_flat = oil_api_gravity.ravel()
    gas_gravity_flat = gas_gravity.ravel()

    for i in numba.prange(flat_size):  # type: ignore
        result[i] = soil.estimate_solution_gor(
            pressure=pressure_flat[i],
            temperature=temperature_flat[i],
            oil_api_gravity=api_gravity_flat[i],
            gas_gravity=gas_gravity_flat[i],
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
        )

    return result.reshape(pressure.shape)  # type: ignore[return-value]


@numba.njit(cache=True)
def _standing_oil_bubble_point_residual(
    pressure: Number,
    oil_api: Number,
    gas_gravity: Number,
    target_rs: Number,
) -> Number:
    """
    Computes scalar residual for Standing correlation: Rs(P) - Rs_target
    """
    gor = soil.compute_gas_to_oil_ratio_standing(
        pressure=pressure,
        oil_api_gravity=oil_api,
        gas_gravity=gas_gravity,
    )
    return gor - target_rs


def estimate_bubble_point_pressure_standing(
    oil_api_gravity: NumberArray[NDimension],
    gas_gravity: NumberArray[NDimension],
    observed_gas_to_oil_ratio: NumberArray[NDimension],
) -> NumberArray[NDimension]:
    """
    Estimate bubble point pressure (Pb) using Standing's correlation
    given observed Rs and known oil API gravity and gas gravity.

    **THIS FUNCTION ESTIMATES THE BUBBLE POINT PRESSURE AND IS NOT A DIRECT
    MEASUREMENT**. It is only valid for light oil (API > 10) and may not be accurate
    for heavy oil or high pressures.

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
    # Allocate output
    bubble_point_pressure = np.empty_like(oil_api_gravity)

    min_pressure = c.MINIMUM_VALID_PRESSURE
    max_pressure = c.MAXIMUM_VALID_PRESSURE
    # Loop over all cells
    it = np.nditer(oil_api_gravity, flags=["multi_index"])  # type: ignore
    while not it.finished:
        idx = it.multi_index
        bubble_point_pressure[idx] = brentq(  # type: ignore
            f=_standing_oil_bubble_point_residual,
            a=min_pressure,
            b=max_pressure,
            args=(
                oil_api_gravity[idx],
                gas_gravity[idx],
                observed_gas_to_oil_ratio[idx],
            ),
            xtol=1e-6,
            full_output=False,
        )
        it.iternext()

    return bubble_point_pressure


@numba.njit(cache=True)
def compute_miscibility_transition_factor(
    pressure: NumberArray[NDimension],
    minimum_miscibility_pressure: NumberOrArray[NDimension],
    transition_width: NumberOrArray[NDimension] = 500.0,
) -> NumberArray[NDimension]:
    """
    Compute pressure-dependent miscibility transition factor.

    Returns a smooth transition from 0 (immiscible) at low pressure
    to 1 (fully miscible) above minimum miscibility pressure.

    This factor represents the degree of miscibility development and should
    be multiplied by the base Todd-Longstaff omega parameter to get the
    effective omega for viscosity calculations.

    Physical Behavior:
        - P << MMP: factor -> 0 (immiscible, no miscible mixing)
        - P ≈ MMP: factor ≈ 0.5 (transition zone, partial miscibility)
        - P >> MMP: factor -> 1 (fully miscible, maximum mixing)

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
    # Smooth transition using hyperbolic tangent
    # Normalize pressure relative to MMP and transition width
    normalized = (pressure - minimum_miscibility_pressure) / transition_width

    # Transition factor varies from 0 (immiscible) to 1 (miscible)
    transition_factor = 0.5 * (1.0 + np.tanh(normalized))
    dtype = pressure.dtype
    return transition_factor.astype(dtype)  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_effective_todd_longstaff_omega(
    pressure: NumberArray[NDimension],
    base_omega: NumberOrArray[NDimension],
    minimum_miscibility_pressure: NumberOrArray[NDimension],
    transition_width: NumberOrArray[NDimension] = 500.0,
) -> NumberArray[NDimension]:
    """
    Compute pressure-dependent effective Todd-Longstaff omega parameter.

    Combines the base mixing parameter (omega) with pressure-dependent
    miscibility to get the effective omega for viscosity calculations.

    Below MMP: omega_eff -> 0 (immiscible behavior, segregated flow)
    Above MMP: omega_eff -> base_omega (miscible behavior, mixed flow)

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
        pressure=2500, base_omega=0.67, minimum_miscibility_pressure=2000, transition_width=500
    )
    0.54  # Partial miscibility developed
    ```
    """
    if min_(base_omega) <= 0.0:
        return np.full_like(pressure, 0.0)

    transition_factor = compute_miscibility_transition_factor(
        pressure=pressure,
        minimum_miscibility_pressure=minimum_miscibility_pressure,
        transition_width=transition_width,
    )
    dtype = pressure.dtype
    return (base_omega * transition_factor).astype(dtype)  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_todd_longstaff_effective_viscosity(
    oil_viscosity: NumberArray[NDimension],
    solvent_viscosity: NumberArray[NDimension],
    solvent_concentration: NumberArray[NDimension],
    omega: NumberArray[NDimension],
) -> NumberArray[NDimension]:
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

    :raises ValidationError: If concentrations or omega are outside [0,1], or viscosities ≤ 0

    Example:
    ```python
    # Immiscible case (omega = 0)
    compute_todd_longstaff_effective_viscosity(
        oil_viscosity=10.0, solvent_viscosity=0.05, solvent_concentration=0.3, omega=0.0
    )
    0.147  # Harmonic mean - segregated flow

    # Fully miscible case (omega = 1)
    compute_todd_longstaff_effective_viscosity(
        oil_viscosity=10.0, solvent_viscosity=0.05, solvent_concentration=0.3, omega=1.0
    )
    7.015  # Arithmetic mean - fully mixed

    # Typical CO2 flood (omega = 0.67)
    compute_todd_longstaff_effective_viscosity(
        oil_viscosity=10.0, solvent_viscosity=0.05, solvent_concentration=0.3, omega=0.67
    )
    0.89  # Realistic mixture viscosity
    ```

    References:
    Todd, M.R. and Longstaff, W.J. (1972). "The Development, Testing and
    Application of a Numerical Simulator for Predicting Miscible Flood Performance."
    JPT, July 1972, pp. 874-882.
    """
    # Validate inputs
    if min_(solvent_concentration) < 0.0 or max_(solvent_concentration) > 1.0:
        raise ValidationError(
            f"Solvent concentration must be in [0,1], got {solvent_concentration}"
        )
    if min_(omega) < 0.0 or max_(omega) > 1.0:
        raise ValidationError(f"Omega must be in [0,1], got {omega}")
    if min_(oil_viscosity) <= 0.0 or min_(solvent_viscosity) <= 0.0:
        raise ValidationError("Viscosities must be positive")

    C_s = solvent_concentration
    C_o = 1.0 - C_s

    # Clamp to avoid 0-division in harmonic mean (pure-phase cells handled by `np.where` below)
    C_s_safe = clip(C_s, 1e-12, 1.0 - 1e-12)
    C_o_safe = 1.0 - C_s_safe

    dtype = oil_viscosity.dtype

    # Fully mixed viscosity (arithmetic/linear mean)
    # Represents ideal miscibility - single homogeneous phase
    mu_mix = C_s_safe * solvent_viscosity + C_o * oil_viscosity

    # Fully segregated viscosity (harmonic mean)
    # Represents parallel flow of two immiscible phases
    # Equivalent to: μ_seg = μ_s * μ_o / (C_s * μ_o + C_o * μ_s)
    mu_segregated = 1.0 / (C_s_safe / solvent_viscosity + C_o_safe / oil_viscosity)

    # Todd-Longstaff interpolation (weighted geometric mean)
    # Special cases:
    #   ω = 0: μ_eff = μ_segregated (immiscible, harmonic mean)
    #   ω = 1: μ_eff = μ_mix (fully mixed, arithmetic mean)
    #   ω = 0.5: μ_eff = sqrt(μ_mix * μ_segregated) (geometric mean)
    mu_effective = (mu_mix**omega) * (mu_segregated ** (1.0 - omega))

    # Handle edge cases element-wise (pure solvent or pure oil cells)
    mu_effective = np.where(C_s >= 1.0 - 1e-12, solvent_viscosity, mu_effective)
    mu_effective = np.where(C_s <= 1e-12, oil_viscosity, mu_effective)
    return mu_effective.astype(dtype)  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_todd_longstaff_effective_density(
    oil_density: NumberArray[NDimension],
    solvent_density: NumberArray[NDimension],
    oil_viscosity: NumberArray[NDimension],
    solvent_viscosity: NumberArray[NDimension],
    solvent_concentration: NumberArray[NDimension],
    omega: NumberArray[NDimension],
) -> NumberArray[NDimension]:
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
    :raises ValidationError: If concentrations or omega are outside [0,1], or inputs ≤ 0

    Example:
    ```python
    # CO2 (light) displacing oil (heavy)
    compute_todd_longstaff_effective_density(
        oil_density=50.0,  # lb/ft³
        solvent_density=30.0,  # lb/ft³ (CO2 is lighter)
        oil_viscosity=10.0,  # cP
        solvent_viscosity=0.05,  # cP (CO2 is much less viscous)
        solvent_concentration=0.3,
        omega=0.67,
    )
    44.2  # Effective density between pure values

    # Fully segregated (omega=0): flow-weighted density
    compute_todd_longstaff_effective_density(
        oil_density=50.0,
        solvent_density=30.0,
        oil_viscosity=10.0,
        solvent_viscosity=0.05,
        solvent_concentration=0.3,
        omega=0.0,
    )
    30.6  # Much closer to solvent (it flows more easily)

    # Fully mixed (omega=1): volume-weighted density
    compute_todd_longstaff_effective_density(
        oil_density=50.0,
        solvent_density=30.0,
        oil_viscosity=10.0,
        solvent_viscosity=0.05,
        solvent_concentration=0.3,
        omega=1.0,
    )
    44.0  # Simple volume average: 0.3*30 + 0.7*50
    ```

    References:
    Todd, M.R. and Longstaff, W.J. (1972). "The Development, Testing and
    Application of a Numerical Simulator for Predicting Miscible Flood Performance."
    JPT, July 1972, pp. 874-882.
    """
    if min_(solvent_concentration) < 0.0 or max_(solvent_concentration) > 1.0:
        raise ValidationError(
            f"Solvent concentration must be in [0,1], got {solvent_concentration}"
        )
    if min_(omega) < 0.0 or max_(omega) > 1.0:
        raise ValidationError(f"Omega must be in [0,1], got {omega}")
    if min_(oil_density) <= 0.0 or min_(solvent_density) <= 0.0:
        raise ValidationError("Densities must be positive")
    if min_(oil_viscosity) <= 0.0 or min_(solvent_viscosity) <= 0.0:
        raise ValidationError("Viscosities must be positive")

    C_s = solvent_concentration
    C_o = 1.0 - solvent_concentration

    dtype = C_s.dtype

    # Fully mixed density (volume-weighted, arithmetic mean)
    # This is the density if phases are perfectly mixed by volume
    rho_mix = (C_s * solvent_density) + (C_o * oil_density)

    # Clamp concentrations to avoid 0-division in flow fraction computation
    # (pure-phase cells are corrected by np.where at the end anyway)
    C_s_safe = np.clip(C_s, 1e-12, 1.0 - 1e-12)
    C_o_safe = 1.0 - C_s_safe

    # Compute phase flow fractions for segregated density
    # These represent how much each phase contributes to flow based on mobility
    # f_s = fraction of flow that is solvent
    # f_o = fraction of flow that is oil
    # Note: More mobile phase (lower viscosity) gets higher flow fraction
    denominator = (C_s_safe * oil_viscosity) + (C_o_safe * solvent_viscosity)

    # Compute flow fractions element-wise, handling near-zero denominators
    # Avoid division by zero (though should never happen with positive viscosities)
    f_s = np.where(denominator < 1e-15, C_s_safe, (C_s_safe * oil_viscosity) / denominator)
    f_o = np.where(denominator < 1e-15, C_o_safe, (C_o_safe * solvent_viscosity) / denominator)

    # Fully segregated density (flow-weighted)
    # This is the density if phases flow separately, weighted by their mobilities
    rho_segregated = (f_s * solvent_density) + (f_o * oil_density)

    # Todd-Longstaff interpolation (weighted geometric mean)
    # Special cases:
    #   ω = 0: ρ_eff = ρ_segregated (flow-weighted, immiscible)
    #   ω = 1: ρ_eff = ρ_mix (volume-weighted, fully mixed)
    rho_effective = (rho_mix**omega) * (rho_segregated ** (1.0 - omega))

    # Handle edge cases element-wise (pure solvent or pure oil cells)
    # Use element-wise check to avoid affecting all cells when any cell is pure
    rho_effective = np.where(C_s >= 1.0, solvent_density, rho_effective)
    rho_effective = np.where(C_s <= 0.0, oil_density, rho_effective)
    return rho_effective.astype(dtype)  # type: ignore[return-value]

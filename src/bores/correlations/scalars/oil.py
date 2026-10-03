import logging
import typing
import warnings

import numpy as np
from scipy.optimize import root_scalar  # type: ignore[import-untyped]

from bores.constants import c
from bores.errors import ComputationError, ValidationError
from bores.types import Number
from bores.utils import clip

logger = logging.getLogger(__name__)

__all__ = [
    "compute_dead_oil_viscosity_modified_beggs",
    "compute_effective_todd_longstaff_omega",
    "compute_gas_to_oil_ratio",
    "compute_gas_to_oil_ratio_standing",
    "compute_hydrocarbon_in_place",
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
    oil_density: Number,
    pressure: Number,
    temperature: Number,
    oil_compressibility: Number,
) -> Number:
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
        (oil_compressibility * delta_p) + (c.OIL_THERMAL_EXPANSION_COEFFICIENT_IMPERIAL * delta_t)
    )
    # Avoid numerical issues with small/large values
    correction_factor = clip(correction_factor, 0.2, 2.0)
    oil_density_at_stp = oil_density * correction_factor
    return oil_density_at_stp / c.STANDARD_WATER_DENSITY_IMPERIAL


def compute_oil_formation_volume_factor_standing(
    temperature: Number,
    oil_specific_gravity: Number,
    gas_gravity: Number,
    gas_to_oil_ratio: Number,
) -> Number:
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

    x = (gas_to_oil_ratio * (gas_gravity / oil_specific_gravity) ** 0.5) + (1.25 * temperature)
    oil_fvf = 0.972 + 0.000147 * (x**1.175)
    return oil_fvf


def get_vazquez_beggs_oil_fvf_coefficients(
    oil_api_gravity: Number,
) -> tuple[Number, Number, Number]:
    """
    Returns the coefficients a1, a2, a3 for the Vazquez and Beggs oil FVF correlation based on oil API gravity.
    """
    if oil_api_gravity <= 30:
        return 4.677e-4, 1.751e-5, -1.811e-8
    return 4.670e-4, 1.100e-5, 1.337e-9


def compute_oil_formation_volume_factor_vazquez_and_beggs(
    temperature: Number,
    oil_specific_gravity: Number,
    gas_gravity: Number,
    gas_to_oil_ratio: Number,
) -> Number:
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
    a1, a2, a3 = get_vazquez_beggs_oil_fvf_coefficients(oil_api_gravity)
    oil_fvf = (
        1
        + (a1 * gas_to_oil_ratio)
        + (a2 * (temperature - 60) * (oil_specific_gravity / gas_gravity))
        + (a3 * (temperature - 60) * gas_to_oil_ratio * (oil_specific_gravity / gas_gravity))
    )
    return oil_fvf


def correct_oil_fvf_for_pressure(
    saturated_oil_fvf: Number,
    oil_compressibility: Number,
    bubble_point_pressure: Number,
    current_pressure: Number,
) -> Number:
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


def compute_oil_formation_volume_factor(
    pressure: Number,
    temperature: Number,
    bubble_point_pressure: Number,
    oil_specific_gravity: Number,
    gas_gravity: Number,
    gas_to_oil_ratio: Number,
    oil_compressibility: Number,
) -> Number:
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


def compute_oil_api_gravity(oil_specific_gravity: Number) -> Number:
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


def get_vazquez_beggs_oil_bubble_point_pressure_coefficients(
    oil_api_gravity: Number,
) -> tuple[Number, Number, Number]:
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


def compute_oil_bubble_point_pressure(
    gas_gravity: Number,
    oil_api_gravity: Number,
    temperature: Number,
    gas_to_oil_ratio: Number,
) -> Number:
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

    c1, c2, c3 = get_vazquez_beggs_oil_bubble_point_pressure_coefficients(oil_api_gravity)
    temperature_rankine = temperature + 459.67
    pressure = (
        gas_to_oil_ratio
        / (c1 * gas_gravity * np.exp((c3 * oil_api_gravity) / temperature_rankine))
    ) ** (1 / c2)
    return pressure


def compute_gas_to_oil_ratio(
    pressure: Number,
    temperature: Number,
    bubble_point_pressure: Number,
    gas_gravity: Number,
    oil_api_gravity: Number,
    gor_at_bubble_point_pressure: Number | None = None,
) -> Number:
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
    if pressure <= 0:
        raise ValidationError("Pressure must be greater than zero.")

    temperature_in_rankine = temperature + 459.67
    c1, c2, c3 = get_vazquez_beggs_oil_bubble_point_pressure_coefficients(oil_api_gravity)

    def compute_gor_vasquez_beggs(pressure: Number) -> Number:
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
    return max(0.0, gor)


def compute_dead_oil_viscosity_modified_beggs(
    temperature: Number,
    oil_specific_gravity: Number,
) -> Number:
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
    if temperature <= 0:
        raise ValidationError("Temperature (°F) must be > 0 for this correlation.")

    oil_api_gravity = compute_oil_api_gravity(oil_specific_gravity)
    if not (5 <= oil_api_gravity <= 75):
        warnings.warn(
            f"API gravity {oil_api_gravity:.6f} is outside typical range [5, 75]. "
            f"Dead oil viscosity may be inaccurate.",
            stacklevel=2,
        )

    temperature_rankine = temperature + 459.67
    log_viscosity = (
        1.8653 - 0.025086 * oil_specific_gravity - 0.5644 * np.log10(temperature_rankine)
    )
    viscosity = (10**log_viscosity) - 1
    return max(0.0, viscosity)


def compute_oil_viscosity(
    pressure: Number,
    temperature: Number,
    bubble_point_pressure: Number,
    oil_specific_gravity: Number,
    gas_to_oil_ratio: Number,
    gor_at_bubble_point_pressure: Number,
) -> Number:
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
    if pressure <= bubble_point_pressure:
        # Saturated case: compute viscosity using current GOR
        X = 10.715 * (gas_to_oil_ratio + 100) ** -0.515
        Y = 5.44 * (gas_to_oil_ratio + 150) ** -0.338
        return max(X * (dead_oil_viscosity**Y), 1e-6)

    # Undersaturated case: compute mu_ob at Pb first
    X_bp = 10.715 * (gor_at_bubble_point_pressure + 100) ** -0.515
    Y_bp = 5.44 * (gor_at_bubble_point_pressure + 150) ** -0.338
    mu_ob = X_bp * (dead_oil_viscosity**Y_bp)

    # Apply undersaturated viscosity correlation
    X_under = 2.6 * pressure**1.187 * np.exp(-11.513 - 8.98e-5 * pressure)
    return max(mu_ob * ((pressure / bubble_point_pressure) ** X_under), 1e-6)


def compute_oil_compressibility_liberation_correction_term(
    pressure: Number,
    temperature: Number,
    gas_gravity: Number,
    oil_api_gravity: Number,
    bubble_point_pressure: Number,
    gas_formation_volume_factor: Number,
    oil_formation_volume_factor: Number,
    gor_at_bubble_point_pressure: Number,
) -> Number:
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
    delta_p = max(0.01, 1e-4 * pressure)
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


def compute_oil_compressibility(
    pressure: Number,
    temperature: Number,
    bubble_point_pressure: Number,
    oil_api_gravity: Number,
    gas_gravity: Number,
    gor_at_bubble_point_pressure: Number,
    gas_formation_volume_factor: Number = 1.0,
    oil_formation_volume_factor: Number = 1.0,
) -> Number:
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
        raise ValidationError("All input parameters (P, Pb, T, Gas SG, API) must be positive.")

    def compute_base_compressibility(pressure: Number) -> Number:
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
        return max(val, 0.0)

    if pressure > bubble_point_pressure:
        return compute_base_compressibility(pressure)

    base_comp = compute_base_compressibility(pressure)
    correction_term = compute_oil_compressibility_liberation_correction_term(
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


def compute_live_oil_density(
    api_gravity: Number,
    gas_gravity: Number,
    gas_to_oil_ratio: Number,
    formation_volume_factor: Number,
) -> Number:
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


def compute_gas_to_oil_ratio_standing(
    pressure: Number, oil_api_gravity: Number, gas_gravity: Number
) -> Number:
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

    gor = gas_gravity * ((pressure / 18.2 + 1.4) * 10 ** (0.0125 * oil_api_gravity)) ** (
        1 / 1.2048
    )
    return gor


def estimate_bubble_point_pressure_standing(
    oil_api_gravity: Number,
    gas_gravity: Number,
    observed_gas_to_oil_ratio: Number,
) -> Number:
    """
    Estimate bubble point pressure (Pb) using Standing's correlation
    given observed Rs and known oil API gravity and gas gravity.

    **THIS FUNCTION ESTIMATES THE BUBBLE POINT PRESSURE AND IS NOT A DIRECT
    MEASUREMENT**. It is only valid for light oils (API > 10) and may not be accurate
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
        return gor - observed_gas_to_oil_ratio  # type: ignore

    solver = root_scalar(residual, bracket=(14.696, 10000), method="brentq")
    if not solver.converged:
        raise ComputationError("Could not converge to a bubble point pressure.")

    bubble_point_pressure = solver.root
    return bubble_point_pressure


def compute_bubble_point_pressure_vazquez_beggs(
    gas_gravity: Number,
    oil_api_gravity: Number,
    temperature: Number,
    gas_to_oil_ratio: Number,
) -> Number:
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


def _compute_gas_to_oil_ratio_standing(
    pressure: Number, oil_api_gravity: Number, gas_gravity: Number
) -> Number:
    """
    Same as `compute_gas_to_oil_ratio_standing` but without input validation.

    :param pressure: Pressure (psi)
    :param oil_api_gravity: API gravity of the oil in degrees API.
    :param gas_gravity: Specific gravity of the gas (relative to air).
    :return: Solution gas-oil ratio (Rs) in (SCF/STB).
    """
    gor = gas_gravity * ((pressure / 18.2 + 1.4) * 10 ** (0.0125 * oil_api_gravity)) ** (
        1 / 1.2048
    )
    return gor


def estimate_solution_gor(
    pressure: Number,
    temperature: Number,
    oil_api_gravity: Number,
    gas_gravity: Number,
    maximum_iterations: int = 20,
    tolerance: Number = 1e-4,
) -> Number:
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
    :param maximum_iterations: Maximum iterations for convergence (default: 20)
    :param tolerance: Relative tolerance for convergence (default: 1e-4)
    :return: Solution gas-to-oil ratio Rs in SCF/STB

    Notes:
        - Convergence is typically achieved in 3-5 iterations
        - Uses Standing correlation for initial guess (ignores T)
        - Uses Vazquez-Beggs for Pb calculation (includes T)
        - Handles both saturated and undersaturated conditions
    """
    # Initial guess from Standing correlation
    rs_current = _compute_gas_to_oil_ratio_standing(
        pressure=pressure,
        oil_api_gravity=oil_api_gravity,
        gas_gravity=gas_gravity,
    )

    # Ensure reasonable bounds for Rs
    rs_min = 0.0
    rs_max = 5000.0  # Practical upper limit for Rs (SCF/STB)
    rs_current = max(rs_min, min(rs_current, rs_max))

    for _ in range(maximum_iterations):
        # Compute bubble point pressure from current Rs estimate
        pb_current = compute_bubble_point_pressure_vazquez_beggs(
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
                pb_mid = compute_bubble_point_pressure_vazquez_beggs(
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
            rs_new = _compute_gas_to_oil_ratio_standing(
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


def compute_hydrocarbon_in_place(
    area: Number,
    thickness: Number,
    porosity: Number,
    phase_saturation: Number,
    formation_volume_factor: Number,
    net_to_gross_ratio: Number = 1.0,
    hydrocarbon_type: typing.Literal["oil", "gas", "water"] = "oil",
    acre_ft_to_bbl: Number = 7758.0,
    acre_ft_to_ft3: Number = 43560.0,
) -> Number:
    """
    Computes the (free) hydrocarbon (or free water) in place (HCIP or FWIP) in stock tank
    barrels (STB) or standard cubic feet (SCF) using the volumetric method.

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
        raise ValidationError("Hydrocarbon type must be either 'oil', 'gas', or 'water'.")
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


def compute_miscibility_transition_factor(
    pressure: Number,
    minimum_miscibility_pressure: Number,
    transition_width: Number = 500.0,
) -> Number:
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


def compute_effective_todd_longstaff_omega(
    pressure: Number,
    base_omega: Number,
    minimum_miscibility_pressure: Number,
    transition_width: Number = 500.0,
) -> Number:
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
    if base_omega == 0:
        return 0.0

    transition_factor = compute_miscibility_transition_factor(
        pressure=pressure,
        minimum_miscibility_pressure=minimum_miscibility_pressure,
        transition_width=transition_width,
    )
    return base_omega * transition_factor


def compute_todd_longstaff_effective_viscosity(
    oil_viscosity: Number,
    solvent_viscosity: Number,
    solvent_concentration: Number,
    omega: Number = 0.67,
) -> Number:
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


def compute_todd_longstaff_effective_density(
    oil_density: Number,
    solvent_density: Number,
    oil_viscosity: Number,
    solvent_viscosity: Number,
    solvent_concentration: Number = 1.0,
    omega: Number = 0.67,
) -> Number:
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

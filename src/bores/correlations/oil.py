import logging

import numba  # type: ignore[import-untyped]
import numpy as np

from bores.constants import c
from bores.correlations.elementwise import elementwise
from bores.types import Number, NumberOrArray

logger = logging.getLogger(__name__)

# Physical constants, read from `bores.constants` when this module is imported.
BARRELS_TO_CUBIC_FEET = c.BARRELS_TO_CUBIC_FEET
CUBIC_FEET_TO_STB = c.CUBIC_FEET_TO_STB
FAHRENHEIT_TO_RANKINE_OFFSET = c.FAHRENHEIT_TO_RANKINE_OFFSET
MOLECULAR_WEIGHT_AIR = c.MOLECULAR_WEIGHT_AIR
OIL_THERMAL_EXPANSION_COEFFICIENT_IMPERIAL = c.OIL_THERMAL_EXPANSION_COEFFICIENT_IMPERIAL
SCF_PER_POUND_MOLE = c.SCF_PER_POUND_MOLE
STANDARD_PRESSURE_IMPERIAL = c.STANDARD_PRESSURE_IMPERIAL
STANDARD_TEMPERATURE_IMPERIAL = c.STANDARD_TEMPERATURE_IMPERIAL
STANDARD_WATER_DENSITY_IMPERIAL = c.STANDARD_WATER_DENSITY_IMPERIAL

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
    "compute_oil_compressibility_liberation_correction_term",
    "compute_oil_formation_volume_factor",
    "compute_oil_formation_volume_factor_standing",
    "compute_oil_formation_volume_factor_vazquez_and_beggs",
    "compute_oil_specific_gravity",
    "compute_oil_viscosity",
    "compute_todd_longstaff_effective_oil_density",
    "compute_todd_longstaff_effective_oil_viscosity",
    "compute_todd_longstaff_effective_solvent_density",
    "compute_todd_longstaff_effective_solvent_viscosity",
    "compute_todd_longstaff_mixture_viscosity",
    "correct_oil_fvf_for_pressure",
    "estimate_bubble_point_pressure_standing",
    "estimate_solution_gor",
    "get_vazquez_beggs_oil_bubble_point_pressure_coefficients",
    "get_vazquez_beggs_oil_fvf_coefficients",
]


@elementwise(
    positive=("oil_density",),
    non_negative=("oil_compressibility",),
    bounds={"temperature": (-FAHRENHEIT_TO_RANKINE_OFFSET, np.inf), "pressure": (0.0, np.inf)},
)
def compute_oil_specific_gravity(
    oil_density: NumberOrArray,
    pressure: NumberOrArray,
    temperature: NumberOrArray,
    oil_compressibility: NumberOrArray,
) -> NumberOrArray:
    """
    Converts oil density at reservoir conditions to the specific gravity of the stock
    tank oil by undoing the pressure and temperature effects.

    The reservoir density follows a linearized (exponential) equation of state:

        ρ = ρ_stp * exp[Co * (P - P_stp) - α * (T - T_stp)]

    so the density at standard conditions is

        ρ_stp = ρ * exp[Co * (P_stp - P) + α * (T - T_stp)]

    and the specific gravity is

        SG = ρ_stp / ρ_water

    where:
        - Co: Oil compressibility (psi⁻¹)
        - α: Oil thermal expansion coefficient (1/°F)
        - P_stp, T_stp: Standard pressure (14.696 psi) and temperature (60 °F)
        - ρ_water: Density of water at standard conditions (lbm/ft³)

    The density correction factor is limited to the range 0.2 to 2.0.

    :param oil_density: Oil density at reservoir conditions (lbm/ft³)
    :param pressure: Reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param oil_compressibility: Oil compressibility (psi⁻¹)
    :return: Specific gravity of oil (dimensionless)
    """
    delta_p = STANDARD_PRESSURE_IMPERIAL - pressure
    delta_t = temperature - STANDARD_TEMPERATURE_IMPERIAL
    correction_factor = np.exp(
        oil_compressibility * delta_p + OIL_THERMAL_EXPANSION_COEFFICIENT_IMPERIAL * delta_t
    )
    correction_factor = min(max(correction_factor, 0.2), 2.0)
    return oil_density * correction_factor / STANDARD_WATER_DENSITY_IMPERIAL


@elementwise(
    positive=("oil_specific_gravity", "gas_gravity"),
    non_negative=("gas_to_oil_ratio",),
    bounds={"temperature": (32.0, np.inf)},
)
def compute_oil_formation_volume_factor_standing(
    temperature: NumberOrArray,
    oil_specific_gravity: NumberOrArray,
    gas_gravity: NumberOrArray,
    gas_to_oil_ratio: NumberOrArray,
) -> NumberOrArray:
    """
    Computes the saturated oil formation volume factor (Bo) with the Standing correlation.

    Formula (Standing, 1981 form):

        Bo = 0.972 + 0.000147 * F^1.175
        F = R_s * (γ_g / γ_o)^0.5 + 1.25 * T

    where:
        - Bo: Oil formation volume factor (bbl/STB)
        - R_s: Solution gas-oil ratio (scf/STB)
        - γ_g: Gas specific gravity (air = 1.0)
        - γ_o: Oil specific gravity (water = 1.0)
        - T: Temperature (°F)

    Typical range: 60-300 °F, 0.5-0.95 oil specific gravity, 20-2000 scf/STB.

    :param temperature: Temperature (°F)
    :param oil_specific_gravity: Oil specific gravity (dimensionless)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :param gas_to_oil_ratio: Solution gas-oil ratio (scf/STB)
    :return: Oil formation volume factor (bbl/STB)
    """
    factor = gas_to_oil_ratio * (gas_gravity / oil_specific_gravity) ** 0.5 + 1.25 * temperature
    return 0.972 + 0.000147 * factor**1.175


@numba.njit(cache=True)
def get_vazquez_beggs_oil_fvf_coefficients(
    oil_api_gravity: Number,
) -> tuple[Number, Number, Number]:
    """
    Returns the coefficients (C1, C2, C3) of the Vazquez and Beggs oil formation
    volume factor correlation for an oil API gravity.

    API ≤ 30: (4.677e-4, 1.751e-5, -1.811e-8). API > 30: (4.670e-4, 1.100e-5, 1.337e-9).

    :param oil_api_gravity: Oil API gravity (°API)
    :return: The coefficients (C1, C2, C3).
    """
    if oil_api_gravity <= 30.0:
        return 4.677e-4, 1.751e-5, -1.811e-8
    return 4.670e-4, 1.100e-5, 1.337e-9


@elementwise(
    positive=("oil_specific_gravity", "gas_gravity"),
    non_negative=("gas_to_oil_ratio",),
)
def compute_oil_formation_volume_factor_vazquez_and_beggs(
    temperature: NumberOrArray,
    oil_specific_gravity: NumberOrArray,
    gas_gravity: NumberOrArray,
    gas_to_oil_ratio: NumberOrArray,
) -> NumberOrArray:
    """
    Computes the saturated oil formation volume factor (Bo) with the Vazquez and Beggs
    (1980) correlation.

        Bo = 1 + C1 * R_s + (T - 60) * (API / γ_g) * (C2 + C3 * R_s)

    where:
        - Bo: Oil formation volume factor (bbl/STB)
        - R_s: Solution gas-oil ratio (scf/STB)
        - API: Oil API gravity (°API)
        - γ_g: Gas specific gravity (air = 1.0). The correlation calls for the gas
          gravity corrected to a separator pressure of 100 psig; the value given is
          used as is.
        - T: Temperature (°F)

    Typical range: 16-58 °API, 70-295 °F, 0-2000 scf/STB.

    :param temperature: Reservoir temperature (°F)
    :param oil_specific_gravity: Oil specific gravity (dimensionless)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :param gas_to_oil_ratio: Solution gas-oil ratio (scf/STB)
    :return: Oil formation volume factor (bbl/STB)
    """
    oil_api_gravity = compute_oil_api_gravity(oil_specific_gravity)
    c1, c2, c3 = get_vazquez_beggs_oil_fvf_coefficients(oil_api_gravity)
    return (
        1.0
        + c1 * gas_to_oil_ratio
        + (temperature - 60.0) * (oil_api_gravity / gas_gravity) * (c2 + c3 * gas_to_oil_ratio)
    )


@elementwise(positive=("saturated_oil_fvf",), non_negative=("oil_compressibility",))
def correct_oil_fvf_for_pressure(
    saturated_oil_fvf: NumberOrArray,
    oil_compressibility: NumberOrArray,
    bubble_point_pressure: NumberOrArray,
    current_pressure: NumberOrArray,
) -> NumberOrArray:
    """
    Applies the exponential shrinkage of undersaturated oil to the oil formation volume
    factor at the bubble point.

        Bo(P) = Bo_sat * exp[Co * (Pb - P)]    for P > Pb
        Bo(P) = Bo_sat                         for P ≤ Pb

    :param saturated_oil_fvf: Oil formation volume factor at the bubble point (bbl/STB)
    :param oil_compressibility: Undersaturated oil compressibility (psi⁻¹)
    :param bubble_point_pressure: Bubble point pressure (psi)
    :param current_pressure: Current pressure (psi)
    :return: Oil formation volume factor at the current pressure (bbl/STB)
    """
    if current_pressure <= bubble_point_pressure:
        return saturated_oil_fvf
    delta_p = bubble_point_pressure - current_pressure
    correction_factor = min(max(np.exp(oil_compressibility * delta_p), 1e-6), 5.0)
    return saturated_oil_fvf * correction_factor


@elementwise(
    positive=("oil_specific_gravity", "gas_gravity"),
    non_negative=("gas_to_oil_ratio", "oil_compressibility"),
    bounds={"temperature": (32.0, np.inf)},
)
def compute_oil_formation_volume_factor(
    pressure: NumberOrArray,
    temperature: NumberOrArray,
    bubble_point_pressure: NumberOrArray,
    oil_specific_gravity: NumberOrArray,
    gas_gravity: NumberOrArray,
    gas_to_oil_ratio: NumberOrArray,
    oil_compressibility: NumberOrArray,
) -> NumberOrArray:
    """
    Computes the oil formation volume factor (Bo) at any pressure.

    The saturated Bo uses the Standing correlation up to 100 °F and the Vazquez and
    Beggs correlation above 100 °F. Above the bubble point, the saturated Bo is
    shrunk with the undersaturated oil compressibility.

    :param pressure: Reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param bubble_point_pressure: Bubble point pressure (psi)
    :param oil_specific_gravity: Oil specific gravity (dimensionless)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :param gas_to_oil_ratio: Solution gas-oil ratio at the pressure (scf/STB)
    :param oil_compressibility: Oil compressibility (psi⁻¹)
    :return: Oil formation volume factor (bbl/STB)
    """
    if temperature <= 100.0:
        saturated_fvf = compute_oil_formation_volume_factor_standing(
            temperature, oil_specific_gravity, gas_gravity, gas_to_oil_ratio
        )
    else:
        saturated_fvf = compute_oil_formation_volume_factor_vazquez_and_beggs(
            temperature, oil_specific_gravity, gas_gravity, gas_to_oil_ratio
        )
    return correct_oil_fvf_for_pressure(
        saturated_fvf, oil_compressibility, bubble_point_pressure, pressure
    )


@elementwise(positive=("oil_specific_gravity",))
def compute_oil_api_gravity(oil_specific_gravity: NumberOrArray) -> NumberOrArray:
    """
    Computes the API gravity from the oil specific gravity.

        API = 141.5 / SG - 131.5

    :param oil_specific_gravity: Oil specific gravity (dimensionless, water = 1.0 at 60 °F)
    :return: API gravity (°API)
    """
    return 141.5 / oil_specific_gravity - 131.5


@numba.njit(cache=True)
def get_vazquez_beggs_oil_bubble_point_pressure_coefficients(
    oil_api_gravity: Number,
) -> tuple[Number, Number, Number]:
    """
    Returns the coefficients (C1, C2, C3) of the Vazquez and Beggs solution gas-oil
    ratio and bubble point pressure correlations for an oil API gravity.

    API ≤ 30: (0.0362, 1.0937, 25.7240). API > 30: (0.0178, 1.1870, 23.9310).

    :param oil_api_gravity: Oil API gravity (°API)
    :return: The coefficients (C1, C2, C3).
    """
    if oil_api_gravity <= 30.0:
        return 0.0362, 1.0937, 25.7240
    return 0.0178, 1.1870, 23.9310


@elementwise(
    positive=("gas_gravity", "oil_api_gravity"),
    non_negative=("gas_to_oil_ratio",),
    bounds={"temperature": (32.0, np.inf)},
)
def compute_oil_bubble_point_pressure(
    gas_gravity: NumberOrArray,
    oil_api_gravity: NumberOrArray,
    temperature: NumberOrArray,
    gas_to_oil_ratio: NumberOrArray,
) -> NumberOrArray:
    """
    Computes the bubble point pressure with the Vazquez and Beggs correlation.

        Pb = [R_s / (C1 * γ_g * exp(C3 * API / T_R))]^(1 / C2)

    where:
        - Pb: Bubble point pressure (psi)
        - R_s: Solution gas-oil ratio at the bubble point (scf/STB)
        - γ_g: Gas specific gravity (air = 1.0)
        - API: Oil API gravity (°API)
        - T_R: Temperature (°R)
        - C1, C2, C3: Coefficients that depend on the API gravity

    Typical range: 16-58 °API, 70-295 °F, 0-2000 scf/STB.

    :param gas_gravity: Gas specific gravity (dimensionless)
    :param oil_api_gravity: Oil API gravity (°API)
    :param temperature: Temperature (°F)
    :param gas_to_oil_ratio: Solution gas-oil ratio at the bubble point (scf/STB)
    :return: Bubble point pressure (psi)
    """
    c1, c2, c3 = get_vazquez_beggs_oil_bubble_point_pressure_coefficients(oil_api_gravity)
    temperature_rankine = temperature + FAHRENHEIT_TO_RANKINE_OFFSET
    return (
        gas_to_oil_ratio / (c1 * gas_gravity * np.exp(c3 * oil_api_gravity / temperature_rankine))
    ) ** (1.0 / c2)


@elementwise(
    positive=("pressure", "gas_gravity", "oil_api_gravity"),
    bounds={"temperature": (-FAHRENHEIT_TO_RANKINE_OFFSET, np.inf)},
)
def compute_gas_to_oil_ratio(
    pressure: NumberOrArray,
    temperature: NumberOrArray,
    bubble_point_pressure: NumberOrArray,
    gas_gravity: NumberOrArray,
    oil_api_gravity: NumberOrArray,
    gor_at_bubble_point_pressure: NumberOrArray = np.nan,
) -> NumberOrArray:
    """
    Computes the solution gas-oil ratio (Rs) with the Vazquez and Beggs correlation.

    Below the bubble point, Rs depends on pressure:

        Rs = C1 * γ_g * P^C2 * exp(C3 * API / T_R)

    At and above the bubble point the oil is undersaturated and Rs stays at its value
    at the bubble point.

    where:
        - Rs: Solution gas-oil ratio (scf/STB)
        - P: Pressure (psi)
        - γ_g: Gas specific gravity (air = 1.0)
        - API: Oil API gravity (°API)
        - T_R: Temperature (°R)
        - C1, C2, C3: Coefficients that depend on the API gravity

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param bubble_point_pressure: Bubble point pressure (psi)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :param oil_api_gravity: Oil API gravity (°API)
    :param gor_at_bubble_point_pressure: Rs at the bubble point (scf/STB). Computed from
        the correlation when omitted (NaN).
    :return: Solution gas-oil ratio (scf/STB)
    """
    c1, c2, c3 = get_vazquez_beggs_oil_bubble_point_pressure_coefficients(oil_api_gravity)
    temperature_rankine = temperature + FAHRENHEIT_TO_RANKINE_OFFSET
    exponential = np.exp(c3 * oil_api_gravity / temperature_rankine)
    if pressure >= bubble_point_pressure:
        if np.isnan(gor_at_bubble_point_pressure):
            gor = c1 * gas_gravity * bubble_point_pressure**c2 * exponential
        else:
            gor = gor_at_bubble_point_pressure
    else:
        gor = c1 * gas_gravity * pressure**c2 * exponential
    return max(0.0, gor)


@elementwise(positive=("oil_specific_gravity", "temperature"))
def compute_dead_oil_viscosity_modified_beggs(
    temperature: NumberOrArray,
    oil_specific_gravity: NumberOrArray,
) -> NumberOrArray:
    """
    Computes the dead oil viscosity with the Egbogah and Ng (1990) correlation, the
    modified Beggs and Robinson correlation.

        log10(log10(μ_od + 1)) = 1.8653 - 0.025086 * API - 0.5644 * log10(T)

    where:
        - μ_od: Dead oil viscosity (cP)
        - API: Oil API gravity (°API)
        - T: Temperature (°F)

    Typical range: 5-58 °API, 70-295 °F.

    :param temperature: Temperature (°F)
    :param oil_specific_gravity: Oil specific gravity (dimensionless)
    :return: Dead oil viscosity (cP)
    """
    oil_api_gravity = compute_oil_api_gravity(oil_specific_gravity)
    exponent = 1.8653 - 0.025086 * oil_api_gravity - 0.5644 * np.log10(temperature)
    viscosity = 10.0 ** (10.0**exponent) - 1.0
    return max(0.0, viscosity)


@elementwise(
    positive=("pressure", "temperature", "bubble_point_pressure", "oil_specific_gravity"),
    non_negative=("gas_to_oil_ratio", "gor_at_bubble_point_pressure"),
)
def compute_oil_viscosity(
    pressure: NumberOrArray,
    temperature: NumberOrArray,
    bubble_point_pressure: NumberOrArray,
    oil_specific_gravity: NumberOrArray,
    gas_to_oil_ratio: NumberOrArray,
    gor_at_bubble_point_pressure: NumberOrArray,
) -> NumberOrArray:
    """
    Computes the oil viscosity of dead, saturated and undersaturated oil.

    Saturated oil (Beggs and Robinson):

        μ_os = A * μ_od^B
        A = 10.715 * (Rs + 100)^-0.515
        B = 5.44 * (Rs + 150)^-0.338

    Undersaturated oil (Vazquez and Beggs):

        μ_o = μ_ob * (P / Pb)^m
        m = 2.6 * P^1.187 * exp(-11.513 - 8.98e-5 * P)

    where:
        - μ_od: Dead oil viscosity (Egbogah and Ng, see
          `compute_dead_oil_viscosity_modified_beggs`)
        - μ_ob: Saturated oil viscosity at the bubble point (cP)
        - Rs: Solution gas-oil ratio (scf/STB)

    :param pressure: Reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param bubble_point_pressure: Bubble point pressure (psi)
    :param oil_specific_gravity: Oil specific gravity (dimensionless)
    :param gas_to_oil_ratio: Solution gas-oil ratio at the pressure (scf/STB)
    :param gor_at_bubble_point_pressure: Solution gas-oil ratio at the bubble point (scf/STB)
    :return: Oil viscosity (cP), at least 1e-6
    """
    dead_oil_viscosity = compute_dead_oil_viscosity_modified_beggs(
        temperature, oil_specific_gravity
    )
    if pressure <= bubble_point_pressure:
        a = 10.715 * (gas_to_oil_ratio + 100.0) ** -0.515
        b = 5.44 * (gas_to_oil_ratio + 150.0) ** -0.338
        return max(a * dead_oil_viscosity**b, 1e-6)

    a_bubble_point = 10.715 * (gor_at_bubble_point_pressure + 100.0) ** -0.515
    b_bubble_point = 5.44 * (gor_at_bubble_point_pressure + 150.0) ** -0.338
    bubble_point_viscosity = a_bubble_point * dead_oil_viscosity**b_bubble_point
    exponent = 2.6 * pressure**1.187 * np.exp(-11.513 - 8.98e-5 * pressure)
    return max(bubble_point_viscosity * (pressure / bubble_point_pressure) ** exponent, 1e-6)


@elementwise(positive=("pressure", "oil_formation_volume_factor"))
def compute_oil_compressibility_liberation_correction_term(
    pressure: NumberOrArray,
    temperature: NumberOrArray,
    gas_gravity: NumberOrArray,
    oil_api_gravity: NumberOrArray,
    bubble_point_pressure: NumberOrArray,
    gas_formation_volume_factor: NumberOrArray,
    oil_formation_volume_factor: NumberOrArray,
    gor_at_bubble_point_pressure: NumberOrArray,
) -> NumberOrArray:
    """
    Computes the gas liberation term of oil compressibility below the bubble point.

        x = (Bg / Bo) * dRs/dP / 5.615

    where `dRs/dP` is a central difference of the solution gas-oil ratio over a small
    pressure step.

    where:
        - Bg: Gas formation volume factor (ft³/scf)
        - Bo: Oil formation volume factor (bbl/STB)
        - Rs: Solution gas-oil ratio (scf/STB)
        - 5.615: Cubic feet per barrel

    :param pressure: Current reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :param oil_api_gravity: Oil API gravity (°API)
    :param bubble_point_pressure: Bubble point pressure (psi)
    :param gas_formation_volume_factor: Gas formation volume factor (ft³/scf)
    :param oil_formation_volume_factor: Oil formation volume factor (bbl/STB)
    :param gor_at_bubble_point_pressure: Solution gas-oil ratio at the bubble point (scf/STB)
    :return: Liberation correction term (psi⁻¹)
    """
    delta_p = max(0.01, 1e-4 * pressure)
    pressure_plus = pressure + delta_p
    pressure_minus = pressure - delta_p
    gor_plus = compute_gas_to_oil_ratio(
        pressure_plus,
        temperature,
        bubble_point_pressure,
        gas_gravity,
        oil_api_gravity,
        gor_at_bubble_point_pressure,
    )
    if pressure_minus > 0.0:
        gor_minus = compute_gas_to_oil_ratio(
            pressure_minus,
            temperature,
            bubble_point_pressure,
            gas_gravity,
            oil_api_gravity,
            gor_at_bubble_point_pressure,
        )
        derivative = (gor_plus - gor_minus) / (2.0 * delta_p)
    else:
        derivative = gor_plus / pressure_plus
    return (
        gas_formation_volume_factor
        / oil_formation_volume_factor
        * derivative
        / BARRELS_TO_CUBIC_FEET
    )


@elementwise(positive=("pressure", "bubble_point_pressure", "gas_gravity", "oil_api_gravity"))
def compute_base_compressibility(
    pressure: NumberOrArray,
    temperature: NumberOrArray,
    bubble_point_pressure: NumberOrArray,
    oil_api_gravity: NumberOrArray,
    gas_gravity: NumberOrArray,
    gor_at_bubble_point_pressure: NumberOrArray,
) -> NumberOrArray:
    """
    Computes the oil compressibility with the Vazquez and Beggs (1980) correlation.

        Co = (-1433 + 5 * Rs + 17.2 * T - 1180 * γ_g + 12.61 * API) / (1e5 * P)

    where Rs is the solution gas-oil ratio at the pressure, held at its bubble point
    value above the bubble point. The correlation is derived for undersaturated oil.
    Below the bubble point it gives the compressibility of the liquid phase alone.

    Typical range: 16-58 °API, 70-295 °F, 126-9500 psi.

    :param pressure: Reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param bubble_point_pressure: Bubble point pressure (psi)
    :param oil_api_gravity: Oil API gravity (°API)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :param gor_at_bubble_point_pressure: Solution gas-oil ratio at the bubble point (scf/STB)
    :return: Oil compressibility (psi⁻¹), not less than zero
    """
    gor = compute_gas_to_oil_ratio(
        pressure,
        temperature,
        bubble_point_pressure,
        gas_gravity,
        oil_api_gravity,
        gor_at_bubble_point_pressure,
    )
    value = (
        -1433.0 + 5.0 * gor + 17.2 * temperature - 1180.0 * gas_gravity + 12.61 * oil_api_gravity
    ) / (1e5 * pressure)
    return max(value, 0.0)


@elementwise(
    positive=(
        "pressure",
        "bubble_point_pressure",
        "gas_gravity",
        "oil_api_gravity",
        "oil_formation_volume_factor",
    ),
)
def compute_oil_compressibility(
    pressure: NumberOrArray,
    temperature: NumberOrArray,
    bubble_point_pressure: NumberOrArray,
    oil_api_gravity: NumberOrArray,
    gas_gravity: NumberOrArray,
    gor_at_bubble_point_pressure: NumberOrArray,
    gas_formation_volume_factor: NumberOrArray,
    oil_formation_volume_factor: NumberOrArray,
) -> NumberOrArray:
    """
    Computes the oil compressibility at any pressure.

    Above the bubble point it is the Vazquez and Beggs compressibility
    (`compute_base_compressibility`). At and below the bubble point the gas liberation
    term is added (`compute_oil_compressibility_liberation_correction_term`):

        Co = Co_base + (Bg / Bo) * dRs/dP / 5.615

    :param pressure: Reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param bubble_point_pressure: Bubble point pressure (psi)
    :param oil_api_gravity: Oil API gravity (°API)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :param gor_at_bubble_point_pressure: Solution gas-oil ratio at the bubble point (scf/STB)
    :param gas_formation_volume_factor: Gas formation volume factor (ft³/scf)
    :param oil_formation_volume_factor: Oil formation volume factor (bbl/STB)
    :return: Oil compressibility (psi⁻¹)
    """
    base = compute_base_compressibility(
        pressure,
        temperature,
        bubble_point_pressure,
        oil_api_gravity,
        gas_gravity,
        gor_at_bubble_point_pressure,
    )
    if pressure > bubble_point_pressure:
        return base
    return base + compute_oil_compressibility_liberation_correction_term(
        pressure,
        temperature,
        gas_gravity,
        oil_api_gravity,
        bubble_point_pressure,
        gas_formation_volume_factor,
        oil_formation_volume_factor,
        gor_at_bubble_point_pressure,
    )


@elementwise(
    positive=("api_gravity", "gas_gravity", "formation_volume_factor"),
    non_negative=("gas_to_oil_ratio",),
    bounds={"api_gravity": (-131.4, np.inf)},
)
def compute_live_oil_density(
    api_gravity: NumberOrArray,
    gas_gravity: NumberOrArray,
    gas_to_oil_ratio: NumberOrArray,
    formation_volume_factor: NumberOrArray,
) -> NumberOrArray:
    """
    Computes the density of live oil at reservoir conditions from a mass balance on one
    stock tank barrel.

        ρ_o = (m_oil + m_gas) / (Bo * 5.615)

    where `m_oil` is the mass of the stock tank oil and `m_gas = Rs * γ_g * 28.96 / 379.49`
    the mass of the dissolved gas, both per stock tank barrel.

    :param api_gravity: Oil API gravity (°API)
    :param gas_gravity: Gas specific gravity (air = 1.0)
    :param gas_to_oil_ratio: Solution gas-oil ratio at the pressure (scf/STB)
    :param formation_volume_factor: Oil formation volume factor at the pressure (bbl/STB)
    :return: Live oil density (lbm/ft³)
    """
    stock_tank_oil_density = 141.5 / (api_gravity + 131.5) * STANDARD_WATER_DENSITY_IMPERIAL
    mass_stock_tank_oil = stock_tank_oil_density / CUBIC_FEET_TO_STB
    gas_mass_per_scf = gas_gravity * MOLECULAR_WEIGHT_AIR / SCF_PER_POUND_MOLE
    mass_dissolved_gas = gas_to_oil_ratio * gas_mass_per_scf
    volume = formation_volume_factor * BARRELS_TO_CUBIC_FEET
    return (mass_stock_tank_oil + mass_dissolved_gas) / volume


@elementwise(
    non_negative=("pressure", "temperature"),
    positive=("gas_gravity",),
    bounds={"oil_api_gravity": (10.0, np.inf)},
)
def compute_gas_to_oil_ratio_standing(
    pressure: NumberOrArray,
    temperature: NumberOrArray,
    oil_api_gravity: NumberOrArray,
    gas_gravity: NumberOrArray,
) -> NumberOrArray:
    """
    Computes the saturated solution gas-oil ratio with the Standing (1947) correlation.

        Rs = γ_g * [(P / 18.2 + 1.4) * 10^(0.0125 * API - 0.00091 * T)]^1.2048

    where:
        - Rs: Solution gas-oil ratio (scf/STB)
        - P: Pressure (psia)
        - API: Oil API gravity (°API)
        - T: Temperature (°F)
        - γ_g: Gas specific gravity (air = 1.0)

    Typical range: 130-7000 psia, 100-258 °F, 16-64 °API, 20-1425 scf/STB.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param oil_api_gravity: Oil API gravity (°API)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :return: Solution gas-oil ratio (scf/STB)
    """
    exponent_base = (pressure / 18.2 + 1.4) * 10.0 ** (
        0.0125 * oil_api_gravity - 0.00091 * temperature
    )
    return gas_gravity * exponent_base**1.2048


@elementwise(
    positive=("gas_gravity", "observed_gas_to_oil_ratio"),
    bounds={"oil_api_gravity": (10.0, np.inf)},
)
def estimate_bubble_point_pressure_standing(
    oil_api_gravity: NumberOrArray,
    gas_gravity: NumberOrArray,
    observed_gas_to_oil_ratio: NumberOrArray,
    temperature: NumberOrArray,
) -> NumberOrArray:
    """
    Estimates the bubble point pressure with the Standing (1947) correlation, the exact
    inverse of `compute_gas_to_oil_ratio_standing`.

        Pb = 18.2 * [(Rs / γ_g)^(1 / 1.2048) * 10^(0.00091 * T - 0.0125 * API) - 1.4]

    Standing publishes the exponent `1 / 1.2048` rounded to 0.83.

    The result is not less than the standard pressure (14.696 psi).

    :param oil_api_gravity: Oil API gravity (°API)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :param observed_gas_to_oil_ratio: Solution gas-oil ratio at the bubble point (scf/STB)
    :param temperature: Temperature (°F)
    :return: Bubble point pressure (psi)
    """
    pressure = 18.2 * (
        (observed_gas_to_oil_ratio / gas_gravity) ** (1.0 / 1.2048)
        * 10.0 ** (0.00091 * temperature - 0.0125 * oil_api_gravity)
        - 1.4
    )
    return max(pressure, STANDARD_PRESSURE_IMPERIAL)


@elementwise(
    positive=("pressure", "gas_gravity", "oil_api_gravity"),
    bounds={"temperature": (-FAHRENHEIT_TO_RANKINE_OFFSET, np.inf)},
)
def estimate_solution_gor(
    pressure: NumberOrArray,
    temperature: NumberOrArray,
    oil_api_gravity: NumberOrArray,
    gas_gravity: NumberOrArray,
) -> NumberOrArray:
    """
    Estimates the saturated solution gas-oil ratio at a pressure with the Vazquez and
    Beggs correlation, for use where the bubble point pressure is not known.

    It is the Rs of oil that is exactly at its bubble point at that pressure, which
    makes it the inverse of `compute_oil_bubble_point_pressure`.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param oil_api_gravity: Oil API gravity (°API)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :return: Saturated solution gas-oil ratio (scf/STB)
    """
    c1, c2, c3 = get_vazquez_beggs_oil_bubble_point_pressure_coefficients(oil_api_gravity)
    temperature_rankine = temperature + FAHRENHEIT_TO_RANKINE_OFFSET
    return c1 * gas_gravity * pressure**c2 * np.exp(c3 * oil_api_gravity / temperature_rankine)


@elementwise(positive=("transition_width",))
def compute_miscibility_transition_factor(
    pressure: NumberOrArray,
    minimum_miscibility_pressure: NumberOrArray,
    transition_width: NumberOrArray = 500.0,
) -> NumberOrArray:
    """
    Computes the degree of miscibility, rising smoothly from 0 well below the minimum
    miscibility pressure (MMP) to 1 well above it.

        f(P) = 0.5 * (1 + tanh((P - MMP) / ΔP))

    The factor is 0.12 at `MMP - ΔP`, 0.5 at the MMP and 0.88 at `MMP + ΔP`.

    :param pressure: Current reservoir pressure (psi)
    :param minimum_miscibility_pressure: Minimum miscibility pressure (psi)
    :param transition_width: Pressure width of the transition zone (psi)
    :return: Miscibility transition factor, between 0 and 1
    """
    return 0.5 * (1.0 + np.tanh((pressure - minimum_miscibility_pressure) / transition_width))


@elementwise(positive=("transition_width",), bounds={"base_omega": (0.0, 1.0)})
def compute_effective_todd_longstaff_omega(
    pressure: NumberOrArray,
    base_omega: NumberOrArray,
    minimum_miscibility_pressure: NumberOrArray,
    transition_width: NumberOrArray = 500.0,
) -> NumberOrArray:
    """
    Computes the Todd-Longstaff mixing parameter at a pressure, as the base value scaled
    by the miscibility transition factor.

    :param pressure: Current reservoir pressure (psi)
    :param base_omega: Todd-Longstaff mixing parameter when fully miscible (0 to 1).
        Typically 0.67.
    :param minimum_miscibility_pressure: Minimum miscibility pressure (psi)
    :param transition_width: Pressure width of the transition zone (psi)
    :return: Effective mixing parameter, between 0 and `base_omega`
    """
    return base_omega * compute_miscibility_transition_factor(
        pressure, minimum_miscibility_pressure, transition_width
    )


@elementwise(
    positive=("oil_viscosity", "solvent_viscosity"),
    bounds={"solvent_concentration": (0.0, 1.0)},
)
def compute_todd_longstaff_mixture_viscosity(
    oil_viscosity: NumberOrArray,
    solvent_viscosity: NumberOrArray,
    solvent_concentration: NumberOrArray,
) -> NumberOrArray:
    """
    Computes the viscosity of a fully mixed oil and solvent with the quarter-power
    mixing rule used by the Todd and Longstaff (1972) model.

        μ_m = μ_o * μ_s / (C_s * μ_o^¼ + C_o * μ_s^¼)^4

    where `C_s` is the solvent volume fraction and `C_o = 1 - C_s`.

    :param oil_viscosity: Oil viscosity (cP)
    :param solvent_viscosity: Solvent viscosity (cP)
    :param solvent_concentration: Solvent volume fraction (0 to 1)
    :return: Fully mixed viscosity (cP)
    """
    oil_fraction = 1.0 - solvent_concentration
    denominator = (
        solvent_concentration * oil_viscosity**0.25 + oil_fraction * solvent_viscosity**0.25
    )
    return oil_viscosity * solvent_viscosity / denominator**4


@elementwise(
    positive=("oil_viscosity", "solvent_viscosity"),
    bounds={"solvent_concentration": (0.0, 1.0), "omega": (0.0, 1.0)},
)
def compute_todd_longstaff_effective_oil_viscosity(
    oil_viscosity: NumberOrArray,
    solvent_viscosity: NumberOrArray,
    solvent_concentration: NumberOrArray,
    omega: NumberOrArray = 0.67,
) -> NumberOrArray:
    """
    Computes the effective viscosity of the oil in the Todd and Longstaff (1972) model.

        μ_o,eff = μ_o^(1 - ω) * μ_m^ω

    where `μ_m` is the fully mixed viscosity (`compute_todd_longstaff_mixture_viscosity`)
    and `ω` the mixing parameter, from 0 (fully segregated, the oil keeps its own
    viscosity) to 1 (fully mixed).

    :param oil_viscosity: Oil viscosity (cP)
    :param solvent_viscosity: Solvent viscosity (cP)
    :param solvent_concentration: Solvent volume fraction (0 to 1)
    :param omega: Mixing parameter (0 to 1). Typically 0.67.
    :return: Effective oil viscosity (cP)
    """
    mixture_viscosity = compute_todd_longstaff_mixture_viscosity(
        oil_viscosity, solvent_viscosity, solvent_concentration
    )
    return oil_viscosity ** (1.0 - omega) * mixture_viscosity**omega


@elementwise(
    positive=("oil_viscosity", "solvent_viscosity"),
    bounds={"solvent_concentration": (0.0, 1.0), "omega": (0.0, 1.0)},
)
def compute_todd_longstaff_effective_solvent_viscosity(
    oil_viscosity: NumberOrArray,
    solvent_viscosity: NumberOrArray,
    solvent_concentration: NumberOrArray,
    omega: NumberOrArray = 0.67,
) -> NumberOrArray:
    """
    Computes the effective viscosity of the solvent in the Todd and Longstaff (1972) model.

        μ_s,eff = μ_s^(1 - ω) * μ_m^ω

    where `μ_m` is the fully mixed viscosity (`compute_todd_longstaff_mixture_viscosity`)
    and `ω` the mixing parameter, from 0 (fully segregated, the solvent keeps its own
    viscosity) to 1 (fully mixed).

    :param oil_viscosity: Oil viscosity (cP)
    :param solvent_viscosity: Solvent viscosity (cP)
    :param solvent_concentration: Solvent volume fraction (0 to 1)
    :param omega: Mixing parameter (0 to 1). Typically 0.67.
    :return: Effective solvent viscosity (cP)
    """
    mixture_viscosity = compute_todd_longstaff_mixture_viscosity(
        oil_viscosity, solvent_viscosity, solvent_concentration
    )
    return solvent_viscosity ** (1.0 - omega) * mixture_viscosity**omega


@elementwise(
    positive=("oil_density", "solvent_density", "oil_viscosity", "solvent_viscosity"),
    bounds={"solvent_concentration": (0.0, 1.0), "omega": (0.0, 1.0)},
)
def compute_todd_longstaff_effective_oil_density(
    oil_density: NumberOrArray,
    solvent_density: NumberOrArray,
    oil_viscosity: NumberOrArray,
    solvent_viscosity: NumberOrArray,
    solvent_concentration: NumberOrArray,
    omega: NumberOrArray = 0.67,
) -> NumberOrArray:
    """
    Computes the effective density of the oil in the Todd and Longstaff (1972) model.

    The effective oil is a blend of oil and solvent whose viscosity is the effective
    oil viscosity. The blend fraction of oil follows from the quarter-power mixing rule:

        f_o = (μ_o,eff^-¼ - μ_s^-¼) / (μ_o^-¼ - μ_s^-¼)
        ρ_o,eff = f_o * ρ_o + (1 - f_o) * ρ_s

    This gives the oil density at `ω = 0` and the volume-weighted mixture density at
    `ω = 1`. When the oil and solvent have the same viscosity, `f_o` is interpolated
    linearly between those two limits.

    :param oil_density: Oil density (lbm/ft³)
    :param solvent_density: Solvent density (lbm/ft³)
    :param oil_viscosity: Oil viscosity (cP)
    :param solvent_viscosity: Solvent viscosity (cP)
    :param solvent_concentration: Solvent volume fraction (0 to 1)
    :param omega: Mixing parameter (0 to 1). Typically 0.67.
    :return: Effective oil density (same units as the input densities)
    """
    effective_viscosity = compute_todd_longstaff_effective_oil_viscosity(
        oil_viscosity, solvent_viscosity, solvent_concentration, omega
    )
    span = oil_viscosity**-0.25 - solvent_viscosity**-0.25
    if abs(span) < 1e-12:
        oil_blend_fraction = (1.0 - omega) + omega * (1.0 - solvent_concentration)
    else:
        oil_blend_fraction = (effective_viscosity**-0.25 - solvent_viscosity**-0.25) / span
    return oil_blend_fraction * oil_density + (1.0 - oil_blend_fraction) * solvent_density


@elementwise(
    positive=("oil_density", "solvent_density", "oil_viscosity", "solvent_viscosity"),
    bounds={"solvent_concentration": (0.0, 1.0), "omega": (0.0, 1.0)},
)
def compute_todd_longstaff_effective_solvent_density(
    oil_density: NumberOrArray,
    solvent_density: NumberOrArray,
    oil_viscosity: NumberOrArray,
    solvent_viscosity: NumberOrArray,
    solvent_concentration: NumberOrArray,
    omega: NumberOrArray = 0.67,
) -> NumberOrArray:
    """
    Computes the effective density of the solvent in the Todd and Longstaff (1972) model.

    The effective solvent is a blend of solvent and oil whose viscosity is the effective
    solvent viscosity. The blend fraction of solvent follows from the quarter-power
    mixing rule:

        f_s = (μ_s,eff^-¼ - μ_o^-¼) / (μ_s^-¼ - μ_o^-¼)
        ρ_s,eff = f_s * ρ_s + (1 - f_s) * ρ_o

    This gives the solvent density at `ω = 0` and the volume-weighted mixture density at
    `ω = 1`. When the oil and solvent have the same viscosity, `f_s` is interpolated
    linearly between those two limits.

    :param oil_density: Oil density (lbm/ft³)
    :param solvent_density: Solvent density (lbm/ft³)
    :param oil_viscosity: Oil viscosity (cP)
    :param solvent_viscosity: Solvent viscosity (cP)
    :param solvent_concentration: Solvent volume fraction (0 to 1)
    :param omega: Mixing parameter (0 to 1). Typically 0.67.
    :return: Effective solvent density (same units as the input densities)
    """
    effective_viscosity = compute_todd_longstaff_effective_solvent_viscosity(
        oil_viscosity, solvent_viscosity, solvent_concentration, omega
    )
    span = solvent_viscosity**-0.25 - oil_viscosity**-0.25
    if abs(span) < 1e-12:
        solvent_blend_fraction = (1.0 - omega) + omega * solvent_concentration
    else:
        solvent_blend_fraction = (effective_viscosity**-0.25 - oil_viscosity**-0.25) / span
    return solvent_blend_fraction * solvent_density + (1.0 - solvent_blend_fraction) * oil_density

import logging
import warnings

import numpy as np
from scipy.optimize import brentq  # type: ignore[import-untyped]

from bores.constants import c
from bores.correlations.core import (
    HENRY_COEFFICIENTS,
    SETSCHENOW_CONSTANTS,
    fahrenheit_to_celsius,
    fahrenheit_to_kelvin,
    get_gas_symbol,
)
from bores.correlations.scalars.core import compute_fluid_density
from bores.errors import ComputationError, ValidationError
from bores.types import Number

logger = logging.getLogger(__name__)

__all__ = [
    "compute_gas_free_water_formation_volume_factor",
    "compute_gas_solubility_in_water",
    "compute_water_bubble_point_pressure",
    "compute_water_bubble_point_pressure_mccain",
    "compute_water_compressibility",
    "compute_water_density",
    "compute_water_density_batzle",
    "compute_water_density_mccain",
    "compute_water_formation_volume_factor",
    "compute_water_formation_volume_factor_mccain",
    "compute_water_viscosity",
]


def compute_water_formation_volume_factor(water_density: Number, salinity: Number) -> Number:
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


def compute_water_formation_volume_factor_mccain(
    pressure: Number,
    temperature: Number,
    salinity: Number = 0.0,
    gas_solubility: Number = 0.0,
) -> Number:
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
        -1.0001e-2 + 1.33391e-4 * temperature_in_celsius + 5.50654e-7 * temperature_in_celsius**2
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
    delta_V_wsp = salinity_wt_percent * (0.1249 + 1.1638e-4 * pressure - 1.1689e-6 * pressure**2)

    # Base FVF (gas-free)
    B_w = (1 + delta_V_wT) * (1 + delta_V_wp) * (1 + delta_V_wsp)

    # Correction for dissolved gas (if present)
    if gas_solubility > 0:
        # Rs_w in SCF/STB
        B_w = B_w - (gas_solubility * 1.0e-6)  # Approximate correction
    return B_w  # type: ignore


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
    bubble_point_pressure = max(0.0, (gas_solubility_in_water - A) / denominator)
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
    gas = get_gas_symbol(gas)
    if gas == "ch4" and 100 <= temperature <= 400:
        # Inverted McCain
        return compute_water_bubble_point_pressure_mccain(
            temperature=temperature,
            gas_solubility_in_water=gas_solubility_in_water,
            salinity=salinity,
        )

    lower_bound_pressure = c.MINIMUM_VALID_PRESSURE
    upper_bound_pressure = c.MAXIMUM_VALID_PRESSURE

    lower_boundsolubility = (
        compute_gas_solubility_in_water(
            pressure=lower_bound_pressure,
            temperature=temperature,
            salinity=salinity,
            gas=gas,
        )
        - c.GAS_SOLUBILITY_TOLERANCE
    )
    upper_boundsolubility = (
        compute_gas_solubility_in_water(
            pressure=upper_bound_pressure,
            temperature=temperature,
            salinity=salinity,
            gas=gas,
        )
        + c.GAS_SOLUBILITY_TOLERANCE
    )

    if not (lower_boundsolubility <= gas_solubility_in_water <= upper_boundsolubility):
        raise ComputationError(
            f"Target gas solubility {gas_solubility_in_water}SCF/STB is outside the range "
            f"[{lower_boundsolubility:.6f}, {upper_boundsolubility:.6f}] "
            f"for gas '{gas}' at T={temperature}°F and salinity={salinity}ppm."
        )

    # Use numerical solver for Duan/Henry
    # For gases like CO₂ and N₂ where no direct analytical formula exists to compute
    # the bubble point pressure, we numerically invert the solubility model (e.g., Duan, Henry's).
    # This inversion finds the pressure at which gas solubility in water equals the specified value.
    # Though these models don't explicitly define a bubble point, this process yields the effective
    # bubble point pressure—i.e., the pressure where gas begins to come out of solution.
    def residual(pressure: Number) -> Number:
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


def _compute_water_viscosity(
    temperature: Number,
    salinity: Number,
    pressure: Number,
    ppm_to_weight_fraction: Number,
) -> Number:
    salinity_fraction = salinity * ppm_to_weight_fraction
    A = 1.0 + 1.17 * salinity_fraction + 3.15e-6 * salinity_fraction**2
    B = 1.48e-3 - 1.8e-7 * salinity_fraction
    C = 2.94e-6

    viscosity_at_standard_pressure = A - (B * temperature) + (C * temperature**2)
    pressure_correction_factor = 0.9994 + (4.0295e-5 * pressure) + (3.1062e-9 * pressure**2)
    viscosity_at_pressure = viscosity_at_standard_pressure * pressure_correction_factor
    return max(viscosity_at_pressure, 1e-6)


def compute_water_viscosity(
    temperature: Number, salinity: Number = 0.0, pressure: Number = 14.7
) -> Number:
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
            f"Temperature {temperature:.6f}°F is outside the valid range for McCain's water viscosity correlation (60°F to 400°F).",
            stacklevel=2,
        )

    if salinity > 300_000:
        warnings.warn(
            f"Salinity {salinity:.6f}ppm is unusually high for McCain's water viscosity correlation.",
            stacklevel=2,
        )

    if pressure is not None and pressure > 10_000:
        warnings.warn(
            f"Pressure {pressure:.6f}psi is unusually high for McCain's water viscosity correlation.",
            stacklevel=2,
        )
    return _compute_water_viscosity(
        temperature=temperature,
        salinity=salinity,
        pressure=pressure,
        ppm_to_weight_fraction=c.PPM_TO_WEIGHT_FRACTION,
    )


def _gas_solubility_in_water_mccain_methane(
    pressure: Number, temperature: Number, salinity: Number = 0.0
) -> Number:
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
        raise ValidationError("Pressure, temperature, and salinity must be non-negative.")

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
    return max(0.0, gas_solubility)


def _gas_solubility_in_water_duan_sun_co2(
    pressure: Number,
    temperature: Number,
    salinity: Number = 0.0,
    nacl_molecular_weight: Number = 58.44,
    psi_to_bar: Number = 0.0689476,
) -> Number:
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
        raise ValidationError("Temperature is out of the valid range for this model (0-260°C).")
    if not (0 < P <= 2000):
        raise ValidationError("Pressure is out of the valid range for this model (0-2000 bar).")

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


def _gas_solubility_in_water_henry_law(
    pressure: Number,
    temperature: Number,
    gas: str,
    molar_masses: dict[str, Number],
    henry_coefficients: dict[str, tuple[Number, Number, Number]],
    salinity: Number = 0.0,
) -> Number:
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
        water_density = c.STANDARD_WATER_DENSITY_SI

    # Setschenow salinity correction
    # Converts salinity from ppm (mg/kg) to mol/kg using molar mass in g/mol
    molarity = salinity / (c.MOLECULAR_WEIGHT_NACL * 1000)

    k_s = SETSCHENOW_CONSTANTS[gas]
    salinity_factor = np.exp(-k_s * molarity)

    gas_solubility = (pressure / H) * (M / water_density) * salinity_factor  # m³ gas / m³ water
    return gas_solubility * c.CUBIC_METER_PER_CUBIC_METER_TO_SCF_PER_STB


def compute_gas_solubility_in_water(
    pressure: Number,
    temperature: Number,
    salinity: Number = 0.0,
    gas: str = "methane",
) -> Number:
    """
    Computes gas solubility in water using McCain, Duan, or Henry's Law based on gas type and temperature.

    :param pressure: Pressure (psi).
    :param temperature: Temperature (°F).
    :param salinity: Salinity in parts per million (ppm).
    :param gas: Type of gas ("methane", "CO2", or "N2"). Default is "methane".
    :return: Gas solubility in water in SCF/STB (standard cubic feet per stock tank barrel).
    """
    gas = get_gas_symbol(gas)
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
        henry_coefficients=HENRY_COEFFICIENTS,  # type: ignore[arg-type]
        salinity=salinity,
    )


def compute_gas_free_water_formation_volume_factor(
    pressure: Number, temperature: Number
) -> Number:
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

    thermal_expansion = -0.010001 + (1.33391e-4 * temperature) + (5.50654e-7 * temperature**2)
    isothermal_compressibility = -(1.95301e-9 * pressure) + (1.72492e-13 * pressure**2)
    gas_free_water_fvf = (1.0 + thermal_expansion) * (1.0 + isothermal_compressibility)
    return max(0.9, gas_free_water_fvf)  # Bw_gas_free is typically close to 1.0


def _compute_dRsw_dP_mccain(temperature: Number, salinity: Number) -> Number:
    """
    Calculates the derivative of gas solubility in water (Rsw) with respect to pressure,
    based on McCain's correlation for Rsw.
    Returns dRsw/dP in scf/(STB*psi).
    """
    if temperature < 0 or salinity < 0:
        raise ValidationError("Temperature and salinity cannot be negative for dRsw/dP.")

    derivative_pure_water = (
        0.0000164 + (0.000000134 * temperature) - (0.00000000185 * temperature**2)
    )
    salinity_correction_factor = 1.0 - 0.000001 * salinity
    # This derivative is positive (Rsw increases with P)
    return derivative_pure_water * salinity_correction_factor


def _compute_dBw_gas_free_dp_mccain(pressure: Number, temperature: Number) -> Number:
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
    pressure: Number,
    temperature: Number,
    bubble_point_pressure: Number,  # This Pwb is for the water's dissolved gas in water.
    gas_formation_volume_factor: Number,  # Bg in ft3/SCF
    gas_solubility_in_water: Number,  # Rsw in SCF/STB
    gas_free_water_formation_volume_factor: Number,  # Bw_gas_free in bbl/STB (output of compute_gas_free_water_formation_volume_factor)
    salinity: Number = 0.0,
) -> Number:
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
            raise ValidationError("Calculated Bw for undersaturated water is non-positive.")
        c_w = -(1.0 / gas_free_water_formation_volume_factor) * dBw_gas_free_dP
    else:
        # Saturated Water (P < Pwb)
        water_fvf_in_bbl_per_stb = gas_free_water_formation_volume_factor + (
            gas_solubility_in_water * gas_fvf_in_bbl_per_scf
        )
        if np.any(water_fvf_in_bbl_per_stb <= 0):
            raise ValidationError("Calculated Bw for saturated water is non-positive.")

        c_w_gas_free_component = -(1.0 / water_fvf_in_bbl_per_stb) * dBw_gas_free_dP
        gas_liberation_component = (gas_fvf_in_bbl_per_scf / water_fvf_in_bbl_per_stb) * dRsw_dP
        c_w = c_w_gas_free_component + gas_liberation_component

    return max(0.0, c_w)


def compute_water_density_mccain(
    pressure: Number, temperature: Number, salinity: Number = 0.0
) -> Number:
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
        c.STANDARD_WATER_DENSITY_IMPERIAL + delta_salinity + delta_pressure + delta_temperature
    )
    return water_density


def compute_water_density_batzle(
    pressure: Number, temperature: Number, salinity: Number
) -> Number:
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
        S * (0.668 + 0.44 * S + 1e-6 * (300 * T - 2400 * T * S + P * (80 + 3 * T - 3300 * S)))
    )
    # Convert to lb/ft³ (1 g/cm³ = 62.42796 lb/ft³)
    water_density = brine_density_g_per_cm3 * 62.42796
    return water_density  # type: ignore


def compute_water_density(
    pressure: Number,
    temperature: Number,
    gas_gravity: Number = 0.0,
    salinity: Number = 0.0,
    gas_solubility_in_water: Number = 0.0,
    gas_free_water_formation_volume_factor: Number = 1.0,
) -> Number:
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
    return max(0.0, live_water_density_in_lb_per_ft3)

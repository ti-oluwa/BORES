import logging
import typing

import numpy as np

from bores.constants import c
from bores.correlations.core import fahrenheit_to_kelvin, fahrenheit_to_rankine
from bores.correlations.scalars.core import compute_fluid_density
from bores.errors import ValidationError
from bores.types import Number, ZFactorMethod
from bores.utils import clip

logger = logging.getLogger(__name__)

__all__ = [
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
    "compute_gas_pseudocritical_properties",
    "compute_gas_viscosity",
]


def compute_gas_gravity(gas: str) -> Number:
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


def compute_gas_gravity_from_density(
    pressure: Number, temperature: Number, density: Number
) -> Number:
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
    air_density = compute_fluid_density(pressure_in_pascals, temperature_in_kelvin, fluid="Air")
    return density / (air_density * c.KILOGRAM_PER_CUBIC_METER_TO_POUNDS_PER_CUBIC_FEET)


def compute_gas_formation_volume_factor(
    pressure: Number,
    temperature: Number,
    gas_compressibility_factor: Number,
) -> Number:
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


def compute_gas_compressibility_factor_papay(
    pressure: Number,
    temperature: Number,
    gas_gravity: Number,
    h2s_mole_fraction: Number = 0.0,
    co2_mole_fraction: Number = 0.0,
    n2_mole_fraction: Number = 0.0,
) -> Number:
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
        raise ValidationError("Pressure, temperature, and gas specific gravity must be positive.")
    if h2s_mole_fraction < 0 or h2s_mole_fraction > 1:
        raise ValidationError("H2S mole fraction must be between 0 and 1.")
    if co2_mole_fraction < 0 or co2_mole_fraction > 1:
        raise ValidationError("CO2 mole fraction must be between 0 and 1.")
    if n2_mole_fraction < 0 or n2_mole_fraction > 1:
        raise ValidationError("N2 mole fraction must be between 0 and 1.")

    pseudo_critical_pressure, pseudo_critical_temperature = compute_gas_pseudocritical_properties(
        gas_gravity=gas_gravity,
        h2s_mole_fraction=h2s_mole_fraction,
        co2_mole_fraction=co2_mole_fraction,
        n2_mole_fraction=n2_mole_fraction,
    )
    pseudo_reduced_pressure = pressure / pseudo_critical_pressure
    pseudo_reduced_temperature = fahrenheit_to_rankine(temperature) / pseudo_critical_temperature

    # Papay's correlation for gas compressibility factor
    compressibility_factor = (
        1
        - (
            (3.52 * pseudo_reduced_pressure * np.exp(-0.869 * pseudo_reduced_temperature))
            / pseudo_reduced_temperature
        )
        + ((0.274 * pseudo_reduced_pressure**2) / pseudo_reduced_temperature**2)
    )
    # Ensure Z is not negative
    return max(0.1, compressibility_factor)  # type: ignore[return-value]


def compute_gas_compressibility_factor_hall_yarborough(
    pressure: Number,
    temperature: Number,
    gas_gravity: Number,
    h2s_mole_fraction: Number = 0.0,
    co2_mole_fraction: Number = 0.0,
    n2_mole_fraction: Number = 0.0,
    maximum_iterations: int = 50,
    tolerance: Number = 1e-10,
) -> Number:
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
    :param maximum_iterations: Maximum Newton-Raphson iterations
    :param tolerance: Convergence tolerance
    :return: Compressibility factor Z (dimensionless)

    References:
        Hall, K.R. and Yarborough, L. (1973). "A New Equation of ReservoirState for Z-factor Calculations."
        Oil & Gas Journal, June 18, 1973, pp. 82-92.
    """
    if pressure <= 0 or temperature <= 0 or gas_gravity <= 0:
        raise ValidationError("Pressure, temperature, and gas gravity must be positive.")

    # Get pseudocritical properties with Wichert-Aziz correction
    pseudo_critical_pressure, pseudo_critical_temperature = compute_gas_pseudocritical_properties(
        gas_gravity=gas_gravity,
        h2s_mole_fraction=h2s_mole_fraction,
        co2_mole_fraction=co2_mole_fraction,
        n2_mole_fraction=n2_mole_fraction,
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
    for _ in range(maximum_iterations):
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


def compute_gas_compressibility_factor_dranchuk_abou_kassem(
    pressure: Number,
    temperature: Number,
    gas_gravity: Number,
    h2s_mole_fraction: Number = 0.0,
    co2_mole_fraction: Number = 0.0,
    n2_mole_fraction: Number = 0.0,
    maximum_iterations: int = 50,
    tolerance: Number = 1e-10,
) -> Number:
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
    :param maximum_iterations: Maximum iterations for density convergence
    :param tolerance: Convergence tolerance
    :return: Compressibility factor Z (dimensionless)

    References:
        Dranchuk, P.M. and Abou-Kassem, J.H. (1975). "Calculation of Z Factors for
        Natural Gases Using Equations of ReservoirState." Journal of Canadian Petroleum Technology,
        July-September 1975, pp. 34-36.
    """
    if pressure <= 0 or temperature <= 0 or gas_gravity <= 0:
        raise ValidationError(
            f"Pressure, temperature, and gas gravity must be positive.\n"
            f"Pressure={float(pressure)}psi, Temperature={float(temperature)}°F, Gas gravity={float(gas_gravity)}"
        )

    # Get pseudocritical properties
    pseudo_critical_pressure, pseudo_critical_temperature = compute_gas_pseudocritical_properties(
        gas_gravity=gas_gravity,
        h2s_mole_fraction=h2s_mole_fraction,
        co2_mole_fraction=co2_mole_fraction,
        n2_mole_fraction=n2_mole_fraction,
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
    for _ in range(maximum_iterations):
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


def compute_gas_compressibility_factor(
    pressure: Number,
    temperature: Number,
    gas_gravity: Number,
    h2s_mole_fraction: Number = 0.0,
    co2_mole_fraction: Number = 0.0,
    n2_mole_fraction: Number = 0.0,
    method: ZFactorMethod = "dak",
) -> Number:
    """
    Computes (natural) gas compressibility factor.

    Method Selection Strategy:
    1. **High Pressure (Pr > 15)**: Use DAK (most accurate for Pr up to 30)
    2. **Medium Pressure (1 < Pr ≤ 15)**: Use Hall-Yarborough (best balance)
    3. **Low Pressure (Pr ≤ 1)**: Use Papay (fast, accurate for low Pr)
    4. **Fallback**: If any method fails validation, try others in order:
        DAK -> Hall-Yarborough -> Papay

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
    - Hall, K.R. and Yarborough, L. (1973). "A New Equation of ReservoirState..."
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


def compute_gas_molecular_weight(gas_gravity: Number) -> Number:
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


def compute_gas_pseudocritical_properties(
    gas_gravity: Number,
    h2s_mole_fraction: Number = 0.0,
    co2_mole_fraction: Number = 0.0,
    n2_mole_fraction: Number = 0.0,
) -> tuple[Number, Number]:
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
    pseudocritical_temperature_rankine = 169.2 + 349.5 * gas_gravity - 74.0 * gas_gravity**2

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
    pressure: Number,
    temperature: Number,
    gas_gravity: Number,
    gas_compressibility_factor: Number,
) -> Number:
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
        gas_compressibility_factor * c.IDEAL_GAS_CONSTANT_IMPERIAL * temperature_in_rankine
    )
    return gas_density


def compute_gas_viscosity(
    temperature: Number,
    gas_density: Number,
    gas_molecular_weight: Number,
) -> Number:
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
    density_in_grams_per_cm3 = gas_density * c.POUNDS_PER_CUBIC_FEET_TO_GRAMS_PER_CUBIC_METER

    k = (
        (9.4 + (0.02 * gas_molecular_weight_lbm_per_lbmole))
        * (temperature_in_rankine**1.5)
        / (209 + (19 * gas_molecular_weight_lbm_per_lbmole) + temperature_in_rankine)
    )

    x = 3.5 + (986 / temperature_in_rankine) + (0.01 * gas_molecular_weight_lbm_per_lbmole)
    y = 2.4 - (0.2 * x)

    exponent = x * (density_in_grams_per_cm3**y)
    exponent = np.minimum(700, max(-700, exponent))  # cap to prevent overflow

    gas_viscosity = (k * 1e-4) * np.exp(exponent)
    return max(0.0, gas_viscosity)


def compute_gas_compressibility(
    pressure: Number,
    temperature: Number,
    gas_gravity: Number,
    gas_compressibility_factor: Number | None = None,
    h2s_mole_fraction: Number = 0.0,
    co2_mole_fraction: Number = 0.0,
    n2_mole_fraction: Number = 0.0,
) -> Number:
    """
    alculates isothermal gas compressibility (C_g) in psi^-1 using a
    numerically consistent derivative of the Z-factor.

    This avoids non-physical negative compressibility caused by
    analytical derivatives of approximate correlations.

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
    if pressure <= 0.0 or temperature <= 0.0 or gas_gravity <= 0.0:
        raise ValidationError("Pressure, temperature, and gas specific gravity must be positive.")

    if gas_compressibility_factor is not None:
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

    # Numerical derivative dZ/dP (central difference)
    # Relative perturbation (stable across pressure scales)
    dP = 1e-4 * pressure
    if dP == 0:
        dP = 1e-6  # absolute fallback

    P_plus = pressure + dP
    P_minus = pressure - dP

    if P_minus <= 0.0:
        P_minus = pressure  # fallback to forward diff if near zero
        Z_plus = compute_gas_compressibility_factor(
            gas_gravity=gas_gravity,
            pressure=P_plus,
            temperature=temperature,
            h2s_mole_fraction=h2s_mole_fraction,
            co2_mole_fraction=co2_mole_fraction,
            n2_mole_fraction=n2_mole_fraction,
        )
        dZ_dP = (Z_plus - Z) / dP
    else:
        Z_plus = compute_gas_compressibility_factor(
            gas_gravity=gas_gravity,
            pressure=P_plus,
            temperature=temperature,
            h2s_mole_fraction=h2s_mole_fraction,
            co2_mole_fraction=co2_mole_fraction,
            n2_mole_fraction=n2_mole_fraction,
        )
        Z_minus = compute_gas_compressibility_factor(
            gas_gravity=gas_gravity,
            pressure=P_minus,
            temperature=temperature,
            h2s_mole_fraction=h2s_mole_fraction,
            co2_mole_fraction=co2_mole_fraction,
            n2_mole_fraction=n2_mole_fraction,
        )
        dZ_dP = (Z_plus - Z_minus) / (2.0 * dP)

    gas_compressibility = (1.0 / pressure) - (1.0 / Z) * dZ_dP
    if gas_compressibility < 0.0:
        gas_compressibility = 0.0

    return gas_compressibility

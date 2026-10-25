"""Physical constants and conversion factors"""

import logging
import typing
import warnings
from contextvars import ContextVar, Token
from uuid import uuid4

import attrs
from typing_extensions import Self

from bores.errors import ValidationError
from bores.precision import get_floating_point_info

__all__ = [
    "CONSTANTS",
    "Constant",
    "Constants",
    "ConstantsContext",
    "c",
    "get_constant",
    "get_constants",
    "set_constants",
]

logger = logging.getLogger(__name__)


@typing.final
@attrs.frozen(slots=True)
class Constant:
    """
    A constant value with optional description and metadata.

    This class wraps a constant value and provides additional context about
    what the constant represents, its units, and any other relevant information.
    """

    value: typing.Any
    """The actual value of the constant."""

    description: str | None = None
    """Optional description of what this constant represents."""

    unit: str | None = None
    """Optional unit of measurement for this constant."""

    aliases: tuple[str, ...] = attrs.field(factory=tuple, converter=tuple)
    """
    Alternate names this constant is also known by (e.g. a unit-suffixed
    name and a system-suffixed name for the same numeric value). This is
    the single source of truth `Constants` reads to build its alias index -
    it does not, by itself, create any dict entries.
    """

    def __str__(self) -> str:
        """Return a human-readable string representation of the `Constant`."""
        return f"{self.value}{self.unit or ''}"

    def __repr__(self) -> str:
        parts = [f"value={self.value}"]
        if self.description:
            parts.append(f"description='{self.description}'")
        if self.unit:
            parts.append(f"unit='{self.unit}'")
        if self.aliases:
            parts.append(f"aliases={self.aliases!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


@typing.final
@attrs.frozen(slots=True)
class ConstantFactory:
    """
    A lazily-evaluated constant whose value is produced by a factory callable
    at access time.

    The factory takes no arguments and reads whatever context it needs
    (e.g. the active floating-point dtype via `get_floating_point_info()`).
    This makes the constant self-aware of the active numerical precision
    without pushing dtype-scaling logic into every call site.

    Caching is deliberately left to the factory itself (e.g. via
    `functools.lru_cache` keyed on dtype) rather than baked into this class,
    because the correct cache invalidation strategy depends on how frequently
    the dtype context changes in the application.

    Serialization evaluates the factory and stores the result, so a
    deserialized `ConstantFactory` becomes a plain `Constant`. This is
    intentional: serialized data represents a snapshot of the value at the
    time of serialization.

    Example:

    ```
    from bores.constants import ConstantFactory
    from bores.precision import get_floating_point_info

    SATURATION_EPSILON = ConstantFactory(
        factory=lambda: get_floating_point_info().eps ** 0.5,
        description="Saturation singularity clamp, dtype-aware.",
        unit="fraction",
    )
    ```
    """

    factory: typing.Callable[[], typing.Any]
    """Zero-argument callable that returns the constant's current value."""

    description: str | None = None
    """Optional description of what this constant represents."""

    unit: str | None = None
    """Optional unit of measurement for this constant."""

    aliases: tuple[str, ...] = attrs.field(factory=tuple, converter=tuple)
    """Alternate names - see `Constant.aliases`."""

    @property
    def value(self) -> typing.Any:
        """Evaluate and return the current value."""
        return self.factory()

    def __str__(self) -> str:
        try:
            v = self.value
        except Exception:
            logger.exception(f"Error evaluating constant: {self.factory!r}")
            v = "<unevaluated>"
        return f"{v}{self.unit or ''}"

    def __repr__(self) -> str:
        parts = [f"factory={self.factory!r}"]
        if self.description:
            parts.append(f"description='{self.description}'")
        if self.unit:
            parts.append(f"unit='{self.unit}'")
        if self.aliases:
            parts.append(f"aliases={self.aliases!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


def _get_saturation_epsilon() -> float:
    """
    Saturation clamp epsilon, scaled to the active floating-point dtype.

    Target: 8 * machine_epsilon, floored at 1e-7 (about the float32 machine epsilon)
    so the clamp is always meaningful even if the context dtype is float64.
    """
    info = get_floating_point_info()
    dtype_based = 8.0 * float(info.eps)
    floor = 1e-7  # meaningful saturation floor, representable in float32
    return max(dtype_based, floor)


def _get_minimum_pore_space() -> float:
    """
    Minimum mobile pore-space guard, matched to the saturation epsilon so
    validity guards and saturation clamps stay numerically consistent.
    """
    return _get_saturation_epsilon()


def _get_finite_difference_epsilon() -> float:
    """
    Central finite-difference step, scaled to the active dtype.
    Optimal step is cbrt(machine_epsilon); floored at 1e-5 for float64.
    """
    info = get_floating_point_info()
    optimal = info.eps ** (1.0 / 3.0)
    floor = 1e-5  # float64 conservative floor
    return float(max(optimal, floor))


# Default constants dictionary
DEFAULT_CONSTANTS: dict[str, typing.Any | Constant | ConstantFactory] = {
    # Standard Conditions
    # Pressure
    "STANDARD_PRESSURE_PASCAL": Constant(
        value=101325,
        description="Standard atmospheric pressure in Pascals (SI units)",
        unit="Pa",
        aliases=("STANDARD_PRESSURE_SI",),
    ),
    "STANDARD_PRESSURE_PSI": Constant(
        value=101325.0 / (0.45359237 * 9.80665 / 0.0254**2),
        description="Standard atmospheric pressure in Pound-per-Square-Inch (psi)",
        unit="psi",
        aliases=('STANDARD_PRESSURE_IMPERIAL',),
    ),
    "STANDARD_PRESSURE_BAR": Constant(
        value=1.01325,
        description="Standard atmospheric pressure in Bars (abs)",
        unit="barsa [bar]",
    ),
    "STANDARD_PRESSURE_ATM": Constant(
        value=1.0000,
        description="Standard atmospheric pressure in Atmospheres",
        unit="atm",
    ),
    # Hydrostatic gradient factors
    "HYDROSTATIC_GRADIENT_FACTOR_FIELD": Constant(
        value=1.0 / 144.0,
        description="Hydrostatic gradient factor for FIELD units: psi per (lbm/ft³ * ft)",
        unit="psi / (lbm/ft³ * ft)",
    ),
    "HYDROSTATIC_GRADIENT_FACTOR_METRIC": Constant(
        value=9.80665e-5,
        description="Hydrostatic gradient factor for METRIC units: bar per (kg/m³ * m)",
        unit="bar / (kg/m³ * m)",
    ),
    "HYDROSTATIC_GRADIENT_FACTOR_LAB": Constant(
        value=9.80665 * 100.0 / (101325.0 * 10.0),
        description="Hydrostatic gradient factor for LAB units: atm per (g/cm³ * cm)",
        unit="atm / (g/cm³ * cm)",
    ),
    "HYDROSTATIC_GRADIENT_FACTOR_SI": Constant(
        value=9.80665,
        description="Hydrostatic gradient factor for SI units: Pa per (kg/m³ * m)",
        unit="Pa / (kg/m³ * m)",
    ),
    # Temperature
    "STANDARD_TEMPERATURE_KELVIN": Constant(
        value=(60.0 - 32.0) * 5.0 / 9.0 + 273.15,
        description="Standard temperature (60°F) in Kelvin",
        unit="K",
        aliases=('STANDARD_TEMPERATURE_SI',),
    ),
    "STANDARD_TEMPERATURE_FAHRENHEIT": Constant(
        value=60.0,
        description="Standard temperature in Fahrenheit",
        unit="°F",
        aliases=("STANDARD_TEMPERATURE_IMPERIAL",),
    ),
    "STANDARD_TEMPERATURE_RANKINE": Constant(
        value=519.67, description="Standard temperature (60°F) in Rankine", unit="°R"
    ),
    "STANDARD_TEMPERATURE_CELSIUS": Constant(
        value=(60.0 - 32.0) * 5.0 / 9.0,
        description="Standard temperature (60°F) in Celsius",
        unit="°C",
    ),
    # Thermal and Compressibility Properties
    # Oil
    "OIL_THERMAL_EXPANSION_COEFFICIENT": Constant(
        value=9.7e-4, description="Thermal expansion coefficient for oil", unit="1/K"
    ),
    "OIL_THERMAL_EXPANSION_COEFFICIENT_IMPERIAL": Constant(
        value=5.39e-4,
        description="Thermal expansion coefficient for oil (Imperial units)",
        unit="1/°F",
    ),
    # Water
    "WATER_THERMAL_EXPANSION_COEFFICIENT": Constant(
        value=1.57e-4,
        description="Thermal expansion coefficient for water at Standard temperature (60°F)",
        unit="1/K",
    ),
    "WATER_THERMAL_EXPANSION_COEFFICIENT_IMPERIAL": Constant(
        value=1.57e-4 / 1.8,
        description="Thermal expansion coefficient for water at Standard temperature (60°F) (Imperial units)",
        unit="1/°F",
    ),
    "WATER_ISOTHERMAL_COMPRESSIBILITY": Constant(
        value=4.6e-10,
        description="Isothermal compressibility of water at Standard temperature (15.6°C) (SI units)",
        unit="1/Pa",
    ),
    "WATER_ISOTHERMAL_COMPRESSIBILITY_IMPERIAL": Constant(
        value=3.17e-6,
        description="Isothermal compressibility of water at Standard temperature (15.6°C) (Imperial units)",
        unit="1/psi",
    ),
    # Standard Densities
    # Water
    "STANDARD_WATER_DENSITY_IMPERIAL": Constant(
        value=62.37,
        description="Standard water density at Standard temperature (15.6°C) (Imperial/field units)",
        unit="lbm/ft³",
    ),
    "STANDARD_WATER_DENSITY_METRIC": Constant(
        value=999.0,
        description="Standard water density at Standard temperature (60°F, 15.6°C) (Metric units)",
        unit="kg/m³",
        aliases=('STANDARD_WATER_DENSITY_SI',),
    ),
    "STANDARD_WATER_DENSITY_LAB": Constant(
        value=0.9990,
        description="Standard water density at Standard temperature (60°F, 15.6°C) (Lab units)",
        unit="g/cm³",
    ),
    # Air
    "STANDARD_AIR_DENSITY_FIELD": Constant(
        value=0.0765,
        description="Standard air density at Standard temperature (15.6°C) (Imperial/field units)",
        unit="lbm/ft³",
        aliases=("STANDARD_AIR_DENSITY_IMPERIAL",),
    ),
    "STANDARD_AIR_DENSITY_METRIC": Constant(
        value=1.225,
        description="Standard air density at Standard temperature (15.6°C) (Metric units)",
        unit="kg/m³",
        aliases=("STANDARD_AIR_DENSITY_SI",),
    ),
    "STANDARD_AIR_DENSITY_LAB": Constant(
        value=1.225e-3,
        description="Standard air density at Standard temperature (15.6°C) (Lab units)",
        unit="g/cm³",
    ),
    # Molecular Weights
    "MOLECULAR_WEIGHT_WATER": Constant(
        value=18.01528, description="Molecular weight of water", unit="g/mol"
    ),
    "MOLECULAR_WEIGHT_CO2": Constant(
        value=44.01, description="Molecular weight of carbon dioxide", unit="g/mol"
    ),
    "MOLECULAR_WEIGHT_N2": Constant(
        value=28.0134, description="Molecular weight of nitrogen", unit="g/mol"
    ),
    "MOLECULAR_WEIGHT_CH4": Constant(
        value=16.04246, description="Molecular weight of methane", unit="g/mol"
    ),
    "MOLECULAR_WEIGHT_NACL": Constant(
        value=58.44,
        description="Molecular weight of sodium chloride (NaCl)",
        unit="g/mol",
    ),
    "MOLECULAR_WEIGHT_O2": Constant(
        value=31.9988, description="Molecular weight of oxygen", unit="g/mol"
    ),
    "MOLECULAR_WEIGHT_ARGON": Constant(
        value=39.948, description="Molecular weight of argon", unit="g/mol"
    ),
    "MOLECULAR_WEIGHT_AIR": Constant(
        value=28.9644, description="Molecular weight of air", unit="g/mol"
    ),
    "MOLECULAR_WEIGHT_HELIUM": Constant(
        value=4.002602, description="Molecular weight of helium", unit="g/mol"
    ),
    "MOLECULAR_WEIGHT_H2": Constant(
        value=2.01588, description="Molecular weight of hydrogen", unit="g/mol"
    ),
    # Pressure Conversions
    "PSI_TO_PASCAL": Constant(
        value=(0.45359237 * 9.80665 / 0.0254**2),
        description="Conversion factor from psi to Pascals (lbf/in2, from the exact lbm, g0 and inch definitions)",
        unit="Pa/psi",
    ),
    "PASCAL_TO_PSI": Constant(
        value=1.0 / (0.45359237 * 9.80665 / 0.0254**2),
        description="Conversion factor from Pascals to psi",
        unit="psi/Pa",
    ),
    "PSI_TO_BAR": Constant(
        value=(0.45359237 * 9.80665 / 0.0254**2) / 1e5,
        description="Conversion factor from psi to bar",
        unit="bar/psi",
    ),
    "ATM_TO_PASCAL": Constant(
        value=101325.0,
        description="Conversion factor from atm to Pascals",
        unit="Pa/atm",
    ),
    # Temperature Conversions
    "RANKINE_TO_KELVIN": Constant(
        value=5 / 9,
        description="Conversion factor from Rankine to Kelvin: T(K) = T(R) * 5/9",
        unit="K/R",
    ),
    "KELVIN_TO_RANKINE": Constant(
        value=9 / 5,
        description="Conversion factor from Kelvin to Rankine: T(R) = T(K) * 9/5",
        unit="R/K",
    ),
    # Viscosity Conversions
    "CENTIPOISE_TO_PASCAL_SECONDS": Constant(
        value=0.001,
        description="Conversion factor from centipoise to Pascal-seconds",
        unit="Pa·s/cP",
    ),
    "PASCAL_SECONDS_TO_CENTIPOISE": Constant(
        value=1000,
        description="Conversion factor from Pascal-seconds to centipoise",
        unit="cP/(Pa·s)",
    ),
    # Permeability Conversions
    "MILLIDARCY_TO_SQUARE_METER": Constant(
        value=(1e-10 / 101325.0),
        description="Conversion factor from millidarcies to square meters (1 D = 1 cP * 1 cm/s / (1 atm/cm))",
        unit="m²/mD",
    ),
    # Gas-Oil Ratio Conversions
    "SCF_PER_STB_TO_CUBIC_METER_PER_CUBIC_METER": Constant(
        value=0.3048**3 / (42.0 * 0.003785411784),
        description="Conversion factor from scf/STB to m³/m³",
        unit="(m³/m³)/(scf/STB)",
    ),
    "CUBIC_METER_PER_CUBIC_METER_TO_SCF_PER_STB": Constant(
        value=(42.0 * 0.003785411784) / 0.3048**3,
        description="Conversion factor from m³/m³ to scf/STB",
        unit="(scf/STB)/(m³/m³)",
    ),
    # Formation Volume Factor Conversions
    "CUBIC_METER_PER_CUBIC_METER_TO_BARRELS_PER_SCF": Constant(
        value=0.3048**3 / (42.0 * 0.003785411784),
        description="Conversion factor from m³/m³ to BBL/scf",
        unit="(BBL/scf)/(m³/m³)",
    ),
    "BARRELS_PER_SCF_TO_CUBIC_METER_PER_CUBIC_METER": Constant(
        value=(42.0 * 0.003785411784) / 0.3048**3,
        description="Conversion factor from BBL/scf to m³/m³",
        unit="(m³/m³)/(BBL/scf)",
    ),
    "CUBIC_METER_PER_CUBIC_METER_TO_BARRELS_PER_STB": Constant(
        value=1.0,
        description="Conversion factor from m³/m³ to BBL/STB",
        unit="(BBL/STB)/(m³/m³)",
    ),
    "BARRELS_PER_STB_TO_CUBIC_METER_PER_CUBIC_METER": Constant(
        value=1.0,
        description="Conversion factor from BBL/STB to m³/m³",
        unit="(m³/m³)/(BBL/STB)",
    ),
    # Volume Conversions
    "MSCF_TO_SCF": Constant(
        value=1000.0,
        description="Conversion factor from thousand standard cubic feet (Mscf) to standard cubic feet (scf)",
        unit="scf/Mscf",
    ),
    "SCF_TO_MSCF": Constant(
        value=1.0 / 1000.0,
        description="Conversion factor from standard cubic feet (scf) to thousand standard cubic feet (Mscf)",
        unit="Mscf/scf",
    ),
    "CUBIC_METER_TO_SCF": Constant(
        value=1.0 / 0.3048**3,
        description="Conversion factor from cubic meters to standard cubic feet",
        unit="scf/m³",
    ),
    "SCF_TO_BARRELS": Constant(
        value=0.3048**3 / (42.0 * 0.003785411784),
        description="Conversion factor from standard cubic feet to barrels",
        unit="BBL/scf",
    ),
    "BARRELS_TO_CUBIC_FEET": Constant(
        value=(42.0 * 0.003785411784) / 0.3048**3,
        description="Conversion factor from barrels to cubic feet",
        unit="ft³/BBL",
    ),
    "CUBIC_FEET_TO_BARRELS": Constant(
        value=0.3048**3 / (42.0 * 0.003785411784),
        description="Conversion factor from cubic feet to barrels",
        unit="BBL/ft³",
    ),
    "STB_TO_CUBIC_FEET": Constant(
        value=(42.0 * 0.003785411784) / 0.3048**3,
        description="Conversion factor from stock tank barrels to (standard) cubic feet",
        unit="ft³/STB",
    ),
    "CUBIC_FEET_TO_STB": Constant(
        value=0.3048**3 / (42.0 * 0.003785411784),
        description="Conversion factor from (standard) cubic feet to stock tank barrels",
        unit="STB/ft³",
    ),
    "STB_TO_CUBIC_METER": Constant(
        value=(42.0 * 0.003785411784),
        description="Conversion factor from stock tank barrels to (standard) cubic meters (42 US gal)",
        unit="m³/STB",
    ),
    "CUBIC_METER_TO_STB": Constant(
        value=1.0 / (42.0 * 0.003785411784),
        description="Conversion factor from (standard) cubic meters to stock tank barrels",
        unit="STB/m³",
    ),
    "BARRELS_TO_CUBIC_METER": Constant(
        value=(42.0 * 0.003785411784),
        description="Conversion factor from barrels to cubic meters (42 US gal)",
        unit="m³/BBL",
    ),
    "CUBIC_METER_TO_BARRELS": Constant(
        value=1.0 / (42.0 * 0.003785411784),
        description="Conversion factor from cubic meters to barrels",
        unit="BBL/m³",
    ),
    "SCF_TO_SCM": Constant(
        value=0.3048**3,
        description="Conversion factor from standard cubic feet to standard cubic meters",
        unit="m³/scf",
    ),
    "DYNE_PER_CENTIMETER_TO_PSI": Constant(
        value=1e-3 / (1e-10 / 101325.0) ** 0.5 / (0.45359237 * 9.80665 / 0.0254**2),
        description="Leverett J-function scaling factor: capillary pressure in psi from an interfacial tension of 1 dyne/cm, with permeability in mD and porosity as a fraction (sigma * sqrt(phi / k)). Its reciprocal is the 0.2166 in the field-unit J-function definition",
        unit="psi / ((dyne/cm) * sqrt(1/mD))",
    ),
    "DYNE_PER_CENTIMETER_TO_NEWTON_PER_METER": Constant(
        value=0.001,
        description="Conversion factor from dyne per centimeter to Newtons per metre",
        unit="(N/m)/(dyne/cm)",
    ),
    # Gas Constant
    "IDEAL_GAS_CONSTANT_LAB": Constant(
        value=8.31446261815324,
        description="Universal gas constant (Lab units)",
        unit="J/(mol·K)",
    ),
    "IDEAL_GAS_CONSTANT_SI": Constant(
        value=8.31446261815324e-3,
        description="Universal gas constant (SI units)",
        unit="kJ/(mol·K)",
    ),
    "IDEAL_GAS_CONSTANT_FIELD": Constant(
        value=8.31446261815324 * 1000.0 * 0.45359237 / 1.8 / (0.45359237 * 9.80665 / 0.0254**2) / 0.3048**3,
        description="Universal gas constant (Imperial units)",
        unit="ft³·psi/(lb·mol·°R)",
        aliases=('IDEAL_GAS_CONSTANT_IMPERIAL',),
    ),
    # Density Conversions
    "POUNDS_PER_CUBIC_FEET_TO_KILOGRAM_PER_CUBIC_METER": Constant(
        value=0.45359237 / 0.3048**3,
        description="Conversion factor from lb/ft³ to kg/m³",
        unit="(kg/m³)/(lb/ft³)",
    ),
    "KILOGRAM_PER_CUBIC_METER_TO_POUNDS_PER_CUBIC_FEET": Constant(
        value=0.3048**3 / 0.45359237,
        description="Conversion factor from kg/m³ to lb/ft³",
        unit="(lb/ft³)/(kg/m³)",
    ),
    "POUNDS_PER_CUBIC_FEET_TO_GRAMS_PER_CUBIC_METER": Constant(
        value=0.45359237 / 0.3048**3 / 1000.0,
        description="Conversion factor from lb/ft³ to g/cm³",
        unit="(g/cm³)/(lb/ft³)",
        aliases=('POUNDS_PER_CUBIC_FEET_TO_GRAMS_PER_CUBIC_CENTIMETER',),
    ),
    # Concentration Conversions
    "PPM_TO_GRAMS_PER_LITER": Constant(
        value=1e-3, description="Conversion factor from ppm to g/L", unit="(g/L)/ppm"
    ),
    "GRAMS_PER_LITER_TO_PPM": Constant(
        value=1e3, description="Conversion factor from g/L to ppm", unit="ppm/(g/L)"
    ),
    "PPM_TO_WEIGHT_FRACTION": Constant(
        value=1e-6,
        description="Conversion factor from ppm to weight fraction",
        unit="fraction/ppm",
    ),
    "PPM_TO_WEIGHT_PERCENT": Constant(
        value=1e-4,
        description="Conversion factor from ppm to weight percent",
        unit="%/ppm",
    ),
    "WEIGHT_PERCENT_TO_PPM": Constant(
        value=1e4,
        description="Conversion factor from weight percent to ppm",
        unit="ppm/%",
    ),
    # Molar Volume
    "SCF_PER_POUND_MOLE": Constant(
        value=379.49,
        description="Standard cubic feet per pound-mole",
        unit="scf/(lb·mol)",
    ),
    # Length Conversions
    "INCHES_TO_METERS": Constant(
        value=0.0254, description="Conversion factor from inches to meters", unit="m/in"
    ),
    "METERS_TO_INCHES": Constant(
        value=1 / 0.0254,
        description="Conversion factor from meters to inches",
        unit="in/m",
    ),
    "FEET_TO_METERS": Constant(
        value=0.3048, description="Conversion factor from feet to meters", unit="m/ft"
    ),
    "METERS_TO_FEET": Constant(
        value=1 / 0.3048,
        description="Conversion factor from meters to feet",
        unit="ft/m",
    ),
    # Area Conversions
    "ACRES_TO_SQUARE_FEET": Constant(
        value=43560,
        description="Conversion factor from acres to square feet",
        unit="ft²/acre",
    ),
    "SQUARE_FEET_TO_ACRES": Constant(
        value=1 / 43560,
        description="Conversion factor from square feet to acres",
        unit="acre/ft²",
    ),
    # Volume-Area Conversions
    "ACRE_FOOT_TO_CUBIC_FEET": Constant(
        value=43560,
        description="Conversion factor from acre-feet to cubic feet",
        unit="ft³/(acre·ft)",
    ),
    "CUBIC_FEET_TO_ACRE_FOOT": Constant(
        value=1 / 43560,
        description="Conversion factor from cubic feet to acre-feet",
        unit="(acre·ft)/ft³",
    ),
    "ACRE_FOOT_TO_BARRELS": Constant(
        value=7758,
        description="Conversion factor from acre-feet to barrels",
        unit="BBL/(acre·ft)",
    ),
    "BARRELS_TO_ACRE_FOOT": Constant(
        value=1 / 7758,
        description="Conversion factor from barrels to acre-feet",
        unit="(acre·ft)/BBL",
    ),
    # Flow Rate Conversions
    "CUBIC_METER_PER_SECOND_TO_STB_PER_DAY": Constant(
        value=86400.0 / (42.0 * 0.003785411784),
        description="Conversion factor from m³/s to STB/day",
        unit="(STB/day)/(m³/s)",
    ),
    "STB_PER_DAY_TO_CUBIC_METER_PER_SECOND": Constant(
        value=(42.0 * 0.003785411784) / 86400.0,
        description="Conversion factor from STB/day to m³/s",
        unit="(m³/s)/(STB/day)",
    ),
    "CUBIC_METER_PER_SECOND_TO_SCF_PER_DAY": Constant(
        value=86400.0 / 0.3048**3,
        description="Conversion factor from m³/s to scf/day",
        unit="(scf/day)/(m³/s)",
    ),
    "SCF_PER_DAY_TO_CUBIC_METER_PER_SECOND": Constant(
        value=0.3048**3 / 86400.0,
        description="Conversion factor from scf/day to m³/s",
        unit="(m³/s)/(scf/day)",
    ),
    # Time Conversions
    "DAYS_PER_WEEK": Constant(value=7, description="Number of days in a week", unit="day/week"),
    "SECONDS_PER_DAY": Constant(
        value=86400.0, description="Number of seconds in a day", unit="s/day"
    ),
    "SECONDS_PER_HOUR": Constant(
        value=3600.0, description="Number of seconds in an hour", unit="s/hr"
    ),
    "SECONDS_PER_MINUTE": Constant(
        value=60.0, description="Number of seconds in a minute", unit="s/min"
    ),
    "DAYS_PER_SECOND": Constant(
        value=1 / 86400.0, description="Number of days in a second", unit="day/s"
    ),
    "HOURS_PER_DAY": Constant(value=24.0, description="Number of hours in a day", unit="hrs/day"),
    "DAYS_PER_HOUR": Constant(
        value=1 / 24.0, description="Number of days in a hour", unit="day/hr"
    ),
    "DAYS_PER_YEAR": Constant(
        value=365.25, description="Number of days in a year", unit="day/year"
    ),
    "MONTHS_PER_YEAR": Constant(
        value=12, description="Number of months in a year", unit="month/year"
    ),
    "SECONDS_PER_YEAR": Constant(
        value=365.25 * 86400.0, description="Number of seconds in a year", unit="s/year"
    ),
    # Transmissibility Conversions
    "MILLIDARCIES_PER_CENTIPOISE_TO_SQUARE_FEET_PER_PSI_PER_DAY": Constant(
        value=(1e-10 / 101325.0) / 1e-3 * (0.45359237 * 9.80665 / 0.0254**2) * 86400.0 / 0.3048**2,
        description="Conversion factor from mD/cP to ft²/(psi·day)",
        unit="(ft²/(psi·day))/(mD/cP)",
    ),
    "MILLIDARCIES_PER_CENTIPOISE_TO_SQUARE_FEET_PER_PSI_PER_SECOND": Constant(
        value=((1e-10 / 101325.0) / 1e-3 * (0.45359237 * 9.80665 / 0.0254**2) * 86400.0 / 0.3048**2) / 86400.0,
        description="Conversion factor from mD/cP to ft²/(psi·s)",
        unit="(ft²/(psi·s))/(mD/cP)",
    ),
    "MILLIDARCIES_FT_PER_CENTIPOISE_TO_CUBIC_FEET_PER_PSI_PER_DAY": Constant(
        value=(1e-10 / 101325.0) / 1e-3 * (0.45359237 * 9.80665 / 0.0254**2) * 86400.0 / 0.3048**2,
        description="Conversion factor from mD·ft/cP to ft³/(psi·day). The well-known 1.127e-3 is this value divided by 5.614583, i.e. the same factor for bbl/day",
        unit="(ft³/(psi·day))/(mD·ft/cP)",
    ),
    # Gravity
    "ACCELERATION_DUE_TO_GRAVITY_METER_PER_SECONDS_SQUARE": Constant(
        value=9.80665,
        description="Standard acceleration due to gravity (on preferred planet - default is Earth) in m/s²",
        unit="m/s²",
    ),
    "ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE": Constant(
        value=32.174,
        description="Standard acceleration due to gravity (on preferred planet - default is Earth) in ft/s²",
        unit="ft/s²",
    ),
    "ACCELERATION_DUE_TO_GRAVITY_FEET_PER_DAY_SQUARE": Constant(
        value=32.174 * 86400.0**2,
        description="Standard acceleration due to gravity (on preferred planet - default is Earth) in ft/day²",
        unit="ft/day²",
    ),
    "GRAVITATIONAL_FACTOR_FIELD": Constant(
        value=32.174,
        description="Gravitational conversion factor in lbm·ft/(lbf·s²). Conversion factor from pound-force to pound-mass or vice versa under Earth's gravity.",
        unit="lbm·ft/(lbf·s²)",
        aliases=("GRAVITATIONAL_FACTOR_LBM_FT_PER_LBF_S2",),
    ),
    "GRAVITATIONAL_FACTOR_METRIC": Constant(
        value=1.0,
        description=(
            "Mass/force coherence factor for METRIC units. kg is already a "
            "coherent SI mass unit (1 N = 1 kg*m/s2 with no separate "
            "conversion factor), so this is 1.0 - included for a uniform "
            "per-system lookup, not because METRIC needs a real correction "
            "the way FIELD's lbm/lbf split does."
        ),
        unit="dimensionless",
    ),
    "GRAVITATIONAL_FACTOR_LAB": Constant(
        value=1.0,
        description=(
            "Mass/force coherence factor for LAB units. Grams are a "
            "coherent CGS mass unit (1 dyne = 1 g*cm/s2), so this is 1.0 - "
            "same reasoning as GRAVITATIONAL_FACTOR_METRIC."
        ),
        unit="dimensionless",
    ),
    "GRAVITATIONAL_FACTOR_SI": Constant(
        value=1.0,
        description="Mass/force coherence factor for SI units - 1.0, SI is coherent by construction.",
        unit="dimensionless",
    ),
    "HYDROSTATIC_AREA_FACTOR_FIELD": Constant(
        value=144.0,
        description=(
            "density * gravitational_acceleration * length, divided by "
            "GRAVITATIONAL_FACTOR_FIELD, lands in lbf/ft2 for FIELD units - "
            "this converts that to lbf/in2 (psi)."
        ),
        unit="in²/ft²",
    ),
    "HYDROSTATIC_AREA_FACTOR_METRIC": Constant(
        value=100_000.0,
        description=(
            "density * gravitational_acceleration * length lands in Pa for "
            "METRIC units (kg/m3 * m/s2 * m is already coherent) - this "
            "converts Pa to bar."
        ),
        unit="Pa/bar",
    ),
    "HYDROSTATIC_AREA_FACTOR_LAB": Constant(
        value=1_013_250.0,
        description=(
            "density * gravitational_acceleration * length lands in barye "
            "(dyne/cm2) for LAB units (g/cm3 * cm/s2 * cm is already "
            "coherent CGS) - this converts barye to atm "
            "(1 atm = 101325 Pa = 1013250 barye)."
        ),
        unit="barye/atm",
    ),
    "HYDROSTATIC_AREA_FACTOR_SI": Constant(
        value=1.0,
        description="density * gravitational_acceleration * length already lands in Pa for SI units - no conversion needed.",
        unit="dimensionless",
    ),
    "RATE_TIME_FACTOR_FIELD": Constant(
        value=1.0 / 86400.0,
        description=(
            "Multiply a rate expressed per FIELD time unit (day) by this "
            "to get a per-second rate (1 / 86400 seconds-per-day), so "
            "velocity/Reynolds-number calculations match the per-second "
            "gravitational_acceleration every wellbore correlation's "
            "hydrostatic term already uses."
        ),
        unit="day/s",
    ),
    "RATE_TIME_FACTOR_METRIC": Constant(
        value=1.0 / 86400.0,
        description="Same reasoning and value as RATE_TIME_FACTOR_FIELD - METRIC also reports rates per day.",
        unit="day/s",
    ),
    "RATE_TIME_FACTOR_LAB": Constant(
        value=1.0 / 3600.0,
        description=(
            "Multiply a rate expressed per LAB time unit (hour) by this "
            "to get a per-second rate (1 / 3600 seconds-per-hour) - same "
            "reasoning as RATE_TIME_FACTOR_FIELD."
        ),
        unit="hr/s",
    ),
    "RATE_TIME_FACTOR_SI": Constant(
        value=1.0,
        description="SI already reports rates per second - already 1:1, included for a uniform per-system lookup.",
        unit="dimensionless",
    ),
    "VISCOSITY_SCALE_FIELD": Constant(
        value=6.71968975e-4,
        description=(
            "Converts viscosity from FIELD's reporting unit (cP) to the "
            "coherent unit FIELD's Reynolds-number calculation needs "
            "(lbm/(ft*s), matching density in lbm/ft3 and velocity in "
            "ft/s): 1 cP = 0.001 Pa*s = 0.001 kg/(m*s), converted to "
            "lbm/(ft*s) via mass and length factors."
        ),
        unit="(lbm/(ft*s))/cP",
    ),
    "VISCOSITY_SCALE_METRIC": Constant(
        value=0.001,
        description=(
            "Converts viscosity from METRIC's reporting unit (cP) to the "
            "coherent unit METRIC's Reynolds-number calculation needs "
            "(Pa*s = kg/(m*s), matching density in kg/m3 and velocity in "
            "m/s): 1 cP = 0.001 Pa*s."
        ),
        unit="Pa*s/cP",
    ),
    "VISCOSITY_SCALE_LAB": Constant(
        value=0.01,
        description=(
            "Converts viscosity from LAB's reporting unit (cP) to the "
            "coherent CGS unit LAB's Reynolds-number calculation needs "
            "(Poise = g/(cm*s), matching density in g/cm3 and velocity "
            "in cm/s): 1 cP = 0.01 Poise."
        ),
        unit="P/cP",
    ),
    "VISCOSITY_SCALE_SI": Constant(
        value=1.0,
        description="SI already reports viscosity in the coherent unit (Pa*s) its Reynolds-number calculation needs - no conversion.",
        unit="dimensionless",
    ),
    "RADIAL_FLOW_CONSTANT_FIELD": Constant(
        value=0.03976180178418527,
        description=(
            "Multiplies a FIELD well index (mD*ft) times mobility (1/cP) times drawdown (psi) to get "
            "reservoir-condition rate (ft3/day). compute_peaceman_well_index's own W = (k*h*N/G) / "
            "(ln(re/rw)+C+s) has no 2*pi in it, unlike the textbook radial Darcy equation, so this "
            "constant carries it: 2*pi times MILLIDARCIES_FT_PER_CENTIPOISE_TO_CUBIC_FEET_PER_PSI_PER_DAY "
            "(6.3283e-3, the ft3/day Darcy constant; the familiar 1.127e-3 is the same factor for bbl/day). "
            "It also equals the SI-pivot derivation: 2*pi * (permeability, length, pressure factors to SI / "
            "viscosity factor to SI) / reservoir_rate factor to SI."
        ),
        unit="(ft3/(psi*day))/(mD*ft/cP)",
    ),
    "RADIAL_FLOW_CONSTANT_METRIC": Constant(
        value=0.05357682808194586,
        description="Same reasoning as RADIAL_FLOW_CONSTANT_FIELD, derived the same way for METRIC's own units (m3/day).",
        unit="(m3/(bar*day))/(mD*m/cP)",
    ),
    "RADIAL_FLOW_CONSTANT_LAB": Constant(
        value=22.619467105846507,
        description="Same reasoning as RADIAL_FLOW_CONSTANT_FIELD, derived the same way for LAB's own units (cm3/hr).",
        unit="(cm3/(atm*hr))/(mD*cm/cP)",
    ),
    "RADIAL_FLOW_CONSTANT_SI": Constant(
        value=6.283185307179586,
        description=(
            "2*pi exactly - SI's own material-property factors are all "
            "1.0 (coherent by construction), so only the geometric 2*pi "
            "compute_peaceman_well_index omits remains."
        ),
        unit="dimensionless",
    ),
    # Reservoir `Fluid` Defaults
    "RESERVOIR_GAS": Constant(
        value="Methane",
        description="Default gas that exists with oil in the reservoir (`Fluid` or CoolProp compatible fluid name)",
        unit=None,
    ),
    # Valid Ranges
    "MINIMUM_VALID_PRESSURE": Constant(
        value=14.5,
        description="Minimum valid pressure (below this, fluid model may be non-reservoir like)",
        unit="psi",
    ),
    "MAXIMUM_VALID_PRESSURE": Constant(
        value=14_700.0,
        description="Maximum valid pressure (above this, fluid model may be non-reservoir like)",
        unit="psi",
    ),
    "MINIMUM_VALID_TEMPERATURE": Constant(
        value=32.0,
        description="Minimum valid temperature (below this, fluid model may be non-reservoir like)",
        unit="°F",
    ),
    "MAXIMUM_VALID_TEMPERATURE": Constant(
        value=482.0,
        description="Maximum valid temperature (above this, fluid model may be non-reservoir like)",
        unit="°F",
    ),
    "GAS_PSEUDO_PRESSURE_THRESHOLD": Constant(
        value=0.0,
        description="Pressure threshold above which gas pseudo-pressure is used (psi)",
        unit="psi",
    ),
    "GAS_PSEUDO_PRESSURE_POINTS": Constant(
        value=200,
        description="Number of points to compute when generating gas pseudo-pressure table internally",
        unit="points",
    ),
    "SATURATION_EPSILON": ConstantFactory(
        factory=_get_saturation_epsilon,
        description=(
            "Clamp distance from 0 and 1 for normalised effective saturations in "
            "capillary-pressure and relative-permeability correlations with "
            "power-law singularities at those boundaries (Brooks-Corey, van "
            "Genuchten, LET). Evaluated at access time from the active dtype "
            "context via `get_floating_point_info()`. "
            "float64: ~1.78e-15 (8*eps) floored to 1e-7. "
            "float32: ~9.54e-7 (8*float32_eps)."
        ),
        unit="fraction",
    ),
    "MINIMUM_MOBILE_PORE_SPACE": ConstantFactory(
        factory=_get_minimum_pore_space,
        description=(
            "Minimum mobile pore-space fraction below which the corresponding "
            "phase relative-permeability or capillary-pressure is forced to zero. Matched to `SATURATION_EPSILON` so "
            "validity guards and clamp bounds are numerically consistent - a "
            "pore space smaller than the clamp floor cannot produce a meaningful "
            "normalised saturation. Dtype-aware via `get_floating_point_info()`."
        ),
        unit="fraction",
    ),
    "FINITE_DIFFERENCE_EPSILON": ConstantFactory(
        factory=_get_finite_difference_epsilon,
        description=(
            "Central finite-difference step for mixing-rule Jacobians and "
            "oil-wet relperm derivatives. Evaluated as max(cbrt(eps), 1e-5) "
            "where eps is the machine epsilon of the active dtype. "
            "float64: 1e-5. float32: ~4.93e-3 (cbrt(float32 eps))."
        ),
        unit="dimensionless",
    ),
    "MINIMUM_TRANSMISSIBILITY_FACTOR": Constant(
        value=1e-12,
        description="Minimum transmissibility factor to prevent numerical issues with very low transmissibility",
        unit="fraction",
    ),
    "GAS_SOLUBILITY_TOLERANCE": Constant(
        value=1e-6,
        description="Tolerance for gas solubility calculations",
        unit="fraction",
    ),
    "DEFAULT_MAX_COMPRESSIBILITY": Constant(
        value=1e-1,
        description=(
            "Compressibility ceiling for derived PVT compressibility tables, in FIELD units. "
            "Converted to the table's unit system with `get_conversion_factors`"
        ),
        unit="1/psi",
    ),
    "GAS_COMPRESSIBILITY_PRESSURE_FACTOR": Constant(
        value=2.0,
        description=(
            "Gas compressibility ceiling as a multiple of `1/P`. Ideal-gas `cg = 1/P` already "
            "exceeds `DEFAULT_MAX_COMPRESSIBILITY` at low pressure, so a fixed ceiling alone "
            "would clip valid gas data"
        ),
        unit="dimensionless",
    ),
    "DEFAULT_WATER_SALINITY_PPM": Constant(
        value=0,
        description="Default water salinity in parts per million (ppm)",
        unit="ppm",
    ),
    "MIN_OIL_ZONE_THICKNESS": Constant(
        value=5,
        description="Minimum oil zone thickness below which a warning is raised to notify that the oil zone is too thin",
        unit="ft",
    ),
    "FLUID_INCOMPRESSIBILITY_THRESHOLD": Constant(
        value=1e-6,
        description="Minimum fluid compressibility below which the fluid should be considered incompressible",
        unit="1/psi",
    ),
    # Wellbore Hydraulics - Friction Correlation
    "WELLBORE_LAMINAR_REYNOLDS_LIMIT": Constant(
        value=2300.0,
        description=(
            "Reynolds number below which tubing flow is treated as laminar "
            "(f = 64/Re) by the simplified Darcy friction-factor correlation."
        ),
        unit="dimensionless",
    ),
    "WELLBORE_TURBULENT_REYNOLDS_LIMIT": Constant(
        value=1.0e5,
        description=(
            "Reynolds number above which the simplified Darcy friction-factor "
            "correlation switches from the Blasius fit to the explicit "
            "Swamee-Jain-style approximation. Between "
            "WELLBORE_LAMINAR_REYNOLDS_LIMIT and this value, Blasius (f = "
            "0.316 * Re^-0.25) is used."
        ),
        unit="dimensionless",
    ),
    "COLEBROOK_MAX_ITERATIONS": Constant(
        value=50,
        description=(
            "Maximum fixed-point iterations for the Colebrook-White friction "
            "factor solve (friction_method='colebrook')."
        ),
        unit="iterations",
    ),
    "COLEBROOK_TOLERANCE": Constant(
        value=1.0e-10,
        description=(
            "Absolute convergence tolerance on the friction factor for the "
            "Colebrook-White fixed-point iteration."
        ),
        unit="dimensionless",
    ),
    "GRIFFITH_BUBBLE_SLIP_VELOCITY_FEET_PER_SECOND": Constant(
        value=0.8,
        description=(
            "Bubble rise (slip) velocity used by the Griffith correlation "
            "for bubble-flow liquid holdup, as adopted in the Hagedorn & "
            "Brown wellbore hydraulics model."
        ),
        unit="ft/s",
    ),
    # Well Control Resolution
    "CONTROL_MAX_FIXED_POINT_ITERATIONS": Constant(
        value=20,
        description=(
            "Maximum fixed-point iterations reconciling perforation flowing "
            "pressures against IPR-derived phase rates at a fixed reference "
            "pressure (BHP)."
        ),
        unit="iterations",
    ),
    "CONTROL_RATE_CONVERGENCE_TOLERANCE": Constant(
        value=1.0e-6,
        description=(
            "Relative convergence tolerance on total phase rate for both the "
            "pressure/rate fixed-point iteration and the BHP bisection search."
        ),
        unit="fraction",
    ),
    "CONTROL_MAX_BISECTION_ITERATIONS": Constant(
        value=40,
        description=(
            "Maximum bisection iterations solving for the reference pressure "
            "(BHP) that delivers a rate-mode control's target rate."
        ),
        unit="iterations",
    ),
    "CONTROL_INJECTOR_BHP_BRACKET_MULTIPLIER": Constant(
        value=10.0,
        description=(
            "Upper bisection bracket for injector rate-mode BHP search, "
            "expressed as a multiple of the highest connected-cell pressure. "
            "Widen this if an injector's target rate isn't reachable within "
            "the default bracket; the bisection is best-effort and returns "
            "the closest bound reached rather than raising if the bracket "
            "doesn't contain the solution."
        ),
        unit="dimensionless",
    ),
    "CONTROL_GROUP_RATE_CUTBACK_FACTOR": Constant(
        value=0.9,
        description=(
            "Fraction a group's own target rate is multiplied by, per "
            "enforcement call, when a `GECON` limit with `WorkoverAction.RATE` "
            "is breached. Applied once per call, not iterated internally, "
            "since satisfying the limit needs the group's targets "
            "reallocated across its members and those members re-resolved, "
            "both outside group-limit enforcement's own scope."
        ),
        unit="dimensionless",
    ),
    "HYSTERESIS_SATURATION_TOLERANCE": Constant(
        value=1e-6,
        description=(
            "Smallest saturation change between two time steps that counts as a phase "
            "moving up or down for hysteresis tracking. Smaller changes are treated as no "
            "movement, so solver round-off cannot flip a cell between drainage and "
            "imbibition or move its reversal saturation."
        ),
        unit="dimensionless",
    ),
    "AQUIFER_BESSEL_SERIES_TERMS": Constant(
        value=30,
        description=("Empirically validated safety margin for the Klins finite-aquifer series"),
        unit="dimensionless",
    ),
}


@typing.final
class Constants:
    """
    Physical constants and conversion factors.

    All constants are stored in an internal dictionary and can be accessed via dot notation.
    `Constants` can be modified at runtime if needed. Use `__getattr__` for value access and
    `__getitem__` for `Constant` object access.

    **Aliases**

    A `Constant`/`ConstantFactory` may declare `aliases=(...)`, alternate
    names for the same value. Aliases are *not* separate entries in the
    store; they are name-resolution redirects to the one canonical entry.
    Reading or writing through an alias always reaches the same underlying
    object as reading or writing through the canonical name. There is
    exactly one value in memory, not two copies to keep in sync.
    """

    __slots__ = ("_aliases", "_store")

    def __new__(cls, *args, **kwargs) -> Self:
        instance = super().__new__(cls)
        instance._store = {}
        instance._aliases = {}
        return instance

    def __init__(
        self,
        defaults: dict[str, typing.Any] | None = None,
    ) -> None:
        defaults = {**DEFAULT_CONSTANTS, **defaults} if defaults is not None else DEFAULT_CONSTANTS
        for name, value in defaults.items():
            if isinstance(value, (Constant, ConstantFactory)):
                wrapped = value
            elif callable(value):
                wrapped = ConstantFactory(factory=value)
            else:
                wrapped = Constant(value=value)
            self._store[name] = wrapped
            self._register_aliases(name, wrapped)

    def _register_aliases(self, canonical: str, constant: Constant | ConstantFactory) -> None:
        """
        Index `constant`'s declared aliases against `canonical` in `_aliases`.

        :raises ValidationError: If an alias collides with an existing
            canonical name, or is already claimed by a different canonical.
        """
        for alias in getattr(constant, "aliases", ()):
            if alias == canonical:
                continue
            if alias in self._store:
                raise ValidationError(
                    f"Alias {alias!r} for {canonical!r} collides with an "
                    "existing canonical constant name."
                )
            existing = self._aliases.get(alias)
            if existing is not None and existing != canonical:
                raise ValidationError(
                    f"Alias {alias!r} is already registered to "
                    f"{existing!r}; cannot also register it to {canonical!r}."
                )
            self._aliases[alias] = canonical

    def _unregister_aliases_for(self, canonical: str) -> None:
        """Drop every alias currently pointing at *canonical* (used on delete)."""
        for alias in (a for a, c in self._aliases.items() if c == canonical):
            del self._aliases[alias]

    def _wrap(self, canonical: str, value: typing.Any) -> Constant | ConstantFactory:
        """
        Wrap a value being assigned to `canonical`.

        A `Constant` or `ConstantFactory` is stored as given. A bare value or a callable replaces
        only the value of the constant already stored under `canonical`, so its description, unit and
        aliases carry over.
        """
        if isinstance(value, (Constant, ConstantFactory)):
            return value
        existing = self._store.get(canonical)
        if callable(value):
            if existing is None:
                return ConstantFactory(factory=value)
            return ConstantFactory(
                factory=value,
                description=existing.description,
                unit=existing.unit,
                aliases=existing.aliases,
            )
        if existing is None:
            return Constant(value=value)
        return Constant(
            value=value,
            description=existing.description,
            unit=existing.unit,
            aliases=existing.aliases,
        )

    def _resolve(self, name: str) -> str:
        """Resolve `name` to its canonical store key (identity if not an alias)."""
        return self._aliases.get(name, name)

    def __getattr__(self, name: str) -> typing.Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        try:
            constant = self._store[self._resolve(name)]
            return (
                constant.value if isinstance(constant, (Constant, ConstantFactory)) else constant
            )
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None

    def __getitem__(self, name: str) -> Constant:
        return self._store[self._resolve(name)]

    def __setattr__(self, name: str, value: typing.Any | Constant) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        canonical = self._resolve(name)
        wrapped = self._wrap(canonical, value)
        self._store[canonical] = wrapped
        # A value set via `c.SOME_ALIAS = ...` doesn't retroactively rename
        # the canonical slot, it just updates the value living there. If the
        # replacement itself declares new aliases, register those too.
        self._register_aliases(canonical, wrapped)

    def __setitem__(self, name: str, value: typing.Any | Constant) -> None:
        canonical = self._resolve(name)
        wrapped = self._wrap(canonical, value)
        self._store[canonical] = wrapped
        self._register_aliases(canonical, wrapped)

    def __delattr__(self, name: str) -> None:
        if name.startswith("_"):
            object.__delattr__(self, name)
            return

        if name in self._aliases:
            # Deleting via an alias only removes that pointer.
            del self._aliases[name]
            return
        try:
            del self._store[name]
            self._unregister_aliases_for(name)
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None

    def __delitem__(self, name: str) -> None:
        if name in self._aliases:
            del self._aliases[name]
            return
        del self._store[name]
        self._unregister_aliases_for(name)

    def __contains__(self, name: str) -> bool:
        return name in self._store or name in self._aliases

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self._store)

    def keys(self) -> typing.KeysView[str]:
        return self._store.keys()

    def values(self) -> typing.ValuesView[Constant]:
        return self._store.values()

    def items(self) -> typing.ItemsView[str, Constant]:
        return self._store.items()

    def get(self, name: str, default: typing.Any = None) -> typing.Any:
        constant = self._store.get(self._resolve(name))
        if constant is None:
            return default
        return constant.value if isinstance(constant, (Constant, ConstantFactory)) else constant

    def get_constant(self, name: str, default: Constant | None = None) -> Constant | None:
        return self._store.get(self._resolve(name), default)

    def __dir__(self):
        default = super().__dir__()
        return sorted({*default, *self._store.keys(), *self._aliases.keys()})

    def _ipython_key_completions_(self) -> list[str]:
        return sorted({*self._store.keys(), *self._aliases.keys()})

    def __len__(self) -> int:
        return len(self._store)


DEFAULT_CONTEXT_ID = uuid4().hex
_constants_context: ContextVar[tuple[Constants, str]] = ContextVar(
    "constants_context", default=(Constants(), DEFAULT_CONTEXT_ID)
)


class ConstantsContext:
    """
    Context manager for temporary process-local `Constants` overrides.

    This context manager allows for temporary overrides of the global `Constants`
    instance within a specific context. Upon exiting the context, the previous
    `Constants` instance is restored.
    """

    __slots__ = ("_constants", "_entry_depth", "_id", "_token")

    def __init__(self, constants: Constants) -> None:
        """
        Initialize the context manager with a new `Constants` instance.

        :param constants: New `Constants` instance to use within the context
        """
        self._constants = constants
        self._entry_depth = 0
        self._id = uuid4().hex
        self._token: Token[tuple[Constants, str]] | None = None

    @property
    def id(self) -> str:
        """The context id"""
        return self._id

    @property
    def constants(self) -> Constants:
        """The context's `Constants`"""
        return self._constants

    def __enter__(self) -> Constants:
        """
        Enter the context, setting the new `Constants` instance.

        :return: The new `Constants` instance
        """
        context_id = _constants_context.get()[1]
        if context_id == self._id:
            warnings.warn(
                f"Constant context {context_id!r} is already active; re-entering it is unnecessary.",
                UserWarning,
                stacklevel=2,
            )
            self._entry_depth += 1
            return self._constants

        if self._entry_depth:
            raise RuntimeError(
                f"Constant context {context_id!r} is already active in another context."
            )

        self._token = _constants_context.set((self._constants, self._id))
        self._entry_depth = 1
        return self._constants

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context, restoring the previous `Constants` instance."""
        if self._entry_depth == 0 or _constants_context.get()[1] != self._id:
            return

        self._entry_depth -= 1
        if self._entry_depth == 0 and self._token is not None:
            _constants_context.reset(self._token)
            self._token = None


@typing.final
class __ConstantsProxy:
    """
    Proxy class to access the current context's `Constants` instance.

    Use the `constants` property to get the current `Constants` instance.

    Override the current `Constants` instance using the `ConstantsContext` context manager.
    """

    @property
    def context_id(self) -> str:
        """
        Get the current context's ID.

        :return: `ConstantsContext` ID.
        """
        return _constants_context.get()[1]

    @property
    def value(self) -> Constants:
        """
        Get the current context's `Constants` instance.

        :return: Current `Constants` instance
        """
        return _constants_context.get()[0]

    def in_default_context(self) -> bool:
        """Returns `True` if we are the default (process local) `Constants` context"""
        return self.context_id == DEFAULT_CONTEXT_ID

    def __getattr__(self, name: str) -> typing.Any:
        """
        Get a constant's value from the current context's `Constants` instance.

        :param name: Name of the constant
        :return: Value of the constant
        :raises AttributeError: If the constant does not exist
        """
        return getattr(self.value, name)

    def __getitem__(self, name: str) -> Constant:
        """
        Get a Constant object from the current context's `Constants` instance.

        :param name: Name of the constant
        :return: Constant object
        :raises KeyError: If the constant does not exist
        """
        return self.value[name]

    def __dir__(self):
        default = super().__dir__()
        return sorted({*default, *self.value.__dir__()})

    def _ipython_key_completions_(self) -> list[str]:
        return self.value._ipython_key_completions_()


CONSTANTS = c = __ConstantsProxy()
"""Global proxy to access physical constants and conversion factors."""


def get_constants() -> Constants:
    """Return the `Constants` instance active in the current context."""
    return _constants_context.get()[0]


def get_constant(name: str) -> Constant | None:
    """
    Get a `Constant` object by name from the global constants.

    :param name: Name of the constant
    :return: `Constant` object or None if not found
    """
    return c.value.get_constant(name)


def set_constants(constants: Constants, /) -> None:
    """
    Set/override the default (process local) `Constants` used.

    Note: This method should not be called inside a `ConstantsContext`.
        An error will be raise if called like so.

    :param constants: The `Constants` object to set as default.
    """
    if c.context_id != DEFAULT_CONTEXT_ID:
        raise ValidationError(
            "Cannot set constants. Are you in a `ConstantsContext`? Only call in default context"
        )
    _constants_context.set((constants, DEFAULT_CONTEXT_ID))

"""Physical constants and conversion factors"""

import logging
import typing
from contextvars import ContextVar, Token
from uuid import uuid4

import attrs
from typing_extensions import Self

from bores.errors import ValidationError
from bores.precision import get_floating_point_info
from bores.serde.base import Serializable
from bores.serde.stores import StoreSerializable
from bores.types import UnitConversionFactors, UnitConversionTable, UnitSystem

__all__ = [
    "UNIT_CONVERSION_TABLE",
    "Constant",
    "Constants",
    "ConstantsContext",
    "build_unit_conversion_table",
    "c",
    "get_constant",
    "get_conversion_factors",
    "set_default_constants",
]

logger = logging.getLogger(__name__)


@typing.final
@attrs.frozen(slots=True)
class Constant(Serializable):
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
class ConstantFactory(Serializable):
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

    def __dump__(self) -> dict[str, typing.Any]:
        """Serialize by evaluating the factory - produces a plain value snapshot."""
        evaluated = Constant(
            value=self.value,
            description=self.description,
            unit=self.unit,
            aliases=self.aliases,
        )
        return evaluated.dump()

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Constant:
        """
        Deserialization always produces a plain ``Constant`` - a factory
        function cannot be reconstructed from serialized data.
        """
        return Constant.load(data)


def _sat_eps_factory() -> float:
    """
    Saturation clamp epsilon, scaled to the active floating-point dtype.

    Target: 8 * machine_epsilon, floored at sqrt(float32 eps) ~ 1e-7 so the
    clamp is always meaningful even if the context dtype is float64.
    """
    info = get_floating_point_info()
    dtype_based = 8.0 * float(info.eps)
    floor = 1e-7  # meaningful saturation floor, representable in float32
    return max(dtype_based, floor)


def _min_pore_space_factory() -> float:
    """
    Minimum mobile pore-space guard, matched to the saturation epsilon so
    validity guards and saturation clamps stay numerically consistent.
    """
    return _sat_eps_factory()


def _fd_eps_factory() -> float:
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
        value=14.6959,
        description="Standard atmospheric pressure in Pound-per-Square-Inch (psi)",
        unit="psi",
        aliases=("STANDARD_PRESSURE_IMPERIAL",),
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
        value=9.6787e-4,
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
        value=288.7056,
        description="Standard temperature in Kelvin",
        unit="K",
        aliases=("STANDARD_TEMPERATURE_SI",),
    ),
    "STANDARD_TEMPERATURE_FAHRENHEIT": Constant(
        value=60.0,
        description="Standard temperature in Fahrenheit",
        unit="°F",
        aliases=("STANDARD_TEMPERATURE_IMPERIAL",),
    ),
    "STANDARD_TEMPERATURE_RANKINE": Constant(
        value=518.67, description="Standard temperature (15.6°C) in Rankine", unit="°R"
    ),
    "STANDARD_TEMPERATURE_CELSIUS": Constant(
        value=15.6, description="Standard temperature in Celsius", unit="°C"
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
        value=3.0e-4, description="Thermal expansion coefficient for water", unit="1/K"
    ),
    "WATER_THERMAL_EXPANSION_COEFFICIENT_IMPERIAL": Constant(
        value=1.67e-4,
        description="Thermal expansion coefficient for water (Imperial units)",
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
        value=998.2,
        description="Standard water density at Standard temperature (15.6°C) (Metric units)",
        unit="kg/m³",
        aliases=("STANDARD_WATER_DENSITY_SI",),
    ),
    "STANDARD_WATER_DENSITY_LAB": Constant(
        value=0.9982,
        description="Standard water density at Standard temperature (15.6°C) (Lab units)",
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
        value=6894.757,
        description="Conversion factor from psi to Pascals",
        unit="Pa/psi",
    ),
    "PASCAL_TO_PSI": Constant(
        value=1 / 6894.757,
        description="Conversion factor from Pascals to psi",
        unit="psi/Pa",
    ),
    "PSI_TO_BAR": Constant(
        value=0.0689476, description="Conversion factor from psi to bar", unit="bar/psi"
    ),
    "ATM_TO_PASCAL": Constant(
        value=101325.0,
        description="Conversion factor from atm to Pascals",
        unit="atm/Pa",
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
        value=9.869233e-16,
        description="Conversion factor from millidarcies to square meters",
        unit="m²/mD",
    ),
    # Gas-Oil Ratio Conversions
    "SCF_PER_STB_TO_CUBIC_METER_PER_CUBIC_METER": Constant(
        value=0.1781076,
        description="Conversion factor from scf/STB to m³/m³",
        unit="(m³/m³)/(scf/STB)",
    ),
    "CUBIC_METER_PER_CUBIC_METER_TO_SCF_PER_STB": Constant(
        value=1 / 0.1781076,
        description="Conversion factor from m³/m³ to scf/STB",
        unit="(scf/STB)/(m³/m³)",
    ),
    # Formation Volume Factor Conversions
    "CUBIC_METER_PER_CUBIC_METER_TO_BARRELS_PER_SCF": Constant(
        value=5.614583,
        description="Conversion factor from m³/m³ to BBL/scf",
        unit="(BBL/scf)/(m³/m³)",
    ),
    "BARRELS_PER_SCF_TO_CUBIC_METER_PER_CUBIC_METER": Constant(
        value=1 / 5.614583,
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
        value=35.3147,
        description="Conversion factor from cubic meters to standard cubic feet",
        unit="scf/m³",
    ),
    "SCF_TO_BARRELS": Constant(
        value=0.1781076,
        description="Conversion factor from standard cubic feet to barrels",
        unit="BBL/scf",
    ),
    "BARRELS_TO_CUBIC_FEET": Constant(
        value=5.614583,
        description="Conversion factor from barrels to cubic feet",
        unit="ft³/BBL",
    ),
    "CUBIC_FEET_TO_BARRELS": Constant(
        value=1 / 5.614583,
        description="Conversion factor from cubic feet to barrels",
        unit="BBL/ft³",
    ),
    "STB_TO_CUBIC_FEET": Constant(
        value=5.614583,
        description="Conversion factor from stock tank barrels to (standard) cubic feet",
        unit="ft³/STB",
    ),
    "CUBIC_FEET_TO_STB": Constant(
        value=1 / 5.614583,
        description="Conversion factor from (standard) cubic feet to stock tank barrels",
        unit="STB/ft³",
    ),
    "STB_TO_CUBIC_METER": Constant(
        value=0.158987,
        description="Conversion factor from stock tank barrels to (standard) cubic meters",
        unit="m³/STB",
    ),
    "CUBIC_METER_TO_STB": Constant(
        value=1 / 0.158987,
        description="Conversion factor from (standard) cubic meters to stock tank barrels",
        unit="STB/m³",
    ),
    "BARRELS_TO_CUBIC_METER": Constant(
        value=0.158987,
        description="Conversion factor from barrels to cubic meters",
        unit="m³/BBL",
    ),
    "CUBIC_METER_TO_BARRELS": Constant(
        value=1 / 0.158987,
        description="Conversion factor from cubic meters to barrels",
        unit="BBL/m³",
    ),
    "SCF_TO_SCM": Constant(
        value=0.0283168,
        description="Conversion factor from standard cubic feet to standard cubic meters",
        unit="m³/scf",
    ),
    "DYNE_PER_CENTIMETER_TO_PSI": Constant(
        value=4.621,
        description="Conversion factor from dyne per centimeter to pounds per square inch",
        unit="dyne/cm·psi",
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
        value=10.73159,
        description="Universal gas constant (Imperial units)",
        unit="ft³·psi/(lb·mol·°R)",
        aliases=("IDEAL_GAS_CONSTANT_IMPERIAL",),
    ),
    # Density Conversions
    "POUNDS_PER_CUBIC_FEET_TO_KILOGRAM_PER_CUBIC_METER": Constant(
        value=16.0185,
        description="Conversion factor from lb/ft³ to kg/m³",
        unit="(kg/m³)/(lb/ft³)",
    ),
    "KILOGRAM_PER_CUBIC_METER_TO_POUNDS_PER_CUBIC_FEET": Constant(
        value=1 / 16.0185,
        description="Conversion factor from kg/m³ to lb/ft³",
        unit="(lb/ft³)/(kg/m³)",
    ),
    "POUNDS_PER_CUBIC_FEET_TO_GRAMS_PER_CUBIC_METER": Constant(
        value=0.01601846,
        description="Conversion factor from lb/ft³ to g/cm³",
        unit="(g/cm³)/(lb/ft³)",
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
        value=543168.384,
        description="Conversion factor from m³/s to STB/day",
        unit="(STB/day)/(m³/s)",
    ),
    "STB_PER_DAY_TO_CUBIC_METER_PER_SECOND": Constant(
        value=1 / 543168.384,
        description="Conversion factor from STB/day to m³/s",
        unit="(m³/s)/(STB/day)",
    ),
    "CUBIC_METER_PER_SECOND_TO_SCF_PER_DAY": Constant(
        value=3049492.8,
        description="Conversion factor from m³/s to scf/day",
        unit="(scf/day)/(m³/s)",
    ),
    "SCF_PER_DAY_TO_CUBIC_METER_PER_SECOND": Constant(
        value=1 / 3049492.8,
        description="Conversion factor from scf/day to m³/s",
        unit="(m³/s)/(scf/day)",
    ),
    # Time Conversions
    "SECONDS_PER_DAY": Constant(
        value=86400.0, description="Number of seconds in a day", unit="s/day"
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
        value=1.127e-3,
        description="Conversion factor from mD/cP to ft²/(psi·day)",
        unit="(ft²/(psi·day))/(mD/cP)",
    ),
    "MILLIDARCIES_PER_CENTIPOISE_TO_SQUARE_FEET_PER_PSI_PER_SECOND": Constant(
        value=1.127e-3 / 86400.0,
        description="Conversion factor from mD/cP to ft²/(psi·s)",
        unit="(ft²/(psi·s))/(mD/cP)",
    ),
    "MILLIDARCIES_FT_PER_CENTIPOISE_TO_CUBIC_FEET_PER_PSI_PER_DAY": Constant(
        value=1.127e-3,
        unit="(ft³/(psi·day))/(mD·ft/cP)",
        description="Conversion factor from mD·ft/cP to ft³/(psi·day)",
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
        factory=_sat_eps_factory,
        description=(
            "Clamp distance from 0 and 1 for normalised effective saturations in "
            "capillary-pressure and relative-permeability correlations with "
            "power-law singularities at those boundaries (Brooks-Corey, van "
            "Genuchten, LET). Evaluated at access time from the active dtype "
            "context via `get_floating_point_info()`. "
            "float64: ~1.78e-15 (8*eps) floored to ~1.49e-8 (sqrt(float32 eps)). "
            "float32: ~9.54e-7 (8*float32_eps)."
        ),
        unit="fraction",
    ),
    "MINIMUM_MOBILE_PORE_SPACE": ConstantFactory(
        factory=_min_pore_space_factory,
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
        factory=_fd_eps_factory,
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
    "AQUIFER_BESSEL_SERIES_TERMS": Constant(
        value=30,
        description=("Empirically validated safety margin for the Klins finite-aquifer series"),
        unit="dimensionless",
    ),
}


@typing.final
class Constants(
    StoreSerializable,
    fields={"_store": dict[str, Constant]},
):
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
        wrapped = value if isinstance(value, Constant) else Constant(value=value)
        self._store[canonical] = wrapped
        # A value set via `c.SOME_ALIAS = ...` doesn't retroactively rename
        # the canonical slot, it just updates the value living there. If the
        # replacement itself declares new aliases, register those too.
        self._register_aliases(canonical, wrapped)

    def __setitem__(self, name: str, value: typing.Any | Constant) -> None:
        canonical = self._resolve(name)
        wrapped = value if isinstance(value, Constant) else Constant(value=value)
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
        return constant.value if isinstance(constant, Constant) else constant

    def get_constant(self, name: str, default: Constant | None = None) -> Constant | None:
        return self._store.get(self._resolve(name), default)

    def __dir__(self):
        default = super().__dir__()
        return sorted({*default, *self._store.keys(), *self._aliases.keys()})

    def _ipython_key_completions_(self) -> list[str]:
        return sorted({*self._store.keys(), *self._aliases.keys()})

    def __len__(self) -> int:
        return len(self._store)

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        """Load constants from dict, re-deriving the alias index from each entry."""
        constants = cls(defaults={})  # type: ignore[arg-type]
        for name, val in data.items():
            wrapped = (
                Constant.load(val)
                if isinstance(val, dict) and "value" in val
                else Constant(value=val)
            )
            constants._store[name] = wrapped
            constants._register_aliases(name, wrapped)
        return constants


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

    __slots__ = ("_constants", "_id", "_token")

    def __init__(self, constants: Constants) -> None:
        """
        Initialize the context manager with a new `Constants` instance.

        :param constants: New `Constants` instance to use within the context
        """
        self._constants = constants
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
        self._token = _constants_context.set((self._constants, self._id))
        return self._constants

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context, restoring the previous `Constants` instance."""
        if self._token is not None:
            _constants_context.reset(self._token)


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
    def _constants(self) -> Constants:
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
        return getattr(self._constants, name)

    def __getitem__(self, name: str) -> Constant:
        """
        Get a Constant object from the current context's `Constants` instance.

        :param name: Name of the constant
        :return: Constant object
        :raises KeyError: If the constant does not exist
        """
        return self._constants[name]

    def __dir__(self):
        default = super().__dir__()
        return sorted({*default, *self._constants.__dir__()})

    def _ipython_key_completions_(self) -> list[str]:
        return self._constants._ipython_key_completions_()


c = __ConstantsProxy()
"""Global proxy to access physical constants and conversion factors."""


def get_constant(name: str) -> Constant | None:
    """
    Get a `Constant` object by name from the global constants.

    :param name: Name of the constant
    :return: `Constant` object or None if not found
    """
    return c._constants.get_constant(name)


def set_default_constants(constants: Constants, /) -> None:
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


def build_unit_conversion_table(constants: Constants | None = None) -> UnitConversionTable:
    """
    Build a complete unit conversion table from the provided or default
    constants registry.

    All numeric values are read from `constants` (or the global `c`
    proxy) so they stay in sync with any application-level overrides made
    via `ConstantsContext`.

    Each entry converts every dimensional quantity from the source
    `UnitSystem` to the target `UnitSystem`. The twelve source -> target
    pairs cover all ordered combinations of FIELD, METRIC, LAB, and SI
    (same-system pairs are handled by `IDENTITY_FACTORS` in
    `get_conversion_factors`).

    Intermediate factors are derived algebraically from the primitives
    stored in the constants registry so no magic numbers are hard-coded
    here.
    """
    con = constants if constants is not None else c

    # Primitive conversion factors from the constants registry
    psi_to_pa: float = con.PSI_TO_PASCAL  # 6894.757 Pa/psi
    psi_to_bar: float = con.PSI_TO_BAR  # 0.0689476 bar/psi
    atm_to_pa: float = con.ATM_TO_PASCAL  # 101 325.0 Pa/atm
    ft_to_m: float = con.FEET_TO_METERS  # 0.3048 m/ft
    m_to_ft: float = con.METERS_TO_FEET  # 3.28084 ft/m
    lbm_ft3_to_kg_m3: float = con.POUNDS_PER_CUBIC_FEET_TO_KILOGRAM_PER_CUBIC_METER
    lbm_ft3_to_g_cm3: float = con.POUNDS_PER_CUBIC_FEET_TO_GRAMS_PER_CUBIC_METER
    cp_to_pas: float = con.CENTIPOISE_TO_PASCAL_SECONDS  # 0.001
    md_to_m2: float = con.MILLIDARCY_TO_SQUARE_METER  # 9.869233e-16
    scf_stb_to_sm3_sm3: float = con.SCF_PER_STB_TO_CUBIC_METER_PER_CUBIC_METER
    stb_to_m3: float = con.STB_TO_CUBIC_METER  # 0.158987
    scf_to_m3: float = con.SCF_TO_SCM  # 0.0283168
    seconds_per_day: float = con.SECONDS_PER_DAY  # 86400.0
    hours_per_day: float = con.HOURS_PER_DAY  # 24.0

    # Derived intermediates (no magic numbers beyond what is above)
    cm_to_m: float = 0.01
    m_to_cm: float = 100.0
    ft_to_cm: float = ft_to_m * m_to_cm
    cm_to_ft: float = cm_to_m * m_to_ft
    kg_m3_to_g_cm3: float = cm_to_m**3  # 1e-6 / 1e-3 = 1e-3

    bar_to_pa: float = psi_to_pa / (psi_to_pa / atm_to_pa * (1.0 / psi_to_bar) * psi_to_bar)
    # Simpler: bar_to_pa = 1e5; but derive from constants to stay consistent
    # 1 bar = 14.5038 psi; bar_to_pa = 14.5038 * psi_to_pa / 14.5038... just:
    bar_to_pa = 1.0 / psi_to_bar * psi_to_pa  # 100 000 Pa/bar
    psi_to_atm: float = psi_to_pa / atm_to_pa
    bar_to_atm: float = bar_to_pa / atm_to_pa

    # Volume (reservoir)
    ft3_to_m3: float = ft_to_m**3
    m3_to_ft3: float = m_to_ft**3
    ft3_to_cm3: float = ft_to_cm**3
    cm3_to_ft3: float = cm_to_ft**3
    m3_to_cm3: float = m_to_cm**3
    cm3_to_m3: float = cm_to_m**3

    # Time
    seconds_per_hour: float = seconds_per_day / hours_per_day  # 3600.0
    days_per_second: float = 1.0 / seconds_per_day

    # Surface volumes
    # STB -> m³: stb_to_m3
    # STB -> cm³:
    stb_to_cm3: float = stb_to_m3 * m3_to_cm3
    # SCF -> m³: scf_to_m3
    # SCF -> cm³:
    scf_to_cm3: float = scf_to_m3 * m3_to_cm3
    # Sm³ -> scc:
    sm3_to_scc: float = m3_to_cm3
    # scc -> Sm³:
    scc_to_sm3: float = cm3_to_m3

    # GOR: SCF/STB -> Sm³/Sm³
    # = (scf_to_m3) / (stb_to_m3)  -- same as scf_stb_to_sm3_sm3
    gor_field_to_metric: float = scf_stb_to_sm3_sm3
    # GOR: Sm³/Sm³ -> SCF/STB
    gor_metric_to_field: float = 1.0 / scf_stb_to_sm3_sm3
    # GOR: SCF/STB -> scc/scc  (scf->scc / stb->scc)
    gor_field_to_lab: float = scf_to_cm3 / stb_to_cm3
    # GOR: scc/scc -> SCF/STB
    gor_lab_to_field: float = 1.0 / gor_field_to_lab
    # GOR: Sm³/Sm³ -> scc/scc  (both dimensionless, same ratio - 1.0)
    # Sm³/Sm³ and scc/scc are both volume/volume in their respective systems;
    # the numerical value of the ratio is unchanged.
    gor_metric_to_lab: float = 1.0
    gor_lab_to_metric: float = 1.0

    # OGR (Rv): STB/SCF -> Sm³/Sm³
    ogr_field_to_metric: float = stb_to_m3 / scf_to_m3
    ogr_metric_to_field: float = 1.0 / ogr_field_to_metric
    ogr_field_to_lab: float = stb_to_cm3 / scf_to_cm3
    ogr_lab_to_field: float = 1.0 / ogr_field_to_lab
    ogr_metric_to_lab: float = 1.0
    ogr_lab_to_metric: float = 1.0

    # FVF
    # liquid FVF: rb/STB -> rm³/Sm³
    # rb = reservoir barrel = ft³/5.614583 ... but FVF is dimensionless ratio
    # rb/STB and rm³/Sm³ are both (reservoir vol)/(surface vol); what changes
    # is the unit of each. rb/STB = 5.614583 ft³ / (5.614583 ft³) = 1 numerically
    # if reservoir and surface are same fluid. The actual conversion factor
    # between rb/STB and rm³/Sm³ is:
    #   (rb -> rm³) / (STB -> Sm³) = (stb_to_m3) / (stb_to_m3) = 1.0
    # Similarly rcf/SCF -> rm³/Sm³ = (ft3_to_m3) / (scf_to_m3)
    liquid_fvf_field_to_metric: float = 1.0  # rb/STB -> rm³/Sm³
    liquid_fvf_field_to_lab: float = 1.0  # rb/STB -> rcc/scc
    liquid_fvf_field_to_si: float = 1.0  # rb/STB -> rm³/Sm³
    gas_fvf_field_to_metric: float = ft3_to_m3 / scf_to_m3  # rcf/SCF -> rm³/Sm³
    gas_fvf_field_to_lab: float = ft3_to_cm3 / scf_to_cm3  # rcf/SCF -> rcc/scc
    gas_fvf_metric_to_field: float = 1.0 / gas_fvf_field_to_metric
    gas_fvf_lab_to_field: float = 1.0 / gas_fvf_field_to_lab

    # Surface liquid rates: STB/day -> Sm³/day, scc/hr, Sm³/s
    liquid_rate_field_to_metric: float = stb_to_m3  # STB/day -> Sm³/day
    liquid_rate_field_to_lab: float = stb_to_cm3 / hours_per_day  # STB/day -> scc/hr
    liquid_rate_field_to_si: float = stb_to_m3 / seconds_per_day  # STB/day -> Sm³/s
    liquid_rate_metric_to_field: float = 1.0 / liquid_rate_field_to_metric
    liquid_rate_metric_to_lab: float = m3_to_cm3 / hours_per_day  # Sm³/day -> scc/hr
    liquid_rate_metric_to_si: float = days_per_second  # Sm³/day -> Sm³/s
    liquid_rate_lab_to_field: float = 1.0 / liquid_rate_field_to_lab
    liquid_rate_lab_to_metric: float = 1.0 / liquid_rate_metric_to_lab
    liquid_rate_lab_to_si: float = cm3_to_m3 * seconds_per_hour  # scc/hr -> Sm³/s
    liquid_rate_si_to_field: float = 1.0 / liquid_rate_field_to_si
    liquid_rate_si_to_metric: float = 1.0 / liquid_rate_metric_to_si
    liquid_rate_si_to_lab: float = 1.0 / liquid_rate_lab_to_si

    # Surface gas rates: SCF/day -> Sm³/day, scc/hr, Sm³/s
    gas_rate_field_to_metric: float = scf_to_m3  # SCF/day -> Sm³/day
    gas_rate_field_to_lab: float = scf_to_cm3 / hours_per_day  # SCF/day -> scc/hr
    gas_rate_field_to_si: float = scf_to_m3 / seconds_per_day  # SCF/day -> Sm³/s
    gas_rate_metric_to_field: float = 1.0 / gas_rate_field_to_metric
    gas_rate_metric_to_lab: float = m3_to_cm3 / hours_per_day  # Sm³/day -> scc/hr
    gas_rate_metric_to_si: float = days_per_second  # Sm³/day -> Sm³/s
    gas_rate_lab_to_field: float = 1.0 / gas_rate_field_to_lab
    gas_rate_lab_to_metric: float = 1.0 / gas_rate_metric_to_lab
    gas_rate_lab_to_si: float = cm3_to_m3 * seconds_per_hour  # scc/hr -> Sm³/s
    gas_rate_si_to_field: float = 1.0 / gas_rate_field_to_si
    gas_rate_si_to_metric: float = 1.0 / gas_rate_metric_to_si
    gas_rate_si_to_lab: float = 1.0 / gas_rate_lab_to_si

    # Reservoir rates: ft³/day -> m³/day, cm³/hr, m³/s
    res_rate_field_to_metric: float = ft3_to_m3  # ft³/day -> m³/day
    res_rate_field_to_lab: float = ft3_to_cm3 / hours_per_day  # ft³/day -> cm³/hr
    res_rate_field_to_si: float = ft3_to_m3 / seconds_per_day  # ft³/day -> m³/s
    res_rate_metric_to_field: float = 1.0 / res_rate_field_to_metric
    res_rate_metric_to_lab: float = m3_to_cm3 / hours_per_day  # m³/day -> cm³/hr
    res_rate_metric_to_si: float = days_per_second  # m³/day -> m³/s
    res_rate_lab_to_field: float = 1.0 / res_rate_field_to_lab
    res_rate_lab_to_metric: float = 1.0 / res_rate_metric_to_lab
    res_rate_lab_to_si: float = cm3_to_m3 * seconds_per_hour  # cm³/hr -> m³/s
    res_rate_si_to_field: float = 1.0 / res_rate_field_to_si
    res_rate_si_to_metric: float = 1.0 / res_rate_metric_to_si
    res_rate_si_to_lab: float = 1.0 / res_rate_lab_to_si

    # Mass
    mass_field_to_metric: float = lbm_ft3_to_kg_m3 * ft3_to_m3  # lbm -> kg
    mass_field_to_lab: float = lbm_ft3_to_g_cm3 * ft3_to_cm3  # lbm -> g
    mass_metric_to_field: float = 1.0 / mass_field_to_metric
    mass_metric_to_lab: float = 1.0 / kg_m3_to_g_cm3  # kg -> g (1000)
    mass_lab_to_field: float = 1.0 / mass_field_to_lab
    mass_lab_to_metric: float = kg_m3_to_g_cm3  # g -> kg

    def _inverse(x: float) -> float:
        return 1.0 / x

    table: UnitConversionTable = {
        ##############
        # FIELD -> *
        ##############
        (UnitSystem.FIELD, UnitSystem.METRIC): UnitConversionFactors(
            pressure=psi_to_bar,
            length=ft_to_m,
            area=ft_to_m**2,
            volume=ft3_to_m3,
            time=1.0,  # day -> day
            mass=mass_field_to_metric,
            temperature=5.0 / 9.0,
            temperature_offset=(-32.0) * (5.0 / 9.0),  # °F -> °C
            density=lbm_ft3_to_kg_m3,
            viscosity=1.0,  # cP -> cP
            permeability=1.0,  # mD -> mD
            compressibility=_inverse(psi_to_bar),
            liquid_surface_volume=stb_to_m3,
            gas_surface_volume=scf_to_m3,
            liquid_fvf=liquid_fvf_field_to_metric,
            gas_fvf=gas_fvf_field_to_metric,
            gas_oil_ratio=gor_field_to_metric,
            oil_gas_ratio=ogr_field_to_metric,
            liquid_surface_rate=liquid_rate_field_to_metric,
            gas_surface_rate=gas_rate_field_to_metric,
            reservoir_rate=res_rate_field_to_metric,
        ),
        (UnitSystem.FIELD, UnitSystem.SI): UnitConversionFactors(
            pressure=psi_to_pa,
            length=ft_to_m,
            area=ft_to_m**2,
            volume=ft3_to_m3,
            time=days_per_second,  # day -> s
            mass=mass_field_to_metric,  # lbm -> kg (SI mass = kg)
            temperature=5.0 / 9.0,
            temperature_offset=(-32.0 * 5.0 / 9.0) + 273.15,  # °F -> K
            density=lbm_ft3_to_kg_m3,
            viscosity=cp_to_pas,
            permeability=md_to_m2,
            compressibility=_inverse(psi_to_pa),
            liquid_surface_volume=stb_to_m3,
            gas_surface_volume=scf_to_m3,
            liquid_fvf=liquid_fvf_field_to_si,
            gas_fvf=gas_fvf_field_to_metric,  # rcf/SCF -> rm³/Sm³ same as metric
            gas_oil_ratio=gor_field_to_metric,
            oil_gas_ratio=ogr_field_to_metric,
            liquid_surface_rate=liquid_rate_field_to_si,
            gas_surface_rate=gas_rate_field_to_si,
            reservoir_rate=res_rate_field_to_si,
        ),
        (UnitSystem.FIELD, UnitSystem.LAB): UnitConversionFactors(
            pressure=psi_to_atm,
            length=ft_to_cm,
            area=ft_to_cm**2,
            volume=ft3_to_cm3,
            time=_inverse(hours_per_day),  # day -> hr
            mass=mass_field_to_lab,
            temperature=5.0 / 9.0,
            temperature_offset=(-32.0) * (5.0 / 9.0),  # °F -> °C
            density=lbm_ft3_to_g_cm3,
            viscosity=1.0,  # cP -> cP
            permeability=1.0,  # mD -> mD
            compressibility=_inverse(psi_to_atm),
            liquid_surface_volume=stb_to_cm3,
            gas_surface_volume=scf_to_cm3,
            liquid_fvf=liquid_fvf_field_to_lab,
            gas_fvf=gas_fvf_field_to_lab,
            gas_oil_ratio=gor_field_to_lab,
            oil_gas_ratio=ogr_field_to_lab,
            liquid_surface_rate=liquid_rate_field_to_lab,
            gas_surface_rate=gas_rate_field_to_lab,
            reservoir_rate=res_rate_field_to_lab,
        ),
        ##############
        # METRIC -> *
        ##############
        (UnitSystem.METRIC, UnitSystem.FIELD): UnitConversionFactors(
            pressure=_inverse(psi_to_bar),
            length=m_to_ft,
            area=m_to_ft**2,
            volume=m3_to_ft3,
            time=1.0,  # day -> day
            mass=mass_metric_to_field,
            temperature=9.0 / 5.0,
            temperature_offset=32.0,  # °C -> °F
            density=_inverse(lbm_ft3_to_kg_m3),
            viscosity=1.0,
            permeability=1.0,
            compressibility=psi_to_bar,
            liquid_surface_volume=_inverse(stb_to_m3),
            gas_surface_volume=_inverse(scf_to_m3),
            liquid_fvf=_inverse(liquid_fvf_field_to_metric),
            gas_fvf=gas_fvf_metric_to_field,
            gas_oil_ratio=gor_metric_to_field,
            oil_gas_ratio=ogr_metric_to_field,
            liquid_surface_rate=liquid_rate_metric_to_field,
            gas_surface_rate=gas_rate_metric_to_field,
            reservoir_rate=res_rate_metric_to_field,
        ),
        (UnitSystem.METRIC, UnitSystem.SI): UnitConversionFactors(
            pressure=bar_to_pa,
            length=1.0,
            area=1.0,
            volume=1.0,
            time=days_per_second,  # day -> s
            mass=1.0,  # kg -> kg
            temperature=1.0,
            temperature_offset=273.15,  # °C -> K
            density=1.0,
            viscosity=cp_to_pas,
            permeability=md_to_m2,
            compressibility=_inverse(bar_to_pa),
            liquid_surface_volume=1.0,  # Sm³ -> Sm³
            gas_surface_volume=1.0,
            liquid_fvf=1.0,
            gas_fvf=1.0,
            gas_oil_ratio=1.0,
            oil_gas_ratio=1.0,
            liquid_surface_rate=liquid_rate_metric_to_si,
            gas_surface_rate=gas_rate_metric_to_si,
            reservoir_rate=res_rate_metric_to_si,
        ),
        (UnitSystem.METRIC, UnitSystem.LAB): UnitConversionFactors(
            pressure=bar_to_atm,
            length=m_to_cm,
            area=m_to_cm**2,
            volume=m3_to_cm3,
            time=_inverse(hours_per_day),  # day -> hr
            mass=mass_metric_to_lab,
            temperature=1.0,
            temperature_offset=0.0,  # °C -> °C
            density=kg_m3_to_g_cm3,
            viscosity=1.0,
            permeability=1.0,
            compressibility=_inverse(bar_to_atm),
            liquid_surface_volume=sm3_to_scc,  # Sm³ -> scc
            gas_surface_volume=sm3_to_scc,
            liquid_fvf=1.0,
            gas_fvf=1.0,
            gas_oil_ratio=gor_metric_to_lab,
            oil_gas_ratio=ogr_metric_to_lab,
            liquid_surface_rate=liquid_rate_metric_to_lab,
            gas_surface_rate=gas_rate_metric_to_lab,
            reservoir_rate=res_rate_metric_to_lab,
        ),
        ##############
        # SI -> *
        ##############
        (UnitSystem.SI, UnitSystem.FIELD): UnitConversionFactors(
            pressure=_inverse(psi_to_pa),
            length=m_to_ft,
            area=m_to_ft**2,
            volume=m3_to_ft3,
            time=seconds_per_day,  # s -> day
            mass=_inverse(mass_field_to_metric),
            temperature=9.0 / 5.0,
            temperature_offset=(-273.15 * 9.0 / 5.0) + 32.0,  # K -> °F
            density=_inverse(lbm_ft3_to_kg_m3),
            viscosity=_inverse(cp_to_pas),
            permeability=_inverse(md_to_m2),
            compressibility=psi_to_pa,
            liquid_surface_volume=_inverse(stb_to_m3),
            gas_surface_volume=_inverse(scf_to_m3),
            liquid_fvf=_inverse(liquid_fvf_field_to_si),
            gas_fvf=_inverse(gas_fvf_field_to_metric),
            gas_oil_ratio=gor_metric_to_field,
            oil_gas_ratio=ogr_metric_to_field,
            liquid_surface_rate=liquid_rate_si_to_field,
            gas_surface_rate=gas_rate_si_to_field,
            reservoir_rate=res_rate_si_to_field,
        ),
        (UnitSystem.SI, UnitSystem.METRIC): UnitConversionFactors(
            pressure=_inverse(bar_to_pa),
            length=1.0,
            area=1.0,
            volume=1.0,
            time=seconds_per_day,  # s -> day
            mass=1.0,
            temperature=1.0,
            temperature_offset=-273.15,  # K -> °C
            density=1.0,
            viscosity=_inverse(cp_to_pas),
            permeability=_inverse(md_to_m2),
            compressibility=bar_to_pa,
            liquid_surface_volume=1.0,
            gas_surface_volume=1.0,
            liquid_fvf=1.0,
            gas_fvf=1.0,
            gas_oil_ratio=1.0,
            oil_gas_ratio=1.0,
            liquid_surface_rate=liquid_rate_si_to_metric,
            gas_surface_rate=gas_rate_si_to_metric,
            reservoir_rate=res_rate_si_to_metric,
        ),
        (UnitSystem.SI, UnitSystem.LAB): UnitConversionFactors(
            pressure=_inverse(atm_to_pa),
            length=m_to_cm,
            area=m_to_cm**2,
            volume=m3_to_cm3,
            time=seconds_per_hour,  # s -> hr
            mass=mass_metric_to_lab,  # kg -> g
            temperature=1.0,
            temperature_offset=-273.15,  # K -> °C
            density=kg_m3_to_g_cm3,
            viscosity=_inverse(cp_to_pas),
            permeability=_inverse(md_to_m2),
            compressibility=atm_to_pa,
            liquid_surface_volume=sm3_to_scc,
            gas_surface_volume=sm3_to_scc,
            liquid_fvf=1.0,
            gas_fvf=1.0,
            gas_oil_ratio=gor_metric_to_lab,
            oil_gas_ratio=ogr_metric_to_lab,
            liquid_surface_rate=liquid_rate_si_to_lab,
            gas_surface_rate=gas_rate_si_to_lab,
            reservoir_rate=res_rate_si_to_lab,
        ),
        ##############
        # LAB -> *
        ##############
        (UnitSystem.LAB, UnitSystem.FIELD): UnitConversionFactors(
            pressure=_inverse(psi_to_atm),
            length=cm_to_ft,
            area=cm_to_ft**2,
            volume=cm3_to_ft3,
            time=hours_per_day,  # hr -> day
            mass=mass_lab_to_field,
            temperature=9.0 / 5.0,
            temperature_offset=32.0,  # °C -> °F
            density=_inverse(lbm_ft3_to_g_cm3),
            viscosity=1.0,
            permeability=1.0,
            compressibility=psi_to_atm,
            liquid_surface_volume=_inverse(stb_to_cm3),
            gas_surface_volume=_inverse(scf_to_cm3),
            liquid_fvf=_inverse(liquid_fvf_field_to_lab),
            gas_fvf=gas_fvf_lab_to_field,
            gas_oil_ratio=gor_lab_to_field,
            oil_gas_ratio=ogr_lab_to_field,
            liquid_surface_rate=liquid_rate_lab_to_field,
            gas_surface_rate=gas_rate_lab_to_field,
            reservoir_rate=res_rate_lab_to_field,
        ),
        (UnitSystem.LAB, UnitSystem.METRIC): UnitConversionFactors(
            pressure=_inverse(bar_to_atm),
            length=cm_to_m,
            area=cm_to_m**2,
            volume=cm3_to_m3,
            time=hours_per_day,  # hr -> day
            mass=mass_lab_to_metric,
            temperature=1.0,
            temperature_offset=0.0,  # °C -> °C
            density=_inverse(kg_m3_to_g_cm3),
            viscosity=1.0,
            permeability=1.0,
            compressibility=bar_to_atm,
            liquid_surface_volume=scc_to_sm3,
            gas_surface_volume=scc_to_sm3,
            liquid_fvf=1.0,
            gas_fvf=1.0,
            gas_oil_ratio=gor_lab_to_metric,
            oil_gas_ratio=ogr_lab_to_metric,
            liquid_surface_rate=liquid_rate_lab_to_metric,
            gas_surface_rate=gas_rate_lab_to_metric,
            reservoir_rate=res_rate_lab_to_metric,
        ),
        (UnitSystem.LAB, UnitSystem.SI): UnitConversionFactors(
            pressure=atm_to_pa,
            length=cm_to_m,
            area=cm_to_m**2,
            volume=cm3_to_m3,
            time=seconds_per_hour,  # hr -> s
            mass=mass_lab_to_metric,  # g -> kg
            temperature=1.0,
            temperature_offset=273.15,  # °C -> K
            density=_inverse(kg_m3_to_g_cm3),
            viscosity=cp_to_pas,
            permeability=md_to_m2,
            compressibility=_inverse(atm_to_pa),
            liquid_surface_volume=scc_to_sm3,
            gas_surface_volume=scc_to_sm3,
            liquid_fvf=1.0,
            gas_fvf=1.0,
            gas_oil_ratio=gor_lab_to_metric,
            oil_gas_ratio=ogr_lab_to_metric,
            liquid_surface_rate=liquid_rate_lab_to_si,
            gas_surface_rate=gas_rate_lab_to_si,
            reservoir_rate=res_rate_lab_to_si,
        ),
    }
    return table


IDENTITY_FACTORS = UnitConversionFactors(
    pressure=1.0,
    length=1.0,
    area=1.0,
    volume=1.0,
    time=1.0,
    mass=1.0,
    temperature=1.0,
    temperature_offset=0.0,
    density=1.0,
    viscosity=1.0,
    permeability=1.0,
    compressibility=1.0,
    liquid_surface_volume=1.0,
    gas_surface_volume=1.0,
    liquid_fvf=1.0,
    gas_fvf=1.0,
    gas_oil_ratio=1.0,
    oil_gas_ratio=1.0,
    liquid_surface_rate=1.0,
    gas_surface_rate=1.0,
    reservoir_rate=1.0,
)
"""Identity unit conversion factors. Has all multiplicative factors as 1.0, and offset 0.0."""

UNIT_CONVERSION_TABLE = build_unit_conversion_table()
"""Default unit conversion table"""


def get_conversion_factors(
    from_system: UnitSystem,
    to_system: UnitSystem,
    /,
    *,
    table: UnitConversionTable | None = None,
) -> UnitConversionFactors:
    """
    Return a dictionary of scalar conversion factors for every physical
    dimension used by the classes in this library.

    Each value converts a quantity expressed in `from_system` to
    `to_system` by **multiplication**, except temperature which uses an
    affine map stored as two keys:

    - "temperature"  - multiplicative factor.
    - "temperature_offset" - additive delta (in target units) applied
    *after* scaling: T_to = T_from * scale + offset.

    :param from_system: Source `UnitSystem`.
    :param to_system: Target `UnitSystem`.
    :returns: Conversion-factor dictionary.
    :raises KeyError: If the (from_system, to_system) pair is not defined.
    """
    if from_system == to_system:
        return IDENTITY_FACTORS

    table = table or build_unit_conversion_table()
    key = (from_system, to_system)
    if key not in table:
        pairs = [f"{a.value} -> {b.value}" for a, b in table]
        raise KeyError(
            f"No unit conversion defined from {from_system.value!r} "
            f"to {to_system.value!r}. Supported pairs: {pairs}."
        )
    return table[key]

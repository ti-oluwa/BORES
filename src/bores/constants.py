"""Physical constants and conversion factors"""

import typing
from contextvars import ContextVar

import attrs
from typing_extensions import Self

from bores.serialization import Serializable
from bores.stores import StoreSerializable

__all__ = ["Constant", "Constants", "ConstantsContext", "c", "get_constant"]


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

    description: typing.Optional[str] = None
    """Optional description of what this constant represents."""

    unit: typing.Optional[str] = None
    """Optional unit of measurement for this constant."""

    def __str__(self) -> str:
        """Return a human-readable string representation of the `Constant`."""
        return f"{self.value}{self.unit or ''}"

    def __repr__(self) -> str:
        """Return a string representation of the `Constant`."""
        parts = [f"value={self.value}"]
        if self.description:
            parts.append(f"description='{self.description}'")
        if self.unit:
            parts.append(f"unit='{self.unit}'")
        return f"{self.__class__.__name__}({', '.join(parts)})"


# Default constants dictionary
DEFAULT_CONSTANTS: typing.Dict[str, typing.Union[typing.Any, Constant]] = {
    # Standard Conditions
    "STANDARD_PRESSURE": Constant(
        value=101325, description="Standard atmospheric pressure (SI units)", unit="Pa"
    ),
    "STANDARD_PRESSURE_IMPERIAL": Constant(
        value=14.696,
        description="Standard atmospheric pressure (Imperial units)",
        unit="psi",
    ),
    "STANDARD_TEMPERATURE": Constant(
        value=288.7056, description="Standard temperature (15.6°C) (SI units)", unit="K"
    ),
    "STANDARD_TEMPERATURE_IMPERIAL": Constant(
        value=60.0, description="Standard temperature (Imperial units)", unit="°F"
    ),
    "STANDARD_TEMPERATURE_RANKINE": Constant(
        value=518.67, description="Standard temperature (15.6°C) in Rankine", unit="°R"
    ),
    "STANDARD_TEMPERATURE_CELSIUS": Constant(
        value=15.6, description="Standard temperature in Celsius", unit="°C"
    ),
    # Thermal and Compressibility Properties
    "OIL_THERMAL_EXPANSION_COEFFICIENT": Constant(
        value=9.7e-4, description="Thermal expansion coefficient for oil", unit="1/K"
    ),
    "WATER_THERMAL_EXPANSION_COEFFICIENT": Constant(
        value=3.0e-4, description="Thermal expansion coefficient for water", unit="1/K"
    ),
    "WATER_ISOTHERMAL_COMPRESSIBILITY": Constant(
        value=4.6e-10,
        description="Isothermal compressibility of water at 15.6°C (SI units)",
        unit="1/Pa",
    ),
    "OIL_THERMAL_EXPANSION_COEFFICIENT_IMPERIAL": Constant(
        value=5.39e-4,
        description="Thermal expansion coefficient for oil (Imperial units)",
        unit="1/°F",
    ),
    "WATER_THERMAL_EXPANSION_COEFFICIENT_IMPERIAL": Constant(
        value=1.67e-4,
        description="Thermal expansion coefficient for water (Imperial units)",
        unit="1/°F",
    ),
    "WATER_ISOTHERMAL_COMPRESSIBILITY_IMPERIAL": Constant(
        value=3.17e-6,
        description="Isothermal compressibility of water at 15.6°C (Imperial units)",
        unit="1/psi",
    ),
    # Standard Densities
    "STANDARD_WATER_DENSITY": Constant(
        value=998.2,
        description="Standard water density at 15.6°C (SI units)",
        unit="kg/m³",
    ),
    "STANDARD_WATER_DENSITY_IMPERIAL": Constant(
        value=62.37,
        description="Standard water density at 15.6°C (Imperial units)",
        unit="lb/ft³",
    ),
    "STANDARD_AIR_DENSITY": Constant(
        value=1.225,
        description="Standard air density at 15.6°C (SI units)",
        unit="kg/m³",
    ),
    "STANDARD_AIR_DENSITY_IMPERIAL": Constant(
        value=0.0765,
        description="Standard air density at 15.6°C (Imperial units)",
        unit="lb/ft³",
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
        description="Conversion factor from stock tank barrels to cubic feet",
        unit="ft³/STB",
    ),
    "CUBIC_FEET_TO_STB": Constant(
        value=1 / 5.614583,
        description="Conversion factor from cubic feet to stock tank barrels",
        unit="STB/ft³",
    ),
    "STB_TO_CUBIC_METER": Constant(
        value=0.158987,
        description="Conversion factor from stock tank barrels to cubic meters",
        unit="m³/STB",
    ),
    "CUBIC_METER_TO_STB": Constant(
        value=1 / 0.158987,
        description="Conversion factor from cubic meters to stock tank barrels",
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
    # Gas Constant
    "IDEAL_GAS_CONSTANT": Constant(
        value=8.31446261815324, description="Universal gas constant", unit="J/(mol·K)"
    ),
    "IDEAL_GAS_CONSTANT_SI": Constant(
        value=8.31446261815324e-3,
        description="Universal gas constant (SI units)",
        unit="kJ/(mol·K)",
    ),
    "IDEAL_GAS_CONSTANT_IMPERIAL": Constant(
        value=10.73159,
        description="Universal gas constant (Imperial units)",
        unit="ft³·psi/(lb·mol·°R)",
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
    "FT_TO_METERS": Constant(
        value=0.3048, description="Conversion factor from feet to meters", unit="m/ft"
    ),
    "METERS_TO_FT": Constant(
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
        value=0.001127,
        description="Conversion factor from mD/cP to ft²/(psi·day)",
        unit="(ft²/(psi·day))/(mD/cP)",
    ),
    "MILLIDARCIES_PER_CENTIPOISE_TO_SQUARE_FEET_PER_PSI_PER_SECOND": Constant(
        value=0.001127 / 86400.0,
        description="Conversion factor from mD/cP to ft²/(psi·s)",
        unit="(ft²/(psi·s))/(mD/cP)",
    ),
    "MILLIDARCIES_FT_PER_CENTIPOISE_TO_CUBIC_FEET_PER_PSI_PER_DAY": Constant(
        value=0.001127,
        unit="(ft³/(psi·day))/(mD·ft/cP)",
        description="Conversion factor from mD·ft/cP to ft³/(psi·day)",
    ),
    # Gravity
    "ACCELERATION_DUE_TO_GRAVITY_METER_PER_SECONDS_SQUARE": Constant(
        value=9.80665,
        description="Standard acceleration due to gravity in m/s²",
        unit="m/s²",
    ),
    "ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE": Constant(
        value=32.174,
        description="Standard acceleration due to gravity in ft/s²",
        unit="ft/s²",
    ),
    "ACCELERATION_DUE_TO_GRAVITY_FEET_PER_DAY_SQUARE": Constant(
        value=32.174 * 86400.0**2,
        description="Standard acceleration due to gravity in ft/day²",
        unit="ft/day²",
    ),
    "GRAVITATIONAL_CONSTANT_LBM_FT_PER_LBF_S2": Constant(
        value=32.174,
        description="Gravitational constant in lbm·ft/(lbf·s²). Conversion factor from pound-force to pound-mass or vice versa under Earth's gravity.",
        unit="lbm·ft/(lbf·s²)",
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
    "SATURATION_EPSILON": Constant(
        value=1e-12,
        description="Small epsilon value to prevent numerical issues with saturations at 0 or 1",
        unit="fraction",
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
}


@typing.final
class Constants(
    StoreSerializable,
    fields={"_store": typing.Dict[str, Constant]},
):
    """
    Physical constants and conversion factors used in reservoir simulations.

    All constants are stored in an internal dictionary and can be accessed via dot notation.
    Constants can be modified at runtime if needed. Use __getattr__ for value access and
    __getitem__ for `Constant` object access.
    """

    __slots__ = ("_store",)

    def __new__(cls, *args, **kwargs) -> "Constants":
        instance = super().__new__(cls)
        instance._store = {}
        return instance

    def __init__(
        self,
        defaults: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ) -> None:
        """
        Initialize the constants store with default values.

        :param defaults: Optional dictionary of default constants to initialize with.
            If None, uses the predefined `DEFAULT_CONSTANTS`.
        """
        defaults = (
            {**DEFAULT_CONSTANTS, **defaults}
            if defaults is not None
            else DEFAULT_CONSTANTS
        )
        for name, value in defaults.items():
            if isinstance(value, Constant):
                self._store[name] = value
            else:
                # Wrap raw values in Constant objects
                self._store[name] = Constant(value=value)

    def __getattr__(self, name: str) -> typing.Any:
        """Get a constant's value using dot notation.

        :param name: Name of the constant
        :return: Value of the constant (unwrapped from Constant object)
        :raises AttributeError: If the constant does not exist
        """
        if name.startswith("_"):
            # Allow access to private attributes normally
            return object.__getattribute__(self, name)

        try:
            constant = self._store[name]
            return constant.value if isinstance(constant, Constant) else constant
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None

    def __getitem__(self, name: str) -> Constant:
        """Get the Constant object (with metadata) using bracket notation.

        :param name: Name of the constant
        :return: Constant object with value, description, and unit
        :raises KeyError: If the constant does not exist
        """
        return self._store[name]

    def __setattr__(self, name: str, value: typing.Union[typing.Any, Constant]) -> None:
        """Set a constant value using dot notation.

        Accepts either a raw value (which will be wrapped in a Constant) or a Constant object.

        :param name: Name of the constant
        :param value: Value to set (raw value or Constant object)
        """
        if name.startswith("_"):
            # Allow setting private attributes normally
            object.__setattr__(self, name, value)
        else:
            if isinstance(value, Constant):
                self._store[name] = value
            else:
                # Wrap raw values in `Constant` objects
                self._store[name] = Constant(value=value)

    def __setitem__(self, name: str, value: typing.Union[typing.Any, Constant]) -> None:
        """Set a constant using bracket notation.

        :param name: Name of the constant
        :param value: Value to set (raw value or Constant object)
        """
        if isinstance(value, Constant):
            self._store[name] = value
        else:
            # Wrap raw values in `Constant` objects
            self._store[name] = Constant(value=value)

    def __delattr__(self, name: str) -> None:
        """Delete a constant from the store.

        :param name: Name of the constant to delete
        :raises AttributeError: If the constant does not exist
        """
        if name.startswith("_"):
            object.__delattr__(self, name)
        else:
            try:
                del self._store[name]
            except KeyError:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                ) from None

    def __delitem__(self, name: str) -> None:
        """Delete a constant using bracket notation.

        :param name: Name of the constant to delete
        :raises KeyError: If the constant does not exist
        """
        del self._store[name]

    def __contains__(self, name: str) -> bool:
        """Check if a constant exists.

        :param name: Name of the constant
        :return: True if the constant exists, False otherwise
        """
        return name in self._store

    def __iter__(self) -> typing.Iterator[str]:
        """Iterate over constant names.

        :return: Iterator over constant names
        """
        return iter(self._store)

    def keys(self) -> typing.KeysView[str]:
        """Get all constant names.

        :return: View of all constant names
        """
        return self._store.keys()

    def values(self) -> typing.ValuesView[Constant]:
        """Get all Constant objects.

        :return: View of all Constant objects
        """
        return self._store.values()

    def items(self) -> typing.ItemsView[str, Constant]:
        """Get all constant name-Constant object pairs.

        :return: View of all constant name-Constant pairs
        """
        return self._store.items()

    def get(self, name: str, default: typing.Any = None) -> typing.Any:
        """Get a constant's value with a default fallback.

        :param name: Name of the constant
        :param default: Default value if constant doesn't exist
        :return: Value of the constant or default
        """
        constant = self._store.get(name)
        if constant is None:
            return default
        return constant.value if isinstance(constant, Constant) else constant

    def get_constant(
        self, name: str, default: typing.Optional[Constant] = None
    ) -> typing.Optional[Constant]:
        """Get a `Constant` object with a default fallback.

        :param name: Name of the constant
        :param default: Default `Constant` if constant doesn't exist
        :return: `Constant` object or default
        """
        return self._store.get(name, default)

    def __dir__(self):
        default = super().__dir__()
        constants = list(self.keys())
        return sorted({*default, *constants})

    def _ipython_key_completions_(self) -> typing.List[str]:
        return sorted(self.keys())

    def __repr__(self) -> str:
        """Return a string representation of the Constants object."""
        return f"{type(self).__name__}(constants={len(self._store)})"

    def __len__(self) -> int:
        """Return the number of constants in the store."""
        return len(self._store)

    def __call__(self) -> "ConstantsContext":
        """
        Create a context manager that within its context, temporarily overrides/set the default constants
        accessed through the global constants proxy `bores.c`  to this `Constants` instance.

        :return: `ConstantsContext` for temporary overrides
        """
        return ConstantsContext(self)

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        """Dump constants to dict."""
        return {
            name: const.dump(recurse) if isinstance(const, Constant) else const
            for name, const in self._store.items()
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        """Load constants from dict."""
        store = {
            name: Constant.load(val)
            if isinstance(val, dict) and "value" in val
            else Constant(value=val)
            for name, val in data.items()
        }
        constants = cls(defaults={})  # type: ignore[arg-type]
        constants._store = store
        return constants


_constants_context: ContextVar[Constants] = ContextVar(
    "constants_context", default=Constants()
)


class ConstantsContext:
    """
    Context manager for temporary global `Constants` overrides.

    This context manager allows for temporary overrides of the global `Constants`
    instance within a specific context. Upon exiting the context, the previous
    `Constants` instance is restored.
    """

    def __init__(self, constants: Constants) -> None:
        """Initialize the context manager with a new Constants instance.

        :param constants: New Constants instance to use within the context
        """
        self._new_constants = constants
        self._token = None

    def __enter__(self) -> Constants:
        """Enter the context, setting the new Constants instance.

        :return: The new Constants instance
        """
        self._token = _constants_context.set(self._new_constants)  # type: ignore[assignment]
        return self._new_constants

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context, restoring the previous Constants instance."""
        if self._token is not None:
            _constants_context.reset(self._token)


class _ConstantsProxy:
    """
    Proxy class to access the current context's `Constants` instance.

    Use the `constants` property to get the current `Constants` instance.

    Override the current `Constants` instance using the `ConstantsContext` context manager.
    """

    @property
    def _constants(self) -> Constants:
        """Get the current context's `Constants` instance.

        :return: Current `Constants` instance
        """
        return _constants_context.get()

    def __getattr__(self, name: str) -> typing.Any:
        """Get a constant's value from the current context's `Constants` instance.

        :param name: Name of the constant
        :return: Value of the constant
        :raises AttributeError: If the constant does not exist
        """
        return getattr(self._constants, name)

    def __getitem__(self, name: str) -> Constant:
        """Get a Constant object from the current context's `Constants` instance.

        :param name: Name of the constant
        :return: Constant object
        :raises KeyError: If the constant does not exist
        """
        return self._constants[name]

    def __dir__(self):
        default = super().__dir__()
        return sorted({*default, *self._constants.__dir__()})

    def _ipython_key_completions_(self) -> typing.List[str]:
        return self._constants._ipython_key_completions_()


c = _ConstantsProxy()
"""Global proxy to access physical constants and conversion factors."""


def get_constant(name: str) -> typing.Optional[Constant]:
    """Get a `Constant` object by name from the global constants.

    :param name: Name of the constant
    :return: `Constant` object or None if not found
    """
    return c._constants.get_constant(name)

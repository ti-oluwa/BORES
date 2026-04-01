# Constants

## Overview

BORES uses a system of physical constants and unit conversion factors throughout its calculations. These constants cover standard conditions (pressure, temperature), fluid densities, molecular weights, conversion factors between SI and imperial units, and numerical thresholds that control simulation behavior. Rather than scattering magic numbers throughout the code, all constants are centralized in a `Constants` class that you can inspect, modify, and even temporarily override during a simulation.

The constants system is designed around the reservoir engineering convention of using imperial (oilfield) units internally: pressures in psi, temperatures in Fahrenheit, permeabilities in millidarcies, viscosities in centipoise, and volumes in barrels and cubic feet. Conversion factors to SI units are provided for every quantity so you can work in whichever system you prefer for input and output.

Understanding the constants is important for two reasons. First, if your reservoir uses non-standard conditions (for example, a different standard temperature or salinity), you can adjust the constants to match. Second, when debugging unexpected results, checking whether the correct constants are being used can quickly identify unit conversion errors.

---

## The Global Constants Proxy

BORES provides a global constants proxy `c` that gives direct access to all default constant values. You can import and use it anywhere:

```python
from bores.constants import c

# Access constant values with dot notation
print(c.STANDARD_PRESSURE_IMPERIAL)       # 14.696
print(c.STANDARD_TEMPERATURE_IMPERIAL)    # 60.0
print(c.MOLECULAR_WEIGHT_CO2)             # 44.01
print(c.BARRELS_TO_CUBIC_FEET)            # 5.614583
```

The proxy automatically returns the unwrapped value of each constant. If you need the full `Constant` object (with description and unit metadata), use bracket notation:

```python
from bores.constants import c

const = c["STANDARD_PRESSURE_IMPERIAL"]
print(const.value)        # 14.696
print(const.description)  # "Standard atmospheric pressure (Imperial units)"
print(const.unit)         # "psi"
```

You can also use `get_constant()` for safe access with a default:

```python
from bores.constants import get_constant

const = get_constant("STANDARD_PRESSURE_IMPERIAL")
if const is not None:
    print(f"{const.description}: {const.value} {const.unit}")
```

---

## Available Constants

### Standard Conditions

| Constant | Value | Unit | Description |
|---|---|---|---|
| `STANDARD_PRESSURE` | 101325 | Pa | Standard atmospheric pressure (SI) |
| `STANDARD_PRESSURE_IMPERIAL` | 14.696 | psi | Standard atmospheric pressure (Imperial) |
| `STANDARD_TEMPERATURE` | 288.7056 | K | Standard temperature 15.6C (SI) |
| `STANDARD_TEMPERATURE_IMPERIAL` | 60.0 | F | Standard temperature (Imperial) |
| `STANDARD_TEMPERATURE_RANKINE` | 518.67 | R | Standard temperature (Rankine) |
| `STANDARD_TEMPERATURE_CELSIUS` | 15.6 | C | Standard temperature (Celsius) |

### Standard Densities

| Constant | Value | Unit | Description |
|---|---|---|---|
| `STANDARD_WATER_DENSITY` | 998.2 | kg/m³ | Water density at standard conditions (SI) |
| `STANDARD_WATER_DENSITY_IMPERIAL` | 62.37 | lb/ft³ | Water density at standard conditions (Imperial) |
| `STANDARD_AIR_DENSITY` | 1.225 | kg/m³ | Air density at standard conditions (SI) |
| `STANDARD_AIR_DENSITY_IMPERIAL` | 0.0765 | lb/ft³ | Air density at standard conditions (Imperial) |

### Molecular Weights

| Constant | Value | Unit | Description |
|---|---|---|---|
| `MOLECULAR_WEIGHT_WATER` | 18.015 | g/mol | Water (H2O) |
| `MOLECULAR_WEIGHT_CO2` | 44.01 | g/mol | Carbon dioxide |
| `MOLECULAR_WEIGHT_N2` | 28.013 | g/mol | Nitrogen |
| `MOLECULAR_WEIGHT_CH4` | 16.042 | g/mol | Methane |
| `MOLECULAR_WEIGHT_AIR` | 28.964 | g/mol | Air |
| `MOLECULAR_WEIGHT_H2` | 2.016 | g/mol | Hydrogen |
| `MOLECULAR_WEIGHT_O2` | 31.999 | g/mol | Oxygen |
| `MOLECULAR_WEIGHT_NACL` | 58.44 | g/mol | Sodium chloride |
| `MOLECULAR_WEIGHT_HELIUM` | 4.003 | g/mol | Helium |
| `MOLECULAR_WEIGHT_ARGON` | 39.948 | g/mol | Argon |

### Thermal and Compressibility Properties

| Constant | Value | Unit | Description |
|---|---|---|---|
| `OIL_THERMAL_EXPANSION_COEFFICIENT` | 9.7e-4 | K⁻¹ | Oil thermal expansion (SI) |
| `OIL_THERMAL_EXPANSION_COEFFICIENT_IMPERIAL` | 5.39e-4 | 1/F | Oil thermal expansion (Imperial) |
| `WATER_THERMAL_EXPANSION_COEFFICIENT` | 3.0e-4 | K⁻¹ | Water thermal expansion (SI) |
| `WATER_THERMAL_EXPANSION_COEFFICIENT_IMPERIAL` | 1.67e-4 | 1/F | Water thermal expansion (Imperial) |
| `WATER_ISOTHERMAL_COMPRESSIBILITY` | 4.6e-10 | Pa⁻¹ | Water compressibility (SI) |
| `WATER_ISOTHERMAL_COMPRESSIBILITY_IMPERIAL` | 3.17e-6 | psi⁻¹ | Water compressibility (Imperial) |

### Gas Constant

| Constant | Value | Unit | Description |
|---|---|---|---|
| `IDEAL_GAS_CONSTANT` | 8.314 | J/(mol K) | Universal gas constant |
| `IDEAL_GAS_CONSTANT_SI` | 8.314e-3 | kJ/(mol K) | Gas constant (SI, kJ) |
| `IDEAL_GAS_CONSTANT_IMPERIAL` | 10.732 | ft³ psi/(lb mol R) | Gas constant (Imperial) |

### Pressure Conversions

| Constant | Value | Description |
|---|---|---|
| `PSI_TO_PASCAL` | 6894.757 | psi to Pa |
| `PASCAL_TO_PSI` | 1.450e-4 | Pa to psi |
| `PSI_TO_BAR` | 0.06895 | psi to bar |

### Temperature Conversions

| Constant | Value | Description |
|---|---|---|
| `RANKINE_TO_KELVIN` | 0.5556 | Rankine to Kelvin |
| `KELVIN_TO_RANKINE` | 1.8 | Kelvin to Rankine |

### Viscosity Conversions

| Constant | Value | Description |
|---|---|---|
| `CENTIPOISE_TO_PASCAL_SECONDS` | 0.001 | cP to Pa s |
| `PASCAL_SECONDS_TO_CENTIPOISE` | 1000 | Pa s to cP |

### Permeability Conversions

| Constant | Value | Description |
|---|---|---|
| `MILLIDARCY_TO_SQUARE_METER` | 9.869e-16 | mD to m² |

### Volume Conversions

| Constant | Value | Description |
|---|---|---|
| `BARRELS_TO_CUBIC_FEET` | 5.6146 | BBL to ft³ |
| `CUBIC_FEET_TO_BARRELS` | 0.1781 | ft³ to BBL |
| `STB_TO_CUBIC_FEET` | 5.6146 | STB to ft³ |
| `STB_TO_CUBIC_METER` | 0.1590 | STB to m³ |
| `CUBIC_METER_TO_STB` | 6.2898 | m³ to STB |
| `SCF_TO_BARRELS` | 0.1781 | SCF to BBL |
| `CUBIC_METER_TO_SCF` | 35.315 | m³ to SCF |
| `SCF_TO_SCM` | 0.02832 | SCF to m³ |
| `ACRE_FOOT_TO_CUBIC_FEET` | 43560 | acre-ft to ft³ |
| `ACRE_FOOT_TO_BARRELS` | 7758 | acre-ft to BBL |
| `ACRES_TO_SQUARE_FEET` | 43560 | acres to ft² |

### Gas-Oil Ratio (GOR) Conversions

| Constant | Value | Description |
|---|---|---|
| `SCF_PER_STB_TO_CUBIC_METER_PER_CUBIC_METER` | 0.1781 | scf/STB to m³/m³ |
| `CUBIC_METER_PER_CUBIC_METER_TO_SCF_PER_STB` | 5.6146 | m³/m³ to scf/STB |

### Formation Volume Factor (FVF) Conversions

| Constant | Value | Description |
|---|---|---|
| `CUBIC_METER_PER_CUBIC_METER_TO_BARRELS_PER_SCF` | 5.6146 | m³/m³ to BBL/scf |
| `BARRELS_PER_SCF_TO_CUBIC_METER_PER_CUBIC_METER` | 0.1781 | BBL/scf to m³/m³ |
| `CUBIC_METER_PER_CUBIC_METER_TO_BARRELS_PER_STB` | 1.0 | m³/m³ to BBL/STB |
| `BARRELS_PER_STB_TO_CUBIC_METER_PER_CUBIC_METER` | 1.0 | BBL/STB to m³/m³ |

### Density Conversions

| Constant | Value | Description |
|---|---|---|
| `POUNDS_PER_CUBIC_FEET_TO_KILOGRAM_PER_CUBIC_METER` | 16.0185 | lb/ft³ to kg/m³ |
| `KILOGRAM_PER_CUBIC_METER_TO_POUNDS_PER_CUBIC_FEET` | 0.0624279 | kg/m³ to lb/ft³ |
| `POUNDS_PER_CUBIC_FEET_TO_GRAMS_PER_CUBIC_METER` | 0.01602 | lb/ft³ to g/cm³ |

### Concentration/Salinity Conversions

| Constant | Value | Description |
|---|---|---|
| `PPM_TO_GRAMS_PER_LITER` | 0.001 | ppm to g/L |
| `GRAMS_PER_LITER_TO_PPM` | 1000 | g/L to ppm |
| `PPM_TO_WEIGHT_FRACTION` | 1e-6 | ppm to weight fraction |
| `PPM_TO_WEIGHT_PERCENT` | 1e-4 | ppm to weight percent |
| `WEIGHT_PERCENT_TO_PPM` | 10000 | weight percent to ppm |

### Interfacial Tension Conversions

| Constant | Value | Description |
|---|---|---|
| `DYNE_PER_CENTIMETER_TO_PSI` | 4.621 | dyne/cm to psi |

### Molar Volume

| Constant | Value | Description |
|---|---|---|
| `SCF_PER_POUND_MOLE` | 379.49 | scf/(lb·mol) |

### Length Conversions

| Constant | Value | Description |
|---|---|---|
| `FT_TO_METERS` | 0.3048 | ft to m |
| `METERS_TO_FT` | 3.2808 | m to ft |
| `INCHES_TO_METERS` | 0.0254 | in to m |
| `METERS_TO_INCHES` | 39.3701 | m to in |

### Time Conversions

| Constant | Value | Description |
|---|---|---|
| `SECONDS_PER_DAY` | 86400 | Seconds in a day |
| `DAYS_PER_YEAR` | 365.25 | Days in a year |
| `MONTHS_PER_YEAR` | 12 | Months in a year |
| `SECONDS_PER_YEAR` | 31557600 | Seconds in a year |
| `DAYS_PER_SECOND` | 1.157e-5 | Days in a second |

### Flow Rate Conversions

| Constant | Value | Description |
|---|---|---|
| `CUBIC_METER_PER_SECOND_TO_STB_PER_DAY` | 543168.4 | m³/s to STB/day |
| `STB_PER_DAY_TO_CUBIC_METER_PER_SECOND` | 1.841e-6 | STB/day to m³/s |
| `CUBIC_METER_PER_SECOND_TO_SCF_PER_DAY` | 3049493 | m³/s to scf/day |
| `SCF_PER_DAY_TO_CUBIC_METER_PER_SECOND` | 3.278e-7 | scf/day to m³/s |

### Transmissibility Conversions

| Constant | Value | Description |
|---|---|---|
| `MILLIDARCIES_PER_CENTIPOISE_TO_SQUARE_FEET_PER_PSI_PER_DAY` | 0.001127 | mD/cP to ft²/(psi·day) |
| `MILLIDARCIES_PER_CENTIPOISE_TO_SQUARE_FEET_PER_PSI_PER_SECOND` | 1.305e-8 | mD/cP to ft²/(psi·s) |
| `MILLIDARCIES_FT_PER_CENTIPOISE_TO_CUBIC_FEET_PER_PSI_PER_DAY` | 0.001127 | (mD·ft)/cP to ft³/(psi·day) |

### Gravity

| Constant | Value | Unit | Description |
|---|---|---|---|
| `ACCELERATION_DUE_TO_GRAVITY_METER_PER_SECONDS_SQUARE` | 9.807 | m/s² | Standard gravity (SI) |
| `ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE` | 32.174 | ft/s² | Standard gravity (Imperial) |
| `GRAVITATIONAL_CONSTANT_LBM_FT_PER_LBF_S2` | 32.174 | lbm ft/(lbf s²) | gc conversion factor |

### Numerical Thresholds

| Constant | Value | Description |
|---|---|---|
| `SATURATION_EPSILON` | 1e-12 | Prevents numerical issues at S=0 or S=1 |
| `MINIMUM_TRANSMISSIBILITY_FACTOR` | 1e-12 | Floor for transmissibility values |
| `GAS_SOLUBILITY_TOLERANCE` | 1e-6 | Tolerance for gas solubility calculations |
| `GAS_PSEUDO_PRESSURE_THRESHOLD` | 0.0 | Pressure above which pseudo-pressure is used |
| `GAS_PSEUDO_PRESSURE_POINTS` | 200 | Points in pseudo-pressure table |
| `DEFAULT_WATER_SALINITY_PPM` | 0.0 | Default water salinity (ppm NaCl) |
| `MIN_OIL_ZONE_THICKNESS` | 5 | Minimum oil zone thickness warning threshold (ft) |
| `FLUID_INCOMPRESSIBILITY_THRESHOLD` | 1e-6 | Minimum fluid compressibility below which the fluid should be considered incompressible |

### Reservoir Fluid Defaults

| Constant | Value | Description |
|---|---|---|
| `RESERVOIR_GAS` | `"Methane"` | Default gas that exists with oil in the reservoir (CoolProp compatible) |

### Valid Ranges

| Constant | Value | Unit | Description |
|---|---|---|---|
| `MINIMUM_VALID_PRESSURE` | 14.5 | psi | Floor for reservoir pressures |
| `MAXIMUM_VALID_PRESSURE` | 14700 | psi | Ceiling for reservoir pressures |
| `MINIMUM_VALID_TEMPERATURE` | 32 | F | Floor for reservoir temperatures |
| `MAXIMUM_VALID_TEMPERATURE` | 482 | F | Ceiling for reservoir temperatures |

---

## The Constants Class

The `Constants` class is a dictionary-like container that stores all constants as `Constant` objects. Each `Constant` wraps a value with optional description and unit metadata.

```python
from bores.constants import Constants, Constant

# Create with default constants
constants = Constants()

# Access a value
print(constants.STANDARD_PRESSURE_IMPERIAL)  # 14.696

# Access the full Constant object
const = constants["STANDARD_PRESSURE_IMPERIAL"]
print(const)  # Constant(value=14.696, description='...', unit='psi')
```

### Modifying Constants

You can modify constants at runtime using dot notation or bracket notation:

```python
from bores.constants import Constants, Constant

constants = Constants()

# Set a raw value (auto-wrapped in Constant)
constants.DEFAULT_WATER_SALINITY_PPM = 50000

# Set with full metadata
constants["MY_CUSTOM_VALUE"] = Constant(
    value=42.0,
    description="Custom parameter for my model",
    unit="psi",
)
```

### Using Custom Constants in a Simulation

The `Config` accepts a `Constants` instance, allowing you to customize constants for a specific simulation:

```python
from bores.constants import Constants

# Create custom constants
custom = Constants()
custom.DEFAULT_WATER_SALINITY_PPM = 100000  # Very saline formation water

config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    constants=custom,
)
```

---

## Temporary Constants Override

For temporary overrides, use the `Constants` instance as a context manager. Within the context, the global proxy `c` points to your custom constants. Outside, the defaults are restored:

```python
from bores.constants import Constants, c

# Create custom constants
custom = Constants()
custom.STANDARD_TEMPERATURE_IMPERIAL = 70.0  # Different standard temp

# Temporarily override global constants
with custom():
    print(c.STANDARD_TEMPERATURE_IMPERIAL)  # 70.0
    # All BORES functions called here use 70 F as standard temperature

# Outside the context, original defaults are restored
print(c.STANDARD_TEMPERATURE_IMPERIAL)  # 60.0
```

This mechanism uses Python's `ContextVar` system and is thread-safe. Each thread maintains its own constants context, so overrides in one thread do not affect other threads.

---

## Iterating Over Constants

The `Constants` class supports iteration, length, and containment checks:

```python
from bores.constants import Constants

constants = Constants()

# Count all constants
print(len(constants))  # Number of defined constants

# Check if a constant exists
if "MOLECULAR_WEIGHT_CO2" in constants:
    print("CO2 molecular weight is defined")

# Iterate over all constant names
for name in constants:
    print(name)

# Iterate over name-value pairs
for name, const in constants.items():
    print(f"{name}: {const.value} {const.unit or ''}")
```

---

## Creating Custom Constants

You can define your own `Constant` objects for project-specific parameters:

```python
from bores.constants import Constant, Constants

# Create a Constants instance with custom values
project_constants = Constants()
project_constants["FORMATION_WATER_VISCOSITY"] = Constant(
    value=0.7,
    description="Measured formation water viscosity at reservoir conditions",
    unit="cP",
)
project_constants["INJECTION_WATER_VISCOSITY"] = Constant(
    value=0.5,
    description="Treated injection water viscosity",
    unit="cP",
)
```

Custom constants coexist with the default constants. You can use them to store project-specific parameters alongside the standard physical constants, keeping all numerical values in one inspectable location.

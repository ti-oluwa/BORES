# Correlations

## Overview

`bores.correlations` holds the empirical correlations BORES uses for the PVT properties of oil, gas and water. Every time the simulator needs an oil formation volume factor, a gas compressibility factor or a water viscosity, it calls one of these functions. You can call them directly for standalone PVT work or to check against laboratory data.

All pressures are in psi, temperatures in °F, salinity in ppm by weight, gas-oil ratios in scf/STB and viscosities in cP unless a function says otherwise.

```python
from bores import correlations

Bo = correlations.compute_oil_formation_volume_factor_vazquez_and_beggs(
    temperature=200.0,
    oil_specific_gravity=0.85,
    gas_gravity=0.75,
    gas_to_oil_ratio=600.0,
)
```

---

## One function for numbers and arrays

Every correlation is written once, for a single set of conditions, and works on numbers and arrays alike. Pass numbers and you get a number back. Pass arrays (any shape, broadcast against each other and against plain numbers) and you get an array.

```python
import numpy as np
from bores import correlations

pressure = np.linspace(500.0, 5000.0, 10)

Z = correlations.compute_gas_compressibility_factor(
    pressure=pressure,
    temperature=200.0,
    gas_gravity=0.7,
)
```

Arguments can be passed by keyword or by position. Float32 arrays give float32 results. Optional arguments (for example `salinity`) can be left out.

Argument ranges are checked once per call, on the whole input, and a `ValidationError` names the argument that is out of range. A correlation can be called from inside your own `numba.njit` code with positional arguments.

The compiled code is cached on disk, so only the first call in a fresh environment pays the compile cost.

!!! note "Constants"
    The correlation modules read the physical constants from `bores.constants` when they are imported.

---

## Core

Unit conversions, CoolProp access and volumetric helpers.

| Function | Description |
| --- | --- |
| `fahrenheit_to_kelvin`, `fahrenheit_to_celsius`, `fahrenheit_to_rankine`, `kelvin_to_fahrenheit` | Temperature conversions |
| `compute_fluid_density`, `compute_fluid_viscosity`, `compute_fluid_compressibility`, `compute_fluid_compressibility_factor` | Equation of state properties through CoolProp, at psi and °F. Needs `bores-framework[coolprop]` |
| `compute_total_fluid_compressibility` | Saturation-weighted sum of the phase compressibilities |
| `compute_hydrocarbon_in_place` | Volumetric oil, free gas or water in place |
| `validate_input_pressure`, `validate_input_temperature` | Range checks used when building tables |
| `get_gas_symbol` | Normalizes a gas name such as "carbon dioxide" to "co2" |

---

## Gas

| Function | Description |
| --- | --- |
| `compute_gas_pseudocritical_properties` | Sutton pseudocritical pressure and temperature, Wichert-Aziz correction for H₂S and CO₂, Carr-Kobayashi-Burrows correction for N₂. One set of conditions only |
| `compute_gas_pseudocritical_pressure`, `compute_gas_pseudocritical_temperature` | The same, for numbers and arrays |
| `compute_gas_compressibility_factor` | Z factor with a chosen method: `"dak"` (default), `"hall-yarborough"` or `"papay"` |
| `compute_gas_compressibility_factor_dranchuk_abou_kassem` | Dranchuk and Abou-Kassem, solved with a safeguarded Newton method |
| `compute_gas_compressibility_factor_hall_yarborough` | Hall and Yarborough, solved with a safeguarded Newton method |
| `compute_gas_compressibility_factor_papay` | Papay, explicit and fast, within about 6% up to a reduced pressure of 8 |
| `compute_gas_formation_volume_factor` | Bg in ft³/scf |
| `compute_gas_density`, `compute_gas_molecular_weight` | Real gas density and apparent molecular weight |
| `compute_gas_viscosity` | Lee, Gonzalez and Eakin |
| `compute_gas_compressibility` | Isothermal compressibility, `1/P - (1/Z) dZ/dP` |
| `compute_gas_gravity`, `compute_gas_gravity_from_density` | Gravity of a named gas, or from a density, through CoolProp |

---

## Oil

| Function | Description |
| --- | --- |
| `compute_oil_api_gravity`, `compute_oil_specific_gravity` | API gravity, and stock tank specific gravity from a reservoir density |
| `compute_oil_bubble_point_pressure` | Vazquez and Beggs bubble point pressure |
| `estimate_bubble_point_pressure_standing` | Standing bubble point pressure, the inverse of `compute_gas_to_oil_ratio_standing` |
| `compute_gas_to_oil_ratio` | Vazquez and Beggs Rs, held at its bubble point value above the bubble point |
| `compute_gas_to_oil_ratio_standing` | Standing Rs |
| `estimate_solution_gor` | Saturated Rs at a pressure, for when the bubble point pressure is not known |
| `compute_oil_formation_volume_factor` | Bo at any pressure: Standing up to 100 °F, Vazquez and Beggs above it, shrunk above the bubble point |
| `compute_oil_formation_volume_factor_standing`, `compute_oil_formation_volume_factor_vazquez_and_beggs` | The saturated Bo correlations |
| `correct_oil_fvf_for_pressure` | Undersaturated shrinkage of Bo |
| `compute_oil_viscosity` | Dead (Egbogah and Ng), saturated (Beggs and Robinson) and undersaturated (Vazquez and Beggs) viscosity |
| `compute_dead_oil_viscosity_modified_beggs` | Egbogah and Ng dead oil viscosity |
| `compute_oil_compressibility` | Compressibility at any pressure, with the gas liberation term below the bubble point |
| `compute_base_compressibility`, `compute_oil_compressibility_liberation_correction_term` | The two parts of the oil compressibility |
| `compute_live_oil_density` | Mass balance density of live oil |

### Miscible flooding

| Function | Description |
| --- | --- |
| `compute_miscibility_transition_factor`, `compute_effective_todd_longstaff_omega` | Smooth tanh transition of the mixing parameter around the minimum miscibility pressure |
| `compute_todd_longstaff_mixture_viscosity` | Fully mixed viscosity with the quarter-power rule |
| `compute_todd_longstaff_effective_oil_viscosity`, `compute_todd_longstaff_effective_solvent_viscosity` | Effective viscosity of each component, `μ^(1-ω) μ_m^ω` |
| `compute_todd_longstaff_effective_oil_density`, `compute_todd_longstaff_effective_solvent_density` | Effective density of each component |

---

## Water

| Function | Description |
| --- | --- |
| `compute_water_viscosity` | McCain brine viscosity |
| `compute_gas_free_water_formation_volume_factor` | McCain Bw |
| `compute_water_compressibility` | McCain compressibility with the dissolved gas and salinity corrections, and the gas liberation term for saturated water |
| `compute_water_standard_density` | Brine density at standard conditions |
| `compute_water_density_mccain`, `compute_water_density_batzle` | Gas-free brine density |
| `compute_water_density` | Density of water holding dissolved gas |
| `compute_water_formation_volume_factor` | Bw from the density |
| `compute_gas_solubility_in_water` | Solubility of CH₄, CO₂, N₂, O₂, Ar, He or H₂ in brine |
| `compute_gas_solubility_in_water_mccain_methane`, `compute_gas_solubility_in_water_henry_law` | The two solubility models behind it |
| `compute_gas_solubility_in_water_pressure_derivative` | `dRsw/dP` |
| `compute_water_bubble_point_pressure` | Pressure at which the water holds a given dissolved gas |

!!! warning "Limits of the solubility models"
    Methane between 100 and 400 °F uses the McCain correlation. Every other case uses Henry's law with an ideal gas, which overestimates solubility at high pressure. For CO₂ near 3000 psi and 200 °F the overestimate is about 25%, and the solubility never saturates.

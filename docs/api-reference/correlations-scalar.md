# Scalar PVT Correlations

## Overview

The `bores.correlations.scalars` module contains scalar PVT correlation functions. Each function accepts single float values as input and returns a single float result. These are the building blocks that the simulator calls internally when computing fluid properties cell by cell, and you can also call them directly in your own scripts for point calculations, unit conversions, validation against laboratory data, or building custom PVT tables.

All functions in this module use oilfield units: pressure in psi, temperature in degrees Fahrenheit, density in lbm/ft³, viscosity in centipoise, and volume factors in bbl/STB or ft³/SCF. The module also provides generic CoolProp-based functions that accept any fluid name supported by the CoolProp thermodynamic library, which are useful for computing properties of non-standard fluids like CO2, nitrogen, or other injection gases.

The functions are organized by property category below. Within each category, you will find correlation-specific variants (e.g., Standing, Vazquez-Beggs) as well as unified dispatch functions that select the best correlation automatically based on conditions. Import them from `bores.correlations.scalars` or from `bores.correlations` directly.

```python
from bores.correlations.scalars import (
    compute_oil_formation_volume_factor_standing,
    compute_gas_compressibility_factor,
    compute_oil_viscosity,
)

# Or equivalently:
from bores.correlations import compute_oil_formation_volume_factor_standing
```

---

## Temperature and Unit Conversions

These utility functions convert between temperature scales. They accept both scalar floats and NumPy arrays.

| Function | Description | Formula |
| --- | --- | --- |
| `kelvin_to_fahrenheit(temp_K)` | Kelvin to Fahrenheit | $(K - 273.15) \times 9/5 + 32$ |
| `fahrenheit_to_kelvin(temp_F)` | Fahrenheit to Kelvin | $(F - 32) \times 5/9 + 273.15$ |
| `fahrenheit_to_celsius(temp_F)` | Fahrenheit to Celsius | $(F - 32) \times 5/9$ |
| `fahrenheit_to_rankine(temp_F)` | Fahrenheit to Rankine | $F + 459.67$ |

```python
from bores.correlations.scalars import fahrenheit_to_kelvin, kelvin_to_fahrenheit

temp_K = fahrenheit_to_kelvin(200.0)  # 366.48 K
temp_F = kelvin_to_fahrenheit(366.48)  # 200.0 F
```

---

## Validation and Clipping

These functions validate or constrain pressure and temperature inputs to physically reasonable ranges.

| Function | Parameters | Description |
| --- | --- | --- |
| `validate_input_temperature(temperature)` | `temperature`: Temperature (F) | Raises `ValidationError` if temperature is outside the valid reservoir range |
| `validate_input_pressure(pressure)` | `pressure`: Pressure (psi) | Raises `ValidationError` if pressure is outside the valid reservoir range |
| `clip_pressure(pressure, fluid)` | `pressure`: Pressure (psi), `fluid`: CoolProp fluid name | Clips pressure to CoolProp's valid range for the given fluid |
| `clip_temperature(temperature, fluid)` | `temperature`: Temperature (K), `fluid`: CoolProp fluid name | Clips temperature to CoolProp's valid range for the given fluid |
| `is_CoolProp_fluid(fluid)` | `fluid`: Fluid name string | Returns `True` if the fluid is supported by CoolProp (cached) |

---

## Generic Fluid Properties (CoolProp)

These functions compute fluid properties for any CoolProp-supported fluid using equation-of-state calculations. They are useful for injection gases (CO2, N2), pure hydrocarbons, and other fluids where black-oil correlations do not apply.

### `compute_fluid_density`

```python
compute_fluid_density(pressure: float, temperature: float, fluid: str) -> float
```

Computes fluid density from the equation of state using CoolProp. Returns density in lbm/ft³.

| Parameter | Type | Unit | Description |
| --- | --- | --- | --- |
| `pressure` | `float` | psi | Reservoir pressure |
| `temperature` | `float` | F | Reservoir temperature |
| `fluid` | `str` | - | CoolProp fluid name (e.g., `"CO2"`, `"Water"`, `"Methane"`) |

```python
from bores.correlations.scalars import compute_fluid_density

co2_density = compute_fluid_density(3000.0, 200.0, "CO2")
print(f"CO2 density: {co2_density:.2f} lbm/ft³")
```

### `compute_fluid_viscosity`

```python
compute_fluid_viscosity(pressure: float, temperature: float, fluid: str) -> float
```

Computes fluid dynamic viscosity from the equation of state using CoolProp. Returns viscosity in centipoise (cP).

| Parameter | Type | Unit | Description |
| --- | --- | --- | --- |
| `pressure` | `float` | psi | Reservoir pressure |
| `temperature` | `float` | F | Reservoir temperature |
| `fluid` | `str` | - | CoolProp fluid name |

### `compute_fluid_compressibility_factor`

```python
compute_fluid_compressibility_factor(pressure: float, temperature: float, fluid: str) -> float
```

Computes the compressibility factor Z from equation of state. Returns a dimensionless value.

### `compute_fluid_compressibility`

```python
compute_fluid_compressibility(pressure: float, temperature: float, fluid: str) -> float
```

Computes isothermal compressibility $C_f = -(1/\rho) \cdot (d\rho/dP)_T$ using CoolProp. Returns compressibility in psi-1.

---

## Gas Gravity

### `compute_gas_gravity`

```python
compute_gas_gravity(gas: str) -> float
```

Computes the specific gravity of a gas relative to air at standard conditions. Accepts any CoolProp-supported gas name.

```python
from bores.correlations.scalars import compute_gas_gravity

methane_gravity = compute_gas_gravity("Methane")  # ~0.554
co2_gravity = compute_gas_gravity("CO2")  # ~1.52
```

### `compute_gas_gravity_from_density`

```python
compute_gas_gravity_from_density(pressure: float, temperature: float, density: float) -> float
```

Computes gas gravity from a measured density at specific conditions, by comparing the density to air density at the same conditions.

| Parameter | Type | Unit | Description |
| --- | --- | --- | --- |
| `pressure` | `float` | psi | Pressure at which density was measured |
| `temperature` | `float` | F | Temperature at which density was measured |
| `density` | `float` | lbm/ft³ | Measured gas density |

---

## Oil Specific Gravity and API Gravity

### `compute_oil_specific_gravity`

```python
compute_oil_specific_gravity(
    oil_density: float, pressure: float, temperature: float, oil_compressibility: float
) -> float
```

Converts oil density at reservoir conditions to specific gravity at standard conditions using a linearized correction for pressure and temperature.

$$\rho_{stp} \approx \rho \cdot \exp\left[C_o \cdot (P_{stp} - P) + \alpha \cdot (T_{stp} - T)\right]$$

$$SG = \rho_{stp} / \rho_{water}$$

### `compute_oil_api_gravity`

```python
compute_oil_api_gravity(oil_specific_gravity: float) -> float
```

Converts oil specific gravity to API gravity in degrees.

$$API = \frac{141.5}{SG} - 131.5$$

```python
from bores.correlations.scalars import compute_oil_api_gravity

api = compute_oil_api_gravity(0.85)  # ~34.97 degrees API
```

---

## Rate Conversions

### `convert_surface_rate_to_reservoir`

```python
convert_surface_rate_to_reservoir(surface_rate: float, formation_volume_factor: float) -> float
```

Converts a surface rate (STB/day) to reservoir conditions (bbl/day). For injection (positive rate), multiplies by FVF. For production (negative rate), divides by FVF.

### `convert_reservoir_rate_to_surface`

```python
convert_reservoir_rate_to_surface(reservoir_rate: float, formation_volume_factor: float) -> float
```

The inverse of `convert_surface_rate_to_reservoir`.

---

## Oil Formation Volume Factor

### `compute_oil_formation_volume_factor_standing`

```python
compute_oil_formation_volume_factor_standing(
    temperature: float, oil_specific_gravity: float,
    gas_gravity: float, gas_to_oil_ratio: float
) -> float
```

Computes $B_o$ using the Standing (1947) correlation.

$$B_o = 0.972 + 0.000147 \cdot \left[R_s \cdot \left(\frac{\gamma_g}{\gamma_o}\right)^{0.5} + 1.25 \cdot T\right]^{1.175}$$

| Parameter | Type | Unit | Description |
| --- | --- | --- | --- |
| `temperature` | `float` | F | Reservoir temperature |
| `oil_specific_gravity` | `float` | - | Oil specific gravity (water = 1.0) |
| `gas_gravity` | `float` | - | Gas specific gravity (air = 1.0) |
| `gas_to_oil_ratio` | `float` | SCF/STB | Solution gas-oil ratio |

**Valid range:** 60-300 F, oil SG 0.5-0.95, GOR 20-2000 SCF/STB.

```python
from bores.correlations.scalars import compute_oil_formation_volume_factor_standing

Bo = compute_oil_formation_volume_factor_standing(
    temperature=200.0,
    oil_specific_gravity=0.85,
    gas_gravity=0.7,
    gas_to_oil_ratio=500.0,
)
print(f"Bo = {Bo:.4f} bbl/STB")
```

### `compute_oil_formation_volume_factor_vazquez_and_beggs`

```python
compute_oil_formation_volume_factor_vazquez_and_beggs(
    temperature: float, oil_specific_gravity: float,
    gas_gravity: float, gas_to_oil_ratio: float
) -> float
```

Computes $B_o$ using the Vazquez and Beggs (1980) correlation. Uses API-gravity-dependent coefficients $(a_1, a_2, a_3)$ with different values for API <= 30 and API > 30.

**Valid range:** API 16-58, temperature 100-300 F, GOR 0-2000 SCF/STB.

### `correct_oil_fvf_for_pressure`

```python
correct_oil_fvf_for_pressure(
    saturated_oil_fvf: float, oil_compressibility: float,
    bubble_point_pressure: float, current_pressure: float
) -> float
```

Applies exponential shrinkage correction for pressures above bubble point.

$$B_o(P) = B_o(P_b) \cdot \exp\left[C_o \cdot (P_b - P)\right]$$

Returns the saturated FVF unchanged if current pressure is below bubble point.

### `compute_oil_formation_volume_factor`

```python
compute_oil_formation_volume_factor(
    pressure: float, temperature: float, bubble_point_pressure: float,
    oil_specific_gravity: float, gas_gravity: float,
    gas_to_oil_ratio: float, oil_compressibility: float
) -> float
```

Unified dispatch: uses Standing for temperatures <= 100 F and Vazquez-Beggs for temperatures > 100 F, then applies the above-bubble-point pressure correction. This is the function the simulator calls internally.

---

## Water Formation Volume Factor

### `compute_water_formation_volume_factor`

```python
compute_water_formation_volume_factor(water_density: float, salinity: float) -> float
```

Computes $B_w$ as the ratio of standard water density to reservoir water density. Uses Batzle-Wang for standard conditions density.

### `compute_water_formation_volume_factor_mccain`

```python
compute_water_formation_volume_factor_mccain(
    pressure: float, temperature: float,
    salinity: float = 0.0, gas_solubility: float = 0.0
) -> float
```

McCain correlation for water FVF. Accounts for temperature, pressure, salinity, and dissolved gas effects. Valid for T: 200-270 F, P: 1000-20,000 psi, salinity: 0-200,000 ppm.

---

## Gas Formation Volume Factor

### `compute_gas_formation_volume_factor`

```python
compute_gas_formation_volume_factor(
    pressure: float, temperature: float, gas_compressibility_factor: float
) -> float
```

Computes $B_g$ in ft³/SCF using the real gas law.

$$B_g = \frac{Z \cdot T \cdot P_{std}}{P \cdot T_{std}}$$

| Parameter | Type | Unit | Description |
| --- | --- | --- | --- |
| `pressure` | `float` | psi | Reservoir pressure |
| `temperature` | `float` | F | Reservoir temperature |
| `gas_compressibility_factor` | `float` | - | Z-factor (dimensionless) |

---

## Gas Compressibility Factor (Z-Factor)

BORES provides three Z-factor correlations of increasing accuracy and computational cost. All three accept sour gas corrections via optional H2S, CO2, and N2 mole fractions using the Wichert-Aziz method.

### `compute_gas_compressibility_factor_papay`

```python
compute_gas_compressibility_factor_papay(
    pressure: float, temperature: float, gas_gravity: float,
    h2s_mole_fraction: float = 0.0, co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0
) -> float
```

Papay (1985) explicit correlation. Fastest, suitable for low-pressure gas.

$$Z = 1 - \frac{3.52 \cdot P_r \cdot e^{-0.869 \cdot T_r}}{T_r} + \frac{0.274 \cdot P_r^2}{T_r^2}$$

**Valid range:** $P_r$: 0.2-15, $T_r$: 1.05-3.0, $\gamma_g$: 0.55-1.0.

### `compute_gas_compressibility_factor_hall_yarborough`

```python
compute_gas_compressibility_factor_hall_yarborough(
    pressure: float, temperature: float, gas_gravity: float,
    h2s_mole_fraction: float = 0.0, co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0,
    maximum_solver_iterations: int = 50, tolerance: float = 1e-10
) -> float
```

Hall-Yarborough (1973) implicit correlation solved with Newton-Raphson. More accurate than Papay, especially at high pressure.

**Valid range:** $P_r$: 0.2-30, $T_r$: 1.0-3.0.

### `compute_gas_compressibility_factor_dranchuk_abou_kassem`

```python
compute_gas_compressibility_factor_dranchuk_abou_kassem(
    pressure: float, temperature: float, gas_gravity: float,
    h2s_mole_fraction: float = 0.0, co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0,
    maximum_solver_iterations: int = 50, tolerance: float = 1e-10
) -> float
```

Dranchuk-Abou-Kassem (DAK, 1975) 11-parameter correlation. Most accurate, industry standard for high-pressure gas.

**Valid range:** $P_r$: 0.2-30, $T_r$: 1.0-3.0.

### `compute_gas_compressibility_factor`

```python
compute_gas_compressibility_factor(
    pressure: float, temperature: float, gas_gravity: float,
    h2s_mole_fraction: float = 0.0, co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0,
    method: ZFactorMethod = "dak"
) -> float
```

Unified dispatch function. The `method` parameter accepts `"papay"`, `"hall-yarborough"`, or `"dak"` (default). DAK is usually sufficient for black-oil simulations.

```python
from bores.correlations.scalars import compute_gas_compressibility_factor

Z = compute_gas_compressibility_factor(
    pressure=2000.0, temperature=150.0, gas_gravity=0.65, method="dak"
)
print(f"Z = {Z:.4f}")
```

---

## Bubble Point Pressure

### `compute_oil_bubble_point_pressure`

```python
compute_oil_bubble_point_pressure(
    gas_gravity: float, oil_api_gravity: float,
    temperature: float, gas_to_oil_ratio: float
) -> float
```

Vazquez-Beggs correlation for oil bubble point pressure.

$$P_b = \left[\frac{R_s}{C_1 \cdot \gamma_g \cdot \exp\left(\frac{C_3 \cdot API}{T_R}\right)}\right]^{1/C_2}$$

Valid for API 16-45, temperature 100-300 F, GOR up to 2000 SCF/STB.

### `estimate_bubble_point_pressure_standing`

```python
estimate_bubble_point_pressure_standing(
    oil_api_gravity: float, gas_gravity: float, observed_gas_to_oil_ratio: float
) -> float
```

Estimates bubble point pressure by numerically inverting the Standing GOR correlation. Uses Brent's root-finding method.

### `compute_water_bubble_point_pressure`

```python
compute_water_bubble_point_pressure(
    temperature: float, gas_solubility_in_water: float,
    salinity: float = 0.0, gas: str = "methane"
) -> float
```

Computes the pressure at which the given gas solubility in water is reached. Uses analytical inversion for methane (McCain), numerical root-finding for other gases.

### `compute_water_bubble_point_pressure_mccain`

```python
compute_water_bubble_point_pressure_mccain(
    temperature: float, gas_solubility_in_water: float, salinity: float
) -> float
```

Inverted McCain correlation for methane bubble point in water. Valid for T: 100-400 F, P: 0-14,700 psi, salinity: 0-200,000 ppm.

---

## Gas-to-Oil Ratio (GOR)

### `compute_gas_to_oil_ratio`

```python
compute_gas_to_oil_ratio(
    pressure: float, temperature: float, bubble_point_pressure: float,
    gas_gravity: float, oil_api_gravity: float,
    gor_at_bubble_point_pressure: float = None
) -> float
```

Vazquez-Beggs correlation for solution GOR. Returns the GOR at bubble point for undersaturated conditions ($P \geq P_b$) and pressure-dependent GOR for saturated conditions ($P < P_b$).

### `compute_gas_to_oil_ratio_standing`

```python
compute_gas_to_oil_ratio_standing(
    pressure: float, oil_api_gravity: float, gas_gravity: float
) -> float
```

Standing correlation for solution GOR. Simplified form that does not require temperature.

$$R_s = \gamma_g \cdot \left[\left(\frac{P}{18.2} + 1.4\right) \cdot 10^{0.0125 \cdot API}\right]^{1/1.2048}$$

### `estimate_solution_gor`

```python
estimate_solution_gor(
    pressure: float, temperature: float, oil_api_gravity: float, gas_gravity: float,
    maximum_solver_iterations: int = 20, tolerance: float = 1e-4
) -> float
```

Iterative estimation of solution GOR that solves the coupled system where GOR depends on pressure and bubble point, and bubble point depends on GOR and temperature. Handles both saturated and undersaturated conditions automatically.

---

## Oil Viscosity

### `compute_dead_oil_viscosity_modified_beggs`

```python
compute_dead_oil_viscosity_modified_beggs(
    temperature: float, oil_specific_gravity: float
) -> float
```

Modified Beggs correlation (Labedi, 1992) for dead oil viscosity. Valid for API 5-75.

$$\log_{10}(\mu_{od} + 1) = 1.8653 - 0.025086 \cdot \gamma_o - 0.5644 \cdot \log_{10}(T_R)$$

### `compute_oil_viscosity`

```python
compute_oil_viscosity(
    pressure: float, temperature: float, bubble_point_pressure: float,
    oil_specific_gravity: float, gas_to_oil_ratio: float,
    gor_at_bubble_point_pressure: float
) -> float
```

Full oil viscosity calculation using Modified Beggs and Robinson. Computes dead oil viscosity first, then applies saturated or undersaturated corrections depending on whether the pressure is above or below the bubble point.

**Saturated** ($P \leq P_b$):

$$\mu_o = X \cdot \mu_{od}^Y \qquad X = 10.715 \cdot (R_s + 100)^{-0.515} \qquad Y = 5.44 \cdot (R_s + 150)^{-0.338}$$

**Undersaturated** ($P > P_b$):

$$\mu_o = \mu_{ob} \cdot \left(\frac{P}{P_b}\right)^{X_{under}}$$

---

## Gas Properties

### `compute_gas_molecular_weight`

```python
compute_gas_molecular_weight(gas_gravity: float) -> float
```

Returns apparent molecular weight: $MW = \gamma_g \times 28.96$ g/mol.

### `compute_gas_pseudocritical_properties`

```python
compute_gas_pseudocritical_properties(
    gas_gravity: float, h2s_mole_fraction: float = 0.0,
    co2_mole_fraction: float = 0.0, n2_mole_fraction: float = 0.0
) -> tuple[float, float]
```

Sutton's correlation for pseudocritical pressure and temperature, with Wichert-Aziz correction for sour gases. Returns $(P_{pc}, T_{pc})$ in psi and Rankine.

### `compute_gas_density`

```python
compute_gas_density(
    pressure: float, temperature: float,
    gas_gravity: float, gas_compressibility_factor: float
) -> float
```

Real gas equation of state density in lbm/ft³.

### `compute_gas_viscosity`

```python
compute_gas_viscosity(
    temperature: float, gas_density: float, gas_molecular_weight: float
) -> float
```

Lee-Gonzalez-Eakin (LGE) correlation for gas viscosity in cP. Valid for T: 100-400 F, P up to 10,000 psi.

$$\mu_g = (k \times 10^{-4}) \cdot \exp(x \cdot \rho_g^y)$$

---

## Water Properties

### `compute_water_viscosity`

```python
compute_water_viscosity(
    temperature: float, salinity: float = 0.0, pressure: float = 14.7
) -> float
```

McCain correlation for water viscosity in cP, corrected for salinity and pressure. Valid for T: 86-350 F, salinity up to 300,000 ppm, pressure up to 10,000 psi.

### `compute_water_density`

```python
compute_water_density(
    pressure: float, temperature: float, gas_gravity: float = 0.0,
    salinity: float = 0.0, gas_solubility_in_water: float = 0.0,
    gas_free_water_formation_volume_factor: float = 1.0
) -> float
```

Live water density at reservoir conditions using McCain's mass balance approach. Accounts for dissolved gas and salinity.

### `compute_water_density_mccain`

```python
compute_water_density_mccain(
    pressure: float, temperature: float, salinity: float = 0.0
) -> float
```

McCain correlation for brine density in lbm/ft³.

### `compute_water_density_batzle`

```python
compute_water_density_batzle(
    pressure: float, temperature: float, salinity: float
) -> float
```

Batzle and Wang (1992) correlation. More accurate at high temperature and pressure conditions.

---

## Compressibility

### `compute_oil_compressibility`

```python
compute_oil_compressibility(
    pressure: float, temperature: float, bubble_point_pressure: float,
    oil_api_gravity: float, gas_gravity: float, gor_at_bubble_point_pressure: float,
    gas_formation_volume_factor: float = 1.0, oil_formation_volume_factor: float = 1.0
) -> float
```

Vasquez and Beggs (1980) correlation for oil compressibility in psi-1. For undersaturated oil ($P > P_b$), uses the standard correlation. For saturated oil ($P \leq P_b$), adds a gas liberation correction term.

Valid for P: 100-5,000 psi, T: 100-300 F, API: 16-58.

### `compute_gas_compressibility`

```python
compute_gas_compressibility(
    pressure: float, temperature: float, gas_gravity: float,
    gas_compressibility_factor: float = None,
    h2s_mole_fraction: float = 0.0, co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0
) -> float
```

Isothermal gas compressibility in psi-1 using Papay's analytical derivative.

$$C_g = \frac{1}{P} - \frac{1}{Z \cdot P_{pc}} \cdot \frac{dZ}{dP_r}$$

### `compute_water_compressibility`

```python
compute_water_compressibility(
    pressure: float, temperature: float, bubble_point_pressure: float,
    gas_formation_volume_factor: float, gas_solubility_in_water: float,
    gas_free_water_formation_volume_factor: float, salinity: float = 0.0
) -> float
```

McCain correlation for water compressibility in psi-1. Handles both undersaturated ($P \geq P_{wb}$) and saturated ($P < P_{wb}$) water conditions.

### `compute_total_fluid_compressibility`

```python
compute_total_fluid_compressibility(
    water_saturation: float, oil_saturation: float,
    water_compressibility: float, oil_compressibility: float,
    gas_saturation: float = None, gas_compressibility: float = None
) -> float
```

Saturation-weighted average of phase compressibilities. Optionally includes gas phase for three-phase systems.

---

## Gas Solubility in Water

### `compute_gas_solubility_in_water`

```python
compute_gas_solubility_in_water(
    pressure: float, temperature: float,
    salinity: float = 0.0, gas: str = "methane"
) -> float
```

Computes gas solubility in water in SCF/STB. Automatically selects the best correlation based on gas type and temperature:

- **Methane** (100-400 F): McCain correlation
- **CO2** (32-572 F): Duan and Sun correlation
- **Other gases** (N2, Ar, O2, He, H2): Henry's law with Setschenow salinity correction

### `compute_gas_free_water_formation_volume_factor`

```python
compute_gas_free_water_formation_volume_factor(
    pressure: float, temperature: float
) -> float
```

McCain correlation for gas-free water FVF ($B_{w,gas-free}$) in bbl/STB. Accounts for thermal expansion and isothermal compressibility without dissolved gas effects.

---

## Oil and Water Density

### `compute_live_oil_density`

```python
compute_live_oil_density(
    api_gravity: float, gas_gravity: float,
    gas_to_oil_ratio: float, formation_volume_factor: float
) -> float
```

Mass balance approach for live oil density in lbm/ft³. Accounts for stock tank oil mass, dissolved gas mass, and volume expansion via FVF.

---

## Volumetrics

### `compute_hydrocarbon_in_place`

```python
compute_hydrocarbon_in_place(
    area: float, thickness: float, porosity: float, phase_saturation: float,
    formation_volume_factor: float, net_to_gross_ratio: float = 1.0,
    hydrocarbon_type: str = "oil",
    acre_ft_to_bbl: float = 7758.0, acre_ft_to_ft3: float = 43560.0
) -> float
```

Volumetric method for original hydrocarbons in place. For oil, returns STB. For gas, returns SCF. For water, returns STB.

$$OIP = 7758 \cdot A \cdot h \cdot \phi \cdot S_o \cdot (N/G) / B_o$$

$$GIP = 43560 \cdot A \cdot h \cdot \phi \cdot S_g \cdot (N/G) / B_g$$

---

## Miscible Flooding (Todd-Longstaff)

These functions implement the Todd-Longstaff (1972) mixing model for miscible gas flooding. They compute effective viscosity and density of oil-solvent mixtures based on a mixing parameter $\omega$ that interpolates between fully segregated (immiscible) and fully mixed (miscible) flow behavior.

### `compute_miscibility_transition_factor`

```python
compute_miscibility_transition_factor(
    pressure: float, minimum_miscibility_pressure: float,
    transition_width: float = 500.0
) -> float
```

Smooth pressure-dependent transition from 0 (immiscible) to 1 (fully miscible) using hyperbolic tangent.

$$f(P) = 0.5 \cdot \left(1 + \tanh\left(\frac{P - MMP}{\Delta P}\right)\right)$$

### `compute_effective_todd_longstaff_omega`

```python
compute_effective_todd_longstaff_omega(
    pressure: float, base_omega: float,
    minimum_miscibility_pressure: float, transition_width: float = 500.0
) -> float
```

Combines the base mixing parameter with the pressure-dependent transition factor: $\omega_{eff} = \omega_{base} \cdot f(P)$.

### `compute_todd_longstaff_effective_viscosity`

```python
compute_todd_longstaff_effective_viscosity(
    oil_viscosity: float, solvent_viscosity: float,
    solvent_concentration: float, omega: float = 0.67
) -> float
```

Todd-Longstaff effective viscosity for oil-solvent mixtures.

$$\mu_{eff} = \mu_{mix}^\omega \cdot \mu_{seg}^{1-\omega}$$

where $\mu_{mix}$ is the arithmetic mean (fully mixed) and $\mu_{seg}$ is the harmonic mean (fully segregated).

| $\omega$ | Behavior | Viscosity Model |
| --- | --- | --- |
| 0.0 | Fully segregated (immiscible) | Harmonic mean |
| 0.5 | Partial mixing | Geometric mean |
| 0.67 | Typical for CO2 floods | Todd-Longstaff interpolation |
| 1.0 | Fully mixed (miscible) | Arithmetic mean |

### `compute_todd_longstaff_effective_density`

```python
compute_todd_longstaff_effective_density(
    oil_density: float, solvent_density: float,
    oil_viscosity: float, solvent_viscosity: float,
    solvent_concentration: float = 1.0, omega: float = 0.67
) -> float
```

Analogous to the viscosity function but for density mixing.

---

## Miscellaneous

### `compute_harmonic_mean`

```python
compute_harmonic_mean(value1: float, value2: float) -> float
```

Computes the harmonic mean of two values: $\frac{2 \cdot v_1 \cdot v_2}{v_1 + v_2}$. Used internally for transmissibility calculations. Returns 0 if both values are zero.

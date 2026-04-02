# Well Fluids

## Overview

Every well in BORES needs to know the properties of the fluids it handles. Production wells define `ProducedFluid` objects for each phase (oil, water, gas) that may flow into the wellbore. Injection wells define a single `InjectedFluid` object describing the fluid being pumped into the reservoir. These fluid objects provide the specific gravity and molecular weight needed to compute PVT properties (formation volume factors, viscosities, densities) at wellbore conditions. Both classes inherit from [Fluid](../fluids.md), which provides the underlying PVT table and pseudo-pressure capabilities.

The distinction between produced and injected fluids matters because of how PVT properties are used. For production wells, the simulator looks up oil, water, and gas properties from the reservoir model's PVT grids. For injection wells, the fluid properties may differ significantly from the in-situ reservoir fluids, particularly for gas injection (where the injected gas may be CO2 or nitrogen with very different properties from the reservoir's solution gas).

---

## ProducedFluid

A `ProducedFluid` describes a fluid phase that flows from the reservoir into a production well. It is a simple data class with four required fields:

```python
import bores

oil = bores.ProducedFluid(
    name="Oil",
    phase=bores.FluidPhase.OIL,
    specific_gravity=0.85,    # Oil gravity relative to water (~35 API)
    molecular_weight=200.0,   # Average molecular weight (g/mol)
)

water = bores.ProducedFluid(
    name="Formation Water",
    phase=bores.FluidPhase.WATER,
    specific_gravity=1.02,    # Slightly saline
    molecular_weight=18.015,  # H2O
)

gas = bores.ProducedFluid(
    name="Associated Gas",
    phase=bores.FluidPhase.GAS,
    specific_gravity=0.65,    # Light hydrocarbon gas
    molecular_weight=16.04,   # Close to methane
)
```

For most simulations, the produced fluid properties should match the reservoir fluid properties you used when building the reservoir model. The specific gravity of oil should match the `oil_specific_gravity_grid` values, and the gas specific gravity should match the `gas_specific_gravity` parameter.

A production well typically includes all three phases, even if you expect only one phase to be produced initially. During a waterflood, for example, water will eventually break through and the well needs water fluid properties to compute the water production rate correctly.

---

## InjectedFluid

An `InjectedFluid` describes the fluid being pumped into an injection well. It extends the base `WellFluid` class with additional parameters for miscible flooding, salinity, and property overrides.

### Water Injection

For water injection, the `InjectedFluid` is straightforward:

```python
import bores

water_fluid = bores.InjectedFluid(
    name="Injection Water",
    phase=bores.FluidPhase.WATER,
    specific_gravity=1.0,     # Fresh water
    molecular_weight=18.015,
)
```

If you are injecting saline water (brine), specify the salinity:

```python
brine_fluid = bores.InjectedFluid(
    name="Seawater",
    phase=bores.FluidPhase.WATER,
    specific_gravity=1.03,
    molecular_weight=18.015,
    salinity=35000.0,  # ppm NaCl
)
```

The salinity affects water density and viscosity calculations. Seawater typically has about 35,000 ppm total dissolved solids, while formation brines can range from 10,000 to over 200,000 ppm.

### Hydrocarbon Gas Injection

For injecting hydrocarbon gas (methane, natural gas, or enriched gas), use standard gas properties:

```python
import bores

natural_gas = bores.InjectedFluid(
    name="Natural Gas",
    phase=bores.FluidPhase.GAS,
    specific_gravity=0.70,
    molecular_weight=20.0,
)
```

The gas properties (compressibility factor, density, viscosity) are computed from correlations using the specific gravity and molecular weight. These correlations work well for hydrocarbon gases because they were developed from large datasets of natural gas measurements.

### CO2 Injection

CO2 injection requires special attention because CO2 is a non-ideal gas whose properties deviate significantly from the standard gas correlations used for hydrocarbon gases. At typical reservoir conditions (above 1100 psi and 90 degrees F), CO2 exists as a supercritical fluid with a density 5 to 10 times higher than what the gas correlations predict, and a viscosity 3 to 5 times higher.

BORES provides density and viscosity override parameters on `InjectedFluid` to handle this:

```python
import bores

co2_fluid = bores.InjectedFluid(
    name="CO2",
    phase=bores.FluidPhase.GAS,
    specific_gravity=1.52,      # CO2 gravity relative to air
    molecular_weight=44.01,     # CO2 molecular weight
    density=35.0,               # lbm/ft³ - from EOS or lab data
    viscosity=0.05,             # cP - from EOS or lab data
)
```

When `density` or `viscosity` is set, BORES uses those values directly instead of computing them from correlations. This is critical for CO2 because the standard gas correlations (Lee-Gonzalez-Eakin for viscosity, Hall-Yarborough for Z-factor) were developed for hydrocarbon gases and produce grossly inaccurate results for CO2 at reservoir conditions.

The density and viscosity values should come from equation-of-state calculations (using a tool like CoolProp, NIST REFPROP, or commercial PVT software) or from laboratory measurements at your specific reservoir temperature and pressure. As a rough guide for CO2 at typical reservoir conditions:

| Condition | Density (lbm/ft³) | Viscosity (cP) |
| --- | --- | --- |
| 2000 psi, 150 degrees F | ~30 | ~0.04 |
| 3000 psi, 150 degrees F | ~38 | ~0.05 |
| 4000 psi, 200 degrees F | ~35 | ~0.06 |
| 5000 psi, 200 degrees F | ~40 | ~0.07 |

!!! warning "Always Override CO2 Properties"

    If you are simulating CO2 injection without setting `density` and `viscosity` on the `InjectedFluid`, the simulator will use standard gas correlations that can underpredict CO2 density by a factor of 5 or more. This leads to incorrect gravity segregation, incorrect injection volumes, and unreliable recovery predictions. Always provide measured or EOS-computed properties for CO2.

### Custom Pseudo-Pressure Tables

For gas wells operating at high pressures or with non-ideal gases (CO2, nitrogen, sour gas), you can provide a custom pseudo-pressure table instead of relying on the default correlations. This is useful when you have:

- Laboratory PVT measurements at specific pressures
- Equation-of-state calculations from commercial software
- High-resolution data that requires finer pressure spacing than the default

BORES supports two ways to build custom pseudo-pressure tables:

**Option 1: From data arrays** (recommended for lab data or EOS output):

```python
import numpy as np
import bores

# From your PVT measurements or EOS calculations
pressures = np.array([100, 500, 1000, 1500, 2000, 3000, 4000, 5000])  # psi
pseudo_pressures = np.array([
    2.1e4, 5.3e5, 1.2e6, 1.9e6, 2.8e6, 4.5e6, 6.4e6, 8.1e6
])  # psi²/cP

# Build table from your data
custom_table = bores.PseudoPressureTable(
    pressures=pressures,
    pseudo_pressures=pseudo_pressures,
    reference_pressure=14.7,  # Standard conditions
)

# Use in injected fluid
co2_fluid = bores.InjectedFluid(
    name="CO2",
    phase=bores.FluidPhase.GAS,
    specific_gravity=1.52,
    molecular_weight=44.01,
    density=35.0,
    viscosity=0.05,
    pseudo_pressure_table=custom_table,  # Provide custom table
)
```

**Option 2: From custom Z-factor and viscosity functions**:

```python
import numpy as np
import bores

# Your custom correlation functions (must support arrays)
def my_z_factor(pressure):
    """Custom Z-factor correlation from EOS."""
    # Your equation-of-state calculations here
    return z_array  # Must return numpy array

def my_viscosity(pressure):
    """Custom viscosity correlation from lab data."""
    # Your viscosity model here
    return mu_array  # Must return numpy array

# Mark functions as array-compatible
my_z_factor._supports_arrays = True
my_viscosity._supports_arrays = True

# Build table with higher resolution
custom_table = bores.PseudoPressureTable(
    z_factor_func=my_z_factor,
    viscosity_func=my_viscosity,
    pressure_range=(100, 8000),  # Wider range
    points=2000,  # Higher resolution (default is 1000)
    reference_pressure=14.7,
)

# Use in injected fluid
co2_fluid = bores.InjectedFluid(
    name="CO2",
    phase=bores.FluidPhase.GAS,
    specific_gravity=1.52,
    molecular_weight=44.01,
    pseudo_pressure_table=custom_table,
)
```

!!! info "When to Use Custom Pseudo-Pressure Tables"

    Custom tables are most valuable for:

    - **CO2 injection**: Standard correlations are inaccurate; use EOS-computed Z and μ
    - **Sour gas**: H2S content affects Z-factor significantly
    - **High pressure** (> 5000 psi): Non-ideal gas behavior increases
    - **Fine-tuning**: Match specific lab PVT measurements exactly

    For typical hydrocarbon gas injection (methane, natural gas) at moderate pressures (< 3000 psi), the default correlations are usually adequate.

!!! tip "Obtaining Pseudo-Pressure Data"

    You can compute pseudo-pressure tables using:

    - **CoolProp**: Open-source thermodynamic library (Python, C++)
    - **NIST REFPROP**: High-accuracy reference fluid properties
    - **Commercial PVT software**: Eclipse PVTi, CMG WinProp, etc.
    - **Laboratory measurements**: Direct PVT cell measurements

    Most commercial tools can export P-Z-μ tables that you can convert to pseudo-pressure using BORES' `PseudoPressureTable` with the value-based mode.

### Miscible Gas Injection

For miscible flooding (CO2 or enriched gas that mixes with oil above the minimum miscibility pressure), set `is_miscible=True` and provide the MMP and Todd-Longstaff mixing parameter:

```python
import bores

co2_miscible = bores.InjectedFluid(
    name="CO2",
    phase=bores.FluidPhase.GAS,
    specific_gravity=1.52,
    molecular_weight=44.01,
    density=35.0,
    viscosity=0.05,
    is_miscible=True,
    minimum_miscibility_pressure=1200.0,  # psi
    todd_longstaff_omega=0.67,            # Mixing parameter (0-1)
    miscibility_transition_width=500.0,   # Pressure range for smooth transition
)
```

The `minimum_miscibility_pressure` (MMP) is the pressure above which the injected gas develops first-contact or multi-contact miscibility with the reservoir oil. Below the MMP, the gas displaces oil immiscibly with high residual oil saturation. Above the MMP, the displacement approaches piston-like efficiency with near-zero residual oil.

The `todd_longstaff_omega` parameter controls the degree of mixing between solvent and oil at the sub-grid scale. A value of 1.0 means complete mixing (the solvent and oil are fully miscible within each grid cell), while 0.0 means no mixing (the fluids remain segregated). The default value of 0.67 is the most commonly used in industry. See [Miscible Flooding](../advanced/miscibility.md) for detailed physics.

The `miscibility_transition_width` controls the pressure range over which miscibility transitions from immiscible to fully miscible. BORES uses a smooth hyperbolic tangent transition centered at the MMP, with the width controlling how abrupt the transition is. A width of 0 gives a sharp step change; values of 300-500 psi give a gradual transition that is more numerically stable.

!!! info "Miscibility Requirements"

    When `is_miscible=True`, the following conditions must be met:

    - The phase must be `GAS` (miscible water injection is not supported)
    - `minimum_miscibility_pressure` must be provided
    - `todd_longstaff_omega` must be provided (defaults to 0.67 if not specified)

---

## Nitrogen Injection

Nitrogen is sometimes used for pressure maintenance or immiscible gas injection. Like CO2, its properties differ from hydrocarbon gas, but the deviation is less severe:

```python
import bores

n2_fluid = bores.InjectedFluid(
    name="Nitrogen",
    phase=bores.FluidPhase.GAS,
    specific_gravity=0.967,     # N2 gravity relative to air
    molecular_weight=28.013,    # N2 molecular weight
)
```

For nitrogen, the standard gas correlations are reasonably accurate because nitrogen is closer to ideal gas behavior than CO2. However, for high-pressure applications (above 5000 psi), you may want to provide density and viscosity overrides from EOS calculations for improved accuracy.

---

## Common Fluid Property Values

| Fluid | Phase | Specific Gravity | Molecular Weight |
| --- | --- | --- | --- |
| Light oil (35 API) | OIL | 0.85 | 180-220 |
| Medium oil (25 API) | OIL | 0.90 | 250-350 |
| Heavy oil (15 API) | OIL | 0.97 | 400-600 |
| Fresh water | WATER | 1.00 | 18.015 |
| Seawater | WATER | 1.03 | 18.015 |
| Formation brine | WATER | 1.05-1.15 | 18.015 |
| Methane | GAS | 0.553 | 16.04 |
| Natural gas (typical) | GAS | 0.60-0.75 | 17-25 |
| CO2 | GAS | 1.52 | 44.01 |
| Nitrogen | GAS | 0.967 | 28.013 |
| H2S | GAS | 1.18 | 34.08 |

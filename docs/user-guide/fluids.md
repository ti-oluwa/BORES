# Fluids

The `Fluid` class represents a named reservoir fluid with an associated phase and, optionally, a set of PVT lookup tables or a pseudo-pressure table. It sits between the general-purpose `PVTTables` system (which provides grid-wide property interpolation) and the well-specific `InjectedFluid` and `ProducedFluid` classes (which describe what a well puts into or takes out of the reservoir).

You will encounter `Fluid` most often in two situations. First, when you want to model a specific injected gas, such as CO2 or nitrogen, that has different PVT properties from the reservoir's native hydrocarbon gas. Second, when you are running a gas well or gas injection scheme and want to use pseudo-pressure to improve the accuracy of near-well flow calculations for compressible gas.

This page explains what `Fluid` is, when you need it, and how to use it in practice.

---

## What Is a Fluid?

A `Fluid` is a lightweight, immutable object that bundles three things together: a human-readable name, a phase identity (`FluidPhase.OIL`, `FluidPhase.GAS`, or `FluidPhase.WATER`), and optionally one or both of a `PVTTable` and a `PseudoPressureTable`.

The phase identity tells BORES how to interpret the PVT data and which correlations to use as fallbacks. The PVT table, if provided, overrides those correlations for the specific fluid's properties. The pseudo-pressure table, which is only meaningful for gas-phase fluids, enables more accurate gas well modeling in high-permeability or high-rate situations where the linear pressure approximation breaks down.

```python
from bores.fluids import Fluid
from bores.types import FluidPhase

# The simplest possible Fluid - just a name and phase
# Properties fall back to built-in correlations using the reservoir_gas identity
methane = Fluid(name="Methane", phase=FluidPhase.GAS)
```

At its most basic, `Fluid` is just a named phase label. Its power comes from the optional PVT and pseudo-pressure tables attached to it.

---

## When Do You Need Fluid?

For most conventional reservoir simulations, you do not need to construct `Fluid` objects directly. The `reservoir_model()` factory handles default gas characterization internally through its `reservoir_gas` parameter, and `PVTTables` handles grid-wide PVT interpolation at the simulation level.

You need `Fluid` when:

- You are injecting a non-hydrocarbon gas (CO2, nitrogen, flue gas) and that gas has meaningfully different PVT properties from the reservoir's native gas.
- You are running a gas production or injection well and want to use pseudo-pressure for well index calculations.
- You are building a `WellFluid` (via `InjectedFluid` or `ProducedFluid`) and want to attach phase-specific PVT behavior to the fluid flowing through that specific well.

---

## Creating a Fluid with a PVT Table

The most common reason to create a `Fluid` is to attach measured or equation-of-state PVT data to a specific gas. This is especially important for CO2 injection, where CO2's PVT behavior (high density near the critical point, strong viscosity variation with pressure) differs substantially from typical natural gas correlations.

```python
import numpy as np
import bores
from bores.fluids import Fluid
from bores.types import FluidPhase
from bores.tables.pvt import build_gas_pvt_data, PVTTable

bores.use_32bit_precision()

# Define pressure and temperature grid for CO2 PVT
pressures = np.linspace(500, 8000, 80)      # psi - cover supercritical range
temperatures = np.linspace(80, 300, 30)     # F

# Build PVT data for CO2 using its specific gravity (air = 1)
co2_pvt_data = build_gas_pvt_data(
    pressures=pressures,
    temperatures=temperatures,
    gas_gravity=1.52,       # CO2 specific gravity relative to air
)

# Wrap in a PVTTable to build the interpolators
co2_pvt_table = PVTTable(
    data=co2_pvt_data,
    interpolation_method="linear",
)

# Create the Fluid object
co2 = Fluid(
    name="CO2",
    phase=FluidPhase.GAS,
    pvt_table=co2_pvt_table,
)

# Query CO2 properties at reservoir conditions
p, T = 3000.0, 150.0
print(f"CO2 viscosity at {p} psi, {T} F: {co2.pvt_table.viscosity(p, T):.4f} cP")
print(f"CO2 FVF:                          {co2.pvt_table.formation_volume_factor(p, T):.4f} ft3/SCF")
print(f"CO2 Z-factor:                     {co2.pvt_table.compressibility_factor(p, T):.4f}")
```

You can also load a pre-built `PVTTable` from a saved file, which is useful if you generated your CO2 PVT from an external EOS package:

```python
from bores.tables.pvt import PVTTable

co2_pvt_table = PVTTable.load("data/co2_pvt.h5")
co2 = Fluid(name="CO2", phase=FluidPhase.GAS, pvt_table=co2_pvt_table)
```

---

## Pseudo-Pressure

For gas reservoirs, the standard Darcy flow equation assumes that gas viscosity and compressibility are constant with pressure, which is a reasonable approximation at low pressures but breaks down significantly at higher pressures. The pseudo-pressure function $m(P)$ absorbs this pressure dependence into a single integral, giving more accurate flow calculations particularly near wells:

$$m(P) = 2 \int_{P_0}^{P} \frac{P}{\mu_g(P) \cdot Z(P)} dP$$

where $\mu_g$ is gas viscosity and $Z$ is the compressibility factor, both functions of pressure. When you use pseudo-pressure, BORES replaces the standard pressure difference $(P - P_{wf})$ driving term in the well equation with the pseudo-pressure difference $m(P) - m(P_{wf})$, which accounts for the nonlinear gas compressibility correctly.

### Building a Pseudo-Pressure Table

You can build a pseudo-pressure table three ways: let BORES build it automatically from the `Fluid`'s PVT table, from a global `PVTTables` bundle, or from correlations using the gas gravity.

The recommended approach is to attach a PVT table to your `Fluid` and then call `get_pseudo_pressure_table()`. BORES builds and caches the table automatically:

```python
from bores.fluids import Fluid
from bores.types import FluidPhase
from bores.tables.pvt import PVTTable

# Fluid with a PVT table attached
reservoir_gas = Fluid(
    name="ReservoirGas",
    phase=FluidPhase.GAS,
    pvt_table=PVTTable.load("data/gas_pvt.h5"),
)

# Build the pseudo-pressure table at reservoir temperature
# BORES uses the Z-factor and viscosity interpolators from pvt_table
ppt = reservoir_gas.get_pseudo_pressure_table(
    temperature=185.0,                          # F - your reservoir temperature
    pressure_range=(200.0, 8000.0),             # psi - bracket your expected range
    points=300,                                 # integration grid density
)

# Query the pseudo-pressure at a specific pressure
m_at_3000 = ppt.pseudo_pressure(3000.0)
print(f"m(3000 psi) = {m_at_3000:.2e} psi2/cP")
```

If you do not have a PVT table for the fluid, you can build pseudo-pressure from correlations by providing the gas gravity and molecular weight via a `WellFluid` subclass:

```python
from bores.fluids import Fluid
from bores.types import FluidPhase

# Without a PVT table, BORES falls back to DAK Z-factor and Lee-Kesler viscosity
# The specific_gravity and molecular_weight attributes come from WellFluid subclasses
# (InjectedFluid, ProducedFluid) that inherit from Fluid
```

!!! info "Pseudo-Pressure and Well Modeling"
    Pseudo-pressure is most important for high-rate gas wells, wells with large pressure drawdowns, and reservoirs where gas viscosity varies significantly across the pressure range of interest (roughly 500 to 5,000 psi). For low-rate wells or when the pressure drawdown is small relative to average reservoir pressure, the linear approximation is adequate and you can skip pseudo-pressure.

### Providing a Pre-Built Table

If you already have a `PseudoPressureTable` from an external calculation, you can attach it directly to the `Fluid` to bypass automatic construction:

```python
from bores.fluids import Fluid
from bores.types import FluidPhase
from bores.tables.pseudo_pressure import PseudoPressureTable

# Load a pre-built pseudo-pressure table
ppt = PseudoPressureTable.load("data/pseudo_pressure.h5")

reservoir_gas = Fluid(
    name="ReservoirGas",
    phase=FluidPhase.GAS,
    pseudo_pressure_table=ppt,      # Attach directly - bypasses automatic construction
)
```

When `pseudo_pressure_table` is set on the `Fluid`, `get_pseudo_pressure_table()` returns it immediately without building anything.

!!! warning "Gas Phase Restriction"
    `pseudo_pressure_table` can only be set on gas-phase fluids. Attempting to attach a pseudo-pressure table to an oil or water `Fluid` raises a `ValidationError` immediately at construction time.

---

## How `get_pseudo_pressure_table()` Resolves Properties

When you call `get_pseudo_pressure_table()`, BORES follows a priority order to find Z-factor and viscosity functions:

1. If `pseudo_pressure_table` is already set on the `Fluid`, return it immediately.
2. If `pvt_table` on the `Fluid` has a `compressibility_factor` interpolator, use it for Z-factor.
3. If `pvt_table` on the `Fluid` has a `viscosity` interpolator, use it for viscosity.
4. If either function is still missing and you passed a `pvt_tables` argument (a global `PVTTables` bundle), check that bundle's gas slot for the missing function.
5. Fall back to correlations (DAK Z-factor, Lee-Kesler viscosity) using `specific_gravity` and `molecular_weight` from the `Fluid` subclass.

If none of these sources can supply both Z-factor and viscosity, a `ValidationError` is raised. The error message tells you exactly which property is missing and how to supply it.

```python
# Pass a global PVTTables bundle as fallback
from bores.tables.pvt import PVTTables

global_pvt = PVTTables.load("run/pvt_tables.h5")

# The Fluid has no pvt_table of its own, so it falls back to global_pvt.gas
lean_gas = Fluid(name="LeanGas", phase=FluidPhase.GAS)
ppt = lean_gas.get_pseudo_pressure_table(
    temperature=175.0,
    pvt_tables=global_pvt,          # Provide global bundle as fallback source
)
```

---

## Caching

`get_pseudo_pressure_table()` caches results by default. Calling it twice with the same arguments returns the same table object without recomputing. The cache key includes the fluid name, phase, temperature, pressure range, number of integration points, and a fingerprint of the PVT table bounds and interpolation method. This means you can call `get_pseudo_pressure_table()` freely inside loops or per-well setup code without worrying about redundant computation.

To bypass the cache (for example, when testing different integration densities), pass `use_cache=False`:

```python
ppt_fine = gas.get_pseudo_pressure_table(
    temperature=185.0,
    points=1000,
    use_cache=False,        # Always recompute
)
```

---

## Serialization

`Fluid` is fully serializable. You can save a `Fluid` (including its attached PVT and pseudo-pressure tables) to disk and reload it later:

```python
from bores.fluids import Fluid

# Save
co2.save("data/co2_fluid.h5")

# Load
co2_loaded = Fluid.load("data/co2_fluid.h5")
```

---

## Using Fluid with Wells

In practice, you will most often encounter `Fluid` as a parent class of `InjectedFluid` and `ProducedFluid`, which are the classes you actually attach to wells. These subclasses add well-specific attributes (specific gravity for the correlation fallback path, injection composition, etc.) on top of the base `Fluid` behavior.

See [Well Fluids](wells/fluids.md) for the full API on how to configure injected and produced fluids, and how to pass them to injection and production wells.

---

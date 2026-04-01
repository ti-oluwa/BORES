---
title: Quick Start - Build Your First Simulation - BORES
description: Build and run your first 3D waterflood reservoir simulation with BORES in under 5 minutes. Complete code walkthrough with grid setup, wells, and result visualization.
tags:
  - tutorial
  - beginner
  - reservoir-simulation
  - python
keywords: reservoir simulation tutorial, waterflood simulation, BORES quickstart, Python petroleum engineering, black-oil simulation, IMPES scheme
author: ti-oluwa
---

# Quick Start

Build and run your first reservoir simulation in under 5 minutes.

---

## Prerequisites

Before starting, make sure you have BORES installed and working. If you have not done this yet, follow the [Installation](installation.md) guide first. You can verify your installation by running:

```python
import bores
print(f"BORES version: {bores.__version__}")
```

You will also need basic familiarity with Python and NumPy. No prior reservoir engineering knowledge is required to follow this tutorial, though it will help you interpret the results.

---

## Your First Simulation

The following example sets up a complete 3D waterflood simulation from scratch. You will define a 10x10x3 reservoir grid, place an injection well and a production well, configure the simulation, and run it for one year. Every line is commented so you can follow along.

This is a single, self-contained script. You can copy it into a Python file or notebook and run it directly.

```python
import bores
import logging
import numpy as np

# Set log level
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Step 1: Set precision
# ---------------------------------------------------------------------------
# BORES defaults to 32-bit floating point for speed.
# You can switch to 64-bit with bores.use_64bit_precision() if you need
# higher numerical accuracy.
bores.use_32bit_precision()

# ---------------------------------------------------------------------------
# Step 2: Define the grid and initial property distributions
# ---------------------------------------------------------------------------
# A 10x10x3 grid means 10 cells in X, 10 in Y, 3 layers in Z.
# Each cell is 100 ft x 100 ft in the horizontal plane.
grid_shape = (10, 10, 3)
cell_dimension = (100.0, 100.0)  # (dx, dy) in feet

# Build uniform grids for each property.
# In a real study you would load heterogeneous data from files.
thickness = bores.build_uniform_grid(grid_shape, value=20.0)        # ft per layer
pressure = bores.build_uniform_grid(grid_shape, value=3000.0)       # psi
porosity = bores.build_uniform_grid(grid_shape, value=0.20)         # fraction
temperature = bores.build_uniform_grid(grid_shape, value=180.0)     # deg F
oil_viscosity = bores.build_uniform_grid(grid_shape, value=1.5)     # cP
bubble_point = bores.build_uniform_grid(grid_shape, value=2500.0)   # psi

# Residual and irreducible saturations
Sorw = bores.build_uniform_grid(grid_shape, value=0.20)  # Residual oil (waterflood)
Sorg = bores.build_uniform_grid(grid_shape, value=0.15)  # Residual oil (gas flood)
Sgr  = bores.build_uniform_grid(grid_shape, value=0.05)  # Residual gas
Swir = bores.build_uniform_grid(grid_shape, value=0.20)  # Irreducible water
Swc  = bores.build_uniform_grid(grid_shape, value=0.20)  # Connate water

# Build depth grid from thickness and a datum (top of reservoir at 5000 ft)
depth = bores.build_depth_grid(thickness, datum=5000.0)

# Build initial saturations from fluid contact depths.
# Place GOC above the reservoir and OWC below it so all cells
# are in the oil zone (undersaturated, no initial gas cap).
Sw, So, Sg = bores.build_saturation_grids(
    depth_grid=depth,
    gas_oil_contact=4999.0,      # Above reservoir top (no gas cap)
    oil_water_contact=5100.0,    # Below reservoir base (all oil zone)
    connate_water_saturation_grid=Swc,
    residual_oil_saturation_water_grid=Sorw,
    residual_oil_saturation_gas_grid=Sorg,
    residual_gas_saturation_grid=Sgr,
    porosity_grid=porosity,
)

# Oil specific gravity (needed for PVT correlations)
oil_sg = bores.build_uniform_grid(grid_shape, value=0.85)  # ~35 deg API

# Isotropic permeability: 100 mD in all directions
perm_grid = bores.build_uniform_grid(grid_shape, value=100.0)
permeability = bores.RockPermeability(x=perm_grid, y=perm_grid, z=perm_grid)

# ---------------------------------------------------------------------------
# Step 3: Build the reservoir model
# ---------------------------------------------------------------------------
# The reservoir_model() factory handles all PVT correlation calculations,
# grid validation, and internal property estimation automatically.
model = bores.reservoir_model(
    grid_shape=grid_shape,
    cell_dimension=cell_dimension,
    thickness_grid=thickness,
    pressure_grid=pressure,
    rock_compressibility=3e-6,            # psi⁻¹
    absolute_permeability=permeability,
    porosity_grid=porosity,
    temperature_grid=temperature,
    water_saturation_grid=Sw,
    gas_saturation_grid=Sg,
    oil_saturation_grid=So,
    oil_viscosity_grid=oil_viscosity,
    oil_specific_gravity_grid=oil_sg,
    oil_bubble_point_pressure_grid=bubble_point,
    residual_oil_saturation_water_grid=Sorw,
    residual_oil_saturation_gas_grid=Sorg,
    residual_gas_saturation_grid=Sgr,
    irreducible_water_saturation_grid=Swir,
    connate_water_saturation_grid=Swc,
    datum_depth=5000,
)

# ---------------------------------------------------------------------------
# Step 4: Define wells
# ---------------------------------------------------------------------------
# Injection well in corner cell (0,0) perforated across all 3 layers.
# Positive rate = injection.
injector = bores.injection_well(
    well_name="INJ-1",
    perforating_intervals=[((0, 0, 0), (0, 0, 2))],
    radius=0.25,  # ft
    control=bores.RateControl(
        target_rate=8000.0, # 8000 STB/day
        bhp_limit=5000,
        clamp=bores.InjectionClamp(),
    ),  
    injected_fluid=bores.InjectedFluid(
        name="Water",
        phase=bores.FluidPhase.WATER,
        specific_gravity=1.0,
        molecular_weight=18.015,
    ),
)

# Production well in opposite corner (9,9) perforated across all 3 layers.
# CoupledRateControl fixes the oil rate; water and gas flow naturally.
producer = bores.production_well(
    well_name="PROD-1",
    perforating_intervals=[((9, 9, 0), (9, 9, 2))],
    radius=0.25,  # ft
    control=bores.CoupledRateControl(
        primary_phase=bores.FluidPhase.OIL,
        primary_control=bores.AdaptiveRateControl(
            target_rate=-10_000.0,    # produce 10,000 STB/day of oil
            target_phase="oil",
            bhp_limit=1000.0,      # never drop below 1000 psi
            clamp=bores.ProductionClamp(),
        ),
        secondary_clamp=bores.ProductionClamp(),
    ),
    produced_fluids=[
        bores.ProducedFluid(
            name="Oil", phase=bores.FluidPhase.OIL,
            specific_gravity=0.85, molecular_weight=200.0,
        ),
        bores.ProducedFluid(
            name="Water", phase=bores.FluidPhase.WATER,
            specific_gravity=1.0, molecular_weight=18.015,
        ),
    ],
)

# Group wells together
wells = bores.wells_(injectors=[injector], producers=[producer])

# ---------------------------------------------------------------------------
# Step 5: Define rock-fluid properties
# ---------------------------------------------------------------------------
# Brooks-Corey relative permeability model with Corey exponents of 2.0.
# The capillary pressure model uses default Brooks-Corey parameters.
rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=bores.BrooksCoreyThreePhaseRelPermModel(
        water_exponent=2.0,
        oil_exponent=2.0,
        gas_exponent=2.0,
    ),
    capillary_pressure_table=bores.BrooksCoreyCapillaryPressureModel(),
)

# ---------------------------------------------------------------------------
# Step 6: Configure the simulation
# ---------------------------------------------------------------------------
# The `Time` helper converts human-readable durations to seconds.
config = bores.Config(
    timer=bores.Timer(
        initial_step_size=bores.Time(days=1),
        maximum_step_size=bores.Time(days=10),
        minimum_step_size=bores.Time(hours=1),
        simulation_time=bores.Time(days=365),
    ),
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",  # Implicit Pressure, Explicit Saturation
)

# ---------------------------------------------------------------------------
# Step 7: Run the simulation
# ---------------------------------------------------------------------------
# bores.run() returns a generator that yields ModelState objects.
# Each state is a snapshot of the reservoir at a specific time step.
states = list(bores.run(model, config))

# ---------------------------------------------------------------------------
# Step 8: Inspect results
# ---------------------------------------------------------------------------
final = states[-1]
print(f"Completed {final.step} steps in {final.time_in_days:.1f} days")
print(f"Final avg pressure: {final.model.fluid_properties.pressure_grid.mean():.1f} psi")
print(f"Final avg oil saturation: {final.model.fluid_properties.oil_saturation_grid.mean():.4f}")
print(f"Final avg water saturation: {final.model.fluid_properties.water_saturation_grid.mean():.4f}")
```

!!! note "First-run compilation"

    The very first time you run a BORES simulation, Numba compiles several internal functions to machine code. This one-time compilation typically takes 10-30 seconds. Subsequent runs reuse the cached compiled code and will be significantly faster.

---

## Understanding the Code

Now that you have a working simulation, let's walk through what each step does and why it matters.

### Setting Precision

BORES uses 32-bit floating point (`float32`) by default. This provides a good balance between speed and accuracy for most reservoir simulations, especially on modern hardware where 32-bit SIMD operations are twice as fast as 64-bit. The precision setting is global and affects all array operations throughout the simulation.

If you are working on a problem where numerical accuracy is critical, for example comparing against analytical solutions or running very long simulations where small errors accumulate, you can switch to 64-bit precision by calling `bores.use_64bit_precision()` before building your model. You can also use the `bores.with_precision()` context manager to temporarily change precision for a specific block of code.

### Building Property Grids

Every cell in the reservoir grid needs a set of physical properties: pressure, temperature, porosity, permeability, saturations, and so on. The `bores.build_uniform_grid()` helper creates a NumPy array of the specified shape filled with a single value. In a real study, you would typically load heterogeneous property distributions from geological models or well log data.

Residual saturations define the minimum amount of each phase that remains trapped in the rock after displacement. These values control the maximum recovery you can achieve and directly influence the shape of relative permeability curves. The irreducible water saturation (`Swir`) is the minimum water saturation achievable by oil drainage, while the connate water saturation (`Swc`) represents the initial water present when the reservoir was formed.

### Building Saturation Grids

The `bores.build_saturation_grids()` function creates physically realistic initial saturation distributions from fluid contact depths. You provide a depth grid (built from `bores.build_depth_grid()`), the gas-oil contact (GOC) and oil-water contact (OWC) depths, and the residual saturation grids. The function divides the reservoir into three zones based on depth:

- **Gas cap** (above GOC): Gas has displaced oil, leaving residual oil to gas displacement ($S_{org}$) and connate water ($S_{wc}$).
- **Oil zone** (between GOC and OWC): Original oil accumulation with connate water and residual gas ($S_{gr}$).
- **Water zone** (below OWC): Water has displaced oil, leaving residual oil to water displacement ($S_{orw}$).

In this example, the GOC is placed above the reservoir top and the OWC below the reservoir base. This means all cells fall in the oil zone, which is the correct initialization for an undersaturated reservoir with no gas cap and no aquifer. The resulting saturations satisfy the constraint $S_o + S_w + S_g = 1.0$ in every cell automatically. You can also enable smooth capillary transition zones at the contacts by passing `use_transition_zones=True`.

### The `reservoir_model()` Factory

The `bores.reservoir_model()` factory function is the primary entry point for constructing a reservoir model. When you call it, several things happen behind the scenes. First, it validates your input grids: checking that saturations are in valid ranges, that pressure and temperature are physically reasonable, and that the grid shape is consistent across all properties.

Second, it computes any fluid properties you did not provide explicitly. In this example, we supplied oil viscosity and bubble point pressure, but omitted properties like gas density, water compressibility, formation volume factors, and solution gas-oil ratio. The factory uses industry-standard PVT correlations (Standing, Vasquez-Beggs, Lee-Gonzalez, and others) to estimate these from pressure, temperature, and fluid gravity. This means you can get a working model with minimal input while retaining the option to override any property with your own data.

Third, it assembles the `FluidProperties`, `RockProperties`, and `SaturationHistory` objects and packages them into an immutable `ReservoirModel`. Because the model is immutable, you can safely pass it around your code without worrying about accidental modifications.

### Defining Wells

Wells are the primary mechanism for injecting fluid into or producing fluid from the reservoir. BORES uses factory functions - `bores.injection_well()` and `bores.production_well()` - to construct well objects with validated parameters.

Each well needs a name, one or more perforating intervals (defined as pairs of grid coordinates specifying where the well connects to the reservoir), a wellbore radius, a control mode, and a fluid specification. The perforating intervals in this example span from layer 0 to layer 2, meaning the well is open across all three layers.

!!! info "Sign Convention"

    BORES uses a consistent sign convention throughout the entire framework: **positive values mean injection** (fluid flowing into the reservoir) and **negative values mean production** (fluid flowing out). This applies to well rates, flux boundaries, and all internal flow calculations. When you specify `target_rate=-500.0` for a producer, the negative sign tells BORES this is a production rate.

The `CoupledRateControl` on the producer is the standard approach in reservoir simulation. You fix the oil rate (the primary phase) at -500 STB/day, and the simulator computes the BHP needed to deliver that rate. Water and gas then produce at their natural rates at the resulting BHP. The inner `AdaptiveRateControl` handles the automatic switch from rate to BHP mode when the well can no longer sustain the target rate without dropping below 1000 psi.

### Rock-Fluid Properties

The `BrooksCoreyThreePhaseRelPermModel` defines how easily each phase flows through the rock as a function of saturation. The Corey exponents control the curvature of the relative permeability curves: higher exponents produce steeper curves, meaning a phase needs higher saturation before it can flow significantly. An exponent of 2.0 is a common starting point for sandstone reservoirs.

The capillary pressure model (`BrooksCoreyCapillaryPressureModel`) accounts for the pressure difference between phases at the pore scale. Capillary forces are especially important in fine-grained rocks and at saturation fronts. The default parameters are reasonable for an initial study, but you should calibrate them against core flood data for any serious analysis.

These rock-fluid models are bundled into a `RockFluidTables` object, which the simulator queries at every time step to determine phase mobilities and capillary pressures at the current saturation state.

### Simulation Configuration

The `bores.Config` class gathers every simulation parameter into a single, immutable object. The `Timer` controls time stepping: it starts with a 1-day step, can grow to 10 days when conditions are stable, and can shrink to 1 hour when the solver needs smaller steps for convergence. The timer adapts automatically based on CFL conditions, pressure changes, and saturation changes.

The `scheme="impes"` setting selects the IMPES (Implicit Pressure, Explicit Saturation) evolution scheme, which solves pressure implicitly and then updates saturations explicitly. This is the default and most commonly used approach in black-oil simulation. It provides a good balance between stability (from implicit pressure) and efficiency (from explicit saturation transport).

The `Config` object is frozen after creation. If you need to modify a parameter, use `config.copy(timer=new_timer)` or `config.with_updates(scheme="explicit")` to create a new configuration with the desired changes.

### Running the Simulation

The `bores.run()` function accepts a `ReservoirModel` and a `Config` and returns a Python generator that yields `ModelState` objects. Each `ModelState` is a complete snapshot of the reservoir at a particular time step, including the updated model, well states, production and injection rates, relative permeabilities, and capillary pressures.

In this example, `list(bores.run(model, config))` collects all states into a list. For large simulations, you may want to process states one at a time to conserve memory, or use `bores.StateStream` to persist them to disk as they are generated.

The generator-based design means the simulation runs lazily. It only computes the next time step when you request the next state. This gives you full control over the simulation loop and lets you insert custom logic between steps, such as checking convergence criteria or modifying well controls.

---

## Streaming Results to Storage

For long-running simulations, collecting all states in memory is impractical. BORES provides `StateStream` to write states to disk as they are computed, keeping memory usage low. Here is how you would modify the run loop to use a HDF5 store:

```python
import bores

# ... (model and config setup from above, just after step 6) ...

# Create an HDF5 store to persist results
store = bores.HDF5Store("waterflood_results.h5")

with bores.StateStream(
    states=bores.run(model, config), 
    store=store, 
    background_io=True,
) as stream:
    for state in stream:
        # Each state is persisted to disk automatically
        if state.step % 10 == 0:
            print(f"Step {state.step}: P_avg = {state.model.fluid_properties.pressure_grid.mean():.1f} psi")
```

`StateStream` currently supports two storage backends `ZarrStore` and `HDF5Store`. Zarr is recommended for large simulations because it supports chunked, compressed storage and is efficient for both writing and later analysis.

---

## Basic Visualization

BORES includes Plotly-based visualization tools for quick inspection of results. The `make_series_plot()` function creates line plots from time-series data:

```python
import bores
import numpy as np

# ... (run the simulation and collect states as before, from step 7) ...

# Extract average pressure over time
time_days = np.array([s.time_in_days for s in states])
avg_pressure = np.array([
    s.model.fluid_properties.pressure_grid.mean() for s in states
])

# Create pressure vs time plot
pressure_series = np.column_stack([time_days, avg_pressure])
fig = bores.make_series_plot(
    data=pressure_series,
    title="Average Reservoir Pressure",
    x_label="Time (days)",
    y_label="Pressure (psi)",
)
fig.show()
```

!!! tip "Multiple Series"

    You can plot multiple series by passing a dictionary of named arrays:

    ```python
    avg_So = np.array([s.model.fluid_properties.oil_saturation_grid.mean() for s in states])
    avg_Sw = np.array([s.model.fluid_properties.water_saturation_grid.mean() for s in states])

    fig = bores.make_series_plot(
        data={
            "Oil Saturation": np.column_stack([time_days, avg_So]),
            "Water Saturation": np.column_stack([time_days, avg_Sw]),
        },
        title="Average Saturations",
        x_label="Time (days)",
        y_label="Saturation (fraction)",
    )
    fig.show()
    ```

---

## What's Next?

You now have a working simulation and a basic understanding of the BORES workflow. Here are the recommended next steps:

- **[Core Concepts](concepts.md)** - Understand the design principles behind BORES: immutable models, factory functions, generics, and the sign convention.
- **[User Guide](../user-guide/index.md)** - Deep dives into wells, boundary conditions, PVT correlations, relative permeability models, and solver configuration.
- **[Tutorials](../tutorials/index.md)** - Step-by-step walkthroughs of common workflows: depletion studies, waterflooding, gas injection, history matching, and more.
- **[API Reference](../api-reference/index.md)** - Complete documentation of every class, function, and parameter in the BORES package.

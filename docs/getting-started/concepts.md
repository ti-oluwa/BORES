# Core Concepts

Understand the foundational ideas behind BORES before diving into advanced features.

---

This page covers the design principles and conventions that run through every part of BORES. Reading it will save you time when you encounter unfamiliar patterns in the API and help you write simulations that are correct, efficient, and maintainable. Each concept below explains both the *what* and the *why*, so you can make informed decisions when building your own workflows.

---

## The Simulation Pipeline

Every BORES simulation follows the same five-stage pipeline: **Model Construction**, **Configuration**, **Simulation Run**, **State Streaming**, and **Analysis**. Understanding this pipeline is the key to understanding how the framework fits together.

In the first stage, you construct a `ReservoirModel` by defining the grid geometry, fluid properties, rock properties, and saturation distributions. The `reservoir_model()` factory function handles the heavy lifting: it runs PVT correlations, validates inputs, builds derived property grids, and assembles everything into an immutable model object. You can provide as few or as many properties as you like, and the factory will estimate the rest from correlations.

In the second stage, you create a `Config` object that specifies how the simulation should run: the time stepping strategy, the evolution scheme, well definitions, boundary conditions, solver settings, and convergence tolerances. The config is also immutable, ensuring that your simulation parameters are locked in before the run begins.

The third stage is the simulation run itself. `bores.run(model, config)` returns a generator that yields `ModelState` snapshots at each output interval. The generator-based design means computation happens lazily - the simulator only advances when you request the next state. The fourth stage, state streaming, optionally pipes these states to a storage backend (Zarr or HDF5) as they are produced, keeping memory usage bounded. The fifth stage is post-simulation analysis, where you compute recovery factors, plot production profiles, track saturation fronts, and extract engineering insights from the stored states.

```mermaid
flowchart LR
    A["Model Construction"] --> B["Configuration"]
    B --> C["Simulation Run"]
    C --> D["State Streaming"]
    D --> E["Analysis"]

    style A fill:#4CAF50,color:#fff,stroke:#388E3C
    style B fill:#2196F3,color:#fff,stroke:#1565C0
    style C fill:#FF9800,color:#fff,stroke:#EF6C00
    style D fill:#9C27B0,color:#fff,stroke:#6A1B9A
    style E fill:#F44336,color:#fff,stroke:#C62828
```

!!! info "Lazy Evaluation"

    The simulation generator does not precompute all states up front. Each call to `next()` advances the simulation by one output interval, solves the pressure and saturation equations, and yields the resulting `ModelState`. This means you can stop the simulation early, inspect intermediate results, or insert custom logic between steps without wasting computation.

---

## Immutable Data Models

BORES represents *almost* all reservoir data using immutable (frozen) classes built with the [attrs](https://www.attrs.org/) library. The core model classes - `ReservoirModel`, `FluidProperties`, `RockProperties`, `SaturationHistory`, and `Config` - are all frozen. Once created, their fields cannot be changed in place.

Immutability matters deeply in simulation software. When you pass a `ReservoirModel` to `bores.run()`, the simulator works on internal copies of the data. Your original model object remains untouched, so you can safely reuse it for parameter sweeps, what-if scenarios, or debugging. There is no hidden state mutation that could silently corrupt your baseline.

When you need a modified version of an immutable object, you use `attrs.evolve()` to create a new instance with specific fields replaced. BORES surfaces this pattern through convenience methods like `config.copy()` and `config.with_updates()`. The original object is never modified; you always get a fresh copy with the changes applied.

```python
import bores
import attrs

# Original config
config = bores.Config(
    timer=bores.Timer(
        initial_step_size=bores.Time(days=1),
        maximum_step_size=bores.Time(days=10),
        minimum_step_size=bores.Time(hours=1),
        simulation_time=bores.Time(days=365),
    ),
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",
)

# Create a modified copy with a different scheme
explicit_config = config.with_updates(scheme="explicit")

# The original is unchanged
assert config.scheme == "impes"
assert explicit_config.scheme == "explicit"
```

!!! tip "Parameter Sweeps"

    Immutability makes parameter sweeps straightforward. Create a base config, then use `config.with_updates()` in a loop to generate variations. Each variation is independent, so you can run them in parallel without synchronization concerns.

---

## Factory Functions

BORES uses factory functions - `reservoir_model()`, `injection_well()`, `production_well()`, and `wells_()` - as the primary way to construct complex objects. You will rarely need to instantiate `ReservoirModel`, `InjectionWell`, or `ProductionWell` directly.

The reason for this design is that building a reservoir model involves far more than assigning values to fields. The `reservoir_model()` factory validates your inputs against physical constraints, estimates missing properties from PVT correlations (Standing, Vasquez-Beggs, Lee-Gonzalez, and others), resolves circular dependencies between oil compressibility and formation volume factor through iterative bootstrapping, normalizes saturations to sum to 1.0, builds depth and elevation grids, and applies fracture transmissibility modifications if fractures are defined. Doing all of this inside a constructor would make the class difficult to understand and test.

Factory functions also provide a clear separation between what you specify (the inputs you care about) and what gets computed (the derived properties you did not provide). This lets you start simple - provide just pressure, temperature, porosity, and a few other essentials - and progressively add detail as your study demands it. The factories handle the transition gracefully, using correlations for anything you omit and your explicit values for anything you provide.

The well factories follow the same principle. `injection_well()` and `production_well()` accept straightforward parameters (name, location, radius, control mode, fluids) and return fully constructed well objects. The `wells_()` function groups injectors and producers into a single `Wells` container that the simulator expects.

```python
import bores

# The factory estimates gas viscosity, density, compressibility, FVF, Rs, Rsw,
# and many other properties from pressure, temperature, and oil/gas gravity
model = bores.reservoir_model(
    grid_shape=(10, 10, 3),
    cell_dimension=(100.0, 100.0),
    thickness_grid=thickness,
    pressure_grid=pressure,
    rock_compressibility=3e-6,
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
)
```

!!! info "Override Any Property"

    If you have measured data or equation-of-state results for a specific property (for example, gas compressibility factor from lab PVT analysis), pass it directly to `reservoir_model()` as the corresponding `*_grid` parameter. The factory will use your value instead of computing it from correlations.

---

## Generics and Dimensions

BORES models are generic over a dimension type parameter called `NDimension`. This means the same classes - `ReservoirModel`, `FluidProperties`, `RockProperties`, wells, boundary conditions, and grid utilities - work across 1D, 2D, and 3D simulations without separate implementations for each dimensionality.

In practice, the dimension is determined by the shape of your grid. A `grid_shape` of `(100,)` creates a 1D model, `(10, 10)` creates a 2D model, and `(10, 10, 3)` creates a 3D model. The type system tracks dimensionality through type aliases: `OneDimension = Tuple[int]`, `TwoDimensions = Tuple[int, int]`, and `ThreeDimensions = Tuple[int, int, int]`. When you build a 3D model, the resulting `ReservoirModel[ThreeDimensions]` carries its dimensionality in the type signature.

This generic design has practical benefits. You can prototype and debug on a fast 1D or 2D model, then scale up to 3D for your final study without changing any of your analysis code. The property grids, well definitions, and boundary conditions all adapt to the dimensionality of the model they belong to.

The full 3D simulation pipeline (`bores.run()`) currently operates on `ThreeDimensions` models. Lower-dimensional models are primarily useful for model construction, property estimation, and analytical comparisons. Future releases will extend the full simulation loop to 1D and 2D natively.

```python
import bores

# 1D model for rapid prototyping (Not yet supported)
model_1d = bores.reservoir_model(grid_shape=(100,), ...)

# 3D model for production studies
model_3d = bores.reservoir_model(grid_shape=(50, 50, 10), ...)

# Type annotations reflect dimensionality
# model_1d: ReservoirModel[OneDimension]
# model_3d: ReservoirModel[ThreeDimensions]
```

---

## Units Convention

BORES uses **field units** throughout the entire codebase. If you work in field units from the start, you never need to convert anything. Here is the complete list of units used in BORES:

| Property | Unit | Symbol |
|---|---|---|
| Pressure | pounds per square inch | psi |
| Length / Thickness | feet | ft |
| Permeability | milliDarcy | mD |
| Viscosity | centipoise | cP |
| Liquid rate | stock-tank barrels per day | STB/day |
| Gas rate | standard cubic feet per day | SCF/day |
| Temperature | degrees Fahrenheit | deg F |
| Density | pounds mass per cubic foot | lbm/ft³ |
| Compressibility | per psi | psi⁻¹ |
| Formation volume factor (oil, water) | reservoir barrels per stock-tank barrel | bbl/STB |
| Formation volume factor (gas) | reservoir cubic feet per standard cubic foot | ft³/SCF |
| Solution GOR | standard cubic feet per stock-tank barrel | SCF/STB |
| Porosity | fraction | - |
| Saturation | fraction | - |
| Relative permeability | fraction | - |

The choice of field units is deliberate. The vast majority of reservoir engineering literature, well reports, and commercial simulator inputs use field units. By aligning with this convention, BORES reduces the friction of importing real-world data and comparing results against published benchmarks.

All PVT correlations, well models, and solver equations inside BORES assume field units. The conversion constants (such as the 5.615 ft³/bbl factor and the 0.001127 Darcy-to-field transmissibility constant) are embedded in the numerical kernels. You do not need to apply them yourself.

!!! warning "SI Units"

    BORES does **not** support SI or metric units natively. If your data is in SI units, you must convert to field units before passing values to BORES functions. Mixing unit systems will produce incorrect results without any error or warning.

---

## Sign Convention

BORES uses a single, consistent sign convention for all flow quantities across the entire framework:

- **Positive** = injection / inflow (fluid entering the reservoir)
- **Negative** = production / outflow (fluid leaving the reservoir)

This convention applies everywhere: well rates, flux boundary conditions, source/sink terms, and all internal flow calculations. It is the most common convention in reservoir simulation and matches what you will find in textbooks like Aziz and Settari.

When defining a production well, specify a negative target rate. When defining an injection well, specify a positive target rate. The `ModelState` objects yielded by the simulator report production rates as positive values (the negation is applied automatically for readability), but the internal sign convention is always as described above.

```python
import bores

# Injection well: positive rate means fluid flows INTO the reservoir
injector_control = bores.RateControl(
    target_rate=500.0,  # +500 STB/day
    clamp=bores.InjectionClamp(),
)

# Production well: fix the oil rate, let other phases flow naturally
producer_control = bores.CoupledRateControl(
    primary_phase=bores.FluidPhase.OIL,
    primary_control=bores.AdaptiveRateControl(
        target_rate=-500.0,       # -500 STB/day of oil (production)
        target_phase="oil",
        bhp_limit=1000.0,         # minimum BHP constraint
        clamp=bores.ProductionClamp(),
    ),
    secondary_clamp=bores.ProductionClamp(),
)
```

!!! danger "Getting the Sign Wrong"

    If you accidentally specify a positive rate for a production well, the well will behave as an injector. If you specify a negative rate for an injection well, it will behave as a producer. BORES does not currently validate the sign against the well type, so always double-check your rate signs when setting up wells.

---

## Precision Control

BORES defaults to 32-bit floating point (`numpy.float32`) for all computations. This is a deliberate performance choice: 32-bit operations use half the memory of 64-bit and are roughly twice as fast for SIMD vectorized calculations. For most reservoir simulations, 32-bit precision provides more than adequate accuracy.

You can switch to 64-bit precision globally by calling `bores.use_64bit_precision()` before constructing your model. This is recommended when you need to match analytical solutions to many significant digits, when running very long simulations (thousands of time steps) where rounding errors can accumulate, or when working with extreme property contrasts (such as permeability ratios exceeding 10^6).

The `bores.with_precision()` context manager lets you temporarily change precision for a specific block of code without affecting the rest of your program. This is useful for running a high-precision validation check alongside a normal-precision production run.

```python
import bores

# Global precision setting
bores.use_64bit_precision()

# Or temporary precision change
with bores.with_precision("float64"):
    model = bores.reservoir_model(...)
    states = list(bores.run(model, config))

# Check current precision
print(bores.get_dtype())  # <class 'numpy.float32'> (back to default outside context)
```

!!! tip "When to Use 64-bit"

    A practical rule of thumb: start with 32-bit (the default). If you observe pressure oscillations, material balance errors exceeding 1%, or results that differ significantly from reference solutions, try switching to 64-bit precision. In most cases, the issue is more likely to be a modeling error or solver configuration problem, but ruling out precision effects is a quick diagnostic step.

---

## Serialization and Registration

BORES provides a two-tier serialization system for saving and loading simulation objects. The base tier, `Serializable`, supports dictionary round-tripping via the [cattrs](https://cattrs.readthedocs.io/) library. Any `Serializable` class can be converted to a plain Python dictionary and reconstructed from one. The second tier, `StoreSerializable`, extends this with `to_store(...)`, `from_store(...)`, `to_file(...)`, and `from_file(...)` methods that write to and read from file-backed storage (HDF5, Zarr, YAML, JSON).

All core BORES classes - `ReservoirModel`, `Config`, `FluidProperties`, `RockProperties`, well classes, boundary conditions, and relative permeability models - are serializable out of the box. You can save a model to disk, load it back, and get an identical object.

When you define custom types that should participate in BORES serialization (for example, a custom relative permeability model or a custom well control), you need to register them using the provided decorators. This ensures that the deserializer knows how to reconstruct your custom type from its dictionary representation. The registration system uses a type registry that maps string type identifiers to Python classes.

```python
import bores

# Save a model to HDF5
model.to_file("my_model.h5")

# Load it back
loaded_model = bores.ReservoirModel.from_file("my_model.h5")

# Save a config to YAML
config.to_file("config.yaml") # or config.save("config.yaml")

# Load it back
loaded_config = bores.Config.from_file("config.yaml")
```

!!! note "Storage Backends"

    BORES supports four storage backends: `ZarrStore` (chunked, compressed arrays - best for large simulations), `HDF5Store` (widely supported scientific format), `JSONStore` (human-readable, good for configs), and `YAMLStore` (human-readable, good for configs). Choose based on your data size and interoperability needs.

---

## Configuration as Code

The `Config` class is a frozen attrs class that holds every parameter controlling a simulation run. Rather than scattering configuration across multiple files, environment variables, or global state, BORES puts everything in one place. This makes simulations reproducible: given the same `ReservoirModel` and `Config`, you will get the same results every time.

The `Config` object includes the time stepping strategy (via `Timer`), well definitions, boundary conditions, rock-fluid tables, solver selection, preconditioner choice, convergence tolerances, evolution scheme, CFL thresholds, maximum saturation and pressure change limits, and many other parameters. Every parameter has a sensible default, so you only need to specify the ones you want to change.

Because `Config` is immutable, changing a parameter requires creating a new config. The `copy()` and `with_updates()` methods make this concise. This pattern eliminates an entire class of bugs where simulation behavior changes unexpectedly because someone modified a shared config object.

```python
import bores

Time = bores.Time

config = bores.Config(
    # Time stepping
    timer=bores.Timer(
        initial_step_size=Time(days=1),
        maximum_step_size=Time(days=10),
        minimum_step_size=Time(hours=1),
        simulation_time=Time(days=365),
    ),

    # Rock-fluid properties
    rock_fluid_tables=bores.RockFluidTables(
        relative_permeability_table=bores.BrooksCoreyRelPermModel(
            water_exponent=2.0, oil_exponent=2.0, gas_exponent=2.0,
        ),
        capillary_pressure_table=bores.BrooksCoreyCapillaryPressureModel(),
    ),

    # Wells
    wells=wells,

    # Solver settings
    scheme="impes",
    pressure_solver="bicgstab",
    pressure_preconditioner="ilu",
    pressure_convergence_tolerance=1e-6,
    maximum_solver_iterations=250,

    # Stability controls
    maximum_pressure_change=100.0,           # psi per step
    maximum_oil_saturation_change=0.5,       # fraction per step
    maximum_water_saturation_change=0.4,     # fraction per step
)
```

!!! example "Common Config Variations"

    === "Fast Screening Run"

        ```python
        screening_config = config.with_updates(
            output_frequency=10,             # Output every 10th step
            maximum_pressure_change=200.0,       # Relax stability constraints
            pressure_preconditioner="diagonal",  # Cheap preconditioner
        )
        ```

    === "High-Accuracy Study"

        ```python
        accurate_config = config.with_updates(
            pressure_convergence_tolerance=1e-8,
            maximum_pressure_change=50.0,        # Tighter stability constraints
            maximum_water_saturation_change=0.2,
            pressure_preconditioner="cpr",   # Strong preconditioner
        )
        ```

    === "Explicit Scheme"

        ```python
        explicit_config = config.with_updates(
            scheme="explicit",
            cfl_threshold=0.6,    # CFL stability limit
            pressure_cfl_threshold=0.9,
        )
        ```

---

## Next Steps

With these concepts in hand, you are ready to explore the rest of BORES:

- **[User Guide](../user-guide/index.md)** - Detailed coverage of wells, boundary conditions, PVT correlations, relative permeability models, solvers, and time stepping strategies.
- **[Tutorials](../tutorials/index.md)** - End-to-end walkthroughs of common reservoir simulation workflows.
- **[API Reference](../api-reference/index.md)** - Complete documentation of every class, function, and parameter.

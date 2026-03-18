---
title: Full API Reference - BORES Documentation
description: Complete API reference for BORES reservoir simulation framework. Includes all classes, functions, constants, PVT correlations, well controls, solvers, and configuration options.
tags:
  - reference
  - advanced
  - python
  - reservoir-simulation
keywords: BORES API, Python reservoir simulator API, PVT correlations API, well controls API, solver configuration, IMPES API reference, black-oil API
author: ti-oluwa
---

# Full API Reference

## Overview

This page lists every class, function, constant, and type exported from the `bores` package. Everything listed here is importable directly from `bores` (for example, `from bores import Config` or `import bores; bores.Config`). The API is organized by functional area to help you find what you need quickly.

The `bores` package uses wildcard re-exports from its internal modules. This means you never need to import from internal paths like `bores.wells.controls` or `bores.solvers.base`. Everything public is available at the top level. The only exception is the array PVT correlations, which must be imported from `bores.correlations.arrays` explicitly, and the visualization submodules (`bores.visualization.plotly1d`, `bores.visualization.plotly2d`, `bores.visualization.plotly3d`) which you can import by name for direct renderer access.

For detailed documentation of individual items, see the relevant sections in the [User Guide](../user-guide/index.md) and [Tutorials](../tutorials/index.md). For PVT correlation functions specifically, see the [Scalar Correlations](correlations-scalar.md) and [Array Correlations](correlations-array.md) pages.

---

## Model Construction

These are the main entry points for building reservoir models and wells. The factory functions handle PVT correlation computation, grid validation, and internal object construction for you.

### Factory Functions

| Name | Type | Description |
| --- | --- | --- |
| `reservoir_model()` | function | Build a `ReservoirModel` from raw grid data, fluid properties, and rock properties. Computes all derived PVT grids internally. |
| `injection_well()` | function | Build an `InjectionWell` with fluid, control, perforations, and schedule. |
| `production_well()` | function | Build a `ProductionWell` with fluid, control, perforations, and schedule. |
| `wells_()` | function | Combine multiple wells into a `Wells` collection for use in `Config`. |

```python
import bores
import numpy as np

# Build input grids (10x10x3 reservoir)
shape = (10, 10, 3)
thickness = bores.build_uniform_grid(shape, 20.0)
pressure = bores.build_uniform_grid(shape, 3000.0)
porosity = bores.build_uniform_grid(shape, 0.20)
temperature = bores.build_uniform_grid(shape, 200.0)
oil_viscosity = bores.build_uniform_grid(shape, 1.5)
bubble_point = bores.build_uniform_grid(shape, 2500.0)

model = bores.reservoir_model(
    grid_shape=shape,
    cell_dimension=(100.0, 100.0),
    thickness_grid=thickness,
    pressure_grid=pressure,
    rock_compressibility=3e-6,
    absolute_permeability=bores.RockPermeability(
        kx=bores.build_uniform_grid(shape, 100.0),
        ky=bores.build_uniform_grid(shape, 100.0),
        kz=bores.build_uniform_grid(shape, 10.0),
    ),
    porosity_grid=porosity,
    temperature_grid=temperature,
    water_saturation_grid=Sw,
    oil_saturation_grid=So,
    gas_saturation_grid=Sg,
    oil_viscosity_grid=oil_viscosity,
    oil_bubble_point_pressure_grid=bubble_point,
    residual_oil_saturation_water_grid=bores.build_uniform_grid(shape, 0.20),
    residual_oil_saturation_gas_grid=bores.build_uniform_grid(shape, 0.10),
    residual_gas_saturation_grid=bores.build_uniform_grid(shape, 0.05),
    irreducible_water_saturation_grid=bores.build_uniform_grid(shape, 0.20),
    connate_water_saturation_grid=bores.build_uniform_grid(shape, 0.20),
)
```

### Core Data Models

| Name | Type | Description |
| --- | --- | --- |
| `ReservoirModel` | class | Immutable model holding all grid data, fluid properties, rock properties, and saturation state. Generic over `NDimension`. |
| `FluidProperties` | class | Oil, gas, and water PVT properties at reservoir conditions. |
| `RockProperties` | class | Porosity, permeability, and compressibility. |
| `RockPermeability` | class | Directional permeability (kx, ky, kz). |
| `SaturationHistory` | class | Historical saturation data for hysteresis tracking. |

All data model classes are frozen `attrs` classes. You modify them using `attrs.evolve()` to create new instances with changed fields.

---

## Simulation

These classes and functions control how simulations are configured and executed.

### Configuration and Execution

| Name | Type | Description |
| --- | --- | --- |
| `Config` | class | Frozen configuration holding timer, wells, boundary conditions, solvers, convergence parameters, evolution scheme, and rock-fluid tables. |
| `Run` | class | Orchestrates the simulation loop. Yields `ModelState` objects per timestep. |
| `run()` | function | Main function to create and start a simulation run. Used by `Run` internally. Returns a generator of model states (`ModelState`). |

```python
config = bores.Config(
    timer=bores.Timer(
        simulation_time=bores.Time(days=365),
        initial_step_size=bores.Time(days=1),
        max_step_size=bores.Time(days=30),
        min_step_size=bores.Time(days=0.01),
    ),
    wells=wells,
    scheme="impes",
)

simulation = bores.run(model=model, config=config)
for step in simulation:
    print(f"Day {step.time:.1f}, Pressure: {step.model.pressure_grid.mean():.1f} psi")
```

### Time Management

| Name | Type | Description |
| --- | --- | --- |
| `Timer` | class | Simulation timer with start, end, and adaptive timestep control. |
| `Time()` | function | Convert time units (days, hours, years, etc.) to seconds. Used to specify `Timer` parameters in readable units. |
| `TimerState` | TypedDict | Current state of the timer during simulation (current time, step count, dt). |

### Model State

| Name | Type | Description |
| --- | --- | --- |
| `ModelState` | class | Snapshot of the model at a single timestep: pressure, saturations, rates, and well data. |
| `validate_state()` | function | Validate that a `ModelState` is physically consistent (saturations sum to 1, pressures positive, etc.). |

---

## Wells

Well types, fluid definitions, controls, and scheduling are all available at the top level.

### Well Types

| Name | Type | Description |
| --- | --- | --- |
| `Well` | class | Base well class. Generic over coordinates and fluid type. |
| `InjectionWell` | class | `Well` specialized for injection with `InjectedFluid`. |
| `ProductionWell` | class | `Well` specialized for production with `ProducedFluid`. |
| `Wells` | class | Collection type for multiple wells (used in `Config`). |
| `well_type()` | function | Decorator to registers a (custom) type of a well ("injection" or "production") for easy serialization/deserialization. |

### Well Fluids

| Name | Type | Description |
| --- | --- | --- |
| `WellFluid` | class | Base class for well fluid definitions. |
| `InjectedFluid` | class | Fluid being injected (phase, gravity, molecular weight, optional miscibility parameters, optional density/viscosity overrides). |
| `ProducedFluid` | class | Fluid being produced (typically just the phase). |

### Well Rate Computation

| Name | Type | Description |
| --- | --- | --- |
| `compute_well_index()` | function | Compute Peaceman well index from permeability, thickness, radii, and skin. |
| `compute_oil_well_rate()` | function | Compute oil production rate from well index, pressures, and mobility. |
| `compute_gas_well_rate()` | function | Compute gas production rate from well index, pressures, and mobility. |
| `compute_required_bhp_for_oil_rate()` | function | Back-calculate BHP needed to achieve a target oil rate. |
| `compute_required_bhp_for_gas_rate()` | function | Back-calculate BHP needed to achieve a target gas rate. |
| `compute_2D_effective_drainage_radius()` | function | Effective drainage radius for 2D grids. |
| `compute_3D_effective_drainage_radius()` | function | Effective drainage radius for 3D grids. |

### Well Controls

| Name | Type | Description |
| --- | --- | --- |
| `WellControl` | protocol | Base protocol for all well control strategies. |
| `BHPControl` | class | Constant bottom-hole pressure control. |
| `RateControl` | class | Constant surface rate control with optional BHP limit. |
| `CoupledRateControl` | class | Rate control on the primary phase with BHP fallback. |
| `MultiPhaseRateControl` | class | Control targeting total liquid or total fluid rate. |
| `AdaptiveRateControl` | class | Starts with rate control, switches to BHP when the rate target cannot be met. |
| `well_control()` | function | Decorator to register a (custom) well control for easy serialization/deserialization. |

### Rate Clamping

| Name | Type | Description |
| --- | --- | --- |
| `RateClamp` | protocol | Base protocol for rate clamping strategies. |
| `ProductionClamp` | class | Clamp that limits production rate to a maximum. |
| `InjectionClamp` | class | Clamp that limits injection rate to a maximum. |
| `rate_clamp()` | function | Decorator to register a (custom) rate clamp for easy serialization/deserialization. |

### Well Scheduling

| Name | Type | Description |
| --- | --- | --- |
| `WellSchedule` | class | Schedule of events for a single well. |
| `WellSchedules` | class | Collection of schedules for multiple wells. |
| `WellEvent` | class | A scheduled event (predicate + action pair). |
| `EventPredicate` | class | Condition that triggers an event. |
| `EventPredicates` | class | Collection of predicates (any/all logic). |
| `EventAction` | class | Action to take when an event triggers. |
| `EventActions` | class | Collection of actions to execute together. |
| `event_predicate()` | function | Factory to create an event predicate from a callable. |
| `event_action()` | function | Factory to create an event action from a callable. |
| `time_predicate()` | function | Create a predicate that triggers at a specific simulation time. |
| `update_well()` | function | Create an action that modifies well parameters at runtime. |

---

## Grid Construction

Functions for building the spatial grids that represent reservoir geometry and initial conditions.

### Grid Builders

| Name | Type | Description |
| --- | --- | --- |
| `build_uniform_grid()` | function | Create a uniform-valued grid of a given shape and value. |
| `build_depth_grid()` | function | Build a depth grid from layer thicknesses and a datum depth. |
| `build_elevation_grid()` | function | Build an elevation grid (increasing upward, opposite of depth). |
| `build_layered_grid()` | function | Build a grid with different values per layer (for heterogeneous properties). |
| `build_saturation_grids()` | function | Build physically consistent three-phase saturation grids from fluid contact depths (GOC, OWC), depth grid, and residual saturations. |

### Grid Aliases

These are aliases that call the same underlying functions. Prefer the `build_*` versions.

| Name | Type | Description |
| --- | --- | --- |
| `uniform_grid()` | function | Alias for `build_uniform_grid()`. |
| `depth_grid()` | function | Alias for `build_depth_grid()`. |
| `elevation_grid()` | function | Alias for `build_elevation_grid()`. |
| `layered_grid()` | function | Alias for `build_layered_grid()`. |
| `array()` | function | Alias for creating a grid array with the current precision dtype. |

### Grid Operations

| Name | Type | Description |
| --- | --- | --- |
| `pad_grid()` | function | Add ghost cells around a grid for boundary condition handling. |
| `unpad_grid()` | function | Remove ghost cells from a padded grid. |
| `get_pad_mask()` | function | Get a boolean mask indicating which cells are ghost cells. |
| `coarsen_grid()` | function | Reduce grid resolution by averaging cells. |
| `coarsen_permeability_grids()` | function | Coarsen permeability grids using harmonic averaging (correct for flow). |
| `flatten_multilayer_grid_to_surface()` | function | Collapse a 3D grid to a 2D surface map (e.g., for visualization). |
| `apply_structural_dip()` | function | Apply a structural dip angle to a depth grid. |

### Grid Data Containers

| Name | Type | Description |
| --- | --- | --- |
| `CapillaryPressureGrids` | class | Container for oil-water and gas-oil capillary pressure grids. |
| `RateGrids` | class | Container for per-cell rate grids (oil, water, gas). |
| `RelativeMobilityGrids` | class | Container for relative mobility grids used in flow calculations. |

---

## Rock-Fluid Properties

Models for relative permeability and capillary pressure.

### Relative Permeability Models

| Name | Type | Description |
| --- | --- | --- |
| `BrooksCoreyThreePhaseRelPermModel` | class | Three-phase relative permeability using Brooks-Corey (Corey) exponents for oil, water, and gas. |
| `LETThreePhaseRelPermModel` | class | Three-phase relative permeability using LET (Lomeland-Ebeltoft-Thomas) three-parameter correlation. More flexible than Brooks-Corey for matching laboratory data. |
| `LETParameters` | class | Frozen container for a single set of LET parameters (L, E, T). Used to group the three parameters when constructing `LETThreePhaseRelPermModel`. |
| `TwoPhaseRelPermTable` | class | Tabular two-phase relative permeability (oil-water or gas-oil). |
| `ThreePhaseRelPermTable` | class | Tabular three-phase relative permeability. |

### Relative Permeability Table Registry

| Name | Type | Description |
| --- | --- | --- |
| `relperm_table()` | decorator | Register a custom relative permeability table function. |
| `get_relperm_table()` | function | Retrieve a registered relative permeability table by name. |
| `list_relperm_tables()` | function | List all registered relative permeability table names. |

### Three-Phase Mixing Rules

These functions compute oil relative permeability in three-phase flow from two-phase (oil-water and gas-oil) curves.

| Name | Type | Description |
| --- | --- | --- |
| `mixing_rule()` | decorator | Register a custom three-phase mixing rule. |
| `stone_I_rule()` | function | Stone's first model. |
| `stone_II_rule()` | function | Stone's second model. |
| `baker_linear_rule()` | function | Baker's linear interpolation. |
| `eclipse_rule()` | function | Eclipse-style modified Stone I. |
| `blunt_rule()` | function | Blunt's saturation-weighted model. |
| `arithmetic_mean_rule()` | function | Simple arithmetic average. |
| `geometric_mean_rule()` | function | Geometric average. |
| `harmonic_mean_rule()` | function | Harmonic average. |
| `max_rule()` | function | Maximum of oil-water and gas-oil curves. |
| `min_rule()` | function | Minimum of oil-water and gas-oil curves. |
| `aziz_settari_rule()` | function | Aziz and Settari's model. |
| `hustad_hansen_rule()` | function | Hustad-Hansen model. |
| `linear_interpolation_rule()` | function | Linear interpolation between two-phase curves. |
| `saturation_weighted_interpolation_rule()` | function | Saturation-weighted interpolation. |
| `product_saturation_weighted_rule()` | function | Product of saturation-weighted curves. |
| `compute_corey_three_phase_relative_permeabilities()` | function | Compute all three-phase kr values from Corey parameters directly. |
| `compute_let_three_phase_relative_permeabilities()` | function | Compute all three-phase kr values from LET parameters directly. |

### Capillary Pressure Models

| Name | Type | Description |
| --- | --- | --- |
| `BrooksCoreyCapillaryPressureModel` | class | Brooks-Corey capillary pressure model with entry pressure and pore-size distribution index. |
| `LeverettJCapillaryPressureModel` | class | Leverett J-function scaling for capillary pressure from dimensionless J(Sw). |
| `VanGenuchtenCapillaryPressureModel` | class | Van Genuchten model for capillary pressure (common in soil science and unconventional reservoirs). |
| `TwoPhaseCapillaryPressureTable` | class | Tabular two-phase capillary pressure (oil-water or gas-oil). |
| `ThreePhaseCapillaryPressureTable` | class | Tabular three-phase capillary pressure. |
| `capillary_pressure_table()` | decorator | Register a custom capillary pressure table function. |

---

## PVT Tables

Tabular property lookup as an alternative to correlations.

| Name | Type | Description |
| --- | --- | --- |
| `PVTTables` | class | Container for pressure-dependent PVT lookup tables (Bo, Bg, Bw, Rs, viscosities, densities, compressibilities). Supports linear and cubic interpolation. |
| `PVTTableData` | class | Single PVT property table: pressure array and corresponding property values. |
| `build_pvt_table_data()` | function | Build a `PVTTableData` from pressure and value arrays with validation. |
| `RockFluidTables` | class | Container for relative permeability and capillary pressure tables, used in `Config`. |
| `PseudoPressureTable` | class | Al-Hussainy real-gas pseudo-pressure lookup table for gas well deliverability calculations. |
| `build_pseudo_pressure_table()` | function | Build a pseudo-pressure table from gas properties over a pressure range. |

---

## Boundary Conditions

Types for specifying reservoir boundary behavior.

### Boundary Classes

| Name | Type | Description |
| --- | --- | --- |
| `Boundary` | enum | All available boundary directions. |
| `BoundaryCondition` | class | A boundary condition applied to a specific face or region. |
| `BoundaryConditions` | class | Collection of boundary conditions for all faces of the grid. |
| `GridBoundaryCondition` | class | Boundary condition applied to the grid with face and type specification. Validates periodic boundary pairing. |
| `BoundaryMetadata` | class | Metadata about a boundary (face, type, parameters). |

### Boundary Types

| Name | Type | Description |
| --- | --- | --- |
| `NoFlowBoundary` | class | Zero-flux boundary (default for all faces). |
| `ConstantBoundary` | class | Constant value (Dirichlet) boundary. |
| `DirichletBoundary` | class | Alias for constant-value boundary. |
| `NeumannBoundary` | class | Constant flux boundary. |
| `FluxBoundary` | class | Specified flux boundary (positive = injection, negative = production). |
| `RobinBoundary` | class | Mixed boundary combining value and flux conditions. |
| `PeriodicBoundary` | class | Periodic boundary connecting opposite faces. |
| `LinearGradientBoundary` | class | Linearly varying boundary value across a face. |
| `SpatialBoundary` | class | Spatially varying boundary specified by a function. |
| `TimeDependentBoundary` | class | Time-varying boundary condition. |
| `VariableBoundary` | class | Boundary that changes based on simulation state. |
| `ParameterizedBoundaryFunction` | class | Boundary defined by a parameterized function. |

### Aquifer Support

| Name | Type | Description |
| --- | --- | --- |
| `CarterTracyAquifer` | class | Carter-Tracy analytical aquifer model with Van Everdingen-Hurst dimensionless influx. Supports physical properties mode and calibrated constant mode. |
| `boundary_function()` | decorator | Register a custom boundary function. |

---

## Faults and Fractures

| Name | Type | Description |
| --- | --- | --- |
| `Fracture` | class | A fault or fracture defined by geometry and transmissibility modification. |
| `FractureGeometry` | class | Geometric specification of a fracture (orientation, extent, position). |
| `apply_fracture()` | function | Apply a single fracture to a transmissibility grid. |
| `apply_fractures()` | function | Apply multiple fractures to a transmissibility grid. |
| `vertical_sealing_fault()` | function | Factory for a vertical sealing fault (zero transmissibility). |
| `inclined_sealing_fault()` | function | Factory for an inclined sealing fault. |
| `damage_zone_fault()` | function | Factory for a fault with an enhanced-permeability damage zone. |
| `conductive_fracture_network()` | function | Factory for a network of conductive fractures. |
| `validate_fracture()` | function | Validate fracture geometry against the grid dimensions. |

---

## Solvers and Preconditioners

Linear solver and preconditioner infrastructure for the pressure equation.

### Solvers

| Name | Type | Description |
| --- | --- | --- |
| `solve_linear_system()` | function | Solve a sparse linear system using the configured solver and preconditioner. |
| `solver_func()` | decorator | Register a custom solver function. |
| `get_solver_func()` | function | Retrieve a registered solver by name. |
| `list_solver_funcs()` | function | List all registered solver names. |

### Preconditioners

| Name | Type | Description |
| --- | --- | --- |
| `CachedPreconditionerFactory` | class | Wraps a preconditioner factory with caching to avoid rebuilding every timestep. |
| `preconditioner_factory()` | decorator | Register a custom preconditioner factory. |
| `get_preconditioner_factory()` | function | Retrieve a registered preconditioner factory by name. |
| `list_preconditioner_factories()` | function | List all registered preconditioner factory names. |
| `build_ilu_preconditioner()` | function | Build an ILU(0) preconditioner. |
| `build_amg_preconditioner()` | function | Build an algebraic multigrid preconditioner. |
| `build_diagonal_preconditioner()` | function | Build a Jacobi (diagonal) preconditioner. |
| `build_block_jacobi_preconditioner()` | function | Build a block Jacobi preconditioner. |
| `build_polynomial_preconditioner()` | function | Build a polynomial preconditioner. |
| `build_cpr_preconditioner()` | function | Build a constrained pressure residual (CPR) preconditioner. |

### Evolution Internals

| Name | Type | Description |
| --- | --- | --- |
| `EvolutionResult` | class | Result of a single evolution step (pressure/saturation update). |
| `to_1D_index_interior_only()` | function | Convert 3D grid indices to 1D indices for the interior cells (excluding ghost cells). |
| `from_1D_index_interior_only()` | function | Convert 1D interior-only indices back to 3D grid indices. |

---

## Analysis

| Name | Type | Description |
| --- | --- | --- |
| `ModelAnalyst` | class | Post-simulation analysis engine. Computes recovery factors, production profiles, front tracking, mobility ratios, sweep efficiency, and more from stored simulation results. |

See the [Model Analysis](../user-guide/advanced/model-analysis.md) page for detailed usage.

---

## Data Streaming and Storage

### Streaming

| Name | Type | Description |
| --- | --- | --- |
| `StateStream` | class | Wraps a simulation generator and streams state snapshots to a storage backend. Supports checkpointing and resume. |
| `StreamProgress` | class | Progress statistics (yield count, saved count, checkpoints). |

### Storage Backends

| Name | Type | Description |
| --- | --- | --- |
| `HDF5Store` | class | HDF5 file storage backend (recommended for large simulations). |
| `ZarrStore` | class | Zarr storage backend (supports cloud storage and parallel I/O). |
| `JSONStore` | class | JSON file storage backend (human-readable, small simulations only). |
| `YAMLStore` | class | YAML file storage backend (human-readable, small simulations only). |
| `new_store()` | function | Create a new store from a file path (auto-detects backend from extension). |
| `data_store()` | decorator | Register a custom storage backend. |

---

## Serialization

The serialization system provides dictionary-based round-tripping for all BORES objects.

| Name | Type | Description |
| --- | --- | --- |
| `Serializable` | class | Base mixin that adds `to_dict()` and `from_dict()` methods to any attrs class. |
| `converter` | object | The cattrs converter instance configured with all BORES type hooks. |
| `dump()` | function | Serialize any `Serializable` object to a dictionary. |
| `load()` | function | Deserialize a dictionary back into the original object type. |

---

## Visualization

The visualization system provides Plotly-based plotting for 1D time series, 2D maps, and 3D volume rendering.

### Core Visualization

| Name | Type | Description |
| --- | --- | --- |
| `ColorScheme` | enum | Available color schemes for visualizations (includes colorblind-friendly options). |
| `PropertyMeta` | class | Metadata about a reservoir property (name, unit, colormap). |
| `PropertyRegistry` | class | Registry of known reservoir properties and their visualization defaults. |
| `property_registry` | object | The global property registry instance. |
| `image_config()` | function | Configure image export settings (format, resolution). |
| `merge_plots()` | function | Combine multiple Plotly figures into a single figure with subplots. |

### 1D Plotting

| Name | Type | Description |
| --- | --- | --- |
| `make_series_plot()` | function | Create a 1D time series plot from data dictionaries or arrays. Supports line, scatter, bar, and tornado plot types. |

For direct renderer access, import from `bores.visualization.plotly1d`:

| Name | Type | Description |
| --- | --- | --- |
| `plotly1d.DataVisualizer` | class | Main 1D visualization orchestrator. |
| `plotly1d.LineRenderer` | class | Line plot renderer. |
| `plotly1d.BarRenderer` | class | Bar chart renderer. |
| `plotly1d.ScatterRenderer` | class | Scatter plot renderer. |
| `plotly1d.TornadoRenderer` | class | Tornado (sensitivity) chart renderer. |

### 2D Plotting

Import from `bores.visualization.plotly2d`:

| Name | Type | Description |
| --- | --- | --- |
| `plotly2d.DataVisualizer` | class | Main 2D visualization orchestrator. |
| `plotly2d.HeatmapRenderer` | class | Heatmap renderer for areal property maps. |
| `plotly2d.ContourRenderer` | class | Contour plot renderer. |
| `plotly2d.ScatterRenderer` | class | 2D scatter plot renderer. |
| `plotly2d.LineRenderer` | class | 2D line plot renderer. |
| `plotly2d.SurfaceRenderer` | class | 3D surface from 2D data renderer. |

### 3D Plotting

Import from `bores.visualization.plotly3d`:

| Name | Type | Description |
| --- | --- | --- |
| `plotly3d.DataVisualizer` | class | Main 3D visualization orchestrator. |
| `plotly3d.VolumeRenderer` | class | Volume rendering for full 3D grids. |
| `plotly3d.IsosurfaceRenderer` | class | Isosurface extraction and rendering. |
| `plotly3d.CellBlockRenderer` | class | Individual cell block rendering. |
| `plotly3d.Scatter3DRenderer` | class | 3D scatter plot renderer. |
| `plotly3d.Labels` | class | Manages text labels and annotations in 3D scenes. |
| `plotly3d.Label` | class | A single text label with position and formatting info. |

### PyVista 3D Plotting

Import from `bores.visualization.pyvista3d`:

| Name | Type | Description |
| --- | --- | --- |
| `pyvista3d.DataVisualizer` | class | PyVista-based 3D visualization orchestrator. Supports volume rendering, isosurfaces, cell blocks, and scatter plots with interactive picking and animation. |
| `pyvista3d.VolumeRenderer` | class | PyVista volume renderer for full 3D grids. |
| `pyvista3d.IsosurfaceRenderer` | class | PyVista isosurface extraction and rendering. |
| `pyvista3d.CellBlockRenderer` | class | PyVista individual cell block renderer. |
| `pyvista3d.Scatter3DRenderer` | class | PyVista 3D scatter plot renderer. |
| `pyvista3d.PlotConfig` | class | Configuration for PyVista 3D plots (width, height, color scheme, opacity, camera, shading, notebook mode, etc.). |
| `pyvista3d.PlotType` | enum | PyVista plot types: `VOLUME`, `ISOSURFACE`, `SCATTER_3D`, `CELL_BLOCKS`. |
| `pyvista3d.Labels` | class | Manages text labels and annotations in PyVista 3D scenes. |
| `pyvista3d.Label` | class | A single text label with position and formatting. |
| `bores.GifExporter` | class | Export animations as animated GIF. |
| `bores.Mp4Exporter` | class | Export animations as MP4 video. |
| `bores.WebPExporter` | class | Export animations as animated WebP. |
| `bores.HtmlExporter` | class | Export animations as interactive HTML. |
| `pyvista3d.viz` | object | Global PyVista visualizer instance with default configuration. |

```python
from bores.visualization.pyvista3d import DataVisualizer, PlotConfig, PlotType

viz = DataVisualizer(config=PlotConfig(
    width=1200,
    height=960,
    plot_type=PlotType.CELL_BLOCKS,
    smooth_shading=True,
    enable_picking=True,
))

# Single plot from a model state
plotter = viz.make_plot(state, property="pressure", title="Reservoir Pressure")
plotter.show()

# Animation from simulation states
plotters = viz.animate(
    states,
    property="oil_saturation",
    step_size=5,
    save=bores.GifExporter("saturation.gif"),
)
```

---

## Precision Control

Functions for controlling the floating-point precision used throughout the simulator.

| Name | Type | Description |
| --- | --- | --- |
| `use_32bit_precision()` | function | Set global precision to float32 (default). Faster computation, lower memory. |
| `use_64bit_precision()` | function | Set global precision to float64. Higher accuracy for ill-conditioned problems. |
| `with_precision()` | context manager | Temporarily set precision within a `with` block. |
| `get_dtype()` | function | Get the currently active NumPy dtype. |
| `set_dtype()` | function | Set the global dtype directly. |
| `get_floating_point_info()` | function | Get machine epsilon and range information for the current dtype. |

```python
import bores

# Default is 32-bit
bores.use_64bit_precision()

# Or temporarily
with bores.with_precision("float64"):
    model = bores.reservoir_model(...)
```

---

## Types and Enums

Type aliases and enumerations used throughout the API.

### Enums

| Name | Type | Description |
| --- | --- | --- |
| `FluidPhase` | enum | Fluid phase: `OIL`, `WATER`, `GAS`. |
| `EvolutionScheme` | literal | Simulation scheme: `"impes"`, `"explicit"`, `"implicit"`. |
| `Orientation` | enum | Spatial orientation for fractures and boundaries. |
| `Wettability` | enum | Rock wettability: `WATER_WET`, `OIL_WET`, `MIXED_WET`. |
| `MiscibilityModel` | literal | Miscibility model type: `"immiscible"` or `"todd_longstaff"` |
| `MixingRule` | protocol | Three-phase mixing rule contract/protocol. |
| `WellFluidType` | literal | Well fluid type. `"oil"`, `"gas"` or `"water"`. |

### Dimension Types

| Name | Type | Description |
| --- | --- | --- |
| `NDimension` | TypeVar | Generic dimension parameter for models, wells, and grids. |
| `ThreeDimensions` | type alias | `Tuple[int, int, int]` for 3D cell coordinates. |
| `TwoDimensions` | type alias | `Tuple[int, int]` for 2D cell coordinates. |
| `OneDimension` | type alias | `Tuple[int]` for 1D cell coordinates. |

### Grid Types

| Name | Type | Description |
| --- | --- | --- |
| `ThreeDimensionalGrid` | type alias | NumPy array of shape `(nx, ny, nz)`. |
| `TwoDimensionalGrid` | type alias | NumPy array of shape `(nx, ny)`. |
| `OneDimensionalGrid` | type alias | NumPy array of shape `(nx,)`. |
| `ArrayLike` | protocol | Protocol for anything that can be converted to a NumPy array. |

### Function Types

| Name | Type | Description |
| --- | --- | --- |
| `Solver` | type alias | A linear system solver callable. |
| `SolverFunc` | type alias | A solver function signature. |
| `Preconditioner` | type alias | A preconditioner callable. |
| `Interpolator` | type alias | An interpolation function. |

### Property Types

| Name | Type | Description |
| --- | --- | --- |
| `CapillaryPressures` | TypedDict | Dictionary of capillary pressure arrays (Pcow, Pcgo). |
| `RelativePermeabilities` | TypedDict | Dictionary of relative permeability arrays (kro, krw, krg). |
| `RelativeMobilityRange` | TypedDict | Dictionary defining range of relative mobility values for each fluid phase. |
| `Range` | class | A `(min, max)` range for quantities. |

---

## Constants

The constants system provides named, documented physical constants with unit information.

| Name | Type | Description |
| --- | --- | --- |
| `Constants` | class | Container for all simulation constants. Access individual constants as attributes. |
| `Constant` | class | A single constant with value, unit, and description. |
| `ConstantsContext` | class | Context manager for temporarily modifying constant values. |
| `c` | object | The global `Constants` instance. Access constants as `bores.c.STANDARD_PRESSURE`, `bores.c.GAS_CONSTANT`, etc. |
| `get_constant()` | function | Retrieve a constant by name string. |

```python
import bores

print(bores.c.STANDARD_PRESSURE)      # 14.696 psi
print(bores.c.STANDARD_TEMPERATURE)   # 60.0 F
print(bores.c.GAS_CONSTANT)           # 10.7316 psi*ft³/(lbmol*R)
```

---

## Errors

All BORES exceptions inherit from `BORESError`. You can catch `BORESError` to handle any BORES-specific error, or catch specific subclasses for targeted handling.

| Name | Type | Description |
| --- | --- | --- |
| `BORESError` | exception | Base class for all BORES errors. |
| `ValidationError` | exception | Invalid input parameters or model configuration. |
| `SimulationError` | exception | Error during simulation execution. |
| `ComputationError` | exception | Numerical computation failure. |
| `SolverError` | exception | Linear solver failed to converge. |
| `PreconditionerError` | exception | Preconditioner construction or application failure. |
| `SerializationError` | exception | Error during object serialization. |
| `DeserializationError` | exception | Error during object deserialization. |
| `SerializableError` | exception | General serializable protocol error. |
| `StorageError` | exception | Storage backend read/write failure. |
| `StreamError` | exception | Data streaming error. |
| `TimingError` | exception | Timer configuration or state error. |
| `StopSimulation` | exception | Raised to halt simulation early (not an error, used for control flow). |

```python
import bores

try:
    model = bores.reservoir_model(porosity=-0.1, ...)
except bores.ValidationError as exc:
    print(f"Invalid model: {exc}")

try:
    for step in bores.run(model=model, config=config):
        pass
except bores.SolverError as exc:
    print(f"Solver failed: {exc}")
except bores.SimulationError as exc:
    print(f"Simulation error: {exc}")
```

---

## Utility Functions

Low-level array utility functions. These are primarily used internally but are available if you need them for custom computations.

| Name | Type | Description |
| --- | --- | --- |
| `apply_mask()` | function | Apply a boolean mask to an array, setting masked cells to a fill value. |
| `clip()` | function | Clip array values to a range (like `np.clip` but numba friendly). |
| `clip_scalar()` | function | Clip a scalar value to a range. |
| `get_mask()` | function | Get a boolean mask from a condition applied to an array. |
| `is_array()` | function | Check if a value is a NumPy array. |
| `max_()` | function | Returns maximum of a scalar/array (like `np.max` but numba friendly). |
| `min_()` | function | Returns mininmum of a scalar/array (like `np.min` but numba friendly). |

---

## PVT Correlations

Scalar and array PVT correlation functions are documented separately due to their size:

- **[Scalar Correlations](correlations-scalar.md)**: 60+ single-value functions for point PVT calculations. Use these for quick checks, building custom tables, or validating against lab data.
- **[Array Correlations](correlations-array.md)**: Vectorized equivalents that operate on entire grids in a single call. The simulator uses these internally, and you can use them for post-processing or custom property computations.

Both modules share the same function names and parameters. The only difference is that scalar functions accept `float` inputs and array functions accept `NDimensionalGrid` inputs.

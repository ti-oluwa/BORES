<p align="center">
  <img src="docs/images/logo.svg" alt="BORES Logo" width="200">
</p>

<h1 align="center">BORES</h1>

<p align="center">
  <strong>3D 3-Phase Black-Oil Reservoir Modelling and Simulation Framework</strong>
</p>

[![Documentation](https://img.shields.io/badge/docs-ti--oluwa.github.io%2Fbores-blue)](https://ti-oluwa.github.io/bores)
[![PyPI](https://img.shields.io/pypi/v/bores-framework)](https://pypi.org/project/bores-framework/)
[![License](https://img.shields.io/github/license/ti-oluwa/bores)](LICENSE)

BORES is a Python framework for 3D black-oil reservoir simulation of three-phase (oil, water, gas) flow in porous media. It provides a clean, modular API for building reservoir models, defining wells and fractures, running simulations, and analyzing results.

> **Disclaimer**: BORES is designed for **educational, research, and prototyping purposes**. It is not production-grade software and should not be used for critical business decisions or regulatory compliance. Results should be validated against established commercial simulators before any real-world application.

**Full documentation @** [https://ti-oluwa.github.io/BORES](https://ti-oluwa.github.io/BORES)

> **NOTE**: The latest release does not match the current docs. The release for the current docs is still in development and is almost ready.

## Installation

```bash
pip install bores-framework
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add bores-framework
```

## Quick Example

A 3D waterflood simulation: 10x10x3 grid, one injector, one producer, 24 years, IMPES scheme.

```python
import typing

import bores

# Set precision (32-bit is the default)
bores.use_32bit_precision()

# Grid dimensions: 10x10x3 cells, each 1000 ft x 1000 ft, 100 ft thick
grid_shape = typing.cast(bores.ThreeDimensions, (10, 10, 3))
cell_dimension = (1000.0, 1000.0)

# Build property grids
thickness = bores.build_uniform_grid(grid_shape, value=100.0)  # ft
pressure = bores.build_uniform_grid(grid_shape, value=3000.0)  # psi
porosity = bores.build_uniform_grid(grid_shape, value=0.20)  # fraction
temperature = bores.build_uniform_grid(grid_shape, value=180.0)  # deg F
oil_viscosity = bores.build_uniform_grid(grid_shape, value=1.5)  # cP
bubble_point = bores.build_uniform_grid(grid_shape, value=2500.0)  # psi
oil_specific_gravity = bores.build_uniform_grid(grid_shape, value=0.85)

# Residual and irreducible saturations
residual_oil_saturation_water = bores.build_uniform_grid(grid_shape, value=0.12)
residual_oil_saturation_gas = bores.build_uniform_grid(grid_shape, value=0.10)
residual_gas_saturation = bores.build_uniform_grid(grid_shape, value=0.05)
irreducible_water_saturation = bores.build_uniform_grid(grid_shape, value=0.06)
connate_water_saturation = bores.build_uniform_grid(grid_shape, value=0.06)

# Build depth grid and compute initial saturations from fluid contacts
depth = bores.build_depth_grid(thickness, datum=5000.0)  # Top at 5000 ft
Sw, So, Sg = bores.build_saturation_grids(
    depth_grid=depth,
    gas_oil_contact=5050.0,  # Above reservoir (no gas cap)
    oil_water_contact=5280.0,  # Below reservoir (all oil zone)
    connate_water_saturation_grid=connate_water_saturation,
    residual_oil_saturation_water_grid=residual_oil_saturation_water,
    residual_oil_saturation_gas_grid=residual_oil_saturation_gas,
    residual_gas_saturation_grid=residual_gas_saturation,
    porosity_grid=porosity,
)

# Isotropic permeability: 100 mD
perm_grid = bores.build_uniform_grid(grid_shape, value=100.0)
permeability = bores.RockPermeability(x=perm_grid)

# Build the reservoir model
model = bores.reservoir_model(
    grid_shape=grid_shape,
    cell_dimension=cell_dimension,
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
    oil_bubble_point_pressure_grid=bubble_point,
    residual_oil_saturation_water_grid=residual_oil_saturation_water,
    residual_oil_saturation_gas_grid=residual_oil_saturation_gas,
    residual_gas_saturation_grid=residual_gas_saturation,
    irreducible_water_saturation_grid=irreducible_water_saturation,
    connate_water_saturation_grid=connate_water_saturation,
    oil_specific_gravity_grid=oil_specific_gravity,
    datum_depth=5000.0,
)

# Define wells
injector = bores.injection_well(
    well_name="INJ-1",
    perforating_intervals=[((0, 0, 2), (0, 0, 2))],
    radius=0.25,
    control=bores.AdaptiveRateControl(
        target_rate=3000,
        bhp_limit=6000.0,
        clamp=bores.InjectionClamp(),
    ),
    injected_fluid=bores.InjectedFluid(
        name="Water",
        phase=bores.FluidPhase.WATER,
        specific_gravity=1.0,
        molecular_weight=18.015,
    ),
)
producer = bores.production_well(
    well_name="PROD-1",
    perforating_intervals=[((9, 9, 1), (9, 9, 1))],
    radius=0.25,
    control=bores.CoupledRateControl(
        primary_phase=bores.FluidPhase.OIL,  # Can be set to "oil" too
        primary_control=bores.AdaptiveRateControl(
            target_rate=-10000.0,
            target_phase="oil",
            bhp_limit=1000.0,
            clamp=bores.ProductionClamp(),
        ),
        secondary_clamp=bores.ProductionClamp(),
    ),
    produced_fluids=[
        bores.ProducedFluid(
            name="Oil",
            phase=bores.FluidPhase.OIL,
            specific_gravity=0.85,
            molecular_weight=200.0,
        ),
        bores.ProducedFluid(
            name="Water",
            phase=bores.FluidPhase.WATER,
            specific_gravity=1.0,
            molecular_weight=18.015,
        ),
    ],
)
wells = bores.wells_(injectors=[injector], producers=[producer])

# Rock-fluid tables (Brooks-Corey relative permeability + capillary pressure)
rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=bores.BrooksCoreyRelPermModel(
        water_exponent=2.0,
        oil_exponent=1.0,
        gas_exponent=1.0,
        wettability=bores.Wettability.WATER_WET,
        mixing_rule="eclipse_rule",
    ),
    capillary_pressure_table=bores.BrooksCoreyCapillaryPressureModel(
        wettability=bores.Wettability.WATER_WET,
    ),
)
timer = bores.Timer(
    initial_step_size=bores.Time(days=5),
    maximum_step_size=bores.Time(months=3),
    minimum_step_size=bores.Time(hours=1),
    simulation_time=bores.Time(years=24),
    maximum_rejections=20,
)

# Simulation configuration
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",
    pressure_solver="direct",
    pressure_preconditioner=None,
    maximum_pressure_change=1800,
)

# Run and monitor the simulation and collect states
states = list(bores.monitor(model, config))
final = states[-1][0]
print(f"Completed {final.step} steps in {final.time_in_days:.2f} days")
print(
    f"Final avg pressure: {final.model.fluid_properties.pressure_grid.mean():.1f} psi"
)
```

## Features

- 3D structured grid with uniform, layered, and heterogeneous property distributions
- Three-phase (oil, water, gas) black-oil PVT correlations (Standing, Vazquez-Beggs, Hall-Yarborough, and more)
- IMPES and Sequential Implicit evolution schemes
- Multiple linear solvers (BiCGSTAB, GMRES, CG, direct) with preconditioner support (ILU, AMG, CPR)
- Well models with BHP control, rate control, schedules, and event-driven actions
- Faults, fractures, and transmissibility modifications
- Boundary conditions including Carter-Tracy analytical aquifer
- Todd-Longstaff miscible flooding with pressure-dependent miscibility
- Relative permeability models (Brooks-Corey, LET, tabular) with 15+ three-phase mixing rules
- Capillary pressure models (Brooks-Corey, Leverett J-function, Van Genuchten, tabular)
- Plotly-based visualization (1D time series, 2D maps, 3D volume rendering)
- HDF5, Zarr, JSON, and YAML storage backends with serialization
- Post-simulation analysis (recovery factors, sweep efficiency, front tracking)

## Citing BORES

If you use BORES in academic work, please cite it as:

```bibtex
@software{bores,
  author = {Daniel Toluwalase Afolayan},
  title = {BORES: 3D 3-Phase Black-Oil Reservoir Modelling and Simulation Framework},
  year = {2026},
  url = {https://github.com/ti-oluwa/bores},
}
```

## Contributing

BORES was developed by a graduate petroleum engineer with just theoretical knowledge and research experience. The project does not have the benefit of decades of field experience backing its implementations, so contributions from practitioners and researchers are welcome.

**Reporting issues**: If you find bugs, inaccuracies in the physics, or unexpected behavior, please [open an issue](https://github.com/ti-oluwa/bores/issues) on GitHub with a clear description and, if possible, a minimal example that reproduces the problem.

**Improvements**: Pull requests for bug fixes, documentation improvements, and enhancements that fall within the scope of a black-oil reservoir simulation framework are welcome. Please keep changes focused and well-tested.

**Out of scope**: Changes that go beyond the black-oil formulation (compositional simulation, thermal recovery, etc.) are outside the current scope of the project.

## License

See [LICENSE](LICENSE) for details.

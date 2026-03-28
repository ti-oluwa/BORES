import logging
import typing

import bores

# Set log level
logging.basicConfig(level=logging.INFO)

# Set precision (32-bit is the default)
bores.use_32bit_precision()

# Grid dimensions: 10x10x3 cells, each 100 ft x 100 ft, 20 ft thick
grid_shape = typing.cast(bores.ThreeDimensions, (10, 10, 3))
cell_dimension = (1000.0, 1000.0)

# Build property grids
thickness = bores.build_uniform_grid(grid_shape, value=100.0)  # ft
pressure = bores.build_uniform_grid(grid_shape, value=3000.0)  # psi
porosity = bores.build_uniform_grid(grid_shape, value=0.20)  # fraction
temperature = bores.build_uniform_grid(grid_shape, value=180.0)  # deg F
oil_viscosity = bores.build_uniform_grid(grid_shape, value=1.5)  # cP
bubble_point = bores.build_uniform_grid(grid_shape, value=2500.0)  # psi

# Residual and irreducible saturations
Sorw = bores.build_uniform_grid(grid_shape, value=0.12)
Sorg = bores.build_uniform_grid(grid_shape, value=0.10)
Sgr = bores.build_uniform_grid(grid_shape, value=0.05)
Swir = bores.build_uniform_grid(grid_shape, value=0.06)
Swc = bores.build_uniform_grid(grid_shape, value=0.06)

# Build depth grid and compute initial saturations from fluid contacts
depth = bores.build_depth_grid(thickness, datum=5000.0)  # Top at 5000 ft
Sw, So, Sg = bores.build_saturation_grids(
    depth_grid=depth,
    gas_oil_contact=5050.0,  # Above reservoir (no gas cap)
    oil_water_contact=5280.0,  # Below reservoir (all oil zone)
    connate_water_saturation_grid=Swc,
    residual_oil_saturation_water_grid=Sorw,
    residual_oil_saturation_gas_grid=Sorg,
    residual_gas_saturation_grid=Sgr,
    porosity_grid=porosity,
)

# Isotropic permeability: 100 mD
perm_grid = bores.build_uniform_grid(grid_shape, value=100.0)
permeability = bores.RockPermeability(x=perm_grid, y=perm_grid, z=perm_grid)

oil_sg_grid = bores.build_uniform_grid(grid_shape, value=0.85)

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
    residual_oil_saturation_water_grid=Sorw,
    residual_oil_saturation_gas_grid=Sorg,
    residual_gas_saturation_grid=Sgr,
    irreducible_water_saturation_grid=Swir,
    connate_water_saturation_grid=Swc,
    oil_specific_gravity_grid=oil_sg_grid,
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
            target_rate=-5000.0,
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
        )
    ],
)
wells = bores.wells_(injectors=[injector], producers=[producer])

# Rock-fluid tables (Brooks-Corey relative permeability + capillary pressure)
rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=bores.BrooksCoreyThreePhaseRelPermModel(
        water_exponent=2.0,
        oil_exponent=2.0,
        gas_exponent=2.0,
        wettability=bores.Wettability.WATER_WET,
        mixing_rule="eclipse_rule"
    ),
    capillary_pressure_table=bores.BrooksCoreyCapillaryPressureModel(
        wettability=bores.Wettability.WATER_WET,
    ),
)

# Simulation configuration
config = bores.Config(
    timer=bores.Timer(
        initial_step_size=bores.Time(days=5),
        max_step_size=bores.Time(months=1),
        min_step_size=bores.Time(hours=1),
        simulation_time=bores.Time(years=24),
        max_rejects=10,
    ),
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="implicit",
    pressure_solver="direct",
    saturation_solver="direct",
    pressure_preconditioner=None,
    saturation_preconditioner=None,
    jacobian_assembly_method="analytical",
    max_pressure_change=1800,
    # disable_capillary_effects=True,
    normalize_saturations=True,
)

# Run the simulation and collect states
states = list(bores.run(model, config))
final = states[-1]
print(f"Completed {final.step} steps in {final.time_in_days:.1f} days")
print(
    f"Final avg pressure: {final.model.fluid_properties.pressure_grid.mean():.1f} psi"
)

import typing

import bores

# bores.use_32bit_precision()

# Grid dimensions: 10x10x3 cells, each 1000 ft x 1000 ft, 100 ft thick
grid_shape = typing.cast(bores.ThreeDimensions, (10, 10, 3))
cell_dimension = (1000.0, 1000.0)

# Build property grids
thickness = bores.uniform_grid(grid_shape, value=100.0)  # ft
porosity = bores.uniform_grid(grid_shape, value=0.20)  # fraction
temperature = bores.uniform_grid(grid_shape, value=180.0)  # deg F
oil_viscosity = bores.uniform_grid(grid_shape, value=1.5)  # cP
bubble_point = bores.uniform_grid(grid_shape, value=2500.0)  # psi
oil_specific_gravity = bores.uniform_grid(grid_shape, value=0.85)
gas_gravity_grid = bores.uniform_grid(grid_shape, value=0.792)

# Residual and irreducible saturations
residual_oil_saturation_water = bores.uniform_grid(grid_shape, value=0.12)
residual_oil_saturation_gas = bores.uniform_grid(grid_shape, value=0.10)
residual_gas_saturation = bores.uniform_grid(grid_shape, value=0.05)
irreducible_water_saturation = bores.uniform_grid(grid_shape, value=0.06)
connate_water_saturation = bores.uniform_grid(grid_shape, value=0.06)

# Build depth grid and compute initial saturations from fluid contacts
depth = bores.depth_grid(thickness, datum=5000.0)  # Top at 5000 ft
water_saturation, oil_saturation, gas_saturation = bores.build_saturation_grids(
    depth_grid=depth,
    gas_oil_contact=5050.0,
    oil_water_contact=5280.0,
    connate_water_saturation_grid=connate_water_saturation,
    residual_oil_saturation_water_grid=residual_oil_saturation_water,
    residual_oil_saturation_gas_grid=residual_oil_saturation_gas,
    residual_gas_saturation_grid=residual_gas_saturation,
    porosity_grid=porosity,
)

pressure = bores.build_pressure_grid(
    depth_grid=depth,
    datum_depth=5150.0,
    datum_pressure=3000.0,
    oil_specific_gravity=0.85,
    water_specific_gravity=1.0,
    gas_specific_gravity=0.792,
    gas_oil_contact=5050.0,
    oil_water_contact=5280.0,
)

# Isotropic permeability: 100 mD
perm_grid = bores.uniform_grid(grid_shape, value=10.0)
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
    water_saturation_grid=water_saturation,
    gas_saturation_grid=gas_saturation,
    oil_saturation_grid=oil_saturation,
    oil_viscosity_grid=oil_viscosity,
    oil_bubble_point_pressure_grid=bubble_point,
    residual_oil_saturation_water_grid=residual_oil_saturation_water,
    residual_oil_saturation_gas_grid=residual_oil_saturation_gas,
    residual_gas_saturation_grid=residual_gas_saturation,
    irreducible_water_saturation_grid=irreducible_water_saturation,
    connate_water_saturation_grid=connate_water_saturation,
    oil_specific_gravity_grid=oil_specific_gravity,
    gas_gravity_grid=gas_gravity_grid,
    datum_depth=5000.0,
)

# Define wells
injector = bores.injection_well(
    well_name="INJ-1",
    perforating_intervals=[((0, 0, 2), (0, 0, 2))],
    radius=0.25,
    control=bores.AdaptiveRateControl(
        target_rate=10000,
        bhp_limit=8000.0,
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
        oil_exponent=2.0,
        gas_exponent=1.5,
        wettability=bores.Wettability.WATER_WET,
        mixing_rule="eclipse_rule",
    ),
    capillary_pressure_table=bores.VanGenuchtenCapillaryPressureModel(
        wettability=bores.Wettability.WATER_WET,
    ),
)
timer = bores.Timer(
    initial_step_size=bores.Time(days=5),
    maximum_step_size=bores.Time(months=3),
    minimum_step_size=bores.Time(hours=1),
    simulation_time=bores.Time(years=30),
    maximum_rejections=20,
)

# Simulation configuration
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",
    pressure_solver="direct",
    saturation_solver="direct",
    pressure_preconditioner=None,
    saturation_preconditioner=None,
    maximum_pressure_change=800,
    # freeze_saturation_pressure=True,
    disable_capillary_effects=True,
    minimum_injector_water_saturation=0.1
)

# Run and monitor the simulation and collect states
states = list(bores.monitor(model, config))
final = states[-1]
print(f"Completed {final.step} steps in {final.time_in_days:.2f} days")
print(
    f"Final avg pressure: {final.model.fluid_properties.pressure_grid.mean():.1f} psi"
)

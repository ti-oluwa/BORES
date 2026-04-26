"""
Buckley-Leverett 1D Waterflood Test Case

This is a standard test problem for reservoir simulators. It tests:
1. Numerical diffusion and shock capturing in saturation transport
2. Fractional flow theory implementation
3. Comparison against analytical Buckley-Leverett solution

Problem setup:
- 1D horizontal core (100 cells x 1 cell x 1 cell)
- Initially at connate water saturation (Swc = 0.2)
- Water injected at constant rate from left
- Producer at right at constant BHP
- Incompressible fluids (simplifies to pure hyperbolic transport)
- No gravity, no capillary pressure

The analytical solution exhibits a shock front (discontinuity) which
tests the numerical scheme's shock-capturing ability.
"""

import typing

import bores
from bores.correlations.core import compute_gas_molecular_weight

# 1000 ft long core, divided into 100 cells (10 ft each)
# Cross-section: 10 ft x 10 ft (100 sq ft area)
nx = 100
ny = 1
nz = 1
grid_shape = typing.cast(bores.ThreeDimensions, (nx, ny, nz))

cell_dimension = (100.0, 100.0)  # DX, DY in feet
thickness = 10.0  # ft

# Uniform properties
porosity = 0.25  # 25% porosity
permeability = 100.0  # mD (isotropic)
rock_compressibility = 1.0e-6  # Very small (nearly incompressible)

# Build grids
thickness_grid = bores.uniform_grid(grid_shape, value=thickness)
porosity_grid = bores.uniform_grid(grid_shape, value=porosity)
perm_grid = bores.uniform_grid(grid_shape, value=permeability)
absolute_permeability = bores.RockPermeability(x=perm_grid)
temperature_grid = bores.uniform_grid(grid_shape, value=150.0)  # deg F

depth_grid = bores.depth_grid(thickness_grid, datum=5000.0)
pressure_grid = bores.uniform_grid(grid_shape, value=5000.0)

# Initial saturations — connate water, rest is oil
connate_water_saturation = 0.20
initial_oil_saturation = 1.0 - connate_water_saturation

water_saturation_grid = bores.uniform_grid(grid_shape, value=connate_water_saturation)
oil_saturation_grid = bores.uniform_grid(grid_shape, value=initial_oil_saturation)
gas_saturation_grid = bores.uniform_grid(grid_shape, value=0.0)

# Residual saturations for rel perm endpoints
connate_water_saturation_grid = bores.uniform_grid(
    grid_shape, value=connate_water_saturation
)
irreducible_water_saturation_grid = connate_water_saturation_grid.copy()
residual_oil_saturation_water = 0.20  # Sor (waterflood residual)
residual_oil_saturation_water_grid = bores.uniform_grid(
    grid_shape, value=residual_oil_saturation_water
)
residual_oil_saturation_gas_grid = bores.uniform_grid(grid_shape, value=0.10)
residual_gas_saturation_grid = bores.uniform_grid(grid_shape, value=0.05)

# Oil
oil_viscosity = 2.0  # cP
oil_specific_gravity = 0.85
oil_density = oil_specific_gravity * 62.4  # lbm/ft³

oil_viscosity_grid = bores.uniform_grid(grid_shape, value=oil_viscosity)
oil_specific_gravity_grid = bores.uniform_grid(grid_shape, value=oil_specific_gravity)

# Water
water_viscosity = 0.5  # cP
water_specific_gravity = 1.0
water_density = water_specific_gravity * 62.4  # lbm/ft³

# Gas (not used, but required by model)
gas_gravity = 0.7
gas_molecular_weight = compute_gas_molecular_weight(gas_gravity=gas_gravity)
gas_gravity_grid = bores.uniform_grid(grid_shape, value=gas_gravity)
oil_bubble_point_grid = bores.uniform_grid(grid_shape, value=2700.0)

rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=bores.LETThreePhaseRelPermModel()
)

model = bores.reservoir_model(
    grid_shape=grid_shape,
    cell_dimension=cell_dimension,
    thickness_grid=thickness_grid,
    pressure_grid=pressure_grid,
    rock_compressibility=rock_compressibility,
    absolute_permeability=absolute_permeability,
    porosity_grid=porosity_grid,
    temperature_grid=temperature_grid,
    water_saturation_grid=water_saturation_grid,
    oil_saturation_grid=oil_saturation_grid,
    gas_saturation_grid=gas_saturation_grid,
    oil_viscosity_grid=oil_viscosity_grid,
    oil_bubble_point_pressure_grid=oil_bubble_point_grid,
    residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
    residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
    residual_gas_saturation_grid=residual_gas_saturation_grid,
    irreducible_water_saturation_grid=irreducible_water_saturation_grid,
    connate_water_saturation_grid=connate_water_saturation_grid,
    oil_specific_gravity_grid=oil_specific_gravity_grid,
    gas_gravity_grid=gas_gravity_grid,
    datum_depth=5000.0,
)

# Water injector at first cell (0, 0, 0)
# Target rate: 1000 STB/D
# This gives a frontal velocity that's easy to track
injector = bores.injection_well(
    well_name="WATER-INJ",
    perforating_intervals=[((0, 0, 0), (0, 0, 0))],
    radius=0.25,  # ft
    control=bores.AdaptiveRateControl(
        target_rate=200.0,  # STB/D
        bhp_limit=6000.0,  # max injection pressure
        clamp=bores.InjectionClamp(),
    ),
    injected_fluid=bores.InjectedFluid(
        name="Water",
        phase=bores.FluidPhase.WATER,
        specific_gravity=1.0,
        molecular_weight=18.015,
    ),
    is_active=True,
    skin_factor=0.0,
)

# Oil producer at last cell (nx-1, 0, 0)
# Controlled by BHP to allow rate to vary
producer = bores.production_well(
    well_name="OIL-PROD",
    perforating_intervals=[((nx - 1, 0, 0), (nx - 1, 0, 0))],
    radius=0.25,  # ft
    control=bores.CoupledRateControl(
        primary_phase=bores.FluidPhase.OIL,
        primary_control=bores.AdaptiveRateControl(
            target_rate=-150.0,  # large negative (effectively BHP control)
            bhp_limit=1000.0,  # min BHP
            clamp=bores.ProductionClamp(),
        ),
        secondary_clamp=bores.ProductionClamp(),
    ),
    produced_fluids=[
        bores.ProducedFluid(
            name="Oil",
            phase=bores.FluidPhase.OIL,
            specific_gravity=oil_specific_gravity,
            molecular_weight=200.0,
        ),
        bores.ProducedFluid(
            name="Water",
            phase=bores.FluidPhase.WATER,
            specific_gravity=1.0,
            molecular_weight=18.015,
        ),
        bores.ProducedFluid(
            name="Gas",
            phase=bores.FluidPhase.GAS,
            specific_gravity=0.7,
            molecular_weight=gas_molecular_weight,
        ),
    ],
    is_active=True,
    skin_factor=0.0,
)
wells = bores.wells_(injectors=[injector], producers=[producer])


# Short timesteps to capture the shock accurately
# Buckley-Leverett breakthrough typically occurs within days to weeks
# depending on rate and pore volume
timer = bores.Timer(
    initial_step_size=bores.Time(hours=0.5),
    maximum_step_size=bores.Time(days=2),
    minimum_step_size=bores.Time(minutes=1),
    simulation_time=bores.Time(days=200),
    maximum_rejections=20,
)

config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",
    pressure_solver="direct",
    saturation_solver="direct",
    maximum_saturation_change=0.5,  # Control numerical diffusion
    disable_capillary_effects=True,  # Classic BL has no Pc
    normalize_saturations=True,
    maximum_pressure_change=2500,
    output_frequency=5,  # Save every 5th step for analysis
    minimum_injector_water_saturation=0.1,
    saturation_cfl_threshold=0.6,
    use_pseudo_pressure=False,
)

print("=" * 70)
print("BUCKLEY-LEVERETT 1D WATERFLOOD TEST")
print("=" * 70)
print(f"Grid: {nx} x {ny} x {nz} cells")
print(f"Total length: {nx * cell_dimension[0]} ft")
print(f"Porosity: {porosity:.2%}")
print(f"Permeability: {permeability} mD")
print(f"Initial Sw: {connate_water_saturation:.2%}")
print(f"Residual So: {residual_oil_saturation_water:.2%}")
print(f"Oil viscosity: {oil_viscosity} cP")
print(f"Water viscosity: {water_viscosity} cP")
print(f"Injection rate: {injector.control.target_rate} STB/D")
print("=" * 70)

states = list(bores.monitor(model, config))

final = states[-1]
print("\n" + "=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)
print(f"Completed {final.step} steps in {final.time_in_days:.2f} days")
print(
    f"Final avg pressure: {final.model.fluid_properties.pressure_grid.mean():.1f} psi"
)
print(f"Final avg Sw: {final.model.fluid_properties.water_saturation_grid.mean():.3f}")
print(f"Water breakthrough at producer: ", end="")

# Check if water has broken through (Sw at producer > connate)
producer_sw = final.model.fluid_properties.water_saturation_grid[-1, 0, 0]
if producer_sw > connate_water_saturation + 0.01:
    print(f"YES (Sw = {producer_sw:.3f})")
else:
    print(f"NO (Sw = {producer_sw:.3f})")

print("=" * 70)
print("\nTo analyze results and compare with analytical solution:")
print("1. Extract water saturation profiles from states at different times")
print("2. Plot Sw vs. distance (x-coordinate) — should show shock front")
print("3. Compare shock position and profile shape with analytical BL solution")
print("4. Calculate numerical diffusion by measuring shock smearing")
print("5. Examine fractional flow at producer vs. time (water cut curve)")

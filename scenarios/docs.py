import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import logging

    import numpy as np

    import bores

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
    thickness = bores.build_uniform_grid(grid_shape, value=20.0)  # ft per layer
    pressure = bores.build_uniform_grid(grid_shape, value=3000.0)  # psi
    porosity = bores.build_uniform_grid(grid_shape, value=0.20)  # fraction
    temperature = bores.build_uniform_grid(grid_shape, value=180.0)  # deg F
    oil_viscosity = bores.build_uniform_grid(grid_shape, value=1.5)  # cP
    bubble_point = bores.build_uniform_grid(grid_shape, value=2500.0)  # psi

    # Residual and irreducible saturations
    Sorw = bores.build_uniform_grid(grid_shape, value=0.20)  # Residual oil (waterflood)
    Sorg = bores.build_uniform_grid(grid_shape, value=0.15)  # Residual oil (gas flood)
    Sgr = bores.build_uniform_grid(grid_shape, value=0.05)  # Residual gas
    Swir = bores.build_uniform_grid(grid_shape, value=0.20)  # Irreducible water
    Swc = bores.build_uniform_grid(grid_shape, value=0.20)  # Connate water

    # Build depth grid from thickness and a datum (top of reservoir at 5000 ft)
    depth = bores.build_depth_grid(thickness, datum=5000.0)

    # Build initial saturations from fluid contact depths.
    # Place GOC above the reservoir and OWC below it so all cells
    # are in the oil zone (undersaturated, no initial gas cap).
    Sw, So, Sg = bores.build_saturation_grids(
        depth_grid=depth,
        gas_oil_contact=4999.0,  # Above reservoir top (no gas cap)
        oil_water_contact=5100.0,  # Below reservoir base (all oil zone)
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
        rock_compressibility=3e-6,  # psi^-1
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

    # ---------------------------------------------------------------------------
    # Step 4: Define wells
    # ---------------------------------------------------------------------------
    # Injection well in corner cell (0,0) perforated across all 3 layers.
    # Positive rate = injection.
    injector = bores.injection_well(
        well_name="INJ-1",
        perforating_intervals=[((0, 0, 0), (0, 0, 0))],
        radius=0.25,  # ft
        control=bores.ConstantRateControl(
            target_rate=500.0,  # 500 STB/day
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
        perforating_intervals=[((9, 9, 2), (9, 9, 2))],
        radius=0.25,  # ft
        control=bores.CoupledRateControl(
            primary_phase=bores.FluidPhase.OIL,
            primary_control=bores.AdaptiveRateControl(
                target_rate=-10_000.0,  # produce 10,000 STB/day of oil
                target_phase="oil",
                bhp_limit=1000.0,  # never drop below 1000 psi
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

    # Group wells together
    wells = bores.wells_(injectors=[injector], producers=[producer])

    # ---------------------------------------------------------------------------
    # Step 5: Define rock-fluid properties
    # ---------------------------------------------------------------------------
    # Brooks-Corey relative permeability model with Corey exponents of 2.0.
    # The capillary pressure model uses default Brooks-Corey parameters.
    rock_fluid = bores.RockFluidTables(
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
    # The Time helper converts human-readable durations to seconds.
    config = bores.Config(
        timer=bores.Timer(
            initial_step_size=bores.Time(days=1),
            maximum_step_size=bores.Time(days=10),
            minimum_step_size=bores.Time(hours=1),
            simulation_time=bores.Time(years=20),
        ),
        rock_fluid_tables=rock_fluid,
        wells=wells,
        scheme="impes",  # Implicit Pressure, Explicit Saturation
    )

    # ---------------------------------------------------------------------------
    # Step 7: Run the simulation
    # ---------------------------------------------------------------------------
    # bores.run() returns a generator that yields ModelState objects.
    # Each state is a snapshot of the reservoir at a specific time step.
    states = list(bores.run(model, config))

    # Extract average pressure over time
    time_days = np.array([s.time_in_days for s in states])
    avg_pressure = np.array(
        [s.model.fluid_properties.pressure_grid.mean() for s in states]
    )

    # Create pressure vs time plot
    pressure_series = np.column_stack([time_days, avg_pressure])
    fig = bores.make_series_plot(
        data=pressure_series,
        title="Average Reservoir Pressure",
        x_label="Time (days)",
        y_label="Pressure (psi)",
    )
    fig.show()
    return bores, np, states, time_days


@app.cell
def _(bores, np, states, time_days):
    avg_So = np.array(
        [s.model.fluid_properties.oil_saturation_grid.mean() for s in states]
    )
    avg_Sw = np.array(
        [s.model.fluid_properties.water_saturation_grid.mean() for s in states]
    )

    fig2 = bores.make_series_plot(
        data={
            "Oil Saturation": np.column_stack([time_days, avg_So]),
            "Water Saturation": np.column_stack([time_days, avg_Sw]),
        },
        title="Average Saturations",
        x_label="Time (days)",
        y_label="Saturation (fraction)",
    )
    fig2.show()
    return


if __name__ == "__main__":
    app.run()

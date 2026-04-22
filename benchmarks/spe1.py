import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def setup_grid():
    import logging
    import typing
    from pathlib import Path

    import numpy as np

    import bores
    from bores.correlations.core import compute_oil_specific_gravity

    logging.basicConfig(level=logging.DEBUG)

    # bores.use_32bit_precision()

    # -------------------------------------------------------------------------
    # Grid geometry — SPE1 benchmark (Odeh, 1981, JPT)
    #
    # 10 × 10 × 3 Cartesian grid
    # DX = DY = 1000 ft (uniform areal)
    # Layer 1: 20 ft thick   (top)
    # Layer 2: 30 ft thick
    # Layer 3: 50 ft thick   (bottom)
    # Top of reservoir at 8325 ft subsea
    # Layer centres at 8335, 8360, 8400 ft subsea
    # -------------------------------------------------------------------------
    cell_dimension = (1000.0, 1000.0)  # DX, DY in feet
    grid_shape = (10, 10, 3)

    layer_thicknesses = bores.array([20.0, 30.0, 50.0])  # ft, layers 1-3
    thickness_grid = bores.layered_grid(
        grid_shape=grid_shape,
        layer_values=layer_thicknesses,
        orientation=bores.Orientation.Z,
    )

    # -------------------------------------------------------------------------
    # Initial pressure
    # Datum: 4800 psia at 8400 ft (centre of Layer 3)
    #
    # Oil gradient:
    #   ρo at Pb (4014.7): 37.046 lbm/ft³ (Table 2)
    #   ρo at 9014.7:      39.768 lbm/ft³ (undersaturated table)
    #   At Pi=4800, interpolate:
    #   ρo = 37.046 + (39.768-37.046)*(4800-4014.7)/(9014.7-4014.7)
    #      = 37.046 + 2.722*0.15710 = 37.474 lbm/ft³
    #   gradient = 37.474/144 = 0.2602 psi/ft
    # -------------------------------------------------------------------------
    reference_pressure = 4800.0  # psia
    reference_depth = 8400.0  # ft (layer 3 centre)

    density_at_pb = 37.046  # lbm/ft³ at 4014.7 psia
    density_at_9015 = 39.768  # lbm/ft³ at 9014.7 psia
    pressure_interpolation_fraction = (reference_pressure - 4014.7) / (
        9014.7 - 4014.7
    )
    initial_oil_density = (
        density_at_pb
        + (density_at_9015 - density_at_pb) * pressure_interpolation_fraction
    )

    # Temperature: 200°F (Table 1, constant throughout)
    temperature_grid = bores.uniform_grid(grid_shape=grid_shape, value=200.0)

    # -------------------------------------------------------------------------
    # Bubble-point pressure
    # From Table 2: Pb = 4014.7 psia (highest saturated Rs table pressure)
    # Pi = 4800 psia > Pb → reservoir initially UNDERSATURATED
    # -------------------------------------------------------------------------
    oil_bubble_point_pressure_grid = bores.uniform_grid(
        grid_shape=grid_shape,
        value=4014.7,  # psia — constant bubble point (Case 1)
    )

    # -------------------------------------------------------------------------
    # Rock properties (Table 1, Odeh 1981)
    # Porosity = 0.3
    # Rock compressibility = 3×10⁻⁶ 1/psi
    # -------------------------------------------------------------------------
    porosity_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.3)
    rock_compressibility = 3.0e-6  # 1/psi

    # -------------------------------------------------------------------------
    # Permeabilities (Table 1, Odeh 1981)
    #
    # Horizontal:  kx = ky = 500, 50, 200 mD  (layers 1, 2, 3)
    # Vertical:    kz = 50, 5, 20 mD
    #   The paper states kv/kh = 0.1 for each layer:
    #     Layer 1: kz = 0.1 × 500 = 50 mD
    #     Layer 2: kz = 0.1 × 50  = 5  mD
    #     Layer 3: kz = 0.1 × 200 = 20 mD
    #
    # Note: The original Odeh paper uses kz = 50, 50, 37.25 in some editions
    # but the widely-cited SPE-CSP (Killough & Kossack 1987) clarification and
    # the ECLIPSE/CMG reference datasets use kv = 0.1*kh per layer.
    # Use the kv/kh = 0.1 values which are more physically consistent.
    # -------------------------------------------------------------------------
    kx_values = bores.array([500.0, 50.0, 200.0])  # mD
    ky_values = bores.array([500.0, 50.0, 200.0])  # mD
    kz_values = bores.array([50.0, 50.0, 25.0])  # mD  (kv = 0.1 × kh)

    kx_grid = bores.layered_grid(
        grid_shape=grid_shape,
        layer_values=kx_values,
        orientation=bores.Orientation.Z,
    )
    ky_grid = bores.layered_grid(
        grid_shape=grid_shape,
        layer_values=ky_values,
        orientation=bores.Orientation.Z,
    )
    kz_grid = bores.layered_grid(
        grid_shape=grid_shape,
        layer_values=kz_values,
        orientation=bores.Orientation.Z,
    )
    absolute_permeability = bores.RockPermeability(x=kx_grid, y=ky_grid, z=kz_grid)

    net_to_gross_grid = bores.uniform_grid(grid_shape=grid_shape, value=1.0)

    # -------------------------------------------------------------------------
    # Initial saturations (Table 1)
    # Sw = 0.12 (connate), So = 0.88, Sg = 0.0
    # -------------------------------------------------------------------------
    connate_water_saturation_grid = bores.uniform_grid(
        grid_shape=grid_shape, value=0.12
    )
    irreducible_water_saturation_grid = connate_water_saturation_grid.copy()
    residual_oil_saturation_water_grid = bores.uniform_grid(
        grid_shape=grid_shape, value=0.0
    )
    residual_oil_saturation_gas_grid = residual_oil_saturation_water_grid.copy()
    residual_gas_saturation_grid = residual_oil_saturation_water_grid.copy()

    # oil_saturation_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.88)
    # water_saturation_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.12)
    # gas_saturation_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.0)
    depth_grid = bores.depth_grid(thickness_grid, datum=8325.0)
    water_saturation_grid, oil_saturation_grid, gas_saturation_grid = (
        bores.build_saturation_grids(
            depth_grid=depth_grid,
            gas_oil_contact=8200.0,
            oil_water_contact=8500.0,
            connate_water_saturation_grid=connate_water_saturation_grid,
            residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
            residual_gas_saturation_grid=residual_gas_saturation_grid,
            porosity_grid=porosity_grid,
            use_transition_zones=True,
            transition_curvature_exponent=1.0,
        )
    )

    pressure_grid = bores.build_pressure_grid(
        depth_grid=depth_grid,
        datum_depth=reference_depth,
        datum_pressure=reference_pressure,
        oil_density=initial_oil_density,
        gas_specific_gravity=0.792,
        water_specific_gravity=1.0,
        gas_oil_contact=8200.0,
        oil_water_contact=8500.0,
    )

    # -------------------------------------------------------------------------
    # Fluid properties (initial estimates — overridden by PVT tables)
    # -------------------------------------------------------------------------
    gas_gravity = 0.792  # Table 1
    gas_gravity_grid = bores.uniform_grid(grid_shape=grid_shape, value=gas_gravity)
    gas_viscosity_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.027)
    oil_viscosity_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.51)

    # Oil specific gravity from dead-oil density at 14.7 psia: 46.244 lb/ft³
    oil_specific_gravity = compute_oil_specific_gravity(
        oil_density=49.244,
        pressure=14.7,
        temperature=200.0,
        oil_compressibility=0.0,
    )
    oil_specific_gravity_grid = bores.uniform_grid(
        grid_shape=grid_shape, value=oil_specific_gravity
    )

    # =========================================================================
    # PVT TABLES — Complete data from Table 2 (Odeh 1981)
    # =========================================================================

    # 9 pressure points: atmospheric → undersaturated
    pvt_pressures = bores.array(
        [
            14.7,  # Atmospheric
            264.7,
            514.7,
            1014.7,
            2014.7,
            2514.7,
            3014.7,
            4014.7,  # Bubble point (Pb)
            9014.7,  # Undersaturated
        ]
    )

    # Two temperature entries (isothermal at 200 °F; need ≥ 2 for 2-D table)
    pvt_temperatures = bores.array([200.0, 201.0])

    # -------------------------------------------------------------------------
    # OIL: Solution GOR (Rs), Bo, μo, ρo — all from Table 2
    # Above Pb: Rs is frozen at the bubble-point value (1270 SCF/STB)
    # -------------------------------------------------------------------------
    solution_gor_values = bores.array(
        [
            1.0,  # 14.7
            90.5,  # 264.7
            180.0,  # 514.7
            371.0,  # 1014.7
            636.0,  # 2014.7
            775.0,  # 2514.7
            930.0,  # 3014.7
            1270.0,  # 4014.7  ← bubble point
            1270.0,  # 9014.7  ← frozen (undersaturated)
        ]
    )

    # Bo (RB/STB): Table 2, saturated + undersaturated column
    # At Pb = 4014.7 → 1.6950; at 9014.7 → 1.5790 (Bo decreases above Pb)
    oil_fvf_values = bores.array(
        [
            1.0620,  # 14.7
            1.1500,  # 264.7
            1.2070,  # 514.7
            1.2950,  # 1014.7
            1.4350,  # 2014.7
            1.5000,  # 2514.7
            1.5650,  # 3014.7
            1.6950,  # 4014.7  ← bubble point
            1.5790,  # 9014.7  ← undersaturated (compression shrinks Bo)
        ]
    )

    # μo (cP): Table 2
    oil_viscosity_values = bores.array(
        [
            1.0400,  # 14.7
            0.9750,  # 264.7
            0.9100,  # 514.7
            0.8300,  # 1014.7
            0.6950,  # 2014.7
            0.6410,  # 2514.7
            0.5940,  # 3014.7
            0.5100,  # 4014.7  ← bubble point
            0.7400,  # 9014.7  ← viscosity rises above Pb
        ]
    )

    # ρo (lbm/ft³): Table 2
    oil_density_values = bores.array(
        [
            46.244,  # 14.7
            43.544,  # 264.7
            42.287,  # 514.7
            41.004,  # 1014.7
            38.995,  # 2014.7
            38.304,  # 2514.7
            37.781,  # 3014.7
            37.046,  # 4014.7
            39.768,  # 9014.7
        ]
    )

    # -------------------------------------------------------------------------
    # GAS: Bg, μg, ρg — from Table 2 "Gas PVT Functions"
    #
    # FIX: Bg in the paper is given in RB/MSCF.
    # Convert correctly: 1 RB/MSCF = 5.614583 ft³ / 1000 SCF = 5.614583e-3 ft³/SCF
    # So: Bg [ft³/SCF] = Bg_paper [RB/MSCF] × 5.614583 / 1000
    # -------------------------------------------------------------------------
    # Raw values from paper (RB/MSCF):
    gas_fvf_values_rb_mscf = bores.array(
        [
            166.666,  # 14.7     (note: 1/6 × 1000 ≈ atmospheric)
            12.093,  # 264.7
            6.274,  # 514.7
            3.197,  # 1014.7
            1.614,  # 2014.7
            1.294,  # 2514.7
            1.080,  # 3014.7
            0.811,  # 4014.7
            0.386,  # 9014.7
        ]
    )
    # Convert RB/MSCF → ft³/SCF
    gas_fvf_values = gas_fvf_values_rb_mscf * (
        bores.c.BARRELS_TO_CUBIC_FEET / 1000.0
    )

    # μg (cP): Table 2
    gas_viscosity_values = bores.array(
        [
            0.00800,  # 14.7
            0.00960,  # 264.7
            0.01120,  # 514.7
            0.01400,  # 1014.7
            0.01890,  # 2014.7
            0.02080,  # 2514.7
            0.02280,  # 3014.7
            0.02680,  # 4014.7
            0.04700,  # 9014.7
        ]
    )

    # ρg (lbm/ft³): Table 2
    gas_density_values = bores.array(
        [
            0.0647,  # 14.7
            0.8916,  # 264.7
            1.7185,  # 514.7
            3.3727,  # 1014.7
            6.8806,  # 2014.7
            8.3326,  # 2514.7
            9.9837,  # 3014.7
            13.2952,  # 4014.7
            27.9483,  # 9014.7
        ]
    )

    # -------------------------------------------------------------------------
    # WATER: Bw, μw, ρw — from Table 2 "Undersaturated Water PVT Functions"
    # -------------------------------------------------------------------------
    water_fvf_values = bores.array(
        [
            1.0410,  # 14.7
            1.0403,  # 264.7
            1.0395,  # 514.7
            1.0380,  # 1014.7
            1.0350,  # 2014.7
            1.0335,  # 2514.7
            1.0320,  # 3014.7
            1.0290,  # 4014.7
            1.0130,  # 9014.7
        ]
    )

    # μw (cP): constant at 0.31 cP per the table
    water_viscosity_values = bores.array([0.3100] * 9)

    # ρw (lbm/ft³): Table 2
    water_density_values = bores.array(
        [
            62.238,  # 14.7
            62.283,  # 264.7
            62.328,  # 514.7
            62.418,  # 1014.7
            62.599,  # 2014.7
            62.690,  # 2514.7
            62.781,  # 3014.7
            62.964,  # 4014.7
            63.959,  # 9014.7
        ]
    )

    # Gas solubility in water (Rsw): zero throughout per Table 2
    gas_solubility_in_water_values = bores.array([0.0] * 9)


    # -------------------------------------------------------------------------
    # Build 2-D tables (n_pressures × n_temperatures)
    # Broadcast each 1-D array across the temperature axis (isothermal)
    # -------------------------------------------------------------------------
    def make_2d(arr):
        return np.column_stack([arr, arr])


    solution_gor_table = typing.cast(
        bores.TwoDimensionalGrid, make_2d(solution_gor_values)
    )
    oil_fvf_table = typing.cast(bores.TwoDimensionalGrid, make_2d(oil_fvf_values))
    oil_viscosity_table = typing.cast(
        bores.TwoDimensionalGrid, make_2d(oil_viscosity_values)
    )
    oil_density_table = typing.cast(
        bores.TwoDimensionalGrid, make_2d(oil_density_values)
    )
    gas_fvf_table = typing.cast(bores.TwoDimensionalGrid, make_2d(gas_fvf_values))
    gas_viscosity_table = typing.cast(
        bores.TwoDimensionalGrid, make_2d(gas_viscosity_values)
    )
    gas_density_table = typing.cast(
        bores.TwoDimensionalGrid, make_2d(gas_density_values)
    )


    # Water tables are 3-D: (n_pressures, n_temperatures, n_salinities)
    # SPE1 uses fresh water → salinity = 0 ppm → n_salinities = 1
    def make_3d(arr):
        return np.stack([make_2d(arr)], axis=2)


    water_fvf_table = typing.cast(
        bores.ThreeDimensionalGrid, make_3d(water_fvf_values)
    )
    water_viscosity_table = typing.cast(
        bores.ThreeDimensionalGrid, make_3d(water_viscosity_values)
    )
    water_density_table = typing.cast(
        bores.ThreeDimensionalGrid, make_3d(water_density_values)
    )
    gas_solubility_in_water_table = typing.cast(
        bores.ThreeDimensionalGrid, make_3d(gas_solubility_in_water_values)
    )

    # -------------------------------------------------------------------------
    # Build PVT dataset
    # -------------------------------------------------------------------------
    pvt_dataset = bores.build_pvt_dataset(
        pressures=pvt_pressures,
        temperatures=pvt_temperatures,
        salinities=bores.array([0.0]),  # fresh water
        bubble_point_pressures=bores.array([4014.7, 4014.7]),
        oil_specific_gravity=oil_specific_gravity,
        gas_gravity=gas_gravity,
        water_salinity=0.0,
        solution_gas_to_oil_ratio_table=solution_gor_table,
        oil_formation_volume_factor_table=oil_fvf_table,
        oil_viscosity_table=oil_viscosity_table,
        oil_density_table=oil_density_table,
        gas_formation_volume_factor_table=gas_fvf_table,
        gas_viscosity_table=gas_viscosity_table,
        gas_density_table=gas_density_table,
        water_formation_volume_factor_table=water_fvf_table,
        water_viscosity_table=water_viscosity_table,
        water_density_table=water_density_table,
        gas_solubility_in_water_table=gas_solubility_in_water_table,
    )
    pvt_tables = bores.PVTTables.from_dataset(
        pvt_dataset, interpolation_method="linear", clamps=False
    )

    # -------------------------------------------------------------------------
    # Build Reservoir Model
    # -------------------------------------------------------------------------
    model = bores.reservoir_model(
        grid_shape=grid_shape,
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        pressure_grid=pressure_grid,
        oil_bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
        absolute_permeability=absolute_permeability,
        porosity_grid=porosity_grid,
        temperature_grid=temperature_grid,
        rock_compressibility=rock_compressibility,
        oil_saturation_grid=oil_saturation_grid,
        water_saturation_grid=water_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        oil_viscosity_grid=oil_viscosity_grid,
        oil_specific_gravity_grid=oil_specific_gravity_grid,
        gas_gravity_grid=gas_gravity_grid,
        gas_viscosity_grid=gas_viscosity_grid,
        residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
        irreducible_water_saturation_grid=irreducible_water_saturation_grid,
        connate_water_saturation_grid=connate_water_saturation_grid,
        residual_gas_saturation_grid=residual_gas_saturation_grid,
        net_to_gross_grid=net_to_gross_grid,
        water_salinity_grid=bores.array(np.zeros(grid_shape)),
        dip_angle=0.0,
        dip_azimuth=0.0,
        pvt_tables=pvt_tables,
        datum_depth=8325.0,  # ft — top of reservoir
    )

    model.save(Path("./benchmarks/runs/spe1/setup/model.h5"))
    pvt_tables.save(Path("./benchmarks/runs/spe1/setup/pvt.h5"))
    return Path, bores, np, oil_specific_gravity, pvt_tables


@app.cell
def setup_config(Path, bores, np, oil_specific_gravity, pvt_tables):
    from bores.correlations.core import compute_gas_molecular_weight

    # -------------------------------------------------------------------------
    # Relative permeability — Table 3 (Odeh 1981)
    # Gas-oil table indexed by gas saturation (Sg)
    # -------------------------------------------------------------------------
    sg_values = bores.array(
        [
            0.000,
            0.001,
            0.020,
            0.050,
            0.120,
            0.200,
            0.250,
            0.300,
            0.400,
            0.450,
            0.500,
            0.600,
            0.700,
            0.850,
            1.000,
        ]
    )
    krg_values = bores.array(
        [
            0.000,
            0.000,
            0.000,
            0.005,
            0.025,
            0.075,
            0.125,
            0.190,
            0.410,
            0.600,
            0.720,
            0.870,
            0.940,
            0.980,
            1.000,
        ]
    )
    krog_values = bores.array(
        [
            1.000,
            1.000,
            0.997,
            0.980,
            0.700,
            0.350,
            0.200,
            0.090,
            0.021,
            0.010,
            0.001,
            0.0001,
            0.000,
            0.000,
            0.000,
        ]
    )

    # Gas-oil table indexed by Sg (non-wetting phase)
    gas_oil_table = bores.TwoPhaseRelPermTable(
        wetting_phase=bores.FluidPhase.OIL,
        non_wetting_phase=bores.FluidPhase.GAS,
        reference_saturation=sg_values,
        reference_phase="non_wetting",  # table is indexed by Sg
        wetting_phase_relative_permeability=krog_values,
        non_wetting_phase_relative_permeability=krg_values,
    )

    # Oil-water table: SPE1 is a gas-oil case with connate water only.
    # krw = 0 everywhere (no water production); kro depends on Sw.
    # Sw spans from Swc (0.12) to 1.0; at Sw = Swc, kro = 1.0.
    # Reference_phase = "wetting" so the table is indexed by Sw.
    so_values = bores.array(
        [
            0.00,
            0.18,
            0.28,
            0.38,
            0.43,
            0.48,
            0.58,
            0.63,
            0.68,
            0.76,
            0.83,
            0.86,
            0.879,
            0.88,
        ]
    )
    krw_values = np.linspace(0.0, 0.00001, 14)
    krow_values = bores.array(
        [
            0.000,
            0.000,
            0.0001,
            0.001,
            0.010,
            0.021,
            0.090,
            0.200,
            0.350,
            0.700,
            0.980,
            0.997,
            1.000,
            1.000,
        ]
    )
    oil_water_table = bores.TwoPhaseRelPermTable(
        wetting_phase=bores.FluidPhase.WATER,
        non_wetting_phase=bores.FluidPhase.OIL,
        reference_saturation=so_values,
        reference_phase="non_wetting",  # indexed by So
        wetting_phase_relative_permeability=krw_values,
        non_wetting_phase_relative_permeability=krow_values,
    )

    relative_permeability_table = bores.ThreePhaseRelPermTable(
        oil_water_table=oil_water_table,
        gas_oil_table=gas_oil_table,
        mixing_rule="eclipse_rule",
    )
    rock_fluid_tables = bores.RockFluidTables(
        relative_permeability_table=relative_permeability_table
    )

    # Gas pseudo-pressure table — Table 2 (Odeh 1981)
    gas_pressures = bores.array(
        [
            14.7,
            264.7,
            514.7,
            1014.7,
            2014.7,
            2514.7,
            3014.7,
            4014.7,
            5014.7,
            9014.7,
        ]
    )
    gas_pseudo_pressures = bores.array(
        [
            0.000e0,  # 14.7
            7.77916e6,  # 264.7
            2.67580e7,  # 514.7
            8.75262e7,  # 1014.7
            2.70709e8,  # 2014.7
            3.86910e8,  # 2514.7
            5.16118e8,  # 3014.7
            8.03963e8,  # 4014.7
            1.12256e9,  # 5014.7
            2.51845e9,  # 9014.7
        ]
    )
    pseudo_pressure_table = bores.PseudoPressureTable(
        pressures=gas_pressures,
        pseudo_pressures=gas_pseudo_pressures,
        reference_pressure=14.7,
    )

    # -------------------------------------------------------------------------
    # Wells
    # -------------------------------------------------------------------------
    gas_molecular_weight = compute_gas_molecular_weight(gas_gravity=0.792)

    # Gas injector: cell (0,0,0), perforating Layer 1
    # Target: 100 MMscf/D; max BHP: 9011 psia (just below lithostatic)
    injector = bores.injection_well(
        well_name="GAS-INJ",
        perforating_intervals=[((0, 0, 0), (0, 0, 0))],
        radius=0.25,
        control=bores.AdaptiveRateControl(
            target_rate=100.0e6,  # 100 MMscf/D (SCF/day)
            bhp_limit=9011.0,  # max injection BHP (psia)
            clamp=bores.InjectionClamp(),
        ),
        injected_fluid=bores.InjectedFluid(
            name="Gas",
            phase=bores.FluidPhase.GAS,
            specific_gravity=0.792,
            molecular_weight=gas_molecular_weight,
            is_miscible=False,
            pseudo_pressure_table=pseudo_pressure_table,
        ),
        is_active=True,
        skin_factor=0.0,
    )

    # Oil producer: cell (9,9,2), perforating Layer 3
    # Target: 20 MSTB/D; min BHP: 1000 psia
    producer = bores.production_well(
        well_name="OIL-PROD",
        perforating_intervals=[((9, 9, 2), (9, 9, 2))],
        radius=0.25,
        control=bores.CoupledRateControl(
            primary_phase="oil",
            primary_control=bores.AdaptiveRateControl(
                target_rate=-20000,  # 20 MSTB/D (negative = production)
                bhp_limit=1000.0,  # min BHP (psia)
                clamp=bores.ProductionClamp(),
            ),
            secondary_clamp=bores.ProductionClamp(),
        ),
        produced_fluids=[
            bores.ProducedFluid(
                name="Oil",
                phase=bores.FluidPhase.OIL,
                specific_gravity=oil_specific_gravity,
                molecular_weight=180.0,
            ),
            bores.ProducedFluid(
                name="Gas",
                phase=bores.FluidPhase.GAS,
                specific_gravity=0.792,
                molecular_weight=gas_molecular_weight,
                pseudo_pressure_table=pseudo_pressure_table,
            ),
            bores.ProducedFluid(
                name="Water",
                phase=bores.FluidPhase.WATER,
                specific_gravity=1.00,
                molecular_weight=bores.c.MOLECULAR_WEIGHT_WATER,
            ),
        ],
        skin_factor=0.0,
        is_active=True,
    )
    wells = bores.wells_(injectors=[injector], producers=[producer])

    # -------------------------------------------------------------------------
    # Timer — 10-year simulation
    # -------------------------------------------------------------------------
    timer = bores.Timer(
        initial_step_size=bores.Time(days=1.0),
        maximum_step_size=bores.Time(days=30.0),
        minimum_step_size=bores.Time(minutes=10.0),
        simulation_time=bores.Time(years=8.0),
        maximum_cfl_number=0.9,
        ramp_up_factor=1.5,
        backoff_factor=0.5,
        aggressive_backoff_factor=0.25,
        maximum_rejections=20,
    )

    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------
    config = bores.Config(
        timer=timer,
        rock_fluid_tables=rock_fluid_tables,
        scheme="si",
        output_frequency=1,
        pressure_solver="direct",
        saturation_solver="direct",
        log_interval=10,
        pvt_tables=pvt_tables,
        wells=wells,
        disable_capillary_effects=True,
        # freeze_saturation_pressure=True,
        # maximum_gas_saturation_change=0.05,
        # maximum_oil_saturation_change=0.05,
        # maximum_water_saturation_change=0.05,
        # maximum_saturation_change=0.5,
        maximum_pressure_change=2000.0,
        use_pseudo_pressure=True,
        normalize_saturations=True,
        phase_appearance_tolerance=1e-6,
        saturation_convergence_tolerance=1e-4,
        saturation_cfl_threshold=0.8,
        minimum_injector_gas_saturation=0.1,
    )
    config.save(Path("./benchmarks/runs/spe1/setup/config.yaml"))
    return (wells,)


@app.cell
def setup_store(Path, bores):
    store = bores.HDF5Store(filepath=Path("./benchmarks/runs/spe1/results/spe1.h5"))
    return


@app.cell
def run_simulation(Path, bores):
    # Load run from saved files
    run = bores.Run.from_files(
        model_path=Path("./benchmarks/runs/spe1/setup/model.h5"),
        config_path=Path("./benchmarks/runs/spe1/setup/config.yaml"),
        pvt_tables_path=Path("./benchmarks/runs/spe1/setup/pvt.h5"),
    )

    # with bores.StateStream(run, store=store, background_io=True) as stream:
    #     last_state = stream.last()
    #     last_state.model.save(Path("./benchmarks/runs/spe1/results/model.h5"))

    # with bores.new_task_pool(concurrency=3) as pool:
    #     states = list(run(config=config.with_updates(task_pool=pool)))

    states = list(run)
    return (states,)


@app.cell
def load_states():
    # store_stream = bores.StateStream(store=store)
    # states = list(store_stream.replay(steps=None))
    return


@app.cell
def setup_analysis(bores, states):
    analyst = bores.ModelAnalyst(states)

    sweep_efficiency_history = analyst.sweep_efficiency_history(
        interval=1, from_step=1, displacing_phase="gas"
    )
    production_rate_history = analyst.instantaneous_rates_history(
        interval=1, from_step=1, rate_type="production"
    )
    injection_rate_history = analyst.instantaneous_rates_history(
        interval=1, from_step=1, rate_type="injection"
    )

    oil_saturation_history = []
    water_saturation_history = []
    gas_saturation_history = []
    avg_pressure_history = []

    volumetric_sweep_efficiency_history = []
    displacement_efficiency_history = []
    recovery_efficiency_history = []

    water_cut_history = []
    gor_history = []
    oil_rate_history = []
    gas_rate_history = []
    gas_injection_rate_history = []
    water_rate_history = []

    for s in states:
        time_step = s.time_in_days
        fluid_properties = s.model.fluid_properties
        avg_oil_sat = fluid_properties.oil_saturation_grid.mean()
        avg_water_sat = fluid_properties.water_saturation_grid.mean()
        avg_gas_sat = fluid_properties.gas_saturation_grid[9, 9, 2]
        avg_pressure = s.production_bhps.gas[9, 9, 2]

        oil_saturation_history.append((time_step, avg_oil_sat))
        water_saturation_history.append((time_step, avg_water_sat))
        gas_saturation_history.append((time_step, avg_gas_sat))
        avg_pressure_history.append((time_step, avg_pressure))

    for time_step, result in sweep_efficiency_history:
        volumetric_sweep_efficiency_history.append(
            (time_step, result.volumetric_sweep_efficiency)
        )
        displacement_efficiency_history.append(
            (time_step, result.displacement_efficiency)
        )
        recovery_efficiency_history.append((time_step, result.recovery_efficiency))

    for time_step, result in production_rate_history:
        oil_rate_history.append((time_step, result.oil_rate))
        water_rate_history.append((time_step, result.water_rate))
        gas_rate_history.append((time_step, result.gas_rate))
        water_cut_history.append((time_step, result.water_cut))
        gor_history.append((time_step, result.gas_oil_ratio))

    for time_step, result in injection_rate_history:
        gas_injection_rate_history.append((time_step, result.gas_rate))
    return (
        analyst,
        avg_pressure_history,
        displacement_efficiency_history,
        gas_injection_rate_history,
        gas_rate_history,
        gas_saturation_history,
        gor_history,
        oil_rate_history,
        oil_saturation_history,
        recovery_efficiency_history,
        volumetric_sweep_efficiency_history,
        water_cut_history,
        water_rate_history,
        water_saturation_history,
    )


@app.cell
def _(avg_pressure_history, bores, np):
    # Pressure
    pressure_fig = bores.make_series_plot(
        data={"Avg. Reservoir Pressure": np.array(avg_pressure_history[1:])},
        title="Pressure Analysis",
        x_label="Time Step",
        y_label="Avg. Pressure (psia)",
        marker_sizes=6,
        width=720,
        height=460,
    )
    pressure_fig.show()
    return


@app.cell
def saturation_plots(
    bores,
    gas_saturation_history,
    np,
    oil_saturation_history,
    water_saturation_history,
):
    # Saturation
    saturation_fig = bores.make_series_plot(
        data={
            "Avg. Water Saturation": np.array(water_saturation_history),
            "Avg. Oil Saturation": np.array(oil_saturation_history),
            "Avg. Gas Saturation": np.array(gas_saturation_history),
        },
        title="Saturation Analysis",
        x_label="Time Step",
        y_label="Saturation",
        marker_sizes=6,
        width=720,
        height=460,
    )
    saturation_fig.show()
    return


@app.cell
def _(bores, gas_rate_history, np, oil_rate_history, water_rate_history):
    oil_rate_fig = bores.make_series_plot(
        data={
            "Oil Rate (STB/day)": np.array(oil_rate_history),
        },
        title="Oil Production Rate Analysis",
        x_label="Time Step",
        y_label="Oil Rate",
        marker_sizes=6,
        width=720,
        height=460,
    )
    water_rate_fig = bores.make_series_plot(
        data={
            "Water Rate (STB/day)": np.array(water_rate_history),
        },
        title="Water Production Rate Analysis",
        x_label="Time Step",
        y_label="Water Rate",
        marker_sizes=6,
        width=720,
        height=460,
    )
    gas_rate_fig = bores.make_series_plot(
        data={
            "Gas Rate (SCF/day)": np.array(gas_rate_history),
        },
        title="Gas Production Rate Analysis",
        x_label="Time Step",
        y_label="Gas Rate",
        marker_sizes=6,
        width=720,
        height=460,
    )

    production_rate_plots = bores.merge_plots(
        oil_rate_fig,
        water_rate_fig,
        gas_rate_fig,
        title="Production Rate Analysis",
    )
    production_rate_plots.show()
    return


@app.cell
def _(bores, gas_injection_rate_history, np):
    gas_injection_rate_fig = bores.make_series_plot(
        data={
            "Gas Rate (SCF/day)": np.array(gas_injection_rate_history),
        },
        title="Gas Injection Rate Analysis",
        x_label="Time Step",
        y_label="Gas Rate",
        marker_sizes=6,
        width=720,
        height=460,
    )
    gas_injection_rate_fig.show()
    return


@app.cell
def fluid_cut_plots(bores, gor_history, np, water_cut_history):
    water_cut_fig = bores.make_series_plot(
        data={
            "Water Cut (WOR)": np.array(water_cut_history),
        },
        title="Water Cut Analysis",
        x_label="Time Step",
        y_label="Water Cut",
        marker_sizes=6,
        width=720,
        height=460,
    )
    gor_fig = bores.make_series_plot(
        data={
            "Gas-Oil Ratio (GOR)": np.array(gor_history),
        },
        title="Gas-Oil Ratio Analysis",
        x_label="Time Step",
        y_label="Gas-Oil Ratio",
        marker_sizes=6,
        width=720,
        height=460,
    )

    fluid_cut_plots = bores.merge_plots(
        water_cut_fig, gor_fig, cols=2, title="Fluid Cut Analysis"
    )
    fluid_cut_plots.show()
    return


@app.cell
def oil_production_plot(analyst, bores, np):
    # Production & Injection
    oil_production_history = analyst.oil_production_history(
        interval=1, cumulative=True, from_step=1
    )
    oil_production_fig = bores.make_series_plot(
        data={
            "Oil Production": np.array(list(oil_production_history)),
        },
        title="Oil Production Analysis",
        x_label="Time Step",
        y_label="Production (STB)",
        marker_sizes=6,
        width=720,
        height=460,
    )
    oil_production_fig.show()
    return


@app.cell
def gas_production_plot(analyst, bores, np):
    gas_production_history = analyst.gas_production_history(
        interval=1, cumulative=False, from_step=1
    )
    gas_production_fig = bores.make_series_plot(
        data={
            "Gas Production": np.array(list(gas_production_history)),
        },
        title="Gas Production Analysis",
        x_label="Time Step",
        y_label="Production (SCF)",
        marker_sizes=6,
        width=720,
        height=460,
    )
    gas_production_fig.show()
    return


@app.cell
def gas_injection_plot(analyst, bores, np):
    gas_injection_history = analyst.gas_injection_history(
        interval=1, cumulative=False, from_step=1
    )
    gas_injection_fig = bores.make_series_plot(
        data={
            "Gas Injection": np.array(list(gas_injection_history)),
        },
        title="Gas Injection Analysis",
        x_label="Time Step",
        y_label="Injection (SCF)",
        marker_sizes=6,
        width=720,
        height=460,
    )
    gas_injection_fig.show()
    return


@app.cell
def reserves_plots(analyst, bores, np):
    # Reserves
    oil_in_place_history = analyst.oil_in_place_history(interval=1, from_step=1)
    gas_in_place_history = analyst.gas_in_place_history(interval=1, from_step=1)
    water_in_place_history = analyst.water_in_place_history(interval=1, from_step=1)

    oil_water_reserves_fig = bores.make_series_plot(
        data={
            "Oil In Place": np.array(list(oil_in_place_history)),
            "Water In Place": np.array(list(water_in_place_history)),
        },
        title="Oil & Water Reserves Analysis",
        x_label="Time Step",
        y_label="OIP/WIP (STB)",
        marker_sizes=6,
        width=720,
        height=460,
    )
    gas_reserve_fig = bores.make_series_plot(
        data={
            "Gas In Place": np.array(list(gas_in_place_history)),
        },
        title="Gas Reserve Analysis",
        x_label="Time Step",
        y_label="GIP (SCF)",
        marker_sizes=6,
        width=720,
        height=460,
    )

    reserves_plots = bores.merge_plots(
        oil_water_reserves_fig,
        gas_reserve_fig,
        cols=2,
        title="Reserves Analysis (CASE 4)",
    )
    reserves_plots.show()
    return


@app.cell
def sweep_efficiency_plots(
    bores,
    displacement_efficiency_history,
    np,
    volumetric_sweep_efficiency_history,
):
    # Sweep efficiencies
    displacement_efficiency_fig = bores.make_series_plot(
        data={
            "Displacement Efficiency": np.array(displacement_efficiency_history),
        },
        title="Displacement Efficiency Analysis",
        x_label="Time Step",
        marker_sizes=6,
        width=720,
        height=460,
        line_colors="green",
    )
    vol_sweep_efficiency_fig = bores.make_series_plot(
        data={
            "Vol. Sweep Efficiency": np.array(volumetric_sweep_efficiency_history),
        },
        title="Volumetric Sweep Efficiency Analysis",
        x_label="Time Step",
        marker_sizes=6,
        width=720,
        height=460,
    )

    sweep_efficiency_plots = bores.merge_plots(
        displacement_efficiency_fig,
        vol_sweep_efficiency_fig,
        cols=2,
        title="Sweep Efficiency Analysis (CASE 4)",
    )
    sweep_efficiency_plots.show()
    return


@app.cell
def recovery_plots(analyst, bores, np, recovery_efficiency_history):
    recovery_efficiency_fig = bores.make_series_plot(
        data={
            "Recovery Efficiency": np.array(recovery_efficiency_history),
        },
        title="Recovery Efficiency Analysis",
        x_label="Time Step",
        marker_sizes=6,
        width=720,
        height=460,
        line_colors="orange",
    )

    oil_recovery_factor_history = analyst.oil_recovery_factor_history(
        interval=1, from_step=1
    )
    recovery_factor_fig = bores.make_series_plot(
        data={
            "Oil Recovery Factor": np.array(list(oil_recovery_factor_history)),
        },
        title="Recovery Factor Analysis",
        x_label="Time Step",
        marker_sizes=6,
        width=720,
        height=460,
    )

    recovery_plots = bores.merge_plots(
        recovery_efficiency_fig,
        recovery_factor_fig,
        cols=2,
        title="Recovery Analysis (CASE 4)",
    )
    recovery_plots.show()
    return


@app.cell
def _(bores):
    viz = bores.pyvista3d.DataVisualizer(bores.pyvista3d.PlotConfig(off_screen=False))
    return (viz,)


@app.cell
def _(bores, states, viz, wells):
    injector_locations, producer_locations = wells.locations
    injector_names, producer_names = wells.names
    well_positions = [*injector_locations, *producer_locations]
    well_names = [*injector_names, *producer_names]
    labels = bores.plotly3d.Labels()
    labels.add_well_labels(well_positions, well_names)

    shared_kwargs = dict(
        plot_type="cell_blocks",
        width=1200,
        height=720,
        # opacity=0.7,
        # labels=labels,
        aspect_mode="data",
        z_scale=10.0,
        # z_slice=(1, 3),
        marker_size=6,
        show_wells=True,
        show_surface_marker=True,
        show_perforations=True,
        # cmin=0.0,
        # cmax=1.0,
    )

    property = "gas-sat"
    figures = []
    timesteps = [295]
    for timestep in timesteps:
        figure = viz.make_plot(
            states[timestep],
            property=property,
            title=f"{property.strip('-').title()} Profile at Timestep {timestep}",
            **shared_kwargs,
        )
        figures.append(figure)

    if len(figures) > 1:
        plots = bores.merge_plots(*figures, cols=2, height=600)
        plots.show()
    else:
        figures[0].show()
    return


@app.cell
def _(bores, states):
    bores.pyvista3d.DataVisualizer(bores.pyvista3d.PlotConfig(off_screen=True)).animate(
        states,
        property="gas-saturation",
        plot_type="cell_blocks",
        save="gas_saturation.mp4",
        frame_duration=250,
        # step_size=5,  # Every 5th timestep
        cmin=0.0,
        cmax=1.0,
        opacity=1.0,
        z_scale=5.0,
        show_wells=True,
        show_surface_marker=True,
        show_perforations=True,
    )
    return


if __name__ == "__main__":
    app.run()

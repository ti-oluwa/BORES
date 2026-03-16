import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def setup_grid():
    import logging
    import typing
    from pathlib import Path

    import numpy as np

    import bores
    from bores.correlations.core import compute_oil_specific_gravity

    logging.basicConfig(level=logging.INFO)
    bores.use_64bit_precision()

    # -------------------------------------------------------------------------
    # Grid geometry — SPE1 benchmark (Odeh, 1981, JPT)
    #
    # 10 × 10 × 3 Cartesian grid
    # DX = DY = 1000 ft (uniform areal)
    # Layer 1: 20 ft thick
    # Layer 2: 30 ft thick
    # Layer 3: 50 ft thick
    # Top of reservoir at 8325 ft; centres at 8335, 8360, 8400 ft
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
    # Oil gradient calculation:
    # Initial reservoir pressure (4800 psia) is UNDERSATURATED (above Pb = 4014.7)
    # From Table 2 "Undersaturated Oil PVT Functions" section:
    #   At 4014.7 psia (Pb): ρₒ = 37.046 lbm/ft³
    #   At 9014.7 psia:      ρₒ = 39.768 lbm/ft³
    #
    # For undersaturated oil, density INCREASES with pressure (compression dominates)
    # Linear interpolation for P = 4800 psia:
    #   ρₒ(4800) = 37.046 + (39.768 - 37.046) × (4800 - 4014.7) / (9014.7 - 4014.7)
    #   ρₒ(4800) = 37.046 + 2.722 × 0.1571 = 37.474 lbm/ft³
    # -------------------------------------------------------------------------
    reference_pressure = 4800.0  # psia
    reference_depth = 8400.0  # ft

    # Interpolate undersaturated density at initial pressure
    rho_at_pb = (
        37.046  # lbm/ft³ at 4014.7 psia (bubble point, undersaturated table)
    )
    rho_at_9015 = 39.768  # lbm/ft³ at 9014.7 psia (undersaturated table)
    p_interp_frac = (reference_pressure - 4014.7) / (9014.7 - 4014.7)
    rho_oil_initial = rho_at_pb + (rho_at_9015 - rho_at_pb) * p_interp_frac

    oil_gradient = rho_oil_initial / 144.0  # psi/ft ≈ 0.260

    layer_centre_depths = np.array([8335.0, 8360.0, 8400.0])  # ft TVDSS
    layer_pressures = (
        reference_pressure + (layer_centre_depths - reference_depth) * oil_gradient
    )
    pressure_grid = bores.layered_grid(
        grid_shape=grid_shape,
        layer_values=bores.array(layer_pressures),
        orientation=bores.Orientation.Z,
    )

    # Temperature: 200°F (Table 1)
    temperature_grid = bores.uniform_grid(grid_shape=grid_shape, value=200.0)

    # -------------------------------------------------------------------------
    # Bubble-point pressure
    # From Table 2: highest saturated pressure = 4014.7 psia
    # Reservoir is initially UNDERSATURATED (Pi = 4800 > Pb = 4014.7)
    # -------------------------------------------------------------------------
    oil_bubble_point_pressure_grid = bores.uniform_grid(
        grid_shape=grid_shape,
        value=4014.7,  # psia — constant bubble point (Case 1)
    )

    # Oil compressibility (undersaturated)
    # Derived from Bo: (1.695-1.579)/(1.695 * 5000) ≈ 1.37e-5 1/psi
    # oil_compressibility_grid = bores.uniform_grid(grid_shape=grid_shape, value=1.37e-5)

    # -------------------------------------------------------------------------
    # Rock properties
    # Porosity = 0.3 (measured at 14.7 psia base pressure)
    # Rock compressibility = 3×10⁻⁶ 1/psi
    # -------------------------------------------------------------------------
    porosity_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.3)
    rock_compressibility = 3.0e-6  # 1/psi

    kx_values = bores.array([500.0, 50.0, 200.0])  # mD per layer
    ky_values = bores.array([500.0, 50.0, 200.0])
    kz_values = bores.array([50.0, 50.0, 25.0])  # vertical permeability

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

    # Net-to-gross = 1.0 (full pay)
    net_to_gross_grid = bores.uniform_grid(grid_shape=grid_shape, value=1.0)

    # -------------------------------------------------------------------------
    # Initial saturations
    # Sw = 0.12, So = 0.88, Sg = 0.0  (uniform, all layers)
    # -------------------------------------------------------------------------
    connate_water_saturation_grid = bores.uniform_grid(
        grid_shape=grid_shape, value=0.12
    )
    irreducible_water_saturation_grid = bores.uniform_grid(
        grid_shape=grid_shape, value=0.12
    )

    # Rel-perm endpoints from Table 3
    residual_oil_saturation_water_grid = bores.uniform_grid(
        grid_shape=grid_shape, value=0.0
    )
    residual_oil_saturation_gas_grid = bores.uniform_grid(
        grid_shape=grid_shape, value=0.0
    )
    residual_gas_saturation_grid = bores.uniform_grid(
        grid_shape=grid_shape, value=0.0
    )

    oil_saturation_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.88)
    water_saturation_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.12)
    gas_saturation_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.0)

    # -------------------------------------------------------------------------
    # Fluid property grids (initial conditions)
    # Will be overridden by PVT tables but needed for initial model setup
    # -------------------------------------------------------------------------

    # Gas gravity: 0.792 (Table 1)
    gas_gravity = 0.792
    gas_gravity_grid = bores.uniform_grid(grid_shape=grid_shape, value=gas_gravity)

    # Gas viscosity initial
    gas_viscosity_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.027)

    oil_viscosity_grid = bores.uniform_grid(
        grid_shape=grid_shape, value=0.51
    )  # cP

    # Oil specific gravity from Table 2 dead oil density at 14.7 psia: 46.244 lb/ft³ (assuming oil is incompressible at STP)
    oil_specific_gravity = compute_oil_specific_gravity(
        oil_density=46.244,
        pressure=14.7,
        temperature=200.0,
        oil_compressibility=0.0,
    )
    oil_specific_gravity_grid = bores.uniform_grid(
        grid_shape=grid_shape, value=oil_specific_gravity
    )

    # =========================================================================
    # PVT TABLES — COMPLETE DATA FROM TABLE 2 (Odeh 1981)
    # =========================================================================

    # Pressure points covering full table range (10 points)
    pvt_pressures = bores.array(
        [
            14.7,  # Atmospheric
            264.7,
            514.7,
            1014.7,
            2014.7,
            2514.7,
            3014.7,
            4014.7,  # Bubble point
            5014.7,  # Undersaturated
            9014.7,  # Undersaturated
        ]
    )

    # Temperature axis (isothermal at 200°F, but need 2 points for 2D tables)
    pvt_temperatures = bores.array([200.0, 201.0])
    n_p = len(pvt_pressures)
    n_t = len(pvt_temperatures)

    # -------------------------------------------------------------------------
    # OIL PROPERTIES (from "Saturated Oil PVT Functions" and "Undersaturated Oil PVT Functions")
    # -------------------------------------------------------------------------

    # Solution GOR (Rs) - SCF/STB
    # Above Pb (4014.7), Rs is frozen at bubble point value
    solution_gor_values = bores.array(
        [
            1.0,  # 14.7 psia
            90.5,  # 264.7
            180.0,  # 514.7
            371.0,  # 1014.7
            636.0,  # 2014.7
            775.0,  # 2514.7
            930.0,  # 3014.7
            1270.0,  # 4014.7 (bubble point)
            1270.0,  # 5014.7 (undersaturated - Rs frozen)
            1270.0,  # 9014.7 (undersaturated - Rs frozen)
        ]
    )

    # Oil Formation Volume Factor (Bo) - RB/STB
    # Above Pb: from "Undersaturated Oil PVT Functions" table
    # At 4014.7: 1.6950, at 9014.7: 1.5790 (Bo DECREASES above Pb - compression)
    oil_fvf_values = bores.array(
        [
            1.0620,  # 14.7 psia
            1.1500,  # 264.7
            1.2070,  # 514.7
            1.2950,  # 1014.7
            1.4350,  # 2014.7
            1.5000,  # 2514.7
            1.5650,  # 3014.7
            1.6950,  # 4014.7 (bubble point) — matches undersaturated table
            1.6718,  # 5014.7 (interpolated: 1.6950 + (1.5790-1.6950)*(1000/5000))
            1.5790,  # 9014.7 (undersaturated table, explicit)
        ]
    )

    # Oil Viscosity (μo) - cP
    # Above Pb: viscosity INCREASES with pressure (undersaturated compression,
    # no more gas liberation to thin the oil).
    # Pb value = 0.5100 cP. At 9014.7 psia, extrapolate using dead oil trend.
    # Standard correlation: μo_us = μo_b * exp(c_visc * (P - Pb))
    # With μo_b=0.5100, approximate c_visc ~ 2.5e-4 from saturated branch slope:
    #   at 5014.7: 0.5100 * exp(2.5e-4 * 1000) = 0.5100 * 1.284 ≈ 0.655 cP
    #   at 9014.7: 0.5100 * exp(2.5e-4 * 5000) = 0.5100 * 3.490 ≈ 1.780 cP
    # These are reasonable for a 35-40 API undersaturated oil.
    oil_viscosity_values = bores.array(
        [
            1.0400,  # 14.7 psia
            0.9750,  # 264.7
            0.9100,  # 514.7
            0.8300,  # 1014.7
            0.6950,  # 2014.7
            0.6410,  # 2514.7
            0.5940,  # 3014.7
            0.5100,  # 4014.7 (bubble point)
            0.6550,  # 5014.7 (undersaturated — increases above Pb)
            1.7800,  # 9014.7 (undersaturated — significantly higher)
        ]
    )

    # Oil Density (ρo) - lbm/ft³
    # Above Pb: from "Undersaturated Oil PVT Functions" — density INCREASES
    # (compression dominates, no more gas liberation to lower density)
    # At 4014.7: 37.046 lbm/ft³, at 9014.7: 39.768 lbm/ft³ (explicit in table)
    # At 5014.7: interpolate = 37.046 + (39.768-37.046)*(1000/5000) = 37.590
    oil_density_values = bores.array(
        [
            46.244,  # 14.7 psia
            43.544,  # 264.7
            42.287,  # 514.7
            41.004,  # 1014.7
            38.995,  # 2014.7
            38.304,  # 2514.7
            37.781,  # 3014.7
            37.046,  # 4014.7 (bubble point) — matches undersaturated table
            37.590,  # 5014.7 (interpolated from undersaturated table)
            39.768,  # 9014.7 (undersaturated table, explicit)
        ]
    )

    # -------------------------------------------------------------------------
    # GAS PROPERTIES (from "Gas PVT Functions")
    # -------------------------------------------------------------------------

    # Gas Formation Volume Factor (Bg) - RB/MSCF -> ft³/SCF
    gas_fvf_values = (
        bores.array(
            [
                0.166666,  # 14.7 psia
                0.012093,  # 264.7
                0.006274,  # 514.7
                0.003197,  # 1014.7
                0.001614,  # 2014.7
                0.001294,  # 2514.7
                0.001080,  # 3014.7
                0.000811,  # 4014.7
                0.000649,  # 5014.7
                0.000386,  # 9014.7
            ]
        )
        * bores.c.BARRELS_TO_CUBIC_FEET
        / 1000
    )

    # Gas Viscosity (μg) - cP
    gas_viscosity_values = bores.array(
        [
            0.008000,  # 14.7 psia
            0.009600,  # 264.7
            0.011200,  # 514.7
            0.014000,  # 1014.7
            0.018900,  # 2014.7
            0.020800,  # 2514.7
            0.022800,  # 3014.7
            0.026800,  # 4014.7
            0.030900,  # 5014.7
            0.047000,  # 9014.7
        ]
    )

    # Gas Density (ρg) - lbm/ft³
    gas_density_values = bores.array(
        [
            0.0647,  # 14.7 psia
            0.8916,  # 264.7
            1.7185,  # 514.7
            3.3727,  # 1014.7
            6.8806,  # 2014.7
            8.3326,  # 2514.7
            9.9837,  # 3014.7
            13.2952,  # 4014.7
            16.6139,  # 5014.7
            27.9483,  # 9014.7
        ]
    )

    # -------------------------------------------------------------------------
    # WATER PROPERTIES (from "Undersaturated Water PVT Functions")
    # -------------------------------------------------------------------------

    # Water Formation Volume Factor (Bw) - RB/STB
    water_fvf_values = bores.array(
        [
            1.0410,  # 14.7 psia
            1.0403,  # 264.7
            1.0395,  # 514.7
            1.0380,  # 1014.7
            1.0350,  # 2014.7
            1.0335,  # 2514.7
            1.0320,  # 3014.7
            1.0290,  # 4014.7
            1.0258,  # 5014.7
            1.0130,  # 9014.7
        ]
    )

    # Water Viscosity (μw) - cP (constant per table)
    water_viscosity_values = bores.array(
        [
            0.3100,  # 14.7 psia
            0.3100,  # 264.7
            0.3100,  # 514.7
            0.3100,  # 1014.7
            0.3100,  # 2014.7
            0.3100,  # 2514.7
            0.3100,  # 3014.7
            0.3100,  # 4014.7
            0.3100,  # 5014.7
            0.3100,  # 9014.7
        ]
    )

    # Water Density (ρw) - lbm/ft³
    water_density_values = bores.array(
        [
            62.238,  # 14.7 psia
            62.283,  # 264.7
            62.328,  # 514.7
            62.418,  # 1014.7
            62.599,  # 2014.7
            62.690,  # 2514.7
            62.781,  # 3014.7
            62.964,  # 4014.7
            63.160,  # 5014.7
            63.959,  # 9014.7
        ]
    )

    # Gas solubility in water (Rsw) - SCF/STB (all zero per table)
    gas_solubility_in_water_values = bores.array([0.0] * 10)

    # -------------------------------------------------------------------------
    # Build 2D Tables (n_pressures × n_temperatures)
    # Broadcast each 1D array to 2D by repeating across temperature axis
    # -------------------------------------------------------------------------

    # OIL TABLES
    solution_gor_table = np.column_stack(
        [solution_gor_values, solution_gor_values]
    )

    oil_fvf_table = np.column_stack([oil_fvf_values, oil_fvf_values])

    oil_viscosity_table = np.column_stack(
        [oil_viscosity_values, oil_viscosity_values]
    )

    oil_density_table = np.column_stack(
        [oil_density_values, oil_density_values]
    )

    # GAS TABLES
    gas_fvf_table = np.column_stack([gas_fvf_values, gas_fvf_values])

    gas_viscosity_table = np.column_stack(
        [gas_viscosity_values, gas_viscosity_values]
    )

    gas_density_table = np.column_stack([gas_density_values, gas_density_values])

    # WATER TABLES (need 3D: n_pressures × n_temperatures × n_salinities)
    # For SPE1, salinity = 0 (fresh water), so n_salinities = 1
    water_fvf_table = np.stack(
        [np.column_stack([water_fvf_values, water_fvf_values])], axis=2
    )  # Shape: (10, 2, 1)

    water_viscosity_table = np.stack(
        [np.column_stack([water_viscosity_values, water_viscosity_values])], axis=2
    )

    water_density_table = np.stack(
        [np.column_stack([water_density_values, water_density_values])], axis=2
    )

    gas_solubility_in_water_table = np.stack(
        [
            np.column_stack(
                [gas_solubility_in_water_values, gas_solubility_in_water_values]
            )
        ],
        axis=2,
    )

    # Cast to typed arrays for bores
    solution_gor_table = typing.cast(bores.TwoDimensionalGrid, solution_gor_table)
    oil_fvf_table = typing.cast(bores.TwoDimensionalGrid, oil_fvf_table)
    oil_viscosity_table = typing.cast(
        bores.TwoDimensionalGrid, oil_viscosity_table
    )
    oil_density_table = typing.cast(bores.TwoDimensionalGrid, oil_density_table)
    gas_fvf_table = typing.cast(bores.TwoDimensionalGrid, gas_fvf_table)
    gas_viscosity_table = typing.cast(
        bores.TwoDimensionalGrid, gas_viscosity_table
    )
    gas_density_table = typing.cast(bores.TwoDimensionalGrid, gas_density_table)
    water_fvf_table = typing.cast(bores.ThreeDimensionalGrid, water_fvf_table)
    water_viscosity_table = typing.cast(
        bores.ThreeDimensionalGrid, water_viscosity_table
    )
    water_density_table = typing.cast(
        bores.ThreeDimensionalGrid, water_density_table
    )
    gas_solubility_in_water_table = typing.cast(
        bores.ThreeDimensionalGrid, gas_solubility_in_water_table
    )

    # -------------------------------------------------------------------------
    # Build PVT Table Data
    # -------------------------------------------------------------------------
    pvt_data = bores.build_pvt_table_data(
        pressures=pvt_pressures,
        temperatures=pvt_temperatures,
        salinities=bores.array([0.0]),  # Fresh water
        bubble_point_pressures=bores.array([4014.7, 4014.7]),
        oil_specific_gravity=oil_specific_gravity,
        gas_gravity=gas_gravity,
        water_salinity=0.0,
        # Provide tabulated properties
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
    pvt_tables = bores.PVTTables(data=pvt_data, interpolation_method="linear")

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
        net_to_gross_ratio_grid=net_to_gross_grid,
        water_salinity_grid=bores.array(np.zeros(grid_shape)),
        dip_angle=0.0,
        dip_azimuth=0.0,
        pvt_tables=pvt_tables,
        datum_depth=8325.0,
    )

    model.save(Path("./benchmarks/runs/spe1/setup/model.h5"))
    pvt_data.save(Path("./benchmarks/runs/spe1/setup/pvt.h5"))
    return Path, bores, np, oil_specific_gravity, pvt_tables


@app.cell
def setup_config(Path, bores, oil_specific_gravity, pvt_tables):
    from bores.correlations.core import compute_gas_molecular_weight

    # -------------------------------------------------------------------------
    # Relative permeability — from Table 3 (Odeh 1981)
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
    kro_values = bores.array(
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

    # Gas-oil table indexed by oil saturation
    so_values = 1.0 - sg_values
    so_ascending = so_values[::-1].copy()
    kro_ascending = kro_values[::-1].copy()
    krg_ascending = krg_values[::-1].copy()

    gas_oil_table = bores.TwoPhaseRelPermTable(
        wetting_phase=bores.FluidPhase.OIL,
        non_wetting_phase=bores.FluidPhase.GAS,
        wetting_phase_saturation=so_ascending,
        wetting_phase_relative_permeability=kro_ascending,
        non_wetting_phase_relative_permeability=krg_ascending,
    )
    # Oil-water table: krw = 0 everywhere (since we dealing with only oil and gas case for this benchmark)
    oil_water_table = bores.TwoPhaseRelPermTable(
        wetting_phase=bores.FluidPhase.WATER,
        non_wetting_phase=bores.FluidPhase.OIL,
        wetting_phase_saturation=[0.0, 1.0],
        wetting_phase_relative_permeability=[0.0, 0.0],
        non_wetting_phase_relative_permeability=[1.0, 0.0],
    )
    relative_permeability_table = bores.ThreePhaseRelPermTable(
        oil_water_table=oil_water_table,
        gas_oil_table=gas_oil_table,
        mixing_rule=bores.eclipse_rule,
    )

    rock_fluid_tables = bores.RockFluidTables(
        relative_permeability_table=relative_permeability_table
    )

    # -------------------------------------------------------------------------
    # Gas Pseudo-Pressure Table (from SPE1 published data)
    # -------------------------------------------------------------------------
    # Data from "Gas PVT Functions" table (Odeh, 1981, JPT Table 1)
    # Column 5: Pseudo-pressure m(P) in psi²/cp
    gas_pressures = bores.array([
        14.7,      # Reference pressure (standard conditions)
        264.7,
        514.7,
        1014.7,
        2014.7,
        2514.7,
        3014.7,
        4014.7,
        5014.7,
        6014.7,
    ])
    gas_pseudo_pressures = bores.array([
        0.0,                 # Zero at reference pressure
        7.7791e6,           # 0.77791E+07
        2.67580e7,          # 0.267580E+08
        6.75220e7,          # 0.675220E+08
        2.70709e8,          # 0.270709E+09
        3.68101e8,          # 0.368101E+09
        5.18118e8,          # 0.518118E+09
        8.03683e8,          # 0.803683E+09
        1.15239e9,          # 1.15239E+10
        2.51848e9,          # 0.251848E+10
    ])
    pseudo_pressure_table = bores.GasPseudoPressureTable(
        pressures=gas_pressures,
        pseudo_pressures=gas_pseudo_pressures,
        reference_pressure=14.7,
    )

    # -------------------------------------------------------------------------
    # Wells
    # -------------------------------------------------------------------------
    gas_molecular_weight = compute_gas_molecular_weight(gas_gravity=0.792)

    injector = bores.injection_well(
        well_name="GAS-INJ",
        perforating_intervals=[((0, 0, 0), (0, 0, 0))],
        radius=0.25,
        control=bores.AdaptiveRateControl(
            target_rate=100.0e6,  # 100 MMscf/D
            bhp_limit=10000.0,  # Max injection BHP (psia)
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

    producer = bores.production_well(
        well_name="OIL-PROD",
        perforating_intervals=[((9, 9, 2), (9, 9, 2))],
        radius=0.25,
        control=bores.CoupledRateControl(
            primary_phase="oil",
            primary_control=bores.AdaptiveRateControl(
                target_rate=-20000,  # 20 MSTB/D production
                bhp_limit=1000.0,  # min BHP 1000 psia
                clamp=bores.ProductionClamp(),
            ),
            secondary_clamp=bores.ProductionClamp(),
        ),
        produced_fluids=(
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
        ),
        skin_factor=0.0,
        is_active=True,
    )
    wells = bores.wells_(injectors=[injector], producers=[producer])

    # -------------------------------------------------------------------------
    # Timer
    # -------------------------------------------------------------------------
    timer = bores.Timer(
        initial_step_size=bores.Time(days=3.0),
        max_step_size=bores.Time(days=30.0),
        min_step_size=bores.Time(minutes=20.0),
        simulation_time=bores.Time(years=10.0),
        max_cfl_number=0.9,
        ramp_up_factor=1.2,
        backoff_factor=0.5,
        aggressive_backoff_factor=0.25,
        max_rejects=20,
    )

    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------
    preconditioner_factory = bores.CachedPreconditionerFactory(
        factory="ilu",
        name="cached_ilu",
        update_frequency=10,
        recompute_threshold=0.3,
    )
    preconditioner_factory.register(override=True)

    config = bores.Config(
        timer=timer,
        rock_fluid_tables=rock_fluid_tables,
        scheme="impes",
        output_frequency=1,
        pressure_solver="bicgstab",
        pressure_preconditioner="cached_ilu",
        log_interval=10,
        pvt_tables=pvt_tables,
        wells=wells,
        boundary_conditions=None,
        disable_capillary_effects=True,
        freeze_saturation_pressure=False,
        miscibility_model="immiscible",
        max_gas_saturation_change=0.1,
        max_oil_saturation_change=0.2,
        max_water_saturation_change=0.1,
        max_pressure_change=1800.0,
        use_pseudo_pressure=True,
        normalize_saturations=True,
    )

    config.save(Path("./benchmarks/runs/spe1/setup/config.yaml"))
    return (wells,)


@app.cell
def setup_store(Path, bores):
    store = bores.ZarrStore(store=Path("./benchmarks/runs/spe1/results/spe1.zarr"))
    return (store,)


@app.cell
def run_simulation(Path, bores, store):
    run = bores.Run.from_files(
        model_path=Path("./benchmarks/runs/spe1/setup/model.h5"),
        config_path=Path("./benchmarks/runs/spe1/setup/config.yaml"),
        pvt_data_path=Path("./benchmarks/runs/spe1/setup/pvt.h5"),
    )


    def GOR_gte_20_000(state) -> bool:
        analyst = bores.ModelAnalyst([state])
        rates = analyst.instantaneous_production_rates(cells=[[(9, 9, 2), (9, 9, 2)]])
        return rates.gas_oil_ratio >= 20_000


    last_state = None
    with bores.StateStream(run, store=store, background_io=True) as stream:
        for state in stream:
            last_state = state

    if last_state is not None:
        last_state.model.save(Path("./benchmarks/runs/spe1/results/model.h5"))
    return


@app.cell
def load_states(bores, store):
    store_stream = bores.StateStream(store=store)
    states = list(store_stream.replay(steps=lambda s: s % 10 == 0))
    return (states,)


@app.cell
def setup_analysis(bores, np, states):
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
        time_step = s.step
        fluid_properties = s.model.fluid_properties
        avg_oil_sat = np.mean(fluid_properties.oil_saturation_grid)
        avg_water_sat = np.mean(fluid_properties.water_saturation_grid)
        avg_gas_sat = np.mean(fluid_properties.gas_saturation_grid)
        avg_pressure = np.mean(fluid_properties.pressure_grid)

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
def pressure_plot(avg_pressure_history, bores, np):
    # Pressure
    pressure_fig = bores.make_series_plot(
        data={"Avg. Reservoir Pressure": np.array(avg_pressure_history)},
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
    water_in_place_history = analyst.water_in_place_history(
        interval=1, from_step=1
    )

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
def _(analyst):
    mbe = analyst.material_balance_error(to_step=1200)
    print(mbe.total_mbe)
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
        z_scale=5.0,
        marker_size=6,
        show_wells=True,
        show_surface_marker=True,
        show_perforations=True,
        # cmin=0.15,
        # cmax=1.1,
    )

    property = "gas-sat"
    figures = []
    timesteps = [67]
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
def _(states, viz):
    viz.animate(
        states,
        property="oil-saturation",
        plot_type="cell_blocks",
        save="oil_saturation.mp4",
        frame_duration=250,
        # step_size=5,  # Every 5th timestep
        cmin=0.0,
        cmax=1.0,  # Fixed range for consistent coloring
        opacity=1.0,
    )
    return


if __name__ == "__main__":
    app.run()

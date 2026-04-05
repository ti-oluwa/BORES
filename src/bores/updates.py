import logging
import typing

import attrs
import numba
import numpy as np
import numpy.typing as npt

from bores.boundary_conditions import BoundaryConditions, BoundaryMetadata
from bores.constants import c
from bores.grids.pvt import (
    build_gas_compressibility_factor_grid,
    build_gas_compressibility_grid,
    build_gas_density_grid,
    build_gas_formation_volume_factor_grid,
    build_gas_free_water_formation_volume_factor_grid,
    build_gas_solubility_in_water_grid,
    build_gas_viscosity_grid,
    build_live_oil_density_grid,
    build_oil_bubble_point_pressure_grid,
    build_oil_compressibility_grid,
    build_oil_effective_density_grid,
    build_oil_effective_viscosity_grid,
    build_oil_formation_volume_factor_grid,
    build_oil_viscosity_grid,
    build_solution_gas_to_oil_ratio_grid,
    build_water_bubble_point_pressure_grid,
    build_water_compressibility_grid,
    build_water_density_grid,
    build_water_formation_volume_factor_grid,
    build_water_viscosity_grid,
)
from bores.grids.rock_fluid import build_effective_residual_saturation_grids
from bores.models import FluidProperties, RockProperties, SaturationHistory
from bores.tables.pvt import PVTTables
from bores.types import (
    MiscibilityModel,
    NDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensions,
)
from bores.wells.base import Wells

logger = logging.getLogger(__name__)


def apply_pressure_boundary_condition(
    padded_pressure_grid: ThreeDimensionalGrid,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    cell_dimension: typing.Tuple[float, float],
    grid_shape: typing.Tuple[int, int, int],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    time: float,
    pad_width: int = 1,
) -> ThreeDimensionalGrid:
    """
    Applies presure boundary condition to the pressure grid (in-place).

    :param padded_pressure_grid: The padded pressure grid.
    :param boundary_conditions: The boundary conditions to apply.
    :param cell_dimension: The dimensions of each grid cell.
    :param grid_shape: The shape of the simulation grid.
    :param thickness_grid: The (unpadded) thickness grid of the reservoir.
    :param pad_width: Number of ghost cells used for grid padding.
    :param time: The current simulation time.
    """
    boundary_conditions["pressure"].apply(
        padded_pressure_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="pressure",
        ),
        pad_width=pad_width,
    )
    return padded_pressure_grid


def apply_saturation_boundary_conditions(
    padded_water_saturation_grid: ThreeDimensionalGrid,
    padded_oil_saturation_grid: ThreeDimensionalGrid,
    padded_gas_saturation_grid: ThreeDimensionalGrid,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    cell_dimension: typing.Tuple[float, float],
    grid_shape: typing.Tuple[int, int, int],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    time: float,
    pad_width: int = 1,
) -> typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid]:
    """
    Applies boundary conditions to the fluid saturation grids (in-place).

    Saturation usually do not have boundary conditions specified, but this is just to support it
    if its needed. By default, the purpose of this function fix the numerically drift between ghost/pad
    cells and edge cells between saturation updates. Ensuring that ghost cells, mirror the edge cells
    all the time (No-flow boundary basically).

    :param padded_water_saturation_grid: The padded water saturation grid.
    :param padded_oil_saturation_grid: The padded oil saturation grid.
    :param padded_gas_saturation_grid: The padded gas saturation grid.
    :param boundary_conditions: The boundary conditions to apply.
    :param cell_dimension: The dimensions of each grid cell.
    :param grid_shape: The shape of the simulation grid.
    :param thickness_grid: The (unpadded) thickness grid of the reservoir.
    :param pad_width: Number of ghost cells used for grid padding.
    :param time: The current simulation time.
    """
    boundary_conditions["oil_saturation"].apply(
        padded_oil_saturation_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="oil_saturation",
        ),
        pad_width=pad_width,
    )
    boundary_conditions["water_saturation"].apply(
        padded_water_saturation_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="water_saturation",
        ),
        pad_width=pad_width,
    )
    boundary_conditions["gas_saturation"].apply(
        padded_gas_saturation_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="gas_saturation",
        ),
        pad_width=pad_width,
    )
    return (
        padded_water_saturation_grid,
        padded_oil_saturation_grid,
        padded_gas_saturation_grid,
    )


def apply_temperature_boundary_condition(
    padded_temperature_grid: ThreeDimensionalGrid,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    cell_dimension: typing.Tuple[float, float],
    grid_shape: typing.Tuple[int, int, int],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    time: float,
    pad_width: int = 1,
) -> ThreeDimensionalGrid:
    """
    Applies presure boundary condition to the temperature grid (in-place).

    :param padded_temperature_grid: The padded temperature grid.
    :param boundary_conditions: The boundary conditions to apply.
    :param cell_dimension: The dimensions of each grid cell.
    :param grid_shape: The shape of the simulation grid.
    :param thickness_grid: The (unpadded) thickness grid of the reservoir.
    :param pad_width: Number of ghost cells used for grid padding.
    :param time: The current simulation time.
    """
    boundary_conditions["temperature"].apply(
        padded_temperature_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="temperature",
        ),
        pad_width=pad_width,
    )
    return padded_temperature_grid


def apply_boundary_conditions(
    padded_fluid_properties: FluidProperties[ThreeDimensions],
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    cell_dimension: typing.Tuple[float, float],
    grid_shape: typing.Tuple[int, int, int],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    time: float,
    pad_width: int = 1,
) -> FluidProperties:
    """
    Applies boundary conditions to the (padded) fluid property grids (in-plcae).

    :param padded_fluid_properties: The padded fluid properties.
    :param boundary_conditions: The boundary conditions to apply.
    :param cell_dimension: The dimensions of each grid cell.
    :param grid_shape: The shape of the simulation grid.
    :param thickness_grid: The (unpadded) thickness grid of the reservoir.
    :param pad_width: Number of ghost cells used for grid padding.
    :param time: The current simulation time.
    """
    apply_pressure_boundary_condition(
        padded_pressure_grid=padded_fluid_properties.pressure_grid,
        boundary_conditions=boundary_conditions,
        cell_dimension=cell_dimension,
        grid_shape=grid_shape,
        thickness_grid=thickness_grid,
        time=time,
        pad_width=pad_width,
    )
    apply_saturation_boundary_conditions(
        padded_water_saturation_grid=padded_fluid_properties.water_saturation_grid,
        padded_oil_saturation_grid=padded_fluid_properties.oil_saturation_grid,
        padded_gas_saturation_grid=padded_fluid_properties.gas_saturation_grid,
        boundary_conditions=boundary_conditions,
        cell_dimension=cell_dimension,
        grid_shape=grid_shape,
        thickness_grid=thickness_grid,
        time=time,
        pad_width=pad_width,
    )
    # Since black-oil system supported is isothermal for now, there's no need for this
    # apply_temperature_boundary_condition(
    #     padded_temperature_grid=padded_fluid_properties.temperature_grid,
    #     boundary_conditions=boundary_conditions,
    #     cell_dimension=cell_dimension,
    #     grid_shape=grid_shape,
    #     thickness_grid=thickness_grid,
    #     time=time,
    #     pad_width=pad_width,
    # )
    return padded_fluid_properties


def update_fluid_properties(
    fluid_properties: FluidProperties[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
    pvt_tables: typing.Optional[PVTTables] = None,
    freeze_saturation_pressure: bool = False,
) -> FluidProperties[ThreeDimensions]:
    """
    Updates PVT fluid properties grids using the current pressure and temperature values.
    This function recalculates the fluid PVT properties in a physically consistent sequence:

    ```markdown
    ┌────────────┐
    │  PRESSURE  │
    └────┬───────┘
         ▼
    ┌─────────────┐
    │  TEMPERATURE│
    └────┬───────┘
         ▼
    ┌──────────────┐
    │ GAS PROPERTIES│
    └────┬─────────┘
         ▼
    ┌──────────────────┐
    │ WATER PROPERTIES │
    └────┬─────────────┘
         ▼
    ┌────────────────────────────────────────────────────────────┐
    │ OIL PROPERTIES                                             │
    │  - Compute oil specific gravity and API gravity            │
    │  - Recalculate bubble point pressure (Pb)                  │
    │  - If pressure < Pb: recompute GOR (Rs) using Vazquez-Beggs│
    │  - Compute FVF using pressure, Rs, Pb                      │
    │  - Then compute oil compressibility, density, viscosity    │
    └────────────────────────────────────────────────────────────┘

    # GAS PROPERTIES
    - Computes gas gravity from density.
    - Uses gas gravity to derive molecular weight.
    - Computes gas z-factor (compressibility factor) from pressure, temperature, and gas gravity.
    - Updates:
        - Formation volume factor (Bg)
        - Compressibility (Cg)
        - Density (ρg)
        - Viscosity (μg)

    # WATER PROPERTIES
    - Computes gas solubility in water (Rs_w) based on salinity, pressure, and temperature.
    - Determines water bubble point pressure from solubility and salinity.
    - Computes gas-free water FVF (for use in density calculation).
    - Updates:
        - Water compressibility (Cw) considering dissolved gas
        - Water FVF (Bw)
        - Water density (ρw)
        - Water viscosity (μw)

    # OIL PROPERTIES
    - Computes oil specific gravity and API gravity from base density.
    - Recalculates bubble point pressure (Pb) using API, temperature, and gas gravity.
    - Determines GOR:
        - If current pressure < Pb: compute GOR using Vazquez-Beggs correlation.
        - If current pressure ≥ Pb: GOR = GOR at Pb (Rs = Rs_b).
    - Computes:
        - Oil formation volume factor (Bo) using pressure, Pb, GOR, and gravity.
        - Oil compressibility (Co) using updated GOR and Pb.
        - Oil density (ρo) using GOR and Bo.
        - Oil viscosity (μo) using Rs, Pb, and API.
    ```

    :param fluid_properties: Current fluid property grids (pressure, temperature, salinity, densities, etc.)
    :param wells: Current wells configuration (used for fluid type information).
    :param miscibility_model: The miscibility model used in the simulation.
    :param pvt_tables: PVT table for property lookups (if None, correlations are used).
    :param freeze_saturation_pressure: If True, keeps oil bubble point pressure (Pb)
        constant at its current value throughout the simulation. This is appropriate for:
        - Natural depletion with no compositional changes
        - Waterflooding where oil composition remains constant
        - Simplified black-oil models without compositional tracking
        If False (default), Pb is recomputed each timestep based on current solution GOR.

        **Properties affected when Pb is frozen:**
        - Bubble point pressure (Pb) - directly frozen
        - Solution GOR (Rs) - computed using frozen Pb as reference
        - Oil FVF (Bo) - uses frozen Pb for undersaturated calculations
        - Oil compressibility (Co) - switches at frozen Pb
        - Oil viscosity (μo) - indirectly through Rs
        - Oil density (ρo) - indirectly through Rs and Bo

    :return: Updated FluidProperties object with recalculated gas, water, and oil properties.
    """
    pressure_grid = fluid_properties.pressure_grid
    temperature_grid = fluid_properties.temperature_grid
    water_salinity_grid = fluid_properties.water_salinity_grid
    oil_pvt_table = pvt_tables.oil if pvt_tables is not None else None
    water_pvt_table = pvt_tables.water if pvt_tables is not None else None
    gas_pvt_table = pvt_tables.gas if pvt_tables is not None else None

    if gas_pvt_table is None:
        # GAS PROPERTIES
        gas_compressibility_factor_grid = build_gas_compressibility_factor_grid(
            gas_gravity_grid=fluid_properties.gas_gravity_grid,
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
        )
        new_gas_formation_volume_factor_grid = build_gas_formation_volume_factor_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            gas_compressibility_factor_grid=gas_compressibility_factor_grid,
        )
        new_gas_compressibility_grid = build_gas_compressibility_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            gas_gravity_grid=fluid_properties.gas_gravity_grid,
            gas_compressibility_factor_grid=gas_compressibility_factor_grid,
        )
        new_gas_density_grid = build_gas_density_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            gas_gravity_grid=fluid_properties.gas_gravity_grid,
            gas_compressibility_factor_grid=gas_compressibility_factor_grid,
        )
        new_gas_viscosity_grid = build_gas_viscosity_grid(
            temperature_grid=temperature_grid,
            gas_density_grid=new_gas_density_grid,
            gas_molecular_weight_grid=fluid_properties.gas_molecular_weight_grid,
        )
        gas_solubility_in_water_grid = build_gas_solubility_in_water_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            salinity_grid=fluid_properties.water_salinity_grid,
            gas=fluid_properties.reservoir_gas,
        )

    else:
        gas_compressibility_factor_grid = gas_pvt_table.compressibility_factor(
            pressure=pressure_grid, temperature=temperature_grid
        )
        new_gas_formation_volume_factor_grid = gas_pvt_table.formation_volume_factor(
            pressure=pressure_grid, temperature=temperature_grid
        )
        new_gas_compressibility_grid = gas_pvt_table.compressibility(
            pressure=pressure_grid, temperature=temperature_grid
        )
        new_gas_density_grid = gas_pvt_table.density(
            pressure=pressure_grid, temperature=temperature_grid
        )
        new_gas_viscosity_grid = gas_pvt_table.viscosity(
            pressure=pressure_grid, temperature=temperature_grid
        )
        gas_solubility_in_water_grid = gas_pvt_table.solubility_in_water(
            pressure=pressure_grid,
            temperature=temperature_grid,
            salinity=water_salinity_grid,
        )

    # WATER PROPERTIES
    if water_pvt_table is None:
        new_water_bubble_point_pressure_grid = build_water_bubble_point_pressure_grid(
            temperature_grid=temperature_grid,
            gas_solubility_in_water_grid=gas_solubility_in_water_grid,  # type: ignore[arg-type]
            salinity_grid=fluid_properties.water_salinity_grid,
        )
        gas_free_water_formation_volume_factor_grid = (
            build_gas_free_water_formation_volume_factor_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
            )
        )
        new_water_compressibility_grid = build_water_compressibility_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=new_water_bubble_point_pressure_grid,
            gas_formation_volume_factor_grid=fluid_properties.gas_formation_volume_factor_grid,
            gas_solubility_in_water_grid=gas_solubility_in_water_grid,  # type: ignore[arg-type]
            gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
            salinity=fluid_properties.water_salinity_grid,
        )
        new_water_density_grid = build_water_density_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            gas_gravity_grid=fluid_properties.gas_gravity_grid,
            salinity_grid=fluid_properties.water_salinity_grid,
            gas_solubility_in_water_grid=gas_solubility_in_water_grid,  # type: ignore[arg-type]
            gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
        )
        new_water_formation_volume_factor_grid = (
            build_water_formation_volume_factor_grid(
                water_density_grid=new_water_density_grid,  # Use new density here
                salinity_grid=fluid_properties.water_salinity_grid,
            )
        )
        new_water_viscosity_grid = build_water_viscosity_grid(
            temperature_grid=temperature_grid,
            salinity_grid=fluid_properties.water_salinity_grid,
            pressure_grid=pressure_grid,
        )
    else:
        new_water_bubble_point_pressure_grid = water_pvt_table.bubble_point_pressure(
            pressure=pressure_grid,
            temperature=temperature_grid,
            salinity=water_salinity_grid,
        )
        gas_free_water_formation_volume_factor_grid = (
            build_gas_free_water_formation_volume_factor_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
            )
        )
        new_water_compressibility_grid = water_pvt_table.compressibility(
            pressure=pressure_grid,
            temperature=temperature_grid,
            salinity=water_salinity_grid,
        )
        new_water_density_grid = water_pvt_table.density(
            pressure=pressure_grid,
            temperature=temperature_grid,
            salinity=water_salinity_grid,
        )
        new_water_formation_volume_factor_grid = (
            water_pvt_table.formation_volume_factor(
                pressure=pressure_grid,
                temperature=temperature_grid,
                salinity=water_salinity_grid,
            )
        )
        new_water_viscosity_grid = water_pvt_table.viscosity(
            pressure=pressure_grid,
            temperature=temperature_grid,
            salinity=water_salinity_grid,
        )

    # OIL PROPERTIES (tricky due to bubble point)
    # Make sure to always compute the oil bubble point pressure grid
    # before the gas to oil ratio grid, as the latter depends on the former.

    if oil_pvt_table is None:
        # Compute New bubble point using Current Rs
        if freeze_saturation_pressure:
            # Keep current bubble point constant
            new_oil_bubble_point_pressure_grid = (
                fluid_properties.oil_bubble_point_pressure_grid
            )
        else:
            # Recompute bubble point using current Rs
            new_oil_bubble_point_pressure_grid = build_oil_bubble_point_pressure_grid(
                gas_gravity_grid=fluid_properties.gas_gravity_grid,
                oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
                temperature_grid=temperature_grid,
                solution_gas_to_oil_ratio_grid=fluid_properties.solution_gas_to_oil_ratio_grid,
            )

        # Compute Rs at New bubble point
        gor_at_bubble_point_pressure_grid = build_solution_gas_to_oil_ratio_grid(
            pressure_grid=new_oil_bubble_point_pressure_grid,  # New bubble point here
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,  # Use same New bubble point here
            gas_gravity_grid=fluid_properties.gas_gravity_grid,
            oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
        )
        # Compute New Rs at current pressure
        new_solution_gas_to_oil_ratio_grid = build_solution_gas_to_oil_ratio_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,  # New bubble point here
            gas_gravity_grid=fluid_properties.gas_gravity_grid,
            oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
            gor_at_bubble_point_pressure_grid=gor_at_bubble_point_pressure_grid,  # GOR at new bubble point here
        )
        # Compute oil FVF (may use lagged compressibility which still acceptable)
        # Oil FVF does not depend necessarily on the new compressibility grid,
        # so we can use the old one (compressibility changes are small, hence, it can be lagged).
        # FVF is a function of pressure and phase behavior. Only when pressure changes,
        # does FVF need to be recalculated.
        new_oil_formation_volume_factor_grid = build_oil_formation_volume_factor_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
            oil_specific_gravity_grid=fluid_properties.oil_specific_gravity_grid,
            gas_gravity_grid=fluid_properties.gas_gravity_grid,
            solution_gas_to_oil_ratio_grid=new_solution_gas_to_oil_ratio_grid,
            oil_compressibility_grid=fluid_properties.oil_compressibility_grid,
        )
        # Compute oil compressibility
        new_oil_compressibility_grid = build_oil_compressibility_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
            oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
            gas_gravity_grid=fluid_properties.gas_gravity_grid,
            gor_at_bubble_point_pressure_grid=gor_at_bubble_point_pressure_grid,
            gas_formation_volume_factor_grid=new_gas_formation_volume_factor_grid,  # type: ignore
            oil_formation_volume_factor_grid=new_oil_formation_volume_factor_grid,
        )
        # Compute oil density and viscosity
        new_oil_density_grid = build_live_oil_density_grid(
            oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
            gas_gravity_grid=fluid_properties.gas_gravity_grid,
            solution_gas_to_oil_ratio_grid=new_solution_gas_to_oil_ratio_grid,
            formation_volume_factor_grid=new_oil_formation_volume_factor_grid,
        )
        new_oil_effective_density_grid = new_oil_density_grid
        new_oil_viscosity_grid = build_oil_viscosity_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
            oil_specific_gravity_grid=fluid_properties.oil_specific_gravity_grid,
            solution_gas_to_oil_ratio_grid=new_solution_gas_to_oil_ratio_grid,
            gor_at_bubble_point_pressure_grid=gor_at_bubble_point_pressure_grid,
        )
    else:
        # Compute New bubble point using Current Rs
        if freeze_saturation_pressure:
            # Keep current bubble point constant
            new_oil_bubble_point_pressure_grid = (
                fluid_properties.oil_bubble_point_pressure_grid
            )
        else:
            # Recompute from tables
            new_oil_bubble_point_pressure_grid = oil_pvt_table.bubble_point_pressure(
                temperature=temperature_grid,
                solution_gor=fluid_properties.solution_gas_to_oil_ratio_grid,
            )

        # Compute New Rs at current pressure
        new_solution_gas_to_oil_ratio_grid = oil_pvt_table.solution_gas_to_oil_ratio(
            pressure=pressure_grid,
            temperature=temperature_grid,
            solution_gor=fluid_properties.solution_gas_to_oil_ratio_grid,
        )
        # Compute oil FVF (may use lagged compressibility which is acceptable)
        # Oil FVF does not depend necessarily on the new compressibility grid,
        # so we can use the old one (compressibility changes are small, hence, it can be lagged).
        # FVF is a function of pressure and phase behavior. Only when pressure changes,
        # does FVF need to be recalculated.
        new_oil_formation_volume_factor_grid = oil_pvt_table.formation_volume_factor(
            pressure=pressure_grid,
            temperature=temperature_grid,
            solution_gor=new_solution_gas_to_oil_ratio_grid,
        )
        # Compute oil compressibility
        new_oil_compressibility_grid = oil_pvt_table.compressibility(
            pressure=pressure_grid,
            temperature=temperature_grid,
        )
        # Compute oil density and viscosity
        new_oil_density_grid = oil_pvt_table.density(
            pressure=pressure_grid,
            temperature=temperature_grid,
        )
        new_oil_effective_density_grid = new_oil_density_grid
        new_oil_viscosity_grid = oil_pvt_table.viscosity(
            pressure=pressure_grid,
            temperature=temperature_grid,
            solution_gor=new_solution_gas_to_oil_ratio_grid,
        )

    new_oil_effective_viscosity_grid = new_oil_viscosity_grid
    # If there are miscible injections, update the effective oil viscosity and density grids
    if miscibility_model == "todd_longstaff":
        for well in wells.injection_wells:
            injected_fluid = well.injected_fluid
            if injected_fluid is not None and injected_fluid.is_miscible:
                injected_fluid_viscosity_grid = injected_fluid.get_viscosity(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                )
                injected_fluid_density_grid = injected_fluid.get_density(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                )
                injected_fluid_viscosity_grid = typing.cast(
                    NDimensionalGrid[ThreeDimensions], injected_fluid_viscosity_grid
                )
                injected_fluid_density_grid = typing.cast(
                    NDimensionalGrid[ThreeDimensions], injected_fluid_density_grid
                )
                # Update effective oil viscosity grid using Todd-Longstaff model
                new_oil_effective_viscosity_grid = build_oil_effective_viscosity_grid(
                    oil_viscosity_grid=new_oil_effective_viscosity_grid,  # type: ignore
                    solvent_viscosity_grid=injected_fluid_viscosity_grid,
                    solvent_concentration_grid=fluid_properties.solvent_concentration_grid,
                    base_omega=injected_fluid.todd_longstaff_omega,
                    pressure_grid=pressure_grid,
                    minimum_miscibility_pressure=injected_fluid.minimum_miscibility_pressure,
                    transition_width=injected_fluid.miscibility_transition_width,
                )
                new_oil_effective_density_grid = build_oil_effective_density_grid(
                    oil_density_grid=new_oil_density_grid,  # type: ignore
                    solvent_density_grid=injected_fluid_density_grid,
                    oil_viscosity_grid=new_oil_effective_viscosity_grid,
                    solvent_viscosity_grid=injected_fluid_viscosity_grid,
                    solvent_concentration_grid=fluid_properties.solvent_concentration_grid,
                    base_omega=injected_fluid.todd_longstaff_omega,
                    pressure_grid=pressure_grid,
                    minimum_miscibility_pressure=injected_fluid.minimum_miscibility_pressure,
                    transition_width=injected_fluid.miscibility_transition_width,
                )

    return attrs.evolve(
        fluid_properties,
        solution_gas_to_oil_ratio_grid=new_solution_gas_to_oil_ratio_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        oil_bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        water_bubble_point_pressure_grid=new_water_bubble_point_pressure_grid,
        oil_viscosity_grid=new_oil_viscosity_grid,
        oil_effective_viscosity_grid=new_oil_effective_viscosity_grid,
        water_viscosity_grid=new_water_viscosity_grid,
        gas_viscosity_grid=new_gas_viscosity_grid,
        oil_formation_volume_factor_grid=new_oil_formation_volume_factor_grid,
        water_formation_volume_factor_grid=new_water_formation_volume_factor_grid,
        gas_formation_volume_factor_grid=new_gas_formation_volume_factor_grid,
        oil_compressibility_grid=new_oil_compressibility_grid,
        water_compressibility_grid=new_water_compressibility_grid,
        gas_compressibility_factor_grid=gas_compressibility_factor_grid,
        gas_compressibility_grid=new_gas_compressibility_grid,
        oil_density_grid=new_oil_density_grid,
        oil_effective_density_grid=new_oil_effective_density_grid,
        water_density_grid=new_water_density_grid,
        gas_density_grid=new_gas_density_grid,
    )


@numba.njit(cache=True)
def _apply_solution_gas_updates(
    oil_saturation_grid: ThreeDimensionalGrid,
    water_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    new_solution_gas_to_oil_ratio_grid: ThreeDimensionalGrid,
    new_oil_formation_volume_factor_grid: ThreeDimensionalGrid,
    new_gas_solubility_in_water_grid: ThreeDimensionalGrid,
    new_water_formation_volume_factor_grid: ThreeDimensionalGrid,
    new_gas_formation_volume_factor_grid: ThreeDimensionalGrid,
    old_solution_gas_to_oil_ratio_grid: ThreeDimensionalGrid,
    old_oil_formation_volume_factor_grid: ThreeDimensionalGrid,
    old_gas_solubility_in_water_grid: ThreeDimensionalGrid,
    old_water_formation_volume_factor_grid: ThreeDimensionalGrid,
    bbl_to_ft3: float,
    dtype: npt.DTypeLike,
) -> typing.Tuple[
    ThreeDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensionalGrid,
]:
    """
    Jit-friendly core of the solution gas liberation/re-dissolution flash.

    Accepts and returns raw grids only; all `attrs` manipulation is handled
    by the calling `apply_solution_gas_updates` wrapper.

    :param oil_saturation_grid: Oil saturation grid (fraction).
    :param water_saturation_grid: Water saturation grid (fraction).
    :param gas_saturation_grid: Gas saturation grid (fraction).
    :param new_solution_gas_to_oil_ratio_grid: Rs grid after PVT update (SCF/STB).
    :param new_oil_formation_volume_factor_grid: Bo grid after PVT update (bbl/STB).
    :param new_gas_solubility_in_water_grid: Rsw grid after PVT update (SCF/STB).
    :param new_water_formation_volume_factor_grid: Bw grid after PVT update (bbl/STB).
    :param new_gas_formation_volume_factor_grid: Bg grid after PVT update (bbl/SCF).
    :param old_solution_gas_to_oil_ratio_grid: Rs grid before PVT update (SCF/STB).
    :param old_oil_formation_volume_factor_grid: Bo grid before PVT update (bbl/STB).
    :param old_gas_solubility_in_water_grid: Rsw grid before PVT update (SCF/STB).
    :param old_water_formation_volume_factor_grid: Bw grid before PVT update (bbl/STB).
    :return: Tuple of (oil_saturation, water_saturation, gas_saturation, corrected_Rs, corrected_Rsw).
    """
    oil_saturation_grid = oil_saturation_grid.copy()
    water_saturation_grid = water_saturation_grid.copy()
    gas_saturation_grid = gas_saturation_grid.copy()
    corrected_rs = new_solution_gas_to_oil_ratio_grid.copy()
    corrected_rsw = new_gas_solubility_in_water_grid.copy()
    zero = dtype.type(0.0)  # type: ignore

    # OIL PHASE
    rs_change = old_solution_gas_to_oil_ratio_grid - new_solution_gas_to_oil_ratio_grid

    oil_liberation_mask = rs_change > 0.0
    oil_redissolution_mask = (rs_change < 0.0) & (gas_saturation_grid > 0.0)
    oil_no_gas_mask = (rs_change < 0.0) & (gas_saturation_grid <= 0.0)

    # Liberation: gas comes out of oil
    liberated_gas_from_oil = np.where(
        oil_liberation_mask,
        rs_change
        * oil_saturation_grid
        * new_gas_formation_volume_factor_grid
        / (old_oil_formation_volume_factor_grid * bbl_to_ft3),
        zero,
    )
    # Oil volume correction due to Bo change (applies to liberation cells)
    oil_saturation_grid = np.where(
        oil_liberation_mask,
        oil_saturation_grid
        * new_oil_formation_volume_factor_grid
        / old_oil_formation_volume_factor_grid,
        oil_saturation_grid,
    )
    gas_saturation_grid = gas_saturation_grid + liberated_gas_from_oil

    # Re-dissolution: gas goes back into oil
    safe_oil_saturation = np.maximum(oil_saturation_grid, 1e-30)
    safe_bg = np.maximum(new_gas_formation_volume_factor_grid, 1e-30)

    max_dissolvable_gor = (
        gas_saturation_grid
        * old_oil_formation_volume_factor_grid
        * bbl_to_ft3
        / (safe_oil_saturation * safe_bg)
    )
    # Actual Rs delta is capped at the available free gas (rs_change is negative here)
    actual_rs_delta = np.where(
        oil_redissolution_mask,
        np.maximum(rs_change, -max_dissolvable_gor),
        zero,
    )
    dissolved_gas_from_oil = np.where(
        oil_redissolution_mask,
        -actual_rs_delta
        * oil_saturation_grid
        * safe_bg
        / (old_oil_formation_volume_factor_grid * bbl_to_ft3),
        zero,
    )
    gas_saturation_grid = gas_saturation_grid - dissolved_gas_from_oil
    # Oil swells on re-dissolution
    oil_saturation_grid = np.where(
        oil_redissolution_mask,
        oil_saturation_grid
        * new_oil_formation_volume_factor_grid
        / old_oil_formation_volume_factor_grid,
        oil_saturation_grid,
    )
    # Correct Rs: re-dissolution cells use actual delta; no-gas cells revert to old Rs
    corrected_rs = np.where(
        oil_redissolution_mask,
        old_solution_gas_to_oil_ratio_grid - actual_rs_delta,
        corrected_rs,
    )
    corrected_rs = np.where(
        oil_no_gas_mask,
        old_solution_gas_to_oil_ratio_grid,
        corrected_rs,
    )

    # WATER PHASE
    rsw_change = old_gas_solubility_in_water_grid - new_gas_solubility_in_water_grid

    water_liberation_mask = rsw_change > zero
    water_redissolution_mask = (rsw_change < zero) & (gas_saturation_grid > zero)
    water_no_gas_mask = (rsw_change < zero) & (gas_saturation_grid <= zero)

    # Liberation: gas comes out of water
    liberated_gas_from_water = np.where(
        water_liberation_mask,
        rsw_change
        * water_saturation_grid
        * new_gas_formation_volume_factor_grid
        / (old_water_formation_volume_factor_grid * bbl_to_ft3),
        zero,
    )
    # Water volume correction due to Bw change (applies to liberation cells)
    water_saturation_grid = np.where(
        water_liberation_mask,
        water_saturation_grid
        * new_water_formation_volume_factor_grid
        / old_water_formation_volume_factor_grid,
        water_saturation_grid,
    )
    gas_saturation_grid = gas_saturation_grid + liberated_gas_from_water

    # Re-dissolution: gas goes back into water
    safe_water_saturation = np.maximum(water_saturation_grid, 1e-30)

    max_dissolvable_rsw = (
        gas_saturation_grid
        * old_water_formation_volume_factor_grid
        * bbl_to_ft3
        / (safe_water_saturation * safe_bg)
    )
    # Actual Rsw delta is capped at the available free gas (rsw_change is negative here)
    actual_rsw_delta = np.where(
        water_redissolution_mask,
        np.maximum(rsw_change, -max_dissolvable_rsw),
        zero,
    )
    dissolved_gas_in_water = np.where(
        water_redissolution_mask,
        -actual_rsw_delta
        * water_saturation_grid
        * safe_bg
        / (old_water_formation_volume_factor_grid * bbl_to_ft3),
        zero,
    )
    gas_saturation_grid = gas_saturation_grid - dissolved_gas_in_water
    # Water swells on re-dissolution
    water_saturation_grid = np.where(
        water_redissolution_mask,
        water_saturation_grid
        * new_water_formation_volume_factor_grid
        / old_water_formation_volume_factor_grid,
        water_saturation_grid,
    )
    # Correct Rsw: re-dissolution cells use actual delta; no-gas cells revert to old Rsw
    corrected_rsw = np.where(
        water_redissolution_mask,
        old_gas_solubility_in_water_grid - actual_rsw_delta,
        corrected_rsw,
    )
    corrected_rsw = np.where(
        water_no_gas_mask,
        old_gas_solubility_in_water_grid,
        corrected_rsw,
    )

    # Clamp saturations (normalization is handled by the caller)
    oil_saturation_grid = np.maximum(oil_saturation_grid, zero)
    water_saturation_grid = np.maximum(water_saturation_grid, zero)
    gas_saturation_grid = np.maximum(gas_saturation_grid, zero)

    return (
        oil_saturation_grid.astype(dtype),
        water_saturation_grid.astype(dtype),
        gas_saturation_grid.astype(dtype),
        corrected_rs.astype(dtype),
        corrected_rsw.astype(dtype),
    )


def apply_solution_gas_updates(
    fluid_properties: FluidProperties[ThreeDimensions],
    old_solution_gas_to_oil_ratio_grid: ThreeDimensionalGrid,
    old_oil_formation_volume_factor_grid: ThreeDimensionalGrid,
    old_gas_solubility_in_water_grid: ThreeDimensionalGrid,
    old_water_formation_volume_factor_grid: ThreeDimensionalGrid,
) -> FluidProperties[ThreeDimensions]:
    """
    Applies solution gas liberation (or re-dissolution) to saturations based on
    changes in the solution gas-to-oil ratio (Rs) and gas solubility in water (Rsw)
    from the PVT update.

    When pressure drops below the bubble point, Rs and Rsw decrease and dissolved gas
    comes out of solution as free gas. When pressure rises and free gas is present,
    gas can re-dissolve into oil and water.

    Physics:
        - Oil shrinks when Rs decreases: So_new = So_old * Bo_new / Bo_old
        - Water shrinks when Rsw decreases: Sw_new = Sw_old * Bw_new / Bw_old
        - Gas appears from oil: dSg_oil = (Rs_old - Rs_new) * So_old * [Bg_new / (Bo_old * bbl_to_ft3)]
        - Gas appears from water: dSg_water = (Rsw_old - Rsw_new) * Sw_old * [Bg_new / (Bw_old * bbl_to_ft3)]
        - Re-dissolution is capped at available free gas
        - Rs and Rsw are corrected if the PVT update assumed more gas could dissolve than exists

    :param fluid_properties: Fluid properties with updated PVT (new Rs, Rsw, Bo, Bw, Bg)
        but old saturations.
    :param old_solution_gas_to_oil_ratio_grid: Solution gas-oil ratio grid before PVT update (SCF/STB).
    :param old_oil_formation_volume_factor_grid: Oil FVF grid before PVT update (bbl/STB).
    :param old_gas_solubility_in_water_grid: Gas solubility in water grid before PVT update (SCF/STB).
    :param old_water_formation_volume_factor_grid: Water FVF grid before PVT update (bbl/STB).
    :return: Updated fluid properties with post-flash saturations and corrected Rs/Rsw.
    """
    (
        new_oil_saturation_grid,
        new_water_saturation_grid,
        new_gas_saturation_grid,
        corrected_solution_gas_to_oil_ratio_grid,
        corrected_gas_solubility_in_water_grid,
    ) = _apply_solution_gas_updates(
        oil_saturation_grid=fluid_properties.oil_saturation_grid,
        water_saturation_grid=fluid_properties.water_saturation_grid,
        gas_saturation_grid=fluid_properties.gas_saturation_grid,
        new_solution_gas_to_oil_ratio_grid=fluid_properties.solution_gas_to_oil_ratio_grid,
        new_oil_formation_volume_factor_grid=fluid_properties.oil_formation_volume_factor_grid,
        new_gas_solubility_in_water_grid=fluid_properties.gas_solubility_in_water_grid,
        new_water_formation_volume_factor_grid=fluid_properties.water_formation_volume_factor_grid,
        new_gas_formation_volume_factor_grid=fluid_properties.gas_formation_volume_factor_grid,
        old_solution_gas_to_oil_ratio_grid=old_solution_gas_to_oil_ratio_grid,
        old_oil_formation_volume_factor_grid=old_oil_formation_volume_factor_grid,
        old_gas_solubility_in_water_grid=old_gas_solubility_in_water_grid,
        old_water_formation_volume_factor_grid=old_water_formation_volume_factor_grid,
        bbl_to_ft3=c.BARRELS_TO_CUBIC_FEET,
        dtype=fluid_properties.oil_saturation_grid.dtype,
    )
    return attrs.evolve(
        fluid_properties,
        oil_saturation_grid=new_oil_saturation_grid,
        water_saturation_grid=new_water_saturation_grid,
        gas_saturation_grid=new_gas_saturation_grid,
        solution_gas_to_oil_ratio_grid=corrected_solution_gas_to_oil_ratio_grid,
        gas_solubility_in_water_grid=corrected_gas_solubility_in_water_grid,
    )


def update_residual_saturation_grids(
    rock_properties: RockProperties[ThreeDimensions],
    saturation_history: SaturationHistory[ThreeDimensions],
    water_saturation_grid: NDimensionalGrid[ThreeDimensions],
    gas_saturation_grid: NDimensionalGrid[ThreeDimensions],
    residual_oil_drainage_ratio_water_flood: float = 0.6,  # Sorw_drainage = 0.6 x Sorw_imbibition
    residual_oil_drainage_ratio_gas_flood: float = 0.6,  # Sorg_drainage = 0.6 x Sorg_imbibition
    residual_gas_drainage_ratio: float = 0.5,  # Sgr_drainage = 0.5 x Sgr_imbibition
    tolerance: float = 1e-6,
) -> typing.Tuple[RockProperties[ThreeDimensions], SaturationHistory[ThreeDimensions]]:
    """
    Updates the effective residual saturation grids based on current displacement regimes
    (drainage or imbibition) determined from saturation history and current saturations.

    :param rock_properties: Current rock properties including residual saturations
    :param saturation_history: Current saturation history including max saturations and imbibition flags
    :param water_saturation_grid: Current water saturation grid
    :param gas_saturation_grid: Current gas saturation grid
    :param residual_oil_drainage_ratio_water_flood: Ratio to compute oil drainage residual from imbibition value.
    :param residual_gas_drainage_ratio: Ratio to compute gas drainage residual from imbibition value.
    :param residual_oil_drainage_ratio_gas_flood: Ratio to compute oil drainage residual from gas flooding imbibition value.
    :param tolerance: Tolerance to determine significant saturation changes.
    :return: Tuple of updated `RockProperties` and `SaturationHistory` with new effective residual saturations
    """
    residual_oil_saturation_water_grid = (
        rock_properties.residual_oil_saturation_water_grid
    )
    residual_oil_saturation_gas_grid = rock_properties.residual_oil_saturation_gas_grid
    residual_gas_saturation_grid = rock_properties.residual_gas_saturation_grid
    max_water_saturation_grid = saturation_history.max_water_saturation_grid
    max_gas_saturation_grid = saturation_history.max_gas_saturation_grid
    water_imbibition_flag_grid = saturation_history.water_imbibition_flag_grid
    gas_imbibition_flag_grid = saturation_history.gas_imbibition_flag_grid

    (
        new_max_water_saturation_grid,
        new_max_gas_saturation_grid,
        effective_residual_oil_saturation_water_grid,
        effective_residual_oil_saturation_gas_grid,
        effective_residual_gas_saturation_grid,
        new_water_imbibition_flag_grid,
        new_gas_imbibition_flag_grid,
    ) = build_effective_residual_saturation_grids(
        water_saturation_grid=water_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
        residual_gas_saturation_grid=residual_gas_saturation_grid,
        max_water_saturation_grid=max_water_saturation_grid,
        max_gas_saturation_grid=max_gas_saturation_grid,
        water_imbibition_flag_grid=water_imbibition_flag_grid,
        gas_imbibition_flag_grid=gas_imbibition_flag_grid,
        residual_oil_drainage_ratio_water_flood=residual_oil_drainage_ratio_water_flood,
        residual_oil_drainage_ratio_gas_flood=residual_oil_drainage_ratio_gas_flood,
        residual_gas_drainage_ratio=residual_gas_drainage_ratio,
        tolerance=tolerance,
    )

    updated_rock_properties = attrs.evolve(
        rock_properties,
        residual_oil_saturation_water_grid=effective_residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=effective_residual_oil_saturation_gas_grid,
        residual_gas_saturation_grid=effective_residual_gas_saturation_grid,
    )
    updated_saturation_history = attrs.evolve(
        saturation_history,
        max_water_saturation_grid=new_max_water_saturation_grid,
        max_gas_saturation_grid=new_max_gas_saturation_grid,
        water_imbibition_flag_grid=new_water_imbibition_flag_grid,
        gas_imbibition_flag_grid=new_gas_imbibition_flag_grid,
    )
    return updated_rock_properties, updated_saturation_history

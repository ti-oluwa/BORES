import logging
import typing

import attrs

from bores.blackoil.tables.pvt import PVTTables
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
    build_water_bubble_point_pressure_grid,
    build_water_compressibility_grid,
    build_water_density_grid,
    build_water_formation_volume_factor_grid,
    build_water_viscosity_grid,
    buildsolution_gas_to_oil_ratio_grid,
)
from bores.grids.rock_fluid import build_effective_residual_saturation_grids
from bores.reservoir import FluidProperties, Hysteresis, RockProperties
from bores.types import MiscibilityModel, NDimensionalGrid, ThreeDimensions
from bores.wells.base import Wells

logger = logging.getLogger(__name__)


def update_fluid_properties(
    fluid_properties: FluidProperties[ThreeDimensions],
    miscibility_model: MiscibilityModel = "immiscible",
    wells: Wells[ThreeDimensions] | None = None,
    pvt_tables: PVTTables | None = None,
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
        - If current pressure >= Pb: GOR = GOR at Pb (Rs = Rs_b).
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

    :return: Updated `FluidProperties` object with recalculated gas, water, and oil properties.
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
        new_water_formation_volume_factor_grid = build_water_formation_volume_factor_grid(
            water_density_grid=new_water_density_grid,  # Use new density here
            salinity_grid=fluid_properties.water_salinity_grid,
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
        new_water_formation_volume_factor_grid = water_pvt_table.formation_volume_factor(
            pressure=pressure_grid,
            temperature=temperature_grid,
            salinity=water_salinity_grid,
            bubble_point_pressure=new_water_bubble_point_pressure_grid,
        )
        new_water_viscosity_grid = water_pvt_table.viscosity(
            pressure=pressure_grid,
            temperature=temperature_grid,
            salinity=water_salinity_grid,
            bubble_point_pressure=new_water_bubble_point_pressure_grid,
        )

    # OIL PROPERTIES (tricky due to bubble point)
    # Make sure to always compute the oil bubble point pressure grid
    # before the gas to oil ratio grid, as the latter depends on the former.

    if oil_pvt_table is None:
        # Compute New bubble point using Current Rs
        if freeze_saturation_pressure:
            # Keep current bubble point constant
            new_oil_bubble_point_pressure_grid = fluid_properties.oil_bubble_point_pressure_grid
        else:
            # print(fluid_properties.solution_gas_to_oil_ratio_grid.dtype)
            # Recompute bubble point using current Rs
            new_oil_bubble_point_pressure_grid = build_oil_bubble_point_pressure_grid(
                gas_gravity_grid=fluid_properties.gas_gravity_grid,
                oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
                temperature_grid=temperature_grid,
                solution_gas_to_oil_ratio_grid=fluid_properties.solution_gas_to_oil_ratio_grid,
            )

        # Compute Rs at New bubble point
        gor_at_bubble_point_pressure_grid = buildsolution_gas_to_oil_ratio_grid(
            pressure_grid=new_oil_bubble_point_pressure_grid,  # New bubble point here
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,  # Use same New bubble point here
            gas_gravity_grid=fluid_properties.gas_gravity_grid,
            oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
        )
        # Compute New Rs at current pressure
        new_solution_gas_to_oil_ratio_grid = buildsolution_gas_to_oil_ratio_grid(
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
            new_oil_bubble_point_pressure_grid = fluid_properties.oil_bubble_point_pressure_grid
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
            bubble_point_pressure=new_oil_bubble_point_pressure_grid,
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
            bubble_point_pressure=new_oil_bubble_point_pressure_grid,
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
            bubble_point_pressure=new_oil_bubble_point_pressure_grid,
        )

    new_oil_effective_viscosity_grid = new_oil_viscosity_grid
    # If there are miscible injections, update the effective oil viscosity and density grids
    if miscibility_model == "todd-longstaff" and wells is not None:
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


def update_residual_saturation_grids(
    rock_properties: RockProperties[ThreeDimensions],
    hysteresis_state: Hysteresis[ThreeDimensions],
    water_saturation_grid: NDimensionalGrid[ThreeDimensions],
    gas_saturation_grid: NDimensionalGrid[ThreeDimensions],
    old_water_saturation_grid: NDimensionalGrid[ThreeDimensions],
    old_gas_saturation_grid: NDimensionalGrid[ThreeDimensions],
    residual_oil_drainage_ratio_water_flood: float = 0.6,  # sorw_drainage = 0.6 x sorw_imbibition
    residual_oil_drainage_ratio_gas_flood: float = 0.6,  # sorg_drainage = 0.6 x sorg_imbibition
    residual_gas_drainage_ratio: float = 0.5,  # sgr_drainage = 0.5 x sgr_imbibition
    tolerance: float = 1e-6,
) -> tuple[RockProperties[ThreeDimensions], Hysteresis[ThreeDimensions]]:
    """
    Updates the effective residual saturation grids based on current displacement regimes
    (drainage or imbibition) determined from hysteresis state and current saturations.

    :param rock_properties: Current rock properties including residual saturations
    :param hysteresis_state: Current hysteresis state including max saturations and imbibition flags
    :param water_saturation_grid: Current water saturation grid
    :param gas_saturation_grid: Current gas saturation grid
    :param residual_oil_drainage_ratio_water_flood: Ratio to compute oil drainage residual from imbibition value.
    :param residual_gas_drainage_ratio: Ratio to compute gas drainage residual from imbibition value.
    :param residual_oil_drainage_ratio_gas_flood: Ratio to compute oil drainage residual from gas flooding imbibition value.
    :param tolerance: Tolerance to determine significant saturation changes.
    :return: Tuple of updated `RockProperties` and `Hysteresis` with new effective residual saturations
    """
    residual_oil_saturation_water_grid = rock_properties.residual_oil_saturation_water_grid
    residual_oil_saturation_gas_grid = rock_properties.residual_oil_saturation_gas_grid
    residual_gas_saturation_grid = rock_properties.residual_gas_saturation_grid
    max_water_saturation_grid = hysteresis_state.max_water_saturation_grid
    max_gas_saturation_grid = hysteresis_state.max_gas_saturation_grid
    water_imbibition_flag_grid = hysteresis_state.water_imbibition_flag_grid
    gas_imbibition_flag_grid = hysteresis_state.gas_imbibition_flag_grid
    water_reversal_saturation_grid = hysteresis_state.water_reversal_saturation_grid
    gas_reversal_saturation_grid = hysteresis_state.gas_reversal_saturation_grid

    (
        new_max_water_saturation_grid,
        new_max_gas_saturation_grid,
        effective_residual_oil_saturation_water_grid,
        effective_residual_oil_saturation_gas_grid,
        effective_residual_gas_saturation_grid,
        new_water_imbibition_flag_grid,
        new_gas_imbibition_flag_grid,
        new_water_reversal_saturation_grid,
        new_gas_reversal_saturation_grid,
    ) = build_effective_residual_saturation_grids(
        water_saturation_grid=water_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        old_water_saturation_grid=old_water_saturation_grid,
        old_gas_saturation_grid=old_gas_saturation_grid,
        residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
        residual_gas_saturation_grid=residual_gas_saturation_grid,
        max_water_saturation_grid=max_water_saturation_grid,
        max_gas_saturation_grid=max_gas_saturation_grid,
        water_imbibition_flag_grid=water_imbibition_flag_grid,
        gas_imbibition_flag_grid=gas_imbibition_flag_grid,
        water_reversal_saturation_grid=water_reversal_saturation_grid,
        gas_reversal_saturation_grid=gas_reversal_saturation_grid,
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
    updated_hysteresis_state = attrs.evolve(
        hysteresis_state,
        max_water_saturation_grid=new_max_water_saturation_grid,
        max_gas_saturation_grid=new_max_gas_saturation_grid,
        water_imbibition_flag_grid=new_water_imbibition_flag_grid,
        gas_imbibition_flag_grid=new_gas_imbibition_flag_grid,
        water_reversal_saturation_grid=new_water_reversal_saturation_grid,
        gas_reversal_saturation_grid=new_gas_reversal_saturation_grid,
    )
    return updated_rock_properties, updated_hysteresis_state

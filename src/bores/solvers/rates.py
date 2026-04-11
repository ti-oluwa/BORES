import typing

import numpy as np

from bores.config import Config
from bores.constants import c
from bores.datastructures import PhaseTensorsProxy
from bores.models import FluidProperties
from bores.types import FluidPhase, ThreeDimensionalGrid, ThreeDimensions
from bores.wells.base import Wells
from bores.wells.core import (
    compute_average_compressibility_factor,
    compute_gas_well_rate,
    compute_oil_well_rate,
    get_pseudo_pressure_table,
)
from bores.wells.indices import WellIndicesCache


def compute_well_rates(
    new_pressure_grid: ThreeDimensionalGrid,
    temperature_grid: ThreeDimensionalGrid,
    water_relative_mobility_grid: ThreeDimensionalGrid,
    oil_relative_mobility_grid: ThreeDimensionalGrid,
    gas_relative_mobility_grid: ThreeDimensionalGrid,
    water_compressibility_grid: ThreeDimensionalGrid,
    oil_compressibility_grid: ThreeDimensionalGrid,
    fluid_properties: FluidProperties[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    config: Config,
    well_indices_cache: WellIndicesCache,
    injection_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    production_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    injection_rates: PhaseTensorsProxy[float, ThreeDimensions],
    production_rates: PhaseTensorsProxy[float, ThreeDimensions],
    injection_fvfs: PhaseTensorsProxy[float, ThreeDimensions],
    production_fvfs: PhaseTensorsProxy[float, ThreeDimensions],
) -> None:
    """
    Compute well flow rates using the new pressure from the pressure solve and
    the BHPs stored during the Jacobian assembly step.

    This function is called after the pressure solve. It uses the new pressure
    grid and the BHPs computed during `compute_well_contributions` to evaluate
    accurate per-phase flow rates for use in the saturation solver.

    Rates are computed using per-phase mobility (not total mobility) and with
    pseudo-pressure enabled for gas phases where appropriate. This gives
    physically accurate rates consistent with the BHP established during the
    pressure solve.

    The split between BHP computation (in `compute_well_contributions`) and
    rate computation (here) ensures:
        - The Jacobian uses total mobility for numerical stability
        - Rates use phase mobility for physical accuracy
        - Gas rates use pseudo-pressure where appropriate
        - All rates are consistent with the new pressure, not the old pressure

    :param new_pressure_grid: New oil pressure grid after pressure solve (psi)
    :param temperature_grid: Temperature grid (°F)
    :param water_relative_mobility_grid: Water relative mobility grid (1/cP)
    :param oil_relative_mobility_grid: Oil relative mobility grid (1/cP)
    :param gas_relative_mobility_grid: Gas relative mobility grid (1/cP)
    :param water_compressibility_grid: Water compressibility grid (1/psi)
    :param oil_compressibility_grid: Oil compressibility grid (1/psi)
    :param gas_compressibility_grid: Gas compressibility grid (1/psi)
    :param fluid_properties: Full fluid properties container
    :param wells: Wells container with injection and production wells
    :param config: Simulation configuration
    :param well_indices_cache: Cache of well indices
    :param injection_bhps: BHPs stored during pressure solve for injection wells
    :param production_bhps: BHPs stored during pressure solve for production wells
    :param injection_rates: Output proxy for injection rates (water, oil, gas) ft³/day
    :param production_rates: Output proxy for production rates (water, oil, gas) ft³/day
    :param injection_fvfs: Output proxy for injection formation volume factors
    :param production_fvfs: Output proxy for production formation volume factors
    """
    bbl_to_ft3 = c.BARRELS_TO_CUBIC_FEET
    incompressibility_threshold = c.FLUID_INCOMPRESSIBILITY_THRESHOLD

    gas_formation_volume_factor_grid = fluid_properties.gas_formation_volume_factor_grid
    water_fvf_grid = fluid_properties.water_formation_volume_factor_grid
    oil_fvf_grid = fluid_properties.oil_formation_volume_factor_grid
    water_bubble_point_pressure_grid = fluid_properties.water_bubble_point_pressure_grid
    gas_solubility_in_water_grid = fluid_properties.gas_solubility_in_water_grid

    # Injection wells
    for well in wells.injection_wells:
        if not well.is_open or well.injected_fluid is None:
            continue

        injected_fluid = well.injected_fluid
        injected_phase = injected_fluid.phase
        use_pseudo_pressure = (
            config.use_pseudo_pressure and injected_phase == FluidPhase.GAS
        )
        well_indices = well_indices_cache.injection[well.name]

        for perforation_index in well_indices:
            i, j, k = perforation_index.cell
            well_index = perforation_index.well_index
            new_pressure = typing.cast(float, new_pressure_grid[i, j, k])
            cell_temperature = typing.cast(float, temperature_grid[i, j, k])

            # Retrieve BHP stored during pressure solve.
            # injection_bhps stores (water_bhp, 0, gas_bhp) depending on phase.
            if injected_phase == FluidPhase.GAS:
                bhp = injection_bhps.gas[i, j, k]
            else:
                bhp = injection_bhps.water[i, j, k]

            if not np.isfinite(bhp) or bhp == 0.0:
                # Perforation was skipped during pressure solve, skip here too.
                continue

            # Compute FVF at new pressure
            phase_fvf = typing.cast(
                float,
                injected_fluid.get_formation_volume_factor(
                    pressure=new_pressure,
                    temperature=cell_temperature,
                ),
            )

            if injected_phase == FluidPhase.GAS:
                phase_mobility = typing.cast(
                    float, gas_relative_mobility_grid[i, j, k]
                )
                # Build pseudo-pressure table if needed
                use_pp, pp_table = get_pseudo_pressure_table(
                    fluid=injected_fluid,
                    pressure=new_pressure,
                    temperature=cell_temperature,
                    use_pseudo_pressure=use_pseudo_pressure,
                    pvt_tables=config.pvt_tables,
                )
                specific_gravity = typing.cast(
                    float,
                    injected_fluid.get_specific_gravity(
                        pressure=new_pressure,
                        temperature=cell_temperature,
                    ),
                )
                avg_z_factor = compute_average_compressibility_factor(
                    pressure=new_pressure,
                    temperature=cell_temperature,
                    gas_gravity=specific_gravity,
                    bottom_hole_pressure=bhp,
                )
                flow_rate = compute_gas_well_rate(
                    well_index=well_index,
                    pressure=new_pressure,
                    temperature=cell_temperature,
                    bottom_hole_pressure=bhp,
                    phase_mobility=phase_mobility,
                    use_pseudo_pressure=use_pp,
                    pseudo_pressure_table=pp_table,
                    average_compressibility_factor=avg_z_factor,
                    formation_volume_factor=phase_fvf,
                )
                injection_rates[i, j, k] = (0.0, 0.0, flow_rate)
                injection_fvfs[i, j, k] = (0.0, 0.0, phase_fvf)

            else:
                # Water injection
                if well.injected_fluid.phase == FluidPhase.WATER:
                    compressibility_kwargs = {
                        "bubble_point_pressure": water_bubble_point_pressure_grid[
                            i, j, k
                        ],
                        "gas_formation_volume_factor": gas_formation_volume_factor_grid[
                            i, j, k
                        ],
                        "gas_solubility_in_water": gas_solubility_in_water_grid[
                            i, j, k
                        ],
                    }
                else:
                    compressibility_kwargs = {}
                
                phase_compressibility = typing.cast(
                    float,
                    injected_fluid.get_compressibility(
                        pressure=new_pressure,
                        temperature=cell_temperature,
                        **compressibility_kwargs,
                    ),
                )
                phase_mobility = typing.cast(
                    float, water_relative_mobility_grid[i, j, k]
                )
                flow_rate = (
                    compute_oil_well_rate(
                        well_index=well_index,
                        pressure=new_pressure,
                        bottom_hole_pressure=bhp,
                        phase_mobility=phase_mobility,
                        fluid_compressibility=phase_compressibility,
                        incompressibility_threshold=incompressibility_threshold,
                    )
                    * bbl_to_ft3
                )
                injection_rates[i, j, k] = (flow_rate, 0.0, 0.0)
                injection_fvfs[i, j, k] = (phase_fvf, 0.0, 0.0)

    # Production wells
    for well in wells.production_wells:
        if not well.is_open:
            continue

        well_indices = well_indices_cache.production[well.name]
        for perforation_index in well_indices:
            i, j, k = perforation_index.cell
            well_index = perforation_index.well_index
            new_pressure = typing.cast(float, new_pressure_grid[i, j, k])
            cell_temperature = typing.cast(float, temperature_grid[i, j, k])

            # Retrieve shared BHP stored during pressure solve.
            # All three entries are the same shared BHP — use index 1 (oil)
            # as the canonical stored value but any index works since they
            # are all written as shared_bhp during compute_well_contributions.
            shared_bhp = production_bhps.oil[i, j, k]
            if not np.isfinite(shared_bhp) or shared_bhp == 0.0:
                # Perforation was skipped during pressure solve, skip here too.
                continue

            water_rate = 0.0
            oil_rate = 0.0
            gas_rate = 0.0
            water_fvf = 0.0
            oil_fvf = 0.0
            gas_fvf = 0.0

            for produced_fluid in well.produced_fluids:
                produced_phase = produced_fluid.phase
                use_pseudo_pressure = (
                    config.use_pseudo_pressure and produced_phase == FluidPhase.GAS
                )

                if produced_phase == FluidPhase.GAS:
                    phase_mobility = typing.cast(
                        float, gas_relative_mobility_grid[i, j, k]
                    )
                    phase_fvf = typing.cast(
                        float, gas_formation_volume_factor_grid[i, j, k]
                    )
                    use_pp, pp_table = get_pseudo_pressure_table(
                        fluid=produced_fluid,
                        pressure=new_pressure,
                        temperature=cell_temperature,
                        use_pseudo_pressure=use_pseudo_pressure,
                        pvt_tables=config.pvt_tables,
                    )
                    specific_gravity = typing.cast(
                        float,
                        produced_fluid.get_specific_gravity(
                            pressure=new_pressure,
                            temperature=cell_temperature,
                        ),
                    )
                    avg_z_factor = compute_average_compressibility_factor(
                        pressure=new_pressure,
                        temperature=cell_temperature,
                        gas_gravity=specific_gravity,
                        bottom_hole_pressure=shared_bhp,
                    )
                    flow_rate = compute_gas_well_rate(
                        well_index=well_index,
                        pressure=new_pressure,
                        temperature=cell_temperature,
                        bottom_hole_pressure=shared_bhp,
                        phase_mobility=phase_mobility,
                        use_pseudo_pressure=use_pp,
                        pseudo_pressure_table=pp_table,
                        average_compressibility_factor=avg_z_factor,
                        formation_volume_factor=phase_fvf,
                    )
                    gas_rate += flow_rate
                    gas_fvf = phase_fvf

                elif produced_phase == FluidPhase.WATER:
                    phase_mobility = typing.cast(
                        float, water_relative_mobility_grid[i, j, k]
                    )
                    phase_fvf = typing.cast(float, water_fvf_grid[i, j, k])
                    phase_compressibility = typing.cast(
                        float, water_compressibility_grid[i, j, k]
                    )
                    flow_rate = (
                        compute_oil_well_rate(
                            well_index=well_index,
                            pressure=new_pressure,
                            bottom_hole_pressure=shared_bhp,
                            phase_mobility=phase_mobility,
                            fluid_compressibility=phase_compressibility,
                            incompressibility_threshold=incompressibility_threshold,
                        )
                        * bbl_to_ft3
                    )
                    water_rate += flow_rate
                    water_fvf = phase_fvf

                else:  # oil
                    phase_mobility = typing.cast(
                        float, oil_relative_mobility_grid[i, j, k]
                    )
                    phase_fvf = typing.cast(float, oil_fvf_grid[i, j, k])
                    phase_compressibility = typing.cast(
                        float, oil_compressibility_grid[i, j, k]
                    )
                    flow_rate = (
                        compute_oil_well_rate(
                            well_index=well_index,
                            pressure=new_pressure,
                            bottom_hole_pressure=shared_bhp,
                            phase_mobility=phase_mobility,
                            fluid_compressibility=phase_compressibility,
                            incompressibility_threshold=incompressibility_threshold,
                        )
                        * bbl_to_ft3
                    )
                    oil_rate += flow_rate
                    oil_fvf = phase_fvf

            production_rates[i, j, k] = (water_rate, oil_rate, gas_rate)
            production_fvfs[i, j, k] = (water_fvf, oil_fvf, gas_fvf)

import typing

import attrs
import numpy as np
import numpy.typing as npt

from bores.config import Config
from bores.constants import c
from bores.datastructures import (
    BottomHolePressure,
    BottomHolePressures,
    FormationVolumeFactors,
    Rates,
    SparseTensor,
)
from bores.models import FluidProperties
from bores.solvers.base import _warn_injection_rate, _warn_production_rate
from bores.types import (
    FluidPhase,
    NDimension,
    NDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensions,
)
from bores.wells.base import Wells
from bores.wells.controls import CoupledRateControl
from bores.wells.indices import WellsIndices


@attrs.frozen
class WellRates(typing.Generic[NDimension]):
    """
    Container for all well rate quantities computed in a single explicit pass
    over all injection and production perforations.

    The scalar grids (`net_well_rate_grid`, `net_water_well_rate_grid`, etc.)
    are 3-D arrays of shape `(nx, ny, nz)` in ft³/day. Positive values denote
    injection (source) and negative values denote production (sink).

    The per-phase tensors (`injection_rates`, `production_rates`, etc.) use
    `(water, oil, gas)` ordering throughout and carry the same sign convention.
    """

    net_well_rate_grid: NDimensionalGrid[ThreeDimensions]
    """Total net volumetric rate per cell across all phases (ft³/day)."""

    net_water_well_rate_grid: NDimensionalGrid[ThreeDimensions]
    """Net volumetric water rate per cell (ft³/day)."""

    net_oil_well_rate_grid: NDimensionalGrid[ThreeDimensions]
    """Net volumetric oil rate per cell (ft³/day)."""

    net_gas_well_rate_grid: NDimensionalGrid[ThreeDimensions]
    """Net volumetric gas rate per cell (ft³/day)."""

    net_water_well_mass_rate_grid: NDimensionalGrid[ThreeDimensions]
    """Net water mass rate per cell (lbm/day)."""

    net_oil_well_mass_rate_grid: NDimensionalGrid[ThreeDimensions]
    """Net oil mass rate per cell (lbm/day)."""

    net_gas_well_mass_rate_grid: NDimensionalGrid[ThreeDimensions]
    """Net gas mass rate per cell (lbm/day)."""

    injection_rates: Rates[float, ThreeDimensions]
    """Per-phase volumetric injection rates (ft³/day)."""

    production_rates: Rates[float, ThreeDimensions]
    """Per-phase volumetric production rates (ft³/day)."""

    injection_mass_rates: Rates[float, ThreeDimensions]
    """Per-phase injection mass rates (lbm/day)."""

    production_mass_rates: Rates[float, ThreeDimensions]
    """Per-phase production mass rates (lbm/day)."""

    injection_fvfs: FormationVolumeFactors[float, ThreeDimensions]
    """Formation volume factors at injection perforation conditions."""

    production_fvfs: FormationVolumeFactors[float, ThreeDimensions]
    """Formation volume factors at production perforation conditions."""

    injection_bhps: BottomHolePressures[float, ThreeDimensions]
    """Effective bottom hole pressures for injection perforations (psi)."""

    production_bhps: BottomHolePressures[float, ThreeDimensions]
    """Effective bottom hole pressures for production perforations (psi)."""


def compute_well_rates(
    fluid_properties: FluidProperties[ThreeDimensions],
    water_relative_mobility_grid: ThreeDimensionalGrid,
    oil_relative_mobility_grid: ThreeDimensionalGrid,
    gas_relative_mobility_grid: ThreeDimensionalGrid,
    wells: Wells[ThreeDimensions],
    time: float,
    config: Config,
    wells_indices: WellsIndices,
    dtype: npt.DTypeLike,
) -> WellRates[ThreeDimensions]:
    """
    Compute explicit well rates for all injection and production perforations
    and return all scalar grids and per-phase tensors packed into a `WellRates` container.

    :param fluid_properties: Fluid properties container holding PVT state at
        the current pressure level.
    :param water_relative_mobility_grid: Water relative mobility grid (1/cP).
    :param oil_relative_mobility_grid: Oil relative mobility grid (1/cP).
    :param gas_relative_mobility_grid: Gas relative mobility grid (1/cP).
    :param wells: Wells container with injection and production well definitions.
    :param time: Total simulation time elapsed, this time step inclusive (s).
    :param config: Simulation configuration.
    :param wells_indices: Cache of well indices and allocation fractions.
    :param dtype: NumPy dtype for the returned grid arrays.
    :return: `WellRates` container holding the total volumetric well-rate grid,
        the six per-phase volumetric and mass rate grids, and the per-phase
        rate, mass rate, fvf, and bhp tensors. All grids use ft³/day for
        volumetric rates and lbm/day for mass rates, with positive values
        denoting injection and negative values denoting production.
    """
    bbl_to_ft3 = c.BARRELS_TO_CUBIC_FEET
    pressure_grid = fluid_properties.pressure_grid
    grid_shape = pressure_grid.shape
    cell_count_x, cell_count_y, cell_count_z = grid_shape

    net_well_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    net_water_well_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    net_oil_well_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    net_gas_well_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    net_water_well_mass_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    net_oil_well_mass_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    net_gas_well_mass_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )

    injection_rates = Rates(
        oil=SparseTensor(grid_shape, dtype=float),
        water=SparseTensor(grid_shape, dtype=float),
        gas=SparseTensor(grid_shape, dtype=float),
    )
    production_rates = Rates(
        oil=SparseTensor(grid_shape, dtype=float),
        water=SparseTensor(grid_shape, dtype=float),
        gas=SparseTensor(grid_shape, dtype=float),
    )
    injection_mass_rates = Rates(
        oil=SparseTensor(grid_shape, dtype=float),
        water=SparseTensor(grid_shape, dtype=float),
        gas=SparseTensor(grid_shape, dtype=float),
    )
    production_mass_rates = Rates(
        oil=SparseTensor(grid_shape, dtype=float),
        water=SparseTensor(grid_shape, dtype=float),
        gas=SparseTensor(grid_shape, dtype=float),
    )
    injection_fvfs = FormationVolumeFactors(
        oil=SparseTensor(grid_shape, dtype=float, default=np.nan),
        water=SparseTensor(grid_shape, dtype=float, default=np.nan),
        gas=SparseTensor(grid_shape, dtype=float, default=np.nan),
    )
    production_fvfs = FormationVolumeFactors(
        oil=SparseTensor(grid_shape, dtype=float, default=np.nan),
        water=SparseTensor(grid_shape, dtype=float, default=np.nan),
        gas=SparseTensor(grid_shape, dtype=float, default=np.nan),
    )
    injection_bhps = BottomHolePressures(
        oil=BottomHolePressure(grid_shape, dtype=float, default=np.nan),
        water=BottomHolePressure(grid_shape, dtype=float, default=np.nan),
        gas=BottomHolePressure(grid_shape, dtype=float, default=np.nan),
    )
    production_bhps = BottomHolePressures(
        oil=BottomHolePressure(grid_shape, dtype=float, default=np.nan),
        water=BottomHolePressure(grid_shape, dtype=float, default=np.nan),
        gas=BottomHolePressure(grid_shape, dtype=float, default=np.nan),
    )

    temperature_grid = fluid_properties.temperature_grid
    water_compressibility_grid = fluid_properties.water_compressibility_grid
    oil_compressibility_grid = fluid_properties.oil_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid
    oil_density_grid = fluid_properties.oil_effective_density_grid
    water_density_grid = fluid_properties.water_density_grid
    gas_density_grid = fluid_properties.gas_density_grid
    oil_viscosity_grid = fluid_properties.oil_effective_viscosity_grid
    water_viscosity_grid = fluid_properties.water_viscosity_grid
    gas_viscosity_grid = fluid_properties.gas_viscosity_grid
    gas_formation_volume_factor_grid = fluid_properties.gas_formation_volume_factor_grid
    oil_formation_volume_factor_grid = fluid_properties.oil_formation_volume_factor_grid
    water_formation_volume_factor_grid = (
        fluid_properties.water_formation_volume_factor_grid
    )
    water_bubble_point_pressure_grid = fluid_properties.water_bubble_point_pressure_grid
    gas_solubility_in_water_grid = fluid_properties.gas_solubility_in_water_grid

    # Injection wells
    for well in wells.injection_wells:
        if not well.is_open or well.injected_fluid is None:
            continue

        injected_fluid = well.injected_fluid
        injected_phase = injected_fluid.phase
        is_gas = injected_phase == FluidPhase.GAS
        use_pseudo_pressure = config.use_pseudo_pressure and is_gas
        indices = wells_indices.injection[well.name]

        for perforation_index in indices:
            i, j, k = perforation_index.cell
            well_index = perforation_index.well_index
            allocation_fraction = indices.get_allocation_fraction(perforation_index)
            cell_pressure = typing.cast(float, pressure_grid[i, j, k])
            cell_temperature = typing.cast(float, temperature_grid[i, j, k])

            phase_fvf = typing.cast(
                float,
                injected_fluid.get_formation_volume_factor(
                    pressure=cell_pressure, temperature=cell_temperature
                ),
            )
            phase_density = typing.cast(
                float,
                injected_fluid.get_density(
                    pressure=cell_pressure, temperature=cell_temperature
                ),
            )
            phase_viscosity = typing.cast(
                float,
                injected_fluid.get_viscosity(
                    pressure=cell_pressure, temperature=cell_temperature
                ),
            )

            if is_gas:
                phase_mobility = typing.cast(float, gas_relative_mobility_grid[i, j, k])
                phase_compressibility = typing.cast(
                    float,
                    injected_fluid.get_compressibility(
                        pressure=cell_pressure, temperature=cell_temperature
                    ),
                )
            else:
                phase_mobility = typing.cast(
                    float, water_relative_mobility_grid[i, j, k]
                )
                phase_compressibility = typing.cast(
                    float,
                    injected_fluid.get_compressibility(
                        pressure=cell_pressure,
                        temperature=cell_temperature,
                        bubble_point_pressure=water_bubble_point_pressure_grid[i, j, k],
                        gas_formation_volume_factor=gas_formation_volume_factor_grid[
                            i, j, k
                        ],
                        gas_solubility_in_water=gas_solubility_in_water_grid[i, j, k],
                    ),
                )

            total_mobility = typing.cast(
                float,
                water_relative_mobility_grid[i, j, k]
                + oil_relative_mobility_grid[i, j, k]
                + gas_relative_mobility_grid[i, j, k],
            )
            flow_rate, effective_bhp = well.get_control(
                pressure=cell_pressure,
                temperature=cell_temperature,
                well_index=well_index,
                phase_viscosity=phase_viscosity,
                phase_mobility=total_mobility,
                fluid=injected_fluid,
                fluid_compressibility=phase_compressibility,
                use_pseudo_pressure=use_pseudo_pressure,
                formation_volume_factor=phase_fvf,
                allocation_fraction=allocation_fraction,
                pvt_tables=None,
            )
            if flow_rate < 0.0 and config.warn_well_anomalies:
                _warn_injection_rate(
                    injection_rate=flow_rate,
                    well_name=well.name,
                    time=time,
                    cell=(i, j, k),
                    rate_unit="ft³/day" if is_gas else "bbls/day",
                )

            if is_gas:
                net_well_rate_grid[i, j, k] += flow_rate
                net_gas_well_rate_grid[i, j, k] += flow_rate
                net_gas_well_mass_rate_grid[i, j, k] += flow_rate * phase_density
                injection_rates.gas[i, j, k] = flow_rate
                injection_mass_rates.gas[i, j, k] = flow_rate * phase_density
                injection_fvfs.gas[i, j, k] = phase_fvf
                if effective_bhp != cell_pressure:
                    injection_bhps.gas[i, j, k] = effective_bhp
            else:
                flow_rate *= bbl_to_ft3
                net_well_rate_grid[i, j, k] += flow_rate
                net_water_well_rate_grid[i, j, k] += flow_rate
                net_water_well_mass_rate_grid[i, j, k] += flow_rate * phase_density
                injection_rates.water[i, j, k] += flow_rate
                injection_mass_rates.water[i, j, k] += flow_rate * phase_density
                injection_fvfs.water[i, j, k] = phase_fvf
                if effective_bhp != cell_pressure:
                    injection_bhps.water[i, j, k] = effective_bhp

    # Production wells
    for well in wells.production_wells:
        if not well.is_open:
            continue

        is_couple_controlled = isinstance(well.control, CoupledRateControl)
        indices = wells_indices.production[well.name]

        for perforation_index in indices:
            i, j, k = perforation_index.cell
            well_index = perforation_index.well_index
            allocation_fraction = indices.get_allocation_fraction(perforation_index)
            cell_pressure = typing.cast(float, pressure_grid[i, j, k])
            cell_temperature = typing.cast(float, temperature_grid[i, j, k])

            context: dict = {}
            if is_couple_controlled:
                context = well.control.build_context(  # type: ignore
                    produced_fluids=well.produced_fluids,
                    oil_mobility=oil_relative_mobility_grid[i, j, k],
                    water_mobility=water_relative_mobility_grid[i, j, k],
                    gas_mobility=gas_relative_mobility_grid[i, j, k],
                    oil_fvf=oil_formation_volume_factor_grid[i, j, k],
                    water_fvf=water_formation_volume_factor_grid[i, j, k],
                    gas_fvf=gas_formation_volume_factor_grid[i, j, k],
                    oil_compressibility=oil_compressibility_grid[i, j, k],
                    water_compressibility=water_compressibility_grid[i, j, k],
                    gas_compressibility=gas_compressibility_grid[i, j, k],
                    oil_viscosity=oil_viscosity_grid[i, j, k],
                    gas_viscosity=gas_viscosity_grid[i, j, k],
                    water_viscosity=water_viscosity_grid[i, j, k],
                )

            water_flow_rate = 0.0
            oil_flow_rate = 0.0
            gas_flow_rate = 0.0
            water_mass_flow_rate = 0.0
            oil_mass_flow_rate = 0.0
            gas_mass_flow_rate = 0.0
            water_phase_fvf = np.nan
            oil_phase_fvf = np.nan
            gas_phase_fvf = np.nan
            water_effective_bhp = np.nan
            oil_effective_bhp = np.nan
            gas_effective_bhp = np.nan

            for produced_fluid in well.produced_fluids:
                produced_phase = produced_fluid.phase
                is_gas = produced_phase == FluidPhase.GAS
                is_water = produced_phase == FluidPhase.WATER
                use_pseudo_pressure = config.use_pseudo_pressure and is_gas

                if is_gas:
                    phase_mobility = typing.cast(
                        float, gas_relative_mobility_grid[i, j, k]
                    )
                    phase_density = typing.cast(float, gas_density_grid[i, j, k])
                    phase_compressibility = typing.cast(
                        float, gas_compressibility_grid[i, j, k]
                    )
                    phase_fvf = typing.cast(
                        float, gas_formation_volume_factor_grid[i, j, k]
                    )
                    phase_viscosity = typing.cast(float, gas_viscosity_grid[i, j, k])
                elif is_water:
                    phase_mobility = typing.cast(
                        float, water_relative_mobility_grid[i, j, k]
                    )
                    phase_density = typing.cast(float, water_density_grid[i, j, k])
                    phase_compressibility = typing.cast(
                        float, water_compressibility_grid[i, j, k]
                    )
                    phase_fvf = typing.cast(
                        float, water_formation_volume_factor_grid[i, j, k]
                    )
                    phase_viscosity = typing.cast(float, water_viscosity_grid[i, j, k])
                else:  # Oil
                    phase_mobility = typing.cast(
                        float, oil_relative_mobility_grid[i, j, k]
                    )
                    phase_density = typing.cast(float, oil_density_grid[i, j, k])
                    phase_compressibility = typing.cast(
                        float, oil_compressibility_grid[i, j, k]
                    )
                    phase_fvf = typing.cast(
                        float, oil_formation_volume_factor_grid[i, j, k]
                    )
                    phase_viscosity = typing.cast(float, oil_viscosity_grid[i, j, k])

                flow_rate, effective_bhp = well.get_control(
                    pressure=cell_pressure,
                    temperature=cell_temperature,
                    well_index=well_index,
                    phase_viscosity=phase_viscosity,
                    phase_mobility=phase_mobility,
                    fluid=produced_fluid,
                    fluid_compressibility=phase_compressibility,
                    use_pseudo_pressure=use_pseudo_pressure,
                    formation_volume_factor=phase_fvf,
                    allocation_fraction=allocation_fraction,
                    pvt_tables=config.pvt_tables,
                    **context,
                )
                if flow_rate > 0.0 and config.warn_well_anomalies:
                    _warn_production_rate(
                        production_rate=flow_rate,
                        well_name=well.name,
                        time=time,
                        cell=(i, j, k),
                        rate_unit="ft³/day" if is_gas else "bbls/day",
                    )

                if is_gas:
                    net_well_rate_grid[i, j, k] += flow_rate
                    gas_flow_rate += flow_rate
                    gas_mass_flow_rate += flow_rate * phase_density
                    gas_phase_fvf = phase_fvf
                    # When bhp returned is same as cell pressure, there no drawdown
                    # so no flow. Hence, bhp should be unset
                    if effective_bhp != cell_pressure:
                        gas_effective_bhp = effective_bhp

                    net_gas_well_rate_grid[i, j, k] += flow_rate
                    net_gas_well_mass_rate_grid[i, j, k] += flow_rate * phase_density
                elif is_water:
                    flow_rate *= bbl_to_ft3
                    net_well_rate_grid[i, j, k] += flow_rate
                    water_flow_rate += flow_rate
                    water_mass_flow_rate += flow_rate * phase_density
                    water_phase_fvf = phase_fvf
                    if effective_bhp != cell_pressure:
                        water_effective_bhp = effective_bhp

                    net_water_well_rate_grid[i, j, k] += flow_rate
                    net_water_well_mass_rate_grid[i, j, k] += flow_rate * phase_density
                else:
                    flow_rate *= bbl_to_ft3
                    net_well_rate_grid[i, j, k] += flow_rate
                    oil_flow_rate += flow_rate
                    oil_mass_flow_rate += flow_rate * phase_density
                    oil_phase_fvf = phase_fvf
                    if effective_bhp != cell_pressure:
                        oil_effective_bhp = effective_bhp

                    net_oil_well_rate_grid[i, j, k] += flow_rate
                    net_oil_well_mass_rate_grid[i, j, k] += flow_rate * phase_density

            production_rates[i, j, k] = (water_flow_rate, oil_flow_rate, gas_flow_rate)
            production_mass_rates[i, j, k] = (
                water_mass_flow_rate,
                oil_mass_flow_rate,
                gas_mass_flow_rate,
            )
            production_fvfs[i, j, k] = (water_phase_fvf, oil_phase_fvf, gas_phase_fvf)
            production_bhps[i, j, k] = (
                water_effective_bhp,
                oil_effective_bhp,
                gas_effective_bhp,
            )

    return WellRates(
        net_well_rate_grid=net_well_rate_grid,
        net_water_well_rate_grid=net_water_well_rate_grid,
        net_oil_well_rate_grid=net_oil_well_rate_grid,
        net_gas_well_rate_grid=net_gas_well_rate_grid,
        net_water_well_mass_rate_grid=net_water_well_mass_rate_grid,
        net_oil_well_mass_rate_grid=net_oil_well_mass_rate_grid,
        net_gas_well_mass_rate_grid=net_gas_well_mass_rate_grid,
        injection_rates=injection_rates,
        production_rates=production_rates,
        injection_mass_rates=injection_mass_rates,
        production_mass_rates=production_mass_rates,
        injection_fvfs=injection_fvfs,
        production_fvfs=production_fvfs,
        injection_bhps=injection_bhps,
        production_bhps=production_bhps,
    )

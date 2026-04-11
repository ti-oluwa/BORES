import logging
import typing

import numba
import numpy as np

from bores.errors import ValidationError
from bores.grids.base import (
    CapillaryPressureGrids,
    RelativeMobilityGrids,
    RelPermGrids,
    build_uniform_grid,
)
from bores.grids.pvt import (
    build_three_phase_capillary_pressure_grids,
    build_three_phase_relative_mobilities_grids,
    build_three_phase_relative_permeabilities_grids,
)
from bores.precision import get_dtype
from bores.tables.rock_fluid import CapillaryPressureTable, RelativePermeabilityTable
from bores.types import NDimensionalGrid, ThreeDimensions

logger = logging.getLogger(__name__)


def build_rock_fluid_properties_grids(
    water_saturation_grid: NDimensionalGrid[ThreeDimensions],
    oil_saturation_grid: NDimensionalGrid[ThreeDimensions],
    gas_saturation_grid: NDimensionalGrid[ThreeDimensions],
    irreducible_water_saturation_grid: NDimensionalGrid[ThreeDimensions],
    residual_oil_saturation_water_grid: NDimensionalGrid[ThreeDimensions],
    residual_oil_saturation_gas_grid: NDimensionalGrid[ThreeDimensions],
    residual_gas_saturation_grid: NDimensionalGrid[ThreeDimensions],
    water_viscosity_grid: NDimensionalGrid[ThreeDimensions],
    oil_viscosity_grid: NDimensionalGrid[ThreeDimensions],
    gas_viscosity_grid: NDimensionalGrid[ThreeDimensions],
    porosity_grid: NDimensionalGrid[ThreeDimensions],
    permeability_grid: NDimensionalGrid[ThreeDimensions],
    relative_permeability_table: RelativePermeabilityTable,
    capillary_pressure_table: typing.Optional[CapillaryPressureTable] = None,
    disable_capillary_effects: bool = False,
    capillary_strength_factor: float = 1.0,
    phase_appearance_tolerance: float = 1e-6,
) -> typing.Tuple[
    RelPermGrids[ThreeDimensions],
    RelativeMobilityGrids[ThreeDimensions],
    CapillaryPressureGrids[ThreeDimensions],
]:
    """
    Builds the rock-fluid properties grids required for simulation.

    :param water_saturation_grid: Water saturation grid.
    :param oil_saturation_grid: Oil saturation grid.
    :param gas_saturation_grid: Gas saturation grid.
    :param irreducible_water_saturation_grid: Irreducible water saturation grid.
    :param residual_oil_saturation_water_grid: Residual oil saturation in water flooding grid.
    :param residual_oil_saturation_gas_grid: Residual oil saturation in gas flooding grid.
    :param residual_gas_saturation_grid: Residual gas saturation grid.
    :param water_viscosity_grid: Water viscosity grid.
    :param oil_viscosity_grid: Oil viscosity grid.
    :param gas_viscosity_grid: Gas viscosity grid.
    :param porosity_grid: Reservoir porosity grid (fraction).
    :param permeability_grid: Reservoir mean permeability grid (mD).
    :param relative_permeability_table: Relative permeability table.
    :param capillary_pressure_table: Optional capillary pressure table. Required if capillary effects are enabled.
    :param disable_capillary_effects: If True, capillary effects are disabled (zero capillary pressures).
    :param capillary_strength_factor: Factor to scale capillary pressure grids.
    :param phase_appearance_tolerance: Tolerance for phase appearance/disappearance.
    :return: A tuple containing:
        - `RelPermGrids`: Relative permeability grids for oil, water, and gas.
        - `RelativeMobilityGrids`: Relative mobility grids for oil, water, and gas.
        - `CapillaryPressureGrids`: Capillary pressure grids for oil-water and gas-oil.
    """
    krw_grid, kro_grid, krg_grid = build_three_phase_relative_permeabilities_grids(
        water_saturation_grid=water_saturation_grid,
        oil_saturation_grid=oil_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        irreducible_water_saturation_grid=irreducible_water_saturation_grid,
        residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
        residual_gas_saturation_grid=residual_gas_saturation_grid,
        relative_permeability_table=relative_permeability_table,
        phase_appearance_tolerance=phase_appearance_tolerance,
    )
    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = build_three_phase_relative_mobilities_grids(
        oil_relative_permeability_grid=kro_grid,
        water_relative_permeability_grid=krw_grid,
        gas_relative_permeability_grid=krg_grid,
        water_viscosity_grid=water_viscosity_grid,
        oil_viscosity_grid=oil_viscosity_grid,
        gas_viscosity_grid=gas_viscosity_grid,
    )

    if disable_capillary_effects:
        logger.debug("Capillary effects disabled; using zero capillary pressure grids")
        oil_water_capillary_pressure_grid = build_uniform_grid(
            grid_shape=water_saturation_grid.shape, value=0.0
        )
        gas_oil_capillary_pressure_grid = oil_water_capillary_pressure_grid.copy()
    else:
        if capillary_pressure_table is None:
            raise ValidationError(
                "Capillary pressure table must be provided if capillary effects are enabled."
            )
        (
            oil_water_capillary_pressure_grid,
            gas_oil_capillary_pressure_grid,
        ) = build_three_phase_capillary_pressure_grids(
            water_saturation_grid=water_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            irreducible_water_saturation_grid=irreducible_water_saturation_grid,
            residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
            residual_gas_saturation_grid=residual_gas_saturation_grid,
            porosity_grid=porosity_grid,
            permeability_grid=permeability_grid,
            capillary_pressure_table=capillary_pressure_table,
        )
        if capillary_strength_factor != 1.0:
            logger.debug(
                f"Scaling capillary pressure grids by factor {capillary_strength_factor}"
            )
            dtype = get_dtype()
            oil_water_capillary_pressure_grid = typing.cast(
                NDimensionalGrid[ThreeDimensions],
                np.multiply(
                    oil_water_capillary_pressure_grid,
                    capillary_strength_factor,
                    dtype=dtype,
                ),
            )
            gas_oil_capillary_pressure_grid = typing.cast(
                NDimensionalGrid[ThreeDimensions],
                np.multiply(
                    gas_oil_capillary_pressure_grid,
                    capillary_strength_factor,
                    dtype=dtype,
                ),
            )

    relperm_grids = RelPermGrids(
        oil_relative_permeability=kro_grid,
        water_relative_permeability=krw_grid,
        gas_relative_permeability=krg_grid,
    )
    relative_mobility_grids = RelativeMobilityGrids(
        water_relative_mobility=water_relative_mobility_grid,
        oil_relative_mobility=oil_relative_mobility_grid,
        gas_relative_mobility=gas_relative_mobility_grid,
    )
    capillary_pressure_grids = CapillaryPressureGrids(
        oil_water_capillary_pressure=oil_water_capillary_pressure_grid,
        gas_oil_capillary_pressure=gas_oil_capillary_pressure_grid,
    )
    return (relperm_grids, relative_mobility_grids, capillary_pressure_grids)


@numba.njit(parallel=True, cache=True)
def build_effective_residual_saturation_grids(
    water_saturation_grid: NDimensionalGrid[ThreeDimensions],
    gas_saturation_grid: NDimensionalGrid[ThreeDimensions],
    max_water_saturation_grid: NDimensionalGrid[ThreeDimensions],
    max_gas_saturation_grid: NDimensionalGrid[ThreeDimensions],
    water_imbibition_flag_grid: np.ndarray[ThreeDimensions, np.dtype[np.bool]],
    gas_imbibition_flag_grid: np.ndarray[ThreeDimensions, np.dtype[np.bool]],
    residual_oil_saturation_water_grid: NDimensionalGrid[ThreeDimensions],
    residual_oil_saturation_gas_grid: NDimensionalGrid[ThreeDimensions],
    residual_gas_saturation_grid: NDimensionalGrid[ThreeDimensions],
    residual_oil_drainage_ratio_water_flood: float = 0.6,  # Sorw_drainage = 0.6 x Sorw_imbibition
    residual_oil_drainage_ratio_gas_flood: float = 0.6,  # Sorg_drainage = 0.6 x Sorg_imbibition
    residual_gas_drainage_ratio: float = 0.5,  # Sgr_drainage = 0.5 x Sgr_imbibition
    tolerance: float = 1e-6,
) -> typing.Tuple[
    NDimensionalGrid[ThreeDimensions],
    NDimensionalGrid[ThreeDimensions],
    NDimensionalGrid[ThreeDimensions],
    NDimensionalGrid[ThreeDimensions],
    NDimensionalGrid[ThreeDimensions],
    np.ndarray[ThreeDimensions, np.dtype[np.bool]],
    np.ndarray[ThreeDimensions, np.dtype[np.bool]],
]:
    """
    Compute effective residual saturations based on displacement regime.

    This function performs the following steps for each grid cell:
    1. Detects if we're in drainage or imbibition
    2. Applies appropriate residual values
    3. Updates historical maxima

    :param water_saturation_grid: Current water saturation grid.
    :param gas_saturation_grid: Current gas saturation grid.
    :param max_water_saturation_grid: Historical maximum water saturation grid.
    :param max_gas_saturation_grid: Historical maximum gas saturation grid.
    :param water_imbibition_flag_grid: Boolean grid indicating if water is in imbibition.
    :param gas_imbibition_flag_grid: Boolean grid indicating if gas is in imbibition.
    :param residual_oil_saturation_water_grid: Residual oil saturation during water flooding grid.
    :param residual_oil_saturation_gas_grid: Residual oil saturation during gas flooding grid.
    :param residual_gas_saturation_grid: Residual gas saturation grid.
    :param residual_oil_drainage_ratio_water_flood: Ratio to compute oil drainage residual from imbibition value.
    :param residual_gas_drainage_ratio: Ratio to compute gas drainage residual from imbibition value.
    :param residual_oil_drainage_ratio_gas_flood: Ratio to compute oil drainage residual from gas flooding imbibition value.
    :param tolerance: Tolerance to determine significant saturation changes.
    :return: A tuple containing:
        - Updated maximum water saturation grid.
        - Updated maximum gas saturation grid.
        - Effective residual oil saturation grid for water flooding.
        - Effective residual oil saturation grid for gas flooding.
        - Effective residual gas saturation grid.
        - Updated water imbibition flag grid.
        - Updated gas imbibition flag grid.
    """
    nx, ny, nz = water_saturation_grid.shape

    new_max_water_saturation_grid = max_water_saturation_grid.copy()
    new_max_gas_saturation_grid = max_gas_saturation_grid.copy()
    new_water_imbibition_flag_grid = water_imbibition_flag_grid.copy()
    new_gas_imbibition_flag_grid = gas_imbibition_flag_grid.copy()

    # Effective residuals (will be populated)
    effective_residual_oil_saturation_water_grid = np.zeros_like(water_saturation_grid)
    effective_residual_oil_saturation_gas_grid = np.zeros_like(water_saturation_grid)
    effective_residual_gas_saturation_grid = np.zeros_like(water_saturation_grid)

    # Parallel loop
    for i in numba.prange(nx):  # type: ignore
        for j in range(ny):
            for k in range(nz):
                Sw = water_saturation_grid[i, j, k]
                Sg = gas_saturation_grid[i, j, k]
                Sw_max = max_water_saturation_grid[i, j, k]
                Sg_max = max_gas_saturation_grid[i, j, k]

                # Get imbibition values from rock properties
                Sorw_imbibition = residual_oil_saturation_water_grid[i, j, k]
                Sorg_imbibition = residual_oil_saturation_gas_grid[i, j, k]
                Sgr_imbibition = residual_gas_saturation_grid[i, j, k]

                # Compute drainage values using ratios
                Sor_drainage = Sorw_imbibition * residual_oil_drainage_ratio_water_flood
                Sorg_drainage = Sorg_imbibition * residual_oil_drainage_ratio_gas_flood
                Sgr_drainage = Sgr_imbibition * residual_gas_drainage_ratio

                # WATER-OIL SYSTEM
                if Sw > (Sw_max + tolerance):
                    # If water saturation is increasing we have water imbibition
                    # hence, water is displacing oil and more oil is trapped
                    effective_residual_oil_saturation_water_grid[i, j, k] = (
                        Sorw_imbibition
                    )
                    new_water_imbibition_flag_grid[i, j, k] = True
                    new_max_water_saturation_grid[i, j, k] = Sw

                elif Sw < (Sw_max - tolerance):
                    # If water saturation is decreasing we have Oil drainage
                    # hence, oil is displacing water and less oil is trapped
                    effective_residual_oil_saturation_water_grid[i, j, k] = Sor_drainage
                    new_water_imbibition_flag_grid[i, j, k] = False
                    # Sw_max stays unchanged (only increases)

                else:
                    # If no significant change, use previous regime
                    if water_imbibition_flag_grid[i, j, k]:
                        effective_residual_oil_saturation_water_grid[i, j, k] = (
                            Sorw_imbibition
                        )
                    else:
                        effective_residual_oil_saturation_water_grid[i, j, k] = (
                            Sor_drainage
                        )

                # GAS-OIL SYSTEM
                if Sg > (Sg_max + tolerance):
                    # If gas saturation is increasing the we have gas imbibition
                    # hence, gas is displacing oil and more oil trapped
                    effective_residual_oil_saturation_gas_grid[i, j, k] = (
                        Sorg_imbibition
                    )
                    effective_residual_gas_saturation_grid[i, j, k] = Sgr_drainage
                    new_gas_imbibition_flag_grid[i, j, k] = True
                    new_max_gas_saturation_grid[i, j, k] = Sg

                elif Sg < (Sg_max - tolerance):
                    # If gas saturation descreasing then we have oil drainage
                    # hence, oil is displacing gas and less oil trapped
                    effective_residual_oil_saturation_gas_grid[i, j, k] = Sorg_drainage
                    effective_residual_gas_saturation_grid[i, j, k] = Sgr_imbibition
                    new_gas_imbibition_flag_grid[i, j, k] = False
                    # Sg_max stays unchanged (only increases)

                else:
                    # If there's no significant change, use previous regime
                    if gas_imbibition_flag_grid[i, j, k]:
                        effective_residual_oil_saturation_gas_grid[i, j, k] = (
                            Sorg_imbibition
                        )
                        effective_residual_gas_saturation_grid[i, j, k] = Sgr_drainage
                    else:
                        effective_residual_oil_saturation_gas_grid[i, j, k] = (
                            Sorg_drainage
                        )
                        effective_residual_gas_saturation_grid[i, j, k] = Sgr_imbibition

    return (
        new_max_water_saturation_grid,
        new_max_gas_saturation_grid,
        effective_residual_oil_saturation_water_grid,
        effective_residual_oil_saturation_gas_grid,
        effective_residual_gas_saturation_grid,
        new_water_imbibition_flag_grid,
        new_gas_imbibition_flag_grid,
    )

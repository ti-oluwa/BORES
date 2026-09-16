import itertools
import logging
import typing

import numba
import numpy as np

from bores.blackoil.satfunc import (
    CapillaryPressureTable,
    RelativePermeabilityTable,
)
from bores.errors import ValidationError
from bores.grids.base import (
    CapillaryPressureGrids,
    RelativeMobilityGrids,
    RelPermGrids,
    build_uniform_grid,
)
from bores.precision import get_dtype
from bores.reservoir import Hysteresis
from bores.types import NDimension, NDimensionalGrid, ThreeDimensions

logger = logging.getLogger(__name__)


def build_three_phase_capillary_pressure_grids(
    water_saturation_grid: NDimensionalGrid[NDimension],
    gas_saturation_grid: NDimensionalGrid[NDimension],
    irreducible_water_saturation_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_water_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_gas_grid: NDimensionalGrid[NDimension],
    residual_gas_saturation_grid: NDimensionalGrid[NDimension],
    porosity_grid: NDimensionalGrid[NDimension],
    permeability_grid: NDimensionalGrid[NDimension],
    capillary_pressure_table: CapillaryPressureTable,
    max_water_saturation: NDimensionalGrid[NDimension] | None = None,
    max_gas_saturation: NDimensionalGrid[NDimension] | None = None,
    water_imbibition_flag: np.ndarray[NDimension, np.dtype[np.bool_]] | None = None,
    gas_imbibition_flag: np.ndarray[NDimension, np.dtype[np.bool_]] | None = None,
    water_reversal_saturation: NDimensionalGrid[NDimension] | None = None,
    gas_reversal_saturation: NDimensionalGrid[NDimension] | None = None,
) -> tuple[NDimensionalGrid[NDimension], NDimensionalGrid[NDimension]]:
    """
    Computes the capillary pressure grids for water, oil, and gas three-phase system.

    This function calculates the capillary pressures based on Corey model with residual saturations.

    :param water_saturation_grid: N-Dimensional array of water saturation values (fraction).
    :param gas_saturation_grid: N-Dimensional array of gas saturation values (fraction).
    :param irreducible_water_saturation_grid: N-Dimensional array of irreducible water saturation values (fraction).
    :param residual_oil_saturation_water_grid: N-Dimensional array of residual oil saturation values during water flooding (fraction).
    :param residual_oil_saturation_gas_grid: N-Dimensional array of residual oil saturation values during gas flooding (fraction).
    :param residual_gas_saturation_grid: N-Dimensional array of residual gas saturation values (fraction).
    :param porosity_grid: N-Dimensional array of reservoir porosity (fraction).
    :param permeability_grid: N-Dimensional array of mean reservoir permeability (mD).
    :param capillary_pressure_table: `CapillaryPressureTable` object containing parameters for capillary pressure calculations.
    :param max_water_saturation: Optional historical maximum water saturation grid for hysteresis effects.
    :param max_gas_saturation: Optional historical maximum gas saturation grid for hysteresis effects.
    :param water_imbibition_flag: Optional boolean grid indicating water imbibition phase for hysteresis.
    :param gas_imbibition_flag: Optional boolean grid indicating gas imbibition phase for hysteresis.
    :param water_reversal_saturation: Optional water saturation at last reversal point for hysteresis.
    :param gas_reversal_saturation: Optional gas saturation at last reversal point for hysteresis.
    :return: Tuple of (oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid)
        where each grid is a N-Dimensional numpy array of capillary pressures (psi).
    """
    one = water_saturation_grid.dtype.type(1.0)
    oil_saturation_grid = one - water_saturation_grid - gas_saturation_grid
    # Check if table supports array/vectorized operations
    supports_vector = getattr(capillary_pressure_table, "supports_vector", False)

    if supports_vector:
        hysteresis_kwargs = {}
        if max_water_saturation is not None:
            hysteresis_kwargs["max_water_saturation"] = max_water_saturation
        if max_gas_saturation is not None:
            hysteresis_kwargs["max_gas_saturation"] = max_gas_saturation
        if water_imbibition_flag is not None:
            hysteresis_kwargs["water_imbibition_flag"] = water_imbibition_flag
        if gas_imbibition_flag is not None:
            hysteresis_kwargs["gas_imbibition_flag"] = gas_imbibition_flag
        if water_reversal_saturation is not None:
            hysteresis_kwargs["water_reversal_saturation"] = water_reversal_saturation
        if gas_reversal_saturation is not None:
            hysteresis_kwargs["gas_reversal_saturation"] = gas_reversal_saturation

        oil_saturation_grid = np.maximum(0.0, oil_saturation_grid)
        capillary_pressures = capillary_pressure_table(
            water_saturation=water_saturation_grid,
            oil_saturation=oil_saturation_grid,
            gas_saturation=gas_saturation_grid,
            irreducible_water_saturation=irreducible_water_saturation_grid,
            residual_oil_saturation_water=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas=residual_oil_saturation_gas_grid,
            residual_gas_saturation=residual_gas_saturation_grid,
            porosity=porosity_grid,
            permeability=permeability_grid,
            **hysteresis_kwargs,
        )
        oil_water_capillary_pressure_grid = capillary_pressures["oil_water"]  # type: ignore[assignment]
        gas_oil_capillary_pressure_grid = capillary_pressures["gas_oil"]  # type: ignore[assignment]
    else:
        # Cell-by-cell approach for tables that don't support arrays
        oil_water_capillary_pressure_grid = build_uniform_grid(
            grid_shape=water_saturation_grid.shape, value=0.0
        )
        gas_oil_capillary_pressure_grid = oil_water_capillary_pressure_grid.copy()

        for indices in itertools.product(*map(range, water_saturation_grid.shape)):
            # Get current saturations for the cell
            water_saturation = water_saturation_grid[indices]
            gas_saturation = gas_saturation_grid[indices]
            oil_saturation = oil_saturation_grid[indices]

            # Get cell-specific rock properties
            irreducible_water_saturation = irreducible_water_saturation_grid[indices]
            residual_oil_saturation_water = residual_oil_saturation_water_grid[indices]
            residual_oil_saturation_gas = residual_oil_saturation_gas_grid[indices]
            residual_gas_saturation = residual_gas_saturation_grid[indices]
            porosity = porosity_grid[indices]
            permeability = permeability_grid[indices]

            # Extract cell-specific hysteresis values if available
            cell_hysteresis_kwargs = {}
            if max_water_saturation is not None:
                cell_hysteresis_kwargs["max_water_saturation"] = max_water_saturation[indices]
            if max_gas_saturation is not None:
                cell_hysteresis_kwargs["max_gas_saturation"] = max_gas_saturation[indices]
            if water_imbibition_flag is not None:
                cell_hysteresis_kwargs["water_imbibition_flag"] = water_imbibition_flag[indices]
            if gas_imbibition_flag is not None:
                cell_hysteresis_kwargs["gas_imbibition_flag"] = gas_imbibition_flag[indices]
            if water_reversal_saturation is not None:
                cell_hysteresis_kwargs["water_reversal_saturation"] = water_reversal_saturation[
                    indices
                ]
            if gas_reversal_saturation is not None:
                cell_hysteresis_kwargs["gas_reversal_saturation"] = gas_reversal_saturation[
                    indices
                ]

            capillary_pressures = capillary_pressure_table(
                water_saturation=water_saturation,
                oil_saturation=max(0.0, oil_saturation),
                gas_saturation=gas_saturation,
                irreducible_water_saturation=irreducible_water_saturation,
                residual_oil_saturation_water=residual_oil_saturation_water,
                residual_oil_saturation_gas=residual_oil_saturation_gas,
                residual_gas_saturation=residual_gas_saturation,
                porosity=porosity,
                permeability=permeability,
                **cell_hysteresis_kwargs,
            )
            oil_water_capillary_pressure_grid[indices] = capillary_pressures["oil_water"]
            gas_oil_capillary_pressure_grid[indices] = capillary_pressures["gas_oil"]

    return oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid  # type: ignore[return-value]


def build_three_phase_relative_permeabilities_grids(
    water_saturation_grid: NDimensionalGrid[NDimension],
    oil_saturation_grid: NDimensionalGrid[NDimension],
    gas_saturation_grid: NDimensionalGrid[NDimension],
    irreducible_water_saturation_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_water_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_gas_grid: NDimensionalGrid[NDimension],
    residual_gas_saturation_grid: NDimensionalGrid[NDimension],
    relative_permeability_table: RelativePermeabilityTable,
    phase_appearance_tolerance: float = 1e-6,
    max_water_saturation: NDimensionalGrid[NDimension] | None = None,
    max_gas_saturation: NDimensionalGrid[NDimension] | None = None,
    water_imbibition_flag: np.ndarray[NDimension, np.dtype[np.bool_]] | None = None,
    gas_imbibition_flag: np.ndarray[NDimension, np.dtype[np.bool_]] | None = None,
    water_reversal_saturation: NDimensionalGrid[NDimension] | None = None,
    gas_reversal_saturation: NDimensionalGrid[NDimension] | None = None,
) -> tuple[
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
]:
    """
    Computes the relative permeability grids for water, oil, and gas phases for a three-phase system.

    This function calculates the relative permeabilities based on Corey model with residual saturations.

    :param water_saturation_grid: N-Dimensional array of water saturation values (fraction).
    :param oil_saturation_grid: N-Dimensional array of oil saturation values (fraction).
    :param gas_saturation_grid: N-Dimensional array of gas saturation values (fraction).
    :param irreducible_water_saturation_grid: N-Dimensional array of irreducible water saturation values (fraction).
    :param residual_oil_saturation_water_grid: N-Dimensional array of residual oil saturation values during water flooding (fraction).
    :param residual_oil_saturation_gas_grid: N-Dimensional array of residual oil saturation values during gas flooding (fraction).
    :param residual_gas_saturation_grid: N-Dimensional array of residual gas saturation values (fraction).
    :param relative_permeability_table: Relative permeability function to use for calculations (e.g., Corey model).
    :param phase_appearance_tolerance: Tolerance for phase appearance/disappearance based on residual/irreducible saturations.
    :param max_water_saturation: Optional historical maximum water saturation grid for hysteresis effects.
    :param max_gas_saturation: Optional historical maximum gas saturation grid for hysteresis effects.
    :param water_imbibition_flag: Optional boolean grid indicating water imbibition phase for hysteresis.
    :param gas_imbibition_flag: Optional boolean grid indicating gas imbibition phase for hysteresis.
    :param water_reversal_saturation: Optional water saturation at last reversal point for hysteresis.
    :param gas_reversal_saturation: Optional gas saturation at last reversal point for hysteresis.
    :return: Tuple of (water_relative_permeability_grid, oil_relative_permeability_grid, gas_relative_permeability_grid) as fractions.
    """
    # Check if table supports array operations
    supports_vector = getattr(relative_permeability_table, "supports_vector", False)
    if supports_vector:
        hysteresis_kwargs = {}
        if max_water_saturation is not None:
            hysteresis_kwargs["max_water_saturation"] = max_water_saturation
        if max_gas_saturation is not None:
            hysteresis_kwargs["max_gas_saturation"] = max_gas_saturation
        if water_imbibition_flag is not None:
            hysteresis_kwargs["water_imbibition_flag"] = water_imbibition_flag
        if gas_imbibition_flag is not None:
            hysteresis_kwargs["gas_imbibition_flag"] = gas_imbibition_flag
        if water_reversal_saturation is not None:
            hysteresis_kwargs["water_reversal_saturation"] = water_reversal_saturation
        if gas_reversal_saturation is not None:
            hysteresis_kwargs["gas_reversal_saturation"] = gas_reversal_saturation

        relative_permeabilities = relative_permeability_table(
            water_saturation=water_saturation_grid,
            oil_saturation=oil_saturation_grid,
            gas_saturation=gas_saturation_grid,
            irreducible_water_saturation=irreducible_water_saturation_grid,
            residual_oil_saturation_water=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas=residual_oil_saturation_gas_grid,
            residual_gas_saturation=residual_gas_saturation_grid,
            **hysteresis_kwargs,
        )

        # Use hysteresis-aware residual oil saturation
        effective_residual_oil_saturation_grid = np.where(
            water_saturation_grid > gas_saturation_grid,
            residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid,
        )

        # Mark phases as inactive by setting relative permeability to zero
        # if below 'residual/irreducible saturations + tolerance'
        water_inactive = (
            water_saturation_grid <= irreducible_water_saturation_grid + phase_appearance_tolerance
        )
        oil_inactive = (
            oil_saturation_grid
            <= effective_residual_oil_saturation_grid + phase_appearance_tolerance
        )
        gas_inactive = (
            gas_saturation_grid <= residual_gas_saturation_grid + phase_appearance_tolerance
        )

        water_relative_permeability_grid = np.where(
            water_inactive, 0.0, relative_permeabilities["water"]
        )
        oil_relative_permeability_grid = np.where(
            oil_inactive, 0.0, relative_permeabilities["oil"]
        )
        gas_relative_permeability_grid = np.where(
            gas_inactive, 0.0, relative_permeabilities["gas"]
        )
    else:
        # Cell-by-cell approach for tables that don't support arrays
        water_relative_permeability_grid = build_uniform_grid(
            grid_shape=water_saturation_grid.shape, value=0.0
        )
        oil_relative_permeability_grid = water_relative_permeability_grid.copy()
        gas_relative_permeability_grid = water_relative_permeability_grid.copy()
        nx, ny, nz = water_saturation_grid.shape

        for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
            # Get current saturations for the cell
            water_saturation = water_saturation_grid[i, j, k]
            oil_saturation = oil_saturation_grid[i, j, k]
            gas_saturation = gas_saturation_grid[i, j, k]

            # Get cell-specific rock properties
            irreducible_water_saturation = irreducible_water_saturation_grid[i, j, k]
            residual_oil_saturation_water = residual_oil_saturation_water_grid[i, j, k]
            residual_oil_saturation_gas = residual_oil_saturation_gas_grid[i, j, k]
            residual_gas_saturation = residual_gas_saturation_grid[i, j, k]

            # Extract cell-specific hysteresis values if available
            cell_hysteresis_kwargs = {}
            if max_water_saturation is not None:
                cell_hysteresis_kwargs["max_water_saturation"] = max_water_saturation[i, j, k]
            if max_gas_saturation is not None:
                cell_hysteresis_kwargs["max_gas_saturation"] = max_gas_saturation[i, j, k]
            if water_imbibition_flag is not None:
                cell_hysteresis_kwargs["water_imbibition_flag"] = water_imbibition_flag[i, j, k]
            if gas_imbibition_flag is not None:
                cell_hysteresis_kwargs["gas_imbibition_flag"] = gas_imbibition_flag[i, j, k]
            if water_reversal_saturation is not None:
                cell_hysteresis_kwargs["water_reversal_saturation"] = water_reversal_saturation[
                    i, j, k
                ]
            if gas_reversal_saturation is not None:
                cell_hysteresis_kwargs["gas_reversal_saturation"] = gas_reversal_saturation[
                    i, j, k
                ]

            # Compute three-phase relative permeabilities
            relative_permeabilities = relative_permeability_table(
                water_saturation=water_saturation,
                oil_saturation=oil_saturation,
                gas_saturation=gas_saturation,
                irreducible_water_saturation=irreducible_water_saturation,
                residual_oil_saturation_water=residual_oil_saturation_water,
                residual_oil_saturation_gas=residual_oil_saturation_gas,
                residual_gas_saturation=residual_gas_saturation,
                **cell_hysteresis_kwargs,
            )
            # Mark phases as inactive by setting relative permeability to the phase as zero
            # if below 'residual/irreducible saturations + tolerance'
            # Use hysteresis-aware residual oil saturation since not flux or mobility info is available here
            if water_saturation > gas_saturation:
                effective_residual_oil_saturation = residual_oil_saturation_water
            else:
                effective_residual_oil_saturation = residual_oil_saturation_gas

            water_inactive = (
                water_saturation <= irreducible_water_saturation + phase_appearance_tolerance
            )
            oil_inactive = (
                oil_saturation <= effective_residual_oil_saturation + phase_appearance_tolerance
            )
            gas_inactive = gas_saturation <= residual_gas_saturation + phase_appearance_tolerance
            water_relative_permeability_grid[i, j, k] = (
                relative_permeabilities["water"] if not water_inactive else 0.0
            )
            oil_relative_permeability_grid[i, j, k] = (
                relative_permeabilities["oil"] if not oil_inactive else 0.0
            )
            gas_relative_permeability_grid[i, j, k] = (
                relative_permeabilities["gas"] if not gas_inactive else 0.0
            )

    return (  # type: ignore[return-value]
        water_relative_permeability_grid,
        oil_relative_permeability_grid,
        gas_relative_permeability_grid,
    )


@numba.njit(cache=True, parallel=True)
def build_three_phase_relative_mobilities_grids(
    water_relative_permeability_grid: NDimensionalGrid[NDimension],
    oil_relative_permeability_grid: NDimensionalGrid[NDimension],
    gas_relative_permeability_grid: NDimensionalGrid[NDimension],
    water_viscosity_grid: NDimensionalGrid[NDimension],
    oil_viscosity_grid: NDimensionalGrid[NDimension],
    gas_viscosity_grid: NDimensionalGrid[NDimension],
) -> tuple[
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
]:
    """
    Computes the relative mobility grids for water, oil, and gas phases for a three-phase system.

    The relative mobility is defined as the ratio of relative permeability to viscosity for each phase.
    :param water_relative_permeability_grid: N-Dimensional array of water relative permeability values (fraction).
    :param oil_relative_permeability_grid: N-Dimensional array of oil relative permeability values (fraction).
    :param gas_relative_permeability_grid: N-Dimensional array of gas relative permeability values (fraction).
    :param water_viscosity_grid: N-Dimensional array of water viscosity values (cP).
    :param oil_viscosity_grid: N-Dimensional array of oil viscosity values (cP).
    :param gas_viscosity_grid: N-Dimensional array of gas viscosity values (cP).
    :return: Tuple of (water_relative_mobility_grid, oil_relative_mobility_grid, gas_relative_mobility_grid) as fractions.
    """
    # Compute mobilities as kr / viscosity, handling division by zero and invalid values
    water_relative_mobility_grid = np.where(
        (water_viscosity_grid > 0.0)
        & np.isfinite(water_viscosity_grid)
        & np.isfinite(water_relative_permeability_grid),
        water_relative_permeability_grid / water_viscosity_grid,
        0.0,
    )
    oil_relative_mobility_grid = np.where(
        (oil_viscosity_grid > 0.0)
        & np.isfinite(oil_viscosity_grid)
        & np.isfinite(oil_relative_permeability_grid),
        oil_relative_permeability_grid / oil_viscosity_grid,
        0.0,
    )
    gas_relative_mobility_grid = np.where(
        (gas_viscosity_grid > 0.0)
        & np.isfinite(gas_viscosity_grid)
        & np.isfinite(gas_relative_permeability_grid),
        gas_relative_permeability_grid / gas_viscosity_grid,
        0.0,
    )

    # Ensure resulting mobilities are also finite (handle any NaN/Inf from division)
    water_relative_mobility_grid = np.where(
        np.isfinite(water_relative_mobility_grid), water_relative_mobility_grid, 0.0
    )
    oil_relative_mobility_grid = np.where(
        np.isfinite(oil_relative_mobility_grid), oil_relative_mobility_grid, 0.0
    )
    gas_relative_mobility_grid = np.where(
        np.isfinite(gas_relative_mobility_grid), gas_relative_mobility_grid, 0.0
    )
    return (  # type: ignore[return-value]
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    )


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
    capillary_pressure_table: CapillaryPressureTable | None = None,
    hysteresis_state: Hysteresis[ThreeDimensions] | None = None,
    disable_capillary_effects: bool = False,
    capillary_strength_factor: float = 1.0,
    phase_appearance_tolerance: float = 1e-6,
) -> tuple[
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
    :param hysteresis_state: Optional `Hysteresis` instance for tracking drainage-imbibition effects. If provided, its historical saturation extrema and displacement flags are used to compute hysteresis-dependent properties.
    :param disable_capillary_effects: If True, capillary effects are disabled (zero capillary pressures).
    :param capillary_strength_factor: Factor to scale capillary pressure grids.
    :param phase_appearance_tolerance: Tolerance for phase appearance/disappearance.
    :return: A tuple containing:
        - `RelPermGrids`: Relative permeability grids for oil, water, and gas.
        - `RelativeMobilityGrids`: Relative mobility grids for oil, water, and gas.
        - `CapillaryPressureGrids`: Capillary pressure grids for oil-water and gas-oil.
    """
    hysteresis_kwargs = {}
    if hysteresis_state is not None:
        hysteresis_kwargs = dict(
            max_water_saturation=hysteresis_state.max_water_saturation_grid,
            max_gas_saturation=hysteresis_state.max_gas_saturation_grid,
            water_imbibition_flag=hysteresis_state.water_imbibition_flag_grid,
            gas_imbibition_flag=hysteresis_state.gas_imbibition_flag_grid,
            water_reversal_saturation=hysteresis_state.water_reversal_saturation_grid,
            gas_reversal_saturation=hysteresis_state.gas_reversal_saturation_grid,
        )

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
        **hysteresis_kwargs,
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
            **hysteresis_kwargs,
        )
        if capillary_strength_factor != 1.0:
            logger.debug(f"Scaling capillary pressure grids by factor {capillary_strength_factor}")
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
    old_water_saturation_grid: NDimensionalGrid[ThreeDimensions],
    old_gas_saturation_grid: NDimensionalGrid[ThreeDimensions],
    max_water_saturation_grid: NDimensionalGrid[ThreeDimensions],
    max_gas_saturation_grid: NDimensionalGrid[ThreeDimensions],
    water_imbibition_flag_grid: np.ndarray[ThreeDimensions, np.dtype[np.bool]],
    gas_imbibition_flag_grid: np.ndarray[ThreeDimensions, np.dtype[np.bool]],
    water_reversal_saturation_grid: NDimensionalGrid[ThreeDimensions],
    gas_reversal_saturation_grid: NDimensionalGrid[ThreeDimensions],
    residual_oil_saturation_water_grid: NDimensionalGrid[ThreeDimensions],
    residual_oil_saturation_gas_grid: NDimensionalGrid[ThreeDimensions],
    residual_gas_saturation_grid: NDimensionalGrid[ThreeDimensions],
    residual_oil_drainage_ratio_water_flood: float = 0.6,  # sorw_drainage = 0.6 x sorw_imbibition
    residual_oil_drainage_ratio_gas_flood: float = 0.6,  # sorg_drainage = 0.6 x sorg_imbibition
    residual_gas_drainage_ratio: float = 0.5,  # gas_saturationr_drainage = 0.5 x gas_saturationr_imbibition
    tolerance: float = 1e-6,
) -> tuple[
    NDimensionalGrid[ThreeDimensions],
    NDimensionalGrid[ThreeDimensions],
    NDimensionalGrid[ThreeDimensions],
    NDimensionalGrid[ThreeDimensions],
    NDimensionalGrid[ThreeDimensions],
    np.ndarray[ThreeDimensions, np.dtype[np.bool]],
    np.ndarray[ThreeDimensions, np.dtype[np.bool]],
    NDimensionalGrid[ThreeDimensions],
    NDimensionalGrid[ThreeDimensions],
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
    :param water_imbibition_flag_grid: Boolean grid indicating if water is in imbibition (True) or drainage (False).
    :param gas_imbibition_flag_grid: Boolean grid indicating if gas is in imbibition (True) or drainage (False).
    :param water_reversal_saturation_grid: Water saturation at the last reversal point for Killough scanning curves.
    :param gas_reversal_saturation_grid: Gas saturation at the last reversal point for Killough scanning curves.
    :param residual_oil_saturation_water_grid: Residual oil saturation during water flooding (imbibition) grid.
    :param residual_oil_saturation_gas_grid: Residual oil saturation during gas flooding (imbibition) grid.
    :param residual_gas_saturation_grid: Residual gas saturation during gas imbibition grid.
    :param residual_oil_drainage_ratio_water_flood: Ratio to compute effective residual oil saturation in water drainage from imbibition value (default: 0.6, meaning Sor_drainage = 0.6 × Sor_imbibition).
    :param residual_oil_drainage_ratio_gas_flood: Ratio to compute effective residual oil saturation in gas drainage from imbibition value (default: 0.6, meaning sorg_drainage = 0.6 × sorg_imbibition).
    :param residual_gas_drainage_ratio: Ratio to compute effective residual gas saturation in drainage from imbibition value (default: 0.5, meaning gas_saturationr_drainage = 0.5 × gas_saturationr_imbibition).
    :param tolerance: Saturation change tolerance to distinguish between drainage and imbibition regimes (fraction).
    :return: A tuple containing:
        - Updated maximum water saturation grid.
        - Updated maximum gas saturation grid.
        - Effective residual oil saturation grid for water flooding.
        - Effective residual oil saturation grid for gas flooding.
        - Effective residual gas saturation grid.
        - Updated water imbibition flag grid.
        - Updated gas imbibition flag grid.
        - Updated water reversal saturation grid.
        - Updated gas reversal saturation grid.
    """
    nx, ny, nz = water_saturation_grid.shape

    new_max_water_saturation_grid = max_water_saturation_grid.copy()
    new_max_gas_saturation_grid = max_gas_saturation_grid.copy()
    new_water_imbibition_flag_grid = water_imbibition_flag_grid.copy()
    new_gas_imbibition_flag_grid = gas_imbibition_flag_grid.copy()
    new_water_reversal_saturation_grid = water_reversal_saturation_grid.copy()
    new_gas_reversal_saturation_grid = gas_reversal_saturation_grid.copy()

    effective_residual_oil_saturation_water_grid = np.zeros_like(water_saturation_grid)
    effective_residual_oil_saturation_gas_grid = np.zeros_like(water_saturation_grid)
    effective_residual_gas_saturation_grid = np.zeros_like(water_saturation_grid)

    for i in numba.prange(nx):  # type: ignore
        for j in range(ny):
            for k in range(nz):
                water_saturation = water_saturation_grid[i, j, k]
                gas_saturation = gas_saturation_grid[i, j, k]
                max_water_saturation = max_water_saturation_grid[i, j, k]
                max_gas_saturation = max_gas_saturation_grid[i, j, k]

                # Get imbibition values from rock properties
                sorw_imbibition = residual_oil_saturation_water_grid[i, j, k]
                sorg_imbibition = residual_oil_saturation_gas_grid[i, j, k]
                gas_saturationr_imbibition = residual_gas_saturation_grid[i, j, k]

                # Compute drainage values using ratios
                Sor_drainage = sorw_imbibition * residual_oil_drainage_ratio_water_flood
                sorg_drainage = sorg_imbibition * residual_oil_drainage_ratio_gas_flood
                gas_saturationr_drainage = gas_saturationr_imbibition * residual_gas_drainage_ratio

                # WATER-OIL SYSTEM
                if water_saturation > (max_water_saturation + tolerance):
                    # If water saturation is increasing we have water imbibition
                    # hence, water is displacing oil and more oil is trapped
                    effective_residual_oil_saturation_water_grid[i, j, k] = sorw_imbibition
                    new_max_water_saturation_grid[i, j, k] = water_saturation

                    # If were just switching to imbibition, record reversal (if coming from drainage)
                    if not water_imbibition_flag_grid[i, j, k]:
                        new_water_reversal_saturation_grid[i, j, k] = old_water_saturation_grid[
                            i, j, k
                        ]

                    new_water_imbibition_flag_grid[i, j, k] = True

                elif water_saturation < (max_water_saturation - tolerance):
                    # If water saturation is decreasing we have Oil drainage
                    # hence, oil is displacing water and less oil is trapped
                    effective_residual_oil_saturation_water_grid[i, j, k] = Sor_drainage

                    # If were just switching to drainage, record reversal (if coming from imbibition)
                    if water_imbibition_flag_grid[i, j, k]:
                        new_water_reversal_saturation_grid[i, j, k] = old_water_saturation_grid[
                            i, j, k
                        ]

                    new_water_imbibition_flag_grid[i, j, k] = False
                    # max_water_saturation stays unchanged (only increases)

                else:
                    # If no significant change, use previous regime
                    if water_imbibition_flag_grid[i, j, k]:
                        effective_residual_oil_saturation_water_grid[i, j, k] = sorw_imbibition
                    else:
                        effective_residual_oil_saturation_water_grid[i, j, k] = Sor_drainage

                # GAS-OIL SYSTEM
                if gas_saturation > (max_gas_saturation + tolerance):
                    # If gas saturation is increasing the we have gas imbibition
                    # hence, gas is displacing oil and more oil trapped
                    effective_residual_oil_saturation_gas_grid[i, j, k] = sorg_imbibition
                    effective_residual_gas_saturation_grid[i, j, k] = gas_saturationr_drainage
                    new_max_gas_saturation_grid[i, j, k] = gas_saturation
                    # If were just switching to imbibition, record reversal (if coming from drainage)
                    if not gas_imbibition_flag_grid[i, j, k]:
                        new_gas_reversal_saturation_grid[i, j, k] = old_gas_saturation_grid[
                            i, j, k
                        ]

                    new_gas_imbibition_flag_grid[i, j, k] = True

                elif gas_saturation < (max_gas_saturation - tolerance):
                    # If gas saturation descreasing then we have oil drainage
                    # hence, oil is displacing gas and less oil trapped
                    effective_residual_oil_saturation_gas_grid[i, j, k] = sorg_drainage
                    effective_residual_gas_saturation_grid[i, j, k] = gas_saturationr_imbibition
                    # max_gas_saturation stays unchanged (only increases)
                    # If were just switching to drainage, record reversal (if coming from imbibition)
                    if gas_imbibition_flag_grid[i, j, k]:
                        new_gas_reversal_saturation_grid[i, j, k] = old_gas_saturation_grid[
                            i, j, k
                        ]

                    new_gas_imbibition_flag_grid[i, j, k] = False

                else:
                    # If there's no significant change, use previous regime
                    if gas_imbibition_flag_grid[i, j, k]:
                        effective_residual_oil_saturation_gas_grid[i, j, k] = sorg_imbibition
                        effective_residual_gas_saturation_grid[i, j, k] = (
                            gas_saturationr_imbibition
                        )
                    else:
                        effective_residual_oil_saturation_gas_grid[i, j, k] = sorg_drainage
                        effective_residual_gas_saturation_grid[i, j, k] = gas_saturationr_drainage

    return (
        new_max_water_saturation_grid,
        new_max_gas_saturation_grid,
        effective_residual_oil_saturation_water_grid,
        effective_residual_oil_saturation_gas_grid,
        effective_residual_gas_saturation_grid,
        new_water_imbibition_flag_grid,
        new_gas_imbibition_flag_grid,
        new_water_reversal_saturation_grid,
        new_gas_reversal_saturation_grid,
    )

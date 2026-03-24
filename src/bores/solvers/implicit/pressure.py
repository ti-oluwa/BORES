import functools
import itertools
import logging
import typing

import attrs
import numba
import numpy as np
from scipy.sparse import coo_matrix

from bores._precision import get_dtype
from bores.boundary_conditions import BoundaryConditions
from bores.config import Config
from bores.constants import c
from bores.correlations.core import compute_harmonic_mean
from bores.errors import PreconditionerError, SolverError
from bores.grids.base import CapillaryPressureGrids, RelativeMobilityGrids
from bores.grids.pvt import build_total_fluid_compressibility_grid
from bores.models import FluidProperties, RockProperties
from bores.solvers.base import (
    EvolutionResult,
    _warn_injection_pressure,
    _warn_injection_rate,
    _warn_production_pressure,
    compute_mobility_grids,
    solve_linear_system,
    to_1D_index_interior_only,
)
from bores.types import (
    FluidPhase,
    OneDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensions,
)
from bores.wells.base import Wells
from bores.wells.controls import CoupledRateControl
from bores.wells.core import (
    compute_average_compressibility_factor,
    compute_gas_productivity_index,
    compute_oil_productivity_index,
    get_pseudo_pressure_table,
)
from bores.wells.indices import WellIndicesCache

logger = logging.getLogger(__name__)


@attrs.frozen
class ImplicitPressureSolution:
    pressure_grid: ThreeDimensionalGrid
    max_pressure_change: float


def evolve_pressure(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step: int,
    time_step_size: float,
    time: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    config: Config,
    well_indices_cache: WellIndicesCache,
    pad_width: int = 1,
) -> EvolutionResult[ImplicitPressureSolution, None]:
    """
    Solves the fully implicit finite-difference pressure equation for a slightly compressible,
    three-phase flow system in a 3D reservoir.

    :param cell_dimension: Tuple of (cell_size_x, cell_size_y) in feet
    :param thickness_grid: 3D grid of cell thicknesses in feet
    :param elevation_grid: 3D grid of cell elevations in feet
    :param time_step: Current time step number (for logging/debugging)
    :param time_step_size: Time step size in seconds.
    :param time: Total simulation time elapsed. This time step inclusive.
    :param rock_properties: `RockProperties` object containing model rock properties
    :param fluid_properties: `FluidProperties` object containing model fluid properties
    :param relative_mobility_grids: Tuple of relative mobility grids for (water, oil, gas)
    :param capillary_pressure_grids: Tuple of capillary pressure grids for (oil-water, gas-oil)
    :param wells: `Wells` object containing well definitions and properties
    :param config: `Config` object containing simulation config
    :param boundary_conditions: Model boundary conditions.
    :param pad_width: Number of ghost cells used for grid padding. Well coordinates are offset by this amount.
    :return: `EvolutionResult` containing the new pressure grid and scheme used
    """
    absolute_permeability = rock_properties.absolute_permeability
    porosity_grid = rock_properties.porosity_grid
    rock_compressibility = rock_properties.compressibility
    oil_density_grid = fluid_properties.oil_effective_density_grid
    water_density_grid = fluid_properties.water_density_grid
    gas_density_grid = fluid_properties.gas_density_grid

    current_oil_pressure_grid = fluid_properties.pressure_grid
    current_water_saturation_grid = fluid_properties.water_saturation_grid
    current_oil_saturation_grid = fluid_properties.oil_saturation_grid
    current_gas_saturation_grid = fluid_properties.gas_saturation_grid

    water_compressibility_grid = fluid_properties.water_compressibility_grid
    oil_compressibility_grid = fluid_properties.oil_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid

    # Determine grid dimensions and cell sizes
    cell_count_x, cell_count_y, cell_count_z = current_oil_pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension
    dtype = get_dtype()

    # Compute total fluid system compressibility for each cell
    total_fluid_compressibility_grid = build_total_fluid_compressibility_grid(
        oil_saturation_grid=current_oil_saturation_grid,
        oil_compressibility_grid=oil_compressibility_grid,
        water_saturation_grid=current_water_saturation_grid,
        water_compressibility_grid=water_compressibility_grid,
        gas_saturation_grid=current_gas_saturation_grid,
        gas_compressibility_grid=gas_compressibility_grid,
    )
    total_compressibility_grid = np.add(
        total_fluid_compressibility_grid, rock_compressibility, dtype=dtype
    )
    # Clamp the compressibility within range
    total_compressibility_grid = config.total_compressibility_range.clip(
        total_compressibility_grid
    )

    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = relative_mobility_grids
    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = (
        capillary_pressure_grids
    )

    md_per_cp_to_ft2_per_psi_per_day = (
        c.MILLIDARCIES_PER_CENTIPOISE_TO_SQUARE_FEET_PER_PSI_PER_DAY
    )
    # Compute mobility grids for x, y, z directions
    x_mobilities, y_mobilities, z_mobilities = compute_mobility_grids(
        absolute_permeability_x=absolute_permeability.x,
        absolute_permeability_y=absolute_permeability.y,
        absolute_permeability_z=absolute_permeability.z,
        water_relative_mobility_grid=water_relative_mobility_grid,
        oil_relative_mobility_grid=oil_relative_mobility_grid,
        gas_relative_mobility_grid=gas_relative_mobility_grid,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
    )
    water_mobility_grid_x, oil_mobility_grid_x, gas_mobility_grid_x = x_mobilities
    water_mobility_grid_y, oil_mobility_grid_y, gas_mobility_grid_y = y_mobilities
    water_mobility_grid_z, oil_mobility_grid_z, gas_mobility_grid_z = z_mobilities

    time_step_size_in_days = time_step_size * c.DAYS_PER_SECOND
    interior_cell_count = (cell_count_x - 2) * (cell_count_y - 2) * (cell_count_z - 2)

    # Compute face transmissibilities (off-diagonal entries and additional diagonal/RHS contributions)
    # Compute gravitational constant conversion factor (ft/s² * lbf·s²/(lbm·ft) = lbf/lbm)
    # On Earth, this should normally be 1.0 in consistent units, but we include it for clarity
    # and say the acceleration due to gravity was changed to 12.0 ft/s² for some reason (say g on Mars)
    # then the conversion factor would be 12.0 / 32.174 = 0.373. Which would scale the gravity terms accordingly.
    gravitational_constant = (
        c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE
        / c.GRAVITATIONAL_CONSTANT_LBM_FT_PER_LBF_S2
    )
    if (pool := config.task_pool) is not None:
        accumulation_future = pool.submit(
            compute_accumulation_arrays,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            thickness_grid=thickness_grid,
            porosity_grid=porosity_grid,
            total_compressibility_grid=total_compressibility_grid,  # type:ignore
            current_oil_pressure_grid=current_oil_pressure_grid,
            cell_size_x=cell_size_x,
            cell_size_y=cell_size_y,
            time_step_size_in_days=time_step_size_in_days,
            dtype=dtype,
        )
        face_transmissibility_future = pool.submit(
            compute_face_transmissibility_arrays,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            thickness_grid=thickness_grid,
            cell_size_x=cell_size_x,
            cell_size_y=cell_size_y,
            water_mobility_grid_x=water_mobility_grid_x,
            oil_mobility_grid_x=oil_mobility_grid_x,
            gas_mobility_grid_x=gas_mobility_grid_x,
            water_mobility_grid_y=water_mobility_grid_y,
            oil_mobility_grid_y=oil_mobility_grid_y,
            gas_mobility_grid_y=gas_mobility_grid_y,
            water_mobility_grid_z=water_mobility_grid_z,
            oil_mobility_grid_z=oil_mobility_grid_z,
            gas_mobility_grid_z=gas_mobility_grid_z,
            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
            oil_density_grid=oil_density_grid,
            water_density_grid=water_density_grid,
            gas_density_grid=gas_density_grid,
            elevation_grid=elevation_grid,
            gravitational_constant=gravitational_constant,
            dtype=dtype,
        )
        wells_future = pool.submit(
            compute_well_contributions,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            current_oil_pressure_grid=current_oil_pressure_grid,
            temperature_grid=fluid_properties.temperature_grid,
            water_relative_mobility_grid=water_relative_mobility_grid,
            oil_relative_mobility_grid=oil_relative_mobility_grid,
            gas_relative_mobility_grid=gas_relative_mobility_grid,
            water_compressibility_grid=water_compressibility_grid,
            oil_compressibility_grid=oil_compressibility_grid,
            gas_compressibility_grid=gas_compressibility_grid,
            fluid_properties=fluid_properties,
            wells=wells,
            time=time,
            config=config,
            well_indices_cache=well_indices_cache,
            md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
            dtype=dtype,
            pad_width=pad_width,
        )

        # Collect all three results
        diagonal_values, rhs_values = accumulation_future.result()
        (
            sparse_rows,
            sparse_cols,
            sparse_off_diag,
            diagonal_additions,
            rhs_additions,
        ) = face_transmissibility_future.result()
        well_contributions = wells_future.result()

    else:
        diagonal_values, rhs_values = compute_accumulation_arrays(
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            thickness_grid=thickness_grid,
            porosity_grid=porosity_grid,
            total_compressibility_grid=total_compressibility_grid,  # type:ignore
            current_oil_pressure_grid=current_oil_pressure_grid,
            cell_size_x=cell_size_x,
            cell_size_y=cell_size_y,
            time_step_size_in_days=time_step_size_in_days,
            dtype=dtype,
        )
        (
            sparse_rows,
            sparse_cols,
            sparse_off_diag,
            diagonal_additions,
            rhs_additions,
        ) = compute_face_transmissibility_arrays(
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            thickness_grid=thickness_grid,
            cell_size_x=cell_size_x,
            cell_size_y=cell_size_y,
            water_mobility_grid_x=water_mobility_grid_x,
            oil_mobility_grid_x=oil_mobility_grid_x,
            gas_mobility_grid_x=gas_mobility_grid_x,
            water_mobility_grid_y=water_mobility_grid_y,
            oil_mobility_grid_y=oil_mobility_grid_y,
            gas_mobility_grid_y=gas_mobility_grid_y,
            water_mobility_grid_z=water_mobility_grid_z,
            oil_mobility_grid_z=oil_mobility_grid_z,
            gas_mobility_grid_z=gas_mobility_grid_z,
            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
            oil_density_grid=oil_density_grid,
            water_density_grid=water_density_grid,
            gas_density_grid=gas_density_grid,
            elevation_grid=elevation_grid,
            gravitational_constant=gravitational_constant,
            dtype=dtype,
        )
        well_contributions = compute_well_contributions(
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            current_oil_pressure_grid=current_oil_pressure_grid,
            temperature_grid=fluid_properties.temperature_grid,
            water_relative_mobility_grid=water_relative_mobility_grid,
            oil_relative_mobility_grid=oil_relative_mobility_grid,
            gas_relative_mobility_grid=gas_relative_mobility_grid,
            water_compressibility_grid=water_compressibility_grid,
            oil_compressibility_grid=oil_compressibility_grid,
            gas_compressibility_grid=gas_compressibility_grid,
            fluid_properties=fluid_properties,
            wells=wells,
            time=time,
            config=config,
            well_indices_cache=well_indices_cache,
            md_per_cp_to_ft2_per_psi_per_day=c.MILLIDARCIES_PER_CENTIPOISE_TO_SQUARE_FEET_PER_PSI_PER_DAY,
            dtype=dtype,
            pad_width=pad_width,
        )

    # Merge into final diagonal and b.
    final_diagonal = diagonal_values + diagonal_additions
    np.add.at(
        final_diagonal,
        well_contributions.diagonal_cell_indices,
        well_contributions.diagonal_values,
    )

    b = rhs_values + rhs_additions
    np.add.at(
        b,
        well_contributions.rhs_cell_indices,
        well_contributions.rhs_values,
    )

    interior_cell_count = (cell_count_x - 2) * (cell_count_y - 2) * (cell_count_z - 2)
    diag_indices = np.arange(interior_cell_count, dtype=np.int32)
    A = coo_matrix(  # type: ignore
        (
            np.concatenate([final_diagonal, sparse_off_diag]),
            (
                np.concatenate([diag_indices, sparse_rows]),
                np.concatenate([diag_indices, sparse_cols]),
            ),
        ),
        shape=(interior_cell_count, interior_cell_count),
        dtype=dtype,  # type: ignore
    )

    # Solve the linear system A·pⁿ⁺¹ = b
    try:
        new_1D_pressure_grid, _ = solve_linear_system(
            A_csr=A.tocsr(),
            b=b,
            rtol=config.pressure_convergence_tolerance,
            max_iterations=config.max_iterations,
            solver=config.pressure_solver,
            preconditioner=config.pressure_preconditioner,
            fallback_to_direct=True,
        )
    except (SolverError, PreconditionerError) as exc:
        logger.error(f"Pressure solve failed at time step {time_step}: {exc}")
        return EvolutionResult(
            value=ImplicitPressureSolution(
                pressure_grid=current_oil_pressure_grid.astype(dtype, copy=False),
                max_pressure_change=0.0,  # No change since solve failed
            ),
            success=False,
            scheme="implicit",
            message=str(exc),
        )

    # Map solution back to 3D grid
    new_pressure_grid = map_1D_solution_to_grid(
        solution_1D=new_1D_pressure_grid,  # type: ignore
        current_grid=current_oil_pressure_grid,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
    )
    max_pressure_change = np.max(np.abs(new_pressure_grid - current_oil_pressure_grid))
    return EvolutionResult(
        value=ImplicitPressureSolution(
            pressure_grid=new_pressure_grid.astype(dtype, copy=False),
            max_pressure_change=max_pressure_change,
        ),
        success=True,
        scheme="implicit",
        message=f"Implicit pressure evolution for time step {time_step} successful.",
    )


@numba.njit(parallel=True, cache=True)
def map_1D_solution_to_grid(
    solution_1D: OneDimensionalGrid,
    current_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
) -> ThreeDimensionalGrid:
    """
    Map the 1D solution array back to a 3D grid, preserving boundary values.

    This function takes the solution from the linear system solver (which only contains
    interior cell values) and fills them into a 3D grid while preserving the boundary
    values from the current grid.

    :param solution_1D: 1D array containing solution for interior cells only
    :param current_grid: Current 3D grid (used to preserve boundary values)
    :param cell_count_x: Number of cells in x-direction (including boundaries)
    :param cell_count_y: Number of cells in y-direction (including boundaries)
    :param cell_count_z: Number of cells in z-direction (including boundaries)
    :return: 3D grid with solution mapped to interior cells and boundaries preserved
    """
    # Initialize with current grid (preserves boundary values)
    new_grid = current_grid.copy()

    # Fill interior cells with solution using parallel processing
    for i in numba.prange(1, cell_count_x - 1):  # type: ignore
        for j in range(1, cell_count_y - 1):
            for k in range(1, cell_count_z - 1):
                # Convert 3D indices to 1D index for solution array
                idx = to_1D_index_interior_only(
                    i=i,
                    j=j,
                    k=k,
                    cell_count_x=cell_count_x,
                    cell_count_y=cell_count_y,
                    cell_count_z=cell_count_z,
                )
                new_grid[i, j, k] = solution_1D[idx]
    return new_grid


@numba.njit(parallel=True, cache=True)
def compute_accumulation_arrays(
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    total_compressibility_grid: ThreeDimensionalGrid,
    current_oil_pressure_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    time_step_size_in_days: float,
    dtype: np.typing.DTypeLike,
) -> typing.Tuple[np.typing.NDArray, np.typing.NDArray]:
    """
    Compute accumulation terms for all interior cells and return as dense arrays.

    This function computes the diagonal coefficients and RHS values that will be used
    to initialize the sparse matrix A and vector b. The accumulation term represents
    the storage capacity of each cell: accumulation_coefficient = (φ * c_t * V) / Δt

    :param cell_count_x: Number of cells in x-direction (including boundaries)
    :param cell_count_y: Number of cells in y-direction (including boundaries)
    :param cell_count_z: Number of cells in z-direction (including boundaries)
    :param thickness_grid: Cell thickness grid (ft)
    :param porosity_grid: Cell porosity grid (fraction)
    :param total_compressibility_grid: Total compressibility grid (1/psi)
    :param current_oil_pressure_grid: Current oil pressure grid (psi)
    :param cell_size_x: Cell size in x-direction (ft)
    :param cell_size_y: Cell size in y-direction (ft)
    :param time_step_size_in_days: Time step size (days)
    :param dtype: Data type for arrays (np.float32 or np.float64)
    :return: Tuple of (diagonal_values, rhs_values) both 1D arrays of length `interior_cell_count`
    """
    interior_cell_count = (cell_count_x - 2) * (cell_count_y - 2) * (cell_count_z - 2)
    diagonal_values = np.zeros(interior_cell_count, dtype=dtype)
    rhs_values = np.zeros(interior_cell_count, dtype=dtype)

    for i in numba.prange(1, cell_count_x - 1):  # type: ignore
        for j in range(1, cell_count_y - 1):
            for k in range(1, cell_count_z - 1):
                cell_1D_index = to_1D_index_interior_only(
                    i=i,
                    j=j,
                    k=k,
                    cell_count_x=cell_count_x,
                    cell_count_y=cell_count_y,
                    cell_count_z=cell_count_z,
                )

                cell_thickness = thickness_grid[i, j, k]
                cell_volume = cell_size_x * cell_size_y * cell_thickness
                cell_porosity = porosity_grid[i, j, k]
                cell_total_compressibility = total_compressibility_grid[i, j, k]
                cell_oil_pressure = current_oil_pressure_grid[i, j, k]

                # Accumulation term coefficient
                accumulation_coefficient = (
                    cell_porosity * cell_total_compressibility * cell_volume
                ) / time_step_size_in_days

                diagonal_values[cell_1D_index] = accumulation_coefficient
                rhs_values[cell_1D_index] = accumulation_coefficient * cell_oil_pressure

    return diagonal_values, rhs_values


@numba.njit(parallel=True, cache=True)
def compute_face_transmissibility_arrays(
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    water_mobility_grid_x: ThreeDimensionalGrid,
    oil_mobility_grid_x: ThreeDimensionalGrid,
    gas_mobility_grid_x: ThreeDimensionalGrid,
    water_mobility_grid_y: ThreeDimensionalGrid,
    oil_mobility_grid_y: ThreeDimensionalGrid,
    gas_mobility_grid_y: ThreeDimensionalGrid,
    water_mobility_grid_z: ThreeDimensionalGrid,
    oil_mobility_grid_z: ThreeDimensionalGrid,
    gas_mobility_grid_z: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    gravitational_constant: float,
    dtype: np.typing.DTypeLike,
) -> typing.Tuple[
    np.typing.NDArray,  # sparse_row_indices
    np.typing.NDArray,  # sparse_col_indices
    np.typing.NDArray,  # sparse_off_diagonal_values
    np.typing.NDArray,  # diagonal_additions       (length = interior_cell_count)
    np.typing.NDArray,  # rhs_additions            (length = interior_cell_count)
]:
    """
    Compute face transmissibility contributions for the implicit pressure matrix.

    For every interior cell we examine its three forward-facing neighbours:
        east   = (i+1, j,   k  )
        south  = (i,   j+1, k  )
        bottom = (i,   j,   k+1)

    Processing only forward-facing neighbours avoids double-counting each face.

    For each face shared between `this_cell` and `neighbour_cell` we compute:

        transmissibility  - total fluid mobility scaled by face geometry (ft²/psi·day)
        rhs_term          - capillary + gravity pseudo-flux across the face (ft²/day)

    These produce six contributions to the linear system A·p = b:

        Sparse off-diagonal:
            A[this_cell, neighbour_cell] -= transmissibility
            A[neighbour_cell, this_cell] -= transmissibility   (matrix is symmetric)

        Diagonal (resistance term, keeps matrix diagonally dominant):
            A[this_cell,      this_cell]      += transmissibility
            A[neighbour_cell, neighbour_cell] += transmissibility

        RHS (non-pressure driving forces — capillary and gravity):
            b[this_cell]      += rhs_term
            b[neighbour_cell] -= rhs_term    (equal and opposite)

    **Race-condition avoidance**

    The diagonal and RHS contributions are scatter-adds: processing the face
    between cells i and j updates both i and j. Running this naively with prange
    would cause two threads to race on those shared diagonal/RHS slots.

    Strategy:
        1. `prange` over i-slices. Each thread writes exclusively into its own
           row of the 2D thread-local buffers (indexed by [ii, local_slot]).
           Zero shared writes and zero race conditions.

        2. After the parallel loop, a single sequential pass over the 2D buffers
           packs the sparse arrays and scatter-adds into `diagonal_additions` and
           `rhs_additions` simultaneously. This is safe because it is single-threaded.

    This avoids any intermediate flat arrays for diagonal/RHS, saving
    two full-size allocations and an extra scan.

    Memory layout note:
        The 2D thread buffers are shaped (nx_interior, max_entries_per_i_slice).
        Accessing `[ii, local_slot]` is row-major and therefore cache-friendly
        since `local_slot` is the inner (fast-moving) index.

    :param cell_count_x: Number of cells in x-direction (including boundaries)
    :param cell_count_y: Number of cells in y-direction (including boundaries)
    :param cell_count_z: Number of cells in z-direction (including boundaries)
    :param thickness_grid: Cell thickness grid (ft)
    :param cell_size_x: Cell size in x-direction (ft)
    :param cell_size_y: Cell size in y-direction (ft)
    :param water_mobility_grid_x: Water mobility grid for x-direction faces (1/(cp*ft))
    :param oil_mobility_grid_x: Oil mobility grid for x-direction faces (1/(cp*ft))
    :param gas_mobility_grid_x: Gas mobility grid for x-direction faces (1/(cp*ft))
    :param water_mobility_grid_y: Water mobility grid for y-direction faces (1/(cp*ft))
    :param oil_mobility_grid_y: Oil mobility grid for y-direction faces (1/(cp*ft))
    :param gas_mobility_grid_y: Gas mobility grid for y-direction faces (1/(cp*ft))
    :param water_mobility_grid_z: Water mobility grid for z-direction faces (1/(cp*ft))
    :param oil_mobility_grid_z: Oil mobility grid for z-direction faces (1/(cp*ft))
    :param gas_mobility_grid_z: Gas mobility grid for z-direction faces (1/(cp*ft))
    :param oil_water_capillary_pressure_grid: Oil-water capillary pressure grid (psi)
    :param gas_oil_capillary_pressure_grid: Gas-oil capillary pressure grid (psi)
    :param oil_density_grid: Oil density grid (lb/ft³)
    :param water_density_grid: Water density grid (lb/ft³)
    :param gas_density_grid: Gas density grid (lb/ft³)
    :param elevation_grid: Cell elevation grid (ft)
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm)
    :param dtype: Data type for arrays (np.float32 or np.float64)
    :return: Tuple of (rows, cols, off_diag_values, diagonal_additions, rhs_additions)
        - rows: Sparse row indices for off-diagonal entries
        - cols: Sparse column indices for off-diagonal entries
        - off_diag_values: Values for sparse off-diagonal entries (-T_face)
        - diagonal_additions: Array to add to diagonal (indexed by cell_1D_index)
        - rhs_additions: Array to add to RHS (indexed by cell_1D_index)
    """
    nx_interior = cell_count_x - 2
    ny_interior = cell_count_y - 2
    nz_interior = cell_count_z - 2
    interior_cell_count = nx_interior * ny_interior * nz_interior

    # Each interior cell owns at most 3 forward faces.
    # Each face generates 2 sparse entries (one per direction of the symmetric pair).
    # The bound is tight for all-interior cells; boundary-adjacent cells generate fewer.
    max_entries_per_i_slice = ny_interior * nz_interior * 3 * 2

    # Thread-local 2D buffers — row ii is private to the thread handling
    # i-slice ii. No two threads ever write the same row.
    # Shaped (nx_interior, max_entries_per_i_slice) so that the inner
    # (slot) dimension is contiguous in memory which translates to cache-friendly writes.

    # Sparse COO entries for the off-diagonal part of A
    thread_sparse_row_indices = np.zeros(
        (nx_interior, max_entries_per_i_slice), dtype=np.int32
    )
    thread_sparse_col_indices = np.zeros(
        (nx_interior, max_entries_per_i_slice), dtype=np.int32
    )
    thread_sparse_off_diag_vals = np.zeros(
        (nx_interior, max_entries_per_i_slice), dtype=dtype
    )

    # Transmissibility values stored alongside the sparse entries.
    # The diagonal contribution for entry at sparse[ii, slot] is:
    #     diagonal_additions[row[ii,slot]] += transmissibility[ii,slot]
    #     diagonal_additions[col[ii,slot]] += transmissibility[ii,slot]
    # We store transmissibility once per face pair (both entries share the same T_face)
    # and reuse it for both the diagonal and RHS during the pack step.
    thread_transmissibility = np.zeros(
        (nx_interior, max_entries_per_i_slice), dtype=dtype
    )

    # RHS term — stored once per face pair, applied with opposite signs to
    # row-cell (+rhs_term) and col-cell (-rhs_term) during the pack step.
    thread_rhs_term = np.zeros((nx_interior, max_entries_per_i_slice), dtype=dtype)

    # How many entries each i-slice wrote — needed to know how far to read
    # during the sequential pack step.
    entries_written_per_i_slice = np.zeros(nx_interior, dtype=np.int32)

    # Parallel loop — each ii maps to grid index i = ii + 1.
    # All writes go to thread_*[ii, local_slot] which is private to this ii.
    for ii in numba.prange(nx_interior):  # type: ignore[attr-defined]
        i = ii + 1
        local_slot = 0

        for j in range(1, cell_count_y - 1):
            for k in range(1, cell_count_z - 1):
                this_cell_1d_index = to_1D_index_interior_only(
                    i=i,
                    j=j,
                    k=k,
                    cell_count_x=cell_count_x,
                    cell_count_y=cell_count_y,
                    cell_count_z=cell_count_z,
                )
                this_cell_thickness = thickness_grid[i, j, k]

                # EAST FACE (i+1, j, k) — x-direction flow
                east_i, east_j, east_k = i + 1, j, k
                if east_i < cell_count_x - 1:
                    east_cell_1d_index = to_1D_index_interior_only(
                        i=east_i,
                        j=east_j,
                        k=east_k,
                        cell_count_x=cell_count_x,
                        cell_count_y=cell_count_y,
                        cell_count_z=cell_count_z,
                    )
                    # Harmonic mean thickness at the shared face — ensures
                    # the effective flow path length is correctly weighted
                    # when adjacent cells have different thicknesses.
                    east_face_thickness = compute_harmonic_mean(
                        this_cell_thickness, thickness_grid[east_i, east_j, east_k]
                    )
                    # x-direction: cross-sectional area = Δy × h_face, distance = Δx
                    east_geometric_factor = (
                        cell_size_y * east_face_thickness / cell_size_x
                    )
                    east_face_mobility, east_cap_flux, east_grav_flux = (
                        compute_pseudo_fluxes_from_neighbour(
                            cell_indices=(i, j, k),
                            neighbour_indices=(east_i, east_j, east_k),
                            water_mobility_grid=water_mobility_grid_x,
                            oil_mobility_grid=oil_mobility_grid_x,
                            gas_mobility_grid=gas_mobility_grid_x,
                            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                            oil_density_grid=oil_density_grid,
                            water_density_grid=water_density_grid,
                            gas_density_grid=gas_density_grid,
                            elevation_grid=elevation_grid,
                            gravitational_constant=gravitational_constant,
                        )
                    )
                    east_transmissibility = east_face_mobility * east_geometric_factor
                    east_rhs_term = (
                        east_cap_flux + east_grav_flux
                    ) * east_geometric_factor

                    # Entry 0 of the pair: A[this_cell, east_cell]
                    thread_sparse_row_indices[ii, local_slot] = this_cell_1d_index
                    thread_sparse_col_indices[ii, local_slot] = east_cell_1d_index
                    thread_sparse_off_diag_vals[ii, local_slot] = -east_transmissibility
                    thread_transmissibility[ii, local_slot] = east_transmissibility
                    thread_rhs_term[ii, local_slot] = east_rhs_term
                    local_slot += 1

                    # Entry 1 of the pair: A[east_cell, this_cell] (symmetric)
                    # Transmissibility and rhs_term are identical in magnitude;
                    # the sign flip for rhs is applied during the pack step.
                    thread_sparse_row_indices[ii, local_slot] = east_cell_1d_index
                    thread_sparse_col_indices[ii, local_slot] = this_cell_1d_index
                    thread_sparse_off_diag_vals[ii, local_slot] = -east_transmissibility
                    thread_transmissibility[ii, local_slot] = east_transmissibility
                    thread_rhs_term[ii, local_slot] = east_rhs_term
                    local_slot += 1

                # SOUTH FACE (i, j+1, k) — y-direction flow
                south_i, south_j, south_k = i, j + 1, k
                if south_j < cell_count_y - 1:
                    south_cell_1d_index = to_1D_index_interior_only(
                        i=south_i,
                        j=south_j,
                        k=south_k,
                        cell_count_x=cell_count_x,
                        cell_count_y=cell_count_y,
                        cell_count_z=cell_count_z,
                    )
                    south_face_thickness = compute_harmonic_mean(
                        this_cell_thickness, thickness_grid[south_i, south_j, south_k]
                    )
                    # y-direction: cross-sectional area = Δx × h_face, distance = Δy
                    south_geometric_factor = (
                        cell_size_x * south_face_thickness / cell_size_y
                    )
                    south_face_mobility, south_cap_flux, south_grav_flux = (
                        compute_pseudo_fluxes_from_neighbour(
                            cell_indices=(i, j, k),
                            neighbour_indices=(south_i, south_j, south_k),
                            water_mobility_grid=water_mobility_grid_y,
                            oil_mobility_grid=oil_mobility_grid_y,
                            gas_mobility_grid=gas_mobility_grid_y,
                            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                            oil_density_grid=oil_density_grid,
                            water_density_grid=water_density_grid,
                            gas_density_grid=gas_density_grid,
                            elevation_grid=elevation_grid,
                            gravitational_constant=gravitational_constant,
                        )
                    )
                    south_transmissibility = (
                        south_face_mobility * south_geometric_factor
                    )
                    south_rhs_term = (
                        south_cap_flux + south_grav_flux
                    ) * south_geometric_factor

                    thread_sparse_row_indices[ii, local_slot] = this_cell_1d_index
                    thread_sparse_col_indices[ii, local_slot] = south_cell_1d_index
                    thread_sparse_off_diag_vals[
                        ii, local_slot
                    ] = -south_transmissibility
                    thread_transmissibility[ii, local_slot] = south_transmissibility
                    thread_rhs_term[ii, local_slot] = south_rhs_term
                    local_slot += 1

                    thread_sparse_row_indices[ii, local_slot] = south_cell_1d_index
                    thread_sparse_col_indices[ii, local_slot] = this_cell_1d_index
                    thread_sparse_off_diag_vals[
                        ii, local_slot
                    ] = -south_transmissibility
                    thread_transmissibility[ii, local_slot] = south_transmissibility
                    thread_rhs_term[ii, local_slot] = south_rhs_term
                    local_slot += 1

                # BOTTOM FACE (i, j, k+1) — z-direction flow
                bottom_i, bottom_j, bottom_k = i, j, k + 1
                if bottom_k < cell_count_z - 1:
                    bottom_cell_1d_index = to_1D_index_interior_only(
                        i=bottom_i,
                        j=bottom_j,
                        k=bottom_k,
                        cell_count_x=cell_count_x,
                        cell_count_y=cell_count_y,
                        cell_count_z=cell_count_z,
                    )
                    bottom_face_thickness = compute_harmonic_mean(
                        this_cell_thickness,
                        thickness_grid[bottom_i, bottom_j, bottom_k],
                    )
                    # z-direction: cross-sectional area = Δx × Δy, distance = h_face
                    bottom_geometric_factor = (
                        cell_size_x * cell_size_y / bottom_face_thickness
                    )
                    bottom_face_mobility, bottom_cap_flux, bottom_grav_flux = (
                        compute_pseudo_fluxes_from_neighbour(
                            cell_indices=(i, j, k),
                            neighbour_indices=(bottom_i, bottom_j, bottom_k),
                            water_mobility_grid=water_mobility_grid_z,
                            oil_mobility_grid=oil_mobility_grid_z,
                            gas_mobility_grid=gas_mobility_grid_z,
                            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                            oil_density_grid=oil_density_grid,
                            water_density_grid=water_density_grid,
                            gas_density_grid=gas_density_grid,
                            elevation_grid=elevation_grid,
                            gravitational_constant=gravitational_constant,
                        )
                    )
                    bottom_transmissibility = (
                        bottom_face_mobility * bottom_geometric_factor
                    )
                    bottom_rhs_term = (
                        bottom_cap_flux + bottom_grav_flux
                    ) * bottom_geometric_factor

                    thread_sparse_row_indices[ii, local_slot] = this_cell_1d_index
                    thread_sparse_col_indices[ii, local_slot] = bottom_cell_1d_index
                    thread_sparse_off_diag_vals[
                        ii, local_slot
                    ] = -bottom_transmissibility
                    thread_transmissibility[ii, local_slot] = bottom_transmissibility
                    thread_rhs_term[ii, local_slot] = bottom_rhs_term
                    local_slot += 1

                    thread_sparse_row_indices[ii, local_slot] = bottom_cell_1d_index
                    thread_sparse_col_indices[ii, local_slot] = this_cell_1d_index
                    thread_sparse_off_diag_vals[
                        ii, local_slot
                    ] = -bottom_transmissibility
                    thread_transmissibility[ii, local_slot] = bottom_transmissibility
                    thread_rhs_term[ii, local_slot] = bottom_rhs_term
                    local_slot += 1

        entries_written_per_i_slice[ii] = local_slot

    # Prefix-sum to find each i-slice's start offset in the flat output.
    # Sequential, O(nx_interior), negligible cost.
    slice_start_offsets = np.zeros(nx_interior + 1, dtype=np.int32)
    for ii in range(nx_interior):
        slice_start_offsets[ii + 1] = (
            slice_start_offsets[ii] + entries_written_per_i_slice[ii]
        )
    total_sparse_entries = slice_start_offsets[nx_interior]

    # Output arrays
    sparse_row_indices = np.empty(total_sparse_entries, dtype=np.int32)
    sparse_col_indices = np.empty(total_sparse_entries, dtype=np.int32)
    sparse_off_diagonal_values = np.empty(total_sparse_entries, dtype=dtype)
    diagonal_additions = np.zeros(interior_cell_count, dtype=dtype)
    rhs_additions = np.zeros(interior_cell_count, dtype=dtype)

    # Single sequential pass: pack sparse arrays + scatter-add diagonal
    # and RHS simultaneously.
    #
    # For each face pair stored at (ii, slot) and (ii, slot+1):
    #   slot+0 → A[this_cell,      neighbour_cell]:  row=this,      col=neighbour
    #   slot+1 → A[neighbour_cell, this_cell     ]:  row=neighbour, col=this
    #
    # Diagonal: both cells receive +transmissibility (stored identically in both slots)
    # RHS:      row-cell receives +rhs_term, col-cell receives -rhs_term.
    #           Since slot+0 has row=this and slot+1 has row=neighbour, we can
    #           apply +rhs_term to row for both slots — the sign is already
    #           encoded by whether the cell is the "owner" (slot+0) or "neighbour"
    #           (slot+1) of the face.
    #
    # Wait — rhs_term has the same magnitude in both slots but opposite sign
    # semantics. We apply:
    #       diagonal_additions[row] += transmissibility   (same for both slots)
    #       rhs_additions[row]      += rhs_term           (slot+0: this gets +)
    #       rhs_additions[col]      -= rhs_term           (slot+0: neighbour gets -)
    # We process pairs (step=2) so we only visit slot+0 and derive slot+1 from it.
    for ii in range(nx_interior):
        out_start = slice_start_offsets[ii]
        count = entries_written_per_i_slice[ii]

        # Process face pairs — entries are always written in pairs (step=2)
        pair = 0
        while pair < count:
            out = out_start + pair

            # Slot 0 of the pair: A[this_cell, neighbour_cell]
            this_cell = thread_sparse_row_indices[ii, pair]
            neighbour_cell = thread_sparse_col_indices[ii, pair]
            transmissibility = thread_transmissibility[ii, pair]
            rhs_term = thread_rhs_term[ii, pair]

            # Pack sparse entry for slot 0
            sparse_row_indices[out] = this_cell
            sparse_col_indices[out] = neighbour_cell
            sparse_off_diagonal_values[out] = thread_sparse_off_diag_vals[ii, pair]

            # Pack sparse entry for slot 1 (symmetric)
            sparse_row_indices[out + 1] = neighbour_cell
            sparse_col_indices[out + 1] = this_cell
            sparse_off_diagonal_values[out + 1] = thread_sparse_off_diag_vals[
                ii, pair + 1
            ]

            # Diagonal: both cells accumulate +transmissibility
            diagonal_additions[this_cell] += transmissibility
            diagonal_additions[neighbour_cell] += transmissibility

            # RHS: this_cell gets +rhs_term, neighbour_cell gets -rhs_term
            rhs_additions[this_cell] += rhs_term
            rhs_additions[neighbour_cell] -= rhs_term

            pair += 2

    return (
        sparse_row_indices,
        sparse_col_indices,
        sparse_off_diagonal_values,
        diagonal_additions,
        rhs_additions,
    )


@attrs.frozen(slots=True)
class WellContributions:
    """
    COO-style arrays representing well contributions to the linear system A·p = b.

    Wells only ever touch the diagonal of A (each perforated cell couples only
    to itself via the productivity index). This means we never need off-diagonal
    entries, just two flat arrays of (index, value) pairs that get scatter-added
    into the final diagonal and RHS vectors before COO matrix construction.

    diagonal_cell_indices[k] and diagonal_values[k] represent:
        A[diagonal_cell_indices[k], diagonal_cell_indices[k]] += diagonal_values[k]

    rhs_cell_indices[k] and rhs_values[k] represent:
        b[rhs_cell_indices[k]] += rhs_values[k]

    In practice `diagonal_cell_indices` == `rhs_cell_indices` since every perforation
    contributes to both A and b at the same cell. They are stored separately to
    keep the interface clean and to allow skipped perforations (non-finite BHP)
    to contribute to neither without special-casing the reduction.
    """

    diagonal_cell_indices: np.typing.NDArray
    diagonal_values: np.typing.NDArray
    rhs_cell_indices: np.typing.NDArray
    rhs_values: np.typing.NDArray


def compute_well_contributions(
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    current_oil_pressure_grid: ThreeDimensionalGrid,
    temperature_grid: ThreeDimensionalGrid,
    water_relative_mobility_grid: ThreeDimensionalGrid,
    oil_relative_mobility_grid: ThreeDimensionalGrid,
    gas_relative_mobility_grid: ThreeDimensionalGrid,
    water_compressibility_grid: ThreeDimensionalGrid,
    oil_compressibility_grid: ThreeDimensionalGrid,
    gas_compressibility_grid: ThreeDimensionalGrid,
    fluid_properties: FluidProperties[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    time: float,
    config: Config,
    well_indices_cache: WellIndicesCache,
    md_per_cp_to_ft2_per_psi_per_day: float,
    dtype: np.typing.DTypeLike,
    pad_width: int = 1,
) -> WellContributions:
    """
    Compute well contributions to the implicit pressure linear system A·p = b
    and return them as COO-style flat arrays.

    Wells contribute only to the diagonal of A and to b.  The semi-implicit
    Robin boundary condition for both injection and production takes the form:

        A[cell, cell] += PI_phase (productivity index adds to diagonal)
        b[cell] += PI_phase * BHP (BHP-weighted PI adds to RHS)

    Perforations where BHP is non-finite are silently skipped. They contribute
    zero to both arrays, which is equivalent to closing that perforation for
    this time step.

    :param cell_count_x: Number of cells in x-direction (including ghost cells)
    :param cell_count_y: Number of cells in y-direction (including ghost cells)
    :param cell_count_z: Number of cells in z-direction (including ghost cells)
    :param thickness_grid: Cell thickness grid (ft)
    :param current_oil_pressure_grid: Current oil pressure grid (psi)
    :param temperature_grid: Temperature grid (°F)
    :param absolute_permeability: Absolute permeability in x, y, z (mD)
    :param water_relative_mobility_grid: Water relative mobility (ft²/psi·day)
    :param oil_relative_mobility_grid: Oil relative mobility (ft²/psi·day)
    :param gas_relative_mobility_grid: Gas relative mobility (ft²/psi·day)
    :param water_compressibility_grid: Water compressibility (1/psi)
    :param oil_compressibility_grid: Oil compressibility (1/psi)
    :param gas_compressibility_grid: Gas compressibility (1/psi)
    :param fluid_properties: Full fluid properties container
    :param wells: Wells container with injection and production wells
    :param cell_size_x: Cell size in x-direction (ft)
    :param cell_size_y: Cell size in y-direction (ft)
    :param time_step: Current time step number (used for anomaly warnings)
    :param time_step_size: Time step size in seconds (used for anomaly warnings)
    :param config: Simulation configuration
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor
    :param pad_width: Ghost cell offset applied to well coordinates
    :return: `WellContributions` with four flat arrays ready for scatter-add
        into the final diagonal and RHS before COO matrix construction.
    """
    _to_1D_index = functools.partial(
        to_1D_index_interior_only,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
    )
    water_bubble_point_pressure_grid = fluid_properties.water_bubble_point_pressure_grid
    gas_formation_volume_factor_grid = fluid_properties.gas_formation_volume_factor_grid
    gas_solubility_in_water_grid = fluid_properties.gas_solubility_in_water_grid
    water_fvf_grid = fluid_properties.water_formation_volume_factor_grid
    oil_fvf_grid = fluid_properties.oil_formation_volume_factor_grid

    # Accumulate contributions into plain Python lists. Well perforation count
    # is O(10–1000) so list append is perfectly fast here and avoids
    # pre-allocating a fixed-size array that we would need to trim later.
    diagonal_cell_indices: typing.List[int] = []
    diagonal_values: typing.List[float] = []
    rhs_cell_indices: typing.List[int] = []
    rhs_values: typing.List[float] = []

    def _add_bhp_contribution(
        cell_1d_index: int,
        productivity_index: float,
        effective_bhp: float,
    ) -> None:
        """
        Local helper that appends one (diagonal, rhs) contribution pair for bhp (Robin) controlled wells.

        Skips automatically when productivity_index is zero (no mobility)
        or when bhp is non-finite (degenerate well state).
        """
        if not np.isfinite(effective_bhp) or productivity_index == 0.0:
            return

        if cell_1d_index not in diagonal_cell_indices:
            diagonal_cell_indices.append(cell_1d_index)
            diagonal_values.append(productivity_index)
        else:
            idx = diagonal_cell_indices.index(cell_1d_index)
            diagonal_values[idx] += productivity_index

        if cell_1d_index not in rhs_cell_indices:
            rhs_cell_indices.append(cell_1d_index)
            rhs_values.append(productivity_index * effective_bhp)
        else:
            idx = rhs_cell_indices.index(cell_1d_index)
            rhs_values[idx] += productivity_index * effective_bhp

    def _add_rate_contribution(
        cell_1d_index: int,
        flow_rate: float,
    ) -> None:
        """
        Local helper that appends a rhs contribution pair for rate (Neumann) controlled wells.
        """
        if cell_1d_index not in rhs_cell_indices:
            rhs_cell_indices.append(cell_1d_index)
            rhs_values.append(flow_rate)
        else:
            idx = rhs_cell_indices.index(cell_1d_index)
            rhs_values[idx] += flow_rate

    bbl_to_ft3 = c.BARRELS_TO_CUBIC_FEET
    for well in wells.injection_wells:
        if not well.is_open or well.injected_fluid is None:
            continue

        injected_fluid = well.injected_fluid
        injected_phase = injected_fluid.phase
        is_bhp_controlled = well.control.is_bhp_control()
        use_pseudo_pressure = (
            config.use_pseudo_pressure and injected_phase == FluidPhase.GAS
        )

        well_indices = well_indices_cache.injection[well.name]
        # Compute BHP and productivity index per perforation.
        for perforation_index in well_indices:
            i, j, k = perforation_index.cell
            cell_1d_index = perforation_index.cell_1d_index
            well_index = perforation_index.well_index
            allocation_fraction = well_indices.allocation_fraction(perforation_index)
            cell_pressure = typing.cast(float, current_oil_pressure_grid[i, j, k])
            cell_temperature = typing.cast(float, temperature_grid[i, j, k])
            phase_fvf = typing.cast(
                float,
                injected_fluid.get_formation_volume_factor(
                    pressure=cell_pressure,
                    temperature=cell_temperature,
                ),
            )

            if injected_phase == FluidPhase.GAS:
                phase_mobility = typing.cast(float, gas_relative_mobility_grid[i, j, k])
                compressibility_kwargs: dict = {}
            else:
                phase_mobility = typing.cast(
                    float, water_relative_mobility_grid[i, j, k]
                )
                compressibility_kwargs = {
                    "bubble_point_pressure": water_bubble_point_pressure_grid[i, j, k],
                    "gas_formation_volume_factor": gas_formation_volume_factor_grid[
                        i, j, k
                    ],
                    "gas_solubility_in_water": gas_solubility_in_water_grid[i, j, k],
                }

            phase_compressibility = typing.cast(
                float,
                injected_fluid.get_compressibility(
                    pressure=cell_pressure,
                    temperature=cell_temperature,
                    **compressibility_kwargs,
                ),
            )
            effective_mobility = typing.cast(
                float,
                water_relative_mobility_grid[i, j, k]
                + oil_relative_mobility_grid[i, j, k]
                + gas_relative_mobility_grid[i, j, k],
            )
            if is_bhp_controlled:
                effective_bhp = well.get_bottom_hole_pressure(
                    pressure=cell_pressure,
                    temperature=cell_temperature,
                    phase_mobility=effective_mobility,
                    well_index=well_index,
                    fluid=injected_fluid,
                    formation_volume_factor=phase_fvf,
                    allocation_fraction=allocation_fraction,
                    use_pseudo_pressure=use_pseudo_pressure,
                    fluid_compressibility=phase_compressibility,
                    pvt_tables=None,
                )

                if not np.isfinite(effective_bhp):
                    logger.error(
                        f"Non-finite BHP for injection well {well.name!r} "
                        f"at cell ({i - pad_width},{j - pad_width},{k - pad_width}): {effective_bhp}. Skipping perforation."
                    )
                    continue

                if abs(effective_bhp - cell_pressure) > 1e6:
                    logger.warning(
                        f"Extreme BHP for injection well {well.name!r} "
                        f"at cell ({i - pad_width}, {j - pad_width}, {k - pad_width}): {effective_bhp:.2e} psi "
                        f"(reservoir pressure: {cell_pressure:.1f} psi)."
                    )

                if cell_pressure > effective_bhp and config.warn_well_anomalies:
                    _warn_injection_pressure(
                        bhp=effective_bhp,
                        cell_pressure=cell_pressure,
                        well_name=well.name,
                        time=time,
                        cell=(i - pad_width, j - pad_width, k - pad_width),
                    )

                if injected_phase == FluidPhase.GAS:
                    use_pp, pp_table = get_pseudo_pressure_table(
                        fluid=injected_fluid,
                        pressure=cell_pressure,
                        temperature=cell_temperature,
                        use_pseudo_pressure=use_pseudo_pressure,
                        pvt_tables=config.pvt_tables,
                    )
                    specific_gravity = typing.cast(
                        float,
                        injected_fluid.get_specific_gravity(
                            pressure=cell_pressure,
                            temperature=cell_temperature,
                        ),
                    )
                    avg_z_factor = compute_average_compressibility_factor(
                        pressure=cell_pressure,
                        temperature=cell_temperature,
                        gas_gravity=specific_gravity,
                    )
                    productivity_index = compute_gas_productivity_index(
                        well_index=well_index,
                        phase_mobility=phase_mobility,
                        pressure=cell_pressure,
                        temperature=cell_temperature,
                        bottom_hole_pressure=effective_bhp,
                        formation_volume_factor=phase_fvf,
                        average_compressibility_factor=avg_z_factor,
                        use_pseudo_pressure=use_pp,
                        pseudo_pressure_table=pp_table,
                    )
                else:
                    productivity_index = compute_oil_productivity_index(
                        well_index=well_index,
                        phase_mobility=phase_mobility,
                        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                    )

                _add_bhp_contribution(cell_1d_index, productivity_index, effective_bhp)

            else:
                # Rate-controlled: We use direct rate injection (Neumann Boundary)
                flow_rate = well.get_flow_rate(
                    pressure=cell_pressure,
                    temperature=cell_temperature,
                    phase_mobility=effective_mobility,
                    well_index=well_index,
                    fluid=injected_fluid,
                    formation_volume_factor=phase_fvf,
                    allocation_fraction=allocation_fraction,
                    use_pseudo_pressure=use_pseudo_pressure,
                    fluid_compressibility=phase_compressibility,
                    pvt_tables=None,
                )

                if flow_rate < 0.0 and config.warn_well_anomalies:
                    _warn_injection_rate(
                        injection_rate=flow_rate,
                        well_name=well.name,
                        time=time,
                        cell=(i - pad_width, j - pad_width, k - pad_width),
                        rate_unit="ft³/day"
                        if injected_phase == FluidPhase.GAS
                        else "bbls/day",
                    )

                if injected_phase != FluidPhase.GAS:
                    flow_rate *= bbl_to_ft3

                _add_rate_contribution(cell_1d_index, flow_rate)

    for well in wells.production_wells:
        if not well.is_open:
            continue

        well_indices = well_indices_cache.production[well.name]
        is_couple_controlled = isinstance(well.control, CoupledRateControl)

        # One contribution per (perforation, produced phase) pair
        for perforation_index in well_indices:
            i, j, k = perforation_index.cell
            cell_1d_index = perforation_index.cell_1d_index
            well_index = perforation_index.well_index
            allocation_fraction = well_indices.allocation_fraction(perforation_index)
            cell_pressure = typing.cast(float, current_oil_pressure_grid[i, j, k])
            cell_temperature = typing.cast(float, temperature_grid[i, j, k])

            primary_phase_context = {}
            if is_couple_controlled:
                primary_phase_context = well.control.build_primary_phase_context(  # type: ignore
                    produced_fluids=well.produced_fluids,
                    oil_mobility=typing.cast(
                        float, oil_relative_mobility_grid[i, j, k]
                    ),
                    water_mobility=typing.cast(
                        float, water_relative_mobility_grid[i, j, k]
                    ),
                    gas_mobility=typing.cast(
                        float, gas_relative_mobility_grid[i, j, k]
                    ),
                    oil_fvf=typing.cast(float, oil_fvf_grid[i, j, k]),
                    water_fvf=typing.cast(float, water_fvf_grid[i, j, k]),
                    gas_fvf=typing.cast(
                        float, gas_formation_volume_factor_grid[i, j, k]
                    ),
                    oil_compressibility=typing.cast(
                        float, oil_compressibility_grid[i, j, k]
                    ),
                    water_compressibility=typing.cast(
                        float, water_compressibility_grid[i, j, k]
                    ),
                    gas_compressibility=typing.cast(
                        float, gas_compressibility_grid[i, j, k]
                    ),
                )

            for produced_fluid in well.produced_fluids:
                produced_phase = produced_fluid.phase

                if produced_phase == FluidPhase.GAS:
                    phase_mobility = typing.cast(
                        float, gas_relative_mobility_grid[i, j, k]
                    )
                    phase_compressibility = typing.cast(
                        float, gas_compressibility_grid[i, j, k]
                    )
                    phase_fvf = typing.cast(
                        float, gas_formation_volume_factor_grid[i, j, k]
                    )
                elif produced_phase == FluidPhase.WATER:
                    phase_mobility = typing.cast(
                        float, water_relative_mobility_grid[i, j, k]
                    )
                    phase_compressibility = typing.cast(
                        float, water_compressibility_grid[i, j, k]
                    )
                    phase_fvf = typing.cast(float, water_fvf_grid[i, j, k])
                else:  # Oil
                    phase_mobility = typing.cast(
                        float, oil_relative_mobility_grid[i, j, k]
                    )
                    phase_compressibility = typing.cast(
                        float, oil_compressibility_grid[i, j, k]
                    )
                    phase_fvf = typing.cast(float, oil_fvf_grid[i, j, k])

                use_pseudo_pressure = (
                    config.use_pseudo_pressure and produced_phase == FluidPhase.GAS
                )
                effective_bhp = well.get_bottom_hole_pressure(
                    pressure=cell_pressure,
                    temperature=cell_temperature,
                    phase_mobility=phase_mobility,
                    well_index=well_index,
                    fluid=produced_fluid,
                    formation_volume_factor=phase_fvf,
                    allocation_fraction=allocation_fraction,
                    use_pseudo_pressure=use_pseudo_pressure,
                    fluid_compressibility=phase_compressibility,
                    pvt_tables=config.pvt_tables,
                    **primary_phase_context,
                )

                if not np.isfinite(effective_bhp):
                    logger.error(
                        f"Non-finite BHP for production well {well.name!r} "
                        f"at cell ({i - pad_width},{j - pad_width},{k - pad_width}): {effective_bhp}. Skipping perforation."
                    )
                    continue

                if abs(effective_bhp - cell_pressure) > 1e6:
                    logger.warning(
                        f"Extreme BHP for production well {well.name!r} "
                        f"at cell ({i - pad_width},{j - pad_width},{k - pad_width}): {effective_bhp:.2e} psi "
                        f"(reservoir pressure: {cell_pressure:.1f} psi)."
                    )

                if cell_pressure < effective_bhp and config.warn_well_anomalies:
                    _warn_production_pressure(
                        bhp=effective_bhp,
                        cell_pressure=cell_pressure,
                        well_name=well.name,
                        time=time,
                        cell=(i - pad_width, j - pad_width, k - pad_width),
                    )

                if produced_phase == FluidPhase.GAS:
                    use_pp, pp_table = get_pseudo_pressure_table(
                        fluid=produced_fluid,
                        pressure=cell_pressure,
                        temperature=cell_temperature,
                        use_pseudo_pressure=use_pseudo_pressure,
                        pvt_tables=config.pvt_tables,
                    )
                    specific_gravity = typing.cast(
                        float,
                        produced_fluid.get_specific_gravity(
                            pressure=cell_pressure,
                            temperature=cell_temperature,
                        ),
                    )
                    avg_z_factor = compute_average_compressibility_factor(
                        pressure=cell_pressure,
                        temperature=cell_temperature,
                        gas_gravity=specific_gravity,
                    )
                    productivity_index = compute_gas_productivity_index(
                        well_index=well_index,
                        phase_mobility=phase_mobility,
                        pressure=cell_pressure,
                        temperature=cell_temperature,
                        bottom_hole_pressure=effective_bhp,
                        formation_volume_factor=phase_fvf,
                        average_compressibility_factor=avg_z_factor,
                        use_pseudo_pressure=use_pp,
                        pseudo_pressure_table=pp_table,
                    )
                else:
                    productivity_index = compute_oil_productivity_index(
                        well_index=well_index,
                        phase_mobility=phase_mobility,
                        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                    )

                _add_bhp_contribution(cell_1d_index, productivity_index, effective_bhp)

                logger.info(
                    f"Producer gas phase: PI={productivity_index:.4f}, "
                    f"BHP={effective_bhp:.2f}, Cell Pressure={cell_pressure:.2f}, "
                    f"∆P={cell_pressure - effective_bhp:.2f}, "
                    f"Krg={gas_relative_mobility_grid[i, j, k] * fluid_properties.gas_viscosity_grid[i, j, k]:.3e}, "
                    f"Sg={fluid_properties.gas_saturation_grid[i, j, k]:.3e}"
                )

    return WellContributions(
        diagonal_cell_indices=np.array(diagonal_cell_indices, dtype=np.int32),
        diagonal_values=np.array(diagonal_values, dtype=dtype),
        rhs_cell_indices=np.array(rhs_cell_indices, dtype=np.int32),
        rhs_values=np.array(rhs_values, dtype=dtype),
    )


@numba.njit(cache=True)
def compute_pseudo_fluxes_from_neighbour(
    cell_indices: ThreeDimensions,
    neighbour_indices: ThreeDimensions,
    water_mobility_grid: ThreeDimensionalGrid,
    oil_mobility_grid: ThreeDimensionalGrid,
    gas_mobility_grid: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    gravitational_constant: float,
) -> typing.Tuple[float, float, float]:
    """
    Computes and returns a tuple of the total harmonic mobility of the phases, the capillary pseudo flux,
    and the gravity pseudo flux from the neighbour to the current cell.

    Pseudo flux comes about from the fact that the fluxes returned are not actual fluxes with units of ft³/day,
    but rather pseudo fluxes with units of ft²/day, which are then used to compute the actual fluxes and subsequently,
    the flow rates in the implicit pressure evolution scheme.

    :param cell_indices: Indices of the current cell (i, j, k)
    :param neighbour_indices: Indices of the neighbouring cell (i±1, j, k) or (i, j±1, k) or (i, j, k±1)
    :param oil_pressure_grid: 3D grid of oil pressures (psi)
    :param water_mobility_grid: 3D grid of water mobilities (ft²/psi.day)
    :param oil_mobility_grid: 3D grid of oil mobilities (ft²/psi.day)
    :param gas_mobility_grid: 3D grid of gas mobilities (ft²/psi.day)
    :param oil_water_capillary_pressure_grid: 3D grid of oil-water capillary pressures (psi)
    :param gas_oil_capillary_pressure_grid: 3D grid of gas-oil capillary pressures (psi)
    :param oil_density_grid: 3D grid of oil densities (lb/ft³)
    :param water_density_grid: 3D grid of water densities (lb/ft³)
    :param gas_density_grid: 3D grid of gas densities (lb/ft³)
    :param elevation_grid: 3D grid of elevations (ft)
    :return: A tuple containing:
        - Total harmonic mobility (ft²/psi.day)
        - Total capillary pseudo flux (ft²/day)
        - Total gravity pseudo flux (ft²/day)
    """
    # Calculate pressure differences relative to current cell (Neighbour - Current)
    # These represent the gradients driving flow from neighbour to current cell, or vice versa
    oil_water_capillary_pressure_difference = (
        oil_water_capillary_pressure_grid[neighbour_indices]
        - oil_water_capillary_pressure_grid[cell_indices]
    )
    gas_oil_capillary_pressure_difference = (
        gas_oil_capillary_pressure_grid[neighbour_indices]
        - gas_oil_capillary_pressure_grid[cell_indices]
    )

    # Calculate the elevation difference between the neighbour and current cell
    elevation_difference = (
        elevation_grid[neighbour_indices] - elevation_grid[cell_indices]
    )
    # Determine the harmonic densities for each phase across the face
    harmonic_water_density = compute_harmonic_mean(
        water_density_grid[neighbour_indices], water_density_grid[cell_indices]
    )
    harmonic_oil_density = compute_harmonic_mean(
        oil_density_grid[neighbour_indices], oil_density_grid[cell_indices]
    )
    harmonic_gas_density = compute_harmonic_mean(
        gas_density_grid[neighbour_indices], gas_density_grid[cell_indices]
    )

    # Calculate harmonic mobilities for each phase across the face
    water_harmonic_mobility = compute_harmonic_mean(
        water_mobility_grid[neighbour_indices], water_mobility_grid[cell_indices]
    )
    oil_harmonic_mobility = compute_harmonic_mean(
        oil_mobility_grid[neighbour_indices], oil_mobility_grid[cell_indices]
    )
    gas_harmonic_mobility = compute_harmonic_mean(
        gas_mobility_grid[neighbour_indices], gas_mobility_grid[cell_indices]
    )
    total_harmonic_mobility = (
        water_harmonic_mobility + oil_harmonic_mobility + gas_harmonic_mobility
    )
    if total_harmonic_mobility <= 0.0:
        # No flow can occur if there is no mobility
        return 0.0, 0.0, 0.0

    # λ_w * (P_cow_{n+1} - P_cow_{n}) (ft²/psi.day * psi = ft²/day)
    water_capillary_pseudo_flux = (
        water_harmonic_mobility * oil_water_capillary_pressure_difference
    )
    # λ_g * (P_cgo_{n+1} - P_cgo_{n}) (ft²/psi.day * psi = ft²/day)
    gas_capillary_pseudo_flux = (
        gas_harmonic_mobility * gas_oil_capillary_pressure_difference
    )
    # Total capillary flux from the neighbour (ft²/day)
    total_capillary_pseudo_flux = (
        water_capillary_pseudo_flux + gas_capillary_pseudo_flux
    )

    # Calculate the phase gravity potentials (hydrostatic/gravity head)
    water_gravity_potential = (
        harmonic_water_density * gravitational_constant * elevation_difference
    ) / 144.0
    oil_gravity_potential = (
        harmonic_oil_density * gravitational_constant * elevation_difference
    ) / 144.0
    gas_gravity_potential = (
        harmonic_gas_density * gravitational_constant * elevation_difference
    ) / 144.0
    # Total gravity pseudo flux (ft²/day)
    water_gravity_pseudo_flux = water_harmonic_mobility * water_gravity_potential
    oil_gravity_pseudo_flux = oil_harmonic_mobility * oil_gravity_potential
    gas_gravity_pseudo_flux = gas_harmonic_mobility * gas_gravity_potential
    total_gravity_pseudo_flux = (
        water_gravity_pseudo_flux + oil_gravity_pseudo_flux + gas_gravity_pseudo_flux
    )
    return (
        total_harmonic_mobility,
        total_capillary_pseudo_flux,
        total_gravity_pseudo_flux,
    )


"""
Implicit finite difference formulation for pressure diffusion in a 3D reservoir
(slightly compressible fluid):

The governing equation for pressure evolution is the linear-flow diffusivity equation:

    ∂p/∂t * (φ·c_t) * V = ∇·(λ·∇p) * A + q * V

Where:
    ∂p/∂t * (φ·c_t) * V — Accumulation term
    ∇·(λ·∇p) * A — Diffusion term (Darcy's law)
    q * V — Source/sink term

Assumptions:
    - Constant porosity (φ), compressibility (c_t), and density (ρ)
    - No reaction or advection terms (pressure-only evolution)
    - Capillary effects optional, appear in source term via pressure corrections

Diffusion term expanded in 3D:

    ∇·(λ·∇p) = ∂/∂x(λ·∂p/∂x) + ∂/∂y(λ·∂p/∂y) + ∂/∂z(λ·∂p/∂z)

Discretization:

Using Backward Euler in time:

    ∂p/∂t ≈ (pⁿ⁺¹_ijk - pⁿ_ijk) / Δt

Using central differences in space:

    ∂/∂x(λ·∂p/∂x) ≈ [λ_{i+½,j,k}·(pⁿ⁺¹_{i+1,j,k} - pⁿ⁺¹_{i,j,k}) - λ_{i⁻½,j,k}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i⁻1,j,k})] / Δx²
    ∂/∂y(λ·∂p/∂y) ≈ [λ_{i,j+½,k}·(pⁿ⁺¹_{i,j+1,k} - pⁿ⁺¹_{i,j,k}) - λ_{i,j⁻½,k}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i,j⁻1,k})] / Δy²
    ∂/∂z(λ·∂p/∂z) ≈ [λ_{i,j,k+½}·(pⁿ⁺¹_{i,j,k+1} - pⁿ⁺¹_{i,j,k}) - λ_{i,j,k⁻½}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i,j,k⁻1})] / Δz²

Putting it all together:

    (pⁿ⁺¹_ijk - pⁿ_ijk) * (φ·c_t·V) / Δt =
        A/Δx · [λ_{i+½,j,k}·(pⁿ⁺¹_{i+1,j,k} - pⁿ⁺¹_{i,j,k}) - λ_{i⁻½,j,k}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i⁻1,j,k})] +
        A/Δy · [λ_{i,j+½,k}·(pⁿ⁺¹_{i,j+1,k} - pⁿ⁺¹_{i,j,k}) - λ_{i,j⁻½,k}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i,j⁻1,k})] +
        A/Δz · [λ_{i,j,k+½}·(pⁿ⁺¹_{i,j,k+1} - pⁿ⁺¹_{i,j,k}) - λ_{i,j,k⁻½}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i,j,k⁻1})] +
        qⁿ⁺¹_ijk * V

Matrix form:

Let:
    Tx⁺ = λ_{i+½,j,k}·A / Δx
    Tx⁻ = λ_{i⁻½,j,k}·A / Δx
    Ty⁺ = λ_{i,j+½,k}·A / Δy
    Ty⁻ = λ_{i,j⁻½,k}·A / Δy
    Tz⁺ = λ_{i,j,k+½}·A / Δz
    Tz⁻ = λ_{i,j,k⁻½}·A / Δz
    β   = φ·c_t·V / Δt

Then:

    A_{ijk,ijk}     = β + Tx⁺ + Tx⁻ + Ty⁺ + Ty⁻ + Tz⁺ + Tz⁻
    A_{ijk,i+1jk}   = -Tx⁺
    A_{ijk,i-1jk}   = -Tx⁻
    A_{ijk,ij+1k}   = -Ty⁺
    A_{ijk,ij-1k}   = -Ty⁻
    A_{ijk,ijk+1}   = -Tz⁺
    A_{ijk,ijk-1}   = -Tz⁻

RHS vector: (Contains terms that actually drive flow)

    b_{ijk} = (β * pⁿ_{ijk}) + (q_{ijk} * V) + Total Capillary Driven Flow + Gravity Driven Flow/Segregation

Capillary pressure driven flow term (if multiphase):

    total_capillary_flow = sum of directional contributions:
        For each direction:
            [(λ_w * ∇P_cow) + (λ_g * ∇P_cgo)] * A / (Δx, Δy, Δz)

Gravity driven segregation (in effect in all directions; dominant in z, non-zero in x/y for dipping reservoirs):

    For each face between cell and neighbour:

    gravity_potential_phase = (harmonic_ρ_phase * g/gc * ∆elevation) / 144   [psi]

    total_gravity_flow = (
            [λ_w * (harmonic_ρ_w * g/gc * ∆elevation) / 144]
            + [λ_o * (harmonic_ρ_o * g/gc * ∆elevation) / 144]
            + [λ_g * (harmonic_ρ_g * g/gc * ∆elevation) / 144]
    ) * A / ΔL

    Where:
    g/gc = 32.174 / 32.174 = 1.0 (dimensionless in consistent imperial units),
    ∆elevation = elevation_neighbour - elevation_current (ft),
    harmonic_ρ_w, harmonic_ρ_o, harmonic_ρ_g are harmonic mean densities at the face (lbm/ft³),
    144 converts lbf/ft² to psi.

This results in a 7-point stencil sparse matrix (in 3D) for solving A·pⁿ⁺¹ = b.

Notes:
    - Harmonic averaging is used for both λ and ρ at cell interfaces
    - Capillary pressure is optional but included via ∇P_cow and ∇P_cgo terms
    - The system is solved each time step to advance pressure implicitly
"""

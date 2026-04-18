import logging
import typing

import attrs
import numba
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix

from bores.config import Config
from bores.constants import c
from bores.correlations.core import compute_harmonic_mean
from bores.datastructures import PhaseTensorsProxy
from bores.errors import PreconditionerError, SolverError
from bores.grids.base import CapillaryPressureGrids, RelativeMobilityGrids
from bores.grids.pvt import build_total_fluid_compressibility_grid
from bores.models import FluidProperties, RockProperties
from bores.solvers.base import (
    EvolutionResult,
    _warn_injection_pressure,
    _warn_production_pressure,
    solve_linear_system,
    to_1D_index,
)
from bores.transmissibility import FaceTransmissibilities
from bores.types import (
    FluidPhase,
    OneDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensions,
)
from bores.wells.base import Wells
from bores.wells.controls import CoupledRateControl
from bores.wells.indices import WellIndicesCache

logger = logging.getLogger(__name__)


@attrs.frozen
class ImplicitPressureSolution:
    pressure_grid: ThreeDimensionalGrid
    maximum_pressure_change: float


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
    face_transmissibilities: FaceTransmissibilities,
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
    wells: Wells[ThreeDimensions],
    config: Config,
    well_indices_cache: WellIndicesCache,
    injection_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    production_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    dtype: npt.DTypeLike = np.float64,
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
    :param face_transmissibilities: `FaceTransmissibilities` object containing face transmissibility grids
    :param pressure_boundaries: 3D grid of boundary pressures (psi) with ghost-cell indexing
    :param flux_boundaries: 3D grid of boundary fluxes (ft³/day) with ghost-cell indexing
    :param wells: `Wells` object containing well definitions and properties
    :param config: `Config` object containing simulation config
    :param boundary_conditions: Model boundary conditions.
    :param well_indices_cache: Cache of well indices for efficient lookup during pressure solve.
    :param injection_bhps: `PhaseTensorsProxy` of injection bottom hole pressures for each phase and cell.
    :param production_bhps: `PhaseTensorsProxy` of production bottom hole pressures for each phase and cell.
    :param pad_width: Number of ghost cells used for grid padding. Well coordinates are offset by this amount.
    :return: `EvolutionResult` containing the new pressure grid and scheme used
    """
    porosity_grid = rock_properties.porosity_grid
    net_to_gross_grid = rock_properties.net_to_gross_grid
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
    time_step_size_in_days = time_step_size * c.DAYS_PER_SECOND

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
            compute_accumulation_contributions,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            thickness_grid=thickness_grid,
            porosity_grid=porosity_grid,
            net_to_gross_grid=net_to_gross_grid,
            total_compressibility_grid=total_compressibility_grid,
            current_oil_pressure_grid=current_oil_pressure_grid,
            cell_size_x=cell_size_x,
            cell_size_y=cell_size_y,
            time_step_size_in_days=time_step_size_in_days,
            dtype=dtype,
        )
        face_transmissibility_future = pool.submit(
            compute_face_flux_contributions,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            pressure_boundaries=pressure_boundaries,
            flux_boundaries=flux_boundaries,
            water_relative_mobility_grid=water_relative_mobility_grid,
            oil_relative_mobility_grid=oil_relative_mobility_grid,
            gas_relative_mobility_grid=gas_relative_mobility_grid,
            face_transmissibilities_x=face_transmissibilities.x,
            face_transmissibilities_y=face_transmissibilities.y,
            face_transmissibilities_z=face_transmissibilities.z,
            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
            oil_density_grid=oil_density_grid,
            water_density_grid=water_density_grid,
            gas_density_grid=gas_density_grid,
            elevation_grid=elevation_grid,
            gravitational_constant=gravitational_constant,
            md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
            dtype=dtype,
        )
        wells_future = pool.submit(
            compute_well_contributions,
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
            injection_bhps=injection_bhps,
            production_bhps=production_bhps,
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
        (
            well_diagonal_cell_indices,
            well_diagonal_values,
            well_rhs_cell_indices,
            well_rhs_values,
        ) = wells_future.result()

    else:
        diagonal_values, rhs_values = compute_accumulation_contributions(
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            thickness_grid=thickness_grid,
            porosity_grid=porosity_grid,
            net_to_gross_grid=net_to_gross_grid,
            total_compressibility_grid=total_compressibility_grid,
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
        ) = compute_face_flux_contributions(
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            pressure_boundaries=pressure_boundaries,
            flux_boundaries=flux_boundaries,
            water_relative_mobility_grid=water_relative_mobility_grid,
            oil_relative_mobility_grid=oil_relative_mobility_grid,
            gas_relative_mobility_grid=gas_relative_mobility_grid,
            face_transmissibilities_x=face_transmissibilities.x,
            face_transmissibilities_y=face_transmissibilities.y,
            face_transmissibilities_z=face_transmissibilities.z,
            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
            oil_density_grid=oil_density_grid,
            water_density_grid=water_density_grid,
            gas_density_grid=gas_density_grid,
            elevation_grid=elevation_grid,
            gravitational_constant=gravitational_constant,
            md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
            dtype=dtype,
        )
        (
            well_diagonal_cell_indices,
            well_diagonal_values,
            well_rhs_cell_indices,
            well_rhs_values,
        ) = compute_well_contributions(
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
            injection_bhps=injection_bhps,
            production_bhps=production_bhps,
        )

    # Merge into final diagonal and b.
    final_diagonal = diagonal_values + diagonal_additions
    np.add.at(
        final_diagonal,
        well_diagonal_cell_indices,
        well_diagonal_values,
    )

    residual_vector = rhs_values + rhs_additions
    np.add.at(
        residual_vector,
        well_rhs_cell_indices,
        well_rhs_values,
    )

    cell_count = cell_count_x * cell_count_y * cell_count_z
    diagional_indices = np.arange(cell_count, dtype=np.int32)
    jacobian = coo_matrix(  # type: ignore
        (
            np.concatenate([final_diagonal, sparse_off_diag]),
            (
                np.concatenate([diagional_indices, sparse_rows]),
                np.concatenate([diagional_indices, sparse_cols]),
            ),
        ),
        shape=(cell_count, cell_count),
        dtype=dtype,  # type: ignore
    )

    # Scale Jacobian and residual by inverse diagonal to improve conditioning for iterative solver
    D = np.abs(jacobian.diagonal())
    D = np.where(D > 0, D, 1.0)
    jacobian = jacobian / D[:, None]
    residual_vector = residual_vector / D
    
    # Solve the linear system A·pⁿ⁺¹ = b
    try:
        new_1D_pressure_grid, _ = solve_linear_system(
            A_csr=jacobian.tocsr(),
            b=residual_vector,
            rtol=config.pressure_convergence_tolerance,
            maximum_iterations=config.maximum_solver_iterations,
            solver=config.pressure_solver,
            preconditioner=config.pressure_preconditioner,
            fallback_to_direct=True,
        )
    except (SolverError, PreconditionerError) as exc:
        logger.error("Pressure solve failed at time step %d: %s", time_step, exc)
        return EvolutionResult(
            value=ImplicitPressureSolution(
                pressure_grid=current_oil_pressure_grid.astype(dtype, copy=False),
                maximum_pressure_change=0.0,  # No change since solve failed
            ),
            success=False,
            scheme="implicit",
            message=str(exc),
        )

    # Map solution back to 3D grid
    new_pressure_grid = map_solution_to_grid(
        solution_1D=new_1D_pressure_grid,
        solution_grid=current_oil_pressure_grid.copy(),
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
    )
    maximum_pressure_change = np.max(
        np.abs(new_pressure_grid - current_oil_pressure_grid)
    )
    return EvolutionResult(
        value=ImplicitPressureSolution(
            pressure_grid=new_pressure_grid.astype(dtype, copy=False),
            maximum_pressure_change=maximum_pressure_change,
        ),
        success=True,
        scheme="implicit",
        message=f"Implicit pressure evolution for time step {time_step} successful.",
    )


@numba.njit(parallel=True, cache=True)
def map_solution_to_grid(
    solution_1D: OneDimensionalGrid,
    solution_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
) -> ThreeDimensionalGrid:
    """
    Map the 1D solution array back to a 3D grid (in-place).

    This function takes the solution from the linear system solver and fills them into a 3D grid.

    :param solution_1D: 1D array containing solution for interior cells only
    :param current_grid: Current 3D grid (used to preserve boundary values)
    :param cell_count_x: Number of cells in x-direction
    :param cell_count_y: Number of cells in y-direction
    :param cell_count_z: Number of cells in z-direction
    :return: 3D grid with solution mapped in place
    """
    # Fill interior cells with solution using parallel processing
    for i in numba.prange(cell_count_x):  # type: ignore
        for j in range(cell_count_y):
            for k in range(cell_count_z):
                # Convert 3D indices to 1D index for solution array
                idx = to_1D_index(
                    i=i,
                    j=j,
                    k=k,
                    cell_count_x=cell_count_x,
                    cell_count_y=cell_count_y,
                    cell_count_z=cell_count_z,
                )
                solution_grid[i, j, k] = solution_1D[idx]
    return solution_grid


@numba.njit(parallel=True, cache=True)
def compute_accumulation_contributions(
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    net_to_gross_grid: ThreeDimensionalGrid,
    total_compressibility_grid: ThreeDimensionalGrid,
    current_oil_pressure_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    time_step_size_in_days: float,
    dtype: npt.DTypeLike,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    """
    Compute accumulation terms for all interior cells and return as dense arrays.

    This function computes the diagonal coefficients and RHS values that will be used
    to initialize the sparse matrix A and vector b. The accumulation term represents
    the storage capacity of each cell: accumulation_coefficient = (φ * c_t * V) / Δt

    :param cell_count_x: Number of cells in x-direction
    :param cell_count_y: Number of cells in y-direction
    :param cell_count_z: Number of cells in z-direction
    :param thickness_grid: Cell thickness grid (ft)
    :param porosity_grid: Cell porosity grid (fraction)
    :param net_to_gross_grid: Cell net-to-gross ratio grid (fraction)
    :param total_compressibility_grid: Total compressibility grid (1/psi)
    :param current_oil_pressure_grid: Current oil pressure grid (psi)
    :param cell_size_x: Cell size in x-direction (ft)
    :param cell_size_y: Cell size in y-direction (ft)
    :param time_step_size_in_days: Time step size (days)
    :param dtype: Data type for arrays (np.float32 or np.float64)
    :return: Tuple of (diagonal_values, rhs_values) both 1D arrays of length `interior_cell_count`
    """
    cell_count = cell_count_x * cell_count_y * cell_count_z
    diagonal_values = np.zeros(cell_count, dtype=dtype)
    rhs_values = np.zeros(cell_count, dtype=dtype)

    for i in numba.prange(cell_count_x):  # type: ignore
        for j in range(cell_count_y):
            for k in range(cell_count_z):
                cell_1D_index = to_1D_index(
                    i=i,
                    j=j,
                    k=k,
                    cell_count_x=cell_count_x,
                    cell_count_y=cell_count_y,
                    cell_count_z=cell_count_z,
                )

                cell_volume = (
                    cell_size_x
                    * cell_size_y
                    * thickness_grid[i, j, k]
                    * net_to_gross_grid[i, j, k]
                )
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
def compute_face_flux_contributions(
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
    water_relative_mobility_grid: ThreeDimensionalGrid,
    oil_relative_mobility_grid: ThreeDimensionalGrid,
    gas_relative_mobility_grid: ThreeDimensionalGrid,
    face_transmissibilities_x: ThreeDimensionalGrid,
    face_transmissibilities_y: ThreeDimensionalGrid,
    face_transmissibilities_z: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    gravitational_constant: float,
    md_per_cp_to_ft2_per_psi_per_day: float,
    dtype: npt.DTypeLike,
) -> typing.Tuple[
    npt.NDArray,  # sparse_row_indices
    npt.NDArray,  # sparse_col_indices
    npt.NDArray,  # sparse_off_diagonal_values
    npt.NDArray,  # diagonal_additions       (length = cell_count)
    npt.NDArray,  # rhs_additions            (length = cell_count)
]:
    """
    Compute face flux contributions for the implicit pressure matrix.

    The grid passed in is the real (unpadded) grid — shape (cell_count_x, cell_count_y,
    cell_count_z) with no ghost-cell layers. Boundary conditions are supplied via two
    dicts keyed on ghost-cell 1D indices computed *as if* the grid were padded by one
    layer on each side (i.e. using cell_count_* + 2 for the padded dimensions).

    For every interior cell we examine its six face neighbours:
        east   = (i+1, j,   k  )
        west   = (i-1, j,   k  )
        south  = (i,   j+1, k  )
        north  = (i,   j-1, k  )
        bottom = (i,   j,   k+1)
        top    = (i,   j,   k-1)

    For each face we classify the neighbour:

    1. **Interior neighbour** (indices in [0, cell_count_*)):
       Standard interior-interior pair — symmetric off-diagonal entries and
       both diagonal contributions added. The RHS gets equal and opposite
       capillary + gravity terms.

    2. **Out-of-bounds neighbour** — boundary face. Convert the out-of-bounds
       indices to a ghost-cell 1D index using the *padded* grid dimensions
       (cell_count_* + 2, with a +1 offset on each axis so that valid cell (0,0,0)
       maps to padded index (1,1,1)):

           ghost_1d = to_1D_index(
               i=i_neighbour + 1,
               j=j_neighbour + 1,
               k=k_neighbour + 1,
               cell_count_x=cell_count_x + 2,
               cell_count_y=cell_count_y + 2,
               cell_count_z=cell_count_z + 2,
           )

       Then look it up:

       a. **Dirichlet** (ghost_1d in pressure_boundaries):
          Known boundary pressure p_bc.
          - diagonal += T_face              (A[cell, cell] += T)
          - rhs      += T_face * p_bc       (known pressure drives flow)
          - rhs      += capillary_flux + gravity_flux          (non-pressure driving forces)
          No off-diagonal entry (neighbour is not a free unknown).

       b. **Neumann** (ghost_1d in flux_boundaries):
          Known boundary flux q_bc in ft³/day already.
          - rhs += q_bc                     (flux applied directly to RHS)
          No diagonal or off-diagonal entry.

    For each face shared between `this_cell` and `neighbour_cell` (interior pair):

        transmissibility  - total fluid mobility scaled by face geometry (ft³/psi·day)
        rhs_term          - capillary + gravity pseudo-flux across the face (ft³/day)

    These produce contributions to the linear system A·p = b:

        Sparse off-diagonal (interior-interior only):
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
        The 2D thread buffers are shaped (cell_count_x, max_entries_per_i_slice).
        Accessing `[ii, local_slot]` is row-major and therefore cache-friendly
        since `local_slot` is the inner (fast-moving) index.

    :param cell_count_x: Number of cells in x-direction (real grid, no ghost cells)
    :param cell_count_y: Number of cells in y-direction (real grid, no ghost cells)
    :param cell_count_z: Number of cells in z-direction (real grid, no ghost cells)
    :param pressure_boundaries: Dict mapping ghost-cell padded-1D-index → boundary pressure (psi).
        Keyed on 1D indices in the (cell_count_x+2, cell_count_y+2, cell_count_z+2) padded space.
        Represents Dirichlet (fixed pressure) boundary conditions.
    :param flux_boundaries: Dict mapping ghost-cell padded-1D-index → boundary flux (ft³/day).
        Keyed on 1D indices in the (cell_count_x+2, cell_count_y+2, cell_count_z+2) padded space.
        Represents Neumann (fixed flux) boundary conditions.
    :param current_oil_pressure_grid: Current oil pressure grid (psi), shape (nx, ny, nz)
    :param water_relative_mobility_grid: Water relative mobility grid (ft²/psi·day)
    :param oil_relative_mobility_grid: Oil relative mobility grid (ft²/psi·day)
    :param gas_relative_mobility_grid: Gas relative mobility grid (ft²/psi·day)
    :param face_transmissibilities_x: Face transmissibilities in x-direction (mD·ft/cp)
    :param face_transmissibilities_y: Face transmissibilities in y-direction (mD·ft/cp)
    :param face_transmissibilities_z: Face transmissibilities in z-direction (mD·ft/cp)
    :param oil_water_capillary_pressure_grid: Oil-water capillary pressure grid (psi)
    :param gas_oil_capillary_pressure_grid: Gas-oil capillary pressure grid (psi)
    :param oil_density_grid: Oil density grid (lb/ft³)
    :param water_density_grid: Water density grid (lb/ft³)
    :param gas_density_grid: Gas density grid (lb/ft³)
    :param elevation_grid: Cell elevation grid (ft)
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm)
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor
    :param dtype: Data type for arrays (np.float32 or np.float64)
    :return: Tuple of (rows, cols, off_diag_values, diagonal_additions, rhs_additions)
        - rows: Sparse row indices for off-diagonal entries
        - cols: Sparse column indices for off-diagonal entries
        - off_diag_values: Values for sparse off-diagonal entries (-T_face)
        - diagonal_additions: Array to add to diagonal (indexed by cell_1D_index), length = cell_count
        - rhs_additions: Array to add to RHS (indexed by cell_1D_index), length = cell_count
    """
    cell_count = cell_count_x * cell_count_y * cell_count_z

    # Each cell owns at most 6 faces, but we use a forward-only pass for interior
    # pairs (east/south/bottom) so each interior face is processed once. Boundary
    # faces (all six directions) are handled per cell when the neighbour is out-of-bounds.
    # Upper bound: 3 forward interior pairs (2 sparse entries each) + 6 boundary singletons.
    max_entries_per_i_slice = cell_count_y * cell_count_z * (3 * 2 + 6)

    # Thread-local 2D buffers — row ii is private to the thread handling i=ii.
    # Inner (slot) dimension is contiguous in memory for cache-friendly writes.

    # Sparse COO entries for off-diagonal part of A (interior-interior pairs only)
    thread_sparse_row_indices = np.zeros(
        (cell_count_x, max_entries_per_i_slice), dtype=np.int32
    )
    thread_sparse_col_indices = np.zeros(
        (cell_count_x, max_entries_per_i_slice), dtype=np.int32
    )
    thread_sparse_off_diag_vals = np.zeros(
        (cell_count_x, max_entries_per_i_slice), dtype=dtype
    )

    # Transmissibility stored once per face (shared by both entries of an interior pair,
    # or the single entry of a boundary singleton). Reused for diagonal and RHS.
    thread_transmissibility = np.zeros(
        (cell_count_x, max_entries_per_i_slice), dtype=dtype
    )

    # RHS (capillary + gravity) term stored once per face.
    # For interior pairs: applied as +rhs_term to this_cell and -rhs_term to neighbour.
    # For boundary singletons: applied as +rhs_term to owner cell (convention: ghost on right).
    thread_rhs_term = np.zeros((cell_count_x, max_entries_per_i_slice), dtype=dtype)

    # Boundary classification flags and data for singleton faces.
    # is_dirichlet: True  → Dirichlet BC (diagonal += T, rhs += T*p_bc + rhs_term)
    # is_neumann:   True  → Neumann BC  (rhs += flux_bc only)
    # (Both False → interior-interior pair)
    thread_is_dirichlet = np.zeros(
        (cell_count_x, max_entries_per_i_slice), dtype=np.bool_
    )
    thread_is_neumann = np.zeros(
        (cell_count_x, max_entries_per_i_slice), dtype=np.bool_
    )

    # For Dirichlet singletons: the known boundary pressure value.
    thread_bc_pressure = np.zeros((cell_count_x, max_entries_per_i_slice), dtype=dtype)

    # For Neumann singletons: the known boundary flux (ft³/day), applied directly to RHS.
    thread_bc_flux = np.zeros((cell_count_x, max_entries_per_i_slice), dtype=dtype)

    # 1D real-grid index of the interior cell that owns the boundary face.
    thread_owner_cell = np.zeros(
        (cell_count_x, max_entries_per_i_slice), dtype=np.int32
    )

    # How many slots each i-slice actually wrote (needed by sequential pack step).
    entries_written_per_i_slice = np.zeros(cell_count_x, dtype=np.int32)

    for ii in numba.prange(cell_count_x):  # type: ignore
        i = ii  # No padding offset — grid indices run 0..cell_count_*-1 directly.
        local_slot = 0

        for j in range(cell_count_y):
            for k in range(cell_count_z):
                this_cell_1d_index = to_1D_index(
                    i=i,
                    j=j,
                    k=k,
                    cell_count_x=cell_count_x,
                    cell_count_y=cell_count_y,
                    cell_count_z=cell_count_z,
                )

                # SIX FACE DIRECTIONS
                # Forward faces (east/south/bottom): processed once per face to avoid
                # double-counting interior pairs. Also catch boundary faces here.
                # Backward faces (west/north/top): only written when the neighbour is
                # out-of-bounds (boundary face), so no double-counting occurs.

                # EAST FACE (i+1, j, k)
                ei, ej, ek = i + 1, j, k
                if ei < cell_count_x:
                    # Interior neighbour: write symmetric off-diagonal pair.
                    transmissibility, capillary_flux, gravity_flux = (
                        compute_fluxes_from_neighbour(
                            cell_indices=(i, j, k),
                            neighbour_indices=(ei, ej, ek),
                            water_relative_mobility_grid=water_relative_mobility_grid,
                            oil_relative_mobility_grid=oil_relative_mobility_grid,
                            gas_relative_mobility_grid=gas_relative_mobility_grid,
                            face_transmissibility=face_transmissibilities_x[
                                i + 1, j + 1, k + 1
                            ],  # face transmissibilities are defined on the padded grid
                            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                            oil_density_grid=oil_density_grid,
                            water_density_grid=water_density_grid,
                            gas_density_grid=gas_density_grid,
                            elevation_grid=elevation_grid,
                            gravitational_constant=gravitational_constant,
                            md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                        )
                    )
                    east_1d = to_1D_index(
                        i=ei,
                        j=ej,
                        k=ek,
                        cell_count_x=cell_count_x,
                        cell_count_y=cell_count_y,
                        cell_count_z=cell_count_z,
                    )
                    # Entry 0: A[this, east]
                    thread_sparse_row_indices[ii, local_slot] = this_cell_1d_index
                    thread_sparse_col_indices[ii, local_slot] = east_1d
                    thread_sparse_off_diag_vals[ii, local_slot] = -transmissibility
                    thread_transmissibility[ii, local_slot] = transmissibility
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    thread_is_dirichlet[ii, local_slot] = False
                    thread_is_neumann[ii, local_slot] = False
                    local_slot += 1
                    # Entry 1: A[east, this]
                    thread_sparse_row_indices[ii, local_slot] = east_1d
                    thread_sparse_col_indices[ii, local_slot] = this_cell_1d_index
                    thread_sparse_off_diag_vals[ii, local_slot] = -transmissibility
                    thread_transmissibility[ii, local_slot] = transmissibility
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    thread_is_dirichlet[ii, local_slot] = False
                    thread_is_neumann[ii, local_slot] = False
                    local_slot += 1
                else:
                    # Out-of-bounds: look up ghost-cell 1D index in padded space.
                    # Ghost cell lives at (ei+1, ej+1, ek+1) in padded grid.
                    pei, pej, pek = ei + 1, ej + 1, ek + 1
                    transmissibility, capillary_flux, gravity_flux = (
                        compute_fluxes_from_neighbour(
                            cell_indices=(i, j, k),
                            # Clamp neighbour to last real cell for mobility/density lookup —
                            # the ghost cell has no real properties; using the boundary cell
                            # itself gives a one-sided (zero-gradient) approximation.
                            neighbour_indices=(cell_count_x - 1, ej, ek),
                            water_relative_mobility_grid=water_relative_mobility_grid,
                            oil_relative_mobility_grid=oil_relative_mobility_grid,
                            gas_relative_mobility_grid=gas_relative_mobility_grid,
                            face_transmissibility=face_transmissibilities_x[
                                i + 1, j + 1, k + 1
                            ],
                            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                            oil_density_grid=oil_density_grid,
                            water_density_grid=water_density_grid,
                            gas_density_grid=gas_density_grid,
                            elevation_grid=elevation_grid,
                            gravitational_constant=gravitational_constant,
                            md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                        )
                    )
                    thread_owner_cell[ii, local_slot] = this_cell_1d_index
                    thread_transmissibility[ii, local_slot] = transmissibility
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    pressure_boundary = pressure_boundaries[pei, pej, pek]
                    if not np.isnan(pressure_boundary):
                        thread_is_dirichlet[ii, local_slot] = True
                        thread_is_neumann[ii, local_slot] = False
                        thread_bc_pressure[ii, local_slot] = pressure_boundary
                    else:
                        # Neumann: flux_boundaries[ghost_1d] is the known flux (ft³/day).
                        thread_is_dirichlet[ii, local_slot] = False
                        thread_is_neumann[ii, local_slot] = True
                        thread_bc_flux[ii, local_slot] = flux_boundaries[pei, pej, pek]
                    local_slot += 1

                # WEST FACE (i-1, j, k) — boundary-only
                # Only written when neighbour is out-of-bounds (boundary face).
                # Interior west faces are handled as the east face of the western cell.
                wi, wj, wk = i - 1, j, k
                if wi < 0:
                    pwi, pwj, pwk = wi + 1, wj + 1, wk + 1
                    transmissibility, capillary_flux, gravity_flux = (
                        compute_fluxes_from_neighbour(
                            cell_indices=(i, j, k),
                            neighbour_indices=(0, wj, wk),  # clamp to first real cell
                            water_relative_mobility_grid=water_relative_mobility_grid,
                            oil_relative_mobility_grid=oil_relative_mobility_grid,
                            gas_relative_mobility_grid=gas_relative_mobility_grid,
                            face_transmissibility=face_transmissibilities_x[
                                pwi, pwj, pwk
                            ],
                            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                            oil_density_grid=oil_density_grid,
                            water_density_grid=water_density_grid,
                            gas_density_grid=gas_density_grid,
                            elevation_grid=elevation_grid,
                            gravitational_constant=gravitational_constant,
                            md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                        )
                    )
                    thread_owner_cell[ii, local_slot] = this_cell_1d_index
                    thread_transmissibility[ii, local_slot] = transmissibility
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    pressure_boundary = pressure_boundaries[pwi, pwj, pwk]
                    if not np.isnan(pressure_boundary):
                        thread_is_dirichlet[ii, local_slot] = True
                        thread_is_neumann[ii, local_slot] = False
                        thread_bc_pressure[ii, local_slot] = pressure_boundary
                    else:
                        thread_is_dirichlet[ii, local_slot] = False
                        thread_is_neumann[ii, local_slot] = True
                        thread_bc_flux[ii, local_slot] = flux_boundaries[pwi, pwj, pwk]
                    local_slot += 1

                # SOUTH FACE (i, j+1, k)
                si, sj, sk = i, j + 1, k
                if sj < cell_count_y:
                    transmissibility, capillary_flux, gravity_flux = (
                        compute_fluxes_from_neighbour(
                            cell_indices=(i, j, k),
                            neighbour_indices=(si, sj, sk),
                            water_relative_mobility_grid=water_relative_mobility_grid,
                            oil_relative_mobility_grid=oil_relative_mobility_grid,
                            gas_relative_mobility_grid=gas_relative_mobility_grid,
                            face_transmissibility=face_transmissibilities_y[
                                i + 1, j + 1, k + 1
                            ],
                            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                            oil_density_grid=oil_density_grid,
                            water_density_grid=water_density_grid,
                            gas_density_grid=gas_density_grid,
                            elevation_grid=elevation_grid,
                            gravitational_constant=gravitational_constant,
                            md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                        )
                    )
                    south_1d = to_1D_index(
                        i=si,
                        j=sj,
                        k=sk,
                        cell_count_x=cell_count_x,
                        cell_count_y=cell_count_y,
                        cell_count_z=cell_count_z,
                    )
                    thread_sparse_row_indices[ii, local_slot] = this_cell_1d_index
                    thread_sparse_col_indices[ii, local_slot] = south_1d
                    thread_sparse_off_diag_vals[ii, local_slot] = -transmissibility
                    thread_transmissibility[ii, local_slot] = transmissibility
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    thread_is_dirichlet[ii, local_slot] = False
                    thread_is_neumann[ii, local_slot] = False
                    local_slot += 1
                    thread_sparse_row_indices[ii, local_slot] = south_1d
                    thread_sparse_col_indices[ii, local_slot] = this_cell_1d_index
                    thread_sparse_off_diag_vals[ii, local_slot] = -transmissibility
                    thread_transmissibility[ii, local_slot] = transmissibility
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    thread_is_dirichlet[ii, local_slot] = False
                    thread_is_neumann[ii, local_slot] = False
                    local_slot += 1
                else:
                    psi, psj, psk = si + 1, sj + 1, sk + 1
                    transmissibility, capillary_flux, gravity_flux = (
                        compute_fluxes_from_neighbour(
                            cell_indices=(i, j, k),
                            neighbour_indices=(si, cell_count_y - 1, sk),
                            water_relative_mobility_grid=water_relative_mobility_grid,
                            oil_relative_mobility_grid=oil_relative_mobility_grid,
                            gas_relative_mobility_grid=gas_relative_mobility_grid,
                            face_transmissibility=face_transmissibilities_y[
                                i + 1, j + 1, k + 1
                            ],
                            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                            oil_density_grid=oil_density_grid,
                            water_density_grid=water_density_grid,
                            gas_density_grid=gas_density_grid,
                            elevation_grid=elevation_grid,
                            gravitational_constant=gravitational_constant,
                            md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                        )
                    )
                    thread_owner_cell[ii, local_slot] = this_cell_1d_index
                    thread_transmissibility[ii, local_slot] = transmissibility
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    pressure_boundary = pressure_boundaries[psi, psj, psk]
                    if not np.isnan(pressure_boundary):
                        thread_is_dirichlet[ii, local_slot] = True
                        thread_is_neumann[ii, local_slot] = False
                        thread_bc_pressure[ii, local_slot] = pressure_boundary
                    else:
                        thread_is_dirichlet[ii, local_slot] = False
                        thread_is_neumann[ii, local_slot] = True
                        thread_bc_flux[ii, local_slot] = flux_boundaries[psi, psj, psk]
                    local_slot += 1

                # NORTH FACE (i, j-1, k) — boundary-only
                ni, nj, nk = i, j - 1, k
                if nj < 0:
                    pni, pnj, pnk = ni + 1, nj + 1, nk + 1
                    transmissibility, capillary_flux, gravity_flux = (
                        compute_fluxes_from_neighbour(
                            cell_indices=(i, j, k),
                            neighbour_indices=(ni, 0, nk),
                            water_relative_mobility_grid=water_relative_mobility_grid,
                            oil_relative_mobility_grid=oil_relative_mobility_grid,
                            gas_relative_mobility_grid=gas_relative_mobility_grid,
                            face_transmissibility=face_transmissibilities_y[
                                pni, pnj, pnk
                            ],
                            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                            oil_density_grid=oil_density_grid,
                            water_density_grid=water_density_grid,
                            gas_density_grid=gas_density_grid,
                            elevation_grid=elevation_grid,
                            gravitational_constant=gravitational_constant,
                            md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                        )
                    )
                    thread_owner_cell[ii, local_slot] = this_cell_1d_index
                    thread_transmissibility[ii, local_slot] = transmissibility
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    pressure_boundary = pressure_boundaries[pni, pnj, pnk]
                    if not np.isnan(pressure_boundary):
                        thread_is_dirichlet[ii, local_slot] = True
                        thread_is_neumann[ii, local_slot] = False
                        thread_bc_pressure[ii, local_slot] = pressure_boundary
                    else:
                        thread_is_dirichlet[ii, local_slot] = False
                        thread_is_neumann[ii, local_slot] = True
                        thread_bc_flux[ii, local_slot] = flux_boundaries[pni, pnj, pnk]
                    local_slot += 1

                # BOTTOM FACE (i, j, k+1)
                bi, bj, bk = i, j, k + 1
                if bk < cell_count_z:
                    transmissibility, capillary_flux, gravity_flux = (
                        compute_fluxes_from_neighbour(
                            cell_indices=(i, j, k),
                            neighbour_indices=(bi, bj, bk),
                            water_relative_mobility_grid=water_relative_mobility_grid,
                            oil_relative_mobility_grid=oil_relative_mobility_grid,
                            gas_relative_mobility_grid=gas_relative_mobility_grid,
                            face_transmissibility=face_transmissibilities_z[
                                i + 1, j + 1, k + 1
                            ],
                            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                            oil_density_grid=oil_density_grid,
                            water_density_grid=water_density_grid,
                            gas_density_grid=gas_density_grid,
                            elevation_grid=elevation_grid,
                            gravitational_constant=gravitational_constant,
                            md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                        )
                    )
                    bottom_1d = to_1D_index(
                        i=bi,
                        j=bj,
                        k=bk,
                        cell_count_x=cell_count_x,
                        cell_count_y=cell_count_y,
                        cell_count_z=cell_count_z,
                    )
                    thread_sparse_row_indices[ii, local_slot] = this_cell_1d_index
                    thread_sparse_col_indices[ii, local_slot] = bottom_1d
                    thread_sparse_off_diag_vals[ii, local_slot] = -transmissibility
                    thread_transmissibility[ii, local_slot] = transmissibility
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    thread_is_dirichlet[ii, local_slot] = False
                    thread_is_neumann[ii, local_slot] = False
                    local_slot += 1
                    thread_sparse_row_indices[ii, local_slot] = bottom_1d
                    thread_sparse_col_indices[ii, local_slot] = this_cell_1d_index
                    thread_sparse_off_diag_vals[ii, local_slot] = -transmissibility
                    thread_transmissibility[ii, local_slot] = transmissibility
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    thread_is_dirichlet[ii, local_slot] = False
                    thread_is_neumann[ii, local_slot] = False
                    local_slot += 1
                else:
                    pbi, pbj, pbk = bi + 1, bj + 1, bk + 1
                    transmissibility, capillary_flux, gravity_flux = (
                        compute_fluxes_from_neighbour(
                            cell_indices=(i, j, k),
                            neighbour_indices=(bi, bj, cell_count_z - 1),
                            water_relative_mobility_grid=water_relative_mobility_grid,
                            oil_relative_mobility_grid=oil_relative_mobility_grid,
                            gas_relative_mobility_grid=gas_relative_mobility_grid,
                            face_transmissibility=face_transmissibilities_z[
                                i + 1, j + 1, k + 1
                            ],
                            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                            oil_density_grid=oil_density_grid,
                            water_density_grid=water_density_grid,
                            gas_density_grid=gas_density_grid,
                            elevation_grid=elevation_grid,
                            gravitational_constant=gravitational_constant,
                            md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                        )
                    )
                    thread_owner_cell[ii, local_slot] = this_cell_1d_index
                    thread_transmissibility[ii, local_slot] = transmissibility
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    pressure_boundary = pressure_boundaries[pbi, pbj, pbk]
                    if not np.isnan(pressure_boundary):
                        thread_is_dirichlet[ii, local_slot] = True
                        thread_is_neumann[ii, local_slot] = False
                        thread_bc_pressure[ii, local_slot] = pressure_boundary
                    else:
                        thread_is_dirichlet[ii, local_slot] = False
                        thread_is_neumann[ii, local_slot] = True
                        thread_bc_flux[ii, local_slot] = flux_boundaries[pbi, pbj, pbk]
                    local_slot += 1

                # TOP FACE (i, j, k-1) — boundary-only
                ti, tj, tk = i, j, k - 1
                if tk < 0:
                    pti, ptj, ptk = ti + 1, tj + 1, tk + 1
                    transmissibility, capillary_flux, gravity_flux = (
                        compute_fluxes_from_neighbour(
                            cell_indices=(i, j, k),
                            neighbour_indices=(ti, tj, 0),
                            water_relative_mobility_grid=water_relative_mobility_grid,
                            oil_relative_mobility_grid=oil_relative_mobility_grid,
                            gas_relative_mobility_grid=gas_relative_mobility_grid,
                            face_transmissibility=face_transmissibilities_z[
                                pti, ptj, ptk
                            ],
                            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                            oil_density_grid=oil_density_grid,
                            water_density_grid=water_density_grid,
                            gas_density_grid=gas_density_grid,
                            elevation_grid=elevation_grid,
                            gravitational_constant=gravitational_constant,
                            md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                        )
                    )
                    thread_owner_cell[ii, local_slot] = this_cell_1d_index
                    thread_transmissibility[ii, local_slot] = transmissibility
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    pressure_boundary = pressure_boundaries[pti, ptj, ptk]
                    if not np.isnan(pressure_boundary):
                        thread_is_dirichlet[ii, local_slot] = True
                        thread_is_neumann[ii, local_slot] = False
                        thread_bc_pressure[ii, local_slot] = pressure_boundary
                    else:
                        thread_is_dirichlet[ii, local_slot] = False
                        thread_is_neumann[ii, local_slot] = True
                        thread_bc_flux[ii, local_slot] = flux_boundaries[pti, ptj, ptk]
                    local_slot += 1

        entries_written_per_i_slice[ii] = local_slot

    # Sequential pack step
    # Count sparse (interior-interior) entries only, to pre-allocate output arrays.
    slice_sparse_counts = np.zeros(cell_count_x, dtype=np.int32)
    for ii in range(cell_count_x):
        count = 0
        for slot in range(entries_written_per_i_slice[ii]):
            if not thread_is_dirichlet[ii, slot] and not thread_is_neumann[ii, slot]:
                count += 1
        slice_sparse_counts[ii] = count

    slice_start_offsets = np.zeros(cell_count_x + 1, dtype=np.int32)
    for ii in range(cell_count_x):
        slice_start_offsets[ii + 1] = slice_start_offsets[ii] + slice_sparse_counts[ii]
    total_sparse_entries = slice_start_offsets[cell_count_x]

    sparse_row_indices = np.empty(total_sparse_entries, dtype=np.int32)
    sparse_col_indices = np.empty(total_sparse_entries, dtype=np.int32)
    sparse_off_diagonal_values = np.empty(total_sparse_entries, dtype=dtype)
    diagonal_additions = np.zeros(cell_count, dtype=dtype)
    rhs_additions = np.zeros(cell_count, dtype=dtype)

    for ii in range(cell_count_x):
        out = slice_start_offsets[ii]
        slot = 0
        while slot < entries_written_per_i_slice[ii]:
            if thread_is_dirichlet[ii, slot]:
                # Dirichlet BC:
                #   diagonal += T  (adds resistance term)
                #   rhs += T * p_bc + rhs_term (known pressure drives flow)
                owner = thread_owner_cell[ii, slot]
                transmissibility = thread_transmissibility[ii, slot]
                diagonal_additions[owner] += transmissibility
                rhs_additions[owner] += (
                    transmissibility * thread_bc_pressure[ii, slot]
                    + thread_rhs_term[ii, slot]
                )
                slot += 1
            elif thread_is_neumann[ii, slot]:
                # Neumann BC:
                #   rhs += q_bc (known flux, ft³/day, applied directly)
                # No diagonal or off-diagonal contribution.
                owner = thread_owner_cell[ii, slot]
                rhs_additions[owner] += thread_bc_flux[ii, slot]
                slot += 1
            else:
                # Interior-interior pair: two consecutive slots share the same face.
                this_cell = thread_sparse_row_indices[ii, slot]
                neighbour_cell = thread_sparse_col_indices[ii, slot]
                transmissibility = thread_transmissibility[ii, slot]
                rhs_term = thread_rhs_term[ii, slot]

                sparse_row_indices[out] = this_cell
                sparse_col_indices[out] = neighbour_cell
                sparse_off_diagonal_values[out] = thread_sparse_off_diag_vals[ii, slot]

                sparse_row_indices[out + 1] = neighbour_cell
                sparse_col_indices[out + 1] = this_cell
                sparse_off_diagonal_values[out + 1] = thread_sparse_off_diag_vals[
                    ii, slot + 1
                ]

                # Both cells share the transmissibility on their diagonal.
                diagonal_additions[this_cell] += transmissibility
                diagonal_additions[neighbour_cell] += transmissibility
                # Equal and opposite capillary+gravity contributions to RHS.
                rhs_additions[this_cell] += rhs_term
                rhs_additions[neighbour_cell] -= rhs_term

                out += 2
                slot += 2

    return (
        sparse_row_indices,
        sparse_col_indices,
        sparse_off_diagonal_values,
        diagonal_additions,
        rhs_additions,
    )


@numba.njit(cache=True, inline="always")
def compute_fluxes_from_neighbour(
    cell_indices: ThreeDimensions,
    neighbour_indices: ThreeDimensions,
    face_transmissibility: float,
    water_relative_mobility_grid: ThreeDimensionalGrid,
    oil_relative_mobility_grid: ThreeDimensionalGrid,
    gas_relative_mobility_grid: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    gravitational_constant: float,
    md_per_cp_to_ft2_per_psi_per_day: float,
) -> typing.Tuple[float, float, float]:
    """
    Computes and returns a tuple of the total transmissibility of the phases, the capillary flux,
    and the gravity flux from the neighbour to the current cell.

    :param cell_indices: Indices of the current cell (i, j, k)
    :param neighbour_indices: Indices of the neighbouring cell (i±1, j, k) or (i, j±1, k) or (i, j, k±1)
    :param face_transmissibility: Transmissibility of the face between the current cell and the neighbour (ft³/psi.day)
    :param water_relative_mobility_grid: 3D grid of water mobilities (ft²/psi.day)
    :param oil_relative_mobility_grid: 3D grid of oil mobilities (ft²/psi.day)
    :param gas_relative_mobility_grid: 3D grid of gas mobilities (ft²/psi.day)
    :param oil_water_capillary_pressure_grid: 3D grid of oil-water capillary pressures (psi)
    :param gas_oil_capillary_pressure_grid: 3D grid of gas-oil capillary pressures (psi)
    :param oil_density_grid: 3D grid of oil densities (lb/ft³)
    :param water_density_grid: 3D grid of water densities (lb/ft³)
    :param gas_density_grid: 3D grid of gas densities (lb/ft³)
    :param elevation_grid: 3D grid of elevations (ft)
    :return: A tuple containing:
        - Total transmissibility (ft³/psi.day)
        - Total capillary flux (ft³/day)
        - Total gravity flux (ft³/day)
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
    # Determine the average densities for each phase across the face
    average_water_density = (
        water_density_grid[neighbour_indices] + water_density_grid[cell_indices]
    ) * 0.5
    average_oil_density = (
        oil_density_grid[neighbour_indices] + oil_density_grid[cell_indices]
    ) * 0.5
    average_gas_density = (
        gas_density_grid[neighbour_indices] + gas_density_grid[cell_indices]
    ) * 0.5

    # Calculate harmonic relative mobilities for each phase across the face
    water_harmonic_relative_mobility = compute_harmonic_mean(
        water_relative_mobility_grid[neighbour_indices],
        water_relative_mobility_grid[cell_indices],
    )
    oil_harmonic_relative_mobility = compute_harmonic_mean(
        oil_relative_mobility_grid[neighbour_indices],
        oil_relative_mobility_grid[cell_indices],
    )
    gas_harmonic_relative_mobility = compute_harmonic_mean(
        gas_relative_mobility_grid[neighbour_indices],
        gas_relative_mobility_grid[cell_indices],
    )
    total_harmonic_relative_mobility = (
        water_harmonic_relative_mobility
        + oil_harmonic_relative_mobility
        + gas_harmonic_relative_mobility
    )
    if total_harmonic_relative_mobility <= 0.0:
        # No flow can occur if there is no mobility
        return 0.0, 0.0, 0.0

    # λ_w * T_face * (P_cow_{n+1} - P_cow_{n}) (ft³/psi.day * psi = ft³/day)
    water_capillary_flux = (
        water_harmonic_relative_mobility
        * face_transmissibility
        * oil_water_capillary_pressure_difference
        * md_per_cp_to_ft2_per_psi_per_day
    )
    # λ_g * T_face * (P_cgo_{n+1} - P_cgo_{n}) (ft³/psi.day * psi = ft³/day)
    gas_capillary_flux = (
        gas_harmonic_relative_mobility
        * face_transmissibility
        * gas_oil_capillary_pressure_difference
        * md_per_cp_to_ft2_per_psi_per_day
    )
    # Total capillary flux from the neighbour (ft³/day)
    total_capillary_flux = water_capillary_flux + gas_capillary_flux

    # Calculate the phase gravity potentials (hydrostatic/gravity head)
    water_gravity_potential = (
        average_water_density * gravitational_constant * elevation_difference
    ) / 144.0
    oil_gravity_potential = (
        average_oil_density * gravitational_constant * elevation_difference
    ) / 144.0
    gas_gravity_potential = (
        average_gas_density * gravitational_constant * elevation_difference
    ) / 144.0
    # Total gravity flux (ft³/day)
    water_gravity_flux = (
        water_harmonic_relative_mobility
        * face_transmissibility
        * water_gravity_potential
        * md_per_cp_to_ft2_per_psi_per_day
    )
    oil_gravity_flux = (
        oil_harmonic_relative_mobility
        * face_transmissibility
        * oil_gravity_potential
        * md_per_cp_to_ft2_per_psi_per_day
    )
    gas_gravity_flux = (
        gas_harmonic_relative_mobility
        * face_transmissibility
        * gas_gravity_potential
        * md_per_cp_to_ft2_per_psi_per_day
    )
    total_gravity_flux = water_gravity_flux + oil_gravity_flux + gas_gravity_flux
    total_transmissibility = (
        total_harmonic_relative_mobility
        * face_transmissibility
        * md_per_cp_to_ft2_per_psi_per_day
    )
    return (total_transmissibility, total_capillary_flux, total_gravity_flux)


def compute_well_contributions(
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
    dtype: npt.DTypeLike,
    injection_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    production_bhps: PhaseTensorsProxy[float, ThreeDimensions],
) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Compute well contributions to the implicit pressure linear system A·p = b
    and return them as COO-style flat arrays.

    Wells contribute only to the diagonal of A and to b. The semi-implicit
    Robin boundary condition for both injection and production takes the form:

        A[cell, cell] += PI_phase (productivity index adds to diagonal)
        b[cell] += PI_phase * BHP (BHP-weighted PI adds to RHS)

    Perforations where BHP is non-finite are silently skipped. They contribute
    zero to both arrays, which is equivalent to closing that perforation for
    this time step.

    :param cell_count_x: Number of cells in x-direction
    :param cell_count_y: Number of cells in y-direction
    :param cell_count_z: Number of cells in z-direction
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
    :param dtype: Desired dtype for output arrays (e.g. np.float32 or np.float64)
    :param injection_bhps: Proxy to write back computed injection BHPs
    :param production_bhps: Proxy to write back computed production BHPs
    :return: Four flat arrays ready for scatter-add into the final diagonal and RHS before COO matrix construction.
    """
    water_bubble_point_pressure_grid = fluid_properties.water_bubble_point_pressure_grid
    gas_formation_volume_factor_grid = fluid_properties.gas_formation_volume_factor_grid
    gas_solubility_in_water_grid = fluid_properties.gas_solubility_in_water_grid
    water_fvf_grid = fluid_properties.water_formation_volume_factor_grid
    oil_fvf_grid = fluid_properties.oil_formation_volume_factor_grid

    diagonal_dict: dict[int, float] = {}
    rhs_dict: dict[int, float] = {}

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

        diagonal_dict[cell_1d_index] = (
            diagonal_dict.get(cell_1d_index, 0.0) + productivity_index
        )
        rhs_dict[cell_1d_index] = (
            rhs_dict.get(cell_1d_index, 0.0) + productivity_index * effective_bhp
        )

    for well in wells.injection_wells:
        if not well.is_open or well.injected_fluid is None:
            continue

        injected_fluid = well.injected_fluid
        injected_phase = injected_fluid.phase
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
                compressibility_kwargs: dict = {}
            else:
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
            total_mobility = typing.cast(
                float,
                water_relative_mobility_grid[i, j, k]
                + oil_relative_mobility_grid[i, j, k]
                + gas_relative_mobility_grid[i, j, k],
            )
            effective_bhp = well.get_bottom_hole_pressure(
                pressure=cell_pressure,
                temperature=cell_temperature,
                phase_mobility=total_mobility,
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
                    "Non-finite BHP for injection well %r "
                    "at cell (%d,%d,%d): %s. Skipping perforation.",
                    well.name,
                    i,
                    j,
                    k,
                    effective_bhp,
                )
                continue

            if abs(effective_bhp - cell_pressure) > 1e6:
                logger.warning(
                    "Extreme BHP for injection well %r "
                    "at cell (%d, %d, %d): %.2e psi "
                    "(reservoir pressure: %.1f psi).",
                    well.name,
                    i,
                    j,
                    k,
                    effective_bhp,
                    cell_pressure,
                )

            if cell_pressure > effective_bhp and config.warn_well_anomalies:
                _warn_injection_pressure(
                    bhp=effective_bhp,
                    cell_pressure=cell_pressure,
                    well_name=well.name,
                    time=time,
                    cell=(i, j, k),
                )

            productivity_index = (
                well_index * total_mobility * md_per_cp_to_ft2_per_psi_per_day
            )
            _add_bhp_contribution(cell_1d_index, productivity_index, effective_bhp)

            if injected_phase == FluidPhase.GAS:
                injection_bhps[i, j, k] = (np.nan, np.nan, effective_bhp)
            else:
                injection_bhps[i, j, k] = (effective_bhp, np.nan, np.nan)

    for well in wells.production_wells:
        if not well.is_open:
            continue

        well_indices = well_indices_cache.production[well.name]
        is_couple_controlled = isinstance(well.control, CoupledRateControl)
        primary_phase: FluidPhase = FluidPhase(
            well.control.primary_phase  # type: ignore
            if is_couple_controlled
            else well.produced_fluids[0].phase
        )
        primary_fluid = next(
            (fluid for fluid in well.produced_fluids if fluid.phase == primary_phase),
            None,
        )
        if primary_fluid is None:
            logger.error(
                "No produced fluid found for controlling phase %r in well %r. "
                "Skipping well.",
                primary_phase,
                well.name,
            )
            continue

        for perforation_index in well_indices:
            i, j, k = perforation_index.cell
            cell_1d_index = perforation_index.cell_1d_index
            well_index = perforation_index.well_index
            allocation_fraction = well_indices.allocation_fraction(perforation_index)
            cell_pressure = typing.cast(float, current_oil_pressure_grid[i, j, k])
            cell_temperature = typing.cast(float, temperature_grid[i, j, k])

            context = {}
            if is_couple_controlled:
                context = well.control.build_primary_phase_context(  # type: ignore
                    produced_fluids=well.produced_fluids,
                    oil_mobility=oil_relative_mobility_grid[i, j, k],
                    water_mobility=water_relative_mobility_grid[i, j, k],
                    gas_mobility=gas_relative_mobility_grid[i, j, k],
                    oil_fvf=oil_fvf_grid[i, j, k],
                    water_fvf=water_fvf_grid[i, j, k],
                    gas_fvf=gas_formation_volume_factor_grid[i, j, k],
                    oil_compressibility=oil_compressibility_grid[i, j, k],
                    water_compressibility=water_compressibility_grid[i, j, k],
                    gas_compressibility=gas_compressibility_grid[i, j, k],
                )

            if primary_phase == FluidPhase.GAS:
                phase_compressibility = typing.cast(
                    float, gas_compressibility_grid[i, j, k]
                )
                phase_fvf = typing.cast(
                    float, gas_formation_volume_factor_grid[i, j, k]
                )
                use_pseudo_pressure = config.use_pseudo_pressure
            elif primary_phase == FluidPhase.WATER:
                phase_compressibility = typing.cast(
                    float, water_compressibility_grid[i, j, k]
                )
                phase_fvf = typing.cast(float, water_fvf_grid[i, j, k])
                use_pseudo_pressure = False
            else:  # Oil
                phase_compressibility = typing.cast(
                    float, oil_compressibility_grid[i, j, k]
                )
                phase_fvf = typing.cast(float, oil_fvf_grid[i, j, k])
                use_pseudo_pressure = False

            total_mobility = typing.cast(
                float,
                water_relative_mobility_grid[i, j, k]
                + oil_relative_mobility_grid[i, j, k]
                + gas_relative_mobility_grid[i, j, k],
            )
            effective_bhp = well.get_bottom_hole_pressure(
                pressure=cell_pressure,
                temperature=cell_temperature,
                phase_mobility=total_mobility,
                well_index=well_index,
                fluid=primary_fluid,
                formation_volume_factor=phase_fvf,
                allocation_fraction=allocation_fraction,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=phase_compressibility,
                pvt_tables=config.pvt_tables,
                **context,
            )
            if not np.isfinite(effective_bhp):
                logger.error(
                    "Non-finite BHP for production well %r "
                    "at cell (%d,%d,%d): %s. Skipping perforation.",
                    well.name,
                    i,
                    j,
                    k,
                    effective_bhp,
                )
                continue

            if abs(effective_bhp - cell_pressure) > 1e6:
                logger.warning(
                    "Extreme BHP for production well %r "
                    "at cell (%d,%d,%d): %.2e psi "
                    "(reservoir pressure: %.1f psi).",
                    well.name,
                    i,
                    j,
                    k,
                    effective_bhp,
                    cell_pressure,
                )

            if cell_pressure < effective_bhp and config.warn_well_anomalies:
                _warn_production_pressure(
                    bhp=effective_bhp,
                    cell_pressure=cell_pressure,
                    well_name=well.name,
                    time=time,
                    cell=(i, j, k),
                )

            if effective_bhp <= 0.0:
                logger.warning(
                    "Controlling phase %s BHP is zero for well %r at cell (%d,%d,%d). "
                    "Controlling phase may not be in produced_fluids. Skipping Robin term.",
                    primary_phase,
                    well.name,
                    i,
                    j,
                    k,
                )
                # Skip Jacobian contribution
            else:
                productivity_index = (
                    well_index * total_mobility * md_per_cp_to_ft2_per_psi_per_day
                )
                _add_bhp_contribution(cell_1d_index, productivity_index, effective_bhp)

            gas_bhp = (
                effective_bhp if FluidPhase.GAS in well.produced_phases else np.nan
            )
            water_bhp = (
                effective_bhp if FluidPhase.WATER in well.produced_phases else np.nan
            )
            oil_bhp = (
                effective_bhp if FluidPhase.OIL in well.produced_phases else np.nan
            )
            production_bhps[i, j, k] = (water_bhp, oil_bhp, gas_bhp)
    return (
        np.array(list(diagonal_dict.keys()), dtype=np.int32),
        np.array(list(diagonal_dict.values()), dtype=dtype),
        np.array(list(rhs_dict.keys()), dtype=np.int32),
        np.array(list(rhs_dict.values()), dtype=dtype),
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
            [λ_w * (average_ρ_w * g/gc * ∆elevation) / 144]
            + [λ_o * (average_ρ_o * g/gc * ∆elevation) / 144]
            + [λ_g * (average_ρ_g * g/gc * ∆elevation) / 144]
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

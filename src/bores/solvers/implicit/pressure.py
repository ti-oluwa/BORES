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
from bores.errors import PreconditionerError, SolverError
from bores.grids.base import CapillaryPressureGrids, RelativeMobilityGrids
from bores.grids.pvt import build_total_fluid_compressibility_grid
from bores.models import FluidProperties, RockProperties
from bores.solvers.base import Solution, solve_linear_system, to_1D_index
from bores.transmissibility import FaceTransmissibilities
from bores.types import ThreeDimensionalGrid, ThreeDimensions

logger = logging.getLogger(__name__)


@attrs.frozen
class ImplicitPressureSolution:
    pressure_grid: ThreeDimensionalGrid
    maximum_pressure_change: float


def solve_pressure(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step: int,
    time_step_size: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    face_transmissibilities: FaceTransmissibilities,
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
    config: Config,
    net_well_rate_grid: typing.Optional[ThreeDimensionalGrid] = None,
    dtype: npt.DTypeLike = np.float64,
) -> Solution[ImplicitPressureSolution, None]:
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
    :param config: `Config` object containing simulation config
    :return: `Solution` containing the new pressure grid and scheme used
    """
    porosity_grid = rock_properties.porosity_grid
    net_to_gross_grid = rock_properties.net_to_gross_grid
    rock_compressibility = rock_properties.compressibility
    oil_density_grid = fluid_properties.oil_effective_density_grid
    water_density_grid = fluid_properties.water_density_grid
    gas_density_grid = fluid_properties.gas_density_grid

    current_pressure_grid = fluid_properties.pressure_grid
    water_saturation_grid = fluid_properties.water_saturation_grid
    oil_saturation_grid = fluid_properties.oil_saturation_grid
    gas_saturation_grid = fluid_properties.gas_saturation_grid

    water_compressibility_grid = fluid_properties.water_compressibility_grid
    oil_compressibility_grid = fluid_properties.oil_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid

    # Determine grid dimensions and cell sizes
    cell_count_x, cell_count_y, cell_count_z = current_pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension

    # Compute total fluid system compressibility for each cell
    total_fluid_compressibility_grid = build_total_fluid_compressibility_grid(
        oil_saturation_grid=oil_saturation_grid,
        oil_compressibility_grid=oil_compressibility_grid,
        water_saturation_grid=water_saturation_grid,
        water_compressibility_grid=water_compressibility_grid,
        gas_saturation_grid=gas_saturation_grid,
        gas_compressibility_grid=gas_compressibility_grid,
    )
    total_compressibility_grid = np.add(
        total_fluid_compressibility_grid, rock_compressibility, dtype=dtype
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

    diagonal_values, rhs_values = compute_accumulation_contributions(
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        porosity_grid=porosity_grid,
        net_to_gross_grid=net_to_gross_grid,
        total_compressibility_grid=total_compressibility_grid,
        current_pressure_grid=current_pressure_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        time_step_size_in_days=time_step_size_in_days,
        dtype=dtype,
    )
    (
        sparse_rows,
        sparse_cols,
        sparse_off_diagonal,
        diagonal_additions,
        rhs_additions,
    ) = assemble_flux_contributions(
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
    if net_well_rate_grid is not None:
        well_rhs_additions = assemble_well_contributions(
            net_well_rate_grid=net_well_rate_grid,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            dtype=dtype,
        )
    else:
        well_rhs_additions = 0

    # Merge into final diagonal and b.
    diagonal = diagonal_values + diagonal_additions
    residual_vector = rhs_values + rhs_additions + well_rhs_additions

    cell_count = cell_count_x * cell_count_y * cell_count_z
    diagional_indices = np.arange(cell_count, dtype=np.int32)
    jacobian = coo_matrix(  # type: ignore
        (
            np.concatenate([diagonal, sparse_off_diagonal]),
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
        pressure_vector, _ = solve_linear_system(
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
        return Solution(
            value=ImplicitPressureSolution(
                pressure_grid=current_pressure_grid.astype(dtype, copy=False),
                maximum_pressure_change=0.0,  # No change since solve failed
            ),
            success=False,
            scheme="implicit",
            message=str(exc),
        )

    # Map solution back to 3D grid
    new_pressure_grid = pressure_vector.reshape(current_pressure_grid.shape)
    maximum_pressure_change = np.max(np.abs(new_pressure_grid - current_pressure_grid))
    return Solution(
        value=ImplicitPressureSolution(
            pressure_grid=new_pressure_grid.astype(dtype, copy=False),  # type: ignore[arg-type]
            maximum_pressure_change=maximum_pressure_change,
        ),
        success=True,
        scheme="implicit",
        message=f"Implicit pressure evolution for time step {time_step} successful.",
    )


@numba.njit(parallel=True, cache=True)
def compute_accumulation_contributions(
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    net_to_gross_grid: ThreeDimensionalGrid,
    total_compressibility_grid: ThreeDimensionalGrid,
    current_pressure_grid: ThreeDimensionalGrid,
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
    :param current_pressure_grid: Current oil pressure grid (psi)
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

                # Accumulation term coefficient
                accumulation_coefficient = (
                    cell_size_x
                    * cell_size_y
                    * thickness_grid[i, j, k]
                    * net_to_gross_grid[i, j, k]
                    * porosity_grid[i, j, k]
                    * total_compressibility_grid[i, j, k]
                ) / time_step_size_in_days

                diagonal_values[cell_1D_index] = accumulation_coefficient
                rhs_values[cell_1D_index] = (
                    accumulation_coefficient * current_pressure_grid[i, j, k]
                )

    return diagonal_values, rhs_values


@numba.njit(parallel=True, cache=True)
def assemble_flux_contributions(
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
    :param current_pressure_grid: Current oil pressure grid (psi), shape (nx, ny, nz)
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
    thread_pressure_bc = np.zeros((cell_count_x, max_entries_per_i_slice), dtype=dtype)

    # For Neumann singletons: the known boundary flux (ft³/day), applied directly to RHS.
    thread_flux_bc = np.zeros((cell_count_x, max_entries_per_i_slice), dtype=dtype)

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
                    T, capillary_flux, gravity_flux = compute_face_fluxes(
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
                    thread_sparse_off_diag_vals[ii, local_slot] = -T
                    thread_transmissibility[ii, local_slot] = T
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    thread_is_dirichlet[ii, local_slot] = False
                    thread_is_neumann[ii, local_slot] = False
                    local_slot += 1
                    # Entry 1: A[east, this]
                    thread_sparse_row_indices[ii, local_slot] = east_1d
                    thread_sparse_col_indices[ii, local_slot] = this_cell_1d_index
                    thread_sparse_off_diag_vals[ii, local_slot] = -T
                    thread_transmissibility[ii, local_slot] = T
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    thread_is_dirichlet[ii, local_slot] = False
                    thread_is_neumann[ii, local_slot] = False
                    local_slot += 1
                else:
                    # Out-of-bounds: look up ghost-cell 1D index in padded space.
                    # Ghost cell lives at (ei+1, ej+1, ek+1) in padded grid.
                    pei, pej, pek = ei + 1, ej + 1, ek + 1
                    T, capillary_flux, gravity_flux = compute_face_fluxes(
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
                    thread_owner_cell[ii, local_slot] = this_cell_1d_index
                    thread_transmissibility[ii, local_slot] = T
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    pressure_boundary = pressure_boundaries[pei, pej, pek]
                    if not np.isnan(pressure_boundary):
                        thread_is_dirichlet[ii, local_slot] = True
                        thread_is_neumann[ii, local_slot] = False
                        thread_pressure_bc[ii, local_slot] = pressure_boundary
                    else:
                        # Neumann: flux_boundaries[ghost_1d] is the known flux (ft³/day).
                        thread_is_dirichlet[ii, local_slot] = False
                        thread_is_neumann[ii, local_slot] = True
                        thread_flux_bc[ii, local_slot] = flux_boundaries[pei, pej, pek]
                    local_slot += 1

                # WEST FACE (i-1, j, k) — boundary-only
                # Only written when neighbour is out-of-bounds (boundary face).
                # Interior west faces are handled as the east face of the western cell.
                wi, wj, wk = i - 1, j, k
                if wi < 0:
                    pwi, pwj, pwk = wi + 1, wj + 1, wk + 1
                    T, capillary_flux, gravity_flux = compute_face_fluxes(
                        cell_indices=(i, j, k),
                        neighbour_indices=(0, wj, wk),  # clamp to first real cell
                        water_relative_mobility_grid=water_relative_mobility_grid,
                        oil_relative_mobility_grid=oil_relative_mobility_grid,
                        gas_relative_mobility_grid=gas_relative_mobility_grid,
                        face_transmissibility=face_transmissibilities_x[pwi, pwj, pwk],
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                    )
                    thread_owner_cell[ii, local_slot] = this_cell_1d_index
                    thread_transmissibility[ii, local_slot] = T
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    pressure_boundary = pressure_boundaries[pwi, pwj, pwk]
                    if not np.isnan(pressure_boundary):
                        thread_is_dirichlet[ii, local_slot] = True
                        thread_is_neumann[ii, local_slot] = False
                        thread_pressure_bc[ii, local_slot] = pressure_boundary
                    else:
                        thread_is_dirichlet[ii, local_slot] = False
                        thread_is_neumann[ii, local_slot] = True
                        thread_flux_bc[ii, local_slot] = flux_boundaries[pwi, pwj, pwk]
                    local_slot += 1

                # SOUTH FACE (i, j+1, k)
                si, sj, sk = i, j + 1, k
                if sj < cell_count_y:
                    T, capillary_flux, gravity_flux = compute_face_fluxes(
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
                    thread_sparse_off_diag_vals[ii, local_slot] = -T
                    thread_transmissibility[ii, local_slot] = T
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    thread_is_dirichlet[ii, local_slot] = False
                    thread_is_neumann[ii, local_slot] = False
                    local_slot += 1
                    thread_sparse_row_indices[ii, local_slot] = south_1d
                    thread_sparse_col_indices[ii, local_slot] = this_cell_1d_index
                    thread_sparse_off_diag_vals[ii, local_slot] = -T
                    thread_transmissibility[ii, local_slot] = T
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    thread_is_dirichlet[ii, local_slot] = False
                    thread_is_neumann[ii, local_slot] = False
                    local_slot += 1
                else:
                    psi, psj, psk = si + 1, sj + 1, sk + 1
                    T, capillary_flux, gravity_flux = compute_face_fluxes(
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
                    thread_owner_cell[ii, local_slot] = this_cell_1d_index
                    thread_transmissibility[ii, local_slot] = T
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    pressure_boundary = pressure_boundaries[psi, psj, psk]
                    if not np.isnan(pressure_boundary):
                        thread_is_dirichlet[ii, local_slot] = True
                        thread_is_neumann[ii, local_slot] = False
                        thread_pressure_bc[ii, local_slot] = pressure_boundary
                    else:
                        thread_is_dirichlet[ii, local_slot] = False
                        thread_is_neumann[ii, local_slot] = True
                        thread_flux_bc[ii, local_slot] = flux_boundaries[psi, psj, psk]
                    local_slot += 1

                # NORTH FACE (i, j-1, k) — boundary-only
                ni, nj, nk = i, j - 1, k
                if nj < 0:
                    pni, pnj, pnk = ni + 1, nj + 1, nk + 1
                    T, capillary_flux, gravity_flux = compute_face_fluxes(
                        cell_indices=(i, j, k),
                        neighbour_indices=(ni, 0, nk),
                        water_relative_mobility_grid=water_relative_mobility_grid,
                        oil_relative_mobility_grid=oil_relative_mobility_grid,
                        gas_relative_mobility_grid=gas_relative_mobility_grid,
                        face_transmissibility=face_transmissibilities_y[pni, pnj, pnk],
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                    )
                    thread_owner_cell[ii, local_slot] = this_cell_1d_index
                    thread_transmissibility[ii, local_slot] = T
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    pressure_boundary = pressure_boundaries[pni, pnj, pnk]
                    if not np.isnan(pressure_boundary):
                        thread_is_dirichlet[ii, local_slot] = True
                        thread_is_neumann[ii, local_slot] = False
                        thread_pressure_bc[ii, local_slot] = pressure_boundary
                    else:
                        thread_is_dirichlet[ii, local_slot] = False
                        thread_is_neumann[ii, local_slot] = True
                        thread_flux_bc[ii, local_slot] = flux_boundaries[pni, pnj, pnk]
                    local_slot += 1

                # BOTTOM FACE (i, j, k+1)
                bi, bj, bk = i, j, k + 1
                if bk < cell_count_z:
                    T, capillary_flux, gravity_flux = compute_face_fluxes(
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
                    thread_sparse_off_diag_vals[ii, local_slot] = -T
                    thread_transmissibility[ii, local_slot] = T
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    thread_is_dirichlet[ii, local_slot] = False
                    thread_is_neumann[ii, local_slot] = False
                    local_slot += 1
                    thread_sparse_row_indices[ii, local_slot] = bottom_1d
                    thread_sparse_col_indices[ii, local_slot] = this_cell_1d_index
                    thread_sparse_off_diag_vals[ii, local_slot] = -T
                    thread_transmissibility[ii, local_slot] = T
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    thread_is_dirichlet[ii, local_slot] = False
                    thread_is_neumann[ii, local_slot] = False
                    local_slot += 1
                else:
                    pbi, pbj, pbk = bi + 1, bj + 1, bk + 1
                    T, capillary_flux, gravity_flux = compute_face_fluxes(
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
                    thread_owner_cell[ii, local_slot] = this_cell_1d_index
                    thread_transmissibility[ii, local_slot] = T
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    pressure_boundary = pressure_boundaries[pbi, pbj, pbk]
                    if not np.isnan(pressure_boundary):
                        thread_is_dirichlet[ii, local_slot] = True
                        thread_is_neumann[ii, local_slot] = False
                        thread_pressure_bc[ii, local_slot] = pressure_boundary
                    else:
                        thread_is_dirichlet[ii, local_slot] = False
                        thread_is_neumann[ii, local_slot] = True
                        thread_flux_bc[ii, local_slot] = flux_boundaries[pbi, pbj, pbk]
                    local_slot += 1

                # TOP FACE (i, j, k-1) — boundary-only
                ti, tj, tk = i, j, k - 1
                if tk < 0:
                    pti, ptj, ptk = ti + 1, tj + 1, tk + 1
                    T, capillary_flux, gravity_flux = compute_face_fluxes(
                        cell_indices=(i, j, k),
                        neighbour_indices=(ti, tj, 0),
                        water_relative_mobility_grid=water_relative_mobility_grid,
                        oil_relative_mobility_grid=oil_relative_mobility_grid,
                        gas_relative_mobility_grid=gas_relative_mobility_grid,
                        face_transmissibility=face_transmissibilities_z[pti, ptj, ptk],
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                    )
                    thread_owner_cell[ii, local_slot] = this_cell_1d_index
                    thread_transmissibility[ii, local_slot] = T
                    thread_rhs_term[ii, local_slot] = capillary_flux + gravity_flux
                    pressure_boundary = pressure_boundaries[pti, ptj, ptk]
                    if not np.isnan(pressure_boundary):
                        thread_is_dirichlet[ii, local_slot] = True
                        thread_is_neumann[ii, local_slot] = False
                        thread_pressure_bc[ii, local_slot] = pressure_boundary
                    else:
                        thread_is_dirichlet[ii, local_slot] = False
                        thread_is_neumann[ii, local_slot] = True
                        thread_flux_bc[ii, local_slot] = flux_boundaries[pti, ptj, ptk]
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
                T = thread_transmissibility[ii, slot]
                diagonal_additions[owner] += T
                rhs_additions[owner] += (
                    T * thread_pressure_bc[ii, slot] + thread_rhs_term[ii, slot]
                )
                slot += 1
            elif thread_is_neumann[ii, slot]:
                # Neumann BC:
                #   rhs += q_bc (known flux, ft³/day, applied directly)
                # No diagonal or off-diagonal contribution.
                owner = thread_owner_cell[ii, slot]
                rhs_additions[owner] += thread_flux_bc[ii, slot]
                slot += 1
            else:
                # Interior-interior pair: two consecutive slots share the same face.
                this_cell = thread_sparse_row_indices[ii, slot]
                neighbour_cell = thread_sparse_col_indices[ii, slot]
                T = thread_transmissibility[ii, slot]
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
                diagonal_additions[this_cell] += T
                diagonal_additions[neighbour_cell] += T
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
def compute_face_fluxes(
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


@numba.njit(cache=True)
def assemble_well_contributions(
    net_well_rate_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    dtype: npt.DTypeLike,
) -> npt.NDArray:
    """
    Build the RHS well source/sink contributions for the implicit pressure
    linear system from the pre-computed explicit well rate grid.

    Because wells are treated explicitly, their rates are frozen at
    start-of-step values and added directly to the RHS vector `b` with no
    diagonal (Jacobian) contribution. This removes all ambiguity around which
    mobility or pseudo-pressure linearisation to use inside the pressure solve.

    The relationship to the pressure equation accumulation term is:

        (phi * c_t * V / dt) * (P^{n+1} - P^n) = sum_faces(T * lambda * dP) + Q_well

    where `Q_well = net_well_rate_grid[i, j, k]` in ft³/day, already converted
    to the correct sign convention (positive = injection, negative = production).

    :param net_well_rate_grid: Total volumetric well rate per cell (ft³/day), positive
        for injection and negative for production. Produced by `compute_well_rates`.
    :param cell_count_x: Number of cells in the x-direction.
    :param cell_count_y: Number of cells in the y-direction.
    :param cell_count_z: Number of cells in the z-direction.
    :param dtype: NumPy dtype for the output array.
    :return: 1D array of length `cell_count_x * cell_count_y * cell_count_z`
        containing the well RHS contribution for each cell, indexed by the
        standard row-major 1D cell index used throughout the pressure Jacobian.
    """
    cell_count = cell_count_x * cell_count_y * cell_count_z
    well_rhs_values = np.zeros(cell_count, dtype=dtype)

    # Flatten to 1D and scatter only non-zero well cells
    flat = net_well_rate_grid.ravel()
    for idx in range(cell_count):
        rate = flat[idx]
        if rate != 0.0:
            well_rhs_values[idx] = rate
    return well_rhs_values

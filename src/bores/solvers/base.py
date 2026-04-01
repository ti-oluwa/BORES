import logging
import threading
import typing
import warnings

import attrs
import numba  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pyamg  # type: ignore[import-untyped]
from scipy.sparse import (  # type: ignore[import-untyped]
    csr_array,
    csr_matrix,
    diags,
    isspmatrix_csr,
)
from scipy.sparse.linalg import (  # type: ignore[import-untyped]
    LinearOperator,
    bicgstab,
    cg,
    cgs,
    gmres,
    lgmres,
    spilu,
    spsolve,
    tfqmr,
)

from bores.errors import PreconditionerError, SolverError, ValidationError
from bores.precision import get_floating_point_info
from bores.types import (
    Preconditioner,
    PreconditionerFactory,
    Solver,
    SolverFunc,
    T,
    ThreeDimensionalGrid,
    ThreeDimensions,
)

logger = logging.getLogger(__name__)


__all__ = [
    "CachedPreconditionerFactory",
    "EvolutionResult",
    "build_amg_preconditioner",
    "build_block_jacobi_preconditioner",
    "build_cpr_preconditioner",
    "build_diagonal_preconditioner",
    "build_ilu_preconditioner",
    "build_polynomial_preconditioner",
    "from_1D_index_interior_only",
    "get_preconditioner_factory",
    "get_solver_func",
    "list_preconditioner_factories",
    "list_solver_funcs",
    "preconditioner_factory",
    "solve_linear_system",
    "solver_func",
    "to_1D_index_interior_only",
]


def _warn_production_rate(
    production_rate: float,
    well_name: str,
    cell: ThreeDimensions,
    time: float,
    rate_unit: str = "ft³/day",
) -> None:
    """
    Issues a warning if a production well is found to be injecting fluid
    instead of producing it. i.e., if the production rate is positive.
    """
    warnings.warn(
        f"Warning: Production well '{well_name}' at cell {cell} has a positive rate of {production_rate:.4f} {rate_unit}, "
        f"indicating it is no longer producing fluid at {time:.3f} seconds. Production rates should be negative. Please check well configuration.",
        UserWarning,
    )


def _warn_injection_rate(
    injection_rate: float,
    well_name: str,
    cell: ThreeDimensions,
    time: float,
    rate_unit: str = "ft³/day",
) -> None:
    """
    Issues a warning if an injection well is found to be producing fluid
    instead of injecting it. i.e., if the injection rate is negative.
    """
    warnings.warn(
        f"Warning: Injection well '{well_name}' at cell {cell} has a negative rate of {injection_rate:.4f} {rate_unit}, "
        f"indicating it is no longer injecting fluid at {time:.3f} seconds. Injection rates should be postive. Please check well configuration.",
        UserWarning,
    )


def _warn_production_pressure(
    bhp: float,
    well_name: str,
    cell: ThreeDimensions,
    cell_pressure: float,
    time: float,
):
    warnings.warn(
        f"Warning: Production well '{well_name}' at cell {cell} has a high BHP of {bhp:.4f}psi, cell pressure is {cell_pressure:.4f}psi, "
        f"indicating it is no longer producing fluid at {time:.3f} seconds. Production pressure should be lower than reservoir pressure. Please check well configuration.",
        UserWarning,
    )


def _warn_injection_pressure(
    bhp: float,
    well_name: str,
    cell: ThreeDimensions,
    cell_pressure: float,
    time: float,
):
    warnings.warn(
        f"Warning: Injection well '{well_name}' at cell {cell} has a low BHP of {bhp:.4f}psi, cell pressure is {cell_pressure:.4f}psi, "
        f"indicating it is no longer injecting fluid at {time:.3f} seconds. Injection pressure should be higher than reservoir pressure. Please check well configuration.",
        UserWarning,
    )


M = typing.TypeVar("M")


@attrs.frozen
class EvolutionResult(typing.Generic[T, M]):
    """
    Result of a single evolution step in the simulation.
    """

    value: T
    """The result value if successful, otherwise None."""
    scheme: typing.Literal["implicit", "explicit"]
    """The numerical scheme used for the evolution step."""
    success: bool = True
    """Indicates if the evolution step was successful."""
    message: typing.Optional[str] = None
    """A message providing additional information about the result."""
    metadata: typing.Optional[M] = None
    """Optional metadata related to the evolution step."""


@numba.njit(inline="always", parallel=True, cache=True)
def to_1D_index_interior_only(
    i: int,
    j: int,
    k: int,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
) -> int:
    """
    Convert 3D interior cell indices to 1D array index.

    For a grid with dimensions (Nx, Ny, Nz), interior cells are
    indexed from (1, 1, 1) to (Nx-2, Ny-2, Nz-2).

    The 1D index starts at 0 for cell (1, 1, 1).

    Padding cells (i=0, i=Nx-1, etc.) return -1.
    Interior cells are mapped to [0, (Nx-2)*(Ny-2)*(Nz-2))
    """
    if not (
        0 < i < cell_count_x - 1
        and 0 < j < cell_count_y - 1
        and 0 < k < cell_count_z - 1
    ):
        return -1  # Padding cell

    # Adjust indices to 0-based for interior grid
    i_interior = i - 1
    j_interior = j - 1
    k_interior = k - 1

    # Interior dimensions
    ny_interior = cell_count_y - 2
    nz_interior = cell_count_z - 2
    return (
        i_interior * (ny_interior * nz_interior)
        + (j_interior * nz_interior)
        + k_interior
    )


@numba.njit(cache=True, parallel=True, inline="always")
def from_1D_index_interior_only(
    idx: int,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
) -> typing.Tuple[int, int, int]:
    """
    Convert 1D interior cell index back to 3D grid indices.

    This is the inverse of to_1D_index_interior_only.

    For a grid with dimensions (Nx, Ny, Nz), interior cells are
    indexed from (1, 1, 1) to (Nx-2, Ny-2, Nz-2).

    The 1D index starts at 0 for cell (1, 1, 1).

    :param idx: 1D array index (0 to interior_cell_count - 1)
    :param cell_count_x: Total number of cells in x-direction (including boundaries)
    :param cell_count_y: Total number of cells in y-direction (including boundaries)
    :param cell_count_z: Total number of cells in z-direction (including boundaries)
    :return: Tuple of (i, j, k) indices in the full grid (interior cells only)
    """
    # Compute interior dimensions
    interior_ny = cell_count_y - 2
    interior_nz = cell_count_z - 2

    # Reverse the row-major ordering
    # idx = i_interior * (interior_ny * interior_nz) + j_interior * interior_nz + k_interior
    i_interior = idx // (interior_ny * interior_nz)
    remainder = idx % (interior_ny * interior_nz)
    j_interior = remainder // interior_nz
    k_interior = remainder % interior_nz

    # Convert back to full grid coordinates (add 1 to shift from interior to full grid)
    i = i_interior + 1
    j = j_interior + 1
    k = k_interior + 1

    return i, j, k


@numba.njit(parallel=True, cache=True)
def compute_mobility_grids(
    absolute_permeability_x: ThreeDimensionalGrid,
    absolute_permeability_y: ThreeDimensionalGrid,
    absolute_permeability_z: ThreeDimensionalGrid,
    water_relative_mobility_grid: ThreeDimensionalGrid,
    oil_relative_mobility_grid: ThreeDimensionalGrid,
    gas_relative_mobility_grid: ThreeDimensionalGrid,
    md_per_cp_to_ft2_per_psi_per_day: float,
) -> typing.Tuple[
    typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
    typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
    typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
]:
    """
    Compute mobility grids for all three phases in all three directions (x, y, z).

    Mobility = (absolute permeability * relative mobility) * conversion factor

    :param absolute_permeability_x: Absolute permeability in x-direction (mD)
    :param absolute_permeability_y: Absolute permeability in y-direction (mD)
    :param absolute_permeability_z: Absolute permeability in z-direction (mD)
    :param water_relative_mobility_grid: Water relative mobility (1/cP)
    :param oil_relative_mobility_grid: Oil relative mobility (1/cP)
    :param gas_relative_mobility_grid: Gas relative mobility (1/cP)
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion constant
    :return: Tuple of 3 direction tuples, each containing (water, oil, gas) mobility grids:
        (x_mobilities, y_mobilities, z_mobilities) where each is (water, oil, gas)
        All with units (ft²/psi·day)
    """
    # X-direction mobilities
    water_mobility_grid_x = (
        absolute_permeability_x
        * water_relative_mobility_grid
        * md_per_cp_to_ft2_per_psi_per_day
    )
    oil_mobility_grid_x = (
        absolute_permeability_x
        * oil_relative_mobility_grid
        * md_per_cp_to_ft2_per_psi_per_day
    )
    gas_mobility_grid_x = (
        absolute_permeability_x
        * gas_relative_mobility_grid
        * md_per_cp_to_ft2_per_psi_per_day
    )

    # Y-direction mobilities
    water_mobility_grid_y = (
        absolute_permeability_y
        * water_relative_mobility_grid
        * md_per_cp_to_ft2_per_psi_per_day
    )
    oil_mobility_grid_y = (
        absolute_permeability_y
        * oil_relative_mobility_grid
        * md_per_cp_to_ft2_per_psi_per_day
    )
    gas_mobility_grid_y = (
        absolute_permeability_y
        * gas_relative_mobility_grid
        * md_per_cp_to_ft2_per_psi_per_day
    )

    # Z-direction mobilities
    water_mobility_grid_z = (
        absolute_permeability_z
        * water_relative_mobility_grid
        * md_per_cp_to_ft2_per_psi_per_day
    )
    oil_mobility_grid_z = (
        absolute_permeability_z
        * oil_relative_mobility_grid
        * md_per_cp_to_ft2_per_psi_per_day
    )
    gas_mobility_grid_z = (
        absolute_permeability_z
        * gas_relative_mobility_grid
        * md_per_cp_to_ft2_per_psi_per_day
    )

    # Group by direction: (water, oil, gas) for each direction
    x_mobilities = (water_mobility_grid_x, oil_mobility_grid_x, gas_mobility_grid_x)
    y_mobilities = (water_mobility_grid_y, oil_mobility_grid_y, gas_mobility_grid_y)
    z_mobilities = (water_mobility_grid_z, oil_mobility_grid_z, gas_mobility_grid_z)
    return (x_mobilities, y_mobilities, z_mobilities)  # type: ignore[return-value]


@numba.njit(parallel=True, cache=True)
def normalize_saturations(
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    saturation_epsilon: float,
) -> typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid]:
    """
    Clamp saturations values (in-place) to zero and normalize saturations to sum = 1.

    :param water_saturation_grid: 3D grid of water saturations.
    :param oil_saturation_grid: 3D grid of oil saturations.
    :param gas_saturation_grid: 3D grid of gas saturations.
    :param saturation_epsilon: Small threshold to avoid division by zero.
    :return: Tuple of normalized saturation grids (water_sat, oil_sat, gas_sat)
    """
    # Get grid shape
    nx, ny, nz = water_saturation_grid.shape

    # Iterate through all cells (parallelized on outermost loop)
    for i in numba.prange(nx):  # type: ignore
        for j in range(ny):
            for k in range(nz):
                # Clamp negatives to zero
                sw = max(0.0, water_saturation_grid[i, j, k])
                so = max(0.0, oil_saturation_grid[i, j, k])
                sg = max(0.0, gas_saturation_grid[i, j, k])

                total = sw + so + sg

                # Normalize if total > epsilon
                if total > saturation_epsilon:
                    water_saturation_grid[i, j, k] = sw / total
                    oil_saturation_grid[i, j, k] = so / total
                    gas_saturation_grid[i, j, k] = sg / total
                else:
                    # Set to zero if total is too small
                    water_saturation_grid[i, j, k] = 0.0
                    oil_saturation_grid[i, j, k] = 0.0
                    gas_saturation_grid[i, j, k] = 0.0

    return water_saturation_grid, oil_saturation_grid, gas_saturation_grid


def build_amg_preconditioner(
    A_csr: typing.Union[csr_array, csr_matrix], cycle: str = "V", **kwargs: typing.Any
) -> LinearOperator:
    """
    Creates an Algebraic Multigrid (AMG) preconditioner using PyAMG.

    :param A_csr: The coefficient matrix in CSR format.
    :param cycle: Multigrid cycle type ('V', 'W', 'F'). V-cycle is standard,
        W-cycle offers better convergence for difficult problems but is more expensive.
    :param kwargs: Additional arguments for `pyamg.smoothed_aggregation_solver`.
    :return: A SciPy `LinearOperator` that represents the AMG preconditioner.
    """
    ml_solver = pyamg.smoothed_aggregation_solver(A_csr, **kwargs)
    return ml_solver.aspreconditioner(cycle=cycle)


def build_diagonal_preconditioner(
    A_csr: typing.Union[csr_array, csr_matrix],
) -> LinearOperator:
    """
    Creates a diagonal preconditioner from the coefficient matrix.

    :param A_csr: The coefficient matrix in CSR format.
    :return: A SciPy `LinearOperator` that represents the diagonal preconditioner.
    """
    diag_elements = A_csr.diagonal()
    # Avoid division by zero by replacing zeros with a small number
    # Use precision-adaptive threshold for float32/float64 compatibility
    epsilon = get_floating_point_info().eps
    threshold = max(1e-10, 100 * epsilon)
    diag_elements = np.where(np.abs(diag_elements) < threshold, 1.0, diag_elements)
    M_diag = diags(1.0 / diag_elements, format="csr")
    return LinearOperator(shape=A_csr.shape, matvec=M_diag.dot)  # type: ignore[arg-type]


def build_block_jacobi_preconditioner(
    A_csr: typing.Union[csr_array, csr_matrix],
    block_size: int = 3,
) -> LinearOperator:
    """
    Creates a Block Jacobi preconditioner for block-structured systems.

    For black-oil reservoir simulation with 3 variables per cell (pressure, So, Sg),
    this preconditioner is more effective than diagonal for coupled systems.

    :param A_csr: The coefficient matrix in CSR format.
    :param block_size: Size of each block (e.g., 3 for black-oil: pressure, So, Sg).
    :return: A SciPy `LinearOperator` that represents the Block Jacobi preconditioner.
    """
    n = A_csr.shape[0]
    num_blocks = n // block_size

    # Pre-compute inverse of diagonal blocks
    block_inverses = []
    epsilon = get_floating_point_info().eps
    threshold = max(1e-10, 100 * epsilon)

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size

        # Extract diagonal block
        block = A_csr[start_idx:end_idx, start_idx:end_idx].toarray()  # type: ignore

        try:
            # Try to invert the block
            block_inv = np.linalg.inv(block)
            block_inverses.append(block_inv)
        except np.linalg.LinAlgError:
            # Block is singular, fall back to diagonal approximation
            diag = np.diag(block)
            diag = np.where(np.abs(diag) < threshold, 1.0, diag)
            block_inv = np.diag(1.0 / diag)
            block_inverses.append(block_inv)

    # Handle remainder cells (if n is not divisible by block_size)
    remainder_start = num_blocks * block_size
    if remainder_start < n:
        remainder_block = A_csr[remainder_start:n, remainder_start:n].toarray()  # type: ignore
        try:
            remainder_inv = np.linalg.inv(remainder_block)
            block_inverses.append(remainder_inv)
        except np.linalg.LinAlgError:
            diag = np.diag(remainder_block)
            diag = np.where(np.abs(diag) < threshold, 1.0, diag)
            remainder_inv = np.diag(1.0 / diag)
            block_inverses.append(remainder_inv)

    def matvec(x: npt.NDArray) -> npt.NDArray:
        """Apply Block Jacobi preconditioner."""
        y = np.zeros_like(x)

        # Apply block inversions
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            y[start_idx:end_idx] = block_inverses[i] @ x[start_idx:end_idx]

        # Handle remainder
        if remainder_start < n:
            y[remainder_start:n] = block_inverses[num_blocks] @ x[remainder_start:n]
        return y

    return LinearOperator(shape=A_csr.shape, matvec=matvec)  # type: ignore[arg-type]


def build_ilu_preconditioner(
    A_csr: typing.Union[csr_array, csr_matrix], **kwargs: typing.Any
) -> LinearOperator:
    """
    Creates an Incomplete LU (ILU) preconditioner using `spilu`.

    :param A_csr: The coefficient matrix in CSR format. It will be
        converted to CSC for efficiency with `spilu`.
    :return: A SciPy `LinearOperator` that solves the preconditioned system.
    """
    # spilu works most efficiently with the CSC matrix format.
    A_csc = A_csr.tocsc()

    # Compute the Incomplete LU factorization
    # You can tune 'drop_tol' and 'fill_factor' for better performance/accuracy trade-offs
    # spilu returns a SuperLU object which has a .solve() method
    kwargs.setdefault("drop_tol", 1e-4)  # Drop tolerance (controls sparsity/accuracy)
    kwargs.setdefault("fill_factor", 10)  # Memory allocation factor
    ilu_factor = spilu(A_csc, **kwargs)
    # Create a `LinearOperator` that uses the .solve() method as the preconditioning step
    return LinearOperator(shape=A_csc.shape, matvec=ilu_factor.solve)  # type: ignore[arg-type]


def build_polynomial_preconditioner(
    A_csr: typing.Union[csr_array, csr_matrix],
    degree: int = 2,
) -> LinearOperator:
    """
    Creates a polynomial preconditioner M = I + αA + α²A².

    Cheap and effective for well-conditioned systems. Uses Neumann series approximation
    of (I - A)^{-1} for properly scaled A.

    :param A_csr: The coefficient matrix in CSR format.
    :param degree: Polynomial degree (1, 2, or 3 recommended).
    :return: A SciPy `LinearOperator` that represents the polynomial preconditioner.
    """
    # Estimate spectral radius for scaling (simple approximation)
    # Use inverse of diagonal norm as scaling factor
    diag = A_csr.diagonal()
    diag_norm = np.max(np.abs(diag))
    alpha = 1.0 / diag_norm if diag_norm > 1e-10 else 1.0

    # Pre-compute polynomial terms: I, αA, α²A², ...
    identity_term = np.ones(A_csr.shape[0])
    terms = [identity_term]

    current_term = alpha
    A_power = A_csr
    for _ in range(degree):
        terms.append(current_term)  # type: ignore
        current_term *= alpha
        if _ < degree - 1:
            A_power = A_power @ A_csr

    def matvec(x: npt.NDArray) -> npt.NDArray:
        """Apply polynomial preconditioner: (I + αA + α²A² + ...) x"""
        result = terms[0] * x  # Identity term

        A_x = x
        for i in range(1, degree + 1):
            A_x = A_csr @ A_x
            result = result + terms[i] * A_x

        return result

    return LinearOperator(shape=A_csr.shape, matvec=matvec)  # type: ignore[arg-type]


_CPR_AMG_KWARGS = {
    "max_coarse": 500,
    "presmoother": ("gauss_seidel", {"sweep": "symmetric", "iterations": 1}),
    "postsmoother": ("gauss_seidel", {"sweep": "symmetric", "iterations": 1}),
}
"""Default AMG parameters for CPR preconditioner."""
_CPR_ILU_KWARGS = {
    "drop_tol": 1e-4,
    "fill_factor": 10,
}
"""Default ILU parameters for CPR preconditioner."""


def build_cpr_preconditioner(
    A_csr: typing.Union[csr_array, csr_matrix],
    *,
    n_variables_per_cell: int = 3,
    pressure_variable_index: int = 0,
    amg_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ilu_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
) -> LinearOperator:
    """
    Creates a Constrained Pressure Residual (CPR) preconditioner for a fully-implicit
    multiphase reservoir Jacobian.

    CPR is a two-stage preconditioner:

        Stage 1 (global): AMG solve of the pressure-pressure sub-block A_pp.
        Stage 2 (local): ILU solve on the full system to smooth high-frequency errors.

    This preconditioner significantly improves convergence for 3D multiphase
    reservoir simulations where the Jacobian is strongly coupled but the pressure
    equation dominates the long-range behavior.

    :param A_csr: The full Jacobian matrix in CSR format.
    :param n_variables_per_cell: Number of unknowns per grid cell
        (e.g., pressure, saturation_o, saturation_g).
    :param pressure_variable_index: Index of the pressure unknown inside each cell's
        variable block. Example: if block = [pressure, So, Sg], then index = 0.
    :param amg_kwargs: Keyword arguments for `pyamg.smoothed_aggregation_solver`.
    :param ilu_kwargs: Keyword arguments for `scipy.sparse.linalg.spilu`.
    :return: A SciPy `LinearOperator` implementing the CPR preconditioner M such that
        x = M @ r approximately solves J^{-1} r.
    :raises ValidationError: If no pressure DOFs are found.
    :raises RuntimeError: If AMG or ILU construction fails.

    Notes
    -----
    CPR workflow:

    1. Restrict residual to pressure DOFs:      r_p = R * r
    2. Solve pressure block:                    z_p = A_pp^{-1} * r_p   (AMG)
    3. Prolongate correction to full space:     z = P * z_p
    4. Compute remaining residual:              w = r - A * z
    5. Smooth locally:                          y = ILU^{-1} * w
    6. Final CPR output:                        x = z + y
    """
    if not isspmatrix_csr(A_csr):
        A_csr = csr_matrix(A_csr)

    amg_kwargs = amg_kwargs or dict(_CPR_AMG_KWARGS)
    ilu_kwargs = ilu_kwargs or dict(_CPR_ILU_KWARGS)
    number_of_equations = A_csr.shape[0]  # type: ignore
    # Identify the pressure DOFs: [p_i, So_i, Sg_i, ...]
    pressure_dof_indices = np.arange(
        pressure_variable_index,
        number_of_equations,
        n_variables_per_cell,
        dtype=np.int64,
    )
    if pressure_dof_indices.size == 0:
        raise ValidationError(
            "No pressure DOFs found. Check `n_variables_per_cell` or `pressure_variable_index`."
        )

    # Extract the pressure-pressure block (A_pp)
    A_pp = A_csr[pressure_dof_indices, :][:, pressure_dof_indices].tocsr()  # type: ignore

    # Build AMG preconditioner for A_pp
    try:
        M_amg = build_amg_preconditioner(A_pp, **amg_kwargs)  # type: ignore
    except Exception as exc:
        raise PreconditionerError(
            f"AMG construction for pressure block failed: {exc}"
        ) from exc

    # Build ILU preconditioner for the full matrix
    try:
        M_ilu = build_ilu_preconditioner(A_csr, **ilu_kwargs)
    except Exception as exc:
        raise PreconditionerError(f"ILU factorization failed for CPR: {exc}") from exc

    # Restriction and prolongation operators (implicit)
    def restrict_to_pressure(vec_full: npt.NDArray) -> npt.NDArray:
        return vec_full[pressure_dof_indices]

    def prolongate_to_full(vec_pressure: npt.NDArray) -> npt.NDArray:
        out = np.zeros(number_of_equations, dtype=vec_pressure.dtype)
        out[pressure_dof_indices] = vec_pressure
        return out

    def matvec(residual: npt.NDArray) -> npt.NDArray:
        """CPR preconditioner application: x = M^{-1} r"""
        # Stage 1: pressure solve
        r_p = restrict_to_pressure(residual)
        try:
            z_p = M_amg.dot(r_p)
        except TypeError:
            z_p = M_amg(r_p)

        z = prolongate_to_full(z_p)  # type: ignore[arg-type]

        # Stage 2: ILU correction
        w = residual - A_csr.dot(z)
        y = M_ilu.dot(w)
        return z + y

    return LinearOperator(shape=A_csr.shape, matvec=matvec)  # type: ignore[arg-type]


def _spsolve(
    A: typing.Any,
    b: typing.Any,
    x0: typing.Optional[typing.Any],
    *,
    rtol: float,
    atol: float,
    maxiter: typing.Optional[int],
    M: typing.Optional[typing.Any],
    callback: typing.Optional[typing.Callable[[npt.NDArray], None]],
) -> npt.NDArray:
    return spsolve(A, b), 0  # type: ignore[return-value]


def _lgmres(
    A: typing.Any,
    b: typing.Any,
    x0: typing.Optional[typing.Any],
    *,
    rtol: float,
    atol: float,
    maxiter: typing.Optional[int],
    M: typing.Optional[typing.Any],
    callback: typing.Optional[typing.Callable[[npt.NDArray], None]],
    inner_m: int = 50,
    outer_k: int = 5,
) -> typing.Tuple[npt.NDArray, int]:
    """
    LGMRES solver with configurable inner/outer iteration parameters.

    :param inner_m: Number of inner GMRES iterations per restart.
    :param outer_k: Number of vectors to carry between inner GMRES iterations.
    """
    return lgmres(  # type: ignore[return-value]
        A,
        b,
        x0=x0,
        M=M,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter or 1000,
        callback=callback,
        inner_m=inner_m,
        outer_k=outer_k,
    )


class CachedPreconditionerFactory:
    """
    Preconditioner factory

    Caches preconditioner and optionally updates it based on matrix changes.

    For most reservoir simulation, the matrix structure (sparsity pattern) stays constant,
    but coefficients (mobility, compressibility) change slowly. Hence this class:

    1. Caches expensive preconditioner setup (ILU, AMG)
    2. Reuses preconditioner across multiple timesteps
    3. Rebuilds only when matrix changes significantly

    **Usage:**
    ```python
    # Option 1: Use string name (recommended for Config integration)
    cached_ilu = CachedPreconditionerFactory(
        factory="ilu",
        update_frequency=10,  # Rebuild every 10 timesteps
        recompute_threshold=0.3,  # Or when matrix changes by 30%
    )

    # Option 2: Use factory function directly
    cached_ilu = CachedPreconditionerFactory(
        factory=build_ilu_preconditioner,
        update_frequency=10,
        recompute_threshold=0.3,
    )

    # In simulation loop
    for timestep in range(num_steps):
        A = build_matrix(...)  # Matrix changes each step
        M = cached_ilu(A)  # Reuses or rebuilds as needed
        x = solve(A, b, M=M)

    # Integration with simulation `Config`
    pressure_preconditioner = CachedPreconditionerFactory("ilu", name="cached_ilu", update_frequency=10)
    # Register the preconditioner factory as the "new" ILU preconditioner factory
    pressure_preconditioner.register(override=True)
    config = Config(pressure_preconditioner="cached_ilu", ...)
    ```
    """

    def __init__(
        self,
        factory: typing.Union[str, PreconditionerFactory],
        name: typing.Optional[str] = None,
        update_frequency: int = 10,
        recompute_threshold: float = 0.5,
    ):
        """
        Initialize cached preconditioner.

        :param factory: Preconditioner factory function or string name (e.g., "ilu", "amg", "block_jacobi")
        :param name: The name of the preconditioner factory been cached or a new name for the cahed version of the factory.
        :param update_frequency: Rebuild preconditioner every N calls (0 = never auto-rebuild)
        :param recompute_threshold: Rebuild if ||A_new - A_old||/||A_old|| > threshold.
            Basically the absolute relative change must be greater than the threshold for a rebuild.
        """
        if isinstance(factory, str):
            self.factory = get_preconditioner_factory(factory)
            self._name = name or factory
        else:
            self.factory = factory
            self._name = name or getattr(factory, "__name__", repr(factory))

        self.update_frequency = update_frequency
        self.recompute_threshold = recompute_threshold

        self._cached_M: typing.Optional[LinearOperator] = None
        self._cached_A_data: typing.Optional[npt.NDArray] = None
        self._call_count = 0

    @property
    def name(self) -> str:
        return self._name

    def __call__(
        self, A_csr: typing.Union[csr_array, csr_matrix]
    ) -> typing.Optional[LinearOperator]:
        """
        Get preconditioner, reusing cache if possible.

        :param A_csr: Current coefficient matrix
        :return: Preconditioner (cached or newly built)
        """
        should_rebuild = self._should_recompute(A_csr)

        if should_rebuild:
            logger.debug(
                f"Rebuilding {self._name} preconditioner (call #{self._call_count}, "
                f"threshold={self.recompute_threshold:.2%})"
            )
            self._cached_M = self.factory(A_csr)
            self._cached_A_data = A_csr.data.copy()
            self._call_count = 0
        else:
            logger.debug(
                f"Reusing cached {self._name} preconditioner (call #{self._call_count}/"
                f"{self.update_frequency})"
            )

        self._call_count += 1
        return self._cached_M

    def _should_recompute(self, A_csr: typing.Union[csr_array, csr_matrix]) -> bool:
        """Determine if preconditioner should be rebuilt."""
        # First call - must build
        if self._cached_M is None:
            return True

        # Frequency-based rebuild
        if self.update_frequency > 0 and self._call_count >= self.update_frequency:
            return True

        # Change-based rebuild (compare matrix coefficients)
        if self._cached_A_data is not None and self.recompute_threshold > 0:
            # Check if matrix data has changed significantly
            # Use Frobenius norm of difference
            if A_csr.data.shape != self._cached_A_data.shape:
                # Structure changed (shouldn't happen in reservoir sim, but handle it)
                return True

            diff_norm = np.linalg.norm(A_csr.data - self._cached_A_data)
            old_norm = np.linalg.norm(self._cached_A_data)

            if old_norm > 1e-10:
                relative_change = diff_norm / old_norm
                if relative_change > self.recompute_threshold:
                    logger.debug(
                        f"Matrix changed by {relative_change:.2%} "
                        f"(threshold: {self.recompute_threshold:.2%})"
                    )
                    return True

        return False

    def reset(self) -> None:
        """Clear cache and force rebuild on next call."""
        self._cached_M = None
        self._cached_A_data = None
        self._call_count = 0

    def register(self, override: bool = False) -> None:
        """
        Register this cached preconditioner (factory) in the preconditioner faactory registry

        :param override: Whether to override exisiting factories with the same name in the registry
        """
        factory = typing.cast(PreconditionerFactory, self)
        preconditioner_factory(factory, name=self._name, override=override)


_preconditoner_registry_lock = threading.Lock()
_PRECONDITIONER_FACTORIES = {
    "cpr": build_cpr_preconditioner,
    "amg": build_amg_preconditioner,
    "ilu": build_ilu_preconditioner,
    "diagonal": build_diagonal_preconditioner,
    "block_jacobi": build_block_jacobi_preconditioner,
    "polynomial": build_polynomial_preconditioner,
}
"""Registered preconditioner factory functions."""

_solver_registry_lock = threading.Lock()
_SOLVER_FUNCS = {
    "lgmres": _lgmres,
    "bicgstab": bicgstab,
    "tfqmr": tfqmr,
    "gmres": gmres,
    "cg": cg,
    "cgs": cgs,
    "direct": _spsolve,
}
"""Registered solver functions."""


@typing.overload
def preconditioner_factory(func: PreconditionerFactory) -> PreconditionerFactory: ...


@typing.overload
def preconditioner_factory(
    func: None = None,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> typing.Callable[[PreconditionerFactory], PreconditionerFactory]: ...


@typing.overload
def preconditioner_factory(
    func: PreconditionerFactory,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> PreconditionerFactory: ...


def preconditioner_factory(
    func: typing.Optional[PreconditionerFactory] = None,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> typing.Union[
    PreconditionerFactory,
    typing.Callable[[PreconditionerFactory], PreconditionerFactory],
]:
    """
    Decorator to register a preconditioner factory function.

    A preconditioner factory is a callable that takes a CSR matrix and returns
    a SciPy `LinearOperator` representing the preconditioner. It simply builds
    the preconditioner when given the system matrix.

    :param func: The preconditioner factory function to decorate.
    :param name: Optional name to register the preconditioner under. If not provided,
        the function's `__name__` attribute is used.
    :param override: If True, allows overriding an existing preconditioner factory
    :return: The original function, unmodified.
    """

    def decorator(func: PreconditionerFactory) -> PreconditionerFactory:
        with _preconditoner_registry_lock:
            key = name or getattr(func, "__name__", None)
            if not key:
                raise ValueError(
                    "Preconditioner factory  must have a `__name__` attribute or a name must be provided."
                )

            if not override and key in _PRECONDITIONER_FACTORIES:
                raise ValueError(
                    f"Preconditioner factory '{name}' is already registered. "
                    f"Use `override=True` to replace it."
                )
            _PRECONDITIONER_FACTORIES[key] = func  # type: ignore
        return func

    if func is not None:
        return decorator(func)
    return decorator


def list_preconditioner_factories() -> typing.List[str]:
    """
    List the names of all registered preconditioner factories.

    :return: List of registered preconditioner factory names.
    """
    with _preconditoner_registry_lock:
        return list(_PRECONDITIONER_FACTORIES.keys())


def get_preconditioner_factory(name: str) -> PreconditionerFactory:
    """
    Get a registered preconditioner factory by name.

    :param name: Name of the preconditioner factory.
    :return: The corresponding preconditioner factory function.
    :raises ValidationError: If the preconditioner factory is unknown.
    """
    with _preconditoner_registry_lock:
        if name not in _PRECONDITIONER_FACTORIES:
            raise ValidationError(
                f"Unknown preconditioner factory: {name!r}. "
                f"Use `@preconditioner_factory` to register new preconditioners. "
                f"Available preconditioners: {list(_PRECONDITIONER_FACTORIES.keys())}"
            )
        return _PRECONDITIONER_FACTORIES[name]  # type: ignore[return-value]


@typing.overload
def solver_func(func: SolverFunc) -> SolverFunc: ...


@typing.overload
def solver_func(
    func: None = None,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> typing.Callable[[SolverFunc], SolverFunc]: ...


@typing.overload
def solver_func(
    func: SolverFunc,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> SolverFunc: ...


def solver_func(
    func: typing.Optional[SolverFunc] = None,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> typing.Union[
    SolverFunc,
    typing.Callable[[SolverFunc], SolverFunc],
]:
    """
    Decorator to register a solver function.

    A solver function is a callable that implements an iterative solver
    interface compatible with SciPy's sparse linear algebra solvers.

    :param func: The solver function to decorate.
    :param name: Optional name to register the solver under. If not provided,
        the function's `__name__` attribute is used.
    :param override: If True, allows overriding an existing solver function.
    :return: The original function, unmodified.
    """

    def decorator(func: SolverFunc) -> SolverFunc:
        with _solver_registry_lock:
            key = name or getattr(func, "__name__", None)
            if not key:
                raise ValueError(
                    "Solver function  must have a `__name__` attribute or a name must be provided."
                )

            if not override and key in _SOLVER_FUNCS:
                raise ValueError(
                    f"Solver function '{name}' is already registered. "
                    f"Use `override=True` to replace it."
                )
            _SOLVER_FUNCS[key] = func
        return func

    if func is not None:
        return decorator(func)
    return decorator


def list_solver_funcs() -> typing.List[str]:
    """
    List the names of all registered solver functions.

    :return: List of registered solver function names.
    """
    with _solver_registry_lock:
        return list(_SOLVER_FUNCS.keys())


def get_solver_func(name: str) -> typing.Optional[SolverFunc]:
    """
    Get a registered solver function by name.

    :param name: Name of the solver function.
    :return: The corresponding solver function.
    """
    with _solver_registry_lock:
        if name not in _SOLVER_FUNCS:
            raise ValidationError(
                f"Unknown solver function: {name!r}. "
                f"Use `@solver_func` to register new solvers. "
                f"Available solvers: {list(_SOLVER_FUNCS.keys())}"
            )
        return _SOLVER_FUNCS[name]  # type: ignore[return-value]


def _get_preconditioner(
    A_csr: typing.Union[csr_array, csr_matrix],
    preconditioner: typing.Optional[Preconditioner],
) -> typing.Optional[LinearOperator]:
    """
    Get or build a preconditioner based on the specification.

    :param A_csr: The coefficient matrix in CSR format.
    :param preconditioner: Preconditioner specification as a string, callable, or None
    :return: A SciPy `LinearOperator` representing the preconditioner, or None.
    :raises ValidationError: If the preconditioner type is unknown.
    """
    if isinstance(preconditioner, (type(None), LinearOperator)):
        return preconditioner
    elif isinstance(preconditioner, str):
        if preconditioner in _PRECONDITIONER_FACTORIES:
            preconditioner_factory = _PRECONDITIONER_FACTORIES[preconditioner]
            M = preconditioner_factory(A_csr)  # type: ignore[operator]
            return M
        else:
            raise ValidationError(
                f"Unknown preconditioner type: {preconditioner!r}. Available preconditioners: {list(_PRECONDITIONER_FACTORIES.keys())}"
            )
    elif callable(preconditioner):
        preconditioner_factory = typing.cast(PreconditionerFactory, preconditioner)
        M = preconditioner_factory(A_csr)
        return M
    return preconditioner  # type: ignore[return-value]


def _get_solver_func(
    solver: typing.Union[Solver, typing.Iterable[Solver]],
) -> typing.List[SolverFunc]:
    """
    Get solver functions from a solver specification.

    :param solver: Solver specification as a string or sequence of strings/callables.
    :return: List of solver functions corresponding to the specification.
    :raises ValidationError: If any solver type is unknown.
    :raises TypeError: If the solver specification is of an invalid type.
    """
    if isinstance(solver, str):
        if solver in _SOLVER_FUNCS:
            solver_func = _SOLVER_FUNCS[solver]
            if isinstance(solver_func, (list, tuple)):
                return list(solver_func)  # type: ignore[return-value]
            return [solver_func]  # type: ignore[return-value]
        raise ValidationError(
            f"Unknown solver type: {solver!r}. Available solvers: {list(_SOLVER_FUNCS.keys())}"
        )
    elif callable(solver):
        return [solver]  # type: ignore[return-value]
    elif isinstance(solver, (list, tuple, set)):
        solver_funcs = []
        for s in solver:
            if isinstance(s, str) and s in _SOLVER_FUNCS:
                solver_funcs.append(_SOLVER_FUNCS[s])
            elif callable(s):
                solver_funcs.append(s)
            else:
                raise ValidationError(f"Unknown solver type in sequence: {s!r}")
        return solver_funcs
    raise TypeError("solver must be a string, callable, or a sequence of strings.")


def solve_linear_system(
    A_csr: typing.Union[csr_array, csr_matrix],
    b: npt.NDArray,
    maximum_iterations: int,
    rtol: typing.Optional[float] = None,
    atol: typing.Optional[float] = None,
    solver: typing.Union[Solver, typing.Iterable[Solver]] = "bicgstab",
    preconditioner: typing.Optional[Preconditioner] = "ilu",
    fallback_to_direct: bool = False,
) -> typing.Tuple[npt.NDArray, typing.Optional[LinearOperator]]:
    """
    Solves the linear system A·x = b using an (iterative) solver with a fallback strategy.

    The function first attempts to solve the system using the BiCGSTAB method, which is
    generally efficient for large, sparse, non-symmetric systems. If BiCGSTAB fails to
    converge within the specified number of iterations, the function falls back to the
    LGMRES method, which is more robust but typically slower.

    Preconditioning is applied using a diagonal preconditioner derived from the diagonal
    elements of matrix A to improve convergence.

    :param A: Coefficient matrix in CSR format.
    :param b: Right-hand side vector.
    :param maximum_iterations: Maximum number of iterations for each solver.
    :param solver: (Iterative) solver or sequence of solvers to use ("bicgstab", "gmres", "lgmres", "tfqmr"), or custom callable(s).
        If a sequence is provided, solvers will be tried in order until one converges.
    :param preconditioner: Type of preconditioner to use ("ilu", "amg", "diagonal"), or None.
        Can also be a preconditioner factory function, that takes A and returns a preconditioner.
        If None, no preconditioning is applied.
    :param rtol: Relative tolerance for convergence (optional).
    :param atol: Absolute tolerance for convergence (optional).
    :param fallback_to_direct: Whether to fall back to a direct solver if all iterative solvers fail.
        Not suitable for large or production use cases due to performance and memory constraints.
    :return: A tuple (x, M) where x is the solution vector and M is the preconditioner used,
    :raises RuntimeError: If both solvers fail to converge.
    """
    solver_funcs = _get_solver_func(solver)
    is_direct = _spsolve in solver_funcs
    if is_direct:
        # No need to build preconditioner for direct solver
        M = None
    else:
        try:
            M = _get_preconditioner(A_csr, preconditioner)
        except Exception as exc:
            raise PreconditionerError(f"Error building preconditioner: {exc}") from exc

    b_norm = np.linalg.norm(b)
    # Too strict
    # epsilon = get_floating_point_info().eps
    # rtol = rtol if rtol is not None else float(epsilon * 50)
    # atol = atol if atol is not None else float(max(1e-8, 20 * epsilon * b_norm))
    rtol = rtol if rtol is not None else 1e-6
    atol = atol if atol is not None else float(max(1e-8, 1e-6 * b_norm))

    for solver_func in solver_funcs:
        x, info = solver_func(
            A=A_csr,
            b=b,
            x0=None,
            M=M,
            rtol=rtol,
            atol=atol,
            maxiter=maximum_iterations,
            callback=None,
        )
        if info == 0:
            return np.ascontiguousarray(x), M
        else:
            logger.warning(
                f"Solver {solver_func!r} failed to converge within {maximum_iterations} iterations. Info: {info}"
            )

    if not fallback_to_direct or is_direct:
        raise SolverError(
            f"All solvers failed to converge within {maximum_iterations} iterations."
        )

    logger.info("Falling back to direct solver (spsolve).")
    n = A_csr.shape[0]
    if n > 100000:
        logger.warning(
            f"Direct solver on large system ({n:,} DOFs) may consume excessive memory "
            f"(estimated: {n * n * 8 / 1e9:.2f} GB for dense factorization). "
            f"Consider improving iterative solver settings or preconditioner."
        )

    try:
        x = spsolve(A_csr, b)
    except Exception as exc:
        logger.error(f"Direct solver failed: {exc}")
        raise SolverError(
            "All iterative solvers and direct solver failed to solve the system."
        ) from exc

    return np.ascontiguousarray(x), None  # type: ignore[return-value]

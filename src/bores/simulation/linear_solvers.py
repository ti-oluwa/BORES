import logging
import threading
import typing
import warnings

import numba  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pyamg  # type: ignore[import-untyped]
from scipy.sparse import (  # type: ignore[import-untyped]
    csr_array,
    csr_matrix,
)
from scipy.sparse.linalg import (  # type: ignore[import-untyped]
    LinearOperator,
    MatrixRankWarning,
    bicg,
    bicgstab,
    cg,
    cgs,
    gcrotmk,
    gmres,
    lgmres,
    minres,
    qmr,
    spilu,
    spsolve,
    tfqmr,
)

from bores.errors import PreconditionerError, SolverError, ValidationError
from bores.types import (
    Preconditioner,
    PreconditionerFactory,
    Solver,
    SolverFunction,
)

logger = logging.getLogger(__name__)


__all__ = [
    "CPR_AMG_KWARGS",
    "CPR_ILU_KWARGS",
    "ILU_QUALITY_ERROR_THRESHOLD",
    "ILU_QUALITY_WARNING_THRESHOLD",
    "RESIDUAL_VERIFICATION_FACTOR",
    "CachedPreconditionerFactory",
    "SystemScalingMethod",
    "VariableOrdering",
    "get_preconditioner_factory",
    "get_solver_function",
    "invert_diagonal_blocks",
    "list_preconditioner_factories",
    "list_solver_functions",
    "make_amg_preconditioner",
    "make_block_jacobi_preconditioner",
    "make_cpr_preconditioner",
    "make_diagonal_preconditioner",
    "make_ilu_preconditioner",
    "make_polynomial_preconditioner",
    "preconditioner_factory",
    "scale_linear_system",
    "solve_linear_system",
    "solver_function",
]


ILU_QUALITY_WARNING_THRESHOLD = 1e3
"""
Relative error of an ILU factorization applied to a probe vector above which a warning
is logged. The error is `||A·M·v - v|| / ||v||`, which is close to zero for an accurate
factorization.
"""

ILU_QUALITY_ERROR_THRESHOLD = 1e6
"""
Relative error of an ILU factorization applied to a probe vector above which the
factorization is rejected as numerically meaningless.
"""

RESIDUAL_VERIFICATION_FACTOR = 10.0
"""
How much larger than the requested tolerance the true residual `||b - A·x||` of an
iterative solution may be before the solution is rejected. Iterative solvers decide
convergence from internal estimates, which can differ from the true residual, so every
solution reported as converged is checked against the true residual.
"""


def _invert_diagonal(diagonal: npt.NDArray) -> npt.NDArray:
    """
    Inverts a vector of diagonal entries, leaving unusable entries unscaled.

    An entry is unusable when it is not finite or when its magnitude is below the
    floating point precision of its dtype relative to the largest entry. Such entries
    are treated as one, so that they neither amplify the residual nor divide by zero.

    :param diagonal: The diagonal entries to invert.
    :return: The element-wise inverse, with unusable entries replaced by one.
    """
    diagonal = np.asarray(diagonal)
    magnitude = np.abs(diagonal)
    finite = np.isfinite(magnitude)
    largest = magnitude[finite].max() if finite.any() else 0.0
    threshold = np.finfo(diagonal.dtype).eps * largest
    unusable = ~finite | (magnitude <= threshold)
    return 1.0 / np.where(unusable, 1.0, diagonal)


def _flatten(vector: npt.NDArray) -> npt.NDArray:
    """Returns `vector` as one dimension, accepting the `(n, 1)` shape SciPy may pass."""
    return np.asarray(vector).reshape(-1)


def _get_matrix_shape(A_csr: csr_array | csr_matrix) -> tuple[int, int]:
    """Return the two-dimensional shape of a CSR matrix, validating its rank."""
    shape = A_csr.shape
    if len(shape) != 2:
        raise ValidationError(f"Matrix must be two-dimensional, got shape {shape}.")
    return typing.cast("tuple[int, int]", shape)


def _get_square_matrix_shape(A_csr: csr_array | csr_matrix) -> tuple[int, int]:
    """Return a CSR matrix's shape, raising if it is not square."""
    shape = _get_matrix_shape(A_csr)
    if shape[0] != shape[1]:
        raise ValidationError(f"Matrix must be square, got shape {shape}.")
    return shape


def _check_matrix_is_square(A_csr: csr_array | csr_matrix) -> None:
    """Validate that a CSR matrix is square when its shape is not otherwise needed."""
    _get_square_matrix_shape(A_csr)


def invert_diagonal_blocks(
    A_csr: csr_array | csr_matrix,
    *,
    block_size: int,
) -> npt.NDArray:
    """
    Inverts every dense diagonal block of a block-structured sparse matrix.

    The matrix is split into consecutive square blocks of `block_size` rows and
    columns along its diagonal. When the matrix dimension is not a multiple of
    `block_size`, the last block is smaller, and it is padded with an identity so all
    blocks can be inverted together. A block that cannot be inverted is replaced by
    the inverse of its diagonal, with unusable diagonal entries treated as one.

    :param A_csr: The coefficient matrix in CSR format.
    :param block_size: Number of rows and columns in each diagonal block.
    :return: Array of shape `(number_of_blocks, block_size, block_size)` holding the
        inverse of each diagonal block.
    :raises ValidationError: If the matrix is not square or `block_size` is below one.
    """
    if block_size < 1:
        raise ValidationError(f"`block_size` must be at least 1, got {block_size}.")

    shape = _get_square_matrix_shape(A_csr)
    n = shape[0]

    number_of_blocks = -(-n // block_size)
    padded_size = number_of_blocks * block_size

    csr = A_csr.tocsr()
    blocks = np.zeros((number_of_blocks, block_size, block_size), dtype=np.float64)
    _gather_diagonal_blocks(
        indptr=csr.indptr,
        indices=csr.indices,
        data=csr.data,
        block_size=block_size,
        blocks=blocks,
    )
    for padded_row in range(n, padded_size):
        blocks[padded_row // block_size, padded_row % block_size, padded_row % block_size] = 1.0

    inverses = np.empty_like(blocks)
    failed = np.zeros(number_of_blocks, dtype=np.bool_)
    _invert_small_blocks(blocks=blocks, inverses=inverses, failed=failed)
    for failed_index in np.flatnonzero(failed):
        inverses[failed_index] = np.diag(_invert_diagonal(np.diag(blocks[failed_index])))
    return inverses.astype(A_csr.dtype, copy=False)


@numba.njit(cache=True, error_model="numpy")
def _gather_diagonal_blocks(
    indptr: npt.NDArray,
    indices: npt.NDArray,
    data: npt.NDArray,
    block_size: int,
    blocks: npt.NDArray,
) -> None:
    """Adds the entries of a CSR matrix that lie inside its diagonal blocks to `blocks`."""
    for row in range(indptr.shape[0] - 1):
        block_index = row // block_size
        first_column = block_index * block_size
        for position in range(indptr[row], indptr[row + 1]):
            column = indices[position]
            if first_column <= column < first_column + block_size:
                blocks[block_index, row - first_column, column - first_column] += data[position]


@numba.njit(cache=True, parallel=True, error_model="numpy")
def _invert_small_blocks(
    blocks: npt.NDArray,
    inverses: npt.NDArray,
    failed: npt.NDArray,
) -> None:
    """
    Inverts each small dense block by Gauss-Jordan elimination with partial pivoting.

    A block is marked in `failed` when a pivot is not larger than the floating point
    round-off of the block, which includes blocks with non-finite entries, and when its
    inverse is not finite. The inverse of a failed block is not meaningful.
    """
    number_of_blocks, size, _ = blocks.shape
    unit_roundoff = 2.220446049250313e-16
    for block_index in numba.prange(number_of_blocks):
        work = np.zeros((size, 2 * size))
        largest = 0.0
        for i in range(size):
            for j in range(size):
                work[i, j] = blocks[block_index, i, j]
                magnitude = abs(work[i, j])
                if magnitude > largest:
                    largest = magnitude
            work[i, size + i] = 1.0
        tolerance = unit_roundoff * size * largest

        valid = True
        for column in range(size):
            pivot_row = column
            pivot_magnitude = abs(work[column, column])
            for row in range(column + 1, size):
                if abs(work[row, column]) > pivot_magnitude:
                    pivot_row = row
                    pivot_magnitude = abs(work[row, column])
            if not pivot_magnitude > tolerance:
                valid = False
                break
            if pivot_row != column:
                for j in range(2 * size):
                    swapped = work[column, j]
                    work[column, j] = work[pivot_row, j]
                    work[pivot_row, j] = swapped
            pivot = work[column, column]
            for j in range(2 * size):
                work[column, j] /= pivot
            for row in range(size):
                if row != column:
                    factor = work[row, column]
                    for j in range(2 * size):
                        work[row, j] -= factor * work[column, j]

        if valid:
            for i in range(size):
                for j in range(size):
                    inverses[block_index, i, j] = work[i, size + j]
                    if not np.isfinite(work[i, size + j]):
                        valid = False
        failed[block_index] = not valid


@numba.njit(cache=True, parallel=True, error_model="numpy")
def _multiply_block_diagonal(
    inverses: npt.NDArray,
    vector: npt.NDArray,
    out: npt.NDArray,
    transpose: bool,
) -> None:
    """Multiplies `vector` by a block diagonal matrix, treating entries past its end as zero."""
    number_of_blocks, size, _ = inverses.shape
    n = vector.shape[0]
    for block_index in numba.prange(number_of_blocks):
        first = block_index * size
        for i in range(size):
            row = first + i
            if row >= n:
                continue
            total = 0.0
            for j in range(size):
                column = first + j
                if column >= n:
                    continue
                if transpose:
                    total += inverses[block_index, j, i] * vector[column]
                else:
                    total += inverses[block_index, i, j] * vector[column]
            out[row] = total


def _apply_block_inverses(
    inverses: npt.NDArray,
    vector: npt.NDArray,
    *,
    transpose: bool = False,
) -> npt.NDArray:
    """Multiplies `vector` by the block diagonal matrix whose blocks are `inverses`."""
    flat = np.ascontiguousarray(_flatten(vector))
    result = np.empty(flat.shape[0], dtype=np.result_type(inverses.dtype, flat.dtype))
    _multiply_block_diagonal(inverses=inverses, vector=flat, out=result, transpose=transpose)
    return result.reshape(np.shape(vector))


def make_amg_preconditioner(
    A_csr: csr_array | csr_matrix, cycle: str = "V", **kwargs: typing.Any
) -> LinearOperator:
    """
    Creates an Algebraic Multigrid (AMG) preconditioner using PyAMG.

    Smoothed aggregation AMG works best for matrices that resemble a diffusion
    operator, such as the pressure block of a reservoir Jacobian. It is a poor choice
    for a full multi-variable Jacobian. The returned operator does not support
    transposed products, so it cannot be used with the `bicg` and `qmr` solvers.

    :param A_csr: The coefficient matrix in CSR format.
    :param cycle: Multigrid cycle type ('V', 'W', 'F'). V-cycle is standard,
        W-cycle offers better convergence for difficult problems but is more expensive.
    :param kwargs: Additional arguments for `pyamg.smoothed_aggregation_solver`.
    :return: A SciPy `LinearOperator` that represents the AMG preconditioner.
    """
    _check_matrix_is_square(A_csr)
    solver = pyamg.smoothed_aggregation_solver(A_csr.tocsr(), **kwargs)
    return solver.aspreconditioner(cycle=cycle)


def make_diagonal_preconditioner(
    A_csr: csr_array | csr_matrix,
) -> LinearOperator:
    """
    Creates a diagonal (Jacobi) preconditioner from the coefficient matrix.

    Diagonal entries that are not finite, or that are negligible compared with the
    largest diagonal entry at the matrix precision, are treated as one.

    :param A_csr: The coefficient matrix in CSR format.
    :return: A SciPy `LinearOperator` that represents the diagonal preconditioner.
    """
    shape = _get_square_matrix_shape(A_csr)
    inverse_diagonal = _invert_diagonal(A_csr.diagonal())

    def matvec(x: npt.NDArray) -> npt.NDArray:
        return (inverse_diagonal * _flatten(x)).reshape(np.shape(x))

    return LinearOperator(  # type: ignore[arg-type]
        shape=shape,
        matvec=matvec,
        rmatvec=matvec,
        dtype=inverse_diagonal.dtype,
    )


def _block_jacobi_operator(
    inverses: npt.NDArray,
    shape: tuple[int, int],
) -> LinearOperator:
    """Wraps precomputed diagonal block inverses as a Block Jacobi `LinearOperator`."""

    def matvec(x: npt.NDArray) -> npt.NDArray:
        return _apply_block_inverses(inverses, x)

    def rmatvec(x: npt.NDArray) -> npt.NDArray:
        return _apply_block_inverses(inverses, x, transpose=True)

    return LinearOperator(  # type: ignore[arg-type]
        shape=shape,
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=inverses.dtype,
    )


def make_block_jacobi_preconditioner(
    A_csr: csr_array | csr_matrix,
    block_size: int = 3,
) -> LinearOperator:
    """
    Creates a Block Jacobi preconditioner for block-structured systems.

    For black-oil reservoir simulation with 3 variables per cell (pressure, So, Sg),
    this preconditioner is more effective than diagonal for coupled systems. When the
    matrix dimension is not a multiple of `block_size`, the last block is smaller.
    A block that cannot be inverted is replaced by the inverse of its diagonal.

    :param A_csr: The coefficient matrix in CSR format.
    :param block_size: Size of each block (e.g., 3 for black-oil: pressure, So, Sg).
    :return: A SciPy `LinearOperator` that represents the Block Jacobi preconditioner.
    :raises ValidationError: If the matrix is not square or `block_size` is below one.
    """
    inverses = invert_diagonal_blocks(A_csr, block_size=block_size)
    return _block_jacobi_operator(inverses, _get_matrix_shape(A_csr))


def make_ilu_preconditioner(A_csr: csr_array | csr_matrix, **kwargs: typing.Any) -> LinearOperator:
    """
    Creates an Incomplete LU (ILU) preconditioner using `spilu`.

    The quality of the factorization is checked by applying it to a fixed probe vector
    `v` and measuring `||A·M·v - v|| / ||v||`. An error above
    `ILU_QUALITY_WARNING_THRESHOLD` is logged as a warning. A factorization that
    produces non-finite values, or an error above `ILU_QUALITY_ERROR_THRESHOLD`, is
    rejected, since it would only feed meaningless corrections to the iterative solver.
    Incomplete factorizations of strongly coupled, non-symmetric reservoir Jacobians
    can be unreliable, so a block Jacobi preconditioner is the safer alternative when
    this check fails.

    :param A_csr: The coefficient matrix in CSR format.
    :param kwargs: Additional arguments for `scipy.sparse.linalg.spilu`. `drop_tol`
        defaults to 1e-4 (smaller values keep more entries and are more accurate) and
        `fill_factor` defaults to 10 (the memory allowed relative to the matrix).
    :return: A SciPy `LinearOperator` that applies the incomplete factorization.
    :raises ValidationError: If the matrix is not square.
    :raises PreconditionerError: If the factorization is numerically unusable.
    """
    shape = _get_square_matrix_shape(A_csr)
    A_csc = A_csr.tocsc()
    kwargs.setdefault("drop_tol", 1e-4)
    kwargs.setdefault("fill_factor", 10)
    try:
        ilu_factor = spilu(A_csc, **kwargs)
    except RuntimeError as exc:
        raise PreconditionerError(f"ILU factorization failed: {exc}") from exc

    def matvec(x: npt.NDArray) -> npt.NDArray:
        return ilu_factor.solve(_flatten(x)).reshape(np.shape(x))

    def rmatvec(x: npt.NDArray) -> npt.NDArray:
        return ilu_factor.solve(_flatten(x), trans="T").reshape(np.shape(x))  # type: ignore[arg-type]

    operator = LinearOperator(  # type: ignore[arg-type]
        shape=shape,
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=A_csc.dtype,
    )

    probe = np.random.default_rng(0).standard_normal(shape[0]).astype(A_csc.dtype)
    probe_norm = np.linalg.norm(probe)
    with np.errstate(all="ignore"):
        correction = operator.matvec(probe)
        error = np.linalg.norm(A_csc @ correction - probe) / probe_norm
    if not np.isfinite(error) or error > ILU_QUALITY_ERROR_THRESHOLD:
        raise PreconditionerError(
            f"ILU factorization is numerically unusable (probe error {error:.3e}). "
            f"Use a smaller `drop_tol`, a larger `fill_factor`, or a different preconditioner."
        )
    if error > ILU_QUALITY_WARNING_THRESHOLD:
        logger.warning(
            f"ILU factorization is a poor approximation of the inverse (probe error {error:.3e})."
        )
    return operator


def make_polynomial_preconditioner(
    A_csr: csr_array | csr_matrix,
    degree: int = 2,
) -> LinearOperator:
    """
    Creates a damped Neumann series polynomial preconditioner.

    The operator approximates the inverse of A with the truncated series
    `(I + G + G² + ... + Gᵈ)·ω·D⁻¹`, where `D` is the diagonal of A, `G = I - ω·D⁻¹·A`,
    and `d` is the degree. The damping factor `ω` is the reciprocal of the largest row
    sum of `|D⁻¹·A|`, which keeps the series from diverging. A degree of zero gives a
    damped diagonal preconditioner.

    This preconditioner is cheap and helps on diagonally dominant systems. It does not
    help on strongly coupled multiphase reservoir Jacobians, where block Jacobi or CPR
    should be used instead.

    :param A_csr: The coefficient matrix in CSR format.
    :param degree: Polynomial degree, zero or greater (1, 2, or 3 recommended).
    :return: A SciPy `LinearOperator` that represents the polynomial preconditioner.
    :raises ValidationError: If the matrix is not square or `degree` is negative.
    """
    _check_matrix_is_square(A_csr)
    if degree < 0:
        raise ValidationError(f"`degree` must be zero or greater, got {degree}.")

    A = A_csr.tocsr()
    inverse_diagonal = _invert_diagonal(A.diagonal())
    row_of_entry = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
    scaled_row_sums = np.bincount(
        row_of_entry,
        weights=np.abs(A.data) * np.abs(inverse_diagonal)[row_of_entry],
        minlength=A.shape[0],
    )
    largest_row_sum = float(scaled_row_sums.max()) if A.shape[0] > 0 else 0.0
    damping = 1.0 / largest_row_sum if largest_row_sum > 0.0 else 1.0
    damped_inverse_diagonal = damping * inverse_diagonal
    A_transpose = A.T.tocsr()

    def matvec(x: npt.NDArray) -> npt.NDArray:
        flat = _flatten(x)
        first_term = damped_inverse_diagonal * flat
        result = first_term
        for _ in range(degree):
            result = first_term + result - damped_inverse_diagonal * (A @ result)
        return result.reshape(np.shape(x))

    def rmatvec(x: npt.NDArray) -> npt.NDArray:
        flat = _flatten(x)
        result = flat
        for _ in range(degree):
            result = flat + result - A_transpose @ (damped_inverse_diagonal * result)
        return (damped_inverse_diagonal * result).reshape(np.shape(x))

    return LinearOperator(  # type: ignore[arg-type]
        shape=A.shape,
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=A.dtype,
    )


VariableOrdering = typing.Literal["cell_major", "variable_major"]
"""
How the entries of a multi-variable system are laid out. With "cell_major", the entries
of one cell are stored together, so entry `i` of cell `c` is at `c * n_variables_per_cell + i`.
With "variable_major", all entries of one variable are stored together, so entry `i`
of cell `c` is at `i * number_of_cells + c`.
"""


def _cell_major_permutation(
    *,
    number_of_cells: int,
    n_variables_per_cell: int,
    ordering: str,
) -> npt.NDArray | None:
    """
    Returns the permutation that lists a system's entries in cell major order.

    :return: `permutation` such that `permutation[k]` is the position, in the given
        ordering, of the entry stored at position `k` in cell major order, or None
        if the ordering already is cell major.
    :raises ValidationError: If the ordering is unknown.
    """
    if ordering == "cell_major":
        return None
    if ordering == "variable_major":
        return (
            np
            .arange(number_of_cells * n_variables_per_cell, dtype=np.int64)
            .reshape(n_variables_per_cell, number_of_cells)
            .T.reshape(-1)
        )
    raise ValidationError(f"Unknown ordering: {ordering!r}. Use 'cell_major' or 'variable_major'.")


CPR_AMG_KWARGS = {
    "max_coarse": 500,
    "presmoother": ("gauss_seidel", {"sweep": "symmetric", "iterations": 1}),
    "postsmoother": ("gauss_seidel", {"sweep": "symmetric", "iterations": 1}),
}
"""Default AMG parameters for CPR preconditioner."""
CPR_ILU_KWARGS = {
    "drop_tol": 1e-4,
    "fill_factor": 10,
}
"""Default ILU parameters for CPR preconditioner."""


def make_cpr_preconditioner(
    A_csr: csr_array | csr_matrix,
    *,
    n_variables_per_cell: int = 3,
    pressure_variable_index: int = 0,
    ordering: VariableOrdering = "cell_major",
    equation_ordering: VariableOrdering | None = None,
    decouple_pressure: bool = True,
    smoother: typing.Literal["block_jacobi", "ilu"] = "block_jacobi",
    amg_kwargs: dict[str, typing.Any] | None = None,
    ilu_kwargs: dict[str, typing.Any] | None = None,
) -> LinearOperator:
    """
    Creates a Constrained Pressure Residual (CPR) preconditioner for a fully-implicit
    multiphase reservoir Jacobian.

    CPR is a two-stage preconditioner:

        Stage 1 (global): AMG solve of the pressure equation.
        Stage 2 (local): a smoother on the full system that removes the error the
        pressure solve leaves behind.

    This preconditioner significantly improves convergence for 3D multiphase
    reservoir simulations where the Jacobian is strongly coupled but the pressure
    equation dominates the long-range behavior.

    The unknowns and equations may be laid out cell by cell or variable by variable, see
    `ordering` and `equation_ordering`. The equations of a cell must be listed in the same
    variable order as its unknowns, so that each cell's diagonal block is square and
    invertible.

    :param A_csr: The full Jacobian matrix in CSR format.
    :param n_variables_per_cell: Number of unknowns per grid cell
        (e.g., pressure, saturation_o, saturation_g).
    :param pressure_variable_index: Index of the pressure unknown inside each cell's
        variable block. Example: if block = [pressure, So, Sg], then index = 0.
    :param ordering: Layout of the unknowns. "cell_major" stores the unknowns of each cell
        together, as in [p0, So0, Sg0, p1, So1, Sg1, ...]. "variable_major" stores each
        variable for all cells together, as in [p0, p1, ..., So0, So1, ..., Sg0, Sg1, ...].
    :param equation_ordering: Layout of the equations, with the same meaning as `ordering`.
        Defaults to the same layout as the unknowns.
    :param decouple_pressure: Whether to form the pressure equation by combining each
        cell's equations so that the dependence on saturations in the cell's own
        block is eliminated (quasi-IMPES). When False, the pressure rows and columns
        of the Jacobian are used directly, which is simpler but is a weaker pressure
        equation.
    :param smoother: The second stage smoother. "block_jacobi" inverts each cell's
        diagonal block and is robust and cheap. "ilu" uses an incomplete LU
        factorization of the full Jacobian, which is more expensive and can be
        unreliable for strongly coupled Jacobians.
    :param amg_kwargs: Keyword arguments for `pyamg.smoothed_aggregation_solver`.
        Defaults to `CPR_AMG_KWARGS`.
    :param ilu_kwargs: Keyword arguments for `scipy.sparse.linalg.spilu`, used only when
        `smoother` is "ilu". Defaults to `CPR_ILU_KWARGS`.
    :return: A SciPy `LinearOperator` implementing the CPR preconditioner M such that
        x = M @ r approximately solves J^{-1} r. It does not support transposed
        products, so it cannot be used with the `bicg` and `qmr` solvers.
    :raises ValidationError: If the matrix is not square, its size is not a multiple of
        `n_variables_per_cell`, `pressure_variable_index` is out of range, or an ordering
        is unknown.
    :raises PreconditionerError: If AMG or smoother construction fails.
    """
    A_csr = A_csr.tocsr()
    matrix_shape = _get_square_matrix_shape(A_csr)
    number_of_equations = matrix_shape[0]
    if n_variables_per_cell < 1:
        raise ValidationError(
            f"`n_variables_per_cell` must be at least 1, got {n_variables_per_cell}."
        )
    if not 0 <= pressure_variable_index < n_variables_per_cell:
        raise ValidationError(
            f"`pressure_variable_index` must be in [0, {n_variables_per_cell}), "
            f"got {pressure_variable_index}."
        )
    if number_of_equations == 0 or number_of_equations % n_variables_per_cell != 0:
        raise ValidationError(
            f"Matrix size {number_of_equations} is not a positive multiple of "
            f"`n_variables_per_cell` ({n_variables_per_cell})."
        )

    if smoother not in ("block_jacobi", "ilu"):
        raise ValidationError(f"Unknown CPR smoother: {smoother!r}. Use 'block_jacobi' or 'ilu'.")

    amg_kwargs = dict(CPR_AMG_KWARGS) if amg_kwargs is None else amg_kwargs
    ilu_kwargs = dict(CPR_ILU_KWARGS) if ilu_kwargs is None else ilu_kwargs
    number_of_cells = number_of_equations // n_variables_per_cell
    unknown_permutation = _cell_major_permutation(
        number_of_cells=number_of_cells,
        n_variables_per_cell=n_variables_per_cell,
        ordering=ordering,
    )
    equation_permutation = _cell_major_permutation(
        number_of_cells=number_of_cells,
        n_variables_per_cell=n_variables_per_cell,
        ordering=ordering if equation_ordering is None else equation_ordering,
    )
    if unknown_permutation is not None or equation_permutation is not None:
        # All work below is done on the system in cell major order.
        if equation_permutation is not None:
            A_csr = A_csr[equation_permutation, :]  # type: ignore
        if unknown_permutation is not None:
            A_csr = csr_matrix(A_csr)[:, unknown_permutation]
        A_csr = A_csr.tocsr()

    pressure_dof_indices = np.arange(
        pressure_variable_index,
        number_of_equations,
        n_variables_per_cell,
        dtype=np.int64,
    )

    block_inverses = None
    if decouple_pressure or smoother == "block_jacobi":
        block_inverses = invert_diagonal_blocks(A_csr, block_size=n_variables_per_cell)

    # Restriction: maps a full residual to the pressure equation residual.
    restriction = None
    if decouple_pressure:
        assert block_inverses is not None
        weights = block_inverses[:, pressure_variable_index, :]
        restriction = csr_matrix(
            (
                weights.reshape(-1),
                (
                    np.repeat(np.arange(number_of_cells), n_variables_per_cell),
                    np.arange(number_of_equations),
                ),
            ),
            shape=(number_of_cells, number_of_equations),
        )
        pressure_matrix = (restriction @ A_csr)[:, pressure_dof_indices].tocsr()
    else:
        pressure_matrix = csr_matrix(A_csr)[pressure_dof_indices, :][
            :, pressure_dof_indices
        ].tocsr()

    try:
        M_amg = make_amg_preconditioner(pressure_matrix, **amg_kwargs)  # type: ignore
    except Exception as exc:
        raise PreconditionerError(f"AMG construction for pressure block failed: {exc}") from exc

    if smoother == "block_jacobi":
        assert block_inverses is not None
        M_smoother = _block_jacobi_operator(block_inverses, matrix_shape)
    else:
        try:
            M_smoother = make_ilu_preconditioner(A_csr, **ilu_kwargs)
        except PreconditionerError:
            raise
        except Exception as exc:
            raise PreconditionerError(f"ILU factorization failed for CPR: {exc}") from exc

    def matvec(residual: npt.NDArray) -> npt.NDArray:
        """CPR preconditioner application: x = M^{-1} r"""
        flat = _flatten(residual)
        if equation_permutation is not None:
            flat = flat[equation_permutation]
        if restriction is not None:
            pressure_residual = restriction @ flat
        else:
            pressure_residual = flat[pressure_dof_indices]

        pressure_correction = M_amg.matvec(pressure_residual)
        correction = np.zeros(number_of_equations, dtype=pressure_correction.dtype)
        correction[pressure_dof_indices] = pressure_correction

        remaining = flat - A_csr @ correction
        result = correction + M_smoother.matvec(remaining)
        if unknown_permutation is not None:
            reordered = np.empty_like(result)
            reordered[unknown_permutation] = result
            result = reordered
        return result.reshape(np.shape(residual))

    return LinearOperator(  # type: ignore[arg-type]
        shape=matrix_shape, matvec=matvec, dtype=A_csr.dtype
    )


def _spsolve(
    A: typing.Any,
    b: typing.Any,
    x0: typing.Any | None,
    *,
    rtol: float,
    atol: float,
    maxiter: int | None,
    M: typing.Any | None,
    callback: typing.Callable[[npt.NDArray], None] | None,
) -> tuple[npt.NDArray, int]:
    """
    Direct sparse solver wrapper compatible with the standard `SolverFunction` interface.

    The tolerances, iteration limit, preconditioner and callback are ignored since the
    solution is exact up to rounding. A singular or otherwise unsolvable system is
    reported through a non-zero status instead of a solution containing NaN values.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            x = spsolve(A.tocsc(), b)
        except (MatrixRankWarning, RuntimeError):
            return np.full(np.shape(b), np.nan, dtype=np.result_type(b, np.float32)), 1

    if not np.all(np.isfinite(x)):
        return x, 1  # type: ignore[return-value]
    return x, 0  # type: ignore[return-value]


def _lgmres(
    A: typing.Any,
    b: typing.Any,
    x0: typing.Any | None,
    *,
    rtol: float,
    atol: float,
    maxiter: int | None,
    M: typing.Any | None,
    callback: typing.Callable[[npt.NDArray], None] | None,
    inner_m: int = 50,
    outer_k: int = 5,
) -> tuple[npt.NDArray, int]:
    """
    LGMRES solver wrapper compatible with the standard `SolverFunction` interface,
    with configurable inner/outer iteration parameters.

    `maxiter` counts outer iterations. Each outer iteration performs up to `inner_m`
    inner GMRES iterations. When `maxiter` is not given, 1000 outer iterations are allowed.

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
        maxiter=1000 if maxiter is None else maxiter,
        callback=callback,
        inner_m=inner_m,
        outer_k=outer_k,
    )


def _minres(
    A: typing.Any,
    b: typing.Any,
    x0: typing.Any | None,
    *,
    rtol: float,
    atol: float,
    maxiter: int | None,
    M: typing.Any | None,
    callback: typing.Callable[[npt.NDArray], None] | None,
    shift: float = 0.0,
) -> tuple[npt.NDArray, int]:
    """
    MINRES solver wrapper compatible with the standard `SolverFunction` interface.

    MINRES is only suitable for symmetric (possibly indefinite) systems, and a
    preconditioner for it must be symmetric positive definite. It does **not** accept
    `atol` directly. SciPy stops when its estimate of the residual falls below `rtol`
    times an estimate of `||A||·||x||`. To approximate the caller's `atol`, `rtol` is
    tightened to the smaller of `rtol` and `atol / ||b||`. The result is only
    approximately bound by `atol`, so the final residual should be verified by the caller.

    :param shift: Solves `(A - shift·I) x = b` instead of `A x = b`.
        Useful when A is nearly singular or when targeting a shifted system
        (e.g. eigenvalue-deflation tricks). Defaults to 0.0 (no shift).
    """
    b_array: npt.NDArray = np.asarray(b)
    b_norm = np.linalg.norm(b_array)
    eps = np.finfo(b_array.dtype).eps if np.issubdtype(b_array.dtype, np.floating) else 1e-15

    effective_rtol = min(rtol, atol / max(b_norm, eps))
    # MINRES requires rtol > 0; clamp to a safe floor.
    effective_rtol = max(effective_rtol, eps * 10)
    return minres(  # type: ignore[return-value]
        A,
        b,
        x0=x0,
        shift=shift,
        M=M,
        rtol=effective_rtol,
        maxiter=1000 if maxiter is None else maxiter,
        callback=callback,
    )


def _qmr(
    A: typing.Any,
    b: typing.Any,
    x0: typing.Any | None,
    *,
    rtol: float,
    atol: float,
    maxiter: int | None,
    M: typing.Any | None,
    callback: typing.Callable[[npt.NDArray], None] | None,
) -> tuple[npt.NDArray, int]:
    """
    QMR solver wrapper compatible with the standard `SolverFunction` interface.

    SciPy's `~scipy.sparse.linalg.qmr` uses a *split* preconditioner
    `(M1, M2)` rather than the single `M` used by every other solver here.
    The preconditioner `M` is supplied as `M1` and `M2` is left as the identity.
    QMR multiplies by the transpose of the preconditioner, so `M` must support
    transposed products. The diagonal, block Jacobi, ILU and polynomial
    preconditioners do, while AMG and CPR do not.

    When `maxiter` is not given, 1000 iterations are allowed.
    """
    right_preconditioner = None
    if M is not None:
        # SciPy requires both halves of the split preconditioner whenever one is given.
        right_preconditioner = LinearOperator(  # type: ignore[arg-type]
            shape=M.shape,
            matvec=lambda x: x,
            rmatvec=lambda x: x,
            dtype=M.dtype,
        )
    return qmr(  # type: ignore[return-value]
        A,
        b,
        x0=x0,
        M1=M,
        M2=right_preconditioner,  # type: ignore[arg-type]
        rtol=rtol,
        atol=atol,
        maxiter=1000 if maxiter is None else maxiter,
        callback=callback,
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

    The preconditioner is always rebuilt when the matrix has a different shape or
    sparsity pattern from the one it was built for, or when its values are not finite.

    A cached factory holds the state of one sequence of related matrices. Registering
    it under a name shared by solves of unrelated systems makes those solves rebuild
    the preconditioner on every call, so use a separate instance for each system.
    Calls are serialized, so one instance can be used from several threads.

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
        factory=make_ilu_preconditioner,
        update_frequency=10,
        recompute_threshold=0.3,
    )

    # In simulation loop
    for timestep in range(num_steps):
        A = make_matrix(...)  # Matrix changes each step
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
        factory: str | PreconditionerFactory,
        name: str | None = None,
        update_frequency: int = 10,
        recompute_threshold: float = 0.5,
    ):
        """
        Initialize cached preconditioner.

        :param factory: Preconditioner factory function or string name (e.g., "ilu", "amg", "block_jacobi")
        :param name: The name of the preconditioner factory been cached or a new name for the cached version of the factory.
        :param update_frequency: Rebuild the preconditioner after it has been used for this
            many calls (0 = never rebuild because of the call count).
        :param recompute_threshold: Rebuild if ||A_new - A_old||/||A_old|| > threshold, where
            the norm is taken over the stored matrix values (0 = never rebuild because of
            matrix changes).
        :raises ValidationError: If `update_frequency` or `recompute_threshold` is negative.
        """
        if update_frequency < 0:
            raise ValidationError(
                f"`update_frequency` must be zero or greater, got {update_frequency}."
            )
        if recompute_threshold < 0:
            raise ValidationError(
                f"`recompute_threshold` must be zero or greater, got {recompute_threshold}."
            )

        if isinstance(factory, str):
            self.factory = get_preconditioner_factory(factory)
            self._name = name or factory
        else:
            self.factory = factory
            self._name = name or str(getattr(factory, "__name__", repr(factory)))

        self.update_frequency = update_frequency
        self.recompute_threshold = recompute_threshold

        self._lock = threading.Lock()
        self._cached_M: LinearOperator | None = None
        self._cached_A_data: npt.NDArray | None = None
        self._cached_A_indices: npt.NDArray | None = None
        self._cached_A_indptr: npt.NDArray | None = None
        self._cached_A_shape: tuple[int, int] | None = None
        self._call_count = 0

    @property
    def name(self) -> str:
        """The name this preconditioner factory is registered under."""
        return self._name

    def __call__(self, A_csr: csr_array | csr_matrix) -> LinearOperator:
        """
        Get preconditioner, reusing cache if possible.

        :param A_csr: Current coefficient matrix
        :return: Preconditioner (cached or newly built)
        """
        with self._lock:
            if self._should_recompute(A_csr):
                logger.debug(
                    f"Rebuilding {self._name} preconditioner "
                    f"(reused {self._call_count} times, threshold={self.recompute_threshold:.2%})"
                )
                self._cached_M = self.factory(A_csr)
                self._cached_A_data = A_csr.data.copy()
                self._cached_A_indices = A_csr.indices.copy()
                self._cached_A_indptr = A_csr.indptr.copy()
                self._cached_A_shape = _get_matrix_shape(A_csr)
                self._call_count = 0
            else:
                logger.debug(
                    f"Reusing cached {self._name} preconditioner "
                    f"(call #{self._call_count}/{self.update_frequency})"
                )

            self._call_count += 1
            return typing.cast(LinearOperator, self._cached_M)

    def should_recompute(self, A_csr: csr_array | csr_matrix) -> bool:
        """
        Determine if preconditioner should be rebuilt.

        :param A_csr: Current coefficient matrix
        :return: True if the next call would rebuild the preconditioner.
        """
        with self._lock:
            return self._should_recompute(A_csr)

    def _should_recompute(self, A_csr: csr_array | csr_matrix) -> bool:
        # First call - must build
        if (
            self._cached_M is None
            or self._cached_A_data is None
            or self._cached_A_indices is None
            or self._cached_A_indptr is None
        ):
            return True

        # Different system: the cached preconditioner is meaningless for it
        if (
            _get_matrix_shape(A_csr) != self._cached_A_shape
            or A_csr.data.shape != self._cached_A_data.shape
            or not np.array_equal(A_csr.indptr, self._cached_A_indptr)
            or not np.array_equal(A_csr.indices, self._cached_A_indices)
        ):
            return True

        # Frequency-based rebuild
        if self.update_frequency > 0 and self._call_count >= self.update_frequency:
            return True

        # Change-based rebuild (compare matrix coefficients with the Frobenius norm of the difference)
        if self.recompute_threshold > 0:
            diff_norm = np.linalg.norm(A_csr.data - self._cached_A_data)
            old_norm = np.linalg.norm(self._cached_A_data)
            if not np.isfinite(diff_norm) or not np.isfinite(old_norm):
                return True

            if old_norm > 0:
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
        with self._lock:
            self._cached_M = None
            self._cached_A_data = None
            self._cached_A_indices = None
            self._cached_A_indptr = None
            self._cached_A_shape = None
            self._call_count = 0

    def register(self, override: bool = False) -> None:
        """
        Register this cached preconditioner (factory) in the preconditioner factory registry

        :param override: Whether to override existing factories with the same name in the registry
        """
        factory = typing.cast(PreconditionerFactory, self)
        preconditioner_factory(factory, name=self._name, override=override)


_preconditioner_registry_lock = threading.Lock()
PRECONDITIONER_FACTORIES = {
    "cpr": make_cpr_preconditioner,
    "amg": make_amg_preconditioner,
    "ilu": make_ilu_preconditioner,
    "diagonal": make_diagonal_preconditioner,
    "block_jacobi": make_block_jacobi_preconditioner,
    "polynomial": make_polynomial_preconditioner,
}
"""Registered preconditioner factory functions."""

_solver_registry_lock = threading.Lock()
SOLVER_FUNCTIONS = {
    "lgmres": _lgmres,
    "bicg": bicg,
    "bicgstab": bicgstab,
    "qmr": _qmr,
    "tfqmr": tfqmr,
    "gmres": gmres,
    "minres": _minres,
    "cg": cg,
    "cgs": cgs,
    "gcrotmk": gcrotmk,
    "direct": _spsolve,
}
"""Registered solver functions."""


@typing.overload
def preconditioner_factory(func: PreconditionerFactory) -> PreconditionerFactory: ...
@typing.overload
def preconditioner_factory(
    func: None = None,
    name: str | None = None,
    override: bool = False,
) -> typing.Callable[[PreconditionerFactory], PreconditionerFactory]: ...
@typing.overload
def preconditioner_factory(
    func: PreconditionerFactory,
    name: str | None = None,
    override: bool = False,
) -> PreconditionerFactory: ...


def preconditioner_factory(
    func: PreconditionerFactory | None = None,
    name: str | None = None,
    override: bool = False,
) -> PreconditionerFactory | typing.Callable[[PreconditionerFactory], PreconditionerFactory]:
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
        with _preconditioner_registry_lock:
            key = name or getattr(func, "__name__", None)
            if not key:
                raise ValueError(
                    "Preconditioner factory must have a `__name__` attribute or a name must be provided."
                )

            if not override and key in PRECONDITIONER_FACTORIES:
                raise ValueError(
                    f"Preconditioner factory '{key}' is already registered. "
                    f"Use `override=True` to replace it."
                )
            PRECONDITIONER_FACTORIES[key] = func  # type: ignore
        return func

    if func is not None:
        return decorator(func)
    return decorator


def list_preconditioner_factories() -> list[str]:
    """
    List the names of all registered preconditioner factories.

    :return: List of registered preconditioner factory names.
    """
    with _preconditioner_registry_lock:
        return list(PRECONDITIONER_FACTORIES.keys())


def get_preconditioner_factory(name: str) -> PreconditionerFactory:
    """
    Get a registered preconditioner factory by name.

    :param name: Name of the preconditioner factory.
    :return: The corresponding preconditioner factory function.
    :raises ValidationError: If the preconditioner factory is unknown.
    """
    with _preconditioner_registry_lock:
        if name not in PRECONDITIONER_FACTORIES:
            raise ValidationError(
                f"Unknown preconditioner factory: {name!r}. "
                f"Use `@preconditioner_factory` to register new preconditioners. "
                f"Available preconditioners: {list(PRECONDITIONER_FACTORIES.keys())}"
            )
        return PRECONDITIONER_FACTORIES[name]  # type: ignore[return-value]


@typing.overload
def solver_function(func: SolverFunction) -> SolverFunction: ...
@typing.overload
def solver_function(
    func: None = None,
    name: str | None = None,
    override: bool = False,
) -> typing.Callable[[SolverFunction], SolverFunction]: ...
@typing.overload
def solver_function(
    func: SolverFunction,
    name: str | None = None,
    override: bool = False,
) -> SolverFunction: ...


def solver_function(
    func: SolverFunction | None = None,
    name: str | None = None,
    override: bool = False,
) -> SolverFunction | typing.Callable[[SolverFunction], SolverFunction]:
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

    def decorator(func: SolverFunction) -> SolverFunction:
        with _solver_registry_lock:
            key = name or getattr(func, "__name__", None)
            if not key:
                raise ValueError(
                    "Solver function must have a `__name__` attribute or a name must be provided."
                )

            if not override and key in SOLVER_FUNCTIONS:
                raise ValueError(
                    f"Solver function '{key}' is already registered. "
                    f"Use `override=True` to replace it."
                )
            SOLVER_FUNCTIONS[key] = func  # type: ignore[assignment]  # ty:ignore[invalid-assignment]
        return func

    if func is not None:
        return decorator(func)
    return decorator


def list_solver_functions() -> list[str]:
    """
    List the names of all registered solver functions.

    :return: List of registered solver function names.
    """
    with _solver_registry_lock:
        return list(SOLVER_FUNCTIONS.keys())


def get_solver_function(name: str) -> SolverFunction:
    """
    Get a registered solver function by name.

    :param name: Name of the solver function.
    :return: The corresponding solver function.
    :raises ValidationError: If the solver function is unknown.
    """
    with _solver_registry_lock:
        if name not in SOLVER_FUNCTIONS:
            raise ValidationError(
                f"Unknown solver function: {name!r}. "
                f"Use `@solver_function` to register new solvers. "
                f"Available solvers: {list(SOLVER_FUNCTIONS.keys())}"
            )
        return SOLVER_FUNCTIONS[name]  # type: ignore[return-value]  # ty:ignore[invalid-return-type]


def _get_preconditioner(
    A_csr: csr_array | csr_matrix,
    preconditioner: Preconditioner | None,
) -> LinearOperator | None:
    """
    Get or build a preconditioner based on the specification.

    :param A_csr: The coefficient matrix in CSR format.
    :param preconditioner: Preconditioner specification as a string, callable, or None
    :return: A SciPy `LinearOperator` representing the preconditioner, or None.
    :raises ValidationError: If the preconditioner type is unknown.
    """
    if preconditioner is None or isinstance(preconditioner, LinearOperator):
        return preconditioner
    elif isinstance(preconditioner, str):
        if preconditioner in PRECONDITIONER_FACTORIES:
            return PRECONDITIONER_FACTORIES[preconditioner](A_csr)  # type: ignore[operator]
        raise ValidationError(
            f"Unknown preconditioner type: {preconditioner!r}. Available preconditioners: {list(PRECONDITIONER_FACTORIES.keys())}"
        )
    elif callable(preconditioner):
        return typing.cast(PreconditionerFactory, preconditioner)(A_csr)
    raise ValidationError(
        f"`preconditioner` must be None, a name, a `LinearOperator` or a factory function, "
        f"got {type(preconditioner).__name__}."
    )


def _get_solver_function(
    solver: Solver | typing.Iterable[Solver],
) -> list[SolverFunction]:
    """
    Get solver functions from a solver specification.

    :param solver: Solver specification as a string, a callable, or an ordered sequence of
        strings and callables.
    :return: List of solver functions corresponding to the specification, in order.
    :raises ValidationError: If any solver type is unknown, the sequence is empty, or the
        sequence is unordered.
    :raises TypeError: If the solver specification is of an invalid type.
    """

    def resolve(item: typing.Any) -> SolverFunction:
        if isinstance(item, str):
            if item in SOLVER_FUNCTIONS:
                return SOLVER_FUNCTIONS[item]  # type: ignore[return-value]  # ty:ignore[invalid-return-type]
            raise ValidationError(
                f"Unknown solver type: {item!r}. Available solvers: {list(SOLVER_FUNCTIONS.keys())}"
            )
        if callable(item):
            return item  # type: ignore[return-value]  # ty:ignore[invalid-return-type]
        raise ValidationError(f"Unknown solver type: {item!r}")

    if isinstance(solver, str) or callable(solver):
        return [resolve(solver)]
    if isinstance(solver, (set, frozenset)):
        raise ValidationError(
            "`solver` must be an ordered sequence, since solvers are tried in order. "
            "Use a list or tuple instead of a set."
        )
    if isinstance(solver, typing.Iterable):
        solver_functions = [resolve(item) for item in solver]
        if not solver_functions:
            raise ValidationError("`solver` must contain at least one solver.")
        return solver_functions
    raise TypeError("`solver` must be a string, callable, or a sequence of strings or callables.")


def _supports_transpose(operator: LinearOperator) -> bool:
    """Whether `operator` can multiply by its transpose, which `bicg` and `qmr` require."""
    try:
        operator.rmatvec(np.zeros(operator.shape[0], dtype=operator.dtype))
    except NotImplementedError:
        return False
    return True


def solve_linear_system(
    A_csr: csr_array | csr_matrix,
    b: npt.NDArray,
    maximum_iterations: int,
    rtol: float | None = None,
    atol: float | None = None,
    solver: Solver | typing.Iterable[Solver] = "bicgstab",
    preconditioner: Preconditioner | None = "ilu",
    fallback_to_direct: bool = False,
) -> tuple[npt.NDArray, LinearOperator | None]:
    """
    Solves the linear system A·x = b using an (iterative) solver with a fallback strategy.

    A solver is accepted only if it reports convergence, its solution is finite, and the
    true residual `||b - A·x||` is within `RESIDUAL_VERIFICATION_FACTOR` times the
    requested tolerance `max(rtol·||b||, atol)`. Otherwise the next solver is tried. The
    preconditioner is built once and shared by all iterative solvers in the sequence.

    :param A_csr: Coefficient matrix in CSR format.
    :param b: Right-hand side vector.
    :param maximum_iterations: Iteration limit passed to each solver. Most solvers count
        single iterations. "gmres", "lgmres" and "gcrotmk" count outer (restart) cycles,
        each of which performs many inner iterations.
    :param rtol: Relative tolerance for convergence. Defaults to 1e-6.
    :param atol: Absolute tolerance for convergence. Defaults to the larger of 1e-8 and
        1e-6 times the norm of `b`.
    :param solver: (Iterative) solver or sequence of solvers to use ("bicgstab", "gmres",
        "lgmres", "tfqmr", "direct", and the other registered names), or custom
        callable(s). If a sequence is provided, solvers will be tried in order until one
        converges. "cg" and "minres" require a symmetric matrix and a symmetric positive
        definite preconditioner. "bicg" and "qmr" require a preconditioner that supports
        transposed products, which the "amg" and "cpr" preconditioners do not, so those
        solvers are skipped when used with them.
    :param preconditioner: Type of preconditioner to use ("ilu", "amg", "diagonal",
        "block_jacobi", "polynomial", "cpr"), or None. Can also be a `LinearOperator`, or
        a preconditioner factory function that takes A and returns a preconditioner.
        If None, no preconditioning is applied. It is ignored by the "direct" solver.
    :param fallback_to_direct: Whether to fall back to a direct solver if all iterative solvers fail.
        Not suitable for large or production use cases due to performance and memory constraints.
    :return: A tuple (x, M) where x is the solution vector and M is the preconditioner used,
        or None if none was used.
    :raises ValidationError: If the inputs are inconsistent, a solver or preconditioner
        name is unknown, or no solver in the sequence can use the preconditioner.
    :raises PreconditionerError: If the preconditioner cannot be built.
    :raises SolverError: If every solver fails to produce an acceptable solution.
    """
    solver_functions = _get_solver_function(solver)

    b = np.asarray(b)
    matrix_shape = _get_square_matrix_shape(A_csr)
    number_of_equations = matrix_shape[0]
    if b.shape != (number_of_equations,):
        raise ValidationError(
            f"Right-hand side has shape {b.shape}, expected {(number_of_equations,)}."
        )
    if not np.all(np.isfinite(b)):
        raise ValidationError("Right-hand side contains non-finite values.")

    M = None
    if any(function is not _spsolve for function in solver_functions):
        try:
            M = _get_preconditioner(A_csr, preconditioner)
        except ValidationError:
            raise
        except Exception as exc:
            raise PreconditionerError(f"Error building preconditioner: {exc}") from exc

        if M is not None and not _supports_transpose(M):
            usable = [function for function in solver_functions if function not in (bicg, _qmr)]
            if not usable:
                raise ValidationError(
                    "The 'bicg' and 'qmr' solvers need a preconditioner that supports transposed "
                    "products, which the chosen preconditioner does not. Use a different solver "
                    "or preconditioner."
                )
            if len(usable) != len(solver_functions):
                logger.warning(
                    "Skipping the 'bicg' and 'qmr' solvers since the preconditioner does not "
                    "support transposed products."
                )
            solver_functions = usable

    b_norm = np.linalg.norm(b)
    # Too strict
    # epsilon = get_floating_point_info().eps
    # rtol = rtol if rtol is not None else float(epsilon * 50)
    # atol = atol if atol is not None else float(max(1e-8, 20 * epsilon * b_norm))
    rtol = rtol if rtol is not None else 1e-6
    atol = atol if atol is not None else float(max(1e-8, 1e-6 * b_norm))
    accepted_residual = max(rtol * b_norm, atol) * RESIDUAL_VERIFICATION_FACTOR

    is_direct = all(function is _spsolve for function in solver_functions)
    for current_solver in solver_functions:
        function_name = getattr(current_solver, "__name__", repr(current_solver))
        x, info = current_solver(
            A=A_csr,
            b=b,
            x0=None,
            M=None if current_solver is _spsolve else M,
            rtol=rtol,
            atol=atol,
            maxiter=maximum_iterations,
            callback=None,
        )
        if info > 0:
            logger.warning(
                f"Solver {function_name!r} did not converge within {maximum_iterations} iterations. Info: {info}"
            )
            continue
        if info < 0:
            logger.warning(
                f"Solver {function_name!r} broke down or received illegal input. Info: {info}"
            )
            continue
        if not np.all(np.isfinite(x)):
            logger.warning(f"Solver {function_name!r} returned a non-finite solution.")
            continue

        residual_norm = np.linalg.norm(b - A_csr @ x)
        if residual_norm > accepted_residual:
            logger.warning(
                f"Solver {function_name!r} reported convergence but its true residual "
                f"{residual_norm:.3e} exceeds the accepted residual {accepted_residual:.3e}."
            )
            continue
        return np.ascontiguousarray(x), M

    if not fallback_to_direct or is_direct:
        raise SolverError("All solvers failed to produce an acceptable solution.")

    logger.info("Falling back to direct solver (spsolve).")
    if number_of_equations > 100000:
        logger.warning(
            f"Direct solver on large system ({number_of_equations:,} DOFs) may be slow and consume excessive "
            f"memory. Consider improving iterative solver settings or preconditioner."
        )

    x, info = _spsolve(
        A=A_csr,
        b=b,
        x0=None,
        M=None,
        rtol=rtol,
        atol=atol,
        maxiter=maximum_iterations,
        callback=None,
    )
    if info != 0:
        logger.error("Direct solver failed: the matrix is singular or the solution is not finite.")
        raise SolverError("All iterative solvers and direct solver failed to solve the system.")
    return np.ascontiguousarray(x), None


@numba.njit(cache=True, error_model="numpy")
def _scale_rows(
    indptr: npt.NDArray,
    data: npt.NDArray,
    epsilon: float,
    inverse_row_scale: npt.NDArray,
) -> None:
    """Divides each row of a CSR matrix by its sum of absolute values, in place."""
    for row in range(indptr.shape[0] - 1):
        total = 0.0
        for position in range(indptr[row], indptr[row + 1]):
            total += abs(data[position])
        scale = 1.0 / total if total >= epsilon else 1.0
        inverse_row_scale[row] = scale
        for position in range(indptr[row], indptr[row + 1]):
            data[position] *= scale


@numba.njit(cache=True, error_model="numpy")
def _largest_column_entries(
    indices: npt.NDArray,
    data: npt.NDArray,
    largest: npt.NDArray,
) -> None:
    """Records the largest absolute entry of each column of a CSR matrix in `largest`."""
    for position in range(data.shape[0]):
        magnitude = abs(data[position])
        if magnitude > largest[indices[position]]:
            largest[indices[position]] = magnitude


@numba.njit(cache=True, error_model="numpy")
def _scale_columns(
    indices: npt.NDArray,
    data: npt.NDArray,
    inverse_column_scale: npt.NDArray,
) -> None:
    """Multiplies each entry of a CSR matrix by the scale of its column, in place."""
    for position in range(data.shape[0]):
        data[position] *= inverse_column_scale[indices[position]]


SystemScalingMethod = typing.Literal["row", "column", "diagonal"]


def scale_linear_system(
    jacobian_csr: csr_array | csr_matrix,
    residual_vector: npt.NDArray[np.floating],
    methods: SystemScalingMethod | typing.Iterable[SystemScalingMethod] | None = None,
    epsilon: float = 1e-12,
) -> tuple[
    csr_array | csr_matrix,
    npt.NDArray[np.floating],
    npt.NDArray[np.floating] | None,
]:
    """
    Scale a linear system to improve numerical conditioning.

    This function applies row scaling and/or column scaling to the
    Jacobian matrix and residual vector. It is designed for use in Newton-based
    solvers for nonlinear systems (e.g., multiphase flow transport equations).
    The inputs are not modified.

    Scaling improves solver stability by normalizing equation magnitudes
    (row scaling) and/or variable sensitivities (column scaling).

    Supported scaling methods:
    - "row": Row scaling by the sum of absolute values in each row (row 1-norm)
    - "column": Column scaling by the largest absolute value in each column
    - "diagonal": Column scaling by the Jacobian diagonal entries, which gives the
      scaled matrix a unit diagonal. This is only sound when every diagonal entry is a
      good measure of the size of its column. A small diagonal entry makes its whole
      column large, which can worsen conditioning by many orders of magnitude and make
      preconditioned solvers fail, so "column" is the safer choice.

    If no methods are provided, ("row", "column") is used.

    Notes:
    - Row scaling modifies both J and R
    - Column scaling modifies only J
    - Column scaling requires post-solve unscaling of the solution vector
    - Methods are applied in the order given, and each method sees the matrix as left
      by the previous one.

    :param jacobian_csr: Sparse Jacobian matrix in CSR format of shape (n, n)
    :param residual_vector: Residual vector of shape (n,)
    :param methods: Scaling methods to apply. Can be:
        - None (auto-select)
        - "row"
        - "column"
        - "diagonal"
        - Iterable like ("row", "column")
        Order matters (applied sequentially)

    :param epsilon: Rows, columns and diagonal entries with a magnitude below this are
        left unscaled to avoid dividing by near-zero values.
    :return: Tuple of:
        - scaled jacobian (CSR matrix)
        - scaled residual (ndarray)
        - column scaling vector (ndarray or None)

        The `column_scaling_vector` must be used to unscale the solution:
            dx = dx_scaled * column_scaling_vector
    :raises ValidationError: If a scaling method is unknown.
    """
    J = jacobian_csr.tocsr(copy=True)
    if not np.issubdtype(J.dtype, np.floating):
        J = J.astype(np.float64)

    R = residual_vector
    n = J.shape[0]

    if methods is None:
        selected: tuple[str, ...] = ("row", "column")
    elif isinstance(methods, str):
        selected = (methods,)
    else:
        selected = tuple(methods)

    unknown = [method for method in selected if method not in ("row", "column", "diagonal")]
    if unknown:
        raise ValidationError(
            f"Unknown scaling method(s): {unknown}. Use 'row', 'column' and/or 'diagonal'."
        )

    column_scaling_vector = None
    for method in selected:
        if method == "row":
            inverse_row_scale = np.empty(n, dtype=J.dtype)
            _scale_rows(
                indptr=J.indptr, data=J.data, epsilon=epsilon, inverse_row_scale=inverse_row_scale
            )
            R = R * inverse_row_scale
            continue

        if method == "column":
            column_size = np.zeros(J.shape[1], dtype=J.dtype)  # type: ignore
            _largest_column_entries(indices=J.indices, data=J.data, largest=column_size)
        else:
            column_size = np.abs(J.diagonal())
        column_size[column_size < epsilon] = 1.0
        inverse_column_scale = (1.0 / column_size).astype(J.dtype, copy=False)
        _scale_columns(indices=J.indices, data=J.data, inverse_column_scale=inverse_column_scale)
        column_scaling_vector = (
            inverse_column_scale
            if column_scaling_vector is None
            else column_scaling_vector * inverse_column_scale
        )

    return J, R, column_scaling_vector  # type: ignore[return-value]  # ty:ignore[invalid-return-type]

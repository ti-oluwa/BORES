import logging
import threading
import typing

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
from bores.precision import get_floating_point_info
from bores.types import (
    Preconditioner,
    PreconditionerFactory,
    Solver,
    SolverFunction,
)

logger = logging.getLogger(__name__)


__all__ = [
    "CachedPreconditionerFactory",
    "get_preconditioner_factory",
    "get_solver_function",
    "list_preconditioner_factories",
    "list_solver_functions",
    "make_amg_preconditioner",
    "make_block_jacobi_preconditioner",
    "make_cpr_preconditioner",
    "make_diagonal_preconditioner",
    "make_ilu_preconditioner",
    "make_polynomial_preconditioner",
    "preconditioner_factory",
    "solve_linear_system",
    "solver_function",
]


def make_amg_preconditioner(
    A_csr: csr_array | csr_matrix, cycle: str = "V", **kwargs: typing.Any
) -> LinearOperator:
    """
    Creates an Algebraic Multigrid (AMG) preconditioner using PyAMG.

    :param A_csr: The coefficient matrix in CSR format.
    :param cycle: Multigrid cycle type ('V', 'W', 'F'). V-cycle is standard,
        W-cycle offers better convergence for difficult problems but is more expensive.
    :param kwargs: Additional arguments for `pyamg.smoothed_aggregation_solver`.
    :return: A SciPy `LinearOperator` that represents the AMG preconditioner.
    """
    solver = pyamg.smoothed_aggregation_solver(A_csr, **kwargs)
    return solver.aspreconditioner(cycle=cycle)


def make_diagonal_preconditioner(
    A_csr: csr_array | csr_matrix,
) -> LinearOperator:
    """
    Creates a diagonal preconditioner from the coefficient matrix.

    :param A_csr: The coefficient matrix in CSR format.
    :return: A SciPy `LinearOperator` that represents the diagonal preconditioner.
    """
    diagonal_elements = A_csr.diagonal()
    # Avoid division by zero by replacing zeros with a small number
    # Use precision-adaptive threshold for float32/float64 compatibility
    epsilon = get_floating_point_info().eps
    threshold = max(1e-10, 100 * epsilon)
    diagonal_elements = np.where(np.abs(diagonal_elements) < threshold, 1.0, diagonal_elements)
    M_diagional = diags(1.0 / diagonal_elements, format="csr")
    return LinearOperator(shape=A_csr.shape, matvec=M_diagional.dot)  # type: ignore[arg-type]


def make_block_jacobi_preconditioner(
    A_csr: csr_array | csr_matrix,
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
    n_blocks = n // block_size

    # Pre-compute inverse of diagonal blocks
    block_inverses = []
    epsilon = get_floating_point_info().eps
    threshold = max(1e-10, 100 * epsilon)

    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size

        # Extract diagonal block
        block = A_csr[start_idx:end_idx, start_idx:end_idx].toarray()  # type: ignore

        try:
            # Try to invert the block
            block_inverse = np.linalg.inv(block)
            block_inverses.append(block_inverse)
        except np.linalg.LinAlgError:
            # Block is singular, fall back to diagonal approximation
            diagonal = np.diag(block)
            diagonal = np.where(np.abs(diagonal) < threshold, 1.0, diagonal)
            block_inverse = np.diag(1.0 / diagonal)
            block_inverses.append(block_inverse)

    # Handle remainder cells (if n is not divisible by block_size)
    remainder_start = n_blocks * block_size
    if remainder_start < n:
        remainder_block = A_csr[remainder_start:n, remainder_start:n].toarray()  # type: ignore
        try:
            remainder_inv = np.linalg.inv(remainder_block)
            block_inverses.append(remainder_inv)
        except np.linalg.LinAlgError:
            diagonal = np.diag(remainder_block)
            diagonal = np.where(np.abs(diagonal) < threshold, 1.0, diagonal)
            remainder_inverse = np.diag(1.0 / diagonal)
            block_inverses.append(remainder_inverse)

    def matvec(x: npt.NDArray) -> npt.NDArray:
        """Apply Block Jacobi preconditioner."""
        y = np.zeros_like(x)

        # Apply block inversions
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            y[start_idx:end_idx] = block_inverses[i] @ x[start_idx:end_idx]

        # Handle remainder
        if remainder_start < n:
            y[remainder_start:n] = block_inverses[n_blocks] @ x[remainder_start:n]
        return y

    return LinearOperator(shape=A_csr.shape, matvec=matvec)  # type: ignore[arg-type]


def make_ilu_preconditioner(A_csr: csr_array | csr_matrix, **kwargs: typing.Any) -> LinearOperator:
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


def make_polynomial_preconditioner(
    A_csr: csr_array | csr_matrix,
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
    diagional = A_csr.diagonal()
    normalized_diagonal = np.max(np.abs(diagional))
    alpha = 1.0 / normalized_diagonal if normalized_diagonal > 1e-10 else 1.0

    # Pre-compute polynomial terms: I, αA, α²A², ...
    identity_term = np.ones(A_csr.shape[0])
    terms = [identity_term]

    current_term = alpha
    A_power = A_csr
    for i in range(degree):
        terms.append(current_term)  # type: ignore
        current_term *= alpha
        if i < degree - 1:
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
    amg_kwargs: dict[str, typing.Any] | None = None,
    ilu_kwargs: dict[str, typing.Any] | None = None,
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

    *Notes*
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

    amg_kwargs = amg_kwargs or dict(CPR_AMG_KWARGS)
    ilu_kwargs = ilu_kwargs or dict(CPR_ILU_KWARGS)
    number_of_equations = A_csr.shape[0]
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
    A_pp = A_csr[pressure_dof_indices, :][:, pressure_dof_indices].tocsr()

    # Build AMG preconditioner for A_pp
    try:
        M_amg = make_amg_preconditioner(A_pp, **amg_kwargs)  # type: ignore
    except Exception as exc:
        raise PreconditionerError(f"AMG construction for pressure block failed: {exc}") from exc

    # Build ILU preconditioner for the full matrix
    try:
        M_ilu = make_ilu_preconditioner(A_csr, **ilu_kwargs)
    except Exception as exc:
        raise PreconditionerError(f"ILU factorization failed for CPR: {exc}") from exc

    # Restriction and prolongation operators (implicit)
    def restrict_to_pressure(vector: npt.NDArray) -> npt.NDArray:
        return vector[pressure_dof_indices]

    def prolongate_to_full(pressure_vector: npt.NDArray) -> npt.NDArray:
        out = np.zeros(number_of_equations, dtype=pressure_vector.dtype)
        out[pressure_dof_indices] = pressure_vector
        return out

    def matvec(residual: npt.NDArray) -> npt.NDArray:
        """CPR preconditioner application: x = M^{-1} r"""
        # Stage 1: pressure solve
        r_p = restrict_to_pressure(residual)
        try:
            z_p = M_amg.dot(r_p)
        except TypeError:
            z_p = M_amg(r_p)

        z = prolongate_to_full(z_p)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

        # Stage 2: ILU correction
        w = residual - A_csr.dot(z)
        y = M_ilu.dot(w)
        return z + y

    return LinearOperator(shape=A_csr.shape, matvec=matvec)  # type: ignore[arg-type]


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
    """Direct (SPSOLVE) solver wrapper compatible with the standard `SolverFunction` interface"""
    return spsolve(A, b), 0  # type: ignore[return-value]


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

    MINRES is only suitable for symmetric (possibly indefinite) systems.  It
    does **not** accept `atol` directly; instead convergence is declared when
    the *relative* residual `||r_k|| / ||b||` drops below `rtol`.  To
    honour the caller's `atol` we tighten `rtol` whenever the absolute
    criterion is the binding one:

        effective_rtol = min(rtol, atol / max(||b||, ε))

    This guarantees the returned solution satisfies *both* tolerances without
    exposing the mismatch to callers.

    :param shift: Solves `(A - shift·I) x = b` instead of `A x = b`.
        Useful when A is nearly singular or when targeting a shifted system
        (e.g. eigenvalue-deflation tricks). Defaults to 0.0 (no shift).
    """
    b_array: npt.NDArray = np.asarray(b)
    b_norm = np.linalg.norm(b_array)
    eps = np.finfo(b_array.dtype).eps if np.issubdtype(b_array.dtype, np.floating) else 1e-15

    # Derive a single effective rtol that covers both the relative and
    # absolute stopping criterion requested by the caller.
    denominator = max(b_norm, eps)
    effective_rtol = min(rtol, atol / denominator)
    # MINRES requires rtol > 0; clamp to a safe floor.
    effective_rtol = max(effective_rtol, eps * 10)
    return minres(  # type: ignore[return-value]
        A,
        b,
        x0=x0,
        shift=shift,
        M=M,
        rtol=effective_rtol,
        maxiter=maxiter or 1000,
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
    The combined effect is equivalent to `M ≈ M1 @ M2`.

    **Mapping strategy**:

    * If `M` is `None`, both `M1` and `M2` are left as `None`
      (no preconditioning).
    * If `M` is provided, it is supplied as `M1`; `M2` is set to the
      identity.  This matches the "left-preconditioned" convention used by the
      rest of the solver infrastructure and keeps the residual norms comparable
      with those reported by other solvers.

    **`atol` handling**:

    Like MINRES, QMR has no native `atol` parameter.  We fold it into a
    tightened `rtol` using the same formula:

        effective_rtol = min(rtol, atol / max(||b||, ε))
    """
    b_array: npt.NDArray = np.asarray(b)
    b_norm = np.linalg.norm(b_array)
    eps = np.finfo(b_array.dtype).eps if np.issubdtype(b_array.dtype, np.floating) else 1e-15

    denominator = max(b_norm, eps)
    effective_rtol = min(rtol, atol / denominator)
    effective_rtol = max(effective_rtol, eps * 10)

    # Split the single preconditioner into the (M1, M2) pair expected by QMR.
    # Using M as M1 (left preconditioner) and leaving M2=None (identity) is the
    # conventional choice and ensures the *left*-preconditioned residual is
    # what gets driven to zero, consistent with the rest of this codebase.
    M1: typing.Any | None = M
    M2: typing.Any | None = None
    return qmr(  # type: ignore[return-value]
        A,
        b,
        x0=x0,
        M1=M1,
        M2=M2,
        rtol=effective_rtol,
        maxiter=maxiter or 1000,
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

        self._cached_M: LinearOperator | None = None
        self._cached_A_data: npt.NDArray | None = None
        self._call_count = 0

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, A_csr: csr_array | csr_matrix) -> LinearOperator | None:
        """
        Get preconditioner, reusing cache if possible.

        :param A_csr: Current coefficient matrix
        :return: Preconditioner (cached or newly built)
        """
        should_rebuild = self.should_recompute(A_csr)
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

    def should_recompute(self, A_csr: csr_array | csr_matrix) -> bool:
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
        with _preconditoner_registry_lock:
            key = name or getattr(func, "__name__", None)
            if not key:
                raise ValueError(
                    "Preconditioner factory  must have a `__name__` attribute or a name must be provided."
                )

            if not override and key in PRECONDITIONER_FACTORIES:
                raise ValueError(
                    f"Preconditioner factory '{name}' is already registered. "
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
    with _preconditoner_registry_lock:
        return list(PRECONDITIONER_FACTORIES.keys())


def get_preconditioner_factory(name: str) -> PreconditionerFactory:
    """
    Get a registered preconditioner factory by name.

    :param name: Name of the preconditioner factory.
    :return: The corresponding preconditioner factory function.
    :raises ValidationError: If the preconditioner factory is unknown.
    """
    with _preconditoner_registry_lock:
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
                    "Solver function  must have a `__name__` attribute or a name must be provided."
                )

            if not override and key in SOLVER_FUNCTIONS:
                raise ValueError(
                    f"Solver function '{name}' is already registered. "
                    f"Use `override=True` to replace it."
                )
            SOLVER_FUNCTIONS[key] = func  # ty:ignore[invalid-assignment]
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


def get_solver_function(name: str) -> SolverFunction | None:
    """
    Get a registered solver function by name.

    :param name: Name of the solver function.
    :return: The corresponding solver function.
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
    if isinstance(preconditioner, (type(None), LinearOperator)):
        return preconditioner
    elif isinstance(preconditioner, str):
        if preconditioner in PRECONDITIONER_FACTORIES:
            preconditioner_factory = PRECONDITIONER_FACTORIES[preconditioner]
            M = preconditioner_factory(A_csr)  # type: ignore[operator]
            return M
        else:
            raise ValidationError(
                f"Unknown preconditioner type: {preconditioner!r}. Available preconditioners: {list(PRECONDITIONER_FACTORIES.keys())}"
            )
    elif callable(preconditioner):
        preconditioner_factory = typing.cast(PreconditionerFactory, preconditioner)
        M = preconditioner_factory(A_csr)
        return M
    return preconditioner  # type: ignore[return-value]


def _get_solver_function(
    solver: Solver | typing.Iterable[Solver],
) -> list[SolverFunction]:
    """
    Get solver functions from a solver specification.

    :param solver: Solver specification as a string or sequence of strings/callables.
    :return: List of solver functions corresponding to the specification.
    :raises ValidationError: If any solver type is unknown.
    :raises TypeError: If the solver specification is of an invalid type.
    """
    if isinstance(solver, str):
        if solver in SOLVER_FUNCTIONS:
            solver_function = SOLVER_FUNCTIONS[solver]
            if isinstance(solver_function, (list, tuple)):
                return list(solver_function)  # type: ignore[return-value]  # ty:ignore[invalid-return-type, invalid-argument-type]
            return [solver_function]  # type: ignore[return-value]  # ty:ignore[invalid-return-type]
        raise ValidationError(
            f"Unknown solver type: {solver!r}. Available solvers: {list(SOLVER_FUNCTIONS.keys())}"
        )
    elif callable(solver):
        return [solver]  # type: ignore[return-value]  # ty:ignore[invalid-return-type]
    elif isinstance(solver, (list, tuple, set)):
        solver_functions = []
        for s in solver:
            if isinstance(s, str) and s in SOLVER_FUNCTIONS:
                solver_functions.append(SOLVER_FUNCTIONS[s])
            elif callable(s):
                solver_functions.append(s)
            else:
                raise ValidationError(f"Unknown solver type in sequence: {s!r}")
        return solver_functions
    raise TypeError("`solver` must be a string, callable, or a sequence of strings.")


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

    Preconditioning is applied using a diagonal preconditioner derived from the diagonal
    elements of matrix A to improve convergence.

    :param A: Coefficient matrix in CSR format.
    :param b: Right-hand side vector.
    :param maximum_iterations: Maximum number of iterations for each solver.
    :param solver: (Iterative) solver or sequence of solvers to use ("bicgstab", "gmres",
        "lgmres", "tfqmr"), or custom callable(s). If a sequence is provided, solvers will
        be tried in order until one converges.
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
    solver_functions = _get_solver_function(solver)
    is_direct = _spsolve in solver_functions
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

    for solver_function in solver_functions:
        x, info = solver_function(
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
                f"Solver {solver_function!r} failed to converge within {maximum_iterations} iterations. Info: {info}"
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


SystemScalingMethod = typing.Literal["row", "diagonal"]


def scale_linear_system(
    jacobian_csr: csr_matrix,
    residual_vector: npt.NDArray[np.floating],
    methods: SystemScalingMethod | list[SystemScalingMethod] | None = None,
    epsilon: float = 1e-12,
) -> tuple[
    csr_matrix,
    npt.NDArray[np.floating],
    npt.NDArray[np.floating] | None,
]:
    """
    Scale a linear system to improve numerical conditioning.

    This function applies row scaling and/or column (diagonal) scaling to the
    Jacobian matrix and residual vector. It is designed for use in Newton-based
    solvers for nonlinear systems (e.g., multiphase flow transport equations).

    Scaling improves solver stability by normalizing equation magnitudes
    (row scaling) and/or variable sensitivities (column scaling).

    Supported scaling methods:
    - "row": Row-infinity norm scaling (max absolute value per row)
    - "diagonal": Diagonal (column) scaling using Jacobian diagonal entries

    If no methods are provided, an automatic heuristic is used:
    - If the Jacobian diagonal is sufficiently strong -> use ("row", "diagonal")
    - Otherwise -> use ("row",) only

    Notes:
    - Row scaling modifies both J and R
    - Column scaling modifies only J
    - Column scaling requires post-solve unscaling of the solution vector

    :param jacobian_csr: Sparse Jacobian matrix in CSR format of shape (n, n)
    :param residual_vector: Residual vector of shape (n,)
    :param methods: Scaling methods to apply. Can be:
        - None (auto-select)
        - "row"
        - "diagonal"
        - Iterable like ("row", "diagonal")
        Order matters (applied sequentially)

    :param epsilon: Small threshold to prevent division by zero or near-zero values
    :return: Tuple of:
        - scaled jacobian (CSR matrix)
        - scaled residual (ndarray)
        - column scaling vector (ndarray or None)

        The column_scaling_vector must be used to unscale the solution:
            dx = dx_scaled * column_scaling_vector
    """
    J = jacobian_csr
    R = residual_vector

    # Auto method selection
    abs_diagonal = None
    if methods is None:
        abs_diagonal = np.abs(J.diagonal())
        if np.median(abs_diagonal) > epsilon:
            methods = ("row", "diagonal")  # type: ignore
        else:
            methods = ("row",)  # type: ignore

    if isinstance(methods, str):
        methods = (methods,)  # type: ignore

    column_scaling_vector = None

    # Row scaling (D_r * J, D_r * R)
    if "row" in methods:  # type: ignore
        row_max = np.abs(J).sum(axis=1).A.ravel()  # type: ignore
        row_max[row_max < epsilon] = 1.0
        inv_row = 1.0 / row_max

        J = diags(inv_row) @ J
        R = R * inv_row

    # Column (diagonal) scaling (J * D_c)
    if "diagonal" in methods:  # type: ignore
        abs_diagonal = abs_diagonal if abs_diagonal is not None else np.abs(J.diagonal())
        abs_diagonal[abs_diagonal < epsilon] = 1.0
        inverse_diagional_scale = 1.0 / abs_diagonal

        J = J @ diags(inverse_diagional_scale)
        column_scaling_vector = inverse_diagional_scale

    return J, R, column_scaling_vector  # type: ignore[return-value]  # ty:ignore[invalid-return-type]

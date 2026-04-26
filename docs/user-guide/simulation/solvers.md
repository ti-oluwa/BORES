# Solvers

## Overview

At each time step, the pressure equation (and in the implicit scheme, the saturation equation) is discretized into a sparse linear system $A \cdot x = b$, where $A$ is a large, sparse matrix representing the transmissibilities between grid cells, $x$ is the unknown (pressure or saturation), and $b$ is the right-hand side containing source terms and accumulation. The solver is the algorithm that finds $x$ given $A$ and $b$.

BORES provides several iterative solvers from SciPy's sparse linear algebra library, plus a direct solver for small problems. You select the solver through the `pressure_solver` and `transport_solver` parameters in the `Config`. The same solver types are available for both systems, but you can (and often should) use different solvers for pressure and saturation because they have different mathematical properties.

```python
import bores

config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",
    pressure_solver="bicgstab",      # Solver for pressure
    transport_solver="bicgstab",    # Solver for saturation
)
```

---

## Available Solvers

| Solver | String Name | Best For |
|---|---|---|
| BiCGSTAB | `"bicgstab"` | General-purpose (default). Works well for non-symmetric systems. |
| GMRES | `"gmres"` | Non-symmetric systems, especially with good preconditioners. |
| LGMRES | `"lgmres"` | Like GMRES but with lower memory usage. Good for large systems. |
| CG | `"cg"` | Symmetric positive definite systems only. Fastest when applicable. |
| CGS | `"cgs"` | Non-symmetric systems. Sometimes faster than BiCGSTAB. |
| TFQMR | `"tfqmr"` | Non-symmetric systems. Smooth convergence, good for stiff problems. |
| Direct | `"direct"` | Small systems only. Exact solution (no iteration). |

### BiCGSTAB (Default)

BiCGSTAB (Biconjugate Gradient Stabilized) is the default solver for both pressure and saturation. It works well for the non-symmetric matrices that arise in reservoir simulation (non-symmetry comes from upwinding in the saturation equation and from certain boundary conditions).

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pressure_solver="bicgstab",
    transport_solver="bicgstab",
)
```

BiCGSTAB converges in a predictable number of iterations for well-conditioned systems and combines well with ILU or AMG preconditioners. It is a safe, reliable choice for most problems.

### GMRES

GMRES (Generalized Minimal Residual) is an alternative that sometimes converges faster than BiCGSTAB, especially for highly non-symmetric systems. Its main disadvantage is memory usage: GMRES stores a history of all previous search directions, which grows linearly with the number of iterations. For very large systems (millions of unknowns), this memory cost can become significant.

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pressure_solver="gmres",
)
```

### LGMRES

LGMRES is a variant of GMRES that uses a fixed-size memory window, making it more memory-efficient for large problems while retaining most of the convergence benefits of full GMRES.

### CG

CG (Conjugate Gradient) is the fastest solver for symmetric positive definite (SPD) matrices. The pressure matrix in single-phase flow is SPD, so CG is an excellent choice for pressure-only problems or depletion studies. However, in multiphase flow with upwinding or certain boundary conditions, the matrix may not be symmetric, causing CG to fail or diverge.

```python
# CG for pressure (works well for SPD pressure matrices)
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pressure_solver="cg",
    transport_solver="bicgstab",  # Saturation matrix is not SPD
)
```

!!! warning "CG Requires Symmetric Matrices"

    Only use CG for the pressure solver, and only when you are confident the pressure matrix is symmetric positive definite. If the solver diverges or produces incorrect results, switch back to BiCGSTAB. The saturation transport matrix is almost never symmetric, so do not use CG for `transport_solver`.

### Direct Solver

The direct solver uses sparse LU factorization (via `spsolve`) to compute the exact solution. It requires no iteration and no convergence tolerance. The trade-off is that direct solvers have $O(N^{1.5})$ to $O(N^2)$ memory and time complexity for sparse matrices, making them impractical for large grids.

```python
# Direct solver for small models (< 5000 cells)
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pressure_solver="direct",
    transport_solver="direct",
)
```

The direct solver is useful for small models (fewer than ~5,000 cells) where the overhead of iterative methods is not justified, for debugging (to verify that iterative solver issues are not causing errors), and as a reference solution to validate iterative solver accuracy.

---

## Solver Chains

You can specify a list of solvers to try in sequence. If the first solver fails to converge, BORES automatically tries the next one. This provides a fallback mechanism for difficult problems.

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pressure_solver=["bicgstab", "gmres", "direct"],
    transport_solver=["bicgstab", "tfqmr"],
)
```

In this example, BORES first tries BiCGSTAB for the pressure equation. If BiCGSTAB does not converge within `maximum_solver_iterations`, it tries GMRES. If GMRES also fails, it falls back to the direct solver. This is particularly useful for simulations where the pressure matrix conditioning varies over time (e.g., when wells switch controls or when a gas front arrives at a producer).

Solver chains add robustness at the cost of potentially longer solve times on difficult steps. For most simulations, a single solver (`"bicgstab"`) with a good preconditioner is sufficient.

---

## Custom Solvers

You can register custom solver functions using the `@solver_func` decorator. A custom solver must follow the SciPy solver interface:

```python
from bores.solvers.base import solver_func

@solver_func(name="my_custom_solver")
def my_solver(A, b, x0=None, *, rtol=1e-6, atol=0.0, maxiter=None, M=None, callback=None):
    # Your solver implementation here
    # Must return the solution array x
    ...

# Then use it in Config
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pressure_solver="my_custom_solver",
)
```

The `@solver_func` decorator registers your function in the global solver registry, making it available by name in the `Config`. You can list all registered solvers with `bores.solvers.base.list_solver_funcs()`.

---

## Solver Tuning

The key parameters that affect solver performance are:

**`pressure_convergence_tolerance`** (default: `1e-6`): The relative residual tolerance for the pressure solver. Tighter tolerances give more accurate pressure but require more iterations. For most simulations, `1e-6` is a good balance. If you notice pressure oscillations, try tightening to `1e-8`. If the solver is spending too many iterations, try relaxing to `1e-5`.

**`transport_convergence_tolerance`** (default: `1e-4`): The tolerance for the saturation solver (implicit scheme only). Saturation transport is typically better conditioned than pressure, so a more relaxed tolerance is appropriate.

**`maximum_solver_iterations`** (default: `250`): The maximum number of iterations before the solver gives up. Well-conditioned problems typically converge in 10 to 50 iterations. If you regularly hit the iteration limit, the problem is likely poorly conditioned and needs a better preconditioner rather than more iterations.

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pressure_solver="bicgstab",
    pressure_convergence_tolerance=1e-6,
    maximum_solver_iterations=250,
)
```

!!! tip "Diagnosing Solver Issues"

    If the solver consistently fails to converge:

    1. Try a stronger preconditioner (switch from `"ilu"` to `"amg"` or `"cpr"`)
    2. Switch to a different solver (`"gmres"` instead of `"bicgstab"`)
    3. Use 64-bit precision (`bores.use_64bit_precision()`) for better numerical conditioning
    4. Reduce the time step to produce a better-conditioned matrix
    5. Check your model for extreme property contrasts (permeability ratios > 10,000:1)

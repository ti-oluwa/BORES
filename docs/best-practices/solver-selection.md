# Solver Selection

## Overview

Every timestep in a reservoir simulation requires solving one or more sparse linear systems of equations. The pressure equation in IMPES, the coupled pressure-saturation system in fully implicit schemes, and various auxiliary calculations all reduce to solving $A \mathbf{x} = \mathbf{b}$ where $A$ is a large sparse matrix. The choice of linear solver and preconditioner has a major impact on both the speed and reliability of your simulation.

BORES provides several iterative Krylov solvers and preconditioners, plus a direct solver for small problems or as a fallback. The solver infrastructure uses a registry system, so you can also register custom solvers and preconditioners if the built-in options do not meet your needs. For most reservoir simulation problems, the default combination of BiCGSTAB with ILU preconditioning works well, but understanding when to use alternatives can save significant computation time or rescue a simulation that fails to converge.

The linear system arising from reservoir simulation has specific properties that guide solver selection. The pressure equation produces a symmetric positive-definite (SPD) matrix in single-phase flow and a nearly symmetric matrix in multi-phase flow. The saturation transport matrix is non-symmetric and often better conditioned than the pressure matrix. These properties mean that different solvers may be optimal for the pressure and saturation equations, and BORES lets you configure them independently.

---

## Available Solvers

BORES registers the following solvers by default. You can list them at runtime with `bores.list_solver_funcs()`.

| Solver | Name | Best For | Properties |
| --- | --- | --- | --- |
| BiCGSTAB | `"bicgstab"` | General non-symmetric systems | Robust, moderate convergence rate. Default for both pressure and saturation. |
| GMRES | `"gmres"` | Non-symmetric systems with many restarts | Fast convergence but uses more memory (stores Krylov basis vectors). |
| LGMRES | `"lgmres"` | Non-symmetric systems | Variant of GMRES with look-ahead restarts. Often faster than standard GMRES. |
| CG | `"cg"` | Symmetric positive-definite systems | Fastest for SPD matrices. Only works for single-phase pressure or symmetric formulations. |
| CGS | `"cgs"` | Non-symmetric systems | Conjugate gradient squared. Can be faster than BiCGSTAB but less stable. |
| TFQMR | `"tfqmr"` | Non-symmetric systems | Transpose-free quasi-minimal residual. Smooth convergence, good stability. |
| Direct | `"direct"` | Small systems (< 10,000 unknowns) | Exact solution via sparse LU factorization. No convergence issues but scales poorly with size. |

### Choosing a Solver

For most problems, start with `"bicgstab"` (the default). It handles non-symmetric systems well and has predictable convergence behavior. If you find that BiCGSTAB is slow to converge or stagnates, try `"lgmres"` or `"gmres"`, which often converge in fewer iterations at the cost of more memory per iteration.

If your problem has a symmetric pressure equation (single-phase, or a well-conditioned multi-phase formulation), switching the pressure solver to `"cg"` can cut solve time roughly in half because CG requires less work per iteration and converges faster on SPD systems.

For small models (fewer than about 10,000 cells), the direct solver `"direct"` is often the fastest option because it avoids the iteration overhead entirely. For larger models, iterative solvers are always faster.

### Solver Fallback Chains

You can specify multiple solvers as a list. BORES will try them in order, falling back to the next solver if the previous one fails to converge:

```python
config = bores.Config(
    pressure_solver=["bicgstab", "lgmres", "gmres"],
    transport_solver="bicgstab",
    # ...
)
```

This configuration tries BiCGSTAB first, then LGMRES, then GMRES for the pressure equation. The fallback strategy adds robustness without the cost of always using the more expensive solver.

You can also enable a final fallback to the direct solver:

```python
# In `solve_linear_system(...)`, set `fallback_to_direct=True`
# The Config doesn't expose this directly, but the solver infrastructure supports it
```

---

## Available Preconditioners

Preconditioners transform the linear system into one that converges faster. A good preconditioner can reduce iteration counts by 10x or more, making it the single most impactful performance choice for solver configuration. BORES registers these preconditioner factories by default:

| Preconditioner | Name | Cost | Effectiveness | Best For |
| --- | --- | --- | --- | --- |
| ILU(0) | `"ilu"` | Moderate | Good | General purpose. Default choice. |
| AMG | `"amg"` | High setup, low per-iteration | Excellent | Large, well-structured pressure systems |
| Diagonal (Jacobi) | `"diagonal"` | Very low | Modest | Simple problems, debugging |
| Block Jacobi | `"block_jacobi"` | Low | Moderate | Parallel or block-structured systems |
| Polynomial | `"polynomial"` | Low | Moderate | Smooth operators |
| CPR | `"cpr"` | High | Excellent | Coupled pressure-saturation systems |

### Choosing a Preconditioner

**ILU (Incomplete LU)** is the default and the best starting point. It constructs an approximate LU factorization of the matrix, which captures most of the matrix structure at a fraction of the cost of a full factorization. For reservoir simulation pressure equations, ILU typically reduces iteration counts from hundreds to tens.

**AMG (Algebraic Multigrid)** is the premium option for pressure equations. It works by constructing a hierarchy of progressively coarser representations of the problem and solving at multiple scales. AMG is more expensive to set up than ILU but can be dramatically faster for large problems (50,000+ cells) because its convergence rate is nearly independent of problem size. The setup cost can be amortized using `CachedPreconditionerFactory`.

**Diagonal (Jacobi)** is the simplest preconditioner. It just divides each equation by its diagonal element. This is very cheap but only modestly effective. Use it when you want to isolate whether solver problems are caused by the preconditioner, or for very well-conditioned systems where preconditioning is barely needed.

**CPR (Constrained Pressure Residual)** is designed specifically for coupled pressure-saturation systems in fully implicit simulations. It separates the pressure and saturation components and applies AMG to the pressure part. This is the recommended preconditioner for fully implicit schemes.

### Preconditioner Caching

Building a preconditioner can be expensive, especially for AMG. Since the matrix structure stays constant in reservoir simulation (only the values change), you can cache the preconditioner and reuse it across multiple timesteps:

```python
import bores

cached_ilu = bores.CachedPreconditionerFactory(
    factory="ilu",
    update_frequency=10,        # Rebuild every 10 timesteps
    recompute_threshold=0.3,    # Or when matrix values change by more than 30%
)
cached_ilu.register(override=True)

config = bores.Config(
    pressure_preconditioner="ilu",  # Will use the cached version
    # ...
)
```

Preconditioner caching can reduce total solver time by 20 to 40% for large models. The `update_frequency` and `recompute_threshold` parameters control how often the preconditioner is rebuilt. More frequent rebuilds maintain better quality at higher cost. See the [Preconditioners](../user-guide/simulation/preconditioners.md) page in the User Guide for detailed configuration options.

---

## Configuration in Config

The `Config` class provides four solver-related parameters:

```python
config = bores.Config(
    timer=timer,
    wells=wells,
    pressure_solver="bicgstab",                  # Solver for the pressure equation
    transport_solver="bicgstab",                 # Solver for the saturation equation
    pressure_preconditioner="ilu",                # Preconditioner for pressure
    transport_preconditioner="ilu",              # Preconditioner for saturation
    pressure_convergence_tolerance=1e-6,          # Relative tolerance for pressure
    transport_convergence_tolerance=1e-4,        # Relative tolerance for saturation
    maximum_solver_iterations=250,                           # Max iterations per solve
)
```

The saturation equation is typically better conditioned than the pressure equation, which is why the default saturation tolerance (1e-4) is looser than the pressure tolerance (1e-6). Tightening the saturation tolerance beyond 1e-4 rarely improves solution quality but increases iteration counts.

---

## Troubleshooting Convergence

### Solver Does Not Converge

If the solver fails to converge within `maximum_solver_iterations`, BORES raises a `SolverError`. Common causes and fixes:

1. **Preconditioner is too weak.** Switch from `"diagonal"` to `"ilu"`, or from `"ilu"` to `"amg"`. A stronger preconditioner reduces iteration counts.

2. **Tolerance is too tight.** If you set `pressure_convergence_tolerance` to 1e-10 or smaller, the solver may struggle to reach that precision, especially in 32-bit mode. Relax to 1e-6 or switch to 64-bit precision.

3. **Matrix is ill-conditioned.** Extreme permeability contrasts (e.g., 0.01 mD next to 10,000 mD) or very thin cells create ill-conditioned matrices. Try AMG preconditioning, which handles these cases better than ILU.

4. **Timestep is too large.** For implicit schemes, very large timesteps produce matrices that are harder to solve. Let the adaptive timer reduce the timestep.

### Solver Converges But Results Look Wrong

If the solver reports convergence but the simulation produces non-physical results (negative saturations, pressures going to zero), the tolerance may be too loose. Tighten `pressure_convergence_tolerance` to 1e-8 and check whether results improve. Also verify that the model setup is correct (boundary conditions, well locations, fluid properties).

### Solver Is Slow But Converges

If the solver converges but takes many iterations (50+), the preconditioner is not capturing the matrix structure well enough. Try a stronger preconditioner, or use `CachedPreconditionerFactory` to amortize setup cost. If you are solving many systems with similar matrices (as in a time-stepping loop), caching is almost always beneficial.

---

## Recommended Configurations

### Small Models (< 5,000 cells)

```python
config = bores.Config(
    pressure_solver="direct",
    transport_solver="bicgstab",
    transport_preconditioner="ilu",
    # ...
)
```

Direct solvers are fast for small problems and eliminate convergence concerns.

### Medium Models (5,000 to 100,000 cells)

```python
config = bores.Config(
    pressure_solver="bicgstab",
    transport_solver="bicgstab",
    pressure_preconditioner="ilu",
    transport_preconditioner="ilu",
    # ...
)
```

The default configuration works well. Consider adding preconditioner caching if the simulation runs for many timesteps.

### Large Models (100,000+ cells)

```python
config = bores.Config(
    pressure_solver=["bicgstab", "lgmres"],
    transport_solver="bicgstab",
    pressure_preconditioner="amg",
    transport_preconditioner="ilu",
    pressure_convergence_tolerance=1e-6,
    maximum_solver_iterations=500,
    # ...
)
```

AMG preconditioning for pressure and a solver fallback chain provide robustness and performance at scale. Always use `CachedPreconditionerFactory` with AMG to avoid rebuilding the multigrid hierarchy every timestep.

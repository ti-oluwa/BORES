# Preconditioners

## Overview

Iterative solvers like BiCGSTAB and GMRES work by repeatedly improving an approximate solution until the residual falls below the convergence tolerance. The number of iterations required depends on the condition number of the matrix: well-conditioned matrices converge quickly, while ill-conditioned matrices require many iterations or may not converge at all.

A preconditioner transforms the linear system into an equivalent but better-conditioned system that converges faster. Instead of solving $A \cdot x = b$ directly, the solver works with $M^{-1} A \cdot x = M^{-1} b$, where $M$ is the preconditioner. A good preconditioner $M$ approximates $A$ well enough that $M^{-1} A$ is close to the identity matrix, but is much cheaper to apply than actually inverting $A$.

In reservoir simulation, preconditioners are essential because the pressure matrix becomes increasingly ill-conditioned as permeability contrasts increase, grids become finer, or wells create large source/sink terms. Without preconditioning, the solver may require hundreds or thousands of iterations, or fail to converge entirely.

BORES provides several built-in preconditioners, selectable by name through the `Config`:

```python
import bores

config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pressure_preconditioner="ilu",
    transport_preconditioner="ilu",
)
```

---

## Available Preconditioners

| Preconditioner | String Name | Cost | Effectiveness | Best For |
|---|---|---|---|---|
| ILU | `"ilu"` | Moderate | Good | General purpose (default) |
| AMG | `"amg"` | High setup, low apply | Very good | Large, smooth problems |
| CPR | `"cpr"` | High | Excellent | Difficult coupled systems |
| Block Jacobi | `"block_jacobi"` | Low | Moderate | Parallel-friendly |
| Diagonal | `"diagonal"` | Very low | Weak | Fast, simple scaling |
| Polynomial | `"polynomial"` | Low | Moderate | Setup-free alternative |

### ILU (Default)

ILU (Incomplete LU Factorization) is the default preconditioner and the most commonly used in reservoir simulation. It computes an approximate LU decomposition of the matrix, keeping only the entries that fall within the original sparsity pattern. This gives a good approximation of the matrix inverse at a fraction of the cost of a full factorization.

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pressure_preconditioner="ilu",
    transport_preconditioner="ilu",
)
```

ILU works well for most reservoir simulation matrices. It typically reduces the iteration count by a factor of 5 to 20 compared to no preconditioning. Its main limitation is that it can struggle with highly anisotropic problems (very large $k_v/k_h$ contrasts) or matrices with extreme coefficient ranges.

### AMG

AMG (Algebraic Multigrid) is a more powerful preconditioner that uses a hierarchy of progressively coarser approximations to accelerate convergence. It is particularly effective for pressure equations in large, smoothly varying grids.

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pressure_preconditioner="amg",
)
```

AMG has a higher setup cost than ILU (building the multigrid hierarchy takes time), but applying the preconditioner is very fast. This makes AMG advantageous for large problems where many solver iterations are needed. For small problems (< 10,000 cells), the setup overhead may outweigh the iteration savings.

AMG works best for the pressure equation, which has the elliptic character that multigrid exploits. It is less effective for the saturation transport equation, which is hyperbolic. A common pattern is to use AMG for pressure and ILU for saturation.

### CPR

CPR (Constrained Pressure Residual) is a two-stage preconditioner designed specifically for coupled pressure-saturation systems. It first applies an AMG preconditioner to the pressure subsystem, then applies ILU to the full system. This combination is very effective for the fully implicit scheme where pressure and saturation are solved simultaneously.

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="sequential-implicit",
    pressure_preconditioner="cpr",
)
```

CPR is the most expensive preconditioner per application, but it can dramatically reduce the number of Newton iterations needed for implicit convergence. Use it when the implicit scheme with ILU preconditioning shows slow or stalled convergence.

### Block Jacobi

Block Jacobi is a simple preconditioner that inverts diagonal blocks of the matrix independently. It is less effective than ILU but is naturally parallel (each block can be inverted independently) and has very low setup cost.

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pressure_preconditioner="block_jacobi",
)
```

### Diagonal

The diagonal (Jacobi) preconditioner simply scales the system by the inverse of the diagonal entries: $M = \text{diag}(A)$. It is the cheapest preconditioner (both to build and to apply) but provides only modest improvement. Use it as a baseline or when you want the fastest possible preconditioner with minimal overhead.

### Polynomial

The polynomial preconditioner approximates $A^{-1}$ using a truncated Neumann series (polynomial expansion). It requires no factorization, making it useful when the matrix changes frequently and factorization costs would be wasted.

---

## Cached Preconditioners

Building a preconditioner (especially ILU or AMG) is expensive, but in reservoir simulation the matrix structure stays constant and the coefficients change slowly between time steps. BORES provides `CachedPreconditionerFactory` to reuse a preconditioner across multiple time steps, rebuilding it only when needed.

```python
from bores.solvers.base import CachedPreconditionerFactory

# Cache ILU, rebuild every 10 steps or when matrix changes by > 30%
cached_ilu = CachedPreconditionerFactory(
    factory="ilu",
    name="cached_ilu",
    update_frequency=10,
    recompute_threshold=0.3,
)

# Register it for use in Config
cached_ilu.register(override=True)

config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pressure_preconditioner="cached_ilu",
)
```

The `update_frequency` parameter controls how many time steps between forced rebuilds. The `recompute_threshold` is a relative norm threshold: if the matrix coefficients change by more than this fraction from the cached version, the preconditioner is rebuilt. Setting `update_frequency=0` disables frequency-based rebuilding, relying solely on the change threshold.

Caching can save 20 to 40% of total simulation time for problems where the matrix changes slowly (most depletion and waterflood studies). For problems with rapid changes (gas injection, well events), use a lower `update_frequency` (3 to 5) to avoid using a stale preconditioner that slows convergence.

!!! tip "Caching Strategy"

    - **Depletion / waterflooding**: `update_frequency=10`, `recompute_threshold=0.3`
    - **Gas injection**: `update_frequency=5`, `recompute_threshold=0.2`
    - **Well events / schedules**: `update_frequency=3`, `recompute_threshold=0.15`

---

## Custom Preconditioners

You can register custom preconditioner factories using the `@preconditioner_factory` decorator:

```python
from bores.solvers.base import preconditioner_factory

@preconditioner_factory(name="my_precond")
def my_preconditioner(A_csr):
    # Build and return a LinearOperator that approximates A^-1
    # A_csr is a scipy.sparse.csr_array or csr_matrix
    ...

# Use in Config
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pressure_preconditioner="my_precond",
)
```

The factory function receives the sparse coefficient matrix and must return a `scipy.sparse.linalg.LinearOperator` that can be applied to a vector (the preconditioner solve step). You can list all registered preconditioner factories with `bores.solvers.base.list_preconditioner_factories()`.

---

## Choosing a Preconditioner

For most simulations, the default `"ilu"` is the right choice. Consider switching when:

| Situation | Recommendation |
|---|---|
| Default / starting point | `"ilu"` |
| Large grids (> 100K cells) | `"amg"` for pressure, `"ilu"` for saturation |
| Fully implicit scheme | `"cpr"` for pressure |
| High permeability contrast | `"amg"` or `"cpr"` |
| Quick runs, small models | `"diagonal"` or no preconditioner (`None`) |
| Matrix changes frequently | `CachedPreconditionerFactory` with `"ilu"` |

You can disable preconditioning entirely by setting the preconditioner to `None`:

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pressure_preconditioner=None,    # No preconditioning
    transport_preconditioner=None,
)
```

This is rarely a good idea for production runs, but can be useful for profiling to understand how much the preconditioner is helping.

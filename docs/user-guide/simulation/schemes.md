# Evolution Schemes

## Overview

An evolution scheme determines how the pressure and saturation equations are discretized and solved at each time step. The choice of scheme affects numerical stability, accuracy, computational cost, and the maximum time step size you can use. BORES supports three evolution schemes, each offering a different trade-off between these factors.

In black-oil reservoir simulation, you are solving two coupled systems of equations: a pressure equation (derived from mass conservation and Darcy's law) and saturation transport equations (one for each mobile phase). These equations are coupled because pressure depends on fluid saturations (through compressibility and density) and saturations depend on pressure (through flow velocities). The evolution scheme defines how this coupling is handled at each time step.

The scheme is set through the `scheme` parameter in the `Config`.

Supported values are:

- `impes` — Implicit Pressure / Explicit Saturation (default)
- `explicit` — Fully explicit pressure and saturation
- `implicit` — Fully implicit pressure and saturation
- `sequential-implicit` or `si` — Sequential Implicit (implicit pressure, implicit saturation)
- `full-sequential-implicit` or `full-si` — Full Sequential Implicit (SI with outer coupling iterations)

Example:

```python
import bores

config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",
)
```

---

## IMPES (Recommended Default)

IMPES stands for **IM**plicit **P**ressure, **E**xplicit **S**aturation. It is the most widely used scheme in black-oil simulation and the recommended default in BORES.

In IMPES, the pressure equation is solved implicitly (using a linear system solver) while the saturation equations are updated explicitly (using the pressure solution from the current step). This gives you the stability benefits of implicit pressure solving while keeping the saturation update simple and fast.

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",
)
```

The implicit pressure step assembles a sparse linear system $A \cdot p = b$ where $A$ is the transmissibility matrix and $b$ contains source terms (wells, boundary conditions) and accumulation terms. This system is solved using an iterative solver (BiCGSTAB by default) with a preconditioner (ILU by default). Because pressure is solved implicitly, there is no CFL stability limit on the pressure equation, which allows larger time steps.

The explicit saturation step uses the pressure solution to compute phase velocities and then advances the saturations forward in time using a first-order upwind scheme. This step is fast (no linear system to solve) but is subject to a CFL stability condition that limits the maximum time step. If the time step is too large, the explicit saturation update can produce unphysical oscillations or negative saturations.

IMPES is the best balance for most problems. It handles pressure diffusion (which is fast and long-range) implicitly for stability, while treating saturation transport (which is local and advective) explicitly for efficiency.

!!! tip "When to Use IMPES"

    IMPES is appropriate for the vast majority of black-oil simulations: primary depletion, waterflooding, gas injection, and miscible flooding. It is the default in BORES and in most commercial simulators. Only switch to another scheme if you encounter specific numerical issues that IMPES cannot handle.

---

## Sequential Implicit (SI)

Sequential Implicit (often abbreviated SI) treats pressure implicitly and then solves saturation implicitly (typically with a Newton-Raphson solver) within the same time step. Compared with IMPES, SI removes the explicit saturation CFL limit because the saturation update is implicit, allowing larger time steps without the oscillations associated with an explicit transport update.

Use SI when IMPES' saturation CFL is too restrictive but you want to avoid the full coupling cost of a monolithic implicit solve.

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="sequential-implicit",  # or "si"
)
```

Key configuration knobs that affect SI behavior:

- `maximum_newton_iterations` — maximum Newton iterations used by the implicit saturation solver.
- `newton_tolerance` / `newton_saturation_change_tolerance` — convergence and per-iteration saturation-change tolerances for the Newton solver.
- Saturation and pressure change limits (`maximum_*_saturation_change`, `maximum_pressure_change`) still apply and trigger timestep reduction if exceeded.

SI is a pragmatic middle ground: it is more expensive per step than IMPES (because it runs Newton iterations for saturation) but typically cheaper than a fully monolithic implicit solve and often allows substantially larger steps than IMPES.

## Full Sequential Implicit (Full-SI)

Full Sequential Implicit (also available as `full-sequential-implicit` or `full-si`) is SI augmented with an outer iteration loop that alternately re-solves pressure and saturation until the inter-iterate drift between pressure and saturation falls below configured tolerances. This enforces stronger coupling between pressure and saturation without forming a single monolithic Jacobian for the entire coupled system.

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="full-sequential-implicit",  # or "full-si"
    maximum_outer_iterations=5,
    pressure_outer_convergence_tolerance=1e-3,
    transport_outer_convergence_tolerance=1e-2,
)
```

Important outer-loop parameters:

- `maximum_outer_iterations` — maximum outer coupling iterations (default 5).
- `pressure_outer_convergence_tolerance` — relative pressure inter-iterate tolerance for outer-loop convergence.
- `transport_outer_convergence_tolerance` — absolute saturation inter-iterate tolerance for outer-loop convergence.

Full-SI is appropriate when pressure–saturation coupling is strong (for example, high-rate gas injection, near-critical PVT behaviour, or situations with strong compositional feedback) and you need better coupling than SI provides but want to avoid the complexity or cost of a fully monolithic implicit Jacobian.

## Explicit

The fully explicit scheme treats both pressure and saturation explicitly. Both equations are advanced forward in time using the values from the previous time step, with no linear systems to solve.

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="explicit",
)
```

The advantage of the explicit scheme is simplicity and low cost per time step. There are no sparse matrix assemblies, no linear system solves, and no preconditioners. Each step is essentially a series of element-wise array operations.

The disadvantage is that the scheme is conditionally stable. Both the pressure and saturation CFL conditions must be satisfied, which often requires very small time steps. The pressure CFL condition is particularly restrictive because pressure diffuses rapidly across the grid. In practice, the explicit scheme often requires time steps 10 to 100 times smaller than IMPES to remain stable.

The CFL thresholds can be tuned in the `Config`:

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="explicit",
    pressure_cfl_threshold=0.9,     # Max pressure CFL number
    cfl_threshold=0.6,   # Max saturation CFL number
)
```

Lowering these thresholds increases stability at the cost of requiring even smaller time steps. Raising them improves performance but risks numerical instability.

!!! warning "Explicit Stability"

    The fully explicit scheme is useful for debugging, for very small models where the cost per step is negligible, or for educational purposes where you want to observe the CFL condition in action. For production simulations, IMPES is almost always a better choice because it allows much larger time steps while maintaining stability.

---

## Fully Monolithic Implicit (Planned Future Feature)

A fully monolithic implicit scheme — which treats pressure and saturation together in a single coupled Jacobian and solves them simultaneously — is architecturally interesting because it is unconditionally stable and allows arbitrarily large time steps. However, it is **not currently implemented** in BORES.

Instead, use **Full Sequential Implicit** (described above) when you need strong pressure–saturation coupling. Full-SI achieves most of the stability and coupling benefits without the complexity of assembling and solving a single monolithic system. For most applications, the outer-iteration approach in Full-SI strikes a good balance between accuracy, stability, and computational cost.

---

## Choosing a Scheme

| Feature | IMPES | Explicit | Sequential Implicit (SI) | Full Sequential Implicit |
| --- | --- | --- | --- | --- |
| Pressure solve | Implicit | Explicit | Implicit | Implicit (outer loop) |
| Saturation solve | Explicit | Explicit | Implicit (Newton) | Implicit with outer coupling |
| Stability | CFL-limited (saturation) | CFL-limited (both) | Unconditionally stable | Unconditionally stable |
| Cost per step | Moderate | Low | Higher (Newton iterations) | Highest (outer + Newton loops) |
| Max time step | Moderate | Small | Large | Large |
| Pressure–saturation coupling | Weak (one-way per step) | Weak | Moderate (implicit) | Strong (outer iterations) |
| Best for | Most depletion / waterflood | Debugging, small models | CFL-restricted problems | Strong coupling, near-critical |
| Default | Yes | No | No | No |

**Recommended progression:** Start with IMPES. If the saturation CFL becomes too restrictive (time steps < 0.01 days for field-scale models), switch to SI. If pressure–saturation coupling is strong (high-rate gas injection, near-critical PVT), use Full-SI with `maximum_outer_iterations=5–10`. The explicit scheme is mainly for debugging and education.

---

## Convergence Controls

The `Config` provides several parameters that control how the solver behaves within each time step, regardless of scheme:

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",

    # Solver convergence
    pressure_convergence_tolerance=1e-6,
    transport_convergence_tolerance=1e-4,
    maximum_solver_iterations=250,

    # Saturation change limits (trigger timestep rejection if exceeded)
    maximum_oil_saturation_change=0.5,
    maximum_water_saturation_change=0.4,
    maximum_gas_saturation_change=0.85,
    maximum_pressure_change=100.0,         # psi per step

    # CFL thresholds (explicit and IMPES saturation)
    cfl_threshold=0.6,
    pressure_cfl_threshold=0.9,

    # Output control
    output_frequency=1,                # Yield state every N steps
    log_interval=5,                    # Log progress every N steps
)
```

The `pressure_convergence_tolerance` controls when the iterative solver considers the pressure solution converged. A tighter tolerance (smaller number) gives more accurate pressure but requires more iterations. The default of `1e-6` is appropriate for most cases.

The `transport_convergence_tolerance` plays the same role for the implicit saturation solver. It can be more relaxed than the pressure tolerance because the saturation transport equation is typically better conditioned.

The `maximum_solver_iterations` parameter caps how many iterations the solver attempts before giving up. If the solver hits this limit, the time step is rejected and retried with a smaller step size. The default of 250 is generous; well-conditioned problems typically converge in 20 to 50 iterations.

The saturation and pressure change limits (`maximum_oil_saturation_change`, `maximum_pressure_change`, etc.) are safety valves. If any cell's saturation or pressure changes by more than these limits in a single step, the step is rejected and retried with a smaller time step. This prevents large, potentially unphysical jumps in the solution. You can tighten these limits for more conservative behavior or relax them for faster simulations.

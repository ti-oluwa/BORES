# Configuration

## Overview

The `Config` class holds every parameter that controls how a BORES simulation runs. It specifies the timer (time stepping behavior), the rock-fluid tables (relative permeability and capillary pressure models), the wells, the numerical scheme, the solvers, the preconditioners, the convergence tolerances, and all the physical constraints that keep the simulation stable and accurate.

`Config` is a frozen (immutable) attrs class. Once you create a `Config`, its fields cannot be changed in place. To modify a configuration, you create a new one using the `copy()` or `with_updates()` methods. This immutability prevents accidental modification of simulation parameters during a run and makes configurations safe to pass between functions without defensive copying.

Every simulation in BORES requires a `Config`. You pass it (along with a reservoir model) to `bores.run()` to start the simulation. The `Config` is the single point of control for all numerical behavior. If two simulations use different schemes, solvers, or convergence criteria, those differences are captured entirely in their respective `Config` objects.

---

## Creating a Config

At minimum, a `Config` requires a `Timer` and a `RockFluidTables`:

```python
import bores

rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=bores.BrooksCoreyRelPermModel(
        irreducible_water_saturation=0.25,
        residual_oil_saturation_water=0.30,
        residual_oil_saturation_gas=0.15,
        residual_gas_saturation=0.05,
        water_exponent=2.5,
        oil_exponent=2.0,
        gas_exponent=2.0,
    ),
    capillary_pressure_table=bores.BrooksCoreyCapillaryPressureModel(
        irreducible_water_saturation=0.25,
        residual_oil_saturation_water=0.30,
        residual_oil_saturation_gas=0.15,
        residual_gas_saturation=0.05,
    ),
)

config = bores.Config(
    timer=bores.Timer(
        initial_step_size=bores.Time(days=1),
        maximum_step_size=bores.Time(days=10),
        minimum_step_size=bores.Time(hours=1),
        simulation_time=bores.Time(years=3),
    ),
    rock_fluid_tables=rock_fluid_tables,
)
```

This creates a valid configuration with all defaults. The IMPES scheme, BiCGSTAB solver with ILU preconditioning, and standard convergence tolerances are used automatically.

### Adding Wells

Most simulations include wells. Pass them through the `wells` parameter:

```python
wells = bores.wells_(injectors=[injector], producers=[producer])

config = bores.Config(
    timer=bores.Timer(
        initial_step_size=bores.Time(days=1),
        maximum_step_size=bores.Time(days=10),
        minimum_step_size=bores.Time(hours=1),
        simulation_time=bores.Time(years=3),
    ),
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
)
```

### Adding Well Schedules

For simulations with time-varying well controls, use `well_schedules`:

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    well_schedules=schedules,
)
```

When `well_schedules` is provided, the simulator automatically switches well controls at the scheduled times. See the [Well Scheduling](wells/schedules.md) page for details.

### Adding Boundary Conditions

Boundary conditions (aquifer support, constant pressure boundaries, etc.) are specified through the `boundary_conditions` parameter:

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    boundary_conditions=boundary_conditions,
)
```

---

## All Config Parameters

### Required Parameters

| Parameter | Type | Description |
|---|---|---|
| `timer` | `Timer` | Time stepping manager (initial/max/min step sizes, simulation time) |
| `rock_fluid_tables` | `RockFluidTables` | Relative permeability and capillary pressure models |

### Optional Model Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `wells` | `Wells` | `None` | Well configuration (injectors and producers) |
| `well_schedules` | `WellSchedules` | `None` | Dynamic well control schedules |
| `boundary_conditions` | `BoundaryConditions` | `None` | Boundary conditions (aquifers, constant pressure, etc.) |
| `pvt_tables` | `PVTTables` | `None` | Tabulated PVT properties (alternative to correlations) |
| `constants` | `Constants` | Default | Physical and conversion constants |

### Numerical Scheme

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scheme` | `str` | `"impes"` | Evolution scheme: `"impes"`, `"explicit"`, or `"implicit"` |
| `use_pseudo_pressure` | `bool` | `True` | Use pseudo-pressure formulation for gas |

See [Schemes](simulation/schemes.md) for detailed information on each evolution scheme.

### Solver Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pressure_solver` | `str` or list | `"bicgstab"` | Solver(s) for the pressure equation |
| `transport_solver` | `str` or list | `"bicgstab"` | Solver(s) for the saturation equation |
| `pressure_preconditioner` | `str` or `None` | `"ilu"` | Preconditioner for pressure solvers |
| `transport_preconditioner` | `str` or `None` | `"ilu"` | Preconditioner for saturation solvers |
| `pressure_convergence_tolerance` | `float` | `1e-6` | Relative convergence tolerance for pressure |
| `transport_convergence_tolerance` | `float` | `1e-4` | Relative convergence tolerance for saturation |
| `maximum_solver_iterations` | `int` | `250` | Maximum solver iterations per step (capped at 500) |
| `task_pool` | `ThreadPoolExecutor` | `None` | Thread pool for concurrent solver matrix assembly |

See [Solvers](simulation/solvers.md) and [Preconditioners](simulation/preconditioners.md) for details.

### Explicit Scheme Controls

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cfl_threshold` | `float` | `0.7` | Maximum saturation CFL number (explicit scheme only) |
| `pressure_cfl_threshold` | `float` | `0.9` | Maximum pressure CFL number (explicit scheme only) |

### Time Step Saturation Controls

| Parameter | Type | Default | Description |
|---|---|---|---|
| `maximum_oil_saturation_change` | `float` | `0.6` | Maximum oil saturation change per step |
| `maximum_water_saturation_change` | `float` | `0.6` | Maximum water saturation change per step |
| `maximum_gas_saturation_change` | `float` | `0.5` | Maximum gas saturation change per step |

### Time Step Pressure Controls

| Parameter | Type | Default | Description |
|---|---|---|---|
| `maximum_pressure_change` | `float` | `1000.0` | Maximum pressure change per step (psi) |

!!! tip "Pressure Change Limits"

    The default `maximum_pressure_change` of 1000 psi is appropriate for most field-scale models. For lower-pressure reservoirs or simulations with rapid pressure transients (well shutins, gas breakthrough), you may need a tighter limit. A useful rule of thumb is to keep `maximum_pressure_change` below 10 to 15% of the initial reservoir pressure. For example, a shallow reservoir at 1,500 psi might use `maximum_pressure_change=150.0`, while a deep HPHT reservoir at 12,000 psi can use higher values like `maximum_pressure_change=1200.0`.

See [Time Step Control](simulation/timestep-control.md) for guidance on adjusting these.

### Physical Controls

| Parameter | Type | Default | Description |
|---|---|---|---|
| `capillary_strength_factor` | `float` | `1.0` | Scale factor for capillary effects (0 to 1) |
| `disable_capillary_effects` | `bool` | `False` | Completely disable capillary pressure |
| `disable_structural_dip` | `bool` | `False` | Disable gravity/structural dip effects |
| `miscibility_model` | `str` | `"immiscible"` | Miscibility model: `"immiscible"` or `"todd_longstaff"` |
| `freeze_saturation_pressure` | `bool` | `False` | Keep bubble point pressure constant |

### Fluid Mobility

| Parameter | Type | Default | Description |
|---|---|---|---|
| `relative_mobility_range` | `PhaseRange` | See below | Min/max relative mobility per phase |
| `total_compressibility_range` | `Range` | `(1e-24, 1e-2)` | Min/max total compressibility |
| `phase_appearance_tolerance` | `float` | `1e-6` | Saturation below which a phase is absent |

The default relative mobility ranges are:

- Oil: $10^{-12}$ to $10^{6}$
- Water: $10^{-12}$ to $10^{6}$
- Gas: $10^{-12}$ to $10^{6}$

These ranges prevent division by zero and numerical overflow in mobility calculations. You rarely need to change them.

### Implicit Solver Configuration

These parameters control Newton-Raphson solvers in implicit and sequential-implicit schemes.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `jacobian_assembly_method` | `str` | `"analytical"` | Jacobian assembly: `"analytical"` or `"numerical"` |
| `maximum_newton_iterations` | `int` | `15` | Maximum Newton-Raphson iterations per solve |
| `newton_tolerance` | `float` | `1e-6` | Relative residual tolerance for Newton convergence |
| `maximum_line_search_cuts` | `int` | `4` | Maximum line search bisections per Newton step |
| `maximum_saturation_change` | `float` | `0.05` | Maximum per-cell saturation change per Newton iteration |
| `newton_saturation_change_tolerance` | `float` | `1e-4` | Saturation change tolerance for dual convergence check |
| `newton_stagnation_patience` | `int` | `3` | Consecutive non-improving iterations before stagnation |
| `newton_stagnation_improvement_threshold` | `float` | `0.01` | Minimum fractional residual reduction per iteration (1% default) |

### Sequential Implicit Outer Iteration Configuration

These parameters control outer loop convergence for the sequential-implicit scheme.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pressure_outer_convergence_tolerance` | `float` | `1e-3` | Relative pressure inter-iterate tolerance |
| `transport_outer_convergence_tolerance` | `float` | `1e-2` | Absolute saturation inter-iterate tolerance |

### Additional Numerical Controls

| Parameter | Type | Default | Description |
|---|---|---|---|
| `normalize_saturations` | `bool` | `True` | Normalize saturations so Sw + So + Sg = 1.0 after each step |
| `log_interval` | `int` | `5` | Log progress every N steps |
| `output_frequency` | `int` | `1` | Yield a state every N steps |
| `warn_well_anomalies` | `bool` | `True` | Warn about anomalous well flow rates |

### Hysteresis

| Parameter | Type | Default | Description |
|---|---|---|---|
| `residual_oil_drainage_ratio_water_flood` | `float` | `0.6` | Oil drainage residual ratio (waterflood) |
| `residual_oil_drainage_ratio_gas_flood` | `float` | `0.6` | Oil drainage residual ratio (gas flood) |
| `residual_gas_drainage_ratio` | `float` | `0.5` | Gas drainage residual ratio |

### Output and Logging

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output_frequency` | `int` | `1` | Yield a state every N steps |
| `log_interval` | `int` | `5` | Log progress every N steps |
| `warn_well_anomalies` | `bool` | `True` | Warn about anomalous well flow rates |

---

## Parallel Assembly with Task Pools

During each timestep, the pressure and saturation solvers assemble coefficient matrices from fluid properties, transmissibilities, and well contributions. In the IMPES scheme, the pressure solver has three independent assembly stages (accumulation terms, face transmissibilities, and well contributions) and the saturation solver has two independent stages (flux contributions and well rate grids). By default, these stages run sequentially on the calling thread. For larger grids, you can run them concurrently using a thread pool.

The `task_pool` parameter accepts a `ThreadPoolExecutor` that BORES uses to submit independent assembly stages in parallel. The calling thread blocks until all submitted stages complete, so the effective assembly time approaches the duration of the slowest stage rather than their sum. This can reduce assembly time by 30 to 50% for grids with 50,000 or more cells.

BORES provides the `new_task_pool()` context manager as the standard way to create and manage a pool. It creates the pool, yields it for use, and shuts it down cleanly when the block exits, whether normally or due to an exception.

```python
import bores

with bores.new_task_pool(concurrency=3) as pool:
    config = bores.Config(
        timer=timer,
        rock_fluid_tables=rock_fluid_tables,
        wells=wells,
        task_pool=pool,
    )

    run = bores.Run(model=model, config=config)
    with bores.StateStream(run, store=store) as stream:
        for state in stream:
            process(state)
# Pool shuts down cleanly here
```

### When to Use a Task Pool

The break-even point depends on grid size. Below about 10,000 cells, the overhead of thread synchronisation and future creation exceeds the time saved by concurrent execution. The benefit becomes noticeable around 50,000 cells and clearly measurable above 200,000 cells.

| Grid Size | Recommendation |
|---|---|
| < 10,000 cells | Leave `task_pool` as `None` (sequential is faster) |
| 10,000 to 50,000 | Marginal benefit, profile before committing |
| 50,000 to 200,000 | Noticeable benefit, 3 workers recommended |
| > 200,000 | Clearly beneficial, assembly cost approaches solve cost |

### Concurrency Settings

Pass `concurrency=3` for IMPES simulations. The pressure solver submits at most 3 tasks per call and the saturation solver submits at most 2, so 3 workers covers both without waste. If your machine has only 2 physical cores, use `concurrency=2`. Higher values provide no additional benefit because the current assembly design never submits more than 3 tasks per solver call.

### Conditional Pool Based on Grid Size

If you want your code to work efficiently across different model sizes, you can conditionally create a pool:

```python
import bores
from contextlib import nullcontext

nx, ny, nz = 50, 50, 20
cell_count = nx * ny * nz  # 50,000

if cell_count > 10_000:
    pool_context = bores.new_task_pool(concurrency=3)
else:
    pool_context = nullcontext(None)

with pool_context as pool:
    config = bores.Config(
        timer=timer,
        rock_fluid_tables=rock_fluid_tables,
        task_pool=pool,  # None for small grids, `ThreadPoolExecutor` for large
    )
    # ... run simulation
```

!!! info "Why Threads, Not Processes"

    BORES uses `ThreadPoolExecutor` rather than `ProcessPoolExecutor` because the assembly functions operate on large numpy arrays that are shared by reference between threads. A process pool would need to pickle and copy those arrays into child processes on every call. For a 100x100x30 grid, this is 30 to 50 MB of serialisation overhead per timestep, which eliminates any parallelism gain. Numba JIT-compiled functions release the Python GIL during execution, so threads achieve true parallel CPU utilisation without GIL contention.

!!! warning "Pool Lifetime"

    Do not share a single pool between concurrent simulation runs. Each simulation submits up to 3 tasks per solver call, so two concurrent runs would need 6 workers to avoid queuing, and the assembly functions are not designed for that usage. Create one pool per simulation run.

---

## Modifying a Config

Since `Config` is immutable, you cannot modify fields directly. Use `copy()` or `with_updates()` to create modified versions:

### `copy()`

```python
# Create a new config with a different scheme
implicit_config = config.copy(scheme="full-sequential-implicit")

# Multiple changes at once
tuned_config = config.copy(
    scheme="full-sequential-implicit",
    pressure_solver="gmres",
    pressure_preconditioner="amg",
    maximum_newton_iterations=20,
)
```

### `with_updates()`

`with_updates()` works the same way as `copy()` but validates that all provided keys are valid `Config` attributes:

```python
# This works
updated = config.with_updates(scheme="full-sequential-implicit")

# This raises AttributeError because "schemee" is not a valid field
updated = config.with_updates(schemee="full-sequential-implicit")  # AttributeError
```

Use `with_updates()` when you want protection against typos in parameter names. Use `copy()` when you prefer the shorter name and are confident in the parameter names.

---

## Freeze Saturation Pressure

The `freeze_saturation_pressure` flag controls whether the oil bubble point pressure (Pb) is recomputed at each time step or held constant at its initial value.

```python
# Keep Pb constant (standard black-oil assumption)
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    freeze_saturation_pressure=True,
)
```

When `freeze_saturation_pressure=True`, the following properties are computed using the initial bubble point pressure rather than a dynamically updated value:

- Bubble point pressure (Pb) itself
- Solution gas-oil ratio (Rs)
- Oil formation volume factor (Bo)
- Oil compressibility (Co)
- Oil viscosity (indirectly through Rs)
- Oil density (indirectly through Rs and Bo)

This is appropriate for natural depletion and waterflooding where oil composition remains constant. Set it to `False` (the default) for miscible injection or any process where dissolved gas content changes significantly during the simulation.

---

## Capillary Strength Factor

The `capillary_strength_factor` scales capillary pressure effects without changing the capillary pressure model itself. It ranges from 0.0 (no capillary effects) to 1.0 (full capillary effects).

```python
# Reduce capillary effects by 50% for numerical stability
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    capillary_strength_factor=0.5,
)
```

Capillary gradients can become numerically dominant in fine meshes or at sharp saturation fronts, causing oscillations or overshoot. Reducing the capillary strength factor damps these effects without removing them entirely. This is a common technique for improving convergence in difficult models while preserving the qualitative influence of capillary pressure on fluid distribution.

Setting `disable_capillary_effects=True` is equivalent to `capillary_strength_factor=0.0` but is more explicit in intent.

---

## Example Configurations

### Simple Depletion Study

```python
config = bores.Config(
    timer=bores.Timer(
        initial_step_size=bores.Time(days=2),
        maximum_step_size=bores.Time(days=30),
        minimum_step_size=bores.Time(days=1),
        simulation_time=bores.Time(years=10),
    ),
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    freeze_saturation_pressure=True,
)
```

### High-Resolution Waterflood

```python
config = bores.Config(
    timer=bores.Timer(
        initial_step_size=bores.Time(days=0.5),
        maximum_step_size=bores.Time(days=5),
        minimum_step_size=bores.Time(hours=1),
        simulation_time=bores.Time(years=5),
    ),
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pressure_solver="gmres",
    pressure_preconditioner="amg",
    maximum_water_saturation_change=0.15,
    maximum_pressure_change=50.0,
)
```

### Miscible Gas Injection

```python
config = bores.Config(
    timer=bores.Timer(
        initial_step_size=bores.Time(hours=6),
        maximum_step_size=bores.Time(days=3),
        minimum_step_size=bores.Time(minutes=30),
        simulation_time=bores.Time(years=3),
    ),
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    miscibility_model="todd_longstaff",
    freeze_saturation_pressure=False,
    maximum_pressure_change=75.0,
)
```

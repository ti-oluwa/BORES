# Running and Monitoring Simulations

BORES executes 3D reservoir simulations through two primary interfaces: `bores.run()` for core simulation execution and `bores.monitor()` for live monitoring with comprehensive diagnostics. This guide covers both APIs and common usage patterns.

## Quick Start

The simplest way to run a simulation is to pass a `ReservoirModel` and `Config`:

```python
import bores

# Load model and configuration
model = bores.ReservoirModel.from_file("reservoir.h5")
config = bores.Config.from_file("simulation.yaml")

# Execute simulation and collect states
for state in bores.run(model, config):
    print(f"Step {state.step}: time={state.time}s")
    # Process pressure, saturation, rates, etc.
```

For more insight into simulation progress with live diagnostics, use `bores.monitor()`:

```python
import bores

model = bores.ReservoirModel.from_file("reservoir.h5")
config = bores.Config.from_file("simulation.yaml")

# Monitor execution with live dashboard
for state in bores.monitor(model, config):
    print(f"Step {state.step}: average pressure = {state.average_pressure:.2f} psi")
```

---

## `bores.run()`: Core Simulation Execution

The `bores.run()` function executes a 3D reservoir simulation using the specified evolution scheme and configuration. It yields `ModelState` objects at intervals determined by `output_frequency`.

### Function Signature

```python
def run(
    input: Union[ReservoirModel[ThreeDimensions], Run],
    config: Optional[Config] = None,
    *,
    on_step_rejected: Optional[StepCallback] = None,
    on_step_accepted: Optional[StepCallback] = None,
) -> Generator[ModelState[ThreeDimensions], None, None]:
    """Run a simulation on a 3D reservoir model."""
```

### Parameters

**`input`** - The simulation input. Accepts either:

- **`ReservoirModel[ThreeDimensions]`** - A 3D static reservoir model. Requires `config` parameter.
- **`Run`** - A pre-configured run specification containing both model and config. See [Run Specification](#run-specification) section.

**`config`** - Optional `Config` instance with simulation parameters, schemes, wells, and PVT properties.

- **Required** if `input` is a `ReservoirModel`.
- **Optional** if `input` is a `Run`; the Run's config is used by default.
- When provided with a `Run` input, overrides the Run's config.

**`on_step_rejected`** - Optional callback invoked when a proposed time step is rejected due to convergence or stability issues. Receives:

- `step_result: StepResult` - Contains residuals, error messages, solver diagnostics.
- `step_size: float` - The rejected step size (seconds).
- `elapsed_time: float` - Total simulation time elapsed (seconds).

Useful for logging, metrics collection, or adaptive strategies:

```python
def on_rejected(result, step_size, elapsed_time):
    print(f"Step rejected at {elapsed_time}s with size {step_size}s: {result.message}")

for state in bores.run(model, config, on_step_rejected=on_rejected):
    pass
```

**`on_step_accepted`** - Optional callback invoked when a time step is successfully accepted. Receives the same arguments as `on_step_rejected`.

```python
def on_accepted(result, step_size, elapsed_time):
    print(f"Step accepted: size={step_size:.3f}s, Newton iterations={result.newton_iterations}")

for state in bores.run(model, config, on_step_accepted=on_accepted):
    pass
```

### Return Value

Yields `ModelState[ThreeDimensions]` objects at intervals specified by `config.output_frequency`. Each state contains:

- **Current simulation step and time** - `step`, `step_size`, `time`
- **Pressure and saturation grids** - Via `state.model.fluid_properties`
- **Rock properties** - Porosity, absolute permeability
- **Well configuration** - `state.wells` with locations and current schedule
- **Flow properties** - Via SparseTensor fields:
  - `injection_rates` / `production_rates` (STB/day for oil/water, SCF/day for gas)
  - `injection_formation_volume_factors` / `production_formation_volume_factors`
  - `injection_bhps` / `production_bhps` (bottom-hole pressures in psi)
- **Relative permeabilities, mobilities, capillary pressures** - Grids for all phases
- **Timers state** - For resume capability

See [ModelState Documentation](../advanced/states-streams.md) for detailed field access patterns.

### Output Frequency

The number of accepted steps between yielded states is determined by `config.output_frequency`. Default is 1 (yield every step):

```python
# Yield every accepted step
config = bores.Config(output_frequency=1)

# Yield every 10 accepted steps (saves memory/I/O)
config = bores.Config(output_frequency=10)

for state in bores.run(model, config):
    # state is yielded every 10 accepted steps
    pass
```

### Step Rejection and Retry

When a step fails (e.g., saturation becomes unphysical, convergence stalls), BORES:

1. Invokes `on_step_rejected` callback (if provided)
2. Reduces the step size adaptively
3. Retries the same time interval with the smaller size

This continues until either:

- The step succeeds (invokes `on_step_accepted` and continues)
- Step size cannot be reduced further (raises `SimulationError`)

### Example: Basic Usage

```python
import bores

# Load simulation components
model = bores.ReservoirModel.from_file("spe1.h5")
config = bores.Config.from_file("config.yaml")

# Execute and collect final states
states = []
for state in bores.run(model, config):
    states.append(state)
    print(f"Step {state.step}: time={state.time:.1f}s, "
          f"avg_pressure={state.average_pressure:.1f}psi, "
          f"avg_Sw={state.average_water_saturation:.4f}")

# Analyze final state
final = states[-1]
print(f"Simulation completed at {final.time}s ({final.step} steps)")
print(f"Final pressure range: {final.min_pressure:.1f} - {final.max_pressure:.1f} psi")
```

### Example: Extracting Well Performance

```python
import bores

model = bores.ReservoirModel.from_file("model.h5")
config = bores.Config.from_file("config.yaml")

# Collect production history
production_history = {}

for state in bores.run(model, config):
    for well_name, well in state.wells.items():
        if well_name not in production_history:
            production_history[well_name] = []
        
        # Access SparseTensor rates
        oil_rate = state.production_rates.oil.get(well.location, 0.0)
        water_rate = state.production_rates.water.get(well.location, 0.0)
        gas_rate = state.production_rates.gas.get(well.location, 0.0)
        
        production_history[well_name].append({
            'time': state.time,
            'oil': oil_rate,
            'water': water_rate,
            'gas': gas_rate
        })

# Plot results
for well_name, history in production_history.items():
    times = [h['time'] for h in history]
    oil_rates = [h['oil'] for h in history]
    print(f"{well_name}: {oil_rates[-1]:.1f} STB/day at end")
```

---

## `Run` Specification

The `Run` class packages a reservoir model and configuration for organized execution. Useful for storing and executing pre-defined scenarios.

### Creating a `Run`

```python
from bores import ReservoirModel, Config, Run

model = ReservoirModel.from_file("path/to/3d_model.h5")
config = Config.from_file("path/to/simulation_config.yaml")

run = Run(
    model=model,
    config=config,
    name="Primary Depletion Scenario",
    description="30-year primary depletion with no intervention",
    tags=("baseline", "primary-only")
)
```

### Fields

**`model: ReservoirModel[ThreeDimensions]`** - The 3D static reservoir model to simulate.

**`config: Config`** - Simulation configuration and parameters.

**`name: Optional[str]`** - Human-readable identifier for the run.

**`description: Optional[str]`** - Detailed description of the simulation scenario.

**`tags: Tuple[str, ...]`** - Tuple of tags for organizing runs (e.g., `("co2-injection", "extreme-pressure")`).

**`created_at: Optional[str]`** - ISO-formatted timestamp (auto-populated on creation).

### Executing a Run

The `Run` class is callable and iterable:

```python
# Calling directly
for state in run():
    print(f"Step {state.step}")

# Iterating
for state in run:
    print(f"Step {state.step}")

# Passing to run() function
for state in bores.run(run):
    print(f"Step {state.step}")

# With config override
new_config = bores.Config(scheme="full-sequential-implicit", ...)
for state in bores.run(run, config=new_config):
    print(f"Using new scheme: {new_config.scheme}")
```

### Loading from Files

```python
from bores import Run

run = Run.from_files(
    model_path="path/to/model.h5",
    config_path="path/to/config.yaml",
    pvt_tables_path="path/to/pvt_tables.h5",  # Optional
    pvt_data_path="path/to/pvt_data.h5",      # Optional
)

# Execute the loaded run
for state in run:
    process(state)
```

---

## `bores.monitor()`: Live Monitoring and Diagnostics

The `bores.monitor()` function wraps `bores.run()` with live progress displays and comprehensive statistics collection. It yields the same `ModelState` objects as `run()`, optionally paired with `RunStats` for post-simulation analysis.

### Function Signature

```python
@overload
def monitor(
    input: Union[ReservoirModel[ThreeDimensions], Run, Iterable[ModelState[ThreeDimensions]]],
    config: Optional[Config] = None,
    *,
    monitor: Optional[MonitorConfig] = None,
    on_step_rejected: Optional[StepCallback] = None,
    on_step_accepted: Optional[StepCallback] = None,
    return_stats: Literal[False] = False,
) -> Generator[ModelState[ThreeDimensions], None, None]: ...

@overload
def monitor(
    input: Union[ReservoirModel[ThreeDimensions], Run, Iterable[ModelState[ThreeDimensions]]],
    config: Optional[Config] = None,
    *,
    monitor: Optional[MonitorConfig] = None,
    on_step_rejected: Optional[StepCallback] = None,
    on_step_accepted: Optional[StepCallback] = None,
    return_stats: Literal[True],
) -> Generator[Tuple[ModelState[ThreeDimensions], RunStats], None, None]: ...
```

### Parameters

**`input`** - Simulation source. Accepts:

- **`ReservoirModel[ThreeDimensions]`** - Requires `config` parameter.
- **`Run`** - Uses embedded config unless overridden.
- **`Iterable[ModelState[ThreeDimensions]]`** - Generator/stream of states from external simulation. Useful for post-processing saved runs.

**`config`** - Optional `Config` instance.

- **Required** if `input` is a `ReservoirModel`.
- **Optional** if `input` is a `Run` (Run's config used unless overridden).
- **Not used** if `input` is an iterable of states.

**`monitor`** - Optional `MonitorConfig` controlling display behavior. Defaults to `MonitorConfig()` (Rich panel enabled, tqdm disabled).

**`on_step_rejected`** / **`on_step_accepted`** - Same callbacks as `bores.run()`.

**`return_stats`** - Boolean flag determining return value:

- **`False` (default)** - Yields only `ModelState` objects.
- **`True`** - Yields `(ModelState, RunStats)` tuples. The `RunStats` object is the same instance throughout the run and accumulates in-place, making it valid for inspection after the loop.

### Return Value

Depending on `return_stats`:

**`return_stats=False`** - Yields `ModelState[ThreeDimensions]` at output intervals:

```python
for state in bores.monitor(model, config):
    print(f"Step {state.step}")
```

**`return_stats=True`** - Yields `(ModelState, RunStats)` tuples:

```python
for state, stats in bores.monitor(model, config, return_stats=True):
    print(f"Step {state.step}")
    print(f"  Wall time: {stats.wall_time_ms:.1f} ms")
    print(f"  Rejections so far: {stats.total_rejections}")
```

The returned `RunStats` object remains valid and updated after the loop completes, enabling post-simulation analysis.

### `MonitorConfig`: Display Control

Controls which display backends are active and refresh frequency.

```python
from bores import MonitorConfig

monitor_cfg = MonitorConfig(
    use_rich=True,              # Rich live panel (default)
    use_tqdm=False,             # tqdm progress bar (optional)
    refresh_interval=1,         # Update every N accepted steps
    extended_every=10,          # Show detailed stats every N steps
    show_wells=True,            # Per-well performance table
    color_theme="dark"          # "dark" or "light"
)

for state in bores.monitor(model, config, monitor=monitor_cfg):
    pass
```

**`use_rich: bool = True`** - Show a live Rich panel with solver diagnostics (pressure/saturation statistics, Newton iterations, wall time). The panel updates in-place every `refresh_interval` steps and persists in terminal history when the run ends.

**`use_tqdm: bool = False`** - Show a tqdm progress bar tracking simulation-time completion (0–100%). Displays per-step postfix with current step, average pressure, average water saturation, and wall time.

**`refresh_interval: int = 1`** - How often (in accepted steps) to refresh the Rich display. Increase for very fast simulations to avoid terminal flicker.

**`extended_every: int = 10`** - Every this many accepted steps, the Rich panel includes extended performance stats: p95 wall time and average Newton iterations. Set to 0 to disable.

**`show_wells: bool = True`** - Include a per-well section in the Rich panel showing injection and production rates by well name (in surface conditions: STB/day for oil/water, SCF/day for gas).

**`color_theme: str = "dark"`** - Color scheme for the Rich panel:

- `"dark"` - Charcoal background with amber accents
- `"light"` - Off-white with navy accents

### `RunStats`: Post-Simulation Diagnostics

The `RunStats` object accumulates diagnostics at each output step and is updated in-place throughout the simulation. Inspect it after the loop for comprehensive run summary:

```python
for state, stats in bores.monitor(model, config, return_stats=True):
    pass

# After loop completes
print(f"Total steps: {stats.total_steps}")
print(f"Accepted steps: {stats.accepted_steps}")
print(f"Rejections: {stats.total_rejections}")
print(f"Total wall time: {stats.wall_time_ms:.1f} ms")
print(f"Average step time: {stats.average_step_time_ms:.2f} ms")
print(f"p95 step time: {stats.p95_step_time_ms:.2f} ms")
print(f"Average Newton iterations: {stats.average_newton_iterations:.2f}")

# Summary table (automatically printed at end)
print(stats.summary_table())
```

---

## Example: Full Monitoring Workflow

```python
import bores

# Setup
model = bores.ReservoirModel.from_file("reservoir.h5")
config = bores.Config.from_file("config.yaml")

# Create monitor config with both displays
monitor_cfg = bores.MonitorConfig(
    use_rich=True,
    use_tqdm=True,
    show_wells=True,
    refresh_interval=1
)

# Track results
states = []
pressure_history = []
saturation_history = []

# Execute with full monitoring
for state, stats in bores.monitor(
    model, 
    config,
    monitor=monitor_cfg,
    return_stats=True
):
    states.append(state)
    pressure_history.append(state.average_pressure)
    saturation_history.append(state.average_water_saturation)

# Post-simulation analysis
print("\n" + "="*60)
print("SIMULATION SUMMARY")
print("="*60)
print(stats.summary_table())
print(f"\nFinal state at {states[-1].time:.1f} seconds:")
print(f"  Pressure: {states[-1].average_pressure:.2f} psi")
print(f"  Water saturation: {states[-1].average_water_saturation:.4f}")
```

---

## Example: Post-Processing Saved Runs

Monitor can also accept an iterable of externally-loaded states for diagnostics on previously-saved runs:

```python
import bores

# Load states from disk (e.g., via HDF5)
states = load_states_from_disk("simulation_states.h5")

# Collect diagnostics without re-running
for state, stats in bores.monitor(
    states,
    monitor=bores.MonitorConfig(use_rich=True),
    return_stats=True
):
    pass

# Inspect results
print(stats.summary_table())
```

---

## Best Practices

### 1. **Choose Output Frequency Appropriately**

Yielding every step generates large amounts of data. For long simulations, increase `output_frequency`:

```python
# Memory-efficient for 10,000+ step simulations
config = bores.Config(
    ...,
    output_frequency=100  # Yield every 100 accepted steps
)
```

### 2. **Use Callbacks for Real-Time Metrics**

Collect diagnostics without storing full states:

```python
metrics = {'rejections': 0, 'max_newton': 0}

def on_rejected(result, step_size, elapsed):
    metrics['rejections'] += 1

def on_accepted(result, step_size, elapsed):
    metrics['max_newton'] = max(metrics['max_newton'], result.newton_iterations)

for state in bores.run(model, config, 
                       on_step_rejected=on_rejected,
                       on_step_accepted=on_accepted):
    pass
```

### 3. **Monitor vs. Run Trade-off**

- Use **`bores.run()`** for headless/batch execution or when you don't need live feedback.
- Use **`bores.monitor()`** for interactive work or when debugging convergence issues.

```python
# Headless (no output)
config = config.with_updates(log_interval=0)
for state in bores.run(model, config):
    pass

# Interactive monitoring
for state in bores.monitor(model, config):
    pass
```

### 4. **Extract Well Data via SparseTensor**

Well-based data (rates, FVF, BHP) are stored efficiently as SparseTensor. Access using well locations:

```python
for state in bores.run(model, config):
    for well_name, well in state.wells.items():
        loc = well.location
        
        # Dictionary-like access
        oil_injection = state.injection_rates.oil[loc]
        gas_production = state.production_rates.gas[loc]
        
        # Convert to dense for numpy operations
        all_oil_injection = state.injection_rates.oil.array()
        mean_injection = all_oil_injection[all_oil_injection > 0].mean()
```

See [SparseTensor Documentation](../advanced/states-streams.md#understanding-sparsetensor) for more patterns.

### 5. **Resume from Checkpoint**

Use `state.timer_state` to resume from a saved state:

```python
# First simulation (stopped early)
states = []
for state in bores.run(model, config):
    states.append(state)
    if state.time > 1e6:  # Stop after 1M seconds
        break

# Resume from last state
last_state = states[-1]
new_timer = bores.Timer.from_state(last_state.timer_state)
new_config = config.with_updates(timer=new_timer)

for state in bores.run(model, new_config):
    process(state)
```

---

## Troubleshooting

### Simulation runs very slowly

Check if step rejection rate is high:

```python
rejection_count = 0

def track_rejections(result, step_size, elapsed):
    global rejection_count
    rejection_count += 1

for state in bores.run(model, config, on_step_rejected=track_rejections):
    pass

print(f"Rejection rate: {rejection_count} rejections")
```

High rejection rates suggest:

- Configuration too aggressive (e.g., `cfl_safety_margin` too high)
- Grid refinement needed near wells
- Evolution scheme mismatch (consider SI/Full-SI for stiff systems)

### Out of memory during long simulations

Increase `output_frequency` to reduce number of yielded states:

```python
config = bores.Config(..., output_frequency=1000)
```

Or process states in streaming fashion without storing them.

### Convergence failures

Monitor provides diagnostics via `RunStats`:

```python
for state, stats in bores.monitor(model, config, return_stats=True):
    pass

print(f"Newton iterations per step: avg={stats.average_newton_iterations:.1f}")
print(f"Wall time per step: avg={stats.average_step_time_ms:.1f} ms")
```

High Newton iterations suggest the scheme or timestep size may need adjustment. See [Solver Selection](../best-practices/solver-selection.md) guide.

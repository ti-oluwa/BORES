# States and Streams

## Overview

When you run a BORES simulation, the simulator produces a sequence of `ModelState` objects, one for each accepted time step. Each state captures a complete snapshot of the reservoir at that moment: pressures, saturations, well rates, relative permeabilities, mobilities, and capillary pressures. Understanding how to work with states is essential for analyzing simulation results.

For large simulations, holding all states in memory simultaneously can be expensive. A 100,000-cell model with 500 time steps generates roughly 2 GB of state data at 32-bit precision. The `StateStream` class solves this by persisting states to disk as they are produced and optionally replaying them later, keeping memory usage constant regardless of simulation length.

Beyond just collecting and storing states, BORES provides `ModelAnalyst` for post-processing simulation results. The analyst accepts states (from a list, a stream, or a store replay) and computes volumetrics, recovery factors, production histories, material balance drive indices, sweep efficiency, decline curves, and well productivity metrics. See the [Model Analysis](model-analysis.md) page for full details on the analyst API.

---

## Model State

The `ModelState` class is a frozen (immutable) attrs class that captures the reservoir state at a single time step. Every field on a `ModelState` is a snapshot in time. Because the class is frozen, you cannot mutate any field after creation. If you need a modified version, use `attrs.evolve()` to create a copy with selected fields changed.

```python
states = list(bores.run(model, config))
state = states[-1]  # Last time step

# Time information
print(state.step)              # Time step index (0-based)
print(state.step_size)         # Step size in seconds
print(state.time)              # Elapsed time in seconds
print(state.time_in_days)      # Elapsed time in days
print(state.time_in_years)     # Elapsed time in years
```

### Accessing Reservoir Properties

The `model` field contains the full reservoir model with all property grids. The model itself is also frozen, so you always get a consistent snapshot of the reservoir at that moment.

```python
# Pressure and saturation grids (3D numpy arrays matching grid shape)
pressure = state.model.fluid_properties.pressure_grid
Sw = state.model.fluid_properties.water_saturation_grid
So = state.model.fluid_properties.oil_saturation_grid
Sg = state.model.fluid_properties.gas_saturation_grid

# PVT properties
oil_visc = state.model.fluid_properties.oil_viscosity_grid
water_visc = state.model.fluid_properties.water_viscosity_grid
oil_fvf = state.model.fluid_properties.oil_formation_volume_factor_grid
gas_fvf = state.model.fluid_properties.gas_formation_volume_factor_grid
Rs = state.model.fluid_properties.solution_gas_to_oil_ratio_grid
Pb = state.model.fluid_properties.oil_bubble_point_pressure_grid

# Rock properties
porosity = state.model.rock_properties.porosity_grid
perm_x = state.model.rock_properties.absolute_permeability.x
perm_y = state.model.rock_properties.absolute_permeability.y
perm_z = state.model.rock_properties.absolute_permeability.z
```

### Accessing Flow Properties

Injection and production rates, formation volume factors, bottom-hole pressures, relative permeabilities, and mobilities are stored directly on the state. All rate and pressure quantities are stored as `SparseTensor` objects for memory efficiency.

```python
# Injection and production rates (ft³/day per cell, SparseTensor)
oil_injection = state.injection_rates.oil
water_injection = state.injection_rates.water
gas_injection = state.injection_rates.gas

oil_production = state.production_rates.oil
water_production = state.production_rates.water
gas_production = state.production_rates.gas

# Formation volume factors (bbls/STB for oil/water, ft³/SCF for gas, SparseTensor)
oil_injection_fvf = state.injection_formation_volume_factors.oil
water_injection_fvf = state.injection_formation_volume_factors.water
gas_injection_fvf = state.injection_formation_volume_factors.gas

oil_production_fvf = state.production_formation_volume_factors.oil
water_production_fvf = state.production_formation_volume_factors.water
gas_production_fvf = state.production_formation_volume_factors.gas

# Bottom-hole pressures at well perforations (psi, SparseTensor)
oil_injection_bhp = state.injection_bhps.oil
water_injection_bhp = state.injection_bhps.water
gas_injection_bhp = state.injection_bhps.gas

oil_production_bhp = state.production_bhps.oil
water_production_bhp = state.production_bhps.water
gas_production_bhp = state.production_bhps.gas

# Relative permeabilities (dimensionless, 0 to 1, regular numpy arrays)
kro = state.relative_permeabilities.oil
krw = state.relative_permeabilities.water
krg = state.relative_permeabilities.gas

# Relative mobilities (cP⁻¹, kr/mu, regular numpy arrays)
lambda_o = state.relative_mobilities.oil
lambda_w = state.relative_mobilities.water
lambda_g = state.relative_mobilities.gas

# Capillary pressures (psi, regular numpy arrays)
pcow = state.capillary_pressures.oil_water
pcog = state.capillary_pressures.gas_oil
```

### Understanding `SparseTensor`

Flow rates, formation volume factors, and bottom-hole pressures are stored as `SparseTensor` objects. A `SparseTensor` is a memory-efficient data structure that only stores non-zero entries (or more generally, non-default entries). This is especially useful for well-based quantities like injection rates, which are zero for most cells in the grid (wells only exist at a few locations).

A `SparseTensor` behaves like a sparse array but remains fully compatible with NumPy. You can think of it as a smart dictionary that also knows its overall shape, default fill value, and data type.

#### Common `SparseTensor` Operations

**Dictionary-like access**: Access values by cell index, getting zero (or the default) for missing entries:

```python
# Get injection rate at a specific cell (returns 0.0 if no entry)
oil_rate_at_cell = state.injection_rates.oil[(10, 20, 5)]

# Check if a cell has an explicit entry
if (10, 20, 5) in state.injection_rates.oil:
    print("Cell has an injection entry")

# Iterate over cells with non-zero rates
for cell_index in state.injection_rates.oil:
    rate = state.injection_rates.oil[cell_index]
    print(f"Cell {cell_index}: oil injection rate = {rate} ft³/day")
```

**Properties and metadata**:

```python
# Number of explicitly stored (non-default) entries
num_injection_cells = state.injection_rates.oil.nnz

# Check if all entries are zero
if state.injection_rates.oil.empty:
    print("No oil injection")

# Get shape (matches grid shape)
tensor_shape = state.injection_rates.oil.shape
```

**Converting to dense NumPy arrays**:

```python
# Convert to a full 3D array (fills missing entries with 0.0)
oil_rate_grid = state.injection_rates.oil.array()
# oil_rate_grid has shape matching the reservoir grid, mostly zeros

# Convert to a Python nested list
oil_rate_list = state.injection_rates.oil.dense()

# Extract as a plain NumPy array with a different dtype
oil_rate_float32 = state.injection_rates.oil.array(dtype=np.float32)
```

**Arithmetic operations**:

```python
# SparseTensors support element-wise operations
total_rate = state.injection_rates.oil + state.injection_rates.water

# Scalar multiplication
doubled_rates = state.injection_rates.oil * 2.0

# Absolute value (useful for production, which is negative by convention)
abs_production = abs(state.production_rates.oil)
```

#### Practical Example: Computing Total Well Rates

```python
# Sum all injection rates across all phases and cells
total_oil_injection = state.injection_rates.oil.sum()
total_water_injection = state.injection_rates.water.sum()
total_gas_injection = state.injection_rates.gas.sum()

print(f"Total oil injection: {total_oil_injection} ft³/day")

# Convert reservoir rates to surface rates using FVF
# (FVF = formation volume factor, which is also a SparseTensor)
surface_oil_rate = state.injection_rates.oil.sum() / state.injection_formation_volume_factors.oil.array().mean()
print(f"Surface oil rate (approximate): {surface_oil_rate} STB/day")

# Get oil injection for a specific well's cells
well_injection_cells = [(10, 20, 5), (10, 20, 6), (10, 20, 7)]  # Perforation cells
well_oil_rate = sum(
    state.injection_rates.oil[cell] 
    for cell in well_injection_cells
)
print(f"Oil rate for this well: {well_oil_rate} ft³/day")
```

#### Working with Formation Volume Factors and Bottom-Hole Pressures

Formation volume factors (FVF) and bottom-hole pressures are also stored as sparse tensors at well locations:

```python
# Check if specific well has injection bhp data
well_cell = (10, 20, 5)
if well_cell in state.injection_bhps.oil:
    bhp = state.injection_bhps.oil[well_cell]
    print(f"Oil injection BHP at {well_cell}: {bhp} psi")

# Get all cells with FVF data (non-zero/non-default entries)
for cell_index in state.injection_formation_volume_factors.oil:
    fvf = state.injection_formation_volume_factors.oil[cell_index]
    print(f"Cell {cell_index}: oil FVF = {fvf} bbls/STB")

# Convert rates to surface production accounting for FVF
# Production rates are negative by convention, so we take absolute value
total_surface_oil = 0.0
for cell in state.production_rates.oil:
    reservoir_rate = abs(state.production_rates.oil[cell])
    fvf = state.production_formation_volume_factors.oil[cell]
    if fvf > 0:
        surface_rate = reservoir_rate * 5.615 / fvf  # 5.615 ft³/bbl conversion
        total_surface_oil += surface_rate
print(f"Total surface oil production: {total_surface_oil} STB/day")
```

#### Number of Non-Zero Entries

Check how many cells have well activity:

```python
# How many cells have oil injection?
num_oil_injection_cells = state.injection_rates.oil.nnz

# How many active well perforations total?
total_perforation_cells = (
    state.injection_rates.oil.nnz +
    state.injection_rates.water.nnz +
    state.injection_rates.gas.nnz +
    state.production_rates.oil.nnz +
    state.production_rates.water.nnz +
    state.production_rates.gas.nnz
)
print(f"Total well perforation cells active: {total_perforation_cells}")
```

#### Why `SparseTensor` Instead of Dense Arrays?

Wells (sources/sinks) only exist at a few grid cells—typically 1-10 cells per well out of 100,000+ total cells. A dense array would waste most of its memory storing zeros. By using a sparse representation, BORES:

- **Saves memory**: Only stores cells with non-zero rates
- **Improves performance**: Iterations only touch cells with data
- **Preserves precision**: No loss of information when converting between sparse and dense forms

When you need a regular NumPy array for analysis, use `.array()` to materialize the dense form. When you're just querying rates for post-processing, iterate directly over the sparse tensor to avoid allocating the full grid.

### Well Information

The well configuration at each state is accessible through the `wells` field. The `wells_exists()` method checks whether any wells are defined without loading the full well data structure.

```python
if state.wells_exists():
    wells = state.wells
    for well in wells.production_wells:
        print(f"Well {well.name}: skin={well.skin_factor}")
```

### `Timer` State

The timer's internal state (step count, proposed next step size, performance history) is optionally captured when `Config.capture_timer_state` is enabled.

```python
if state.timer_state is not None:
    ts = state.timer_state
    print(f"Next proposed step: {ts['next_step_size']:.2f} seconds")
    print(f"Steps since failure: {ts['steps_since_last_failure']}")
```

---

## Collecting States

The simplest approach is to collect all states in a list. The `bores.run()` function returns a generator, so calling `list()` forces the entire simulation to run and stores every state in memory.

```python
states = list(bores.run(model, config))

# Extract time series data
import numpy as np

time_days = np.array([s.time_in_days for s in states])
avg_pressure = np.array([
    s.model.fluid_properties.pressure_grid.mean() for s in states
])
avg_So = np.array([
    s.model.fluid_properties.oil_saturation_grid.mean() for s in states
])
```

This is fine for simulations that produce a manageable number of states (up to a few hundred). For longer runs, consider using `output_frequency` in the `Config` to reduce the number of states, or use `StateStream` for disk-backed persistence.

### Output Frequency

The `output_frequency` parameter in `Config` controls how often states are yielded:

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    output_frequency=10,  # Yield every 10th step
)
```

With `output_frequency=10`, the simulator still runs all time steps internally but only yields a state every 10th step, reducing memory usage by 10x. The intermediate steps are computed but not stored.

---

## State Stream

`StateStream` provides memory-efficient state iteration with optional disk persistence. It wraps the simulation generator and writes states to a `DataStore` as they are produced, immediately freeing the memory used by each state. The stream acts as both a context manager and an iterator, and offers several methods for controlling how you consume and replay states.

The core idea is that you iterate through states exactly once during the simulation. As each state passes through, the stream optionally persists it to a store. After the simulation is exhausted, you can replay any or all of the saved states from the store without re-running the simulation. This keeps peak memory usage proportional to `batch_size` rather than the total number of states.

```python
from bores.streams import StateStream
from bores.stores import HDF5Store

store = HDF5Store("simulation_run.h5")

with StateStream(
    states=bores.run(model, config),
    store=store,
    batch_size=10,
) as stream:
    for state in stream:
        print(f"Step {state.step}: P_avg = {state.model.fluid_properties.pressure_grid.mean():.1f}")
```

### Configuration Options

| Parameter | Default | Description |
|---|---|---|
| `states` | (required) | Generator or iterator of `ModelState` instances |
| `store` | `None` | Storage backend for persistence. Must support appending |
| `batch_size` | 10 | States accumulated before flushing to disk |
| `validate` | `False` | Validate states before saving (checks grid shapes, dtype consistency) |
| `auto_save` | `True` | Flush remaining states on context exit |
| `auto_replay` | `True` | Replay from store when iterating after consumption |
| `save` | `True` | Boolean or filter function for selective saving |
| `background_io` | `False` | Write to disk in a background thread |
| `max_queue_size` | 50 | Back-pressure limit for background I/O queue |
| `checkpoint_interval` | `None` | Create a checkpoint every N states |
| `checkpoint_store` | `None` | Separate store for checkpoints |
| `max_batch_memory_usage` | `None` | Force flush when batch exceeds this size (MB) |

---

## Stream Methods

### Iterating with `for`

The most common way to use a stream is the `for` loop. On the first pass, the stream consumes the underlying generator, yielding each state and persisting it to the store according to the `save` and `batch_size` settings. If you iterate again after the generator is exhausted, the stream automatically replays from the store (if `auto_replay=True`).

```python
with StateStream(states=bores.run(model, config), store=store) as stream:
    # First pass: runs the simulation, saves states
    for state in stream:
        process(state)

    # Second pass: replays from store (no re-simulation)
    for state in stream:
        plot(state)
```

If `auto_replay` is `False`, iterating a second time raises a `StreamError`. In that case, you must call `replay()` explicitly.

### `consume()`

The `consume()` method exhausts the entire stream without yielding any states to your code. All configured side effects (persistence, checkpointing, validation) still occur normally. This is useful when you want to run a simulation purely for the purpose of saving its output to disk.

```python
stream = StateStream(
    states=bores.run(model, config),
    store=HDF5Store("run.h5"),
    checkpoint_store=HDF5Store("checkpoints.h5"),
    checkpoint_interval=100,
)
stream.consume()  # States saved and checkpointed, nothing returned
```

After calling `consume()`, the stream is marked as consumed. Calling `consume()` again has no effect. You can still call `replay()` to load the saved states.

### `last()`

The `last()` method returns the final state from the stream. If the stream has not been consumed yet, it iterates through the entire simulation (triggering all side effects) and returns the last state yielded. If the stream has already been consumed and a store is available, it loads only the last entry from the store without replaying everything.

```python
with StateStream(states=bores.run(model, config), store=store) as stream:
    final_state = stream.last()
    print(f"Final pressure: {final_state.model.fluid_properties.pressure_grid.mean():.1f}")
```

This is particularly useful when you only need the end result of a simulation but still want all intermediate states saved to disk.

### `until(condition)`

The `until()` method iterates through the stream, yielding states one at a time, until the `condition` function returns `True`. The state that satisfies the condition is yielded as well, then iteration stops. All configured side effects (persistence, checkpointing) apply to every state that passes through, including those before the stop condition is met.

```python
with StateStream(states=bores.run(model, config), store=store) as stream:
    # Run until average pressure drops below 1500 psi
    for state in stream.until(
        lambda s: s.model.fluid_properties.pressure_grid.mean() < 1500.0
    ):
        print(f"Step {state.step}: P = {state.model.fluid_properties.pressure_grid.mean():.1f}")
```

The condition receives a `ModelState` and returns a boolean. Use `until()` when you have a physical stopping criterion ("stop when water cut exceeds 95%") rather than a fixed number of steps. Note that the remaining states in the underlying generator are not consumed, so the simulation stops early. If the condition never becomes `True`, the entire stream is consumed.

### `while_(condition)`

The `while_()` method is the complement of `until()`. It iterates as long as the `condition` returns `True`, and stops when the condition becomes `False`. The final state (where the condition failed) is also yielded.

```python
with StateStream(states=bores.run(model, config), store=store) as stream:
    # Run while oil saturation is above residual
    for state in stream.while_(
        lambda s: s.model.fluid_properties.oil_saturation_grid.mean() > 0.15
    ):
        analyze(state)
```

Like `until()`, all side effects apply to every state that passes through. The key difference is the semantics: `until()` runs until something happens, `while_()` runs while something holds. Choose whichever reads more naturally for your use case.

### `replay(indices, predicate, steps, validator)`

The `replay()` method loads previously saved states from the store. It returns an iterator, so you can process states one at a time without loading everything into memory. Filtering happens before deserialization, so skipped entries have no I/O cost.

```python
# Replay all saved states
for state in stream.replay():
    plot(state)

# Replay specific entries by insertion-order index
for state in stream.replay(indices=[0, 50, 99]):
    compare(state)

# Replay by simulation step number
for state in stream.replay(steps=[0, 100, 200, 300]):
    analyze(state)

# Replay using a step filter function
for state in stream.replay(steps=lambda s: s % 50 == 0):
    log(state)

# Replay with metadata predicate
for state in stream.replay(predicate=lambda e: e.meta.get("step", 0) > 100):
    late_analysis(state)
```

The parameters can be combined. `indices` takes priority and bypasses `steps` and `predicate`. When both `steps` and `predicate` are provided, they are composed with a logical AND (both must pass for a state to be yielded).

| Parameter | Type | Description |
|---|---|---|
| `indices` | `Sequence[int]` | Zero-based insertion-order positions to load |
| `steps` | `Sequence[int]` or `Callable[[int], bool]` | Filter by simulation step number |
| `predicate` | `Callable[[EntryMeta], bool]` | Filter on stored entry metadata |
| `validator` | `Callable[[ModelState], ModelState]` | Post-load validation/transformation |

### `flush(block)`

The `flush()` method manually writes the current batch buffer to the store. In normal operation, flushing happens automatically when the batch reaches `batch_size` or when the stream exits its context manager. Call `flush()` explicitly when you need to guarantee that data has been written at a specific point.

```python
with StateStream(states=bores.run(model, config), store=store, batch_size=50) as stream:
    for state in stream:
        if state.step % 100 == 0:
            stream.flush(block=True)  # Ensure data is on disk now
            print(f"Flushed through step {state.step}")
```

The `block` parameter controls behavior when `background_io=True`. With `block=False` (the default), the batch is enqueued and the method returns immediately. With `block=True`, the method waits until all pending writes have completed.

### `progress()`

The `progress()` method returns a `StreamProgress` dictionary with real-time statistics about the stream's state. Use this for monitoring long-running simulations or building progress bars.

```python
with StateStream(states=bores.run(model, config), store=store) as stream:
    for state in stream:
        if state.step % 50 == 0:
            p = stream.progress()
            print(
                f"Yielded: {p['yield_count']}, "
                f"Saved: {p['saved_count']}, "
                f"Checkpoints: {p['checkpoints_count']}, "
                f"Pending: {p['batch_pending']}, "
                f"Memory: {p['memory_usage']:.1f} MB"
            )
```

The returned dictionary contains:

| Key | Type | Description |
|---|---|---|
| `yield_count` | `int` | Total states yielded (including replays) |
| `saved_count` | `int` | Total states written to store |
| `checkpoints_count` | `int` | Total checkpoints created |
| `batch_pending` | `int` | States in current batch (not yet flushed) |
| `store_backend` | `str` or `None` | Name of the store class being used |
| `memory_usage` | `float` | Estimated batch memory in MB |

When `background_io=True`, two additional keys are available: `io_queue_size` (current items in the I/O queue) and `io_thread_alive` (whether the background thread is running).

### Properties

The stream also exposes several read-only properties for quick status checks:

```python
stream.yield_count       # Total states yielded so far
stream.saved_count       # Total states saved to store
stream.checkpoints_count # Total checkpoints created
stream.is_consumed       # Whether the underlying generator is exhausted
```

---

## Background I/O

When `background_io=True`, disk writes happen in a separate thread, allowing the simulation to continue while data is being written. This provides a 2 to 3x speedup when I/O is slower than the simulation itself. The simulation thread fills a queue; the I/O worker thread drains it.

```python
with StateStream(
    states=bores.run(model, config),
    store=HDF5Store("run.h5"),
    background_io=True,
    max_queue_size=50,
) as stream:
    for state in stream:
        analyze(state)
```

The `max_queue_size` parameter limits the number of batches waiting to be written. When the queue is full, the simulation thread blocks until the I/O worker drains an item. This back-pressure mechanism prevents unbounded memory growth when the simulation produces states faster than they can be written to disk. If the I/O worker encounters an error, the exception is re-raised on the next iteration of the simulation loop so you do not silently lose data.

---

## Selective Saving

You can filter which states are saved using a predicate function passed to the `save` parameter. All states are still yielded to your code, but only those matching the predicate are persisted.

```python
# Save every 10th state
with StateStream(
    states=bores.run(model, config),
    store=store,
    save=lambda s: s.step % 10 == 0,
) as stream:
    for state in stream:
        pass  # All states yielded, but only every 10th saved
```

You can also disable saving entirely by passing `save=False`. The stream will still iterate through all states but will not write anything to the store.

---

## Checkpointing

For long-running simulations, checkpointing saves the state at regular intervals to a separate store, enabling crash recovery. Checkpoints are written to a dedicated `checkpoint_store` independently of the main store.

```python
with StateStream(
    states=bores.run(model, config),
    store=HDF5Store("full_run.h5"),
    checkpoint_store=HDF5Store("checkpoints.h5"),
    checkpoint_interval=100,
) as stream:
    for state in stream:
        pass
```

If the simulation crashes at step 450, the checkpoint store contains the state at steps 100, 200, 300, and 400 (checkpointing starts at step 1, not step 0).

### Accessing Checkpoints

The stream provides methods for working with checkpoints:

```python
# List available checkpoint step numbers
available = stream.list_checkpoints()
print(f"Checkpoints at steps: {available}")

# Load a specific checkpoint by step number
state_at_200 = stream.checkpoint(200)

# Iterate over all checkpoints
for checkpoint_state in stream.checkpoints():
    print(f"Checkpoint at step {checkpoint_state.step}")
```

---

## Idiomatic Patterns

### Run, Save, and Analyze

The most common workflow is to run a simulation with streaming persistence, then analyze the saved results:

```python
import bores
from bores.stores import HDF5Store
from bores.streams import StateStream
from bores.analyses import ModelAnalyst

store = HDF5Store("simulation.h5")
with StateStream(
    states=bores.run(model, config),
    store=store,
    background_io=True,
) as stream:
    stream.consume()  # Save everything, process nothing

# Analyze saved results
from bores.states import ModelState

with store(mode="r") as s:
    analyst = ModelAnalyst(s.load(ModelState))
print(f"Recovery factor: {analyst.oil_recovery_factor:.2%}")
```

### Monitor During Simulation

If you want to process states as they arrive while also saving them:

```python
with StateStream(states=bores.run(model, config), store=store) as stream:
    for state in stream:
        if state.step % 10 == 0:
            p = state.model.fluid_properties.pressure_grid.mean()
            print(f"Step {state.step}: P_avg = {p:.1f} psi")
```

### Early Stopping

Use `until()` or `while_()` to stop the simulation when a physical criterion is met:

```python
with StateStream(states=bores.run(model, config), store=store) as stream:
    # Stop when water cut exceeds 95%
    final = None
    for state in stream.until(
        lambda s: (s.production.water.sum() / max(s.production.oil.sum() + s.production.water.sum(), 1e-10)) > 0.95
    ):
        final = state
```

### Sparse Replay

When analyzing long simulations, load only the states you need:

```python
# Every 50th step for trend analysis
analyst = ModelAnalyst(stream.replay(steps=lambda s: s % 50 == 0))

# First and last only for delta comparison
initial = next(stream.replay(indices=[0]))
final = stream.last()
```

### Get Just the Final State

When you only care about the end result:

```python
stream = StateStream(states=bores.run(model, config), store=store)
final_state = stream.last()
```

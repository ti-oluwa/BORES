# Serialization and Storage

## Overview

BORES provides a comprehensive serialization system for saving and loading simulation data. You can save reservoir models, configurations, simulation states, and analysis results to disk in multiple formats, then load them later to continue a simulation, reproduce results, or share data with collaborators.

The serialization system has two layers. The base layer (`Serializable`) handles conversion between Python objects and dictionaries. The storage layer (`StoreSerializable`) adds the ability to write those dictionaries to various file backends. Most BORES objects inherit from both, so they support both in-memory round-tripping and file-based persistence.

---

## Storage Backends

BORES supports four storage backends, each with different trade-offs:

| Backend | Format | Append | Best For |
|---|---|---|---|
| `ZarrStore` | Zarr (chunked arrays) | Yes | Large simulations, streaming, array-heavy data |
| `HDF5Store` | HDF5 | Yes | Interoperability, scientific data exchange |
| `JSONStore` | JSON | No | Small configs, human-readable, debugging |
| `YAMLStore` | YAML | No | Configuration files, human-readable |

### ZarrStore

Zarr is the recommended backend for large simulation state data. It stores arrays in a chunked, compressed format that is fast to write and read, supports appending without rewriting existing data, and handles large datasets efficiently. Apart from saving in local directories, `ZarrStore` should support any `BaseStore` type support by zarr.

```python
from bores.stores import ZarrStore
from bores.states import ModelState

store = ZarrStore("simulation.zarr")

# Save states (context manager keeps the handle open for the batch)
with store(mode="a"):
    store.dump(states)

# Load states
with store(mode="r") as s:
    loaded_states = list(s.load(ModelState))
```

### HDF5Store

HDF5 is widely supported in the scientific computing ecosystem. Use it when you need to share data with other tools (MATLAB, Paraview, commercial simulators) or when your organization has standardized on HDF5.

```python
from bores.stores import HDF5Store
from bores.states import ModelState

store = HDF5Store("simulation.h5")

with store(mode="a"):
    store.dump(states)

with store(mode="r") as s:
    loaded_states = list(s.load(ModelState))
```

### JSONStore

JSON is human-readable and useful for saving small objects like configurations. It does not handle large numpy arrays efficiently, so avoid using it for state data.

```python
from bores.stores import JSONStore

store = JSONStore("config.json")

with store(mode="a"):
    store.dump([config])

with store(mode="r") as s:
    loaded_config = next(s.load(type(config)))
```

### YAMLStore

YAML is similar to JSON but more human-friendly. Use it for configuration files that you want to edit by hand.

```python
from bores.stores import YAMLStore

store = YAMLStore("config.yaml")

with store(mode="a"):
    store.dump([config])
```

### Creating Stores with `new_store`

The `new_store` factory function creates the appropriate store based on the file extension:

```python
from bores.stores import new_store

store = new_store("output.zarr")   # ZarrStore
store = new_store("output.h5")    # HDF5Store
store = new_store("output.json")  # JSONStore
store = new_store("output.yaml")  # YAMLStore
```

---

## Store Operations

### Persistent Handles

Every store supports a persistent handle pattern. By default, each call to `dump`, `load`, `append`, or `entries` opens and closes the underlying file or directory independently. This is convenient for one-off calls, but for workloads that perform many operations in sequence (like appending states in a loop), the per-call open/close overhead is significant.

Call `open()` once to obtain a persistent handle, then `close()` when finished. The `__call__` context manager does both automatically and is the recommended approach:

```python
# Context manager (preferred)
with store(mode="a") as s:
    for state in states:
        s.append(state)
# Handle closed automatically here

# Low-level equivalent
store.open(mode="a")
for state in states:
    store.append(state)
store.close()
```

When no handle is open, every method falls back to opening and closing internally, so existing call-sites work without changes.

### Dumping Data

The `dump()` method writes an iterable of serializable objects to the store, overwriting any existing content:

```python
with store(mode="a"):
    store.dump(states)
```

You can optionally provide a validator (called on each item before writing) and a metadata function (for lightweight filtering without deserializing):

```python
with store(mode="a"):
    store.dump(
        states,
        validator=lambda s: validate_state(s, dtype="global"),
        meta=lambda s: {"step": s.step, "time": s.time_in_days},
    )
```

### Loading Data

The `load()` method returns a generator that yields deserialized objects:

```python
from bores.states import ModelState

# Load all states
with store(mode="r") as s:
    for state in s.load(ModelState):
        process(state)

# Load specific indices
with store(mode="r") as s:
    for state in s.load(ModelState, indices=[0, 50, 99]):
        analyze(state)

# Load with a predicate on metadata
with store(mode="r") as s:
    for state in s.load(ModelState, predicate=lambda e: e.idx < 10):
        plot_early(state)
```

The `indices` parameter loads only the items at the specified positions (zero-based). The `predicate` parameter filters based on `EntryMeta` objects, which contain the index and any metadata stored during `dump()`. Both options avoid deserializing items that you do not need.

### Appending Data

Stores that support appending (`ZarrStore`, `HDF5Store`) can add items without rewriting existing content. For bulk appends, always use a persistent handle to avoid the open/close cost on every call:

```python
with store(mode="a") as s:
    for new_state in new_states:
        entry_meta = s.append(new_state)
        print(f"Appended at index {entry_meta.idx}")
```

### Inspecting Entries

The `entries()` method returns metadata for all stored items without deserializing any array data:

```python
entries = store.entries()
print(f"Store contains {len(entries)} items")
for entry in entries:
    print(f"  Index {entry.idx}: {entry.group_name}")
```

---

## Serializable Protocol

All BORES objects that support serialization inherit from the `Serializable` base class. This class provides `dump()` and `load()` methods for dictionary round-tripping:

```python
# Serialize to dictionary
data = config.dump()

# Deserialize from dictionary
restored_config = type(config).load(data)
```

### Custom Serialization

If you create custom classes that need to be serialized, you can implement the `__dump__` and `__load__` methods:

```python
from bores.serialization import Serializable

class MyData(Serializable):
    def __init__(self, values):
        self.values = values

    def __dump__(self, recurse=True):
        return {"values": self.values.tolist()}

    @classmethod
    def __load__(cls, data):
        import numpy as np
        return cls(values=np.array(data["values"]))
```

### Custom Storage Backends

You can register custom storage backends using the `@data_store` decorator:

```python
from bores.stores import data_store, DataStore

@data_store("parquet")
class ParquetStore(DataStore):
    def dump(self, data, **kwargs):
        ...

    def load(self, typ, **kwargs):
        ...

    def entries(self):
        ...

    def flush(self):
        ...
```

---

## Array Serialization

By default, BORES registers a smart ndarray serializer that compresses numpy arrays using the most compact encoding available. The serializer tries four encodings in order and picks the first one that applies:

1. **Scalar**: If every element in the array has the same value, stores only the fill value and shape. A 100x100x50 pressure grid initialized to 3000.0 psi is stored as a single number plus the shape tuple, rather than 500,000 identical floats.

2. **Layered**: If the array is constant along one axis (for example, a depth grid where every cell in a layer has the same depth), stores only the unique values along that axis. A 50-layer depth grid that varies only in z is stored as 50 values instead of the full 3D array.

3. **Sparse**: If most elements share a common fill value (for example, a rate grid where only a few cells near wells have non-zero values), stores only the indices and values of the non-fill cells. This is effective when fewer than 50% of cells differ from the fill value.

4. **Dense**: Falls back to base64-encoded raw bytes. This is lossless and handles all arrays, but produces the largest output. It is still more compact than storing arrays as Python lists in JSON because it uses binary encoding.

All four encodings are exact. There is no approximation or lossy compression. The deserializer detects the encoding automatically from the `encoding` key in the serialized dictionary.

### `BORES_SAVE_RAW_NDARRAY`

If you set the `BORES_SAVE_RAW_NDARRAY` environment variable to `true` before importing BORES, the smart serializer is not registered and arrays are serialized using the default cattrs behavior (nested Python lists). This produces larger, slower output but can be useful for debugging serialization issues or for interoperability with tools that expect plain JSON arrays.

```bash
# Disable smart array serialization (use plain lists)
export BORES_SAVE_RAW_NDARRAY=true
```

Accepted truthy values are `t`, `y`, `yes`, `true`, and `1` (case-insensitive). Any other value (or not setting the variable at all) leaves the smart serializer active, which is the recommended default.

!!! tip "When to Use Raw Arrays"

    Leave `BORES_SAVE_RAW_NDARRAY` unset for normal usage. The smart serializer typically reduces JSON/YAML file sizes by 10x to 100x for reservoir grids, and deserialization is faster because there is less data to parse. Only set it to `true` when you need to inspect the raw array values in a text editor or when debugging a serialization round-trip issue.

---

## File-Based Persistence

Most BORES objects inherit from `StoreSerializable`, which provides convenient methods for saving and loading directly to and from files. The file extension determines which storage backend is used automatically.

### `to_file` and `from_file`

The `to_file()` method saves any serializable object to a file. The `from_file()` class method loads it back. The file extension determines the storage format: `.zarr` uses ZarrStore, `.h5` uses HDF5Store, `.json` uses JSONStore, and `.yaml` uses YAMLStore.

```python
import bores
from bores.models import ReservoirModel

# Save a model to HDF5
model.to_file("my_model.h5")

# Load it back
loaded_model = ReservoirModel.from_file("my_model.h5")
```

The `save` method is an alias for `to_file`, so you can use whichever reads better in your code:

```python
model.save("my_model.h5")  # Same as model.to_file("my_model.h5")
```

### `to_store` and `from_store`

When you already have a store object (for example, because you want to reuse a connection or configure specific options), use `to_store()` and `from_store()` directly:

```python
from bores.stores import ZarrStore

store = ZarrStore("models.zarr")

# Save to an existing store
model.to_store(store)

# Load from an existing store
loaded_model = ReservoirModel.from_store(store)
```

The `from_store()` method returns the first item in the store (or `None` if the store is empty). If the store contains multiple items, only the first is returned.

### Saving and Loading Configurations

Configurations are best saved as YAML or JSON for human readability. You can edit the YAML file by hand and reload it:

```python
# Save config to YAML
config.to_file("simulation_config.yaml")

# Edit the YAML file by hand, then reload
loaded_config = bores.Config.from_file("simulation_config.yaml")
```

### Saving and Loading Models

Reservoir models contain large arrays (porosity, permeability, saturations), so HDF5 or Zarr are the best formats:

```python
# Save model to HDF5
model.to_file("reservoir_model.h5")

# Load model
from bores.models import ReservoirModel
loaded_model = ReservoirModel.from_file("reservoir_model.h5")
```

### `Timer` State

The timer supports its own state serialization for checkpointing mid-simulation:

```python
# Save timer state
timer_state = timer.dump_state() # or `timer.dump()` which does same thing

# Restore timer
restored_timer = bores.Timer.load_state(timer_state) # or `timer.load()
```

---

## `Run` Objects

The `Run` class bundles a reservoir model and a simulation configuration into a single, serializable unit. You can think of a `Run` as a simulation definition: it contains everything needed to reproduce a simulation. Because `Run` inherits from `StoreSerializable`, you can save and load entire simulation setups with a single call.

### Creating a `Run`

```python
import bores

run = bores.Run(
    model=model,
    config=config,
    name="Base Case Waterflood",
    description="5-spot pattern, 1000 STB/day injection",
    tags=("waterflood", "base-case", "field-A"),
)
```

The `name`, `description`, and `tags` fields are optional metadata for organizing your simulation library. The `created_at` field is set automatically to the current UTC timestamp.

### Executing a `Run`

A `Run` is callable and iterable. Calling it returns a generator of `ModelState` objects, exactly like `bores.run(model, config)`. Iterating over it does the same thing.

```python
# As a callable
for state in run():
    process(state)

# As an iterable
for state in run:
    process(state)

# With bores.run() (accepts Run objects directly)
for state in bores.run(run):
    process(state)

# Override config at execution time
new_config = config.with_updates(output_frequency=5)
for state in bores.run(run, config=new_config):
    process(state)
```

### Saving and Loading `Run`s

Because `Run` is a `StoreSerializable`, you can save and load complete simulation definitions:

```python
# Save run to HDF5
run.to_file("base_case.h5")

# Load run later
loaded_run = bores.Run.from_file("base_case.h5")
print(f"Run: {loaded_run.name}")
print(f"Created: {loaded_run.created_at}")

# Execute the loaded run
for state in loaded_run:
    process(state)
```

### Loading from Separate Files

When your model and config are stored in separate files, use the `from_files()` class method:

```python
run = bores.Run.from_files(
    model_path="reservoir_model.h5",
    config_path="simulation_config.yaml",
    pvt_data_path="pvt_tables.h5",  # Optional
)
```

The `pvt_data_path` parameter is optional. When provided, the PVT tables are loaded and attached to the config automatically. This is useful when PVT data comes from a separate laboratory analysis file.

---

## Complete Workflows

### Run, Save States, and Analyze Later

The most common workflow is to run a simulation with streaming persistence, then analyze the saved results in a separate session:

```python
import bores
from bores.stores import ZarrStore
from bores.streams import StateStream

# Session 1: Run and save
store = ZarrStore("simulation.zarr")
with StateStream(
    states=bores.run(model, config),
    store=store,
    checkpoint_store=ZarrStore("checkpoints.zarr"),
    checkpoint_interval=50,
) as stream:
    stream.consume()

# Session 2: Analyze saved results (no re-simulation needed)
from bores.states import ModelState
from bores.analyses import ModelAnalyst

store = ZarrStore("simulation.zarr")
with store(mode="r") as s:
    analyst = ModelAnalyst(s.load(ModelState))
print(f"Recovery factor: {analyst.oil_recovery_factor:.2%}")
```

### Save and Reproduce a Simulation

Package a simulation as a `Run` object so anyone can reproduce it:

```python
# Save the simulation definition
run = bores.Run(
    model=model,
    config=config,
    name="SPE1 Benchmark",
    tags=("benchmark", "spe1"),
)
run.to_file("spe1_run.h5")

# Later (or on another machine): reproduce exactly
loaded_run = bores.Run.from_file("spe1_run.h5")
states = list(bores.run(loaded_run))
```

### Load Just the Final State

When you only need the end result from a previously saved simulation:

```python
store = ZarrStore("simulation.zarr")
max_idx = store.max_index()

with store(mode="r") as s:
    final_state = next(s.load(ModelState, indices=[max_idx]))
print(f"Final pressure: {final_state.model.fluid_properties.pressure_grid.mean():.1f} psi")
```

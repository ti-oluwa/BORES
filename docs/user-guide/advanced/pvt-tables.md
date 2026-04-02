# PVT Tables

By default, BORES computes fluid properties at every time step using empirical correlations. Those correlations are fast and broadly applicable, but they are generalizations - they were derived from hundreds of reservoirs and represent average behavior, not your specific fluids. When you have laboratory PVT measurements from your actual reservoir, you can hand those measurements directly to BORES as lookup tables, and the simulator will interpolate from your data instead of estimating from correlations.

This distinction matters more than it might seem at first. Two reservoirs at the same pressure and temperature can have vastly different oil viscosities, bubble points, or solution gas-oil ratios depending on fluid composition. A 35 API gravity oil from the Permian Basin behaves differently from a 35 API oil from the North Sea. Correlations cannot capture those differences - but your lab data can. PVT tables let you bring that specificity into the simulation.

There is also a practical benefit to using tables in workflow pipelines. If you are running dozens of sensitivity cases, generating tables once from your lab data (or from an equation-of-state package like PVTSim or ECLIPSE PVTi) and reusing them across cases is far more reproducible than depending on correlation choices that might drift between runs. The table approach makes your simulation input explicit and auditable.

The PVT system in BORES is built around three classes. `PVTData` is a plain data container that holds raw tabulated arrays for a single phase. `PVTTable` wraps one `PVTData` object and builds fast scipy interpolators from it, ready for runtime lookups. `PVTTables` is a bundle that holds one `PVTTable` per fluid phase (oil, gas, water) and is the object you pass to your simulation `Config`.

---

## Building PVT Data

### From Correlations as a Baseline

The quickest way to get started with PVT tables is to generate them from correlations using the phase-specific builder functions. This gives you a complete, consistent set of tables covering your pressure-temperature range, which you can then selectively replace with lab data for specific properties.

```python
import bores
import numpy as np
from bores.tables.pvt import build_pvt_dataset, PVTTables

bores.use_32bit_precision()

# Define the pressure and temperature grid for your simulation conditions.
# Always extend slightly beyond your expected simulation range to avoid extrapolation.
pressures = np.linspace(500, 5000, 50)      # 500 to 5000 psi
temperatures = np.linspace(100, 250, 30)    # 100 to 250 F

# Build all three phase datasets in one call
pvt_dataset = build_pvt_dataset(
    pressures=pressures,
    temperatures=temperatures,
    oil_specific_gravity=0.85,
    gas_gravity=0.65,           # Relative to air
    water_salinity=50000.0,     # ppm NaCl
)
```

The `build_pvt_dataset` function returns a `PVTDataSet` containing three `PVTData` objects - one for oil, one for gas, and one for water. Each object holds 2D arrays of shape `(n_pressures, n_temperatures)` for properties like viscosity, density, and formation volume factor. Water properties can optionally be 3D when salinity varies across the grid (more on that below).

You can also build individual phase datasets if you only need certain phases, or if your phases have different grid resolutions:

```python
from bores.tables.pvt import build_oil_pvt_data, build_gas_pvt_data, build_water_pvt_data

oil_data = build_oil_pvt_data(
    pressures=pressures,
    temperatures=temperatures,
    oil_specific_gravity=0.87,
    gas_gravity=0.65,
    estimated_solution_gor=500.0,   # SCF/STB - helps estimate bubble point
)

gas_data = build_gas_pvt_data(
    pressures=pressures,
    temperatures=temperatures,
    gas_gravity=0.65,
)

water_data = build_water_pvt_data(
    pressures=pressures,
    temperatures=temperatures,
    water_salinity=50000.0,         # ppm NaCl
)
```

The `estimated_solution_gor` parameter for the oil builder deserves attention. BORES uses it to estimate the bubble point pressure across the temperature range via Standing's correlation. If you leave it out, BORES will estimate a solution GOR from your API gravity, but you will see a warning. If you have any knowledge of your reservoir's GOR, pass it in here for better bubble point estimates.

### Mixing Lab Data with Correlations

In practice, you often have lab data for some properties but not all. Perhaps you ran a differential liberation test and measured oil FVF and GOR, but you do not have a full viscosity curve. BORES handles this gracefully: any property table you supply overrides the correlation-computed one, and properties you omit are computed from correlations automatically.

```python
import numpy as np
from bores.tables.pvt import build_oil_pvt_data, PVTDataSet

# Load your lab-measured tables. Shape must be (n_pressures, n_temperatures).
# If your lab only measured at one temperature, you can broadcast.
lab_oil_fvf = np.loadtxt("lab_fvf.csv")           # (50, 30) array in bbl/STB
lab_oil_gor = np.loadtxt("lab_gor.csv")            # (50, 30) array in SCF/STB
lab_bubble_points = np.loadtxt("lab_pb.csv")       # 1D array of length 30 (one per temperature)

oil_data = build_oil_pvt_data(
    pressures=pressures,
    temperatures=temperatures,
    oil_specific_gravity=0.87,
    gas_gravity=0.65,
    # Lab-measured properties override correlations
    bubble_point_pressures=lab_bubble_points,
    formation_volume_factor_table=lab_oil_fvf,
    solution_gas_to_oil_ratio_table=lab_oil_gor,
    # Viscosity and density will fall back to correlations
)
```

The `bubble_point_pressures` argument can be either a 1D array of shape `(n_temperatures,)` representing $P_b(T)$, or a 2D array of shape `(n_rs, n_temperatures)` representing $P_b(R_s, T)$. If you pass a 2D table, you must also supply `solution_gas_to_oil_ratios` - the Rs axis for that table.

!!! info "What is Bubble Point Pressure?"
    The bubble point pressure $P_b$ is the pressure at which the first gas bubble comes out of solution from undersaturated oil. Above the bubble point, all gas is dissolved in the oil phase (undersaturated oil). Below it, free gas exists alongside oil (saturated oil). BORES uses $P_b$ to switch between correlation regimes for viscosity and FVF, so accurate bubble point data significantly improves simulation fidelity.

### Salinity-Dependent Water Properties

Water properties - viscosity, density, FVF, and compressibility - all depend on salinity in addition to pressure and temperature. If your reservoir has spatially varying formation water salinity, or if you want to cover a range of salinities for generality, you can supply a salinity grid to make the water tables 3D.

```python
from bores.tables.pvt import build_water_pvt_data
import numpy as np

salinities = np.array([10000, 50000, 100000, 200000], dtype=np.float32)  # ppm NaCl

water_data = build_water_pvt_data(
    pressures=pressures,
    temperatures=temperatures,
    salinities=salinities,          # Makes all water tables 3D: (n_p, n_t, n_s)
    gas_gravity=0.65,
)
```

With a salinity array provided, all water property tables become 3D arrays of shape `(n_pressures, n_temperatures, n_salinities)`. The interpolators built from these tables then accept a salinity argument at lookup time, enabling full three-dimensional interpolation for water properties.

---

## Building `PVTTables` (the Interpolators)

Once you have a `PVTDataSet`, you need to wrap it in a `PVTTables` object. This step builds the scipy interpolators that do the actual work during simulation runtime. It is a one-time cost - after the interpolators are built, each property lookup is essentially O(1).

```python
from bores.tables.pvt import PVTTables

pvt_tables = PVTTables.from_dataset(
    dataset=pvt_dataset,
    interpolation_method="linear",
    validate=True,
    warn_on_extrapolation=False,
)
```

You can also build `PVTTables` directly from individual `PVTData` files saved to disk:

```python
pvt_tables = PVTTables.from_files(
    oil="data/oil_pvt.h5",
    gas="data/gas_pvt.h5",
    water="data/water_pvt.h5",
    interpolation_method="linear",
)
```

### Interpolation Methods

BORES supports two interpolation methods, each appropriate for different scenarios.

| Method | Speed | Accuracy | Minimum Grid Points |
|---|---|---|---|
| `"linear"` | Fast | First-order (piecewise bilinear) | 2 per dimension |
| `"cubic"` | Slower | Third-order (smooth derivatives) | 4 per dimension |

Linear interpolation is the right choice for most production simulations. It is robust, does not overshoot between data points, and performs well even with coarser grids. Use cubic interpolation when your solver depends on smooth property derivatives - for example, in fully implicit schemes where the Jacobian includes PVT derivatives. Cubic interpolation requires at least 4 grid points along each axis.

!!! tip "Grid Density for Cubic Interpolation"
    If you choose cubic interpolation, increase your grid density in regions with strong property gradients. Oil viscosity near the bubble point, for instance, can change rapidly with pressure. A grid spacing of 50-100 psi in that region will give a much more accurate cubic fit than 500 psi steps.

### Validation

When `validate=True` (the default), `PVTTables` runs a physical consistency check on your data before building interpolators. The checks include:

- All viscosities are strictly positive
- All densities are strictly positive
- All formation volume factors are strictly positive
- Gas density is less than oil density at all conditions
- 2D tables have shape `(n_pressures, n_temperatures)`
- 3D tables have shape `(n_pressures, n_temperatures, n_salinities)`

If any check fails, a `ValidationError` is raised with a clear message pointing to the problem. You can set `validate=False` to skip these checks if you are confident in your data quality and want faster initialization during batch runs.

### Property Clamping

To prevent physically impossible values from extrapolation artifacts at the boundaries of your table, `PVTTables` clamps all interpolated values to physically reasonable ranges. The defaults are:

| Property | Minimum | Maximum |
|---|---|---|
| Oil viscosity | 1e-6 cP | 10,000 cP |
| Oil compressibility | 0 psi-1 | 0.1 psi-1 |
| Oil density | 1.0 lb/ft3 | 80.0 lb/ft3 |
| Oil FVF | 0.5 bbl/STB | 5.0 bbl/STB |
| Solution GOR | 0 SCF/STB | 5,000 SCF/STB |
| Gas viscosity | 1e-6 cP | 100 cP |
| Gas z-factor | 0.1 | 3.0 |
| Water viscosity | 1e-6 cP | 10.0 cP |
| Water FVF | 0.9 bbl/STB | 2.0 bbl/STB |

You can override these defaults on a per-phase basis by passing phase-specific clamp dictionaries. Each phase table accepts its own `clamps` argument when building from raw `PVTData`:

```python
from bores.tables.pvt import PVTTable, PVTTables

# Custom clamps for oil (for a heavy oil study)
oil_clamps = {
    "viscosity": (0.1, 500.0),     # Tighter range for heavy oil
    "density": (30.0, 60.0),
}

# Custom clamps for gas
gas_clamps = {
    "compressibility_factor": (0.5, 2.5),
}

# Build phase tables with custom clamps
oil_table = PVTTable(oil_data, clamps=oil_clamps)
gas_table = PVTTable(gas_data, clamps=gas_clamps)
water_table = PVTTable(water_data)  # Uses default clamps

# Then assemble into PVTTables
pvt_tables = PVTTables(oil=oil_table, gas=gas_table, water=water_table)
```

Alternatively, if you want to apply the same clamp overrides to all three phases uniformly during dataset conversion, you can pass `clamps=False` to disable clamping entirely and handle bounds checking separately:

```python
# Disable clamping if you prefer to manage property bounds yourself
pvt_tables = PVTTables.from_dataset(
    dataset=pvt_dataset,
    clamps=False,  # Disable automatic clamping
)
```

The clamp overrides are merged on top of the phase-appropriate defaults, so you only need to specify the properties you want to change within each phase's clamps dictionary.

### Memory Management

After building the interpolators, the raw `PVTDataSet` is no longer needed for simulation. If memory is tight - common when running large 3D models or ensembles - you can discard the raw data:

```python
import gc

pvt_dataset = build_pvt_dataset(...)
pvt_tables = PVTTables.from_dataset(pvt_dataset)

# Free approximately 50% of PVT-related memory
del pvt_dataset
gc.collect()
```

If you need the raw data later (for inspection or re-building with different interpolation settings), you can always recover it from the `PVTTables` object via the `dataset` property, which reconstructs a `PVTDataSet` from the data stored inside each `PVTTable`.

---

## Querying PVT Properties

The `PVTTables` object exposes individual phase tables through its `.oil`, `.gas`, and `.water` attributes. Each phase table provides a consistent set of methods for looking up properties at given pressure and temperature conditions.

```python
# Build tables first (as shown above)
pvt_tables = PVTTables.from_dataset(pvt_dataset)

# Look up oil properties at a single point
pressure = 2500.0       # psi
temperature = 180.0     # F

oil_viscosity = pvt_tables.oil.viscosity(pressure, temperature)
oil_fvf = pvt_tables.oil.formation_volume_factor(pressure, temperature)
oil_density = pvt_tables.oil.density(pressure, temperature)
solution_gor = pvt_tables.oil.solution_gas_to_oil_ratio(pressure, temperature)
bubble_point = pvt_tables.oil.bubble_point_pressure(temperature)

print(f"Oil viscosity:   {oil_viscosity:.3f} cP")
print(f"Oil FVF:         {oil_fvf:.4f} bbl/STB")
print(f"Solution GOR:    {solution_gor:.1f} SCF/STB")
print(f"Bubble point:    {bubble_point:.1f} psi")
```

All lookup methods also accept numpy arrays for vectorized evaluation over a pressure-temperature grid:

```python
import numpy as np

# Evaluate over a 2D pressure-temperature grid
p_grid, t_grid = np.meshgrid(pressures, temperatures, indexing="ij")

# Batch lookup - result has the same shape as the input arrays
viscosity_surface = pvt_tables.oil.viscosity(p_grid.ravel(), t_grid.ravel())
viscosity_surface = viscosity_surface.reshape(p_grid.shape)
```

### Oil Phase Methods

The oil `PVTTable` provides the following methods:

- **`viscosity(pressure, temperature)`** - Oil viscosity in cP. Above the bubble point, applies the Beggs-Robinson undersaturated correction automatically.
- **`density(pressure, temperature)`** - Live oil density in lb/ft3.
- **`formation_volume_factor(pressure, temperature)`** - Oil FVF in bbl/STB. Above the bubble point, applies the McCain compressibility correction.
- **`compressibility(pressure, temperature)`** - Oil compressibility in psi-1.
- **`solution_gas_to_oil_ratio(pressure, temperature)`** - Solution GOR in SCF/STB. Frozen at the bubble point value for undersaturated conditions.
- **`bubble_point_pressure(temperature)`** - Bubble point pressure in psi.
- **`is_saturated(pressure, temperature)`** - Returns a boolean (or boolean array) indicating whether conditions are at or below the bubble point.
- **`specific_gravity(pressure, temperature)`** - Dimensionless specific gravity relative to water.
- **`molecular_weight(pressure, temperature)`** - Molecular weight in lbm/lb-mol.

!!! info "Undersaturated Oil Corrections"
    When reservoir pressure exceeds the bubble point, oil is undersaturated - all gas is dissolved and the oil is compressed as a single-phase liquid. BORES automatically applies two standard corrections in this regime. Viscosity uses the Beggs-Robinson (1975) / Vazquez-Beggs (1980) correction: $\mu_o = \mu_{ob} (P / P_b)^X$ where X captures pressure sensitivity. FVF uses the McCain compressibility correction: $B_o = B_{ob} \exp(-\bar{c}_o (P - P_b))$ where $\bar{c}_o$ is the average oil compressibility. Both corrections require a bubble point table to be provided.

### Gas Phase Methods

- **`viscosity(pressure, temperature)`** - Gas viscosity in cP via Lee-Kesler correlation.
- **`density(pressure, temperature)`** - Gas density in lb/ft3.
- **`formation_volume_factor(pressure, temperature)`** - Gas FVF in ft3/SCF.
- **`compressibility(pressure, temperature)`** - Gas compressibility in psi-1.
- **`compressibility_factor(pressure, temperature)`** - Z-factor (dimensionless).
- **`solubility_in_water(pressure, temperature, salinity=None)`** - Gas solubility in formation water in SCF/STB. Requires a salinity grid to have been provided when building the gas data.
- **`specific_gravity(pressure, temperature)`** - Gas specific gravity relative to air.
- **`molecular_weight(pressure, temperature)`** - Molecular weight in lbm/lb-mol.

### Water Phase Methods

Water lookups require a salinity argument. If you omit it, BORES falls back to the `default_salinity` set when the table was built (the first value in the salinities array).

```python
# Water lookup with explicit salinity
salinity = 50000.0   # ppm NaCl

water_viscosity = pvt_tables.water.viscosity(pressure, temperature, salinity=salinity)
water_fvf = pvt_tables.water.formation_volume_factor(pressure, temperature, salinity=salinity)
water_density = pvt_tables.water.density(pressure, temperature, salinity=salinity)
water_compressibility = pvt_tables.water.compressibility(pressure, temperature, salinity=salinity)
```

Available water methods: `viscosity`, `density`, `formation_volume_factor`, `compressibility`, `specific_gravity`, `molecular_weight`, and `bubble_point_pressure` (which additionally requires the `pressure` argument).

---

## Using PVT Tables in a Simulation

Pass the `PVTTables` object to your `Config` using the `pvt_tables` parameter. When present, BORES routes all fluid property lookups through the interpolators instead of the built-in correlations.

```python
import bores
import numpy as np
from bores.tables.pvt import build_pvt_dataset, PVTTables

bores.use_32bit_precision()

pressures = np.linspace(400, 3500, 60)
temperatures = np.linspace(120, 220, 20)

pvt_dataset = build_pvt_dataset(
    pressures=pressures,
    temperatures=temperatures,
    oil_specific_gravity=0.85,
    gas_gravity=0.65,
    water_salinity=40000.0,
)

pvt_tables = PVTTables.from_dataset(pvt_dataset)

# Pass to Config
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pvt_tables=pvt_tables,      # Tables override correlations everywhere
)

model = bores.ReservoirModel.from_file("reservoir.h5")
for state in bores.run(model, config):
    print(f"Step {state.step}: avg pressure = {state.average_pressure:.1f} psi")
```

PVT tables affect every part of the simulation that uses fluid properties: the pressure equation coefficients, mobility calculations, well rate conversions, capillary pressure evaluation, and phase flux computations. You are replacing the entire fluid property engine, not just a subset of it.

If you need to attach PVT tables to a specific injected fluid rather than the entire simulation, see [Fluids](../fluids.md).

---

## Saving and Loading PVT Data

PVT data and tables are fully serializable. The recommended workflow is to build your tables once, save them to disk, and load them for each simulation run. This avoids recomputing the same correlation-based tables repeatedly and ensures reproducibility.

```python
# Build and save the raw data (no interpolators)
pvt_dataset = build_pvt_dataset(...)
pvt_dataset.save("run/pvt_data.h5")

# Load and build tables with desired interpolation settings
from bores.tables.pvt import PVTDataSet

pvt_dataset = PVTDataSet.load("run/pvt_data.h5")
pvt_tables = PVTTables.from_dataset(pvt_dataset, interpolation_method="linear")
```

You can also save the full `PVTTables` object (including interpolator metadata):

```python
pvt_tables.save("run/pvt_tables.h5")

# Load later - interpolators are rebuilt automatically
pvt_tables_loaded = PVTTables.load("run/pvt_tables.h5")
```

The `PVTDataSet` approach is generally preferred for long-term storage because it stores only the raw numbers without depending on scipy's internal interpolator format, which can change between library versions. Use `PVTTables.save()` when you need to share a fully configured setup with a specific interpolation method already chosen.

For loading per-phase files individually - for example, when oil and water data were generated by different tools:

```python
pvt_tables = PVTTables.from_files(
    oil="data/oil_pvt.h5",
    gas="data/gas_pvt.h5",
    water="data/water_pvt.h5",
)
```

---

## Extrapolation Behavior and Table Range Selection

When simulation pressure or temperature falls outside your table bounds, the interpolators extrapolate using the same method (linear or cubic) configured for interpolation. Modest extrapolation is usually fine. Significant extrapolation - more than 10-15% beyond your grid boundaries - can produce physically unreasonable values.

To detect extrapolation during a run, enable warnings:

```python
pvt_tables = PVTTables.from_dataset(
    dataset=pvt_dataset,
    warn_on_extrapolation=True,
)
```

This logs a warning each time a query falls outside the table bounds, showing the queried range and the table limits. If you see these warnings, extend your grid to cover the full simulation range.

!!! tip "Choosing Table Ranges"
    Build your pressure grid to extend 10-15% beyond your initial and final expected pressures. If your reservoir starts at 3,000 psi and you expect depletion to 500 psi, set your pressure range from 400 to 3,500 psi. Similarly, if you expect temperature variations due to injection (cold water injected into a hot reservoir), make sure your temperature range covers the coldest injected fluid temperature. This margin prevents extrapolation during normal operation without making the table unnecessarily large.

!!! warning "Temperature Variation in Isothermal Simulations"
    Even if you are running an isothermal simulation (constant temperature), BORES still requires a valid temperature range in your PVT tables because the table system is general. For isothermal cases, you only need a narrow temperature range - for example, two temperature points bracketing your reservoir temperature by a few degrees. A single-point temperature table is not supported because the bivariate spline interpolator requires at least 2 points per dimension (4 for cubic).

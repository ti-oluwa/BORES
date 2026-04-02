# Fluid Properties

In reservoir simulation, fluid properties describe how oil, gas, and water behave as pressure and temperature change throughout the reservoir. The industry term for these is PVT properties, which stands for Pressure-Volume-Temperature, because those are the primary variables that govern fluid behavior. Getting PVT properties right matters: the same reservoir with different fluid characterization can yield very different pressure responses, recovery estimates, and production forecasts.

BORES takes a practical approach to fluid properties. When you call `bores.reservoir_model()`, you provide the minimum field-available inputs - initial pressure, temperature, oil gravity, and gas gravity - and BORES computes the full suite of derived properties automatically using industry-standard correlations. You do not need a laboratory PVT report to run your first simulation. As your study matures and you acquire measured data, you can replace correlation-computed properties with your own values selectively, one property at a time, without changing anything else.

For cases where you have full PVT laboratory reports or equation-of-state results, BORES also supports tabulated PVT data through its `PVTTables` system. See [PVT Tables](advanced/pvt-tables.md) for that workflow. And for simulations involving gas injection or CO2 flooding where you need to model specific injected fluid behavior, the `Fluid` class described in the [Fluids](fluids.md) page extends these properties with injection-specific capabilities. See [Fluids](fluids.md) for details on custom gas characterization and pseudo-pressure too.

---

## Oil Properties

Oil is the most complex phase to characterize because its properties depend on whether it is saturated (at or below the bubble point, with free gas present) or undersaturated (above the bubble point, with all gas in solution). The correlations BORES uses automatically account for this distinction.

### Key Oil Inputs

| Parameter | Units | Typical Range | Description |
|---|---|---|---|
| `oil_viscosity_grid` | cP | 0.5 to 100+ | Dead oil viscosity at reservoir temperature |
| `oil_specific_gravity_grid` | dimensionless | 0.75 to 0.95 | Oil specific gravity relative to water |
| `oil_bubble_point_pressure_grid` | psi | 500 to 5,000 | Pressure at which gas first liberates from oil |

Oil specific gravity and API gravity are related by:

$$\text{API} = \frac{141.5}{\gamma_o} - 131.5$$

where $\gamma_o$ is the oil specific gravity (water = 1.0). An oil specific gravity of 0.85 corresponds to approximately 35 degrees API, which is a light crude oil. Values above 45 API are condensates; values below 20 API are heavy crudes that may require specialized viscosity treatment.

```python
import bores
import numpy as np

grid_shape = (20, 20, 5)

# 35 API light crude
oil_sg = bores.build_uniform_grid(grid_shape, value=0.85)

# Dead oil viscosity at reservoir temperature (cP)
oil_visc = bores.build_uniform_grid(grid_shape, value=1.5)

# Bubble point pressure - typical for a solution-gas drive reservoir
bubble_pt = bores.build_uniform_grid(grid_shape, value=2500.0)
```

### Properties Computed by the Factory

From your oil inputs plus pressure and temperature, `bores.reservoir_model()` computes the following properties automatically:

The **solution gas-oil ratio** $R_s$ (SCF/STB) represents the volume of gas dissolved per volume of oil at surface conditions. BORES uses Standing's correlation to compute $R_s$ as a function of pressure, temperature, API gravity, and gas gravity. Below the bubble point, $R_s$ decreases as pressure drops and gas liberates from solution.

The **oil formation volume factor** $B_o$ (bbl/STB) is the ratio of oil volume at reservoir conditions to oil volume at surface conditions. It is always greater than 1.0 because dissolved gas expands the oil at reservoir pressure. BORES uses an iterative bootstrap procedure to resolve the circular dependency between $B_o$ and oil compressibility $c_o$: above the bubble point, $B_o = B_{ob} \exp(-c_o (P - P_b))$ requires $c_o$, while the liberation term in $c_o$ below $P_b$ requires $B_o$. The bootstrap converges in 2-3 iterations for typical light to medium oils.

The **oil compressibility** $c_o$ (psi⁻¹) quantifies how much oil volume changes per unit pressure change. It is small for liquid oil (typically $10^{-5}$ to $10^{-6}$ psi⁻¹) but becomes larger near the bubble point due to the gas liberation contribution.

The **live oil density** (lb/ft³) accounts for dissolved gas, which is lighter than oil. BORES computes it from $B_o$, $R_s$, and the phase gravities.

The **live oil viscosity** (cP) is adjusted from your input dead oil viscosity using the Beggs-Robinson correlation, which accounts for gas in solution reducing viscosity significantly below the bubble point.

!!! info "Bubble Point and Solution Gas"
    The bubble point pressure $P_b$ is the pressure at which the first bubble of free gas forms in a previously undersaturated oil. Above $P_b$, the oil is a single compressed liquid phase. Below $P_b$, free gas coexists with oil, and the solution GOR $R_s$ decreases as more gas liberates. This liberation drives reservoir energy (solution gas drive) and is one of the primary production mechanisms in many conventional reservoirs.

### Providing Measured Values

If you have laboratory measurements for any oil property, pass them as grids and BORES will use them instead of the correlation-computed values. You can mix measured and correlation-computed properties freely:

```python
import bores
import numpy as np

grid_shape = (20, 20, 5)

# Lab-measured bubble point and solution GOR
lab_pb = np.loadtxt("measured_pb.csv").reshape(grid_shape)
lab_rs = np.loadtxt("measured_rs.csv").reshape(grid_shape)

model = bores.reservoir_model(
    grid_shape=grid_shape,
    # ...other required parameters...
    oil_bubble_point_pressure_grid=lab_pb,      # Uses your lab data
    solution_gas_to_oil_ratio_grid=lab_rs,      # Uses your lab data
    # Bo, Co, viscosity still computed from correlations using your Pb and Rs
)
```

If you supply `oil_bubble_point_pressure_grid` but omit `solution_gas_to_oil_ratio_grid`, BORES derives $R_s$ from your $P_b$ using the Standing correlation. If you supply $R_s$ but not $P_b$, BORES derives $P_b$ from $R_s$. If you supply neither, BORES uses an iterative coupled Vazquez-Beggs and Standing solver that accounts for temperature variation across the grid.

---

## Gas Properties

Gas behavior deviates substantially from ideal gas at reservoir conditions because molecules are close together at high pressure. The compressibility factor $Z$ captures this deviation. The real gas law is:

$$PV = ZnRT$$

where $P$ is pressure, $V$ is volume, $n$ is moles, $R$ is the universal gas constant, and $T$ is absolute temperature. When $Z = 1$ the gas behaves ideally. At typical reservoir conditions (2,000 to 5,000 psi, 150 to 250 F), $Z$ ranges from about 0.7 to 0.95 for typical natural gas.

### Key Gas Inputs

| Parameter | Units | Typical Range | Description |
|---|---|---|---|
| `gas_gravity_grid` | dimensionless | 0.55 to 1.5 | Gas specific gravity relative to air |

The gas gravity is the primary input for all gas property correlations. If you do not provide `gas_gravity_grid`, BORES defaults to methane ($\gamma_g = 0.553$). You can also supply `gas_gravity` as a scalar to the `reservoir_model()` factory and BORES will broadcast it to a uniform grid.

!!! info "Common Gas Gravity Values"
    Methane (CH4): 0.553. Typical natural gas: 0.60 to 0.75. The gas gravity increases with heavier components (ethane, propane, CO2). CO2 has a gas gravity of 1.52 relative to air, which is important if you are modeling CO2 injection. Nitrogen has a gas gravity of 0.967.

### Properties Computed by the Factory

The **gas compressibility factor** $Z$ is computed using the Hall-Yarborough correlation, which is based on pseudo-reduced pressure $P_{pr}$ and pseudo-reduced temperature $T_{pr}$ calculated from the gas gravity.

The **gas formation volume factor** $B_g$ (ft³/SCF) relates gas volume at reservoir conditions to surface conditions:

$$B_g = \frac{ZT}{P} \cdot 0.02829$$

where $T$ is in Rankine and $P$ is in psia. $B_g$ typically ranges from 0.003 to 0.01 ft³/SCF at reservoir conditions, meaning one SCF of surface gas expands to 100 to 300 SCF at reservoir pressure when brought to surface.

The **gas viscosity** $\mu_g$ (cP) is computed using the Lee-Gonzalez-Eakin correlation. Gas viscosity is much lower than oil viscosity, typically 0.01 to 0.05 cP at reservoir conditions, which is why gas has very high mobility.

The **gas compressibility** $c_g$ (psi⁻¹) is approximately $1/P$ for ideal gas and is corrected for Z-factor deviation at actual conditions.

---

## Water Properties

Formation water typically has higher salinity than seawater, which affects its density, viscosity, and FVF compared to fresh water. BORES computes water properties from pressure, temperature, and salinity.

### Key Water Inputs

| Parameter | Units | Typical Range | Description |
|---|---|---|---|
| `water_salinity_grid` | ppm NaCl | 0 to 300,000 | Formation water salinity |

If you do not provide `water_salinity_grid`, BORES defaults to the internal constant `c.DEFAULT_WATER_SALINITY_PPM`. You can override salinity spatially if your reservoir has different salinity zones across the grid.

```python
import bores
import numpy as np

grid_shape = (20, 20, 5)

# Spatially varying salinity - higher in deeper zones
salinity = np.ones(grid_shape) * 50000.0    # 50,000 ppm background
salinity[:, :, 3:] = 100000.0               # 100,000 ppm in bottom two layers

model = bores.reservoir_model(
    grid_shape=grid_shape,
    # ...other required parameters...
    water_salinity_grid=salinity,
)
```

### Properties Computed by the Factory

The **water formation volume factor** $B_w$ (bbl/STB) is close to 1.0 for typical formation water (1.02 to 1.08) because water is nearly incompressible. It increases slightly with temperature and decreases with salinity.

The **water viscosity** $\mu_w$ (cP) ranges from about 0.4 cP at high reservoir temperatures to 1.0 cP at lower temperatures. Salinity has a modest effect on viscosity.

The **water compressibility** $c_w$ (psi⁻¹) is computed by the McCain correlation and includes a dissolved gas liberation correction term.

---

## PVT Correlations Reference

BORES uses the following correlations for properties not supplied directly:

| Property | Correlation |
|---|---|
| Bubble point pressure | Standing (1947) |
| Solution GOR ($R_s$) | Standing (1947) / Vazquez-Beggs (1980) |
| Oil FVF ($B_o$) | Standing (1947) / Vasquez-Beggs (1980) |
| Oil compressibility ($c_o$) | Vasquez-Beggs (1980) |
| Dead oil viscosity | Beggs-Robinson (1975) |
| Live oil viscosity | Beggs-Robinson (1975) |
| Gas Z-factor | Dranchuk-Abou-Kassem (1975) |
| Gas viscosity | Lee-Gonzalez-Eakin (1966) |
| Water properties | McCain (1990) |

!!! warning "Correlation Limitations"
    These correlations are empirical fits derived from large datasets of conventional light-to-medium crude oils (roughly 25 to 45 API) with typical hydrocarbon gases. Accuracy degrades for heavy oils below 20 API, near-critical fluids, CO2-rich systems, and high-temperature high-pressure environments. If your reservoir falls outside these ranges, provide measured PVT data via [PVT Tables](advanced/pvt-tables.md) rather than relying on correlations.

---

## Accessing Fluid Properties on a Model

After building a model, all computed fluid properties are available as numpy arrays on `model.fluid_properties`. Every array has the same shape as your grid.

```python
import bores

model = bores.reservoir_model(...)

fp = model.fluid_properties

# Pressure and saturation
pressure = fp.pressure_grid                         # psi
So = fp.oil_saturation_grid                         # fraction
Sw = fp.water_saturation_grid                       # fraction
Sg = fp.gas_saturation_grid                         # fraction

# Oil PVT
Rs  = fp.solution_gas_to_oil_ratio_grid             # SCF/STB
Bo  = fp.oil_formation_volume_factor_grid           # bbl/STB
co  = fp.oil_compressibility_grid                   # psi⁻¹
Pb  = fp.oil_bubble_point_pressure_grid             # psi
rho_o = fp.oil_density_grid                         # lb/ft³
mu_o  = fp.oil_viscosity_grid                       # cP

# Gas PVT
Bg    = fp.gas_formation_volume_factor_grid         # ft³/SCF
z     = fp.gas_compressibility_factor_grid          # dimensionless
cg    = fp.gas_compressibility_grid                 # psi⁻¹
rho_g = fp.gas_density_grid                         # lb/ft³
mu_g  = fp.gas_viscosity_grid                       # cP

# Water PVT
Bw    = fp.water_formation_volume_factor_grid       # bbl/STB
cw    = fp.water_compressibility_grid               # psi⁻¹
rho_w = fp.water_density_grid                       # lb/ft³
mu_w  = fp.water_viscosity_grid                     # cP

# Saliniy and gas gravity (constant for a given fluid type)
salinity  = fp.water_salinity_grid                  # ppm NaCl
gg        = fp.gas_gravity_grid                     # dimensionless
```

These arrays reflect the initial conditions you provided. As the simulation runs, BORES updates them at each time step to reflect the evolving pressure and saturation distribution. When you access fluid properties on a `ModelState` yielded during simulation, you are reading the state at that specific time step.

---

## How `reservoir_model()` Assembles Fluid Properties

The factory follows a deterministic resolution order, checking at each step whether you supplied a value and falling back to correlations only if you did not. Understanding this order helps you predict which correlation will be used for any given property:

1. Validate grid shapes, pressure and temperature ranges, and saturation consistency.
2. Resolve gas gravity: use `gas_gravity_grid` if provided, otherwise compute from `reservoir_gas` identity.
3. Compute gas Z-factor, $B_g$, $c_g$, and $\mu_g$ from gas gravity and reservoir conditions.
4. Resolve oil API gravity and specific gravity.
5. Resolve $P_b$ and $R_s$: four cases based on which combination you provided (both, only $P_b$, only $R_s$, or neither). See the parameter docstring for `reservoir_model()` for the full logic.
6. Resolve $B_o$ and $c_o$ using iterative bootstrap if both are missing, or a single pass if one is provided.
7. Compute oil density, live oil viscosity, and gas solubility in water.
8. Resolve water density, $B_w$, $c_w$, and $\mu_w$ from pressure, temperature, and salinity.
9. Assemble `FluidProperties`, `RockProperties`, and `SaturationHistory` into a `ReservoirModel`.

Because all grid fields in `FluidProperties` are stored as numpy arrays, you can inspect the full set of computed properties before running any simulation by examining `model.fluid_properties` directly after calling `bores.reservoir_model()`.

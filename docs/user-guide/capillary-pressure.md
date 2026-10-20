# Capillary Pressure

## What is Capillary Pressure?

When two immiscible fluids share the same pore space, the interface between them curves due to differences in how strongly each fluid adheres to the rock surface. This curvature creates a pressure difference across the fluid-fluid interface, and that pressure difference is called capillary pressure. It is defined as the pressure in the non-wetting phase minus the pressure in the wetting phase:

$$P_c = P_{nw} - P_w$$

In a water-wet rock, water preferentially coats the grain surfaces and occupies the smaller pores, making water the wetting phase and oil the non-wetting phase. The oil-water capillary pressure $P_{cow} = P_o - P_w$ is positive, meaning oil pressure exceeds water pressure. This pressure difference is what allows oil to occupy larger pores while water fills the smaller ones. The magnitude of capillary pressure depends on the pore throat size, the interfacial tension between the fluids, the contact angle of the fluid-rock system, and the saturation state.

Capillary pressure plays a critical role in controlling the initial fluid distribution in a reservoir. Before any production begins, the reservoir is in capillary-gravity equilibrium: the capillary pressure at every point balances the gravity head caused by the density difference between the fluids. This equilibrium determines the transition zone, the region between the free water level and the depth where oil saturation reaches its maximum. In thick transition zones (common in tight rocks with high capillary pressure), water saturation decreases gradually with height above the free water level, and the oil column is never fully "clean" of water. In high-permeability rocks with low capillary pressure, the transition zone is thin and the contacts are sharp.

During simulation, capillary pressure affects flow by modifying the pressure field. The pressure equation for each phase includes a capillary pressure correction: $P_o = P_w + P_{cow}$ and $P_g = P_o + P_{cgo}$. This means capillary forces act as an additional driving force (or resistance) to flow. In waterflooding, capillary pressure causes imbibition, where water spontaneously enters oil-filled pores even without an external pressure gradient. In gas injection, capillary entry pressure can prevent gas from entering tight rock layers, creating capillary barriers that trap gas.

In a three-phase system (oil, water, gas), BORES tracks two capillary pressures:

- **Oil-water capillary pressure**: $P_{cow} = P_o - P_w$ (positive in water-wet rock)
- **Gas-oil capillary pressure**: $P_{cgo} = P_g - P_o$ (positive because gas is always non-wetting relative to oil)

Both depend on saturation and are computed by the capillary pressure model at every grid cell and every timestep.

---

## The Brooks-Corey Model

The Brooks-Corey capillary pressure model is the most widely used analytical correlation in petroleum engineering. It relates capillary pressure to the effective saturation through a power law based on two parameters: the entry pressure and the pore size distribution index.

The model is expressed as:

$$P_c = P_d \cdot S_e^{-1/\lambda}$$

where $P_d$ is the entry (or displacement) pressure in psi, $S_e$ is the effective (normalized) saturation, and $\lambda$ is the pore size distribution index. The entry pressure $P_d$ represents the minimum capillary pressure needed for the non-wetting phase to enter the largest pore throats. Rocks with large, well-connected pores have low entry pressures (1-5 psi), while tight rocks with small pore throats have high entry pressures (10-50+ psi).

For the gas-oil system, $S_e$ is the effective saturation of the wetting liquid: $S_{e} = \dfrac{1 - S_{wc} - S_g - S_{or,g}}{1 - S_{wc} - S_{gr} - S_{or,g}}$. It falls as gas displaces oil, so $P_{cgo}$ rises with gas saturation (and is zero for Brooks-Corey where there is no mobile gas). The capillary pressure has no slope where $S_e$ is clipped.

The pore size distribution index $\lambda$ controls how rapidly capillary pressure changes with saturation. A large $\lambda$ (say 4-6) means the pore sizes are narrowly distributed (uniform rock), so capillary pressure changes gradually and the transition zone is narrow. A small $\lambda$ (say 0.5-1.5) means the pore sizes vary widely, producing a broad transition zone where capillary pressure changes steeply near residual saturation. Most reservoir sandstones have $\lambda$ values between 1.5 and 4.0, while carbonates tend to have lower values due to their more heterogeneous pore structure.

### Basic Usage

```python
import bores

capillary_pressure = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet=5.0,  # psi
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=1.0,  # psi
    gas_oil_pore_size_distribution_index=2.0,
)
```

This creates a water-wet Brooks-Corey model with an oil-water entry pressure of 5 psi, a gas-oil entry pressure of 1 psi, and a pore size distribution index of 2.0 for both systems. The gas-oil entry pressure is typically lower than the oil-water entry pressure because the gas-oil interfacial tension is lower than oil-water interfacial tension.

Like the relative permeability model, residual saturations can be set on the model or inherited from the reservoir model. When the saturation endpoints are set to `None` (the default), BORES uses the endpoint grids from the reservoir model automatically. If you provide explicit values, those override the grid-level defaults.

```python
capillary_pressure = bores.BrooksCoreyCapillaryPressureModel(
    irreducible_water_saturation=0.25,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.05,
    oil_water_entry_pressure_water_wet=5.0,
    oil_water_pore_size_distribution_index_water_wet=2.5,
    gas_oil_entry_pressure=1.5,
    gas_oil_pore_size_distribution_index=2.0,
)
```

### Brooks-Corey Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `irreducible_water_saturation` | `None` | Connate water saturation $S_{wc}$. If `None`, uses grid values. |
| `residual_oil_saturation_water` | `None` | Residual oil to waterflood $S_{or,w}$. If `None`, uses grid values. |
| `residual_oil_saturation_gas` | `None` | Residual oil to gas flood $S_{or,g}$. If `None`, uses grid values. |
| `residual_gas_saturation` | `None` | Trapped gas saturation $S_{gr}$. If `None`, uses grid values. |
| `oil_water_entry_pressure_water_wet` | `5.0` | Entry pressure for oil-water system, water-wet (psi) |
| `oil_water_entry_pressure_oil_wet` | `5.0` | Entry pressure for oil-water system, oil-wet (psi) |
| `oil_water_pore_size_distribution_index_water_wet` | `2.0` | Pore size distribution index $\lambda$ for oil-water, water-wet |
| `oil_water_pore_size_distribution_index_oil_wet` | `2.0` | Pore size distribution index $\lambda$ for oil-water, oil-wet |
| `gas_oil_entry_pressure` | `1.0` | Entry pressure for gas-oil system (psi) |
| `gas_oil_pore_size_distribution_index` | `2.0` | Pore size distribution index $\lambda$ for gas-oil |
| `wettability` | `WATER_WET` | Rock wettability (`WATER_WET`, `OIL_WET`, or `MIXED_WET`) |
| `mixed_wet_water_fraction` | `0.5` | Fraction of pore space that is water-wet (for `MIXED_WET` only) |

### Wettability Effects on Capillary Pressure

Wettability fundamentally changes the sign and magnitude of capillary pressure. In a water-wet system, oil must overcome the capillary entry pressure to enter a pore, so $P_{cow} > 0$. In an oil-wet system, the rock surface prefers oil, and water must overcome an entry pressure to displace oil, so $P_{cow} < 0$. In a mixed-wet system, BORES computes a weighted average of the water-wet and oil-wet capillary pressures.

```python
import bores

# Water-wet system: positive Pcow
capillary_pressure_water_wet = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet=5.0,
    oil_water_pore_size_distribution_index_water_wet=2.5,
    gas_oil_entry_pressure=1.0,
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.Wettability.WATER_WET,
)

# Oil-wet system: negative Pcow
capillary_pressure_oil_wet = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_oil_wet=4.0,
    oil_water_pore_size_distribution_index_oil_wet=2.0,
    gas_oil_entry_pressure=1.0,
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.Wettability.OIL_WET,
)

# Mixed-wet system: 60% water-wet, 40% oil-wet pore surfaces
capillary_pressure_mixed_wet = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet=5.0,
    oil_water_entry_pressure_oil_wet=4.0,
    oil_water_pore_size_distribution_index_water_wet=2.5,
    oil_water_pore_size_distribution_index_oil_wet=2.0,
    gas_oil_entry_pressure=1.0,
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.Wettability.MIXED_WET,
    mixed_wet_water_fraction=0.6,
)
```

The mixed-wet option is particularly useful for carbonate reservoirs and aged sandstones where the wettability is intermediate. The `mixed_wet_water_fraction` parameter controls the weighting: a value of 0.6 means 60% of the pore surface area is water-wet and 40% is oil-wet. The resulting capillary pressure is a linear combination of the two end-member curves, which can produce the characteristic "crossover" behavior seen in mixed-wet SCAL data where capillary pressure changes sign at intermediate saturations.

!!! info "Entry Pressure and Rock Quality"

    Entry pressure correlates inversely with permeability. High-permeability sands (500+ mD) may have entry pressures below 1 psi, while tight sandstones (1-10 mD) can have entry pressures of 20-50 psi. A useful rule of thumb: $P_d \approx C / \sqrt{k}$ where $C$ is an empirical constant (roughly 3-10 depending on the rock type) and $k$ is permeability in mD.

---

## The Van Genuchten Model

The Van Genuchten model is an alternative to Brooks-Corey that provides smoother capillary pressure curves, particularly near the residual saturations where Brooks-Corey produces infinite values. This smoothness makes the Van Genuchten model numerically better-behaved and is often preferred for simulations where convergence near residual saturation is important.

The model is expressed as:

$$P_c = \frac{1}{\alpha} \left[ S_e^{-1/m} - 1 \right]^{1/n}$$

where $\alpha$ is an inverse pressure parameter (in psi⁻¹), $n$ is a shape parameter that must be greater than 1, and $m = 1 - 1/n$ is derived from $n$. The parameter $\alpha$ is roughly the inverse of the entry pressure: larger $\alpha$ values produce lower capillary pressures. The parameter $n$ controls the curve shape and has a similar role to the pore size distribution index in Brooks-Corey: higher values produce narrower transition zones.

The key advantage of Van Genuchten over Brooks-Corey is its behavior at the endpoints. Brooks-Corey capillary pressure goes to infinity as saturation approaches the residual value ($S_e \to 0$), which can cause numerical difficulties. Van Genuchten also approaches infinity but does so more gradually, which makes the transition smoother and reduces the risk of solver convergence issues.

### Van Genuchten Usage

```python
import bores

capillary_pressure = bores.VanGenuchtenCapillaryPressureModel(
    oil_water_alpha_water_wet=0.01,  # psi⁻¹ (roughly: entry pressure ~ 1/alpha)
    oil_water_n_water_wet=2.0,  # Shape parameter (must be > 1)
    gas_oil_alpha=0.02,  # psi⁻¹
    gas_oil_n=2.0,  # Shape parameter
)
```

The Van Genuchten model supports the same wettability options as Brooks-Corey, with separate parameters for the water-wet and oil-wet components. For water-wet systems, $P_{cow} > 0$; for oil-wet, $P_{cow} < 0$; and for mixed-wet, a weighted average is used.

### Van Genuchten Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `irreducible_water_saturation` | `None` | Connate water saturation $S_{wc}$. |
| `residual_oil_saturation_water` | `None` | Residual oil to waterflood $S_{or,w}$. |
| `residual_oil_saturation_gas` | `None` | Residual oil to gas flood $S_{or,g}$. |
| `residual_gas_saturation` | `None` | Trapped gas saturation $S_{gr}$. |
| `oil_water_alpha_water_wet` | `0.01` | Van Genuchten $\alpha$ for oil-water, water-wet (psi⁻¹) |
| `oil_water_alpha_oil_wet` | `0.01` | Van Genuchten $\alpha$ for oil-water, oil-wet (psi⁻¹) |
| `oil_water_n_water_wet` | `2.0` | Van Genuchten $n$ for oil-water, water-wet |
| `oil_water_n_oil_wet` | `2.0` | Van Genuchten $n$ for oil-water, oil-wet |
| `gas_oil_alpha` | `0.01` | Van Genuchten $\alpha$ for gas-oil (psi⁻¹) |
| `gas_oil_n` | `2.0` | Van Genuchten $n$ for gas-oil |
| `wettability` | `WATER_WET` | Rock wettability |
| `mixed_wet_water_fraction` | `0.5` | Water-wet pore fraction (for `MIXED_WET`) |

!!! tip "Converting Between Brooks-Corey and Van Genuchten"

    There is no exact conversion between the two models, but approximate relationships exist. The Van Genuchten $\alpha$ is roughly the inverse of the Brooks-Corey entry pressure: $\alpha \approx 1 / P_d$. The shape parameters are related by $n \approx \lambda + 1$ for $\lambda > 1$. These are rough guides for getting a starting point when switching between models.

---

## The Leverett J-Function Model

The Leverett J-function is a dimensionless correlation that normalizes capillary pressure by rock and fluid properties. This makes it possible to transfer capillary pressure data measured on one core sample to other parts of the reservoir with different porosity and permeability, without needing separate measurements for each rock type.

The J-function relates capillary pressure to rock quality through:

$$P_c = \sigma \cos\theta \sqrt{\frac{\phi}{k}} \cdot J(S_e)$$

where $\sigma$ is the interfacial tension (dyne/cm), $\theta$ is the contact angle (degrees), $\phi$ is porosity, $k$ is permeability (mD), and $J(S_e)$ is the dimensionless J-function evaluated at effective saturation. The $\sqrt{\phi/k}$ factor captures the relationship between rock quality and pore throat size: tight rocks (low $k$, low $\phi$) have smaller pores and therefore higher capillary pressures.

In BORES, the J-function uses a power-law form: $J(S_e) = a \cdot S_e^{-b}$, where $a$ is the `j_function_coefficient` and $b$ is the `j_function_exponent`. These empirical parameters are typically calibrated by fitting the J-function model to laboratory capillary pressure data from one or more core plugs.

The Leverett J-function approach is most valuable in heterogeneous reservoirs where porosity and permeability vary spatially. Because the model scales capillary pressure by the local $\sqrt{\phi/k}$, it automatically produces higher capillary pressures in tight zones and lower values in high-permeability zones, without needing separate capillary pressure curves for each rock type.

### Leverett J-Function Usage

```python
import bores

capillary_pressure = bores.LeverettJCapillaryPressureModel(
    permeability=100.0,  # mD (reference permeability)
    porosity=0.2,  # Reference porosity
    oil_water_interfacial_tension=30.0,  # dyne/cm
    gas_oil_interfacial_tension=20.0,  # dyne/cm
    oil_water_contact_angle=0.0,  # degrees (0 = water-wet)
    gas_oil_contact_angle=0.0,  # degrees
    j_function_coefficient=0.5,  # Empirical coefficient 'a'
    j_function_exponent=0.5,  # Empirical exponent 'b'
)
```

### Leverett J-Function Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `irreducible_water_saturation` | `None` | Connate water saturation $S_{wc}$. |
| `residual_oil_saturation_water` | `None` | Residual oil to waterflood $S_{or,w}$. |
| `residual_oil_saturation_gas` | `None` | Residual oil to gas flood $S_{or,g}$. |
| `residual_gas_saturation` | `None` | Trapped gas saturation $S_{gr}$. |
| `permeability` | `100.0` | Absolute permeability (mD) |
| `porosity` | `0.2` | Porosity (fraction) |
| `oil_water_interfacial_tension` | `30.0` | Oil-water IFT (dyne/cm) |
| `gas_oil_interfacial_tension` | `20.0` | Gas-oil IFT (dyne/cm) |
| `oil_water_contact_angle` | `0.0` | Oil-water contact angle (degrees) |
| `gas_oil_contact_angle` | `0.0` | Gas-oil contact angle (degrees) |
| `j_function_coefficient` | `0.5` | Coefficient $a$ in $J = a \cdot S_e^{-b}$ |
| `j_function_exponent` | `0.5` | Exponent $b$ in $J = a \cdot S_e^{-b}$ |
| `wettability` | `WATER_WET` | Rock wettability |
| `mixed_wet_water_fraction` | `0.5` | Water-wet pore fraction (for `MIXED_WET`) |

!!! warning "Contact Angle Convention"

    A contact angle of 0 degrees means the wetting phase perfectly wets the rock (the interface is flat against the surface). A contact angle of 90 degrees means neutral wettability. Contact angles above 90 degrees indicate the non-wetting phase is actually more wetting. In practice, petroleum engineers often specify wettability through the `wettability` parameter rather than manipulating contact angles directly.

---

## Tabular Capillary Pressure

When you have laboratory-measured capillary pressure data from mercury injection, centrifuge, or porous plate experiments, you can use tabular capillary pressure instead of an analytical model. BORES provides `TwoPhaseCapillaryPressureTable` for single phase-pair data and `ThreePhaseCapillaryPressureTable` for combining oil-water and gas-oil data into a complete three-phase model.

Tabular capillary pressure is the preferred approach when you have high-quality SCAL data that does not fit well to a Brooks-Corey or Van Genuchten curve. Laboratory data often shows features that analytical models cannot capture: multiple inflection points from bimodal pore size distributions, sudden changes in slope at specific saturations, or asymmetric behavior at the drainage and imbibition endpoints. Using the raw data as a table preserves all of these features.

### `TwoPhaseCapillaryPressureTable`

A `TwoPhaseCapillaryPressureTable` stores wetting phase saturation values and the corresponding capillary pressure at each saturation. You specify which fluid is the wetting phase and which is the non-wetting phase.

```python
import bores
import numpy as np

# Oil-water capillary pressure from mercury injection data
oil_water_capillary_pressure_table = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    reference_saturation=np.array([0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]),
    capillary_pressure=np.array([25.0, 12.0, 7.5, 4.2, 2.8, 1.8, 1.0, 0.3]),
    reference_phase="wetting",
)
```

The `reference_saturation` array must be monotonically increasing and contain at least two points. Capillary pressure values should decrease as wetting phase saturation increases (more water means lower capillary pressure in a water-wet system). BORES uses `np.interp` for fast linear interpolation between table points, with constant extrapolation beyond the table endpoints.

You can query the table at any saturation value or with grid arrays:

```python
# Single point query
capillary_pressure_at_04 = oil_water_capillary_pressure_table.get_capillary_pressure(0.4)

# Grid array query
Sw_grid = np.random.uniform(0.2, 0.8, size=(20, 20, 5))
capillary_pressure_grid = oil_water_capillary_pressure_table.get_capillary_pressure(Sw_grid)
```

### `ThreePhaseCapillaryPressureTable`

For three-phase simulation, combine two `TwoPhaseCapillaryPressureTable` objects into a `ThreePhaseCapillaryPressureTable`. This is the capillary pressure equivalent of the `ThreePhaseRelPermTable` and follows the same pattern.

```python
import bores
import numpy as np

# Oil-water capillary pressure (water is wetting phase, indexed by Sw)
oil_water_capillary_pressure = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    reference_saturation=np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]),
    capillary_pressure=np.array([20.0, 8.5, 4.5, 2.8, 1.5, 0.7, 0.1]),
    reference_phase="wetting",
)

# Gas-oil capillary pressure (oil is wetting phase, indexed by So)
gas_oil_capillary_pressure = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.OIL,
    non_wetting_phase=bores.FluidPhase.GAS,
    reference_saturation=np.array([0.15, 0.25, 0.40, 0.55, 0.70, 0.85]),
    capillary_pressure=np.array([8.0, 4.5, 2.0, 1.0, 0.4, 0.05]),
    reference_phase="wetting",
)

# Combine into three-phase table
three_phase_capillary_pressure = bores.ThreePhaseCapillaryPressureTable(
    oil_water_table=oil_water_capillary_pressure,
    gas_oil_table=gas_oil_capillary_pressure,
)
```

The `ThreePhaseCapillaryPressureTable` validates the phase assignments on construction. The oil-water table must have water and oil as its phases, the gas-oil table must have oil and gas. If any of these constraints are violated, BORES raises a `ValidationError`.

When computing capillary pressures during simulation, the three-phase table looks up $P_{cow}$ from the oil-water table using the wetting phase saturation and $P_{cgo}$ from the gas-oil table using the oil saturation of the gas-oil system. Each table is queried at its own system's saturations, whichever phase its reference axis uses: $S_o = 1 - S_w$ for the oil-water table and $S_o = 1 - S_{wc} - S_g$ for the gas-oil table, so $P_{cow}$ depends on $S_w$ only and $P_{cgo}$ on $S_g$ only. The two capillary pressures are independent (no mixing rule is needed, unlike relative permeability).

!!! tip "When to Use Tables vs. Analytical Models"

    Use tabular capillary pressure when you have high-quality laboratory SCAL data (mercury injection, centrifuge, or porous plate), when your pore size distribution is bimodal or otherwise poorly described by a single power law, or when you need to exactly reproduce measured capillary pressure curves for history matching. Use analytical models (Brooks-Corey or Van Genuchten) when lab data is unavailable, when you want to parametrically study the effect of entry pressure or pore size distribution, or when you need smooth derivatives for numerical stability.

---

## Direct Usage (Outside Simulation)

You do not need to run a full simulation to evaluate capillary pressure models. All models (analytical and tabular) can be called directly with saturation values, which is useful for plotting curves, comparing models, validating against laboratory data, or building custom analysis workflows. Every model supports both scalar and grid-array inputs.

### Calling Analytical Models Directly

The `BrooksCoreyCapillaryPressureModel`, `VanGenuchtenCapillaryPressureModel`, and `LeverettJCapillaryPressureModel` all support the same calling interface. Use `get_capillary_pressures()` or `__call__` with water, oil, and gas saturations. They return a dictionary with `"oil_water"` and `"gas_oil"` keys.

```python
import bores
import numpy as np

capillary_pressure = bores.BrooksCoreyCapillaryPressureModel(
    irreducible_water_saturation=0.25,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.05,
    oil_water_entry_pressure_water_wet=5.0,
    oil_water_pore_size_distribution_index_water_wet=2.5,
    gas_oil_entry_pressure=1.5,
    gas_oil_pore_size_distribution_index=2.0,
)

# Scalar evaluation
result = capillary_pressure.get_capillary_pressures(
    water_saturation=0.4,
    oil_saturation=0.55,
    gas_saturation=0.05,
)
print(f"Pcow = {result['oil_water']:.2f} psi")
print(f"Pcgo = {result['gas_oil']:.2f} psi")

# Using __call__ (same result)
result = capillary_pressure(
    water_saturation=0.4,
    oil_saturation=0.55,
    gas_saturation=0.05,
)

# Grid array evaluation (3D)
Sw = bores.build_uniform_grid((20, 20, 5), value=0.4)
So = bores.build_uniform_grid((20, 20, 5), value=0.55)
Sg = bores.build_uniform_grid((20, 20, 5), value=0.05)

result = capillary_pressure(water_saturation=Sw, oil_saturation=So, gas_saturation=Sg)
pcoil_water_grid = result["oil_water"]  # Shape: (20, 20, 5)
pcgas_oil_grid = result["gas_oil"]  # Shape: (20, 20, 5)
```

The `get_capillary_pressures()` method also accepts optional residual saturation overrides that take precedence over the model defaults. This is useful when the simulator passes cell-level saturation endpoints from the grid:

```python
result = capillary_pressure.get_capillary_pressures(
    water_saturation=0.35,
    oil_saturation=0.60,
    gas_saturation=0.05,
    irreducible_water_saturation=0.20,  # Override model default
    residual_oil_saturation_water=0.30,  # Override model default
)
```

The Van Genuchten and Leverett J-function models use the exact same calling convention:

```python
van_genuchten = bores.VanGenuchtenCapillaryPressureModel(
    irreducible_water_saturation=0.25,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.05,
    oil_water_alpha_water_wet=0.01,
    oil_water_n_water_wet=2.0,
    gas_oil_alpha=0.02,
    gas_oil_n=2.0,
)

result = van_genuchten(water_saturation=0.4, oil_saturation=0.55, gas_saturation=0.05)
```

### Calling Tabular Models Directly

The `TwoPhaseCapillaryPressureTable` can be queried at any saturation using `get_capillary_pressure()` or `__call__`:

```python
# Using the oil_water_capillary_pressure_table from earlier
capillary_pressure_scalar = oil_water_capillary_pressure_table.get_capillary_pressure(0.45)
print(f"Pcow at Sw=0.45: {capillary_pressure_scalar:.2f} psi")

# Grid array query
Sw_grid = np.random.uniform(0.2, 0.8, size=(20, 20, 5))
capillary_pressure_grid = oil_water_capillary_pressure_table.get_capillary_pressure(
    Sw_grid
)  # Shape: (20, 20, 5)
```

The `ThreePhaseCapillaryPressureTable` uses `get_capillary_pressures()` or `__call__` with all three saturations:

```python
# Using the three_phase_capillary_pressure from earlier
result = three_phase_capillary_pressure.get_capillary_pressures(
    water_saturation=0.4,
    oil_saturation=0.5,
    gas_saturation=0.1,
)
print(f"Pcow = {result['oil_water']:.2f} psi")
print(f"Pcgo = {result['gas_oil']:.2f} psi")

# Using __call__
result = three_phase_capillary_pressure(
    water_saturation=Sw,
    oil_saturation=So,
    gas_saturation=Sg,
)
```

This direct evaluation capability is essential for building capillary pressure curves for reports, comparing different models or parameter sets, and verifying that your model produces physically consistent pressures before running a simulation.

---

## Integrating with `RockFluidTables`

Capillary pressure is passed to the simulation through the `RockFluidTables` object alongside the relative permeability model. You can mix analytical and tabular approaches freely: for example, you could use Brooks-Corey relative permeability with tabular capillary pressure, or tabular relative permeability with a Van Genuchten capillary pressure model.

```python
import bores

# Analytical relative permeability + analytical capillary pressure
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
        oil_water_entry_pressure_water_wet=5.0,
        oil_water_pore_size_distribution_index_water_wet=2.5,
        gas_oil_entry_pressure=1.0,
        gas_oil_pore_size_distribution_index=2.0,
    ),
)
```

You can also combine tabular relative permeability with an analytical capillary pressure model, or vice versa:

```python
# Tabular capillary pressure with analytical relative permeability
rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=bores.BrooksCoreyRelPermModel(
        irreducible_water_saturation=0.25,
        residual_oil_saturation_water=0.30,
        residual_oil_saturation_gas=0.15,
        residual_gas_saturation=0.05,
        water_exponent=2.0,
        oil_exponent=2.0,
        gas_exponent=2.0,
    ),
    capillary_pressure_table=three_phase_capillary_pressure,  # ThreePhaseCapillaryPressureTable from lab data
)
```

Pass the `RockFluidTables` to the `Config` to use it in simulation:

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",
)
```

Both analytical models and tabular data are serializable. When you save a `Config`, all capillary pressure parameters or table data are preserved and can be reloaded exactly.

---

## Visualizing Capillary Pressure Curves

Visualizing your capillary pressure curves before running a simulation is essential for verifying that the model matches your physical expectations. The best approach is to create the actual model you plan to use and call it directly across a saturation sweep. This ensures the plotted curves are exactly what the simulator will use.

### Oil-Water Capillary Pressure

```python
import bores
import numpy as np

# Create the model
capillary_pressure = bores.BrooksCoreyCapillaryPressureModel(
    irreducible_water_saturation=0.25,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.05,
    oil_water_entry_pressure_water_wet=5.0,
    oil_water_pore_size_distribution_index_water_wet=2.5,
    gas_oil_entry_pressure=1.5,
    gas_oil_pore_size_distribution_index=2.0,
)

# Sweep water saturation across the mobile range (no free gas)
Sw_values = np.linspace(0.26, 0.75, 100)  # Slightly above swc to avoid singularity
pcoil_water_values = np.zeros_like(Sw_values)

for i, sw in enumerate(Sw_values):
    so = 1.0 - sw  # No free gas
    result = capillary_pressure.get_capillary_pressures(
        water_saturation=sw,
        oil_saturation=so,
        gas_saturation=0.0,
    )
    pcoil_water_values[i] = result["oil_water"]

fig = bores.make_series_plot(
    data=np.column_stack([Sw_values, pcoil_water_values]),
    title="Brooks-Corey Oil-Water Capillary Pressure",
    x_label="Water Saturation (fraction)",
    y_label="Capillary Pressure (psi)",
)
fig.show()
# Output: [PLACEHOLDER: Insert capillary_pressure_curve.png]
```

This plot should show capillary pressure decreasing from a high value at low water saturation (near residual) toward zero as water saturation approaches the maximum mobile value. If the curve shows unexpected shapes (negative values in a water-wet system, or unreasonably high pressures), check your entry pressure and pore size distribution index.

### Comparing Models

You can plot multiple capillary pressure models together to compare their behavior. This is useful when deciding between Brooks-Corey and Van Genuchten, or when evaluating how different wettability settings affect the curves:

```python
import bores
import numpy as np

# Brooks-Corey (water-wet)
brookes_corey_water_wet = bores.BrooksCoreyCapillaryPressureModel(
    irreducible_water_saturation=0.25,
    residual_oil_saturation_water=0.25,
    oil_water_entry_pressure_water_wet=5.0,
    oil_water_pore_size_distribution_index_water_wet=2.5,
)

# Van Genuchten (water-wet)
van_genuchten_water_wet = bores.VanGenuchtenCapillaryPressureModel(
    irreducible_water_saturation=0.25,
    residual_oil_saturation_water=0.25,
    oil_water_alpha_water_wet=0.2,
    oil_water_n_water_wet=2.5,
)

# Evaluate both across the same saturation range
Sw_range = np.linspace(0.26, 0.74, 80)
pcoil_water_brookes_corey = np.zeros_like(Sw_range)
pcoil_water_van_genuchten = np.zeros_like(Sw_range)

for i, sw in enumerate(Sw_range):
    so = 1.0 - sw
    result_brookes_corey = brookes_corey_water_wet.get_capillary_pressures(
        water_saturation=sw,
        oil_saturation=so,
        gas_saturation=0.0,
    )
    result_van_genuchten = van_genuchten_water_wet.get_capillary_pressures(
        water_saturation=sw,
        oil_saturation=so,
        gas_saturation=0.0,
    )
    pcoil_water_brookes_corey[i] = result_brookes_corey["oil_water"]
    pcoil_water_van_genuchten[i] = result_van_genuchten["oil_water"]

fig = bores.make_series_plot(
    data={
        "Brooks-Corey": np.column_stack([Sw_range, pcoil_water_brookes_corey]),
        "Van Genuchten": np.column_stack([Sw_range, pcoil_water_van_genuchten]),
    },
    title="Capillary Pressure Model Comparison",
    x_label="Water Saturation (fraction)",
    y_label="Capillary Pressure (psi)",
)
fig.show()
# Output: [PLACEHOLDER: Insert capillary_pressure_comparison.png]
```

You should see that both models produce similar general shapes (high Pc at low Sw, low Pc at high Sw), but the Van Genuchten curve is smoother near the residual saturation where Brooks-Corey approaches infinity. This smoother behavior is why Van Genuchten is sometimes preferred for numerical stability.

---

## Choosing a Capillary Pressure Model

The choice of capillary pressure model depends on the data available and the simulation objectives.

**Brooks-Corey** is the most common starting point. It is simple, well-understood, and has only two fitting parameters per phase pair (entry pressure and pore size distribution index). Use it when you have limited data or need a quick, physically reasonable approximation. Its main limitation is the infinite capillary pressure at residual saturation, which can cause numerical issues in some cases.

**Van Genuchten** is preferred when numerical stability near residual saturations is important. Its smoother curve shape avoids the singularity at $S_e = 0$ and produces better convergence in implicit solvers. It is widely used in environmental and groundwater modeling and is increasingly common in petroleum applications. Use it when your simulation shows convergence difficulties near residual saturation with Brooks-Corey.

**Leverett J-function** is the best choice when you have heterogeneous rock properties and want capillary pressure to scale automatically with local porosity and permeability. If your reservoir has significant spatial variation in rock quality (e.g., layered sandstone-shale sequences), the J-function approach produces physically consistent capillary pressure distributions without needing separate models for each rock type.

**Tabular data** should be used when you have high-quality laboratory measurements that do not fit any analytical model well, or when you need exact reproduction of measured curves for regulatory or history-matching purposes.

!!! example "Quick Reference: Typical Parameter Sets"

    === "Consolidated Sandstone"

        ```python
        capillary_pressure = bores.BrooksCoreyCapillaryPressureModel(
            oil_water_entry_pressure_water_wet=5.0,
            oil_water_pore_size_distribution_index_water_wet=2.5,
            gas_oil_entry_pressure=1.5,
            gas_oil_pore_size_distribution_index=2.0,
            wettability=bores.Wettability.WATER_WET,
        )
        ```

    === "Tight Carbonate"

        ```python
        capillary_pressure = bores.BrooksCoreyCapillaryPressureModel(
            oil_water_entry_pressure_water_wet=25.0,
            oil_water_pore_size_distribution_index_water_wet=1.2,
            gas_oil_entry_pressure=10.0,
            gas_oil_pore_size_distribution_index=1.5,
            wettability=bores.Wettability.OIL_WET,
            oil_water_entry_pressure_oil_wet=20.0,
            oil_water_pore_size_distribution_index_oil_wet=1.0,
        )
        ```

    === "High-Perm Unconsolidated"

        ```python
        capillary_pressure = bores.VanGenuchtenCapillaryPressureModel(
            oil_water_alpha_water_wet=0.5,  # Low capillary_pressure pressure
            oil_water_n_water_wet=3.0,  # Narrow pore size distribution
            gas_oil_alpha=0.8,
            gas_oil_n=2.5,
        )
        ```

    === "Heterogeneous with J-Function"

        ```python
        capillary_pressure = bores.LeverettJCapillaryPressureModel(
            permeability=150.0,
            porosity=0.22,
            oil_water_interfacial_tension=30.0,
            gas_oil_interfacial_tension=20.0,
            j_function_coefficient=0.4,
            j_function_exponent=0.5,
        )
        ```

!!! warning "Capillary Pressure and Numerical Stability"

    Very high capillary pressures (above 50 psi) can cause pressure oscillations and timestep cuts in explicit saturation schemes. If you encounter convergence problems, consider: (1) switching from Brooks-Corey to Van Genuchten for smoother behavior, (2) reducing entry pressure if the high values are not well-constrained by data, (3) using smaller timesteps in regions with high capillary gradients, or (4) using 64-bit precision via `bores.use_64bit_precision()` for better numerical accuracy.

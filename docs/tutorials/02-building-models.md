# Building Reservoir Models

Construct realistic reservoir models with heterogeneous properties, anisotropic permeability, structural dip, and 3D visualization.

---

## Overview

In the [previous tutorial](01-first-simulation.md), you built a homogeneous reservoir with uniform properties everywhere. Real reservoirs are never that simple. Rock properties vary from layer to layer due to depositional history, permeability is typically lower in the vertical direction than horizontal, and the reservoir surface may be tilted (dipping) rather than perfectly flat.

This tutorial teaches you how to construct models that capture these geological complexities using BORES's grid-building utilities. You will create layered porosity and permeability distributions, set up anisotropic permeability using `RockPermeability`, apply structural dip with `apply_structural_dip()`, and visualize the resulting model in 3D before running any simulation.

Building a realistic model is the foundation of any meaningful simulation study. The quality of your results depends directly on how well your model captures the key geological features that control fluid flow. Even when you are working with simplified models for screening studies, understanding how to introduce heterogeneity helps you assess whether your simplifications are reasonable.

We will work with a larger 20x20x5 grid to give the layering and dip enough room to be visually apparent.

---

## Physical Setup

Our model represents a layered sandstone reservoir with the following characteristics:

- **Grid**: 20 cells in X, 20 cells in Y, 5 layers in Z (2,000 total cells)
- **Cell size**: 50 ft x 50 ft horizontally
- **Layer thicknesses**: Varying from 15 ft to 25 ft per layer
- **Porosity**: Varies by layer (15% to 25%)
- **Permeability**: Varies by layer and direction (anisotropic)
- **Structural dip**: 3 degrees toward the east
- **Initial pressure**: 3,500 psi
- **Temperature**: 200 F

---

## Step 1 - Layered Property Grids

Real reservoirs are deposited in layers, and each layer can have different rock quality. BORES provides `build_layered_grid()` to create grids where property values vary along a chosen axis.

```python
import bores
import numpy as np

bores.use_32bit_precision()

grid_shape = (20, 20, 5)
cell_dimension = (50.0, 50.0)  # ft

# Layer thicknesses (5 layers, varying from 15 to 25 ft)
thickness = bores.build_layered_grid(
    grid_shape=grid_shape,
    layer_values=[25.0, 20.0, 15.0, 20.0, 25.0],
    orientation="z",
)

# Porosity varies by layer: good sand, tight, moderate, good, excellent
porosity = bores.build_layered_grid(
    grid_shape=grid_shape,
    layer_values=[0.22, 0.15, 0.18, 0.22, 0.25],
    orientation="z",
)
```

The `orientation="z"` parameter tells BORES that the layering runs along the vertical (z) axis. The `layer_values` list must have exactly as many entries as there are cells in that direction - in this case, 5 values for 5 layers. You can also layer along `"x"` or `"y"` for lateral property variations, though vertical layering is the most common geological pattern.

Notice how porosity varies from 15% (a tight, cemented zone in layer 2) to 25% (an excellent sand in layer 5). This kind of variation is typical of fluvial or deltaic depositional environments where sand quality changes with the energy of the depositing current.

The thickness grid controls how much pore volume each layer contains. Thicker layers with higher porosity dominate the flow behavior because they hold more fluid and transmit it more easily.

---

## Step 2 - Anisotropic Permeability

In most sedimentary rocks, horizontal permeability is significantly higher than vertical permeability. This anisotropy arises from the way sediment grains settle and compact. Shale laminae and clay drapes between sand layers further reduce vertical flow.

```python
# Horizontal permeability varies by layer (mD)
kx_values = [150.0, 50.0, 80.0, 200.0, 300.0]
ky_values = [150.0, 50.0, 80.0, 200.0, 300.0]

# Vertical permeability is typically 10-20% of horizontal
kz_values = [15.0, 5.0, 8.0, 20.0, 30.0]

kx_grid = bores.build_layered_grid(
    grid_shape=grid_shape, layer_values=kx_values, orientation="z"
)
ky_grid = bores.build_layered_grid(
    grid_shape=grid_shape, layer_values=ky_values, orientation="z"
)
kz_grid = bores.build_layered_grid(
    grid_shape=grid_shape, layer_values=kz_values, orientation="z"
)

permeability = bores.RockPermeability(x=kx_grid, y=ky_grid, z=kz_grid)
```

The `RockPermeability` object holds separate permeability grids for each direction. Here, horizontal permeability (kx and ky) ranges from 50 mD in the tight zone to 300 mD in the best sand, while vertical permeability (kz) is set to 10% of horizontal in each layer. This 10:1 anisotropy ratio is a common starting assumption for sandstones, though real ratios can range from 2:1 in clean sands to 1000:1 in thinly laminated sequences.

The anisotropy has important implications for fluid flow. Water or gas injected at one location will spread more easily in the horizontal plane than it will migrate vertically between layers. This affects sweep efficiency, breakthrough times, and ultimate recovery.

!!! info "Permeability Anisotropy Ratio"

    The ratio $k_v/k_h$ (vertical-to-horizontal permeability) is one of the most important parameters controlling vertical conformance in reservoir simulation. Common ranges:

    - **Clean, massive sandstone**: $k_v/k_h$ = 0.3 - 1.0
    - **Laminated sandstone**: $k_v/k_h$ = 0.01 - 0.1
    - **Layered with shale breaks**: $k_v/k_h$ = 0.001 - 0.01

---

## Step 3 - Remaining Property Grids

```python
# Uniform properties for simplicity
pressure = bores.build_uniform_grid(grid_shape, value=3500.0)       # psi
temperature = bores.build_uniform_grid(grid_shape, value=200.0)     # deg F
oil_viscosity = bores.build_uniform_grid(grid_shape, value=1.2)     # cP
bubble_point = bores.build_uniform_grid(grid_shape, value=2800.0)   # psi
oil_sg = bores.build_uniform_grid(grid_shape, value=0.82)           # ~40 deg API

# Residual saturations
Sorw = bores.build_uniform_grid(grid_shape, value=0.20)
Sorg = bores.build_uniform_grid(grid_shape, value=0.15)
Sgr  = bores.build_uniform_grid(grid_shape, value=0.05)
Swir = bores.build_uniform_grid(grid_shape, value=0.25)
Swc  = bores.build_uniform_grid(grid_shape, value=0.25)

# Build initial saturations from fluid contacts
depth = bores.build_depth_grid(thickness, datum=5000.0)

Sw, So, Sg = bores.build_saturation_grids(
    depth_grid=depth,
    gas_oil_contact=4900.0,       # GOC above reservoir (no initial gas cap)
    oil_water_contact=5120.0,     # OWC below reservoir
    connate_water_saturation_grid=Swc,
    residual_oil_saturation_water_grid=Sorw,
    residual_oil_saturation_gas_grid=Sorg,
    residual_gas_saturation_grid=Sgr,
    porosity_grid=porosity,
)
```

For this tutorial, we keep some properties uniform to focus on the model-building aspects. The `build_saturation_grids()` function computes physically consistent saturations from fluid contact depths, ensuring that $S_o + S_w + S_g = 1.0$ in every cell. Here we place the GOC above the reservoir (no gas cap) and the OWC below, so most cells are in the oil zone with connate water. In a real study, you might also use spatially varying temperature (geothermal gradient) or place the contacts within the reservoir to create initial gas caps or water legs.

---

## Step 4 - Build the Model with Structural Dip

Structural dip means the reservoir surface is tilted relative to horizontal. This tilt creates a gravity component that drives fluids: gas migrates updip (toward higher elevation) while water drains downdip (toward lower elevation). Structural dip is one of the primary mechanisms for gas-oil and oil-water contact movement in real reservoirs.

```python
model = bores.reservoir_model(
    grid_shape=grid_shape,
    cell_dimension=cell_dimension,
    thickness_grid=thickness,
    pressure_grid=pressure,
    rock_compressibility=3e-6,
    absolute_permeability=permeability,
    porosity_grid=porosity,
    temperature_grid=temperature,
    water_saturation_grid=Sw,
    gas_saturation_grid=Sg,
    oil_saturation_grid=So,
    oil_viscosity_grid=oil_viscosity,
    oil_specific_gravity_grid=oil_sg,
    oil_bubble_point_pressure_grid=bubble_point,
    residual_oil_saturation_water_grid=Sorw,
    residual_oil_saturation_gas_grid=Sorg,
    residual_gas_saturation_grid=Sgr,
    irreducible_water_saturation_grid=Swir,
    connate_water_saturation_grid=Swc,
    dip_angle=3.0,        # 3 degrees from horizontal
    dip_azimuth=90.0,     # dipping toward East
    datum_depth=5000,
)
```

The `dip_angle` parameter specifies the tilt in degrees from horizontal (0 = flat, 90 = vertical). A 3-degree dip is moderate and common in many real reservoirs. The `dip_azimuth` parameter specifies the direction of dip using compass convention: 0 = North, 90 = East, 180 = South, 270 = West. So `dip_azimuth=90.0` means the reservoir dips toward the east, with the updip (highest) side on the west.

When you later run a simulation with this model, the simulator will account for gravity effects along the dip. Gas will tend to migrate westward (updip) while water will tend to flow eastward (downdip). This has practical implications for well placement: producers are often placed updip to take advantage of gravity drainage, while injectors are placed downdip so that injected fluid sweeps upward through the oil column.

!!! warning "Dip and Gravity"

    Structural dip effects are computed using the depth grid and the gravity term in the flow equations. If you set `disable_structural_dip=True` in the `Config`, the simulator ignores gravity effects **due to dipping** regardless of the dip angle specified in the model. This can be useful for debugging or for studying horizontal flow in isolation.

---

## Step 5 - Visualize the Model Before Running

One of the most valuable practices in reservoir simulation is inspecting your model visually before running any simulation. This catches errors in property assignments, grid construction, and well placement before they lead to confusing results.

### Porosity Distribution

```python
viz = bores.plotly3d.DataVisualizer()

fig = viz.make_plot(
    source=model,
    property="porosity",
    plot_type="volume",
    title="Porosity Distribution (Layered)",
)
fig.show()
```

You should see five distinct horizontal bands of color corresponding to the five porosity values. The tight zone (layer 2, 15% porosity) should appear as a cooler color band sandwiched between better quality sands.

### Permeability Distribution

```python
fig = viz.make_plot(
    source=model,
    property="permeability-x",
    plot_type="volume",
    title="Horizontal Permeability (X-direction)",
)
fig.show()
```

The permeability visualization reveals the same layered pattern but with a wider range of values - from 50 mD in the tight zone to 300 mD in the best sand. This layered contrast will strongly influence how fluids flow through the reservoir during simulation.

### Depth Grid (showing dip)

```python
depth_grid = model.get_depth_grid(apply_dip=True)
fig = viz.make_plot(
    source=depth_grid,
    plot_type="scatter_3d",
    title="Depth Grid with 3-Degree Eastward Dip",
)
fig.show()
```

The depth grid visualization shows the structural dip clearly. Cells on the western side of the model should be at shallower depth (higher elevation) than cells on the eastern side, reflecting the 3-degree eastward tilt.

---

## Step 6 - Adding Random Heterogeneity

For more realistic models, you can add random perturbations to the layered base values. This simulates the natural variability within each geological layer.

```python
rng = np.random.default_rng(seed=42)

# Start from layered porosity and add +/- 2% random variation
porosity_heterogeneous = porosity.copy()
noise = rng.normal(loc=0.0, scale=0.02, size=grid_shape).astype(np.float32)
porosity_heterogeneous = np.clip(porosity_heterogeneous + noise, 0.05, 0.35)

# Permeability: log-normal perturbation (more physically realistic)
kx_heterogeneous = kx_grid.copy()
log_noise = rng.normal(loc=0.0, scale=0.3, size=grid_shape).astype(np.float32)
kx_heterogeneous = kx_heterogeneous * np.exp(log_noise)
kx_heterogeneous = np.clip(kx_heterogeneous, 1.0, 1000.0)
```

Permeability in real rocks follows a log-normal distribution, meaning that the logarithm of permeability is normally distributed. Multiplying by `exp(noise)` preserves this statistical property. The `np.clip()` calls ensure values stay within physically reasonable bounds.

!!! tip "Reproducible Random Models"

    Always use a seeded random number generator (`np.random.default_rng(seed=42)`) when building stochastic models. This ensures your results are reproducible across runs, which is essential for debugging, peer review, and comparison studies.

---

## Step 7 - Tips for Real-World Model Building

When transitioning from tutorial models to real-world applications, keep these guidelines in mind:

**Start simple, add complexity incrementally.** Begin with a uniform model to verify your simulation runs correctly. Then add layering, then heterogeneity, then structural features. If something breaks, you know which addition caused the problem.

**Validate property ranges.** Before building the model, check that your property values are physically reasonable:

| Property | Typical Range | Units |
|----------|--------------|-------|
| Porosity | 0.05 - 0.35 | fraction |
| Permeability | 0.1 - 10,000 | mD |
| Rock compressibility | 1e-7 - 1e-5 | psi$^{-1}$ |
| Oil viscosity | 0.2 - 100 | cP |
| Oil specific gravity | 0.7 - 1.0 | dimensionless |
| Temperature | 100 - 400 | F |

**Mind the saturation constraint.** In every cell, $S_o + S_w + S_g = 1.0$. If you build saturation grids from different sources, they may not sum to exactly 1.0. The `reservoir_model()` factory will normalize them and warn you, but it is better to ensure consistency upfront.

**Use layered grids for geological layering.** The `build_layered_grid()` function with `orientation="z"` is the natural way to represent vertical heterogeneity. For lateral variations (facies changes), use `orientation="x"` or `orientation="y"`, or build grids from external geological model data.

---

## Key Takeaways

1. **`build_layered_grid()`** creates grids with property values that vary along a specified axis, which is the natural way to represent geological layering.

2. **`RockPermeability(x=..., y=..., z=...)`** defines anisotropic permeability, which is essential for realistic vertical flow modeling. Vertical permeability is typically 10-100x lower than horizontal.

3. **Structural dip** (`dip_angle` and `dip_azimuth` in `reservoir_model()`) introduces gravity-driven flow that affects gas migration, water drainage, and well placement strategy.

4. **Always visualize your model** using `bores.plotly3d.DataVisualizer` before running simulations. Visual inspection catches errors that are invisible in the numbers.

5. **Log-normal perturbations** (multiply by `exp(noise)`) are more physically realistic for permeability heterogeneity than additive Gaussian noise.

---

## Next Steps

With a realistic heterogeneous model in hand, you are ready to study secondary recovery. In the [next tutorial](03-waterflood.md), you will add injection wells to maintain reservoir pressure and sweep oil toward the producers - the fundamental waterflood workflow.

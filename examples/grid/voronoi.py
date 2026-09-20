"""
Complex multi-layer 2-D extruded Voronoi (PEBI) grid example for bores.

**Grid description**
- 93 seed points arranged as:
    * 1  centre cell (well location)
    * 12 inner ring  at r = 150 m  (wellbore refinement)
    * 24 middle ring at r = 450 m  (near-well flow region)
    * 36 outer ring  at r = 900 m  (reservoir flank)
    * 20 random background cells
- 5 vertical layers with variable thickness:
    Layer 1:  8 m  — cap rock / seal
    Layer 2: 15 m  — upper reservoir
    Layer 3: 25 m  — main pay zone
    Layer 4: 20 m  — lower reservoir
    Layer 5: 40 m  — aquifer
"""

import numpy as np
import pyvista as pv

from bores.grids.factories.voronoi import make_voronoi_grid
from bores.grids.utils import make_pyvista_grid

# Seed geometry
rng = np.random.default_rng(42)


# Radial rings
def _ring(n: int, r: float) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


seeds_centre = np.array([[0.0, 0.0]])  # well cell
seeds_inner = _ring(12, 150.0)  # wellbore refinement
seeds_mid = _ring(24, 450.0)  # near-well region
seeds_outer = _ring(36, 900.0)  # reservoir flank
seeds_rand = rng.uniform(-950, 950, (20, 2))  # background scatter
seeds_2d = np.vstack([seeds_centre, seeds_inner, seeds_mid, seeds_outer, seeds_rand])

# Layer definition
z_top = 2100.0  # top of cap rock (m depth, positive down)
layer_thicknesses = np.array([8.0, 15.0, 25.0, 20.0, 40.0])
layer_names = ["Cap rock", "Upper reservoir", "Main pay", "Lower reservoir", "Aquifer"]

bounding_box = (-1000.0, 1000.0, -1000.0, 1000.0)

# Build grid

print("Building Voronoi grid …")
grid = make_voronoi_grid(
    seeds_2d,
    bounding_box=bounding_box,
    z_top=z_top,
    layer_thicknesses=layer_thicknesses,
)

print(f"  cells    : {grid.n_cells}")
print(f"  faces    : {grid.n_faces}")
print(f"  vertices : {grid.n_vertices}")
print(f"  bbox     : {grid.bounding_box}")

n_layers = len(layer_thicknesses)
n_cols = grid.n_cells // n_layers
print(f"  columns  : {n_cols}  x  {n_layers} layers  = {grid.n_cells} cells")

# Synthetic property fields

assert grid.cell_centroids is not None
cx = grid.cell_centroids[:, 0]
cy = grid.cell_centroids[:, 1]
cz = grid.cell_centroids[:, 2]
r = np.sqrt(cx**2 + cy**2)  # radial distance from well

# Porosity: higher in main pay (layer 3), radially symmetric with scatter
layer_index = np.arange(grid.n_cells) % n_layers  # 0 = cap, 4 = aquifer
layer_porosity = np.array([0.04, 0.18, 0.28, 0.22, 0.10])
porosity = layer_porosity[layer_index] + 0.03 * rng.standard_normal(grid.n_cells)
porosity = np.clip(porosity, 0.02, 0.40)

# Permeability: log-normal, correlated with porosity, higher near well bore
radial_perm_factor = np.exp(-r / 600.0)  # near-well enhancement
perm_base = 80 * (porosity / 0.22) ** 3  # Kozeny-Carman proxy
perm = perm_base * (1.0 + 2.0 * radial_perm_factor)
perm *= np.exp(0.5 * rng.standard_normal(grid.n_cells))  # log-normal scatter
perm = np.clip(perm, 0.1, 2000.0)

# Water saturation: increases with depth and radius (oil in structural high)
sw = 0.15 + 0.4 * (cz - z_top) / layer_thicknesses.sum() + 0.1 * r / 1000.0
sw += 0.05 * rng.standard_normal(grid.n_cells)
sw = np.clip(sw, 0.10, 0.95)

# Pressure: hydrostatic + small depletion cone around well
hydrostatic = 200.0 + 0.1 * cz  # bar  (approx 0.1 bar/m)
drawdown = -5.0 * np.exp(-r / 200.0)  # bar  (depletion near well)
pressure = hydrostatic + drawdown + 0.5 * rng.standard_normal(grid.n_cells)

# PyVista visualisation

print("\nConverting to PyVista …")
pv_grid = make_pyvista_grid(
    grid,
    cell_data={
        "porosity": porosity,
        "permeability_x": perm,
        "water_sat": sw,
        "pressure_bar": pressure,
    },
)

# Plot 1: porosity coloured by layer, z-exaggerated
pl = pv.Plotter(shape=(1, 2), window_size=(1600, 700))  # type: ignore

pl.subplot(0, 0)
pl.add_text("Voronoi PEBI Grid — Porosity", font_size=10)
pl.add_mesh(
    pv_grid,
    scalars="porosity",
    cmap="viridis",
    show_edges=True,
    edge_color="lightgrey",
    clim=[0.0, 0.40],
)
pl.set_scale(zscale=-5)  # flip z and exaggerate depth # type: ignore
pl.view_isometric()  # type: ignore

pl.subplot(0, 1)
pl.add_text("Voronoi PEBI Grid — Pressure (bar)", font_size=10)
pl.add_mesh(
    pv_grid,
    scalars="pressure_bar",
    cmap="RdYlGn_r",
    show_edges=True,
)
pl.set_scale(zscale=-5)  # type: ignore
pl.view_isometric()  # type: ignore

print("Rendering …  (close window to exit)")
pl.show()

# Summary stats
print("\n── Property summary")
print(f"  Porosity   : {porosity.mean():.3f}  ±  {porosity.std():.3f}")
print(
    f"  Perm (mD)  : {perm.mean():.1f}  ±  {perm.std():.1f}  "
    f"  [p10={np.percentile(perm, 10):.1f}, p90={np.percentile(perm, 90):.1f}]"
)
print(f"  Water sat  : {sw.mean():.3f}  ±  {sw.std():.3f}")
print(f"  Pressure   : {pressure.mean():.1f}  ±  {pressure.std():.1f} bar")
print(f"\nPore volume (m³): {grid.compute_pore_volume(porosity, np.ones(grid.n_cells)).sum():.3e}")  # type: ignore

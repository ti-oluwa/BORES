"""
Loads and visualises a Cartesian GRDECL test grid.

**Grid description**:

- 20 x 15 x 8  =  2 400 cells  (2 112 active, 88 %)
- Variable cell spacing:
    DX: 80-200 m  (tighter near fault flanks)
    DY: 100-160 m (symmetric, wider in centre)
    DZ: 5-30 m    (thin cap + aquifer, thick reservoir)
- Dipping dome structure with regional tilt (TOPS array)
- Geological features:
    * N-S fault zone (i=9-10 inactive, transmissibility via NNCs)
    * Cap rock seal  (MULTZ = 0.001 on layer 1)
    * Near-fault MULTX reduction (0.05 on columns 7-8 and 11-12)
    * Channel-flow MULTY enhancement (1.5 along j=6-8, reservoir layers)
    * 55 cross-fault NNCs connecting i=8 to i=11
    * PINCH tolerance 1e-3 m
"""

import pathlib

import numpy as np
import pyvista as pv

from bores.grids.io.grdecl import load_grdecl
from bores.grids.utils import make_pyvista_grid
from bores.units import UNIT_SYSTEM

# Load
grdecl_path = pathlib.Path("data/cartesian.grdecl")

print(f"Loading {grdecl_path.name} …")
grid = load_grdecl(grdecl_path, encoding="ascii", unit_system=UNIT_SYSTEM.value)

print(f"  cells        : {grid.n_cells}")
print(f"  faces        : {grid.n_faces}")
print(f"  vertices     : {grid.n_vertices}")
print(f"  n_faults     : {grid.n_faults}")
print(f"  n_nnc        : {grid.n_nnc}")
print(f"  bbox         : {tuple(f'{v:.1f}' for v in grid.bounding_box)}")
print(f"  unit system  : {grid.unit_system}")
print(f"  has_mult     : {grid.has_transmissibility_multipliers}")

if grid.fault_face_indices:
    for name, idxs in grid.fault_face_indices.items():
        mult = (grid.fault_transmissibility_multipliers or {}).get(name, "-")
        print(f"  fault '{name}': {len(idxs)} faces, MULTFLT={mult}")

# Synthetic property fields
rng = np.random.default_rng(13)
n = grid.n_cells

assert grid.cell_centroids is not None
cx = grid.cell_centroids[:, 0]
cy = grid.cell_centroids[:, 1]
cz = grid.cell_centroids[:, 2]  # depth, positive down

# Reference depth and dome centre (approximate from bbox)
bb = grid.bounding_box
x_mid = (bb[0] + bb[1]) / 2
y_mid = (bb[2] + bb[3]) / 2
z_ref = bb[4]

# Porosity: dome-shaped, higher at crest, decreasing with depth and toward fault
r_dome = np.sqrt(((cx - x_mid) / 800) ** 2 + ((cy - y_mid) / 600) ** 2)
base_porosity = 0.28 * np.exp(-r_dome) - 0.04 * (cz - z_ref) / 130.0
porosity = np.clip(base_porosity + 0.02 * rng.standard_normal(n), 0.02, 0.38)

# Permeability: log-normal, strongly correlated with porosity
log_k = np.log(50.0) + 6.0 * (porosity - 0.15) + 0.6 * rng.standard_normal(n)
permeability = np.exp(log_k)
permeability = np.clip(permeability, 0.01, 5000.0)

# Water saturation: structurally controlled (rises away from dome crest)
base_water_saturation = 0.12 + 0.5 * r_dome + 0.25 * (cz - z_ref) / 130.0
water_saturation = np.clip(base_water_saturation + 0.04 * rng.standard_normal(n), 0.10, 1.0)

# Pressure: hydrostatic gradient + slight over-pressure at dome crest
pressure = 195.0 + 0.098 * cz - 4.0 * np.exp(-r_dome * 2) + 0.3 * rng.standard_normal(n)

# Highlight fault zone in a separate array
fault_flag = np.zeros(n)
if grid.fault_face_indices:
    for face_idxs in grid.fault_face_indices.values():
        for face_idx in face_idxs:
            for side in range(2):
                ci = int(grid.face_cell_indices[face_idx, side])
                if ci >= 0:
                    fault_flag[ci] = 1.0

# Convert to PyVista grid
print("\nConverting to PyVista …")
pore_volume_grid = make_pyvista_grid(
    grid,
    cell_data={
        "porosity": porosity,
        "permeability_x": permeability,
        "water_saturation": water_saturation,
        "pressure": pressure,
        "fault_flag": fault_flag,
    },
)

# Plot: 2×2 panel
pl = pv.Plotter(shape=(2, 2), window_size=(1600, 1000))  # type: ignore

panel_cfg = [
    (0, 0, "porosity", "viridis", "Porosity", (0.0, 0.38)),
    (0, 1, "permeability_x", "plasma", "Permeability (mD)", None),
    (1, 0, "water_saturation", "Blues", "Water Saturation", (0.1, 1.0)),
    (1, 1, "pressure", "RdYlGn_r", "Pressure (bar)", None),
]
for row, col, scalar, cmap, title, clim in panel_cfg:
    pl.subplot(row, col)
    pl.add_text(f"Cartesian Grid — {title}", font_size=9)
    kwargs = dict(
        scalars=scalar,
        cmap=cmap,
        show_edges=True,
        edge_color="lightgrey",
    )
    if clim:
        kwargs["clim"] = clim  # type: ignore
    pl.add_mesh(pore_volume_grid, **kwargs)  # type: ignore
    # flip z (depth down → screen up) and exaggerate
    pl.set_scale(zscale=-3)  # type: ignore
    pl.view_isometric()  # type: ignore

print("Rendering …  (close window to exit)")
pl.show()

# Summary
print("\n── Property summary")
print(f"  Porosity   : {porosity.mean():.3f}  ±  {porosity.std():.3f}")
print(
    f"  Perm (mD)  : {permeability.mean():.1f}  "
    f"  [p10={np.percentile(permeability, 10):.1f}, p90={np.percentile(permeability, 90):.1f}]"
)
print(f"  Water sat  : {water_saturation.mean():.3f}  ±  {water_saturation.std():.3f}")
print(f"  Pressure   : {pressure.mean():.1f}  ±  {pressure.std():.1f} bar")

net_to_gross = np.where(water_saturation < 0.60, 1.0, 0.3)
total_pore_volume = grid.compute_pore_volume(porosity, net_to_gross).sum()  # type: ignore[arg-type]
print(f"\nNet pore volume (m³): {total_pore_volume:.3e}")
print(
    f"Oil in place (m³ at res. cond.): {(total_pore_volume * (1 - water_saturation)).mean():.3e}"
)

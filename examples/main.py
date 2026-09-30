"""
Fuller walkthrough: load a case, inspect the model and schedule, and plot
the grid with PyVista.

See case.py for the minimal version of this, matching the README.
"""

import pyvista as pv

from bores.deck import DeckFile
from bores.grids.utils import make_pyvista_grid
from bores.simulation.case import SimulationCase
from bores.types import UnitSystem
from bores.wells.hydraulics.homogeneous import homogeneous_wellbore

df = DeckFile(
    "/home/tioluwa/Projects/nagcsu/runs_new/auto_final/NigerDelta UGH1 Composite Field.DATA",
    encoding="utf-8",
    unit_system=UnitSystem.FIELD,
)


# Load the simultion case from the deck.
wellbore = homogeneous_wellbore(tubing_inner_diameter=2.5, unit_system=UnitSystem.FIELD)
case = SimulationCase.from_deck(df, default_wellbore=wellbore, temperature=200.0)

# Fluid model, from the compiled case
pvt = case.model.fluid.pvt
oil_table = pvt.region(1).tables.oil
assert oil_table is not None, "`oil_table` should not be None"
print(oil_table.viscosity([4700, 200, 3456, 10000, 4000], 200, solution_gor=800))

# Wells and schedule
wells = case.model.wells
print(f"wells: {list(wells.names) if wells else []}")

for item in case.schedule:
    print(item, "\n")

print(f"{len(case.schedule)} scheduled item(s)")

# Plot the grid
grid = case.model.reservoir.grid
print(f"cells   : {grid.n_cells}")
print(f"faces   : {grid.n_faces}")
print(f"bbox    : {grid.bounding_box}")

pv_grid = make_pyvista_grid(grid, cell_data={"pressure": case.initial_state.pressure})
pl = pv.Plotter()
pl.add_mesh(pv_grid, scalars="pressure", show_edges=True)
pl.set_scale(zscale=15, xscale=2, yscale=2)  # type:ignore
pl.show()

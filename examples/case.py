import pyvista as pv

from bores.deck import DeckFile
from bores.grids.utils import make_pyvista_grid
from bores.reservoir import Temperature
from bores.simulation.case import SimulationCase
from bores.types import UnitSystem
from bores.wells.hydraulics.homogeneous import homogeneous_wellbore

df = DeckFile(
    "/home/tioluwa/Projects/nagscu/Phase One/Data/NigerDelta UGH1 Composite Field.DATA",
    encoding="utf-8",
    unit_system=UnitSystem.METRIC,
)

temperature = Temperature(200, unit_system=UnitSystem.METRIC)
wellbore = homogeneous_wellbore(tubing_inner_diameter=0.5, unit_system=UnitSystem.METRIC)
# Load simulation case
case = SimulationCase.from_deck(df, default_wellbore=wellbore, temperature=temperature)

# Test the fluid
pvt = case.model.fluid.pvt
table = pvt.region(1).tables.oil
assert table is not None, "`table` should not be None"
print(table.viscosity([4700, 200, 3456, 10000, 4000], 200, solution_gor=800))

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

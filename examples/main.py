import pyvista as pv

from bores.deck import DeckFile
from bores.grids.utils import make_pyvista_grid
from bores.simulation.case import SimulationCase
from bores.wells.hydraulics.homogeneous import homogeneous_wellbore

df = DeckFile(
    "/home/tioluwa/Projects/opm-data/norne/NORNE_ATW2013.DATA",
    encoding="latin-1",
    # unit_system=UnitSystem.METRIC,
)


# Load the simultion case from the deck.
wellbore = homogeneous_wellbore(tubing_inner_diameter=2.5, unit_system=df.unit_system)
case = SimulationCase.from_deck(df, title="Norne", tubing_performance=wellbore, temperature=200.0)

# Fluid model, from the compiled case
pvt = case.model.fluid.pvt
oil_table = pvt.region(1).tables.oil
assert oil_table is not None, "`oil_table` should not be None"
print(
    oil_table.density(
        pressure=[4700, 200, 3456, 10000, 4000],
        temperature=[[200] * 5, [400] * 5],
        solution_gor=[[800] * 5, [1270] * 5],
    )
)

# Wells and schedule
wells = case.model.wells
print(f"Wells: {wells.names if wells else []}")

# for item in case.schedule:
#     print(item, "\n")

# print(f"{len(case.schedule)} scheduled item(s)")

# Plot the grid
grid = case.model.reservoir.grid
print(f"cells: {grid.n_cells}")
print(f"faces: {grid.n_faces}")
print(f"bbox: {grid.bounding_box}")

pv_grid = make_pyvista_grid(
    grid,
    cell_data={"pressure": case.model.reservoir.regions.equilibrium_region},
    use_hexahedra=True,
    hollow=True,
)
pl = pv.Plotter()
pl.add_mesh(pv_grid, scalars="pressure", show_edges=True)
pl.set_scale(zscale=5, xscale=2, yscale=2)  # type:ignore
pl.show()

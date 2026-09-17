import pyvista as pv

from bores.deck.file import DeckFile
from bores.grids.io.grdecl import load_grdecl
from bores.grids.utils import make_pyvista_grid
from bores.types import UnitSystem

df = DeckFile("./data/Johansen.grdecl", unit_system=UnitSystem.FIELD)
grid = load_grdecl(source=df, encoding="utf-8")
print(f"cells   : {grid.n_cells}")
print(f"faces   : {grid.n_faces}")
print(f"volumes : {grid.cell_volumes}")
print(f"bbox    : {grid.bounding_box}")

pv_grid = make_pyvista_grid(grid)
pl = pv.Plotter()
pl.add_mesh(pv_grid, scalars="cell_depth", show_edges=True)
pl.set_scale(zscale=5, xscale=2, yscale=2)  # type:ignore
pl.show()

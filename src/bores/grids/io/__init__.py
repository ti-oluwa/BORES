"""
Grid import/export functions. Each sub-module handles one file family.

GRDECL / Eclipse text:

    from bores.grids.io.grdecl import load_grdecl, dump_grdecl

Gmsh (.msh):

    from bores.grids.io.gmsh import load_msh

VTK XML polyhedron cells (.vtu), exact for any cell shape, no extra dependencies:

    from bores.grids.io.vtu import dump_polyhedral, load_polyhedral

meshio-readable formats (optional dependency, not re-exported here):

    from bores.grids.io.meshio import load_mesh, dump_mesh
"""

from bores.grids.io.gmsh import load_msh
from bores.grids.io.grdecl import dump_grdecl, load_grdecl
from bores.grids.io.vtu import dump_polyhedral, load_polyhedral

__all__ = ["dump_grdecl", "dump_polyhedral", "load_grdecl", "load_msh", "load_polyhedral"]

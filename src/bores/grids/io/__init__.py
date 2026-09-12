"""
Grid import/export functions. Each sub-module handles one file family.

GRDECL / Eclipse text:

    from bores.grids.io.grdecl import load_grdecl, dump_grdecl

Gmsh (.msh):

    from bores.grids.io.gmsh import load_msh

meshio-readable formats (optional dependency, not re-exported here):

    from bores.grids.io.meshio import load_mesh, dump_mesh
"""

from bores.grids.io.gmsh import load_msh
from bores.grids.io.grdecl import dump_grdecl, load_grdecl

__all__ = ["dump_grdecl", "load_grdecl", "load_msh"]

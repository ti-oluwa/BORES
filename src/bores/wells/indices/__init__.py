"""Perforation and well connection-factor resolution."""

from bores.wells.indices.perforations import (
    PerforationIndex,
    resolve_md_perforations_indices,
    resolve_perforation_orientation,
    resolve_perforations_indices,
)
from bores.wells.indices.wells import (
    WellIndex,
    build_wells_indices,
    compute_2D_effective_drainage_radius,
    compute_3D_effective_drainage_radius,
    compute_effective_permeability_for_well,
    compute_equivalent_radius_well_index,
    compute_peaceman_well_index,
    is_locally_cartesian,
    resolve_connection_factor,
    resolve_well_index_direction,
)

__all__ = [
    "PerforationIndex",
    "WellIndex",
    "build_wells_indices",
    "compute_2D_effective_drainage_radius",
    "compute_3D_effective_drainage_radius",
    "compute_effective_permeability_for_well",
    "compute_equivalent_radius_well_index",
    "compute_peaceman_well_index",
    "is_locally_cartesian",
    "resolve_connection_factor",
    "resolve_md_perforations_indices",
    "resolve_perforation_orientation",
    "resolve_perforations_indices",
    "resolve_well_index_direction",
]

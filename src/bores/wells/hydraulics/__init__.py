"""
Wellbore hydraulics: shared primitives plus per-correlation models.

`compute_perforation_pressures`, `compute_segment_drop`, and
`compute_tubing_head_pressure` are defined once per correlation module
(`beggs_and_brill`, `hagedorn_brown`, `homogeneous`, `gray`,
`woldesemayat_ghajar`) and are not re-exported here, since the five
definitions collide on name. Import them from their own submodule.
"""

from bores.wells.hydraulics.base import (
    PressureDrop,
    SurfaceFluidProperties,
    WellBoreModel,
    WellBoreModelOptions,
    compute_friction_factor,
    compute_hydrostatic_pressure,
    compute_mixture_density,
    compute_mixture_velocity,
    compute_mixture_viscosity,
    compute_segment_pressure_drop,
    compute_static_hydrostatic_drop,
    compute_static_mixture_density,
    compute_surface_mixture_density,
    compute_surface_mixture_viscosity,
    get_unit_system_constant,
)
from bores.wells.hydraulics.beggs_and_brill import (
    BeggsAndBrillWellbore,
    beggs_and_brill_wellbore,
    compute_beggs_brill_holdup,
    compute_horizontal_holdup,
    compute_two_phase_friction_factor,
    get_flow_pattern,
)
from bores.wells.hydraulics.gray import (
    GrayWellbore,
    compute_gray_effective_roughness,
    compute_gray_holdup,
    gray_wellbore,
)
from bores.wells.hydraulics.hagedorn_brown import (
    HagedornBrownWellbore,
    compute_griffith_holdup,
    compute_hagedorn_brown_holdup,
    hagedorn_brown_wellbore,
    is_griffith_bubble_flow,
)
from bores.wells.hydraulics.homogeneous import HomogeneousWellbore, homogeneous_wellbore
from bores.wells.hydraulics.vfp import VFPData, VFPTable, VFPTables
from bores.wells.hydraulics.woldesemayat_ghajar import (
    WoldesemayatGhajarWellbore,
    compute_woldesemayat_ghajar_void_fraction,
    woldesemayat_ghajar_wellbore,
)

__all__ = [
    "BeggsAndBrillWellbore",
    "GrayWellbore",
    "HagedornBrownWellbore",
    "HomogeneousWellbore",
    "PressureDrop",
    "SurfaceFluidProperties",
    "VFPData",
    "VFPTable",
    "VFPTables",
    "WellBoreModel",
    "WellBoreModelOptions",
    "WoldesemayatGhajarWellbore",
    "beggs_and_brill_wellbore",
    "compute_beggs_brill_holdup",
    "compute_friction_factor",
    "compute_gray_effective_roughness",
    "compute_gray_holdup",
    "compute_griffith_holdup",
    "compute_hagedorn_brown_holdup",
    "compute_horizontal_holdup",
    "compute_hydrostatic_pressure",
    "compute_mixture_density",
    "compute_mixture_velocity",
    "compute_mixture_viscosity",
    "compute_segment_pressure_drop",
    "compute_static_hydrostatic_drop",
    "compute_static_mixture_density",
    "compute_surface_mixture_density",
    "compute_surface_mixture_viscosity",
    "compute_two_phase_friction_factor",
    "compute_woldesemayat_ghajar_void_fraction",
    "get_flow_pattern",
    "get_unit_system_constant",
    "gray_wellbore",
    "hagedorn_brown_wellbore",
    "homogeneous_wellbore",
    "is_griffith_bubble_flow",
    "woldesemayat_ghajar_wellbore",
]

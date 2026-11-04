"""Tests that parameters whose default lives in `bores.constants` resolve it at call time."""

import importlib
import inspect

import pytest

from bores.constants import Constants, ConstantsContext, c

MOVED_DEFAULTS = [
    (
        "bores.wells.factories",
        "make_perforation",
        "wellbore_radius",
        "DEFAULT_WELLBORE_RADIUS",
        0.25,
    ),
    ("bores.wells.factories", "make_producer", "wellbore_radius", "DEFAULT_WELLBORE_RADIUS", 0.25),
    ("bores.wells.factories", "make_injector", "wellbore_radius", "DEFAULT_WELLBORE_RADIUS", 0.25),
    (
        "bores.blackoil.pvt.factories",
        "build_pvt_dataset",
        "oil_specific_gravity",
        "DEFAULT_OIL_SPECIFIC_GRAVITY",
        0.85,
    ),
    (
        "bores.correlations.core",
        "compute_fluid_in_place",
        "acre_ft_to_bbl",
        "ACRE_FOOT_TO_BARRELS",
        7758,
    ),
    (
        "bores.correlations.core",
        "compute_fluid_in_place",
        "acre_ft_to_ft3",
        "ACRE_FOOT_TO_CUBIC_FEET",
        43560,
    ),
    (
        "bores.grids.containment",
        "cell_contains_point",
        "tolerance",
        "CELL_CONTAINMENT_TOLERANCE",
        1e-6,
    ),
    (
        "bores.grids.containment",
        "get_first_cell_containing_point",
        "tolerance",
        "CELL_CONTAINMENT_TOLERANCE",
        1e-6,
    ),
    (
        "bores.wells.indices.perforations",
        "check_cell_contains_point",
        "tolerance",
        "CELL_CONTAINMENT_TOLERANCE",
        1e-6,
    ),
    (
        "bores.grids.factories.corner_point",
        "make_corner_point_grid",
        "vertex_tolerance",
        "CORNER_POINT_VERTEX_TOLERANCE",
        1e-8,
    ),
    (
        "bores.wells.indices.wells",
        "is_locally_cartesian",
        "tolerance",
        "LOCALLY_CARTESIAN_TOLERANCE",
        0.05,
    ),
    (
        "bores.simulation.linear_solvers",
        "scale_linear_system",
        "epsilon",
        "LINEAR_SYSTEM_SCALING_EPSILON",
        1e-12,
    ),
    (
        "bores.blackoil.pvt.regions",
        "build_co2_temperature_axis",
        "step_fraction",
        "CO2_TEMPERATURE_STEP_FRACTION",
        0.12,
    ),
    (
        "bores.blackoil.pvt.regions",
        "build_co2_temperature_axis",
        "minimum_step",
        "CO2_TEMPERATURE_MINIMUM_STEP",
        0.18,
    ),
    (
        "bores.blackoil.pvt.co2_storage",
        "build_co2_salinity_axis",
        "spacing",
        "CO2_SALINITY_SPACING",
        10000.0,
    ),
    (
        "bores.blackoil.pvt.live_oil",
        "build_live_oil_branch_tables",
        "floor_fraction",
        "LIVE_OIL_FLOOR_FRACTION",
        0.05,
    ),
]


@pytest.mark.parametrize(("module", "function", "parameter", "constant", "value"), MOVED_DEFAULTS)
def test_default_is_unset_and_the_constant_keeps_the_old_value(
    module, function, parameter, constant, value
):
    signature = inspect.signature(getattr(importlib.import_module(module), function))
    assert signature.parameters[parameter].default is None
    assert getattr(c, constant) == pytest.approx(value)


def test_acre_foot_conversions_follow_the_constants():
    from bores.correlations.core import compute_fluid_in_place

    kwargs = dict(
        area=10.0, thickness=20.0, porosity=0.2, phase_saturation=0.7, formation_volume_factor=1.2
    )
    assert compute_fluid_in_place(**kwargs) == pytest.approx(
        10.0 * 20.0 * 0.2 * 0.7 * 7758.0 / 1.2
    )
    with ConstantsContext(Constants({"ACRE_FOOT_TO_BARRELS": 8000.0})):
        assert compute_fluid_in_place(**kwargs) == pytest.approx(
            10.0 * 20.0 * 0.2 * 0.7 * 8000.0 / 1.2
        )

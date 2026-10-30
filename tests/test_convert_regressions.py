import numpy as np

import bores.units as units
from bores.blackoil.pseudo_pressure import PseudoPressureTable
from bores.blackoil.satfunc.capillary_pressure.models.leverett_j import (
    LeverettJCapillaryPressureTable,
    convert_dyne_per_cm_to_pressure_unit,
)
from bores.types import UnitSystem
from bores.units import get_conversion_factors
from bores.wells.base import Well, WellType
from bores.wells.controls import EconomicLimit, EconomicQuantity
from bores.wells.factories import make_perforation
from bores.wells.groups import GroupControl, GroupProducerControlMode

FIELD_TO_METRIC = get_conversion_factors(UnitSystem.FIELD, UnitSystem.METRIC)


def test_pseudo_pressure_table_scales_pressures_and_pseudo_pressures_separately():
    pressures = np.linspace(500.0, 5000.0, 20)
    table = PseudoPressureTable(
        pressures=pressures,
        pseudo_pressures=pressures**2 / 0.02,
        reference_pressure=500.0,
        unit_system=UnitSystem.FIELD,
    )
    converted = table.convert(UnitSystem.METRIC)
    factors = FIELD_TO_METRIC
    assert np.allclose(converted.pressures, pressures * factors["pressure"])
    assert np.isclose(converted.reference_pressure, 500.0 * factors["pressure"])
    assert np.allclose(
        converted.pseudo_pressures,
        table.pseudo_pressures * factors["pressure"] ** 2 / factors["viscosity"],
    )


def test_well_converts_connection_factor_override_and_d_factor():
    perforation = make_perforation(1000.0, 1050.0, connection_factor_override=10.0)
    well = Well(
        name="P1",
        well_type=WellType.PRODUCER,
        surface_location=(100.0, 200.0),
        reference_depth=1000.0,
        perforations=(perforation,),
        d_factor=2.0e-3,
    )
    converted = well.convert(UnitSystem.METRIC)
    factors = FIELD_TO_METRIC
    assert np.isclose(
        converted.perforations[0].connection_factor_override,
        10.0 * factors["permeability"] * factors["length"],
    )
    assert np.isclose(converted.d_factor, 2.0e-3 / factors["gas_surface_rate"])
    back = converted.convert(UnitSystem.FIELD)
    assert np.isclose(back.d_factor, 2.0e-3)
    assert np.isclose(back.perforations[0].connection_factor_override, 10.0)


def test_group_control_converts_its_economic_limits_even_without_a_target_rate():
    limit = EconomicLimit(quantity=EconomicQuantity.OIL_RATE, min_value=100.0)
    for control in (
        GroupControl(mode=GroupProducerControlMode.OIL_RATE, target_rate=5000.0, limits=(limit,)),
        GroupControl(mode=GroupProducerControlMode.NONE, limits=(limit,)),
    ):
        converted = control.convert(UnitSystem.METRIC)
        assert converted.unit_system == UnitSystem.METRIC
        assert converted.limits[0].unit_system == UnitSystem.METRIC
        assert np.isclose(
            converted.limits[0].min_value, 100.0 * FIELD_TO_METRIC["liquid_surface_rate"]
        )


def test_leverett_j_follows_its_unit_system_without_rebuilding_the_table(monkeypatch):
    table = LeverettJCapillaryPressureTable().convert(UnitSystem.METRIC)
    assert table.dyne_per_cm_to_pressure_unit == convert_dyne_per_cm_to_pressure_unit(
        UnitSystem.METRIC
    )
    builds = []
    original = units.build_unit_conversion_table
    monkeypatch.setattr(
        units, "build_unit_conversion_table", lambda *a, **k: builds.append(1) or original(*a, **k)
    )
    table.dyne_per_cm_to_pressure_unit  # noqa: B018
    assert not builds

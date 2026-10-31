"""Tests that the hysteresis modules, now packages, keep their old import paths and share a base."""

import importlib

import pytest

from bores.blackoil.satfunc.capillary_pressure.hysteresis import (
    HysteresisCapillaryPressureTable,
    KilloughCapillaryPressureTable,
)
from bores.blackoil.satfunc.relperm.hysteresis import (
    HysteresisRelPermTable,
    KilloughLandRelPermTable,
    compute_land_residual_saturation,
    compute_land_trapping_coefficient,
)
from tests.satfunc.test_capillary_hysteresis import _hysteresis as _capillary_table
from tests.satfunc.test_relperm_hysteresis import _table as _relperm_table
from tests.satfunc.test_relperm_hysteresis import _tabular as _tabular_relperm


def test_the_models_share_their_family_base():
    assert issubclass(KilloughLandRelPermTable, HysteresisRelPermTable)
    assert issubclass(KilloughCapillaryPressureTable, HysteresisCapillaryPressureTable)


def test_the_old_module_paths_and_the_new_modules_give_the_same_objects():
    land = importlib.import_module("bores.blackoil.satfunc.relperm.hysteresis.land")
    killough = importlib.import_module("bores.blackoil.satfunc.relperm.hysteresis.killough")
    assert land.compute_land_residual_saturation is compute_land_residual_saturation
    assert land.compute_land_trapping_coefficient is compute_land_trapping_coefficient
    assert killough.KilloughLandRelPermTable is KilloughLandRelPermTable


def test_the_names_the_package_exposes_are_still_exported_from_satfunc():
    satfunc = importlib.import_module("bores.blackoil.satfunc")
    assert satfunc.KilloughLandRelPermTable is KilloughLandRelPermTable
    assert satfunc.KilloughCapillaryPressureTable is KilloughCapillaryPressureTable


def test_a_model_without_a_compute_is_not_usable():
    table = _tabular_relperm("baker_linear_rule")
    base = HysteresisRelPermTable(oil_water_drainage_table=table, gas_oil_drainage_table=table)
    with pytest.raises(NotImplementedError):
        base.evaluate(0.3, 0.5, 0.2)
    capillary = _capillary_table()
    bare = HysteresisCapillaryPressureTable(
        oil_water_drainage_table=capillary.oil_water_drainage_table,
        gas_oil_drainage_table=capillary.gas_oil_drainage_table,
    )
    with pytest.raises(NotImplementedError):
        bare.evaluate(0.3, 0.5, 0.2)


def test_the_killough_specific_exponent_lives_on_the_killough_subclasses_only():
    table = _relperm_table(
        _tabular_relperm("baker_linear_rule"), scanning_interpolation_exponent=2.0
    )
    assert table.scanning_interpolation_exponent == pytest.approx(2.0)
    assert not hasattr(HysteresisRelPermTable, "scanning_interpolation_exponent")
    assert not hasattr(HysteresisCapillaryPressureTable, "scanning_interpolation_exponent")

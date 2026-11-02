"""Tests for assembling hysteresis tables from a deck and picking them per `(SATNUM, IMBNUM)` pair."""

import logging

import numpy as np
import pytest

from bores.blackoil.caches.satfunc import compute_satfunc_cache
from bores.blackoil.satfunc.capillary_pressure.hysteresis import CurvatureCapillaryPressureTable
from bores.blackoil.satfunc.hysteresis import (
    HysteresisOptions,
    as_hysteresis_capillary_pressure_table,
    as_hysteresis_relperm_table,
    as_hysteresis_tables,
    derive_land_parameters,
    hysteresis_roles,
)
from bores.blackoil.satfunc.regions import SatFunc
from bores.blackoil.satfunc.relperm.hysteresis import (
    CarlsonRelPermTable,
    EclipseKilloughRelPermTable,
    KilloughLandRelPermTable,
    compute_land_residual_saturation,
)
from bores.deck import DeckFile
from bores.errors import ValidationError
from bores.reservoir.regions import Regions
from bores.types import FluidPhase, RelPermHysteresisModel

DRAINAGE_WATER_OIL = """
0.20 0.00 1.00 4.0
0.30 0.05 0.70 2.0
0.50 0.20 0.30 1.0
0.70 0.50 0.05 0.5
0.80 0.70 0.00 0.0 /
"""
IMBIBITION_WATER_OIL = """
0.20 0.00 1.00 3.0
0.30 0.08 0.60 1.5
0.50 0.30 0.20 0.7
0.65 0.55 0.00 0.3 /
"""
DRAINAGE_GAS_OIL = """
0.05 0.00 0.80 0.1
0.30 0.20 0.30 0.4
0.55 0.70 0.00 0.8
0.80 0.90 0.00 1.0 /
"""
IMBIBITION_GAS_OIL = """
0.15 0.00 0.75 0.1
0.40 0.25 0.20 0.3
0.55 0.60 0.00 0.5
0.80 0.90 0.00 1.0 /
"""


def _deck(
    *, satopts: str = " HYSTER /", ehystr: str | None = "0.2 0 1.0 0.1 BOTH RETR DRAIN WATER"
) -> DeckFile:
    text = (
        "METRIC\nRUNSPEC\nDIMENS\n 6 1 1 /\nOIL\nWATER\nGAS\nTABDIMS\n 2 1 /\n"
        f"SATOPTS\n{satopts}\nPROPS\n"
        f"SWOF\n{DRAINAGE_WATER_OIL}{IMBIBITION_WATER_OIL}"
        f"SGOF\n{DRAINAGE_GAS_OIL}{IMBIBITION_GAS_OIL}"
    )
    if ehystr is not None:
        text += f"EHYSTR\n {ehystr} /\n"
    return DeckFile(text)


def _satfunc(**options) -> SatFunc:
    return SatFunc.from_deck(_deck(**options), dtype=np.float64)


def test_a_deck_with_hyster_carries_its_options_and_one_without_does_not():
    assert _satfunc().hysteresis is not None
    assert _satfunc(satopts=" /").hysteresis is None


def test_land_is_derived_so_the_largest_start_traps_the_imbibition_tables_critical_saturation():
    satfunc = _satfunc()
    drainage = satfunc[1].relative_permeability
    imbibition = satfunc[2].relative_permeability
    land = derive_land_parameters(drainage=drainage, imbibition=imbibition)
    assert land.maximum_residual_gas_saturation == pytest.approx(0.15)
    assert land.maximum_residual_oil_saturation_water == pytest.approx(0.35)
    largest_gas = (
        1.0 - drainage.get_connate_water_saturation() - drainage.get_residual_oil_saturation_gas()
    )
    trapped = compute_land_residual_saturation(
        initial_non_wetting_saturation=largest_gas,
        maximum_residual_saturation=land.maximum_residual_gas_saturation,
        land_trapping_coefficient=land.land_trapping_coefficient_gas,
    )
    assert trapped == pytest.approx(land.maximum_residual_gas_saturation, rel=1e-6)
    assert land.land_trapping_coefficient_gas > 0.0


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (RelPermHysteresisModel.CARLSON, CarlsonRelPermTable),
        (RelPermHysteresisModel.KILLOUGH, EclipseKilloughRelPermTable),
        (RelPermHysteresisModel.BLENDED_SCANNING, KilloughLandRelPermTable),
    ],
)
def test_the_chosen_model_builds_its_table(model, expected):
    satfunc = _satfunc()
    table = as_hysteresis_relperm_table(
        drainage=satfunc[1].relative_permeability,
        imbibition=satfunc[2].relative_permeability,
        options=HysteresisOptions(relperm_model=model, wetting_phase=FluidPhase.WATER),
    )
    assert type(table) is expected


def test_relperm_stays_on_the_drainage_table_when_it_does_not_follow_hysteresis():
    satfunc = _satfunc()
    drainage, imbibition = satfunc[1].relative_permeability, satfunc[2].relative_permeability
    for options in (
        HysteresisOptions(relperm_model=RelPermHysteresisModel.NONE),
        HysteresisOptions(apply_to_relperm=False),
    ):
        assert (
            as_hysteresis_relperm_table(drainage=drainage, imbibition=imbibition, options=options)
            is drainage
        )


def test_the_killough_options_reach_the_eclipse_killough_table():
    satfunc = _satfunc()
    options = HysteresisOptions(
        relperm_model=RelPermHysteresisModel.KILLOUGH,
        wetting_phase=FluidPhase.WATER,
        wetting_phase_uses_imbibition=True,
        killough_wetting_curvature=2.5,
    )
    table = as_hysteresis_relperm_table(
        drainage=satfunc[1].relative_permeability,
        imbibition=satfunc[2].relative_permeability,
        options=options,
    )
    assert isinstance(table, EclipseKilloughRelPermTable)
    assert table.wetting_phase_hysteresis
    assert table.wetting_phase_curvature == pytest.approx(2.5)


def test_capillary_pressure_follows_the_options():
    satfunc = _satfunc()
    drainage, imbibition = satfunc[1].capillary_pressure, satfunc[2].capillary_pressure
    options = HysteresisOptions(
        capillary_pressure_curvature=0.3,
        retrace_capillary_pressure=False,
        wetting_phase=FluidPhase.WATER,
    )
    table = as_hysteresis_capillary_pressure_table(
        drainage=drainage, imbibition=imbibition, options=options
    )
    assert isinstance(table, CurvatureCapillaryPressureTable)
    assert table.scanning_curvature == pytest.approx(0.3)
    assert not table.retrace_scanning_curves
    off = HysteresisOptions(apply_to_capillary_pressure=False)
    assert (
        as_hysteresis_capillary_pressure_table(
            drainage=drainage, imbibition=imbibition, options=off
        )
        is drainage
    )
    assert (
        as_hysteresis_capillary_pressure_table(
            drainage=None, imbibition=imbibition, options=options
        )
        is None
    )


def test_a_pair_of_tables_is_assembled_into_one_set_of_tables():
    satfunc = _satfunc()
    assembled = as_hysteresis_tables(
        drainage=satfunc[1],
        imbibition=satfunc[2],
        options=satfunc.hysteresis,
    )
    assert isinstance(assembled.relative_permeability, CarlsonRelPermTable)
    assert isinstance(assembled.capillary_pressure, CurvatureCapillaryPressureTable)


def test_a_pair_is_assembled_once_and_a_cell_without_hysteresis_gets_its_drainage_tables():
    satfunc = _satfunc()
    assert satfunc.resolve(1, 2) is satfunc.resolve(1, 2)
    assert satfunc.resolve(1, 2) is not satfunc.resolve(2, 1)
    plain = _satfunc(satopts=" /")
    assert plain.resolve(1, 2) is plain[1]
    assert plain.resolve(2) is plain[2]


def test_hysteresis_needs_an_imbibition_region_that_exists():
    satfunc = _satfunc()
    with pytest.raises(ValidationError):
        satfunc.resolve(1)
    with pytest.raises(KeyError):
        satfunc.resolve(1, 7)
    groups = Regions(
        saturation_region=np.array([1, 1, 2, 2, 1, 2], dtype=np.int32)
    ).group_saturation_cells()
    with pytest.raises(ValidationError, match="IMBNUM"):
        satfunc.validate_region_groups(groups)
    complete = Regions(
        saturation_region=np.array([1, 1, 2, 2, 1, 2], dtype=np.int32),
        imbibition_region=np.array([2, 2, 1, 1, 2, 1], dtype=np.int32),
    ).group_saturation_cells(use_imbibition_region=True)
    satfunc.validate_region_groups(complete)


def test_the_wetting_roles_come_from_the_tables_and_a_disagreeing_ehystr_is_only_a_warning(caplog):
    water_wet = _satfunc()
    assert water_wet.hysteresis_roles() == (True, False)
    assert hysteresis_roles(water_wet[1].relative_permeability) == (True, False)
    disagreeing = _satfunc(ehystr="0.2 0 1.0 0.1 BOTH RETR DRAIN OIL")
    with caplog.at_level(logging.WARNING):
        assert disagreeing.hysteresis_roles() == (True, False)
    assert "wetting phase" in caplog.text


def _history(size):
    rng = np.random.default_rng(2)
    water = rng.uniform(0.3, 0.6, size)
    gas = rng.uniform(0.1, 0.3, size)
    return (
        water,
        gas,
        dict(
            max_water_saturation=water + 0.1,
            min_water_saturation=water - 0.08,
            water_imbibition_flag=rng.random(size) > 0.5,
            water_reversal_saturation=water - 0.08,
            max_gas_saturation=gas + 0.1,
            min_gas_saturation=gas - 0.05,
            gas_imbibition_flag=rng.random(size) > 0.5,
            gas_reversal_saturation=gas + 0.1,
        ),
    )


@pytest.mark.parametrize(
    "model", [None, RelPermHysteresisModel.BLENDED_SCANNING], ids=["carlson", "blended"]
)
def test_the_cache_evaluates_each_cell_with_the_tables_of_its_own_pair(model):
    satfunc = SatFunc.from_deck(_deck(), dtype=np.float64, relperm_model=model)
    size = 6
    saturation_region = np.array([1, 1, 2, 2, 1, 2], dtype=np.int32)
    imbibition_region = np.array([2, 2, 1, 1, 2, 1], dtype=np.int32)
    groups = Regions(
        saturation_region=saturation_region, imbibition_region=imbibition_region
    ).group_saturation_cells(use_imbibition_region=True)
    water, gas, history = _history(size)
    oil = 1.0 - water - gas
    cache = compute_satfunc_cache(
        water_saturation=water,
        oil_saturation=oil,
        gas_saturation=gas,
        saturation_region=saturation_region,
        satfunc=satfunc,
        region_groups=groups,
        dtype=np.float64,
        **history,
    )
    for pair in ((1, 2), (2, 1)):
        cells = (saturation_region == pair[0]) & (imbibition_region == pair[1])
        tables = satfunc.resolve(*pair)
        expected = tables.relative_permeability.evaluate(
            water[cells],
            oil[cells],
            gas[cells],
            **{name: value[cells] for name, value in history.items()},
        )
        np.testing.assert_allclose(
            cache.water_relative_permeability[cells], expected["water"], rtol=1e-12
        )
        np.testing.assert_allclose(
            cache.oil_relative_permeability[cells], expected["oil"], rtol=1e-12
        )
        np.testing.assert_allclose(
            cache.gas_relative_permeability[cells], expected["gas"], rtol=1e-12
        )
        assert not np.isnan(cache.oil_water_capillary_pressure[cells]).any()


def test_the_cache_groups_by_both_regions_itself_when_no_groups_are_given():
    satfunc = _satfunc()
    saturation_region = np.array([1, 1, 2, 2, 1, 2], dtype=np.int32)
    imbibition_region = np.array([2, 2, 1, 1, 2, 1], dtype=np.int32)
    water, gas, history = _history(6)
    arguments = dict(
        water_saturation=water,
        oil_saturation=1.0 - water - gas,
        gas_saturation=gas,
        saturation_region=saturation_region,
        satfunc=satfunc,
        dtype=np.float64,
        **history,
    )
    groups = Regions(
        saturation_region=saturation_region, imbibition_region=imbibition_region
    ).group_saturation_cells(use_imbibition_region=True)
    explicit = compute_satfunc_cache(**arguments, region_groups=groups)
    implicit = compute_satfunc_cache(**arguments, imbibition_region=imbibition_region)
    for field in explicit._fields:
        np.testing.assert_array_equal(getattr(explicit, field), getattr(implicit, field))


def test_hysteresis_changes_the_result_only_where_a_cell_has_a_history_to_follow():
    satfunc = _satfunc()
    plain = _satfunc(satopts=" /")
    size = 6
    saturation_region = np.ones(size, dtype=np.int32)
    imbibition_region = np.full(size, 2, dtype=np.int32)
    water, gas, history = _history(size)
    arguments = dict(
        water_saturation=water,
        oil_saturation=1.0 - water - gas,
        gas_saturation=gas,
        saturation_region=saturation_region,
        dtype=np.float64,
    )
    without_history = compute_satfunc_cache(
        **arguments, satfunc=satfunc, imbibition_region=imbibition_region
    )
    drainage = compute_satfunc_cache(**arguments, satfunc=plain)
    np.testing.assert_array_equal(
        without_history.gas_relative_permeability, drainage.gas_relative_permeability
    )
    with_history = compute_satfunc_cache(
        **arguments, satfunc=satfunc, imbibition_region=imbibition_region, **history
    )
    assert not np.allclose(
        with_history.gas_relative_permeability, drainage.gas_relative_permeability
    )


def test_converting_units_keeps_the_hysteresis_options_and_converts_the_scanned_pressures():
    from bores.types import UnitSystem
    from bores.units import get_conversion_factors

    metric = _satfunc()
    field = metric.convert(UnitSystem.FIELD)
    assert field.hysteresis is metric.hysteresis
    assert metric.resolve(1, 2).capillary_pressure.unit_system == UnitSystem.METRIC
    assert field.resolve(1, 2).capillary_pressure.unit_system == UnitSystem.FIELD

    water = np.array([0.35, 0.45])
    gas = np.array([0.1, 0.1])
    history = dict(
        max_water_saturation=np.full(2, 0.6),
        min_water_saturation=np.full(2, 0.3),
        water_imbibition_flag=np.full(2, True),
        water_reversal_saturation=np.full(2, 0.3),
    )
    in_metric = metric.resolve(1, 2).capillary_pressure.evaluate(
        water, 1.0 - water - gas, gas, **history
    )["oil_water"]
    in_field = field.resolve(1, 2).capillary_pressure.evaluate(
        water, 1.0 - water - gas, gas, **history
    )["oil_water"]
    factor = get_conversion_factors(UnitSystem.METRIC, UnitSystem.FIELD)["pressure"]
    np.testing.assert_allclose(in_field, np.asarray(in_metric) * factor, rtol=1e-9)

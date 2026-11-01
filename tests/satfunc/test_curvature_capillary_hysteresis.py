"""Tests for capillary pressure hysteresis with Killough's curvature-weighted scanning curves."""

import numpy as np
import pytest

from bores.blackoil.satfunc.capillary_pressure.hysteresis import (
    CurvatureCapillaryPressureTable,
    HysteresisCapillaryPressureTable,
)
from bores.blackoil.satfunc.capillary_pressure.hysteresis.curvature import (
    compute_curvature_scanning_weight,
)
from bores.blackoil.satfunc.capillary_pressure.models import BrooksCoreyCapillaryPressureTable
from bores.blackoil.satfunc.tables import as_three_phase_capillary_pressure_table
from bores.errors import ValidationError
from tests.satfunc.test_capillary_hysteresis import ENDPOINTS


def _tabular(entry_pressure_scale: float = 1.0):
    defaults = BrooksCoreyCapillaryPressureTable(**ENDPOINTS)
    return as_three_phase_capillary_pressure_table(
        BrooksCoreyCapillaryPressureTable(
            **ENDPOINTS,
            oil_water_entry_pressure_water_wet=defaults.oil_water_entry_pressure_water_wet
            * entry_pressure_scale,
            gas_oil_entry_pressure=defaults.gas_oil_entry_pressure * entry_pressure_scale,
        ),
        n_points=500,
        n_endpoint_extra=0,
        dtype=np.float64,
    )


def _table(curvature=0.1, with_imbibition=True):
    drainage = _tabular()
    imbibition = _tabular(entry_pressure_scale=0.5)
    return (
        CurvatureCapillaryPressureTable(
            oil_water_drainage_table=drainage.oil_water_table,
            gas_oil_drainage_table=drainage.gas_oil_table,
            oil_water_imbibition_table=imbibition.oil_water_table if with_imbibition else None,
            gas_oil_imbibition_table=imbibition.gas_oil_table if with_imbibition else None,
            scanning_curvature=curvature,
        ),
        drainage,
        imbibition,
    )


def _history(size, *, minimum=0.3, maximum=0.7, reversal=None, imbibing=True):
    return dict(
        max_water_saturation=np.full(size, maximum),
        min_water_saturation=np.full(size, minimum),
        water_imbibition_flag=np.full(size, imbibing),
        water_reversal_saturation=np.full(size, minimum if reversal is None else reversal),
    )


def _pcow(table, water, **history):
    water = np.atleast_1d(np.asarray(water, dtype=np.float64))
    gas = np.zeros_like(water)
    return np.asarray(table.evaluate(water, 1.0 - water - gas, gas, **history)["oil_water"])


def test_the_scanning_weight_runs_from_zero_at_the_start_to_one_at_the_far_end():
    for imbibing in (1.0, 0.0):
        start, end = (0.3, 0.7) if imbibing else (0.7, 0.3)
        low, _ = compute_curvature_scanning_weight(start, 0.3, 0.7, imbibing, 0.1, 1e-12)
        high, _ = compute_curvature_scanning_weight(end, 0.3, 0.7, imbibing, 0.1, 1e-12)
        # An imbibition scan starts on the drainage curve (weight 0), a drainage scan on the
        # imbibition curve (weight 1).
        assert low == pytest.approx(0.0 if imbibing else 1.0)
        assert high == pytest.approx(1.0 if imbibing else 0.0)


@pytest.mark.parametrize("curvature", [0.01, 0.1, 1.0])
def test_the_imbibition_weight_matches_killoughs_equation(curvature):
    low, high = 0.3, 0.7
    for saturation in (0.35, 0.5, 0.65):
        weight, _ = compute_curvature_scanning_weight(saturation, low, high, 1.0, curvature, 1e-12)
        numerator = 1.0 / (saturation - low + curvature) - 1.0 / curvature
        denominator = 1.0 / (high - low + curvature) - 1.0 / curvature
        assert weight == pytest.approx(numerator / denominator, rel=1e-12)


@pytest.mark.parametrize("curvature", [0.01, 0.1, 1.0])
def test_the_drainage_weight_matches_killoughs_equation(curvature):
    low, high = 0.3, 0.7
    for saturation in (0.35, 0.5, 0.65):
        weight, _ = compute_curvature_scanning_weight(saturation, low, high, 0.0, curvature, 1e-12)
        numerator = 1.0 / (high - saturation + curvature) - 1.0 / curvature
        denominator = 1.0 / (high - low + curvature) - 1.0 / curvature
        # Killough's F is the weight of the drainage curve on a drainage scan.
        assert 1.0 - weight == pytest.approx(numerator / denominator, rel=1e-12)


def test_a_smaller_curvature_moves_the_scan_over_to_the_other_bound_sooner():
    midway = [
        compute_curvature_scanning_weight(0.4, 0.3, 0.7, 1.0, curvature, 1e-12)[0]
        for curvature in (1.0, 0.1, 0.01)
    ]
    assert midway[0] < midway[1] < midway[2] < 1.0


@pytest.mark.parametrize("imbibing", [1.0, 0.0])
@pytest.mark.parametrize("saturation", [0.32, 0.4, 0.55, 0.68])
def test_the_weight_derivative_matches_finite_differences(imbibing, saturation):
    step = 1e-7
    _, slope = compute_curvature_scanning_weight(saturation, 0.3, 0.7, imbibing, 0.1, 1e-12)
    above, _ = compute_curvature_scanning_weight(saturation + step, 0.3, 0.7, imbibing, 0.1, 1e-12)
    below, _ = compute_curvature_scanning_weight(saturation - step, 0.3, 0.7, imbibing, 0.1, 1e-12)
    assert slope == pytest.approx((above - below) / (2.0 * step), rel=1e-5)


def test_outside_the_range_or_without_one_the_drainage_curve_is_used():
    weight, slope = compute_curvature_scanning_weight(0.5, 0.5, 0.5, 1.0, 0.1, 1e-12)
    assert (weight, slope) == (0.0, 0.0)
    below, _ = compute_curvature_scanning_weight(0.2, 0.3, 0.7, 1.0, 0.1, 1e-12)
    above, _ = compute_curvature_scanning_weight(0.9, 0.3, 0.7, 1.0, 0.1, 1e-12)
    assert below == pytest.approx(0.0)
    assert above == pytest.approx(1.0)


def test_an_imbibition_scan_leaves_the_drainage_curve_and_arrives_on_the_imbibition_curve():
    table, drainage, imbibition = _table()
    water = np.array([0.3, 0.7])
    scanned = _pcow(table, water, **_history(2))
    np.testing.assert_allclose(scanned[0], _pcow(drainage, water[:1])[0], rtol=1e-9)
    np.testing.assert_allclose(scanned[1], _pcow(imbibition, water[1:])[0], rtol=1e-9)
    inside = np.array([0.4, 0.5, 0.6])
    scanned = _pcow(table, inside, **_history(3))
    low, high = _pcow(imbibition, inside), _pcow(drainage, inside)
    assert np.all(scanned >= np.minimum(low, high) - 1e-12)
    assert np.all(scanned <= np.maximum(low, high) + 1e-12)


def test_a_drainage_scan_leaves_the_imbibition_curve_and_returns_to_the_drainage_curve():
    table, drainage, imbibition = _table()
    history = _history(2, reversal=0.7, imbibing=False)
    water = np.array([0.7, 0.3])
    scanned = _pcow(table, water, **history)
    np.testing.assert_allclose(scanned[0], _pcow(imbibition, water[:1])[0], rtol=1e-9)
    np.testing.assert_allclose(scanned[1], _pcow(drainage, water[1:])[0], rtol=1e-9)


def test_without_history_or_an_imbibition_curve_the_drainage_curve_is_used():
    table, drainage, _ = _table()
    water = np.array([0.35, 0.5])
    np.testing.assert_array_equal(_pcow(table, water), _pcow(drainage, water))
    no_imbibition, _, _ = _table(with_imbibition=False)
    np.testing.assert_array_equal(
        _pcow(no_imbibition, water, **_history(2)), _pcow(drainage, water)
    )
    incomplete = _history(2)
    del incomplete["min_water_saturation"]
    np.testing.assert_array_equal(_pcow(table, water, **incomplete), _pcow(drainage, water))


@pytest.mark.parametrize("imbibing", [True, False])
@pytest.mark.parametrize("water", [0.36, 0.5, 0.64])
def test_the_derivative_matches_finite_differences(imbibing, water):
    table, _, _ = _table()
    step = 1e-6
    history = dict(
        max_water_saturation=0.7,
        min_water_saturation=0.3,
        water_imbibition_flag=imbibing,
        water_reversal_saturation=0.3 if imbibing else 0.7,
    )

    def pc(value):
        return float(table.evaluate(value, 1.0 - value, 0.0, **history)["oil_water"])

    analytic = float(table.derivatives(water, 1.0 - water, 0.0, **history)["dpcow_dsw"])
    numeric = (pc(water + step) - pc(water - step)) / (2.0 * step)
    assert analytic == pytest.approx(numeric, rel=1e-4)


def _gas_history(size, *, peak=0.4, minimum=0.1, imbibing=True):
    return dict(
        max_gas_saturation=np.full(size, peak),
        gas_reversal_saturation=np.full(size, peak),
        min_gas_saturation=np.full(size, minimum),
        gas_imbibition_flag=np.full(size, imbibing),
    )


def _pcgo(table, gas, **history):
    gas = np.atleast_1d(np.asarray(gas, dtype=np.float64))
    water = np.full_like(gas, 0.3)
    return np.asarray(table.evaluate(water, 1.0 - water - gas, gas, **history)["gas_oil"])


def test_gas_scans_down_from_its_peak_on_the_drainage_curve_to_its_minimum_on_the_imbibition_curve():
    table, drainage, imbibition = _table()
    gas = np.array([0.4, 0.1])
    scanned = _pcgo(table, gas, **_gas_history(2))
    np.testing.assert_allclose(scanned[0], _pcgo(drainage, gas[:1])[0], rtol=1e-9)
    np.testing.assert_allclose(scanned[1], _pcgo(imbibition, gas[1:])[0], rtol=1e-9)
    inside = np.array([0.15, 0.25, 0.35])
    scanned = _pcgo(table, inside, **_gas_history(3))
    low, high = _pcgo(imbibition, inside), _pcgo(drainage, inside)
    assert np.all(scanned >= np.minimum(low, high) - 1e-12)
    assert np.all(scanned <= np.maximum(low, high) + 1e-12)
    assert not np.allclose(scanned, high)


def test_the_gas_scanning_curve_is_retraced_whichever_way_gas_moves():
    table, _, _ = _table()
    gas = np.array([0.15, 0.25, 0.35])
    falling = _pcgo(table, gas, **_gas_history(3, imbibing=True))
    rising = _pcgo(table, gas, **_gas_history(3, imbibing=False))
    np.testing.assert_array_equal(falling, rising)


def test_gas_above_its_peak_is_on_the_drainage_curve():
    table, drainage, _ = _table()
    gas = np.array([0.45, 0.5])
    np.testing.assert_allclose(
        _pcgo(table, gas, **_gas_history(2)), _pcgo(drainage, gas), rtol=1e-9
    )


def test_gas_without_its_history_or_a_distinct_imbibition_curve_uses_the_drainage_curve():
    table, drainage, _ = _table()
    gas = np.array([0.15, 0.3])
    np.testing.assert_array_equal(_pcgo(table, gas), _pcgo(drainage, gas))
    without_minimum = _gas_history(2)
    del without_minimum["min_gas_saturation"]
    np.testing.assert_array_equal(_pcgo(table, gas, **without_minimum), _pcgo(drainage, gas))
    no_imbibition, _, _ = _table(with_imbibition=False)
    np.testing.assert_array_equal(
        _pcgo(no_imbibition, gas, **_gas_history(2)), _pcgo(drainage, gas)
    )


def test_the_gas_peak_defaults_to_the_reversal_saturation_or_the_maximum():
    table, _, _ = _table()
    gas = np.array([0.15, 0.3])
    full = _gas_history(2)
    only_reversal = dict(
        gas_reversal_saturation=full["gas_reversal_saturation"],
        min_gas_saturation=full["min_gas_saturation"],
    )
    only_maximum = dict(
        max_gas_saturation=full["max_gas_saturation"],
        min_gas_saturation=full["min_gas_saturation"],
    )
    expected = _pcgo(table, gas, **full)
    np.testing.assert_array_equal(_pcgo(table, gas, **only_reversal), expected)
    np.testing.assert_array_equal(_pcgo(table, gas, **only_maximum), expected)


@pytest.mark.parametrize("gas", [0.14, 0.2, 0.3, 0.38])
def test_the_gas_derivative_matches_finite_differences(gas):
    table, _, _ = _table()
    step = 1e-6
    history = dict(max_gas_saturation=0.4, gas_reversal_saturation=0.4, min_gas_saturation=0.1)

    def pc(value):
        return float(table.evaluate(0.3, 0.7 - value, value, **history)["gas_oil"])

    analytic = table.derivatives(0.3, 0.7 - gas, gas, **history)
    slope = float(analytic["dpcgo_dsg"] - analytic["dpcgo_dso"])
    numeric = (pc(gas + step) - pc(gas - step)) / (2.0 * step)
    assert slope == pytest.approx(numeric, rel=1e-4)


def test_it_is_a_hysteresis_table_with_its_own_curvature():
    table, _, _ = _table(curvature=0.25)
    assert isinstance(table, HysteresisCapillaryPressureTable)
    assert table.scanning_curvature == pytest.approx(0.25)
    default, _, _ = _table(curvature=0.1)
    assert default.scanning_curvature == pytest.approx(0.1)


def test_invalid_configurations_are_rejected():
    drainage = _tabular()
    imbibition = _tabular(entry_pressure_scale=0.5)
    with pytest.raises(ValidationError):
        CurvatureCapillaryPressureTable(
            oil_water_drainage_table=drainage.oil_water_table,
            gas_oil_drainage_table=drainage.gas_oil_table,
            scanning_curvature=-0.1,
        )
    gas_wet = as_three_phase_capillary_pressure_table(
        BrooksCoreyCapillaryPressureTable(**ENDPOINTS),
        n_points=100,
        n_endpoint_extra=0,
        dtype=np.float64,
        gas_oil_wetting_phase="gas",
    )
    with pytest.raises(ValidationError):
        CurvatureCapillaryPressureTable(
            oil_water_drainage_table=gas_wet.oil_water_table,
            gas_oil_drainage_table=gas_wet.gas_oil_table,
            gas_oil_imbibition_table=imbibition.gas_oil_table,
        )

"""Tests for Carlson relative permeability hysteresis (translated imbibition scanning curves)."""

import numpy as np
import pytest

from bores.blackoil.satfunc.relperm.hysteresis import CarlsonRelPermTable
from bores.blackoil.satfunc.relperm.hysteresis.carlson import (
    tabulate_translation,
    translate_scanning_saturation,
)
from bores.blackoil.satfunc.tables import as_three_phase_relperm_table
from bores.errors import ValidationError
from bores.types import Wettability
from tests.satfunc import test_mixing_rules as mixing
from tests.satfunc.test_relperm_hysteresis import _tabular

CONNATE_WATER = 0.2
PEAK = 0.4


def _imbibition_table(maximum_gas_relperm: float = 0.9, maximum_oil_relperm: float = 0.8):
    """
    A table whose gas curve lies below the drainage gas curve (it has a larger critical gas
    saturation) and reaches the same maximum, as real
    imbibition data does, so a translated curve can pass through any reversal point.
    """
    return as_three_phase_relperm_table(
        mixing.LETThreePhaseRelPermTable(
            **{
                **mixing.ENDPOINTS,
                "residual_gas_saturation": 0.15,
                "residual_oil_saturation_gas": 0.2,
                "residual_oil_saturation_water": 0.35,
            },
            **mixing.NO_FLOORS,
            wettability=Wettability.WATER_WET,
            mixing_rule="baker_linear_rule",
            maximum_oil_relperm=maximum_oil_relperm,
            maximum_water_relperm=0.7,
            maximum_gas_relperm=maximum_gas_relperm,
        ),
        n_points=500,
        n_endpoint_extra=0,
        mixing_rule="baker_linear_rule",
        dtype=np.float64,
    )


def _table(imbibition=None, with_oil_water=False, **options):
    base = _tabular("baker_linear_rule")
    imbibition = _imbibition_table() if imbibition is None else imbibition
    return CarlsonRelPermTable(
        oil_water_drainage_table=base,
        gas_oil_drainage_table=base,
        gas_oil_imbibition_table=imbibition,
        oil_water_imbibition_table=imbibition if with_oil_water else None,
        **options,
    ), base


def _history(peak, size, *, imbibing=True):
    return dict(
        max_gas_saturation=np.full(size, peak),
        gas_imbibition_flag=np.full(size, imbibing),
        gas_reversal_saturation=np.full(size, peak),
    )


def _krg(table, gas, **history):
    gas = np.atleast_1d(np.asarray(gas, dtype=np.float64))
    water = np.full_like(gas, CONNATE_WATER)
    return np.asarray(table.evaluate(water, 1.0 - water - gas, gas, **history)["gas"])


def test_the_translated_curve_passes_through_the_reversal_point():
    table, base = _table()
    for peak in (0.2, 0.3, 0.4, 0.5):
        peak_array = np.array([peak])
        shifted = _imbibition_krg_at(
            table, peak_array + np.interp(peak, table.gas_shift_axis, table.gas_shift)
        )
        assert shifted[0] == pytest.approx(_krg(base, peak_array)[0], rel=1e-3, abs=1e-6)


def _imbibition_krg_at(table, gas):
    return _krg(table.gas_oil_imbibition_table, gas)


def test_gas_is_on_the_drainage_curve_at_and_above_its_peak():
    table, base = _table()
    gas = np.array([PEAK, 0.45, 0.5])
    scanned = _krg(table, gas, **_history(PEAK, gas.size))
    np.testing.assert_allclose(scanned, _krg(base, gas), rtol=1e-9)


def test_below_the_peak_the_scanning_curve_is_the_imbibition_curve_translated():
    table, _ = _table()
    gas = np.array([0.38, 0.33, 0.28, 0.22])
    delta = float(np.interp(PEAK, table.gas_shift_axis, table.gas_shift))
    scanned = _krg(table, gas, **_history(PEAK, gas.size))
    expected = _imbibition_krg_at(table, gas + delta)
    np.testing.assert_allclose(scanned, expected, rtol=1e-9)
    assert delta > 0.0


def test_the_scanning_curve_leaves_the_drainage_curve_continuously_and_stays_below_it():
    table, base = _table()
    gas = np.array([PEAK - 1e-6, 0.37, 0.3, 0.2])
    scanned = _krg(table, gas, **_history(PEAK, gas.size))
    drainage = _krg(base, gas)
    assert scanned[0] == pytest.approx(drainage[0], rel=1e-3)
    assert np.all(scanned <= drainage + 1e-12)
    assert np.all(np.diff(scanned) <= 0.0)


def test_scanning_curves_from_different_peaks_are_parallel_copies_of_the_imbibition_curve():
    table, _ = _table()
    gas = np.array([0.25])
    for peak in (0.3, 0.4, 0.5):
        delta = float(np.interp(peak, table.gas_shift_axis, table.gas_shift))
        scanned = _krg(table, gas, **_history(peak, 1))
        np.testing.assert_allclose(scanned, _imbibition_krg_at(table, gas + delta), rtol=1e-9)


def test_trapped_gas_is_where_the_translated_curve_reaches_zero():
    table, _ = _table()
    delta = float(np.interp(PEAK, table.gas_shift_axis, table.gas_shift))
    trapped = table.gas_oil_imbibition_table.get_residual_gas_saturation() - delta
    below = np.array([max(trapped - 0.01, 0.0), max(trapped, 0.0)])
    np.testing.assert_allclose(_krg(table, below, **_history(PEAK, 2)), 0.0, atol=1e-12)
    above = np.array([trapped + 0.02])
    assert _krg(table, above, **_history(PEAK, 1))[0] > 0.0


def test_the_scanning_curve_is_retraced_when_gas_rises_again():
    table, _ = _table()
    gas = np.array([0.35, 0.28])
    falling = _krg(table, gas, **_history(PEAK, 2, imbibing=True))
    rising = _krg(table, gas, **_history(PEAK, 2, imbibing=False))
    np.testing.assert_array_equal(falling, rising)


def test_without_history_or_an_imbibition_curve_the_drainage_curve_is_used():
    table, base = _table()
    gas = np.array([0.15, 0.3])
    np.testing.assert_array_equal(_krg(table, gas), _krg(base, gas))
    drainage_only = CarlsonRelPermTable(oil_water_drainage_table=base, gas_oil_drainage_table=base)
    np.testing.assert_array_equal(_krg(drainage_only, gas, **_history(PEAK, 2)), _krg(base, gas))


@pytest.mark.parametrize("gas", [0.18, 0.27, 0.33, 0.38])
def test_the_gas_derivative_matches_finite_differences(gas):
    table, _ = _table()
    step = 1e-6
    history = dict(max_gas_saturation=PEAK, gas_imbibition_flag=True, gas_reversal_saturation=PEAK)

    def krg(value):
        return float(table.evaluate(0.2, 0.8 - value, value, **history)["gas"])

    analytic = table.derivatives(0.2, 0.8 - gas, gas, **history)
    slope = float(analytic["dkrg_dsg"] - analytic["dkrg_dso"])
    numeric = (krg(gas + step) - krg(gas - step)) / (2.0 * step)
    assert slope == pytest.approx(numeric, rel=1e-3)


def test_the_wetting_phase_takes_its_imbibition_curve_only_when_asked():
    table, base = _table()
    asked, _ = _table(wetting_phase_uses_imbibition=True)
    imbibition = table.gas_oil_imbibition_table
    water = np.array([CONNATE_WATER])
    gas = np.array([0.3])
    history = _history(PEAK, 1)
    oil_default = asked_oil = None
    oil_default = float(table.evaluate(water, 1.0 - water - gas, gas, **history)["oil"][0])
    asked_oil = float(asked.evaluate(water, 1.0 - water - gas, gas, **history)["oil"][0])
    assert oil_default == pytest.approx(
        float(base.evaluate(water, 1.0 - water - gas, gas)["oil"][0])
    )
    assert asked_oil == pytest.approx(
        float(imbibition.evaluate(water, 1.0 - water - gas, gas)["oil"][0])
    )
    assert asked_oil != pytest.approx(oil_default)


def test_an_oil_water_imbibition_table_is_accepted_when_water_is_the_wetting_phase():
    base = _tabular("baker_linear_rule")
    CarlsonRelPermTable(
        oil_water_drainage_table=base,
        gas_oil_drainage_table=base,
        oil_water_imbibition_table=_imbibition_table(),
    )


def test_oil_wet_oil_water_scanning_is_rejected_unless_the_table_only_serves_the_wetting_phase():
    oil_wet = _tabular("baker_linear_rule", oil_water_wetting_phase="oil")
    with pytest.raises(ValidationError):
        CarlsonRelPermTable(
            oil_water_drainage_table=oil_wet,
            gas_oil_drainage_table=oil_wet,
            oil_water_imbibition_table=_imbibition_table(),
        )
    CarlsonRelPermTable(
        oil_water_drainage_table=oil_wet,
        gas_oil_drainage_table=oil_wet,
        oil_water_imbibition_table=_imbibition_table(),
        wetting_phase_uses_imbibition=True,
    )


def test_the_translation_of_a_curve_shifted_by_a_constant_is_that_constant():
    axis = np.linspace(0.0, 1.0, 101)
    drainage = axis.copy()
    imbibition = np.clip(axis - 0.1, 0.0, None)
    shift, scale = tabulate_translation(
        axis=axis, drainage_values=drainage, imbibition_values=imbibition
    )
    np.testing.assert_allclose(shift[5:90], 0.1, atol=1e-9)
    # Where the imbibition curve tops out at 0.9, the top is lifted onto the drainage value.
    np.testing.assert_allclose(scale[:90], 1.0)
    assert scale[-1] == pytest.approx(1.0 / 0.9)
    assert shift[-1] == pytest.approx(0.1 - 0.1) or shift[-1] <= 0.0


def test_the_translation_is_read_by_interpolation_and_clamped_beyond_the_axis():
    axis = np.array([0.0, 0.5, 1.0])
    shift = np.array([0.0, 0.2, 0.1])
    scale = np.array([1.0, 1.0, 2.0])
    saturation = np.array([0.3, 0.3, 0.3, 0.3])
    reversal = np.array([-0.2, 0.25, 0.75, 2.0])
    out = np.empty(4)
    out_scale = np.empty(4)
    translate_scanning_saturation(
        size=4,
        saturation=saturation,
        reversal_saturation=reversal,
        shift_axis=axis,
        shift=shift,
        vertical_scale=scale,
        reflect=False,
        out_saturation=out,
        out_vertical_scale=out_scale,
    )
    np.testing.assert_allclose(out, [0.3 + 0.0, 0.3 + 0.1, 0.3 + 0.15, 0.3 + 0.1])
    np.testing.assert_allclose(out_scale, [1.0, 1.0, 1.5, 2.0])


def test_the_reflected_axis_translates_the_complement_of_the_saturation():
    axis = np.array([0.0, 1.0])
    shift = np.array([0.2, 0.2])
    out = np.empty(1)
    out_scale = np.empty(1)
    translate_scanning_saturation(
        size=1,
        saturation=np.array([0.5]),
        reversal_saturation=np.array([0.3]),
        shift_axis=axis,
        shift=shift,
        vertical_scale=np.ones(2),
        reflect=True,
        out_saturation=out,
        out_vertical_scale=out_scale,
    )
    # Oil is 0.5, translated to 0.7, which is a water saturation of 0.3.
    assert out[0] == pytest.approx(0.3)


WATER_MINIMUM = 0.3


def _oil_history(size, minimum=WATER_MINIMUM):
    return dict(
        max_water_saturation=np.full(size, 0.6),
        water_imbibition_flag=np.ones(size, dtype=bool),
        min_water_saturation=np.full(size, minimum),
    )


def _kro(table, water, **history):
    water = np.atleast_1d(np.asarray(water, dtype=np.float64))
    gas = np.zeros_like(water)
    return np.asarray(table.evaluate(water, 1.0 - water - gas, gas, **history)["oil"])


def test_oil_is_on_the_drainage_curve_at_and_below_the_minimum_water_saturation():
    table, base = _table(with_oil_water=True)
    water = np.array([WATER_MINIMUM, 0.27, 0.22])
    scanned = _kro(table, water, **_oil_history(water.size))
    np.testing.assert_allclose(scanned, _kro(base, water), rtol=1e-9)


def test_above_the_minimum_water_saturation_oil_follows_the_translated_imbibition_curve():
    table, base = _table(with_oil_water=True)
    water = np.array([0.34, 0.42, 0.5, 0.58])
    delta = float(np.interp(1.0 - WATER_MINIMUM, table.oil_shift_axis, table.oil_shift))
    factor = float(np.interp(1.0 - WATER_MINIMUM, table.oil_shift_axis, table.oil_scale))
    scanned = _kro(table, water, **_oil_history(water.size))
    expected = _kro(table.oil_water_imbibition_table, water - delta) * factor
    np.testing.assert_allclose(scanned, expected, rtol=1e-9)
    assert np.all(scanned <= _kro(base, water) + 1e-12)
    assert delta > 0.0


def test_oil_leaves_the_drainage_curve_continuously_at_the_minimum_water_saturation():
    table, base = _table(with_oil_water=True)
    water = np.array([WATER_MINIMUM + 1e-6])
    scanned = _kro(table, water, **_oil_history(1))
    assert scanned[0] == pytest.approx(_kro(base, water)[0], rel=1e-3)


def test_oil_scanning_needs_the_minimum_water_saturation_history():
    table, base = _table(with_oil_water=True)
    water = np.array([0.4, 0.5])
    history = _oil_history(2)
    del history["min_water_saturation"]
    np.testing.assert_array_equal(_kro(table, water, **history), _kro(base, water))
    np.testing.assert_array_equal(_kro(table, water), _kro(base, water))


@pytest.mark.parametrize("water", [0.34, 0.42, 0.5])
def test_the_oil_derivative_matches_finite_differences(water):
    table, _ = _table(with_oil_water=True)
    step = 1e-6
    history = dict(
        max_water_saturation=0.6, water_imbibition_flag=True, min_water_saturation=WATER_MINIMUM
    )

    def kro(value):
        return float(table.evaluate(value, 1.0 - value, 0.0, **history)["oil"])

    analytic = table.derivatives(water, 1.0 - water, 0.0, **history)
    slope = float(analytic["dkro_dsw"] - analytic["dkro_dso"])
    numeric = (kro(water + step) - kro(water - step)) / (2.0 * step)
    assert slope == pytest.approx(numeric, rel=1e-3)


def test_a_wetting_phase_choice_does_not_change_the_oil_scanning_curve():
    plain, _ = _table(with_oil_water=True)
    asked, _ = _table(with_oil_water=True, wetting_phase_uses_imbibition=True)
    water = np.array([0.4, 0.5])
    np.testing.assert_allclose(
        _kro(plain, water, **_oil_history(2)), _kro(asked, water, **_oil_history(2)), rtol=1e-9
    )


def test_an_imbibition_curve_that_tops_out_low_is_lifted_onto_the_drainage_curve():
    table, base = _table(imbibition=_imbibition_table(maximum_gas_relperm=0.5))
    # Drainage gas reaches 0.5959 at the peak, more than the 0.5 the imbibition curve ever reaches.
    assert float(np.interp(PEAK, table.gas_shift_axis, table.gas_scale)) > 1.0
    gas = np.array([PEAK, PEAK - 1e-6, PEAK - 0.02, 0.3])
    scanned = _krg(table, gas, **_history(PEAK, gas.size))
    drainage = _krg(base, gas)
    assert scanned[0] == pytest.approx(drainage[0], rel=1e-9)
    assert scanned[1] == pytest.approx(drainage[1], rel=1e-3)
    assert np.all(np.diff(scanned[1:]) <= 0.0)
    # Where the imbibition curve can reach the drainage value the factor is exactly one.
    assert float(np.interp(0.2, table.gas_shift_axis, table.gas_scale)) == pytest.approx(1.0)


@pytest.mark.parametrize("gas", [0.3, 0.35, 0.38])
def test_the_lifted_curve_derivative_matches_finite_differences(gas):
    table, _ = _table(imbibition=_imbibition_table(maximum_gas_relperm=0.5))
    step = 1e-6
    history = dict(max_gas_saturation=PEAK, gas_imbibition_flag=True, gas_reversal_saturation=PEAK)

    def krg(value):
        return float(table.evaluate(0.2, 0.8 - value, value, **history)["gas"])

    analytic = table.derivatives(0.2, 0.8 - gas, gas, **history)
    slope = float(analytic["dkrg_dsg"] - analytic["dkrg_dso"])
    numeric = (krg(gas + step) - krg(gas - step)) / (2.0 * step)
    assert slope == pytest.approx(numeric, rel=1e-3)

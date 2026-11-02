"""Tests for Killough relative permeability hysteresis as Eclipse defines it."""

import numpy as np
import pytest

from bores.blackoil.satfunc.relperm.hysteresis import (
    EclipseKilloughRelPermTable,
    KilloughLandRelPermTable,
)
from bores.blackoil.satfunc.relperm.hysteresis.eclipse_killough import killough_axis_kernel
from bores.blackoil.satfunc.tables import as_three_phase_relperm_table
from bores.errors import ValidationError
from bores.types import Wettability
from tests.satfunc import test_mixing_rules as mixing
from tests.satfunc.test_relperm_hysteresis import _table as _blended_table
from tests.satfunc.test_relperm_hysteresis import _tabular

CONNATE_WATER = 0.2
PEAK = 0.4
WATER_MINIMUM = 0.3


def _imbibition_table():
    """A table with larger critical gas and residual oil, so it traps more than drainage."""
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
            maximum_oil_relperm=0.8,
            maximum_water_relperm=0.7,
            maximum_gas_relperm=0.9,
        ),
        n_points=500,
        n_endpoint_extra=0,
        mixing_rule="baker_linear_rule",
        dtype=np.float64,
    )


def _table(**options):
    base = _tabular("baker_linear_rule")
    imbibition = _imbibition_table()
    return (
        EclipseKilloughRelPermTable(
            oil_water_drainage_table=base,
            gas_oil_drainage_table=base,
            oil_water_imbibition_table=imbibition,
            gas_oil_imbibition_table=imbibition,
            **options,
        ),
        base,
        imbibition,
    )


def _gas_history(size, peak=PEAK, imbibing=True):
    return dict(
        max_gas_saturation=np.full(size, peak),
        gas_imbibition_flag=np.full(size, imbibing),
        gas_reversal_saturation=np.full(size, peak),
    )


def _oil_history(size, minimum=WATER_MINIMUM):
    return dict(
        max_water_saturation=np.full(size, 0.6),
        water_imbibition_flag=np.ones(size, dtype=bool),
        min_water_saturation=np.full(size, minimum),
    )


def _relperms(table, water, gas, **history):
    water = np.atleast_1d(np.asarray(water, dtype=np.float64))
    gas = np.atleast_1d(np.asarray(gas, dtype=np.float64))
    return table.evaluate(water, 1.0 - water - gas, gas, **history)


def _gas(table, gas, **history):
    gas = np.atleast_1d(np.asarray(gas, dtype=np.float64))
    return _relperms(table, np.full_like(gas, CONNATE_WATER), gas, **history)


def _trapped(system, peak):
    excess = peak - system.critical_drainage
    return system.critical_drainage + excess / (1.0 + system.land_coefficient * excess)


def test_the_endpoints_come_from_the_two_tables_and_the_largest_start_traps_the_critical_saturation():
    table, base, _ = _table()
    system = table.gas_system
    assert system.critical_drainage == pytest.approx(base.get_residual_gas_saturation())
    assert system.critical_imbibition == pytest.approx(0.15)
    assert system.maximum == pytest.approx(1.0 - CONNATE_WATER - 0.15)
    assert system.land_coefficient == pytest.approx(
        1.0 / (0.15 - system.critical_drainage) - 1.0 / (system.maximum - system.critical_drainage)
    )
    # Land measured from the drainage critical saturation traps `critical_imbibition` at the maximum.
    assert _trapped(system, system.maximum) == pytest.approx(system.critical_imbibition)
    assert table.oil_system.critical_imbibition == pytest.approx(0.35)


def test_gas_is_on_the_drainage_curve_at_and_above_its_peak():
    table, base, _ = _table()
    gas = np.array([PEAK, 0.45, 0.5])
    scanned = _gas(table, gas, **_gas_history(3))["gas"]
    np.testing.assert_allclose(scanned, _gas(base, gas)["gas"], rtol=1e-9)


def test_gas_below_its_peak_is_the_stretched_and_scaled_imbibition_curve():
    table, base, imbibition = _table()
    system = table.gas_system
    gas = np.array([0.37, 0.3, 0.25, 0.2])
    trapped = _trapped(system, PEAK)
    stretch = (system.maximum - system.critical_imbibition) / (PEAK - trapped)
    mapped = system.critical_imbibition + (gas - trapped) * stretch
    peak_value = _gas(base, np.array([PEAK]))["gas"][0]
    imbibition_value = _gas(imbibition, mapped)["gas"]
    at_critical = _gas(imbibition, np.array([system.critical_imbibition]))["gas"][0]
    at_maximum = _gas(imbibition, np.array([system.maximum]))["gas"][0]
    expected = peak_value * (imbibition_value - at_critical) / (at_maximum - at_critical)
    scanned = _gas(table, gas, **_gas_history(4))["gas"]
    np.testing.assert_allclose(scanned, expected, rtol=1e-9)
    assert np.all(np.diff(scanned) < 0.0)


def test_gas_is_continuous_at_the_reversal_and_immobile_at_and_below_the_trapped_saturation():
    table, base, _ = _table()
    system = table.gas_system
    trapped = _trapped(system, PEAK)
    near_peak = _gas(table, np.array([PEAK - 1e-7]), **_gas_history(1))["gas"][0]
    assert near_peak == pytest.approx(_gas(base, np.array([PEAK]))["gas"][0], rel=1e-4)
    low = np.array([trapped, trapped - 0.02, system.critical_drainage])
    np.testing.assert_allclose(_gas(table, low, **_gas_history(3))["gas"], 0.0, atol=1e-9)
    assert trapped > system.critical_drainage


def test_the_scanning_curve_is_retraced_whichever_way_gas_moves():
    table, _, _ = _table()
    gas = np.array([0.35, 0.3, 0.25])
    falling = _gas(table, gas, **_gas_history(3, imbibing=True))["gas"]
    rising = _gas(table, gas, **_gas_history(3, imbibing=False))["gas"]
    np.testing.assert_array_equal(falling, rising)


def test_without_history_or_an_imbibition_table_the_drainage_curves_are_used():
    table, base, _ = _table()
    gas = np.array([0.15, 0.3])
    np.testing.assert_array_equal(_gas(table, gas)["gas"], _gas(base, gas)["gas"])
    drainage_only = EclipseKilloughRelPermTable(
        oil_water_drainage_table=base, gas_oil_drainage_table=base
    )
    np.testing.assert_array_equal(
        _gas(drainage_only, gas, **_gas_history(2))["gas"], _gas(base, gas)["gas"]
    )


def test_the_wetting_phase_stays_on_drainage_unless_it_is_asked_to_scan():
    plain, base, _ = _table()
    scanning, _, _ = _table(wetting_phase_hysteresis=True)
    gas = np.array([0.35, 0.28, 0.22])
    drainage_oil = _gas(base, gas)["oil"]
    np.testing.assert_array_equal(_gas(plain, gas, **_gas_history(3))["oil"], drainage_oil)
    scanned_oil = _gas(scanning, gas, **_gas_history(3))["oil"]
    assert not np.allclose(scanned_oil, drainage_oil)


def test_the_wetting_phase_scans_from_the_drainage_value_to_its_value_at_the_trapped_saturation():
    table, base, _ = _table(wetting_phase_hysteresis=True, wetting_phase_curvature=1.5)
    system = table.gas_system
    trapped = _trapped(system, PEAK)
    at_peak = _gas(table, np.array([PEAK]), **_gas_history(1))["oil"][0]
    assert at_peak == pytest.approx(_gas(base, np.array([PEAK]))["oil"][0], rel=1e-9)
    drainage_at_trapped = _gas(base, np.array([trapped]))["oil"][0]
    expected_at_trapped = (
        drainage_at_trapped
        + (system.imbibition_wetting_at_critical - system.drainage_wetting_at_critical)
        * (trapped / system.critical_imbibition) ** 1.5
    )
    at_trapped = _gas(table, np.array([trapped]), **_gas_history(1))["oil"][0]
    assert at_trapped == pytest.approx(expected_at_trapped, rel=1e-6)


def test_it_is_not_the_blended_scheme():
    killough, base, imbibition = _table()
    blended = _blended_table(
        base,
        imbibition=imbibition,
        maximum_residual_gas_saturation=0.15,
        maximum_residual_oil_saturation_gas=0.2,
        maximum_residual_oil_saturation_water=0.35,
    )
    assert isinstance(blended, KilloughLandRelPermTable)
    gas = np.array([0.35, 0.3, 0.25])
    water = np.full(3, CONNATE_WATER)
    history = _gas_history(3)
    ours = _gas(killough, gas, **history)["gas"]
    theirs = np.asarray(blended.evaluate(water, 1.0 - water - gas, gas, **history)["gas"])
    assert not np.allclose(ours, theirs, rtol=1e-3)
    # The two meet at the reversal, where both are on the drainage curve.
    peak = np.array([PEAK])
    np.testing.assert_allclose(
        _gas(killough, peak, **_gas_history(1))["gas"],
        np.asarray(
            blended.evaluate(
                peak * 0 + CONNATE_WATER, 1.0 - CONNATE_WATER - peak, peak, **_gas_history(1)
            )["gas"]
        ),
        rtol=1e-6,
    )


@pytest.mark.parametrize("scan_wetting", [False, True])
@pytest.mark.parametrize("gas", [0.18, 0.25, 0.33, 0.38])
def test_the_gas_derivatives_match_finite_differences(gas, scan_wetting):
    table, _, _ = _table(wetting_phase_hysteresis=scan_wetting)
    step = 1e-6
    history = dict(max_gas_saturation=PEAK, gas_imbibition_flag=True, gas_reversal_saturation=PEAK)

    def value(name, point):
        return float(
            table.evaluate(CONNATE_WATER, 1.0 - CONNATE_WATER - point, point, **history)[name]
        )

    analytic = table.derivatives(CONNATE_WATER, 1.0 - CONNATE_WATER - gas, gas, **history)
    for name, key in (("gas", "dkrg_dsg"), ("oil", "dkro_dsg")):
        numeric = (value(name, gas + step) - value(name, gas - step)) / (2.0 * step)
        # The derivative with respect to gas at fixed water includes oil's response to the change.
        other = "dkrg_dso" if name == "gas" else "dkro_dso"
        assert float(analytic[key] - analytic[other]) == pytest.approx(numeric, rel=2e-3, abs=1e-6)


def _oil(table, water, **history):
    water = np.atleast_1d(np.asarray(water, dtype=np.float64))
    gas = np.zeros_like(water)
    return table.evaluate(water, 1.0 - water - gas, gas, **history)


def test_oil_is_on_the_drainage_curve_at_and_below_the_minimum_water_saturation():
    table, base, _ = _table()
    water = np.array([WATER_MINIMUM, 0.27, 0.22])
    np.testing.assert_allclose(
        _oil(table, water, **_oil_history(3))["oil"], _oil(base, water)["oil"], rtol=1e-9
    )


def test_above_the_minimum_water_saturation_oil_is_the_stretched_and_scaled_imbibition_curve():
    table, base, imbibition = _table()
    system = table.oil_system
    water = np.array([0.35, 0.45, 0.55])
    oil_peak = 1.0 - WATER_MINIMUM
    trapped = _trapped(system, oil_peak)
    stretch = (system.maximum - system.critical_imbibition) / (oil_peak - trapped)
    mapped_oil = system.critical_imbibition + ((1.0 - water) - trapped) * stretch
    peak_value = _oil(base, np.array([WATER_MINIMUM]))["oil"][0]
    imbibition_value = _oil(imbibition, 1.0 - mapped_oil)["oil"]
    at_critical = _oil(imbibition, np.array([1.0 - system.critical_imbibition]))["oil"][0]
    at_maximum = _oil(imbibition, np.array([1.0 - system.maximum]))["oil"][0]
    expected = peak_value * (imbibition_value - at_critical) / (at_maximum - at_critical)
    np.testing.assert_allclose(_oil(table, water, **_oil_history(3))["oil"], expected, rtol=1e-9)


@pytest.mark.parametrize("scan_wetting", [False, True])
@pytest.mark.parametrize("water", [0.34, 0.42, 0.5])
def test_the_oil_water_derivatives_match_finite_differences(water, scan_wetting):
    table, _, _ = _table(wetting_phase_hysteresis=scan_wetting)
    step = 1e-6
    history = dict(
        max_water_saturation=0.6, water_imbibition_flag=True, min_water_saturation=WATER_MINIMUM
    )

    def value(name, point):
        return float(table.evaluate(point, 1.0 - point, 0.0, **history)[name])

    analytic = table.derivatives(water, 1.0 - water, 0.0, **history)
    for name, key, other in (("oil", "dkro_dsw", "dkro_dso"), ("water", "dkrw_dsw", "dkrw_dso")):
        numeric = (value(name, water + step) - value(name, water - step)) / (2.0 * step)
        assert float(analytic[key] - analytic[other]) == pytest.approx(numeric, rel=2e-3, abs=1e-6)


def test_oil_scanning_needs_the_minimum_water_saturation_history():
    table, base, _ = _table()
    water = np.array([0.4, 0.5])
    history = _oil_history(2)
    del history["min_water_saturation"]
    np.testing.assert_array_equal(_oil(table, water, **history)["oil"], _oil(base, water)["oil"])


def test_the_axis_kernel_maps_the_reversal_to_the_maximum_and_the_trapped_saturation_to_critical():
    size = 3
    trapped = np.empty(size)
    mapped = np.empty(size)
    scale = np.empty(size)
    scanning = np.empty(size)
    killough_axis_kernel(
        size=size,
        saturation=np.array([0.4, 0.3, 0.45]),
        peak=np.array([0.4]),
        reflect=False,
        critical_drainage=0.05,
        critical_imbibition=0.15,
        maximum=0.65,
        land_coefficient=1.0 / 0.10 - 1.0 / 0.60,
        has_trapping=True,
        out_trapped=trapped,
        out_mapped=mapped,
        out_scale=scale,
        out_scanning=scanning,
    )
    expected_trapped = 0.05 + 0.35 / (1.0 + (1.0 / 0.10 - 1.0 / 0.60) * 0.35)
    assert trapped[0] == pytest.approx(expected_trapped)
    # At the peak itself, and above it, the cell is on the drainage curve.
    np.testing.assert_allclose(scanning, [0.0, 1.0, 0.0])
    assert mapped[1] == pytest.approx(
        0.15 + (0.3 - expected_trapped) * 0.5 / (0.4 - expected_trapped)
    )


def test_nothing_is_trapped_beyond_the_critical_saturation_when_the_curves_share_it():
    size = 1
    out = [np.empty(size) for _ in range(4)]
    killough_axis_kernel(
        size=size,
        saturation=np.array([0.3]),
        peak=np.array([0.4]),
        reflect=False,
        critical_drainage=0.05,
        critical_imbibition=0.05,
        maximum=0.65,
        land_coefficient=0.0,
        has_trapping=False,
        out_trapped=out[0],
        out_mapped=out[1],
        out_scale=out[2],
        out_scanning=out[3],
    )
    assert out[0][0] == pytest.approx(0.05)


def test_invalid_configurations_are_rejected():
    base = _tabular("baker_linear_rule")
    with pytest.raises(ValidationError):
        EclipseKilloughRelPermTable(
            oil_water_drainage_table=base,
            gas_oil_drainage_table=base,
            wetting_phase_curvature=0.0,
        )
    oil_wet = _tabular("baker_linear_rule", oil_water_wetting_phase="oil")
    with pytest.raises(ValidationError):
        EclipseKilloughRelPermTable(
            oil_water_drainage_table=oil_wet,
            gas_oil_drainage_table=oil_wet,
            oil_water_imbibition_table=_imbibition_table(),
        )


def test_gas_that_has_shrunk_to_below_the_trapped_saturation_is_immobile_and_does_not_break_the_wetting_phase():
    table, _, _ = _table(wetting_phase_hysteresis=True)
    system = table.gas_system
    trapped = _trapped(system, PEAK)
    gas = np.array([trapped, trapped - 0.05, system.critical_drainage, 0.01])
    result = _gas(table, gas, **_gas_history(4))
    np.testing.assert_allclose(result["gas"], 0.0, atol=1e-9)
    # The wetting phase has reached its value at the trapped saturation and stays there.
    assert np.all(np.isfinite(result["oil"]))
    np.testing.assert_allclose(result["oil"][1:], result["oil"][0], rtol=1e-6)

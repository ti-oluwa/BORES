"""Tests for the Killough/Land relative permeability hysteresis table."""

import numpy as np
import pytest

from bores.blackoil.satfunc.relperm.hysteresis import (
    KilloughLandRelPermTable,
    compute_land_residual_saturation,
    compute_land_trapping_coefficient,
)
from bores.blackoil.satfunc.relperm.tables import RelativePermeabilityTable
from bores.blackoil.satfunc.tables import as_three_phase_relperm_table
from bores.blackoil.satfunc.utils import compute_killough_scanning_weight
from bores.types import Wettability
from tests.satfunc.test_mixing_rules import RULES, _model, _points, _values


def _tabular(rule_name, **options):
    return as_three_phase_relperm_table(
        _model("let", Wettability.WATER_WET, rule_name),
        n_points=500,
        n_endpoint_extra=0,
        mixing_rule=rule_name,
        dtype=np.float64,
        **options,
    )


def _table(base, imbibition=None, **options):
    extra = {}
    if imbibition is not None:
        extra = {
            "oil_water_imbibition_table": imbibition.oil_water_table,
            "gas_oil_imbibition_table": imbibition.gas_oil_table,
        }
    return KilloughLandRelPermTable(
        oil_water_drainage_table=base.oil_water_table,
        gas_oil_drainage_table=base.gas_oil_table,
        **extra,
        **options,
    )


def _history(sw, sg):
    return dict(
        max_water_saturation=sw + 0.08,
        max_gas_saturation=sg + 0.08,
        water_imbibition_flag=np.ones_like(sw, dtype=bool),
        gas_imbibition_flag=np.ones_like(sg, dtype=bool),
        water_reversal_saturation=sw - 0.06,
        gas_reversal_saturation=sg + 0.08,
    )


def test_it_is_a_relative_permeability_table():
    assert issubclass(KilloughLandRelPermTable, RelativePermeabilityTable)


@pytest.mark.parametrize("rule_name", RULES)
def test_evaluate_with_derivatives_equals_the_separate_calls(rule_name):
    base = _tabular(rule_name)
    other = _tabular(rule_name, oil_water_reference_phase="wetting")
    table = _table(
        base, imbibition=other, mixing_rule=rule_name, scanning_interpolation_exponent=1.5
    )
    points = np.array(_points(30))
    sw, so, sg = points[:, 0], points[:, 1], points[:, 2]
    history = _history(sw, sg)
    values, slopes = table.evaluate_with_derivatives(sw, so, sg, **history)
    expected_values = table.evaluate(sw, so, sg, **history)
    expected_slopes = table.derivatives(sw, so, sg, **history)
    for key in expected_values:
        np.testing.assert_array_equal(values[key], expected_values[key])
    for key in expected_slopes:
        np.testing.assert_array_equal(slopes[key], expected_slopes[key])


@pytest.mark.parametrize("rule_name", ["baker_linear_rule", "stone_II_rule"])
def test_scanning_derivatives_match_finite_differences(rule_name):
    """The scanning variables are Sw and Sg, with So carrying the balance."""
    base = _tabular(rule_name)
    scaled = as_three_phase_relperm_table(
        _model("brooks_corey", Wettability.WATER_WET, rule_name),
        n_points=400,
        n_endpoint_extra=0,
        mixing_rule=rule_name,
        dtype=np.float64,
    )
    table = _table(base, imbibition=scaled, mixing_rule=rule_name)
    step = 1e-6
    for sw, so, sg in _points(12):
        history = _history(np.array([sw]), np.array([sg]))
        history["max_water_saturation"] = np.array([sw + 0.1])
        history["max_gas_saturation"] = np.array([sg + 0.1])

        def at(w, o, g):
            result = table.evaluate(np.array([w]), np.array([o]), np.array([g]), **history)
            return np.array([float(result[key][0]) for key in ("water", "oil", "gas")])

        slopes = table.derivatives(np.array([sw]), np.array([so]), np.array([sg]), **history)
        along_sw = np.array([
            float(slopes[f"dkr{p}_dsw"][0] - slopes[f"dkr{p}_dso"][0]) for p in "wog"
        ])
        fd = (at(sw + step, so - step, sg) - at(sw - step, so + step, sg)) / (2 * step)
        np.testing.assert_allclose(along_sw, fd, atol=2e-4)


@pytest.mark.parametrize("rule_name", ["baker_linear_rule", "stone_II_rule"])
def test_reoriented_drainage_tables_match_the_three_phase_table(rule_name):
    """A two-phase table is queried at its own system's saturations whatever its axis."""
    reoriented = _tabular(
        rule_name, oil_water_reference_phase="non_wetting", gas_oil_reference_phase="wetting"
    )
    table = _table(reoriented, mixing_rule=rule_name)
    for point in _points(15):
        np.testing.assert_allclose(_values(table, *point), _values(reoriented, *point), atol=1e-12)
    step = 1e-6
    for sw, so, sg in _points(8):
        slopes = table.derivatives(sw, so, sg)
        along_sw = np.array([slopes[f"dkr{p}_dsw"] - slopes[f"dkr{p}_dso"] for p in "wog"])
        fd = (
            _values(table, sw + step, so - step, sg) - _values(table, sw - step, so + step, sg)
        ) / (2 * step)
        np.testing.assert_allclose(along_sw, fd, atol=1e-5)


def test_missing_history_for_one_system_keeps_it_on_drainage():
    base = _tabular("baker_linear_rule")
    other = _tabular("baker_linear_rule", oil_water_reference_phase="wetting")
    table = _table(base, imbibition=other)
    points = np.array(_points(10))
    sw, so, sg = points[:, 0], points[:, 1], points[:, 2]
    history = _history(sw, sg)
    gas_only = {key: value for key, value in history.items() if "gas" in key}
    drainage = table.evaluate(sw, so, sg)
    scanned = table.evaluate(sw, so, sg, **gas_only)
    np.testing.assert_array_equal(scanned["water"], drainage["water"])


def test_scalar_and_array_inputs_agree():
    base = _tabular("stone_II_rule")
    table = _table(base, imbibition=_tabular("stone_II_rule", oil_water_reference_phase="wetting"))
    kwargs = dict(
        max_water_saturation=0.55,
        max_gas_saturation=0.2,
        water_imbibition_flag=True,
        gas_imbibition_flag=False,
        water_reversal_saturation=0.4,
        gas_reversal_saturation=0.1,
    )
    scalar = table.evaluate(0.45, 0.4, 0.15, **kwargs)
    array = table.evaluate(np.array([0.45]), np.array([0.4]), np.array([0.15]), **kwargs)
    for key in scalar:
        assert np.isscalar(scalar[key])
        assert scalar[key] == pytest.approx(float(array[key][0]))


@pytest.mark.parametrize("coefficient", [0.0, 1.5])
def test_land_residual_scalar_and_array_agree(coefficient):
    initial = np.array([0.0, 0.01, 0.3, 0.6])
    array = compute_land_residual_saturation(initial, 0.25, coefficient)
    for value, expected in zip(initial, array, strict=False):
        assert compute_land_residual_saturation(float(value), 0.25, coefficient) == pytest.approx(
            float(expected)
        )
    assert np.all(array <= initial + 1e-15)


def test_analytical_tables_with_land_trapping_evaluate():
    """Analytical (non-tabular) backing tables take the Land residual as an endpoint override."""
    drainage = _model("brooks_corey", Wettability.WATER_WET, "stone_II_rule")
    imbibition = _model("let", Wettability.WATER_WET, "stone_II_rule")
    table = KilloughLandRelPermTable(
        oil_water_drainage_table=drainage,
        gas_oil_drainage_table=drainage,
        oil_water_imbibition_table=imbibition,
        gas_oil_imbibition_table=imbibition,
        maximum_residual_oil_saturation_water=0.3,
        maximum_residual_oil_saturation_gas=0.2,
        maximum_residual_gas_saturation=0.15,
        mixing_rule="stone_II_rule",
    )
    sw, sg = np.array([0.4, 0.5]), np.array([0.1, 0.2])
    history = _history(sw, sg)
    values, slopes = table.evaluate_with_derivatives(sw, 1.0 - sw - sg, sg, **history)
    assert np.all(np.isfinite(values["oil"])) and np.all(np.isfinite(slopes["dkro_dsw"]))


def test_land_form_is_the_standard_one():
    maximum_residual, maximum_saturation = 0.25, 0.7
    coefficient = compute_land_trapping_coefficient(
        maximum_residual_saturation=maximum_residual,
        maximum_non_wetting_saturation=maximum_saturation,
    )
    assert coefficient == pytest.approx(1 / 0.25 - 1 / 0.7)
    at_maximum = compute_land_residual_saturation(
        initial_non_wetting_saturation=maximum_saturation,
        maximum_residual_saturation=maximum_residual,
        land_trapping_coefficient=coefficient,
    )
    assert at_maximum == pytest.approx(maximum_residual)
    low = compute_land_residual_saturation(
        initial_non_wetting_saturation=0.2,
        maximum_residual_saturation=maximum_residual,
        land_trapping_coefficient=coefficient,
    )
    assert low == pytest.approx(0.2 / (1 + coefficient * 0.2))
    assert low < 0.2


@pytest.mark.parametrize("exponent", [1.0, 1.8])
@pytest.mark.parametrize("imbibing", [True, False])
def test_scanning_weight_is_continuous_and_bounded(exponent, imbibing):
    saturations = np.linspace(0.2, 0.6, 4001)
    weights = np.array([
        compute_killough_scanning_weight(
            saturation=value,
            reversal_saturation=0.3,
            maximum_historical_saturation=0.5,
            is_imbibition=1.0 if imbibing else 0.0,
            scanning_exponent=exponent,
            numerical_epsilon=1e-12,
        )[0]
        for value in saturations
    ])
    assert weights.min() >= 0.0 and weights.max() <= 1.0
    assert np.max(np.abs(np.diff(weights))) < 5e-3
    assert weights[0] == 0.0 and weights[-1] == 1.0


@pytest.mark.parametrize("imbibing", [True, False])
def test_collapsed_scanning_range_sits_on_the_flagged_curve(imbibing):
    weight, slope = compute_killough_scanning_weight(
        saturation=0.4,
        reversal_saturation=0.4,
        maximum_historical_saturation=0.4,
        is_imbibition=1.0 if imbibing else 0.0,
        scanning_exponent=1.0,
        numerical_epsilon=1e-12,
    )
    assert weight == (1.0 if imbibing else 0.0)
    assert slope == 0.0


def _gas_scanning_table(*, tabular_land_trapping: bool = True):
    """A table whose gas imbibition curve lies well below its gas drainage curve."""
    from bores.blackoil.satfunc.tables import as_three_phase_relperm_table
    from tests.satfunc.test_mixing_rules import ENDPOINTS, NO_FLOORS, LETThreePhaseRelPermTable

    base = _tabular("baker_linear_rule")
    imbibition = as_three_phase_relperm_table(
        LETThreePhaseRelPermTable(
            **ENDPOINTS,
            **NO_FLOORS,
            wettability=Wettability.WATER_WET,
            mixing_rule="baker_linear_rule",
            maximum_oil_relperm=0.8,
            maximum_water_relperm=0.7,
            maximum_gas_relperm=0.45,
        ),
        n_points=500,
        n_endpoint_extra=0,
        mixing_rule="baker_linear_rule",
        dtype=np.float64,
    )
    table = _table(
        base,
        imbibition=imbibition,
        maximum_residual_gas_saturation=0.25,
        maximum_residual_oil_saturation_gas=0.3,
        maximum_residual_oil_saturation_water=0.3,
        tabular_land_trapping=tabular_land_trapping,
    )
    return table, base, imbibition


def test_non_wetting_gas_scans_down_from_the_drainage_curve_at_its_peak():
    table, base, imbibition = _gas_scanning_table(tabular_land_trapping=False)
    peak = 0.4
    sg = np.array([0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1])
    sw = np.full_like(sg, 0.2)
    so = 1.0 - sw - sg
    history = dict(
        max_gas_saturation=np.full_like(sg, peak),
        gas_imbibition_flag=np.ones_like(sg, dtype=bool),
        gas_reversal_saturation=np.full_like(sg, peak),
    )
    drainage = np.asarray(base.evaluate(sw, so, sg)["gas"])
    bounding_imbibition = np.asarray(imbibition.evaluate(sw, so, sg)["gas"])
    scanned = np.asarray(table.evaluate(sw, so, sg, **history)["gas"])

    # At the peak the cell is still on the drainage curve.
    assert scanned[0] == pytest.approx(drainage[0], rel=1e-9)
    # Below the peak it leaves the drainage curve and stays between the two bounding curves.
    assert np.all(scanned[1:] < drainage[1:])
    assert np.all(scanned[1:] > bounding_imbibition[1:])
    # The further gas falls, the closer it is to the imbibition curve.
    weight = (drainage - scanned) / (drainage - bounding_imbibition + 1e-300)
    assert np.all(np.diff(weight[1:]) > 0.0)


def test_rising_non_wetting_gas_retraces_back_onto_the_drainage_curve():
    table, base, _ = _gas_scanning_table()
    sg = np.array([0.1, 0.2, 0.3, 0.4, 0.45])
    sw = np.full_like(sg, 0.2)
    so = 1.0 - sw - sg
    history = dict(
        max_gas_saturation=np.full_like(sg, 0.4),
        gas_imbibition_flag=np.zeros_like(sg, dtype=bool),
        gas_reversal_saturation=np.full_like(sg, 0.4),
    )
    drainage = np.asarray(base.evaluate(sw, so, sg)["gas"])
    scanned = np.asarray(table.evaluate(sw, so, sg, **history)["gas"])
    # Gas beyond its previous peak is on the primary drainage curve again.
    assert scanned[-2:] == pytest.approx(drainage[-2:], rel=1e-9)
    assert np.all(scanned[:-2] <= drainage[:-2])


def test_gas_scanning_derivative_matches_finite_differences():
    table, _, _ = _gas_scanning_table()
    sg, step = 0.27, 1e-6
    history = dict(
        max_gas_saturation=0.4,
        gas_imbibition_flag=True,
        gas_reversal_saturation=0.4,
    )

    def krg(gas):
        return float(table.evaluate(0.2, 0.8 - gas, gas, **history)["gas"])

    analytic = table.derivatives(0.2, 0.8 - sg, sg, **history)
    slope = float(analytic["dkrg_dsg"] - analytic["dkrg_dso"])
    numeric = (krg(sg + step) - krg(sg - step)) / (2.0 * step)
    assert slope == pytest.approx(numeric, rel=1e-4)


def _gas_history(peak, size):
    return dict(
        max_gas_saturation=np.full(size, peak),
        gas_imbibition_flag=np.ones(size, dtype=bool),
        gas_reversal_saturation=np.full(size, peak),
    )


def test_tabular_land_trapping_ends_the_imbibition_curve_at_the_trapped_gas():
    table, base, _ = _gas_scanning_table()
    peak = 0.4
    trapped = float(
        table.land_residual_gas(
            size=1, water_saturation=np.array([0.2]), gas_reversal=np.array([peak])
        )[0]
    )
    below = np.linspace(0.0, trapped, 6)
    sw = np.full_like(below, 0.2)
    scanned = np.asarray(
        table.evaluate(sw, 1.0 - sw - below, below, **_gas_history(peak, below.size))["gas"]
    )
    # Gas at or below the trapped saturation is immobile, which the unscaled table puts at 0.05.
    np.testing.assert_allclose(scanned, 0.0, atol=1e-12)
    above = np.array([trapped + 0.01])
    sw = np.full_like(above, 0.2)
    mobile = table.evaluate(sw, 1.0 - sw - above, above, **_gas_history(peak, 1))["gas"]
    assert float(mobile[0]) > 0.0
    assert trapped > 0.05
    assert base.get_residual_gas_saturation() == pytest.approx(0.05)


def test_tabular_land_trapping_traps_more_gas_than_the_table_alone_when_asked():
    on, _, _ = _gas_scanning_table(tabular_land_trapping=True)
    off, _, _ = _gas_scanning_table(tabular_land_trapping=False)
    sg = np.array([0.1, 0.15, 0.2])
    sw = np.full_like(sg, 0.2)
    so = 1.0 - sw - sg
    history = _gas_history(0.4, sg.size)
    trapped_on = np.asarray(on.evaluate(sw, so, sg, **history)["gas"])
    trapped_off = np.asarray(off.evaluate(sw, so, sg, **history)["gas"])
    # The table alone is critical at 0.05, Land traps up to the larger saturation.
    assert np.all(trapped_on < trapped_off)


def test_switching_tabular_land_trapping_off_leaves_the_table_as_tabulated():
    off, base, _ = _gas_scanning_table(tabular_land_trapping=False)
    drainage_only = _table(
        base,
        imbibition=_gas_scanning_table()[2],
        maximum_residual_gas_saturation=None,
        maximum_residual_oil_saturation_gas=None,
        maximum_residual_oil_saturation_water=None,
    )
    sg = np.array([0.1, 0.2, 0.3])
    sw = np.full_like(sg, 0.2)
    so = 1.0 - sw - sg
    history = _gas_history(0.4, sg.size)
    np.testing.assert_array_equal(
        off.evaluate(sw, so, sg, **history)["gas"],
        drainage_only.evaluate(sw, so, sg, **history)["gas"],
    )


def test_the_rescaled_axis_maps_the_cell_span_onto_the_table_span():
    from bores.blackoil.satfunc.relperm.hysteresis.land import rescale_scanning_saturation

    reversal = np.array([0.4, 0.4])
    trapped = np.array([0.1, 0.4])
    saturation = np.array([0.4, 0.3])
    scaled = np.empty(2)
    scale = np.empty(2)
    rescale_scanning_saturation(
        size=2,
        saturation=saturation,
        reversal_saturation=reversal,
        trapped_saturation=trapped,
        on_wetting_axis=False,
        table_start=0.05,
        table_end=0.65,
        saturation_epsilon=1e-12,
        out_saturation=scaled,
        out_scale=scale,
    )
    # Cell one: the reversal maps to the table end, the span 0.3 stretches to 0.6.
    assert scaled[0] == pytest.approx(0.65)
    assert scale[0] == pytest.approx(2.0)
    # Cell two has nothing trapped between its reversal and its trapped saturation: unscaled.
    assert scaled[1] == pytest.approx(0.3)
    assert scale[1] == pytest.approx(1.0)


def test_the_wetting_axis_runs_from_the_reversal_up_to_one_minus_the_trapped_saturation():
    from bores.blackoil.satfunc.relperm.hysteresis.land import rescale_scanning_saturation

    scaled = np.empty(2)
    scale = np.empty(2)
    rescale_scanning_saturation(
        size=2,
        saturation=np.array([0.3, 0.7]),
        reversal_saturation=np.array([0.3, 0.3]),
        trapped_saturation=np.array([0.2, 0.2]),
        on_wetting_axis=True,
        table_start=0.2,
        table_end=0.8,
        saturation_epsilon=1e-12,
        out_saturation=scaled,
        out_scale=scale,
    )
    # Imbibition runs from 0.3 to 0.8, so the span is 0.5 and the table's 0.6 is stretched by 1.2.
    np.testing.assert_allclose(scaled, [0.2, 0.2 + 0.4 * 1.2])
    np.testing.assert_allclose(scale, [1.2, 1.2])


def test_tabular_gas_derivative_matches_finite_differences_with_the_land_rescaling():
    table, _, _ = _gas_scanning_table()
    step = 1e-6
    history = dict(max_gas_saturation=0.4, gas_imbibition_flag=True, gas_reversal_saturation=0.4)

    def krg(gas):
        return float(table.evaluate(0.2, 0.8 - gas, gas, **history)["gas"])

    for sg in (0.18, 0.27, 0.33):
        analytic = table.derivatives(0.2, 0.8 - sg, sg, **history)
        slope = float(analytic["dkrg_dsg"] - analytic["dkrg_dso"])
        numeric = (krg(sg + step) - krg(sg - step)) / (2.0 * step)
        assert slope == pytest.approx(numeric, rel=1e-3)


def test_tabular_land_trapping_makes_oil_immobile_at_the_land_residual_on_the_water_side():
    table, _, _ = _gas_scanning_table()
    reversal = 0.3
    land_residual = float(
        table.land_residual_oil_water(
            size=1, gas_saturation=np.array([0.0]), water_reversal=np.array([reversal])
        )[0]
    )
    end = 1.0 - land_residual
    history = dict(
        max_water_saturation=np.array([end]),
        water_imbibition_flag=np.array([True]),
        water_reversal_saturation=np.array([reversal]),
    )
    water = np.array([end])
    gas = np.zeros(1)
    at_end = table.evaluate(water, 1.0 - water - gas, gas, **history)
    assert float(at_end["oil"][0]) == pytest.approx(0.0, abs=1e-9)
    below = np.array([end - 0.02])
    assert float(table.evaluate(below, 1.0 - below - gas, gas, **history)["oil"][0]) > 0.0
    assert land_residual > 0.0


def test_the_land_model_traps_the_most_from_the_maximum_and_never_more_than_it_started_with():
    from bores.blackoil.satfunc.relperm.hysteresis import LandTrapping

    model = LandTrapping.from_endpoints(maximum_trapped_saturation=0.25, maximum_saturation=0.8)
    assert model.coefficient == pytest.approx(1.0 / 0.25 - 1.0 / 0.8)
    assert model.trapped_saturation(0.8) == pytest.approx(0.25)
    starts = np.array([0.0, 0.05, 0.2, 0.5, 0.8])
    trapped = model.trapped_saturation(starts)
    assert np.all(trapped <= starts + 1e-12)
    assert np.all(trapped <= 0.25 + 1e-12)
    assert np.all(np.diff(trapped) >= 0.0)
    for start, value in zip(starts, trapped, strict=True):
        assert model.trapped_saturation(float(start)) == pytest.approx(float(value))


def test_a_land_model_with_no_room_to_trap_traps_everything_it_can_up_to_the_maximum():
    from bores.blackoil.satfunc.relperm.hysteresis import LandTrapping

    model = LandTrapping.from_endpoints(maximum_trapped_saturation=0.5, maximum_saturation=0.5)
    assert model.coefficient == pytest.approx(0.0)
    assert model.trapped_saturation(0.3) == pytest.approx(0.3)
    assert model.trapped_saturation(0.9) == pytest.approx(0.5)

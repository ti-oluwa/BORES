"""Tests for the three-phase oil relative permeability mixing rules and their consumers."""

import numpy as np
import pytest

from bores.blackoil.satfunc.relperm.mixing_rules import (
    MixingRule,
    compute_central_difference_partial_derivatives,
    get_mixing_rule,
    list_mixing_rules,
    mixing_rule,
)
from bores.blackoil.satfunc.relperm.models import (
    BrooksCoreyRelPermTable,
    LETThreePhaseRelPermTable,
)
from bores.blackoil.satfunc.tables import as_three_phase_relperm_table
from bores.errors import ValidationError
from bores.types import Wettability

RULES = ["baker_linear_rule", "stone_I_rule", "stone_II_rule", "min_rule"]
ENDPOINTS = {
    "irreducible_water_saturation": 0.2,
    "residual_oil_saturation_water": 0.25,
    "residual_oil_saturation_gas": 0.15,
    "residual_gas_saturation": 0.05,
}
SWC = ENDPOINTS["irreducible_water_saturation"]
SORW = ENDPOINTS["residual_oil_saturation_water"]
SORG = ENDPOINTS["residual_oil_saturation_gas"]
NO_FLOORS = {
    "minimum_water_relperm": None,
    "minimum_oil_relperm": None,
    "minimum_gas_relperm": None,
}


def _normalized(saturation, low, high, exponent):
    return np.clip((saturation - low) / (high - low), 0.0, 1.0) ** exponent


def _rule_inputs(sw: float, sg: float, krocw: float) -> dict:
    """Corey-type two-phase curves: kro_w depends on Sw only and kro_g on Sg only."""
    return {
        "kro_w": krocw * _normalized(1.0 - sw, SORW, 1.0 - SWC, 2.5),
        "kro_g": krocw * _normalized(1.0 - SWC - sg, SORG, 1.0 - SWC, 2.0),
        "krw": 0.6 * _normalized(sw, SWC, 1.0 - SORW, 2.2),
        "krg": 0.9 * _normalized(sg, 0.05, 1.0 - SWC - SORG, 2.0),
        "kr_max": krocw,
        "water_saturation": sw,
        "oil_saturation": 1.0 - sw - sg,
        "gas_saturation": sg,
        "irreducible_water_saturation": SWC,
        "residual_oil_saturation_water": SORW,
        "residual_oil_saturation_gas": SORG,
    }


def test_registry_holds_exactly_the_supported_rules():
    assert sorted(list_mixing_rules()) == sorted(RULES)


def test_duplicate_registration_is_rejected():
    with pytest.raises(ValidationError, match="already registered"):
        mixing_rule(lambda **kwargs: 0.0, name="min_rule")


@pytest.mark.parametrize("krocw", [1.0, 0.8])
@pytest.mark.parametrize("rule_name", RULES)
class TestTwoPhaseLimits:
    def test_no_gas_gives_oil_water_kro(self, rule_name, krocw):
        rule = get_mixing_rule(rule_name)
        for sw in (0.3, 0.45, 0.6):
            inputs = _rule_inputs(sw, 0.0, krocw)
            assert float(rule(**inputs)) == pytest.approx(inputs["kro_w"], abs=1e-12)

    def test_connate_water_gives_gas_oil_kro(self, rule_name, krocw):
        rule = get_mixing_rule(rule_name)
        for sg in (0.1, 0.3, 0.5):
            inputs = _rule_inputs(SWC, sg, krocw)
            assert float(rule(**inputs)) == pytest.approx(inputs["kro_g"], abs=1e-12)

    def test_pure_oil_gives_the_oil_endpoint(self, rule_name, krocw):
        rule = get_mixing_rule(rule_name)
        assert float(rule(**_rule_inputs(SWC, 0.0, krocw))) == pytest.approx(krocw, abs=1e-12)

    def test_result_is_bounded_by_the_oil_endpoint(self, rule_name, krocw):
        rule = get_mixing_rule(rule_name)
        for sw in np.linspace(SWC, 0.75, 9):
            for sg in np.linspace(0.0, 0.8 - sw + SWC, 9):
                if 1.0 - sw - sg < 0.0:
                    continue
                value = float(rule(**_rule_inputs(sw, sg, krocw)))
                assert -1e-12 <= value <= krocw + 1e-12


@pytest.mark.parametrize("rule_name", RULES)
class TestRuleContract:
    def test_analytical_partials_match_central_differences(self, rule_name):
        """The partials are the exact gradient, even where the saturations do not sum to 1."""
        rule = get_mixing_rule(rule_name)
        rng = np.random.default_rng(1)
        for _ in range(200):
            inputs = _rule_inputs(rng.uniform(0.22, 0.7), rng.uniform(0.02, 0.6), 0.8)
            inputs["oil_saturation"] = rng.uniform(0.1, 0.6)
            inputs["kro_w"] = rng.uniform(0.02, 0.8)
            inputs["kro_g"] = rng.uniform(0.02, 0.8)
            analytical = rule.partial_derivatives(**inputs)
            numerical = compute_central_difference_partial_derivatives(
                rule=rule, **inputs, epsilon=1e-6
            )
            for name, value in analytical.items():
                assert float(value) == pytest.approx(float(numerical[name]), abs=1e-6)

    def test_positively_homogeneous_in_the_oil_curves(self, rule_name):
        rule = get_mixing_rule(rule_name)
        inputs = _rule_inputs(0.4, 0.2, 1.0)
        scaled = dict(inputs, kro_w=0.7 * inputs["kro_w"], kro_g=0.7 * inputs["kro_g"], kr_max=0.7)
        assert float(rule(**scaled)) == pytest.approx(0.7 * float(rule(**inputs)), abs=1e-12)

    def test_array_inputs_keep_shape_and_dtype(self, rule_name):
        rule = get_mixing_rule(rule_name)
        sw = np.array([0.3, 0.4, 0.5], dtype=np.float32)
        sg = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        inputs = {
            key: (
                np.array(
                    [_rule_inputs(a, b, 1.0)[key] for a, b in zip(sw, sg, strict=True)],
                    dtype=np.float32,
                )
            )
            for key in ("kro_w", "kro_g", "krw", "krg", "oil_saturation")
        }
        inputs.update(
            kr_max=1.0,
            water_saturation=sw,
            gas_saturation=sg,
            irreducible_water_saturation=SWC,
            residual_oil_saturation_water=SORW,
            residual_oil_saturation_gas=SORG,
        )
        result = rule(**inputs)
        assert result.shape == (3,)
        assert result.dtype == np.float32

    def test_endpoints_may_vary_per_cell(self, rule_name):
        rule = get_mixing_rule(rule_name)
        swc = np.array([0.15, 0.2, 0.25])
        inputs = _rule_inputs(0.45, 0.15, 1.0)
        inputs["irreducible_water_saturation"] = swc
        # A rule that does not use the endpoint returns a single value
        result = np.broadcast_to(rule(**inputs), swc.shape)
        for cell, value in enumerate(swc):
            single = dict(inputs, irreducible_water_saturation=float(value))
            assert float(result[cell]) == pytest.approx(float(rule(**single)), abs=1e-12)


@pytest.mark.parametrize("rule_name", RULES)
class TestInputHandling:
    def test_scalar_inputs_give_a_scalar(self, rule_name):
        rule = get_mixing_rule(rule_name)
        result = rule(**_rule_inputs(0.4, 0.2, 1.0))
        assert np.ndim(result) == 0

    def test_multidimensional_inputs_keep_their_shape(self, rule_name):
        rule = get_mixing_rule(rule_name)
        sw = np.linspace(0.3, 0.5, 6).reshape(2, 3)
        sg = np.linspace(0.1, 0.25, 6).reshape(2, 3)
        inputs = {
            key: np.array([
                [_rule_inputs(a, b, 1.0)[key] for a, b in zip(ra, rb, strict=True)]
                for ra, rb in zip(sw, sg, strict=True)
            ])
            for key in ("kro_w", "kro_g", "krw", "krg", "oil_saturation")
        }
        inputs.update(
            kr_max=1.0,
            water_saturation=sw,
            gas_saturation=sg,
            irreducible_water_saturation=SWC,
            residual_oil_saturation_water=SORW,
            residual_oil_saturation_gas=SORG,
        )
        assert rule(**inputs).shape == (2, 3)
        for derivative in rule.partial_derivatives(**inputs).values():
            assert np.asarray(derivative).shape == (2, 3)


class TestRuleFormulas:
    def test_baker_is_the_eclipse_saturation_weighted_interpolation(self):
        inputs = _rule_inputs(0.4, 0.2, 1.0)
        mobile_water = 0.4 - SWC
        expected = (0.2 * inputs["kro_g"] + mobile_water * inputs["kro_w"]) / (0.2 + mobile_water)
        assert float(get_mixing_rule("baker_linear_rule")(**inputs)) == pytest.approx(expected)

    def test_stone_I_matches_the_aziz_settari_form(self):
        sw, sg = 0.4, 0.2
        inputs = _rule_inputs(sw, sg, 1.0)
        so = 1.0 - sw - sg
        alpha = 1.0 - sg / (1.0 - SWC - SORG)
        som = alpha * SORW + (1.0 - alpha) * SORG
        mobile = 1.0 - SWC - som
        so_star, sw_star, sg_star = (so - som) / mobile, (sw - SWC) / mobile, sg / mobile
        beta_w = inputs["kro_w"] / (1.0 - sw_star)
        beta_g = inputs["kro_g"] / (1.0 - sg_star)
        expected = so_star * beta_w * beta_g
        assert float(get_mixing_rule("stone_I_rule")(**inputs)) == pytest.approx(expected)

    def test_stone_I_is_zero_at_or_below_the_minimum_oil_saturation(self):
        inputs = _rule_inputs(0.55, 0.3, 1.0)
        inputs["oil_saturation"] = 0.1  # Som is above this
        assert float(get_mixing_rule("stone_I_rule")(**inputs)) == pytest.approx(0.0, abs=1e-15)

    def test_stone_II_matches_the_normalized_form(self):
        inputs = _rule_inputs(0.4, 0.2, 0.8)
        krocw = 0.8
        expected = krocw * (
            (inputs["kro_w"] / krocw + inputs["krw"]) * (inputs["kro_g"] / krocw + inputs["krg"])
            - inputs["krw"]
            - inputs["krg"]
        )
        assert float(get_mixing_rule("stone_II_rule")(**inputs)) == pytest.approx(expected)

    def test_min_rule_uses_the_subgradient_at_a_tie(self):
        inputs = _rule_inputs(0.4, 0.2, 1.0)
        inputs["kro_g"] = inputs["kro_w"]
        derivatives = get_mixing_rule("min_rule").partial_derivatives(**inputs)
        assert float(derivatives["dkro_dkro_w"]) == pytest.approx(0.5)
        assert float(derivatives["dkro_dkro_g"]) == pytest.approx(0.5)

    def test_a_rule_without_an_analytical_derivative_falls_back_to_finite_differences(self):
        def weighted(kro_w, kro_g, **kwargs):
            return 0.3 * np.asarray(kro_w) + 0.7 * np.asarray(kro_g)

        rule = MixingRule(func=weighted)  # type: ignore[arg-type]
        derivatives = rule.partial_derivatives(**_rule_inputs(0.4, 0.2, 1.0))
        assert float(derivatives["dkro_dkro_w"]) == pytest.approx(0.3, abs=1e-6)
        assert float(derivatives["dkro_dkro_g"]) == pytest.approx(0.7, abs=1e-6)
        assert float(derivatives["dkro_dsw_explicit"]) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def _model(kind: str, wettability: Wettability, rule_name: str):
    options = dict(
        **ENDPOINTS,
        **NO_FLOORS,
        wettability=wettability,
        mixing_rule=rule_name,
        maximum_oil_relperm=0.8,
        maximum_water_relperm=0.7,
        maximum_gas_relperm=0.9,
    )
    return (BrooksCoreyRelPermTable if kind == "brooks_corey" else LETThreePhaseRelPermTable)(
        **options
    )


def _points(count: int = 25, seed: int = 3):
    rng = np.random.default_rng(seed)
    points = []
    while len(points) < count:
        sw, sg = rng.uniform(0.24, 0.66), rng.uniform(0.07, 0.55)
        if 1.0 - sw - sg > 0.08:
            points.append((sw, 1.0 - sw - sg, sg))
    return points


def _values(table, sw, so, sg) -> np.ndarray:
    result = table.evaluate(sw, so, sg)
    return np.array([float(result["water"]), float(result["oil"]), float(result["gas"])])


def _assert_derivatives_match_finite_differences(table, points, step=1e-6, tolerance=1e-6):
    """The solver uses `dkr_dsw - dkr_dso` and `dkr_dsg - dkr_dso`."""
    for sw, so, sg in points:
        d = table.derivatives(sw, so, sg)
        along_sw = np.array(
            [
                d["dkrw_dsw"] - d["dkrw_dso"],
                d["dkro_dsw"] - d["dkro_dso"],
                d["dkrg_dsw"] - d["dkrg_dso"],
            ],
            dtype=float,
        )
        along_sg = np.array(
            [
                d["dkrw_dsg"] - d["dkrw_dso"],
                d["dkro_dsg"] - d["dkro_dso"],
                d["dkrg_dsg"] - d["dkrg_dso"],
            ],
            dtype=float,
        )
        fd_sw = (
            _values(table, sw + step, so - step, sg) - _values(table, sw - step, so + step, sg)
        ) / (2 * step)
        fd_sg = (
            _values(table, sw, so - step, sg + step) - _values(table, sw, so + step, sg - step)
        ) / (2 * step)
        np.testing.assert_allclose(along_sw, fd_sw, atol=tolerance)
        np.testing.assert_allclose(along_sg, fd_sg, atol=tolerance)


@pytest.mark.parametrize("rule_name", RULES)
@pytest.mark.parametrize("wettability", list(Wettability), ids=lambda w: w.name)
@pytest.mark.parametrize("kind", ["brooks_corey", "let"])
def test_model_derivatives_match_finite_differences(kind, wettability, rule_name):
    _assert_derivatives_match_finite_differences(_model(kind, wettability, rule_name), _points())


@pytest.mark.parametrize("rule_name", RULES)
@pytest.mark.parametrize("kind", ["brooks_corey", "let"])
def test_water_wet_model_matches_its_tabular_conversion(kind, rule_name):
    """Same two-phase curves and same rule, so the two representations must agree."""
    model = _model(kind, Wettability.WATER_WET, rule_name)
    table = as_three_phase_relperm_table(
        model, n_points=400, n_endpoint_extra=0, mixing_rule=rule_name, dtype=np.float64
    )
    for sw, so, sg in _points():
        np.testing.assert_allclose(
            _values(table, sw, so, sg), _values(model, sw, so, sg), atol=1e-6
        )


@pytest.mark.parametrize("kind", ["brooks_corey", "let"])
def test_pure_oil_gives_the_oil_endpoint(kind):
    model = _model(kind, Wettability.WATER_WET, "baker_linear_rule")
    assert float(model.evaluate(SWC, 1.0 - SWC, 0.0)["oil"]) == pytest.approx(0.8)


# --------------------------------------------------------------------------------------
# Tabular tables
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("rule_name", RULES)
def test_tabular_derivatives_match_finite_differences(rule_name):
    table = as_three_phase_relperm_table(
        _model("brooks_corey", Wettability.WATER_WET, rule_name),
        n_points=400,
        n_endpoint_extra=0,
        mixing_rule=rule_name,
        dtype=np.float64,
    )
    _assert_derivatives_match_finite_differences(table, _points(), tolerance=1e-5)


@pytest.mark.parametrize("rule_name", ["baker_linear_rule", "stone_II_rule"])
def test_tabular_orientation_does_not_change_three_phase_results(rule_name):
    """
    A two-phase system is queried at its own saturations whatever its reference axis,
    so So-indexed oil-water and Sw-indexed gas-oil tables must agree with the defaults.
    """
    model = _model("let", Wettability.WATER_WET, rule_name)
    default = as_three_phase_relperm_table(
        model, n_points=500, n_endpoint_extra=0, mixing_rule=rule_name, dtype=np.float64
    )
    reoriented = as_three_phase_relperm_table(
        model,
        n_points=500,
        n_endpoint_extra=0,
        mixing_rule=rule_name,
        dtype=np.float64,
        oil_water_reference_phase="non_wetting",
        gas_oil_reference_phase="wetting",
    )
    assert reoriented.oil_water_table.reference_phase == "non_wetting"
    assert reoriented.gas_oil_table.reference_phase == "wetting"
    for sw, so, sg in _points():
        np.testing.assert_allclose(
            _values(reoriented, sw, so, sg), _values(default, sw, so, sg), atol=1e-4
        )
    _assert_derivatives_match_finite_differences(reoriented, _points(), tolerance=1e-4)


# --------------------------------------------------------------------------------------
# Hysteresis
# --------------------------------------------------------------------------------------
def _hysteresis_pair(rule_name: str, with_imbibition: bool = False):
    from bores.blackoil.satfunc.relperm.hysteresis import KilloughLandRelPermTable

    base = as_three_phase_relperm_table(
        _model("let", Wettability.WATER_WET, rule_name),
        n_points=400,
        n_endpoint_extra=0,
        mixing_rule=rule_name,
        dtype=np.float64,
    )
    options = {}
    if with_imbibition:
        options = {
            "oil_water_imbibition_table": base.oil_water_table,
            "gas_oil_imbibition_table": base.gas_oil_table,
        }
    table = KilloughLandRelPermTable(
        oil_water_drainage_table=base.oil_water_table,
        gas_oil_drainage_table=base.gas_oil_table,
        mixing_rule=rule_name,
        **options,
    )
    return base, table


@pytest.mark.parametrize("rule_name", RULES)
def test_hysteresis_drainage_matches_the_tabular_table(rule_name):
    base, table = _hysteresis_pair(rule_name)
    for point in _points():
        np.testing.assert_allclose(_values(table, *point), _values(base, *point), atol=1e-12)
    _assert_derivatives_match_finite_differences(table, _points(15), tolerance=1e-5)


def test_hysteresis_endpoints_match_the_tabular_table():
    base, table = _hysteresis_pair("stone_I_rule")
    assert table.get_connate_water_saturation() == pytest.approx(SWC)
    assert table.get_residual_oil_saturation_water() == pytest.approx(SORW)
    # The gas-oil table alone cannot know swc, so this depends on the shared resolution
    assert table.get_residual_oil_saturation_gas() == pytest.approx(SORG)
    assert table.get_residual_oil_saturation_gas() == pytest.approx(
        base.get_residual_oil_saturation_gas()
    )


@pytest.mark.parametrize("rule_name", ["baker_linear_rule", "stone_II_rule"])
def test_hysteresis_scanning_between_identical_curves_reproduces_the_curves(rule_name):
    """Boolean imbibition flags must work, and identical drainage/imbibition curves give the same kr."""
    base, table = _hysteresis_pair(rule_name, with_imbibition=True)
    sw = np.array([0.35, 0.45, 0.55])
    sg = np.array([0.15, 0.2, 0.1])
    so = 1.0 - sw - sg
    result = table.evaluate(
        sw,
        so,
        sg,
        max_water_saturation=sw + 0.05,
        max_gas_saturation=sg + 0.05,
        water_imbibition_flag=np.array([True, False, True]),
        gas_imbibition_flag=np.array([False, True, True]),
        water_reversal_saturation=sw + 0.05,
        gas_reversal_saturation=sg + 0.05,
    )
    expected = base.evaluate(sw, so, sg)
    for phase in ("water", "oil", "gas"):
        np.testing.assert_allclose(result[phase], expected[phase], atol=1e-9)

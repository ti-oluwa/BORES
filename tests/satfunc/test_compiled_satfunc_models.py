"""Tests for the compiled relative permeability models and their consistency guarantees."""

import inspect

import numpy as np
import pytest

from bores.blackoil.satfunc.capillary_pressure.models import (
    BrooksCoreyCapillaryPressureTable,
    LeverettJCapillaryPressureTable,
    VanGenuchtenCapillaryPressureTable,
)
from bores.blackoil.satfunc.relperm.mixing_rules import (
    MixingRule,
    baker_linear_rule,
    get_builtin_rule_id,
    min_rule,
    stone_I_rule,
    stone_II_rule,
)
from bores.blackoil.satfunc.relperm.models import (
    BrooksCoreyRelPermTable,
    LETParameters,
    LETThreePhaseRelPermTable,
)
from bores.errors import ValidationError
from bores.types import Wettability

ENDPOINTS = dict(
    irreducible_water_saturation=0.2,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.05,
)
RULES = [min_rule, baker_linear_rule, stone_II_rule, stone_I_rule]


def make_table(model: str, **kwargs):
    if model == "brooks_corey":
        return BrooksCoreyRelPermTable(
            water_exponent=2.3, oil_exponent=1.7, gas_exponent=3.1, **ENDPOINTS, **kwargs
        )
    return LETThreePhaseRelPermTable(
        water=LETParameters(L=2.1, E=1.3, T=2.4),
        oil_water=LETParameters(L=3.0, E=0.9, T=1.8),
        gas_oil=LETParameters(L=1.6, E=1.7, T=2.9),
        gas=LETParameters(L=2.7, E=1.1, T=2.2),
        **ENDPOINTS,
        **kwargs,
    )


@pytest.fixture(scope="module")
def saturations():
    rng = np.random.default_rng(11)
    p = rng.dirichlet([2.0, 2.0, 1.0], size=300)
    return p[:, 0].copy(), p[:, 1].copy(), p[:, 2].copy()


@pytest.mark.parametrize("model", ["brooks_corey", "let"])
@pytest.mark.parametrize("wettability", list(Wettability))
@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.name if hasattr(r, "name") else str(r))
def test_fused_call_equals_separate_calls(model, wettability, rule, saturations):
    table = make_table(
        model, wettability=wettability, mixing_rule=rule, mixed_wet_water_fraction=0.4
    )
    relperm, derivatives = table.evaluate_with_derivatives(*saturations)
    expected_relperm = table.evaluate(*saturations)
    expected_derivatives = table.derivatives(*saturations)
    for key in expected_relperm:
        np.testing.assert_array_equal(relperm[key], expected_relperm[key])
    for key in expected_derivatives:
        np.testing.assert_array_equal(derivatives[key], expected_derivatives[key])


@pytest.mark.parametrize("model", ["brooks_corey", "let"])
def test_scalar_and_array_calls_agree(model, saturations):
    table = make_table(model, wettability=Wettability.MIXED_WET, mixing_rule=stone_II_rule)
    sw, so, sg = saturations
    array_relperm = table.evaluate(sw, so, sg)
    array_derivatives = table.derivatives(sw, so, sg)
    for i in (0, 7, 123):
        scalar_relperm = table.evaluate(float(sw[i]), float(so[i]), float(sg[i]))
        scalar_derivatives = table.derivatives(float(sw[i]), float(so[i]), float(sg[i]))
        assert isinstance(scalar_relperm["oil"], float)
        for key in array_relperm:
            assert scalar_relperm[key] == pytest.approx(array_relperm[key][i], rel=1e-12)
        for key in array_derivatives:
            assert scalar_derivatives[key] == pytest.approx(
                array_derivatives[key][i], rel=1e-12, abs=1e-14
            )


@pytest.mark.parametrize("model", ["brooks_corey", "let"])
def test_float32_input_gives_float32_output(model, saturations):
    table = make_table(model, mixing_rule=stone_II_rule)
    sw, so, sg = (s.astype(np.float32) for s in saturations)
    assert table.evaluate(sw, so, sg)["oil"].dtype == np.float32
    assert table.derivatives(sw, so, sg)["dkro_dso"].dtype == np.float32


def test_builtin_rules_are_identified_and_modified_ones_are_not():
    ids = [get_builtin_rule_id(rule) for rule in RULES]
    assert None not in ids
    assert len(set(ids)) == len(RULES)
    assert get_builtin_rule_id(MixingRule(func=baker_linear_rule.func)) is None
    original = stone_II_rule._dfunc
    try:
        stone_II_rule._dfunc = lambda **kwargs: None
        assert get_builtin_rule_id(stone_II_rule) is None
    finally:
        stone_II_rule._dfunc = original
    assert get_builtin_rule_id(stone_II_rule) is not None


@pytest.mark.parametrize("model", ["brooks_corey", "let"])
def test_custom_rule_takes_the_python_path_and_agrees_on_values(model, saturations):
    custom = MixingRule(func=baker_linear_rule.func)
    builtin_table = make_table(model, mixing_rule=baker_linear_rule)
    custom_table = make_table(model, mixing_rule=custom)
    expected = builtin_table.evaluate(*saturations)
    actual = custom_table.evaluate(*saturations)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
    fused_relperm, _ = custom_table.evaluate_with_derivatives(*saturations)
    for key in expected:
        np.testing.assert_array_equal(fused_relperm[key], actual[key])


@pytest.mark.parametrize("model", ["brooks_corey", "let"])
@pytest.mark.parametrize("wettability", list(Wettability))
def test_derivatives_are_evaluated_at_the_normalised_saturations(model, wettability):
    table = make_table(model, wettability=wettability, mixing_rule=stone_II_rule)
    sw, so, sg = np.array([0.3, 0.25]), np.array([0.4, 0.3]), np.array([0.2, 0.3])
    total = sw + so + sg
    assert np.all(np.abs(total - 1.0) > 0.05)
    raw = table.derivatives(sw, so, sg)
    normalised = table.derivatives(sw / total, so / total, sg / total)
    for key in raw:
        np.testing.assert_allclose(raw[key], normalised[key], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("model", ["brooks_corey", "let"])
@pytest.mark.parametrize("method", ["evaluate", "derivatives"])
def test_out_of_range_saturation_raises_in_both_paths(model, method):
    table = make_table(model, mixing_rule=stone_II_rule)
    with pytest.raises(ValidationError, match="between 0 and 1"):
        getattr(table, method)(np.array([1.2]), np.array([0.1]), np.array([0.1]))


@pytest.mark.parametrize("model", ["brooks_corey", "let"])
def test_unknown_wettability_raises_in_the_derivative_path(model, saturations):
    table = make_table(model, mixing_rule=stone_II_rule)
    object.__setattr__(table, "wettability", "neutral")
    with pytest.raises(ValidationError, match="not implemented"):
        table.derivatives(*saturations)


@pytest.mark.parametrize("model", ["brooks_corey", "let"])
def test_degenerate_mobile_range_is_consistent_between_value_and_derivative(model):
    degenerate = dict(
        irreducible_water_saturation=0.6,
        residual_oil_saturation_water=0.4,
        residual_oil_saturation_gas=0.1,
        residual_gas_saturation=0.05,
    )
    kwargs = dict(wettability=Wettability.WATER_WET, mixing_rule=baker_linear_rule, **degenerate)
    if model == "brooks_corey":
        table = BrooksCoreyRelPermTable(
            water_exponent=2.0, oil_exponent=2.0, gas_exponent=2.0, **kwargs
        )
        python_table = BrooksCoreyRelPermTable(
            water_exponent=2.0,
            oil_exponent=2.0,
            gas_exponent=2.0,
            **{
                **kwargs,
                "mixing_rule": MixingRule(
                    func=baker_linear_rule.func, dfunc=baker_linear_rule._dfunc
                ),
            },
        )
    else:
        table = LETThreePhaseRelPermTable(**kwargs)
        python_table = LETThreePhaseRelPermTable(**{
            **kwargs,
            "mixing_rule": MixingRule(func=baker_linear_rule.func, dfunc=baker_linear_rule._dfunc),
        })
    sw, so = np.array([0.7, 0.8]), np.array([0.2, 0.1])
    sg = 1.0 - sw - so
    compiled = table.derivatives(sw, so, sg)
    python = python_table.derivatives(sw, so, sg)
    for key in compiled:
        np.testing.assert_allclose(compiled[key], python[key], atol=1e-9)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(maximum_water_relperm=0.0), "maximum_water_relperm"),
        (dict(maximum_oil_relperm=1.2), "maximum_oil_relperm"),
        (dict(mixed_wet_water_fraction=1.5), "mixed_wet_water_fraction"),
        (dict(irreducible_water_saturation=1.3), "irreducible_water_saturation"),
    ],
)
@pytest.mark.parametrize("model", ["brooks_corey", "let"])
def test_invalid_table_parameters_are_rejected(model, overrides, message):
    with pytest.raises(ValidationError, match=message):
        if model == "brooks_corey":
            BrooksCoreyRelPermTable(**overrides)
        else:
            LETThreePhaseRelPermTable(**overrides)


@pytest.mark.parametrize("name", ["water_exponent", "oil_exponent", "gas_exponent"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_non_positive_corey_exponents_are_rejected(name, value):
    with pytest.raises(ValidationError, match=name):
        BrooksCoreyRelPermTable(**{name: value})


@pytest.mark.parametrize(
    "table_class",
    [
        BrooksCoreyCapillaryPressureTable,
        VanGenuchtenCapillaryPressureTable,
        LeverettJCapillaryPressureTable,
    ],
)
@pytest.mark.parametrize("wettability", list(Wettability))
def test_capillary_derivatives_are_evaluated_at_the_normalised_saturations(
    table_class, wettability
):
    table = table_class(wettability=wettability, **ENDPOINTS)
    sw, so, sg = np.array([0.3, 0.25]), np.array([0.4, 0.3]), np.array([0.2, 0.3])
    total = sw + so + sg
    raw = table.derivatives(sw, so, sg)
    normalised = table.derivatives(sw / total, so / total, sg / total)
    for key in raw:
        np.testing.assert_allclose(raw[key], normalised[key], rtol=1e-12, atol=1e-14)


def test_leverett_j_functions_take_the_unit_conversion_without_a_default():
    from bores.blackoil.satfunc.capillary_pressure.models import leverett_j

    checked = 0
    for name in dir(leverett_j):
        function = getattr(leverett_j, name)
        if not name.startswith("compute_leverett_j") or not callable(function):
            continue
        parameters = inspect.signature(getattr(function, "py_func", function)).parameters
        if "dyne_per_cm_to_pressure_unit" in parameters:
            default = parameters["dyne_per_cm_to_pressure_unit"].default
            assert default is inspect.Parameter.empty, name
            checked += 1
    assert checked >= 8


@pytest.mark.parametrize(
    "unit_system", list(__import__("bores.units", fromlist=["UnitSystem"]).UnitSystem)
)
def test_leverett_j_table_uses_the_conversion_of_its_unit_system(unit_system):
    from bores.blackoil.satfunc.capillary_pressure.models.leverett_j import (
        compute_leverett_j_capillary_pressures,
        convert_dyne_per_cm_to_pressure_unit,
    )
    from bores.constants import c

    factor = convert_dyne_per_cm_to_pressure_unit(unit_system)
    assert factor != 4.725e-4
    if unit_system.name == "FIELD":
        assert factor == c.DYNE_PER_CENTIMETER_TO_PSI
    table = LeverettJCapillaryPressureTable(
        wettability=Wettability.WATER_WET, unit_system=unit_system, **ENDPOINTS
    )
    sw, so, sg = np.array([0.35]), np.array([0.5]), np.array([0.15])
    from_table = table.evaluate(sw, so, sg)["oil_water"]
    doubled = compute_leverett_j_capillary_pressures(
        water_saturation=sw,
        oil_saturation=so,
        gas_saturation=sg,
        **ENDPOINTS,
        wettability=Wettability.WATER_WET,
        permeability=table.permeability,
        porosity=table.porosity,
        oil_water_interfacial_tension=table.oil_water_interfacial_tension,
        gas_oil_interfacial_tension=table.gas_oil_interfacial_tension,
        dyne_per_cm_to_pressure_unit=2.0 * factor,
        saturation_epsilon=c.SATURATION_EPSILON,
        minimum_mobile_pore_space=c.MINIMUM_MOBILE_PORE_SPACE,
    )[0]
    np.testing.assert_allclose(doubled, 2.0 * from_table, rtol=1e-12)

"""Tests for the three-phase oil relative permeability mixing rules."""

import numpy as np
import pytest

from bores.blackoil.satfunc.relperm.mixing_rules import list_mixing_rules
from bores.blackoil.satfunc.relperm.models import BrooksCoreyRelPermTable
from bores.blackoil.satfunc.tables import as_three_phase_relperm_table

ENDPOINTS = {
    "irreducible_water_saturation": 0.2,
    "residual_oil_saturation_water": 0.25,
    "residual_oil_saturation_gas": 0.15,
    "residual_gas_saturation": 0.05,
}
CONNATE_WATER = ENDPOINTS["irreducible_water_saturation"]

# Rules built only from the two two-phase oil kr values (and krw, krg, krocw) that must
# reduce to the two-phase curves at the limits
TWO_PHASE_CONSISTENT_RULES = ["min_rule", "stone_I_rule", "stone_II_rule", "hustad_hansen_rule"]


def _table(rule: str, krocw: float):
    source = BrooksCoreyRelPermTable(**ENDPOINTS, maximum_oil_relperm=krocw)
    return as_three_phase_relperm_table(
        source, n_points=300, n_endpoint_extra=0, mixing_rule=rule, dtype=np.float64
    )


def _kro(table, sw, so, sg) -> float:
    return float(table.evaluate(sw, so, sg)["oil"])


@pytest.mark.parametrize("krocw", [1.0, 0.8])
@pytest.mark.parametrize("rule", TWO_PHASE_CONSISTENT_RULES)
class TestTwoPhaseLimits:
    def test_no_gas_gives_oil_water_kro(self, rule, krocw):
        table = _table(rule, krocw)
        for sw in (0.3, 0.45, 0.6):
            so = 1.0 - sw
            expected = float(
                table.oil_water_table.get_non_wetting_phase_relative_permeability(
                    sw, non_wetting_saturation=so
                )
            )
            assert _kro(table, sw, so, 0.0) == pytest.approx(expected, abs=1e-6)

    def test_connate_water_gives_gas_oil_kro(self, rule, krocw):
        table = _table(rule, krocw)
        for sg in (0.1, 0.3, 0.5):
            so = 1.0 - CONNATE_WATER - sg
            expected = float(
                table.gas_oil_table.get_wetting_phase_relative_permeability(
                    so, non_wetting_saturation=sg
                )
            )
            assert _kro(table, CONNATE_WATER, so, sg) == pytest.approx(expected, abs=1e-6)

    def test_pure_oil_gives_the_oil_endpoint(self, rule, krocw):
        table = _table(rule, krocw)
        assert _kro(table, CONNATE_WATER, 1.0 - CONNATE_WATER, 0.0) == pytest.approx(
            krocw, abs=1e-6
        )


@pytest.mark.parametrize("rule", list_mixing_rules())
def test_analytic_derivatives_match_finite_differences(rule):
    """The solver uses `dkro_dsw - dkro_dso` and `dkro_dsg - dkro_dso`."""
    table = _table(rule, krocw=0.8)
    rng = np.random.default_rng(0)
    points = []
    while len(points) < 30:
        sw, sg = rng.uniform(0.22, 0.7), rng.uniform(0.02, 0.6)
        so = 1.0 - sw - sg
        if so > 0.05:
            points.append((sw, so, sg))

    step = 1e-6
    for sw, so, sg in points:
        derivatives = table.derivatives(sw, so, sg)
        along_sw = (
            _kro(table, sw + step, so - step, sg) - _kro(table, sw - step, so + step, sg)
        ) / (2 * step)
        along_sg = (
            _kro(table, sw, so - step, sg + step) - _kro(table, sw, so + step, sg - step)
        ) / (2 * step)
        assert float(derivatives["dkro_dsw"]) - float(derivatives["dkro_dso"]) == pytest.approx(
            along_sw, abs=1e-6
        )
        assert float(derivatives["dkro_dsg"]) - float(derivatives["dkro_dso"]) == pytest.approx(
            along_sg, abs=1e-6
        )

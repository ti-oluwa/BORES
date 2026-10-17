"""Tests for resampling tabular three-phase relperm and capillary pressure tables."""

import attrs
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from bores.blackoil.satfunc.capillary_pressure.models import BrooksCoreyCapillaryPressureTable
from bores.blackoil.satfunc.relperm.models import BrooksCoreyRelPermTable
from bores.blackoil.satfunc.tables import (
    as_three_phase_capillary_pressure_table,
    as_three_phase_relperm_table,
    sample_oil_water_relative_permeabilities,
)
from bores.blackoil.satfunc.utils import pchip_resample
from bores.errors import ValidationError
from bores.types import FluidPhase, UnitSystem

ENDPOINTS = {
    "irreducible_water_saturation": 0.2,
    "residual_oil_saturation_water": 0.25,
    "residual_oil_saturation_gas": 0.15,
    "residual_gas_saturation": 0.05,
}


def _relperm_source(**kwargs):
    kwargs.setdefault("n_points", 60)
    kwargs.setdefault("n_endpoint_extra", 0)
    return as_three_phase_relperm_table(
        BrooksCoreyRelPermTable(**ENDPOINTS), dtype=np.float64, **kwargs
    )


def _capillary_source(**kwargs):
    kwargs.setdefault("n_points", 60)
    kwargs.setdefault("n_endpoint_extra", 0)
    return as_three_phase_capillary_pressure_table(
        BrooksCoreyCapillaryPressureTable(**ENDPOINTS), dtype=np.float64, **kwargs
    )


def _count_near_start(axis, span):
    """Number of knots in the first 10 % of the range."""
    return int(np.sum(axis < axis[0] + 0.1 * span))


def _assert_relperm_matches_source(result, source):
    """Resampled arrays equal an independent PCHIP of the source evaluated on the new axis."""
    for name in ("oil_water_table", "gas_oil_table"):
        new, old = getattr(result, name), getattr(source, name)
        axis = np.clip(
            new.reference_saturation, old.reference_saturation[0], old.reference_saturation[-1]
        )
        for attribute in (
            "wetting_phase_relative_permeability",
            "non_wetting_phase_relative_permeability",
        ):
            expected = PchipInterpolator(old.reference_saturation, getattr(old, attribute))(axis)
            np.testing.assert_allclose(getattr(new, attribute), expected, atol=1e-9)


def _assert_capillary_matches_source(result, source):
    for name in ("oil_water_table", "gas_oil_table"):
        new, old = getattr(result, name), getattr(source, name)
        axis = np.clip(
            new.reference_saturation, old.reference_saturation[0], old.reference_saturation[-1]
        )
        expected = PchipInterpolator(old.reference_saturation, old.capillary_pressure)(axis)
        np.testing.assert_allclose(new.capillary_pressure, expected, rtol=1e-9)


class TestPchipResample:
    def test_target_saturations_are_used_and_clipped_for_evaluation(self):
        knots = np.array([0.2, 0.4, 0.6, 0.8])
        values = knots**2
        targets = np.array([0.1, 0.3, 0.9])
        axis, resampled = pchip_resample(knots, values, target_saturations=targets)
        np.testing.assert_array_equal(axis, targets)
        # Outside the source range the curve is held at the nearest end value
        assert resampled[0] == pytest.approx(values[0])
        assert resampled[2] == pytest.approx(values[-1])

    def test_requires_a_grid(self):
        with pytest.raises(ValueError, match="number_of_output_points"):
            pchip_resample(np.array([0.0, 1.0]), np.array([0.0, 1.0]))


class TestTabularRelPerm:
    def test_needs_no_saturation_endpoints(self):
        source = _relperm_source()
        result = as_three_phase_relperm_table(source, n_points=80, n_endpoint_extra=0)
        assert len(result.oil_water_table.reference_saturation) == 80

    def test_axis_spans_source_range_and_matches_source_values(self):
        source = _relperm_source()
        result = as_three_phase_relperm_table(
            source, n_points=120, n_endpoint_extra=0, dtype=np.float64
        )
        for name in ("oil_water_table", "gas_oil_table"):
            new = getattr(result, name).reference_saturation
            old = getattr(source, name).reference_saturation
            assert new[0] == pytest.approx(old[0])
            assert new[-1] == pytest.approx(old[-1])
        _assert_relperm_matches_source(result, source)

    def test_endpoint_enrichment_is_honored(self):
        source = _relperm_source()
        plain = as_three_phase_relperm_table(
            source, n_points=60, n_endpoint_extra=0, dtype=np.float64
        )
        enriched = as_three_phase_relperm_table(
            source, n_points=60, n_endpoint_extra=20, dtype=np.float64
        )
        plain_axis = plain.oil_water_table.reference_saturation
        enriched_axis = enriched.oil_water_table.reference_saturation
        assert len(enriched_axis) > len(plain_axis)
        span = plain_axis[-1] - plain_axis[0]
        assert _count_near_start(enriched_axis, span) > _count_near_start(plain_axis, span)
        _assert_relperm_matches_source(enriched, source)

    def test_custom_axes_are_honored(self):
        source = _relperm_source()
        oil_water_axis = np.linspace(0.2, 0.75, 11)
        gas_oil_axis = np.linspace(0.05, 0.65, 7)
        result = as_three_phase_relperm_table(
            source,
            oil_water_reference_saturation=oil_water_axis,
            gas_oil_reference_saturation=gas_oil_axis,
            dtype=np.float64,
        )
        np.testing.assert_allclose(result.oil_water_table.reference_saturation, oil_water_axis)
        np.testing.assert_allclose(result.gas_oil_table.reference_saturation, gas_oil_axis)
        _assert_relperm_matches_source(result, source)

    def test_orientation_defaults_to_the_source(self):
        source = _relperm_source(
            oil_water_reference_phase="non_wetting", gas_oil_reference_phase="wetting"
        )
        result = as_three_phase_relperm_table(
            source, n_points=70, n_endpoint_extra=0, dtype=np.float64
        )
        for name in ("oil_water_table", "gas_oil_table"):
            assert getattr(result, name).reference_phase == getattr(source, name).reference_phase
            assert getattr(result, name).wetting_phase == getattr(source, name).wetting_phase
        _assert_relperm_matches_source(result, source)

    def test_matching_explicit_orientation_is_accepted(self):
        source = _relperm_source(oil_water_reference_phase="non_wetting")
        as_three_phase_relperm_table(
            source,
            oil_water_reference_phase="non_wetting",
            oil_water_wetting_phase=source.oil_water_table.wetting_phase,
            gas_oil_reference_phase="non_wetting",
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"oil_water_reference_phase": "non_wetting"},
            {"gas_oil_reference_phase": "wetting"},
            {"oil_water_wetting_phase": FluidPhase.OIL},
            {"gas_oil_wetting_phase": FluidPhase.GAS},
        ],
    )
    def test_conflicting_explicit_orientation_raises(self, kwargs):
        source = _relperm_source()
        with pytest.raises(ValidationError, match="re-orient"):
            as_three_phase_relperm_table(source, **kwargs)

    def test_minimum_relperm_floors_are_carried_over(self):
        source = _relperm_source()
        floored_oil_water = attrs.evolve(
            source.oil_water_table, min_wetting_relperm=1e-3, min_non_wetting_relperm=2e-3
        )
        floored = attrs.evolve(source, oil_water_table=floored_oil_water)
        result = as_three_phase_relperm_table(floored, n_points=80, n_endpoint_extra=0)
        assert result.oil_water_table.min_wetting_relperm == pytest.approx(1e-3)
        assert result.oil_water_table.min_non_wetting_relperm == pytest.approx(2e-3)
        residual = result.oil_water_table.reference_saturation[0]
        assert (
            float(result.oil_water_table.get_wetting_phase_relative_permeability(residual)) >= 1e-3
        )

    def test_mixing_rule_is_kept(self):
        source = as_three_phase_relperm_table(
            BrooksCoreyRelPermTable(**ENDPOINTS),
            mixing_rule="stone_II_rule",
            n_points=40,
            n_endpoint_extra=0,
        )
        result = as_three_phase_relperm_table(source, n_points=50)
        assert result.mixing_rule is source.mixing_rule


class TestAnalyticRelPerm:
    def test_default_orientation_is_unchanged(self):
        result = _relperm_source()
        assert result.oil_water_table.reference_phase == "wetting"
        assert result.gas_oil_table.reference_phase == "non_wetting"

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_samplers_honor_dtype(self, dtype):
        axis = np.linspace(0.2, 0.75, 9)
        _, wetting_kr, non_wetting_kr = sample_oil_water_relative_permeabilities(
            relperm_table=BrooksCoreyRelPermTable(**ENDPOINTS),
            oil_water_reference_saturations=axis,
            sweep_axis_is_water_saturation=True,
            irreducible_water_saturation=0.2,
            residual_oil_saturation_water=0.25,
            model_kwargs=dict(ENDPOINTS),
            oil_water_wetting_phase=FluidPhase.WATER,
            dtype=dtype,
        )
        assert wetting_kr.dtype == dtype
        assert non_wetting_kr.dtype == dtype


class TestTabularCapillaryPressure:
    def test_needs_no_saturation_endpoints(self):
        source = _capillary_source()
        result = as_three_phase_capillary_pressure_table(source, n_points=80, n_endpoint_extra=0)
        assert len(result.oil_water_table.reference_saturation) == 80

    def test_values_match_source_and_enrichment_is_honored(self):
        source = _capillary_source()
        plain = as_three_phase_capillary_pressure_table(
            source, n_points=60, n_endpoint_extra=0, dtype=np.float64
        )
        enriched = as_three_phase_capillary_pressure_table(
            source, n_points=60, n_endpoint_extra=30, dtype=np.float64
        )
        assert len(enriched.gas_oil_table.reference_saturation) > len(
            plain.gas_oil_table.reference_saturation
        )
        _assert_capillary_matches_source(plain, source)
        _assert_capillary_matches_source(enriched, source)

    def test_custom_axes_are_honored(self):
        source = _capillary_source()
        axis = np.linspace(0.2, 0.75, 9)
        result = as_three_phase_capillary_pressure_table(
            source, oil_water_reference_saturation=axis, dtype=np.float64
        )
        np.testing.assert_allclose(result.oil_water_table.reference_saturation, axis)
        _assert_capillary_matches_source(result, source)

    def test_orientation_defaults_to_the_source(self):
        source = _capillary_source(
            oil_water_reference_phase="non_wetting", gas_oil_reference_phase="wetting"
        )
        result = as_three_phase_capillary_pressure_table(
            source, n_points=70, n_endpoint_extra=0, dtype=np.float64
        )
        for name in ("oil_water_table", "gas_oil_table"):
            assert getattr(result, name).reference_phase == getattr(source, name).reference_phase
        _assert_capillary_matches_source(result, source)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"oil_water_reference_phase": "non_wetting"},
            {"gas_oil_reference_phase": "wetting"},
            {"oil_water_wetting_phase": FluidPhase.OIL},
        ],
    )
    def test_conflicting_explicit_orientation_raises(self, kwargs):
        # Previously this silently relabeled the data
        source = _capillary_source()
        with pytest.raises(ValidationError, match="re-orient"):
            as_three_phase_capillary_pressure_table(source, **kwargs)

    def test_unit_conversion_is_applied_to_the_resampled_table(self):
        source = _capillary_source()
        assert source.unit_system == UnitSystem.FIELD
        same = as_three_phase_capillary_pressure_table(
            source, n_points=50, n_endpoint_extra=0, dtype=np.float64
        )
        converted = as_three_phase_capillary_pressure_table(
            source, n_points=50, n_endpoint_extra=0, dtype=np.float64, unit_system=UnitSystem.SI
        )
        assert converted.unit_system == UnitSystem.SI
        ratio = (
            converted.oil_water_table.capillary_pressure / same.oil_water_table.capillary_pressure
        )
        assert ratio[0] != pytest.approx(1.0)
        np.testing.assert_allclose(ratio, ratio[0])

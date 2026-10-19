"""Tests for the capillary pressure models and their tabular conversion."""

import numpy as np
import pytest

from bores.blackoil.satfunc.capillary_pressure.models import (
    BrooksCoreyCapillaryPressureTable,
    LeverettJCapillaryPressureTable,
    VanGenuchtenCapillaryPressureTable,
)
from bores.blackoil.satfunc.tables import as_three_phase_capillary_pressure_table
from bores.types import FluidPhase, Wettability

ENDPOINTS = {
    "irreducible_water_saturation": 0.2,
    "residual_oil_saturation_water": 0.25,
    "residual_oil_saturation_gas": 0.15,
    "residual_gas_saturation": 0.05,
}
SWC = ENDPOINTS["irreducible_water_saturation"]

# Distinct oil-wet parameters and an unbalanced blend, so the mixed-wet curves are not zero
MODEL_OPTIONS = {
    "brooks_corey": (
        BrooksCoreyCapillaryPressureTable,
        {
            "oil_water_entry_pressure_oil_wet": 3.0,
            "oil_water_pore_size_distribution_index_oil_wet": 1.5,
            "mixed_wet_water_fraction": 0.7,
        },
    ),
    "van_genuchten": (
        VanGenuchtenCapillaryPressureTable,
        {
            "oil_water_alpha_oil_wet": 0.02,
            "oil_water_n_oil_wet": 2.5,
            "mixed_wet_water_fraction": 0.7,
        },
    ),
    "leverett_j": (LeverettJCapillaryPressureTable, {"mixed_wet_water_fraction": 0.7}),
}


def _model(name: str, wettability: Wettability = Wettability.WATER_WET):
    cls, options = MODEL_OPTIONS[name]
    return cls(**ENDPOINTS, **options, wettability=wettability)


def _points(count: int = 25, seed: int = 5):
    rng = np.random.default_rng(seed)
    points = []
    while len(points) < count:
        sw, sg = rng.uniform(0.26, 0.64), rng.uniform(0.08, 0.5)
        if 1.0 - sw - sg > 0.1:
            points.append((sw, 1.0 - sw - sg, sg))
    return points


def _pressures(table, sw, so, sg) -> np.ndarray:
    result = table.evaluate(sw, so, sg)
    return np.array([float(result["oil_water"]), float(result["gas_oil"])])


def _assert_derivatives_match_finite_differences(table, points, step=1e-6, tolerance=1e-5):
    """
    The solver uses the derivatives with respect to the independent `Sw` and `Sg`:
    `dpcow_dsw - dpcow_dso`, `-dpcow_dso`, `-dpcgo_dso` and `dpcgo_dsg - dpcgo_dso`.
    """
    for sw, so, sg in points:
        d = {key: float(value) for key, value in table.derivatives(sw, so, sg).items()}
        analytical = np.array([
            d["dpcow_dsw"] - d["dpcow_dso"],
            -d["dpcow_dso"],
            -d["dpcgo_dso"],
            d["dpcgo_dsg"] - d["dpcgo_dso"],
        ])
        along_sw = (
            _pressures(table, sw + step, so - step, sg)
            - _pressures(table, sw - step, so + step, sg)
        ) / (2 * step)
        along_sg = (
            _pressures(table, sw, so - step, sg + step)
            - _pressures(table, sw, so + step, sg - step)
        ) / (2 * step)
        numerical = np.array([along_sw[0], along_sg[0], along_sw[1], along_sg[1]])
        error = np.abs(analytical - numerical) / np.maximum(1.0, np.abs(numerical))
        assert error.max() < tolerance, (
            f"at (Sw, So, Sg) = {(sw, so, sg)}: analytical {analytical}, numerical {numerical}"
        )


@pytest.mark.parametrize("wettability", list(Wettability), ids=lambda w: w.name)
@pytest.mark.parametrize("name", list(MODEL_OPTIONS))
def test_model_derivatives_match_finite_differences(name, wettability):
    _assert_derivatives_match_finite_differences(_model(name, wettability), _points())


def test_mixed_wet_curve_is_not_trivially_zero():
    for name in MODEL_OPTIONS:
        assert abs(_pressures(_model(name, Wettability.MIXED_WET), 0.4, 0.4, 0.2)[0]) > 1e-6


@pytest.mark.parametrize("name", list(MODEL_OPTIONS))
class TestTabularConversion:
    def test_matches_the_analytic_model(self, name):
        model = _model(name)
        table = as_three_phase_capillary_pressure_table(
            model, n_points=400, n_endpoint_extra=0, dtype=np.float64
        )
        for point in _points():
            np.testing.assert_allclose(
                _pressures(table, *point), _pressures(model, *point), rtol=1e-2
            )

    def test_pcow_depends_on_sw_only_and_pcgo_on_sg_only(self, name):
        table = as_three_phase_capillary_pressure_table(
            _model(name), n_points=300, n_endpoint_extra=0, dtype=np.float64
        )
        assert _pressures(table, 0.4, 0.6, 0.0)[0] == pytest.approx(
            _pressures(table, 0.4, 0.4, 0.2)[0]
        )
        assert _pressures(table, 0.2, 0.6, 0.2)[1] == pytest.approx(
            _pressures(table, 0.4, 0.4, 0.2)[1]
        )

    def test_orientation_does_not_change_the_pressures(self, name):
        """A table is queried at its own system's saturations whatever its reference axis."""
        model = _model(name)
        default = as_three_phase_capillary_pressure_table(
            model, n_points=500, n_endpoint_extra=0, dtype=np.float64
        )
        reoriented = as_three_phase_capillary_pressure_table(
            model,
            n_points=500,
            n_endpoint_extra=0,
            dtype=np.float64,
            oil_water_reference_phase="non_wetting",
            gas_oil_reference_phase="wetting",
        )
        assert reoriented.oil_water_table.get_reference_axis_phase() == FluidPhase.OIL
        assert reoriented.gas_oil_table.get_reference_axis_phase() == FluidPhase.OIL
        for point in _points():
            np.testing.assert_allclose(
                _pressures(reoriented, *point), _pressures(default, *point), rtol=1e-9
            )

    @pytest.mark.parametrize("oriented", [False, True], ids=["default", "reoriented"])
    def test_tabular_derivatives_match_finite_differences(self, name, oriented):
        options = (
            {"oil_water_reference_phase": "non_wetting", "gas_oil_reference_phase": "wetting"}
            if oriented
            else {}
        )
        table = as_three_phase_capillary_pressure_table(
            _model(name), n_points=500, n_endpoint_extra=0, dtype=np.float64, **options
        )
        _assert_derivatives_match_finite_differences(table, _points())


@pytest.mark.parametrize("oil_water_reference_phase", ["wetting", "non_wetting"])
def test_connate_water_is_recovered_from_either_axis(oil_water_reference_phase):
    table = as_three_phase_capillary_pressure_table(
        _model("brooks_corey"),
        n_points=300,
        n_endpoint_extra=0,
        oil_water_reference_phase=oil_water_reference_phase,
    )
    assert table.get_connate_water_saturation() == pytest.approx(SWC)
    assert table.oil_water_table.get_connate_water_saturation() == pytest.approx(SWC)

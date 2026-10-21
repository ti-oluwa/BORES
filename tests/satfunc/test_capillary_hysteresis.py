"""Tests for the Killough capillary pressure hysteresis table."""

import numpy as np
import pytest

from bores.blackoil.satfunc.capillary_pressure.hysteresis import KilloughCapillaryPressureTable
from bores.blackoil.satfunc.capillary_pressure.models import BrooksCoreyCapillaryPressureTable
from bores.blackoil.satfunc.tables import as_three_phase_capillary_pressure_table

ENDPOINTS = {
    "irreducible_water_saturation": 0.2,
    "residual_oil_saturation_water": 0.25,
    "residual_oil_saturation_gas": 0.15,
    "residual_gas_saturation": 0.05,
}
SWC = ENDPOINTS["irreducible_water_saturation"]
REORIENTED = {"oil_water_reference_phase": "non_wetting", "gas_oil_reference_phase": "wetting"}


def _tabular(**options):
    return as_three_phase_capillary_pressure_table(
        BrooksCoreyCapillaryPressureTable(**ENDPOINTS),
        n_points=500,
        n_endpoint_extra=0,
        dtype=np.float64,
        **options,
    )


def _hysteresis(oriented: bool = False, with_imbibition: bool = False):
    base = _tabular(**(REORIENTED if oriented else {}))
    options = (
        {
            "oil_water_imbibition_table": base.oil_water_table,
            "gas_oil_imbibition_table": base.gas_oil_table,
        }
        if with_imbibition
        else {}
    )
    return KilloughCapillaryPressureTable(
        oil_water_drainage_table=base.oil_water_table,
        gas_oil_drainage_table=base.gas_oil_table,
        **options,
    )


def _points(count: int = 20, seed: int = 5):
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


@pytest.mark.parametrize("oriented", [False, True], ids=["default", "reoriented"])
def test_drainage_matches_the_tabular_table(oriented):
    """Whatever the reference axes, a two-phase table is queried at its own system's saturations."""
    reference = _tabular()
    table = _hysteresis(oriented=oriented)
    for point in _points():
        np.testing.assert_allclose(
            _pressures(table, *point), _pressures(reference, *point), rtol=1e-9
        )


@pytest.mark.parametrize("oriented", [False, True], ids=["default", "reoriented"])
def test_derivatives_match_finite_differences(oriented):
    """The solver uses `dpcow_dsw - dpcow_dso`, `-dpcow_dso`, `-dpcgo_dso`, `dpcgo_dsg - dpcgo_dso`."""
    table = _hysteresis(oriented=oriented)
    step = 1e-6
    for sw, so, sg in _points():
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
        assert error.max() < 1e-5, f"at {(sw, so, sg)}: {analytical} vs {numerical}"


@pytest.mark.parametrize("oriented", [False, True], ids=["default", "reoriented"])
def test_scanning_between_identical_curves_reproduces_the_curve(oriented):
    """Boolean imbibition flags must work, and identical drainage and imbibition curves give the same Pc."""
    table = _hysteresis(oriented=oriented, with_imbibition=True)
    reference = _tabular()
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
    expected = reference.evaluate(sw, so, sg)
    np.testing.assert_allclose(result["oil_water"], expected["oil_water"], rtol=1e-9)
    np.testing.assert_allclose(result["gas_oil"], expected["gas_oil"], rtol=1e-9)


def test_connate_water_is_read_from_the_oil_water_table():
    assert _hysteresis().get_connate_water_saturation() == pytest.approx(SWC)
    assert _hysteresis(oriented=True).get_connate_water_saturation() == pytest.approx(SWC)


def test_a_parametric_oil_water_table_with_a_tabular_gas_oil_table():
    """The connate water saturation comes from the parametric model when it is the oil-water table."""
    reoriented = _tabular(**REORIENTED)
    table = KilloughCapillaryPressureTable(
        oil_water_drainage_table=BrooksCoreyCapillaryPressureTable(**ENDPOINTS),
        gas_oil_drainage_table=reoriented.gas_oil_table,
    )
    reference = _tabular()
    assert table.get_connate_water_saturation() == pytest.approx(SWC)
    for point in _points():
        np.testing.assert_allclose(
            _pressures(table, *point)[1], _pressures(reference, *point)[1], rtol=1e-9
        )

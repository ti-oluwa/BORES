"""Tests for the frozen and unfrozen saturation pressure modes of the PVT, mobility and transmissibility caches."""

import numpy as np
import pytest

from bores.blackoil.caches import transmissibility as transmissibility_module
from bores.blackoil.caches.mobility import compute_mobility_cache
from bores.blackoil.caches.pvt import compute_pvt_cache, compute_saturation_pressures
from bores.blackoil.caches.satfunc import make_new_cache as make_satfunc_cache

from .synthetic_tables import (
    SingleRegionPVT,
    build_gas_table,
    build_oil_table,
    solution_gor_at_saturation,
)

DTYPE = np.float64
STATE_FIELDS = ("dbo_drs", "dμo_drs", "dρo_drs", "dbg_drv", "dμg_drv", "dρg_drv")
PRESSURE = np.array([1500.0, 4000.0, 4000.0, 3000.0, 2500.0, 5000.0], dtype=DTYPE)
TEMPERATURE = np.full(6, 200.0, dtype=DTYPE)
SOLUTION_GOR = np.array(
    [solution_gor_at_saturation(1500.0, 200.0), 150.0, 300.0, 250.0, 150.0, 400.0], dtype=DTYPE
)
VAPORIZED_OIL_RATIO = np.array([2e-5, 7e-5, 3e-5, 5e-5, 1e-5, 8e-5], dtype=DTYPE)
SALINITY = np.full(6, 35000.0, dtype=DTYPE)
REGION = np.ones(6, dtype=int)


@pytest.fixture(scope="module")
def pvt():
    return SingleRegionPVT(
        oil=build_oil_table(dtype=np.dtype(DTYPE)), gas=build_gas_table(dtype=np.dtype(DTYPE))
    )


def _cache(pvt, **options):
    return compute_pvt_cache(
        PRESSURE, TEMPERATURE, SOLUTION_GOR, REGION, pvt, salinity=SALINITY, dtype=DTYPE, **options
    )


@pytest.fixture(scope="module")
def unfrozen(pvt):
    return _cache(pvt, vaporized_oil_ratio=VAPORIZED_OIL_RATIO)


@pytest.fixture(scope="module")
def saturation_pressures(pvt):
    return compute_saturation_pressures(
        TEMPERATURE,
        SOLUTION_GOR,
        REGION,
        pvt,
        vaporized_oil_ratio=VAPORIZED_OIL_RATIO,
        dtype=DTYPE,
    )


@pytest.fixture(scope="module")
def frozen(pvt, saturation_pressures):
    bubble_point, dew_point = saturation_pressures
    return _cache(
        pvt,
        freeze_saturation_pressure=True,
        frozen_bubble_point_pressure=bubble_point,
        frozen_dew_point_pressure=dew_point,
    )


class TestUnfrozen:
    def test_oil_derivatives_exist_only_for_undersaturated_cells(self, unfrozen):
        undersaturated = ~unfrozen.is_saturated
        assert undersaturated.any()
        assert (~undersaturated).any()
        for name in ("dbo_drs", "dμo_drs"):
            values = getattr(unfrozen, name)
            assert np.isfinite(values[undersaturated]).all()
            assert np.isnan(values[~undersaturated]).all()

    def test_gas_derivatives_exist_only_for_undersaturated_cells(self, unfrozen):
        for name in ("dbg_drv", "dμg_drv"):
            values = getattr(unfrozen, name)
            assert np.isfinite(values).any()
            assert np.isnan(values).any()

    def test_the_cache_matches_the_table_derivatives(self, pvt, unfrozen):
        oil = pvt[1].tables.oil
        expected = np.asarray(oil.db_drs(PRESSURE, TEMPERATURE, SOLUTION_GOR))
        np.testing.assert_allclose(unfrozen.dbo_drs, expected, equal_nan=True)

    def test_the_derivatives_have_the_expected_sign(self, unfrozen):
        finite = np.isfinite(unfrozen.dbo_drs)
        assert (unfrozen.dbo_drs[finite] > 0.0).all()  # more dissolved gas swells the oil
        assert (unfrozen.dμo_drs[finite] < 0.0).all()  # and thins it


class TestFrozen:
    def test_the_state_derivatives_are_not_defined(self, frozen):
        for name in STATE_FIELDS:
            assert np.isnan(getattr(frozen, name)).all(), name

    def test_matches_the_unfrozen_cache_when_the_state_is_that_of_the_frozen_saturation_pressure(
        self, unfrozen, frozen
    ):
        """Each cell was frozen at its own `Rs` and `Rv`, so the two modes must agree exactly."""
        assert frozen.is_saturated.tolist() == unfrozen.is_saturated.tolist()
        for name in (
            "oil_formation_volume_factor",
            "oil_viscosity",
            "dbo_dp",
            "dμo_dp",
            "solution_gas_oil_ratio",
            "gas_formation_volume_factor",
            "gas_viscosity",
        ):
            np.testing.assert_allclose(
                getattr(frozen, name),
                getattr(unfrozen, name),
                rtol=1e-12,
                equal_nan=True,
                err_msg=name,
            )

    def test_a_pressure_drop_below_the_frozen_bubble_point_gives_saturated_oil(
        self, pvt, saturation_pressures
    ):
        bubble_point, dew_point = saturation_pressures
        below = np.minimum(PRESSURE, bubble_point - 100.0)
        cache = compute_pvt_cache(
            below,
            TEMPERATURE,
            SOLUTION_GOR,
            REGION,
            pvt,
            salinity=SALINITY,
            freeze_saturation_pressure=True,
            frozen_bubble_point_pressure=bubble_point,
            frozen_dew_point_pressure=dew_point,
            dtype=DTYPE,
        )
        assert cache.is_saturated.all()
        # Saturated oil follows the table's own `Rs(P)` curve
        expected = np.asarray(
            pvt[1].tables.oil.solution_gas_to_oil_ratio(
                below, TEMPERATURE, bubble_point_pressure=bubble_point
            )
        )
        np.testing.assert_allclose(cache.solution_gas_oil_ratio, expected, rtol=1e-9)

    def test_the_frozen_bubble_point_is_required(self, pvt):
        with pytest.raises(ValueError, match="frozen_bubble_point_pressure"):
            _cache(pvt, freeze_saturation_pressure=True)


def test_saturation_pressures_are_taken_from_the_state(pvt):
    bubble_point, dew_point = compute_saturation_pressures(
        TEMPERATURE,
        SOLUTION_GOR,
        REGION,
        pvt,
        vaporized_oil_ratio=VAPORIZED_OIL_RATIO,
        dtype=DTYPE,
    )
    expected_bubble_point = np.asarray(
        pvt[1].tables.oil.bubble_point_pressure(temperature=TEMPERATURE, solution_gor=SOLUTION_GOR)
    )
    np.testing.assert_allclose(bubble_point, expected_bubble_point)
    assert np.isfinite(dew_point).all()
    _, without_rv = compute_saturation_pressures(
        TEMPERATURE, SOLUTION_GOR, REGION, pvt, dtype=DTYPE
    )
    assert np.isnan(without_rv).all()


def _mobility(pvt_cache):
    satfunc = make_satfunc_cache(len(PRESSURE), DTYPE)
    satfunc.water_relative_permeability[:] = 0.2
    satfunc.oil_relative_permeability[:] = 0.5
    satfunc.gas_relative_permeability[:] = 0.1
    return compute_mobility_cache(pvt_cache, satfunc, dtype=DTYPE)


class TestMobility:
    def test_the_state_derivatives_follow_the_quotient_rule(self, unfrozen):
        mobility = _mobility(unfrozen)
        oil_expected = -0.5 * unfrozen.dμo_drs / unfrozen.oil_viscosity**2
        gas_expected = -0.1 * unfrozen.dμg_drv / unfrozen.gas_viscosity**2
        np.testing.assert_allclose(mobility.dʎo_drs, oil_expected, equal_nan=True)
        np.testing.assert_allclose(mobility.dʎg_drv, gas_expected, equal_nan=True)
        # Only oil depends on Rs and only gas on Rv
        np.testing.assert_allclose(mobility.dʎT_drs, oil_expected, equal_nan=True)
        np.testing.assert_allclose(mobility.dʎT_drv, gas_expected, equal_nan=True)

    def test_there_is_no_state_derivative_when_frozen(self, frozen):
        mobility = _mobility(frozen)
        assert np.isnan(mobility.dʎo_drs).all()
        assert np.isnan(mobility.dʎg_drv).all()


class TestTransmissibility:
    def test_the_state_derivatives_use_the_upstream_cell(self, unfrozen):
        mobility = _mobility(unfrozen)
        owners = np.array([0, 1])
        neighbours = np.array([1, 2])
        geometric = np.array([2.0, 3.0], dtype=DTYPE)
        # Pressure falls along the cells, so the first cell of each face is upstream
        falling = np.array([4000.0, 3000.0, 2000.0], dtype=DTYPE)
        cache = transmissibility_module.make_new_cache(2, DTYPE)
        transmissibility_module.update_transmissibility_cache(
            owners,
            neighbours,
            geometric,
            np.zeros(3, dtype=DTYPE),
            0.0,
            _with_densities(_slice(unfrozen, 3)),
            _slice(mobility, 3),
            falling,
            falling,
            falling,
            cache,
        )
        np.testing.assert_allclose(
            cache.dTo_drs, geometric * mobility.dʎo_drs[owners], equal_nan=True
        )
        np.testing.assert_allclose(
            cache.dTg_drv, geometric * mobility.dʎg_drv[owners], equal_nan=True
        )


def _slice(cache, count):
    """The first `count` cells of a per-cell NamedTuple cache."""
    return type(cache)(*(np.asarray(field)[:count] for field in cache))


def _with_densities(cache):
    """The synthetic tables have no density, which would make the potential (and the upstream cell) undefined."""
    ones = np.ones(len(cache.oil_density), dtype=DTYPE)
    return cache._replace(
        oil_density=50.0 * ones, gas_density=5.0 * ones, water_density=62.0 * ones
    )

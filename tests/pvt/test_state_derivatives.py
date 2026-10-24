"""Tests for the `Rs`/`Rv` derivatives of the oil and gas PVT tables."""

import numpy as np
import pytest

from .synthetic_tables import (
    RS_AXIS,
    RV_AXIS,
    bubble_point,
    build_gas_table,
    build_oil_table,
    dew_point,
)

# States at the middle of a table interval, so a reference difference never straddles a knot
RS_UNDERSATURATED = np.array([RS_AXIS[2] + 30.0, RS_AXIS[4] + 30.0, RS_AXIS[6] + 30.0])
RV_UNDERSATURATED = np.array([RV_AXIS[1] + 6e-6, RV_AXIS[3] + 6e-6, RV_AXIS[5] + 6e-6])
TEMPERATURE = 200.0


def _reference(value, state, step):
    """Central difference of the table's own value function, with the saturation pressure following the state."""
    return (value(state + step) - value(state - step)) / (2.0 * step)


@pytest.fixture(params=[False, True], ids=["formula", "undersaturated-tables"])
def oil(request):
    return build_oil_table(with_undersaturated_tables=request.param, dtype=np.dtype("float64"))


@pytest.fixture
def gas():
    return build_gas_table(dtype=np.dtype("float64"))


class TestOil:
    def _undersaturated_pressure(self):
        return bubble_point(RS_UNDERSATURATED, TEMPERATURE) + 800.0

    def test_formation_volume_factor_derivative_matches_the_value_function(self, oil):
        pressure = self._undersaturated_pressure()
        temperature = np.full(3, TEMPERATURE)
        expected = _reference(
            lambda rs: np.asarray(
                oil.formation_volume_factor(pressure, temperature, solution_gor=rs)
            ),
            RS_UNDERSATURATED,
            1e-4,
        )
        np.testing.assert_allclose(
            oil.db_drs(pressure, temperature, RS_UNDERSATURATED), expected, rtol=1e-4
        )

    def test_viscosity_derivative_matches_the_value_function(self, oil):
        pressure = self._undersaturated_pressure()
        temperature = np.full(3, TEMPERATURE)
        expected = _reference(
            lambda rs: np.asarray(oil.viscosity(pressure, temperature, solution_gor=rs)),
            RS_UNDERSATURATED,
            1e-4,
        )
        np.testing.assert_allclose(
            oil.dμ_drs(pressure, temperature, RS_UNDERSATURATED), expected, rtol=1e-4
        )

    def test_density_derivative_matches_the_value_function(self, oil):
        """The density depends on `Rs` directly and through `Bo`: `(rho_sto + Rs * rho_gas / f) / Bo`."""
        pressure = self._undersaturated_pressure()
        temperature = np.full(3, TEMPERATURE)
        expected = _reference(
            lambda rs: np.asarray(oil.density(pressure, temperature, solution_gor=rs)),
            RS_UNDERSATURATED,
            1e-4,
        )
        derivative = np.asarray(oil.dρ_drs(pressure, temperature, RS_UNDERSATURATED))
        assert (np.abs(derivative) > 0.0).all()
        np.testing.assert_allclose(derivative, expected, rtol=1e-4)

    def test_saturated_cells_have_no_derivative(self, oil):
        rs = RS_UNDERSATURATED
        pressure = bubble_point(rs, TEMPERATURE) - 50.0  # below the bubble point
        derivative = np.asarray(oil.db_drs(pressure, np.full(3, TEMPERATURE), rs))
        assert np.isnan(derivative).all()

    def test_just_undersaturated_cell_uses_a_one_sided_step(self, oil):
        """A forward step would cross the bubble point, so the derivative is taken backwards."""
        rs = np.array([RS_AXIS[4] + 30.0])
        pressure = bubble_point(rs, TEMPERATURE) + 1e-3
        temperature = np.full(1, TEMPERATURE)

        def fvf(state):
            return float(
                np.asarray(oil.formation_volume_factor(pressure, temperature, solution_gor=state))
            )

        step = 1e-4
        backward = (fvf(rs) - fvf(rs - step)) / step
        derivative = float(np.asarray(oil.db_drs(pressure, temperature, rs)))
        assert np.isfinite(derivative)
        assert derivative == pytest.approx(backward, rel=1e-3)

    def test_multidimensional_inputs_keep_their_shape(self, oil):
        pressure = np.broadcast_to(self._undersaturated_pressure()[:, None], (3, 2))
        state = np.broadcast_to(RS_UNDERSATURATED[:, None], (3, 2))
        derivative = np.asarray(oil.db_drs(pressure, np.full((3, 2), TEMPERATURE), state))
        assert derivative.shape == (3, 2)
        np.testing.assert_allclose(derivative[:, 0], derivative[:, 1])

    def test_scalar_inputs_give_a_scalar(self, oil):
        pressure = float(self._undersaturated_pressure()[0])
        assert np.ndim(oil.db_drs(pressure, TEMPERATURE, float(RS_UNDERSATURATED[0]))) == 0

    def test_other_phases_have_no_oil_derivatives(self, gas):
        assert gas.db_drs(3000.0, TEMPERATURE, 200.0) is None
        assert gas.dμ_drs(3000.0, TEMPERATURE, 200.0) is None
        assert gas.dρ_drs(3000.0, TEMPERATURE, 200.0) is None


class TestGas:
    def _undersaturated_pressure(self):
        return dew_point(RV_UNDERSATURATED, TEMPERATURE) + 1500.0

    def test_formation_volume_factor_derivative_matches_the_value_function(self, gas):
        pressure = self._undersaturated_pressure()
        temperature = np.full(3, TEMPERATURE)
        expected = _reference(
            lambda rv: np.asarray(
                gas.formation_volume_factor(pressure, temperature, vaporized_oil_ratio=rv)
            ),
            RV_UNDERSATURATED,
            1e-8,
        )
        np.testing.assert_allclose(
            gas.db_drv(pressure, temperature, RV_UNDERSATURATED), expected, rtol=1e-4
        )

    def test_viscosity_derivative_matches_the_value_function(self, gas):
        pressure = self._undersaturated_pressure()
        temperature = np.full(3, TEMPERATURE)
        expected = _reference(
            lambda rv: np.asarray(gas.viscosity(pressure, temperature, vaporized_oil_ratio=rv)),
            RV_UNDERSATURATED,
            1e-8,
        )
        np.testing.assert_allclose(
            gas.dμ_drv(pressure, temperature, RV_UNDERSATURATED), expected, rtol=1e-4
        )

    def test_the_step_scales_with_the_small_magnitude_of_rv(self, gas):
        """`Rv` is of order 1e-5, so a step relative to a state of 1 would span the whole axis."""
        pressure = self._undersaturated_pressure()[:1]
        temperature = np.full(1, TEMPERATURE)
        derivative = float(np.asarray(gas.dμ_drv(pressure, temperature, RV_UNDERSATURATED[:1])))
        # The 3-D table is linear in Rv: mu = mu_dry * (1 + 500 Rv)
        mu_dry = 0.012 + 2e-6 * float(pressure[0]) + 1e-5 * (TEMPERATURE - 150.0)
        assert derivative == pytest.approx(mu_dry * 500.0, rel=1e-3)

    def test_density_derivative_matches_the_value_function(self, gas):
        pressure = self._undersaturated_pressure()
        temperature = np.full(3, TEMPERATURE)
        expected = _reference(
            lambda rv: np.asarray(gas.density(pressure, temperature, vaporized_oil_ratio=rv)),
            RV_UNDERSATURATED,
            1e-8,
        )
        derivative = np.asarray(gas.dρ_drv(pressure, temperature, RV_UNDERSATURATED))
        assert (np.abs(derivative) > 0.0).all()
        np.testing.assert_allclose(derivative, expected, rtol=1e-4)

    def test_saturated_gas_has_no_derivative(self, gas):
        pressure = dew_point(RV_UNDERSATURATED, TEMPERATURE) - 50.0
        assert np.isnan(
            np.asarray(gas.db_drv(pressure, np.full(3, TEMPERATURE), RV_UNDERSATURATED))
        ).all()

    def test_other_phases_have_no_gas_derivatives(self, oil):
        assert oil.db_drv(3000.0, TEMPERATURE, 2e-5) is None
        assert oil.dμ_drv(3000.0, TEMPERATURE, 2e-5) is None
        assert oil.dρ_drv(3000.0, TEMPERATURE, 2e-5) is None

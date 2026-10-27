import numpy as np
import pytest
from scipy.optimize import brentq

from bores.correlations import gas
from bores.errors import ValidationError


def pseudocritical(gas_gravity):
    return (
        756.8 - 131.0 * gas_gravity - 3.6 * gas_gravity**2,
        169.2 + 349.5 * gas_gravity - 74.0 * gas_gravity**2,
    )


def hall_yarborough_reference(reduced_pressure, reduced_temperature):
    """Hall-Yarborough solved with a bracketing root finder."""
    t = 1.0 / reduced_temperature
    a = 0.06125 * t * np.exp(-1.2 * (1.0 - t) ** 2)
    b = t * (14.76 - 9.76 * t + 4.58 * t**2)
    c = t * (90.7 - 242.2 * t + 42.4 * t**2)
    d = 2.18 + 2.82 * t

    def residual(y):
        return (
            -a * reduced_pressure + (y + y**2 + y**3 - y**4) / (1 - y) ** 3 - b * y**2 + c * y**d
        )

    return a * reduced_pressure / brentq(residual, 1e-9, 0.95)


Z_GRID = [(1000.0, 150.0), (3000.0, 200.0), (5000.0, 250.0), (9000.0, 300.0), (14000.0, 350.0)]


class TestPseudocriticalProperties:
    def test_sutton_for_clean_gas(self):
        for gravity in (0.6, 0.7, 0.9):
            ppc, tpc = gas.compute_gas_pseudocritical_properties(gravity)
            assert (ppc, tpc) == pytest.approx(pseudocritical(gravity))
            assert gas.compute_gas_pseudocritical_pressure(gravity) == pytest.approx(ppc)
            assert gas.compute_gas_pseudocritical_temperature(gravity) == pytest.approx(tpc)

    def test_wichert_aziz_uses_the_uncorrected_temperature_in_the_pressure_term(self):
        ppc0, tpc0 = pseudocritical(0.7)
        a, b = 0.2, 0.1
        epsilon = 120 * (a**0.9 - a**1.6) + 15 * (b**0.5 - b**4)
        expected_t = tpc0 - epsilon
        expected_p = ppc0 * expected_t / (tpc0 + b * (1 - b) * epsilon)
        ppc, tpc = gas.compute_gas_pseudocritical_properties(0.7, 0.1, 0.1)
        assert (ppc, tpc) == pytest.approx((expected_p, expected_t))

    def test_nitrogen_lowers_both_pseudocritical_properties(self):
        clean = gas.compute_gas_pseudocritical_properties(0.7)
        diluted = gas.compute_gas_pseudocritical_properties(0.7, 0.0, 0.0, 0.1)
        assert diluted[0] == pytest.approx(clean[0] - 17.0)
        assert diluted[1] == pytest.approx(clean[1] - 25.0)

    def test_correction_is_continuous_at_zero_acid_gas(self):
        clean = gas.compute_gas_pseudocritical_properties(0.7)
        tiny = gas.compute_gas_pseudocritical_properties(0.7, 1e-9, 1e-9)
        assert tiny == pytest.approx(clean, rel=1e-4)

    def test_arrays_and_defaults(self):
        gravity = np.array([0.6, 0.7, 0.8])
        np.testing.assert_allclose(
            gas.compute_gas_pseudocritical_pressure(gravity), pseudocritical(gravity)[0]
        )

    def test_invalid_contaminants_raise(self):
        with pytest.raises(ValidationError):
            gas.compute_gas_pseudocritical_pressure(0.7, h2s_mole_fraction=0.3)
        with pytest.raises(ValidationError):
            gas.compute_gas_pseudocritical_temperature(0.7, 0.2, 0.3)
        with pytest.raises(ValidationError):
            gas.compute_gas_pseudocritical_pressure(0.0)


class TestCompressibilityFactor:
    @pytest.mark.parametrize(("pressure", "temperature"), Z_GRID)
    def test_hall_yarborough_matches_an_independent_solution(self, pressure, temperature):
        ppc, tpc = pseudocritical(0.7)
        reference = hall_yarborough_reference(pressure / ppc, (temperature + 459.67) / tpc)
        result = gas.compute_gas_compressibility_factor_hall_yarborough(pressure, temperature, 0.7)
        assert result == pytest.approx(reference, rel=1e-9)

    @pytest.mark.parametrize(("pressure", "temperature"), Z_GRID)
    def test_dak_agrees_with_hall_yarborough_within_one_percent(self, pressure, temperature):
        dak = gas.compute_gas_compressibility_factor_dranchuk_abou_kassem(
            pressure, temperature, 0.7
        )
        hall = gas.compute_gas_compressibility_factor_hall_yarborough(pressure, temperature, 0.7)
        assert dak == pytest.approx(hall, rel=1e-2)

    def test_dak_satisfies_its_own_equation_of_state(self):
        pressure, temperature = 7000.0, 220.0
        ppc, tpc = pseudocritical(0.75)
        pr, tr = pressure / ppc, (temperature + 459.67) / tpc
        z = gas.compute_gas_compressibility_factor_dranchuk_abou_kassem(
            pressure, temperature, 0.75
        )
        rho = 0.27 * pr / (z * tr)
        a = [
            0.3265,
            -1.0700,
            -0.5339,
            0.01569,
            -0.05165,
            0.5475,
            -0.7361,
            0.1844,
            0.1056,
            0.6134,
            0.7210,
        ]
        rhs = (
            1
            + (a[0] + a[1] / tr + a[2] / tr**3 + a[3] / tr**4 + a[4] / tr**5) * rho
            + (a[5] + a[6] / tr + a[7] / tr**2) * rho**2
            - a[8] * (a[6] / tr + a[7] / tr**2) * rho**5
            + a[9] * (1 + a[10] * rho**2) * (rho**2 / tr**3) * np.exp(-a[10] * rho**2)
        )
        assert z == pytest.approx(rhs, rel=1e-9)

    def test_dak_converges_where_a_fixed_point_iteration_does_not(self):
        # Pr = 7.0 and Tr = 1.43 sits where successive substitution diverges.
        ppc, tpc = pseudocritical(0.7)
        pressure = 7.01 * ppc
        temperature = 1.43 * tpc - 459.67
        z = gas.compute_gas_compressibility_factor_dranchuk_abou_kassem(pressure, temperature, 0.7)
        assert z == pytest.approx(0.91, abs=0.02)

    @pytest.mark.parametrize(("pressure", "temperature"), Z_GRID[:3])
    def test_papay_matches_published_formula(self, pressure, temperature):
        ppc, tpc = pseudocritical(0.7)
        pr, tr = pressure / ppc, (temperature + 459.67) / tpc
        expected = 1 - 3.52 * pr / 10 ** (0.9813 * tr) + 0.274 * pr**2 / 10 ** (0.8157 * tr)
        assert gas.compute_gas_compressibility_factor_papay(pressure, temperature, 0.7) == (
            pytest.approx(expected)
        )

    def test_papay_is_within_six_percent_up_to_a_reduced_pressure_of_eight(self):
        for pressure, temperature in [(1000.0, 150.0), (3000.0, 200.0), (5000.0, 250.0)]:
            papay = gas.compute_gas_compressibility_factor_papay(pressure, temperature, 0.7)
            hall = gas.compute_gas_compressibility_factor_hall_yarborough(
                pressure, temperature, 0.7
            )
            assert papay == pytest.approx(hall, rel=0.06)

    @pytest.mark.parametrize(
        "correlation",
        [
            gas.compute_gas_compressibility_factor_hall_yarborough,
            gas.compute_gas_compressibility_factor_dranchuk_abou_kassem,
        ],
    )
    def test_low_pressure_limit_is_ideal(self, correlation):
        assert correlation(1.0, 200.0, 0.7) == pytest.approx(1.0, abs=1e-15)
        assert correlation(0.0, 200.0, 0.7) == pytest.approx(1.0, abs=1e-15)

    def test_papay_low_pressure_limit_is_nearly_ideal(self):
        assert gas.compute_gas_compressibility_factor_papay(1.0, 200.0, 0.7) == pytest.approx(
            1.0, abs=1e-3
        )

    @pytest.mark.parametrize(
        "correlation",
        [
            gas.compute_gas_compressibility_factor_papay,
            gas.compute_gas_compressibility_factor_hall_yarborough,
            gas.compute_gas_compressibility_factor_dranchuk_abou_kassem,
        ],
    )
    def test_results_are_limited_to_a_physical_range(self, correlation, conditions):
        z = correlation(
            conditions["pressure"], conditions["temperature"], conditions["gas_gravity"]
        )
        assert np.all((z >= 0.2) & (z <= 3.0))

    def test_acid_gas_changes_the_factor(self):
        clean = gas.compute_gas_compressibility_factor(4000.0, 200.0, 0.7)
        sour = gas.compute_gas_compressibility_factor(4000.0, 200.0, 0.7, h2s_mole_fraction=0.1)
        assert sour != pytest.approx(clean)

    def test_dispatcher_selects_the_method(self):
        kw = dict(pressure=3000.0, temperature=200.0, gas_gravity=0.7)
        assert gas.compute_gas_compressibility_factor(**kw) == pytest.approx(
            gas.compute_gas_compressibility_factor_dranchuk_abou_kassem(**kw)
        )
        assert gas.compute_gas_compressibility_factor(
            **kw, method="hall-yarborough"
        ) == pytest.approx(gas.compute_gas_compressibility_factor_hall_yarborough(**kw))
        assert gas.compute_gas_compressibility_factor(**kw, method="papay") == pytest.approx(
            gas.compute_gas_compressibility_factor_papay(**kw)
        )
        with pytest.raises(ValidationError, match="Unknown method"):
            gas.compute_gas_compressibility_factor(**kw, method="nope")  # type: ignore[arg-type]

    def test_arrays_match_scalars(self, conditions):
        c = conditions
        z = gas.compute_gas_compressibility_factor(
            c["pressure"], c["temperature"], c["gas_gravity"]
        )
        for i in (0, 7, 24):
            assert z[i] == pytest.approx(
                gas.compute_gas_compressibility_factor(
                    float(c["pressure"][i]), float(c["temperature"][i]), float(c["gas_gravity"][i])
                )
            )

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValidationError):
            gas.compute_gas_compressibility_factor_dranchuk_abou_kassem(-1.0, 200.0, 0.7)
        with pytest.raises(ValidationError):
            gas.compute_gas_compressibility_factor_papay(3000.0, 200.0, 0.0)


class TestGasProperties:
    def test_formation_volume_factor(self):
        expected = 0.9 * (200.0 + 459.67) * 14.696 / (3000.0 * 519.67)
        assert gas.compute_gas_formation_volume_factor(3000.0, 200.0, 0.9) == pytest.approx(
            expected, rel=1e-3
        )

    def test_formation_volume_factor_is_one_for_ideal_gas_at_standard_conditions(self):
        assert gas.compute_gas_formation_volume_factor(14.696, 60.0, 1.0) == pytest.approx(
            1.0, rel=1e-3
        )

    def test_molecular_weight(self):
        assert gas.compute_gas_molecular_weight(1.0) == pytest.approx(28.9647, rel=1e-3)
        assert gas.compute_gas_molecular_weight(0.6) == pytest.approx(0.6 * 28.9647, rel=1e-3)

    def test_density_from_real_gas_law(self):
        pressure, temperature, gravity, z = 3000.0, 200.0, 0.7, 0.9
        expected = pressure * gravity * 28.9647 / (z * 10.732 * (temperature + 459.67))
        assert gas.compute_gas_density(pressure, temperature, gravity, z) == pytest.approx(
            expected, rel=1e-3
        )

    def test_density_and_formation_volume_factor_are_consistent(self):
        # Standard density times Bg is the reservoir density.
        pressure, temperature, gravity, z = 3000.0, 200.0, 0.7, 0.9
        bg = gas.compute_gas_formation_volume_factor(pressure, temperature, z)
        standard = gas.compute_gas_density(14.696, 60.0, gravity, 1.0)
        reservoir = gas.compute_gas_density(pressure, temperature, gravity, z)
        assert reservoir == pytest.approx(standard / bg, rel=2e-3)

    def test_viscosity_is_lee_gonzalez_eakin(self):
        temperature, gravity, pressure = 200.0, 0.65, 3000.0
        z = gas.compute_gas_compressibility_factor(pressure, temperature, gravity)
        density = gas.compute_gas_density(pressure, temperature, gravity, z)
        molecular_weight = gas.compute_gas_molecular_weight(gravity)
        t = temperature + 459.67
        k = (9.4 + 0.02 * molecular_weight) * t**1.5 / (209 + 19 * molecular_weight + t)
        x = 3.5 + 986 / t + 0.01 * molecular_weight
        expected = 1e-4 * k * np.exp(x * (density * 0.0160185) ** (2.4 - 0.2 * x))
        result = gas.compute_gas_viscosity(temperature, density, molecular_weight)
        assert result == pytest.approx(expected, rel=1e-3)
        assert 0.012 < result < 0.03

    def test_viscosity_rises_with_density(self):
        low = gas.compute_gas_viscosity(200.0, 2.0, 19.0)
        high = gas.compute_gas_viscosity(200.0, 12.0, 19.0)
        assert high > low

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValidationError):
            gas.compute_gas_formation_volume_factor(0.0, 200.0, 0.9)
        with pytest.raises(ValidationError):
            gas.compute_gas_density(3000.0, 200.0, 0.7, 0.0)
        with pytest.raises(ValidationError):
            gas.compute_gas_viscosity(200.0, 0.0, 19.0)
        with pytest.raises(ValidationError):
            gas.compute_gas_molecular_weight(-0.1)


class TestGasCompressibility:
    def test_matches_a_finite_difference_of_the_factor(self):
        pressure, temperature, gravity = 3000.0, 200.0, 0.7

        def z(p):
            return gas.compute_gas_compressibility_factor(p, temperature, gravity)

        dz = (z(pressure + 0.5) - z(pressure - 0.5)) / 1.0
        expected = 1.0 / pressure - dz / z(pressure)
        assert gas.compute_gas_compressibility(pressure, temperature, gravity) == pytest.approx(
            expected, rel=1e-4
        )

    def test_ideal_gas_limit_is_one_over_pressure(self):
        assert gas.compute_gas_compressibility(5.0, 200.0, 0.7) == pytest.approx(
            1.0 / 5.0, rel=1e-3
        )

    def test_supplied_factor_is_used(self):
        computed = gas.compute_gas_compressibility(3000.0, 200.0, 0.7)
        z = gas.compute_gas_compressibility_factor(3000.0, 200.0, 0.7)
        assert gas.compute_gas_compressibility(3000.0, 200.0, 0.7, z) == pytest.approx(computed)

    def test_never_negative(self, conditions):
        c = conditions
        result = gas.compute_gas_compressibility(c["pressure"], c["temperature"], c["gas_gravity"])
        assert np.all(result >= 0)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValidationError):
            gas.compute_gas_compressibility(0.0, 200.0, 0.7)


class TestGasGravity:
    coolprop = pytest.importorskip("CoolProp")

    def test_methane_gravity(self):
        assert gas.compute_gas_gravity("Methane") == pytest.approx(0.5547, abs=2e-3)
        assert gas.compute_gas_gravity("CO2") == pytest.approx(1.53, abs=0.02)
        assert gas.compute_gas_gravity("Nitrogen") == pytest.approx(0.967, abs=0.01)

    def test_unknown_gas_raises(self):
        with pytest.raises(ValidationError, match="not supported"):
            gas.compute_gas_gravity("Unobtainium")

    def test_gravity_from_density_inverts_the_density_of_a_known_gas(self):
        from bores.correlations import core

        for pressure, temperature in [(14.696, 60.0), (3000.0, 200.0)]:
            density = core.compute_fluid_density(pressure, temperature, "Methane")
            gravity = gas.compute_gas_gravity_from_density(pressure, temperature, density)
            expected = density / core.compute_fluid_density(pressure, temperature, "Air")
            assert gravity == pytest.approx(expected)
        standard = core.compute_fluid_density(14.696, 60.0, "Methane")
        assert gas.compute_gas_gravity_from_density(14.696, 60.0, standard) == pytest.approx(
            0.5547, abs=2e-3
        )

    def test_gravity_from_density_accepts_arrays(self):
        pressure = np.array([500.0, 3000.0])
        gravity = gas.compute_gas_gravity_from_density(pressure, 200.0, np.array([1.0, 6.0]))
        assert gravity.shape == (2,)

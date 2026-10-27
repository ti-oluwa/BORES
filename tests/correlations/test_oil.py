import typing

import numpy as np
import pytest

from bores.correlations import oil
from bores.errors import ValidationError

STANDARD_PRESSURE = 14.696
STANDARD_TEMPERATURE = 60.0


def api_of(specific_gravity):
    return 141.5 / specific_gravity - 131.5


def vb_bubble_coefficients(api):
    return (0.0362, 1.0937, 25.7240) if api <= 30 else (0.0178, 1.1870, 23.9310)


class TestApiGravity:
    def test_known_values(self):
        assert oil.compute_oil_api_gravity(1.0) == pytest.approx(10.0)
        assert oil.compute_oil_api_gravity(141.5 / 131.5 / 1.0 * 0 + 0.85) == pytest.approx(
            34.97058823529412
        )

    def test_arrays(self):
        sg = np.array([0.8, 0.85, 0.9])
        np.testing.assert_allclose(oil.compute_oil_api_gravity(sg), 141.5 / sg - 131.5)

    def test_non_positive_gravity_raises(self):
        with pytest.raises(ValidationError):
            oil.compute_oil_api_gravity(0.0)


class TestOilSpecificGravity:
    def test_round_trip_through_pressure_and_temperature(self):
        # rho(P, T) = rho_stp * exp[Co (P - Pstp) - alpha (T - Tstp)]
        specific_gravity, compressibility, alpha = 0.85, 1e-5, 5.39e-4
        for pressure, temperature in [(3000.0, 200.0), (5000.0, 250.0), (1000.0, 120.0)]:
            density = (
                specific_gravity
                * 62.37
                * np.exp(compressibility * (pressure - STANDARD_PRESSURE))
                * np.exp(-alpha * (temperature - STANDARD_TEMPERATURE))
            )
            result = oil.compute_oil_specific_gravity(
                oil_density=density,
                pressure=pressure,
                temperature=temperature,
                oil_compressibility=compressibility,
            )
            assert result == pytest.approx(specific_gravity, rel=2e-3)

    def test_hotter_oil_has_higher_standard_gravity_than_its_reservoir_density_suggests(self):
        cold = oil.compute_oil_specific_gravity(50.0, STANDARD_PRESSURE, 60.0, 1e-5)
        hot = oil.compute_oil_specific_gravity(50.0, STANDARD_PRESSURE, 250.0, 1e-5)
        assert hot > cold

    def test_invalid_density_raises(self):
        with pytest.raises(ValidationError):
            oil.compute_oil_specific_gravity(-1.0, 3000.0, 200.0, 1e-5)


class TestFormationVolumeFactor:
    def test_standing(self):
        sg, gg, rs, t = 0.85, 0.75, 600.0, 200.0
        factor = rs * np.sqrt(gg / sg) + 1.25 * t
        expected = 0.972 + 0.000147 * factor**1.175
        assert oil.compute_oil_formation_volume_factor_standing(t, sg, gg, rs) == pytest.approx(
            expected
        )

    def test_vazquez_beggs_uses_api_gravity_not_specific_gravity(self):
        sg, gg, rs, t = 0.85, 0.75, 600.0, 200.0
        api = api_of(sg)
        c1, c2, c3 = 4.670e-4, 1.100e-5, 1.337e-9
        expected = 1.0 + c1 * rs + (t - 60.0) * (api / gg) * (c2 + c3 * rs)
        result = oil.compute_oil_formation_volume_factor_vazquez_and_beggs(t, sg, gg, rs)
        assert result == pytest.approx(expected)
        assert result == pytest.approx(1.3572, abs=1e-4)

    def test_vazquez_beggs_coefficients_switch_at_30_api(self):
        assert oil.get_vazquez_beggs_oil_fvf_coefficients(30.0) == (4.677e-4, 1.751e-5, -1.811e-8)
        assert oil.get_vazquez_beggs_oil_fvf_coefficients(30.1) == (4.670e-4, 1.100e-5, 1.337e-9)

    def test_vazquez_beggs_is_continuous_enough_with_standing_at_the_switch_temperature(self):
        args = dict(oil_specific_gravity=0.85, gas_gravity=0.75, gas_to_oil_ratio=600.0)
        standing = oil.compute_oil_formation_volume_factor_standing(temperature=100.0, **args)
        vazquez = oil.compute_oil_formation_volume_factor_vazquez_and_beggs(
            temperature=100.0, **args
        )
        assert abs(standing - vazquez) / standing < 0.05

    def test_pressure_correction_only_above_bubble_point(self):
        kw = dict(saturated_oil_fvf=1.3, oil_compressibility=1e-5, bubble_point_pressure=2500.0)
        assert oil.correct_oil_fvf_for_pressure(current_pressure=2000.0, **kw) == pytest.approx(
            1.3
        )
        assert oil.correct_oil_fvf_for_pressure(current_pressure=2500.0, **kw) == pytest.approx(
            1.3
        )
        assert oil.correct_oil_fvf_for_pressure(current_pressure=4500.0, **kw) == pytest.approx(
            1.3 * np.exp(-1e-5 * 2000.0)
        )

    def test_full_factor_picks_standing_below_and_vazquez_above_100_f(self):
        common = dict(
            pressure=2000.0,
            bubble_point_pressure=2500.0,
            oil_specific_gravity=0.85,
            gas_gravity=0.75,
            gas_to_oil_ratio=600.0,
            oil_compressibility=1e-5,
        )
        cold = oil.compute_oil_formation_volume_factor(temperature=90.0, **common)
        hot = oil.compute_oil_formation_volume_factor(temperature=200.0, **common)
        assert cold == pytest.approx(
            oil.compute_oil_formation_volume_factor_standing(90.0, 0.85, 0.75, 600.0)
        )
        assert hot == pytest.approx(
            oil.compute_oil_formation_volume_factor_vazquez_and_beggs(200.0, 0.85, 0.75, 600.0)
        )

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValidationError):
            oil.compute_oil_formation_volume_factor_standing(200.0, 0.0, 0.75, 600.0)
        with pytest.raises(ValidationError):
            oil.compute_oil_formation_volume_factor_vazquez_and_beggs(200.0, 0.85, 0.75, -1.0)


class TestBubblePointAndSolutionGor:
    def test_vazquez_beggs_bubble_point(self):
        api, gg, t, rs = 35.0, 0.75, 200.0, 600.0
        c1, c2, c3 = vb_bubble_coefficients(api)
        expected = (rs / (c1 * gg * np.exp(c3 * api / (t + 459.67)))) ** (1 / c2)
        assert oil.compute_oil_bubble_point_pressure(gg, api, t, rs) == pytest.approx(expected)

    def test_gor_is_inverse_of_bubble_point(self):
        api, gg, t, rs = 35.0, 0.75, 200.0, 600.0
        pb = oil.compute_oil_bubble_point_pressure(gg, api, t, rs)
        below = oil.compute_gas_to_oil_ratio(pb - 1.0, t, pb, gg, api)
        assert below == pytest.approx(rs, rel=2e-3)
        assert oil.estimate_solution_gor(pb, t, api, gg) == pytest.approx(rs, rel=1e-9)

    def test_bubble_point_coefficients_switch_at_30_api(self):
        assert oil.get_vazquez_beggs_oil_bubble_point_pressure_coefficients(30.0) == (
            0.0362,
            1.0937,
            25.7240,
        )
        assert oil.get_vazquez_beggs_oil_bubble_point_pressure_coefficients(30.5) == (
            0.0178,
            1.1870,
            23.9310,
        )

    def test_gor_is_capped_above_bubble_point(self):
        api, gg, t, pb = 35.0, 0.75, 200.0, 2500.0
        at_bubble_point = oil.compute_gas_to_oil_ratio(pb, t, pb, gg, api)
        assert oil.compute_gas_to_oil_ratio(5000.0, t, pb, gg, api) == pytest.approx(
            at_bubble_point
        )
        assert oil.compute_gas_to_oil_ratio(5000.0, t, pb, gg, api, 650.0) == pytest.approx(650.0)

    def test_gor_increases_with_pressure_below_bubble_point(self):
        pressure = np.linspace(200.0, 2400.0, 8)
        rs = oil.compute_gas_to_oil_ratio(pressure, 200.0, 2500.0, 0.75, 35.0)
        assert np.all(np.diff(rs) > 0)

    def test_standing_solution_gor_matches_published_form(self):
        pressure, t, api, gg = 2000.0, 200.0, 35.0, 0.75
        expected = gg * ((pressure / 18.2 + 1.4) * 10 ** (0.0125 * api - 0.00091 * t)) ** 1.2048
        result = oil.compute_gas_to_oil_ratio_standing(pressure, t, api, gg)
        assert result == pytest.approx(expected)
        assert 300.0 < result < 700.0

    def test_standing_bubble_point_is_exact_inverse_of_standing_gor(self):
        api, gg = 35.0, 0.75
        for rs, t in [(200.0, 150.0), (600.0, 200.0), (1000.0, 250.0)]:
            pb = oil.estimate_bubble_point_pressure_standing(api, gg, rs, t)
            assert oil.compute_gas_to_oil_ratio_standing(pb, t, api, gg) == pytest.approx(
                rs, rel=1e-9
            )

    def test_standing_bubble_point_never_below_standard_pressure(self):
        assert oil.estimate_bubble_point_pressure_standing(
            35.0, 0.75, 0.01, 100.0
        ) == pytest.approx(14.696, rel=1e-4)

    def test_arrays_match_scalars(self, conditions):
        c = conditions
        arrays = oil.compute_oil_bubble_point_pressure(
            c["gas_gravity"], c["oil_api_gravity"], c["temperature"], c["gas_to_oil_ratio"]
        )
        for i in range(len(arrays)):
            scalar = oil.compute_oil_bubble_point_pressure(
                float(c["gas_gravity"][i]),
                float(c["oil_api_gravity"][i]),
                float(c["temperature"][i]),
                float(c["gas_to_oil_ratio"][i]),
            )
            assert arrays[i] == pytest.approx(scalar)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValidationError):
            oil.compute_oil_bubble_point_pressure(0.75, 35.0, 20.0, 600.0)
        with pytest.raises(ValidationError):
            oil.compute_gas_to_oil_ratio(0.0, 200.0, 2500.0, 0.75, 35.0)
        with pytest.raises(ValidationError):
            oil.compute_gas_to_oil_ratio_standing(2000.0, 200.0, 5.0, 0.75)


class TestViscosity:
    def test_dead_oil_viscosity_matches_egbogah_ng(self):
        sg, t = 0.85, 200.0
        z = 1.8653 - 0.025086 * api_of(sg) - 0.5644 * np.log10(t)
        expected = 10 ** (10**z) - 1
        result = oil.compute_dead_oil_viscosity_modified_beggs(t, sg)
        assert result == pytest.approx(expected)
        assert result == pytest.approx(2.08, abs=0.01)

    def test_dead_oil_viscosity_falls_with_temperature_and_api(self):
        assert oil.compute_dead_oil_viscosity_modified_beggs(
            250.0, 0.85
        ) < oil.compute_dead_oil_viscosity_modified_beggs(150.0, 0.85)
        assert oil.compute_dead_oil_viscosity_modified_beggs(
            200.0, 0.80
        ) < oil.compute_dead_oil_viscosity_modified_beggs(200.0, 0.92)

    def test_saturated_viscosity_is_beggs_robinson(self):
        sg, t, rs = 0.85, 200.0, 500.0
        dead = oil.compute_dead_oil_viscosity_modified_beggs(t, sg)
        a = 10.715 * (rs + 100.0) ** -0.515
        b = 5.44 * (rs + 150.0) ** -0.338
        result = oil.compute_oil_viscosity(2000.0, t, 2500.0, sg, rs, 600.0)
        assert result == pytest.approx(a * dead**b)

    def test_undersaturated_viscosity_rises_with_pressure(self):
        kw = dict(temperature=200.0, bubble_point_pressure=2500.0, oil_specific_gravity=0.85)
        at_bubble_point = oil.compute_oil_viscosity(
            pressure=2500.0, gas_to_oil_ratio=600.0, gor_at_bubble_point_pressure=600.0, **kw
        )
        above = oil.compute_oil_viscosity(
            pressure=5000.0, gas_to_oil_ratio=600.0, gor_at_bubble_point_pressure=600.0, **kw
        )
        assert above > at_bubble_point

    def test_undersaturated_viscosity_follows_vazquez_beggs(self):
        t, sg, pb, p, rsb = 200.0, 0.85, 2500.0, 4000.0, 600.0
        dead = oil.compute_dead_oil_viscosity_modified_beggs(t, sg)
        mu_b = 10.715 * (rsb + 100.0) ** -0.515 * dead ** (5.44 * (rsb + 150.0) ** -0.338)
        m = 2.6 * p**1.187 * np.exp(-11.513 - 8.98e-5 * p)
        result = oil.compute_oil_viscosity(p, t, pb, sg, rsb, rsb)
        assert result == pytest.approx(mu_b * (p / pb) ** m)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValidationError):
            oil.compute_dead_oil_viscosity_modified_beggs(-10.0, 0.85)
        with pytest.raises(ValidationError):
            oil.compute_oil_viscosity(2000.0, 200.0, 2500.0, 0.85, -1.0, 600.0)


class TestCompressibility:
    args: typing.ClassVar[dict[str, float]] = dict(
        temperature=200.0,
        bubble_point_pressure=2500.0,
        oil_api_gravity=35.0,
        gas_gravity=0.75,
        gor_at_bubble_point_pressure=600.0,
    )

    def test_base_compressibility_is_vazquez_beggs(self):
        pressure = 4000.0
        expected = (-1433.0 + 5.0 * 600.0 + 17.2 * 200.0 - 1180.0 * 0.75 + 12.61 * 35.0) / (
            1e5 * pressure
        )
        assert oil.compute_base_compressibility(pressure=pressure, **self.args) == pytest.approx(
            expected
        )

    def test_undersaturated_compressibility_is_base_only(self):
        extra = dict(gas_formation_volume_factor=0.004, oil_formation_volume_factor=1.3)
        total = oil.compute_oil_compressibility(pressure=4000.0, **self.args, **extra)
        base = oil.compute_base_compressibility(pressure=4000.0, **self.args)
        assert total == pytest.approx(base)

    def test_saturated_compressibility_adds_the_liberation_term(self):
        extra = dict(gas_formation_volume_factor=0.004, oil_formation_volume_factor=1.3)
        total = oil.compute_oil_compressibility(pressure=2000.0, **self.args, **extra)
        base = oil.compute_base_compressibility(pressure=2000.0, **self.args)
        liberation = oil.compute_oil_compressibility_liberation_correction_term(
            pressure=2000.0,
            temperature=200.0,
            gas_gravity=0.75,
            oil_api_gravity=35.0,
            bubble_point_pressure=2500.0,
            gas_formation_volume_factor=0.004,
            oil_formation_volume_factor=1.3,
            gor_at_bubble_point_pressure=600.0,
        )
        assert liberation > 0
        assert total == pytest.approx(base + liberation)

    def test_liberation_term_matches_finite_difference(self):
        pressure = 1500.0

        def rs(p):
            return oil.compute_gas_to_oil_ratio(p, 200.0, 2500.0, 0.75, 35.0, 600.0)

        derivative = (rs(pressure + 1.0) - rs(pressure - 1.0)) / 2.0
        expected = 0.004 / 1.3 * derivative / 5.6146
        result = oil.compute_oil_compressibility_liberation_correction_term(
            pressure=pressure,
            temperature=200.0,
            gas_gravity=0.75,
            oil_api_gravity=35.0,
            bubble_point_pressure=2500.0,
            gas_formation_volume_factor=0.004,
            oil_formation_volume_factor=1.3,
            gor_at_bubble_point_pressure=600.0,
        )
        assert result == pytest.approx(expected, rel=1e-3)

    def test_compressibility_is_never_negative(self):
        assert (
            oil.compute_base_compressibility(pressure=5000.0, **{**self.args, "temperature": 60.0})
            >= 0
        )


class TestLiveOilDensity:
    def test_mass_balance_on_one_stock_tank_barrel(self):
        api, gg, rs, bo = 35.0, 0.75, 600.0, 1.3
        stock_tank = 141.5 / (api + 131.5) * 62.37
        mass = stock_tank * 5.6146 + rs * gg * 28.9647 / 379.49
        assert oil.compute_live_oil_density(api, gg, rs, bo) == pytest.approx(
            mass / (bo * 5.6146), rel=2e-3
        )

    def test_dissolved_gas_with_same_volume_makes_oil_denser(self):
        assert oil.compute_live_oil_density(35.0, 0.75, 800.0, 1.3) > oil.compute_live_oil_density(
            35.0, 0.75, 0.0, 1.3
        )

    def test_dead_oil_density_at_unit_fvf_is_stock_tank_density(self):
        assert oil.compute_live_oil_density(35.0, 0.75, 0.0, 1.0) == pytest.approx(
            141.5 / 166.5 * 62.37, rel=1e-3
        )

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValidationError):
            oil.compute_live_oil_density(35.0, 0.75, 600.0, 0.0)


class TestMiscibility:
    def test_transition_factor_values(self):
        assert oil.compute_miscibility_transition_factor(2500.0, 2500.0, 500.0) == pytest.approx(
            0.5
        )
        assert oil.compute_miscibility_transition_factor(3000.0, 2500.0, 500.0) == pytest.approx(
            0.5 * (1 + np.tanh(1.0))
        )
        assert oil.compute_miscibility_transition_factor(2000.0, 2500.0, 500.0) == pytest.approx(
            0.5 * (1 - np.tanh(1.0))
        )

    def test_transition_factor_is_smooth_and_monotonic(self):
        pressure = np.linspace(500.0, 6000.0, 400)
        factor = oil.compute_miscibility_transition_factor(pressure, 2500.0, 500.0)
        assert np.all(np.diff(factor) > 0)
        assert np.max(np.abs(np.diff(factor))) < 0.015

    def test_default_width_is_500(self):
        assert oil.compute_miscibility_transition_factor(2800.0, 2500.0) == pytest.approx(
            oil.compute_miscibility_transition_factor(2800.0, 2500.0, 500.0)
        )

    def test_effective_omega_scales_base_value(self):
        factor = oil.compute_miscibility_transition_factor(3000.0, 2500.0, 500.0)
        assert oil.compute_effective_todd_longstaff_omega(
            3000.0, 0.67, 2500.0, 500.0
        ) == pytest.approx(0.67 * factor)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValidationError):
            oil.compute_miscibility_transition_factor(2500.0, 2500.0, 0.0)
        with pytest.raises(ValidationError):
            oil.compute_effective_todd_longstaff_omega(3000.0, 1.5, 2500.0)


class TestToddLongstaff:
    viscosities: typing.ClassVar[dict[str, float]] = dict(
        oil_viscosity=10.0, solvent_viscosity=0.05
    )
    densities: typing.ClassVar[dict[str, float]] = dict(oil_density=50.0, solvent_density=20.0)

    def test_mixture_viscosity_uses_quarter_power_rule(self):
        cs = 0.3
        expected = 10.0 * 0.05 / (cs * 10.0**0.25 + (1 - cs) * 0.05**0.25) ** 4
        assert oil.compute_todd_longstaff_mixture_viscosity(
            **self.viscosities, solvent_concentration=cs
        ) == pytest.approx(expected)

    def test_mixture_viscosity_limits_are_the_pure_fluids(self):
        assert oil.compute_todd_longstaff_mixture_viscosity(
            **self.viscosities, solvent_concentration=0.0
        ) == pytest.approx(10.0)
        assert oil.compute_todd_longstaff_mixture_viscosity(
            **self.viscosities, solvent_concentration=1.0
        ) == pytest.approx(0.05)

    def test_effective_viscosities_at_omega_limits(self):
        kw = dict(**self.viscosities, solvent_concentration=0.3)
        mixture = oil.compute_todd_longstaff_mixture_viscosity(**kw)
        assert oil.compute_todd_longstaff_effective_oil_viscosity(
            **kw, omega=0.0
        ) == pytest.approx(10.0)
        assert oil.compute_todd_longstaff_effective_oil_viscosity(
            **kw, omega=1.0
        ) == pytest.approx(mixture)
        assert oil.compute_todd_longstaff_effective_solvent_viscosity(
            **kw, omega=0.0
        ) == pytest.approx(0.05)
        assert oil.compute_todd_longstaff_effective_solvent_viscosity(
            **kw, omega=1.0
        ) == pytest.approx(mixture)

    def test_effective_viscosity_is_geometric_interpolation(self):
        kw = dict(**self.viscosities, solvent_concentration=0.3)
        mixture = oil.compute_todd_longstaff_mixture_viscosity(**kw)
        omega = 0.67
        assert oil.compute_todd_longstaff_effective_oil_viscosity(
            **kw, omega=omega
        ) == pytest.approx(10.0 ** (1 - omega) * mixture**omega)
        assert oil.compute_todd_longstaff_effective_solvent_viscosity(
            **kw, omega=omega
        ) == pytest.approx(0.05 ** (1 - omega) * mixture**omega)

    def test_default_omega_is_two_thirds(self):
        kw = dict(**self.viscosities, solvent_concentration=0.3)
        assert oil.compute_todd_longstaff_effective_oil_viscosity(**kw) == pytest.approx(
            oil.compute_todd_longstaff_effective_oil_viscosity(**kw, omega=0.67)
        )

    def test_effective_densities_at_omega_limits(self):
        kw = dict(**self.densities, **self.viscosities, solvent_concentration=0.3)
        volume_average = 0.7 * 50.0 + 0.3 * 20.0
        assert oil.compute_todd_longstaff_effective_oil_density(**kw, omega=0.0) == pytest.approx(
            50.0
        )
        assert oil.compute_todd_longstaff_effective_solvent_density(
            **kw, omega=0.0
        ) == pytest.approx(20.0)
        assert oil.compute_todd_longstaff_effective_oil_density(**kw, omega=1.0) == pytest.approx(
            volume_average
        )
        assert oil.compute_todd_longstaff_effective_solvent_density(
            **kw, omega=1.0
        ) == pytest.approx(volume_average)

    def test_effective_densities_stay_between_the_pure_densities(self):
        kw = dict(**self.densities, **self.viscosities, solvent_concentration=0.3)
        for omega in np.linspace(0, 1, 6):
            assert (
                20.0 <= oil.compute_todd_longstaff_effective_oil_density(**kw, omega=omega) <= 50.0
            )
            assert (
                20.0
                <= oil.compute_todd_longstaff_effective_solvent_density(**kw, omega=omega)
                <= 50.0
            )

    def test_equal_viscosities_do_not_divide_by_zero(self):
        kw = dict(
            **self.densities, oil_viscosity=1.0, solvent_viscosity=1.0, solvent_concentration=0.3
        )
        assert np.isfinite(oil.compute_todd_longstaff_effective_oil_density(**kw, omega=0.5))
        assert np.isfinite(oil.compute_todd_longstaff_effective_solvent_density(**kw, omega=0.5))

    def test_arrays(self):
        concentration = np.linspace(0.0, 1.0, 5)
        result = oil.compute_todd_longstaff_mixture_viscosity(10.0, 0.05, concentration)
        assert result.shape == (5,)
        assert np.all(np.diff(result) < 0)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValidationError):
            oil.compute_todd_longstaff_mixture_viscosity(10.0, 0.05, 1.5)
        with pytest.raises(ValidationError):
            oil.compute_todd_longstaff_effective_oil_viscosity(0.0, 0.05, 0.3)
        with pytest.raises(ValidationError):
            oil.compute_todd_longstaff_effective_oil_density(50.0, 20.0, 10.0, 0.05, 0.3, 2.0)

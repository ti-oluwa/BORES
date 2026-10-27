import typing

import numpy as np
import pytest

from bores.correlations import core, water
from bores.errors import ComputationError, ValidationError


def mccain_bw(pressure, temperature):
    delta_t = -0.010001 + 1.33391e-4 * temperature + 5.50654e-7 * temperature**2
    delta_p = (
        -1.95301e-9 * pressure * temperature
        - 1.72834e-13 * pressure**2 * temperature
        - 3.58922e-7 * pressure
        - 2.25341e-10 * pressure**2
    )
    return (1 + delta_t) * (1 + delta_p)


class TestStandardDensity:
    def test_fresh_water(self):
        assert water.compute_water_standard_density(0.0) == pytest.approx(62.368)

    def test_salinity_raises_density(self):
        weight_percent = 3.0
        expected = 62.368 + 0.438603 * weight_percent + 1.60074e-3 * weight_percent**2
        assert water.compute_water_standard_density(30000.0) == pytest.approx(expected)

    def test_negative_salinity_raises(self):
        with pytest.raises(ValidationError):
            water.compute_water_standard_density(-1.0)


class TestViscosity:
    def test_mccain_reference_values(self):
        for temperature, expected in [(100.0, 0.626), (200.0, 0.288), (300.0, 0.182)]:
            result = water.compute_water_viscosity(temperature, 0.0, 14.696)
            assert result == pytest.approx(expected * (0.9994 + 4.0295e-5 * 14.696), rel=2e-2)

    def test_formula(self):
        t, s_ppm, p = 200.0, 30000.0, 3000.0
        s = s_ppm / 1e4
        a = 109.574 - 0.840564 * s + 0.313314 * s**2 + 8.72213e-3 * s**3
        b = -1.12166 + 2.63951e-2 * s - 6.79461e-4 * s**2 - 5.47119e-5 * s**3 + 1.55586e-6 * s**4
        expected = a * t**b * (0.9994 + 4.0295e-5 * p + 3.1062e-9 * p**2)
        assert water.compute_water_viscosity(t, s_ppm, p) == pytest.approx(expected)

    def test_falls_with_temperature_rises_with_salinity(self):
        assert water.compute_water_viscosity(250.0, 0.0, 3000.0) < water.compute_water_viscosity(
            150.0, 0.0, 3000.0
        )
        assert water.compute_water_viscosity(
            200.0, 50000.0, 3000.0
        ) > water.compute_water_viscosity(200.0, 0.0, 3000.0)

    def test_matches_coolprop_within_the_accuracy_of_the_correlation(self):
        pytest.importorskip("CoolProp")
        reference = core.compute_fluid_viscosity(14.696, 200.0, "Water")
        assert water.compute_water_viscosity(200.0, 0.0, 14.696) == pytest.approx(
            reference, rel=0.1
        )

    def test_arrays_and_invalid_inputs(self, conditions):
        result = water.compute_water_viscosity(
            conditions["temperature"], conditions["salinity"], conditions["pressure"]
        )
        assert result.shape == conditions["pressure"].shape
        assert np.all(result > 0)
        with pytest.raises(ValidationError):
            water.compute_water_viscosity(0.0, 0.0, 3000.0)


class TestFormationVolumeFactor:
    def test_matches_mccain(self):
        for pressure, temperature in [(14.696, 60.0), (3000.0, 200.0), (5000.0, 250.0)]:
            assert water.compute_gas_free_water_formation_volume_factor(
                pressure, temperature
            ) == pytest.approx(mccain_bw(pressure, temperature))

    def test_is_close_to_one_at_standard_conditions(self):
        assert water.compute_gas_free_water_formation_volume_factor(14.696, 60.0) == pytest.approx(
            1.0, abs=1e-3
        )

    def test_thermal_expansion_and_pressure_compression(self):
        assert water.compute_gas_free_water_formation_volume_factor(
            3000.0, 250.0
        ) > water.compute_gas_free_water_formation_volume_factor(3000.0, 150.0)
        assert water.compute_gas_free_water_formation_volume_factor(
            5000.0, 200.0
        ) < water.compute_gas_free_water_formation_volume_factor(1000.0, 200.0)

    def test_temperature_is_in_fahrenheit_not_celsius(self):
        assert water.compute_gas_free_water_formation_volume_factor(
            3000.0, 200.0
        ) == pytest.approx(1.0339, abs=1e-4)

    def test_invalid_pressure_raises(self):
        with pytest.raises(ValidationError):
            water.compute_gas_free_water_formation_volume_factor(-1.0, 200.0)


class TestGasSolubility:
    def test_mccain_methane(self):
        t, p = 200.0, 3000.0
        a = 8.15839 - 6.12265e-2 * t + 1.91663e-4 * t**2 - 2.1654e-7 * t**3
        b = 1.01021e-2 - 7.44241e-5 * t + 3.05553e-7 * t**2 - 2.94883e-10 * t**3
        c = (
            -9.02505 + 0.130237 * t - 8.53425e-4 * t**2 + 2.34122e-6 * t**3 - 2.37049e-9 * t**4
        ) * 1e-7
        expected = a + b * p + c * p**2
        result = water.compute_gas_solubility_in_water_mccain_methane(p, t, 0.0)
        assert result == pytest.approx(expected)
        assert result == pytest.approx(15.1, abs=0.3)

    def test_mccain_salting_out(self):
        fresh = water.compute_gas_solubility_in_water_mccain_methane(3000.0, 200.0, 0.0)
        brine = water.compute_gas_solubility_in_water_mccain_methane(3000.0, 200.0, 30000.0)
        assert brine == pytest.approx(fresh * 10 ** (-0.0840655 * 3.0 * 200.0**-0.285854))
        assert brine < fresh

    def test_henry_law_formula(self):
        kh, coefficient, ks = 1.4e-5, 1600.0, 0.11
        t_k = (200.0 - 32.0) / 1.8 + 273.15
        concentration = kh * np.exp(coefficient * (1 / t_k - 1 / 298.15)) * 3000.0 * 6894.757
        result = water.compute_gas_solubility_in_water_henry_law(
            3000.0, 200.0, 0.0, kh, coefficient, ks
        )
        batzle = water.compute_water_density_batzle(3000.0, 200.0, 0.0) * 16.01846
        molality = concentration / batzle
        expected = molality * 0.158987294928 * 999.0 * 379.49 / 453.59237
        assert result == pytest.approx(expected, rel=1e-3)

    def test_henry_law_salting_out_uses_base_ten(self):
        fresh = water.compute_gas_solubility_in_water_henry_law(
            3000.0, 200.0, 0.0, 1.4e-5, 1600.0, 0.11
        )
        brine = water.compute_gas_solubility_in_water_henry_law(
            3000.0, 200.0, 58440.0, 1.4e-5, 1600.0, 0.11
        )
        salt_mass_fraction = 0.05844
        molality = salt_mass_fraction / (1 - salt_mass_fraction) * 1000 / 58.44
        assert brine == pytest.approx(fresh * 10 ** (-0.11 * molality), rel=1e-9)

    def test_henry_law_agrees_with_mccain_for_methane_within_twenty_percent(self):
        henry = water.compute_gas_solubility_in_water(3000.0, 200.0, 0.0, "ch4")
        mccain = water.compute_gas_solubility_in_water_mccain_methane(3000.0, 200.0, 0.0)
        forced_henry = water.compute_gas_solubility_in_water_henry_law(
            3000.0, 200.0, 0.0, *core.HENRY_CONSTANTS["ch4"], core.SETSCHENOW_CONSTANTS["ch4"]
        )
        assert henry == pytest.approx(mccain)
        assert forced_henry == pytest.approx(mccain, rel=0.2)

    def test_dispatcher_uses_mccain_for_methane_in_range_and_henry_outside(self):
        inside = water.compute_gas_solubility_in_water(3000.0, 200.0, 0.0, "methane")
        outside = water.compute_gas_solubility_in_water(3000.0, 80.0, 0.0, "methane")
        assert inside == pytest.approx(
            water.compute_gas_solubility_in_water_mccain_methane(3000.0, 200.0, 0.0)
        )
        assert outside == pytest.approx(
            water.compute_gas_solubility_in_water_henry_law(
                3000.0, 80.0, 0.0, *core.HENRY_CONSTANTS["ch4"], core.SETSCHENOW_CONSTANTS["ch4"]
            )
        )

    def test_dispatcher_handles_arrays_with_mixed_temperatures(self):
        temperature = np.array([80.0, 200.0, 450.0])
        result = water.compute_gas_solubility_in_water(3000.0, temperature, 0.0, "methane")
        assert result.shape == (3,)
        assert np.all(result > 0)

    def test_other_gases_give_physical_magnitudes(self):
        nitrogen = water.compute_gas_solubility_in_water(3000.0, 200.0, 0.0, "n2")
        carbon_dioxide = water.compute_gas_solubility_in_water(3000.0, 200.0, 0.0, "CO2")
        assert 3.0 < nitrogen < 12.0
        assert 100.0 < carbon_dioxide < 400.0
        assert carbon_dioxide > nitrogen

    def test_solubility_rises_with_pressure(self):
        pressure = np.linspace(500.0, 5000.0, 6)
        for gas_name in ("methane", "n2", "co2", "h2"):
            result = water.compute_gas_solubility_in_water(pressure, 200.0, 20000.0, gas_name)
            assert np.all(np.diff(result) > 0)

    def test_unsupported_gas_raises(self):
        with pytest.raises(ValidationError, match="not available"):
            water.compute_gas_solubility_in_water(3000.0, 200.0, 0.0, "xenon")

    def test_derivative_matches_finite_difference(self):
        result = water.compute_gas_solubility_in_water_pressure_derivative(
            3000.0, 200.0, 0.0, "methane"
        )
        reference = (
            water.compute_gas_solubility_in_water(3001.0, 200.0, 0.0, "methane")
            - water.compute_gas_solubility_in_water(2999.0, 200.0, 0.0, "methane")
        ) / 2.0
        assert result == pytest.approx(reference, rel=1e-3)
        assert result > 0

    def test_derivative_accepts_arrays(self):
        result = water.compute_gas_solubility_in_water_pressure_derivative(
            np.array([500.0, 3000.0]), 200.0, 0.0, "co2"
        )
        assert result.shape == (2,)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValidationError):
            water.compute_gas_solubility_in_water_mccain_methane(-1.0, 200.0, 0.0)
        with pytest.raises(ValidationError):
            water.compute_gas_solubility_in_water_henry_law(3000.0, 200.0, 0.0, 0.0, 1600.0, 0.1)


class TestBubblePointPressure:
    @pytest.mark.parametrize("gas_name", ["methane", "co2", "n2"])
    def test_inverts_the_solubility(self, gas_name):
        pressure = np.array([800.0, 2500.0, 4800.0])
        temperature = np.array([150.0, 200.0, 250.0])
        salinity = np.array([0.0, 30000.0, 90000.0])
        solubility = water.compute_gas_solubility_in_water(
            pressure, temperature, salinity, gas_name
        )
        result = water.compute_water_bubble_point_pressure(
            temperature, solubility, salinity, gas_name
        )
        np.testing.assert_allclose(result, pressure, rtol=1e-9)

    def test_scalar_in_scalar_out(self):
        solubility = water.compute_gas_solubility_in_water(3000.0, 200.0, 0.0, "methane")
        result = water.compute_water_bubble_point_pressure(200.0, solubility, 0.0, "methane")
        assert np.ndim(result) == 0
        assert result == pytest.approx(3000.0, rel=1e-9)

    def test_unreachable_solubility_raises(self):
        with pytest.raises(ComputationError):
            water.compute_water_bubble_point_pressure(200.0, 1e6, 0.0, "methane")
        with pytest.raises(ComputationError):
            water.compute_water_bubble_point_pressure(200.0, 0.0, 0.0, "methane")


class TestCompressibility:
    base: typing.ClassVar[dict[str, float]] = dict(
        pressure=3000.0,
        temperature=200.0,
        gas_formation_volume_factor=0.005,
        gas_solubility_in_water=15.0,
        gas_solubility_pressure_derivative=0.004,
        salinity=0.0,
    )

    def test_undersaturated_value_follows_mccain_with_dissolved_gas_correction(self):
        t = 200.0

        def bw(p):
            return mccain_bw(p, t)

        pure = -(bw(3001.0) - bw(2999.0)) / 2.0 / bw(3000.0)
        result = water.compute_water_compressibility(bubble_point_pressure=2000.0, **self.base)
        assert result == pytest.approx(pure * (1 + 8.9e-3 * 15.0), rel=1e-4)
        assert 2e-6 < result < 4e-6

    def test_saturated_water_adds_the_gas_liberation_term(self):
        undersaturated = water.compute_water_compressibility(
            bubble_point_pressure=2000.0, **self.base
        )
        saturated = water.compute_water_compressibility(bubble_point_pressure=3000.0, **self.base)
        liberation = 0.005 / mccain_bw(3000.0, 200.0) * 0.004 / 5.6146
        assert saturated == pytest.approx(undersaturated + liberation, rel=1e-4)

    def test_pressure_equal_to_bubble_point_within_rounding_counts_as_saturated(self):
        exact = water.compute_water_compressibility(bubble_point_pressure=3000.0, **self.base)
        rounded = water.compute_water_compressibility(
            bubble_point_pressure=3000.0 * (1 + 1e-9), **self.base
        )
        assert rounded == pytest.approx(exact)

    def test_salinity_reduces_compressibility(self):
        fresh = water.compute_water_compressibility(bubble_point_pressure=2000.0, **self.base)
        brine = water.compute_water_compressibility(
            bubble_point_pressure=2000.0, **{**self.base, "salinity": 50000.0}
        )
        factor = -0.052 + 2.7e-4 * 200 - 1.14e-6 * 200**2 + 1.121e-9 * 200**3
        assert brine == pytest.approx(fresh * (1 + 5.0 * factor), rel=1e-9)
        assert brine < fresh

    def test_default_salinity_is_fresh(self):
        kw = {k: v for k, v in self.base.items() if k != "salinity"}
        assert water.compute_water_compressibility(
            bubble_point_pressure=2000.0, **kw
        ) == pytest.approx(
            water.compute_water_compressibility(bubble_point_pressure=2000.0, **self.base)
        )

    def test_matches_coolprop_order_of_magnitude(self):
        pytest.importorskip("CoolProp")
        reference = core.compute_fluid_compressibility(3000.0, 200.0, "Water")
        pure = water.compute_water_compressibility(
            bubble_point_pressure=2000.0, **{**self.base, "gas_solubility_in_water": 0.0}
        )
        assert pure == pytest.approx(reference, rel=0.35)

    def test_never_negative(self, conditions):
        c = conditions
        result = water.compute_water_compressibility(
            pressure=c["pressure"],
            temperature=c["temperature"],
            bubble_point_pressure=c["bubble_point_pressure"],
            gas_formation_volume_factor=np.full_like(c["pressure"], 0.005),
            gas_solubility_in_water=np.full_like(c["pressure"], 10.0),
            gas_solubility_pressure_derivative=np.full_like(c["pressure"], 0.004),
            salinity=c["salinity"],
        )
        assert np.all(result >= 0)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValidationError):
            water.compute_water_compressibility(
                bubble_point_pressure=2000.0, **{**self.base, "gas_formation_volume_factor": 0.0}
            )


class TestDensity:
    def test_mccain_density_is_standard_density_over_formation_volume_factor(self):
        expected = water.compute_water_standard_density(30000.0) / mccain_bw(3000.0, 200.0)
        assert water.compute_water_density_mccain(3000.0, 200.0, 30000.0) == pytest.approx(
            expected
        )

    def test_batzle_wang_is_close_to_coolprop_for_fresh_water(self):
        pytest.importorskip("CoolProp")
        for pressure, temperature in [(14.696, 60.0), (3000.0, 200.0), (6000.0, 250.0)]:
            reference = core.compute_fluid_density(pressure, temperature, "Water")
            result = water.compute_water_density_batzle(pressure, temperature, 0.0)
            assert result == pytest.approx(reference, rel=3e-3)

    def test_batzle_wang_density_depends_on_temperature_and_pressure(self):
        base = water.compute_water_density_batzle(3000.0, 200.0, 30000.0)
        assert water.compute_water_density_batzle(3000.0, 300.0, 30000.0) < base
        assert water.compute_water_density_batzle(6000.0, 200.0, 30000.0) > base

    def test_batzle_wang_brine_published_value(self):
        # 10% NaCl at 20 °C is 1.0707 g/cm³ (CRC); the correlation is good to about 0.3%.
        result = water.compute_water_density_batzle(14.5038, 68.0, 100000.0)
        assert result / 62.428 == pytest.approx(1.0707, abs=4e-3)

    def test_mccain_and_batzle_agree_within_one_percent(self, conditions):
        c = conditions
        mccain = water.compute_water_density_mccain(c["pressure"], c["temperature"], c["salinity"])
        batzle = water.compute_water_density_batzle(c["pressure"], c["temperature"], c["salinity"])
        np.testing.assert_allclose(mccain, batzle, rtol=1.2e-2)

    def test_live_density_mass_balance(self):
        bw = mccain_bw(3000.0, 200.0)
        standard = water.compute_water_standard_density(30000.0)
        gas_mass = 12.0 * 0.7 * 28.9647 / 379.49
        expected = (standard * 5.6146 + gas_mass) / (bw * 5.6146)
        result = water.compute_water_density(3000.0, 200.0, 0.7, 30000.0, 12.0, bw)
        assert result == pytest.approx(expected, rel=2e-3)

    def test_live_density_exceeds_gas_free_density(self):
        bw = mccain_bw(3000.0, 200.0)
        free = water.compute_water_density(3000.0, 200.0, 0.7, 30000.0, 0.0, bw)
        live = water.compute_water_density(3000.0, 200.0, 0.7, 30000.0, 20.0, bw)
        assert live > free
        assert free == pytest.approx(water.compute_water_density_mccain(3000.0, 200.0, 30000.0))

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValidationError):
            water.compute_water_density(3000.0, 200.0, 0.0, 30000.0, 12.0, 1.03)
        with pytest.raises(ValidationError):
            water.compute_water_density_batzle(-1.0, 200.0, 0.0)


class TestWaterFormationVolumeFactor:
    def test_inverts_the_density(self):
        bw = mccain_bw(3000.0, 200.0)
        density = water.compute_water_density_mccain(3000.0, 200.0, 30000.0)
        assert water.compute_water_formation_volume_factor(density, 30000.0) == pytest.approx(bw)

    def test_realistic_value(self):
        density = water.compute_water_density_batzle(3000.0, 200.0, 30000.0)
        assert water.compute_water_formation_volume_factor(density, 30000.0) == pytest.approx(
            1.03, abs=0.01
        )

    def test_user_supplied_standard_density_is_used(self):
        default = water.compute_water_formation_volume_factor(60.0, 30000.0)
        custom = water.compute_water_formation_volume_factor(60.0, 30000.0, 63.0)
        assert custom == pytest.approx(63.0 / 60.0)
        assert default == pytest.approx(water.compute_water_standard_density(30000.0) / 60.0)

    def test_invalid_density_raises(self):
        with pytest.raises(ValidationError):
            water.compute_water_formation_volume_factor(0.0, 30000.0)

import typing

import numpy as np
import pytest

from bores.correlations import core
from bores.errors import ValidationError

coolprop = pytest.importorskip("CoolProp")


class TestTemperatureConversions:
    def test_known_points(self):
        assert core.fahrenheit_to_kelvin(32.0) == pytest.approx(273.15)
        assert core.fahrenheit_to_kelvin(212.0) == pytest.approx(373.15)
        assert core.fahrenheit_to_celsius(212.0) == pytest.approx(100.0)
        assert core.fahrenheit_to_rankine(0.0) == pytest.approx(459.67)
        assert core.kelvin_to_fahrenheit(273.15) == pytest.approx(32.0)

    def test_round_trip_and_arrays(self):
        temperature = np.linspace(-40.0, 400.0, 11)
        np.testing.assert_allclose(
            core.kelvin_to_fahrenheit(core.fahrenheit_to_kelvin(temperature)), temperature
        )


class TestInputValidation:
    def test_valid_values_pass(self):
        core.validate_input_temperature(np.array([100.0, 200.0]))
        core.validate_input_pressure(np.array([500.0, 5000.0]))

    def test_out_of_range_raises(self):
        with pytest.raises(ValidationError, match="Temperature"):
            core.validate_input_temperature(1e6)
        with pytest.raises(ValidationError, match="Pressure"):
            core.validate_input_pressure(-5.0)


class TestCoolPropHelpers:
    def test_propssi_forwards(self):
        assert core.PropsSI("D", "T", 300, "P", 101325, "Water") == pytest.approx(996.5, rel=1e-2)

    def test_is_coolprop_fluid(self):
        assert core.is_CoolProp_fluid("Water")
        assert not core.is_CoolProp_fluid("NotAFluid")

    def test_clipping_limits_to_valid_range(self):
        assert core.clip_pressure(1e-3, "Water") > 0
        assert core.clip_temperature(1.0, "Water") > 200.0
        assert core.clip_temperature(300.0, "Water") == pytest.approx(300.0)

    def test_water_properties_at_standard_conditions(self):
        assert core.compute_fluid_density(14.696, 60.0, "Water") == pytest.approx(62.37, rel=5e-3)
        assert core.compute_fluid_viscosity(14.696, 60.0, "Water") == pytest.approx(1.12, rel=2e-2)
        assert core.compute_fluid_compressibility(14.696, 60.0, "Water") == pytest.approx(
            3.2e-6, rel=5e-2
        )

    def test_compressibility_factor_of_methane_is_near_one_at_low_pressure(self):
        assert core.compute_fluid_compressibility_factor(14.696, 60.0, "Methane") == pytest.approx(
            1.0, abs=5e-3
        )

    def test_arrays_broadcast_and_scalars_return_scalars(self):
        pressure = np.array([[500.0], [3000.0]])
        temperature = np.array([100.0, 200.0, 300.0])
        density = core.compute_fluid_density(pressure, temperature, "Water")
        assert density.shape == (2, 3)
        assert np.ndim(core.compute_fluid_density(500.0, 100.0, "Water")) == 0
        assert density[0, 0] == pytest.approx(core.compute_fluid_density(500.0, 100.0, "Water"))

    def test_evaluate_coolprop_returns_si_units(self):
        density = core.evaluate_coolprop("D", 14.696, 60.0, "Water")
        assert density == pytest.approx(999.0, rel=2e-3)

    def test_gas_below_the_triple_point_pressure_is_not_clipped_upward(self):
        # CO2 melts above 5.2 bar, but its gas phase at 1 atm is well defined.
        density = core.compute_fluid_density(14.696, 60.0, "CO2")
        assert density == pytest.approx(0.1166, rel=1e-2)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.longdouble])
    def test_result_dtype_follows_the_inputs(self, dtype):
        result = core.compute_fluid_density(
            np.full(2, 3000.0, dtype), np.full(2, 200.0, dtype), "Water"
        )
        assert result.dtype == dtype

    def test_mixed_precision_inputs_promote(self):
        pressure = np.full(2, 3000.0, dtype=np.float32)
        assert core.compute_fluid_density(pressure, np.full(2, 200.0), "Water").dtype == np.float64
        assert (
            core.compute_fluid_density(np.full(2, 3000.0), np.float32(200.0), "Water").dtype
            == np.float64
        )

    def test_float32_input_gives_float32(self):
        result = core.compute_fluid_density(
            np.array([500.0, 1000.0], dtype=np.float32), np.float32(150.0), "Water"
        )
        assert result.dtype == np.float32


class TestTotalCompressibility:
    def test_two_phase(self):
        total = core.compute_total_fluid_compressibility(
            water_saturation=0.3,
            oil_saturation=0.7,
            water_compressibility=3e-6,
            oil_compressibility=1e-5,
        )
        assert total == pytest.approx(0.3 * 3e-6 + 0.7 * 1e-5)

    def test_three_phase(self):
        total = core.compute_total_fluid_compressibility(
            water_saturation=0.3,
            oil_saturation=0.5,
            water_compressibility=3e-6,
            oil_compressibility=1e-5,
            gas_saturation=0.2,
            gas_compressibility=1e-4,
        )
        assert total == pytest.approx(0.3 * 3e-6 + 0.5 * 1e-5 + 0.2 * 1e-4)


class TestHydrocarbonInPlace:
    base: typing.ClassVar[dict[str, float]] = dict(
        area=100.0,
        thickness=50.0,
        porosity=0.2,
        phase_saturation=0.7,
        formation_volume_factor=1.2,
    )

    def test_oil_in_place(self):
        expected = 7758.0 * 100.0 * 50.0 * 0.2 * 0.7 / 1.2
        assert core.compute_fluid_in_place(**self.base) == pytest.approx(expected)

    def test_gas_in_place_uses_cubic_feet_per_acre_foot(self):
        base = {**self.base, "formation_volume_factor": 0.005}
        expected = 43560.0 * 100.0 * 50.0 * 0.2 * 0.7 / 0.005
        result = core.compute_fluid_in_place(**base, hydrocarbon_type="gas")
        assert result == pytest.approx(expected)

    def test_water_in_place_and_net_to_gross(self):
        full = core.compute_fluid_in_place(**self.base, hydrocarbon_type="water")
        half = core.compute_fluid_in_place(
            **self.base, hydrocarbon_type="water", net_to_gross_ratio=0.5
        )
        assert half == pytest.approx(0.5 * full)

    def test_arrays(self):
        base = {**self.base, "area": np.array([10.0, 20.0])}
        result = core.compute_fluid_in_place(**base)
        assert result[1] == pytest.approx(2 * result[0])

    @pytest.mark.parametrize(
        "change",
        [
            {"area": 0.0},
            {"thickness": -1.0},
            {"porosity": 1.5},
            {"phase_saturation": -0.1},
            {"net_to_gross_ratio": 2.0},
            {"formation_volume_factor": 0.0},
            {"hydrocarbon_type": "steam"},
        ],
    )
    def test_invalid_inputs_raise(self, change):
        with pytest.raises(ValidationError):
            core.compute_fluid_in_place(**{**self.base, **change})


class TestGasNames:
    @pytest.mark.parametrize(
        ("name", "symbol"),
        [
            ("Methane", "ch4"),
            ("CO2", "co2"),
            ("carbon dioxide", "co2"),
            ("N2", "n2"),
            ("H-2", "h2"),
        ],
    )
    def test_symbol_lookup(self, name, symbol):
        assert core.get_gas_symbol(name) == symbol

    def test_henry_reference_temperature_is_25_celsius(self):
        assert core.HENRY_REFERENCE_TEMPERATURE == pytest.approx(25.0 + 273.15)

    def test_aliases_point_at_known_gases(self):
        assert set(core.GAS_ALIASES.values()) <= set(core.HENRY_CONSTANTS)

    def test_every_henry_gas_has_a_setschenow_constant(self):
        assert set(core.HENRY_CONSTANTS) == set(core.SETSCHENOW_CONSTANTS)

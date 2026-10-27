import inspect

import numpy as np
import pytest

import bores.correlations as correlations
from bores.correlations.elementwise import Elementwise

N = 12


def make_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    r = rng.random(N)
    return {
        "pressure": 800.0 + 4200.0 * r,
        "current_pressure": 800.0 + 4200.0 * r,
        "temperature": 130.0 + 170.0 * rng.random(N),
        "bubble_point_pressure": 2200.0 + 1200.0 * rng.random(N),
        "oil_specific_gravity": 0.78 + 0.12 * r,
        "oil_api_gravity": 25.0 + 20.0 * r,
        "api_gravity": 25.0 + 20.0 * r,
        "gas_gravity": 0.6 + 0.3 * rng.random(N),
        "gas_to_oil_ratio": 150.0 + 800.0 * r,
        "observed_gas_to_oil_ratio": 150.0 + 800.0 * r,
        "gor_at_bubble_point_pressure": 500.0 + 300.0 * r,
        "oil_compressibility": (5.0 + 10.0 * r) * 1e-6,
        "saturated_oil_fvf": 1.1 + 0.4 * r,
        "oil_formation_volume_factor": 1.2 + 0.3 * r,
        "formation_volume_factor": 1.2 + 0.3 * r,
        "oil_density": 45.0 + 5.0 * r,
        "gas_formation_volume_factor": 0.003 + 0.004 * r,
        "gas_compressibility_factor": 0.8 + 0.15 * r,
        "gas_density": 4.0 + 6.0 * r,
        "gas_molecular_weight": 16.0 + 10.0 * r,
        "salinity": 5000.0 + 90000.0 * rng.random(N),
        "gas_solubility_in_water": 5.0 + 15.0 * r,
        "gas_solubility_pressure_derivative": 0.002 + 0.004 * r,
        "gas_free_water_formation_volume_factor": 1.02 + 0.02 * r,
        "water_density": 60.0 + 2.0 * r,
        "oil_viscosity": 1.0 + 9.0 * r,
        "solvent_viscosity": 0.02 + 0.05 * r,
        "solvent_density": 15.0 + 10.0 * r,
        "solvent_concentration": 0.1 + 0.8 * r,
        "omega": 0.2 + 0.6 * rng.random(N),
        "base_omega": 0.2 + 0.6 * rng.random(N),
        "minimum_miscibility_pressure": 2500.0 + 500.0 * r,
        "henry_constant": 1e-5 + 1e-5 * r,
        "henry_temperature_coefficient": 1500.0 + 500.0 * r,
        "setschenow_constant": 0.1 + 0.05 * r,
    }


ELEMENTWISE = {
    name: value for name, value in vars(correlations).items() if isinstance(value, Elementwise)
}


def test_every_exported_name_exists():
    for name in correlations.__all__:
        assert hasattr(correlations, name), name


def test_correlations_are_registered():
    assert len(ELEMENTWISE) >= 40


@pytest.mark.parametrize("name", sorted(ELEMENTWISE))
def test_array_result_equals_scalar_results_and_is_finite(name):
    correlation = ELEMENTWISE[name]
    inputs = make_inputs()
    arguments = {}
    for parameter, spec in inspect.signature(correlation.py_func).parameters.items():
        if parameter in inputs:
            arguments[parameter] = inputs[parameter]
        elif spec.default is inspect.Parameter.empty:
            pytest.fail(f"no test input for argument {parameter!r} of {name}")
    result = correlation(**arguments)
    assert result.shape == (N,)
    assert np.all(np.isfinite(result)), name
    for i in range(N):
        scalar = correlation(**{key: float(value[i]) for key, value in arguments.items()})
        assert np.ndim(scalar) == 0
        assert scalar == pytest.approx(result[i], rel=1e-12, abs=1e-300), (name, i)


@pytest.mark.parametrize("name", sorted(ELEMENTWISE))
def test_positional_and_keyword_calls_agree(name):
    correlation = ELEMENTWISE[name]
    inputs = make_inputs()
    names = [
        parameter
        for parameter, spec in inspect.signature(correlation.py_func).parameters.items()
        if parameter in inputs
    ]
    required = [
        parameter
        for parameter, spec in inspect.signature(correlation.py_func).parameters.items()
        if spec.default is inspect.Parameter.empty
    ]
    keyword = {parameter: float(inputs[parameter][0]) for parameter in names}
    positional = [float(inputs[parameter][0]) for parameter in required]
    assert correlation(*positional, **{k: v for k, v in keyword.items() if k not in required}) == (
        pytest.approx(correlation(**keyword), rel=1e-12)
    )

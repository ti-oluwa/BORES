"""Shared fixtures for the correlation tests."""

import numpy as np
import pytest


@pytest.fixture
def conditions() -> dict[str, np.ndarray]:
    """Reservoir-like inputs spread over the valid range of the correlations."""
    rng = np.random.default_rng(12345)
    n = 25
    r = rng.random(n)
    oil_specific_gravity = 0.78 + 0.12 * r
    return {
        "pressure": 800.0 + 4200.0 * rng.random(n),
        "temperature": 120.0 + 180.0 * rng.random(n),
        "gas_gravity": 0.6 + 0.3 * rng.random(n),
        "oil_specific_gravity": oil_specific_gravity,
        "oil_api_gravity": 141.5 / oil_specific_gravity - 131.5,
        "gas_to_oil_ratio": 100.0 + 900.0 * r,
        "bubble_point_pressure": 2000.0 + 1500.0 * rng.random(n),
        "salinity": 5000.0 + 95000.0 * rng.random(n),
    }

"""Tests for how PVT table queries combine their inputs."""

import numpy as np
import pytest

from bores.blackoil.pvt.tables import broadcast_query_arrays
from bores.errors import ValidationError

PRESSURES = np.array([4700.0, 200.0, 3456.0, 10000.0, 4000.0])


def test_inputs_of_the_same_shape_are_paired_element_by_element():
    temperatures = np.array([200.0, 210.0, 220.0, 230.0, 240.0])
    pressure, temperature = broadcast_query_arrays(pressure=PRESSURES, temperature=temperatures)
    assert pressure.shape == temperature.shape == (5,)
    np.testing.assert_array_equal(pressure, PRESSURES)
    np.testing.assert_array_equal(temperature, temperatures)


def test_a_scalar_is_broadcast_against_an_array():
    pressure, temperature = broadcast_query_arrays(pressure=PRESSURES, temperature=200.0)
    assert pressure.shape == temperature.shape == (5,)
    np.testing.assert_array_equal(temperature, np.full(5, 200.0))


def test_a_column_and_a_row_give_a_grid():
    pressure, temperature = broadcast_query_arrays(
        pressure=PRESSURES[:, None], temperature=np.array([200.0, 400.0, 600.0])[None, :]
    )
    assert pressure.shape == temperature.shape == (5, 3)
    assert temperature[2, 1] == pytest.approx(400.0)
    assert pressure[2, 1] == pytest.approx(3456.0)


def test_the_order_of_the_inputs_is_kept():
    first, second, third = broadcast_query_arrays(a=1.0, b=np.array([1.0, 2.0]), c=3.0)
    assert (first[0], second[0], third[0]) == pytest.approx((1.0, 1.0, 3.0))


def test_unpairable_shapes_raise_a_clear_error():
    """The reported case: 5 pressures with 3 temperatures."""
    with pytest.raises(ValidationError) as error:
        broadcast_query_arrays(pressure=PRESSURES, temperature=np.array([200.0, 400.0, 600.0]))
    message = str(error.value)
    assert "pressure (5,)" in message
    assert "temperature (3,)" in message
    assert "pressure[:, None]" in message
    assert "temperature[None, :]" in message


def test_the_error_names_every_input():
    with pytest.raises(
        ValidationError, match=r"pressure \(5,\), temperature \(5,\), salinity \(3,\)"
    ):
        broadcast_query_arrays(pressure=PRESSURES, temperature=PRESSURES, salinity=np.ones(3))

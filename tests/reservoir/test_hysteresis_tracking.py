"""Tests for the saturation history tracker that feeds the hysteresis scanning curves."""

import numpy as np
import pytest

from bores.errors import ValidationError
from bores.reservoir.state import Hysteresis, advance_phase_hysteresis
from bores.reservoir.workspace import HysteresisWorkspace

WATER_PATH = [0.5, 0.3, 0.2, 0.35, 0.5, 0.4]
GAS_PATH = [0.10, 0.25, 0.40, 0.30, 0.20, 0.30, 0.45]


def _walk(
    *, water_path, gas_path, water_imbibition_when_increasing, gas_imbibition_when_increasing
):
    state = Hysteresis.from_saturation(
        water_saturation=np.array([water_path[0]]),
        gas_saturation=np.array([gas_path[0]]),
    )
    history = []
    for step in range(1, len(gas_path)):
        state = state.advance(
            water_saturation=np.array([water_path[step]]),
            gas_saturation=np.array([gas_path[step]]),
            previous_water_saturation=np.array([water_path[step - 1]]),
            previous_gas_saturation=np.array([gas_path[step - 1]]),
            water_imbibition_when_increasing=water_imbibition_when_increasing,
            gas_imbibition_when_increasing=gas_imbibition_when_increasing,
        )
        history.append(state)
    return history


def test_non_wetting_gas_imbibes_while_it_falls_from_its_peak():
    history = _walk(
        water_path=[0.2] * len(GAS_PATH),
        gas_path=GAS_PATH,
        water_imbibition_when_increasing=True,
        gas_imbibition_when_increasing=False,
    )
    assert [bool(state.gas_imbibition_flag[0]) for state in history] == [
        False,
        False,
        True,
        True,
        False,
        False,
    ]
    assert [float(state.max_gas_saturation[0]) for state in history] == pytest.approx([
        0.25,
        0.40,
        0.40,
        0.40,
        0.40,
        0.45,
    ])
    # The reversal is the peak, which is what Land turns into trapped gas.
    assert [float(state.gas_reversal_saturation[0]) for state in history] == pytest.approx([
        0.25,
        0.40,
        0.40,
        0.40,
        0.40,
        0.45,
    ])


def test_wetting_water_imbibes_while_it_rises_from_its_minimum():
    history = _walk(
        water_path=WATER_PATH,
        gas_path=[0.1] * len(WATER_PATH),
        water_imbibition_when_increasing=True,
        gas_imbibition_when_increasing=False,
    )
    assert [bool(state.water_imbibition_flag[0]) for state in history] == [
        False,
        False,
        True,
        True,
        False,
    ]
    assert [float(state.water_reversal_saturation[0]) for state in history] == pytest.approx([
        0.5,
        0.5,
        0.2,
        0.2,
        0.5,
    ])
    assert [float(state.max_water_saturation[0]) for state in history] == pytest.approx([
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
    ])


def test_a_change_inside_the_tolerance_does_not_flip_the_regime():
    state = Hysteresis.from_saturation(
        water_saturation=np.array([0.2]), gas_saturation=np.array([0.3])
    )
    wobbled = state.advance(
        water_saturation=np.array([0.2]),
        gas_saturation=np.array([0.3 - 1e-12]),
        previous_water_saturation=np.array([0.2]),
        previous_gas_saturation=np.array([0.3]),
        water_imbibition_when_increasing=True,
        gas_imbibition_when_increasing=False,
        tolerance=1e-6,
    )
    assert not wobbled.gas_imbibition_flag[0]


def test_each_cell_is_tracked_on_its_own():
    previous = np.array([0.1, 0.4, 0.4])
    new = np.array([0.3, 0.3, 0.4])
    flag = np.zeros(3, dtype=bool)
    maximum = previous.copy()
    reversal = previous.copy()
    new_max, new_flag, new_reversal = advance_phase_hysteresis(
        new_saturation=new,
        previous_saturation=previous,
        max_saturation=maximum,
        imbibition_flag=flag,
        reversal_saturation=reversal,
        imbibition_when_increasing=False,
    )
    np.testing.assert_array_equal(new_flag, [False, True, False])
    np.testing.assert_allclose(new_max, [0.3, 0.4, 0.4])
    np.testing.assert_allclose(new_reversal, [0.3, 0.4, 0.4])


def test_the_workspace_updates_in_place_and_matches_the_immutable_state():
    rng = np.random.default_rng(7)
    shape = (3, 4)
    water = rng.uniform(0.2, 0.6, shape)
    gas = rng.uniform(0.0, 0.3, shape)
    state = Hysteresis.from_saturation(water_saturation=water, gas_saturation=gas)
    workspace = HysteresisWorkspace.from_state(state)
    arrays = (
        workspace.max_water_saturation,
        workspace.max_gas_saturation,
        workspace.water_imbibition_flag,
        workspace.gas_imbibition_flag,
        workspace.water_reversal_saturation,
        workspace.gas_reversal_saturation,
    )
    previous_water, previous_gas = water, gas
    for _ in range(6):
        new_water = np.clip(previous_water + rng.normal(0.0, 0.05, shape), 0.0, 1.0)
        new_gas = np.clip(previous_gas + rng.normal(0.0, 0.05, shape), 0.0, 1.0)
        arguments = dict(
            water_saturation=new_water,
            gas_saturation=new_gas,
            previous_water_saturation=previous_water,
            previous_gas_saturation=previous_gas,
            water_imbibition_when_increasing=True,
            gas_imbibition_when_increasing=False,
        )
        state = state.advance(**arguments)
        workspace.update(**arguments)
        previous_water, previous_gas = new_water, new_gas

    snapshot = workspace.snapshot()
    for name in (
        "max_water_saturation",
        "max_gas_saturation",
        "water_imbibition_flag",
        "gas_imbibition_flag",
        "water_reversal_saturation",
        "gas_reversal_saturation",
    ):
        np.testing.assert_array_equal(getattr(snapshot, name), getattr(state, name))
    assert all(
        before is after
        for before, after in zip(
            arrays,
            (
                workspace.max_water_saturation,
                workspace.max_gas_saturation,
                workspace.water_imbibition_flag,
                workspace.gas_imbibition_flag,
                workspace.water_reversal_saturation,
                workspace.gas_reversal_saturation,
            ),
            strict=True,
        )
    )


def test_an_output_that_cannot_be_written_in_place_is_rejected():
    saturation = np.full(4, 0.3)
    with pytest.raises(ValidationError):
        advance_phase_hysteresis(
            new_saturation=saturation,
            previous_saturation=saturation,
            max_saturation=saturation,
            imbibition_flag=np.zeros(4, dtype=bool),
            reversal_saturation=saturation,
            imbibition_when_increasing=False,
            out_max_saturation=np.zeros(8)[::2],
        )

import numpy as np
import pytest

from bores.errors import ValidationError
from bores.grids.factories.cartesian import make_cartesian_grid
from bores.reservoir.state import (
    DEFAULT_DATAFRAME_FIELDS,
    HYSTERESIS_DATAFRAME_FIELDS,
    as_dataframe,
)
from bores.types import UnitSystem

from .helpers import N_CELLS, as_columns, make_grid, make_reservoir_state


def test_a_single_state_has_a_row_for_each_cell_and_no_label(backend):
    state = make_reservoir_state()
    columns = as_columns(as_dataframe(state, backend=backend))
    assert list(columns) == ["cell_index", *DEFAULT_DATAFRAME_FIELDS]
    assert columns["cell_index"] == list(range(N_CELLS))
    assert columns["pressure"] == state.pressure.tolist()
    assert columns["water_saturation"] == pytest.approx(state.water_saturation.tolist())


def test_several_states_stack_in_order_with_a_snapshot_label(backend):
    first, second = make_reservoir_state(), make_reservoir_state(shift=-100.0)
    columns = as_columns(as_dataframe([first, second], backend=backend))
    assert columns["snapshot"] == [0] * N_CELLS + [1] * N_CELLS
    assert columns["cell_index"] == list(range(N_CELLS)) * 2
    assert columns["pressure"] == first.pressure.tolist() + second.pressure.tolist()


def test_times_replace_the_snapshot_label(backend):
    states = [make_reservoir_state(), make_reservoir_state(shift=-1.0)]
    columns = as_columns(as_dataframe(states, backend=backend, times=[0.0, 30.0]))
    assert "snapshot" not in columns
    assert columns["time"] == [0.0] * N_CELLS + [30.0] * N_CELLS
    single = as_columns(as_dataframe(states[0], backend=backend, times=5.0))
    assert single["time"] == [5.0] * N_CELLS


def test_a_generator_of_states_is_accepted(backend):
    states = (make_reservoir_state(shift=float(i)) for i in range(3))
    assert as_dataframe(states, backend=backend).shape[0] == 3 * N_CELLS


def test_fields_choose_the_columns_and_include_derived_ones(backend):
    state = make_reservoir_state()
    columns = as_columns(
        as_dataframe(state, backend=backend, fields=["total_gas_mass", "pressure"])
    )
    assert list(columns) == ["cell_index", "total_gas_mass", "pressure"]
    assert columns["total_gas_mass"] == state.total_gas_mass.tolist()


def test_all_fields_leave_out_those_without_values(backend):
    columns = as_columns(as_dataframe(make_reservoir_state(), backend=backend, fields="all"))
    assert "solvent_concentration" not in columns
    assert "pressure" in columns and "oil_mass" in columns
    assert not set(HYSTERESIS_DATAFRAME_FIELDS) & set(columns)


def test_cells_by_index_and_by_mask(backend):
    state = make_reservoir_state()
    by_index = as_columns(as_dataframe(state, backend=backend, cells=[5, 2]))
    assert by_index["cell_index"] == [5, 2]
    assert by_index["pressure"] == [state.pressure[5], state.pressure[2]]
    mask = np.zeros(N_CELLS, dtype=bool)
    mask[[1, 2]] = True
    assert as_columns(as_dataframe(state, backend=backend, cells=mask))["cell_index"] == [1, 2]


def test_grid_adds_ijk_and_optionally_centroids(backend):
    grid = make_grid()
    columns = as_columns(
        as_dataframe(make_reservoir_state(), backend=backend, grid=grid, centroids=True)
    )
    assert list(columns)[:8] == ["cell_index", "i", "j", "k", "x", "y", "z", "pressure"]
    for cell in range(N_CELLS):
        assert (columns["i"][cell], columns["j"][cell], columns["k"][cell]) == grid.ijk_index(cell)
        assert [columns["x"][cell], columns["y"][cell], columns["z"][cell]] == pytest.approx(
            grid.cell_centroids[cell].tolist()
        )


def test_grid_columns_repeat_for_every_stacked_state(backend):
    states = [make_reservoir_state(), make_reservoir_state(shift=1.0)]
    columns = as_columns(as_dataframe(states, backend=backend, grid=make_grid(), cells=[0, 11]))
    assert columns["cell_index"] == [0, 11, 0, 11]
    assert columns["k"] == [0, 1, 0, 1]


def test_units_are_converted_before_reporting(backend):
    state = make_reservoir_state()
    columns = as_columns(as_dataframe(state, backend=backend, unit_system=UnitSystem.METRIC))
    expected = state.convert(UnitSystem.METRIC).pressure.tolist()
    assert columns["pressure"] == pytest.approx(expected)
    assert columns["pressure"][0] != state.pressure[0]


def test_dtype_is_kept_unless_overridden(backend):
    state = make_reservoir_state(dtype=np.float32)
    kept = as_dataframe(state, backend=backend, fields=["pressure"])
    assert "32" in str(kept["pressure"].dtype)
    forced = as_dataframe(state, backend=backend, fields=["pressure"], dtype=np.float64)
    assert "64" in str(forced["pressure"].dtype)


def test_hysteresis_arrays_are_exported_when_the_state_tracks_them(backend):
    state = make_reservoir_state(hysteresis=True)
    frame = as_dataframe(state, backend=backend, fields=list(HYSTERESIS_DATAFRAME_FIELDS))
    columns = as_columns(frame)
    assert list(columns) == ["cell_index", *HYSTERESIS_DATAFRAME_FIELDS]
    assert columns["max_water_saturation"] == pytest.approx(
        state.hysteresis.max_water_saturation.tolist()
    )
    assert columns["gas_reversal_saturation"] == pytest.approx(
        state.hysteresis.gas_reversal_saturation.tolist()
    )
    assert columns["water_imbibition_flag"] == [False] * N_CELLS
    assert "bool" in str(frame["gas_imbibition_flag"].dtype).lower()


def test_hysteresis_export_follows_an_advance_of_the_tracked_history(backend):
    state = make_reservoir_state(hysteresis=True)
    rising = np.zeros(N_CELLS)
    rising[:4] = 0.1
    water = state.water_saturation + rising
    advanced = state.hysteresis.advance(
        water_saturation=water,
        gas_saturation=state.gas_saturation,
        previous_water_saturation=state.water_saturation,
        previous_gas_saturation=state.gas_saturation,
        water_imbibition_when_increasing=True,
        gas_imbibition_when_increasing=False,
    )
    assert (
        advanced.water_imbibition_flag[:4].all() and not advanced.water_imbibition_flag[4:].any()
    )

    columns = as_columns(
        as_dataframe(
            state.evolve(water_saturation=water, hysteresis=advanced),
            backend=backend,
            fields=["max_water_saturation", "water_imbibition_flag"],
        )
    )
    assert columns["water_imbibition_flag"] == advanced.water_imbibition_flag.tolist()
    assert columns["max_water_saturation"] == pytest.approx(advanced.max_water_saturation.tolist())
    assert columns["max_water_saturation"][0] > state.hysteresis.max_water_saturation[0]


def test_all_fields_include_hysteresis_only_when_every_state_has_it(backend):
    tracked, untracked = make_reservoir_state(hysteresis=True), make_reservoir_state()
    assert "max_water_saturation" in as_columns(
        as_dataframe(tracked, backend=backend, fields="all")
    )
    mixed = as_columns(as_dataframe([tracked, untracked], backend=backend, fields="all"))
    assert "max_water_saturation" not in mixed


def test_asking_for_hysteresis_a_state_does_not_track_is_rejected(backend):
    with pytest.raises(ValidationError, match="hysteresis"):
        as_dataframe(make_reservoir_state(), backend=backend, fields=["max_water_saturation"])
    with pytest.raises(ValidationError, match="state 1"):
        as_dataframe(
            [make_reservoir_state(hysteresis=True), make_reservoir_state()],
            backend=backend,
            fields=["max_water_saturation"],
        )


def test_empty_input_gives_an_empty_frame_with_the_columns(backend):
    frame = as_dataframe([], backend=backend)
    assert frame.shape == (0, 2 + len(DEFAULT_DATAFRAME_FIELDS))
    assert list(frame.columns)[:2] == ["snapshot", "cell_index"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fields": ["nope"]}, "Unknown"),
        ({"fields": ["pressure", "pressure"]}, "repeats"),
        ({"fields": ["solvent_concentration"]}, "no values"),
        ({"fields": "some"}, "all"),
        ({"times": [1.0]}, "times"),
        ({"cells": [99]}, "outside"),
        ({"active_only": True}, "grid"),
        ({"centroids": True}, "grid"),
        ({"backend": "arrow"}, "backend"),
    ],
)
def test_invalid_requests_are_rejected(kwargs, message):
    states = [make_reservoir_state(), make_reservoir_state(shift=1.0)]
    with pytest.raises(ValidationError, match=message):
        as_dataframe(states, **kwargs)


def test_inconsistent_inputs_are_rejected():
    with pytest.raises(ValidationError, match="ReservoirState"):
        as_dataframe([make_reservoir_state(), "state"])
    with pytest.raises(ValidationError, match="unit systems"):
        as_dataframe([make_reservoir_state(), make_reservoir_state(unit_system=UnitSystem.METRIC)])
    with pytest.raises(ValidationError, match="cells"):
        as_dataframe(make_reservoir_state(), grid=make_cartesian_grid(nx=2, ny=2, nz=2))


def test_the_method_gives_the_same_frame_as_the_function(backend):
    state = make_reservoir_state(hysteresis=True)
    kwargs = {"backend": backend, "fields": ["pressure", "max_gas_saturation"], "cells": [1, 4]}
    assert as_columns(state.as_dataframe(time=7.0, **kwargs)) == as_columns(
        as_dataframe(state, times=7.0, **kwargs)
    )

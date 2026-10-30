import pytest
from typing_extensions import get_overloads

import bores.blackoil.state as model_module
import bores.reservoir.state as reservoir_module
import bores.schedule.summary as summary_module
import bores.wells.state as wells_module
from bores.blackoil.state import BlackOilModelState, as_dataframe
from bores.errors import ValidationError
from bores.reservoir.state import ReservoirState
from bores.reservoir.state import as_dataframe as reservoir_as_dataframe
from bores.schedule.summary import SummaryReport
from bores.types import UnitSystem
from bores.wells.state import WellStates
from bores.wells.state import as_dataframe as wells_as_dataframe

from .helpers import N_CELLS, as_columns, make_grid, make_reservoir_state, make_well_states


def make_model_states(count=3, with_wells=True):
    return [
        BlackOilModelState(
            reservoir=make_reservoir_state(shift=-10.0 * i),
            wells=make_well_states(shift=-5.0 * i) if with_wells else None,
            time=30.0 * i,
        )
        for i in range(count)
    ]


def test_reservoir_part_is_labelled_with_each_states_time(backend):
    columns = as_columns(as_dataframe(make_model_states(), backend=backend))
    assert columns["time"] == [0.0] * N_CELLS + [30.0] * N_CELLS + [60.0] * N_CELLS
    assert "snapshot" not in columns


def test_a_single_state_also_carries_its_time(backend):
    state = make_model_states(1)[0]
    columns = as_columns(as_dataframe(state, backend=backend))
    assert columns["time"] == [0.0] * N_CELLS


def test_reservoir_part_equals_the_reservoir_function(backend):
    states = make_model_states()
    expected = reservoir_as_dataframe(
        [s.reservoir for s in states], backend=backend, times=[s.time for s in states]
    )
    assert as_columns(as_dataframe(states, backend=backend)) == as_columns(expected)


def test_wells_and_perforation_parts_equal_the_wells_function(backend):
    states = make_model_states()
    times = [s.time for s in states]
    wells = [s.wells for s in states]
    assert as_columns(as_dataframe(states, backend=backend, part="wells")) == as_columns(
        wells_as_dataframe(wells, backend=backend, times=times)
    )
    assert as_columns(as_dataframe(states, backend=backend, part="perforations")) == as_columns(
        wells_as_dataframe(wells, backend=backend, level="perforation", times=times)
    )


def test_states_without_wells_add_no_rows_to_the_wells_parts(backend):
    states = [*make_model_states(2), *make_model_states(1, with_wells=False)]
    states[-1] = BlackOilModelState(reservoir=make_reservoir_state(), wells=None, time=90.0)
    columns = as_columns(as_dataframe(states, backend=backend, part="wells"))
    assert sorted(set(columns["time"])) == [0.0, 30.0]
    assert as_dataframe(states, backend=backend).shape[0] == 3 * N_CELLS


def test_options_pass_through_to_the_part(backend):
    grid = make_grid()
    states = make_model_states(2)
    reservoir = as_columns(
        as_dataframe(states, backend=backend, grid=grid, fields=["pressure"], cells=[0, 11])
    )
    assert reservoir["cell_index"] == [0, 11, 0, 11] and reservoir["k"] == [0, 1, 0, 1]
    perforations = as_columns(
        as_dataframe(states, backend=backend, part="perforations", grid=grid, wells="INJ1")
    )
    assert perforations["well_name"] == ["INJ1", "INJ1"] and "i" in perforations
    wells = as_columns(
        as_dataframe(states, backend=backend, part="wells", open_only=True, categorical=True)
    )
    assert "PROD2" not in wells["well_name"]


def test_units_convert_the_time_as_well_as_the_values(backend):
    states = make_model_states(2)
    columns = as_columns(
        as_dataframe(states, backend=backend, unit_system=UnitSystem.METRIC, fields=["pressure"])
    )
    converted = states[1].convert(UnitSystem.METRIC)
    assert columns["time"][-1] == pytest.approx(converted.time)
    assert columns["pressure"][-1] == pytest.approx(converted.reservoir.pressure[-1])


def test_empty_input(backend):
    assert as_dataframe([], backend=backend).shape[0] == 0
    assert as_dataframe([], backend=backend, part="wells").shape[0] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"part": "blocks"}, "part"),
        ({"part": "wells", "fields": ["pressure"]}, "fields"),
        ({"part": "wells", "cells": [0]}, "cells"),
        ({"part": "wells", "grid": make_grid()}, "grid"),
        ({"part": "perforations", "open_only": True}, "open_only"),
        ({"wells": "PROD*"}, "wells"),
        ({"open_only": True}, "open_only"),
        ({"categorical": True}, "categorical"),
        ({"backend": "arrow"}, "backend"),
    ],
)
def test_options_that_do_not_apply_to_the_part_are_rejected(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        as_dataframe(make_model_states(), **kwargs)


def test_wells_parts_need_at_least_one_state_with_wells():
    with pytest.raises(ValidationError, match="well states"):
        as_dataframe(make_model_states(2, with_wells=False), part="wells")


def test_inconsistent_inputs_are_rejected():
    with pytest.raises(ValidationError, match="BlackOilModelState"):
        as_dataframe([make_model_states(1)[0], "state"])
    metric = make_model_states(1)[0].convert(UnitSystem.METRIC)
    with pytest.raises(ValidationError, match="unit systems"):
        as_dataframe([make_model_states(1)[0], metric])


def test_the_method_gives_the_same_frame_as_the_function(backend):
    state = make_model_states(1)[0]
    for part in ("reservoir", "wells", "perforations"):
        assert as_columns(state.as_dataframe(backend=backend, part=part)) == as_columns(
            as_dataframe(state, backend=backend, part=part)
        )


@pytest.mark.parametrize(
    "function",
    [
        summary_module.as_dataframe,
        reservoir_module.as_dataframe,
        wells_module.as_dataframe,
        model_module.as_dataframe,
        SummaryReport.as_dataframe,
        ReservoirState.as_dataframe,
        WellStates.as_dataframe,
        BlackOilModelState.as_dataframe,
    ],
    ids=lambda f: f"{f.__module__}.{f.__qualname__}",
)
def test_every_as_dataframe_has_overloads_for_each_backend(function):
    overloads = get_overloads(function)
    assert len(overloads) == 3
    returns = [overload.__annotations__["return"] for overload in overloads]
    assert returns[:2] == ["pd.DataFrame", "pl.DataFrame"]

import pytest

from bores.errors import ValidationError
from bores.types import UnitSystem
from bores.wells.state import as_dataframe

from .helpers import as_columns, is_categorical, make_grid, make_well_states

WELL_COLUMNS = [
    "well_name",
    "well_type",
    "is_open",
    "control_mode",
    "injected_phase",
    "target_rate",
    "target_bhp",
    "target_thp",
    "bhp",
    "thp",
    "oil_rate",
    "water_rate",
    "gas_rate",
    "surface_oil_rate",
    "surface_water_rate",
    "surface_gas_rate",
    "active_limit",
    "active_limit_quantity",
]


def test_well_level_has_a_row_for_each_well_with_its_control_and_rates(backend):
    columns = as_columns(as_dataframe(make_well_states(), backend=backend))
    assert list(columns) == WELL_COLUMNS
    assert columns["well_name"] == ["PROD1", "INJ1", "PROD2"]
    assert columns["well_type"] == ["producer", "injector", "producer"]
    assert columns["is_open"] == [True, True, False]
    assert columns["control_mode"] == ["oil_rate", "bhp", "bhp"]
    assert columns["injected_phase"] == [None, "water", None]


def test_targets_the_control_does_not_have_are_missing(backend):
    columns = as_columns(as_dataframe(make_well_states(), backend=backend))
    assert columns["target_rate"] == [500.0, None, None]
    assert columns["target_bhp"] == [None, 4500.0, 1500.0]
    assert columns["target_thp"] == [None, None, None]
    assert columns["thp"] == [300.0, None, None]


def test_rates_and_the_active_limit(backend):
    columns = as_columns(as_dataframe(make_well_states(), backend=backend))
    assert columns["oil_rate"] == [250.0, 0.0, 0.0]
    assert columns["water_rate"] == [11.0, -800.0, 0.0]
    assert columns["surface_oil_rate"] == [240.0, 0.0, 0.0]
    assert columns["surface_water_rate"] == [11.0, -790.0, 0.0]
    assert columns["active_limit"] == ["rate", None, "economic"]
    assert columns["active_limit_quantity"] == ["liquid", None, "water_cut"]


def test_a_single_wellstates_is_not_mistaken_for_a_sequence_of_names(backend):
    frame = as_dataframe(make_well_states(), backend=backend)
    assert frame.shape == (3, len(WELL_COLUMNS))


def test_several_times_stack_in_order(backend):
    states = [make_well_states(), make_well_states(shift=-50.0)]
    columns = as_columns(as_dataframe(states, backend=backend, times=[0.0, 30.0]))
    assert columns["time"] == [0.0] * 3 + [30.0] * 3
    assert columns["bhp"] == [2000.0, 4400.0, 1500.0, 1950.0, 4400.0, 1500.0]
    snapshot = as_columns(as_dataframe(states, backend=backend))
    assert snapshot["snapshot"] == [0, 0, 0, 1, 1, 1]


def test_perforation_level_has_a_row_for_each_open_perforation(backend):
    columns = as_columns(as_dataframe(make_well_states(), backend=backend, level="perforation"))
    assert list(columns) == [
        "well_name",
        "perforation",
        "cell_index",
        "flowing_pressure",
        "oil_rate",
        "water_rate",
        "gas_rate",
    ]
    assert columns["well_name"] == ["PROD1", "PROD1", "INJ1"]
    assert columns["perforation"] == [0, 1, 0]
    assert columns["cell_index"] == [3, 9, 1]
    assert columns["flowing_pressure"] == [2100.0, 2200.0, 4300.0]
    assert columns["water_rate"] == [5.0, 6.0, -800.0]


def test_perforation_level_stacks_times_and_adds_grid_columns(backend):
    grid = make_grid()
    states = [make_well_states(), make_well_states(shift=-50.0)]
    columns = as_columns(
        as_dataframe(
            states,
            backend=backend,
            level="perforation",
            times=[0.0, 30.0],
            grid=grid,
            centroids=True,
        )
    )
    assert columns["time"] == [0.0] * 3 + [30.0] * 3
    assert columns["flowing_pressure"] == [2100.0, 2200.0, 4300.0, 2050.0, 2150.0, 4300.0]
    for row, cell in enumerate(columns["cell_index"]):
        assert (columns["i"][row], columns["j"][row], columns["k"][row]) == grid.ijk_index(cell)
    assert {"x", "y", "z"} <= set(columns)


def test_wells_can_be_chosen_by_name_or_pattern_and_shut_ones_dropped(backend):
    states = make_well_states()
    assert as_columns(as_dataframe(states, backend=backend, wells="PROD*"))["well_name"] == [
        "PROD1",
        "PROD2",
    ]
    assert as_columns(as_dataframe(states, backend=backend, wells=["INJ1", "PROD1"]))[
        "well_name"
    ] == ["PROD1", "INJ1"]
    assert as_columns(as_dataframe(states, backend=backend, open_only=True))["well_name"] == [
        "PROD1",
        "INJ1",
    ]
    perforations = as_columns(
        as_dataframe(states, backend=backend, level="perforation", wells="INJ1")
    )
    assert perforations["well_name"] == ["INJ1"]


def test_units_are_converted_before_reporting(backend):
    states = make_well_states()
    columns = as_columns(as_dataframe(states, backend=backend, unit_system=UnitSystem.METRIC))
    expected = states.convert(UnitSystem.METRIC)["PROD1"].bhp
    assert columns["bhp"][0] == pytest.approx(expected)
    assert columns["bhp"][0] < 1000.0


def test_categorical_text_columns(backend):
    frame = as_dataframe(make_well_states(), backend=backend, categorical=True)
    assert is_categorical(frame["well_name"])
    assert as_columns(frame)["well_name"] == ["PROD1", "INJ1", "PROD2"]


def test_empty_input(backend):
    assert as_dataframe([], backend=backend).shape == (0, 1 + len(WELL_COLUMNS))
    assert as_dataframe([], backend=backend, level="perforation").shape[0] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"level": "cell"}, "level"),
        ({"grid": make_grid()}, "perforation"),
        ({"centroids": True}, "perforation"),
        ({"level": "perforation", "open_only": True}, "open_only"),
        ({"wells": "NOPE*"}, "No well matches"),
        ({"times": [1.0]}, "times"),
        ({"backend": "arrow"}, "backend"),
    ],
)
def test_invalid_requests_are_rejected(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        as_dataframe([make_well_states(), make_well_states()], **kwargs)


def test_inconsistent_inputs_are_rejected():
    with pytest.raises(ValidationError, match="WellStates"):
        as_dataframe([make_well_states(), "wells"])
    metric = make_well_states().convert(UnitSystem.METRIC)
    with pytest.raises(ValidationError, match="unit systems"):
        as_dataframe([make_well_states(), metric])


def test_the_method_gives_the_same_frame_as_the_function(backend):
    states = make_well_states()
    kwargs = {"backend": backend, "level": "perforation", "wells": "PROD*", "grid": make_grid()}
    assert as_columns(states.as_dataframe(time=3.0, **kwargs)) == as_columns(
        as_dataframe(states, times=3.0, **kwargs)
    )

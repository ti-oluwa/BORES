import datetime
import subprocess
import sys

import numpy as np
import pytest

from bores.dataframes import (
    Column,
    build_dataframe,
    get_grid_columns,
    get_label_column,
    resolve_backend,
    resolve_times,
    select_cells,
)
from bores.errors import ValidationError

from .helpers import N_CELLS, as_columns, is_categorical, make_grid


def test_unknown_backend_is_rejected():
    with pytest.raises(ValidationError, match="backend"):
        resolve_backend("arrow")


@pytest.mark.parametrize("name", ["pandas", "polars"])
def test_missing_library_gives_an_install_hint(monkeypatch, name):
    monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=rf"bores-framework\[{name}\]"):
        build_dataframe([Column(name="a", values=np.zeros(1))], backend=name)


def test_importing_bores_does_not_import_either_library():
    code = (
        "import sys, bores.dataframes, bores.schedule.summary, bores.reservoir.state, "
        "bores.wells.state, bores.blackoil.state\n"
        "assert 'pandas' not in sys.modules and 'polars' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_columns_keep_their_values_and_missing_entries(backend):
    dates = [datetime.datetime(2020, 1, 1), None, datetime.datetime(2020, 1, 3)]
    frame = build_dataframe(
        [
            Column(name="x", values=np.array([1.0, np.nan, 3.0]), missing_is_null=True),
            Column(name="step", values=[1, None, 3], kind="int"),
            Column(name="n", values=np.array([4, 5, 6]), kind="int"),
            Column(name="flag", values=[True, False, True], kind="bool"),
            Column(name="name", values=["a", None, "c"], kind="str"),
            Column(name="date", values=dates, kind="datetime"),
        ],
        backend=backend,
    )
    columns = as_columns(frame)
    assert columns["x"] == [1.0, None, 3.0]
    assert columns["step"] == [1, None, 3]
    assert columns["n"] == [4, 5, 6]
    assert columns["flag"] == [True, False, True]
    assert columns["name"] == ["a", None, "c"]
    assert [None if d is None else d.day for d in columns["date"]] == [1, None, 3]


def test_missing_float_is_a_null_in_polars_and_nan_in_pandas(backend):
    frame = build_dataframe(
        [Column(name="x", values=np.array([1.0, np.nan]), missing_is_null=True)], backend=backend
    )
    if backend == "polars":
        assert frame["x"].null_count() == 1 and frame["x"].is_nan().sum() == 0
    else:
        assert np.isnan(frame["x"].iloc[1])


def test_float_column_that_is_not_marked_keeps_nan_in_polars(backend):
    frame = build_dataframe([Column(name="x", values=np.array([1.0, np.nan]))], backend=backend)
    if backend == "polars":
        assert frame["x"].null_count() == 0 and frame["x"].is_nan().sum() == 1


def test_categorical_text_column(backend):
    frame = build_dataframe(
        [Column(name="well", values=["a", "b", "a"], kind="str", categorical=True)],
        backend=backend,
    )
    assert is_categorical(frame["well"])
    assert as_columns(frame)["well"] == ["a", "b", "a"]


def test_repeated_names_and_unequal_lengths_are_rejected(backend):
    with pytest.raises(ValidationError, match="distinct"):
        build_dataframe(
            [Column(name="a", values=np.zeros(2)), Column(name="a", values=np.zeros(2))],
            backend=backend,
        )
    with pytest.raises(ValidationError, match="equal lengths"):
        build_dataframe(
            [Column(name="a", values=np.zeros(2)), Column(name="b", values=np.zeros(3))],
            backend=backend,
        )


def test_no_columns_gives_an_empty_frame(backend):
    assert build_dataframe([], backend=backend).shape == (0, 0)


class FakeGrid:
    """Just what `select_cells` reads of a grid: a cell count and which cells are active."""

    def __init__(self, active):
        self.active = np.asarray(active)
        self.n_cells = self.active.size

    def are_cells_active(self, indices):
        return self.active[indices]


def test_select_cells_by_default_takes_every_cell():
    chosen = select_cells(n_cells=5, cells=None, grid=None, active_only=False)
    assert chosen.tolist() == [0, 1, 2, 3, 4]


def test_select_cells_by_indices_and_by_mask_keep_the_request_order():
    assert select_cells(n_cells=5, cells=[3, 1], grid=None, active_only=False).tolist() == [3, 1]
    mask = np.array([False, True, False, True, True])
    assert select_cells(n_cells=5, cells=mask, grid=None, active_only=False).tolist() == [1, 3, 4]


def test_select_cells_drops_inactive_cells():
    grid = FakeGrid([True, False, True, True, False])
    chosen = select_cells(n_cells=5, cells=None, grid=grid, active_only=True)
    assert chosen.tolist() == [0, 2, 3]
    chosen = select_cells(n_cells=5, cells=[4, 3, 1, 0], grid=grid, active_only=True)
    assert chosen.tolist() == [3, 0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cells": [5]}, "outside"),
        ({"cells": [-1]}, "outside"),
        ({"cells": [1, 1]}, "repeat"),
        ({"cells": np.ones(3, dtype=bool)}, "mask"),
        ({"cells": [0.5]}, "integer"),
        ({"cells": [[0, 1]]}, "one-dimensional"),
        ({"active_only": True}, "grid"),
    ],
)
def test_select_cells_rejects_bad_requests(kwargs, message):
    arguments = {"n_cells": 5, "cells": None, "grid": None, "active_only": False} | kwargs
    with pytest.raises(ValidationError, match=message):
        select_cells(**arguments)


def test_select_cells_rejects_a_grid_of_another_size():
    with pytest.raises(ValidationError, match="cells"):
        select_cells(n_cells=5, cells=None, grid=FakeGrid([True] * 4), active_only=False)


def test_grid_columns_give_ijk_that_round_trip_and_centroids():
    grid = make_grid()
    indices = np.array([0, 4, 7, 11], dtype=np.int64)
    columns = {
        c.name: c.values for c in get_grid_columns(grid=grid, cell_indices=indices, centroids=True)
    }
    assert list(columns) == ["i", "j", "k", "x", "y", "z"]
    for row, cell in enumerate(indices):
        assert (columns["i"][row], columns["j"][row], columns["k"][row]) == grid.ijk_index(
            int(cell)
        )
    np.testing.assert_allclose(
        np.column_stack([columns["x"], columns["y"], columns["z"]]), grid.cell_centroids[indices]
    )


def test_grid_columns_without_a_grid():
    indices = np.array([0], dtype=np.int64)
    assert get_grid_columns(grid=None, cell_indices=indices, centroids=False) == []
    with pytest.raises(ValidationError, match="grid"):
        get_grid_columns(grid=None, cell_indices=indices, centroids=True)


def test_resolve_times():
    assert resolve_times(None, count=2) is None
    assert resolve_times(5.0, count=1) == [5.0]
    assert resolve_times([1.0, 2.0], count=2) == [1.0, 2.0]
    with pytest.raises(ValidationError, match="times"):
        resolve_times([1.0], count=2)


def test_label_column_is_time_or_snapshot_or_nothing():
    time = get_label_column(times=[0.0, 30.0], row_counts=[2, 1], is_single=False)
    assert time.name == "time" and time.values.tolist() == [0.0, 0.0, 30.0]
    snapshot = get_label_column(times=None, row_counts=[2, 1], is_single=False)
    assert snapshot.name == "snapshot" and snapshot.values.tolist() == [0, 0, 1]
    assert get_label_column(times=None, row_counts=[N_CELLS], is_single=True) is None

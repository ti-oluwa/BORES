"""Shared support for turning result models into pandas or polars data frames."""

import importlib
import typing

import attrs
import numpy as np
import numpy.typing as npt

from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.types import Boolean, IntArray, Integer, Number, OneDimension

if typing.TYPE_CHECKING:
    import pandas as pd  # type: ignore[import]
    import polars as pl  # type: ignore[import]

__all__ = [
    "BACKENDS",
    "AnyDataFrame",
    "Backend",
    "Column",
    "build_dataframe",
    "get_grid_columns",
    "get_label_column",
    "resolve_backend",
    "resolve_times",
    "select_cells",
]

Backend = typing.Literal["pandas", "polars"]
"""The data frame library a data frame is built with."""

BACKENDS: tuple[str, ...] = ("pandas", "polars")
"""Every supported `Backend`."""

AnyDataFrame = typing.Union["pd.DataFrame", "pl.DataFrame"]
"""A `pandas.DataFrame` or a `polars.DataFrame`."""

ColumnKind = typing.Literal["float", "int", "bool", "str", "datetime"]
"""What a `Column` holds."""


@attrs.frozen(kw_only=True, slots=True)
class Column:
    """One column of a data frame, before it is given to a data frame library."""

    name: str
    """The column's name."""

    values: typing.Sequence[typing.Any] | npt.NDArray
    """
    The column's values. A `float` column is a NumPy array, where NaN marks a missing value.
    An `int`, `str` or `datetime` column may be a list holding `None` for a missing value.
    """

    kind: ColumnKind = "float"
    """What the column holds. This decides the column's type in either library."""

    missing_is_null: Boolean = False
    """
    For a `float` column, whether NaN means "no value here" rather than a number that is not
    a number. Polars tells the two apart, so such a column holds nulls there. Pandas does not,
    so it keeps NaN.
    """

    categorical: Boolean = False
    """For a `str` column, whether to store it as a categorical, which saves memory when the same few strings repeat."""


def resolve_backend(backend: str) -> typing.Any:
    """
    Imports the data frame library behind a backend.

    :param backend: `"pandas"` or `"polars"`.
    :returns: The imported library module.
    :raises ValidationError: If `backend` is not one of the supported names.
    :raises ImportError: If the library is not installed, with how to install it.
    """
    if backend not in BACKENDS:
        raise ValidationError(f"`backend` must be one of {list(BACKENDS)}; got {backend!r}.")
    try:
        return importlib.import_module(backend)
    except ImportError as exc:
        raise ImportError(
            f"`backend={backend!r}` needs {backend} installed. Install it with "
            f"`pip install 'bores-framework[{backend}]'`, or pass the other backend."
        ) from exc


def build_dataframe(columns: typing.Sequence[Column], *, backend: str) -> AnyDataFrame:
    """
    Builds a data frame from columns.

    :param columns: The columns, in order. All must have the same length.
    :param backend: `"pandas"` or `"polars"`.
    :returns: A `pandas.DataFrame` or a `polars.DataFrame`.
    :raises ValidationError: If `backend` is unknown, two columns share a name, or the columns
        differ in length.
    :raises ImportError: If the library behind `backend` is not installed.
    """
    library = resolve_backend(backend)

    names = [column.name for column in columns]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValidationError(
            f"Data frame columns must have distinct names; repeated: {repeated}."
        )
    lengths = {column.name: len(column.values) for column in columns}
    if len(set(lengths.values())) > 1:
        raise ValidationError(f"Data frame columns must have equal lengths; got {lengths}.")

    if backend == "pandas":
        return library.DataFrame({column.name: to_pandas(library, column) for column in columns})
    return library.DataFrame([to_polars(library, column) for column in columns])


def to_pandas(pd: typing.Any, column: Column) -> typing.Any:
    """
    :param pd: The imported pandas module.
    :param column: The column to convert.
    :returns: The column's values in the type pandas should store them as.
    """
    values = column.values
    if column.kind == "float":
        if isinstance(values, np.ndarray) and values.dtype.kind == "f":
            return values
        return np.asarray(values, dtype=np.float64)
    if column.kind == "int":
        if isinstance(values, np.ndarray) or None not in values:
            return np.asarray(values, dtype=np.int64)
        return pd.array(values, dtype="Int64")
    if column.kind == "bool":
        return np.asarray(values, dtype=np.bool_)
    if column.kind == "datetime":
        return pd.to_datetime(list(values))
    if column.categorical:
        return pd.Categorical(list(values))
    return list(values)


def to_polars(pl: typing.Any, column: Column) -> typing.Any:
    """
    :param pl: The imported polars module.
    :param column: The column to convert.
    :returns: The column as a polars series of the type polars should store it as.
    """
    values = column.values
    name = column.name
    if column.kind == "float":
        array = values if isinstance(values, np.ndarray) else np.asarray(values, dtype=np.float64)
        return pl.Series(name, array, nan_to_null=column.missing_is_null)
    if column.kind == "int":
        return pl.Series(name, values, dtype=pl.Int64)
    if column.kind == "bool":
        return pl.Series(name, values, dtype=pl.Boolean)
    if column.kind == "datetime":
        return pl.Series(name, list(values), dtype=pl.Datetime("us"))
    return pl.Series(name, list(values), dtype=pl.Categorical if column.categorical else pl.String)


def select_cells(
    *,
    n_cells: Integer,
    cells: typing.Sequence[Integer] | np.ndarray | None,
    grid: Grid | None,
    active_only: Boolean,
) -> IntArray[OneDimension]:
    """
    Works out which cells of a state a data frame reports.

    :param n_cells: How many cells the state has.
    :param cells: Flat cell indices, or a boolean mask with one entry for every cell, or
        `None` for every cell.
    :param grid: The grid the state belongs to, needed to drop inactive cells.
    :param active_only: Whether to drop the cells the grid marks inactive.
    :returns: The flat indices of the chosen cells, in increasing order of the request.
    :raises ValidationError: If `cells` has an index outside the state, a mask of the wrong
        length, or repeats a cell, or if `active_only` is set without a `grid` or the grid's
        cell count differs from the state's.
    """
    if grid is not None and grid.n_cells != n_cells:
        raise ValidationError(
            f"`grid` has {grid.n_cells} cells but the state has {n_cells}. "
            "Pass the grid this state was computed on."
        )

    if cells is None:
        selected = np.arange(n_cells, dtype=np.int64)
    else:
        requested = np.asarray(cells)
        if requested.dtype == np.bool_:
            if requested.shape != (n_cells,):
                raise ValidationError(
                    f"A boolean `cells` mask needs one entry for each of the {n_cells} cells; "
                    f"got shape {requested.shape}."
                )
            selected = np.flatnonzero(requested).astype(np.int64)
        else:
            if requested.ndim != 1 or (requested.size and requested.dtype.kind not in "iu"):
                raise ValidationError(
                    "`cells` must be a one-dimensional sequence of integer cell indices "
                    "or a boolean mask."
                )
            selected = requested.astype(np.int64)
            if selected.size and (selected.min() < 0 or selected.max() >= n_cells):
                raise ValidationError(
                    f"`cells` has an index outside 0 to {n_cells - 1}: "
                    f"{selected[(selected < 0) | (selected >= n_cells)][:5].tolist()}."
                )
            if np.unique(selected).size != selected.size:
                raise ValidationError("`cells` must not repeat a cell.")

    if active_only:
        if grid is None:
            raise ValidationError("`active_only` needs the `grid` the state belongs to.")
        selected = selected[grid.are_cells_active(selected)]
    return typing.cast("IntArray[OneDimension]", selected)


def get_grid_columns(
    *,
    grid: Grid | None,
    cell_indices: IntArray[OneDimension],
    centroids: Boolean,
) -> list[Column]:
    """
    Builds the columns that say where cells are, from the grid they belong to.

    :param grid: The grid, or `None` for no location columns.
    :param cell_indices: Flat indices of the cells, one for each data frame row.
    :param centroids: Whether to add the cell centroid coordinates as `x`, `y` and `z`.
    :returns: `i`, `j` and `k` (0-based, when the grid is structured) and, if asked for,
        `x`, `y` and `z`. Empty if there is no `grid`.
    :raises ValidationError: If `centroids` is set without a `grid`.
    """
    if grid is None:
        if centroids:
            raise ValidationError("`centroids` needs the `grid` the state belongs to.")
        return []

    columns: list[Column] = []
    if grid.dimensions is not None:
        i, j, k = grid.ijk_indices(cell_indices)
        columns += [
            Column(name="i", values=i, kind="int"),
            Column(name="j", values=j, kind="int"),
            Column(name="k", values=k, kind="int"),
        ]
    if centroids:
        assert grid.cell_centroids is not None
        points = grid.cell_centroids[cell_indices]
        columns += [
            Column(name="x", values=points[:, 0]),
            Column(name="y", values=points[:, 1]),
            Column(name="z", values=points[:, 2]),
        ]
    return columns


def resolve_times(
    times: Number | typing.Sequence[Number] | None, *, count: Integer
) -> list[Number] | None:
    """
    Checks the elapsed times given for a number of states.

    :param times: One time for each state, or a single number for a single state, or `None`.
    :param count: How many states there are.
    :returns: The times as a list, or `None` if none were given.
    :raises ValidationError: If the number of times differs from the number of states.
    """
    if times is None:
        return None
    values = [times] if np.isscalar(times) else list(typing.cast("typing.Sequence[Number]", times))
    if len(values) != count:
        raise ValidationError(f"`times` has {len(values)} entries but {count} states were given.")
    return typing.cast("list[Number]", values)


def get_label_column(
    *,
    times: typing.Sequence[Number] | None,
    row_counts: typing.Sequence[Integer],
    is_single: Boolean,
) -> Column | None:
    """
    Builds the column that says which state each row came from.

    :param times: The elapsed time of each state, if known.
    :param row_counts: How many rows each state contributes.
    :param is_single: Whether one state, not a sequence, was given.
    :returns: A `time` column if `times` is given, otherwise a `snapshot` column counting the
        states from 0, or `None` for a single state without times, which needs no label.
    """
    counts = np.asarray(row_counts, dtype=np.int64)
    if times is not None:
        return Column(name="time", values=np.repeat(np.asarray(times, dtype=np.float64), counts))
    if is_single:
        return None
    return Column(
        name="snapshot",
        values=np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts),
        kind="int",
    )

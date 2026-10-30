"""
Summary-vector recording API.

A summary vector (Eclipse's own term: `FOPR`, `WBHP`, `WWCT`, and so on) is
just another thing a schedule can read at a point in time. It fires on the
same `Event`s as any other scheduled item, and the values it reads are recorded in
a `SummaryReport` rather than mutating the model.

`Summary` is the read-only counterpart to `Action`. An `Action` changes the model
an `Event` fires against; a `Summary` reports on it instead, via `RecordSummary`,
an `Action` that evaluates one or more `Summary` quantities and records what they
return in a `SummaryReport`.
"""

import datetime
import fnmatch
import typing

import attrs
import numpy as np

from bores.dataframes import AnyDataFrame, Backend, Column, build_dataframe, resolve_backend
from bores.errors import SummaryError, ValidationError
from bores.schedule.base import (
    Action,
    ModelT,
    ModelTcon,
    ScheduleContext,
    ScheduleItem,
)
from bores.schedule.events import IntervalEvent, TimeEvent, TimeStepEvent
from bores.serde.base import Serializable
from bores.serde.stores.base import StoreSerializable
from bores.types import Boolean, IntArray, Number, NumberArray, OneDimension
from bores.utils import get_current_date

if typing.TYPE_CHECKING:
    import pandas as pd  # type: ignore[import]
    import polars as pl  # type: ignore[import]

__all__ = [
    "RecordSummary",
    "Summary",
    "SummaryRecord",
    "SummaryReport",
    "as_dataframe",
    "record_at",
    "record_every",
]

SummaryLayout = typing.Literal["wide", "long"]
"""How `as_dataframe` lays a `SummaryReport` out: one column for each vector, or one row for each record."""


@typing.runtime_checkable
class Summary(typing.Protocol[ModelTcon]):
    """
    A summary vector.

    Any callable matching this signature, with a `key` attribute, satisfies it.
    """

    @property
    def key(self) -> str:
        """
        This vector's identifier in a `SummaryReport`, in Eclipse's own
        `MNEMONIC` or `MNEMONIC:QUALIFIER` form (`"FOPR"`, `"WBHP:PROD1"`).
        """
        ...

    def __call__(self, model: ModelTcon, context: ScheduleContext) -> Number:
        """
        Reads this vector's current value.

        :param model: The model being scheduled against.
        :param context: The current moment's context.
        :returns: This vector's value at `context.time`.
        """
        ...


@attrs.frozen(kw_only=True, slots=True)
class SummaryRecord(Serializable):
    """One recorded value of one summary vector, at one point in time."""

    key: str
    """The vector this value belongs to, e.g. `"FOPR"` or `"WBHP:PROD1"`."""

    time: Number
    """Elapsed time this value was recorded at, in the run's unit system."""

    value: Number
    """The vector's value at `time`."""

    time_step: int | None = None
    """The time-step index this value was recorded at, if tracked."""

    date: datetime.datetime | None = None
    """The calendar date this value was recorded at, if tracked."""


@attrs.define(kw_only=True, slots=True)
class SummaryReport(
    StoreSerializable,
    dump_exclude={"key_indices", "indexed_count", "series_cache"},
    load_exclude={"key_indices", "indexed_count", "series_cache"},
):
    """
    A run's recorded summary vectors, in the order they were recorded.

    Mutable and append-only. Built empty and handed to a run through
    `ScheduleContext.extra["summary_report"]`.

    Each vector's values must be recorded at strictly increasing times, since `at` and
    `between` search them by time. Recording a vector at or before the time of its latest
    record is an error, which also catches a report time recorded twice, for example by two
    items that record the same vector at the same time. To redo part of a run, such as after
    a restart or a rejected step, first remove the records that no longer apply with
    `discard_after`. A record added by appending to `records` directly is not checked.

    Keeps an index from each vector key to the positions of its records,
    so per-vector queries cost time proportional to that vector's own
    record count, not the whole report's. Records appended directly to
    `records` instead of through `record(...)` are picked up on the next query.
    """

    records: list[SummaryRecord] = attrs.field(factory=list)
    """Every value recorded so far, across every vector, in recording order."""

    key_indices: dict[str, list[int]] = attrs.field(init=False, factory=dict, repr=False, eq=False)
    """Positions in `records` of each vector's records, in recording order."""

    indexed_count: int = attrs.field(init=False, default=0, repr=False, eq=False)
    """How many leading entries of `records` `key_indices` already covers."""

    series_cache: dict[str, tuple[int, NumberArray[OneDimension], NumberArray[OneDimension]]] = (
        attrs.field(init=False, factory=dict, repr=False, eq=False)
    )
    """Each vector's last built `(record_count, times, values)`, reused while no new record arrives."""

    def update_index(self) -> None:
        """Indexes any records added since the last query or append."""
        records = self.records
        for position in range(self.indexed_count, len(records)):
            self.key_indices.setdefault(records[position].key, []).append(position)
        self.indexed_count = len(records)

    def record(
        self,
        *,
        key: str,
        time: Number,
        value: Number,
        time_step: int | None = None,
        date: datetime.datetime | None = None,
    ) -> SummaryRecord:
        """
        Appends one value to the ledger.

        :param key: The vector this value belongs to.
        :param time: Elapsed time this value was recorded at.
        :param value: The vector's value at `time`.
        :param time_step: The time-step index this value was recorded at, if tracked.
        :param date: The calendar date this value was recorded at, if tracked.
        :returns: The `SummaryRecord` that was appended.
        :raises SummaryError: If `key` already has a record at `time` or later.
        """
        previous = self.last(key)
        if previous is not None and time <= previous.time:
            raise SummaryError(
                f"Cannot record {key!r} at time {time}: it already has a record at time "
                f"{previous.time}. A vector's records must be at strictly increasing times. "
                "Use `discard_after` to remove records before recording them again."
            )
        record = SummaryRecord(
            key=key,
            time=time,
            value=value,
            time_step=time_step,
            date=date,
        )
        self.records.append(record)
        self.update_index()
        return record

    def discard_after(self, *, time: Number) -> int:
        """
        Removes every record made after `time`, of every vector, so that time can be recorded
        again. Use it when a run goes back to an earlier time, such as a restart.

        :param time: Elapsed time to keep records up to. A record at exactly `time` is kept.
        :returns: How many records were removed.
        """
        kept = [record for record in self.records if record.time <= time]
        removed = len(self.records) - len(kept)
        if removed:
            self.records[:] = kept
            self.key_indices.clear()
            self.series_cache.clear()
            self.indexed_count = 0
            self.update_index()
        return removed

    def keys(self) -> frozenset[str]:
        """
        Every distinct vector key recorded so far.

        :returns: The set of recorded keys.
        """
        self.update_index()
        return frozenset(self.key_indices)

    def count(self, key: str, /) -> int:
        """
        How many values of this vector have been recorded.

        :param key: The vector to count.
        :returns: The number of records for `key`, `0` if never recorded.
        """
        self.update_index()
        return len(self.key_indices.get(key, ()))

    def get(self, key: str, /) -> list[SummaryRecord]:
        """
        Every record of this vector, in recording order.

        :param key: The vector to retrieve.
        :returns: The vector's records. Empty if `key` was never recorded.
        """
        self.update_index()
        records = self.records
        return [records[position] for position in self.key_indices.get(key, ())]

    def series(self, key: str, /) -> tuple[NumberArray[OneDimension], NumberArray[OneDimension]]:
        """
        This vector's full time series, in recording order.

        :param key: The vector to retrieve.
        :returns: `(times, values)`, each shape `(n_records_for_key,)`, always in double
            precision whatever the run's working precision, so reported values keep the precision
            they were recorded with. Both empty if `key` was never recorded. Rebuilt only when
            `key` has gained records since the last call, so treat the
            returned arrays as read-only.
        """
        self.update_index()
        positions = self.key_indices.get(key, ())
        cached = self.series_cache.get(key)
        if cached is not None and cached[0] == len(positions):
            return cached[1], cached[2]

        records = self.records
        times = typing.cast(
            "NumberArray[OneDimension]",
            np.fromiter((records[position].time for position in positions), dtype=np.float64),
        )
        values = typing.cast(
            "NumberArray[OneDimension]",
            np.fromiter((records[position].value for position in positions), dtype=np.float64),
        )
        self.series_cache[key] = (len(positions), times, values)
        return times, values

    def last(self, key: str, /) -> SummaryRecord | None:
        """
        This vector's most recently recorded value.

        :param key: The vector to retrieve.
        :returns: The last `SummaryRecord` for `key`, or `None` if never recorded.
        """
        self.update_index()
        positions = self.key_indices.get(key)
        if not positions:
            return None
        return self.records[positions[-1]]

    def at(self, key: str, /, *, time: Number) -> SummaryRecord | None:
        """
        This vector's most recent record at or before `time`.

        :param key: The vector to retrieve.
        :param time: Elapsed time to look up.
        :returns: The latest `SummaryRecord` for `key` with `record.time <= time`,
            or `None` if there is none.
        """
        self.update_index()
        positions = self.key_indices.get(key)
        if not positions:
            return None

        records = self.records
        low, high = 0, len(positions)
        while low < high:
            middle = (low + high) // 2
            if records[positions[middle]].time <= time:
                low = middle + 1
            else:
                high = middle

        if low == 0:
            return None
        return records[positions[low - 1]]

    def between(
        self, key: str, /, *, start: Number, end: Number
    ) -> tuple[NumberArray[OneDimension], NumberArray[OneDimension]]:
        """
        This vector's time series restricted to `start <= time <= end`.

        :param key: The vector to retrieve.
        :param start: Earliest elapsed time to include.
        :param end: Latest elapsed time to include.
        :returns: `(times, values)` within the window, in recording order.
        """
        times, values = self.series(key)
        mask = (times >= start) & (times <= end)
        return typing.cast("NumberArray[OneDimension]", times[mask]), typing.cast(
            "NumberArray[OneDimension]", values[mask]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> typing.Iterator[SummaryRecord]:
        return iter(self.records)

    def __contains__(self, key: object, /) -> bool:
        if not isinstance(key, str):
            return False
        self.update_index()
        return key in self.key_indices

    @typing.overload
    def as_dataframe(
        self,
        *,
        backend: typing.Literal["pandas"] = "pandas",
        layout: SummaryLayout = ...,
        keys: str | typing.Sequence[str] | None = ...,
        start: Number | None = ...,
        end: Number | None = ...,
        include_dates: Boolean | None = ...,
        include_time_steps: Boolean | None = ...,
        split_keys: Boolean = ...,
        categorical: Boolean = ...,
    ) -> "pd.DataFrame": ...
    @typing.overload
    def as_dataframe(
        self,
        *,
        backend: typing.Literal["polars"],
        layout: SummaryLayout = ...,
        keys: str | typing.Sequence[str] | None = ...,
        start: Number | None = ...,
        end: Number | None = ...,
        include_dates: Boolean | None = ...,
        include_time_steps: Boolean | None = ...,
        split_keys: Boolean = ...,
        categorical: Boolean = ...,
    ) -> "pl.DataFrame": ...
    @typing.overload
    def as_dataframe(
        self,
        *,
        backend: Backend,
        layout: SummaryLayout = ...,
        keys: str | typing.Sequence[str] | None = ...,
        start: Number | None = ...,
        end: Number | None = ...,
        include_dates: Boolean | None = ...,
        include_time_steps: Boolean | None = ...,
        split_keys: Boolean = ...,
        categorical: Boolean = ...,
    ) -> AnyDataFrame: ...
    def as_dataframe(
        self,
        *,
        backend: Backend = "pandas",
        layout: SummaryLayout = "wide",
        keys: str | typing.Sequence[str] | None = None,
        start: Number | None = None,
        end: Number | None = None,
        include_dates: Boolean | None = None,
        include_time_steps: Boolean | None = None,
        split_keys: Boolean = False,
        categorical: Boolean = False,
    ) -> AnyDataFrame:
        """
        Turns this report into a pandas or polars data frame.

        :param backend: `"pandas"` or `"polars"`. The library must be installed.
        :param layout: `"wide"` for a row for each time and a column for each vector, or `"long"`
            for a row for each record.
        :param keys: Vector keys or patterns such as `"WBHP:*"` to include. Every vector if not given.
        :param start: Earliest elapsed time to include.
        :param end: Latest elapsed time to include.
        :param include_dates: Whether to add a `date` column. Added when any record has a date if not given.
        :param include_time_steps: Whether to add a `time_step` column. Added when any record
            has a time-step index if not given.
        :param split_keys: For the `"long"` layout, add `vector` and `qualifier` columns.
        :param categorical: For the `"long"` layout, store the key columns as categoricals.
        :returns: A `pandas.DataFrame` or `polars.DataFrame`.
        """
        return as_dataframe(
            self,
            backend=backend,
            layout=layout,
            keys=keys,
            start=start,
            end=end,
            include_dates=include_dates,
            include_time_steps=include_time_steps,
            split_keys=split_keys,
            categorical=categorical,
        )


@attrs.frozen(kw_only=True, slots=True)
class RecordSummary(Action[ModelT]):
    """
    Evaluates `quantities` and records what they return to the run's `SummaryReport`.

    Pair with whichever `Event` should drive reporting cadence (`IntervalEvent`
    for "every N time units", `TimeStepEvent` for "every timestep", and so on).
    """

    quantities: tuple[Summary[ModelT], ...] = attrs.field(converter=tuple)
    """Every vector to evaluate and record when this action fires."""

    def __call__(self, model: ModelT, context: ScheduleContext) -> ModelT:
        """
        Evaluates every one of `quantities` and records each under its own key.

        :param model: The model to read from. Returned unchanged.
        :param context: The current moment's context. `context.extra` must
            carry a `"summary_report"` key with a `SummaryReport` value.
        :returns: `model`, unchanged.
        :raises SummaryError: If `context.extra["summary_report"]` is
            missing or isn't a `SummaryReport`.
        """
        report = context.extra.get("summary_report")
        if not isinstance(report, SummaryReport):
            raise SummaryError(
                f"{type(self).__name__} needs key 'summary_report' in `context.extra`, "
                f"with a `SummaryReport` value, but got {report!r}."
            )

        date = None
        if context.start_date is not None:
            date = get_current_date(
                start_date=context.start_date,
                elapsed_time=context.time,
                unit_system=context.unit_system,
            )

        for quantity in self.quantities:
            value = quantity(model, context)
            report.record(
                key=quantity.key,
                time=context.time,
                value=value,
                time_step=context.time_step,
                date=date,
            )
        return model


def record_every(
    *,
    every: Number,
    quantities: typing.Sequence[Summary[ModelT]],
    start: Number = 0.0,
    name: str | None = None,
    max_firings: Number | None = None,
) -> ScheduleItem[ModelT]:
    """
    Builds a `ScheduleItem` recording `quantities` every `every` time units.

    The common case: Eclipse-style reporting at a fixed cadence, all
    requested vectors recorded together at each report time.

    :param every: How often to record, in the schedule's unit system.
    :param quantities: Every vector to record at each report time.
    :param start: The first time recording is eligible to fire.
    :param name: Optional label for the item.
    :param max_firings: The most times to record. Defaults to no limit.
    :returns: A `ScheduleItem` pairing an `IntervalEvent` with `RecordSummary`.
    """
    return ScheduleItem(
        event=IntervalEvent(every=every, start=start),
        action=RecordSummary(quantities=quantities),
        name=name
        or f"record_every(every={every!r}, {','.join(quantity.key for quantity in quantities)})",
        max_firings=max_firings,
    )


def record_at(
    *,
    quantities: typing.Sequence[Summary[ModelT]],
    time: Number | None = None,
    time_step: int | None = None,
    name: str | None = None,
) -> ScheduleItem[ModelT]:
    """
    Builds a `ScheduleItem` recording `quantities` once, at an elapsed time
    or at a time step.

    :param quantities: Every vector to record.
    :param time: Elapsed time to record at. Give exactly one of `time` or `time_step`.
    :param time_step: Time-step index to record at.
    :param name: Optional label for the item.
    :returns: A `ScheduleItem` pairing a `TimeEvent` or `TimeStepEvent`
        with `RecordSummary`.
    :raises ValidationError: If neither or both of `time` and `time_step` are given.
    """
    if (time is None) == (time_step is None):
        raise ValidationError("Give exactly one of `time` or `time_step`.")

    event: TimeEvent | TimeStepEvent
    if time is not None:
        event = TimeEvent(at=time)
        label = f"record_at(time={time!r})"
    else:
        event = TimeStepEvent(at=typing.cast(int, time_step))
        label = f"record_at(time_step={time_step!r})"
    return ScheduleItem(
        event=event,
        action=RecordSummary(quantities=quantities),
        name=name or f"{label[:-1]}, {','.join(quantity.key for quantity in quantities)})",
    )


def select_summary_keys(
    report: SummaryReport, keys: str | typing.Sequence[str] | None
) -> list[str]:
    """
    Works out which of a report's vectors a data frame covers.

    :param report: The report to select from.
    :param keys: Vector keys, or Eclipse-style patterns such as `"WBHP:*"` or `"F*PR"`, or
        `None` for every recorded vector.
    :returns: The matching keys, in the order they were first recorded for `None` and in the
        order they were asked for otherwise, each once.
    :raises ValidationError: If a key was never recorded or a pattern matches nothing.
    """
    report.update_index()
    available = list(report.key_indices)
    if keys is None:
        return available

    chosen: list[str] = []
    for pattern in [keys] if isinstance(keys, str) else list(keys):
        if any(character in pattern for character in "*?["):
            matches = [key for key in available if fnmatch.fnmatchcase(key, pattern)]
        else:
            matches = [pattern] if pattern in report.key_indices else []
        if not matches:
            shown = available[:20]
            more = f" and {len(available) - 20} more" if len(available) > 20 else ""
            raise ValidationError(
                f"No recorded summary vector matches {pattern!r}. Recorded vectors: {shown}{more}."
            )
        chosen.extend(key for key in matches if key not in chosen)
    return chosen


@typing.overload
def as_dataframe(
    report: SummaryReport,
    *,
    backend: typing.Literal["pandas"] = "pandas",
    layout: SummaryLayout = ...,
    keys: str | typing.Sequence[str] | None = ...,
    start: Number | None = ...,
    end: Number | None = ...,
    include_dates: Boolean | None = ...,
    include_time_steps: Boolean | None = ...,
    split_keys: Boolean = ...,
    categorical: Boolean = ...,
) -> "pd.DataFrame": ...
@typing.overload
def as_dataframe(
    report: SummaryReport,
    *,
    backend: typing.Literal["polars"],
    layout: SummaryLayout = ...,
    keys: str | typing.Sequence[str] | None = ...,
    start: Number | None = ...,
    end: Number | None = ...,
    include_dates: Boolean | None = ...,
    include_time_steps: Boolean | None = ...,
    split_keys: Boolean = ...,
    categorical: Boolean = ...,
) -> "pl.DataFrame": ...
@typing.overload
def as_dataframe(
    report: SummaryReport,
    *,
    backend: Backend,
    layout: SummaryLayout = ...,
    keys: str | typing.Sequence[str] | None = ...,
    start: Number | None = ...,
    end: Number | None = ...,
    include_dates: Boolean | None = ...,
    include_time_steps: Boolean | None = ...,
    split_keys: Boolean = ...,
    categorical: Boolean = ...,
) -> AnyDataFrame: ...
def as_dataframe(
    report: SummaryReport,
    *,
    backend: Backend = "pandas",
    layout: SummaryLayout = "wide",
    keys: str | typing.Sequence[str] | None = None,
    start: Number | None = None,
    end: Number | None = None,
    include_dates: Boolean | None = None,
    include_time_steps: Boolean | None = None,
    split_keys: Boolean = False,
    categorical: Boolean = False,
) -> AnyDataFrame:
    """
    Turns a `SummaryReport` into a pandas or polars data frame.

    In the `"wide"` layout, the default, there is one row for each time any chosen vector was
    recorded and one column for each vector, named by its key. A vector that was not recorded at
    some time has a missing value there: NaN in pandas and null in polars. This is the form to
    plot or to compare vectors in.

    In the `"long"` layout there is one row for each record, with the columns `key`, `time` and
    `value`. This is the form to group by well or to filter on a vector.

    Values are double precision. Times are in the run's unit system, as recorded.

    :param report: The report to convert.
    :param backend: `"pandas"` or `"polars"`. The library must be installed.
    :param layout: `"wide"` or `"long"`.
    :param keys: Vector keys or patterns such as `"WBHP:*"`, one or a sequence, to include. Every
        recorded vector if not given.
    :param start: Earliest elapsed time to include. No lower limit if not given.
    :param end: Latest elapsed time to include. No upper limit if not given.
    :param include_dates: Whether to add a `date` column. If not given, it is added when any
        included record has a date.
    :param include_time_steps: Whether to add a `time_step` column. If not given, it is added
        when any included record has a time-step index.
    :param split_keys: For the `"long"` layout, add `vector` and `qualifier` columns, which
        split `"WBHP:PROD1"` into `"WBHP"` and `"PROD1"`. The qualifier is missing for a key with
        no colon.
    :param categorical: For the `"long"` layout, store the `key`, `vector` and `qualifier`
        columns as categoricals, which saves memory for a large report.
    :returns: A `pandas.DataFrame` or `polars.DataFrame`, ordered by time. Records of the
        `"long"` layout keep the order they were recorded in.
    :raises ValidationError: If an option is not valid for the layout, `start` is after `end`,
        or a key matches no recorded vector.
    :raises ImportError: If the library behind `backend` is not installed.
    """
    resolve_backend(backend)
    if layout not in ("wide", "long"):
        raise ValidationError(f"`layout` must be 'wide' or 'long'; got {layout!r}.")
    if layout == "wide" and (split_keys or categorical):
        raise ValidationError(
            "`split_keys` and `categorical` apply to the 'long' layout, where keys are values "
            "of a column. The 'wide' layout names a column for each key."
        )
    if start is not None and end is not None and start > end:
        raise ValidationError(f"`start` ({start}) must not be after `end` ({end}).")

    chosen = select_summary_keys(report, keys)
    records = report.records
    positions_by_key: dict[str, IntArray[OneDimension]] = {}
    for key in chosen:
        positions = np.asarray(report.key_indices[key], dtype=np.int64)
        times, _ = report.series(key)
        keep = np.ones(times.shape[0], dtype=np.bool_)
        if start is not None:
            keep &= times >= start
        if end is not None:
            keep &= times <= end
        positions_by_key[key] = typing.cast("IntArray[OneDimension]", positions[keep])

    all_positions = (
        np.sort(np.concatenate(list(positions_by_key.values())))
        if positions_by_key
        else np.empty(0, dtype=np.int64)
    )
    if include_dates is None:
        include_dates = any(records[position].date is not None for position in all_positions)
    if include_time_steps is None:
        include_time_steps = any(
            records[position].time_step is not None for position in all_positions
        )

    if layout == "long":
        selected = [records[position] for position in all_positions]
        columns = [
            Column(
                name="key",
                values=[record.key for record in selected],
                kind="str",
                categorical=categorical,
            )
        ]
        if split_keys:
            vectors, qualifiers = [], []
            for record in selected:
                vector, separator, qualifier = record.key.partition(":")
                vectors.append(vector)
                qualifiers.append(qualifier if separator else None)
            columns += [
                Column(name="vector", values=vectors, kind="str", categorical=categorical),
                Column(name="qualifier", values=qualifiers, kind="str", categorical=categorical),
            ]
        columns += [
            Column(
                name="time",
                values=np.fromiter((record.time for record in selected), dtype=np.float64),
            ),
            Column(
                name="value",
                values=np.fromiter((record.value for record in selected), dtype=np.float64),
            ),
        ]
        if include_time_steps:
            columns.append(
                Column(
                    name="time_step", values=[record.time_step for record in selected], kind="int"
                )
            )
        if include_dates:
            columns.append(
                Column(name="date", values=[record.date for record in selected], kind="datetime")
            )
        return build_dataframe(columns, backend=backend)

    series: dict[str, tuple[NumberArray[OneDimension], NumberArray[OneDimension]]] = {}
    for key in chosen:
        times, values = report.series(key)
        keep = np.isin(
            np.asarray(report.key_indices[key], dtype=np.int64),
            positions_by_key[key],
            assume_unique=True,
        )
        series[key] = typing.cast(
            "tuple[NumberArray[OneDimension], NumberArray[OneDimension]]",
            (times[keep], values[keep]),
        )

    times_present = [times for times, _ in series.values() if times.size]
    row_times = np.unique(np.concatenate(times_present)) if times_present else np.empty(0)
    n_rows = row_times.shape[0]

    columns = [Column(name="time", values=row_times)]
    if include_dates or include_time_steps:
        dates: list[datetime.datetime | None] = [None] * n_rows
        time_steps: list[int | None] = [None] * n_rows
        for key in chosen:
            rows = np.searchsorted(row_times, series[key][0])
            for row, position in zip(rows, positions_by_key[key], strict=True):
                record = records[position]
                if record.date is not None and dates[row] is None:
                    dates[row] = record.date
                if record.time_step is not None and time_steps[row] is None:
                    time_steps[row] = record.time_step
        if include_dates:
            columns.append(Column(name="date", values=dates, kind="datetime"))
        if include_time_steps:
            columns.append(Column(name="time_step", values=time_steps, kind="int"))

    for key in chosen:
        times, values = series[key]
        column = np.full(n_rows, np.nan, dtype=np.float64)
        column[np.searchsorted(row_times, times)] = values
        columns.append(Column(name=key, values=column, missing_is_null=True))
    return build_dataframe(columns, backend=backend)

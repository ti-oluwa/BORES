"""
Summary-vector recording API.

A summary vector (Eclipse's own term: `FOPR`, `WBHP`, `WWCT`, and so on) is
just another thing a schedule can read at a point in time. It fires on the
same `Event`s as any other scheduled item, and the values it reads a recorded in
a `SummaryReport` rather than mutating the model.

`Summary` is the read-only counterpart to `Action`. An `Action` changes the model
an `Event` fires against; a `Summary` reports on it instead, via `RecordSummary`,
an `Action` that evaluates one or more `Summary` quantities and records what they
return in a `SummaryReport`.
"""

import datetime
import typing

import attrs
import numpy as np

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
from bores.types import Number, NumberArray, OneDimension
from bores.utils import get_current_date

__all__ = [
    "RecordSummary",
    "Summary",
    "SummaryRecord",
    "SummaryReport",
    "record_at",
    "record_every",
]


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
    `ScheduleContext.extra["summary_report"]`; `RecordSummary` is the
    only thing that appends to it.

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
        """
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
        :returns: `(times, values)`, each shape `(n_records_for_key,)`.
            Both empty if `key` was never recorded. Rebuilt only when
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
            NumberArray[OneDimension],
            np.fromiter((records[position].time for position in positions), dtype=np.float64),
        )
        values = typing.cast(
            NumberArray[OneDimension],
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
        return typing.cast(NumberArray[OneDimension], times[mask]), typing.cast(
            NumberArray[OneDimension], values[mask]
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


@attrs.frozen(kw_only=True, slots=True)
class RecordSummary(Action[ModelT]):
    """
    Evaluates `quantities` and records what they return to the run's `SummaryReport`.

    Pair with whichever `Event` should drive reporting cadence (`IntervalEvent`
    for "every N time units", `TimeStepEvent` for "every timestep", and so on).
    """

    __type__: typing.ClassVar[str] = "record_summary"

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
) -> ScheduleItem[ModelT]:
    """
    Builds a `ScheduleItem` recording `quantities` every `every` time units.

    The common case: Eclipse-style reporting at a fixed cadence, all
    requested vectors recorded together at each report time.

    :param every: How often to record, in the schedule's unit system.
    :param quantities: Every vector to record at each report time.
    :param start: The first time recording is eligible to fire.
    :param name: Optional label for the item.
    :returns: A `ScheduleItem` pairing an `IntervalEvent` with `RecordSummary`.
    """
    return ScheduleItem(
        event=IntervalEvent(every=every, start=start),
        action=RecordSummary(quantities=tuple(quantities)),
        name=name or f"record_every(every={every!r})",
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
        action=RecordSummary(quantities=tuple(quantities)),
        name=name or label,
    )

"""Times a simulation must land on exactly, such as event times, report times and dates."""

import datetime
import math
import numbers
import typing
from bisect import bisect_right
from collections.abc import Iterable

import attrs
from typing_extensions import Self

from bores.errors import ValidationError
from bores.types import Number, UnitSystem
from bores.units import get_conversion_factors
from bores.utils import get_current_time

__all__ = [
    "PeriodicTarget",
    "SupportsReportTimes",
    "SupportsTimeTargets",
    "TimeTargetSource",
    "TimeTargets",
    "build_time_targets",
]


@attrs.frozen(kw_only=True, slots=True)
class PeriodicTarget:
    """A target that recurs at a fixed interval, without listing every occurrence."""

    every: Number = attrs.field(validator=attrs.validators.gt(0))
    """Time between two occurrences. Must be positive."""

    start: Number = 0.0
    """Time of the first occurrence."""

    end: Number | None = None
    """Time after which no more occurrences are wanted. `None` keeps them going for the whole run."""

    def next_after(self, time: Number, *, tolerance: Number) -> Number | None:
        """
        Return the first recurring target after `time`.

        :param time: The time to look beyond.
        :param tolerance: Occurrences within this distance after `time` count as `time` itself.
        :returns: The first occurrence later than `time`, or `None` if there is none.
        """
        if time + tolerance < self.start:
            candidate = self.start
        else:
            count = math.floor((time + tolerance - self.start) / self.every) + 1
            candidate = self.start + count * self.every

        if self.end is not None and candidate > self.end + tolerance:
            return None
        return candidate

    def snap(self, time: Number, *, tolerance: Number) -> Number | None:
        """
        Return the recurring target nearest `time` within `tolerance`.

        :param time: A time that may have landed just beside an occurrence.
        :param tolerance: How close counts as landing on it.
        :returns: The occurrence within `tolerance` of `time`, or `None` if there is none.
        """
        if time < self.start - tolerance:
            return None

        count = round((time - self.start) / self.every)
        candidate = self.start + count * self.every
        if abs(candidate - time) > tolerance:
            return None
        if self.end is not None and candidate > self.end + tolerance:
            return None
        return candidate


def _sorted_unique(values: Iterable[Number]) -> tuple[Number, ...]:
    """Sorts times and drops repeats."""
    return tuple(sorted({float(value) for value in values}))


def _as_periodic_tuple(values: Iterable[PeriodicTarget]) -> tuple[PeriodicTarget, ...]:
    """Holds recurring targets as a tuple."""
    return tuple(values)


@attrs.frozen(slots=True)
class TimeTargets:
    """
    Every time a time stepper must land on exactly.

    This includes the times events fire at, report times, and anything else that changes
    the model at a known moment. Immutable, so combine two with `|` rather than mutating.
    """

    times: tuple[Number, ...] = attrs.field(converter=_sorted_unique, factory=tuple)
    """Individual times, sorted and without repeats, in the run's own time unit."""

    periodic: tuple[PeriodicTarget, ...] = attrs.field(converter=_as_periodic_tuple, factory=tuple)
    """Recurring targets, such as an event that fires every 30 days."""

    def __attrs_post_init__(self) -> None:
        for target in self.times:
            if not math.isfinite(target):
                raise ValidationError(f"Time targets must be finite, but got {target!r}.")

    def __bool__(self) -> bool:
        return bool(self.times or self.periodic)

    def __or__(self, other: typing.Any) -> Self:
        if not isinstance(other, TimeTargets):
            return NotImplemented
        return self.__class__(
            times=self.times + other.times,
            periodic=self.periodic + other.periodic,  # type: ignore[arg-type,misc,assignment]
        )

    def scaled(self, factor: Number) -> Self:
        """
        Return a copy with all target times scaled by `factor`.

        :param factor: Multiplier from the current time unit to the new one.
        :returns: A copy with every time and recurrence expressed in the new unit.
        """
        return self.__class__(
            times=[time * factor for time in self.times],
            periodic=[
                PeriodicTarget(  # type: ignore[arg-type,misc,assignment]
                    every=target.every * factor,
                    start=target.start * factor,
                    end=None if target.end is None else target.end * factor,
                )
                for target in self.periodic
            ],
        )

    def next_after(self, time: Number, *, tolerance: Number = 0.0) -> Number | None:
        """
        Return the earliest target after `time`.

        :param time: The time to look beyond.
        :param tolerance: A target within this distance after `time` counts as `time` itself,
            so a clock that landed a hair short of a target does not return that target again.
        :returns: The earliest target later than `time`, or `None` if there is none.
        """
        best: Number | None = None
        index = bisect_right(self.times, time + tolerance)  # type: ignore[arg-type]
        if index < len(self.times):
            best = self.times[index]
        for target in self.periodic:
            candidate = target.next_after(time, tolerance=tolerance)
            if candidate is not None and (best is None or candidate < best):
                best = candidate
        return best

    def snap(self, time: Number, *, tolerance: Number) -> Number | None:
        """
        Return the latest target within `tolerance` of `time`.

        :param time: A time that may have landed just beside a target.
        :param tolerance: How close counts as landing on it.
        :returns: The target `time` landed on, or `None` if it landed on none. When several
            targets lie within `tolerance` of each other, the latest is returned, so that an
            event at each of them has happened by the returned time.
        """
        best: Number | None = None
        index = bisect_right(self.times, time + tolerance)  # type: ignore[arg-type]
        while index > 0 and self.times[index - 1] >= time - tolerance:
            candidate = self.times[index - 1]
            if best is None or candidate > best:
                best = candidate
            index -= 1
        for target in self.periodic:
            candidate = target.snap(time, tolerance=tolerance)  # type: ignore[arg-type,misc,assignment]
            if candidate is not None and (best is None or candidate > best):
                best = candidate
        return best


@typing.runtime_checkable
class SupportsTimeTargets(typing.Protocol):
    """Anything that can say which times it needs a time stepper to land on, such as a `Schedule`."""

    def get_time_targets(
        self, *, start_date: datetime.datetime | None = None, unit_system: UnitSystem = ...
    ) -> TimeTargets:
        """
        :param start_date: The simulation's start date.
        :param unit_system: Unit system the times are expressed in.
        :returns: The times to land on.
        """
        ...


@typing.runtime_checkable
class SupportsReportTimes(typing.Protocol):
    """Anything that carries report times, such as a `DeckFile`."""

    @property
    def report_times(self) -> typing.Sequence[Number]:
        """Report times, in the source's own time unit."""
        ...


TimeTargetSource: typing.TypeAlias = (
    TimeTargets
    | PeriodicTarget
    | datetime.datetime
    | Number
    | SupportsTimeTargets
    | SupportsReportTimes
    | Iterable["TimeTargetSource"]
)
"""
Anything `build_time_targets` accepts: a schedule, a deck file, an elapsed time, a date, a
`PeriodicTarget`, `TimeTargets`, or a collection of any of those.
"""


def build_time_targets(
    *sources: TimeTargetSource,
    start_date: datetime.datetime | None = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
) -> TimeTargets:
    """
    Combines every source of times a simulation must land on into one `TimeTargets`.

    A source can be any of the following, and `sources` can mix them freely:

    - A `Schedule`, which contributes the times its events fire at.
    - A `DeckFile`, which contributes its report times (every `TSTEP` entry and `DATES` date).
      They are converted from the deck's own unit system to `unit_system`.
    - A number, which is an elapsed time in `unit_system`'s native time unit.
    - A `datetime`, which is converted to an elapsed time from `start_date`.
    - A `PeriodicTarget`, for a time that recurs at a fixed interval.
    - A `TimeTargets`, which is merged as it is.
    - Any collection of the above, such as a list, tuple, set or generator.

    :param sources: What to collect times from.
    :param start_date: The simulation's start date. Required if a source is or contains a
        `datetime`, or if a schedule has a `DateEvent`.
    :param unit_system: Unit system the result is expressed in.
    :returns: Every time from every source, sorted and without repeats.
    :raises ValidationError: If a source is not one of the kinds above, or a `datetime` is given
        without `start_date`.
    """
    times: list[Number] = []
    periodic: list[PeriodicTarget] = []

    def collect(source: TimeTargetSource) -> None:
        if isinstance(source, TimeTargets):
            times.extend(source.times)
            periodic.extend(source.periodic)
        elif isinstance(source, PeriodicTarget):
            periodic.append(source)
        elif isinstance(source, datetime.datetime):
            if start_date is None:
                raise ValidationError(
                    f"`start_date` is needed to turn the date {source!r} into an elapsed time."
                )
            times.append(
                get_current_time(
                    start_date=start_date, current_date=source, unit_system=unit_system
                )
            )
        elif isinstance(source, SupportsTimeTargets):
            targets = source.get_time_targets(start_date=start_date, unit_system=unit_system)
            times.extend(targets.times)
            periodic.extend(targets.periodic)
        elif isinstance(source, SupportsReportTimes):
            source_unit_system = getattr(source, "unit_system", unit_system)
            factor = get_conversion_factors(source_unit_system, unit_system)["time"]
            times.extend(float(time) * factor for time in source.report_times)
        elif isinstance(source, numbers.Real) and not isinstance(source, bool):
            times.append(float(source))
        elif isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
            for item in source:
                collect(item)
        else:
            raise ValidationError(
                f"Cannot get time targets from {source!r}. Use a schedule, a deck file, a number, "
                "a datetime, a `PeriodicTarget`, `TimeTargets`, or a list of those."
            )

    for source in sources:
        collect(source)
    return TimeTargets(times=times, periodic=periodic)

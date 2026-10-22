"""Times a simulation must land on exactly, such as event times, report times and dates."""

import math
import typing
from bisect import bisect_right

import attrs
from typing_extensions import Self

from bores.errors import ValidationError
from bores.types import Number

__all__ = ["PeriodicTarget", "TimeTargets"]


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


@attrs.frozen(slots=True)
class TimeTargets:
    """
    Every time a time stepper must land on exactly.
     
    This includes the times events fire at, report times, and anything else that changes 
    the model at a known moment. Immutable, so combine two with `|` rather than mutating.
    """

    times: tuple[Number, ...] = attrs.field(
        converter=lambda values: tuple(sorted(set(float(value) for value in values))),
        factory=tuple,
    )
    """Individual times, sorted and without repeats, in the run's own time unit."""

    periodic: tuple[PeriodicTarget, ...] = attrs.field(converter=tuple, factory=tuple)
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

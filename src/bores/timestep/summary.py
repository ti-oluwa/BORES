"""
Summary vectors that report on the time stepping of a run, read from its `Timer`.

The run must give the timer to the schedule under `context.extra["timer"]`, next to the
workspace under `"workspace"`. Both vectors read the most recent accepted step, so they hold the
same value until the next step is accepted.
"""

import typing

import attrs

from bores.errors import SummaryError
from bores.schedule.base import ScheduleContext
from bores.schedule.summary import Summary
from bores.timestep.control import StepMetrics, Timer
from bores.types import Number

if typing.TYPE_CHECKING:
    from bores.blackoil.compile import CompiledBlackOilModel

__all__ = [
    "NEWTON",
    "TIMESTEP",
    "NewtonIterations",
    "TimeStepSize",
    "get_last_accepted_step",
    "get_timer",
]


def get_timer(*, context: ScheduleContext) -> Timer:
    """
    Reads the run's `Timer` from `context.extra["timer"]`.

    :param context: The current moment's context.
    :returns: The run's `Timer`.
    :raises SummaryError: If the key is missing or isn't a `Timer`.
    """
    timer = context.extra.get("timer")
    if not isinstance(timer, Timer):
        raise SummaryError(
            "Time stepping summary vectors need key 'timer' in `context.extra`, with a "
            f"`Timer` value, but got {timer!r}."
        )
    return timer


def get_last_accepted_step(*, timer: Timer) -> StepMetrics | None:
    """
    Finds the most recent accepted step in a timer's recent step metrics.

    :param timer: The run's timer.
    :returns: The metrics of the latest accepted step, or `None` if no step has been accepted yet
        (or the timer has since dropped every accepted step from its bounded history).
    """
    for metrics in reversed(timer.recent_metrics):
        if metrics.success:
            return metrics
    return None


@attrs.frozen(kw_only=True, slots=True)
class TimeStepSize(Summary["CompiledBlackOilModel"]):
    """Length of the latest accepted time step (`TIMESTEP`), in the timer's native time unit."""

    @property
    def key(self) -> str:
        """
        Returns this vector's report key.

        :returns: `"TIMESTEP"`.
        """
        return "TIMESTEP"

    def __call__(self, model: "CompiledBlackOilModel", context: ScheduleContext) -> Number:
        """
        Reads the latest accepted step's size.

        :param model: Unused. Present to satisfy the `Summary` protocol.
        :param context: The current moment's context, carrying the timer.
        :returns: The step size, or `NaN` before the first step is accepted.
        """
        metrics = get_last_accepted_step(timer=get_timer(context=context))
        return float("nan") if metrics is None else metrics.step_size


@attrs.frozen(kw_only=True, slots=True)
class NewtonIterations(Summary["CompiledBlackOilModel"]):
    """Newton iterations the latest accepted time step took (`NEWTON`)."""

    @property
    def key(self) -> str:
        """
        Returns this vector's report key.

        :returns: `"NEWTON"`.
        """
        return "NEWTON"

    def __call__(self, model: "CompiledBlackOilModel", context: ScheduleContext) -> Number:
        """
        Reads the latest accepted step's Newton iteration count.

        :param model: Unused. Present to satisfy the `Summary` protocol.
        :param context: The current moment's context, carrying the timer.
        :returns: The iteration count, or `NaN` before the first step is accepted or when the
            solver did not report a count for it.
        """
        metrics = get_last_accepted_step(timer=get_timer(context=context))
        if metrics is None or metrics.newton_iterations is None:
            return float("nan")
        return metrics.newton_iterations


TIMESTEP = TimeStepSize
NEWTON = NewtonIterations

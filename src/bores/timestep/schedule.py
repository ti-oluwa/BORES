"""Changing a `Timer` part-way through a run, as a scheduled action."""

import typing

import attrs

from bores.errors import ActionError
from bores.schedule.base import Action, ModelT, ScheduleContext
from bores.timestep.control import Timer
from bores.types import Number

__all__ = ["TIMER_CONTEXT_KEY", "RetuneTimer"]

TIMER_CONTEXT_KEY = "timer"
"""The key of `ScheduleContext.extra` that holds the run's `Timer` for `RetuneTimer` to change."""


@attrs.frozen(kw_only=True, slots=True)
class RetuneTimer(Action[ModelT]):
    """
    Changes some of the run's `Timer` settings when applied, the schedule's form of a `TUNING`
    that appears after the start of the run.

    The timer is not part of the model, so the code that advances the schedule has to hand it over
    in `ScheduleContext.extra` under `TIMER_CONTEXT_KEY`. The model is returned unchanged.
    """

    settings: typing.Mapping[str, Number]
    """New timer settings by name (see `bores.timestep.control.RETUNABLE_SETTINGS`)."""

    def __call__(self, model: ModelT, context: ScheduleContext) -> ModelT:
        """
        Applies `settings` to the timer in the context.

        :param model: The model being scheduled against. Returned as it is.
        :param context: The current moment's context, holding the timer in `extra`.
        :returns: `model`, unchanged.
        :raises ActionError: If the context carries no timer.
        """
        timer = context.extra.get(TIMER_CONTEXT_KEY)
        if not isinstance(timer, Timer):
            raise ActionError(
                f"`RetuneTimer` needs the run's `Timer` in `context.extra[{TIMER_CONTEXT_KEY!r}]`, "
                f"but found {type(timer).__name__}."
            )
        timer.retune(self.settings)
        return model

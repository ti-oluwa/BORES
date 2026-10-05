"""Builtin `Action` implementations."""

import attrs

from bores.errors import StopSimulation
from bores.schedule.base import Action, ModelT, ScheduleContext

__all__ = ["EndRun", "NoOp", "RunSequence"]


@attrs.frozen(kw_only=True, slots=True)
class EndRun(Action[ModelT]):
    """
    Ends the simulation unconditionally when applied. Pair with an event
    for a manual stop condition, the schedule-driven equivalent of
    Eclipse's `END` keyword, separate from the automatic end-run an
    economic limit's own `end_run` flag can already raise.
    """

    reason: str = "Schedule requested an end to the run."
    """Carried on the `StopSimulation` this action raises."""

    def __call__(self, model: ModelT, context: ScheduleContext) -> ModelT:
        """
        Always raises. Never returns.

        :param model: The model being scheduled against. Unused.
        :param context: The current moment's context. Unused.
        :raises StopSimulation: Always, with `reason` as its message.
        """
        raise StopSimulation(self.reason)


@attrs.frozen(kw_only=True, slots=True)
class NoOp(Action[ModelT]):
    """Does nothing. Useful as a placeholder, or to pair with an event you only want logged."""

    label: str | None = None
    """Optional note on what this no-op stands in for."""

    def __call__(self, model: ModelT, context: ScheduleContext) -> ModelT:
        """
        Returns `model` unchanged.

        :param model: The model being scheduled against.
        :param context: The current moment's context. Unused.
        :returns: `model`, unchanged.
        """
        return model


@attrs.frozen(kw_only=True, slots=True)
class RunSequence(Action[ModelT]):
    """Applies `actions` in order, threading the model through each in turn."""

    actions: tuple[Action[ModelT], ...] = attrs.field(converter=tuple)
    """Every action to apply, in order."""

    def __call__(self, model: ModelT, context: ScheduleContext) -> ModelT:
        """
        Applies every one of `actions` in order.

        :param model: The model to change.
        :param context: The current moment's context.
        :returns: The model after every action in `actions` has run.
        """
        for action in self.actions:
            model = action(model, context)
        return model

"""Builtin `Action` implementations."""

import attrs

from bores.schedule.base import ModelT, ScheduleContext, SerializableAction, action_type

__all__ = ["NoOp", "RunSequence"]


@action_type
@attrs.frozen(kw_only=True, slots=True)
class NoOp(SerializableAction[ModelT]):
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


@action_type
@attrs.frozen(kw_only=True, slots=True)
class RunSequence(SerializableAction[ModelT]):
    """Applies `actions` in order, threading the model through each in turn."""

    actions: tuple[SerializableAction, ...] = attrs.field(converter=tuple)
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

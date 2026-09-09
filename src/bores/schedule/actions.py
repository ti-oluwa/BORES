"""Builtin `Action` implementations: `NoOpAction`, `Actions`."""

import attrs

from bores.schedule.base import ModelT, ScheduleContext, SerializableAction, action_type

__all__ = ["Actions", "NoOpAction"]


@action_type
@attrs.frozen(kw_only=True, slots=True)
class NoOpAction(SerializableAction[ModelT]):
    """Does nothing. Useful as a placeholder, or to pair with an event you only want logged."""

    label: str | None = None
    """Optional note on what this no-op stands in for."""

    def __call__(self, model: ModelT, context: ScheduleContext) -> ModelT:
        return model


@action_type
@attrs.frozen(kw_only=True, slots=True)
class Actions(SerializableAction[ModelT]):
    """Applies `actions` in order, threading the model through each in turn."""

    actions: tuple[SerializableAction, ...] = attrs.field(converter=tuple)

    def __call__(self, model: ModelT, context: ScheduleContext) -> ModelT:
        for action in self.actions:
            model = action(model, context)
        return model

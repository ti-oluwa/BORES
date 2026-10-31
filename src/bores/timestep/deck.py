"""Building a `Timer` from the time-stepping keywords of a deck."""

import typing
import warnings

import attrs

from bores.deck.file import DeckFile
from bores.deck.keywords.schedule import TUNING
from bores.errors import ValidationError
from bores.schedule.base import ScheduleItem
from bores.schedule.events import TimeEvent
from bores.timestep.control import Timer
from bores.timestep.schedule import RetuneTimer
from bores.types import Number

__all__ = [
    "TUNING_DEFAULTS",
    "get_tuning_at_start",
    "load_timer",
    "load_timer_schedule",
    "tuning_to_timer_settings",
]

TUNING_DEFAULTS: dict[str, typing.Any] = {
    field.name: field.default for _, fields in TUNING.records for field in fields
}
"""
What each `TUNING` item is when the deck leaves it out, keyed by item name. A deck value equal to
its default is taken as "not customised", because `TUNING` fills in a default for every item and a
defaulted item cannot be told from one written out.
"""


def get_tuning_occurrences(deck_file: DeckFile) -> list[dict[str, typing.Any]]:
    """
    Every `TUNING` in the deck, in file order.

    :param deck_file: The deck to read.
    :returns: One dict per occurrence, holding the `time_step`, `convergence` and `iteration`
        groups and the `schedule_time` it takes effect at.
    """
    return list(deck_file.get_shared("TUNING") or [])


def get_tuning_at_start(deck_file: DeckFile) -> dict[str, dict[str, typing.Any]] | None:
    """
    The `TUNING` settings that apply from the start of the run.

    A `TUNING` in the schedule changes the settings from the time it appears at. This is the one
    in effect at time zero; the later ones come from `load_timer_schedule`.

    :param deck_file: The deck to read.
    :returns: The settings grouped as `time_step`, `convergence` and `iteration` (each a dict of
        item name to value), or `None` if the deck has no `TUNING` at time zero.
    """
    at_start = [
        occurrence
        for occurrence in get_tuning_occurrences(deck_file)
        if occurrence.get("schedule_time", 0.0) == 0
    ]
    if not at_start:
        return None
    # Several at time zero: the last one written wins, as it would in the deck
    return {group: dict(items) for group, items in at_start[-1].items() if isinstance(items, dict)}


def tuning_to_timer_settings(
    tuning: dict[str, dict[str, typing.Any]] | None, *, include_step_sizes: bool
) -> dict[str, Number]:
    """
    Turn the `TUNING` items a `Timer` can use into `Timer` settings.

    - `initial_time_step`, `maximum_time_step` and `minimum_time_step` become the step sizes.
    - `maximum_time_step_increase_factor` becomes `maximum_growth_per_step`,
      `convergence_failure_cut_factor` the `backoff_factor` (with `aggressive_backoff_factor` kept
      from exceeding it) and `maximum_material_balance_error` `maximum_total_relative_mbe`.

    Only an item the deck sets to something other than Eclipse's default is taken, since `TUNING`
    fills in a default for every item and a defaulted item cannot be told from one written out.
    The step sizes are the exception when `include_step_sizes` is set, as a `Timer` cannot be
    built without them. The other `TUNING` items have no `Timer` setting and are not carried over.

    :param tuning: One `TUNING` occurrence's groups, or `None` for none.
    :param include_step_sizes: Whether to give the three step sizes even where they are the
        defaults.
    :returns: Timer settings by name.
    """
    tuning = tuning or {}
    time_step_items = tuning.get("time_step", {})
    convergence_items = tuning.get("convergence", {})

    def item_value(items: dict[str, typing.Any], name: str) -> float:
        return float(items.get(name, TUNING_DEFAULTS[name]))

    def differs_from_default(items: dict[str, typing.Any], name: str) -> bool:
        return item_value(items, name) != TUNING_DEFAULTS[name]

    settings: dict[str, Number] = {}
    for item_name, setting_name in (
        ("initial_time_step", "initial_step_size"),
        ("maximum_time_step", "maximum_step_size"),
        ("minimum_time_step", "minimum_step_size"),
    ):
        if include_step_sizes or differs_from_default(time_step_items, item_name):
            settings[setting_name] = item_value(time_step_items, item_name)

    increase_factor = item_value(time_step_items, "maximum_time_step_increase_factor")
    if (
        differs_from_default(time_step_items, "maximum_time_step_increase_factor")
        and increase_factor > 1.0
    ):
        settings["maximum_growth_per_step"] = increase_factor

    cut_factor = item_value(time_step_items, "convergence_failure_cut_factor")
    if (
        differs_from_default(time_step_items, "convergence_failure_cut_factor")
        and 0.0 < cut_factor < 1.0
    ):
        settings["backoff_factor"] = cut_factor
        # The harder cut after a failed step can never be milder than the ordinary one
        settings["aggressive_backoff_factor"] = min(
            attrs.fields(Timer).aggressive_backoff_factor.default, cut_factor
        )

    if differs_from_default(convergence_items, "maximum_material_balance_error"):
        settings["maximum_total_relative_mbe"] = item_value(
            convergence_items, "maximum_material_balance_error"
        )
    return settings


def load_timer(deck_file: DeckFile) -> Timer | None:
    """
    Build a `Timer` from the deck's schedule length and the `TUNING` in effect at the start.

    `simulation_time` is the last report time (the end of the final `TSTEP` or `DATES`). The
    other settings follow `tuning_to_timer_settings`, with Eclipse's own defaults for the step
    sizes when the deck has no `TUNING`. Times are in the deck's native time unit (days for
    `FIELD` and `METRIC`, hours for `LAB`), which is also what the `Timer` uses for the same unit
    system.

    A `TUNING` that appears later in the schedule does not change this timer. It is warned about,
    and `load_timer_schedule` gives the scheduled changes that follow it.

    :param deck_file: The deck to read.
    :returns: The timer, or `None` if the deck has no schedule length to run for.
    :raises ValidationError: If the deck's step sizes cannot all hold (a minimum above the
        maximum).
    """
    simulation_time = deck_file.report_times[-1] if deck_file.report_times else 0.0
    if not simulation_time > 0.0:
        return None

    settings = tuning_to_timer_settings(get_tuning_at_start(deck_file), include_step_sizes=True)
    minimum_step_size = settings["minimum_step_size"]
    maximum_step_size = settings["maximum_step_size"]
    if not 0.0 < minimum_step_size <= maximum_step_size:
        raise ValidationError(
            f"`TUNING` step sizes are inconsistent: the minimum time step {minimum_step_size:g} "
            f"must be positive and no larger than the maximum time step {maximum_step_size:g}."
        )

    # Eclipse takes the initial step as a wish and keeps it inside the limits
    settings["initial_step_size"] = min(
        max(settings["initial_step_size"], minimum_step_size), maximum_step_size
    )

    later_times = sorted({
        occurrence["schedule_time"]
        for occurrence in get_tuning_occurrences(deck_file)
        if occurrence.get("schedule_time", 0.0) != 0
    })
    if later_times:
        warnings.warn(
            "`TUNING` appears again at "
            + ", ".join(f"{time:g}" for time in later_times)
            + f" ({deck_file.unit_system.name} time unit). The timer built here only has the "
            "settings from the start; `load_timer_schedule` gives the changes to apply at those "
            "times.",
            stacklevel=2,
        )
    return Timer(
        simulation_time=simulation_time,
        unit_system=deck_file.unit_system,
        **settings,  # type: ignore[arg-type]
    )


def load_timer_schedule(deck_file: DeckFile) -> tuple[ScheduleItem[typing.Any], ...]:
    """
    The scheduled timer changes from every `TUNING` that appears after the start of the run.

    Each becomes one item that applies `RetuneTimer` at the time the `TUNING` takes effect, with
    only the settings that `TUNING` sets to something other than Eclipse's default (see
    `tuning_to_timer_settings`). An occurrence that sets none of them gives no item. The items
    only work if the code advancing the schedule puts the run's `Timer` in
    `ScheduleContext.extra` under `TIMER_CONTEXT_KEY`.

    :param deck_file: The deck to read.
    :returns: The items, in the order the `TUNING` records appear.
    """
    items: list[ScheduleItem[typing.Any]] = []
    for occurrence in get_tuning_occurrences(deck_file):
        time = occurrence.get("schedule_time", 0.0)
        if time == 0:
            continue

        groups = {
            group: dict(values) for group, values in occurrence.items() if isinstance(values, dict)
        }
        settings = tuning_to_timer_settings(groups, include_step_sizes=False)
        if not settings:
            continue
        items.append(
            ScheduleItem(
                event=TimeEvent(at=float(time)),
                action=RetuneTimer(settings=settings),
                name=f"tuning at {float(time):g}",
            )
        )
    return tuple(items)

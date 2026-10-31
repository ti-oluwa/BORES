"""Building a `Timer` from the time-stepping keywords of a deck."""

import typing
import warnings

from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.timestep.control import Timer

__all__ = ["TUNING_DEFAULTS", "get_tuning_at_start", "load_timer"]

TUNING_DEFAULTS: dict[str, float] = {
    "tsfmax": 3.0,
    "tsfcnv": 0.1,
    "xxxmbe": 1e-06,
}
"""
What `TUNING` assumes for the items a `Timer` can use optionally. A deck value equal to one of
these is taken as "not customised" and leaves the `Timer`'s own setting alone, because `TUNING`
carries a default for every item and a defaulted item cannot be told from one written out.
"""


def get_tuning_at_start(deck_file: DeckFile) -> dict[str, dict[str, typing.Any]] | None:
    """
    The `TUNING` settings that apply from the start of the run.

    A `TUNING` record changes the settings from the time it appears at, which a single `Timer`
    cannot follow, so only the one in effect at time zero is returned. The ones that come later
    are reported with a warning.

    :param deck_file: The deck to read.
    :returns: The settings grouped as `time_step`, `convergence` and `iteration` (each a dict of
        item name to value), or `None` if the deck has no `TUNING` at time zero.
    """
    occurrences: list[dict[str, typing.Any]] = deck_file.get_shared("TUNING") or []
    at_start = [item for item in occurrences if item.get("schedule_time", 0) == 0]
    later = sorted({
        item["schedule_time"] for item in occurrences if item.get("schedule_time", 0) != 0
    })
    if later:
        warnings.warn(
            "`TUNING` appears again at "
            + ", ".join(f"{time:g}" for time in later)
            + f" ({deck_file.unit_system.name} time unit); only the one in effect at the start of "
            "the run is used, because a timer is configured once.",
            stacklevel=2,
        )
    if not at_start:
        return None
    # Several at time zero: the last one written wins, as it would in the deck
    return {
        group: dict(values) for group, values in at_start[-1].items() if isinstance(values, dict)
    }


def load_timer(deck_file: DeckFile) -> Timer | None:
    """
    Build a `Timer` from the deck's schedule length and its `TUNING`.

    - `simulation_time` is the last report time (the end of the final `TSTEP` or `DATES`).
    - `TSINIT`, `TSMAXZ` and `TSMINZ` give the initial, maximum and minimum step sizes. They are
      always read, with Eclipse's own defaults when the deck has no `TUNING`, since a `Timer`
      cannot be built without them.
    - `TSFMAX` becomes `maximum_growth_per_step`, `TSFCNV` the `backoff_factor` and `XXXMBE`
      `maximum_total_relative_mbe`, each only if the deck sets it to something other than
      Eclipse's default. The `Timer`'s own settings stand otherwise.

    Times are in the deck's native time unit (days for `FIELD` and `METRIC`, hours for `LAB`),
    which is also what the `Timer` uses for the same unit system. Items `Timer` has no use for
    (`TSMCHP`, `TSFMIN`, `TFDIFF`, `THRUPT`, `TMAXWC`, the `TRG*` targets and the other `XXX*`
    limits) are not carried over.

    :param deck_file: The deck to read.
    :returns: The timer, or `None` if the deck has no schedule length to run for.
    :raises ValidationError: If the deck's step sizes cannot all hold (a minimum above the
        maximum).
    """
    simulation_time = float(deck_file.report_times[-1]) if deck_file.report_times else 0.0
    if not simulation_time > 0.0:
        return None

    tuning = get_tuning_at_start(deck_file)
    time_step = (tuning or {}).get("time_step", {})
    convergence = (tuning or {}).get("convergence", {})
    initial = float(time_step.get("tsinit", 1.0))
    maximum = float(time_step.get("tsmaxz", 365.0))
    minimum = float(time_step.get("tsminz", 0.1))
    if not 0.0 < minimum <= maximum:
        raise ValidationError(
            f"`TUNING` step sizes are inconsistent: minimum (`TSMINZ`) {minimum:g} must be "
            f"positive and no larger than maximum (`TSMAXZ`) {maximum:g}."
        )
    # Eclipse takes TSINIT as a wish and keeps it inside the limits
    initial = min(max(initial, minimum), maximum)

    options: dict[str, typing.Any] = {}
    growth = float(time_step.get("tsfmax", TUNING_DEFAULTS["tsfmax"]))
    if growth != TUNING_DEFAULTS["tsfmax"] and growth > 1.0:
        options["maximum_growth_per_step"] = growth

    backoff = float(time_step.get("tsfcnv", TUNING_DEFAULTS["tsfcnv"]))
    if backoff != TUNING_DEFAULTS["tsfcnv"] and 0.0 < backoff < 1.0:
        options["backoff_factor"] = backoff
        # The harder cut after a failed step can never be milder than the ordinary one
        options["aggressive_backoff_factor"] = min(0.25, backoff)

    mass_balance = convergence.get("xxxmbe", TUNING_DEFAULTS["xxxmbe"])
    if mass_balance is not None and float(mass_balance) != TUNING_DEFAULTS["xxxmbe"]:  # noqa
        options["maximum_total_relative_mbe"] = float(mass_balance)

    return Timer(
        initial_step_size=initial,
        maximum_step_size=maximum,
        minimum_step_size=minimum,
        simulation_time=simulation_time,
        unit_system=deck_file.unit_system,
        **options,
    )

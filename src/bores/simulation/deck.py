"""Building a `RunSpec` from a deck."""

import typing

from bores.deck.file import DeckFile
from bores.deck.phases import declares_phases, read_deck_phases
from bores.simulation.spec import RunSpec
from bores.timestep.deck import get_tuning_at_start, load_timer
from bores.wells.deck import load_well_control_spec

__all__ = ["NEWTON_ITERATIONS_DEFAULT", "load_run_spec"]

NEWTON_ITERATIONS_DEFAULT = 12
"""`NEWTMX`'s default in Eclipse. A deck value equal to it is taken as not customised."""


def load_run_spec(deck_file: DeckFile) -> RunSpec:
    """
    Build the run configuration a deck describes.

    A deck only speaks to some of what a `RunSpec` holds. These come from it:

    - `unit_system`: the deck's unit keyword (`FIELD`, `METRIC` or `LAB`).
    - `start_date`: `START`, if the deck has one.
    - `phases`: the phases the deck declares (see `read_deck_phases`), or `None` if it declares
      none.
    - `timer`: the schedule length and `TUNING` (see `load_timer`), or `None` if the deck has no
      schedule length to run for.
    - `well_control`: see `load_well_control_spec`.
    - `hysteresis_enabled`: whether `SATOPTS` lists `HYSTER`.
    - `maximum_newton_iterations`: `NEWTMX` from the `TUNING` in effect at the start, only if the
      deck sets it to something other than Eclipse's default and within the range `RunSpec`
      accepts.

    Everything else (solvers, tolerances, capillary and gravity switches, ...) has no deck
    equivalent, or none that means the same thing here, and keeps its `RunSpec` default.

    :param deck_file: The deck to read.
    :returns: The run spec, in the deck's unit system.
    :raises ValidationError: If the phases are declared inconsistently or the `TUNING` step sizes
        cannot all hold.
    """
    options: dict[str, typing.Any] = {}
    start_date = deck_file.get("START")
    if start_date is not None:
        options["start_date"] = start_date

    saturation_options = deck_file.get("SATOPTS") or []
    if "HYSTER" in saturation_options:
        options["hysteresis_enabled"] = True

    tuning = get_tuning_at_start(deck_file)
    newton = ((tuning or {}).get("iteration", {})).get("newtmx")
    if newton is not None and int(newton) != NEWTON_ITERATIONS_DEFAULT and 1 <= int(newton) <= 50:
        options["maximum_newton_iterations"] = int(newton)

    return RunSpec(
        unit_system=deck_file.unit_system,
        phases=read_deck_phases(deck_file) if declares_phases(deck_file) else None,
        timer=load_timer(deck_file),
        well_control=load_well_control_spec(deck_file),
        **options,
    )

"""Building a `RunSpec` from a deck."""

import typing

from bores.deck.file import DeckFile
from bores.deck.phases import declares_phases, read_deck_phases
from bores.simulation.options import (
    NewtonOptions,
    PhysicsOptions,
    load_equilibration_options,
    load_minimum_pore_volume,
    load_model_dimensions,
    load_saturation_options,
)
from bores.simulation.spec import RunSpec
from bores.timestep.deck import TUNING_DEFAULTS, get_tuning_at_start, load_timer
from bores.wells.deck import load_well_control_spec

__all__ = ["load_run_spec"]


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
    - `saturation_options`, `equilibration_options`: `SATOPTS` (including whether hysteresis is
      on), `ENDSCALE` and `SCALECRS`, and `EQLOPTS`.
    - `dimensions`: `TABDIMS`, `EQLDIMS`, `REGDIMS`, `WELLDIMS`, `FAULTDIM`, `PIMTDIMS`, `VFPPDIMS`
      and `VFPIDIMS`.
    - `data_check_only`: whether the deck has `NOSIM`.
    - `physics_options.minimum_pore_volume`: `MINPV`, in pore volume units.
    - `newton_options.maximum_newton_iterations`: the `TUNING` item of that name in effect at the
      start, only if the deck sets it to something other than Eclipse's default and within the
      range `NewtonOptions` accepts.

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

    newton_settings: dict[str, typing.Any] = {}
    tuning = get_tuning_at_start(deck_file) or {}
    maximum_newton_iterations = tuning.get("iteration", {}).get("maximum_newton_iterations")
    if (
        maximum_newton_iterations is not None
        and int(maximum_newton_iterations) != TUNING_DEFAULTS["maximum_newton_iterations"]
        and 1 <= int(maximum_newton_iterations) <= 50
    ):
        newton_settings["maximum_newton_iterations"] = int(maximum_newton_iterations)

    physics_settings: dict[str, typing.Any] = {}
    minimum_pore_volume = load_minimum_pore_volume(deck_file)
    if minimum_pore_volume is not None:
        physics_settings["minimum_pore_volume"] = minimum_pore_volume

    return RunSpec(
        unit_system=deck_file.unit_system,
        phases=read_deck_phases(deck_file).to_phases() if declares_phases(deck_file) else None,
        timer=load_timer(deck_file),
        well_control=load_well_control_spec(deck_file),
        saturation_options=load_saturation_options(deck_file),
        equilibration_options=load_equilibration_options(deck_file),
        dimensions=load_model_dimensions(deck_file),
        data_check_only=deck_file.has("NOSIM"),
        newton_options=NewtonOptions(**newton_settings),
        physics_options=PhysicsOptions(**physics_settings),
        **options,
    )

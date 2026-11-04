"""
Which fluid phases a deck models, read once from the `RUNSPEC` keywords.

A deck declares its phases with `OIL`, `WATER`, `GAS` (or `GASWAT`) and the options that let
components move between them (`DISGAS`, `DISGASW`, `VAPOIL`, `VAPWAT`). A CO2 storage deck adds
`CO2STORE` (or `CO2STOR`). `read_deck_phases` turns those keywords into one `DeckPhases` value that
the saturation function loader, the PVT loader and the initializer all share.
"""

import typing
import warnings

import attrs

from bores.constants import c
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.types import Boolean, Phases, UnitSystem
from bores.units import get_conversion_factors

__all__ = [
    "SUPPORTED_PHASE_SETS",
    "DeckPhases",
    "check_co2_storage_standard_conditions",
    "declares_phases",
    "default_standard_conditions",
    "read_deck_phases",
]

SUPPORTED_PHASE_SETS = (
    "oil, water and gas",
    "oil and water",
    "oil and gas",
    "gas and water",
    "water only",
)
"""The phase combinations a model can be built with, as they are named in error messages."""


@attrs.define(frozen=True, kw_only=True, slots=True)
class DeckPhases(Phases):
    """
    The phases a deck models and the components that can move between them.

    The phase flags describe the model BORES builds, not the keywords as written. A CO2 storage
    deck that declares `OIL GAS CO2STOR` uses the oil slot for brine, so it is reported here as a
    gas and water model with `aqueous_slot` set to `"oil"`.
    """

    aqueous_slot: typing.Literal["oil", "water"]
    """
    The phase the deck itself uses for the aqueous fluid, `"water"` for an ordinary deck.

    It is `"oil"` for a CO2 storage deck that declares `OIL GAS CO2STOR`, where the deck's oil
    tables describe brine. Such a deck is read as a gas and water model, so `oil` is `False`.
    """

    def to_phases(self) -> Phases:
        """Returns the phases without the deck's own labelling, for a model that has no use for it."""
        return Phases(**{field.name: getattr(self, field.name) for field in attrs.fields(Phases)})


STANDARD_CONDITIONS_TOLERANCE = 1.0e-3
"""Relative difference from the default standard conditions that is still taken as the default."""


def default_standard_conditions(unit_system: UnitSystem) -> tuple[float, float]:
    """
    Returns the standard temperature and pressure a deck gets when it does not set `STCOND`.

    They are 60 F and 1 atm, expressed in the unit system of the deck.

    :param unit_system: The unit system of the deck.
    :returns: The standard temperature and the standard pressure, in the deck's units.
    """
    factors = get_conversion_factors(UnitSystem.FIELD, unit_system)
    temperature = (
        c.STANDARD_TEMPERATURE_FAHRENHEIT * factors["temperature"] + factors["temperature_offset"]
    )
    pressure = c.STANDARD_PRESSURE_PSI * factors["pressure"]
    return float(temperature), float(pressure)


def check_co2_storage_standard_conditions(deck_file: DeckFile) -> None:
    """
    Checks that a CO2 storage deck leaves its standard conditions at their defaults.

    The CO2 and brine properties are computed at the default standard conditions, so a deck that
    sets `STCOND` to anything else cannot be honoured.

    :param deck_file: The deck to check.
    :raises ValidationError: If `STCOND` sets a temperature or pressure other than the default.
    """
    record = deck_file.get("STCOND")
    if record is None:
        return

    defaults = default_standard_conditions(deck_file.unit_system)
    for name, default in zip(("temperature", "pressure"), defaults, strict=True):
        value = record[name]
        if value is not None and abs(float(value) - default) > STANDARD_CONDITIONS_TOLERANCE * abs(
            default
        ):
            raise ValidationError(
                f"A CO2 storage deck must keep the default standard {name} ({default:g}) in "
                f"`STCOND`, but the deck sets {float(value):g}."
            )


def declares_phases(deck_file: DeckFile) -> Boolean:
    """
    Checks whether a deck declares any phase keyword (`OIL`, `WATER`, `GAS` or `GASWAT`).

    A standalone include file that holds only fluid or rock tables declares none, and has no
    phase set to read.

    :param deck_file: The deck to check.
    :returns: `True` if at least one phase keyword is present.
    """
    return deck_file.has_any("OIL", "WATER", "GAS", "GASWAT")


def read_deck_phases(deck_file: DeckFile) -> DeckPhases:
    """
    Reads which phases a deck models from its `RUNSPEC` keywords.

    `GASWAT` declares gas and water together. For a CO2 storage deck, `DISGAS` and `DISGASW` both
    mean that CO2 dissolves in the brine. A deck with `OIL GAS CO2STOR` and no `WATER` is
    translated to a gas and water model, because its oil slot holds brine.

    :param deck_file: The deck to read.
    :returns: The phases the model has and which components move between them.
    :raises ValidationError: If the deck declares no phase, declares a phase combination BORES
        does not model (gas only, oil only, or CO2 storage with all three phases), or declares a
        dissolution or vaporization option that needs a phase the deck does not have.
    """
    present = deck_file.has_many(
        "OIL",
        "WATER",
        "GAS",
        "GASWAT",
        "CO2STORE",
        "CO2STOR",
        "DISGAS",
        "DISGASW",
        "VAPOIL",
        "VAPWAT",
    )
    declared_oil = present["OIL"]
    declared_water = present["WATER"] or present["GASWAT"]
    declared_gas = present["GAS"] or present["GASWAT"]
    co2_storage = present["CO2STORE"] or present["CO2STOR"]
    disgas = present["DISGAS"]
    disgasw = present["DISGASW"]
    vapoil = present["VAPOIL"]
    vapwat = present["VAPWAT"]

    if (present["CO2STORE"] or present["CO2STOR"]) and declared_gas:
        check_co2_storage_standard_conditions(deck_file)

    if not (declared_oil or declared_water or declared_gas):
        raise ValidationError(
            "The deck declares no phase. Declare the phases in `RUNSPEC` with `OIL`, `WATER`, "
            "`GAS` or `GASWAT`."
        )

    if co2_storage and declared_gas and declared_oil and declared_water:
        raise ValidationError(
            "CO2 storage decks model gas with either an oil phase (`OIL GAS CO2STOR`, where the "
            "oil slot holds brine) or a water phase (`GAS WATER CO2STORE`), not all three."
        )

    aqueous_slot: typing.Literal["oil", "water"] = "water"
    oil = declared_oil
    water = declared_water
    gas = declared_gas
    if co2_storage and declared_gas and declared_oil and not declared_water:
        aqueous_slot = "oil"
        oil = False
        water = True

    if not is_supported_phase_set(oil=oil, water=water, gas=gas):
        raise ValidationError(
            f"The deck declares {describe_phase_set(oil=oil, water=water, gas=gas)}, which is not "
            f"supported. Supported phase sets are {', '.join(SUPPORTED_PHASE_SETS)}."
        )

    if co2_storage and not gas:
        raise ValidationError("CO2 storage needs a gas phase. Declare `GAS` in `RUNSPEC`.")

    if co2_storage:
        dissolved_gas_in_water = disgas or disgasw
        dissolved_gas_in_oil = False
    else:
        dissolved_gas_in_water = disgasw
        dissolved_gas_in_oil = disgas

    if dissolved_gas_in_oil and not (oil and gas):
        raise ValidationError(
            "`DISGAS` dissolves gas in oil and needs both `OIL` and `GAS`. For gas dissolving in "
            "water, use `DISGASW`."
        )
    if dissolved_gas_in_water and not (water and gas):
        raise ValidationError("Gas dissolving in water needs both a water phase and `GAS`.")
    if vapoil and not (oil and gas):
        if aqueous_slot == "oil":
            warnings.warn(
                "`VAPOIL` is ignored for a CO2 storage deck whose oil slot holds brine. Use "
                "`VAPWAT` for water vaporizing into the gas.",
                stacklevel=2,
            )
            vapoil = False
        else:
            raise ValidationError(
                "`VAPOIL` vaporizes oil into gas and needs both `OIL` and `GAS`."
            )
    if vapwat and not (water and gas):
        raise ValidationError(
            "`VAPWAT` vaporizes water into gas and needs both `WATER` and `GAS`."
        )

    return DeckPhases(
        oil=oil,
        water=water,
        gas=gas,
        dissolved_gas_in_oil=dissolved_gas_in_oil,
        vaporized_oil=vapoil,
        dissolved_gas_in_water=dissolved_gas_in_water,
        vaporized_water=vapwat,
        co2_storage=co2_storage,
        aqueous_slot=aqueous_slot,
    )


def is_supported_phase_set(*, oil: Boolean, water: Boolean, gas: Boolean) -> Boolean:
    """Checks whether the phase combination is one the model can be built with."""
    count = int(oil) + int(water) + int(gas)
    if count >= 2:
        return True
    return water and not oil and not gas


def describe_phase_set(*, oil: Boolean, water: Boolean, gas: Boolean) -> str:
    """Names the phase combination for an error message, such as `gas only`."""
    names = [name for name, active in (("oil", oil), ("water", water), ("gas", gas)) if active]
    return f"{' and '.join(names)} only"

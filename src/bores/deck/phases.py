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

from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.types import Boolean, FluidPhase, Integer

__all__ = ["SUPPORTED_PHASE_SETS", "DeckPhases", "declares_phases", "read_deck_phases"]

SUPPORTED_PHASE_SETS = (
    "oil, water and gas",
    "oil and water",
    "oil and gas",
    "gas and water",
    "water only",
)
"""The phase combinations a model can be built with, as they are named in error messages."""


@attrs.define(frozen=True, kw_only=True, slots=True)
class DeckPhases:
    """
    The phases a deck models and the components that can move between them.

    The phase flags describe the model BORES builds, not the keywords as written. A CO2 storage
    deck that declares `OIL GAS CO2STOR` uses the oil slot for brine, so it is reported here as a
    gas and water model with `aqueous_slot` set to `"oil"`.
    """

    oil: Boolean
    """Whether the model has an oil phase."""

    water: Boolean
    """Whether the model has a water (aqueous) phase."""

    gas: Boolean
    """Whether the model has a gas phase."""

    dissolved_gas_in_oil: Boolean
    """Whether gas can dissolve in the oil phase (`DISGAS` in a model with oil)."""

    vaporized_oil: Boolean
    """Whether oil can vaporize into the gas phase (`VAPOIL`)."""

    dissolved_gas_in_water: Boolean
    """Whether gas can dissolve in the water phase (`DISGASW`, or `DISGAS` in a CO2 storage deck)."""

    vaporized_water: Boolean
    """Whether water can vaporize into the gas phase (`VAPWAT`)."""

    co2_storage: Boolean
    """Whether the deck models CO2 storage in brine (`CO2STORE` or `CO2STOR`)."""

    aqueous_slot: typing.Literal["oil", "water"]
    """
    The phase the deck itself uses for the aqueous fluid, `"water"` for an ordinary deck.

    It is `"oil"` for a CO2 storage deck that declares `OIL GAS CO2STOR`, where the deck's oil
    tables describe brine. Such a deck is read as a gas and water model, so `oil` is `False`.
    """

    @property
    def phases(self) -> tuple[FluidPhase, ...]:
        """The active phases in the order oil, water, gas."""
        active: list[FluidPhase] = []
        if self.oil:
            active.append(FluidPhase.OIL)
        if self.water:
            active.append(FluidPhase.WATER)
        if self.gas:
            active.append(FluidPhase.GAS)
        return tuple(active)

    @property
    def phase_count(self) -> Integer:
        """How many phases the model has."""
        return int(self.oil) + int(self.water) + int(self.gas)

    @property
    def is_three_phase(self) -> Boolean:
        """Whether the model has oil, water and gas."""
        return self.oil and self.water and self.gas

    @property
    def is_gas_water(self) -> Boolean:
        """Whether the model has gas and water and no oil."""
        return self.gas and self.water and not self.oil

    @property
    def is_single_phase(self) -> Boolean:
        """Whether the model has only one phase."""
        return self.phase_count == 1


def declares_phases(deck_file: DeckFile) -> Boolean:
    """
    Whether a deck declares any phase keyword (`OIL`, `WATER`, `GAS` or `GASWAT`).

    A standalone include file that holds only fluid or rock tables declares none, and has no
    phase set to read.

    :param deck_file: The deck to check.
    :returns: `True` if at least one phase keyword is present.
    """
    return deck_file.has_any("OIL", "WATER", "GAS", "GASWAT")


def read_deck_phases(deck_file: DeckFile) -> DeckPhases:
    """
    Read which phases a deck models from its `RUNSPEC` keywords.

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
    """Whether the phase combination is one the model can be built with."""
    count = int(oil) + int(water) + int(gas)
    if count >= 2:
        return True
    return water and not oil and not gas


def describe_phase_set(*, oil: Boolean, water: Boolean, gas: Boolean) -> str:
    """Name the phase combination for an error message, such as `gas only`."""
    names = [name for name, active in (("oil", oil), ("water", water), ("gas", gas)) if active]
    return f"{' and '.join(names)} only"

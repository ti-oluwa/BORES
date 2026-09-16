"""The wells sub-system bundle."""

import typing

import attrs
from typing_extensions import Self

from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.serde.stores.base import StoreSerializable
from bores.types import UnitConversionTable, UnitSystem
from bores.wells.base import Wells
from bores.wells.controls import WellControls
from bores.wells.groups import GroupControls, WellGroups
from bores.wells.hydraulics.base import WellBoreModel
from bores.wells.resolution.spec import WellControlSpec

__all__ = ["WellSystem"]


@attrs.frozen(kw_only=True, slots=True)
class WellSystem(StoreSerializable):
    """A complete wells sub-system: wells, controls, hydraulics, and groups."""

    wells: Wells
    """Every well in the system."""

    well_controls: WellControls
    """Current control target for each well."""

    default_wellbore: WellBoreModel
    """Hydraulics model used by any well without an entry in `wellbore_overrides`."""

    wellbore_overrides: typing.Mapping[str, WellBoreModel] = attrs.field(factory=dict)
    """Per-well hydraulics override - e.g. a gas well on Beggs & Brill while
    every oil well uses the homogeneous no-slip model."""

    groups: WellGroups | None = None
    """The well-group hierarchy, if the deck defined one."""

    group_controls: GroupControls | None = None
    """Current control target for each group, if the deck defined any."""

    control_spec: WellControlSpec = attrs.field(factory=WellControlSpec)
    """Configuration for how well controls are resolved during simulation."""

    def __attrs_post_init__(self) -> None:
        if self.well_controls.unit_system != self.wells.unit_system:
            raise ValidationError(
                f"`well_controls.unit_system` ({self.well_controls.unit_system.value}) != "
                f"`wells.unit_system` ({self.wells.unit_system.value})."
            )
        if (
            self.group_controls is not None
            and self.group_controls.unit_system != self.wells.unit_system
        ):
            raise ValidationError(
                f"`group_controls.unit_system` ({self.group_controls.unit_system.value}) "
                f"!= `wells.unit_system` ({self.wells.unit_system.value})."
            )

    @property
    def unit_system(self) -> UnitSystem:
        """Unit system shared by `wells`/`well_controls`/`group_controls`."""
        return self.wells.unit_system

    def get_wellbore_model(self, well_name: str) -> WellBoreModel:
        """
        Gets the hydraulics model to use for a well.

        :param well_name: Name of the well.
        :returns: `wellbore_overrides[well_name]` if present, else `default_wellbore`.
        """
        return self.wellbore_overrides.get(well_name, self.default_wellbore)

    def get_wells_in_group(self, group_name: str) -> tuple[str, ...]:
        """
        Gets every well belonging to a group or any group under it.

        :param group_name: Name of the group.
        :returns: Names of every well whose group is `group_name` or a
            descendant of it, per `groups`.
        :raises ValidationError: If `groups` is `None`.
        """
        if self.groups is None:
            raise ValidationError(
                "`get_wells_in_group` requires `groups` to be set on this WellSystem."
            )
        member_group_names = {group_name, *self.groups.descendants(group_name)}
        return tuple(name for name in self.wells if self.wells[name].group in member_group_names)

    @classmethod
    def from_deck(
        cls, deck_file: DeckFile, *, grid: Grid, default_wellbore: WellBoreModel
    ) -> Self:
        """
        Builds a `WellSystem` from a parsed deck.

        :param deck_file: Parsed deck.
        :param grid: `Grid` built from the same deck.
        :param default_wellbore: Hydraulics model for every well - build
            one via `wells.hydraulics.homogeneous.homogeneous_model()`,
            `wells.hydraulics.beggs_and_brill.beggs_and_brill_model()`, or
            `wells.hydraulics.hagedorn_brown.hagedorn_brown_model()`.
        :returns: `WellSystem` built from every well/control/group keyword
            in the deck. `groups`/`group_controls` are `None` if the deck
            has no `GRUPTREE`/`GCONPROD`/`GCONINJE`.
        """
        groups = WellGroups.from_deck(deck_file) if deck_file.has("GRUPTREE") else None
        group_controls = (
            GroupControls.from_deck(deck_file)
            if (deck_file.has("GCONPROD") or deck_file.has("GCONINJE"))
            else None
        )
        return cls(
            wells=Wells.from_deck(deck_file, grid=grid),
            well_controls=WellControls.from_deck(deck_file),
            default_wellbore=default_wellbore,
            groups=groups,
            group_controls=group_controls,
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Converts this system to a different unit system.

        `groups` (a pure hierarchy, with no dimensioned data) is unchanged.

        :param target: Target unit system.
        :param table: Optional custom unit-conversion table.
        :returns: This system, with `wells`/`well_controls`/`default_wellbore`/
            `wellbore_overrides`/`group_controls` converted to `target`.
        """
        if target == self.unit_system:
            return self
        return attrs.evolve(
            self,
            wells=self.wells.convert(target, table=table),
            well_controls=self.well_controls.convert(target, table=table),
            default_wellbore=self.default_wellbore.convert(target, table=table),
            wellbore_overrides={
                well_name: wellbore.convert(target, table=table)
                for well_name, wellbore in self.wellbore_overrides.items()
            },
            group_controls=(
                self.group_controls.convert(target, table=table)
                if self.group_controls is not None
                else None
            ),
        )

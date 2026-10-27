"""The wells sub-system bundle."""

import typing

import attrs
from typing_extensions import Self

from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.types import UnitConversionTable, UnitSystem
from bores.wells.base import Wells
from bores.wells.controls import WellControls
from bores.wells.groups import GroupControls, WellGroups
from bores.wells.hydraulics.base import WellBoreModel
from bores.wells.hydraulics.vfp import VFPTables
from bores.wells.resolution.spec import WellControlSpec

__all__ = ["TubingPerformance", "WellSystem"]

TubingPerformance: typing.TypeAlias = WellBoreModel | VFPTables
"""What relates a well's bottom-hole pressure to its tubing head pressure: a correlation
(`WellBoreModel`) or the deck's vertical flow performance tables (`VFPTables`)."""


@attrs.frozen(kw_only=True, slots=True)
class WellSystem:
    """A complete wells sub-system: wells, controls, hydraulics, and groups."""

    wells: Wells
    """Every well in the system."""

    well_controls: WellControls
    """Current control target for each well."""

    tubing_performance: (
        TubingPerformance | typing.Mapping[str, TubingPerformance | None] | None
    ) = None
    """
    What relates each well's bottom-hole pressure to its tubing head pressure.

    - A single `WellBoreModel` or `VFPTables` applies to every well. A `VFPTables` is a
      collection indexed by table number, and each well picks its table through the `vfp_table`
      of its control.
    - A mapping assigns per well. Its keys are well names, and the key `"*"` is the fallback for
      wells not named. A well mapped to `None` has none, even when `"*"` is set.
    - `None` means no well in this system needs one. Every well is then subject to
      `spec.connection_pressure_mode` for connection-pressure distribution, and can't
      use THP control, a `THPLimit`, or THP reporting.
    """

    groups: WellGroups | None = None
    """The well-group hierarchy, if the deck defined one."""

    group_controls: GroupControls | None = None
    """Current control target for each group, if the deck defined any."""

    spec: WellControlSpec = attrs.field(factory=WellControlSpec)
    """Configuration for how well controls are resolved during simulation."""

    def __attrs_post_init__(self) -> None:
        if isinstance(self.tubing_performance, typing.Mapping):
            unknown = set(self.tubing_performance) - set(self.wells) - {"*"}
            if unknown:
                raise ValidationError(
                    f"`tubing_performance` names wells that are not in this system: "
                    f"{sorted(unknown)}."
                )
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

    def get_tubing_performance(self, well_name: str) -> TubingPerformance | None:
        """
        Gets what relates a well's bottom-hole pressure to its tubing head pressure.

        :param well_name: Name of the well.
        :returns: The well's own entry if `tubing_performance` is a mapping that names it
            (`None` counts as named), else the mapping's `"*"` entry, else, if it is a single
            item, that item. `None` if the well has none.
        """
        if isinstance(self.tubing_performance, typing.Mapping):
            if well_name in self.tubing_performance:
                return self.tubing_performance[well_name]
            return self.tubing_performance.get("*")
        return self.tubing_performance

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
        cls,
        deck_file: DeckFile,
        *,
        grid: Grid,
        tubing_performance: (
            TubingPerformance | typing.Mapping[str, TubingPerformance | None] | None
        ) = None,
    ) -> Self:
        """
        Builds a `WellSystem` from a parsed deck.

        :param deck_file: Parsed deck.
        :param grid: `Grid` built from the same deck.
        :param tubing_performance: What relates each well's BHP to its THP (see
            `WellSystem.tubing_performance`). If omitted and the deck defines `VFPPROD` or
            `VFPINJ` tables, those are used. Omit it entirely if nothing in this deck needs one,
            e.g, no well under THP control, no `THPLimit`, and
            `spec.connection_pressure_mode` set to cover connection distribution
            without one.
        :returns: `WellSystem` built from every well/control/group keyword
            in the deck. `groups`/`group_controls` are `None` if the deck
            has no `GRUPTREE`/`GCONPROD`/`GCONINJE`.
        """
        if tubing_performance is None and (deck_file.has("VFPPROD") or deck_file.has("VFPINJ")):
            tubing_performance = VFPTables.from_deck(deck_file)
        groups = WellGroups.from_deck(deck_file) if deck_file.has("GRUPTREE") else None
        group_controls = (
            GroupControls.from_deck(deck_file)
            if (deck_file.has("GCONPROD") or deck_file.has("GCONINJE"))
            else None
        )
        return cls(
            wells=Wells.from_deck(deck_file, grid=grid),
            well_controls=WellControls.from_deck(deck_file),
            tubing_performance=tubing_performance,
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
        :returns: This system, with `wells`/`well_controls`/`tubing_performance`/
            `group_controls` converted to `target`. A `None` entry stays `None`.
        """
        if target == self.unit_system:
            return self
        return attrs.evolve(
            self,
            wells=self.wells.convert(target, table=table),
            well_controls=self.well_controls.convert(target, table=table),
            tubing_performance=convert_tubing_performance(
                self.tubing_performance, target, table=table
            ),
            group_controls=(
                self.group_controls.convert(target, table=table)
                if self.group_controls is not None
                else None
            ),
        )


def convert_tubing_performance(
    tubing_performance: TubingPerformance | typing.Mapping[str, TubingPerformance | None] | None,
    target: UnitSystem,
    /,
    *,
    table: UnitConversionTable | None = None,
) -> TubingPerformance | typing.Mapping[str, TubingPerformance | None] | None:
    """
    Converts a system's tubing performance to a different unit system.

    :param tubing_performance: A single item, a per-well mapping of them, or `None`.
    :param target: Target unit system.
    :param table: Optional custom unit-conversion table.
    :returns: The same shape, with every item converted to `target`. `None` stays `None`.
    """
    if tubing_performance is None:
        return None
    if isinstance(tubing_performance, typing.Mapping):
        return {
            name: convert_tubing_performance(item, target, table=table)  # type: ignore[misc]
            for name, item in tubing_performance.items()
        }
    return tubing_performance.convert(target, table=table)

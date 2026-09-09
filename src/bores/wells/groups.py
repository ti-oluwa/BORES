"""Well-group hierarchy and group-level control targets."""

import enum
import typing

import attrs
from typing_extensions import Self

from bores.constants import get_conversion_factors
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.serde.stores import StoreSerializable
from bores.types import FluidPhase, Number, UnitConversionTable, UnitSystem

__all__ = [
    "FIELD_GROUP",
    "GroupControl",
    "GroupControls",
    "GroupInjectorControlMode",
    "GroupProducerControlMode",
    "WellGroup",
    "WellGroups",
]

FIELD_GROUP = "FIELD"
"""The implicit root group name every top-level group's `parent` resolves to."""


@attrs.frozen(kw_only=True, slots=True, hash=True)
class WellGroup(Serializable):
    """One node in the group hierarchy - deck `GRUPTREE` child/parent link."""

    name: str
    """Group name"""

    parent: str | None = None
    """
    `None` only for the implicit root (`name == "FIELD"`); every other
    group must have a parent (possibly `"FIELD"` itself).
    """

    def __attrs_post_init__(self) -> None:
        if not self.name:
            raise ValidationError("`name` must be a non-empty string.")
        if self.parent is None and self.name != FIELD_GROUP:
            raise ValidationError(
                f"`WellGroup` {self.name!r} must have a `parent` (only the "
                f"{FIELD_GROUP!r} root may omit one)."
            )


WellGroupTree = dict[str | WellGroup, "WellGroupTree"]


class WellGroups(
    StoreSerializable,
    fields={"groups": typing.Mapping[str, WellGroup]},
):
    """Name-keyed group hierarchy. Lookup and traversal only."""

    def __init__(self, groups: typing.Mapping[str, WellGroup]) -> None:
        groups = dict(groups)
        groups.setdefault(FIELD_GROUP, WellGroup(name=FIELD_GROUP, parent=None))
        for name, group in groups.items():
            if group.parent is not None and group.parent not in groups:
                raise ValidationError(
                    f"WellGroup {name!r}'s parent {group.parent!r} is not itself a declared group."
                )
        self.groups = groups

    def group(self, name: str) -> WellGroup:
        """
        Returns group with the given name.

        :param name: The group name
        :raises KeyError: If no group with that name exists.
        """
        group = self.groups.get(name)
        if group is None:
            raise KeyError(f"No group named {name!r}.")
        return group

    def children(self, name: str) -> tuple[str, ...]:
        """Direct child group names of `name` only (not grandchildren)."""
        if name not in self.groups:
            raise KeyError(f"No group named {name!r}.")
        return tuple(name for name, group in self.groups.items() if group.parent == name)

    def descendants(self, name: str) -> tuple[str, ...]:
        """Every group under `name`, at any depth (not including `name` itself)."""
        result: list[str] = []
        frontier = list(self.children(name))
        while frontier:
            current = frontier.pop()
            result.append(current)
            frontier.extend(self.children(current))
        return tuple(result)

    def tree(self, root: str = FIELD_GROUP, *, objects: bool = False) -> WellGroupTree:
        """
        Return the group hierarchy rooted at `root` as a nested dictionary.

        The returned structure is keyed either by group names (default) or by
        `WellGroup` objects when `objects=True`, with each value being a dictionary
        containing that group's descendants recursively.

        Example (`objects=False`):

        ```python
        {
            "FIELD": {
                "PROD": {
                    "NORTH": {},
                    "SOUTH": {},
                },
                "INJ": {},
            },
        }
        ```

        Example (`objects=True`):

        ```python
        {
            WellGroup(name="FIELD", parent=None): {
                WellGroup(name="PROD", parent="FIELD"): {},
                WellGroup(name="INJ", parent="FIELD"): {},
            },
        }
        ```

        :param root: Root group of the returned hierarchy.
        :param objects: Whether dictionary keys should be `WellGroup` objects instead
            of group names.
        :returns: Nested dictionary representing the group hierarchy.
        :raises KeyError: If `root` is not a known group.
        """
        self.group(root)  # Validate.

        def build(name: str) -> WellGroupTree:
            return {
                (self.group(child) if objects else child): build(child)
                for child in sorted(self.children(name))
            }

        root_key: str | WellGroup
        root_key = self.group(root) if objects else root
        return {root_key: build(root)}

    def format_tree(self, root: str = FIELD_GROUP, *, ascii: bool = False) -> str:
        """
        Return a human-readable rendering of the group hierarchy.

        Example (`ascii=False`):

        ```text
        FIELD
        ├── PROD
        │   ├── NORTH
        │   └── SOUTH
        └── INJ
            └── WATER
        ```

        Example (`ascii=True`):

        ```text
        FIELD
        |-- PROD
        |   |-- NORTH
        |   `-- SOUTH
        `-- INJ
            `-- WATER
        ```

        :param root: Root group of the rendered hierarchy.
        :param ascii: Use an ASCII-only tree instead of Unicode box-drawing
            characters.
        :returns: Formatted tree.
        :raises KeyError: If `root` is not a known group.
        """
        self.group(root)  # Validate.
        if ascii:
            tee = "|-- "
            elbow = "`-- "
            pipe = "|   "
            blank = "    "
        else:
            tee = "├── "
            elbow = "└── "
            pipe = "│   "
            blank = "    "

        lines: list[str] = [root]

        def visit(name: str, prefix: str) -> None:
            children = sorted(self.children(name))
            for index, child in enumerate(children):
                last = index == len(children) - 1
                lines.append(prefix + (elbow if last else tee) + child)
                visit(child, prefix + (blank if last else pipe))

        visit(root, "")
        return "\n".join(lines)

    @classmethod
    def from_deck(cls, deck_file: DeckFile) -> Self:
        """
        Load `WellGroups` from a parsed `DeckFile` if any exists.

        :param deck_file: Parsed deck.
        :returns: `WellGroups` from `GRUPTREE`, or None if absent.
        """
        from bores.wells.deck import load_groups

        return typing.cast(Self, load_groups(deck_file))

    def __format__(self, spec: str) -> str:
        """
        Format the group hierarchy.

        Supported format specifiers:

        - ``""``      : default string representation.
        - ``"tree"``  : Unicode tree.
        - ``"ascii"`` : ASCII-only tree.

        :param spec: Format specifier.
        :returns: Formatted string.
        :raises ValueError: If the format specifier is unknown.
        """
        if spec == "":
            return str(self)

        if spec == "tree":
            return self.format_tree()

        if spec == "ascii":
            return self.format_tree(ascii=True)

        raise ValueError(f"Unknown `WellGroups` format specifier {spec!r}.")

    def __getitem__(self, name: str) -> WellGroup:
        return self.group(name)

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self.groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __contains__(self, name: object) -> bool:
        return name in self.groups


class GroupProducerControlMode(enum.Enum):
    """Deck `GCONPROD` item 2 - which target field is active for a production group."""

    OIL_RATE = "oil_rate"
    WATER_RATE = "water_rate"
    GAS_RATE = "gas_rate"
    LIQUID_RATE = "liquid_rate"
    RESERVOIR_VOLUME_RATE = "reservoir_volume_rate"
    FIELD = "field"
    """Controlled by its parent group's allocation instead of its own target."""
    NONE = "none"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        lowered = str(value).lower()
        for member in cls:
            if member.value == lowered:
                return member
        return None


class GroupInjectorControlMode(enum.Enum):
    """Deck `GCONINJE` item 3 - which target field is active for an injection group."""

    RATE = "rate"
    RESERVOIR_VOLUME_RATE = "reservoir_volume_rate"
    VOIDAGE_REPLACEMENT = "voidage_replacement"
    REINJECTION = "reinjection"
    FIELD = "field"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        lowered = str(value).lower()
        for member in cls:
            if member.value == lowered:
                return member
        return None


@attrs.frozen(kw_only=True, slots=True)
class GroupControl(Serializable):
    """
    A group's production or injection target.

    Same "all targets may be populated, `mode` selects the active one"
    shape as `ProducerControl`/`InjectorControl` - deck `GCONPROD`/
    `GCONINJE` supplies several target items regardless of which one is
    active.
    """

    mode: GroupProducerControlMode | GroupInjectorControlMode
    target_rate: Number | None = None
    injected_phase: FluidPhase | None = None
    """Set only for an injection group (`mode` is a `GroupInjectorControlMode`)."""
    unit_system: UnitSystem = UnitSystem.FIELD

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Returns a  new `GroupControl` in the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `GroupControl` with `target_rate` converted to target.
        """
        if target == self.unit_system or self.target_rate is None:
            return attrs.evolve(self, unit_system=target)

        factors = get_conversion_factors(self.unit_system, target, table=table)
        if self.injected_phase is not None:
            factor = (
                factors["reservoir_rate"]
                if self.mode is GroupInjectorControlMode.RESERVOIR_VOLUME_RATE
                else factors["gas_surface_rate"]
                if self.injected_phase is FluidPhase.GAS
                else factors["liquid_surface_rate"]
            )
        else:
            factor = (
                factors["reservoir_rate"]
                if self.mode is GroupProducerControlMode.RESERVOIR_VOLUME_RATE
                else factors["gas_surface_rate"]
                if self.mode is GroupProducerControlMode.GAS_RATE
                else factors["liquid_surface_rate"]
            )
        return attrs.evolve(self, target_rate=self.target_rate * factor, unit_system=target)


class GroupControls(
    StoreSerializable,
    fields={
        "controls": typing.Mapping[str, GroupControl],
        "unit_system": typing.Optional[UnitSystem],  # noqa: UP045
    },
):
    """
    Name-keyed, mutable mapping from group name to its current `GroupControl`.
    """

    __slots__ = ("controls", "unit_system")

    def __init__(
        self,
        controls: typing.Mapping[str, GroupControl],
        unit_system: UnitSystem | None = None,
    ) -> None:
        """
        Initialize the `GroupControls` object.

        :param controls: Mapping from group name to GroupControl.
        :param unit_system: Target unit system for every control. None
            requires all controls to already share the same unit system.
        :raises ValidationError: If unit_system is None and the controls
            don't all share one unit system.
        """
        if unit_system is None:
            systems = {control.unit_system for control in controls.values()}
            if len(systems) > 1:
                raise ValidationError(
                    "All controls must share the same unit system when "
                    "`unit_system` is not explicitly provided. Found: "
                    f"{sorted(s.value for s in systems)}."
                )
            unit_system = systems.pop() if systems else UnitSystem.FIELD
        else:
            controls = {
                name: control
                if control.unit_system == unit_system
                else control.convert(unit_system)
                for name, control in controls.items()
            }

        self.controls = dict(controls)
        self.unit_system = unit_system

    def get(self, name: str) -> GroupControl | None:
        return self.controls.get(name)

    def set(self, name: str, control: GroupControl) -> None:
        self.controls[name] = control

    @classmethod
    def from_deck(cls, deck_file: DeckFile) -> Self:
        """
        Load `GroupControls` from a parsed `DeckFile` if any exists.

        :param deck_file: Parsed deck.
        :returns: `GroupControls` from `GCONPROD`/`GCONINJE`, or None if absent.
        """
        from bores.wells.deck import load_group_controls

        return typing.cast(Self, load_group_controls(deck_file))

    def __getitem__(self, name: str) -> GroupControl:
        control = self.get(name)
        if control is None:
            raise KeyError(f"No control set for group {name!r}.")
        return control

    def __setitem__(self, name: str, control: GroupControl) -> None:
        self.set(name, control)

    def __delitem__(self, name: str) -> None:
        if name not in self.controls:
            raise KeyError(f"No control set for group {name!r}.")
        del self.controls[name]

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self.controls)

    def __len__(self) -> int:
        return len(self.controls)

    def __contains__(self, name: object) -> bool:
        return name in self.controls

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Returns a  new `GroupControls` object in the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `GroupControls` with every control converted to target.
        """
        if target == self.unit_system:
            return self
        return self.__class__(
            controls={
                name: control.convert(target, table=table)
                for name, control in self.controls.items()
            },
            unit_system=target,
        )

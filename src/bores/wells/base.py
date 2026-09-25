"""Static well identity and completion data."""

import enum
import typing
from collections.abc import Mapping

import attrs
from typing_extensions import Self

from bores.constants import get_conversion_factors
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.serde.base import Serializable
from bores.serde.stores import StoreSerializable
from bores.types import (
    FluidPhase,
    Integer,
    Number,
    Orientation,
    UnitConversionTable,
    UnitSystem,
)
from bores.utils import scale
from bores.wells.trajectory import WellTrajectory

__all__ = [
    "CompletionStatus",
    "Perforation",
    "Well",
    "WellStatus",
    "WellType",
    "Wells",
]


class WellType(enum.Enum):
    """Producer/injector identity."""

    PRODUCER = "producer"
    INJECTOR = "injector"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        lowered = str(value).lower()
        for member in cls:
            if member.value == lowered:
                return member
        return None


class CompletionStatus(enum.Enum):
    """
    Static intent for a perforation (deck `COMPDAT` item 6 `OPEN`/`SHUT`).

    This is not the same as a well being shut in by a control action at
    runtime, that's `WellState.is_open`. `CompletionStatus.SHUT`
    means **"this completion was never meant to flow"** (e.g. a deck author
    disabling one layer of a multi-layer completion); `WellState.is_open =
    False` means **"the whole well is currently shut for operational reasons"**.

    A perforation with `CompletionStatus.SHUT` is excluded from
    the perforation indices resolution output entirely; one with `OPEN` is still
    subject to the well-level open/shut flag in `WellState` at simulation time.
    """

    OPEN = "open"
    SHUT = "shut"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        lowered = str(value).lower()
        for member in cls:
            if member.value == lowered:
                return member
        return None


class WellStatus(enum.Enum):
    """
    Schedule-activation status for a well or perforation under the
    load-once compiled architecture.

    Orthogonal to both `CompletionStatus` (a perforation's static,
    deck-authored intent to ever flow) and `WellState.is_open` (a well's
    runtime operational open/shut state, driven by control/limit resolution
    during simulation). `WellStatus` answers a third, independent question:
    has the schedule clock reached the point in the deck where this well or
    perforation's defining record (`WELSPECS`/`COMPDAT`) takes effect yet?

    Exists for the "load the full roster once" architecture: every well and
    perforation that will ever appear across the whole schedule is loaded
    up front into the compiled arrays as `PENDING`, then flipped to
    `ACTIVE` in place as the schedule clock passes each entity's
    introduction point, rather than reallocating compiled structures
    mid-run. Every solver kernel skips `PENDING` rows entirely, the same
    way it already skips `CompletionStatus.SHUT` perforations.
    """

    PENDING = "pending"
    """Declared somewhere in the deck but not yet reached by the schedule clock."""

    ACTIVE = "active"
    """Reached by the schedule clock; subject to normal processing."""

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
class Perforation(Serializable):
    """
    A single completion interval on a well, defined by true vertical
    depth, measured depth, or both.

    Give `top_depth`/`bottom_depth` for a well with no `trajectory.
    Give `top_md`/`bottom_md` for a well with one,
    since true vertical depth is not invertible along a horizontal or
    S-shaped section (multiple measured depths can share the same
    TVD), so measured depth is the only interval representation that
    identifies a unique location on an arbitrary path there; that
    well's own TVD at this interval is then derived from
    `top_md`/`bottom_md` through the trajectory's own interpolation,
    not given directly. Both may be given together, on a trajectory
    well, if the TVD is already known from some other source.

    Two `Perforation` instances with identical fields are
    interchangeable; nothing about a `Perforation` depends on which
    well it belongs to, beyond `top_md`/`bottom_md` needing that well's
    own `trajectory` to mean anything.
    """

    top_depth: Number | None = None
    """
    Positive-down depth, same convention as `Grid.vertex_coordinates`
    z-axis and `Grid.cell_center_depths`. Required on a well with no
    `trajectory`; derived from `top_md` there instead, on one that has one.
    """

    bottom_depth: Number | None = None
    """Equals `top_depth` for a point perforation. Validated `>= top_depth`, when given."""

    top_md: Number | None = None
    """
    Measured depth. Must fall within the owning `Well`'s trajectory
    range. Only valid on a well with a `trajectory`.
    """

    bottom_md: Number | None = None
    """Equals `top_md` for a point perforation. Validated `>= top_md`, when given."""

    skin: Number = 0.0
    """
    Dimensionless skin factor. Deck `COMPDAT` has no direct skin item in
    the base keyword, hence this is carried for use by `WPIMULT`/manual skin
    workflows; harmless default.
    """

    wellbore_radius: Number = 0.25
    """Perforation radius."""

    status: CompletionStatus = CompletionStatus.OPEN
    """See `CompletionStatus`."""

    schedule_status: WellStatus = WellStatus.ACTIVE
    """
    See `WellStatus`. Independent of `status`: a perforation added by a
    `COMPDAT` record later in the schedule can be `WellStatus.PENDING`
    while every already-active sibling perforation on the same well is
    `ACTIVE`, and a `WellStatus.PENDING` perforation can still carry
    `CompletionStatus.SHUT` for when it does activate.
    """

    saturation_region: int | None = None

    connection_factor_override: Number | None = None
    """
    Deck `COMPDAT` item 8 (`CF`). When present, wells indices computation uses this
    directly instead of computing a Peaceman/equivalent-radius well index.
    """

    connection_factor_multiplier: Number | None = None
    """
    Deck `WPIMULT`. Scales the computed well index rather than replacing
    it. Applied after `connection_factor_override`, if that's also set,
    though the two would not normally both be present on one perforation.
    """

    direction: Orientation | None = None
    """
    `bores.typing.Orientation` (`X`/`Y`/`Z`/`UNSET`). `None` means
    `wells.indices` resolves a direction. Only meaningful when
    `top_md`/`bottom_md` aren't given: Peaceman's formula assumes a
    wellbore aligned with a principal permeability axis, which an
    arbitrary trajectory azimuth generally isn't, so a completion with
    `top_md`/`bottom_md` set always resolves through the isotropic
    equivalent-radius well index instead, never Peaceman's formula,
    regardless of `direction`.
    """

    partial_penetration_fraction: Number | None = None
    """
    **Not to be set by the user.** 

    Populated during perforation indices computation (overlap-length /
    cell-thickness ratio). `None` on a freshly constructed `Perforation` is
    the correct/expected state. Validated: if set, must be in `(0, 1]`.
    """

    cell_index: Integer | None = None
    """
    **Not to be set by the user.**

    The grid cell this completion connects to, when that is already
    exact and known upfront (a single-layer structured-grid connection,
    such as one `COMPDAT` record with `k1 == k2`). `wells.indices`
    resolves straight to this cell instead of re-deriving it from
    geometry, which a horizontal or multi-segment well's completions
    otherwise cannot do reliably: several completions on the same well
    often share a true vertical depth, and measured depth on its own
    carries no lateral position to search with. `None` means the cell
    is derived geometrically instead, the same as before this field
    existed.
    """

    def __attrs_post_init__(self) -> None:
        has_tvd = self.top_depth is not None or self.bottom_depth is not None
        has_md = self.top_md is not None or self.bottom_md is not None
        if not has_tvd and not has_md:
            raise ValidationError(
                "Give `top_depth`/`bottom_depth`, `top_md`/`bottom_md`, or both."
            )
        if (self.top_depth is None) != (self.bottom_depth is None):
            raise ValidationError("`top_depth` and `bottom_depth` must be given together.")
        if self.top_depth is not None and self.bottom_depth < self.top_depth:  # type: ignore[operator]
            raise ValidationError(
                f"`bottom_depth` ({self.bottom_depth}) must be >= `top_depth` ({self.top_depth})."
            )
        if (self.top_md is None) != (self.bottom_md is None):
            raise ValidationError("`top_md` and `bottom_md` must be given together.")
        if self.top_md is not None and self.bottom_md < self.top_md:  # type: ignore[operator]
            raise ValidationError(
                f"`bottom_md` ({self.bottom_md}) must be >= `top_md` ({self.top_md})."
            )
        if self.direction is not None and has_md:
            raise ValidationError(
                "`direction` only applies to a TVD-based completion; it has no effect once "
                "`top_md`/`bottom_md` are given, since that always resolves through the "
                "isotropic equivalent-radius well index instead of Peaceman's formula."
            )
        if self.wellbore_radius <= 0:
            raise ValidationError("`wellbore_radius` must be positive.")
        if self.connection_factor_override is not None and (self.connection_factor_override <= 0):
            raise ValidationError("`connection_factor_override` must be positive.")
        if self.partial_penetration_fraction is not None and not (
            0 < self.partial_penetration_fraction <= 1
        ):
            raise ValidationError(
                "`partial_penetration_fraction` must be in (0, 1]; got "
                f"{self.partial_penetration_fraction}."
            )
        if self.cell_index is not None and self.cell_index < 0:
            raise ValidationError(f"`cell_index` must be >= 0; got {self.cell_index}.")

    @property
    def is_point_perforation(self) -> bool:
        """`True` if the given interval (measured depth, if given, else true
        vertical depth) has zero length."""
        if self.top_md is not None:
            return self.top_md == self.bottom_md
        return self.top_depth == self.bottom_depth

    @property
    def is_active(self) -> bool:
        """`True` if `schedule_status is WellStatus.ACTIVE`."""
        return self.schedule_status is WellStatus.ACTIVE

    @property
    def length(self) -> Number:
        """
        `bottom_md - top_md` if measured depth is given, else `bottom_depth
        - top_depth`. Zero for a point perforation. Measured-depth length,
        when given, is not a true-vertical-depth length; along a
        horizontal section these differ substantially.
        """
        if self.top_md is not None:
            return self.bottom_md - self.top_md  # type: ignore[operator]
        return self.bottom_depth - self.top_depth  # type: ignore[operator]


@attrs.frozen(kw_only=True, slots=True)
class Well(Serializable):
    """Static well identity and configuration."""

    name: str
    """Unique identifier, deck `WELSPECS` item 1."""

    well_type: WellType

    surface_location: tuple[Number, Number] = attrs.field(converter=tuple)  # type: ignore
    """
    `(x, y)` in `Grid` coordinate units - the wellhead location,
    regardless of whether the well is vertical or has a `trajectory`.
    """

    reference_depth: Number
    """BHP/THP reporting datum, deck `WELSPECS` item 5."""

    perforations: tuple[Perforation, ...] = attrs.field(converter=tuple)
    """
    Each with `top_depth`/`bottom_depth` set if `trajectory` is `None`,
    or `top_md`/`bottom_md` set if it isn't (both may be set either way).

    Must not be empty. 
    """

    trajectory: WellTrajectory | None = None
    """
    Deviation survey. 

    **`None`** (default): a vertical well at `surface_location`, and every
    entry in `perforations` must have `top_depth`/`bottom_depth` set. 
    
    **Set**: a deviated/horizontal well, and every entry in `perforations`
    must have `top_md`/`bottom_md` set instead.
    """

    preferred_phase: FluidPhase | None = None
    """
    Deck `WELSPECS` item 6. `None` allowed for manual construction where
    it's not yet decided.
    """

    group: str | None = None
    """
    Deck `WELSPECS` item 2. `None` if ungrouped.
    """

    pvt_region: int | None = None

    unit_system: UnitSystem = UnitSystem.FIELD

    schedule_status: WellStatus = WellStatus.ACTIVE
    """
    See `WellStatus`. Under the load-once roster architecture, `Wells.from_deck`
    constructs every well the schedule will ever introduce up front; a well
    whose `WELSPECS` hasn't been reached yet by the schedule clock is
    `WellStatus.PENDING`, and is flipped to `ACTIVE` in place once it is.
    A `Well` built directly (not via schedule loading) defaults to `ACTIVE`,
    matching current/pre-schedule-aware behavior.
    """

    metadata: typing.Mapping[str, typing.Any] | None = None
    """Free-form, mirrors `Grid.metadata`."""

    d_factor: Number | None = None
    """Non-Darcy flow coefficient, deck `WDFAC`. `None` if not set."""

    def __attrs_post_init__(self) -> None:
        if not self.name:
            raise ValidationError("`name` must be a non-empty string.")
        if not self.perforations:
            raise ValidationError(f"Well {self.name!r} must have at least one perforation.")

        if self.trajectory is not None:
            if not all(perforation.top_md is not None for perforation in self.perforations):
                raise ValidationError(
                    f"Well {self.name!r} has a `trajectory`; every entry in `perforations` "
                    "must have `top_md`/`bottom_md` set."
                )

            for perforation in self.perforations:
                assert perforation.top_md is not None and perforation.bottom_md is not None
                if not (
                    self.trajectory.top_measured_depth
                    <= perforation.top_md
                    <= perforation.bottom_md
                    <= self.trajectory.bottom_measured_depth
                ):
                    raise ValidationError(
                        f"Perforation measured-depth range "
                        f"[{perforation.top_md}, {perforation.bottom_md}] falls "
                        f"outside well {self.name!r}'s trajectory range "
                        f"[{self.trajectory.top_measured_depth}, "
                        f"{self.trajectory.bottom_measured_depth}]."
                    )
        else:
            if not all(perforation.top_depth is not None for perforation in self.perforations):
                raise ValidationError(
                    f"Well {self.name!r} has no `trajectory`; every entry in `perforations` "
                    "must have `top_depth`/`bottom_depth` set. Set `trajectory` to use "
                    "`top_md`/`bottom_md` instead."
                )

    @property
    def n_perforations(self) -> int:
        """Total perforation count, including any with `CompletionStatus.SHUT`."""
        return len(self.perforations)

    @property
    def open_perforations(self) -> tuple[Perforation, ...]:
        """Perforations with `CompletionStatus.OPEN` only."""
        return tuple(
            perforation
            for perforation in self.perforations
            if perforation.status is CompletionStatus.OPEN
        )

    @property
    def active_perforations(self) -> tuple[Perforation, ...]:
        """Perforations with `WellStatus.ACTIVE` only."""
        return tuple(
            perforation
            for perforation in self.perforations
            if perforation.schedule_status is WellStatus.ACTIVE
        )

    @property
    def is_active(self) -> bool:
        """`True` if `schedule_status is WellStatus.ACTIVE`."""
        return self.schedule_status is WellStatus.ACTIVE

    @property
    def min_perforation_depth(self) -> Number:
        """Shallowest true vertical depth across all perforations (open or shut)."""
        if self.trajectory is not None:
            return min(
                self.trajectory.position_at(perforation.top_md)[2]  # type: ignore[union-attr]
                for perforation in self.perforations
            )
        return min(perforation.top_depth for perforation in self.perforations)  # type: ignore[union-attr]

    @property
    def max_perforation_depth(self) -> Number:
        """Deepest true vertical depth across all perforations (open or shut)."""
        if self.trajectory is not None:
            return max(
                self.trajectory.position_at(perforation.bottom_md)[2]  # type: ignore[union-attr]
                for perforation in self.perforations
            )
        return max(perforation.bottom_depth for perforation in self.perforations)  # type: ignore[union-attr]

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `Well` with all dimensioned fields converted to *target*.

        :param target: Target `UnitSystem`.
        :param table: Optional explicit `UnitConversionTable` override,
            same as every other `.convert()` in this codebase.
        :returns: New `Well` in `target` units, or `self` if already there.
        """
        if self.unit_system == target:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        length_factor = factors["length"]

        trajectory: WellTrajectory | None
        perforations: tuple[Perforation, ...]

        if self.trajectory is not None:
            trajectory = WellTrajectory(
                stations=tuple(
                    attrs.evolve(
                        station,
                        x=scale(station.x, length_factor),
                        y=scale(station.y, length_factor),
                        z=scale(station.z, length_factor),
                        measured_depth=scale(station.measured_depth, length_factor),
                    )
                    for station in self.trajectory.stations
                )
            )
            perforations = tuple(
                attrs.evolve(
                    perforation,
                    top_md=scale(perforation.top_md, length_factor),  # type: ignore[union-attr]
                    bottom_md=scale(perforation.bottom_md, length_factor),  # type: ignore[union-attr]
                    wellbore_radius=scale(perforation.wellbore_radius, length_factor),
                )
                for perforation in self.perforations
            )
        else:
            trajectory = None
            perforations = tuple(
                attrs.evolve(
                    perforation,
                    top_depth=scale(perforation.top_depth, length_factor),  # type: ignore[union-attr]
                    bottom_depth=scale(perforation.bottom_depth, length_factor),  # type: ignore[union-attr]
                    wellbore_radius=scale(perforation.wellbore_radius, length_factor),
                )
                for perforation in self.perforations
            )
        return attrs.evolve(
            self,
            surface_location=(
                scale(self.surface_location[0], length_factor),
                scale(self.surface_location[1], length_factor),
            ),
            reference_depth=scale(self.reference_depth, length_factor),
            perforations=perforations,
            trajectory=trajectory,
            unit_system=target,
        )


class Wells(
    StoreSerializable,
    fields={
        "wells": typing.Mapping[str, Well],
        "unit_system": typing.Optional[UnitSystem],  # noqa: UP045
    },
):
    """Name-keyed container of `Well` objects"""

    __slots__ = ("unit_system", "wells")

    def __init__(
        self,
        wells: typing.Mapping[str, Well] | typing.Sequence[Well],
        unit_system: UnitSystem | None = None,
    ) -> None:
        """
        :param wells: Sequence of `Well`s or a mapping from well name to `Well`.
        :param unit_system: Target unit system for every well. None
            requires all wells to already share the same unit system.
        :raises ValidationError: If wells is empty, any key doesn't match
            its value's `Well.name`, or (unit_system is None) the wells
            don't all share one unit system.
        """
        if not wells:
            raise ValidationError("`wells` must contain at least one entry.")

        if isinstance(wells, Mapping):
            mismatched = {key: well.name for key, well in wells.items() if key != well.name}
            if mismatched:
                raise ValidationError(
                    f"`wells` dict keys must match `Well.name`; mismatches "
                    f"(key -> well.name): {mismatched}."
                )
            all_wells = wells
        else:
            all_wells = {well.name: well for well in wells}

        if unit_system is None:
            systems = {well.unit_system for well in all_wells.values()}
            if len(systems) > 1:
                raise ValidationError(
                    "All wells must share the same unit system when "
                    "`unit_system` is not explicitly provided. Found: "
                    f"{sorted(s.value for s in systems)}."
                )
            unit_system = systems.pop()
            all_wells = dict(all_wells)
        else:
            all_wells = {
                name: well if well.unit_system == unit_system else well.convert(unit_system)
                for name, well in all_wells.items()
            }

        self.wells = all_wells
        self.unit_system = unit_system

    def get(self, name: str) -> Well:
        """
        Retrieve a registered `Well` by name.

        :param name: `Well` name.
        :returns: `Well` for that well.
        :raises KeyError: If no well with that name exists.
        """
        well = self.wells.get(name)
        if well is None:
            raise KeyError(f"No well named {name!r}. Available: {sorted(self.wells)}.")
        return well

    def add(self, well: Well) -> None:
        """
        Add a new `Well` to the container. If a well with the same name already exists, it is replaced.

        :param well: `Well` to add.
        """
        if well.unit_system != self.unit_system:
            raise ValidationError(
                f"`Well` {well.name!r} has unit system {well.unit_system}, expected {self.unit_system}."
            )
        self.wells[well.name] = well

    def set(self, name: str, well: Well) -> None:
        """
        Set an existing `Well` in the container by name. The name must match the `Well.name`.

        :param name: Name of the well to set.
        :param well: `Well` to set.
        """
        if name != well.name:
            raise ValidationError(
                f"Name mismatch: key {name!r} does not match Well.name {well.name!r}."
            )
        self.add(well)

    @property
    def names(self) -> tuple[str, ...]:
        """All well names, in insertion order."""
        return tuple(self.wells.keys())

    @property
    def producers(self) -> tuple[Well, ...]:
        """All wells with `well_type is WellType.PRODUCER`."""
        return tuple(well for well in self.wells.values() if well.well_type is WellType.PRODUCER)

    @property
    def injectors(self) -> tuple[Well, ...]:
        """All wells with `well_type is WellType.INJECTOR`."""
        return tuple(well for well in self.wells.values() if well.well_type is WellType.INJECTOR)

    @property
    def active(self) -> tuple[Well, ...]:
        """All wells with `schedule_status is WellStatus.ACTIVE`."""
        return tuple(well for well in self.wells.values() if well.is_active)

    @property
    def pending(self) -> tuple[Well, ...]:
        """All wells with `schedule_status is WellStatus.PENDING`."""
        return tuple(well for well in self.wells.values() if not well.is_active)

    @classmethod
    def from_deck(cls, deck_file: DeckFile, *, grid: Grid) -> Self:
        """
        Load the `Wells` object from a parsed `DeckFile`.

        :param deck_file: Parsed deck containing `WELSPECS`/`COMPDAT`/`WCONINJE`.
        :param grid: Grid built from the same deck.
        :param well_kwargs: Forwarded to `wells_from_records` and passed
            to the loaded `Well` instance.
        :returns: `Wells` for every well in the deck.
        """
        from bores.wells.deck import load_wells

        return typing.cast(Self, load_wells(deck_file, grid))

    def __getitem__(self, name: str) -> Well:
        return self.get(name)

    def __setitem__(self, name: str, well: Well) -> None:
        self.set(name, well)

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self.wells)

    def __len__(self) -> int:
        return len(self.wells)

    def __add__(self, other: Self | Well) -> Self:
        """
        Return a new `Wells` object containing the union of this and *other*.

        :param other: Another `Wells` or a single `Well`.
        :returns: New `Wells` with all wells from both.
        :raises ValidationError: If any well names collide.
        """
        if isinstance(other, Well):
            other_wells = {other.name: other}
        elif isinstance(other, Wells):
            other_wells = other.wells
        else:
            raise TypeError(f"Cannot add {type(other).__name__} to `{type(self).__name__}`.")

        overlapping_names = set(self.wells) & set(other_wells)
        if overlapping_names:
            raise ValidationError(
                f"Cannot add `Well`s with overlapping names: {sorted(overlapping_names)}."
            )

        combined_wells = {**self.wells, **other_wells}
        return self.__class__(wells=combined_wells, unit_system=self.unit_system)

    def __iadd__(self, other: Self | Well) -> Self:
        """
        In-place addition of another `Wells` or a single `Well`.

        :param other: Another `Wells` or a single `Well`.
        :returns: Self with wells from *other* added.
        :raises ValidationError: If any well names collide.
        """
        if isinstance(other, Well):
            other_wells = {other.name: other}
        elif isinstance(other, Wells):
            other_wells = other.wells
        else:
            raise TypeError(f"Cannot add {type(other).__name__} to `{type(self).__name__}`.")

        overlapping_names = set(self.wells) & set(other_wells)
        if overlapping_names:
            raise ValidationError(
                f"Cannot add `Well`s with overlapping names: {sorted(overlapping_names)}."
            )

        for name, well in other_wells.items():
            self.set(name, well)
        return self

    def __contains__(self, name: object) -> bool:
        return name in self.wells

    def __dump__(self) -> dict[str, typing.Any]:
        return {"wells": {name: well.dump() for name, well in self.wells.items()}}

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        wells = {name: Well.load(well_data) for name, well_data in data["wells"].items()}
        return cls(wells=wells)

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Returns a new `Wells` object in the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `Wells` with every well converted to target.
        """
        if target == self.unit_system:
            return self
        return self.__class__(
            wells={name: well.convert(target, table=table) for name, well in self.wells.items()},
            unit_system=target,
        )

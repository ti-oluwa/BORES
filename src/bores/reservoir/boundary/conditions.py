import logging
import typing
import warnings

import attrs
import numpy as np
from typing_extensions import Self

from bores.errors import ValidationError
from bores.reservoir.boundary.base import BoundaryCondition
from bores.reservoir.boundary.types import ConstantFluxBoundary
from bores.serde.stores import StoreSerializable
from bores.types import IntArray, NDimension, OneDimension, UnitConversionTable, UnitSystem

logger = logging.getLogger(__name__)

__all__ = ["BoundaryConditions", "BoundaryRegion"]


@attrs.frozen
class BoundaryRegion(StoreSerializable):
    """
    Associates a named `BoundaryCondition` with a set of boundary faces.

    A `BoundaryRegion` is the unit of assignment. It defines what faces
    (identified by their positions in `Grid.boundary_face_indices`)
    are governed by what condition.

    **Face positions vs global face indices**

    `face_positions` stores positions (0-based offsets) into
    `Grid.boundary_face_indices`, not global face indices. This means that:

    - The solver can look up the boundary half-transmissibility directly as
      `transmissibilities.boundary[face_positions]` without a secondary
      mapping step.
    - Changing the grid (e.g. refining or coarsening) invalidates the
      positions, so regions must be rebuilt whenever the grid changes.

    **Name**

    The `name` field is purely informational. It is used in log messages
    and error reporting. It does not need to match any other identifier in
    the model and does not need to be unique within a `BoundaryConditions`
    container (though uniqueness is recommended for clarity).

    :param name: Human-readable label (e.g. `"south_aquifer"`, `"producer_flank"`).
    :param face_positions: Shape `(n_faces,)` int32 - 0-based positions
        into `Grid.boundary_face_indices` that belong to this region.
    :param condition: The `BoundaryCondition` applied at these faces.
    """

    name: str
    """Human-readable label for this region."""

    face_positions: IntArray[OneDimension]
    """
    Shape `(n_faces,)` int32 - positions into `Grid.boundary_face_indices`.

    These are not global face indices. Use `grid.boundary_face_indices[face_positions]` 
    to recover global indices.
    """

    condition: BoundaryCondition
    """The boundary condition applied at the faces in this region."""

    def __attrs_post_init__(self) -> None:
        face_positions = np.asarray(self.face_positions, dtype=np.int32)
        if face_positions.ndim != 1:
            raise ValidationError("`face_positions` must be a 1-D array.")
        object.__setattr__(self, "face_positions", face_positions)

    @property
    def unit_system(self) -> UnitSystem:
        return self.condition.unit_system

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `BoundaryRegion` with the condition rescaled to *target*.

        `face_positions` and `name` are copied unchanged.

        :param target: Target `UnitSystem`.
        :returns: New `BoundaryRegion` in *target* units.
        """
        return self.__class__(
            name=self.name,
            face_positions=self.face_positions.copy(),
            condition=self.condition.convert(target, table=table),
        )

    @classmethod
    def no_flow(
        cls,
        name: str,
        face_positions: IntArray[NDimension],
        unit_system: UnitSystem = UnitSystem.FIELD,
    ) -> Self:
        """
        Convenience constructor for a sealed (no-flow) boundary region.

        Equivalent to `BoundaryRegion(name, face_positions, ConstantFluxBoundary(flux=0.0))`.

        :param name: Region label.
        :param face_positions: Boundary face positions.
        :param unit_system: Unit system label for the condition.
        :returns: `BoundaryRegion` with a no-flow `ConstantFluxBoundary`.
        """
        return cls(
            name=name,
            face_positions=typing.cast(
                IntArray[OneDimension], face_positions.astype(np.int32, copy=False)
            ),
            condition=ConstantFluxBoundary(flux=0.0, unit_system=unit_system),
        )

    def __dump__(self) -> dict[str, typing.Any]:
        return {
            "name": self.name,
            "face_positions": self.face_positions.tolist(),
            "condition": self.condition.dump(),
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        return cls(
            name=data["name"],
            face_positions=typing.cast(
                IntArray[OneDimension],
                np.asarray(data["face_positions"], dtype=np.int32),
            ),
            condition=BoundaryCondition.load(data["condition"]),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"n_faces={len(self.face_positions)}, "
            f"condition={self.condition!r}, "
            f"unit_system={self.unit_system!r}"
            f")"
        )


@attrs.frozen(slots=True)
class BoundaryConditions(StoreSerializable):
    """
    Ordered collection of `BoundaryRegion` objects governing all boundary
    faces of a reservoir model.

    **Default behaviour**

    Any boundary face not covered by any `BoundaryRegion` in `regions`
    defaults to **zero flux** (no-flow, sealed boundary). This means an
    empty `BoundaryConditions()` gives a fully sealed reservoir.

    **Overlap handling**

    If two regions assign different conditions to the same face position,
    the region that appears later in `regions` wins at compile time.

    `compile_boundary_conditions` processes regions in order and
    overwrites earlier assignments. Overlapping regions are not an error
    but produce a warning here, at construction.

    **Unit system**

    All regions must share the same `unit_system` as the model. Use
    `convert(target)` to rescale the whole container before compiling it
    for a model that operates in a different unit system.
    """

    regions: list[BoundaryRegion] = attrs.field(factory=list)
    """
    Ordered list of boundary regions. Compiled in list order; 
    later regions override earlier ones on overlapping faces.
    """
    unit_system: UnitSystem | None = None
    """
    Target unit system for all regions. `None` requires every region in
    `regions` to already share the same unit system and then resolves to that
    shared value post-initialization. When given, every region is converted to it.
    """

    _regions_map: dict[str, BoundaryRegion] = attrs.field(
        factory=dict, init=False, repr=False, eq=False
    )
    """
    Precomputed `{region.name: region}`, built once here rather than a
    linear scan on every `get_region` call. The last region with a given
    name wins, matching `remove_region`'s/`get_region`'s own
    first-match-in-list-order semantics only for a unique name.
    """

    def __attrs_post_init__(self) -> None:
        # Warn on overlapping face assignments
        seen: dict[int, str] = {}
        for region in self.regions:
            for position in region.face_positions:
                position = int(position)
                if position in seen:
                    warnings.warn(
                        f"Boundary face position {position} is assigned by both "
                        f"region {seen[position]!r} and region {region.name!r}. "
                        f"Region {region.name!r} (later in list) will take effect.",
                        UserWarning,
                        stacklevel=3,
                    )
                seen[position] = region.name

        unit_system = self.unit_system
        if unit_system is None:
            systems = {region.unit_system for region in self.regions}
            if len(systems) > 1:
                raise ValidationError(
                    "All regions must share the same unit system when "
                    "`unit_system` is not explicitly provided. "
                    f"Found: {sorted(s.value for s in systems)}. "
                    "Pass `unit_system` explicitly to convert all regions to "
                    "a common system."
                )
            resolved = systems.pop() if systems else UnitSystem.FIELD
            object.__setattr__(self, "unit_system", resolved)
        else:
            converted = [
                region if region.unit_system == unit_system else region.convert(unit_system)
                for region in self.regions
            ]
            object.__setattr__(self, "regions", converted)

        object.__setattr__(self, "_regions_map", {region.name: region for region in self.regions})

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `BoundaryConditions` with every region's condition
        rescaled to *target*.

        Useful when the model unit system changes after the boundary conditions
        have been defined (e.g. loading a FIELD-unit deck and converting the
        entire model to `METRIC` for a METRIC-unit solver).

        :param target: Target `UnitSystem`.
        :returns: New `BoundaryConditions` in *target* units.
        """
        return attrs.evolve(
            self,
            regions=[region.convert(target, table=table) for region in self.regions],
        )

    def add_region(self, region: BoundaryRegion, unit_system: UnitSystem | None = None) -> Self:
        """
        Return a new `BoundaryConditions` with *region* appended.

        Since `BoundaryConditions` is frozen, this does not mutate the
        current instance.

        :param region: `BoundaryRegion` to append.
        :returns: New `BoundaryConditions` with *region* at the end of the
            list (lowest override priority among overlapping faces relative to
            later additions).
        """
        return attrs.evolve(self, regions=[*self.regions, region], unit_system=unit_system)

    def remove_region(self, name: str) -> Self:
        """
        Return a new `BoundaryConditions` with the named region removed.

        If no region with `name` exists, returns `self` unchanged and
        logs a warning.

        :param name: `BoundaryRegion.name` to remove.
        :returns: New `BoundaryConditions` without the named region.
        """
        remaining = [region for region in self.regions if region.name != name]
        if len(remaining) == len(self.regions):
            warnings.warn(
                f"No boundary region named {name!r} found. `remove_region` had no effect.",
                UserWarning,
                stacklevel=2,
            )
            return self
        return attrs.evolve(self, regions=remaining)

    def get_region(self, name: str) -> BoundaryRegion:
        """
        Return the region with the given name.

        :param name: Region name.
        :returns: Matching `BoundaryRegion`.
        :raises KeyError: If no region with `name` is found.
        """
        try:
            return self._regions_map[name]
        except KeyError:
            available = list(self._regions_map)
            raise KeyError(
                f"No boundary region named {name!r}. Available regions: {available}."
            ) from None

    @property
    def n_regions(self) -> int:
        """Number of boundary regions."""
        return len(self.regions)

    @property
    def all_no_flow(self) -> bool:
        """
        Return `True` if every region applies a no-flow flux condition.

        An empty `BoundaryConditions` (no regions) also returns `True`
        since the default is no-flow everywhere.
        """
        return all(
            isinstance(region.condition, ConstantFluxBoundary) and region.condition.is_no_flow()
            for region in self.regions
        )

    def __repr__(self) -> str:
        region_names = [region.name for region in self.regions]
        return f"""
        {self.__class__.__name__}(
            n_regions={self.n_regions}, 
            regions={region_names}, 
            unit_system={self.unit_system!r},
        )
        """

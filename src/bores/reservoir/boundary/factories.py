"""
Convenience factories for building `BoundaryRegion`/`BoundaryConditions`
objects from a `Grid`, without hand-deriving boundary face positions.

Building a `BoundaryRegion` by hand requires pulling `Grid.boundary_face_indices`,
figuring out which of those global face indices actually sit on the flank of
the domain you care about, then converting the survivors back into
positions within `boundary_face_indices`, which is the indexing
`BoundaryRegion.face_positions` actually expects, not global face indices.
"""

import logging
import typing

from bores.grids.base import Grid
from bores.grids.utils import classify_boundary_faces, resolve_side
from bores.reservoir.boundary.base import BoundaryCondition
from bores.reservoir.boundary.conditions import BoundaryConditions, BoundaryRegion
from bores.types import IntArray, OneDimension, Side, UnitSystem

logger = logging.getLogger(__name__)

__all__ = ["make_axis_aligned_boundary_conditions", "make_boundary_region"]


def make_boundary_region(
    name: str,
    grid: Grid,
    side: Side | str,
    condition: BoundaryCondition,
    *,
    classified: typing.Mapping[Side, IntArray[OneDimension]] | None = None,
) -> BoundaryRegion:
    """
    Build a `BoundaryRegion` covering one axis-aligned flank of `grid`.

    ```python
    aquifer_region = make_boundary_region(
        name="south_aquifer",
        grid=grid,
        side="south",
        condition=CarterTracyAquifer(
            initial_pressure=4500.0,
            aquifer_constant=1.2e6,
        ),
    )
    ```
    :param name: `BoundaryRegion.name`.
    :param grid: Grid to classify. Ignored if `classified` is supplied.
    :param side: Which flank - a `Side` member, its `.value` string
        ('west'/'east'/'south'/'north'/'top'/'bottom'), or a common alias
        ('left'/'right'/'front'/'back'/'up'/'down').
    :param condition: `BoundaryCondition` applied at every face on that side.
    :param classified: Pre-computed `classify_boundary_faces(grid)` result.
        Pass this when building several regions on the same grid to avoid
        re-classifying every boundary face once per region.
    :returns: `BoundaryRegion` covering every face on that side. May have
        zero faces if the grid has no boundary faces on that flank -
        construction still succeeds; an empty region simply contributes
        nothing when evaluated.
    :raises ValidationError: If `side` doesn't resolve to a known `Side`.
    """
    resolved_side = resolve_side(side)
    faces = classified if classified is not None else classify_boundary_faces(grid)
    return BoundaryRegion(
        name=name,
        face_positions=faces[resolved_side],
        condition=condition,
    )


def make_axis_aligned_boundary_conditions(
    grid: Grid,
    sides: typing.Mapping[Side | str, BoundaryCondition],
    *,
    default: BoundaryCondition | None = None,
    unit_system: UnitSystem | None = None,
) -> BoundaryConditions:
    """
    Build a complete `BoundaryConditions` from a `{side: condition}` mapping
    in one call.

    ```python
    boundary_conditions = make_axis_aligned_boundary_conditions(
        grid,
        sides={
            "south": CarterTracyAquifer(initial_pressure=4500.0, aquifer_constant=1.2e6),
            "north": ConstantPressureBoundary(pressure=4200.0),
        },
        default=ConstantFluxBoundary(flux=0.0),  # explicit no-flow elsewhere
    )
    ```
    :param grid: Grid to classify.
    :param sides: Maps each flank you want to assign to its
        `BoundaryCondition`. Keys accept a `Side`, its `.value` string, or a
        common alias, same as `make_boundary_region`.
    :param default: `BoundaryCondition` applied to every `Side` not present
        in `sides`. When `None` (the default), sides not present in `sides`
        get no explicit region at all - they fall through to
        `BoundaryConditions`' own implicit zero-flux default rather than an
        explicit `ConstantFluxBoundary(flux=0.0)` region. The two are
        physically equivalent; passing an explicit `default` is mainly
        useful when you want every side to show up by name in
        `BoundaryConditions.regions` (e.g. for logging/introspection).
    :param unit_system: Forwarded to `BoundaryConditions`. Every region's
        condition is converted to this unit system if it isn't already in
        it; when `None`, every condition passed in `sides`/`default` must
        already share the same unit system.
    :returns: `BoundaryConditions` with one `BoundaryRegion` per side that
        has a condition (from `sides` or `default`) and at least one
        classified face. Sides with zero classified faces are skipped with
        a warning rather than added as an empty region.
    :raises ValidationError: If any key in `sides` doesn't resolve to a
        known `Side`, or the regions end up with mismatched unit systems
        and `unit_system` is `None`.
    """
    classified = classify_boundary_faces(grid)
    resolved_sides = {resolve_side(key): condition for key, condition in sides.items()}

    regions: list[BoundaryRegion] = []
    for side in Side:
        condition = resolved_sides.get(side, default)
        if condition is None:
            continue

        positions = classified[side]
        if len(positions) == 0:
            logger.warning(
                "No boundary faces classified for side %r on this grid; skipping region for it.",
                side.value,
            )
            continue

        regions.append(
            BoundaryRegion(
                name=side.value,
                face_positions=positions,
                condition=condition,
            )
        )
    return BoundaryConditions(regions=regions, unit_system=unit_system)

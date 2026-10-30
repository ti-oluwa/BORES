"""Resolve well perforations to grid cells."""

import itertools
import math
import typing

import attrs
import numpy as np
from typing_extensions import Self

from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.grids.containment import cell_contains_point, get_first_cell_containing_point
from bores.grids.traversal import check_point_in_polygon_2d, walk_segment_cells
from bores.types import (
    Boolean,
    GridIntersectionMethod,
    IntArray,
    Integer,
    Number,
    OneDimension,
    Orientation,
    UnitConversionTable,
    UnitSystem,
)
from bores.units import get_conversion_factors
from bores.utils import scale
from bores.wells.base import Perforation, Well, convert_perforation

__all__ = [
    "PerforationIndex",
    "resolve_known_perforations_indices",
    "resolve_md_perforations_indices",
    "resolve_perforation_orientation",
    "resolve_perforations_indices",
]


@attrs.frozen(kw_only=True, slots=True)
class PerforationIndex:
    """One resolved (perforation, cell) pair, Basically, a connection."""

    perforation: Perforation
    """The perforation this connection belongs to."""

    cell_index: Integer
    """The grid cell this connection resolves to."""

    partial_penetration_fraction: Number
    """
    Fraction of the cell's completion-direction length that `perforation`
    actually overlaps, in `(0, 1]`.
    """

    representative_depth: Number
    """
    Midpoint true vertical depth of the overlap between `perforation`'s
    interval and this cell.
    """

    representative_measured_depth: Number
    """
    Midpoint measured depth of the overlap between `perforation`'s interval and this cell, along
    the wellbore. Equals `representative_depth` for a vertical well. The distance between two
    connections along the wellbore is the difference of these.
    """

    inclination_from_vertical: Number
    """
    Angle in radians between this connection's local wellbore direction
    and the vertical. `0` for a straight-down connection, `pi/2` for a
    horizontal one. `wells.hydraulics` reads this directly for computations.
    """

    well_index: Number | None = None
    """Connection factor for this connection. `None` until computed."""

    conductivity: Number | None = None
    """
    Peaceman numerator (`k*h*N/G`) for this connection, isolated from its
    skin/geometry denominator. `None` for an overridden connection
    (`Perforation.connection_factor_override`), where `well_index` bypasses
    the Peaceman formula entirely and has no such decomposition. Used to
    recompute an effective `well_index` under a rate-dependent skin
    addition (non-Darcy `D*q`) without rederiving the whole formula.
    """

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system this connection's dimensioned fields are expressed in."""

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Returns a  new `PerforationIndex` in the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `PerforationIndex` with `representative_depth` and
            `well_index` converted to target.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        length_factor = factors["length"]
        connection_factor = factors["permeability"] * length_factor
        return attrs.evolve(
            self,
            perforation=convert_perforation(
                self.perforation, length_factor=length_factor, connection_factor=connection_factor
            ),
            representative_depth=scale(self.representative_depth, length_factor),
            representative_measured_depth=scale(self.representative_measured_depth, length_factor),
            well_index=scale(self.well_index, connection_factor),
            conductivity=scale(self.conductivity, connection_factor),
            unit_system=target,
        )


def resolve_perforation_orientation(perforation: Perforation) -> Orientation:
    """
    Resolve which axis a `Perforation`'s completion runs along.

    Resolution order: `perforation.direction` if set (and not
    `Orientation.UNSET`); otherwise `Orientation.Z`.

    :param perforation: The perforation being resolved.
    :returns: `Orientation.X`, `Orientation.Y`, or `Orientation.Z`.
    """
    if perforation.direction is not None and perforation.direction is not Orientation.UNSET:
        return perforation.direction
    return Orientation.Z


def get_default_horizontal_tolerance(grid: Grid, x: Number, y: Number) -> Number:
    """
    Derive a default horizontal search radius for `resolve_perforations_indices`.

    Evaluated locally at the well's own position (half the diagonal of the
    nearest cell), since cell size can vary across an unstructured grid.

    :param grid: Grid to sample local cell size from.
    :param x: Well surface x-coordinate.
    :param y: Well surface y-coordinate.
    :returns: Radius in the grid's length units.
    """
    z = grid.bounding_box[4]
    nearest = grid.find_nearest_cell(x, y, z)
    dx = grid.cell_length_x[nearest]
    dy = grid.cell_length_y[nearest]
    return 0.5 * math.sqrt(dx**2 + dy**2)


def get_vertical_column_candidates(
    grid: Grid, x: Number, y: Number, tolerance: Number
) -> IntArray[OneDimension]:
    """
    Return every cell index whose centroid falls within `tolerance` of
    `(x, y)` in the horizontal plane, across all depths.

    `grid.find_cells_in_radius` is a 3-D sphere query, not a vertical
    column query, so this queries once with a radius large enough to span
    the full depth range from the grid's mid-depth, then filters down to
    the true horizontal distance using `grid.cell_centroids[:, :2]`.

    :param grid: Grid to query.
    :param x: Well surface x-coordinate.
    :param y: Well surface y-coordinate.
    :param tolerance: Horizontal radius (grid length units).
    :returns: Sorted 1-D int array of candidate cell indices, any depth, no
        duplicates, not yet filtered by perforation depth.
    """
    z_min, z_max = grid.bounding_box[4], grid.bounding_box[5]
    mid_z = 0.5 * (z_min + z_max)
    half_depth_span = 0.5 * (z_max - z_min)
    query_radius = math.sqrt(tolerance**2 + half_depth_span**2)

    raw_candidates = grid.find_cells_in_radius(x, y, mid_z, query_radius)
    if raw_candidates.size == 0:
        return raw_candidates

    assert grid.cell_centroids is not None
    centroids_xy = grid.cell_centroids[raw_candidates, :2]
    horizontal_distances = np.sqrt((centroids_xy[:, 0] - x) ** 2 + (centroids_xy[:, 1] - y) ** 2)
    filtered_candidates = raw_candidates[horizontal_distances <= tolerance]
    return typing.cast("IntArray[OneDimension]", np.sort(filtered_candidates))


def compute_local_vertical_extent_aabb(
    grid: Grid, cell_index: Integer, x: Number, y: Number
) -> tuple[Number, Number] | None:
    """
    Get vertical extent of a cell's axis-aligned bounding box.

    Every point within the cell's horizontal footprint sees the same
    `[cell_min_z, cell_max_z]` window under this method. `x`/`y` are
    accepted for signature parity with `compute_local_vertical_extent_exact` and
    otherwise unused.

    :param grid: Grid providing `cell_min_xyz`/`cell_max_xyz`.
    :param cell_index: Cell to measure.
    :param x: Unused.
    :param y: Unused.
    :returns: `(top_depth, bottom_depth)`, always.
    """
    return (
        grid.cell_min_xyz[cell_index, 2],
        grid.cell_max_xyz[cell_index, 2],
    )


def compute_local_vertical_extent_exact(
    grid: Grid, cell_index: Integer, x: Number, y: Number
) -> tuple[Number, Number] | None:
    """
    Get exact vertical extent of a cell at horizontal position
    `(x, y)`.

    A vertical line through `(x, y)` pierces a convex cell's boundary at
    exactly two faces. For each face of the cell, intersects the line
    with that face's plane (via the face centroid and outward normal),
    then keeps the intersection only if `(x, y)` falls inside that face's
    actual polygon (projected to the horizontal plane) rather than just
    its infinite plane.

    Faces whose outward normal is close to horizontal
    are skipped outright as a vertical line doesn't pierce a vertical
    (side) face at a well-defined single point, and those faces bound the
    cell's sides, not its top or bottom.

    :param grid: Grid providing face topology and geometry.
    :param cell_index: Cell to measure.
    :param x: Horizontal x-coordinate of the vertical line.
    :param y: Horizontal y-coordinate of the vertical line.
    :returns: `(top_depth, bottom_depth)`, or `None` if `(x, y)` does not actually
        fall within this cell's horizontal footprint.
    """
    crossings: list[float] = []
    for face_idx in grid.get_cell_face_indices(cell_index):
        normal = grid.get_face_normal_for_cell(face_idx, cell_index)
        normal_z = normal[2]
        if abs(normal_z) < 1e-9:
            continue

        centroid = grid.face_centroids[face_idx]
        z_intersect = (
            centroid[2]
            - (normal[0] * (x - centroid[0]) + normal[1] * (y - centroid[1])) / normal_z
        )

        face_vertex_xy = grid.get_face_vertex_coordinates(face_idx)[:, :2]
        if check_point_in_polygon_2d(x, y, polygon_xy=face_vertex_xy):  # type: ignore[arg-type]
            crossings.append(z_intersect)

    if not crossings:
        return None
    return min(crossings), max(crossings)


def resolve_known_perforations_indices(
    *,
    grid: Grid,
    well: Well,
    perforations: typing.Sequence[Perforation] | None = None,
) -> tuple[PerforationIndex, ...]:
    """
    Resolve every open perforation on `well` straight to its own
    `Perforation.cell_index`, skipping geometric derivation entirely.

    For a completion built from a structured-grid deck record
    (`COMPDAT`), the grid cell a single-layer connection connects to is
    already known exactly. Re-deriving it geometrically would need the
    well's true lateral path. `resolve_perforations_indices` only looks
    near one `surface_location`, and a `WELSEGS`/`COMPSEGS`-derived
    trajectory carries measured depth and true vertical depth only, no
    azimuth, so `resolve_md_perforations_indices`'s spatial walk can't
    place it either. Every returned entry has `partial_penetration_fraction
    = 1.0`, since `Perforation.cell_index` is only ever set for a
    completion that fully occupies exactly one grid cell.

    :param grid: Grid to resolve against, only for `well.unit_system`'s
        consistency check; the cell itself is already known, so nothing
        here is derived from `grid`'s geometry.
    :param well: `Well` whose perforations are resolved. Every open
        perforation must already have `cell_index` set; see
        `resolve_perforations_indices`/`resolve_md_perforations_indices`
        for a well without that.
    :param perforations: Perforations to resolve, open or shut. Defaults to
        `well.open_perforations`.
    :returns: Tuple of `PerforationIndex`, one per resolved perforation.
    """
    if grid.unit_system != well.unit_system:
        raise ValidationError(
            f"Grid `unit_system` ({grid.unit_system.value}) != well "
            f"{well.name!r}'s `unit_system` ({well.unit_system.value})."
        )

    results: list[PerforationIndex] = []
    for perforation in well.open_perforations if perforations is None else perforations:
        assert perforation.cell_index is not None
        assert perforation.top_depth is not None and perforation.bottom_depth is not None
        representative_depth = 0.5 * (perforation.top_depth + perforation.bottom_depth)
        representative_measured_depth = (
            0.5 * (perforation.top_md + perforation.bottom_md)  # type: ignore[operator]
            if perforation.top_md is not None
            else representative_depth
        )
        if perforation.top_md is not None:
            # A `WELSEGS`-derived trajectory only carries measured depth and
            # true vertical depth, no azimuth, so its position/tangent along
            # a mostly-horizontal leg is degenerate (a real lateral reach
            # with no x/y to show for it). The completion's own true
            # vertical depth against its own measured-depth span gives the
            # same inclination without needing that geometry at all.
            md_length = perforation.bottom_md - perforation.top_md  # type: ignore[operator]
            tvd_length = perforation.bottom_depth - perforation.top_depth
            inclination = (
                math.acos(max(-1.0, min(1.0, tvd_length / md_length))) if md_length > 0 else 0.0
            )
        else:
            inclination = (
                0.0
                if resolve_perforation_orientation(perforation) is Orientation.Z
                else math.pi / 2.0
            )

        results.append(
            PerforationIndex(
                perforation=perforation,
                cell_index=perforation.cell_index,
                partial_penetration_fraction=1.0,
                representative_depth=representative_depth,
                representative_measured_depth=representative_measured_depth,
                inclination_from_vertical=inclination,
                unit_system=well.unit_system,
            )
        )
    return tuple(results)


def resolve_perforations_indices(
    grid: Grid,
    well: Well,
    *,
    horizontal_tolerance: Number | None = None,
    intersection_method: GridIntersectionMethod = "aabb",
    perforations: typing.Sequence[Perforation] | None = None,
) -> tuple[PerforationIndex, ...]:
    """
    Resolve every open `Perforation` on `well` to the `Grid` cell(s) it
    intersects. Only valid for a `well` with no `trajectory`.
    See `resolve_md_perforations_indices` for one with a trajectory.

    For each open perforation, this finds every cell whose vertical extent
    overlaps `[top_depth, bottom_depth]` at `well.surface_location`, and
    computes a partial-penetration fraction and a representative depth per
    cell from the overlap window. A point perforation (`top_depth == bottom_depth`)
    resolves to exactly one cell with `partial_penetration_fraction = 1.0`.

    :param grid: Grid to resolve against.
    :param well: `Well` whose perforations are resolved. Not modified.
    :param horizontal_tolerance: Search radius around `surface_location`
        used to gather vertical-column candidates, in `well.unit_system`
        length units. `None` derives a default from the grid's local cell
        size at `surface_location`.
    :param method: `"aabb"` (default) or `"exact"`.
    :param perforations: Perforations to resolve, open or shut. Defaults to
        `well.open_perforations`.
    :returns: Tuple of `PerforationIndex`, one per (perforation, cell) pair
        that overlaps. A single `Perforation` spanning multiple layers
        yields multiple `PerforationIndex` entries.
    :raises ValidationError: If `well.trajectory` is set (use
        `resolve_md_perforations_indices` instead), or a perforation's
        depth interval overlaps no cell at all (dangling completion).
    """
    if well.trajectory is not None:
        raise ValidationError(
            f"Well {well.name!r} has a trajectory; use `resolve_md_perforations_indices` instead."
        )

    if grid.unit_system != well.unit_system:
        raise ValidationError(
            f"Grid `unit_system` ({grid.unit_system.value}) != well "
            f"{well.name!r}'s `unit_system` ({well.unit_system.value})."
        )

    x, y = well.surface_location
    tolerance = (
        horizontal_tolerance
        if horizontal_tolerance is not None
        else get_default_horizontal_tolerance(grid, x, y)
    )
    candidates = get_vertical_column_candidates(grid, x, y, tolerance)
    compute_extent = (
        compute_local_vertical_extent_aabb
        if intersection_method == "aabb"
        else compute_local_vertical_extent_exact
    )

    results: list[PerforationIndex] = []
    for perforation in well.open_perforations if perforations is None else perforations:
        assert perforation.top_depth is not None and perforation.bottom_depth is not None
        inclination = (
            0.0 if resolve_perforation_orientation(perforation) is Orientation.Z else math.pi / 2.0
        )
        matches: list[PerforationIndex] = []

        if perforation.is_point_perforation:
            depth = perforation.top_depth
            for cell_index in candidates:
                extent = compute_extent(grid, cell_index, x, y)
                if extent is None:
                    continue

                top_depth, bottom_depth = extent
                if top_depth <= depth <= bottom_depth:
                    matches.append(
                        PerforationIndex(
                            perforation=perforation,
                            cell_index=cell_index,
                            partial_penetration_fraction=1.0,
                            representative_depth=depth,
                            representative_measured_depth=depth,
                            inclination_from_vertical=inclination,
                            unit_system=well.unit_system,
                        )
                    )
                    break
        else:
            for cell_index in candidates:
                extent = compute_extent(grid, cell_index, x, y)
                if extent is None:
                    continue

                top_depth, bottom_depth = extent
                overlap_top = max(perforation.top_depth, top_depth)
                overlap_bottom = min(perforation.bottom_depth, bottom_depth)
                length = max(overlap_bottom - overlap_top, 0.0)
                if length <= 0.0:
                    continue
                matches.append(
                    PerforationIndex(
                        perforation=perforation,
                        cell_index=cell_index,
                        partial_penetration_fraction=length / perforation.length,
                        representative_depth=0.5 * (overlap_top + overlap_bottom),
                        representative_measured_depth=0.5 * (overlap_top + overlap_bottom),
                        inclination_from_vertical=inclination,
                        unit_system=well.unit_system,
                    )
                )

        if not matches:
            raise ValidationError(
                f"`Perforation` [{perforation.top_depth}, {perforation.bottom_depth}] "
                f"on well {well.name!r} does not overlap any grid cell near "
                f"surface_location={well.surface_location} (dangling completion)."
            )
        results.extend(matches)

    return tuple(results)


def check_cell_contains_point(
    grid: Grid,
    cell_index: Integer,
    point: tuple[Number, Number, Number],
    tolerance: Number = 1e-6,
) -> Boolean:
    """
    `True` if `point` lies inside cell `cell_index`, or on its boundary within `tolerance`.

    The test follows the cell's own faces, so it stays correct for warped and non-convex cells.
    A cell with no faces (an inactive or pinched-out cell) contains nothing.

    :param grid: Grid providing face geometry.
    :param cell_index: Cell to test.
    :param point: `(x, y, z)` to test.
    :param tolerance: Distance from the cell's surface up to which a point still counts as inside.
    :returns: Whether the cell contains the point.
    """
    return cell_contains_point(grid, cell_index, point, tolerance)


def find_first_containing_cell(
    grid: Grid,
    candidates: IntArray[OneDimension],
    point: tuple[Number, Number, Number],
    tolerance: Number = 1e-6,
) -> Integer | None:
    """
    Find the first of `candidates` whose cell contains `point`.

    A cell with no faces (an inactive or pinched-out cell) contains nothing, so it is never
    returned.

    :param grid: Grid providing face geometry.
    :param candidates: Cell indices to test, in priority order.
    :param point: Point to locate.
    :param tolerance: Distance from a cell's surface up to which a point still counts as inside.
    :returns: The first containing cell, or `None` if none of them contains `point`.
    """
    return get_first_cell_containing_point(grid, candidates, point, tolerance)


def locate_cell(
    grid: Grid, point: tuple[Number, Number, Number], search_radius: Number
) -> Integer | None:
    """
    Find the grid cell containing `point`.

    Starts from the nearest-centroid guess (cheap, right most of the
    time); if that guess isn't actually the containing cell (common near a
    distorted or unstructured cell boundary), falls back to an
    expanding-radius search over every cell whose centroid is nearby,
    testing each with `check_cell_contains_point`.

    :param grid: Grid to search.
    :param point: Point to locate.
    :param search_radius: Initial fallback search radius; doubles each
        retry, up to 6 attempts.
    :returns: Containing cell index, or `None` if not found within the
        search budget (point outside the grid, inside an inactive or
        pinched-out cell, or resolution too coarse).
    """
    nearest = grid.find_nearest_cell(*point)
    if check_cell_contains_point(grid, nearest, point):
        return nearest

    radius = search_radius
    for _ in range(6):
        candidates = grid.find_cells_in_radius(point[0], point[1], point[2], radius)
        containing = find_first_containing_cell(grid, candidates, point)
        if containing is not None:
            return containing
        radius *= 2.0
    return None


def walk_segment_through_grid(
    grid: Grid,
    start: tuple[Number, Number, Number],
    end: tuple[Number, Number, Number],
    start_md: Number,
    end_md: Number,
    search_radius: Number,
) -> list[tuple[Integer, Number, Number]]:
    """
    Walk the straight 3-D segment from `start` to `end` through `grid`,
    cell by cell.

    Locates the cell containing `start`, then repeatedly finds the nearest
    face (in walk order) the segment exits the current cell through and
    crosses to that face's neighbor (`face_cell_indices`), accumulating
    `(cell_index, entry_md, exit_md)` for each cell entered, until the
    segment reaches `end` or exits the grid through a boundary face (no
    neighbor so the trajectory leaves the active domain; the walk simply
    stops there rather than guessing where it re-enters).

    :param grid: Grid to walk.
    :param start: Segment start point.
    :param end: Segment end point.
    :param start_md: Measured depth at `start`.
    :param end_md: Measured depth at `end`.
    :param search_radius: Forwarded to `locate_cell` for the initial cell.
    :returns: Ordered list of `(cell_index, entry_md, exit_md)`. Empty if
        `start` isn't inside any active cell (outside the grid, or inside an
        inactive or pinched-out cell).
    """
    direction = (end[0] - start[0], end[1] - start[1], end[2] - start[2])
    if math.sqrt(sum(component**2 for component in direction)) == 0:
        return []

    first_cell = locate_cell(grid, start, search_radius)
    if first_cell is None:
        return []

    cells, entry_fractions, exit_fractions = walk_segment_cells(
        grid, start=start, end=end, first_cell=first_cell
    )
    md_span = end_md - start_md
    return list(
        zip(
            cells.tolist(),
            (start_md + entry_fractions * md_span).tolist(),
            (start_md + exit_fractions * md_span).tolist(),
            strict=True,
        )
    )


def resolve_md_perforations_indices(
    grid: Grid,
    well: Well,
    *,
    search_radius: Number | None = None,
    perforations: typing.Sequence[Perforation] | None = None,
) -> tuple[PerforationIndex, ...]:
    """
    Resolve every open perforation on `well` to the `Grid` cell(s) its
    measured-depth interval passes through. Only valid for a `well` with a
    `trajectory` set. See `resolve_perforations_indices` for one without.

    For each open perforation, it gets the trajectory polyline vertices
    covering `[top_md, bottom_md]` (`WellTrajectory.stations_between`),
    then walks each leg of that polyline through `grid`
    (`walk_segment_through_grid`) to find every cell it passes through,
    the measured-depth sub-range within each, and (from the leg's
    direction) the local inclination from vertical.

    :param grid: Grid to resolve against.
    :param well: `Well` whose perforations are resolved. Not modified.
    :param search_radius: Forwarded to `locate_cell` for each leg's
        starting-cell search. `None` derives a default from the grid's
        local cell size at `well.surface_location`.
    :param perforations: Perforations to resolve, open or shut. Defaults to
        `well.open_perforations`.
    :returns: Tuple of `PerforationIndex`, one per (perforation, cell) the
        trajectory passes through in that perforation's MD range, in
        along-hole order. A perforation crossing several cells yields
        several entries.
    :raises ValidationError: If `well.trajectory` is `None` (use
        `resolve_perforations_indices` instead), or a perforation's MD
        range doesn't intersect any grid cell at all (trajectory passes
        entirely outside the active grid over that range).
    """
    if well.trajectory is None:
        raise ValidationError(
            f"Well {well.name!r} has no trajectory; use `resolve_perforations_indices` instead."
        )

    if grid.unit_system != well.unit_system:
        raise ValidationError(
            f"Grid `unit_system` ({grid.unit_system.value}) != well "
            f"{well.name!r}'s `unit_system` ({well.unit_system.value})."
        )

    trajectory = well.trajectory
    x, y = well.surface_location
    radius = (
        search_radius
        if search_radius is not None
        else get_default_horizontal_tolerance(grid, x, y)
    )

    results: list[PerforationIndex] = []
    for perforation in well.open_perforations if perforations is None else perforations:
        assert perforation.top_md is not None and perforation.bottom_md is not None
        top_md = perforation.top_md
        true_bottom_md = perforation.bottom_md
        bottom_md = max(true_bottom_md, top_md + 1e-9)
        # A point perforation (top_md == bottom_md) has no direction to
        # walk, so nudge to an infinitesimal interval so it still resolves
        # to exactly one cell via the same walk machinery, rather than a
        # separate point-lookup code path.

        stations = trajectory.stations_between(top_md, bottom_md)
        matches: list[PerforationIndex] = []

        for previous, current in itertools.pairwise(stations):
            start = (previous.x, previous.y, previous.z)
            end = (current.x, current.y, current.z)
            leg_dx, leg_dy, leg_dz = (
                end[0] - start[0],
                end[1] - start[1],
                end[2] - start[2],
            )
            leg_length = math.sqrt(leg_dx**2 + leg_dy**2 + leg_dz**2)
            if leg_length == 0:
                continue
            inclination = math.acos(max(-1.0, min(1.0, leg_dz / leg_length)))

            crossings = walk_segment_through_grid(
                grid=grid,
                start=start,
                end=end,
                start_md=previous.measured_depth,
                end_md=current.measured_depth,
                search_radius=radius,
            )
            for cell_index, entry_md, exit_md in crossings:
                sub_length = exit_md - entry_md
                if sub_length <= 0.0:
                    continue
                matches.append(
                    PerforationIndex(
                        perforation=perforation,
                        cell_index=cell_index,
                        partial_penetration_fraction=(
                            1.0
                            if perforation.is_point_perforation
                            else sub_length / (true_bottom_md - top_md)
                        ),
                        representative_depth=trajectory.position_at(0.5 * (entry_md + exit_md))[2],
                        representative_measured_depth=0.5 * (entry_md + exit_md),
                        inclination_from_vertical=inclination,
                        unit_system=well.unit_system,
                    )
                )

        if not matches:
            raise ValidationError(
                f"Perforation [{perforation.top_md}, {perforation.bottom_md}] on "
                f"well {well.name!r} does not intersect any grid cell along its "
                "trajectory (dangling completion, or the trajectory passes "
                "outside the active grid over this range)."
            )
        results.extend(matches)

    return tuple(results)

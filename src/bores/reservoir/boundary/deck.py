"""Load `BoundaryConditions` from a parsed deck."""

import typing
import warnings

import numpy as np

from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.grids.utils import resolve_boundary_faces_for_box
from bores.reservoir.boundary.aquifers.carter_tracy import CarterTracyAquifer
from bores.reservoir.boundary.aquifers.fetkovich import FetkovichAquifer
from bores.reservoir.boundary.base import BoundaryCondition
from bores.reservoir.boundary.conditions import BoundaryConditions, BoundaryRegion
from bores.reservoir.boundary.types import ConstantFluxBoundary
from bores.types import IntArray, Number, NumberArray, OneDimension, UnitSystem
from bores.utils import get_hydrostatic_gradient_factor

if typing.TYPE_CHECKING:
    from bores.blackoil.pvt.regions import PVT

__all__ = [
    "AquiferConnections",
    "compute_default_aquifer_pressure",
    "load_boundary_conditions",
    "load_flux_aquifer_from_record",
    "resolve_aquancon_connections",
    "resolve_aquancon_face_positions",
    "resolve_aquancon_region",
]


def load_flux_aquifer_from_record(
    record: typing.Mapping[str, typing.Any], unit_system: UnitSystem
) -> ConstantFluxBoundary:
    """
    Build a `ConstantFluxBoundary` from one `AQUFLUX` record.

    `AQUFLUX` gives a fixed influx rate directly.

    :param record: One parsed `AQUFLUX` record.
    :param unit_system: The deck's unit system.
    :returns: `ConstantFluxBoundary` with `flux=record["flux"]`.
    """
    return ConstantFluxBoundary(flux=record["flux"], unit_system=unit_system)


class AquiferConnections(typing.NamedTuple):
    """The boundary faces one aquifer is attached to by `AQUANCON`, with their weights."""

    face_positions: IntArray[OneDimension]
    """Sorted boundary face positions (into `Grid.boundary_face_indices`)."""

    influx_coefficients: NumberArray[OneDimension]
    """
    Each face's influx coefficient, in the same order as `face_positions`. The
    aquifer's influx is shared between its faces in proportion to these.
    """


def resolve_aquancon_connections(
    deck_file: DeckFile,
    grid: Grid,
    aquifer_id: int,
    *,
    claimed_faces: typing.Mapping[int, int] | None = None,
) -> AquiferConnections | None:
    """
    Resolve the boundary faces one aquifer is attached to by `AQUANCON`, and how much
    of the aquifer's influx each face takes.

    Every `AQUANCON` record with a matching `aquifer_id` is resolved via
    `resolve_boundary_faces_for_box`. See that function for what gets skipped
    (out-of-bounds or inactive cells, and cells whose face in the given direction isn't
    a genuine grid-boundary face).

    A face's influx coefficient is the record's `influx_coefficient` when given, or the
    face's own area when it is defaulted, multiplied by the record's
    `connection_multiplier`. A face attached by more than one record of the same aquifer
    adds the coefficients up.

    A record with `allow_already_connected` set to `NO` leaves out every face that
    `claimed_faces` says another aquifer already holds. A record set to `YES` keeps
    them, but a face can only be held by one aquifer in the compiled model, so the
    aquifer that is loaded last keeps it.

    :param deck_file: Parsed deck, read for its `AQUANCON` records.
    :param grid: Grid to resolve face positions against.
    :param aquifer_id: Aquifer id to collect `AQUANCON` records for.
    :param claimed_faces: Face position to the id of the aquifer that already holds it.
    :returns: The aquifer's connections, or `None` if `AQUANCON` has no records for
        `aquifer_id` or none of them resolved to a face.
    :raises ValidationError: If a record gives a negative influx coefficient or
        connection multiplier.
    """
    records = deck_file.get("AQUANCON") or []
    matching = [record for record in records if record["aquifer_id"] == aquifer_id]
    if not matching:
        return None

    coefficients: dict[int, float] = {}
    for record in matching:
        explicit = record["influx_coefficient"]
        multiplier = record["connection_multiplier"]
        if (explicit is not None and explicit < 0.0) or multiplier < 0.0:
            raise ValidationError(
                f"`AQUANCON` aquifer {aquifer_id!r}: `influx_coefficient` and "
                "`connection_multiplier` can not be negative."
            )

        box_positions = resolve_boundary_faces_for_box(
            grid,
            i1=record["i1"],
            i2=record["i2"],
            j1=record["j1"],
            j2=record["j2"],
            k1=record["k1"],
            k2=record["k2"],
            face_direction=record["face"],
            label=f"AQUANCON aquifer {aquifer_id}",
        )
        positions = [int(position) for position in box_positions]

        held_elsewhere = [
            position
            for position in positions
            if claimed_faces is not None and claimed_faces.get(position, aquifer_id) != aquifer_id
        ]
        if held_elsewhere and record["allow_already_connected"] == "NO":
            warnings.warn(
                f"`AQUANCON` aquifer {aquifer_id!r}: {len(held_elsewhere)} face(s) are already "
                "connected to another aquifer and `allow_already_connected` is `NO`. "
                "Leaving them out.",
                stacklevel=2,
            )
            skipped = set(held_elsewhere)
            positions = [position for position in positions if position not in skipped]
        elif held_elsewhere:
            warnings.warn(
                f"`AQUANCON` aquifer {aquifer_id!r}: {len(held_elsewhere)} face(s) are also "
                "connected to another aquifer. A face can only be held by one aquifer, so "
                "the one loaded last keeps it.",
                stacklevel=2,
            )

        if not positions:
            continue
        areas = grid.face_areas[grid.boundary_face_indices[positions]]
        for position, area in zip(positions, areas.tolist(), strict=True):
            base = explicit if explicit is not None else area
            coefficients[position] = coefficients.get(position, 0.0) + base * multiplier

    if not coefficients:
        warnings.warn(
            f"`AQUANCON` aquifer {aquifer_id!r}: every record resolved to no "
            "boundary faces. Not attaching this aquifer.",
            stacklevel=2,
        )
        return None

    ordered = sorted(coefficients)
    return AquiferConnections(
        face_positions=typing.cast(IntArray[OneDimension], np.asarray(ordered, dtype=np.int32)),
        influx_coefficients=typing.cast(
            NumberArray[OneDimension],
            np.asarray([coefficients[position] for position in ordered], dtype=np.float64),
        ),
    )


def resolve_aquancon_face_positions(
    deck_file: DeckFile,
    grid: Grid,
    aquifer_id: int,
) -> IntArray[OneDimension] | None:
    """
    Resolve the boundary face positions one aquifer is attached to by `AQUANCON`.

    :param deck_file: Parsed deck, read for its `AQUANCON` records.
    :param grid: Grid to resolve face positions against.
    :param aquifer_id: Aquifer id to collect `AQUANCON` records for.
    :returns: Sorted boundary face positions (into `Grid.boundary_face_indices`), or
        `None` if `AQUANCON` has no records for `aquifer_id` or none of them resolved
        to a face.
    """
    connections = resolve_aquancon_connections(deck_file, grid, aquifer_id)
    return connections.face_positions if connections is not None else None


def resolve_aquancon_region(
    deck_file: DeckFile,
    grid: Grid,
    aquifer_id: int,
    condition: BoundaryCondition,
    *,
    region_name: str | None = None,
    connections: AquiferConnections | None = None,
) -> BoundaryRegion | None:
    """
    Build a `BoundaryRegion` for one aquifer from its `AQUANCON` records.

    :param deck_file: Parsed deck, read for its `AQUANCON` records.
    :param grid: Grid to resolve face positions against.
    :param aquifer_id: Aquifer id to collect `AQUANCON` records for.
    :param condition: The `BoundaryCondition` (aquifer or flux) to attach.
    :param region_name: `BoundaryRegion.name`. Defaults to `f"aquifer_{aquifer_id}"`.
    :param connections: Connections already resolved by `resolve_aquancon_connections`,
        to avoid resolving them again.
    :returns: A `BoundaryRegion`, or `None` if `AQUANCON` has no records
        for `aquifer_id`, or every one of them resolved to no faces.
    """
    if connections is None:
        connections = resolve_aquancon_connections(deck_file, grid, aquifer_id)
    if connections is None:
        return None
    label = region_name or f"aquifer_{aquifer_id}"
    return BoundaryRegion(
        name=label,
        face_positions=connections.face_positions,
        condition=condition,
        face_weights=connections.influx_coefficients,
    )


def compute_default_aquifer_pressure(
    grid: Grid,
    *,
    face_positions: IntArray[OneDimension],
    reservoir_pressure: NumberArray[OneDimension],
    datum_depth: Number,
    water_gradient: Number,
) -> Number:
    """
    Initial aquifer pressure, at the aquifer's datum depth, that is in equilibrium with
    the reservoir cells the aquifer is attached to.

    Each attached cell's pressure is moved to the datum depth along the water pressure
    gradient and the results are averaged.

    :param grid: The model's `Grid`.
    :param face_positions: Boundary face positions the aquifer is attached to.
    :param reservoir_pressure: Initial pressure of every cell, in the grid's unit system.
    :param datum_depth: Aquifer datum depth (positive down).
    :param water_gradient: Water pressure gradient, pressure per unit depth.
    :returns: Aquifer pressure at `datum_depth`.
    :raises ValidationError: If no attached cell is found.
    """
    face_indices = grid.boundary_face_indices[face_positions]
    cell_indices = grid.face_cell_indices[face_indices, 0]
    cell_indices = np.unique(cell_indices[cell_indices >= 0])
    if len(cell_indices) == 0:
        raise ValidationError("The aquifer is not attached to any cell.")

    depths = grid.cell_center_depths[cell_indices]
    pressures = reservoir_pressure[cell_indices] + water_gradient * (datum_depth - depths)
    return np.mean(pressures)


def load_boundary_conditions(
    deck_file: DeckFile,
    grid: Grid,
    *,
    pvt: "PVT | None" = None,
    reservoir_pressure: NumberArray[OneDimension] | None = None,
    extra_regions: typing.Sequence[BoundaryRegion] | None = None,
) -> BoundaryConditions | None:
    """
    Build a `BoundaryConditions` from every boundary condition a deck defines.

    Covers every deck-sourced boundary condition this codebase knows how
    to load: `AQUCT` and `AQUFETP` analytic aquifers, and `AQUFLUX`
    flux-specified aquifers, each attached to grid faces via `AQUANCON`.

    There is no real Eclipse keyword for a plain constant-pressure
    boundary (`ConstantPressureBoundary`) outside of the aquifer models
    already covered here, so this function can't populate one from a
    deck. Pass one in via `extra_regions` if the model needs it.

    :param deck_file: Parsed deck.
    :param grid: The model's `Grid`, read for `AQUANCON`'s face resolution.
    :param pvt: The model's PVT tables. Required if the deck has an
        `AQUCT` keyword. See `load_carter_tracy_aquifer` for why.
    :param reservoir_pressure: Initial pressure of every cell, in the deck's unit system.
        Needed when an `AQUCT` record defaults its initial pressure (`1*`), which is then
        set to the water-gradient-corrected average pressure of the cells the aquifer is
        attached to, at the aquifer's datum depth.
    :param extra_regions: Additional `BoundaryRegion`s to include as-is
        (e.g. a manually-built `ConstantPressureBoundary` region). Appended
        after every deck-sourced region, so they win on overlapping faces.
    :returns: `BoundaryConditions` covering every attached aquifer/flux
        region and `extra_regions`, or `None` if the deck defines no
        boundary conditions and `extra_regions` is empty.
    :raises ValidationError: If the deck has an `AQUCT` keyword but `pvt`
        is `None`, if an `AQUCT` record defaults its initial pressure but
        `reservoir_pressure` is `None`, or if the same `aquifer_id` is defined by more
        than one of `AQUCT`/`AQUFETP`/`AQUFLUX`.
    """
    unit_system = deck_file.unit_system
    regions: list[BoundaryRegion] = []
    resolved_connections: dict[int, AquiferConnections | None] = {}
    claimed_faces: dict[int, int] = {}

    def get_aquancon_connections(aquifer_id: int) -> AquiferConnections | None:
        if aquifer_id not in resolved_connections:
            connections = resolve_aquancon_connections(
                deck_file, grid, aquifer_id, claimed_faces=claimed_faces
            )
            resolved_connections[aquifer_id] = connections
            if connections is not None:
                claimed_faces.update(
                    dict.fromkeys(connections.face_positions.tolist(), aquifer_id)
                )
        return resolved_connections[aquifer_id]

    def get_initial_pressure(record: typing.Mapping[str, typing.Any]) -> Number:
        aquifer_id = record["aquifer_id"]
        if reservoir_pressure is None or pvt is None:
            raise ValidationError(
                f"`AQUCT` aquifer {aquifer_id!r} defaults its initial pressure (`1*`), "
                "which is taken from the initial reservoir pressure. Pass "
                "`reservoir_pressure` to `load_boundary_conditions`."
            )

        connections = get_aquancon_connections(aquifer_id)
        if connections is None:
            raise ValidationError(
                f"`AQUCT` aquifer {aquifer_id!r} defaults its initial pressure (`1*`) but "
                "no `AQUANCON` record attaches it to any cell to take it from."
            )

        static = pvt.region(record["pvt_table_number"]).static
        water_density = static.stock_tank_water_density
        if water_density is None:
            raise ValidationError(
                f"`AQUCT` aquifer {aquifer_id!r} defaults its initial pressure (`1*`) but the "
                "deck gives no water density (`DENSITY`) to correct it to the datum depth."
            )
        if static.water_reference_fvf:
            water_density = water_density / static.water_reference_fvf

        return compute_default_aquifer_pressure(
            grid,
            face_positions=connections.face_positions,
            reservoir_pressure=reservoir_pressure,
            datum_depth=record["datum_depth"],
            water_gradient=water_density * get_hydrostatic_gradient_factor(unit_system),
        )

    defined_ids: dict[int, str] = {}

    def check_id(aquifer_id: int, source: str) -> None:
        if aquifer_id in defined_ids:
            raise ValidationError(
                f"Aquifer id {aquifer_id!r} is defined by both "
                f"{defined_ids[aquifer_id]!r} and {source!r}. Each aquifer id "
                "must come from exactly one of `AQUCT`/`AQUFETP`/`AQUFLUX`."
            )
        defined_ids[aquifer_id] = source

    if deck_file.get("AQUCT"):
        if pvt is None:
            raise ValidationError(
                "Deck defines an `AQUCT` keyword but no `pvt` was given to "
                "resolve water viscosity from. See `load_carter_tracy_aquifer`."
            )
        for record in deck_file.get("AQUCT") or []:
            get_aquancon_connections(record["aquifer_id"])
        carter_tracy_aquifers = CarterTracyAquifer.from_deck(
            deck_file, pvt=pvt, get_initial_pressure=get_initial_pressure
        )
        for aquifer_id, aquifer in carter_tracy_aquifers.items():
            check_id(aquifer_id, "AQUCT")
            region = resolve_aquancon_region(
                deck_file,
                grid,
                aquifer_id,
                aquifer,
                connections=get_aquancon_connections(aquifer_id),
            )
            if region is not None:
                regions.append(region)

    if deck_file.get("AQUFETP"):
        fetkovich_aquifers = FetkovichAquifer.from_deck(deck_file)
        for aquifer_id in fetkovich_aquifers:
            get_aquancon_connections(aquifer_id)
        for aquifer_id, aquifer in fetkovich_aquifers.items():
            check_id(aquifer_id, "AQUFETP")
            region = resolve_aquancon_region(
                deck_file,
                grid,
                aquifer_id,
                aquifer,
                connections=get_aquancon_connections(aquifer_id),
            )
            if region is not None:
                regions.append(region)

    if deck_file.get("AQUFLUX"):
        for record in deck_file.get("AQUFLUX") or []:
            aquifer_id = record["aquifer_id"]
            check_id(aquifer_id, "AQUFLUX")
            condition = load_flux_aquifer_from_record(record, unit_system)
            region = resolve_aquancon_region(
                deck_file,
                grid,
                aquifer_id,
                condition,
                connections=get_aquancon_connections(aquifer_id),
            )
            if region is not None:
                regions.append(region)

    dangling = {
        record["aquifer_id"]
        for record in (deck_file.get("AQUANCON") or [])
        if record["aquifer_id"] not in defined_ids
    }
    if dangling:
        warnings.warn(
            f"`AQUANCON` references aquifer id(s) {sorted(dangling)}, but none "
            "of them are defined by an `AQUCT`, `AQUFETP`, or `AQUFLUX` keyword. "
            "These `AQUANCON` records are ignored.",
            stacklevel=2,
        )

    if extra_regions:
        regions.extend(extra_regions)

    if not regions:
        return None
    return BoundaryConditions(regions=regions, unit_system=unit_system)

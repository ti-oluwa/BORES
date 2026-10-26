"""Load `BoundaryConditions` from a parsed deck."""

import typing
import warnings

import numpy as np

from bores.constants import c
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.grids.utils import resolve_boundary_faces_for_box
from bores.reservoir.boundary.aquifers.carter_tracy import CarterTracyAquifer
from bores.reservoir.boundary.aquifers.fetkovich import FetkovichAquifer
from bores.reservoir.boundary.base import BoundaryCondition, FluxBasis
from bores.reservoir.boundary.conditions import BoundaryConditions, BoundaryRegion
from bores.reservoir.boundary.types import ConstantFluxBoundary
from bores.types import IntArray, Integer, Number, NumberArray, OneDimension, UnitSystem
from bores.units import get_conversion_factors
from bores.utils import get_hydrostatic_gradient_factor

if typing.TYPE_CHECKING:
    from bores.blackoil.pvt.regions import PVT

__all__ = [
    "AquiferConnections",
    "compute_default_aquifer_pressure",
    "load_boundary_conditions",
    "load_flux_aquifer",
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

    `AQUFLUX` gives the influx per unit of face area, as liquid surface volume per time.
    The returned boundary has a `PER_AREA` basis, so each face passes that rate times its
    own area. `load_boundary_conditions` scales it to each connection's `AQUANCON` influx
    coefficient (its area, unless given) through the `face_weights` of the region it builds.

    :param record: One parsed `AQUFLUX` record.
    :param unit_system: The deck's unit system.
    :returns: `ConstantFluxBoundary` with a `PER_AREA` basis whose `flux` is the record's
        flux, with stock-tank barrels turned into cubic feet in the field unit system.
    """
    flux = record["flux"]
    if unit_system is UnitSystem.FIELD:
        flux = flux * c.STB_TO_CUBIC_FEET
    return ConstantFluxBoundary(flux=flux, basis=FluxBasis.PER_AREA, unit_system=unit_system)


@typing.overload
def load_flux_aquifer(deck_file: DeckFile, *, aquifer_id: Integer) -> ConstantFluxBoundary: ...
@typing.overload
def load_flux_aquifer(
    deck_file: DeckFile, *, aquifer_id: None = None
) -> dict[int, ConstantFluxBoundary]: ...


def load_flux_aquifer(
    deck_file: DeckFile, *, aquifer_id: Integer | None = None
) -> ConstantFluxBoundary | dict[int, ConstantFluxBoundary]:
    """
    Load one or all `AQUFLUX` aquifers of a deck as `ConstantFluxBoundary`s.

    :param deck_file: Parsed `bores.deck.file.DeckFile`.
    :param aquifer_id: Id of a specific aquifer to load, or `None` for all.
    :returns: A single `ConstantFluxBoundary` if `aquifer_id` is given; otherwise a
        `dict[int, ConstantFluxBoundary]` keyed by aquifer id.
    :raises ValidationError: If the deck has no `AQUFLUX` keyword, or `aquifer_id` is
        given but not found in it.
    """
    records = deck_file.get("AQUFLUX")
    if not records:
        raise ValidationError("No `AQUFLUX` keyword found in the provided deck.")

    if aquifer_id is not None:
        matching = [record for record in records if record["aquifer_id"] == aquifer_id]
        if not matching:
            available = sorted(record["aquifer_id"] for record in records)
            raise ValidationError(
                f"Aquifer {aquifer_id!r} not found in `AQUFLUX`. Available: {available}."
            )
        return load_flux_aquifer_from_record(matching[0], deck_file.unit_system)

    return {
        record["aquifer_id"]: load_flux_aquifer_from_record(record, deck_file.unit_system)
        for record in records
    }


class AquiferConnections(typing.NamedTuple):
    """The boundary faces one aquifer is attached to by `AQUANCON`, with their weights."""

    face_positions: IntArray[OneDimension]
    """Sorted boundary face positions (into `Grid.boundary_face_indices`)."""

    influx_coefficients: NumberArray[OneDimension]
    """
    Each face's influx coefficient, in the same order as `face_positions`. The
    influx of an `AQUCT` or `AQUFETP` aquifer is shared between its faces in proportion
    to these.
    """

    area_ratios: NumberArray[OneDimension]
    """
    Each face's influx coefficient divided by the face's own area, in the same order as
    `face_positions`. This is the connection multiplier when the coefficient is defaulted
    to the face's area, and otherwise the explicit coefficient replaces the area. An `AQUFLUX`
    aquifer's rate through a face is its flux times the face's area times this, which is its
    flux times the influx coefficient.
    """


def count_active_cells_in_box(grid: Grid, record: typing.Mapping[str, typing.Any]) -> Integer:
    """
    Number of active cells in the IJK box of an `AQUANCON` record.

    :param grid: Grid with structured dimensions and cell statuses. Without them nothing
        can be counted and `0` is returned.
    :param record: One parsed `AQUANCON` record, whose box is 1-based and inclusive.
    :returns: How many of the box's cells are active.
    """
    if grid.dimensions is None or grid.cell_statuses is None:
        return 0
    nx, ny, nz = grid.dimensions
    i = np.arange(max(record["i1"], 1) - 1, min(record["i2"], nx))
    j = np.arange(max(record["j1"], 1) - 1, min(record["j2"], ny))
    k = np.arange(max(record["k1"], 1) - 1, min(record["k2"], nz))
    cells = (i[:, None, None] + nx * (j[None, :, None] + ny * k[None, None, :])).ravel()
    return np.count_nonzero(grid.cell_statuses[cells] == 1)


def resolve_aquancon_connections(
    deck_file: DeckFile,
    grid: Grid,
    aquifer_id: int,
    *,
    claimed_cells: typing.Mapping[int, int] | None = None,
    length_factor: Number = 1.0,
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

    Only faces on the edge of the active grid can be connected, which is what
    `connect_adjoining_active_cell = NO` (the default) asks for. A record that sets it to
    `YES` and has cells in its box whose face adjoins another active cell is an error,
    since connecting those faces is not supported. A cell that `claimed_cells` says another
    aquifer already holds is also an error, whichever of its faces is connected.

    :param deck_file: Parsed deck, read for its `AQUANCON` records.
    :param grid: Grid to resolve face positions against.
    :param aquifer_id: Aquifer id to collect `AQUANCON` records for.
    :param claimed_cells: Cell index to the id of the aquifer that holds it.
    :param length_factor: Multiplies lengths measured on `grid` to give the deck's length
        unit, for a grid that is not in the deck's unit system. Face areas are scaled by
        its square before they are compared with a record's `influx_coefficient`.
    :returns: The aquifer's connections, or `None` if `AQUANCON` has no records for
        `aquifer_id` or none of them resolved to a face.
    :raises ValidationError: If a record gives a negative influx coefficient or
        connection multiplier, sets `connect_adjoining_active_cell` to `YES` where that
        would connect a face that adjoins an active cell, or connects a cell that another
        aquifer holds.
    """
    records = deck_file.get("AQUANCON") or []
    matching = [record for record in records if record["aquifer_id"] == aquifer_id]
    if not matching:
        return None

    coefficients: dict[int, float] = {}
    face_areas: dict[int, float] = {}
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

        if record["connect_adjoining_active_cell"] == "YES":
            n_adjoining = count_active_cells_in_box(grid, record) - len(positions)
            if n_adjoining > 0:
                raise ValidationError(
                    f"`AQUANCON` aquifer {aquifer_id!r}: `connect_adjoining_active_cell` is `YES` "
                    f"and {n_adjoining} face(s) in the box adjoin an active cell, but connecting "
                    "those faces is not supported. Only faces on the edge of the active grid can "
                    "be connected."
                )

        owner_cells = grid.face_cell_indices[grid.boundary_face_indices[positions], 0].tolist()
        holders = {
            cell: claimed_cells[cell]
            for cell in owner_cells
            if claimed_cells is not None and claimed_cells.get(cell, aquifer_id) != aquifer_id
        }
        if holders:
            raise ValidationError(
                f"`AQUANCON` aquifer {aquifer_id!r} connects {len(holders)} cell(s) that "
                f"aquifer {sorted(set(holders.values()))} is already connected to. A cell can "
                "only be connected to one aquifer."
            )

        if not positions:
            continue
        areas = grid.face_areas[grid.boundary_face_indices[positions]] * length_factor**2
        for position, area in zip(positions, areas.tolist(), strict=True):
            base = explicit if explicit is not None else area
            coefficients[position] = coefficients.get(position, 0.0) + base * multiplier
            face_areas[position] = area

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
        area_ratios=typing.cast(
            NumberArray[OneDimension],
            np.asarray(
                [coefficients[position] / face_areas[position] for position in ordered],
                dtype=np.float64,
            ),
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
    face_weights: NumberArray[OneDimension] | None = None,
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
    :param face_weights: The region's `face_weights`. Defaults to the connections' influx
        coefficients.
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
        face_weights=face_weights if face_weights is not None else connections.influx_coefficients,
    )


def compute_default_aquifer_pressure(
    grid: Grid,
    *,
    face_positions: IntArray[OneDimension],
    reservoir_pressure: NumberArray[OneDimension],
    datum_depth: Number,
    water_gradient: Number,
    length_factor: Number = 1.0,
    weights: NumberArray[OneDimension] | None = None,
) -> Number:
    """
    Initial aquifer pressure, at the aquifer's datum depth, that is in equilibrium with
    the reservoir cells the aquifer is attached to.

    Each attached cell's water-phase pressure is moved to the datum depth along the water
    pressure gradient and the results are averaged, weighted by how much of the aquifer's
    influx each connection takes.

    :param grid: The model's `Grid`.
    :param face_positions: Boundary face positions the aquifer is attached to.
    :param reservoir_pressure: Initial water-phase pressure of every cell, in the deck's
        unit system.
    :param datum_depth: Aquifer datum depth (positive down).
    :param water_gradient: Water pressure gradient, pressure per unit depth.
    :param length_factor: Multiplies depths measured on `grid` to give the units of
        `datum_depth`, for a grid that is not in the same unit system.
    :param weights: One weight for each of `face_positions`, usually the connections'
        influx coefficients. `None` weights every connection equally.
    :returns: Aquifer pressure at `datum_depth`.
    :raises ValidationError: If no attached cell is found.
    """
    face_indices = grid.boundary_face_indices[face_positions]
    cell_indices = grid.face_cell_indices[face_indices, 0]
    attached = cell_indices >= 0
    if not attached.any():
        raise ValidationError("The aquifer is not attached to any cell.")

    cell_indices = cell_indices[attached]
    connection_weights = (
        np.ones(cell_indices.shape[0])
        if weights is None
        else np.asarray(weights, dtype=np.float64)[attached]
    )
    if not connection_weights.sum() > 0.0:
        raise ValidationError("The aquifer's connection weights add up to zero.")

    depths = grid.cell_center_depths[cell_indices] * length_factor
    pressures = reservoir_pressure[cell_indices] + water_gradient * (datum_depth - depths)
    return np.average(pressures, weights=connection_weights)


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
    :param grid: The model's `Grid`, read for `AQUANCON`'s face resolution and for the
        areas and depths of the faces and cells aquifers attach to. It may be in any unit
        system, as these are converted to the deck's.
    :param pvt: The model's PVT tables. Required if the deck has an
        `AQUCT` keyword, and if an `AQUFETP` record defaults its initial pressure. See
        `load_carter_tracy_aquifer_from_record` for why.
    :param reservoir_pressure: Initial water-phase pressure of every cell, in the deck's
        unit system (see `bores.initialization.compute_water_phase_pressure`). Needed when
        an `AQUCT` or `AQUFETP` record defaults its initial pressure (`1*`), which is then
        set to the water-gradient-corrected average pressure of the cells the aquifer is
        attached to, at the aquifer's datum depth, weighted by the connections' influx
        coefficients.
    :param extra_regions: Additional `BoundaryRegion`s to include as-is
        (e.g. a manually-built `ConstantPressureBoundary` region). Appended
        after every deck-sourced region, so they win on overlapping faces.
    :returns: `BoundaryConditions` covering every attached aquifer/flux
        region and `extra_regions`, or `None` if the deck defines no
        boundary conditions and `extra_regions` is empty.
    :raises ValidationError: If the deck has an `AQUCT` keyword but `pvt`
        is `None`, if an `AQUCT` or `AQUFETP` record defaults its initial pressure but
        `reservoir_pressure` is `None`, or if the same `aquifer_id` is defined by more
        than one of `AQUCT`/`AQUFETP`/`AQUFLUX`.
    """
    unit_system = deck_file.unit_system
    length_factor = get_conversion_factors(grid.unit_system, unit_system)["length"]
    regions: list[BoundaryRegion] = []
    resolved_connections: dict[int, AquiferConnections | None] = {}
    claimed_cells: dict[int, int] = {}

    def get_aquancon_connections(aquifer_id: int) -> AquiferConnections | None:
        if aquifer_id not in resolved_connections:
            connections = resolve_aquancon_connections(
                deck_file,
                grid,
                aquifer_id,
                claimed_cells=claimed_cells,
                length_factor=length_factor,
            )
            resolved_connections[aquifer_id] = connections
            if connections is not None:
                owner_cells = grid.face_cell_indices[
                    grid.boundary_face_indices[connections.face_positions], 0
                ]
                claimed_cells.update(dict.fromkeys(owner_cells.tolist(), aquifer_id))
        return resolved_connections[aquifer_id]

    def get_initial_pressure(record: typing.Mapping[str, typing.Any]) -> Number:
        aquifer_id = record["aquifer_id"]
        if reservoir_pressure is None or pvt is None:
            raise ValidationError(
                f"Aquifer {aquifer_id!r} defaults its initial pressure (`1*`), "
                "which is taken from the initial reservoir pressure. Pass "
                "`reservoir_pressure` to `load_boundary_conditions`."
            )

        connections = get_aquancon_connections(aquifer_id)
        if connections is None:
            raise ValidationError(
                f"Aquifer {aquifer_id!r} defaults its initial pressure (`1*`) but "
                "no `AQUANCON` record attaches it to any cell to take it from."
            )

        static = pvt.region(record["pvt_table_number"]).static
        stock_tank_water_density = static.stock_tank_water_density
        if stock_tank_water_density is None:
            raise ValidationError(
                f"Aquifer {aquifer_id!r} defaults its initial pressure (`1*`) but the "
                "deck gives no water density (`DENSITY`) to correct it to the datum depth."
            )
        water_density = float(stock_tank_water_density)
        if static.water_reference_fvf:
            water_density /= float(static.water_reference_fvf)

        return compute_default_aquifer_pressure(
            grid,
            face_positions=connections.face_positions,
            reservoir_pressure=reservoir_pressure,
            datum_depth=record["datum_depth"],
            water_gradient=water_density * get_hydrostatic_gradient_factor(unit_system),
            length_factor=length_factor,
            weights=connections.influx_coefficients,
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
        for record in deck_file.get("AQUFETP") or []:
            get_aquancon_connections(record["aquifer_id"])
        fetkovich_aquifers = FetkovichAquifer.from_deck(
            deck_file, get_initial_pressure=get_initial_pressure
        )
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
        flux_aquifers = load_flux_aquifer(deck_file)
        for aquifer_id in flux_aquifers:
            get_aquancon_connections(aquifer_id)
        for aquifer_id, condition in flux_aquifers.items():
            check_id(aquifer_id, "AQUFLUX")
            flux_connections = get_aquancon_connections(aquifer_id)
            region = resolve_aquancon_region(
                deck_file,
                grid,
                aquifer_id,
                condition,
                connections=flux_connections,
                face_weights=(
                    flux_connections.area_ratios if flux_connections is not None else None
                ),
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

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
from bores.types import IntArray, OneDimension, UnitSystem

if typing.TYPE_CHECKING:
    from bores.blackoil.pvt.regions import PVT

__all__ = [
    "load_boundary_conditions",
    "load_flux_aquifer",
    "resolve_aquancon_region",
]


def load_flux_aquifer(
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


def resolve_aquancon_region(
    deck_file: DeckFile,
    grid: Grid,
    aquifer_id: int,
    condition: BoundaryCondition,
    *,
    region_name: str | None = None,
) -> BoundaryRegion | None:
    """
    Build a `BoundaryRegion` for one aquifer from its `AQUANCON` records.

    Every `AQUANCON` record with a matching `aquifer_id` is resolved via
    `resolve_boundary_faces_for_box` and unioned into one region. See that
    function for what gets skipped (out-of-bounds or inactive cells, and
    cells whose face in the given direction isn't a genuine grid-boundary
    face).

    `allow_already_connected="YES"` on a record is not enforced - this
    function doesn't track which faces other aquifers have already
    claimed, so a `YES` record is treated the same as the `NO` default,
    with a warning.

    :param deck_file: Parsed deck, read for its `AQUANCON` records.
    :param grid: Grid to resolve face positions against.
    :param aquifer_id: Aquifer id to collect `AQUANCON` records for.
    :param condition: The `BoundaryCondition` (aquifer or flux) to attach.
    :param region_name: `BoundaryRegion.name`. Defaults to `f"aquifer_{aquifer_id}"`.
    :returns: A `BoundaryRegion`, or `None` if `AQUANCON` has no records
        for `aquifer_id`, or every one of them resolved to no faces.
    """
    records = deck_file.get("AQUANCON") or []
    matching = [record for record in records if record["aquifer_id"] == aquifer_id]
    if not matching:
        return None

    label = region_name or f"aquifer_{aquifer_id}"
    positions: set[int] = set()
    for record in matching:
        if record["allow_already_connected"] == "YES":
            warnings.warn(
                f"`AQUANCON` aquifer {aquifer_id!r}: `allow_already_connected=YES` "
                "is not enforced. Connections aren't checked against other "
                "aquifers' claimed faces.",
                stacklevel=2,
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
        positions.update(int(p) for p in box_positions)

    if not positions:
        warnings.warn(
            f"`AQUANCON` aquifer {aquifer_id!r}: every record resolved to no "
            "boundary faces. Not attaching this aquifer.",
            stacklevel=2,
        )
        return None

    face_positions = typing.cast(
        IntArray[OneDimension], np.asarray(sorted(positions), dtype=np.int32)
    )
    return BoundaryRegion(name=label, face_positions=face_positions, condition=condition)


def load_boundary_conditions(
    deck_file: DeckFile,
    grid: Grid,
    *,
    pvt: "PVT | None" = None,
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
        `AQUCT` keyword - see `load_carter_tracy_aquifer` for why.
    :param extra_regions: Additional `BoundaryRegion`s to include as-is
        (e.g. a manually-built `ConstantPressureBoundary` region). Appended
        after every deck-sourced region, so they win on overlapping faces.
    :returns: `BoundaryConditions` covering every attached aquifer/flux
        region and `extra_regions`, or `None` if the deck defines no
        boundary conditions and `extra_regions` is empty.
    :raises ValidationError: If the deck has an `AQUCT` keyword but `pvt`
        is `None`, or if the same `aquifer_id` is defined by more than one
        of `AQUCT`/`AQUFETP`/`AQUFLUX`.
    """
    unit_system = deck_file.unit_system
    regions: list[BoundaryRegion] = []

    defined_ids: dict[int, str] = {}

    def check_id(aquifer_id: int, source: str) -> None:
        if aquifer_id in defined_ids:
            raise ValidationError(
                f"Aquifer id {aquifer_id!r} is defined by both "
                f"{defined_ids[aquifer_id]!r} and {source!r}. Each aquifer id "
                "must come from exactly one of AQUCT/AQUFETP/AQUFLUX."
            )
        defined_ids[aquifer_id] = source

    if deck_file.get("AQUCT"):
        if pvt is None:
            raise ValidationError(
                "Deck defines an AQUCT keyword but no `pvt` was given to "
                "resolve water viscosity from. See `load_carter_tracy_aquifer`."
            )
        for aquifer_id, aquifer in CarterTracyAquifer.from_deck(deck_file, pvt=pvt).items():
            check_id(aquifer_id, "AQUCT")
            region = resolve_aquancon_region(deck_file, grid, aquifer_id, aquifer)
            if region is not None:
                regions.append(region)

    if deck_file.get("AQUFETP"):
        for aquifer_id, aquifer in FetkovichAquifer.from_deck(deck_file).items():
            check_id(aquifer_id, "AQUFETP")
            region = resolve_aquancon_region(deck_file, grid, aquifer_id, aquifer)
            if region is not None:
                regions.append(region)

    if deck_file.get("AQUFLUX"):
        for record in deck_file.get("AQUFLUX") or []:
            aquifer_id = record["aquifer_id"]
            check_id(aquifer_id, "AQUFLUX")
            condition = load_flux_aquifer(record, unit_system)
            region = resolve_aquancon_region(deck_file, grid, aquifer_id, condition)
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

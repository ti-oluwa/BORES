import typing

import attrs
import numba
import numpy as np
from typing_extensions import Self

from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.types import (
    Boolean,
    BooleanCellArray,
    IntArray,
    IntCellArray,
    Integer,
    OneDimension,
)

__all__ = ["RegionGroups", "Regions", "group_cells_by_region"]


def load_region_array(
    deck_file: DeckFile,
    keyword: str,
    n_cells: int,
    default_region: int | None = None,
    *,
    active: BooleanCellArray | None = None,
) -> IntCellArray | None:
    """
    Load a region number array (`SATNUM`, `EQLNUM`, ...) from a deck.

    :param deck_file: Parsed deck.
    :param keyword: Region keyword, e.g. `"EQLNUM"`.
    :param n_cells: Number of cells the array must have (every cell of the grid, active or not).
    :param default_region: Region of every cell when the deck does not have the keyword. `None`
        returns `None` in that case.
    :param active: Which cells are active. Eclipse ignores the region number of an inactive cell,
        and decks routinely write `0` there (Norne's `EQLNUM` and `FIPNUM` do), whereas every cell
        here needs a region of at least 1. When given, an inactive cell with a number below 1 gets
        region 1. Active cells are never changed, so a bad number in one is still reported.
    :returns: Region numbers as `int32`, or `None`.
    :raises ValidationError: If the array does not have `n_cells` values.
    """
    array = deck_file.get(keyword)
    if array is None:
        if default_region is not None:
            return np.full(n_cells, default_region, dtype=np.int32)
        return None

    array = array.astype(np.int32, copy=False)
    if array.size != n_cells:
        raise ValidationError(f"{keyword} has {array.size} values; expected {n_cells}.")

    if active is not None:
        if active.shape != (n_cells,):
            raise ValidationError(
                f"`active` has shape {active.shape}; expected ({n_cells},) to match {keyword}."
            )
        unused = ~active & (array < 1)
        if unused.any():
            array = np.where(unused, 1, array).astype(np.int32)
    return typing.cast(IntCellArray, array)


class RegionGroups(typing.NamedTuple):
    """
    Cells grouped by the pair of regions that decides which saturation function table a cell
    uses. Region numbers never change during a run, so the grouping is built once and then reused
    on every timestep instead of being rediscovered from the region arrays each time.
    """

    saturation_regions: tuple[int, ...]
    """Saturation region number (`SATNUM`) of each group."""

    imbibition_regions: tuple[int, ...]
    """Imbibition region number (`IMBNUM`) of each group, `0` when no imbibition regions are used."""

    cell_indices: tuple[IntCellArray | None, ...]
    """
    Ascending cell indices of each group. `None` means the group holds every cell, which is the
    case whenever there is a single group, so callers can use whole-array views and skip the
    gather and scatter.
    """

    def __len__(self) -> int:
        return len(self.saturation_regions)


@numba.njit(cache=True)
def sort_cells_by_region(
    saturation_region: IntCellArray,
    imbibition_region: IntCellArray,
    use_imbibition_region: Boolean,
    stride: Integer,
    offsets: IntArray[OneDimension],
    order: IntCellArray,
) -> None:
    """
    Counting sort of the cells by `saturation_region * stride + imbibition_region`.

    The sort is stable, so the cells of each group stay in ascending order. `offsets` has one entry more
    than there are keys, and must be zero on entry. On exit the cells of key `k` are
    `order[offsets[k] : offsets[k + 1]]`.
    """
    size = saturation_region.size
    for cell in range(size):
        key = saturation_region[cell] * stride
        if use_imbibition_region:
            key += imbibition_region[cell]
        offsets[key + 1] += 1
    for key in range(offsets.size - 1):
        offsets[key + 1] += offsets[key]
    # `cursor` walks each key's slot, leaving `offsets` untouched for the caller.
    cursor = offsets[:-1].copy()
    for cell in range(size):
        key = saturation_region[cell] * stride
        if use_imbibition_region:
            key += imbibition_region[cell]
        order[cursor[key]] = cell
        cursor[key] += 1


def group_cells_by_region(
    *,
    saturation_region: IntCellArray,
    imbibition_region: IntCellArray | None = None,
) -> RegionGroups:
    """
    Group cells by `(SATNUM, IMBNUM)`, once, so that later timesteps can evaluate each group's
    tables without searching the region arrays again.

    Groups are ordered by saturation region and then imbibition region, and the cells in each
    group keep their original order.

    :param saturation_region: Saturation region number of every cell.
    :param imbibition_region: Imbibition region number of every cell. Cells are grouped by
        saturation region alone when this is not given.
    :returns: The cell groups.
    :raises ValidationError: If the two arrays differ in size, or a region number is negative.
    """
    size = saturation_region.size
    use_imbibition_region = imbibition_region is not None
    if imbibition_region is None:
        imbibition_region = typing.cast(IntCellArray, saturation_region[:0])
    elif imbibition_region.size != size:
        raise ValidationError(
            f"`imbibition_region` has {imbibition_region.size} values; "
            f"`saturation_region` has {size}."
        )
    if size == 0:
        raise ValidationError("`saturation_region` is empty, so there are no cells to group.")
    if saturation_region.min() < 0 or (use_imbibition_region and imbibition_region.min() < 0):
        raise ValidationError("Region numbers must not be negative.")

    stride = int(imbibition_region.max()) + 1 if use_imbibition_region else 1
    key_count = (int(saturation_region.max()) + 1) * stride
    offsets = np.zeros(key_count + 1, dtype=np.int64)
    order = np.empty(size, dtype=np.int64)
    sort_cells_by_region(
        saturation_region=saturation_region,
        imbibition_region=imbibition_region,
        use_imbibition_region=use_imbibition_region,
        stride=stride,
        offsets=offsets,
        order=order,
    )

    saturation_regions: list[int] = []
    imbibition_regions: list[int] = []
    cell_indices: list[IntCellArray | None] = []
    occupied = np.flatnonzero(offsets[1:] != offsets[:-1])
    for key in occupied.tolist():
        saturation_regions.append(key // stride)
        imbibition_regions.append(key % stride if use_imbibition_region else 0)
        cell_indices.append(typing.cast(IntCellArray, order[offsets[key] : offsets[key + 1]]))

    if len(occupied) == 1:
        cell_indices = [None]
    return RegionGroups(
        saturation_regions=tuple(saturation_regions),
        imbibition_regions=tuple(imbibition_regions),
        cell_indices=tuple(cell_indices),
    )


@attrs.frozen(slots=True)
class Regions:
    """
    Reservoir per-cell region assignments metadata.

    Populated from the `REGIONS` section of an Eclipse deck, or supplied
    directly by the user. All region arrays are 1-based integer indices
    selecting which PVT, saturation-function, equilibration, or rock
    compaction table applies to each cell.

    All fields are optional, but an important behaviour to note is that when absent,
    region 1 is assumed for every cell in the `get_*_region` methods.
    """

    pvt_region: IntCellArray | None = None
    """
    Shape (n_cells,) - PVT region index per cell (1-based).
    Selects which PVTTables entry from PVT applies.
    Read from PVTNUM. Default: 1 everywhere.
    """

    saturation_region: IntCellArray | None = None
    """
    Shape (n_cells,) - saturation function region index (1-based).
    Selects SWOF/SGOF/SWFN/SGFN table. Read from SATNUM.
    """

    imbibition_region: IntCellArray | None = None
    """
    Shape (n_cells,) - imbibition saturation function region index (1-based).
    Used for hysteresis scanning curves. Read from IMBNUM.
    """

    equilibrium_region: IntCellArray | None = None
    """
    Shape (n_cells,) - equilibration region index (1-based).
    Selects which EQUIL record governs initialisation. Read from EQLNUM.
    """

    rock_region: IntCellArray | None = None
    """
    Shape (n_cells,) - rock compaction region index (1-based).
    Selects ROCK/ROCKTAB table. Read from ROCKNUM.
    """

    fluid_in_place_region: IntCellArray | None = None
    """
    Shape (n_cells,) - fluid-in-place reporting region (1-based).
    Controls which cells contribute to ROIP/RGIP/RWIP output groups.
    Read from FIPNUM.
    """

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        n_cells: int,
        use_default: bool = False,
        active: BooleanCellArray | None = None,
    ) -> Self:
        """
        Build `Regions` from a parsed DeckFile.

        Missing keywords default to None (region 1 is assumed by callers).

        :param deck_file: Parsed DeckFile.
        :param n_cells: Number of cells in the grid (active or not), for validation.
        :param use_default: If True, missing keywords are filled with region 1.
            Else, they are left as None. Default: False.
        :param active: Which cells of the grid are active. Pass it when the grid has inactive
            cells: decks write `0` as the region number of those (Eclipse ignores it), which
            would otherwise fail validation, so inactive cells with a number below 1 are given
            region 1. `None` leaves every number as written.
        :returns: `Regions` object loaded from ECLIPSE deck.
        """
        default_region = 1 if use_default else None
        return cls(
            pvt_region=load_region_array(
                deck_file, "PVTNUM", n_cells=n_cells, default_region=default_region, active=active
            ),
            saturation_region=load_region_array(
                deck_file, "SATNUM", n_cells=n_cells, default_region=default_region, active=active
            ),
            imbibition_region=load_region_array(
                deck_file, "IMBNUM", n_cells=n_cells, default_region=default_region, active=active
            ),
            equilibrium_region=load_region_array(
                deck_file, "EQLNUM", n_cells=n_cells, default_region=default_region, active=active
            ),
            rock_region=load_region_array(
                deck_file, "ROCKNUM", n_cells=n_cells, default_region=default_region, active=active
            ),
            fluid_in_place_region=load_region_array(
                deck_file, "FIPNUM", n_cells=n_cells, default_region=default_region, active=active
            ),
        )

    def group_saturation_cells(self, *, use_imbibition_region: bool = False) -> RegionGroups:
        """
        Group the cells by the regions that decide their saturation function tables.

        The regions do not change during a run, so call this once and reuse the result.

        :param use_imbibition_region: Group by `(SATNUM, IMBNUM)` rather than by `SATNUM` alone.
            Needed once imbibition tables are evaluated, a cell then being described by both.
        :returns: The cell groups.
        :raises ValidationError: If `saturation_region` is not set, or `use_imbibition_region`
            is set while `imbibition_region` is not.
        """
        if self.saturation_region is None:
            raise ValidationError("`saturation_region` is not set, so cells cannot be grouped.")
        if use_imbibition_region and self.imbibition_region is None:
            raise ValidationError(
                "`imbibition_region` is not set, so cells cannot be grouped by it."
            )
        return group_cells_by_region(
            saturation_region=self.saturation_region,
            imbibition_region=self.imbibition_region if use_imbibition_region else None,
        )

    def get_pvt_region(self, cell_idx: Integer) -> Integer:
        """Return the PVT region for a cell, defaulting to 1."""
        if self.pvt_region is None:
            return 1
        return self.pvt_region[cell_idx]

    get_pvtnum = get_pvt_region

    def get_saturation_region(self, cell_idx: Integer) -> Integer:
        """Return the saturation function region for a cell, defaulting to 1."""
        if self.saturation_region is None:
            return 1
        return self.saturation_region[cell_idx]

    get_satnum = get_saturation_region

    def get_imbibition_region(self, cell_idx: Integer) -> Integer:
        """Return the imbibition region for a cell, defaulting to 1."""
        if self.imbibition_region is None:
            return 1
        return self.imbibition_region[cell_idx]

    def get_equilibration_region(self, cell_idx: Integer) -> Integer:
        """Return the equilibration region for a cell, defaulting to 1."""
        if self.equilibrium_region is None:
            return 1
        return self.equilibrium_region[cell_idx]

    get_eqlnum = get_equilibration_region

    def get_fluid_in_place_region(self, cell_idx: Integer) -> Integer:
        """Return the fluid-in-place region for a cell, defaulting to 1."""
        if self.fluid_in_place_region is None:
            return 1
        return self.fluid_in_place_region[cell_idx]

    get_fipnum = get_fluid_in_place_region

    def get_rock_region(self, cell_idx: Integer) -> Integer:
        """Return the rock function region for a cell, defaulting to 1."""
        if self.rock_region is None:
            return 1
        return self.rock_region[cell_idx]

    get_rocknum = get_rock_region

import typing

import attrs
import numpy as np
from typing_extensions import Self

from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.types import IntCellArray

__all__ = ["Regions"]


def load_region_array(
    deck_file: DeckFile, keyword: str, n_cells: int, default_region: int | None = None
) -> IntCellArray | None:
    array = deck_file.get(keyword)
    if array is None:
        if default_region is not None:
            return np.full(n_cells, default_region, dtype=np.int32)
        return None

    array = array.astype(np.int32, copy=False)
    if array.size != n_cells:
        raise ValidationError(f"{keyword} has {array.size} values; expected {n_cells}.")
    return typing.cast(IntCellArray, array)


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
    def from_deck(cls, deck_file: DeckFile, *, n_cells: int, use_default: bool = False) -> Self:
        """
        Build `Regions` from a parsed DeckFile.

        Missing keywords default to None (region 1 is assumed by callers).

        :param deck_file: Parsed DeckFile.
        :param n_cells: Number of active cells, for validation.
        :param use_default: If True, missing keywords are filled with region 1.
            Else, they are left as None. Default: False.
        :returns: `Regions` object loaded from ECLIPSE deck.
        """
        default_region = 1 if use_default else None
        return cls(
            pvt_region=load_region_array(
                deck_file, "PVTNUM", n_cells=n_cells, default_region=default_region
            ),
            saturation_region=load_region_array(
                deck_file, "SATNUM", n_cells=n_cells, default_region=default_region
            ),
            imbibition_region=load_region_array(
                deck_file, "IMBNUM", n_cells=n_cells, default_region=default_region
            ),
            equilibrium_region=load_region_array(
                deck_file, "EQLNUM", n_cells=n_cells, default_region=default_region
            ),
            rock_region=load_region_array(
                deck_file, "ROCKNUM", n_cells=n_cells, default_region=default_region
            ),
            fluid_in_place_region=load_region_array(
                deck_file, "FIPNUM", n_cells=n_cells, default_region=default_region
            ),
        )

    def get_pvt_region(self, cell_idx: int) -> int:
        """Return the PVT region for a cell, defaulting to 1."""
        if self.pvt_region is None:
            return 1
        return int(self.pvt_region[cell_idx])

    get_pvtnum = get_pvt_region

    def get_saturation_region(self, cell_idx: int) -> int:
        """Return the saturation function region for a cell, defaulting to 1."""
        if self.saturation_region is None:
            return 1
        return int(self.saturation_region[cell_idx])

    get_satnum = get_saturation_region

    def get_imbibition_region(self, cell_idx: int) -> int:
        """Return the imbibition region for a cell, defaulting to 1."""
        if self.imbibition_region is None:
            return 1
        return int(self.imbibition_region[cell_idx])

    def get_equilibration_region(self, cell_idx: int) -> int:
        """Return the equilibration region for a cell, defaulting to 1."""
        if self.equilibrium_region is None:
            return 1
        return int(self.equilibrium_region[cell_idx])

    get_eqlnum = get_equilibration_region

    def get_fluid_in_place_region(self, cell_idx: int) -> int:
        """Return the fluid-in-place region for a cell, defaulting to 1."""
        if self.fluid_in_place_region is None:
            return 1
        return int(self.fluid_in_place_region[cell_idx])

    get_fipnum = get_fluid_in_place_region

    def get_rock_region(self, cell_idx: int) -> int:
        """Return the rock function region for a cell, defaulting to 1."""
        if self.rock_region is None:
            return 1
        return int(self.rock_region[cell_idx])

    get_rocknum = get_rock_region

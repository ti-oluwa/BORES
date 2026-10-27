import numpy as np
import pytest

from bores.deck import DeckFile
from bores.errors import ValidationError
from bores.reservoir.model import validate_regions
from bores.reservoir.regions import Regions, load_region_array

# A 2x2 grid whose first and last cells are inactive. Like Norne's `EQLNUM` and `FIPNUM`, the deck
# writes 0 as the region of those cells, which Eclipse ignores.
DECK = "METRIC\nSPECGRID\n 2 2 1 1 F /\nEQLNUM\n 0 2 3 0 /\nFIPNUM\n 0 1 1 0 /\n"
ACTIVE = np.array([False, True, True, False])


def deck() -> DeckFile:
    return DeckFile(DECK)


def test_inactive_cells_with_region_zero_get_region_one():
    regions = Regions.from_deck(deck(), n_cells=4, use_default=True, active=ACTIVE)
    assert regions.equilibrium_region.tolist() == [1, 2, 3, 1]
    assert regions.fluid_in_place_region.tolist() == [1, 1, 1, 1]
    validate_regions(regions, 4)


def test_active_cells_are_never_changed():
    regions = Regions.from_deck(deck(), n_cells=4, use_default=True, active=ACTIVE)
    assert regions.equilibrium_region[ACTIVE].tolist() == [2, 3]


def test_without_the_active_mask_numbers_are_left_as_written_and_validation_explains():
    regions = Regions.from_deck(deck(), n_cells=4, use_default=True)
    assert regions.equilibrium_region.tolist() == [0, 2, 3, 0]
    with pytest.raises(ValidationError, match=r"2 region number\(s\) below 1 .*active=\.\.\."):
        validate_regions(regions, 4)


def test_a_zero_in_an_active_cell_is_still_an_error():
    everything_active = np.ones(4, dtype=bool)
    regions = Regions.from_deck(deck(), n_cells=4, use_default=True, active=everything_active)
    with pytest.raises(ValidationError, match="below 1"):
        validate_regions(regions, 4)


def test_the_active_mask_must_match_the_grid():
    with pytest.raises(ValidationError, match="`active` has shape"):
        load_region_array(deck(), "EQLNUM", 4, active=np.ones(3, dtype=bool))


def test_missing_keywords_still_default_to_region_one():
    only_grid = DeckFile("METRIC\nSPECGRID\n 2 2 1 1 F /\n")
    regions = Regions.from_deck(only_grid, n_cells=4, use_default=True, active=ACTIVE)
    assert regions.equilibrium_region.tolist() == [1, 1, 1, 1]

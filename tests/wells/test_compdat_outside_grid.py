"""A `COMPDAT` or `WELSPECS` location outside the grid is an error, not a different cell."""

import warnings

import pytest

from bores.deck import DeckFile
from bores.deck import keywords as k
from bores.errors import ValidationError
from bores.grids.factories.cartesian import make_cartesian_grid
from bores.wells.deck import load_wells_from_records

HEADER = "RUNSPEC\nFIELD\nSPECGRID\n 10 10 3 1 F /\n"


def load(*, i=1, j=1, k1=1, k2=None, head_i=1, head_j=1):
    grid = make_cartesian_grid(nx=10, ny=10, nz=3, dx=100.0, dy=100.0, dz=10.0)
    deck = DeckFile(
        HEADER
        + f"WELSPECS\n 'W' 'G' {head_i} {head_j} 1* 'OIL' /\n/\n"
        + f"COMPDAT\n 'W' {i} {j} {k1} {k1 if k2 is None else k2} 'OPEN' 1* 1* 0.5 /\n/\n",
        encoding="utf-8",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wells = load_wells_from_records(grid, deck.get(k.WELSPECS), deck.get(k.COMPDAT))
    (name,) = [well for well in wells]
    well = wells[name] if isinstance(name, str) else name
    return grid, well


def test_the_last_cell_of_the_grid_is_accepted():
    grid, well = load(i=10, j=10, k1=3, head_i=10, head_j=10)
    assert grid.ijk_index(well.perforations[0].cell_index) == (9, 9, 2)
    assert tuple(well.surface_location) == pytest.approx((950.0, 950.0))


@pytest.mark.parametrize(
    "location",
    [
        {"i": 11},
        {"j": 11},
        {"k1": 4},
        {"i": 0},
        {"k1": 3, "k2": 2},
    ],
)
def test_a_connection_outside_the_grid_is_rejected(location):
    with pytest.raises(ValidationError, match="`COMPDAT` connection.*10 x 10 x 3 grid"):
        load(**location)


def test_a_well_head_outside_the_grid_is_rejected():
    with pytest.raises(ValidationError, match="`WELSPECS` head.*10 x 10 grid"):
        load(head_i=11)

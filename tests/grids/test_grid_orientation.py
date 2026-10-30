import warnings

import pytest

from bores.datastructures import (
    DepthDirection,
    GridOrientation,
    Handedness,
    IndexDirection,
)
from bores.grids.io.grdecl import dump_grdecl, load_grdecl
from bores.types import UnitSystem

BASE = """FIELD
SPECGRID
 3 2 2 1 F /
GRIDUNIT
 'FEET' /
{orientation}
DXV
 100 100 100 /
DYV
 50 50 /
DZV
 10 10 /
TOPS
 6*1000 /
"""


def build(orientation_block: str = "") -> str:
    return BASE.format(orientation=orientation_block)


def load(orientation_block: str = ""):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_grdecl(build(orientation_block))


def test_absent_keyword_gives_no_orientation():
    assert load().orientation is None


def test_default_record_is_recorded_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        grid = load_grdecl(build("GDORIENT\n INC INC INC DOWN RIGHT /"))
    assert grid.orientation == GridOrientation()
    assert grid.orientation.is_default


def test_bare_record_takes_defaults():
    assert load("GDORIENT\n /").orientation == GridOrientation()


def test_items_are_translated_from_deck_spelling():
    orientation = load("GDORIENT\n DEC INC DEC UP LEFT /").orientation
    assert orientation == GridOrientation(
        i_direction=IndexDirection.DECREASING,
        j_direction=IndexDirection.INCREASING,
        k_direction=IndexDirection.DECREASING,
        depth_direction=DepthDirection.UP,
        handedness=Handedness.LEFT,
    )
    assert orientation.non_default_items == {
        "i_direction": "decreasing",
        "k_direction": "decreasing",
        "depth_direction": "up",
        "handedness": "left",
    }


def test_non_default_orientation_warns_that_it_is_not_applied():
    with pytest.warns(UserWarning, match=r"GDORIENT.*i_direction=decreasing.*used as written"):
        load_grdecl(build("GDORIENT\n DEC INC INC DOWN RIGHT /"))


def test_upward_depth_warning_mentions_gravity():
    with pytest.warns(UserWarning, match="gravity"):
        load_grdecl(build("GDORIENT\n INC INC INC UP RIGHT /"))


def test_geometry_is_not_changed_by_orientation():
    plain = load()
    flipped = load("GDORIENT\n DEC DEC DEC UP LEFT /")
    assert (plain.cell_centroids == flipped.cell_centroids).all()
    assert (plain.cell_volumes == flipped.cell_volumes).all()


def test_dump_writes_non_default_orientation_and_round_trips():
    grid = load("GDORIENT\n DEC INC DEC UP LEFT /")
    text = dump_grdecl(grid)
    assert "GDORIENT" in text
    assert "DEC  INC  DEC  UP  LEFT" in text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reloaded = load_grdecl(text, unit_system=UnitSystem.FIELD)
    assert reloaded.orientation == grid.orientation


@pytest.mark.parametrize("block", ["", "GDORIENT\n INC INC INC DOWN RIGHT /"])
def test_dump_omits_default_or_absent_orientation(block):
    assert "GDORIENT" not in dump_grdecl(load(block))


def test_invalid_item_is_rejected():
    with pytest.raises(Exception, match="GDORIENT"):
        load_grdecl(build("GDORIENT\n SIDEWAYS /"))

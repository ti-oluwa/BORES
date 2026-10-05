import pytest

from bores.errors import GridImportError
from bores.grids.io.grdecl import load_grdecl
from bores.types import UnitSystem

NX, NY, NZ = 3, 2, 2


def keyword(name, values):
    return f"{name}\n" + " ".join(str(value) for value in values) + "\n/\n\n"


def cartesian_deck_text(comment=""):
    cells = NX * NY * NZ
    text = f"{comment}SPECGRID\n{NX} {NY} {NZ} 1 F /\n\nGRIDUNIT\nFEET /\n\n"
    text += keyword("DX", [10] * cells) + keyword("DY", [12] * cells)
    text += keyword("DZ", [2] * cells) + keyword("TOPS", [2000] * (NX * NY))
    return text


def test_raw_text_loads_the_same_grid_as_the_file(tmp_path):
    text = cartesian_deck_text()
    path = tmp_path / "deck.grdecl"
    path.write_text(text)

    from_text = load_grdecl(text, unit_system=UnitSystem.FIELD)
    from_file = load_grdecl(path, unit_system=UnitSystem.FIELD)

    assert from_text.cell_volumes.sum() == pytest.approx(from_file.cell_volumes.sum())
    assert from_text.dimensions == from_file.dimensions


def test_raw_text_longer_than_the_filename_limit_loads():
    text = cartesian_deck_text(comment="-- " + "x" * 5000 + "\n")
    assert len(text) > 4096
    grid = load_grdecl(text, unit_system=UnitSystem.FIELD)
    assert grid.cell_volumes.sum() == pytest.approx(NX * 10 * NY * 12 * NZ * 2)


def test_string_path_still_loads(tmp_path):
    path = tmp_path / "deck.grdecl"
    path.write_text(cartesian_deck_text())
    grid = load_grdecl(str(path), unit_system=UnitSystem.FIELD)
    assert grid.cell_volumes.sum() == pytest.approx(NX * 10 * NY * 12 * NZ * 2)


def test_missing_file_path_is_reported_as_missing(tmp_path):
    with pytest.raises(GridImportError, match="No file exists"):
        load_grdecl(str(tmp_path / "absent.grdecl"), unit_system=UnitSystem.FIELD)


def test_very_long_single_line_string_is_reported_not_crashed():
    with pytest.raises(GridImportError, match="No file exists"):
        load_grdecl("x" * 5000, unit_system=UnitSystem.FIELD)

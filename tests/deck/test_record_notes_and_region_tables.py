import warnings

import numpy as np
import pytest

from bores.blackoil.pvt.static import StaticPVT
from bores.blackoil.satfunc.regions import SatFunc
from bores.deck import Deck, DeckFile, DeckParseError
from bores.deck.core import find_terminator, strip_comments, strip_record_notes

# Two stretches of the Norne deck (opm-data, NORNE_ATW2013.DATA), exactly as written there: every
# record carries a note after its closing `/`, and one of them contains a tab.
NORNE_PERMZ_BLOCK = """COPY
   PERMX PERMY /
   PERMX PERMZ /
/

-- based on same kv/kh factor
MULTIPLY
   'PERMZ' 0.2    1 46 1 112  1  1 /    Garn 3
   'PERMZ' 0.04   1 46 1 112  2  2 /    Garn 2
   'PERMZ' 0.25   1 46 1 112  3  3 /    Garn 1
   'PERMZ' 0.0    1 46 1 112  4  4 /    Not \t(inactive anyway)
   'PERMZ' 0.13   1 46 1 112  5  5 /    Ile 2.2
   'PERMZ' 0.13   1 46 1 112  6  6 /    Ile 2.1.3
   'PERMZ' 0.13   1 46 1 112  7  7 /    Ile 2.1.2
   'PERMZ' 0.13   1 46 1 112  8  8 /    Ile 2.1.1
   'PERMZ' 0.09   1 46 1 112  9  9 /    Ile 1.3
   'PERMZ' 0.07   1 46 1 112 10 10 /    Ile 1.2
   'PERMZ' 0.19   1 46 1 112 11 11 /    Ile 1.1
   'PERMZ' 0.13   1 46 1 112 12 12 /    Tofte 2.2
   'PERMZ' 0.64   1 46 1 112 13 13 /\tTofte 2.1.3
   'PERMZ' 0.64   1 46 1 112 14 14 /\tTofte 2.1.2
   'PERMZ' 0.64   1 46 1 112 15 15 /\tTofte 2.1.1
   'PERMZ' 0.64   1 46 1 112 16 16 /\tTofte 1.2.2
   'PERMZ' 0.64   1 46 1 112 17 17 /\tTofte 1.2.1
   'PERMZ' 0.016  1 46 1 112 18 18 /\tTofte 1.1
   'PERMZ' 0.004  1 46 1 112 19 19 /\tTilje 4
   'PERMZ' 0.004  1 46 1 112 20 20 /\tTilje 3
   'PERMZ' 1.0    1 46 1 112 21 21 /\tTilje 2
   'PERMZ' 1.0    1 46 1 112 22 22 /\tTilje 1
/
"""
NORNE_MULTZ_BLOCK = """EQUALS
  'MULTZ'    1.0      1  46  1 112   1   1  / Garn3       - Garn 2
  'MULTZ'    0.05     1  46  1 112  15  15  / Tofte 2.1.1 - Tofte 1.2.2
  'MULTZ'    0.001    1  46  1 112  18  18  / Tofte 1.1   - Tilje 4
  'MULTZ'    0.00001  1  46  1 112  20  20  / Tilje 3     - Tilje 2
-- The Top Tilje 2 barrier is included as MULTREGT = 0.0
/
"""

NX, NY, NZ = 46, 112, 22
PERMZ_PER_LAYER = [
    0.2, 0.04, 0.25, 0.0, 0.13, 0.13, 0.13, 0.13, 0.09, 0.07, 0.19, 0.13,
    0.64, 0.64, 0.64, 0.64, 0.64, 0.016, 0.004, 0.004, 1.0, 1.0,
]  # fmt: skip


def norne_grid_deck(body: str, **kwargs) -> DeckFile:
    return DeckFile(f"METRIC\nSPECGRID\n {NX} {NY} {NZ} 1 F /\n{body}", **kwargs)


def small(body: str, grid: str = "3 1 1", **kwargs) -> DeckFile:
    return DeckFile(f"METRIC\nSPECGRID\n {grid} 1 F /\n{body}", **kwargs)


def captured(function):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function()
    return result, [str(w.message) for w in caught]


# --- the Norne records -------------------------------------------------------------------------


def test_norne_vertical_permeability_multipliers_all_apply_whatever_follows_each_slash():
    deck = norne_grid_deck(f"PERMX\n {NX * NY * NZ}*100 /\n{NORNE_PERMZ_BLOCK}", strict=True)
    permz = deck.get("PERMZ").reshape(NZ, NY, NX)
    # One multiplier per layer, uniform across the layer.
    assert np.allclose(permz.min(axis=(1, 2)), permz.max(axis=(1, 2)))
    assert np.allclose(permz[:, 0, 0] / 100.0, PERMZ_PER_LAYER)
    assert np.allclose(deck.get("PERMY"), 100.0)


def test_norne_notes_cause_no_warnings_and_no_phantom_operators():
    _, messages = captured(
        lambda: norne_grid_deck(f"PERMX\n {NX * NY * NZ}*100 /\n{NORNE_PERMZ_BLOCK}")
    )
    assert messages == []
    deck = norne_grid_deck(f"PERMX\n {NX * NY * NZ}*100 /\n{NORNE_PERMZ_BLOCK}")
    # `Garn`, `Ile`, `Tofte`, ... were once read as the targets of operators of their own.
    assert deck.get("GARN") is None and deck.get("TOFTE") is None


def test_norne_multz_barriers_apply_to_the_layers_they_name():
    deck = norne_grid_deck(NORNE_MULTZ_BLOCK, strict=True)
    multz = deck.get("MULTZ").reshape(NZ, NY, NX)[:, 0, 0]
    expected = np.ones(NZ)
    expected[0], expected[14], expected[17], expected[19] = 1.0, 0.05, 0.001, 0.00001
    assert np.allclose(multz, expected)


# --- notes after the closing slash -------------------------------------------------------------


def test_text_after_a_records_closing_slash_is_dropped():
    body = "\n 'PERMZ' 0.2 1 1 /  Garn 3\n 'PERMZ' 0.04 2 2 /\tGarn 2 (x)\n/\n"
    assert strip_record_notes(body) == "\n 'PERMZ' 0.2 1 1 /\n 'PERMZ' 0.04 2 2 /\n/\n"


def test_a_second_slash_is_not_a_note_and_a_quoted_slash_is_not_a_record_end():
    assert strip_record_notes("\n 'A' 1 / /  \n") == "\n 'A' 1 / /  \n"
    assert strip_record_notes("\n 'a / b' 2 / note\n") == "\n 'a / b' 2 /\n"
    assert strip_record_notes("\n 1 2 3\n") == "\n 1 2 3\n"


def test_a_note_never_makes_a_single_record_keyword_a_multi_record_one():
    deck = Deck(strip_comments("PORO\n 1 2 3 / porosity, from the core data\nPERMX\n 4 5 6 /\n"))
    poro = deck.get_first_record_for("PORO")
    assert poro is not None and "/" not in poro.body
    assert poro.body.split() == ["1", "2", "3"]


def test_notes_are_dropped_in_every_kind_of_keyword():
    deck = small(
        "PORO\n 0.1 0.2 0.3 / the porosity\n"
        "EQUALS\n 'PERMX' 5 / first layer\n 'PERMY' 6 / second layer\n/\n"
        "TSTEP\n 10 / ten days\n",
        strict=True,
    )
    assert deck.get("PORO").tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert deck.get("PERMX").tolist() == pytest.approx([5.0] * 3)
    assert deck.get("PERMY").tolist() == pytest.approx([6.0] * 3)
    assert deck.report_times == [0.0, 10.0]


def test_a_note_with_an_apostrophe_is_not_an_unterminated_quote():
    _, messages = captured(lambda: small("PORO\n 0.1 0.2 0.3 / don't round it\n", strict=True))
    assert messages == []


def test_a_slash_in_a_note_that_is_not_a_record_is_dropped_without_comment():
    deck, messages = captured(
        lambda: small("EQUALS\n 'PERMX' 5 / kv/kh factor\n 'PERMY' 6 /\n/\n", strict=True)
    )
    assert messages == []
    assert deck.get("PERMX").tolist() == pytest.approx([5.0] * 3)
    assert deck.get("PERMY").tolist() == pytest.approx([6.0] * 3)


def test_a_second_record_on_the_same_line_is_ignored_with_a_warning():
    body = "EQUALS\n 'PERMX' 5 / 'PERMY' 6 /\n/\n"
    deck, messages = captured(lambda: small(body))
    assert any("looks like another record" in m and "PERMY" in m for m in messages)
    assert deck.get("PERMX").tolist() == pytest.approx([5.0] * 3)
    assert deck.get("PERMY") is None
    with pytest.raises(DeckParseError, match="looks like another record"):
        small(body, strict=True)


def test_a_second_record_after_a_single_record_is_also_reported():
    _, messages = captured(lambda: small("PORO\n 0.1 0.2 0.3 / 0.4 0.5 0.6 /\n"))
    assert any("looks like another record" in m for m in messages)


def test_free_text_keeps_a_slash_in_its_text():
    deck = small("TITLE\nbase case / sensitivity 2\n")
    assert deck.get("TITLE") == "base case / sensitivity 2"


def test_include_ignores_a_note_after_its_closing_slash(tmp_path):
    (tmp_path / "poro.inc").write_text("PORO\n 0.1 0.2 0.3 /")  # no trailing newline
    main = tmp_path / "main.DATA"
    main.write_text(
        "METRIC\nSPECGRID\n 3 1 1 1 F /\nINCLUDE\n 'poro.inc' /  the porosity, 2.1.3\nPERMX\n 4 5 6 /\n"
    )
    deck = DeckFile(main, strict=True)
    assert deck.get("PORO").tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert deck.get("PERMX").tolist() == pytest.approx([4.0, 5.0, 6.0])


def test_terminator_search_is_unchanged_by_notes():
    assert find_terminator("'a/b' 1 / note") == 8


# --- one table per region ----------------------------------------------------------------------


def rows_per_region(deck: DeckFile, keyword: str) -> list[int]:
    return [len(table) for table in deck.get(keyword)]


def test_one_row_per_region_keywords_give_one_entry_per_region():
    deck = DeckFile(
        "FIELD\nDENSITY\n 50 62 0.06 /\n 45 63 0.07 /\n"
        "PVTW\n 4000 1.0 3e-6 0.5 0 /\n 3000 1.1 3e-6 0.6 0 /\n"
        "ROCK\n 4000 3e-6 /\n 3000 5e-6 /\n"
    )
    for keyword in ("DENSITY", "PVTW", "ROCK"):
        assert rows_per_region(deck, keyword) == [1, 1], keyword
    assert [table[0]["oil"] for table in deck.get("DENSITY")] == [50.0, 45.0]
    assert [table[0]["reference_pressure"] for table in deck.get("PVTW")] == [4000.0, 3000.0]


def test_one_table_per_region_keywords_give_one_entry_per_region():
    deck = DeckFile(
        "FIELD\nSWOF\n 0.2 0 1 0\n 0.6 0.3 0.3 0\n 1.0 1 0 0 /\n 0.1 0 1 0\n 0.9 1 0 0 /\n"
        "PVDO\n 100 1.2 0.8\n 200 1.1 0.9 /\n 100 1.3 0.7\n 200 1.2 0.8\n 300 1.1 0.9 /\n"
        "RSVD\n 1000 0.5\n 5000 0.6 /\n 1000 0.4\n 5000 0.5 /\n"
    )
    assert rows_per_region(deck, "SWOF") == [3, 2]
    assert rows_per_region(deck, "PVDO") == [2, 3]
    assert rows_per_region(deck, "RSVD") == [2, 2]
    assert deck.get("SWOF")[1][0]["sw"] == pytest.approx(0.1)


def test_a_single_region_is_a_single_entry_however_it_is_written():
    packed = DeckFile("FIELD\nSWOF\n 0.2 0 1 0\n 0.6 0.3 0.3 0\n 1.0 1 0 0 /\n").get("SWOF")
    compact = DeckFile("FIELD\nSWOF\n 0.2 0 1 0 /\n 0.6 0.3 0.3 0 /\n 1.0 1 0 0 /\n/\n").get(
        "SWOF"
    )
    assert len(packed) == len(compact) == 1
    assert packed == compact


def test_compact_style_needs_the_closing_lone_slash():
    # Without it, rows each closed by a `/` are separate regions, as in Eclipse.
    rows = DeckFile("FIELD\nSWOF\n 0.2 0 1 0 /\n 1.0 1 0 0 /\n").get("SWOF")
    assert [len(table) for table in rows] == [1, 1]


def test_a_segment_with_a_partial_row_is_rejected():
    with pytest.raises(DeckParseError, match="expected a multiple of 4"):
        DeckFile("FIELD\nSWOF\n 0.2 0 1 0\n 0.6 0.3 /\n").get("SWOF")


def test_empty_segments_are_not_regions():
    tables = DeckFile("FIELD\nSWOF\n 0.2 0 1 0\n 1.0 1 0 0 /\n /\n").get("SWOF")
    assert [len(table) for table in tables] == [2]


def test_tracer_tables_are_one_per_region():
    deck = small("TVDPFSEA\n 1000 0.0\n 5000 1.5 /\n 1000 2.0\n 5000 3.0 /\n")
    first, second = deck.get("TVDPFSEA")
    assert [row["concentration"] for row in first] == [0.0, 1.5]
    assert [row["concentration"] for row in second] == [2.0, 3.0]


def test_the_pvt_of_each_region_comes_from_its_own_entries():
    deck = DeckFile(
        "FIELD\nSPECGRID\n 2 1 1 1 F /\n"
        "DENSITY\n 50 62 0.06 /\n 45 63 0.07 /\n"
        "PVTW\n 4000 1.0 3e-6 0.5 0 /\n 3000 1.1 3e-6 0.6 0 /\n"
    )
    first, second = (StaticPVT.from_deck(deck, pvtnum=n) for n in (1, 2))
    assert (first.stock_tank_oil_density, second.stock_tank_oil_density) == (50.0, 45.0)
    assert (first.water_reference_pressure, second.water_reference_pressure) == (4000.0, 3000.0)
    assert (first.water_reference_fvf, second.water_reference_fvf) == (1.0, 1.1)


def test_two_saturation_tables_make_two_saturation_regions():
    deck = DeckFile(
        "FIELD\nSPECGRID\n 2 1 1 1 F /\n"
        "SWOF\n 0.2 0 1 0\n 0.6 0.3 0.3 0\n 1.0 1 0 0 /\n 0.1 0 1 0\n 0.5 0.2 0.4 0\n 1.0 1 0 0 /\n"
        "SGOF\n 0 0 1 0\n 0.5 0.4 0.1 0\n 0.9 1 0 0 /\n 0 0 1 0\n 0.4 0.3 0.1 0\n 0.9 1 0 0 /\n"
    )
    assert len(SatFunc.from_deck(deck)) == 2

import warnings

import numpy as np
import pytest

from bores.deck import DeckFile, DeckParseError
from bores.deck.core import (
    INCLUDE_SCAN_CHUNK,
    find_terminator,
    first_segment,
    may_contain_include,
    split_segments,
    strip_comments,
)
from bores.deck.keywords.base import ArrayKeyword
from bores.errors import GridImportError
from bores.grids.io.grdecl import load_grdecl
from bores.types import UnitSystem

HEADER = "METRIC\nSPECGRID\n 3 1 1 1 F /\n"


def make(body: str, **kwargs) -> DeckFile:
    return DeckFile(HEADER + body, **kwargs)


def captured(function):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function()
    return result, [str(w.message) for w in caught]


# --- quote-aware lexing ------------------------------------------------------------------------


def test_comment_markers_inside_quotes_are_kept():
    text = "RESTART\n 'CASE--A' 5 / -- trailing\n"
    assert strip_comments(text) == "RESTART\n 'CASE--A' 5 / \n"
    assert strip_comments('X "a -- b" -- c\n') == 'X "a -- b" \n'


def test_unclosed_quote_does_not_hide_a_comment():
    assert strip_comments("A 'oops -- gone\nB\n") == "A 'oops \nB\n"


def test_stripping_comments_from_text_without_any_is_free():
    text = "PORO\n1 2 3 /\n"
    assert strip_comments(text) is text


def test_quoted_values_may_contain_comment_markers_and_slashes():
    deck = make("RESTART\n 'run/CASE--1' 5 / -- comment\n")
    assert deck.get("RESTART") == {"root_name": "run/CASE--1", "report_step": 5}


def test_slash_helpers_ignore_quoted_slashes():
    assert find_terminator("'a/b' 1 / 2 /") == 8
    assert find_terminator("'a/b' 1") == -1
    assert first_segment("'a/b' 1 / 2 /") == "'a/b' 1 "
    assert first_segment("1 2 3") == "1 2 3"
    assert split_segments("'a/b' 1 / 2 / ") == ["'a/b' 1 ", " 2 ", " "]
    assert split_segments("1 / 2 /") == ["1 ", " 2 ", ""]


# --- includes ----------------------------------------------------------------------------------


def test_include_detection_is_case_insensitive_and_spans_chunks():
    assert may_contain_include("x\nInClUdE\n")
    assert not may_contain_include("PORO\n1 2 3 /\n")
    padding = "x" * (INCLUDE_SCAN_CHUNK - 3)
    assert may_contain_include(padding + "include" + "y" * 10)
    assert may_contain_include(padding + "INCLUDE")
    assert not may_contain_include(padding + "includ")


def test_include_cycle_is_reported_with_the_chain(tmp_path):
    (tmp_path / "a.inc").write_text("INCLUDE\n 'b.inc' /\n")
    (tmp_path / "b.inc").write_text("INCLUDE\n 'a.inc' /\n")
    main = tmp_path / "main.DATA"
    main.write_text("METRIC\nINCLUDE\n 'a.inc' /\n")
    with pytest.raises(DeckParseError, match=r"cycle detected.*a\.inc.*b\.inc.*a\.inc"):
        DeckFile(main)


def test_a_file_that_includes_itself_is_a_cycle(tmp_path):
    main = tmp_path / "main.DATA"
    main.write_text("METRIC\nINCLUDE\n 'main.DATA' /\n")
    with pytest.raises(DeckParseError, match="cycle detected"):
        DeckFile(main)


def test_the_same_file_included_twice_is_not_a_cycle(tmp_path):
    (tmp_path / "poro.inc").write_text("PORO\n 3*0.2 /\n")
    main = tmp_path / "main.DATA"
    main.write_text(
        "METRIC\nSPECGRID\n 3 1 1 1 F /\nINCLUDE\n 'poro.inc' /\nINCLUDE\n 'poro.inc' /\n"
    )
    assert DeckFile(main).get("PORO").tolist() == [0.2, 0.2, 0.2]


def test_commented_out_include_is_not_followed(tmp_path):
    main = tmp_path / "main.DATA"
    main.write_text("METRIC\n-- INCLUDE\n-- 'missing.inc' /\n")
    assert DeckFile(main).keywords == ["METRIC"]


def test_include_in_raw_text_warns_and_is_an_error_when_strict():
    body = "METRIC\nINCLUDE\n 'x.inc' /\n"
    deck, messages = captured(lambda: DeckFile(body))
    assert deck.keywords == ["METRIC"]
    assert any("raw-text input" in message for message in messages)
    with pytest.raises(DeckParseError, match="raw-text input"):
        DeckFile(body, strict=True)


# --- records without a terminator and unclosed quotes ------------------------------------------


UNTERMINATED_MIDDLE = "PORO\n 0.1 0.2 0.3\nPERMX\n 5 5 5 /\n"


def test_unterminated_data_in_the_middle_of_a_deck_is_reported():
    deck, messages = captured(lambda: make(UNTERMINATED_MIDDLE))
    assert any("`PORO` has data but no terminating '/'" in m and "`PERMX`" in m for m in messages)
    assert deck.get("PERMX").tolist() == [5.0, 5.0, 5.0]


def test_unterminated_data_at_the_end_of_a_deck_is_reported():
    _, messages = captured(lambda: make("PORO\n 0.1 0.2 0.3\n"))
    assert any("before the end of the deck" in m for m in messages)


def test_unterminated_data_is_an_error_when_strict():
    with pytest.raises(DeckParseError, match="`PORO` has data but no terminating"):
        make(UNTERMINATED_MIDDLE, strict=True)


def test_nullary_keywords_and_free_text_are_not_unterminated():
    _, messages = captured(lambda: make("OIL\nWATER\nTITLE\nmy model\n", strict=True))
    assert messages == []


def test_unclosed_quote_is_reported_and_an_error_when_strict():
    body = "RESTART\n 'CASE 5 /\n"
    _, messages = captured(lambda: make(body))
    assert any("never closed" in m for m in messages)
    with pytest.raises(DeckParseError, match="never closed"):
        make(body, strict=True)


def test_balanced_quotes_with_apostrophes_in_the_other_quote_style_are_fine():
    _, messages = captured(lambda: make('RESTART\n "it\'s" 5 /\n', strict=True))
    assert messages == []


# --- unit system and dimensions ----------------------------------------------------------------


def test_conflicting_unit_system_keywords_are_reported():
    deck, messages = captured(lambda: DeckFile("METRIC\nFIELD\n"))
    assert deck.unit_system == UnitSystem.METRIC
    assert any("more than one unit system" in m for m in messages)
    with pytest.raises(DeckParseError, match="more than one unit system"):
        DeckFile("METRIC\nFIELD\n", strict=True)


def test_a_single_unit_system_keyword_is_silent():
    deck, messages = captured(lambda: DeckFile("FIELD\n", strict=True))
    assert deck.unit_system == UnitSystem.FIELD
    assert messages == []


def test_explicit_unit_system_overrides_the_deck_without_complaint():
    deck, messages = captured(
        lambda: DeckFile("METRIC\n", unit_system=UnitSystem.FIELD, strict=True)
    )
    assert deck.unit_system == UnitSystem.FIELD
    assert messages == []


def test_specgrid_and_dimens_disagreeing_is_reported():
    body = "METRIC\nSPECGRID\n 2 1 1 1 F /\nDIMENS\n 3 3 3 /\n"
    deck, messages = captured(lambda: DeckFile(body))
    assert (deck.dimensions.nx, deck.dimensions.ny, deck.dimensions.nz) == (2, 1, 1)
    assert any("`DIMENS` declares 3x3x3" in m for m in messages)
    with pytest.raises(DeckParseError, match="DIMENS"):
        DeckFile(body, strict=True)


def test_specgrid_and_dimens_agreeing_is_silent():
    body = "METRIC\nSPECGRID\n 2 1 1 1 F /\nDIMENS\n 2 1 1 /\n"
    _, messages = captured(lambda: DeckFile(body, strict=True))
    assert messages == []


def test_a_repeated_single_use_keyword_is_reported():
    body = "METRIC\nSPECGRID\n 2 1 1 1 F /\nSPECGRID\n 3 3 3 1 F /\n"
    deck, messages = captured(lambda: DeckFile(body))
    assert deck.dimensions.nx == 2
    assert any("`SPECGRID` appears 2 times" in m for m in messages)
    with pytest.raises(DeckParseError, match="appears 2 times"):
        DeckFile(body, strict=True)


def test_registering_a_dimension_keyword_later_re_reads_the_grid():
    deck = DeckFile("METRIC\nSPECGRID\n 2 1 1 1 F /\n", keywords=[])
    assert deck.dimensions is None
    from bores.deck import keywords as k

    deck.add_keywords(k.SPECGRID, ArrayKeyword("PORO"))
    assert deck.dimensions is not None and deck.dimensions.n_cells == 2


# --- BOX and operator records ------------------------------------------------------------------


def test_defaulted_box_bounds_default_to_the_whole_grid():
    body = "BOX\n 1 2 1 1 1* 1 /\nEQUALS\n 'PORO' 0.4 /\n/\n"
    assert make(body, strict=True).get("PORO").tolist() == [0.4, 0.4, 0.0]


def test_box_items_left_off_the_end_are_defaulted():
    body = "BOX\n 2 3 /\nEQUALS\n 'PORO' 0.4 /\n/\n"
    deck = DeckFile("METRIC\nSPECGRID\n 3 2 1 1 F /\n" + body, strict=True)
    assert deck.get("PORO").tolist() == [0.0, 0.4, 0.4, 0.0, 0.4, 0.4]


def test_box_outside_the_grid_is_clamped_with_a_warning_or_an_error_when_strict():
    body = "BOX\n 1 9 1 1 1 1 /\nEQUALS\n 'PORO' 0.4 /\n/\n"
    deck, messages = captured(lambda: make(body))
    assert deck.get("PORO").tolist() == [0.4, 0.4, 0.4]
    assert any("fall outside 1-3" in m for m in messages)
    with pytest.raises(DeckParseError, match="fall outside 1-3"):
        make(body, strict=True)


def test_reversed_box_bounds_are_reported():
    body = "BOX\n 3 1 1 1 1 1 /\nEQUALS\n 'PORO' 0.4 /\n/\n"
    _, messages = captured(lambda: make(body))
    assert any("I1 (3) is above I2 (1)" in m for m in messages)


def test_unreadable_box_says_what_happens_next_and_is_an_error_when_strict():
    body = "BOX\n 1 x 1 1 1 1 /\nEQUALS\n 'PORO' 0.4 /\n/\n"
    _, messages = captured(lambda: make(body))
    assert any("non-integer box bound 'x'" in m for m in messages)
    assert any("apply to the previous box" in m for m in messages)
    with pytest.raises(DeckParseError, match="non-integer box bound"):
        make(body, strict=True)


def test_inline_box_with_missing_items_keeps_the_ambient_bounds():
    body = "BOX\n 1 2 1 1 1 1 /\nEQUALS\n 'PORO' 0.4 2 /\n/\n"
    # `I1` is 2, every other bound is the ambient BOX's.
    assert make(body, strict=True).get("PORO").tolist() == [0.0, 0.4, 0.0]


def test_malformed_operator_record_is_skipped_with_a_warning_or_an_error_when_strict():
    body = "EQUALS\n 'PORO' abc /\n/\n"
    _, messages = captured(lambda: make(body))
    assert any("non-numeric value 'abc'" in m for m in messages)
    with pytest.raises(DeckParseError, match="non-numeric value"):
        make(body, strict=True)


# --- looking up keywords stays lenient ---------------------------------------------------------


def test_unknown_unregistered_and_absent_keywords_return_none_without_noise():
    deck, messages = captured(lambda: make("FOOBAR\n 1 /\nEQUALS\n 'PORRO' 0.1 /\n/\n"))
    assert messages == []
    strict_deck, strict_messages = captured(
        lambda: make("OIL\nEQUALS\n 'PORRO' 0.1 /\n/\n", strict=True)
    )
    assert strict_messages == []
    for d in (deck, strict_deck):
        assert d.get("FOOBAR") is None
        assert d.get("PORRO") is None
        assert d.get("NOT_A_KEYWORD") is None
        assert d.get("PORO") is None


# --- constructors ------------------------------------------------------------------------------


def test_plain_constructor_accepts_single_line_raw_text():
    assert DeckFile("METRIC").unit_system == UnitSystem.METRIC
    assert DeckFile("field").keywords == ["FIELD"]


def test_plain_constructor_still_reads_a_path_given_as_str(tmp_path):
    path = tmp_path / "model.DATA"
    path.write_text("FIELD\n")
    assert DeckFile(str(path)).unit_system == UnitSystem.FIELD


def test_missing_file_named_by_a_path_like_str_is_an_error_not_raw_text(tmp_path):
    with pytest.raises(DeckParseError, match="No file exists"):
        DeckFile(str(tmp_path / "missing.DATA"))
    with pytest.raises(DeckParseError, match="No file exists"):
        DeckFile("model.DATA")


def test_from_text_never_treats_text_as_a_path(tmp_path, monkeypatch):
    (tmp_path / "METRIC").write_text("FIELD\n")  # a file named like the text
    monkeypatch.chdir(tmp_path)
    assert DeckFile.from_text("METRIC").unit_system == UnitSystem.METRIC


def test_from_text_rejects_non_text():
    with pytest.raises(TypeError):
        DeckFile.from_text(b"METRIC")  # type: ignore[arg-type]


def test_from_file_and_from_bytes(tmp_path):
    path = tmp_path / "model.DATA"
    path.write_text("FIELD\nSPECGRID\n 2 1 1 1 F /\n")
    assert DeckFile.from_file(path).unit_system == UnitSystem.FIELD
    assert DeckFile.from_file(str(path)).dimensions.n_cells == 2
    assert DeckFile.from_path(path).unit_system == UnitSystem.FIELD
    assert DeckFile.from_bytes(path.read_bytes()).unit_system == UnitSystem.FIELD
    with pytest.raises(DeckParseError, match="Cannot read"):
        DeckFile.from_file(tmp_path / "missing.DATA")


def test_strict_is_carried_by_the_constructors(tmp_path):
    assert DeckFile.from_text("METRIC\n", strict=True).strict is True
    assert DeckFile.from_text("METRIC\n").strict is False
    with pytest.raises(DeckParseError):
        DeckFile.from_bytes(b"METRIC\nFIELD\n", strict=True)


# --- caching -----------------------------------------------------------------------------------


def test_uncached_reads_return_independent_writable_arrays():
    deck = make("PORO\n 0.1 0.2 0.3 /\n")
    first = deck.get("PORO")
    first[0] = 9.0
    assert deck.get("PORO").tolist() == [0.1, 0.2, 0.3]


def test_cached_arrays_are_read_only_so_they_cannot_be_overwritten():
    deck = make("PORO\n 0.1 0.2 0.3 /\n")
    cached = deck.get("PORO", use_cache=True)
    assert not cached.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        cached[0] = 9.0
    assert deck.get("PORO", use_cache=True) is cached
    modified = cached.copy()
    modified[0] = 9.0
    assert deck.get("PORO", use_cache=True).tolist() == [0.1, 0.2, 0.3]
    assert deck.get("PORO").flags.writeable


def test_cached_non_array_values_are_copied_on_every_read():
    deck = make("RESTART\n 'CASE' 5 /\n")
    first = deck.get("RESTART", use_cache=True)
    first["root_name"] = "CHANGED"
    assert deck.get("RESTART", use_cache=True)["root_name"] == "CASE"
    assert deck.get("RESTART", use_cache=True) is not deck.get("RESTART", use_cache=True)


# --- GRDECL loader -----------------------------------------------------------------------------


def test_load_grdecl_threads_strict_to_the_deck():
    text = (
        "SPECGRID\n 1 1 1 1 F /\n"
        "COORD\n 0 0 0 0 0 1 1 0 0 1 0 1 0 1 0 0 1 1 1 1 0 1 1 1 /\n"
        "ZCORN\n 0 0 0 0 1 1 1 1 /\n"
        "ACTNUM\n 1\n"  # no terminating slash
    )
    with pytest.raises(GridImportError, match="no terminating"):
        load_grdecl(text, unit_system=UnitSystem.METRIC, strict=True)
    with pytest.warns(UserWarning, match="no terminating"):
        grid = load_grdecl(text, unit_system=UnitSystem.METRIC)
    assert grid.dimensions == (1, 1, 1)


def test_arrays_read_back_unchanged_by_the_lexical_changes():
    deck = make("PORO\n 3*0.25 / -- note\nPERMX\n 100 200 300 /\n")
    assert np.allclose(deck.get("PORO"), 0.25)
    assert deck.get("PERMX").tolist() == [100.0, 200.0, 300.0]

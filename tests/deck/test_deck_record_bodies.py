import pytest

from bores.deck import Deck, DeckFile
from bores.deck.core import first_segment, strip_comments

HEADER = "METRIC\nSPECGRID\n 2 1 1 1 F /\n"


def scan(text: str, **kwargs) -> Deck:
    return Deck(strip_comments(text), **kwargs)


def test_deck_does_not_keep_a_second_copy_of_the_text():
    deck = scan(HEADER + "PORO\n 0.1 0.2 /\n")
    assert not hasattr(deck, "text")
    # Still usable as a dict key or set member, by identity.
    assert deck in {deck}


def test_single_record_body_stops_before_its_terminator():
    deck = scan(HEADER + "PORO\n 0.1 0.2 /\n")
    record = deck.get_first_record_for("PORO")
    assert record is not None
    assert "/" not in record.body
    assert record.body.split() == ["0.1", "0.2"]
    # So parsers can use the body as it is, with no second copy cut out of it.
    assert first_segment(record.body) is record.body


def test_body_of_a_keyword_with_several_records_keeps_all_of_them():
    deck = scan(HEADER + "EQUALS\n 'PORO' 0.2 /\n 'PERMX' 100 /\n/\nPORO\n 0.1 0.2 /\n")
    record = deck.get_first_record_for("EQUALS")
    assert record is not None
    assert record.body.count("/") == 3
    assert first_segment(record.body).split() == ["'PORO'", "0.2"]


def test_quoted_slash_does_not_end_a_single_record_body():
    deck = scan("RESTART\n 'run/CASE1' 5 /\n")
    record = deck.get_first_record_for("RESTART")
    assert record is not None
    assert record.body == "\n 'run/CASE1' 5 "


def test_free_text_body_is_left_alone():
    deck = scan("TITLE\nmy case /\nFIELD\n")
    record = deck.get_first_record_for("TITLE")
    assert record is not None
    assert record.body == "\nmy case /\n"
    assert DeckFile(HEADER.replace("METRIC", "FIELD") + "TITLE\nmy case /\n").get("TITLE")


def test_nullary_and_unterminated_records_have_an_empty_body():
    deck = scan("OIL\nWATER\n")
    assert [record.body for record in deck.records] == ["", ""]


def test_a_final_terminator_without_a_trailing_newline_is_trimmed():
    deck = scan(HEADER + "PORO\n 0.1 0.2 /")
    record = deck.get_first_record_for("PORO")
    assert record is not None
    assert record.body.split() == ["0.1", "0.2"]
    assert DeckFile(HEADER + "PORO\n 0.1 0.2 /").get("PORO").tolist() == pytest.approx([0.1, 0.2])


def test_tables_with_one_or_several_rows_read_the_same():
    one = DeckFile("FIELD\nPVDO\n 100 1.2 0.8 /\n")
    two = DeckFile("FIELD\nPVDO\n 100 1.2 0.8\n 200 1.1 0.9 /\n")
    assert len(one.get("PVDO")[0]) == 1
    assert len(two.get("PVDO")[0]) == 2

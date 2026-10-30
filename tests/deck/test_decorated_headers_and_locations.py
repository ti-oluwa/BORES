import pytest

from bores.deck import DeckFile
from bores.deck.core import DeckParseError
from bores.deck.keywords import DISGASW, OLDTRAN, RUNSUM

DECK = """RUNSPEC
FIELD
DIMENS
 1 1 1 /
GRID
OLDTRAN
PROPS
ROCK
   6000  0.000001
/

SOLUTION ==========================================
SUMMARY ===========================================
RUNSUM
"""


def test_header_with_trailing_equals_ends_previous_keyword():
    deck_file = DeckFile(DECK)
    rock = deck_file.get("ROCK")
    assert rock == [[{"reference_pressure": 6000.0, "compressibility": 1e-06}]]


def test_new_flag_keywords_are_read():
    deck_file = DeckFile(DECK + "\n")
    assert deck_file.get(OLDTRAN) is True
    assert deck_file.get(RUNSUM) is True
    assert deck_file.get(DISGASW) is None


def test_records_carry_line_numbers():
    deck_file = DeckFile(DECK)
    lines = {record.keyword: record.line for record in deck_file.deck.records}
    assert lines["RUNSPEC"] == 1
    assert lines["ROCK"] == 8


def test_parse_error_names_the_line():
    deck_file = DeckFile("FIELD\nDIMENS\n 1 1 1 /\nROCK\n 6000 abc /\n/\n")
    with pytest.raises(DeckParseError, match=r"ROCK` \(line 4\)"):
        deck_file.get("ROCK")

"""`RPTRST`/`RPTSCHED` occurrences, case-insensitive keywords and encoding errors."""

import pytest

from bores.deck import DeckFile
from bores.deck import keywords as k
from bores.deck.core import DeckParseError

HEADER = "RUNSPEC\nFIELD\nSPECGRID\n 10 10 3 1 F /\n"


def parse(body: str, keyword):
    return DeckFile(HEADER + body, encoding="utf-8").get(keyword)


def test_every_occurrence_is_read_with_its_schedule_time():
    records = parse(
        "RPTRST\n 'BASIC=1' /\nTSTEP\n 5 /\nRPTRST\n 'BASIC=2' 'FREQ=3' ALLPROPS /\n", k.RPTRST
    )
    assert records == [
        {"mnemonics": {"BASIC": 1}, "flags": [], "schedule_time": 0.0},
        {
            "mnemonics": {"BASIC": 2, "FREQ": 3, "ALLPROPS": None},
            "flags": [],
            "schedule_time": 5.0,
        },
    ]


def test_mnemonics_are_upper_cased_and_empty_occurrence_is_kept():
    (record,) = parse("RPTSCHED\n 'pres' 'fip=2' /\n", k.RPTSCHED)
    assert record["mnemonics"] == {"PRES": None, "FIP": 2}
    (record,) = parse("RPTRST\n /\n", k.RPTRST)
    assert record["mnemonics"] == {} and record["flags"] == []


def test_positional_integer_form_keeps_order_and_defaults():
    (record,) = parse("RPTSCHED\n 1 1 1 0 0 0 /\n", k.RPTSCHED)
    assert record["flags"] == [1, 1, 1, 0, 0, 0] and record["mnemonics"] == {}
    (record,) = parse("RPTRST\n 1 1* 0 2* /\n", k.RPTRST)
    assert record["flags"] == [1, None, 0, None, None]


def test_mixed_forms_and_bad_levels_are_rejected():
    with pytest.raises(DeckParseError, match="cannot be mixed"):
        parse("RPTRST\n 1 BASIC=2 /\n", k.RPTRST)
    with pytest.raises(DeckParseError, match="non-integer"):
        parse("RPTRST\n BASIC=x /\n", k.RPTRST)


def test_keywords_are_case_insensitive():
    deck = DeckFile(
        "runspec\nfield\nSpecGrid\n 2 2 1 1 F /\nporo\n4*0.25 /\nbpr\n1 1 1 /\n/\ninit\n",
        encoding="utf-8",
    )
    assert deck.deck.keywords == ["RUNSPEC", "FIELD", "SPECGRID", "PORO", "BPR", "INIT"]
    assert list(deck.get("PORO")) == [0.25] * 4
    assert deck.get(k.BPR) == [{"i": 1, "j": 1, "k": 1}]
    assert deck.get(k.INIT) is True


def test_title_text_is_not_mistaken_for_a_keyword_in_any_case():
    deck = DeckFile("title\nspe1\nrunspec\nfield\n", encoding="utf-8")
    assert deck.deck.keywords == ["TITLE", "RUNSPEC", "FIELD"]


def test_wrong_encoding_error_says_what_to_do(tmp_path):
    path = tmp_path / "case.data"
    path.write_bytes("RUNSPEC\n-- caf\u00e9\nFIELD\n".encode("utf-8"))
    for source in (path, str(path), path.read_bytes()):
        with pytest.raises(DeckParseError, match="pass it as `encoding=`") as info:
            DeckFile(source)
        assert "ascii" in str(info.value)
    assert DeckFile(path, encoding="utf-8").deck.keywords == ["RUNSPEC", "FIELD"]


def test_unknown_encoding_is_reported():
    with pytest.raises(DeckParseError, match="not a known text encoding"):
        DeckFile(b"RUNSPEC\n", encoding="no-such-codec")

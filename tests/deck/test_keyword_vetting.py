import datetime

import numpy as np
import pytest

from bores.deck import DeckFile
from bores.deck.core import DeckParseError
from bores.deck.keywords.base import timedelta_to_timeunit


def make(body: str) -> DeckFile:
    return DeckFile("METRIC\n" + body + "\n")


def test_commented_include_is_not_followed(tmp_path):
    (tmp_path / "main.DATA").write_text("METRIC\n-- INCLUDE 'missing.inc' /\nOIL\n")
    deck_file = DeckFile(tmp_path / "main.DATA")
    assert deck_file.get("OIL") is True


def test_title_that_looks_like_a_keyword_is_kept():
    deck_file = make("TITLE\nSPE1\nOIL")
    assert deck_file.get("TITLE") == "SPE1"
    assert deck_file.get("OIL") is True


def test_bare_repeat_in_array_is_dropped_with_warning():
    deck_file = make("SPECGRID\n2 1 1 /\nPORO\n2*0.3 1* /")
    with pytest.warns(UserWarning, match="bare 'N\\*'"):
        poro = deck_file.get("PORO")
    assert poro.tolist() == [0.3, 0.3]


def test_equals_with_defaulted_box_bounds_keeps_ambient_box():
    deck_file = make("SPECGRID\n2 2 1 /\nPORO\n4*0.1 /\nEQUALS\n'PORO' 0.5 2 2 1* 1* 1* 1* /\n/")
    assert deck_file.get("PORO").tolist() == [0.1, 0.5, 0.1, 0.5]


def test_wecon_reads_min_gas_rate_before_water_cut():
    deck_file = make("WECON\n'W1' 10 20 0.9 1* 1* NONE NO /\n/")
    (record,) = deck_file.get("WECON")
    assert record["min_oil_rate"] == 10
    assert record["min_gas_rate"] == 20
    assert record["max_water_cut"] == pytest.approx(0.9)
    assert record["max_gor"] is None
    assert record["workover_action"] == "NONE"


def test_welspecs_defaults_and_aliases():
    deck_file = make("WELSPECS\n'W1' 'G' 1 1 1* WAT /\n'W2' 'G' 1 1 1* LIQ /\n/")
    first, second = deck_file.get("WELSPECS")
    assert first["phase"] == "WATER"
    assert second["phase"] == "LIQ"
    assert first["enable_crossflow"] is True


def test_welspecs_rejects_unknown_phase():
    with pytest.raises(DeckParseError):
        make("WELSPECS\n'W1' 'G' 1 1 1* LIO /\n/").get("WELSPECS")


def test_saturation_table_rejects_values_outside_unit_interval():
    deck_file = make("SWOF\n0.2 0 1 0 /\n1.4 1 0 0 /\n/")
    with pytest.raises(DeckParseError):
        deck_file.get("SWOF")


def test_saturation_table_accepts_round_off():
    deck_file = make("SWOF\n0.2 0 1.0000000001 0 /\n1 1 0 0 /\n/")
    assert len(deck_file.get("SWOF")[0]) == 2


def test_fault_face_accepts_ijk_spellings():
    deck_file = make("FAULTS\n'F' 1 1 1 2 1 2 I /\n'G' 1 1 1 2 1 2 K- /\n/")
    first, second = deck_file.get("FAULTS")
    assert (first["face"], second["face"]) == ("X", "Z-")


def test_pinch_follows_eclipse_item_order():
    deck_file = make("PINCH\n0.001 NOGAP 5 ALL ALL /")
    record = deck_file.get("PINCH")
    assert record["gap_mode"] == "NOGAP"
    assert record["max_empty_gap"] == 5
    assert record["pinchout_option"] == "ALL"
    assert record["multz_option"] == "ALL"
    assert "composition_option" not in record


def test_gridunit_validates_unit():
    assert make("GRIDUNIT\n'METRES' /").get("GRIDUNIT")["unit"] == "METRES"
    with pytest.raises(DeckParseError):
        make("GRIDUNIT\n'YARDS' /").get("GRIDUNIT")


def test_wpave_uses_eclipse_items():
    # The depth correction is the third item and the connections flag the fourth, as in OPM's
    # definition and in Norne (`WPAVE 1* 0.000 'WELL' 'ALL' /`).
    record = make("WPAVE\n0.5 1 RES ALL /").get("WPAVE")
    assert (record["connections"], record["depth_correction"]) == ("ALL", "RES")


def test_summary_vector_requested_twice_is_joined():
    deck_file = make("WOPR\n'A' /\nWOPR\n'B' /")
    assert deck_file.get("WOPR") == ["A", "B"]
    assert make("WOPR\n'A' /\nWOPR\n/").get("WOPR") == []


def test_date_difference_in_each_time_unit():
    delta = datetime.date(2020, 1, 3) - datetime.date(2020, 1, 1)
    assert timedelta_to_timeunit(delta, "days") == 2
    assert timedelta_to_timeunit(delta, "hours") == 48
    assert timedelta_to_timeunit(delta, "seconds") == 2 * 86400


def test_tokenize_fast_path_matches_quoted_path():
    from bores.deck.core import tokenize

    assert tokenize("3*1.5 2") == ["1.5", "1.5", "1.5", "2"]
    assert tokenize("2*'A B' C") == ["A B", "A B", "C"]
    assert np.allclose([float(t) for t in tokenize("1 2")], [1, 2])


def test_welopen_and_wpimult_use_single_k_and_completion_range():
    welopen = make("WELOPEN\n'W1' SHUT 2 3 4 1 5 /\n/").get("WELOPEN")[0]
    assert (welopen["i"], welopen["j"], welopen["k"]) == (2, 3, 4)
    assert (welopen["c1"], welopen["c2"]) == (1, 5)
    wpimult = make("WPIMULT\n'W1' 0.5 2 3 4 1 5 /\n/").get("WPIMULT")[0]
    assert (wpimult["k"], wpimult["c1"], wpimult["c2"]) == (4, 1, 5)


def test_connection_layer_rejects_completion_numbers():
    from bores.errors import NotSupportedError
    from bores.wells.deck import connection_layers

    assert connection_layers({"well": "W1", "k": 3}, keyword="WELOPEN") == (3, 3)
    with pytest.raises(NotSupportedError):
        connection_layers({"well": "W1", "k": 0, "c1": 2, "c2": 2}, keyword="WELOPEN")


def test_wconinjh_control_mode_is_item_twelve():
    deck_file = make("WCONINJH\n'I1' WATER OPEN 500 1* 1* 1* 1* 1* 1* 1* BHP /\n/")
    (record,) = deck_file.get("WCONINJH")
    assert record["control_mode"] == "BHP"
    assert record["rate"] == pytest.approx(500)


def test_wconhist_control_mode_cannot_be_defaulted():
    with pytest.raises(DeckParseError):
        make("WCONHIST\n'P1' OPEN 1* 100 /\n/").get("WCONHIST")


def test_wtest_reason_and_extra_items():
    (record,) = make("WTEST\n'P1' 30 PEGDC 3 5 /\n/").get("WTEST")
    assert (record["reason"], record["test_count"]) == ("PEGDC", 3)
    assert make("WTEST\n'P1' 30 /\n/").get("WTEST")[0]["reason"] == ""


def test_gravity_regions_and_defaults():
    first, second = make("GRAVITY\n35 1.02 0.8 /\n30 /\n/").get("GRAVITY")
    assert first["gas_specific_gravity"] == pytest.approx(0.8)
    assert second["water_specific_gravity"] == pytest.approx(1.0)
    assert second["gas_specific_gravity"] == pytest.approx(0.7773)


def test_ehystr_options():
    record = make("EHYSTR\n0.2 1 1* 1* PC RETR IMBIB WATER /").get("EHYSTR")
    assert record["limiting_curves"] == "PC"
    assert record["wetting_phase"] == "WATER"
    with pytest.raises(DeckParseError):
        make("EHYSTR\n0.2 1 1* 1* BAD /").get("EHYSTR")

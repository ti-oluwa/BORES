import pytest

from bores.deck import DeckFile


def read(body: str, keyword: str):
    return DeckFile("METRIC\n" + body).get(keyword)


# --- positions and defaults confirmed against OPM's keyword definitions -----------------------


def test_wpave_item_order_follows_the_deck_norne_writes_depth_correction_before_connections():
    # Norne: `WPAVE 1* 0.000 'WELL' 'ALL' /`
    record = read("WPAVE\n 1* 0.000 'WELL' 'ALL' /\n", "WPAVE")
    assert record["f1"] == pytest.approx(0.5)  # defaulted
    assert record["f2"] == pytest.approx(0.0)
    assert record["depth_correction"] == "WELL"
    assert record["connections"] == "ALL"


def test_wpave_defaults():
    record = read("WPAVE\n /\n", "WPAVE")
    assert (record["f1"], record["f2"]) == (pytest.approx(0.5), pytest.approx(1.0))
    assert (record["depth_correction"], record["connections"]) == ("WELL", "OPEN")


def test_regdims_item_five_is_max_etrack_and_nplmix_is_the_tenth():
    record = read("REGDIMS\n 1 2 3 4 5 6 7 8 9 10 /\n", "REGDIMS")
    assert record["max_etrack"] == 5
    assert record["ntcreg"] == 6
    assert record["nplmix"] == 10
    assert read("REGDIMS\n /\n", "REGDIMS")["nmfip"] == 1


def test_dimension_defaults():
    assert read("EQLDIMS\n /\n", "EQLDIMS")["ndprvd"] == 2000
    assert read("EQLDIMS\n /\n", "EQLDIMS")["nttrvd"] == 1
    welldims = read("WELLDIMS\n /\n", "WELLDIMS")
    assert [welldims[name] for name in ("nwmaxz", "ncwmax", "ngmaxz", "nwgmax")] == [0, 0, 0, 0]
    assert welldims["max_separator_stages"] == 5 and welldims["max_gpp_table_rows"] == 201


def test_defaults_of_items_that_used_to_be_required():
    assert read("GRIDUNIT\n /\n", "GRIDUNIT")["unit"] == "METRES"
    assert read("MAPUNITS\n /\n", "MAPUNITS")["unit"] == "METRES"
    assert read("GRUPTREE\n 'G1' /\n", "GRUPTREE")[0]["parent"] == "FIELD"
    assert read("WELOPEN\n 'W1' /\n", "WELOPEN")[0]["status"] == "OPEN"
    assert read("WPIMULT\n 'W1' /\n", "WPIMULT")[0]["multiplier"] == pytest.approx(1.0)
    assert read("AQUCT\n 1 2000 1* 100 1* 1e-5 1000 10 /\n", "AQUCT")[0][
        "porosity"
    ] == pytest.approx(1.0)


# --- trailing items that used to be dropped silently --------------------------------------------


def test_tabdims_items_after_the_sixth_are_read():
    record = read("TABDIMS\n 1 1 20 20 1 20 30 7 /\n", "TABDIMS")
    assert record["max_vaporized_oil_ratio_nodes"] == 30
    assert record["max_endpoint_depth_tables"] == 7


def test_welldims_items_after_the_fourth_are_read():
    record = read("WELLDIMS\n 10 20 3 5 1 2 3 4 5 6 7 8 9 10 /\n", "WELLDIMS")
    assert record["nwgmax"] == 5
    assert record["max_separator_stages"] == 1
    assert record["max_gpp_table_rows"] == 10


def test_well_keywords_read_their_later_items():
    welspecs = read(
        "WELSPECS\n 'W1' 'G' 1 1 1* OIL 0 STD SHUT YES 0 SEG 2 1* 1* STD /\n", "WELSPECS"
    )[0]
    assert welspecs["fluid_in_place_region"] == 2
    assert welspecs["well_model"] == "STD"
    wconhist = read("WCONHIST\n 'W1' OPEN ORAT 1000 0 0 1* 0 0 0 7 8 /\n", "WCONHIST")[0]
    assert wconhist["wet_gas_rate"] == pytest.approx(7.0)
    assert wconhist["natural_gas_liquid_rate"] == pytest.approx(8.0)


def test_unit_dimensioned_defaults_are_not_filled_in_with_opms_metric_values():
    # OPM states these in its default (metric) units, which would be wrong in a FIELD deck, so a
    # defaulted item stays `None` for the consumer to resolve in the deck's own units.
    assert read("WCONPROD\n 'P1' OPEN ORAT 100 /\n", "WCONPROD")[0]["bhp"] is None
    assert read("PINCH\n /\n", "PINCH")["max_empty_gap"] is None

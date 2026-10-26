"""Deck support for `INIT`, `DRSDT`, `DRVDT` and the `B*` block summary vectors."""

import pathlib
import warnings

import numpy as np
import pytest

from bores.deck import DeckFile
from bores.deck import keywords as k
from bores.deck.core import DeckParseError
from bores.errors import SummaryError
from bores.simulation.case import SimulationCase
from bores.types import UnitSystem
from bores.wells import summary as well_summary
from bores.wells.deck import BLOCK_SUMMARY_VECTORS, load_summary_items
from bores.wells.hydraulics.homogeneous import homogeneous_wellbore

SPE1_DECK = pathlib.Path(__file__).parents[2] / "data" / "SPE1CASE1.DATA"
HEADER = "RUNSPEC\nFIELD\nSPECGRID\n 10 10 3 1 F /\n"


def parse(body: str, keyword):
    return DeckFile(HEADER + body, encoding="utf-8").get(keyword)


@pytest.mark.parametrize("mnemonic", sorted(BLOCK_SUMMARY_VECTORS))
def test_every_block_mnemonic_parses_cells(mnemonic):
    cells = parse(f"{mnemonic}\n1 1 1 /\n10 10 3 /\n/\n", mnemonic)
    assert cells == [{"i": 1, "j": 1, "k": 1}, {"i": 10, "j": 10, "k": 3}]


def test_block_cells_repeat_across_lines_and_blocks():
    assert parse("BPR\n1 1 1 /\n2 2 2 /\n/\nBPR\n3 3 3 /\n/\n", k.BPR) == [
        {"i": 1, "j": 1, "k": 1},
        {"i": 2, "j": 2, "k": 2},
        {"i": 3, "j": 3, "k": 3},
    ]


def test_a_second_cell_on_the_same_line_is_ignored_as_eclipse_does_with_a_warning():
    # Whatever follows a record's closing `/` on its line is a note to Eclipse and to OPM, so a
    # second cell written there is not read. Silently dropping it would be worse, so it is said.
    with pytest.warns(UserWarning, match="'2 2 2 /' looks like another record"):
        cells = parse("BPR\n1 1 1 / 2 2 2 /\n/\n", k.BPR)
    assert cells == [{"i": 1, "j": 1, "k": 1}]


def test_block_cell_missing_an_index_is_rejected():
    with pytest.raises(DeckParseError):
        parse("BPR\n1 1 /\n/\n", k.BPR)


def test_block_keyword_with_no_cells_reads_as_absent():
    assert parse("BPR\n/\n", k.BPR) is None


def test_init_is_a_flag():
    assert parse("INIT\nPORO\n300*0.2 /\n", k.INIT) is True
    assert parse("PORO\n300*0.2 /\n", k.INIT) is None


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("0 /", (0.0, "ALL")),
        ("0 ALL /", (0.0, "ALL")),
        ("0.5 free /", (0.5, "FREE")),
        ("0.5 'FREE' /", (0.5, "FREE")),
        ("0.5 1* /", (0.5, "ALL")),
        ("1e-4 /", (1e-4, "ALL")),
    ],
)
def test_drsdt_variants(body, expected):
    (record,) = parse(f"DRSDT\n {body}\n", k.DRSDT)
    assert (record["max_rate"], record["option"]) == expected


def test_drsdt_rejects_bad_option_and_defaulted_rate():
    with pytest.raises(DeckParseError):
        parse("DRSDT\n 0 BAD /\n", k.DRSDT)
    with pytest.raises(DeckParseError):
        parse("DRSDT\n 1* ALL /\n", k.DRSDT)


def test_drsdt_and_drvdt_are_stamped_with_their_schedule_time():
    records = parse(
        "START\n 1 JAN 2020 /\nDRSDT\n 0 /\nDATES\n 1 FEB 2020 /\n/\nDRSDT\n 0.2 FREE /\nDRVDT\n 0 /\n",
        k.DRSDT,
    )
    assert [(r["max_rate"], r["option"], r["schedule_time"]) for r in records] == [
        (0.0, "ALL", 0.0),
        (0.2, "FREE", 31.0),
    ]
    assert parse("DRVDT\n 0 /\n", k.DRVDT) == [{"max_rate": 0.0, "schedule_time": 0.0}]


def test_summary_loader_places_block_vectors_and_dedupes():
    deck = DeckFile(
        HEADER
        + "SUMMARY\nBPR\n1 1 1 /\n1 1 1 /\n/\nBPRESSUR\n1 1 1 /\n2 1 1 /\n/\nBGSAT\n10 10 3 /\n/\n"
        + "SCHEDULE\nTSTEP\n 1 /\n",
        encoding="utf-8",
    )
    (item,) = load_summary_items(deck)
    assert [quantity.key for quantity in item.action.quantities] == [
        "BPR:1,1,1",
        "BPR:2,1,1",
        "BGSAT:10,10,3",
    ]


def test_summary_loader_rejects_cells_outside_the_grid():
    deck = DeckFile(
        "RUNSPEC\nFIELD\nSPECGRID\n 2 2 2 1 F /\nSUMMARY\nBPR\n3 1 1 /\n/\nSCHEDULE\nTSTEP\n 1 /\n",
        encoding="utf-8",
    )
    with pytest.raises(DeckParseError, match="outside"):
        load_summary_items(deck)


@pytest.fixture(scope="module")
def spe1():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deck = DeckFile(SPE1_DECK, encoding="utf-8")
        wellbore = homogeneous_wellbore(tubing_inner_diameter=2.5, unit_system=UnitSystem.FIELD)
        case = SimulationCase.from_deck(
            deck,
            tubing_performance=wellbore,
            temperature=200.0,
            salinity=np.full(300, 30000.0),
        )
        workspace = case.build_workspace()

    class Context:
        extra = {"workspace": workspace}

    return deck, case.model, Context()


def test_spe1_deck_keywords(spe1):
    deck, _, _ = spe1
    assert deck.get(k.INIT) is True
    assert deck.get(k.RPTSCHED) == [
        {
            "mnemonics": {"PRES": None, "SGAS": None, "RS": None, "WELLS": None},
            "flags": [],
            "schedule_time": 0.0,
        }
    ]
    assert deck.get(k.RPTRST) == [{"mnemonics": {"BASIC": 1}, "flags": [], "schedule_time": 0.0}]
    assert deck.get(k.DRSDT) == [{"max_rate": 0.0, "option": "ALL", "schedule_time": 0.0}]


def test_spe1_block_vectors_report_the_initial_state(spe1):
    deck, model, context = spe1
    quantities = {
        quantity.key: quantity
        for quantity in load_summary_items(deck)[0].action.quantities
        if quantity.key.startswith("B")
    }
    assert len(quantities) == 2 + 9
    assert quantities["BPR:10,10,3"](model, context) == pytest.approx(4800.0)
    assert 4700.0 < quantities["BPR:1,1,1"](model, context) < 4800.0
    # SPE1 starts undersaturated, so there is no free gas anywhere.
    assert all(
        quantity(model, context) == 0.0 for key, quantity in quantities.items() if "BGSAT" in key
    )

    cell = {"i": 9, "j": 9, "k": 2}
    assert well_summary.BlockWaterSaturation(**cell)(model, context) == pytest.approx(0.12)
    assert well_summary.BlockOilSaturation(**cell)(model, context) == pytest.approx(0.88)
    assert well_summary.BlockGasOilRatio(**cell)(model, context) == pytest.approx(1270.0)
    assert well_summary.BlockVaporizedOilGasRatio(**cell)(model, context) == 0.0
    assert well_summary.BlockOilRelativePermeability(**cell)(model, context) == pytest.approx(1.0)
    assert well_summary.BlockGasRelativePermeability(**cell)(model, context) == 0.0
    assert well_summary.BlockWaterRelativePermeability(**cell)(model, context) == 0.0
    assert well_summary.BPRESSUR is well_summary.BPR


def test_block_vector_outside_the_grid_raises(spe1):
    _, model, context = spe1
    with pytest.raises(SummaryError, match="outside"):
        well_summary.BlockPressure(i=10, j=0, k=0)(model, context)

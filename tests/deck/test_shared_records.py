"""`DeckFile.get_shared` and the loaders that rely on it to read each keyword only once."""

import collections
import pathlib
from unittest import mock

from bores.blackoil.satfunc.regions import SatFunc
from bores.deck import DeckFile
from bores.deck.file import DEFAULT_KEYWORDS
from bores.deck.keywords.grid import PORO
from bores.simulation.case import load_case
from bores.types import UnitSystem

DATA_DIRECTORY = pathlib.Path(__file__).resolve().parents[2] / "data"
GRID = "METRIC\nSPECGRID\n 2 2 1 1 F /\n"


class GetCounter:
    """Counts `DeckFile.get` calls by keyword name while active."""

    def __init__(self):
        self.counts: collections.Counter[str] = collections.Counter()
        self.original = DeckFile.get

    def __enter__(self):
        counter, original = self, self.original

        def counting(deck, k, /, **kwargs):
            counter.counts[k.upper() if isinstance(k, str) else k.name] += 1
            return original(deck, k, **kwargs)

        self.patch = mock.patch.object(DeckFile, "get", counting)
        self.patch.start()
        return self

    def __exit__(self, *exc):
        self.patch.stop()


class TestGetShared:
    def test_returns_the_same_object_and_parses_once(self):
        deck = DeckFile(GRID + "SWOF\n 0.2 0 1 0\n 1 1 0 0 /\n")
        with GetCounter() as counter:
            first = deck.get_shared("swof")
            second = deck.get_shared("SWOF")
        assert first is second
        assert counter.counts["SWOF"] == 1
        assert first == deck.get("SWOF")

    def test_absent_keyword_is_none_and_remembered(self):
        deck = DeckFile(GRID)
        with GetCounter() as counter:
            assert deck.get_shared("SWOF") is None
            assert deck.get_shared("SWOF") is None
        assert counter.counts["SWOF"] == 1

    def test_get_still_returns_a_value_of_its_own(self):
        deck = DeckFile(GRID + "SWOF\n 0.2 0 1 0\n 1 1 0 0 /\n")
        shared = deck.get_shared("SWOF")
        assert deck.get("SWOF") is not shared

    def test_registering_a_keyword_drops_what_was_remembered_for_it(self):
        deck = DeckFile(
            GRID + "PORO\n 4*0.2 /\n",
            keywords=[k for k in DEFAULT_KEYWORDS if k.name != "PORO"],
        )
        assert deck.get_shared("PORO") is None
        deck.add_keywords(PORO)
        assert deck.get_shared("PORO") is not None

    def test_a_dimension_change_drops_everything(self):
        deck = DeckFile(GRID + "PORO\n 4*0.2 /\n")
        before = deck.get_shared("PORO")
        deck.add_keywords(*[k for k in deck._registry.values() if k.name == "SPECGRID"])
        assert deck.get_shared("PORO") is not before


def three_region_deck() -> DeckFile:
    swof = "".join(" 0.2 0 1 0\n 0.6 0.3 0.3 0\n 1 1 0 0 /\n" for _ in range(3))
    sgof = "".join(" 0 0 1 0\n 0.4 0.2 0.2 0\n 0.8 1 0 0 /\n" for _ in range(3))
    return DeckFile(GRID + "SWOF\n" + swof + "SGOF\n" + sgof)


def test_satfunc_reads_each_keyword_once_however_many_regions():
    deck = three_region_deck()
    with GetCounter() as counter:
        satfunc = SatFunc.from_deck(deck)
    assert len(satfunc) == 3
    assert counter.counts["SWOF"] == 1 and counter.counts["SGOF"] == 1


def test_satfunc_result_is_unchanged_by_sharing():
    shared = SatFunc.from_deck(three_region_deck())
    with mock.patch.object(DeckFile, "get_shared", lambda self, k, /: self.get(k)):
        fresh = SatFunc.from_deck(three_region_deck())
    assert shared.tables.keys() == fresh.tables.keys()
    for satnum in shared.tables:
        a = shared.tables[satnum].relative_permeability
        b = fresh.tables[satnum].relative_permeability
        for system in ("oil_water_table", "gas_oil_table"):
            table_a, table_b = getattr(a, system), getattr(b, system)
            assert (table_a.reference_saturation == table_b.reference_saturation).all()
            assert (
                table_a.wetting_phase_relative_permeability
                == table_b.wetting_phase_relative_permeability
            ).all()


TWO_AQUIFERS = (
    "\nAQUCT\n 1 9035 4800 100 0.25 3.0E-6 5000 100 360 1 1 /\n"
    " 2 9035 4800 100 0.25 3.0E-6 5000 100 360 1 1 /\n/\n\n"
    "AQUANCON\n 1 10 10 1 10 1 3 I+ 1* 1.0 NO /\n 2 1 1 1 10 1 3 I- 1* 1.0 NO /\n/\n\n"
)


def spe1_with_two_aquifers() -> DeckFile:
    text = (DATA_DIRECTORY / "SPE1CASE1.DATA").read_text()
    assert "\nSUMMARY" in text
    return DeckFile(
        text.replace("\nSUMMARY", TWO_AQUIFERS + "\nSUMMARY", 1), unit_system=UnitSystem.FIELD
    )


def test_a_full_case_reads_each_record_keyword_once_and_never_alters_shared_records():
    deck = spe1_with_two_aquifers()
    with GetCounter() as counter:
        case = load_case(deck, temperature=200.0)
    assert len(case.model.boundary_conditions.aquifers.initial_pressures) == 2

    for keyword in (
        "WELSPECS",
        "COMPDAT",
        "WCONPROD",
        "WCONINJE",
        "SWOF",
        "SGOF",
        "AQUCT",
        "AQUANCON",
    ):
        assert counter.counts[keyword] == 1, keyword

    # A loader that changed a shared record would make it differ from a fresh parse
    assert deck._shared, "nothing was shared"
    for keyword, value in deck._shared.items():
        assert value == deck.get(keyword), keyword

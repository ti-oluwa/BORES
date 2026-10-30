"""`DeckFile.has_many` / `has_any` / `missing`, `find_missing`, and the fail-fast checks built on them."""

from unittest import mock

import pytest

from bores.deck.file import DeckFile
from bores.deck.keywords.grid import PORO
from bores.deck.precheck import find_missing
from bores.errors import GridImportError, ValidationError
from bores.grids.base import Grid
from bores.reservoir.rock.model import Rock
from bores.simulation.case import CaseLoadError, check_deck_requirements

GRID = "METRIC\nSPECGRID\n 2 2 1 1 F /\n"


def deck(body: str = "") -> DeckFile:
    return DeckFile(GRID + body)


class TestHasMany:
    def test_maps_each_name_to_presence_in_order_given(self):
        d = deck("PORO\n 4*0.2 /\nPERMX\n 4*100 /\n")
        assert list(d.has_many("PERMX", "NTG", "PORO").items()) == [
            ("PERMX", True),
            ("NTG", False),
            ("PORO", True),
        ]

    def test_names_are_case_insensitive_and_keyword_objects_are_accepted(self):
        d = deck("PORO\n 4*0.2 /\n")
        assert d.has_many("poro", PORO) == {"PORO": True}

    def test_duplicates_collapse(self):
        assert list(deck().has_many("A", "a", "B")) == ["A", "B"]

    def test_operator_only_targets_count_as_present(self):
        d = deck("EQUALS\n 'DZ' 10 /\n/\n")
        assert d.has_many("DZ", "DX") == {"DZ": True, "DX": False}
        assert d.has("DZ") is d.has_many("DZ")["DZ"]

    def test_agrees_with_has_for_every_keyword(self):
        d = deck("PORO\n 4*0.2 /\nEQUALS\n 'DZ' 10 /\n/\n")
        names = ["PORO", "DZ", "PERMX", "SPECGRID", "NOPE"]
        assert d.has_many(*names) == {n: d.has(n) for n in names}

    def test_does_not_parse_anything(self):
        d = deck("PORO\n 4*0.2 /\n")
        with mock.patch.object(type(PORO), "parse", side_effect=AssertionError("parsed")):
            d.has_many("PORO", "PERMX")
            d.missing("PORO")
            d.has_any("PORO")

    def test_has_any_and_missing(self):
        d = deck("PORO\n 4*0.2 /\n")
        assert d.has_any("NTG", "PORO") and not d.has_any("NTG", "PERMX")
        assert d.missing("PORO", "ntg", "PERMX") == ["NTG", "PERMX"]
        assert d.missing("PORO") == []


class TestFindMissing:
    def test_all_of_and_any_of(self):
        d = deck("DX\n 4*10 /\nPORO\n 4*0.2 /\n")
        assert find_missing(d, all_of=["PORO"], any_of=[("DX", "DXV")]) == []
        assert find_missing(d, all_of=["PORO", "PERMX"], any_of=[("DY", "DYV"), ("DX",)]) == [
            "`PERMX`",
            "one of `DY`, `DYV`",
        ]

    def test_empty_inputs_are_satisfied(self):
        assert find_missing(deck()) == []

    def test_a_section_header_is_not_a_requirement_in_itself(self):
        # a keyword-only include has no `PROPS` header; the keywords are what matter
        d = deck("PORO\n 4*0.2 /\n")
        assert not d.has("PROPS")
        assert find_missing(d, all_of=["PORO"]) == []


class TestFailsBeforeParsing:
    def test_rock_reports_missing_poro_without_parsing_permx(self):
        d = deck("PERMX\n 4*100 /\n")
        grid = mock.Mock(n_cells=4)
        with mock.patch("bores.reservoir.rock.model.load_cell_array") as load:
            with pytest.raises(ValidationError, match="`PORO` is required"):
                Rock.from_deck(d, grid=grid)
        load.assert_not_called()

    def test_rock_lists_every_missing_keyword(self):
        with pytest.raises(ValidationError, match="`PORO`, `PERMX` are required"):
            Rock.from_deck(deck(), grid=mock.Mock(n_cells=4))

    def test_cartesian_grid_missing_dy_is_reported_before_dx_is_parsed(self):
        d = deck("DX\n 4*10 /\nDZ\n 4*5 /\nTOPS\n 4*1000 /\n")
        with mock.patch.object(DeckFile, "get", side_effect=AssertionError("parsed")):
            with pytest.raises(GridImportError, match="`DY`, `DYV`"):
                Grid.from_deck(d)

    def test_corner_point_grid_missing_zcorn(self):
        d = deck("COORD\n 1 /\n")
        with pytest.raises(GridImportError, match="`ZCORN`"):
            Grid.from_deck(d)

    def test_grid_with_no_geometry_keeps_its_message(self):
        with pytest.raises(GridImportError, match="neither `COORD`"):
            Grid.from_deck(deck())


class TestCaseRequirements:
    def check(self, d: DeckFile, **needs: bool):
        defaults = dict(
            needs_temperature=True,
            needs_grid=True,
            needs_satfunc=True,
            needs_rock=True,
            needs_pvt=True,
            needs_equilibrium=True,
            needs_wells=True,
        )
        defaults.update(needs)
        check_deck_requirements(d, title="t", **defaults)

    def test_reports_every_missing_stage_at_once(self):
        with pytest.raises(CaseLoadError) as info:
            self.check(deck())
        message = str(info.value)
        for fragment in (
            "grid:",
            "temperature: one of `TEMPVD`",
            "saturation functions: one of `SWOF`",
            "rock: `PORO`",
            "PVT: one of `PVTO`",
            "initial state: `EQUIL`",
        ):
            assert fragment in message

    def test_supplied_components_are_not_required(self):
        self.check(
            deck(),
            needs_temperature=False,
            needs_grid=False,
            needs_satfunc=False,
            needs_rock=False,
            needs_pvt=False,
            needs_equilibrium=False,
            needs_wells=False,
        )

    def test_a_deck_with_everything_passes(self):
        body = (
            "DX\n 4*10 /\nDY\n 4*10 /\nDZ\n 4*5 /\nPORO\n 4*0.2 /\nPERMX\n 4*100 /\n"
            "RTEMP\n 80 /\nSWOF\n 0.2 0 1 0\n 1 1 0 0 /\nPVDO\n 100 1.2 1 /\nEQUIL\n 1000 300 1100 0 900 0 /\nWELSPECS\n 'P' 'G' 1 1 1* OIL /\n/\nCOMPDAT\n 'P' 1 1 1 1 OPEN /\n/\n"
        )
        self.check(deck(body))

    def test_missing_dimensions_is_reported(self):
        with pytest.raises(CaseLoadError, match="`SPECGRID` or `DIMENS`"):
            self.check(
                DeckFile("METRIC\n"),
                needs_temperature=False,
                needs_satfunc=False,
                needs_rock=False,
                needs_pvt=False,
                needs_equilibrium=False,
            )


class TestWellsAreOptional:
    WELLS = "WELSPECS\n 'P' 'G' 1 1 1* OIL /\n/\nCOMPDAT\n 'P' 1 1 1 1 OPEN /\n/\n"

    def test_a_deck_with_no_well_keywords_has_no_wells(self):
        from bores.wells.deck import deck_defines_wells

        assert deck_defines_wells(deck()) is False
        assert deck_defines_wells(deck(self.WELLS)) is True

    def test_a_deck_without_wells_passes_the_case_precheck(self):
        TestCaseRequirements().check(
            deck(),
            needs_temperature=False,
            needs_grid=False,
            needs_satfunc=False,
            needs_rock=False,
            needs_pvt=False,
            needs_equilibrium=False,
        )

    @pytest.mark.parametrize(
        "body, fragment",
        [
            ("WELSPECS\n 'P' 'G' 1 1 1* OIL /\n/\n", "`COMPDAT` is missing"),
            ("COMPDAT\n 'P' 1 1 1 1 OPEN /\n/\n", "`WELSPECS` is missing"),
            (
                "WCONPROD\n 'P' OPEN BHP 5* 100 /\n/\n",
                "`WCONPROD` is present but the deck defines no wells",
            ),
        ],
    )
    def test_inconsistent_well_keywords_are_reported_up_front(self, body, fragment):
        with pytest.raises(CaseLoadError, match=fragment):
            TestCaseRequirements().check(
                deck(body),
                needs_temperature=False,
                needs_grid=False,
                needs_satfunc=False,
                needs_rock=False,
                needs_pvt=False,
                needs_equilibrium=False,
            )

    def test_inconsistency_is_not_checked_when_wells_are_supplied(self):
        TestCaseRequirements().check(
            deck("WCONPROD\n 'P' OPEN BHP 5* 100 /\n/\n"),
            needs_temperature=False,
            needs_grid=False,
            needs_satfunc=False,
            needs_rock=False,
            needs_pvt=False,
            needs_equilibrium=False,
            needs_wells=False,
        )

    def test_well_system_from_a_deck_without_wells_says_so_before_reading_vfp_tables(self):
        from bores.wells.model import WellSystem

        d = deck("VFPPROD\n 1 /\n")
        with mock.patch(
            "bores.wells.model.VFPTables.from_deck", side_effect=AssertionError("parsed")
        ):
            with pytest.raises(ValidationError, match="defines no wells"):
                WellSystem.from_deck(d, grid=mock.Mock())


def spe1_without_wells() -> DeckFile:
    import pathlib
    import re

    text = (pathlib.Path(__file__).resolve().parents[2] / "data" / "SPE1CASE1.DATA").read_text()
    for keyword in ("WELSPECS", "COMPDAT", "WCONPROD", "WCONINJE"):
        text = re.sub(rf"(?ms)^{keyword}\s*\n.*?^/\s*$", "", text)
    return DeckFile(text)


def test_a_real_deck_without_wells_loads_as_a_case_with_no_wells():
    from bores.simulation.case import load_case

    d = spe1_without_wells()
    assert not d.has_any("WELSPECS", "COMPDAT", "WCONPROD", "WCONINJE")
    case = load_case(d, temperature=200.0)
    assert case.model.wells is None
    assert case.schedule is not None

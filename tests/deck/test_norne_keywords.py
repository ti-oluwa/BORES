import warnings

import numpy as np
import pytest

import bores.deck.keywords as keywords
from bores.deck import DEFAULT_KEYWORDS, DeckFile, DeckParseError, tvdpf

# A stretch of the Norne deck, exactly as it is written there.
NORNE_SNIPPET = """GDORIENT
INC INC INC DOWN LEFT / support this keyword

ENDSCALE
NODIR   REVERS /

SATOPTS
HYSTER  /

EQLOPTS
 'THPRES'  /   no fine equilibration if swatinit is being used

VFPIDIMS
 30    20   20 /

VFPPDIMS
 19  10  10  10  0  50 /

FAULTDIM
10000 /

PIMTDIMS
1  51 /

NSTACK
 30 /

UNIFIN
UNIFOUT

--FMTOUT
--FMTIN

NEWTRAN

-- Ask for an EGRID file; no .GRID output.
GRIDFILE
  0  1 /

MESSAGES
 8*10000  20000 10000 1000 1* /

NOECHO

MINPV
  500 /

TRACER
SEA  WAT  /
HTO  WAT  /
S36  WAT  /
2FB  WAT  /
4FB  WAT  /
DFB  WAT  /
TFB  WAT  /
/

SCALECRS
YES /

COPY
  SWL SWCR /
  SWL SGU  /
/


ADD
SWCR   0.08  1 46 1 112 1 22  /
/

-- SGU = 1 - SWL
MULTIPLY
SGU  -1  1 46 1 112 1 22  /
/

ADD
SGU   1  1 46 1 112 1 22  /
/

EQUALS
SGL   0.0  1 46 1 112 1 22  /
SGCR  0.03 1 46 1 112 1 22  /
SOWCR 0.13 1 46 1 112 1 22  /
SOGCR 0.07 1 46 1 112 1 22  /
SWU   1.0  1 46 1 112 1 22  /
/

-- Hysteresis input
EHYSTR
   0.1   0  0.1 1* KR /

EQUALS
ISGCR 0.22 1 46 1 112  1  22 /
/

RPTPROPS
1 1 1 5*0 0 /

RPTRST
BASIC=2 KRO KRW KRG /

RPTSOL

FIP=3  SWAT / 

-- initialise injected tracers to zero
TVDPFSEA
1000   0.0
5000   0.0 /
TVDPFHTO
1000   0.0
5000   0.0 /
TVDPFS36
1000   0.0
5000   0.0 /
TVDPF2FB
1000   0.0
5000   0.0 /
TVDPF4FB
1000   0.0
5000   0.0 /
TVDPFDFB
1000   0.0
5000   0.0 /
TVDPFTFB
1000   0.0
5000   0.0 /

-------------------------------------------------------------------------------

SUMMARY

NEWTON
MLINEARS


TUNING
1 10  0.1  0.15  3  0.3  0.3  1.20  /
5*   0.1   0.0001   0.02  0.02  /
2* 40 1* 15 /
/

-- only possible for ECL 2006.2+ version
ZIPPY2
'SIM=4.2' 'MINSTEP=1E-6' /
"""

NX, NY, NZ = 46, 112, 22


def norne_deck(**kwargs) -> DeckFile:
    # `SWL` is set elsewhere in the Norne deck. The snippet copies it, so it is given here.
    header = f"METRIC\nSPECGRID\n {NX} {NY} {NZ} 1 F /\nSWL\n {NX * NY * NZ}*0.2 /\n"
    return DeckFile(header + NORNE_SNIPPET, **kwargs)


def make(body: str, grid: str = "3 1 1", **kwargs) -> DeckFile:
    return DeckFile(f"METRIC\nSPECGRID\n {grid} 1 F /\n{body}", **kwargs)


def captured(function):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function()
    return result, [str(w.message) for w in caught]


# --- the Norne snippet -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def norne() -> DeckFile:
    return norne_deck()


def test_the_snippet_reads_in_strict_mode_without_a_single_warning():
    deck, messages = captured(lambda: norne_deck(strict=True))
    assert messages == []
    assert deck.dimensions is not None and deck.dimensions.n_cells == NX * NY * NZ


def test_every_keyword_of_the_snippet_is_registered():
    names = {keyword.name for keyword in DEFAULT_KEYWORDS}
    snippet_keywords = {
        "GDORIENT", "ENDSCALE", "SATOPTS", "EQLOPTS", "VFPIDIMS", "VFPPDIMS", "FAULTDIM",
        "PIMTDIMS", "NSTACK", "UNIFIN", "UNIFOUT", "FMTOUT", "FMTIN", "NEWTRAN", "GRIDFILE",
        "MESSAGES", "NOECHO", "MINPV", "TRACER", "SCALECRS", "EHYSTR", "RPTPROPS", "RPTRST",
        "RPTSOL", "TVDPFSEA", "TVDPFHTO", "TVDPFS36", "TVDPF2FB", "TVDPF4FB", "TVDPFDFB",
        "TVDPFTFB", "SUMMARY", "NEWTON", "MLINEARS", "TUNING", "ZIPPY2", "SWL", "SWCR", "SGU",
        "SGL", "SGCR", "SOWCR", "SOGCR", "SWU", "ISGCR",
    }  # fmt: skip
    assert snippet_keywords <= names


def test_grid_and_runspec_values(norne):
    assert norne.get("GDORIENT") == {
        "i_direction": "INC",
        "j_direction": "INC",
        "k_direction": "INC",
        "z_direction": "DOWN",
        "handedness": "LEFT",
    }
    assert norne.get("ENDSCALE") == {
        "directional": "NODIR",
        "reversibility": "REVERS",
        "ntendp": 1,
        "nsendp": 20,
        "combination_mode": 0,
    }
    assert norne.get("SATOPTS") == ["HYSTER"]
    assert norne.get("EQLOPTS") == ["THPRES"]
    assert norne.get("VFPIDIMS") == {"mxmflo": 30, "mxmthp": 20, "nmmvft": 20}
    assert norne.get("VFPPDIMS") == {
        "mxmflo": 19,
        "mxmthp": 10,
        "mxmwfr": 10,
        "mxmgfr": 10,
        "mxmalq": 0,
        "nmmvft": 50,
    }
    assert norne.get("FAULTDIM") == {"mfsegs": 10000}
    assert norne.get("PIMTDIMS") == {"ntpimt": 1, "nppimt": 51}
    assert norne.get("NSTACK") == {"nstack": 30}
    assert norne.get("GRIDFILE") == {"grid": 0, "egrid": 1}
    assert norne.get("MINPV") == {"minpv": 500.0}
    assert norne.get("MESSAGES") == {
        "message_print_limit": 10000,
        "comment_print_limit": 10000,
        "warning_print_limit": 10000,
        "problem_print_limit": 10000,
        "error_print_limit": 10000,
        "bug_print_limit": 10000,
        "message_stop_limit": 10000,
        "comment_stop_limit": 10000,
        "warning_stop_limit": 20000,
        "problem_stop_limit": 10000,
        "error_stop_limit": 1000,
        "bug_stop_limit": 1,  # defaulted with `1*`
        "group_print_limit": 10,
    }


def test_flags_and_sections(norne):
    for name in ("UNIFIN", "UNIFOUT", "NEWTRAN", "NOECHO", "SUMMARY"):
        assert norne.get(name) is True, name
    # Written only as a comment in the snippet.
    assert norne.get("FMTOUT") is None
    assert norne.get("FMTIN") is None
    assert norne.get("NEWTON") == [] and norne.get("MLINEARS") == []


def test_props_and_solution_values(norne):
    assert [(t["name"], t["fluid"]) for t in norne.get("TRACER")] == [
        (name, "WAT") for name in ("SEA", "HTO", "S36", "2FB", "4FB", "DFB", "TFB")
    ]
    assert norne.get("SCALECRS") == {"enabled": True}
    hysteresis = norne.get("EHYSTR")
    assert hysteresis["limiting_curves"] == "KR"
    assert hysteresis["killough_wetting_curvature"] == pytest.approx(0.1)
    for name in (
        "TVDPFSEA",
        "TVDPFHTO",
        "TVDPFS36",
        "TVDPF2FB",
        "TVDPF4FB",
        "TVDPFDFB",
        "TVDPFTFB",
    ):
        (table,) = norne.get(name)
        assert [(row["depth"], row["concentration"]) for row in table] == [(1000, 0), (5000, 0)]


def test_report_controls(norne):
    (props,) = norne.get("RPTPROPS")
    assert props["flags"] == [1, 1, 1, 0, 0, 0, 0, 0, 0] and props["mnemonics"] == {}
    (restart,) = norne.get("RPTRST")
    assert restart["mnemonics"] == {"BASIC": 2, "KRO": None, "KRW": None, "KRG": None}
    (solution,) = norne.get("RPTSOL")
    assert solution["mnemonics"] == {"FIP": 3, "SWAT": None}
    assert props["schedule_time"] == pytest.approx(0.0)
    assert solution["schedule_time"] == pytest.approx(0.0)


def test_tuning_and_zippy2(norne):
    (tuning,) = norne.get("TUNING")
    time_step = tuning["time_step"]
    assert [time_step[k] for k in ("tsinit", "tsmaxz", "tsminz", "tsmchp")] == [1, 10, 0.1, 0.15]
    assert [time_step[k] for k in ("tsfmax", "tsfmin", "tsfcnv", "tfdiff")] == [3, 0.3, 0.3, 1.2]
    assert time_step["thrupt"] == pytest.approx(1e20)  # defaulted
    assert time_step["tmaxwc"] is None  # no default
    convergence = tuning["convergence"]
    # The five items skipped with `5*` take their defaults.
    assert [convergence[k] for k in ("trgtte", "trgcnv", "trgmbe", "trglcv", "xxxtte")] == [
        pytest.approx(v) for v in (0.1, 0.001, 1e-7, 1e-4, 10.0)
    ]
    assert convergence["trgsft"] is None
    assert [convergence[k] for k in ("xxxcnv", "xxxmbe", "xxxlcv", "xxxwfl")] == [
        0.1,
        0.0001,
        0.02,
        0.02,
    ]
    assert convergence["trwght"] == 1
    iteration = tuning["iteration"]
    assert [iteration[k] for k in ("newtmx", "newtmn", "litmax", "litmin")] == [12, 1, 40, 1]
    assert [iteration[k] for k in ("mxwsit", "mxwpit", "mnwrfp")] == [15, 8, 4]
    assert tuning["schedule_time"] == pytest.approx(0.0)
    assert norne.get("ZIPPY2") == [
        {"settings": {"SIM": "4.2", "MINSTEP": "1E-6"}, "schedule_time": 0.0}
    ]


def test_end_point_scaling_arrays_follow_the_operators_of_the_snippet(norne):
    # SWCR = SWL + 0.08, and SGU = 1 - SWL, both set through COPY / ADD / MULTIPLY.
    assert np.allclose(norne.get("SWL"), 0.2)
    assert np.allclose(norne.get("SWCR"), 0.28)
    assert np.allclose(norne.get("SGU"), 0.8)
    for name, value in {
        "SGL": 0.0,
        "SGCR": 0.03,
        "SOWCR": 0.13,
        "SOGCR": 0.07,
        "SWU": 1.0,
        "ISGCR": 0.22,
    }.items():
        array = norne.get(name)
        assert array.shape == (NX * NY * NZ,)
        assert np.allclose(array, value), name


# --- end-point scaling arrays ------------------------------------------------------------------


def test_cells_no_array_ever_set_have_no_end_point_rather_than_zero():
    deck = make("BOX\n 1 1 1 1 1 1 /\nEQUALS\n 'SWL' 0.1 /\n/\nENDBOX\n")
    swl = deck.get("SWL")
    assert swl[0] == pytest.approx(0.1)
    assert np.isnan(swl[1:]).all()


def test_operators_leave_a_cell_without_an_end_point_without_one():
    deck = make(
        "BOX\n 1 1 1 1 1 1 /\nEQUALS\n 'SWL' 0.1 /\n/\nENDBOX\n"
        "COPY\n 'SWL' 'SWCR' /\n/\nADD\n 'SWCR' 0.08 /\n/\nMULTIPLY\n 'SWCR' 2 /\n/\n"
    )
    swcr = deck.get("SWCR")
    assert swcr[0] == pytest.approx(0.36)
    assert np.isnan(swcr[1:]).all()


def test_an_unspecified_end_point_array_is_none():
    assert make("OIL\n").get("SWL") is None


# --- option lists ------------------------------------------------------------------------------


def test_option_lists():
    assert make("SATOPTS\n HYSTER Direct /\n").get("SATOPTS") == ["HYSTER", "DIRECT"]
    assert make("EQLOPTS\n 1* QUIESC /\n").get("EQLOPTS") == ["QUIESC"]
    assert make("EQLOPTS\n /\n").get("EQLOPTS") == []
    assert make("OIL\n").get("EQLOPTS") is None


def test_option_list_repeated_is_reported():
    body = "SATOPTS\n HYSTER /\nSATOPTS\n DIRECT /\n"
    values, messages = captured(lambda: make(body).get("SATOPTS"))
    assert values == ["HYSTER"]
    assert any("appears 2 times" in m for m in messages)
    with pytest.raises(DeckParseError, match="appears 2 times"):
        make(body, strict=True).get("SATOPTS")


# --- fixed-layout records ----------------------------------------------------------------------


def test_gdorient_defaults_and_validation():
    assert make("GDORIENT\n /\n").get("GDORIENT") == {
        "i_direction": "INC",
        "j_direction": "INC",
        "k_direction": "INC",
        "z_direction": "DOWN",
        "handedness": "RIGHT",
    }
    assert make("GDORIENT\n dec 1* inc up /\n").get("GDORIENT")["i_direction"] == "DEC"
    with pytest.raises(DeckParseError, match="'i_direction' got unrecognised value 'UP'"):
        make("GDORIENT\n UP /\n").get("GDORIENT")
    with pytest.raises(DeckParseError, match="'handedness'"):
        make("GDORIENT\n INC INC INC DOWN MIDDLE /\n").get("GDORIENT")


def test_endscale_defaults_and_validation():
    assert make("ENDSCALE\n /\n").get("ENDSCALE") == {
        "directional": "NODIR",
        "reversibility": "REVERS",
        "ntendp": 1,
        "nsendp": 20,
        "combination_mode": 0,
    }
    deck = make("ENDSCALE\n DIRECT IRREVERS 3 40 1 /\n")
    assert deck.get("ENDSCALE")["directional"] == "DIRECT"
    assert deck.get("ENDSCALE")["reversibility"] == "IRREVERS"
    assert deck.get("ENDSCALE")["ntendp"] == 3
    with pytest.raises(DeckParseError, match="'directional'"):
        make("ENDSCALE\n SIDEWAYS /\n").get("ENDSCALE")


def test_dimension_records_default_the_items_not_given():
    # Defaults as declared in OPM's keyword definitions.
    assert make("VFPIDIMS\n 30 /\n").get("VFPIDIMS") == {"mxmflo": 30, "mxmthp": 0, "nmmvft": 0}
    assert make("VFPPDIMS\n /\n").get("VFPPDIMS") == dict.fromkeys(
        ("mxmflo", "mxmthp", "mxmwfr", "mxmgfr", "mxmalq", "nmmvft"), 0
    )
    assert make("FAULTDIM\n 1* /\n").get("FAULTDIM") == {"mfsegs": 0}
    assert make("PIMTDIMS\n /\n").get("PIMTDIMS") == {"ntpimt": 0, "nppimt": 0}
    assert make("NSTACK\n /\n").get("NSTACK") == {"nstack": 10}
    assert make("MINPV\n /\n").get("MINPV") == {"minpv": pytest.approx(1e-6)}


def test_messages_defaults():
    messages = make("MESSAGES\n /\n").get("MESSAGES")
    assert messages["warning_print_limit"] == 10000
    assert messages["error_stop_limit"] == 10
    assert messages["bug_stop_limit"] == 1


def test_gridfile_defaults():
    assert make("GRIDFILE\n /\n").get("GRIDFILE") == {"grid": 0, "egrid": 1}
    assert make("GRIDFILE\n 1 0 /\n").get("GRIDFILE") == {"grid": 1, "egrid": 0}


def test_scalecrs():
    assert make("SCALECRS\n NO /\n").get("SCALECRS") == {"enabled": False}
    assert make("SCALECRS\n /\n").get("SCALECRS") == {"enabled": False}
    assert make("SCALECRS\n yes /\n").get("SCALECRS") == {"enabled": True}
    with pytest.raises(DeckParseError):
        make("SCALECRS\n MAYBE /\n").get("SCALECRS")


def test_tracer_defaults_case_and_validation():
    (record,) = make("TRACER\n sea wat /\n/\n").get("TRACER")
    assert record == {
        "name": "SEA",
        "fluid": "WAT",
        "unit": "",
        "solution_phase": "",
        "num_part_table": 0,
        "adsorb_phase": "",
    }
    (record,) = make("TRACER\n X GAS 'mg/l' OIL 2 /\n/\n").get("TRACER")
    assert (record["unit"], record["solution_phase"], record["num_part_table"]) == (
        "mg/l",
        "OIL",
        2,
    )
    with pytest.raises(DeckParseError, match="'fluid' got unrecognised value 'STEAM'"):
        make("TRACER\n X STEAM /\n/\n").get("TRACER")


# --- tracer concentration tables ---------------------------------------------------------------


def test_tvdpf_tables_one_per_region():
    deck = make("TVDPFSEA\n 1000 0.0\n 5000 1.5 /\n 1000 2.0\n 5000 3.0 /\n")
    first, second = deck.get("TVDPFSEA")
    assert [row["concentration"] for row in first] == [0.0, 1.5]
    assert [row["concentration"] for row in second] == [2.0, 3.0]


def test_tvdpf_builds_the_keyword_of_any_tracer():
    keyword = tvdpf("xyz")
    assert keyword.name == "TVDPFXYZ"
    deck = make("TVDPFXYZ\n 1000 0.5 /\n")
    assert deck.get("TVDPFXYZ") is None  # not registered, so not read
    deck.add_keywords(keyword)
    assert deck.get("TVDPFXYZ")[0][0]["concentration"] == pytest.approx(0.5)


# --- TUNING and ZIPPY2 over time ---------------------------------------------------------------


def test_tuning_is_stamped_with_the_time_it_takes_effect():
    body = "TUNING\n 1 /\n /\n /\nTSTEP\n 10 20 /\nTUNING\n 2 /\n 5* 0.5 /\n 5 /\n"
    first, second = make(body).get("TUNING")
    assert first["schedule_time"] == pytest.approx(0.0)
    assert second["schedule_time"] == pytest.approx(30.0)
    assert first["time_step"]["tsinit"] == 1 and second["time_step"]["tsinit"] == 2
    assert second["convergence"]["xxxcnv"] == pytest.approx(0.5)
    assert second["iteration"]["newtmx"] == 5
    assert first["iteration"]["newtmx"] == 12


def test_tuning_empty_records_leave_everything_at_its_default():
    (tuning,) = make("TUNING\n /\n /\n /\n").get("TUNING")
    time_step, convergence, iteration = (
        tuning["time_step"],
        tuning["convergence"],
        tuning["iteration"],
    )
    assert [time_step[k] for k in ("tsinit", "tsmaxz", "tsminz", "tsmchp")] == [
        pytest.approx(v) for v in (1.0, 365.0, 0.1, 0.15)
    ]
    assert time_step["tfdiff"] == pytest.approx(1.25)
    assert time_step["tmaxwc"] is None  # no default
    assert convergence["trgmbe"] == pytest.approx(1e-7)
    assert convergence["trgsft"] is None
    assert convergence["trwght"] == 1
    assert iteration["ddplim"] == pytest.approx(1e6)
    assert iteration["xxxdpr"] is None
    assert iteration["litmax"] == 25


def test_tuning_with_a_stray_closing_slash_is_read_without_complaint():
    deck, messages = captured(
        lambda: make("TUNING\n 1 /\n /\n /\n/\n", strict=True),
    )
    (tuning,) = deck.get("TUNING")
    assert tuning["time_step"]["tsinit"] == 1
    assert messages == []


def test_tuning_with_too_few_records_is_reported():
    body = "TUNING\n 1 /\n 2 /\n"
    (tuning,), messages = captured(lambda: make(body).get("TUNING"))
    assert any("has 2 `/`-terminated record(s) but needs 3" in m for m in messages)
    assert tuning["iteration"]["newtmx"] == 12  # the missing record is read as empty
    with pytest.raises(DeckParseError, match="needs 3"):
        make(body, strict=True).get("TUNING")


def test_tuning_with_data_after_its_records_is_reported():
    body = "TUNING\n 1 /\n /\n /\n 99 /\n"
    _, messages = captured(lambda: make(body).get("TUNING"))
    assert any("data after its 3 records" in m for m in messages)


def test_tuning_rejects_a_value_that_is_not_a_number():
    with pytest.raises(DeckParseError, match="'tsinit' got invalid value 'abc'"):
        make("TUNING\n abc /\n /\n /\n").get("TUNING")


def test_zippy2_settings_per_occurrence():
    body = "ZIPPY2\n 'SIM=4.2' flag 'minstep = 1E-6' /\nTSTEP\n 5 /\nZIPPY2\n 'SIM=5' /\n"
    first, second = make(body).get("ZIPPY2")
    assert first == {
        "settings": {"SIM": "4.2", "FLAG": None, "MINSTEP": "1E-6"},
        "schedule_time": 0.0,
    }
    assert second == {"settings": {"SIM": "5"}, "schedule_time": 5.0}


# --- report controls ---------------------------------------------------------------------------


def test_rptsol_and_rptprops_read_both_forms():
    deck = make("RPTPROPS\n 1 1* 0 /\nRPTSOL\n FIP=3 SWAT /\n")
    assert deck.get("RPTPROPS")[0]["flags"] == [1, None, 0]
    assert deck.get("RPTSOL")[0]["mnemonics"] == {"FIP": 3, "SWAT": None}
    with pytest.raises(DeckParseError, match="cannot be mixed"):
        make("RPTSOL\n FIP=3 1 /\n").get("RPTSOL")


# --- the registry ------------------------------------------------------------------------------


def test_section_keywords_say_which_sections_a_deck_has():
    deck = make("PROPS\nSOLUTION\n")
    assert deck.get("PROPS") is True and deck.get("SOLUTION") is True
    assert deck.get("SCHEDULE") is None and deck.get("SUMMARY") is None


def test_every_registered_keyword_can_be_imported_from_the_keywords_package():
    exported = [getattr(keywords, name) for name in keywords.__all__]
    unreachable = [
        keyword.name
        for keyword in DEFAULT_KEYWORDS
        if not any(keyword is item for item in exported)
    ]
    assert unreachable == []


def test_registered_keyword_names_are_unique():
    names = [keyword.name for keyword in DEFAULT_KEYWORDS]
    assert len(names) == len(set(names))

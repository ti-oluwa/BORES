import itertools
import random
import warnings

import numpy as np
import pytest

from bores.deck import DeckFile, DeckParseError
from bores.deck.core import parse_floats, parse_numeric_runs


def make(body: str, grid: str = "3 1 1", **kwargs) -> DeckFile:
    return DeckFile(f"METRIC\nSPECGRID\n {grid} 1 F /\n{body}", **kwargs)


def captured(function):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function()
    return result, [str(w.message) for w in caught]


def runs_of(text: str):
    return [
        (
            run.entry_count,
            run.values.tolist() if isinstance(run.values, np.ndarray) else run.values,
        )
        for run in parse_numeric_runs(text, name="X")
    ]


# --- numeric reader ----------------------------------------------------------------------------


def test_numbers_are_read_in_bulk_across_any_whitespace():
    assert parse_floats(" 1  2.5 \n 3e2\r\n-4\t.5 ", "X").tolist() == [1.0, 2.5, 300.0, -4.0, 0.5]


def test_blank_text_has_no_numbers():
    # `numpy.fromstring` reads blank text as [-1.0]; that must never leak through.
    assert parse_floats("", "X").size == 0
    assert parse_floats("  \n\t ", "X").size == 0
    assert runs_of("  \n ") == []


def test_non_numbers_are_named_in_the_error():
    with pytest.raises(DeckParseError, match="`X` contains a non-numeric value 'abc'"):
        parse_floats("1 2 abc 4", "`X`")
    with pytest.raises(DeckParseError, match="'1,2'"):
        parse_floats("1,2", "X")


def test_repeat_groups_stay_single_runs():
    assert runs_of("1 2 3*7 4") == [(2, [1.0, 2.0]), (3, 7.0), (1, [4.0])]
    assert runs_of("1000000*0.25") == [(1000000, 0.25)]
    assert runs_of("10*1e-3 5 5") == [(10, 0.001), (2, [5.0, 5.0])]
    assert runs_of("  4*2\n 3 ") == [(4, 2.0), (1, [3.0])]


def test_bare_repeat_groups_are_defaulted_entries():
    assert runs_of("0.1 2* 0.2") == [(1, [0.1]), (2, None), (1, [0.2])]
    assert runs_of("5*") == [(5, None)]


def test_empty_repeat_groups_are_left_out():
    assert runs_of("0*5 1 2") == [(2, [1.0, 2.0])]


@pytest.mark.parametrize("text", ["x*3", "*5", "1 *5", "1.5*2", "a" * 40 + "*3", "2*3*4", "2*x"])
def test_malformed_repeat_groups_are_rejected(text):
    with pytest.raises(DeckParseError):
        parse_numeric_runs(text, name="X")


# --- array data blocks -------------------------------------------------------------------------


def test_arrays_mixing_values_and_repeat_groups():
    deck = make("PORO\n 0.1 2*0.2 0.3 2*0.4 /\n", grid="3 2 1")
    assert deck.get("PORO").tolist() == [0.1, 0.2, 0.2, 0.3, 0.4, 0.4]


def test_wrong_value_count_message_names_the_expected_counts():
    with pytest.raises(DeckParseError, match=r"PORO expected 1 or 3 value\(s\); got 2"):
        make("PORO\n 1 2 /\n").get("PORO")


def test_non_numeric_array_value_is_an_error():
    with pytest.raises(DeckParseError, match="non-numeric value 'x'"):
        make("PORO\n 1 x 3 /\n").get("PORO")


def test_defaulted_entries_are_dropped_with_a_warning_or_an_error_when_strict():
    # Exactly as before: the defaulted entry is dropped (and everything after it moves).
    values, messages = captured(lambda: make("PORO\n 0.1 0.2 0.3 1* /\n").get("PORO"))
    assert values.tolist() == [0.1, 0.2, 0.3]
    assert any("bare 'N*'" in m and "1 token(s) dropped" in m for m in messages)
    with pytest.raises(DeckParseError, match="bare 'N\\*'"):
        make("PORO\n 0.1 0.2 0.3 1* /\n", strict=True).get("PORO")


def test_single_value_fills_the_grid_silently_unless_strict():
    values, messages = captured(lambda: make("PORO\n 0.25 /\n").get("PORO"))
    assert values.tolist() == [0.25] * 3
    assert messages == []
    with pytest.raises(DeckParseError, match="single value but the grid has 3 cells"):
        make("PORO\n 0.25 /\n", strict=True).get("PORO")


def test_single_value_in_a_one_cell_grid_is_a_full_array():
    values, messages = captured(
        lambda: make("PORO\n 0.25 /\n", grid="1 1 1", strict=True).get("PORO")
    )
    assert values.tolist() == [0.25]
    assert messages == []


def test_repeat_group_covering_the_grid_is_not_a_single_value():
    assert make("PORO\n 3*0.25 /\n", strict=True).get("PORO").tolist() == [0.25] * 3


def test_empty_array_record_is_skipped_with_a_warning_or_an_error_when_strict():
    values, messages = captured(lambda: make("PORO\n /\n").get("PORO"))
    assert values.tolist() == [0.0] * 3
    assert any("zero tokens" in m for m in messages)
    with pytest.raises(DeckParseError, match="zero tokens"):
        make("PORO\n /\n", strict=True).get("PORO")


def test_short_form_array_still_tiles_over_the_layers():
    deck = make("TOPS\n 100 200 /\n", grid="2 1 2")
    assert deck.get("TOPS").tolist() == [100.0, 200.0, 100.0, 200.0]
    full = make("TOPS\n 1 2 3 4 /\n", grid="2 1 2")
    assert full.get("TOPS").tolist() == [1.0, 2.0, 3.0, 4.0]


def test_short_form_in_a_single_column_grid_is_not_a_strict_error():
    deck = make("TOPS\n 100 /\n", grid="1 1 3", strict=True)
    assert deck.get("TOPS").tolist() == [100.0] * 3


# --- COORD and ZCORN ---------------------------------------------------------------------------


COORD_1x1 = "0 0 0 0 0 1 1 0 0 1 0 1 0 1 0 0 1 1 1 1 0 1 1 1"


def test_coord_and_zcorn_accept_repeat_groups():
    deck = make(f"COORD\n {COORD_1x1} /\nZCORN\n 4*0 4*1 /\n", grid="1 1 1")
    assert deck.get("COORD").shape == (2, 2, 6)
    assert deck.get("ZCORN").tolist() == [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]


def test_coord_and_zcorn_count_mismatch_is_reported():
    with pytest.raises(DeckParseError, match="`ZCORN` expected 8 values for a 1x1x1 grid; got 7"):
        make("ZCORN\n 7*1 /\n", grid="1 1 1").get("ZCORN")
    with pytest.raises(DeckParseError, match="`COORD` expected 24 values for a 1x1 grid; got 2"):
        make("COORD\n 1 2 /\n", grid="1 1 1").get("COORD")


def test_coord_rejects_defaulted_and_non_numeric_values():
    with pytest.raises(DeckParseError, match="defaulted"):
        make("ZCORN\n 7*1 1* /\n", grid="1 1 1").get("ZCORN")
    with pytest.raises(DeckParseError, match="non-numeric value 'q'"):
        make("ZCORN\n 7*1 q /\n", grid="1 1 1").get("ZCORN")


def test_returned_coord_array_is_writable():
    zcorn = make("ZCORN\n 8*1 /\n", grid="1 1 1").get("ZCORN")
    zcorn[0, 0, 0] = 5.0
    assert zcorn[0, 0, 0] == pytest.approx(5.0)


# --- COPY --------------------------------------------------------------------------------------


def test_copy_reads_the_source_as_of_the_copy_not_as_it_ends_up():
    deck = make(
        "PORO\n 0.1 0.2 0.3 /\nCOPY\n 'PORO' 'PERMX' /\n/\nEQUALS\n 'PORO' 9 /\n/\n",
    )
    assert deck.get("PERMX").tolist() == [0.1, 0.2, 0.3]
    assert deck.get("PORO").tolist() == [9.0, 9.0, 9.0]


def test_copy_from_a_partly_set_source_uses_the_sources_own_default():
    # NTG starts at 1.0, not 0.0, so the cells the BOX never touched copy as 1.0.
    deck = make(
        "BOX\n 1 1 1 1 1 1 /\nEQUALS\n 'NTG' 0.5 /\n/\nENDBOX\nCOPY\n 'NTG' 'PORO' /\n/\n",
    )
    assert deck.get("PORO").tolist() == [0.5, 1.0, 1.0]
    multiplier = make(
        "BOX\n 1 1 1 1 1 1 /\nEQUALS\n 'MULTX' 0.5 /\n/\nENDBOX\nCOPY\n 'MULTX' 'PORO' /\n/\n"
    )
    assert multiplier.get("PORO").tolist() == [0.5, 1.0, 1.0]


def test_copy_from_a_short_form_source():
    deck = make("TOPS\n 100 200 /\nCOPY\n 'TOPS' 'PERMX' /\n/\n", grid="2 1 2")
    assert deck.get("PERMX").tolist() == [100.0, 200.0, 100.0, 200.0]


def test_a_source_registered_after_construction_is_read_with_its_own_default():
    from bores.deck.keywords.base import ArrayKeyword

    deck = make(
        "BOX\n 1 1 1 1 1 1 /\nEQUALS\n 'MYSRC' 0.5 /\n/\nENDBOX\nCOPY\n 'MYSRC' 'PORO' /\n/\n"
    )
    deck.add_keywords(ArrayKeyword("MYSRC", default_value=7.0))
    assert deck.get("PORO").tolist() == [0.5, 7.0, 7.0]


def test_chains_and_fan_out_of_copies():
    body = (
        "PORO\n 0.1 0.2 0.3 /\n"
        "COPY\n 'PORO' 'PERMX' /\n 'PORO' 'PERMY' /\n/\n"
        "ADD\n 'PERMX' 1 /\n/\n"
        "COPY\n 'PERMX' 'PERMZ' /\n 'PERMY' 'NTG' /\n/\n"
        "MULTIPLY\n 'PORO' 10 /\n/\n"
        "COPY\n 'PORO' 'DZ' /\n 'PERMZ' 'DX' /\n/\n"
    )
    deck = make(body)
    assert np.allclose(deck.get("PERMX"), [1.1, 1.2, 1.3])
    assert np.allclose(deck.get("PERMY"), [0.1, 0.2, 0.3])
    assert np.allclose(deck.get("PERMZ"), [1.1, 1.2, 1.3])
    assert np.allclose(deck.get("NTG"), [0.1, 0.2, 0.3])
    assert np.allclose(deck.get("DZ"), [1.0, 2.0, 3.0])
    assert np.allclose(deck.get("DX"), [1.1, 1.2, 1.3])
    assert np.allclose(deck.get("PORO"), [1.0, 2.0, 3.0])


def test_copy_results_are_writable_and_independent():
    deck = make("PORO\n 0.1 0.2 0.3 /\nCOPY\n 'PORO' 'PERMX' /\n 'PORO' 'PERMY' /\n/\n")
    permx = deck.get("PERMX")
    permx[:] = -1.0
    assert deck.get("PERMY").tolist() == [0.1, 0.2, 0.3]
    assert deck.get("PORO").tolist() == [0.1, 0.2, 0.3]


def test_copy_with_no_source_data_still_warns():
    _, messages = captured(lambda: make("COPY\n 'PORO' 'PERMX' /\n/\n").get("PERMX"))
    assert any("has no data" in m for m in messages)


# --- operators against an independent per-cell reference ---------------------------------------


def reference(nx, ny, nz, initial, ops):
    cells = {(i, j, k): initial for i, j, k in itertools.product(range(nx), range(ny), range(nz))}
    for op, value, (i1, i2, j1, j2, k1, k2) in ops:
        for i, j, k in itertools.product(range(i1 - 1, i2), range(j1 - 1, j2), range(k1 - 1, k2)):
            old = cells[i, j, k]
            cells[i, j, k] = {
                "EQUALS": value,
                "ADD": old + value,
                "MULTIPLY": old * value,
                "MAXVALUE": max(old, value),
                "MINVALUE": min(old, value),
            }[op]
    return [cells[i, j, k] for k in range(nz) for j in range(ny) for i in range(nx)]


@pytest.mark.parametrize("seed", range(40))
def test_operators_match_a_per_cell_reference(seed):
    rng = random.Random(seed)
    nx, ny, nz = rng.randint(1, 5), rng.randint(1, 4), rng.randint(1, 3)
    ops = []
    lines = [f"PORO\n {nx * ny * nz}*0.5 /"]
    for _ in range(rng.randint(1, 8)):
        op = rng.choice(["EQUALS", "ADD", "MULTIPLY", "MAXVALUE", "MINVALUE"])
        value = round(rng.uniform(0.1, 3.0), 3)
        i1 = rng.randint(1, nx)
        j1 = rng.randint(1, ny)
        k1 = rng.randint(1, nz)
        box = (i1, rng.randint(i1, nx), j1, rng.randint(j1, ny), k1, rng.randint(k1, nz))
        ops.append((op, value, box))
        lines.append(f"{op}\n 'PORO' {value} {' '.join(map(str, box))} /\n/")
    deck = make("\n".join(lines) + "\n", grid=f"{nx} {ny} {nz}", strict=True)
    assert np.allclose(deck.get("PORO"), reference(nx, ny, nz, 0.5, ops))


def test_full_grid_operators_and_boxed_copy():
    deck = make(
        "PORO\n 0.1 0.2 0.3 /\nBOX\n 2 3 1 1 1 1 /\nCOPY\n 'PORO' 'PERMX' /\n/\nENDBOX\n"
        "ADD\n 'PORO' 1 /\n/\nMULTIPLY\n 'PORO' 2 /\n/\nMAXVALUE\n 'PORO' 2.5 /\n/\n"
        "MINVALUE\n 'PORO' 3 /\n/\n",
    )
    assert deck.get("PERMX").tolist() == [0.0, 0.2, 0.3]
    assert np.allclose(deck.get("PORO"), [2.5, 2.5, 2.6])


# --- schedule ----------------------------------------------------------------------------------


def clock(body: str, **kwargs) -> list[float]:
    return DeckFile("METRIC\n" + body, **kwargs).report_times


def test_a_default_or_non_numeric_tstep_entry_is_always_an_error():
    with pytest.raises(DeckParseError, match=r"'1\*'.*defaulted"):
        clock("TSTEP\n 10 1* /\n")
    with pytest.raises(DeckParseError, match="non-numeric time-step value 'abc'"):
        clock("TSTEP\n 10 abc /\n")


def test_a_negative_tstep_is_reported():
    times, messages = captured(lambda: clock("TSTEP\n 10 -2 /\n"))
    assert times == [0.0, 8.0, 10.0]
    assert any("moves the clock backwards" in m for m in messages)
    with pytest.raises(DeckParseError, match="backwards"):
        clock("TSTEP\n 10 -2 /\n", strict=True)


def test_an_unreadable_start_is_reported_and_the_first_dates_becomes_time_zero():
    body = "START\n 1 FOO 2020 /\nDATES\n 1 FEB 2020 /\n/\n"
    times, messages = captured(lambda: clock(body))
    assert times == [0.0]
    assert any("`START` cannot be read" in m for m in messages)
    with pytest.raises(DeckParseError, match="`START` cannot be read"):
        clock(body, strict=True)


def test_a_start_with_too_few_items_is_reported():
    _, messages = captured(lambda: clock("START\n 1 JAN /\n"))
    assert any("`START` cannot be read" in m for m in messages)


def test_an_unreadable_dates_entry_is_skipped_with_a_warning_or_an_error_when_strict():
    body = "START\n 1 JAN 2020 /\nDATES\n 1 FOO 2020 /\n 1 FEB 2020 /\n/\n"
    times, messages = captured(lambda: clock(body))
    assert times == [0.0, 31.0]
    assert any("`DATES` entry" in m and "It is skipped" in m for m in messages)
    with pytest.raises(DeckParseError, match="`DATES` entry"):
        clock(body, strict=True)


def test_dates_that_go_back_in_time_are_reported():
    body = "START\n 1 JAN 2020 /\nDATES\n 1 FEB 2020 /\n 15 JAN 2020 /\n/\n"
    _, messages = captured(lambda: clock(body))
    assert any("earlier than the deck's clock" in m for m in messages)
    with pytest.raises(DeckParseError, match="earlier"):
        clock(body, strict=True)


def test_a_well_formed_schedule_is_silent_even_when_strict():
    times, messages = captured(
        lambda: clock(
            "START\n 1 JAN 2020 /\nTSTEP\n 5*30 /\nDATES\n 1 JUL 2020 /\n/\n", strict=True
        )
    )
    assert times == [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 182.0]
    assert messages == []

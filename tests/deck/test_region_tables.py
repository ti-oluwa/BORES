"""Tests that tables for several regions are read as one table per region."""

import pytest

from bores.deck import DeckFile

TABLES = [
    [(0.15, 0.0, 1.0, 2.0), (0.30, 0.05, 0.5, 1.0), (1.0, 1.0, 0.0, 0.0)],
    [(0.10, 0.0, 1.0, 2.0), (0.25, 0.03, 0.6, 1.2), (1.0, 1.0, 0.0, 0.0)],
    [(0.12, 0.0, 1.0, 2.0), (0.40, 0.10, 0.3, 0.8), (1.0, 1.0, 0.0, 0.0)],
]


def row_text(row) -> str:
    return " ".join(str(value) for value in row)


def packed(table) -> str:
    return "\n".join(row_text(row) for row in table)


def make(tabdims: str | None, body: str) -> DeckFile:
    header = "METRIC\n"
    if tabdims is not None:
        header += f"TABDIMS\n {tabdims} /\n"
    return DeckFile(header + body + "\n")


def first_columns(deck_file: DeckFile, keyword: str) -> list[list[float]]:
    return [[row[next(iter(row))] for row in table] for table in deck_file.get(keyword)]


EXPECTED = [[row[0] for row in table] for table in TABLES]


@pytest.mark.parametrize(
    "layout",
    [
        "closing slash on the last row",
        "closing slash on its own line",
        "extra slash closing the keyword",
    ],
)
def test_each_region_gets_its_own_table(layout):
    if layout == "closing slash on the last row":
        body = "\n".join(packed(table) + " /" for table in TABLES)
    elif layout == "closing slash on its own line":
        body = "\n".join(packed(table) + "\n/" for table in TABLES)
    else:
        body = "\n".join(packed(table) + "\n/" for table in TABLES) + "\n/"
    deck_file = make("3 1", "PROPS\nSWOF\n" + body)
    assert first_columns(deck_file, "SWOF") == EXPECTED
    for table, expected in zip(deck_file.get("SWOF"), TABLES, strict=True):
        assert [row["krow"] for row in table] == [row[2] for row in expected]


def test_missing_tabdims_means_one_table():
    deck_file = make(None, "PROPS\nSWOF\n" + packed(TABLES[0]) + " /")
    assert first_columns(deck_file, "SWOF") == [EXPECTED[0]]


def test_defaulted_table_count_means_one_table():
    deck_file = make("1*", "PROPS\nSWOF\n" + packed(TABLES[0]) + " /")
    assert first_columns(deck_file, "SWOF") == [EXPECTED[0]]


def test_keyword_repeated_once_per_region_gives_one_table_each():
    body = "\n".join("SWOF\n" + packed(table) + " /" for table in TABLES)
    deck_file = make("3 1", "PROPS\n" + body)
    assert first_columns(deck_file, "SWOF") == EXPECTED


def test_gas_oil_and_pvt_tables_are_split_per_region():
    sgof = [[(0.0, 0.0, 1.0, 0.0), (0.5, 0.4, 0.2, 0.2), (0.9, 1.0, 0.0, 0.5)]] * 2
    pvdo = [[(100.0, 1.2, 1.0), (4000.0, 1.1, 1.2)], [(110.0, 1.3, 1.0), (4100.0, 1.15, 1.2)]]
    body = (
        "PROPS\nSGOF\n"
        + "\n".join(packed(table) + "\n/" for table in sgof)
        + "\nPVDO\n"
        + "\n".join(packed(table) + "\n/" for table in pvdo)
    )
    deck_file = make("2 2", body)
    assert [len(table) for table in deck_file.get("SGOF")] == [3, 3]
    assert [[row["pressure"] for row in table] for table in deck_file.get("PVDO")] == [
        [100.0, 4000.0],
        [110.0, 4100.0],
    ]

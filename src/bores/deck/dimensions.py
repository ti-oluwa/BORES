"""
Checking a deck against the sizes it declares for itself.

The `RUNSPEC` dimension keywords (`TABDIMS`, `EQLDIMS`, `WELLDIMS`, `VFPPDIMS`, `VFPIDIMS`) tell
Eclipse how much room to reserve, and Eclipse refuses a deck that then exceeds them. Nothing here
depends on those limits to run, and OPM Flow ignores them, so a deck that exceeds one still loads.
But it is a deck Eclipse would reject, and usually a sign of a table or well added without updating
the limit, so it is reported.
"""

import typing
import warnings

from bores.deck.file import DeckFile

__all__ = ["DeckDimensionWarning", "check_deck_dimensions", "warn_deck_dimensions"]

SATURATION_TABLE_KEYWORDS = ("SWOF", "SGOF", "SWFN", "SGFN", "SOF2", "SOF3")
PVT_TABLE_KEYWORDS = ("PVTO", "PVCO", "PVDO", "PVCDO", "PVTG", "PVDG", "PVTW")


class DeckDimensionWarning(UserWarning):
    """A deck holds more than one of its own dimension keywords says it should."""


def get_region_counts(deck_file: DeckFile, keywords: typing.Iterable[str]) -> dict[str, int]:
    """Counts how many regions (tables) each present keyword of `keywords` defines."""
    counts: dict[str, int] = {}
    for keyword in keywords:
        records = deck_file.get_shared(keyword) if deck_file.has(keyword) else None
        if records:
            counts[keyword] = len(records)
    return counts


def check_deck_dimensions(deck_file: DeckFile) -> list[str]:
    """
    Find where the deck holds more than its dimension keywords declare.

    A dimension keyword the deck does not write is not checked, since Eclipse then has nothing
    to enforce.

    :param deck_file: The deck to check.
    :returns: One message per exceeded limit. Empty if the deck is within all of them.
    """
    problems: list[str] = []

    def exceeded(keyword: str, item: str, declared: typing.Any, needed: int, what: str) -> None:
        if declared is not None and needed > int(declared):
            problems.append(
                f"`{keyword}` `{item}` is {int(declared)} but the deck has {needed} {what}."
            )

    tabdims = deck_file.get_shared("TABDIMS")
    if tabdims is not None:
        saturation = get_region_counts(deck_file, SATURATION_TABLE_KEYWORDS)
        if saturation:
            keyword = max(saturation, key=lambda name: saturation[name])
            exceeded(
                "TABDIMS",
                "num_saturation_tables",
                tabdims.get("num_saturation_tables"),
                saturation[keyword],
                f"saturation function regions (`{keyword}`)",
            )
            longest = max(
                len(rows)
                for name in saturation
                for rows in deck_file.get_shared(name)  # type: ignore
            )
            exceeded(
                "TABDIMS",
                "max_saturation_table_nodes",
                tabdims.get("max_saturation_table_nodes"),
                longest,
                "rows in its longest saturation function table",
            )

        pvt = get_region_counts(deck_file, PVT_TABLE_KEYWORDS)
        if pvt:
            keyword = max(pvt, key=lambda name: pvt[name])
            exceeded(
                "TABDIMS",
                "num_pvt_tables",
                tabdims.get("num_pvt_tables"),
                pvt[keyword],
                f"PVT regions (`{keyword}`)",
            )

    eqldims = deck_file.get_shared("EQLDIMS")
    equil = deck_file.get_shared("EQUIL")
    if eqldims is not None and equil:
        exceeded(
            "EQLDIMS",
            "max_equilibration_regions",
            eqldims.get("max_equilibration_regions"),
            len(equil),
            "equilibration regions (`EQUIL` records)",
        )

    welldims = deck_file.get_shared("WELLDIMS")
    welspecs = deck_file.get_shared("WELSPECS")
    if welldims is not None and welspecs:
        wells = {record["well"] for record in welspecs}
        exceeded(
            "WELLDIMS", "max_wells", welldims.get("max_wells"), len(wells), "wells (`WELSPECS`)"
        )

    for dimensions, table_keyword, item in (
        ("VFPPDIMS", "VFPPROD", "max_production_vfp_tables"),
        ("VFPIDIMS", "VFPINJ", "max_injection_vfp_tables"),
    ):
        declared = deck_file.get_shared(dimensions)
        if declared is not None and deck_file.has(table_keyword):
            tables = len(deck_file.deck.get_records_for(table_keyword))
            exceeded(dimensions, item, declared.get(item), tables, f"`{table_keyword}` tables")
    return problems


def warn_deck_dimensions(deck_file: DeckFile) -> list[str]:
    """
    Warn about each limit the deck exceeds, and return the messages.

    :param deck_file: The deck to check.
    :returns: The messages, as from `check_deck_dimensions`.
    """
    problems = check_deck_dimensions(deck_file)
    for problem in problems:
        warnings.warn(
            f"{problem} Eclipse would reject this deck.", DeckDimensionWarning, stacklevel=2
        )
    return problems

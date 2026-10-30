"""
Up-front checks that a deck holds what a loader is about to read.

A loader that reads its keywords one after another can parse several large arrays before it
reaches the one the deck does not have. `find_missing` answers the same question from the deck's
index alone, so a loader (or a stage that calls several loaders) can fail before parsing anything,
and report everything that is missing at once.

Only keywords that are always needed belong in a check. A keyword that is read only when another
is present, or only when its component was not supplied by the caller, is left out. Section
keywords (`GRID`, `PROPS`, ...) are never a reliable requirement: a standalone grid or PVT include
file has no section header yet loads fine, so the keywords themselves are what is checked.
"""

import typing

from bores.deck.file import DeckFile
from bores.deck.keywords.base import Keyword

__all__ = ["find_missing"]

KeywordLike = str | Keyword[typing.Any]


def keyword_name(keyword: KeywordLike) -> str:
    """Return the upper-case name of a keyword given as a name or a `Keyword`."""
    return keyword.name.upper() if isinstance(keyword, Keyword) else keyword.upper()


def find_missing(
    deck_file: DeckFile,
    *,
    all_of: typing.Iterable[KeywordLike] = (),
    any_of: typing.Iterable[typing.Iterable[KeywordLike]] = (),
) -> list[str]:
    """
    Describe what a deck lacks out of what a loader needs.

    The whole question is answered with a single `DeckFile.has_many` call.

    :param deck_file: The deck to check.
    :param all_of: Keywords that must all be present.
    :param any_of: Groups of alternatives. Each group is satisfied when at least one of its
        keywords is present (for example `("DX", "DXV")`).
    :returns: One entry per unmet requirement, in the order given: the keyword in backticks for an
        `all_of` keyword, or ``one of `A`, `B` `` for a group. Empty when everything is present.
    """
    required = [keyword_name(keyword) for keyword in all_of]
    groups = [[keyword_name(keyword) for keyword in group] for group in any_of]
    present = deck_file.has_many(*required, *(name for group in groups for name in group))

    unmet = [f"`{name}`" for name in dict.fromkeys(required) if not present[name]]
    for group in groups:
        if group and not any(present[name] for name in group):
            unmet.append("one of " + ", ".join(f"`{name}`" for name in group))
    return unmet

"""
SUMMARY section keyword implementations.

The SUMMARY section declares which result vectors Eclipse should write to
the summary file (`.SMSPEC` / `.UNSMRY`) for later plotting and analysis.
Most SUMMARY keywords are *selector* keywords: their bare presence in the
deck means "output this vector for every relevant object" (every well,
every region, the whole field, etc.), optionally restricted to a list of
named/numbered objects.

**Field vectors** (whole-reservoir totals, no object list):

- `FOPR` / `FWPR` / `FGPR` - field oil/water/gas production rate.
- `FOPT` / `FWPT` / `FGPT` - field oil/water/gas production cumulative total.

**Well vectors** (one series per well; optionally restricted to named wells):

- `WOPR` / `WWPR` / `WGPR` - well oil/water/gas production rate.
- `WBHP` - well bottom-hole pressure.
- `WTHP` - well tubing-head pressure.

**Region vectors** (one series per FIP region; optionally restricted to
named region numbers):

- `ROIP` / `RGIP` / `RWIP` - reservoir oil/gas/water in place.

**Reporting controls** (`RPTRST` / `RPTSCHED`) configure restart-file /
print-summary mnemonics rather than naming a result vector; they are kept
in this module because they only ever appear in the SUMMARY (and
SCHEDULE) sections and share no shape with the vector selectors above.
"""

from bores.datastructures import GridDimensions
from bores.deck.core import Deck, DeckParseError, tokenize
from bores.deck.keywords.base import Keyword
from bores.deck.operators import Operation

__all__ = [
    "FGPR",
    "FGPT",
    "FOPR",
    "FOPT",
    "FWPR",
    "FWPT",
    "RGIP",
    "ROIP",
    "RPTRST",
    "RPTSCHED",
    "RWIP",
    "WBHP",
    "WGPR",
    "WOPR",
    "WTHP",
    "WWPR",
]


class SummaryVectorKeyword(Keyword[list[str]]):
    """
    A SUMMARY-section vector selector (`FOPR`, `WOPR`, `ROIP`, ...).

    Eclipse allows an optional list of object names/numbers (well names for
    `W*` vectors, region numbers for `R*` vectors) immediately following the
    keyword, terminated by `/`. A bare keyword with no list (or an empty
    list before the `/`) means "every object of the relevant type" and is
    conventionally written as a lone `/` or nothing at all.

    `parse` returns the requested object list (each entry as a string,
    since well names are strings and region numbers come through as
    their original token text), or an empty list `[]` meaning "all
    objects". It is never `None` when the keyword is merely unrestricted,
    since the keyword *is* present and the requested vector should still
    be activated. `None` is reserved for "keyword absent from the deck".
    """

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> list[str] | None:
        record = deck.get_first_record_for(self.name)
        if record is None:
            return None
        return tokenize(record.body.split("/", 1)[0])


FOPR = SummaryVectorKeyword("FOPR")
"""`FOPR` - field oil production rate. Takes no object list (whole field)."""

FWPR = SummaryVectorKeyword("FWPR")
"""`FWPR` - field water production rate. Takes no object list (whole field)."""

FGPR = SummaryVectorKeyword("FGPR")
"""`FGPR` - field gas production rate. Takes no object list (whole field)."""

FOPT = SummaryVectorKeyword("FOPT")
"""`FOPT` - field cumulative oil production total. Takes no object list."""

FWPT = SummaryVectorKeyword("FWPT")
"""`FWPT` - field cumulative water production total. Takes no object list."""

FGPT = SummaryVectorKeyword("FGPT")
"""`FGPT` - field cumulative gas production total. Takes no object list."""

WOPR = SummaryVectorKeyword("WOPR")
"""
`WOPR ['WELL1' 'WELL2' ...] /` - well oil production rate.

`parse` returns the requested well-name list, or `[]` for "every well"
when the keyword appears with no names before its `/`.
"""

WWPR = SummaryVectorKeyword("WWPR")
"""`WWPR ['WELL1' ...] /` - well water production rate (see `WOPR`)."""

WGPR = SummaryVectorKeyword("WGPR")
"""`WGPR ['WELL1' ...] /` - well gas production rate (see `WOPR`)."""

WBHP = SummaryVectorKeyword("WBHP")
"""`WBHP ['WELL1' ...] /` - well bottom-hole pressure (see `WOPR`)."""

WTHP = SummaryVectorKeyword("WTHP")
"""`WTHP ['WELL1' ...] /` - well tubing-head pressure (see `WOPR`)."""

ROIP = SummaryVectorKeyword("ROIP")
"""
`ROIP [region1 region2 ...] /` - reservoir oil in place, per `FIPNUM`
region.

`parse` returns the requested region-number list (as strings), or `[]`
for "every region" when the keyword appears with no numbers before its
`/`.
"""

RGIP = SummaryVectorKeyword("RGIP")
"""`RGIP [region1 ...] /` - reservoir gas in place, per region (see `ROIP`)."""

RWIP = SummaryVectorKeyword("RWIP")
"""`RWIP [region1 ...] /` - reservoir water in place, per region (see `ROIP`)."""


class MnemonicReportKeyword(Keyword[dict[str, int | None]]):
    """
    A reporting-control keyword whose body is a list of `MNEMONIC` or
    `MNEMONIC=N` entries (`RPTRST`, `RPTSCHED`).

    Example body: `BASIC=2 FREQ=3 ALLPROPS`.

    `parse` returns `{mnemonic: level_or_None}`, mapping each mnemonic
    to its integer level (e.g. `2` for `BASIC=2`) or `None` for a
    bare flag mnemonic with no `=value` (e.g. `ALLPROPS`).
    """

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> dict[str, int | None] | None:
        record = deck.get_first_record_for(self.name)
        if record is None:
            return None

        tokens = tokenize(record.body.split("/", 1)[0])
        result: dict[str, int | None] = {}
        for token in tokens:
            if "=" in token:
                mnemonic, _, raw_value = token.partition("=")
                try:
                    result[mnemonic.upper()] = int(raw_value)
                except ValueError as exc:
                    raise DeckParseError(
                        f"{self.name}: mnemonic {mnemonic!r} has non-integer "
                        f"level {raw_value!r}: {exc}"
                    ) from exc
            else:
                result[token.upper()] = None
        return result


RPTRST = MnemonicReportKeyword("RPTRST")
"""
`RPTRST  MNEMONIC[=N] ... /` - restart-file output control.

Selects which arrays are written to the restart file and at what detail
level. `parse` returns `{mnemonic: level_or_None}`, e.g.
`{"BASIC": 2}` for `RPTRST BASIC=2 /`.
"""

RPTSCHED = MnemonicReportKeyword("RPTSCHED")
"""
`RPTSCHED  MNEMONIC[=N] ... /` - print-summary (.PRT file) output control
for the SCHEDULE section.

Same mnemonic/level shape as `RPTRST`, but governs printed report
content rather than the binary restart file.
"""

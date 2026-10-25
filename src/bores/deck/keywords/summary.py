"""
SUMMARY section keyword implementations.

The SUMMARY section declares which result vectors Eclipse should write to
the summary file (`.SMSPEC` / `.UNSMRY`) for later plotting and analysis.
Most SUMMARY keywords are *selector* keywords: their bare presence in the
deck means "output this vector for every relevant object" (every well,
every region, the whole field, etc.), optionally restricted to a list of
named/numbered objects.

**Field vectors** (whole-reservoir totals, no object list):

- `FOPR` / `FWPR` / `FGPR` / `FLPR`: field oil/water/gas/liquid production rate.
- `FWIR` / `FGIR`: field water/gas injection rate.
- `FOPT` / `FWPT` / `FGPT`: field oil/water/gas production cumulative total.
- `FWIT` / `FGIT`: field water/gas injection cumulative total.
- `FWCT`: field water cut.
- `FGOR`: field gas-oil ratio.
- `FOIR` / `FOIT`: field oil injection rate/total. Rare in practice; Eclipse still defines them.

**Well vectors** (one series per well; optionally restricted to named wells):

- `WOPR` / `WWPR` / `WGPR` / `WLPR`: well oil/water/gas/liquid production rate.
- `WWIR` / `WGIR`: well water/gas injection rate.
- `WOPT` / `WWPT` / `WGPT`: well oil/water/gas production cumulative total.
- `WWIT` / `WGIT`: well water/gas injection cumulative total.
- `WWCT`: well water cut.
- `WGOR`: well gas-oil ratio.
- `WBHP`: well bottom-hole pressure.
- `WTHP`: well tubing-head pressure.
- `WOIR` / `WOIT`: well oil injection rate/total. Rare in practice; Eclipse still defines them.

Every vector above has a matching `bores.wells.summary` `Summary`
implementation of the same mnemonic.

**Region vectors** (one series per FIP region; optionally restricted to
named region numbers):

- `ROIP` / `RGIP` / `RWIP`: reservoir oil/gas/water in place.

**Block vectors** (one series per listed grid cell, given as `I J K /` lines closed
by a final `/`):

- `BPR` / `BPRESSUR`: oil phase pressure in the cell.
- `BGSAT` / `BSGAS`: gas saturation in the cell.
- `BOSAT` / `BSOIL`: oil saturation in the cell.
- `BWSAT` / `BSWAT`: water saturation in the cell.
- `BRS` / `BRV`: dissolved gas-oil ratio / vaporized oil-gas ratio in the cell.
- `BOKR` / `BWKR` / `BGKR`: oil/water/gas relative permeability in the cell.

Every block vector has a matching `bores.wells.summary` `Summary` implementation.

**Reporting controls** (`RPTRST` / `RPTSCHED`) configure restart-file /
print-summary mnemonics rather than naming a result vector; they are kept
in this module because they only ever appear in the SUMMARY (and
SCHEDULE) sections and share no shape with the vector selectors above.
"""

import typing

from bores.datastructures import GridDimensions
from bores.deck.core import Deck, DeckParseError, first_segment, tokenize
from bores.deck.keywords.base import (
    Field,
    Keyword,
    RepeatedRecordKeyword,
    TimeUnit,
    get_schedule_times,
)
from bores.deck.operators import Operation

__all__ = [
    "BGKR",
    "BGSAT",
    "BOKR",
    "BOSAT",
    "BPR",
    "BPRESSUR",
    "BRS",
    "BRV",
    "BSGAS",
    "BSOIL",
    "BSWAT",
    "BWKR",
    "BWSAT",
    "FGIR",
    "FGIT",
    "FGOR",
    "FGPR",
    "FGPT",
    "FLPR",
    "FOIR",
    "FOIT",
    "FOPR",
    "FOPT",
    "FWCT",
    "FWIR",
    "FWIT",
    "FWPR",
    "FWPT",
    "RGIP",
    "ROIP",
    "RPTRST",
    "RPTSCHED",
    "RWIP",
    "WBHP",
    "WGIR",
    "WGIT",
    "WGOR",
    "WGPR",
    "WGPT",
    "WLPR",
    "WOIR",
    "WOIT",
    "WOPR",
    "WOPT",
    "WTHP",
    "WWCT",
    "WWIR",
    "WWIT",
    "WWPR",
    "WWPT",
]


class SummaryVectorKeyword(Keyword[list[str]]):
    """
    A `SUMMARY`-section vector selector (`FOPR`, `WOPR`, `ROIP`, ...).

    Eclipse allows an optional list of object names/numbers (well names for
    `W*` vectors, region numbers for `R*` vectors) immediately following the
    keyword, terminated by `/`. A bare keyword with no list (or an empty
    list before the `/`) means "every object of the relevant type" and is
    conventionally written as a lone `/` or nothing at all.

    `parse` returns the requested object list (each entry as a string,
    since well names are strings and region numbers come through as
    their original token text), joined across every occurrence of the keyword,
    or an empty list `[]` meaning "all objects". It is never `None` when the keyword is merely unrestricted,
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
        records = deck.get_records_for(self.name)
        if not records:
            return None

        # A vector may be requested more than once (e.g. one `WOPR` block per
        # group of wells); every request counts.
        names: list[str] = []
        for record in records:
            requested = tokenize(first_segment(record.body))
            if not requested:
                return []  # An unrestricted request covers every object.
            names.extend(requested)
        return names


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

FLPR = SummaryVectorKeyword("FLPR")
"""`FLPR` - field liquid (oil plus water) production rate. Takes no object list."""

FWIR = SummaryVectorKeyword("FWIR")
"""`FWIR` - field water injection rate. Takes no object list."""

FOIR = SummaryVectorKeyword("FOIR")
"""`FOIR` - field oil injection rate. Rare in practice; Eclipse still defines it. Takes no object list."""

FGIR = SummaryVectorKeyword("FGIR")
"""`FGIR` - field gas injection rate. Takes no object list."""

FWIT = SummaryVectorKeyword("FWIT")
"""`FWIT` - field cumulative water injection total. Takes no object list."""

FOIT = SummaryVectorKeyword("FOIT")
"""`FOIT` - field cumulative oil injection total. Rare in practice; Eclipse still defines it. Takes no object list."""

FGIT = SummaryVectorKeyword("FGIT")
"""`FGIT` - field cumulative gas injection total. Takes no object list."""

FWCT = SummaryVectorKeyword("FWCT")
"""`FWCT` - field water cut (water over oil plus water production). Takes no object list."""

FGOR = SummaryVectorKeyword("FGOR")
"""`FGOR` - field gas-oil ratio (gas over oil production). Takes no object list."""

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

WLPR = SummaryVectorKeyword("WLPR")
"""`WLPR ['WELL1' ...] /` - well liquid (oil plus water) production rate (see `WOPR`)."""

WWIR = SummaryVectorKeyword("WWIR")
"""`WWIR ['WELL1' ...] /` - well water injection rate (see `WOPR`)."""

WOIR = SummaryVectorKeyword("WOIR")
"""`WOIR ['WELL1' ...] /` - well oil injection rate. Rare in practice; Eclipse still defines it (see `WOPR`)."""

WGIR = SummaryVectorKeyword("WGIR")
"""`WGIR ['WELL1' ...] /` - well gas injection rate (see `WOPR`)."""

WOPT = SummaryVectorKeyword("WOPT")
"""`WOPT ['WELL1' ...] /` - well cumulative oil production total (see `WOPR`)."""

WWPT = SummaryVectorKeyword("WWPT")
"""`WWPT ['WELL1' ...] /` - well cumulative water production total (see `WOPR`)."""

WGPT = SummaryVectorKeyword("WGPT")
"""`WGPT ['WELL1' ...] /` - well cumulative gas production total (see `WOPR`)."""

WWIT = SummaryVectorKeyword("WWIT")
"""`WWIT ['WELL1' ...] /` - well cumulative water injection total (see `WOPR`)."""

WOIT = SummaryVectorKeyword("WOIT")
"""`WOIT ['WELL1' ...] /` - well cumulative oil injection total. Rare in practice; Eclipse still defines it (see `WOPR`)."""

WGIT = SummaryVectorKeyword("WGIT")
"""`WGIT ['WELL1' ...] /` - well cumulative gas injection total (see `WOPR`)."""

WWCT = SummaryVectorKeyword("WWCT")
"""`WWCT ['WELL1' ...] /` - well water cut (see `WOPR`)."""

WGOR = SummaryVectorKeyword("WGOR")
"""`WGOR ['WELL1' ...] /` - well gas-oil ratio (see `WOPR`)."""

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


def block_vector(name: str) -> RepeatedRecordKeyword[int]:
    """
    Build a `SUMMARY`-section block vector selector.

    Block vectors list the grid cells to report, one `I J K /` line per cell, closed
    by a final `/`.

    :param name: Keyword name (for example `"BPR"`).
    :returns: A keyword whose `parse` returns one `{"i": ..., "j": ..., "k": ...}`
        dict per listed cell, using the deck's own 1-based cell indices, in file order.
        Repeated occurrences of the keyword are concatenated. `None` if the keyword is
        absent.
    """
    return RepeatedRecordKeyword[int](
        name,
        fields=[Field("i", int), Field("j", int), Field("k", int)],
    )


BGKR = block_vector("BGKR")
"""
`BGKR` - gas relative permeability in each listed grid cell.

Takes one `I J K /` line per cell, closed by a final `/`. `parse` returns a list of
`{"i", "j", "k"}` dicts.
"""

BGSAT = block_vector("BGSAT")
"""
`BGSAT` - gas saturation in each listed grid cell.

Takes one `I J K /` line per cell, closed by a final `/`. `parse` returns a list of
`{"i", "j", "k"}` dicts.
"""

BOKR = block_vector("BOKR")
"""
`BOKR` - oil relative permeability in each listed grid cell.

Takes one `I J K /` line per cell, closed by a final `/`. `parse` returns a list of
`{"i", "j", "k"}` dicts.
"""

BOSAT = block_vector("BOSAT")
"""
`BOSAT` - oil saturation in each listed grid cell.

Takes one `I J K /` line per cell, closed by a final `/`. `parse` returns a list of
`{"i", "j", "k"}` dicts.
"""

BPR = block_vector("BPR")
"""
`BPR` - oil phase pressure in each listed grid cell.

Takes one `I J K /` line per cell, closed by a final `/`. `parse` returns a list of
`{"i", "j", "k"}` dicts.
"""

BPRESSUR = block_vector("BPRESSUR")
"""
`BPRESSUR` - oil phase pressure in each listed grid cell.

Takes one `I J K /` line per cell, closed by a final `/`. `parse` returns a list of
`{"i", "j", "k"}` dicts.
"""

BRS = block_vector("BRS")
"""
`BRS` - dissolved gas-oil ratio in each listed grid cell.

Takes one `I J K /` line per cell, closed by a final `/`. `parse` returns a list of
`{"i", "j", "k"}` dicts.
"""

BRV = block_vector("BRV")
"""
`BRV` - vaporized oil-gas ratio in each listed grid cell.

Takes one `I J K /` line per cell, closed by a final `/`. `parse` returns a list of
`{"i", "j", "k"}` dicts.
"""

BSGAS = block_vector("BSGAS")
"""
`BSGAS` - gas saturation in each listed grid cell.

Takes one `I J K /` line per cell, closed by a final `/`. `parse` returns a list of
`{"i", "j", "k"}` dicts.
"""

BSOIL = block_vector("BSOIL")
"""
`BSOIL` - oil saturation in each listed grid cell.

Takes one `I J K /` line per cell, closed by a final `/`. `parse` returns a list of
`{"i", "j", "k"}` dicts.
"""

BSWAT = block_vector("BSWAT")
"""
`BSWAT` - water saturation in each listed grid cell.

Takes one `I J K /` line per cell, closed by a final `/`. `parse` returns a list of
`{"i", "j", "k"}` dicts.
"""

BWKR = block_vector("BWKR")
"""
`BWKR` - water relative permeability in each listed grid cell.

Takes one `I J K /` line per cell, closed by a final `/`. `parse` returns a list of
`{"i", "j", "k"}` dicts.
"""

BWSAT = block_vector("BWSAT")
"""
`BWSAT` - water saturation in each listed grid cell.

Takes one `I J K /` line per cell, closed by a final `/`. `parse` returns a list of
`{"i", "j", "k"}` dicts.
"""


class MnemonicReportKeyword(Keyword[list[dict[str, typing.Any]]]):
    """
    A reporting-control keyword (`RPTRST`, `RPTSCHED`) that may appear many times in the
    `SCHEDULE` section, each occurrence replacing the settings from its `schedule_time` onwards.

    One occurrence is written either with mnemonics, `MNEMONIC` or `MNEMONIC=N` entries such as
    `BASIC=2 FREQ=3 ALLPROPS`, or in the older form of positional integers such as `1 1 0 1`,
    where `1*` (or `N*`) leaves a position at its default. The two forms cannot be mixed.

    `parse` returns one dict per occurrence, in file order:

    - `"mnemonics"`: `{mnemonic: level_or_None}`, the integer level (e.g. `2` for `BASIC=2`),
      or `None` for a flag with no `=value` (e.g. `ALLPROPS`). Empty for the integer form.
    - `"flags"`: the positional integers, `None` where a position is defaulted. Empty for the
      mnemonic form. Their meaning depends on their position, which is kept as written.
    - `"schedule_time"`: elapsed time at which the occurrence takes effect, in the deck's time
      unit.
    """

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
        time_unit: TimeUnit = "days",
    ) -> list[dict[str, typing.Any]] | None:
        records = deck.get_records_for(self.name)
        if not records:
            return None

        times = (
            schedule_times if schedule_times is not None else get_schedule_times(deck, time_unit)
        )
        return [
            {
                **self.parse_tokens(tokenize(first_segment(record.body))),
                "schedule_time": times.get(record.start, 0.0),
            }
            for record in records
        ]

    def parse_tokens(self, tokens: list[str]) -> dict[str, typing.Any]:
        """
        Reads the entries of one occurrence.

        :param tokens: The occurrence's entries.
        :returns: Its `"mnemonics"` and `"flags"`.
        :raises DeckParseError: If a level is not an integer, or the two forms are mixed.
        """
        is_flag = [token == "1*" or token.lstrip("+-").isdigit() for token in tokens]
        if tokens and all(is_flag):
            return {
                "mnemonics": {},
                "flags": [None if token == "1*" else int(token) for token in tokens],
            }
        if any(is_flag):
            raise DeckParseError(
                f"{self.name}: positional integers cannot be mixed with mnemonics, got {tokens}."
            )

        mnemonics: dict[str, int | None] = {}
        for token in tokens:
            mnemonic, equals, raw_level = token.partition("=")
            if not equals:
                mnemonics[mnemonic.upper()] = None
                continue
            try:
                mnemonics[mnemonic.upper()] = int(raw_level)
            except ValueError as exc:
                raise DeckParseError(
                    f"{self.name}: mnemonic {mnemonic!r} has non-integer level {raw_level!r}."
                ) from exc
        return {"mnemonics": mnemonics, "flags": []}


RPTRST = MnemonicReportKeyword("RPTRST")
"""
`RPTRST  MNEMONIC[=N] ... /` - restart-file output control.

Selects which arrays are written to the restart file and at what detail level. It may appear
many times in `SCHEDULE`. `parse` returns one dict per occurrence, e.g.
`[{"mnemonics": {"BASIC": 2}, "flags": [], "schedule_time": 0.0}]` for `RPTRST BASIC=2 /`.
"""

RPTSCHED = MnemonicReportKeyword("RPTSCHED")
"""
`RPTSCHED  MNEMONIC[=N] ... /` - print-summary (`.PRT` file) output control for the
`SCHEDULE` section.

Same shape as `RPTRST`, but governs printed report content rather than the binary restart
file.
"""

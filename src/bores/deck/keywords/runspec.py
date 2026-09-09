"""
RUNSPEC section keyword implementations.

The RUNSPEC section is the first section in an Eclipse black-oil data file.
It declares the run's meta-data: grid dimensions, active phases, unit system,
table-sizing hints, well/group dimensioning, and the simulation start date.
"""

from bores.datastructures import GridDimensions
from bores.deck.core import Deck
from bores.deck.keywords.base import (
    DateKeyword,
    Field,
    FlagKeyword,
    Keyword,
    RecordKeyword,
)
from bores.deck.operators import Operation

__all__ = [
    "DISGAS",
    "EQLDIMS",
    "FIELD",
    "GAS",
    "LAB",
    "METRIC",
    "NOSIM",
    "OIL",
    "REGDIMS",
    "START",
    "TABDIMS",
    "TITLE",
    "UNIFIN",
    "UNIFOUT",
    "VAPOIL",
    "WATER",
    "WELLDIMS",
]


OIL = FlagKeyword("OIL")
"""
`OIL` - enable the oil phase.

When present, Eclipse activates oil conservation equations and expects
oil PVT data in the PROPS section.
"""

GAS = FlagKeyword("GAS")
"""
`GAS` - enable the gas phase.

When present, Eclipse activates gas conservation equations and expects
gas PVT data in the PROPS section.
"""

WATER = FlagKeyword("WATER")
"""
`WATER` - enable the water phase.

When present, Eclipse activates water conservation equations and expects
water PVT data (`PVTW`) in the PROPS section.
"""

DISGAS = FlagKeyword("DISGAS")
"""
`DISGAS` - enable dissolved-gas-in-oil modelling (live oil).

Requires `OIL` and `GAS` to also be active.  Live-oil PVT data
must be supplied via `PVTO` (or `PVCO`) in the PROPS section.
"""

VAPOIL = FlagKeyword("VAPOIL")
"""
`VAPOIL` - enable vaporized-oil-in-gas modelling (wet gas / condensate).

Requires `OIL` and `GAS` to also be active.  Wet-gas PVT data
must be supplied via `PVTG` in the PROPS section.
"""

FIELD = FlagKeyword("FIELD")
"""
`FIELD` - use the Field unit system.

Lengths in ft, pressures in psi, volumes in bbl / Mscf,
permeability in mD, temperatures in °F.
"""

METRIC = FlagKeyword("METRIC")
"""
`METRIC` - use the Metric (SI-like) unit system.

Lengths in m, pressures in bar(a), volumes in m³,
permeability in mD, temperatures in °C.
"""

LAB = FlagKeyword("LAB")
"""
`LAB` - use the Laboratory unit system.

Lengths in cm, pressures in atm, volumes in cc,
permeability in mD, temperatures in °C.
"""

NOSIM = FlagKeyword("NOSIM")
"""
`NOSIM` - data-check mode: parse and validate the deck without
running the simulation.

Useful during deck development to catch input errors quickly.
"""

UNIFIN = FlagKeyword("UNIFIN")
"""
`UNIFIN` - read unified (single-file) restart input.

When present, Eclipse reads restart data from a single `.UNRST`
file rather than individual step files.
"""

UNIFOUT = FlagKeyword("UNIFOUT")
"""
`UNIFOUT` - write unified (single-file) restart output.

When present, Eclipse writes restart data to a single `.UNRST`
file rather than individual step files.
"""


class TitleKeyword(Keyword[str]):
    """
    The `TITLE` keyword: a single free-text line immediately after the
    keyword name, terminated by end-of-line (no `/` required).

    Eclipse allows the title to contain any characters; it is used as a
    label on simulator output. The parsed value is the stripped raw text
    of the body, or an empty string if the keyword has no body.
    """

    def __init__(self) -> None:
        super().__init__("TITLE")

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> str | None:
        record = deck.get_first_record_for(self.name)
        if record is None:
            return None
        # The body is everything after the keyword name up to the scanner's
        # first '/'. For `TITLE` there is no '/' so the scanner will treat it
        # as a bare keyword with an empty body when the next keyword follows
        # immediately. Either way, strip surrounding whitespace.
        return record.body.strip()


TITLE = TitleKeyword()
"""
`TITLE` - free-text simulation case label.

The value on the line immediately following `TITLE` is used as a
header on Eclipse print-file output.  No `/` terminator is required.

`parse` returns the stripped title string, or `None` if the
keyword is absent.
"""

START = DateKeyword("START")
"""
`START D MON YYYY /` - simulation start date.

Sets the calendar date corresponding to time zero of the run.
Affects how `DATES` entries are interpreted and how output is labelled.

`parse` returns a :class:`datetime.date`, or `None` if absent.

Example:

    START
        1 JAN 2020 /
"""

TABDIMS = RecordKeyword[int](
    "TABDIMS",
    fields=[
        Field("ntsfun", int, required=False, default=1),
        Field("ntpvt", int, required=False, default=1),
        Field("nssfun", int, required=False, default=20),
        Field("nppvt", int, required=False, default=20),
        Field("ntfip", int, required=False, default=1),
        Field("nrpvt", int, required=False, default=20),
    ],
)
"""
`TABDIMS NTSFUN NTPVT NSSFUN NPPVT NTFIP NRPVT ... /`
- saturation/PVT table dimensioning.

Provides upper bounds used by Eclipse to pre-allocate table memory.
All parameters are optional with simulator defaults; only the first
six are parsed here (the most commonly needed ones).

Fields:

- `ntsfun` - number of saturation-function families / regions.
- `ntpvt`  - number of PVT-function families / regions.
- `nssfun` - max number of nodes in any saturation table.
- `nppvt`  - max number of pressure nodes in any PVT table.
- `ntfip`  - max number of FIP regions (`FIPNUM`).
- `nrpvt`  - max number of Rs/Rv nodes in a live-oil / wet-gas table.
"""

WELLDIMS = RecordKeyword[int](
    "WELLDIMS",
    fields=[
        Field("nwmaxz", int, required=False, default=1),
        Field("ncwmax", int, required=False, default=1),
        Field("ngmaxz", int, required=False, default=1),
        Field("nwgmax", int, required=False, default=1),
    ],
)
"""
`WELLDIMS NWMAXZ NCWMAX NGMAXZ NWGMAX ... /`
- well and group dimensioning.

Provides upper bounds for well/group memory allocation.

Fields:

- `nwmaxz` - maximum number of wells.
- `ncwmax` - maximum number of connections (completions) per well.
- `ngmaxz` - maximum number of groups.
- `nwgmax` - maximum number of wells in any one group.
"""

EQLDIMS = RecordKeyword[int](
    "EQLDIMS",
    fields=[
        Field("ntequl", int, required=False, default=1),
        Field("ndprvd", int, required=False, default=20),
        Field("ndrxvd", int, required=False, default=20),
        Field("nttrvd", int, required=False, default=20),
        Field("nstrvd", int, required=False, default=20),
    ],
)
"""
`EQLDIMS NTEQUL NDPRVD NDRXVD NTTRVD NSTRVD /`
- equilibration region dimensioning.

All parameters are optional with simulator defaults.

Fields:

- `ntequl` - maximum number of equilibration regions.
- `ndprvd` - maximum number of depth nodes in pressure-depth tables.
- `ndrxvd` - maximum number of depth nodes in Rs/Rv-depth tables.
- `nttrvd` - maximum number of depth nodes in temp-depth tables.
- `nstrvd` - maximum number of depth nodes in saturation-depth tables.
"""

REGDIMS = RecordKeyword[int](
    "REGDIMS",
    fields=[
        Field("ntfip", int, required=False, default=1),
        Field("nmfip", int, required=False, default=0),
        Field("nrfreg", int, required=False, default=0),
        Field("ntfreg", int, required=False, default=0),
        Field("nplmix", int, required=False, default=0),
    ],
)
"""
`REGDIMS NTFIP NMFIP NRFREG NTFREG NPLMIX /`
- FIP/region dimensioning.

All parameters are optional with simulator defaults.

Fields:

- `ntfip`  - maximum number of FIP regions.
- `nmfip`  - maximum number of multiply-defined FIP regions.
- `nrfreg` - maximum number of reservoir fluid-in-place regions.
- `ntfreg` - maximum number of tracer FIP regions.
- `nplmix` - maximum number of polymer mixing regions.
"""

"""
RUNSPEC section keyword implementations.

The `RUNSPEC` section is the first section in an Eclipse black-oil data file.
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
    OptionListKeyword,
    RecordKeyword,
)
from bores.deck.keywords.parsers import uppercase
from bores.deck.operators import Operation

__all__ = [
    "DISGAS",
    "ENDSCALE",
    "EQLDIMS",
    "EQLOPTS",
    "FAULTDIM",
    "FIELD",
    "FMTIN",
    "FMTOUT",
    "GAS",
    "LAB",
    "MESSAGES",
    "METRIC",
    "NOECHO",
    "NOSIM",
    "NSTACK",
    "OIL",
    "PIMTDIMS",
    "REGDIMS",
    "SATOPTS",
    "START",
    "TABDIMS",
    "TITLE",
    "UNIFIN",
    "UNIFOUT",
    "VAPOIL",
    "VFPIDIMS",
    "VFPPDIMS",
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

Requires `OIL` and `GAS` to also be active. Live-oil PVT data
must be supplied via `PVTO` (or `PVCO`) in the PROPS section.
"""

VAPOIL = FlagKeyword("VAPOIL")
"""
`VAPOIL` - enable vaporized-oil-in-gas modelling (wet gas / condensate).

Requires `OIL` and `GAS` to also be active. Wet-gas PVT data
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
header on Eclipse print-file output. No `/` terminator is required.

`parse` returns the stripped title string, or `None` if the
keyword is absent.
"""

START = DateKeyword("START")
"""
`START D MON YYYY /` - simulation start date.

Sets the calendar date corresponding to time zero of the run.
Affects how `DATES` entries are interpreted and how output is labelled.

`parse` returns a `datetime.date`, or `None` if absent.

Example:

    START
        1 JAN 2020 /
"""

TABDIMS = RecordKeyword[int | str](
    "TABDIMS",
    fields=[
        Field("ntsfun", int, required=False, default=1),
        Field("ntpvt", int, required=False, default=1),
        Field("nssfun", int, required=False, default=20),
        Field("nppvt", int, required=False, default=20),
        Field("ntfip", int, required=False, default=1),
        Field("nrpvt", int, required=False, default=20),
        Field("max_rv_nodes", int, required=False, default=20),
        Field("ntendp", int, required=False, default=1),
        Field("num_eos_res", int, required=False, default=1),
        Field("num_eos_surface", int, required=False, default=1),
        Field("max_flux_regions", int, required=False, default=10),
        Field("max_thermal_regions", int, required=False, default=1),
        Field("ntrocc", int, required=False),
        Field("max_pressure_maintainance_regions", int, required=False, default=0),
        Field("max_kvalue_tables", int, required=False, default=0),
        Field("ntalpha", int, required=False),
        Field("asphaltene_aspkdam_max_rows", int, required=False, default=10),
        Field("asphaltene_asprewg_max_rows", int, required=False, default=10),
        Field("asphaltene_aspviso_max_rows", int, required=False, default=10),
        Field("item20_not_used", str, required=False),
        Field("asphaltene_asppw2d_max_columns", int, required=False, default=5),
        Field("asphaltene_asppw2d_max_rows", int, required=False, default=5),
        Field("asphaltene_aspwetf_max_rows", int, required=False, default=5),
        Field("num_kvalue_tables", int, required=False, default=0),
    ],
)
"""
`TABDIMS NTSFUN NTPVT NSSFUN NPPVT NTFIP NRPVT ... /`
- saturation/PVT table dimensioning.

Provides upper bounds used by Eclipse to pre-allocate table memory.
All parameters are optional with simulator defaults; only the first
six are parsed here (the most commonly needed ones).

Fields:

- `ntsfun`: number of saturation-function families / regions.
- `ntpvt`: number of PVT-function families / regions.
- `nssfun`: max number of nodes in any saturation table.
- `nppvt`: max number of pressure nodes in any PVT table.
- `ntfip`: max number of FIP regions (`FIPNUM`).
- `nrpvt`: max number of Rs/Rv nodes in a live-oil / wet-gas table.

Further items, read by position after the ones above and named as in OPM's keyword definition: `max_rv_nodes`, `ntendp`, `num_eos_res`, `num_eos_surface`, `max_flux_regions`, `max_thermal_regions`, `ntrocc`, `max_pressure_maintainance_regions`, `max_kvalue_tables`, `ntalpha`, `asphaltene_aspkdam_max_rows`, `asphaltene_asprewg_max_rows`, `asphaltene_aspviso_max_rows`, `item20_not_used`, `asphaltene_asppw2d_max_columns`, `asphaltene_asppw2d_max_rows`, `asphaltene_aspwetf_max_rows`, `num_kvalue_tables`.
"""

WELLDIMS = RecordKeyword[int](
    "WELLDIMS",
    fields=[
        Field("nwmaxz", int, required=False, default=0),
        Field("ncwmax", int, required=False, default=0),
        Field("ngmaxz", int, required=False, default=0),
        Field("nwgmax", int, required=False, default=0),
        Field("max_stages", int, required=False, default=5),
        Field("max_streams", int, required=False, default=10),
        Field("max_mixtures", int, required=False, default=5),
        Field("max_separators", int, required=False, default=4),
        Field("max_mixture_items", int, required=False, default=3),
        Field("max_completion_x", int, required=False, default=0),
        Field("max_wellist_pr_well", int, required=False, default=1),
        Field("max_dynamic_wellist", int, required=False, default=1),
        Field("max_secondary_wells", int, required=False, default=10),
        Field("max_gpp_rows", int, required=False, default=201),
    ],
)
"""
`WELLDIMS NWMAXZ NCWMAX NGMAXZ NWGMAX ... /`
- well and group dimensioning.

Provides upper bounds for well/group memory allocation.

Fields:

- `nwmaxz`: maximum number of wells.
- `ncwmax`: maximum number of connections (completions) per well.
- `ngmaxz`: maximum number of groups.
- `nwgmax`: maximum number of wells in any one group.

Further items, read by position after the ones above and named as in OPM's keyword definition: `max_stages`, `max_streams`, `max_mixtures`, `max_separators`, `max_mixture_items`, `max_completion_x`, `max_wellist_pr_well`, `max_dynamic_wellist`, `max_secondary_wells`, `max_gpp_rows`.
"""

EQLDIMS = RecordKeyword[int](
    "EQLDIMS",
    fields=[
        Field("ntequl", int, required=False, default=1),
        Field("ndprvd", int, required=False, default=2000),
        Field("ndrxvd", int, required=False, default=20),
        Field("nttrvd", int, required=False, default=1),
        Field("nstrvd", int, required=False, default=20),
    ],
)
"""
`EQLDIMS NTEQUL NDPRVD NDRXVD NTTRVD NSTRVD /`
- equilibration region dimensioning.

All parameters are optional with simulator defaults.

Fields:

- `ntequl`: maximum number of equilibration regions.
- `ndprvd`: maximum number of depth nodes in pressure-depth tables.
- `ndrxvd`: maximum number of depth nodes in Rs/Rv-depth tables.
- `nttrvd`: maximum number of depth nodes in temp-depth tables.
- `nstrvd`: maximum number of depth nodes in saturation-depth tables.
"""

REGDIMS = RecordKeyword[int](
    "REGDIMS",
    fields=[
        Field("ntfip", int, required=False, default=1),
        Field("nmfip", int, required=False, default=1),
        Field("nrfreg", int, required=False, default=0),
        Field("ntfreg", int, required=False, default=0),
        Field("max_etrack", int, required=False, default=0),
        Field("ntcreg", int, required=False, default=1),
        Field("max_opernum", int, required=False, default=0),
        Field("max_operate_dwork", int, required=False, default=0),
        Field("max_operate_iwork", int, required=False, default=0),
        Field("nplmix", int, required=False, default=1),
    ],
)
"""
`REGDIMS NTFIP NMFIPR NRFREG NTFREG MAX_ETRACK NTCREG ... NPLMIX /`
- FIP/region dimensioning.

All parameters are optional with simulator defaults.

Fields:

- `ntfip`: maximum number of FIP regions.
- `nmfip`: maximum number of multiply-defined FIP regions.
- `nrfreg`: maximum number of reservoir fluid-in-place regions.
- `ntfreg`: maximum number of tracer FIP regions.
- `max_etrack`: maximum number of extra tracked quantities.
- `ntcreg`, `max_opernum`, `max_operate_dwork`, `max_operate_iwork`: dimensions of the
  coal, `OPERNUM` and `OPERATE` features.
- `nplmix`: maximum number of polymer mixing regions (the tenth item).
"""

ENDSCALE = RecordKeyword[int | str](
    "ENDSCALE",
    fields=[
        Field(
            "directional",
            uppercase,
            required=False,
            default="NODIR",
            options={"NODIR", "DIRECT"},
        ),
        Field(
            "reversibility",
            uppercase,
            required=False,
            default="REVERS",
            options={"REVERS", "IRREVERS"},
        ),
        Field("ntendp", int, required=False, default=1),
        Field("nsendp", int, required=False, default=20),
        Field("combination_mode", int, required=False, default=0),
    ],
)
"""
`ENDSCALE DIRECT IRREVERS NTENDP NSENDP COMB_MODE /` - switch on saturation end-point scaling.

Enables the end-point scaling arrays of the `PROPS` section (`SWL`, `SWCR`, `SGU`, ...).

Fields:

- `directional`: `"NODIR"` (default), one set of end-point arrays for all directions, or
  `"DIRECT"`, a set per direction.
- `reversibility`: `"REVERS"` (default), the scaling of an imbibition curve is reversible, or
  `"IRREVERS"`.
- `ntendp`: number of end-point versus depth tables (default 1).
- `nsendp`: number of nodes in each of those tables (default 20).
- `combination_mode`: end-point scaling combination mode (default 0).
"""

SATOPTS = OptionListKeyword("SATOPTS")
"""
`SATOPTS OPTION ... /` - switch on optional saturation function features.

`parse` returns the options written, upper-cased, e.g. `["HYSTER"]` for `SATOPTS HYSTER /` (see
`EHYSTR` for the hysteresis controls). Options are not checked against a list, since the set
Eclipse accepts depends on its version.
"""

EQLOPTS = OptionListKeyword("EQLOPTS")
"""
`EQLOPTS OPTION ... /` - switch on optional equilibration features (`THPRES`, `QUIESC`,
`MOBILE`, `IRREVERS`).

`parse` returns the options written, upper-cased, e.g. `["THPRES"]` for `EQLOPTS 'THPRES' /`.
Options are not checked against a list, since the set Eclipse accepts depends on its version.
"""

VFPIDIMS = RecordKeyword[int](
    "VFPIDIMS",
    fields=[
        Field("mxmflo", int, required=False, default=0),
        Field("mxmthp", int, required=False, default=0),
        Field("nmmvft", int, required=False, default=0),
    ],
)
"""
`VFPIDIMS MXMFLO MXMTHP NMMVFT /` - injection well VFP table dimensioning.

Every item defaults to 0.

Fields:

- `mxmflo`: maximum number of flow-rate values in any table.
- `mxmthp`: maximum number of tubing-head-pressure values in any table.
- `nmmvft`: maximum number of injection VFP tables.
"""

VFPPDIMS = RecordKeyword[int](
    "VFPPDIMS",
    fields=[
        Field("mxmflo", int, required=False, default=0),
        Field("mxmthp", int, required=False, default=0),
        Field("mxmwfr", int, required=False, default=0),
        Field("mxmgfr", int, required=False, default=0),
        Field("mxmalq", int, required=False, default=0),
        Field("nmmvft", int, required=False, default=0),
    ],
)
"""
`VFPPDIMS MXMFLO MXMTHP MXMWFR MXMGFR MXMALQ NMMVFT /` - production well VFP table dimensioning.

Every item defaults to 0.

Fields:

- `mxmflo`: maximum number of flow-rate values in any table.
- `mxmthp`: maximum number of tubing-head-pressure values in any table.
- `mxmwfr`: maximum number of water-fraction values in any table.
- `mxmgfr`: maximum number of gas-fraction values in any table.
- `mxmalq`: maximum number of artificial-lift-quantity values in any table.
- `nmmvft`: maximum number of production VFP tables.
"""

FAULTDIM = RecordKeyword[int](
    "FAULTDIM",
    fields=[Field("mfsegs", int, required=False, default=0)],
)
"""
`FAULTDIM MFSEGS /` - fault dimensioning. `mfsegs` is the maximum number of fault segments
(default 0).
"""

PIMTDIMS = RecordKeyword[int](
    "PIMTDIMS",
    fields=[
        Field("ntpimt", int, required=False, default=0),
        Field("nppimt", int, required=False, default=0),
    ],
)
"""
`PIMTDIMS NTPIMT NPPIMT /` - dimensioning of the pressure-induced multiplier (`PIMULTAB`)
tables. Every item defaults to 0.

Fields:

- `ntpimt`: maximum number of multiplier tables.
- `nppimt`: maximum number of rows in any multiplier table.
"""

NSTACK = RecordKeyword[int](
    "NSTACK",
    fields=[Field("nstack", int, required=False, default=10)],
)
"""
`NSTACK N /` - size of the linear solver's search-direction stack (default 10). It may also
appear in `SCHEDULE` to change the size from that point on; only the first occurrence is read.
"""

MESSAGES = RecordKeyword[int](
    "MESSAGES",
    fields=[
        Field("message_print_limit", int, required=False, default=1000000),
        Field("comment_print_limit", int, required=False, default=1000000),
        Field("warning_print_limit", int, required=False, default=10000),
        Field("problem_print_limit", int, required=False, default=100),
        Field("error_print_limit", int, required=False, default=100),
        Field("bug_print_limit", int, required=False, default=100),
        Field("message_stop_limit", int, required=False, default=1000000),
        Field("comment_stop_limit", int, required=False, default=1000000),
        Field("warning_stop_limit", int, required=False, default=10000),
        Field("problem_stop_limit", int, required=False, default=100),
        Field("error_stop_limit", int, required=False, default=10),
        Field("bug_stop_limit", int, required=False, default=1),
        Field("group_print_limit", int, required=False, default=10),
    ],
)
"""
`MESSAGES ...  /` - limits on how many messages of each severity are printed, and at how many the
run is stopped. It may appear in any section; only the first occurrence is read.

Positional, in the order of the fields: a print limit for each of message, comment, warning,
problem, error and bug, then a stop limit for each in the same order, then `group_print_limit`.
Items left out (or written `1*`) take the defaults shown in the field definitions.
"""

FMTIN = FlagKeyword("FMTIN")
"""`FMTIN` - read restart / initial files in formatted (text) form."""

FMTOUT = FlagKeyword("FMTOUT")
"""`FMTOUT` - write output files in formatted (text) form."""

NOECHO = FlagKeyword("NOECHO")
"""
`NOECHO` - stop echoing the deck to the print file from here on. It has no effect on what is
read; it may appear in any section.
"""

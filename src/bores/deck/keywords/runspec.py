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
    "CO2STOR",
    "CO2STORE",
    "DISGAS",
    "DISGASW",
    "ENDSCALE",
    "EQLDIMS",
    "EQLOPTS",
    "FAULTDIM",
    "FIELD",
    "FMTIN",
    "FMTOUT",
    "GAS",
    "GASWAT",
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
    "VAPWAT",
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

DISGASW = FlagKeyword("DISGASW")
"""
`DISGASW` - enable dissolved-gas-in-water modelling (gas solubility in the aqueous phase).

Requires `WATER` and `GAS` to also be active. Bare/nullary keyword (no data section). `parse`
returns `True` if present in the deck, `False` otherwise.
"""

VAPOIL = FlagKeyword("VAPOIL")
"""
`VAPOIL` - enable vaporized-oil-in-gas modelling (wet gas / condensate).

Requires `OIL` and `GAS` to also be active. Wet-gas PVT data
must be supplied via `PVTG` in the PROPS section.
"""

GASWAT = FlagKeyword("GASWAT")
"""
`GASWAT` - enable the gas and water phases together.

Shorthand for declaring `GAS` and `WATER`, for a model with no oil.
"""

VAPWAT = FlagKeyword("VAPWAT")
"""
`VAPWAT` - enable vaporized-water-in-gas modelling (water that evaporates into the gas phase).

Requires `WATER` and `GAS` to also be active.
"""

CO2STORE = FlagKeyword("CO2STORE")
"""
`CO2STORE` - model CO2 storage in brine.

The fluid properties of CO2 and brine are computed from pressure, temperature and `SALINITY`
instead of being read from PVT tables. The gas phase is CO2 and the aqueous phase is brine.
"""

CO2STOR = FlagKeyword("CO2STOR")
"""
`CO2STOR` - the OPM Flow spelling of `CO2STORE`, with the same meaning.
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
        Field("num_saturation_tables", int, required=False, default=1),
        Field("num_pvt_tables", int, required=False, default=1),
        Field("max_saturation_table_nodes", int, required=False, default=20),
        Field("max_pvt_pressure_nodes", int, required=False, default=20),
        Field("max_fluid_in_place_regions", int, required=False, default=1),
        Field("max_dissolved_gas_ratio_nodes", int, required=False, default=20),
        Field("max_vaporized_oil_ratio_nodes", int, required=False, default=20),
        Field("max_endpoint_depth_tables", int, required=False, default=1),
        Field("num_reservoir_eos_regions", int, required=False, default=1),
        Field("num_surface_eos_regions", int, required=False, default=1),
        Field("max_flux_regions", int, required=False, default=10),
        Field("max_thermal_regions", int, required=False, default=1),
        Field("max_rock_compaction_regions", int, required=False),
        Field("max_pressure_maintenance_regions", int, required=False, default=0),
        Field("max_kvalue_tables", int, required=False, default=0),
        Field("max_alpha_tables", int, required=False),
        Field("asphaltene_aspkdam_max_rows", int, required=False, default=10),
        Field("asphaltene_asprewg_max_rows", int, required=False, default=10),
        Field("asphaltene_aspviso_max_rows", int, required=False, default=10),
        Field("unused_item_20", str, required=False),
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
All parameters are optional with simulator defaults.

Fields:

- `num_saturation_tables` (`NTSFUN`): number of saturation function tables (`SATNUM` regions).
- `num_pvt_tables` (`NTPVT`): number of PVT tables (`PVTNUM` regions).
- `max_saturation_table_nodes` (`NSSFUN`): max number of nodes in any saturation table.
- `max_pvt_pressure_nodes` (`NPPVT`): max number of pressure nodes in any PVT table.
- `max_fluid_in_place_regions` (`NTFIP`): max number of fluid-in-place regions (`FIPNUM`).
- `max_dissolved_gas_ratio_nodes` (`NRPVT`): max number of `Rs` nodes in a live-oil table.
- `max_vaporized_oil_ratio_nodes`: max number of `Rv` nodes in a wet-gas table.
- `max_endpoint_depth_tables`: max number of end-point scaling versus depth tables.
- `num_reservoir_eos_regions`: number of equation-of-state regions for the reservoir
  (compositional runs).
- `num_surface_eos_regions`: number of equation-of-state regions for the surface (compositional
  runs).
- `max_flux_regions`: max number of flux regions (`FLUXNUM`).
- `max_thermal_regions`: max number of thermal regions (`THERMNUM`).
- `max_rock_compaction_regions`: max number of rock compaction regions.
- `max_pressure_maintenance_regions`: max number of pressure maintenance regions.
- `max_kvalue_tables`: max number of K-value tables.
- `max_alpha_tables`: max number of `ALPHA` tables.
- `asphaltene_aspkdam_max_rows`: max number of rows in an asphaltene damage (`ASPKDAM`) table.
- `asphaltene_asprewg_max_rows`: max number of rows in an `ASPREWG` table.
- `asphaltene_aspviso_max_rows`: max number of rows in an `ASPVISO` table.
- `unused_item_20`: not used by the simulator; kept so the items after it keep their positions.
- `asphaltene_asppw2d_max_columns`: max number of columns in an `ASPPW2D` table.
- `asphaltene_asppw2d_max_rows`: max number of rows in an `ASPPW2D` table.
- `asphaltene_aspwetf_max_rows`: max number of rows in an `ASPWETF` table.
- `num_kvalue_tables`: number of K-value tables in use.
"""

WELLDIMS = RecordKeyword[int](
    "WELLDIMS",
    fields=[
        Field("max_wells", int, required=False, default=0),
        Field("max_connections_per_well", int, required=False, default=0),
        Field("max_groups", int, required=False, default=0),
        Field("max_wells_per_group", int, required=False, default=0),
        Field("max_separator_stages", int, required=False, default=5),
        Field("max_surface_streams", int, required=False, default=10),
        Field("max_mixtures", int, required=False, default=5),
        Field("max_separators", int, required=False, default=4),
        Field("max_mixture_items", int, required=False, default=3),
        Field("max_completion_x", int, required=False, default=0),
        Field("max_well_lists_per_well", int, required=False, default=1),
        Field("max_dynamic_well_lists", int, required=False, default=1),
        Field("max_secondary_wells", int, required=False, default=10),
        Field("max_gpp_table_rows", int, required=False, default=201),
    ],
)
"""
`WELLDIMS NWMAXZ NCWMAX NGMAXZ NWGMAX ... /`
- well and group dimensioning.

Provides upper bounds for well/group memory allocation.

Fields:

- `max_wells` (`NWMAXZ`): maximum number of wells.
- `max_connections_per_well` (`NCWMAX`): maximum number of connections (completions) per well.
- `max_groups` (`NGMAXZ`): maximum number of groups.
- `max_wells_per_group` (`NWGMAX`): maximum number of wells in any one group.
- `max_separator_stages`: maximum number of stages in a surface separator system (default 5).
- `max_surface_streams`: maximum number of surface streams (default 10).
- `max_mixtures`: maximum number of stream mixtures (default 5).
- `max_separators`: maximum number of separators (default 4).
- `max_mixture_items`: maximum number of items in a mixture (default 3).
- `max_completion_x`: completion-related dimension that the simulator reserves; not used here
  (default 0).
- `max_well_lists_per_well`: maximum number of well lists any one well can belong to (default
  1).
- `max_dynamic_well_lists`: maximum number of dynamic well lists (default 1).
- `max_secondary_wells`: maximum number of secondary wells (default 10).
- `max_gpp_table_rows`: maximum number of rows in a gas plant processing (`GPP`) table (default
  201).
"""

EQLDIMS = RecordKeyword[int](
    "EQLDIMS",
    fields=[
        Field("max_equilibration_regions", int, required=False, default=1),
        Field("max_pressure_depth_nodes", int, required=False, default=2000),
        Field("max_rs_rv_depth_nodes", int, required=False, default=20),
        Field("max_tracer_concentration_depth_tables", int, required=False, default=1),
        Field("max_tracer_concentration_depth_nodes", int, required=False, default=20),
    ],
)
"""
`EQLDIMS NTEQUL NDPRVD NDRXVD NTTRVD NSTRVD /`
- equilibration region dimensioning.

All parameters are optional with simulator defaults.

Fields:

- `max_equilibration_regions` (`NTEQUL`): maximum number of equilibration regions (`EQLNUM`).
  Default 1.
- `max_pressure_depth_nodes` (`NDPRVD`): maximum number of depth nodes in pressure-depth tables.
  Default 2000.
- `max_rs_rv_depth_nodes` (`NDRXVD`): maximum number of depth nodes in the `Rs`/`Rv`-depth
  tables (`RSVD`, `RVVD`, ...). Default 20.
- `max_tracer_concentration_depth_tables` (`NTTRVD`): maximum number of tracer concentration-
  depth tables (`TVDP<name>`). Default 1.
- `max_tracer_concentration_depth_nodes` (`NSTRVD`): maximum number of depth nodes in any tracer
  concentration-depth table. Default 20.
"""

REGDIMS = RecordKeyword[int](
    "REGDIMS",
    fields=[
        Field("max_fluid_in_place_regions", int, required=False, default=1),
        Field("max_fluid_in_place_region_sets", int, required=False, default=1),
        Field("max_reservoir_regions", int, required=False, default=0),
        Field("max_flux_regions", int, required=False, default=0),
        Field("max_etrack", int, required=False, default=0),
        Field("max_coal_regions", int, required=False, default=1),
        Field("max_opernum_regions", int, required=False, default=0),
        Field("max_operate_real_work_size", int, required=False, default=0),
        Field("max_operate_integer_work_size", int, required=False, default=0),
        Field("max_polymer_mixing_regions", int, required=False, default=1),
    ],
)
"""
`REGDIMS NTFIP NMFIPR NRFREG NTFREG MAX_ETRACK NTCREG ... NPLMIX /`
- FIP/region dimensioning.

All parameters are optional with simulator defaults.

Fields:

- `max_fluid_in_place_regions` (`NTFIP`): maximum number of fluid-in-place regions (`FIPNUM`).
- `max_fluid_in_place_region_sets` (`NMFIPR`): maximum number of fluid-in-place region sets
  (`FIPNUM` and any other `FIP<name>` array, such as `FIPGL`). Default 1.
- `max_reservoir_regions` (`NRFREG`): maximum number of independent reservoir regions. Default 0.
- `max_flux_regions` (`NTFREG`): maximum number of flux regions (`FLUXNUM`), as used by the flux
  options. Default 0.
- `max_etrack`: maximum number of extra tracked quantities.
- `max_coal_regions` (`NTCREG`): maximum number of coal regions (`COALNUM`). Default 1.
- `max_opernum_regions`: maximum number of `OPERNUM` regions.
- `max_operate_real_work_size`: size of the real-valued work space for `OPERATE`.
- `max_operate_integer_work_size`: size of the integer work space for `OPERATE`.
- `max_polymer_mixing_regions` (`NPLMIX`): maximum number of polymer mixing regions (the tenth
  item). Default 1.
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
        Field("num_endpoint_depth_tables", int, required=False, default=1),
        Field("num_endpoint_depth_nodes", int, required=False, default=20),
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
- `num_endpoint_depth_tables` (`NTENDP`): number of end-point versus depth tables (default 1).
- `num_endpoint_depth_nodes` (`NSENDP`): number of nodes in each of those tables (default 20).
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
        Field("max_flow_rate_values", int, required=False, default=0),
        Field("max_tubing_head_pressure_values", int, required=False, default=0),
        Field("max_injection_vfp_tables", int, required=False, default=0),
    ],
)
"""
`VFPIDIMS MXMFLO MXMTHP NMMVFT /` - injection well VFP table dimensioning.

Every item defaults to 0.

Fields:

- `max_flow_rate_values` (`MXMFLO`): maximum number of flow-rate values in any table.
- `max_tubing_head_pressure_values` (`MXMTHP`): maximum number of tubing-head-pressure values in
  any table.
- `max_injection_vfp_tables` (`NMMVFT`): maximum number of injection VFP tables.
"""

VFPPDIMS = RecordKeyword[int](
    "VFPPDIMS",
    fields=[
        Field("max_flow_rate_values", int, required=False, default=0),
        Field("max_tubing_head_pressure_values", int, required=False, default=0),
        Field("max_water_fraction_values", int, required=False, default=0),
        Field("max_gas_fraction_values", int, required=False, default=0),
        Field("max_artificial_lift_values", int, required=False, default=0),
        Field("max_production_vfp_tables", int, required=False, default=0),
    ],
)
"""
`VFPPDIMS MXMFLO MXMTHP MXMWFR MXMGFR MXMALQ NMMVFT /` - production well VFP table dimensioning.

Every item defaults to 0.

Fields:

- `max_flow_rate_values` (`MXMFLO`): maximum number of flow-rate values in any table.
- `max_tubing_head_pressure_values` (`MXMTHP`): maximum number of tubing-head-pressure values in
  any table.
- `max_water_fraction_values` (`MXMWFR`): maximum number of water-fraction values in any table.
- `max_gas_fraction_values` (`MXMGFR`): maximum number of gas-fraction values in any table.
- `max_artificial_lift_values` (`MXMALQ`): maximum number of artificial-lift-quantity values in
  any table.
- `max_production_vfp_tables` (`NMMVFT`): maximum number of production VFP tables.
"""

FAULTDIM = RecordKeyword[int](
    "FAULTDIM",
    fields=[Field("max_fault_segments", int, required=False, default=0)],
)
"""
`FAULTDIM MFSEGS /` - fault dimensioning. `max_fault_segments` is the maximum number of fault segments
(default 0).
"""

PIMTDIMS = RecordKeyword[int](
    "PIMTDIMS",
    fields=[
        Field("max_multiplier_tables", int, required=False, default=0),
        Field("max_multiplier_table_rows", int, required=False, default=0),
    ],
)
"""
`PIMTDIMS NTPIMT NPPIMT /` - dimensioning of the pressure-induced multiplier (`PIMULTAB`)
tables. Every item defaults to 0.

Fields:

- `max_multiplier_tables` (`NTPIMT`): maximum number of multiplier tables.
- `max_multiplier_table_rows` (`NPPIMT`): maximum number of rows in any multiplier table.
"""

NSTACK = RecordKeyword[int](
    "NSTACK",
    fields=[Field("stack_size", int, required=False, default=10)],
)
"""
`NSTACK N /` - size of the linear solver's search-direction stack, `stack_size` (default 10). It may also
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

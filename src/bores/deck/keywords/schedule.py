"""
SCHEDULE section keyword implementations.

The SCHEDULE section is the last section of an Eclipse black-oil deck. It
advances simulated time (`DATES` / `TSTEP`) and defines/controls wells and
groups as the run proceeds. Unlike the GRID/PROPS sections, most SCHEDULE
keywords are *time-ordered events* rather than static data: the same well
can be opened, closed, and re-targeted by repeated keyword occurrences at
different points in the deck, and callers are expected to apply them in
file order alongside the `DATES`/`TSTEP` timeline.

**Time advancement**:

- `DATES` - see `bores.deck.keywords.base.DatesKeyword`, registered
  here as part of the SCHEDULE keyword set.
- `TSTEP` - see `bores.deck.keywords.base.TStepKeyword`, likewise.

**Well/group definition**:

- `WELSPECS` - declare a well (location, group, preferred phase).
- `COMPDAT`  - declare/modify well connections (completions).
- `GRUPTREE` - declare the group hierarchy.

**Well/group control**:

- `WCONPROD` - producer rate/pressure targets.
- `WCONINJE` - injector rate/pressure targets.
- `WELOPEN`  - open/shut/stop a well or specific connections.
- `WELTARG`  - modify a single control target on an existing well.
- `WPIMULT`  - well productivity-index multiplier.
- `GCONPROD` - group production targets/guide rates.
- `GCONINJE` - group injection targets.

**Economic limits / well testing**:

- `WECON` - well economic limits (auto-shut-in thresholds).
- `WTEST` - automatic well re-opening / testing schedule.
"""

import typing

import numpy as np
import numpy.typing as npt

from bores.datastructures import GridDimensions
from bores.deck.core import Deck, DeckParseError, tokenize
from bores.deck.keywords.base import (
    DatesKeyword,
    Field,
    HeaderedScheduledRecordKeyword,
    Keyword,
    RecordKeyword,
    ScheduledRecordKeyword,
)
from bores.deck.operators import Operation
from bores.errors import ValidationError

__all__ = [
    "COMPDAT",
    "COMPSEGS",
    "DATES",
    "GCONINJE",
    "GCONPROD",
    "GECON",
    "GRUPTREE",
    "TSTEP",
    "VFPINJ",
    "VFPPROD",
    "WCONHIST",
    "WCONINJE",
    "WCONINJH",
    "WCONPROD",
    "WDFAC",
    "WECON",
    "WELOPEN",
    "WELPI",
    "WELSEGS",
    "WELSPECS",
    "WELTARG",
    "WGRUPCON",
    "WLIFT",
    "WPAVE",
    "WPIMULT",
    "WTEST",
    "VFPInjectorDeckTable",
    "VFPProducerDeckTable",
]


DATES = DatesKeyword("DATES")
"""
`DATES  D MON YYYY / ... /` - advance simulated time to one or more
explicit calendar dates.

See `bores.deck.keywords.base.DatesKeyword`. `parse` returns a
`List[datetime.date]`, or `None` if absent.
"""


class TStepKeyword(Keyword[list[float]]):
    """
    The `TSTEP` keyword: a flat list of time-step sizes terminated by
    `/`.

    `N*value` repeat syntax is already expanded by
    `bores.deck.core.tokenize`, so `30*30` correctly yields
    thirty entries of `30.0`.

    Multiple `TSTEP` blocks in the same deck are concatenated in file
    order, consistent with Eclipse semantics.

    `parse` returns a `List[float]`, or `None` when the keyword
    is absent.

    Example deck fragment:

        TSTEP
         30 30 30 90 /
    """

    def __init__(self) -> None:
        super().__init__("TSTEP")

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> list[float] | None:
        records = deck.get_records_for(self.name)
        if not records:
            return None

        steps: list[float] = []
        for record in records:
            body = record.body.split("/", 1)[0]
            tokens = tokenize(body)
            for token in tokens:
                try:
                    steps.append(float(token))
                except ValueError as exc:
                    raise DeckParseError(
                        f"{self.name}: non-numeric time-step value {token!r}: {exc}"
                    ) from exc

        return steps or None


TSTEP = TStepKeyword()
"""
`TSTEP  dt1 dt2 ... /` - advance simulated time by one or more explicit
step sizes (in the deck's declared time unit, days for FIELD/METRIC).
"""


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    value = str(value).upper()
    if value == "YES":
        return True
    if value == "NO":
        return False
    raise ValidationError(f"Expected 'YES' or 'NO', got {value!r}")


WELSPECS = ScheduledRecordKeyword[str | float | bool](
    "WELSPECS",
    fields=[
        Field("well", str),
        Field("group", str),
        Field("i", int),
        Field("j", int),
        Field("reference_depth", np.float64, required=False, default=None),
        Field(
            "phase",
            lambda v: str(v).upper(),
            required=False,
            default="OIL",
            options={"OIL", "WAT", "GAS", "LIO"},
        ),
        Field("drainage_radius", np.float64, required=False, default=None),
        Field(
            "inflow_equation",
            lambda v: str(v).upper(),
            required=False,
            default="STD",
            options={"STD", "NO"},
        ),
        Field(
            "auto_shut",
            lambda v: str(v).upper(),
            required=False,
            default="SHUT",
            options={"SHUT", "STOP"},
        ),
        Field("enable_crossflow", type=parse_bool, required=False, default="YES"),
        Field("pvt_table", int, required=False, default=0),
        Field(
            "density_calculation_method",
            lambda v: str(v).upper(),
            required=False,
            default="SEG",
            options={"SEG", "AVG"},
        ),
    ],
)
"""
`WELSPECS 'WELL' 'GROUP' I J [ref_depth] [phase] ... / ... /`
- declare a well's name, parent group, surface location, and defaults.

Multiple `WELSPECS` blocks in the same deck are concatenated in file
order (a deck typically only declares each well once, but Eclipse permits
later blocks to add more wells as drilling progresses).

Fields:

- `well`            - well name.
- `group`           - parent group name.
- `i` / `j`         - 1-based structured grid location of the wellhead.
- `reference_depth`       - BHP reference depth; defaults to the first
  completion's mid-perforation depth when absent (`None` here).
- `phase`           - preferred phase (`OIL`, `WATER`, `GAS`).
- `drainage_radius` - well drainage radius for PI calculations.
- `inflow_equation`       - inflow equation type (`STD` or `NO`).
- `auto_shut`       - automatic shut-in behaviour (`SHUT` or `STOP`).
- `enable_crossflow`       - whether crossflow between completions is allowed.
- `pvt_table`       - PVT region override (`0` = use cell's `PVTNUM`).
- `density_calculation_method`    - wellbore density calculation method (`SEG` or `AVG`).
"""

COMPDAT = ScheduledRecordKeyword[str | float](
    "COMPDAT",
    fields=[
        Field("well", str),
        Field("i", int),
        Field("j", int),
        Field("k1", int),
        Field("k2", int),
        Field(
            "status",
            lambda v: str(v).upper(),
            required=False,
            default="OPEN",
            options={"OPEN", "SHUT", "STOP", "AUTO"},
        ),
        Field("saturation_table", int, required=False, default=0),
        Field("connection_factor", np.float64, required=False, default=None),
        Field("diameter", np.float64, required=False, default=0.0),
        Field("kh", np.float64, required=False, default=None),
        Field("skin", np.float64, required=False, default=0.0),
        Field("d_factor", np.float64, required=False, default=0.0),
        Field(
            "direction",
            lambda v: str(v).upper(),
            required=False,
            default="Z",
            options={"X", "Y", "Z"},
        ),
        Field("kh_multiplier", np.float64, required=False, default=1.0),
    ],
)
"""
`COMPDAT 'WELL' I J K1 K2 [status] [sat_table] [conn_factor] ... / ... /`
- declare or modify well connections (completions).

14 standard items; the last several are very commonly defaulted (`1*`).
Multiple `COMPDAT` blocks are concatenated in file order, since the same
well/connection can be re-completed or re-parametrised later in the
schedule.

Fields:

- `well`                - well name.
- `i` / `j`             - 1-based structured grid column of the connection.
- `k1` / `k2`           - 1-based top/bottom layer of the connection
  interval (a single-layer connection has `k1 == k2`).
- `status`              - `OPEN` or `SHUT`.
- `saturation_table`           - saturation table number override (`0` = use
  the cell's `SATNUM`).
- `connection_factor`   - explicit transmissibility/connection factor;
  `None` means Eclipse computes it from geometry and `PERMX`/`PERMY`.
- `diameter`            - wellbore diameter at this connection.
- `kh`                  - explicit permeability-thickness product;
  `None` means Eclipse computes it from the grid.
- `skin`                - mechanical skin factor.
- `d_factor`            - non-Darcy (rate-dependent) skin factor.
- `direction`           - completion direction (`X`, `Y`, or `Z`).
- `kh_multiplier` - additional permeability-thickness multiplier.
"""

WELSEGS = HeaderedScheduledRecordKeyword(
    "WELSEGS",
    header_fields=[
        Field("well", str),
        Field("reference_depth", np.float64),
        Field("tubing_length_to_first_segment", np.float64),
        Field("first_segment_volume", np.float64, required=False, default=None),
        Field(
            "length_depth_mode",
            lambda v: str(v).upper(),
            required=False,
            default="INC",
            options={"INC", "ABS"},
        ),
        Field(
            "pressure_drop_model",
            lambda v: str(v).upper(),
            required=False,
            default="HFA",
        ),
    ],
    detail_fields=[
        Field("first_segment", int),
        Field("last_segment", int),
        Field("branch", int),
        Field("outlet_segment", int),
        Field("length", np.float64),
        Field("depth_change", np.float64),
        Field("diameter", np.float64),
        Field("roughness", np.float64, required=False, default=0.0),
    ],
)
"""
`WELSEGS 'WELL' DEPTH1 TLEN1 [VOL1] [LEN&DEP] [PRESDROP] / segment records... /`
- defines a multi-segment well's segment tree: one header record for the
well, then one record per segment (or contiguous segment range).

Only the main bore (`branch == 1`) is supported by
`bores.wells.deck.load_well_segments` today; a lateral branch (`branch
!= 1`) parses without error here, since that's still a faithful reading
of the keyword, but is rejected where it's actually used.

Header fields:

- `well`                          - well name.
- `reference_depth`               - depth of the first segment node (deck
  item 2, `DEPTH1`) - the same BHP/THP reference `WELSPECS` establishes.
- `tubing_length_to_first_segment` - along-hole length from the wellhead
  to the first segment node (deck item 3, `TLEN1`).
- `first_segment_volume`          - wellbore volume of the first segment
  (deck item 4, `VOL1`); `None` means Eclipse computes it from geometry.
- `length_depth_mode`             - `INC`: each segment record's own
  `length`/`depth_change` are relative to its outlet segment. `ABS`:
  they're cumulative from the first segment node instead.
- `pressure_drop_model`           - which pressure-drop components apply
  (`H`ydrostatic/`F`riction/`A`cceleration, e.g. `HFA`, `HF-`, `H--`).
  Parsed through but not yet consumed by `load_well_segments`.

Segment record fields:

- `first_segment` / `last_segment` - segment number, or an inclusive
  range of them sharing this record's own values.
- `branch`                         - which branch this segment belongs
  to (`1` is always the main bore).
- `outlet_segment`                 - the segment this one flows into,
  forming the tree; the main bore's own first segment outlets to `1`
  (the reference node itself).
- `length` / `depth_change`        - this segment's own along-hole
  length and true-vertical-depth change, interpreted per
  `length_depth_mode`.
- `diameter` / `roughness`         - tubing inner diameter and absolute
  roughness for this segment.
"""

COMPSEGS = HeaderedScheduledRecordKeyword(
    "COMPSEGS",
    header_fields=[
        Field("well", str),
    ],
    detail_fields=[
        Field("i", int),
        Field("j", int),
        Field("k", int),
        Field("branch", int, required=False, default=1),
        Field("start_length", np.float64, required=False, default=None),
        Field("end_length", np.float64, required=False, default=None),
    ],
)
"""
`COMPSEGS 'WELL' / I J K [branch] [start_length] [end_length] ... / ... /`
- maps a well's existing `COMPDAT` connections onto its `WELSEGS`
segment tree, one record per connection.

Real decks may add further, rarely-used trailing items (direction,
end-range, and an explicit connection depth); they refine cases this parser 
does not attempt to support yet (see its own docstring).

Header field:

- `well` - well name; the only thing on this record.

Detail fields:

- `i` / `j` / `k`         - the `COMPDAT` connection this record refines
  - must already exist.
- `branch`                - which `WELSEGS` branch this connection maps
  onto; only `1` (the main bore) is supported by `load_well_segments` today.
- `start_length` / `end_length` - along-branch length where this
  connection starts/ends. `None` (`1*`) means Eclipse would derive it
  from the connection's own grid-block geometry; `load_well_segments`
  requires both explicitly instead of attempting that derivation.
"""

WCONPROD = ScheduledRecordKeyword[str | float](
    "WCONPROD",
    fields=[
        Field("well", str),
        Field(
            "status",
            lambda v: str(v).upper(),
            required=False,
            default="OPEN",
            options={"OPEN", "SHUT", "STOP", "AUTO"},
        ),
        Field(
            "control_mode",
            lambda v: str(v).upper(),
            required=False,
            default=None,
            options={
                "ORAT",
                "BHP",
                "RESV",
                "WRAT",
                "LRAT",
                "GRAT",
                "THP",
                "GRUP",
            },
        ),
        Field("oil_rate", np.float64, required=False, default=0.0),
        Field("water_rate", np.float64, required=False, default=0.0),
        Field("gas_rate", np.float64, required=False, default=0.0),
        Field("liquid_rate", np.float64, required=False, default=0.0),
        Field("reservoir_volume_rate", np.float64, required=False, default=0.0),
        Field("bhp", np.float64, required=False, default=None),
        Field("thp", np.float64, required=False, default=None),
        Field("vfp_table", int, required=False, default=0),
        Field("alq", np.float64, required=False, default=0.0),
    ],
)
"""
`WCONPROD 'WELL' [status] [control_mode] [orat] ... / ... /`
- producer rate/pressure targets and the active control mode.

Fields:

- `well`         - well name.
- `status`       - `OPEN`, `SHUT`, `STOP`, or `AUTO`.
- `control_mode` - the constraint Eclipse actively controls the well
  by (e.g. `ORAT`, `WRAT`, `GRAT`, `LRAT`, `RESV`, `BHP`, `THP`,
  `GRUP`); `None` is only valid if the well is shut/stopped.
- `oil_rate` / `water_rate` / `gas_rate` / `liquid_rate` - oil/water/gas/liquid rate
  upper-limit targets.
- `reservoir_volume_rate`         - reservoir-volume rate upper-limit target.
- `bhp` / `thp`  - bottom-hole / tubing-head pressure limits;
  `None` means no limit.
- `vfp_table`    - VFP (vertical flow performance) table number for
  THP-to-BHP conversion (`0` = none assigned).
- `alq`          - artificial lift quantity the well currently operates at, used with the
  VFP table's own ALQ axis (`0` = no artificial lift). The `VFPPROD` table only defines
  how BHP varies with ALQ; this item is the value the well actually runs at.
"""

WCONINJE = ScheduledRecordKeyword[str | float](
    "WCONINJE",
    fields=[
        Field("well", str),
        Field(
            "injector_type",
            lambda v: str(v).upper(),
            options={"OIL", "WATER", "GAS"},
        ),
        Field(
            "status",
            lambda v: str(v).upper(),
            required=False,
            default="OPEN",
            options={"OPEN", "SHUT", "STOP", "AUTO"},
        ),
        Field(
            "control_mode",
            lambda v: str(v).upper(),
            required=False,
            default=None,
            options={
                "BHP",
                "RESV",
                "RATE",
                "THP",
                "GRUP",
            },
        ),
        Field("rate", np.float64, required=False, default=0.0),
        Field("reservoir_volume_rate", np.float64, required=False, default=0.0),
        Field("bhp", np.float64, required=False, default=None),
        Field("thp", np.float64, required=False, default=None),
        Field("vfp_table", int, required=False, default=0),
    ],
)
"""
`WCONINJE 'WELL' TYPE [status] [control_mode] [rate] ... / ... /`
- injector rate/pressure targets, active control mode, and injected
fluid type.

Fields:

- `well`          - well name.
- `injector_type` - injected fluid: `WATER`, `GAS`, or `OIL`.
- `status`        - `OPEN`, `SHUT`, `STOP`, or `AUTO`.
- `control_mode`  - the constraint Eclipse actively controls the well
  by (`RATE`, `RESV`, `BHP`, `THP`, or `GRUP`).
- `rate`          - surface injection rate upper-limit target.
- `reservoir_volume_rate`          - reservoir-volume injection rate upper-limit target.
- `bhp` / `thp`   - bottom-hole / tubing-head pressure limits;
  `None` means no limit.
- `vfp_table`     - VFP table number for THP-to-BHP conversion
  (`0` = none assigned).
"""

WELOPEN = ScheduledRecordKeyword[str | int](
    "WELOPEN",
    fields=[
        Field("well", str),
        Field(
            "status",
            lambda v: str(v).upper(),
            options={"OPEN", "SHUT", "STOP", "AUTO"},
        ),
        Field("i", int, required=False, default=0),
        Field("j", int, required=False, default=0),
        Field("k1", int, required=False, default=0),
        Field("k2", int, required=False, default=0),
    ],
)
"""
`WELOPEN 'WELL' STATUS [I J K1 K2] ... / ... /`
- open, shut, or stop a well, or one of its connections.

Fields:

- `well`   - well name.
- `status` - `OPEN`, `SHUT`, `STOP`, or `AUTO`.
- `i` / `j` / `k1` / `k2` - optional connection location/layer range to
  restrict the action to a single connection (or range of layers) rather
  than the whole well; all default to `0`, meaning "whole well".

When `i`/`j`/`k1`/`k2` are all absent (or `0`), the action applies to
the whole well; when given, it applies only to the connection(s) at
that location (or layer range `k1`-`k2` at column `(i, j)`).
"""

WELTARG = ScheduledRecordKeyword[str | float](
    "WELTARG",
    fields=[
        Field("well", str),
        Field(
            "control_mode",
            lambda v: str(v).upper(),
            options={
                "ORAT",
                "BHP",
                "RESV",
                "WRAT",
                "LRAT",
                "GRAT",
                "THP",
                "GRUP",
            },
        ),
        Field("value", np.float64),
    ],
)
"""
`WELTARG 'WELL' CONTROL_MODE VALUE / ... /`
- modify a single existing control target on a well without re-stating
its full `WCONPROD` / `WCONINJE` record.

Fields:

- `well`         - well name.
- `control_mode` - target being modified (e.g. `ORAT`, `BHP`, `RESV`).
- `value`        - new value for that target.
"""

WPIMULT = ScheduledRecordKeyword[str | float](
    "WPIMULT",
    fields=[
        Field("well", str),
        Field("multiplier", np.float64),
        Field("i", int, required=False, default=0),
        Field("j", int, required=False, default=0),
        Field("k1", int, required=False, default=0),
        Field("k2", int, required=False, default=0),
    ],
)
"""
`WPIMULT 'WELL' MULTIPLIER [I J K] / ... /`
- multiply a well's (or a single connection's) productivity index.

Fields:

- `well`       - well name.
- `multiplier` - PI multiplier applied on top of the existing value.
- `i` / `j` / `k1` / `k2` - optional connection location to restrict the
  multiplier to a single connection; all default to `0`, meaning
  "every connection on this well".
"""

GRUPTREE = ScheduledRecordKeyword[str](
    "GRUPTREE",
    fields=[
        Field("child", str),
        Field("parent", str),
    ],
)
"""
`GRUPTREE 'CHILD' 'PARENT' / ... /`
- declare one parent/child link in the well-group hierarchy.

Each record adds one group (or well group membership) under a parent
group; the implicit root group is always named `'FIELD'`. Multiple
`GRUPTREE` blocks are concatenated in file order.

Fields:

- `child`  - group (or sub-tree) name being attached.
- `parent` - parent group name (`'FIELD'` for top-level groups).
"""

GCONPROD = ScheduledRecordKeyword[str | float](
    "GCONPROD",
    fields=[
        Field("group", str),
        Field(
            "control_mode",
            lambda v: str(v).upper(),
            options={
                "ORAT",
                "BHP",
                "RESV",
                "WRAT",
                "LRAT",
                "GRAT",
                "FLD",
                "NONE",
            },
        ),
        Field("oil_rate", np.float64, required=False, default=0.0),
        Field("water_rate", np.float64, required=False, default=0.0),
        Field("gas_rate", np.float64, required=False, default=0.0),
        Field("liquid_rate", np.float64, required=False, default=0.0),
        Field(
            "exceed_action",
            lambda v: str(v).upper(),
            required=False,
            default="NONE",
            options={"RATE", "NONE", "CON"},
        ),
    ],
)
"""
`GCONPROD 'GROUP' CONTROL_MODE [orat] [wrat] [grat] [lrat] [exceed_action] / ... /`
- group-level production targets / guide-rate control.

Fields:

- `group`         - group name.
- `control_mode`  - constraint the group is controlled by (`ORAT`,
  `WRAT`, `GRAT`, `LRAT`, `RESV`, `FLD`, or `NONE`).
- `oil_rate` / `water_rate` / `gas_rate` / `liquid_rate` - oil/water/gas/liquid rate
  upper-limit targets for the group.
- `exceed_action` - action when an individual well in the group would
  exceed its share of the group target (`NONE`, `RATE`, `CON`, ...).
"""

GCONINJE = ScheduledRecordKeyword[str | float](
    "GCONINJE",
    fields=[
        Field("group", str),
        Field(
            "injector_type",
            lambda v: str(v).upper(),
            options={"OIL", "WATER", "GAS"},
        ),
        Field(
            "control_mode",
            lambda v: str(v).upper(),
            options={
                "RATE",
                "RESV",
                "VREP",
                "REIN",
                "FLD",
                "NONE",
            },
        ),
        Field("rate", np.float64, required=False, default=0.0),
        Field("reservoir_volume_rate", np.float64, required=False, default=0.0),
    ],
)
"""
`GCONINJE 'GROUP' TYPE CONTROL_MODE [rate] [resv] / ... /`
- group-level injection targets.

Fields:

- `group`         - group name.
- `injector_type` - injected fluid: `WATER`, `GAS`, or `OIL`.
- `control_mode`  - constraint the group is controlled by (`RATE`,
  `RESV`, `VREP`, `REIN`, or `FLD`).
- `rate`          - surface injection rate upper-limit target for the
  group.
- `reservoir_volume_rate`          - reservoir-volume injection rate upper-limit target
  for the group.
"""

WECON = ScheduledRecordKeyword[str | float | bool](
    "WECON",
    fields=[
        Field("well", str),
        Field("min_oil_rate", np.float64, required=False, default=0.0),
        Field("max_water_cut", np.float64, required=False, default=None),
        Field("max_gor", np.float64, required=False, default=None),
        Field("max_wgr", np.float64, required=False, default=None),
        Field(
            "workover_action",
            lambda v: str(v).upper(),
            required=False,
            default="WELL",
            options={"CON", "+CON", "WELL", "PLUG"},
        ),
        Field(
            "end_run",
            type=parse_bool,
            required=False,
            default=False,
        ),
    ],
)
"""
`WECON 'WELL' [min_oil_rate] [max_water_cut] [max_gor] [max_wgr] [workover_action] [end_run] / ... /`
- economic limits that automatically shut in or work over a well.

Fields:

- `well`            - well name.
- `min_oil_rate`    - minimum economic oil production rate; the well is
  shut in (per `workover_action`) once production falls below this.
- `max_water_cut`   - maximum water cut before workover/shut-in;
  `None` means no limit.
- `max_gor`         - maximum gas-oil ratio before workover/shut-in;
  `None` means no limit.
- `max_wgr`         - maximum water-gas ratio before workover/shut-in;
  `None` means no limit.
- `workover_action` - action taken when a limit is breached (`NONE`,
  `CON`, `+CON`, `WELL`, `PLUG`).
- `end_run`    - whether breaching this limit should end the
  simulation run (`YES` or `NO`).
"""

WTEST = ScheduledRecordKeyword[str | float](
    "WTEST",
    fields=[
        Field("well", str),
        Field("interval", np.float64),
        Field("reason", lambda v: str(v).upper(), required=False, default="PEW"),
    ],
)
"""
`WTEST 'WELL' INTERVAL [reason] / ... /`
- schedule automatic periodic re-opening ("testing") of a well that was
shut in for an economic or operational reason.

Fields:

- `well`     - well name.
- `interval` - time between re-open attempts, in the deck's declared
  time unit.
- `reason`   - which shut-in reason(s) this test schedule applies to,
  as a string of one-letter codes: `P` (economic), `E` (group control
  efficiency), `W` (workover/economic limit, `WECON`); default `"PEW"`
  applies to all three.
"""

WGRUPCON = ScheduledRecordKeyword[str | float](
    "WGRUPCON",
    fields=[
        Field(name="well", type=str),
        Field(
            name="available_for_group_control",
            type=lambda s: s.upper() == "YES",
            required=False,
            default=True,
        ),
        Field(name="guide_rate", type=np.float64, required=False, default=None),
        Field(
            name="guide_rate_phase",
            type=lambda v: str(v).upper(),
            required=False,
            default=None,
            options={"OIL", "WAT", "GAS", "LIQ", "RES", "COMB", "FORM"},
        ),
    ],
)
"""
`WGRUPCON 'WELL' [available_for_group_control] [guide_rate] [guide_rate_phase] / ... /`
- configure whether a well participates in group control and, optionally,
assign an explicit guide rate.

Group-control algorithms use guide rates to distribute production or
injection targets among wells belonging to the same group. A well may be
excluded from group allocation while still remaining under its own local
controls.

Fields:

- `well`                        - well name.
- `available_for_group_control` - whether the well may participate in
  automatic group control (`YES`/`NO`); defaults to `YES`.
- `guide_rate`                  - explicit guide rate used when allocating
  group targets; `None` means Eclipse computes or inherits the guide rate.
- `guide_rate_phase`            - phase used for the guide rate (`OIL`,
  `WAT`, `GAS`, `LIQ`, `RES`, `COMB`, or `FORM`).
"""

GECON = ScheduledRecordKeyword[str | float | bool](
    "GECON",
    fields=[
        Field(name="group", type=str),
        Field(name="min_oil_rate", type=np.float64, required=False, default=None),
        Field(name="min_gas_rate", type=np.float64, required=False, default=None),
        Field(name="max_water_cut", type=np.float64, required=False, default=None),
        Field(name="max_gor", type=np.float64, required=False, default=None),
        Field(name="max_wgr", type=np.float64, required=False, default=None),
        Field(
            name="workover_procedure",
            type=lambda v: str(v).upper(),
            required=False,
            default="NONE",
            options={"NONE", "CON", "+CON", "WELL", "PLUG", "RATE"},
        ),
        Field(
            name="end_run",
            type=lambda s: s.upper() == "YES",
            required=False,
            default=False,
        ),
    ],
)
"""
`GECON 'GROUP' [min_oil_rate] [min_gas_rate] [max_water_cut] [max_gor] [max_wgr] [workover_procedure] [end_run] / ... /`
- define economic operating limits for an entire production group.

When one of the specified limits is exceeded, Eclipse performs the selected
workover action on wells within the group. These limits are analogous to
`WECON`, but apply collectively to all wells in the group.

Fields:

- `group`                  - group name.
- `min_oil_rate`           - minimum economic oil production rate.
- `min_gas_rate`           - minimum economic gas production rate.
- `max_water_cut`          - maximum allowable water cut.
- `max_gor`                - maximum allowable gas-oil ratio.
- `max_wgr`    - maximum allowable water-gas ratio.
- `workover_procedure`     - action taken when a limit is exceeded
  (`NONE`, `CON`, `+CON`, `WELL`, `PLUG`, or `RATE`).
- `end_run`                - whether exceeding the limit terminates the
  simulation (`YES` or `NO`).
"""

WELPI = ScheduledRecordKeyword[str | float](
    "WELPI",
    fields=[
        Field(name="well", type=str),
        Field(name="productivity_index", type=np.float64),
    ],
)
"""
`WELPI 'WELL' TARGET_PI / ... /`
- explicitly assign a productivity index (PI) to a well.

Normally the productivity index is computed automatically from the reservoir
geometry, permeability, completion data, and well properties. `WELPI`
overrides that calculation with a user-specified target value.

Fields:

- `well`      - well name.
- `productivity_index` - explicit productivity index assigned to the well.
"""

WPAVE = RecordKeyword[str | float | bool](
    "WPAVE",
    fields=[
        Field(name="f1", type=np.float64, required=False, default=1.0),
        Field(
            name="procedure",
            type=lambda v: str(v).upper(),
            required=False,
            default="WBP4",
            options={"WBP", "WBP4", "WBP5", "WBP9", "PBHP"},
        ),
        Field(name="f2", type=np.float64, required=False, default=0.0),
        Field(
            name="depth_correction",
            type=lambda v: str(v).upper(),
            required=False,
            default="WELL",
            options={"WELL", "RES"},
        ),
        Field(
            name="open_connections_only",
            type=parse_bool,
            required=False,
            default=True,
        ),
    ],
)
"""
`WPAVE [f1] [procedure] [f2] [depth_correction] [open_connections_only] /`
- configure how average well pressure is calculated.

Average well pressure is used by several well-control algorithms and
reporting functions. This keyword selects the averaging procedure and
controls whether only open completions contribute to the calculation.

Fields:

- `f1`                    - procedure-specific weighting factor.
- `procedure`             - averaging method (`WBP`, `WBP4`, `WBP5`,
  `WBP9`, or `PBHP`).
- `f2`                    - additional procedure-specific parameter.
- `depth_correction`      - apply depth correction using either the well
  reference depth (`WELL`) or reservoir depth (`RES`).
- `open_connections_only` - whether only open completions contribute to the
  average pressure (`YES` or `NO`); defaults to `YES`.
"""

WCONHIST = ScheduledRecordKeyword[str | float](
    "WCONHIST",
    fields=[
        Field(name="well", type=str),
        Field(
            name="status",
            type=lambda v: str(v).upper(),
            required=False,
            default="OPEN",
            options={"OPEN", "SHUT"},
        ),
        Field(
            name="control_mode",
            type=lambda v: str(v).upper(),
            required=False,
            default="RESV",
            options={"ORAT", "WRAT", "GRAT", "RESV", "BHP"},
        ),
        Field(name="oil_rate", type=np.float64, required=False, default=0.0),
        Field(name="water_rate", type=np.float64, required=False, default=0.0),
        Field(name="gas_rate", type=np.float64, required=False, default=0.0),
        Field(name="vfp_table", type=int, required=False, default=None),
        Field(name="alq", type=np.float64, required=False, default=None),
        Field(name="thp", type=np.float64, required=False, default=None),
        Field(name="bhp", type=np.float64, required=False, default=None),
    ],
)
"""
`WCONHIST 'WELL' [status] [control_mode] [orat] [wrat] [grat] [vfp_table] [alq] [thp] [bhp] / ... /`
- specify historical production data for history matching.

Unlike `WCONPROD`, which defines simulation targets, `WCONHIST` supplies
observed production rates and operating conditions that the simulator
attempts to reproduce during history matching.

Fields:

- `well`         - well name.
- `status`       - well status (`OPEN` or `SHUT`).
- `control_mode` - historical control mode (`ORAT`, `WRAT`, `GRAT`,
  `RESV`, or `BHP`).
- `oil_rate`         - observed oil production rate.
- `water_rate`         - observed water production rate.
- `gas_rate`         - observed gas production rate.
- `vfp_table`    - VFP table used for THP/BHP calculations.
- `alq`          - artificial lift quantity.
- `thp`          - observed tubing-head pressure.
- `bhp`          - observed bottom-hole pressure.
"""

WCONINJH = ScheduledRecordKeyword[str | float](
    "WCONINJH",
    fields=[
        Field(name="well", type=str),
        Field(
            name="phase",
            type=lambda v: str(v).upper(),
            options={"OIL", "WAT", "GAS"},
        ),
        Field(
            name="status",
            type=lambda v: str(v).upper(),
            required=False,
            default="OPEN",
            options={"OPEN", "SHUT"},
        ),
        Field(name="rate", type=np.float64, required=False, default=0.0),
        Field(name="bhp", type=np.float64, required=False, default=None),
        Field(name="thp", type=np.float64, required=False, default=None),
        Field(name="vfp_table", type=int, required=False, default=None),
        Field(
            name="control_mode",
            type=lambda v: str(v).upper(),
            required=False,
            default="RATE",
            options={"RATE", "BHP"},
        ),
    ],
)
"""
`WCONINJH 'WELL' PHASE [status] [rate] [bhp] [thp] [vfp_table] [control_mode] / ... /`
- specify historical injection data for history matching.

Unlike `WCONINJE`, which defines simulator control targets, `WCONINJH`
describes measured injection performance that the simulator should honour
during history matching.

Fields:

- `well`         - well name.
- `phase`        - injected fluid (`OIL`, `WAT`, `GAS`, or `DISGAS`).
- `status`       - injector status (`OPEN` or `SHUT`).
- `rate`         - observed surface injection rate.
- `bhp`          - observed bottom-hole pressure.
- `thp`          - observed tubing-head pressure.
- `vfp_table`    - VFP table used for THP/BHP calculations.
- `control_mode` - historical injection control mode (`RATE` or `BHP`).
"""

WDFAC = ScheduledRecordKeyword[str | float](
    "WDFAC",
    fields=[
        Field(name="well", type=str),
        Field(name="d_factor", type=np.float64),
    ],
)
"""
`WDFAC 'WELL' D_FACTOR / ... /`
- assign a non-Darcy flow (D-factor) coefficient to a well.

The D-factor models additional pressure losses caused by high-velocity,
non-Darcy flow near the wellbore. It supplements the mechanical skin factor
and is primarily used for high-rate gas wells.

Fields:

- `well`     - well name.
- `d_factor` - non-Darcy flow coefficient assigned to the well.
"""


WLIFT = ScheduledRecordKeyword[str | float](
    "WLIFT",
    fields=[
        Field("well", str),
        Field("trigger_limit", np.float64, required=False, default=0.0),
        Field(
            "trigger_phase",
            lambda v: str(v).upper(),
            required=False,
            default="OIL",
            options={"OIL", "GAS", "WATER", "LIQ"},
        ),
        Field("new_vfp_table", int, required=False, default=0),
        Field("new_alq", np.float64, required=False, default=0.0),
        Field("new_efficiency_factor", np.float64, required=False, default=0.0),
        Field("water_cut_limit", np.float64, required=False, default=0.0),
        Field("new_thp_limit", np.float64, required=False, default=0.0),
        Field("gas_oil_ratio_limit", np.float64, required=False, default=0.0),
        Field("alq_shift", np.float64, required=False, default=1.0e20),
        Field("thp_shift", np.float64, required=False, default=1.0e20),
    ],
)
"""
`WLIFT 'WELL' [trigger_limit] [trigger_phase] [new_vfp_table] [new_alq] [new_efficiency_factor]
[water_cut_limit] [new_thp_limit] [gas_oil_ratio_limit] [alq_shift] [thp_shift] / ... /`
- re-tubing, THP and lift-switching workover: replace a well's VFP table and artificial lift
quantity once a trigger is met.

Item order and defaults follow OPM Flow's `WLIFT`.

Fields:

- `well`                  - well name or template.
- `trigger_limit`         - rate of `trigger_phase` below which the workover is triggered
  (`0` = no rate trigger).
- `trigger_phase`         - phase whose rate is tested against `trigger_limit`.
- `new_vfp_table`         - VFP table number to switch to (`0` = keep the current table).
- `new_alq`               - ALQ to apply (`0` = unchanged).
- `new_efficiency_factor` - well efficiency factor after the workover (`0` = unchanged).
- `water_cut_limit`       - water cut above which the workover is triggered (`0` = none).
- `new_thp_limit`         - THP limit applied after the workover (`0` = unchanged).
- `gas_oil_ratio_limit`   - gas-oil ratio above which the workover is triggered (`0` = none).
- `alq_shift`             - ALQ change applied if the well still violates after the workover.
- `thp_shift`             - THP change applied if the well still violates after the workover.

A deck's `WLIFT` is an update to a well's lift settings, so (like `WELTARG`) it is a
scheduled record: it takes effect at the schedule time it appears at.
"""


class VFPProducerDeckTable(typing.NamedTuple):
    """One `VFPPROD` table exactly as the deck declares it, before any unit or axis mapping."""

    table_number: int
    """Table number wells refer to through `WCONPROD`'s `vfp_table` item."""

    datum_depth: float | None
    """Depth the BHP values refer to, or `None` if defaulted."""

    flow_type: str
    """Meaning of the flow axis: `OIL`, `LIQ`, `GAS`, `WG` or `TM`."""

    water_fraction_type: str
    """Meaning of the water axis: `WOR`, `WCT` or `WGR`."""

    gas_fraction_type: str
    """Meaning of the gas axis: `GOR`, `GLR` or `OGR`."""

    thp_type: str
    """Meaning of the pressure axis (`THP`)."""

    alq_type: str
    """Meaning of the artificial lift axis: `GRAT`, `IGLR`, `TGLR`, `PUMP`, `COMP`, `BEAN` or blank."""

    units: str | None
    """Unit system the table is written in (`METRIC`, `FIELD`, `LAB`, `PVT-M`), or `None` if defaulted."""

    bhp_type: str
    """What the table values are (`BHP`)."""

    flow: npt.NDArray[np.float64]
    """Flow axis values."""

    thp: npt.NDArray[np.float64]
    """Tubing head pressure axis values."""

    water_fraction: npt.NDArray[np.float64]
    """Water axis values, in the units of `water_fraction_type`."""

    gas_fraction: npt.NDArray[np.float64]
    """Gas axis values, in the units of `gas_fraction_type`."""

    alq: npt.NDArray[np.float64]
    """Artificial lift axis values."""

    bhps: npt.NDArray[np.float64]
    """BHP values shaped `(len(flow), len(thp), len(water_fraction), len(gas_fraction), len(alq))`."""


class VFPInjectorDeckTable(typing.NamedTuple):
    """One `VFPINJ` table exactly as the deck declares it, before any unit mapping."""

    table_number: int
    """Table number wells refer to through `WCONINJE`'s `vfp_table` item."""

    datum_depth: float | None
    """Depth the BHP values refer to, or `None` if defaulted."""

    flow_type: str
    """Injected phase the flow axis refers to: `OIL`, `WAT` or `GAS`."""

    thp_type: str
    """Meaning of the pressure axis (`THP`)."""

    units: str | None
    """Unit system the table is written in, or `None` if defaulted."""

    bhp_type: str
    """What the table values are (`BHP`)."""

    flow: npt.NDArray[np.float64]
    """Flow axis values."""

    thp: npt.NDArray[np.float64]
    """Tubing head pressure axis values."""

    bhps: npt.NDArray[np.float64]
    """BHP values shaped `(len(flow), len(thp))`."""


class VFPKeyword(Keyword[list[VFPProducerDeckTable] | list[VFPInjectorDeckTable]]):
    """
    A `VFPPROD` or `VFPINJ` keyword: one vertical-flow-performance table per occurrence.

    A table has no terminating record. Its length follows from its axes: a header record,
    one record per axis, then one record per combination of the non-flow axes holding the BHP
    values along the flow axis. Each such record starts with the 1-based positions on those axes.
    """

    __slots__ = ("injector",)

    def __init__(self, name: str, *, injector: bool) -> None:
        """
        :param name: `"VFPPROD"` or `"VFPINJ"`.
        :param injector: `True` for `VFPINJ` (flow and pressure axes only).
        """
        super().__init__(name)
        self.injector = injector

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> list[VFPProducerDeckTable] | list[VFPInjectorDeckTable] | None:
        records = deck.get_records_for(self.name)
        if not records:
            return None
        tables = [self.parse_table(record.body) for record in records]
        return tables  # type: ignore[return-value]

    def parse_table(self, body: str) -> VFPProducerDeckTable | VFPInjectorDeckTable:
        """
        Parse one table's body.

        :param body: The text between this occurrence's keyword line and the next keyword.
        :returns: The table as declared.
        :raises DeckParseError: If the axes and rows are inconsistent.
        """
        segments: list[list[str]] = [[]]
        for token in tokenize(body.replace("/", " / ")):
            if token == "/":
                segments.append([])
            else:
                segments[-1].append(token)

        segments = [segment for segment in segments if segment]
        n_axes = 2 if self.injector else 5
        if len(segments) < 1 + n_axes + 1:
            raise DeckParseError(f"{self.name}: a table needs a header, its axes and data rows.")

        def item(segment: list[str], index: int, default: str | None = None) -> str | None:
            if index >= len(segment) or segment[index] == "1*":
                return default
            return segment[index].upper()

        def numbers(segment: list[str], label: str) -> npt.NDArray[np.float64]:
            try:
                return np.array(segment, dtype=np.float64)
            except ValueError as exc:
                raise DeckParseError(f"{self.name}: bad {label} axis values {segment}.") from exc

        header = segments[0]
        table_number = int(float(header[0]))
        datum_text = item(header, 1)
        datum_depth = float(datum_text) if datum_text is not None else None
        axes = [numbers(segments[1 + i], f"axis {i + 1}") for i in range(n_axes)]
        rows = segments[1 + n_axes :]

        try:
            if self.injector:
                flow, thp = axes
                bhps = np.full((len(flow), len(thp)), np.nan)
                for row in rows:
                    thp_index = int(float(row[0])) - 1
                    values = np.array(row[1:], dtype=np.float64)
                    if values.size != len(flow) or not 0 <= thp_index < len(thp):
                        raise DeckParseError(
                            f"{self.name} table {table_number}: bad data row {row}."
                        )
                    bhps[:, thp_index] = values

                if np.isnan(bhps).any():
                    raise DeckParseError(
                        f"{self.name} table {table_number}: the data rows do not cover every THP."
                    )
                return VFPInjectorDeckTable(
                    table_number=table_number,
                    datum_depth=datum_depth,
                    flow_type=item(header, 2, "") or "",
                    thp_type=item(header, 3, "THP") or "THP",
                    units=item(header, 4),
                    bhp_type=item(header, 5, "BHP") or "BHP",
                    flow=flow,
                    thp=thp,
                    bhps=bhps,
                )

            flow, thp, water_fraction, gas_fraction, alq = axes
            shape = (len(flow), len(thp), len(water_fraction), len(gas_fraction), len(alq))
            bhps = np.full(shape, np.nan)
            for row in rows:
                indices = [int(float(value)) - 1 for value in row[:4]]
                values = np.array(row[4:], dtype=np.float64)
                limits = shape[1:]
                if values.size != len(flow) or any(
                    not 0 <= index < limit for index, limit in zip(indices, limits, strict=True)
                ):
                    raise DeckParseError(f"{self.name} table {table_number}: bad data row {row}.")

                bhps[:, indices[0], indices[1], indices[2], indices[3]] = values

            if np.isnan(bhps).any():
                raise DeckParseError(
                    f"{self.name} table {table_number}: the data rows do not cover every "
                    "combination of the THP, water, gas and ALQ axes."
                )
            return VFPProducerDeckTable(
                table_number=table_number,
                datum_depth=datum_depth,
                flow_type=item(header, 2, "") or "",
                water_fraction_type=item(header, 3, "") or "",
                gas_fraction_type=item(header, 4, "") or "",
                thp_type=item(header, 5, "THP") or "THP",
                alq_type=item(header, 6, "") or "",
                units=item(header, 7),
                bhp_type=item(header, 8, "BHP") or "BHP",
                flow=flow,
                thp=thp,
                water_fraction=water_fraction,
                gas_fraction=gas_fraction,
                alq=alq,
                bhps=bhps,
            )
        except ValueError as exc:
            raise DeckParseError(f"{self.name} table {table_number}: {exc}") from exc


VFPPROD = VFPKeyword("VFPPROD", injector=False)
"""
`VFPPROD` - a producer vertical flow performance table: BHP as a function of flow rate, THP,
water fraction, gas fraction and artificial lift quantity (ALQ).

`parse` returns a list of `VFPProducerDeckTable`, one per occurrence, or `None` if absent. The table
only defines how BHP varies with ALQ; the ALQ a well actually runs at comes from `WCONPROD`
(or `WLIFT`).

Fields:

- `table_number`         - table ID referenced by `WCONPROD` / `WLIFT` via `vfp_table`.
- `datum_depth`          - reference depth for the reported BHP values, or `None` if defaulted.
- `flow_type`            - meaning of the flow axis (`OIL`, `LIQ`, `GAS`, `WG`, `TM`, etc.).
- `water_fraction_type`   - meaning of the water-fraction axis (`WOR`, `WCT`, `WGR`, etc.).
- `gas_fraction_type`    - meaning of the gas-fraction axis (`GOR`, `GLR`, `OGR`, etc.).
- `thp_type`             - meaning of the THP axis (`THP`).
- `alq_type`             - meaning of the ALQ axis (`GRAT`, `IGLR`, `TGLR`, `PUMP`, `COMP`, `BEAN`, or blank).
- `units`                - unit system declared for the table (`FIELD`, `METRIC`, `LAB`, `PVT-M`, or `None`).
- `bhp_type`             - what the table values represent (`BHP`).
- `flow`                 - flow-rate axis values.
- `thp`                  - tubing-head pressure axis values.
- `water_fraction`       - water-fraction axis values.
- `gas_fraction`         - gas-fraction axis values.
- `alq`                  - artificial-lift axis values.
- `bhps`                 - BHP table values shaped by the flow / THP / water / gas / ALQ axes.
"""

VFPINJ = VFPKeyword("VFPINJ", injector=True)
"""
`VFPINJ` - an injector vertical flow performance table: BHP as a function of flow rate and THP.

`parse` returns a list of `VFPInjectorDeckTable`, one per occurrence, or `None` if absent.

Fields:

- `table_number`     - table ID referenced by `WCONINJE` via `vfp_table`.
- `datum_depth`      - reference depth for the reported BHP values, or `None` if defaulted.
- `flow_type`        - meaning of the flow axis (`OIL`, `WAT`, `GAS`, etc.).
- `thp_type`         - meaning of the THP axis (`THP`).
- `units`            - unit system declared for the table (`FIELD`, `METRIC`, `LAB`, `PVT-M`, or `None`).
- `bhp_type`         - what the table values represent (`BHP`).
- `flow`             - flow-rate axis values.
- `thp`              - tubing-head pressure axis values.
- `bhps`             - BHP table values shaped by the flow and THP axes.
"""

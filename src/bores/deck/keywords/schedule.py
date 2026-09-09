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

import numpy as np

from bores.datastructures import GridDimensions
from bores.deck.core import Deck, DeckParseError, tokenize
from bores.deck.keywords.base import (
    DatesKeyword,
    Field,
    Keyword,
    RecordKeyword,
    ScheduledRecordKeyword,
)
from bores.deck.operators import Operation
from bores.errors import ValidationError

__all__ = [
    "COMPDAT",
    "DATES",
    "GCONINJE",
    "GCONPROD",
    "GECON",
    "GRUPTREE",
    "TSTEP",
    "WCONHIST",
    "WCONINJE",
    "WCONINJH",
    "WCONPROD",
    "WDFAC",
    "WECON",
    "WELOPEN",
    "WELPI",
    "WELSPECS",
    "WELTARG",
    "WGRUPCON",
    "WPAVE",
    "WPIMULT",
    "WTEST",
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

"""
Killough (1976) relative permeability scanning curves, as Eclipse defines them.

These differ from the blended scheme of `KilloughLandRelPermTable`, which weighs the drainage and
imbibition curves of every phase against each other. Here the two kinds of phase are treated
differently, and nothing is blended with the drainage curve at the same saturation.

**Non-wetting phase.** Imbibition from a reversal at the saturation `s_hy` traps `s_t` of the phase,
by Land's relation measured from the drainage critical saturation `s_crd`:

    s_t = s_crd + (s_hy - s_crd) / (1 + C * (s_hy - s_crd)),
    C = 1 / (s_cri - s_crd) - 1 / (s_max - s_crd)

where `s_cri` is the critical saturation of the imbibition curve and `s_max` the largest saturation.
The scanning curve is the imbibition curve itself, with its saturation axis stretched so that
`[s_t, s_hy]` covers `[s_cri, s_max]`, and with its values scaled so that it starts at the drainage
value at the reversal:

    s* = s_cri + (s - s_t) * (s_max - s_cri) / (s_hy - s_t)
    kr(s) = kr_drn(s_hy) * (kr_imb(s*) - kr_imb(s_cri)) / (kr_imb(s_max) - kr_imb(s_cri))

The curve is retraced when the saturation rises again, and the drainage curve is rejoined at `s_hy`.

**Wetting phase.** It scans from its drainage value at the reversal to a value at the trapped
saturation, `kr_w(s_t) = kr_drn(s_t) + [kr_imb(s_cri) - kr_drn(s_cri)] * (s_t / s_cri)^a`, with the
position along the way following the imbibition curve's own progress over the same stretched axis:

    kr_w(s) = kr_drn(s_hy) + (kr_imb(s*) - kr_imb(s_max)) / (kr_imb(s_cri) - kr_imb(s_max))
              * (kr_w(s_t) - kr_drn(s_hy))

- Killough, J.E. (1976). "Reservoir Simulation With History-Dependent Saturation Functions".
  SPE 5106.
- Land, C.S. (1968). "Calculation of Imbibition Relative Permeability for Two- and Three-Phase
  Flow from Rock Properties". SPE 1942.
"""

import typing

import attrs
import numba
import numpy as np

from bores.blackoil.satfunc.relperm.hysteresis.base import (
    BoundingCurves,
    HysteresisRelPermTable,
    query_gas_oil_curves,
    query_oil_water_curves,
)
from bores.blackoil.satfunc.relperm.tables import (
    RelativePermeabilityTable,
    resolve_residual_oil_saturation_gas,
)
from bores.errors import ValidationError
from bores.types import (
    Boolean,
    BooleanArray,
    FluidPhase,
    Integer,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
    RelativePermeabilities,
    RelativePermeabilityDerivatives,
)

__all__ = [
    "EclipseKilloughRelPermTable",
    "KilloughSystem",
    "killough_axis_kernel",
    "killough_scan_kernel",
]

SATURATION_EPSILON = 1e-12
"""Smallest saturation span that is scanned."""


class KilloughSystem(typing.NamedTuple):
    """
    What Killough's scheme needs to know about one two-phase system, read off its two tables.

    Saturations are those of the system's non-wetting phase: gas in the gas-oil system, and oil in
    the oil-water system, where the saturation is `1 - Sw`.
    """

    critical_drainage: Number
    """Critical saturation of the drainage curve, `s_crd`."""

    critical_imbibition: Number
    """Critical saturation of the imbibition curve, `s_cri`. The most that can be trapped."""

    maximum: Number
    """Largest saturation of the non-wetting phase, `s_max`."""

    land_coefficient: Number | None
    """Land's coefficient `C`, or `None` when nothing is trapped beyond the critical saturation."""

    imbibition_non_wetting_at_critical: Number
    imbibition_non_wetting_at_maximum: Number
    imbibition_wetting_at_critical: Number
    imbibition_wetting_at_maximum: Number
    drainage_wetting_at_critical: Number


@numba.njit(cache=True, error_model="numpy")
def killough_axis_kernel(
    size: Integer,
    saturation: NumberArray[OneDimension],
    peak: NumberArray[OneDimension],
    reflect: Boolean,
    critical_drainage: Number,
    critical_imbibition: Number,
    maximum: Number,
    land_coefficient: Number,
    has_trapping: Boolean,
    out_trapped: NumberArray[OneDimension],
    out_mapped: NumberArray[OneDimension],
    out_scale: NumberArray[OneDimension],
    out_scanning: NumberArray[OneDimension],
) -> None:
    """
    Trapped saturation, and the stretched saturation to read the imbibition curve at, for every
    cell, in the saturation of the non-wetting phase.

    With `reflect` set the non-wetting phase is the complement of the saturation given, which is how
    oil in a water-wet system is read off the water saturation and its minimum.

    :param size: Number of cells.
    :param saturation: Each cell's saturation, or the water saturation when `reflect` is set.
    :param peak: Each cell's reversal saturation, or the minimum water saturation when `reflect`
        is set. May be one shared value.
    :param reflect: Whether the non-wetting phase is the complement of the saturation given.
    :param critical_drainage: `s_crd`.
    :param critical_imbibition: `s_cri`.
    :param maximum: `s_max`.
    :param land_coefficient: Land's `C`, read only when `has_trapping` is set.
    :param has_trapping: Whether anything is trapped beyond `s_crd`.
    :param out_trapped: Receives the trapped saturation.
    :param out_mapped: Receives the saturation to read the imbibition curve at, which is never
        below `critical_imbibition`.
    :param out_scale: Receives the slope of the stretch.
    :param out_scanning: Receives 1 for a cell that is on a scanning curve and 0 for one on the
        drainage curve.
    """
    imbibition_span = maximum - critical_imbibition
    for i in range(size):
        current = saturation[i] if saturation.size > 1 else saturation[0]
        reversal = peak[i] if peak.size > 1 else peak[0]
        if reflect:
            current = 1.0 - current
            reversal = 1.0 - reversal

        excess = reversal - critical_drainage
        if excess > SATURATION_EPSILON:
            if has_trapping:
                trapped = critical_drainage + excess / (1.0 + land_coefficient * excess)
            else:
                trapped = critical_drainage
        else:
            trapped = reversal

        span = reversal - trapped
        out_trapped[i] = trapped
        if (
            span > SATURATION_EPSILON
            and imbibition_span > SATURATION_EPSILON
            and current < reversal
        ):
            scale = imbibition_span / span
            # Below the trapped saturation the phase is immobile and the stretched saturation
            # would fall below the imbibition table's own critical saturation, so it stops there.
            out_mapped[i] = max(
                critical_imbibition + (current - trapped) * scale, critical_imbibition
            )
            out_scale[i] = scale
            out_scanning[i] = 1.0
        else:
            out_mapped[i] = current
            out_scale[i] = 1.0
            out_scanning[i] = 0.0


@numba.njit(cache=True, error_model="numpy")
def killough_scan_kernel(
    size: Integer,
    scanning: NumberArray[OneDimension],
    scale: NumberArray[OneDimension],
    trapped: NumberArray[OneDimension],
    drainage_non_wetting: NumberArray[OneDimension],
    drainage_wetting: NumberArray[OneDimension],
    drainage_non_wetting_slope: NumberArray[OneDimension],
    drainage_wetting_slope: NumberArray[OneDimension],
    peak_drainage_non_wetting: NumberArray[OneDimension],
    peak_drainage_wetting: NumberArray[OneDimension],
    trapped_drainage_wetting: NumberArray[OneDimension],
    imbibition_non_wetting: NumberArray[OneDimension],
    imbibition_wetting: NumberArray[OneDimension],
    imbibition_non_wetting_slope: NumberArray[OneDimension],
    imbibition_wetting_slope: NumberArray[OneDimension],
    critical_imbibition: Number,
    imbibition_non_wetting_at_critical: Number,
    imbibition_non_wetting_at_maximum: Number,
    imbibition_wetting_at_critical: Number,
    imbibition_wetting_at_maximum: Number,
    drainage_wetting_at_critical: Number,
    wetting_curvature: Number,
    scan_wetting: Boolean,
    with_derivatives: Boolean,
    out_non_wetting: NumberArray[OneDimension],
    out_wetting: NumberArray[OneDimension],
    out_non_wetting_slope: NumberArray[OneDimension],
    out_wetting_slope: NumberArray[OneDimension],
) -> None:
    """
    Killough's scanning curves of the non-wetting and the wetting phase, for every cell, in the
    saturation of the non-wetting phase and in one pass.

    A cell that is not on a scanning curve keeps its drainage values and slopes. Every slope is with
    respect to the non-wetting phase's saturation. See the module notes for the equations.

    :param size: Number of cells.
    :param scanning: 1 for a cell on a scanning curve, from `killough_axis_kernel`.
    :param scale: Slope of the stretched axis, from `killough_axis_kernel`.
    :param trapped: Trapped saturation, from `killough_axis_kernel`.
    :param drainage_non_wetting: Drainage relative permeability of the non-wetting phase.
    :param drainage_wetting: Drainage relative permeability of the wetting phase.
    :param drainage_non_wetting_slope: Slope of `drainage_non_wetting`.
    :param drainage_wetting_slope: Slope of `drainage_wetting`.
    :param peak_drainage_non_wetting: Drainage non-wetting value at the reversal saturation.
    :param peak_drainage_wetting: Drainage wetting value at the reversal saturation.
    :param trapped_drainage_wetting: Drainage wetting value at the trapped saturation.
    :param imbibition_non_wetting: Imbibition non-wetting value at the stretched saturation.
    :param imbibition_wetting: Imbibition wetting value at the stretched saturation.
    :param imbibition_non_wetting_slope: Slope of `imbibition_non_wetting`.
    :param imbibition_wetting_slope: Slope of `imbibition_wetting`.
    :param critical_imbibition: `s_cri`.
    :param imbibition_non_wetting_at_critical: Imbibition non-wetting value at `s_cri`.
    :param imbibition_non_wetting_at_maximum: Imbibition non-wetting value at `s_max`.
    :param imbibition_wetting_at_critical: Imbibition wetting value at `s_cri`.
    :param imbibition_wetting_at_maximum: Imbibition wetting value at `s_max`.
    :param drainage_wetting_at_critical: Drainage wetting value at `s_cri`.
    :param wetting_curvature: The exponent `a` of the wetting phase's trapped value.
    :param scan_wetting: Whether the wetting phase scans or stays on its drainage curve.
    :param with_derivatives: Whether the slopes are present and wanted.
    :param out_non_wetting: Receives the non-wetting relative permeability.
    :param out_wetting: Receives the wetting relative permeability.
    :param out_non_wetting_slope: Receives its slope, when wanted.
    :param out_wetting_slope: Receives its slope, when wanted.
    """
    non_wetting_range = imbibition_non_wetting_at_maximum - imbibition_non_wetting_at_critical
    wetting_range = imbibition_wetting_at_critical - imbibition_wetting_at_maximum
    for i in range(size):
        out_non_wetting[i] = drainage_non_wetting[i]
        out_wetting[i] = drainage_wetting[i]
        if with_derivatives:
            out_non_wetting_slope[i] = drainage_non_wetting_slope[i]
            out_wetting_slope[i] = drainage_wetting_slope[i]
        if scanning[i] < 0.5:
            continue
        stretch = scale[i]

        if abs(non_wetting_range) > SATURATION_EPSILON:
            ratio = (imbibition_non_wetting[i] - imbibition_non_wetting_at_critical) / (
                non_wetting_range
            )
            if ratio > 0.0:
                out_non_wetting[i] = peak_drainage_non_wetting[i] * ratio
                if with_derivatives:
                    out_non_wetting_slope[i] = (
                        peak_drainage_non_wetting[i]
                        * imbibition_non_wetting_slope[i]
                        * stretch
                        / non_wetting_range
                    )
            else:
                out_non_wetting[i] = 0.0
                if with_derivatives:
                    out_non_wetting_slope[i] = 0.0

        if scan_wetting and abs(wetting_range) > SATURATION_EPSILON:
            if critical_imbibition > SATURATION_EPSILON:
                factor = (trapped[i] / critical_imbibition) ** wetting_curvature
            else:
                factor = 1.0

            trapped_value = (
                trapped_drainage_wetting[i]
                + (imbibition_wetting_at_critical - drainage_wetting_at_critical) * factor
            )
            travel = trapped_value - peak_drainage_wetting[i]
            progress = (imbibition_wetting[i] - imbibition_wetting_at_maximum) / wetting_range
            if progress < 0.0:
                progress = 0.0
                inside = False
            elif progress > 1.0:
                progress = 1.0
                inside = False
            else:
                inside = True

            out_wetting[i] = peak_drainage_wetting[i] + progress * travel
            if with_derivatives:
                out_wetting_slope[i] = (
                    imbibition_wetting_slope[i] * stretch / wetting_range * travel
                    if inside
                    else 0.0
                )


def _land_coefficient(
    *, critical_drainage: Number, critical_imbibition: Number, maximum: Number
) -> Number | None:
    """Land's coefficient measured from the drainage critical saturation, or `None` for no trapping."""
    trapped_range = critical_imbibition - critical_drainage
    largest_range = maximum - critical_drainage
    if trapped_range <= SATURATION_EPSILON or largest_range <= SATURATION_EPSILON:
        return None
    return max(1.0 / trapped_range - 1.0 / largest_range, 0.0)


@attrs.frozen(slots=True)
class EclipseKilloughRelPermTable(HysteresisRelPermTable):
    """
    Killough relative permeability hysteresis as Eclipse defines it (see the module notes).

    The non-wetting phase of each system scans: gas in the gas-oil system, and oil in the oil-water
    system when water is the wetting phase. Gas peaks at its historical maximum saturation and oil
    at one minus the historical minimum water saturation, so the saturation history must carry
    both. The wetting phase follows its drainage curve unless `wetting_phase_hysteresis` is set. The
    three-phase mixing rule then combines the curves exactly as it does for the other models.

    Land trapping is part of the model, and is derived from the tables: the imbibition table's
    critical saturation is what is trapped when imbibition starts from the largest saturation of the
    drainage table. The maximum residual and Land coefficient fields of the base class are not used.

    The imbibition table has to share its endpoints with the drainage table's range, so that the
    stretched curve is read where the imbibition table is defined. A system without an imbibition
    table, without history, or with nothing between its critical and its largest saturation stays
    on its drainage curves.

    Not implemented: the Eclipse modification of the trapped saturation (`EHYSTR` item 4).
    """

    wetting_phase_hysteresis: bool = False
    """Whether the wetting phase scans too, as `EHYSTR` code 3 asks, rather than staying on drainage."""

    wetting_phase_curvature: Number = 1.0
    """The exponent `a` of the wetting phase's value at the trapped saturation (`EHYSTR` item 3)."""

    gas_system: KilloughSystem | None = attrs.field(init=False, repr=False, eq=False)
    """The gas-oil system's endpoints, or `None` when it does not scan."""

    oil_system: KilloughSystem | None = attrs.field(init=False, repr=False, eq=False)
    """The oil-water system's endpoints, or `None` when it does not scan."""

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.wetting_phase_curvature <= 0.0:
            raise ValidationError("`wetting_phase_curvature` must be positive.")
        if (
            self.oil_water_imbibition_table is not None
            and self.get_oil_water_wetting_phase() != FluidPhase.WATER
        ):
            raise ValidationError(
                "Killough scanning of water as the non-wetting phase of an oil-wet oil-water "
                "system is not supported. Give no `oil_water_imbibition_table`."
            )
        if (
            self.gas_oil_imbibition_table is not None
            and self.get_gas_oil_wetting_phase() == FluidPhase.GAS
        ):
            raise ValidationError(
                "Killough scanning of oil as the non-wetting phase of a gas-wet gas-oil system is "
                "not supported. Give no `gas_oil_imbibition_table`."
            )
        object.__setattr__(self, "gas_system", self._tabulate_gas_system())
        object.__setattr__(self, "oil_system", self._tabulate_oil_system())

    def _gas_curves(
        self,
        table: RelativePermeabilityTable,
        gas: typing.Any,
        *,
        water: typing.Any,
        with_derivatives: bool,
    ) -> tuple[typing.Any, typing.Any, typing.Any, typing.Any]:
        """`(wetting, non_wetting, wetting_slope, non_wetting_slope)` of the gas-oil system at `gas`."""
        connate_water = self.get_connate_water_saturation()
        return query_gas_oil_curves(
            table,
            water_saturation=water,
            oil_saturation=1.0 - water - gas,
            gas_saturation=gas,
            connate_water_saturation=connate_water,
            with_derivatives=with_derivatives,
        )

    def _oil_curves(
        self,
        table: RelativePermeabilityTable,
        oil: typing.Any,
        *,
        gas: typing.Any,
        with_derivatives: bool,
    ) -> tuple[typing.Any, typing.Any, typing.Any, typing.Any]:
        """
        `(wetting, non_wetting, wetting_slope, non_wetting_slope)` of the oil-water system at the
        oil saturation `oil`, which is `1 - Sw`. Slopes are with respect to `oil`.
        """
        water = 1.0 - oil
        krw, kro, dkrw, dkro = query_oil_water_curves(
            table,
            water_saturation=water,
            oil_saturation=oil - gas,
            gas_saturation=gas,
            with_derivatives=with_derivatives,
        )
        if with_derivatives:
            return krw, kro, -dkrw, -dkro
        return krw, kro, dkrw, dkro

    def _tabulate_gas_system(self) -> KilloughSystem | None:
        imbibition = self.gas_oil_imbibition_table
        if imbibition is None:
            return None
        drainage = self.gas_oil_drainage_table
        connate_water = self.get_connate_water_saturation()
        critical_drainage = float(drainage.get_residual_gas_saturation())
        critical_imbibition = float(imbibition.get_residual_gas_saturation())
        maximum = float(
            1.0 - connate_water - resolve_residual_oil_saturation_gas(drainage, connate_water)
        )
        if maximum - critical_imbibition <= SATURATION_EPSILON:
            return None

        def at(table: RelativePermeabilityTable, gas: Number) -> tuple[float, float]:
            wetting, non_wetting, _, _ = self._gas_curves(
                table, np.array([gas]), water=np.array([connate_water]), with_derivatives=False
            )
            return float(np.asarray(wetting)[0]), float(np.asarray(non_wetting)[0])

        wetting_at_critical, non_wetting_at_critical = at(imbibition, critical_imbibition)
        wetting_at_maximum, non_wetting_at_maximum = at(imbibition, maximum)
        return KilloughSystem(
            critical_drainage=critical_drainage,
            critical_imbibition=critical_imbibition,
            maximum=maximum,
            land_coefficient=_land_coefficient(
                critical_drainage=critical_drainage,
                critical_imbibition=critical_imbibition,
                maximum=maximum,
            ),
            imbibition_non_wetting_at_critical=non_wetting_at_critical,
            imbibition_non_wetting_at_maximum=non_wetting_at_maximum,
            imbibition_wetting_at_critical=wetting_at_critical,
            imbibition_wetting_at_maximum=wetting_at_maximum,
            drainage_wetting_at_critical=at(drainage, critical_imbibition)[0],
        )

    def _tabulate_oil_system(self) -> KilloughSystem | None:
        imbibition = self.oil_water_imbibition_table
        if imbibition is None or self.get_oil_water_wetting_phase() != FluidPhase.WATER:
            return None
        drainage = self.oil_water_drainage_table
        critical_drainage = float(drainage.get_residual_oil_saturation_water())
        critical_imbibition = float(imbibition.get_residual_oil_saturation_water())
        maximum = float(1.0 - self.get_connate_water_saturation())
        if maximum - critical_imbibition <= SATURATION_EPSILON:
            return None

        def at(table: RelativePermeabilityTable, oil: Number) -> tuple[float, float]:
            wetting, non_wetting, _, _ = self._oil_curves(
                table, np.array([oil]), gas=np.zeros(1), with_derivatives=False
            )
            return float(np.asarray(wetting)[0]), float(np.asarray(non_wetting)[0])

        wetting_at_critical, non_wetting_at_critical = at(imbibition, critical_imbibition)
        wetting_at_maximum, non_wetting_at_maximum = at(imbibition, maximum)
        return KilloughSystem(
            critical_drainage=critical_drainage,
            critical_imbibition=critical_imbibition,
            maximum=maximum,
            land_coefficient=_land_coefficient(
                critical_drainage=critical_drainage,
                critical_imbibition=critical_imbibition,
                maximum=maximum,
            ),
            imbibition_non_wetting_at_critical=non_wetting_at_critical,
            imbibition_non_wetting_at_maximum=non_wetting_at_maximum,
            imbibition_wetting_at_critical=wetting_at_critical,
            imbibition_wetting_at_maximum=wetting_at_maximum,
            drainage_wetting_at_critical=at(drainage, critical_imbibition)[0],
        )

    def _scan_system(
        self,
        *,
        system: KilloughSystem,
        size: Integer,
        saturation: NumberArray[OneDimension],
        peak: NumberArray[OneDimension],
        reflect: Boolean,
        drainage: tuple[typing.Any, typing.Any, typing.Any, typing.Any],
        curves_at: typing.Callable[..., tuple[typing.Any, typing.Any, typing.Any, typing.Any]],
        drainage_table: RelativePermeabilityTable,
        imbibition_table: RelativePermeabilityTable,
        with_derivatives: Boolean,
    ) -> tuple[typing.Any, typing.Any, typing.Any, typing.Any]:
        """
        Killough's scanning curves of one system, in the saturation of its non-wetting phase.

        :param drainage: The system's drainage curves at the cells' own saturations, as
            `(wetting, non_wetting, wetting_slope, non_wetting_slope)` with slopes with respect to
            the non-wetting saturation.
        :param curves_at: Reads a table's curves at an array of non-wetting saturations, returning
            the same four, with slopes with respect to that saturation.
        :returns: The scanned `(wetting, non_wetting, wetting_slope, non_wetting_slope)`.
        """
        dtype = np.asarray(drainage[0]).dtype
        trapped = np.empty(size, dtype=dtype)
        mapped = np.empty(size, dtype=dtype)
        scale = np.empty(size, dtype=dtype)
        scanning = np.empty(size, dtype=dtype)
        killough_axis_kernel(
            size=size,
            saturation=saturation,
            peak=peak,
            reflect=reflect,
            critical_drainage=system.critical_drainage,
            critical_imbibition=system.critical_imbibition,
            maximum=system.maximum,
            land_coefficient=system.land_coefficient or 0.0,
            has_trapping=system.land_coefficient is not None,
            out_trapped=trapped,
            out_mapped=mapped,
            out_scale=scale,
            out_scanning=scanning,
        )
        peak_saturation = 1.0 - peak if reflect else peak
        peak_wetting, peak_non_wetting, _, _ = curves_at(
            drainage_table, peak_saturation, with_derivatives=False
        )
        trapped_wetting, _, _, _ = curves_at(drainage_table, trapped, with_derivatives=False)
        wetting_i, non_wetting_i, wetting_slope_i, non_wetting_slope_i = curves_at(
            imbibition_table, mapped, with_derivatives=with_derivatives
        )

        def flat(array: typing.Any) -> typing.Any:
            return np.ascontiguousarray(np.broadcast_to(np.asarray(array), (size,)))

        placeholder = np.zeros(1, dtype=dtype)
        wetting_d, non_wetting_d, wetting_slope_d, non_wetting_slope_d = drainage
        out_non_wetting = np.empty(size, dtype=dtype)
        out_wetting = np.empty(size, dtype=dtype)
        out_non_wetting_slope = np.empty(size if with_derivatives else 1, dtype=dtype)
        out_wetting_slope = np.empty(size if with_derivatives else 1, dtype=dtype)
        killough_scan_kernel(
            size=size,
            scanning=scanning,
            scale=scale,
            trapped=trapped,
            drainage_non_wetting=flat(non_wetting_d),
            drainage_wetting=flat(wetting_d),
            drainage_non_wetting_slope=flat(non_wetting_slope_d)
            if with_derivatives
            else placeholder,
            drainage_wetting_slope=flat(wetting_slope_d) if with_derivatives else placeholder,
            peak_drainage_non_wetting=flat(peak_non_wetting),
            peak_drainage_wetting=flat(peak_wetting),
            trapped_drainage_wetting=flat(trapped_wetting),
            imbibition_non_wetting=flat(non_wetting_i),
            imbibition_wetting=flat(wetting_i),
            imbibition_non_wetting_slope=flat(non_wetting_slope_i)
            if with_derivatives
            else placeholder,
            imbibition_wetting_slope=flat(wetting_slope_i) if with_derivatives else placeholder,
            critical_imbibition=system.critical_imbibition,
            imbibition_non_wetting_at_critical=system.imbibition_non_wetting_at_critical,
            imbibition_non_wetting_at_maximum=system.imbibition_non_wetting_at_maximum,
            imbibition_wetting_at_critical=system.imbibition_wetting_at_critical,
            imbibition_wetting_at_maximum=system.imbibition_wetting_at_maximum,
            drainage_wetting_at_critical=system.drainage_wetting_at_critical,
            wetting_curvature=self.wetting_phase_curvature,
            scan_wetting=self.wetting_phase_hysteresis,
            with_derivatives=with_derivatives,
            out_non_wetting=out_non_wetting,
            out_wetting=out_wetting,
            out_non_wetting_slope=out_non_wetting_slope,
            out_wetting_slope=out_wetting_slope,
        )
        return (
            out_wetting,
            out_non_wetting,
            out_wetting_slope if with_derivatives else None,
            out_non_wetting_slope if with_derivatives else None,
        )

    def compute(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        *,
        with_derivatives: Boolean,
        max_water_saturation: NumberOrArray[NDimension] | None = None,
        max_gas_saturation: NumberOrArray[NDimension] | None = None,
        water_imbibition_flag: bool | BooleanArray[NDimension] | None = None,
        gas_imbibition_flag: bool | BooleanArray[NDimension] | None = None,
        water_reversal_saturation: NumberOrArray[NDimension] | None = None,
        gas_reversal_saturation: NumberOrArray[NDimension] | None = None,
        min_water_saturation: NumberOrArray[NDimension] | None = None,
        min_gas_saturation: NumberOrArray[NDimension] | None = None,
        **kwargs: typing.Any,
    ) -> tuple[RelativePermeabilities, RelativePermeabilityDerivatives | None]:
        """
        Relative permeabilities, and optionally their nine derivatives, in one pass.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param with_derivatives: Whether to also compute the derivatives.
        :param max_water_saturation: Historical maximum water saturation per cell. Switches oil
            hysteresis on together with `water_imbibition_flag` and `min_water_saturation`.
        :param max_gas_saturation: Historical maximum gas saturation per cell. Switches gas
            hysteresis on together with `gas_imbibition_flag`.
        :param water_imbibition_flag: Per-cell water imbibition flag.
        :param gas_imbibition_flag: Per-cell gas imbibition flag.
        :param water_reversal_saturation: Water saturation at the last reversal. Not used: the
            oil peak is read off `min_water_saturation`.
        :param gas_reversal_saturation: Gas saturation at the last reversal, the peak the
            scanning curve starts from. Defaults to `max_gas_saturation`.
        :param min_water_saturation: Historical minimum water saturation per cell, where oil, the
            non-wetting phase of a water-wet system, peaked.
        :param min_gas_saturation: Historical minimum gas saturation per cell. Not used.
        :param kwargs: Extra arguments forwarded to the backing tables.
        :return: The relative permeabilities and, when requested, their derivatives
            (otherwise `None`).
        """
        inputs, history = self.prepare_scan_inputs(
            water_saturation,
            oil_saturation,
            gas_saturation,
            max_water_saturation=max_water_saturation,
            max_gas_saturation=max_gas_saturation,
            water_imbibition_flag=water_imbibition_flag,
            gas_imbibition_flag=gas_imbibition_flag,
            water_reversal_saturation=water_reversal_saturation,
            gas_reversal_saturation=gas_reversal_saturation,
            min_water_saturation=min_water_saturation,
        )
        size = inputs.size
        sw, so, sg = inputs.sw, inputs.so, inputs.sg
        use_gas = (
            inputs.use_gas_hysteresis and self.gas_system is not None and inputs.gas_is_non_wetting
        )
        use_oil = (
            inputs.use_water_hysteresis
            and self.oil_system is not None
            and min_water_saturation is not None
        )
        inputs = inputs._replace(use_water_hysteresis=False, use_gas_hysteresis=False)

        curve_arguments = dict(
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            with_derivatives=with_derivatives,
        )
        connate_water = self.get_connate_water_saturation()
        gas_oil_arguments = {**curve_arguments, "connate_water_saturation": connate_water}
        krw, kro_water, dkrw, dkro_water = query_oil_water_curves(
            self.oil_water_drainage_table, **curve_arguments, **kwargs
        )
        kro_gas, krg, dkro_gas, dkrg = query_gas_oil_curves(
            self.gas_oil_drainage_table, **gas_oil_arguments, **kwargs
        )

        if use_oil:
            drainage = (
                krw,
                kro_water,
                None if dkrw is None else -np.asarray(dkrw),
                None if dkro_water is None else -np.asarray(dkro_water),
            )
            wetting, non_wetting, wetting_slope, non_wetting_slope = self._scan_system(
                system=typing.cast(KilloughSystem, self.oil_system),
                size=size,
                saturation=sw,
                peak=inputs.water_minimum,
                reflect=True,
                drainage=drainage,
                curves_at=lambda table, oil, with_derivatives: self._oil_curves(
                    table, oil, gas=sg, with_derivatives=with_derivatives
                ),
                drainage_table=self.oil_water_drainage_table,
                imbibition_table=typing.cast(
                    RelativePermeabilityTable, self.oil_water_imbibition_table
                ),
                with_derivatives=with_derivatives,
            )
            kro_water = non_wetting
            if self.wetting_phase_hysteresis:
                krw = wetting
            if with_derivatives:
                dkro_water = -np.asarray(non_wetting_slope)
                if self.wetting_phase_hysteresis:
                    dkrw = -np.asarray(wetting_slope)

        if use_gas:
            wetting, non_wetting, wetting_slope, non_wetting_slope = self._scan_system(
                system=typing.cast(KilloughSystem, self.gas_system),
                size=size,
                saturation=sg,
                peak=inputs.gas_reversal,
                reflect=False,
                drainage=(kro_gas, krg, dkro_gas, dkrg),
                curves_at=lambda table, gas, with_derivatives: self._gas_curves(
                    table, gas, water=sw, with_derivatives=with_derivatives
                ),
                drainage_table=self.gas_oil_drainage_table,
                imbibition_table=typing.cast(
                    RelativePermeabilityTable, self.gas_oil_imbibition_table
                ),
                with_derivatives=with_derivatives,
            )
            krg = non_wetting
            if self.wetting_phase_hysteresis:
                kro_gas = wetting
            if with_derivatives:
                dkrg = non_wetting_slope
                if self.wetting_phase_hysteresis:
                    dkro_gas = wetting_slope

        # Scanning has produced the final curves, so the shared kernel only has to mix them.
        curves = BoundingCurves(
            krw_d=krw,
            krw_i=krw,
            kro_water_d=kro_water,
            kro_water_i=kro_water,
            kro_gas_d=kro_gas,
            kro_gas_i=kro_gas,
            krg_d=krg,
            krg_i=krg,
            dkrw_d=dkrw,
            dkrw_i=dkrw,
            dkro_water_d=dkro_water,
            dkro_water_i=dkro_water,
            dkro_gas_d=dkro_gas,
            dkro_gas_i=dkro_gas,
            dkrg_d=dkrg,
            dkrg_i=dkrg,
        )
        return self.scan_bounding_curves(
            inputs=inputs,
            history=history,
            curves=curves,
            scanning_exponent=1.0,
            gas_translated_scanning=False,
            with_derivatives=with_derivatives,
        )

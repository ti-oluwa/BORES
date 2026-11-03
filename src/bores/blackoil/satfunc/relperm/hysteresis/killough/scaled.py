"""
Killough (1976) scanning curves for relative permeability hysteresis.

The blended scheme of `BlendedKilloughRelPermTable` weighs the drainage and imbibition curves of
every phase against each other. Here the two kinds of phase are treated differently, and nothing is
blended with the drainage curve at the same saturation.

**Non-wetting phase.** Imbibition from a reversal at the saturation `s_hy` traps `s_t` of the
phase, by Land's relation measured from the drainage critical saturation `s_crd`:

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
    TwoPhaseRelPerms,
    query_gas_oil_curves,
    query_oil_water_curves,
)
from bores.blackoil.satfunc.relperm.hysteresis.land import (
    compute_land_residual_saturation_scalar,
    compute_land_trapping_coefficient,
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
    "KilloughEndpoints",
    "KilloughRelPermTable",
    "scan_killough_curves_kernel",
    "stretch_saturation_axis_kernel",
]

SATURATION_EPSILON = 1e-12
"""Smallest saturation span that is scanned."""


class KilloughEndpoints(typing.NamedTuple):
    """
    What Killough's scheme needs to know about one two-phase system, read off its two tables.

    Saturations are those of the system's non-wetting phase: gas in the gas-oil system, and oil in
    the oil-water system, where the saturation is `1 - Sw`.
    """

    drainage_critical_saturation: Number
    """Non-wetting saturation below which the drainage curve is immobile, `s_crd`."""

    imbibition_critical_saturation: Number
    """
    Non-wetting saturation below which the imbibition curve is immobile, `s_cri`. It is the most
    that can be trapped, which happens when imbibition starts from `maximum_saturation`.
    """

    maximum_saturation: Number
    """Largest non-wetting saturation the system reaches, `s_max`."""

    land_trapping_coefficient: Number
    """
    Land's coefficient `C`, measured from `drainage_critical_saturation`. It is very large when the
    imbibition table traps nothing beyond the drainage one, which leaves nothing extra trapped.
    """

    imbibition_non_wetting_relperm_at_critical_saturation: Number
    """Non-wetting relative permeability of the imbibition table at its critical saturation."""

    imbibition_non_wetting_relperm_at_maximum_saturation: Number
    """Non-wetting relative permeability of the imbibition table at the largest saturation."""

    imbibition_wetting_relperm_at_critical_saturation: Number
    """Wetting relative permeability of the imbibition table at its critical saturation."""

    imbibition_wetting_relperm_at_maximum_saturation: Number
    """Wetting relative permeability of the imbibition table at the largest saturation."""

    drainage_wetting_relperm_at_imbibition_critical_saturation: Number
    """Wetting relative permeability of the drainage table at the imbibition critical saturation."""


@numba.njit(cache=True, error_model="numpy")
def stretch_saturation_axis_kernel(
    size: Integer,
    saturation: NumberArray[OneDimension],
    reversal_saturation: NumberArray[OneDimension],
    non_wetting_is_complement: Boolean,
    endpoints: KilloughEndpoints,
    out_trapped_saturation: NumberArray[OneDimension],
    out_mapped_saturation: NumberArray[OneDimension],
    out_axis_scale: NumberArray[OneDimension],
    out_is_scanning: NumberArray[OneDimension],
) -> None:
    """
    For every cell, the saturation trapped by Land's relation and the stretched saturation to read
    the imbibition curve at, both in the saturation of the non-wetting phase.

    The cell is on a scanning curve while its saturation is below its reversal saturation, and on the
    drainage curve at and above it.

    :param size: Number of cells.
    :param saturation: Each cell's saturation, or its water saturation when
        `non_wetting_is_complement` is set.
    :param reversal_saturation: Each cell's reversal saturation, or its minimum water saturation
        when `non_wetting_is_complement` is set. May be one shared value.
    :param non_wetting_is_complement: Whether the non-wetting phase's saturation is one minus the
        saturation given, which is how oil in a water-wet system is read off water.
    :param endpoints: The system's endpoints.
    :param out_trapped_saturation: Receives the trapped saturation.
    :param out_mapped_saturation: Receives the saturation to read the imbibition curve at, which is
        never below the imbibition critical saturation.
    :param out_axis_scale: Receives the slope of the stretch.
    :param out_is_scanning: Receives 1 for a cell on a scanning curve and 0 for one on the drainage
        curve.
    """
    drainage_critical = endpoints.drainage_critical_saturation
    imbibition_critical = endpoints.imbibition_critical_saturation
    imbibition_span = endpoints.maximum_saturation - imbibition_critical
    for i in range(size):
        current = saturation[i] if saturation.size > 1 else saturation[0]
        reversal = (
            reversal_saturation[i] if reversal_saturation.size > 1 else reversal_saturation[0]
        )
        if non_wetting_is_complement:
            current = 1.0 - current
            reversal = 1.0 - reversal
        excess = reversal - drainage_critical
        if excess > SATURATION_EPSILON:
            trapped_excess = compute_land_residual_saturation_scalar(
                initial_non_wetting_saturation=excess,
                maximum_residual_saturation=imbibition_critical - drainage_critical,
                land_trapping_coefficient=endpoints.land_trapping_coefficient,
                saturation_epsilon=SATURATION_EPSILON,
            )
            trapped = drainage_critical + trapped_excess
        else:
            trapped = reversal
        span = reversal - trapped
        out_trapped_saturation[i] = trapped
        if (
            span > SATURATION_EPSILON
            and imbibition_span > SATURATION_EPSILON
            and current < reversal
        ):
            scale = imbibition_span / span
            # Below the trapped saturation the phase is immobile and the stretched saturation would
            # fall below the imbibition table's own critical saturation, so it stops there.
            out_mapped_saturation[i] = max(
                imbibition_critical + (current - trapped) * scale, imbibition_critical
            )
            out_axis_scale[i] = scale
            out_is_scanning[i] = 1.0
        else:
            out_mapped_saturation[i] = current
            out_axis_scale[i] = 1.0
            out_is_scanning[i] = 0.0


@numba.njit(cache=True, error_model="numpy")
def scan_killough_curves_kernel(
    size: Integer,
    is_scanning: NumberArray[OneDimension],
    axis_scale: NumberArray[OneDimension],
    trapped_saturation: NumberArray[OneDimension],
    drainage_non_wetting_relperm: NumberArray[OneDimension],
    drainage_wetting_relperm: NumberArray[OneDimension],
    drainage_non_wetting_relperm_slope: NumberArray[OneDimension],
    drainage_wetting_relperm_slope: NumberArray[OneDimension],
    reversal_drainage_non_wetting_relperm: NumberArray[OneDimension],
    reversal_drainage_wetting_relperm: NumberArray[OneDimension],
    trapped_drainage_wetting_relperm: NumberArray[OneDimension],
    mapped_imbibition_non_wetting_relperm: NumberArray[OneDimension],
    mapped_imbibition_wetting_relperm: NumberArray[OneDimension],
    mapped_imbibition_non_wetting_relperm_slope: NumberArray[OneDimension],
    mapped_imbibition_wetting_relperm_slope: NumberArray[OneDimension],
    endpoints: KilloughEndpoints,
    wetting_phase_curvature: Number,
    scan_wetting_phase: Boolean,
    with_derivatives: Boolean,
    out_non_wetting_relperm: NumberArray[OneDimension],
    out_wetting_relperm: NumberArray[OneDimension],
    out_non_wetting_relperm_slope: NumberArray[OneDimension],
    out_wetting_relperm_slope: NumberArray[OneDimension],
) -> None:
    """
    Killough's scanning curves of the non-wetting and the wetting phase for every cell, in the
    saturation of the non-wetting phase and in one pass.

    A cell that is not on a scanning curve keeps its drainage values and slopes. Every slope is with
    respect to the non-wetting phase's saturation. See the module notes for the equations.

    :param size: Number of cells.
    :param is_scanning: 1 for a cell on a scanning curve, from `stretch_saturation_axis_kernel`.
    :param axis_scale: Slope of the stretched axis, from `stretch_saturation_axis_kernel`.
    :param trapped_saturation: Trapped saturation, from `stretch_saturation_axis_kernel`.
    :param drainage_non_wetting_relperm: Drainage non-wetting value at each cell's saturation.
    :param drainage_wetting_relperm: Drainage wetting value at each cell's saturation.
    :param drainage_non_wetting_relperm_slope: Slope of `drainage_non_wetting_relperm`.
    :param drainage_wetting_relperm_slope: Slope of `drainage_wetting_relperm`.
    :param reversal_drainage_non_wetting_relperm: Drainage non-wetting value at the reversal.
    :param reversal_drainage_wetting_relperm: Drainage wetting value at the reversal.
    :param trapped_drainage_wetting_relperm: Drainage wetting value at the trapped saturation.
    :param mapped_imbibition_non_wetting_relperm: Imbibition non-wetting value at the stretched
        saturation.
    :param mapped_imbibition_wetting_relperm: Imbibition wetting value at the stretched saturation.
    :param mapped_imbibition_non_wetting_relperm_slope: Slope of the imbibition non-wetting value.
    :param mapped_imbibition_wetting_relperm_slope: Slope of the imbibition wetting value.
    :param endpoints: The system's endpoints.
    :param wetting_phase_curvature: The exponent `a` of the wetting phase's trapped value.
    :param scan_wetting_phase: Whether the wetting phase scans or stays on its drainage curve.
    :param with_derivatives: Whether the slopes are present and wanted.
    :param out_non_wetting_relperm: Receives the non-wetting relative permeability.
    :param out_wetting_relperm: Receives the wetting relative permeability.
    :param out_non_wetting_relperm_slope: Receives its slope, when wanted.
    :param out_wetting_relperm_slope: Receives its slope, when wanted.
    """
    imbibition_critical = endpoints.imbibition_critical_saturation
    non_wetting_at_critical = endpoints.imbibition_non_wetting_relperm_at_critical_saturation
    non_wetting_range = (
        endpoints.imbibition_non_wetting_relperm_at_maximum_saturation - non_wetting_at_critical
    )
    wetting_at_maximum = endpoints.imbibition_wetting_relperm_at_maximum_saturation
    wetting_range = (
        endpoints.imbibition_wetting_relperm_at_critical_saturation - wetting_at_maximum
    )
    for i in range(size):
        out_non_wetting_relperm[i] = drainage_non_wetting_relperm[i]
        out_wetting_relperm[i] = drainage_wetting_relperm[i]
        if with_derivatives:
            out_non_wetting_relperm_slope[i] = drainage_non_wetting_relperm_slope[i]
            out_wetting_relperm_slope[i] = drainage_wetting_relperm_slope[i]
        if is_scanning[i] < 0.5:
            continue
        stretch = axis_scale[i]

        if abs(non_wetting_range) > SATURATION_EPSILON:
            progress = (mapped_imbibition_non_wetting_relperm[i] - non_wetting_at_critical) / (
                non_wetting_range
            )
            if progress > 0.0:
                out_non_wetting_relperm[i] = reversal_drainage_non_wetting_relperm[i] * progress
                if with_derivatives:
                    out_non_wetting_relperm_slope[i] = (
                        reversal_drainage_non_wetting_relperm[i]
                        * mapped_imbibition_non_wetting_relperm_slope[i]
                        * stretch
                        / non_wetting_range
                    )
            else:
                out_non_wetting_relperm[i] = 0.0
                if with_derivatives:
                    out_non_wetting_relperm_slope[i] = 0.0

        if scan_wetting_phase and abs(wetting_range) > SATURATION_EPSILON:
            if imbibition_critical > SATURATION_EPSILON:
                trapped_fraction = (trapped_saturation[i] / imbibition_critical) ** (
                    wetting_phase_curvature
                )
            else:
                trapped_fraction = 1.0
            trapped_relperm = (
                trapped_drainage_wetting_relperm[i]
                + (
                    endpoints.imbibition_wetting_relperm_at_critical_saturation
                    - endpoints.drainage_wetting_relperm_at_imbibition_critical_saturation
                )
                * trapped_fraction
            )
            travel = trapped_relperm - reversal_drainage_wetting_relperm[i]
            progress = (mapped_imbibition_wetting_relperm[i] - wetting_at_maximum) / wetting_range
            on_curve = 0.0 < progress < 1.0
            progress = min(max(progress, 0.0), 1.0)
            out_wetting_relperm[i] = reversal_drainage_wetting_relperm[i] + progress * travel
            if with_derivatives:
                out_wetting_relperm_slope[i] = (
                    mapped_imbibition_wetting_relperm_slope[i] * stretch / wetting_range * travel
                    if on_curve
                    else 0.0
                )


@attrs.frozen(slots=True)
class KilloughRelPermTable(HysteresisRelPermTable):
    """
    Killough relative permeability hysteresis (see the module notes).

    The non-wetting phase of each system scans: gas in the gas-oil system, and oil in the oil-water
    system when water is the wetting phase. Gas peaks at its historical maximum saturation and oil at
    one minus the historical minimum water saturation, so the saturation history must carry both.
    The wetting phase follows its drainage curve unless `wetting_phase_hysteresis` is set. The
    three-phase mixing rule then combines the curves as it does for the other models.

    Land trapping is part of the model and is derived from the tables: the imbibition table's critical
    saturation is what is trapped when imbibition starts from the largest saturation of the drainage
    table. The maximum residual and Land coefficient fields of the base class are not used.

    A system without an imbibition table, without history, or with nothing between its critical and
    its largest saturation stays on its drainage curves.

    Not implemented: the modification of the trapped saturation (`EHYSTR` item 4).
    """

    __type__ = "killough_relperm_model"

    wetting_phase_hysteresis: Boolean = False
    """Whether the wetting phase scans too, rather than staying on its drainage curve."""

    wetting_phase_curvature: Number = 1.0
    """The exponent `a` of the wetting phase's value at the trapped saturation (`EHYSTR` item 3)."""

    gas_oil_endpoints: KilloughEndpoints | None = attrs.field(init=False, repr=False, eq=False)
    """The gas-oil system's endpoints, or `None` when it does not scan."""

    oil_water_endpoints: KilloughEndpoints | None = attrs.field(init=False, repr=False, eq=False)
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
        object.__setattr__(self, "gas_oil_endpoints", self._read_gas_oil_endpoints())
        object.__setattr__(self, "oil_water_endpoints", self._read_oil_water_endpoints())

    def gas_oil_relperms_at(
        self,
        table: RelativePermeabilityTable,
        gas_saturation: NumberArray[OneDimension],
        *,
        water_saturation: NumberArray[OneDimension],
        with_derivatives: bool,
    ) -> TwoPhaseRelPerms:
        """
        Relative permeabilities of the gas-oil system at the given gas saturations.

        :param table: The table to read.
        :param gas_saturation: Gas saturation of every cell.
        :param water_saturation: Water saturation of every cell.
        :param with_derivatives: Whether to also read the slopes.
        :returns: The oil and gas relative permeabilities, with slopes with respect to gas.
        """
        return query_gas_oil_curves(
            table,
            water_saturation=water_saturation,
            oil_saturation=1.0 - water_saturation - gas_saturation,
            gas_saturation=gas_saturation,
            connate_water_saturation=self.get_connate_water_saturation(),
            with_derivatives=with_derivatives,
        )

    def oil_water_relperms_at(
        self,
        table: RelativePermeabilityTable,
        oil_saturation: NumberArray[OneDimension],
        *,
        gas_saturation: NumberArray[OneDimension],
        with_derivatives: bool,
    ) -> TwoPhaseRelPerms:
        """
        Relative permeabilities of the oil-water system at the given oil saturations, where oil
        saturation means one minus water saturation.

        :param table: The table to read.
        :param oil_saturation: One minus the water saturation of every cell.
        :param gas_saturation: Gas saturation of every cell.
        :param with_derivatives: Whether to also read the slopes.
        :returns: The water and oil relative permeabilities, with slopes with respect to the oil
            saturation, not to water.
        """
        relperms = query_oil_water_curves(
            table,
            water_saturation=1.0 - oil_saturation,
            oil_saturation=oil_saturation - gas_saturation,
            gas_saturation=gas_saturation,
            with_derivatives=with_derivatives,
        )
        if relperms.wetting_relperm_slope is None or relperms.non_wetting_relperm_slope is None:
            return relperms
        return TwoPhaseRelPerms(
            relperms.wetting_relperm,
            relperms.non_wetting_relperm,
            -relperms.wetting_relperm_slope,
            -relperms.non_wetting_relperm_slope,
        )

    def _read_gas_oil_endpoints(self) -> KilloughEndpoints | None:
        imbibition = self.gas_oil_imbibition_table
        if imbibition is None:
            return None
        drainage = self.gas_oil_drainage_table
        connate_water = self.get_connate_water_saturation()
        drainage_critical = drainage.get_residual_gas_saturation()
        imbibition_critical = imbibition.get_residual_gas_saturation()
        maximum = (
            1.0 - connate_water - resolve_residual_oil_saturation_gas(drainage, connate_water)
        )
        if maximum - imbibition_critical <= SATURATION_EPSILON:
            return None

        def relperms_at(table: RelativePermeabilityTable, gas: Number) -> TwoPhaseRelPerms:
            return self.gas_oil_relperms_at(
                table,
                np.array([gas]),
                water_saturation=np.array([connate_water]),
                with_derivatives=False,
            )

        at_critical = relperms_at(imbibition, imbibition_critical)
        at_maximum = relperms_at(imbibition, maximum)
        return KilloughEndpoints(
            drainage_critical_saturation=drainage_critical,
            imbibition_critical_saturation=imbibition_critical,
            maximum_saturation=maximum,
            land_trapping_coefficient=compute_land_trapping_coefficient(
                maximum_residual_saturation=imbibition_critical - drainage_critical,
                maximum_non_wetting_saturation=maximum - drainage_critical,
            ),
            imbibition_non_wetting_relperm_at_critical_saturation=at_critical.non_wetting_relperm[
                0
            ],
            imbibition_non_wetting_relperm_at_maximum_saturation=at_maximum.non_wetting_relperm[0],
            imbibition_wetting_relperm_at_critical_saturation=at_critical.wetting_relperm[0],
            imbibition_wetting_relperm_at_maximum_saturation=at_maximum.wetting_relperm[0],
            drainage_wetting_relperm_at_imbibition_critical_saturation=(
                relperms_at(drainage, imbibition_critical).wetting_relperm[0]
            ),
        )

    def _read_oil_water_endpoints(self) -> KilloughEndpoints | None:
        imbibition = self.oil_water_imbibition_table
        if imbibition is None or self.get_oil_water_wetting_phase() != FluidPhase.WATER:
            return None
        drainage = self.oil_water_drainage_table
        drainage_critical = drainage.get_residual_oil_saturation_water()
        imbibition_critical = imbibition.get_residual_oil_saturation_water()
        maximum = 1.0 - self.get_connate_water_saturation()
        if maximum - imbibition_critical <= SATURATION_EPSILON:
            return None

        def relperms_at(table: RelativePermeabilityTable, oil: Number) -> TwoPhaseRelPerms:
            return self.oil_water_relperms_at(
                table, np.array([oil]), gas_saturation=np.zeros(1), with_derivatives=False
            )

        at_critical = relperms_at(imbibition, imbibition_critical)
        at_maximum = relperms_at(imbibition, maximum)
        return KilloughEndpoints(
            drainage_critical_saturation=drainage_critical,
            imbibition_critical_saturation=imbibition_critical,
            maximum_saturation=maximum,
            land_trapping_coefficient=compute_land_trapping_coefficient(
                maximum_residual_saturation=imbibition_critical - drainage_critical,
                maximum_non_wetting_saturation=maximum - drainage_critical,
            ),
            imbibition_non_wetting_relperm_at_critical_saturation=at_critical.non_wetting_relperm[
                0
            ],
            imbibition_non_wetting_relperm_at_maximum_saturation=at_maximum.non_wetting_relperm[0],
            imbibition_wetting_relperm_at_critical_saturation=at_critical.wetting_relperm[0],
            imbibition_wetting_relperm_at_maximum_saturation=at_maximum.wetting_relperm[0],
            drainage_wetting_relperm_at_imbibition_critical_saturation=(
                relperms_at(drainage, imbibition_critical).wetting_relperm[0]
            ),
        )

    def scan_system(
        self,
        *,
        endpoints: KilloughEndpoints,
        size: int,
        saturation: NumberArray[OneDimension],
        reversal_saturation: NumberArray[OneDimension],
        non_wetting_is_complement: bool,
        drainage_relperms: TwoPhaseRelPerms,
        relperms_at: typing.Callable[
            [RelativePermeabilityTable, NumberArray[OneDimension], bool], TwoPhaseRelPerms
        ],
        drainage_table: RelativePermeabilityTable,
        imbibition_table: RelativePermeabilityTable,
        with_derivatives: bool,
    ) -> TwoPhaseRelPerms:
        """
        Killough's scanning curves of one two-phase system, in the saturation of its non-wetting
        phase.

        :param endpoints: The system's endpoints.
        :param size: Number of cells.
        :param saturation: Each cell's saturation, or its water saturation for the oil-water system.
        :param reversal_saturation: Each cell's reversal saturation, or its minimum water saturation
            for the oil-water system.
        :param non_wetting_is_complement: Whether the non-wetting saturation is one minus the
            saturation given.
        :param drainage_relperms: The system's drainage curves at the cells' own saturations, with
            slopes with respect to the non-wetting saturation.
        :param relperms_at: Reads a table's curves at an array of non-wetting saturations, with
            slopes with respect to that saturation, and whether to read the slopes.
        :param drainage_table: The drainage table.
        :param imbibition_table: The imbibition table.
        :param with_derivatives: Whether to also compute the slopes.
        :returns: The scanned curves, with slopes with respect to the non-wetting saturation.
        """
        dtype = drainage_relperms.wetting_relperm.dtype
        trapped_saturation = np.empty(size, dtype=dtype)
        mapped_saturation = np.empty(size, dtype=dtype)
        axis_scale = np.empty(size, dtype=dtype)
        is_scanning = np.empty(size, dtype=dtype)
        stretch_saturation_axis_kernel(
            size=size,
            saturation=saturation,
            reversal_saturation=reversal_saturation,
            non_wetting_is_complement=non_wetting_is_complement,
            endpoints=endpoints,
            out_trapped_saturation=trapped_saturation,
            out_mapped_saturation=mapped_saturation,
            out_axis_scale=axis_scale,
            out_is_scanning=is_scanning,
        )
        reversal_non_wetting = (
            1.0 - reversal_saturation if non_wetting_is_complement else reversal_saturation
        )
        at_reversal = relperms_at(drainage_table, reversal_non_wetting, False)
        at_trapped = relperms_at(drainage_table, trapped_saturation, False)
        mapped = relperms_at(imbibition_table, mapped_saturation, with_derivatives)

        def per_cell(array: NumberArray[OneDimension]) -> NumberArray[OneDimension]:
            return np.ascontiguousarray(np.broadcast_to(array, (size,)))

        placeholder = np.zeros(1, dtype=dtype)

        def slope(array: NumberArray[OneDimension] | None) -> NumberArray[OneDimension]:
            return placeholder if array is None else per_cell(array)

        out_non_wetting = np.empty(size, dtype=dtype)
        out_wetting = np.empty(size, dtype=dtype)
        out_non_wetting_slope = np.empty(size if with_derivatives else 1, dtype=dtype)
        out_wetting_slope = np.empty(size if with_derivatives else 1, dtype=dtype)
        scan_killough_curves_kernel(
            size=size,
            is_scanning=is_scanning,
            axis_scale=axis_scale,
            trapped_saturation=trapped_saturation,
            drainage_non_wetting_relperm=per_cell(drainage_relperms.non_wetting_relperm),
            drainage_wetting_relperm=per_cell(drainage_relperms.wetting_relperm),
            drainage_non_wetting_relperm_slope=slope(drainage_relperms.non_wetting_relperm_slope),
            drainage_wetting_relperm_slope=slope(drainage_relperms.wetting_relperm_slope),
            reversal_drainage_non_wetting_relperm=per_cell(at_reversal.non_wetting_relperm),
            reversal_drainage_wetting_relperm=per_cell(at_reversal.wetting_relperm),
            trapped_drainage_wetting_relperm=per_cell(at_trapped.wetting_relperm),
            mapped_imbibition_non_wetting_relperm=per_cell(mapped.non_wetting_relperm),
            mapped_imbibition_wetting_relperm=per_cell(mapped.wetting_relperm),
            mapped_imbibition_non_wetting_relperm_slope=slope(mapped.non_wetting_relperm_slope),
            mapped_imbibition_wetting_relperm_slope=slope(mapped.wetting_relperm_slope),
            endpoints=endpoints,
            wetting_phase_curvature=self.wetting_phase_curvature,
            scan_wetting_phase=self.wetting_phase_hysteresis,
            with_derivatives=with_derivatives,
            out_non_wetting_relperm=out_non_wetting,
            out_wetting_relperm=out_wetting,
            out_non_wetting_relperm_slope=out_non_wetting_slope,
            out_wetting_relperm_slope=out_wetting_slope,
        )
        return TwoPhaseRelPerms(
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
        gas_oil_endpoints = self.gas_oil_endpoints
        oil_water_endpoints = self.oil_water_endpoints
        scan_gas = (
            inputs.use_gas_hysteresis
            and gas_oil_endpoints is not None
            and inputs.gas_is_non_wetting
        )
        scan_oil = (
            inputs.use_water_hysteresis
            and oil_water_endpoints is not None
            and min_water_saturation is not None
        )
        inputs = inputs._replace(use_water_hysteresis=False, use_gas_hysteresis=False)

        curve_arguments = dict(
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            with_derivatives=with_derivatives,
        )
        oil_water = query_oil_water_curves(
            self.oil_water_drainage_table, **curve_arguments, **kwargs
        )
        gas_oil = query_gas_oil_curves(
            self.gas_oil_drainage_table,
            **curve_arguments,
            connate_water_saturation=self.get_connate_water_saturation(),
            **kwargs,
        )

        if scan_oil:
            scanned = self.scan_system(
                endpoints=typing.cast(KilloughEndpoints, oil_water_endpoints),
                size=size,
                saturation=sw,
                reversal_saturation=inputs.water_minimum,
                non_wetting_is_complement=True,
                drainage_relperms=TwoPhaseRelPerms(
                    oil_water.wetting_relperm,
                    oil_water.non_wetting_relperm,
                    None
                    if oil_water.wetting_relperm_slope is None
                    else -oil_water.wetting_relperm_slope,
                    None
                    if oil_water.non_wetting_relperm_slope is None
                    else -oil_water.non_wetting_relperm_slope,
                ),
                relperms_at=lambda table, oil, with_slopes: self.oil_water_relperms_at(
                    table, oil, gas_saturation=sg, with_derivatives=with_slopes
                ),
                drainage_table=self.oil_water_drainage_table,
                imbibition_table=typing.cast(
                    RelativePermeabilityTable, self.oil_water_imbibition_table
                ),
                with_derivatives=with_derivatives,
            )
            # The scanned slopes are with respect to oil, and the models take them with respect to water.
            oil_water = TwoPhaseRelPerms(
                scanned.wetting_relperm
                if self.wetting_phase_hysteresis
                else oil_water.wetting_relperm,
                scanned.non_wetting_relperm,
                oil_water.wetting_relperm_slope
                if scanned.wetting_relperm_slope is None or not self.wetting_phase_hysteresis
                else -scanned.wetting_relperm_slope,
                None
                if scanned.non_wetting_relperm_slope is None
                else -scanned.non_wetting_relperm_slope,
            )

        if scan_gas:
            scanned = self.scan_system(
                endpoints=typing.cast(KilloughEndpoints, gas_oil_endpoints),
                size=size,
                saturation=sg,
                reversal_saturation=inputs.gas_reversal,
                non_wetting_is_complement=False,
                drainage_relperms=gas_oil,
                relperms_at=lambda table, gas, with_slopes: self.gas_oil_relperms_at(
                    table, gas, water_saturation=sw, with_derivatives=with_slopes
                ),
                drainage_table=self.gas_oil_drainage_table,
                imbibition_table=typing.cast(
                    RelativePermeabilityTable, self.gas_oil_imbibition_table
                ),
                with_derivatives=with_derivatives,
            )
            gas_oil = TwoPhaseRelPerms(
                scanned.wetting_relperm
                if self.wetting_phase_hysteresis
                else gas_oil.wetting_relperm,
                scanned.non_wetting_relperm,
                scanned.wetting_relperm_slope
                if self.wetting_phase_hysteresis
                else gas_oil.wetting_relperm_slope,
                scanned.non_wetting_relperm_slope,
            )

        # Scanning has produced the final curves, so the shared kernel only has to mix them.
        curves = BoundingCurves(
            krw_d=oil_water.wetting_relperm,
            krw_i=oil_water.wetting_relperm,
            kro_water_d=oil_water.non_wetting_relperm,
            kro_water_i=oil_water.non_wetting_relperm,
            kro_gas_d=gas_oil.wetting_relperm,
            kro_gas_i=gas_oil.wetting_relperm,
            krg_d=gas_oil.non_wetting_relperm,
            krg_i=gas_oil.non_wetting_relperm,
            dkrw_d=oil_water.wetting_relperm_slope,
            dkrw_i=oil_water.wetting_relperm_slope,
            dkro_water_d=oil_water.non_wetting_relperm_slope,
            dkro_water_i=oil_water.non_wetting_relperm_slope,
            dkro_gas_d=gas_oil.wetting_relperm_slope,
            dkro_gas_i=gas_oil.wetting_relperm_slope,
            dkrg_d=gas_oil.non_wetting_relperm_slope,
            dkrg_i=gas_oil.non_wetting_relperm_slope,
        )
        return self.scan_bounding_curves(
            inputs=inputs,
            history=history,
            curves=curves,
            scanning_exponent=1.0,
            gas_translated_scanning=False,
            with_derivatives=with_derivatives,
        )

"""
Carlson (1981) scanning curves for the non-wetting phase.

The non-wetting phase is the only one that scans. Drainage follows the drainage curve. After a
reversal at the saturation `s_hst` the scanning curve is the imbibition curve translated along the
saturation axis by `Δ`, chosen so that the translated curve passes through the reversal point:

    kr_imb(s_hst + Δ) = kr_drain(s_hst),        kr_scan(s) = kr_imb(s + Δ)

The scanning curve is retraced when the saturation rises again, and the drainage curve is rejoined at
`s_hst`. Every scanning curve is the same imbibition curve, only shifted, so scanning curves are
parallel. The trapped saturation is where the translated curve reaches zero, so it needs no separate
trapping model.

- Carlson, F.M. (1981). "Simulation of Relative Permeability Hysteresis to the Non-Wetting Phase".
  SPE 10157.
- Rasmussen, A.F. et al. (2021). "The Open Porous Media Flow Reservoir Simulator", Computers &
  Mathematics with Applications 81, section 2.1.5, which describes the same translation.
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
from bores.errors import ValidationError
from bores.types import (
    Boolean,
    BooleanArray,
    FluidPhase,
    Integer,
    NDimension,
    NumberArray,
    NumberOrArray,
    OneDimension,
    RelativePermeabilities,
    RelativePermeabilityDerivatives,
)

__all__ = ["CarlsonRelPermTable", "translate_scanning_saturation"]

SHIFT_CURVE_POINTS = 513
"""Points on the saturation axis at which the translation `Δ` is tabulated, once per table."""


@numba.njit(cache=True, error_model="numpy")
def translate_scanning_saturation(
    size: Integer,
    saturation: NumberArray[OneDimension],
    reversal_saturation: NumberArray[OneDimension],
    shift_axis: NumberArray[OneDimension],
    shift: NumberArray[OneDimension],
    out_saturation: NumberArray[OneDimension],
) -> None:
    """
    Saturation at which to read the imbibition curve for each cell, `s + Δ(s_hst)`.

    `Δ` is read off the tabulated translation curve by linear interpolation at the cell's reversal
    saturation, in one pass and without intermediate arrays. A reversal outside the tabulated axis
    takes the nearest tabulated translation.

    :param size: Number of cells.
    :param saturation: Each cell's saturation.
    :param reversal_saturation: Each cell's reversal (peak) saturation, or one shared value.
    :param shift_axis: Ascending saturations at which the translation is tabulated.
    :param shift: Translation `Δ` at each tabulated saturation.
    :param out_saturation: Receives the saturation to read the imbibition curve at.
    """
    last = shift_axis.size - 1
    for i in range(size):
        reversal = (
            reversal_saturation[i] if reversal_saturation.size > 1 else reversal_saturation[0]
        )
        current = saturation[i] if saturation.size > 1 else saturation[0]
        if reversal <= shift_axis[0]:
            delta = shift[0]
        elif reversal >= shift_axis[last]:
            delta = shift[last]
        else:
            upper = np.searchsorted(shift_axis, reversal)
            lower = upper - 1
            fraction = (reversal - shift_axis[lower]) / (shift_axis[upper] - shift_axis[lower])
            delta = shift[lower] + fraction * (shift[upper] - shift[lower])
        out_saturation[i] = current + delta


def tabulate_translation(
    *,
    axis: NumberArray[OneDimension],
    drainage_values: NumberArray[OneDimension],
    imbibition_values: NumberArray[OneDimension],
) -> NumberArray[OneDimension]:
    """
    Tabulate `Δ(s_hst)` from `kr_imb(s_hst + Δ) = kr_drain(s_hst)` along `axis`.

    The imbibition curve is inverted by interpolation. A relative permeability the imbibition curve
    never reaches, because it tops out lower, is read off at the end of the axis.

    :param axis: Ascending saturations, from zero to the largest the system reaches.
    :param drainage_values: Drainage relative permeability along `axis`.
    :param imbibition_values: Imbibition relative permeability along `axis`.
    :returns: The translation at every point of `axis`.
    """
    rising = np.maximum.accumulate(np.asarray(imbibition_values, dtype=np.float64))
    levels, first = np.unique(rising, return_index=True)
    inverse = np.interp(
        np.asarray(drainage_values, dtype=np.float64), levels, np.asarray(axis)[first]
    )
    return typing.cast("NumberArray[OneDimension]", inverse - np.asarray(axis, dtype=np.float64))


@attrs.frozen(slots=True)
class CarlsonRelPermTable(HysteresisRelPermTable):
    """
    Carlson relative permeability hysteresis for the non-wetting phase, which here is gas in the
    gas-oil system.

    Gas follows its drainage curve while its saturation rises and, once it falls from a peak, the
    imbibition curve translated to pass through the reversal point (see the module notes). The oil
    in the gas-oil system, the wetting phase there, follows its drainage curve, or its imbibition
    curve when `wetting_phase_uses_imbibition` is set. The three-phase mixing rule then combines
    the curves exactly as it does for the Killough model.

    The translation is tabulated once, when the table is built, from the two gas curves, so a
    timestep only reads it. Land trapping is not used: the trapped saturation is where the
    translated curve reaches zero. Oil, the non-wetting phase of the oil-water system, scans the
    same way, but that needs the historical minimum water saturation, which the saturation
    history does not track yet, so only an oil-water imbibition table that serves the wetting phase is
    accepted.

    The translated curve can pass through the reversal point only if the imbibition curve reaches
    the drainage value there. Where the imbibition curve tops out lower, which real data should not
    do, the scanning curve starts from that top and kr drops at the reversal.

    The saturation history is passed as keyword arguments, as for the Killough model. A system
    without history, or without an imbibition table, stays on its drainage curves.
    """

    __type__ = "carlson_relperm_model"

    wetting_phase_uses_imbibition: bool = False
    """
    Whether the wetting phase takes its imbibition curve rather than its drainage curve, as
    `EHYSTR` codes 1 and 3 ask. It applies to water in the oil-water system and to oil in the
    gas-oil system.
    """

    gas_shift_axis: NumberArray[OneDimension] = attrs.field(init=False, repr=False, eq=False)
    """Gas saturations at which the translation of the gas imbibition curve is tabulated."""

    gas_shift: NumberArray[OneDimension] = attrs.field(init=False, repr=False, eq=False)
    """Translation `Δ` of the gas imbibition curve at each of `gas_shift_axis`."""

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if (
            self.oil_water_imbibition_table is not None
            and not self.wetting_phase_uses_imbibition
            and self.get_oil_water_wetting_phase() == FluidPhase.WATER
        ):
            raise ValidationError(
                "Carlson scanning of oil in the oil-water system needs the historical minimum "
                "water saturation, which is not tracked yet. Give no `oil_water_imbibition_table`, "
                "or set `wetting_phase_uses_imbibition` so that it serves water."
            )
        axis, shift = self._tabulate_gas_translation()
        object.__setattr__(self, "gas_shift_axis", axis)
        object.__setattr__(self, "gas_shift", shift)

    def _tabulate_gas_translation(
        self,
    ) -> tuple[NumberArray[OneDimension], NumberArray[OneDimension]]:
        """Tabulate the translation of the gas imbibition curve, or none when there is none."""
        imbibition_table = self.gas_oil_imbibition_table
        axis = np.zeros(2)
        if imbibition_table is None or self.get_gas_oil_wetting_phase() == FluidPhase.GAS:
            return typing.cast("NumberArray[OneDimension]", axis), typing.cast(
                "NumberArray[OneDimension]", np.zeros(2)
            )
        connate_water = self.get_connate_water_saturation()
        axis = np.linspace(0.0, 1.0 - connate_water, SHIFT_CURVE_POINTS)
        arguments = dict(
            water_saturation=np.full_like(axis, connate_water),
            oil_saturation=1.0 - connate_water - axis,
            gas_saturation=axis,
            connate_water_saturation=connate_water,
            with_derivatives=False,
        )
        _, drainage_krg, _, _ = query_gas_oil_curves(self.gas_oil_drainage_table, **arguments)  # type: ignore[arg-type]
        _, imbibition_krg, _, _ = query_gas_oil_curves(imbibition_table, **arguments)  # type: ignore[arg-type]
        shift = tabulate_translation(
            axis=typing.cast("NumberArray[OneDimension]", axis),
            drainage_values=typing.cast("NumberArray[OneDimension]", np.asarray(drainage_krg)),
            imbibition_values=typing.cast("NumberArray[OneDimension]", np.asarray(imbibition_krg)),
        )
        return typing.cast("NumberArray[OneDimension]", axis), shift

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
        **kwargs: typing.Any,
    ) -> tuple[RelativePermeabilities, RelativePermeabilityDerivatives | None]:
        """
        Relative permeabilities, and optionally their nine derivatives, in one pass.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param with_derivatives: Whether to also compute the derivatives.
        :param max_water_saturation: Historical maximum water saturation per cell. Not used.
        :param max_gas_saturation: Historical maximum gas saturation per cell. Switches gas
            hysteresis on together with `gas_imbibition_flag`.
        :param water_imbibition_flag: Per-cell water imbibition flag. Not used.
        :param gas_imbibition_flag: Per-cell gas imbibition flag. Switches gas hysteresis on
            together with `max_gas_saturation`.
        :param water_reversal_saturation: Water saturation at the last reversal. Not used.
        :param gas_reversal_saturation: Gas saturation at the last reversal, the peak the
            scanning curve starts from. Defaults to `max_gas_saturation`.
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
        )
        size = inputs.size
        sw, so, sg = inputs.sw, inputs.so, inputs.sg
        # Only the non-wetting gas scans, and only when there is an imbibition curve to translate.
        gas_table = self.gas_oil_imbibition_table
        use_gas_hysteresis = (
            inputs.use_gas_hysteresis and gas_table is not None and inputs.gas_is_non_wetting
        )
        inputs = inputs._replace(use_water_hysteresis=False, use_gas_hysteresis=use_gas_hysteresis)

        drainage_water_system = self.oil_water_drainage_table
        drainage_gas_system = self.gas_oil_drainage_table
        curve_arguments = dict(
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            with_derivatives=with_derivatives,
        )
        gas_oil_arguments = {
            **curve_arguments,
            "connate_water_saturation": self.get_connate_water_saturation(),
        }
        krw_d, kro_water_d, dkrw_d, dkro_water_d = query_oil_water_curves(
            drainage_water_system, **curve_arguments, **kwargs
        )
        kro_gas_d, krg_d, dkro_gas_d, dkrg_d = query_gas_oil_curves(
            drainage_gas_system, **gas_oil_arguments, **kwargs
        )

        # The wetting phase has no scanning curve, only a choice of curve.
        if self.wetting_phase_uses_imbibition:
            water_table = self.oil_water_imbibition_table
            if water_table is not None and self.get_oil_water_wetting_phase() == FluidPhase.WATER:
                krw_d, _, dkrw_d, _ = query_oil_water_curves(
                    water_table, **curve_arguments, **kwargs
                )
        krw_i, kro_water_i, dkrw_i, dkro_water_i = krw_d, kro_water_d, dkrw_d, dkro_water_d

        kro_gas_i, krg_i, dkro_gas_i, dkrg_i = kro_gas_d, krg_d, dkro_gas_d, dkrg_d
        if gas_table is not None and self.get_gas_oil_wetting_phase() != FluidPhase.GAS:
            if self.wetting_phase_uses_imbibition:
                kro_gas_d, _, dkro_gas_d, _ = query_gas_oil_curves(
                    gas_table, **gas_oil_arguments, **kwargs
                )
                kro_gas_i, dkro_gas_i = kro_gas_d, dkro_gas_d
        if use_gas_hysteresis:
            translated = np.empty_like(sg)
            translate_scanning_saturation(
                size=size,
                saturation=sg,
                reversal_saturation=inputs.gas_reversal,
                shift_axis=self.gas_shift_axis,
                shift=self.gas_shift,
                out_saturation=translated,
            )
            translated_arguments = {
                **gas_oil_arguments,
                "gas_saturation": translated,
                "oil_saturation": 1.0 - sw - translated,
            }
            assert gas_table is not None
            _, krg_i, _, dkrg_i = query_gas_oil_curves(gas_table, **translated_arguments, **kwargs)

        curves = BoundingCurves(
            krw_d=krw_d,
            krw_i=krw_i,
            kro_water_d=kro_water_d,
            kro_water_i=kro_water_i,
            kro_gas_d=kro_gas_d,
            kro_gas_i=kro_gas_i,
            krg_d=krg_d,
            krg_i=krg_i,
            dkrw_d=dkrw_d,
            dkrw_i=dkrw_i,
            dkro_water_d=dkro_water_d,
            dkro_water_i=dkro_water_i,
            dkro_gas_d=dkro_gas_d,
            dkro_gas_i=dkro_gas_i,
            dkrg_d=dkrg_d,
            dkrg_i=dkrg_i,
        )
        return self.scan_bounding_curves(
            inputs=inputs,
            history=history,
            curves=curves,
            scanning_exponent=1.0,
            gas_translated_scanning=True,
            with_derivatives=with_derivatives,
        )

"""
Carlson (1981) scanning curves for the non-wetting phases.

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
    vertical_scale: NumberArray[OneDimension],
    reflect: Boolean,
    out_saturation: NumberArray[OneDimension],
    out_vertical_scale: NumberArray[OneDimension],
) -> None:
    """
    Returns saturation at which to read the imbibition curve for each cell, and the factor that keeps the
    scanning curve on the drainage curve at the reversal.

    The translation `Δ` and the vertical factor are read off their tabulated curves by linear
    interpolation at the cell's reversal saturation, in one pass and without intermediate arrays.
    A reversal outside the tabulated axis takes the nearest tabulated value.

    The non-wetting phase scans along its own saturation, which is the saturation given for gas.
    Oil in a water-wet system scans along its own saturation too, `1 - s` of the water saturation
    that is given, so `reflect` maps the water saturation and its minimum onto the oil axis, and
    the translated oil saturation back onto water.

    :param size: Number of cells.
    :param saturation: Each cell's saturation.
    :param reversal_saturation: Each cell's reversal saturation: the peak of a non-wetting phase,
        or the minimum water saturation when `reflect` is set. May be one shared value.
    :param shift_axis: Ascending saturations of the scanning phase at which the curves are
        tabulated.
    :param shift: Translation `Δ` of the scanning phase at each tabulated saturation.
    :param vertical_scale: Factor on the imbibition curve at each tabulated saturation, 1 where
        the translated curve reaches the drainage value by itself.
    :param reflect: Whether the scanning phase is the complement of the saturation given.
    :param out_saturation: Receives the saturation to read the imbibition curve at.
    :param out_vertical_scale: Receives the factor to multiply the imbibition values and slopes by.
    """
    last = shift_axis.size - 1
    for i in range(size):
        reversal = (
            reversal_saturation[i] if reversal_saturation.size > 1 else reversal_saturation[0]
        )
        current = saturation[i] if saturation.size > 1 else saturation[0]
        if reflect:
            reversal = 1.0 - reversal
        if reversal <= shift_axis[0]:
            delta = shift[0]
            factor = vertical_scale[0]
        elif reversal >= shift_axis[last]:
            delta = shift[last]
            factor = vertical_scale[last]
        else:
            upper = np.searchsorted(shift_axis, reversal)
            lower = upper - 1
            fraction = (reversal - shift_axis[lower]) / (shift_axis[upper] - shift_axis[lower])
            delta = shift[lower] + fraction * (shift[upper] - shift[lower])
            factor = vertical_scale[lower] + fraction * (
                vertical_scale[upper] - vertical_scale[lower]
            )
        out_saturation[i] = 1.0 - ((1.0 - current) + delta) if reflect else current + delta
        out_vertical_scale[i] = factor


def tabulate_translation(
    *,
    axis: NumberArray[OneDimension],
    drainage_values: NumberArray[OneDimension],
    imbibition_values: NumberArray[OneDimension],
) -> tuple[NumberArray[OneDimension], NumberArray[OneDimension]]:
    """
    Tabulates the translation `Δ(s_hst)` from `kr_imb(s_hst + Δ) = kr_drain(s_hst)` along `axis`,
    and the vertical factor that makes the scanning curve meet the drainage curve when the
    imbibition curve cannot reach it.

    The imbibition curve is inverted by interpolation. Where the drainage value is above the most
    the imbibition curve ever reaches, the translation puts the curve's top at the reversal
    saturation, and the factor `kr_drain(s_hst) / max(kr_imb)` lifts that top onto the drainage
    value, so the scanning curve starts on the drainage curve instead of below it. Everywhere else
    the factor is exactly 1.

    :param axis: Ascending saturations, from zero to the largest the system reaches.
    :param drainage_values: Drainage relative permeability along `axis`.
    :param imbibition_values: Imbibition relative permeability along `axis`.
    :returns: The translation and the vertical factor at every point of `axis`.
    """
    rising = np.maximum.accumulate(np.asarray(imbibition_values, dtype=np.float64))
    levels, first = np.unique(rising, return_index=True)
    drainage = np.asarray(drainage_values, dtype=np.float64)
    inverse = np.interp(drainage, levels, np.asarray(axis)[first])
    shift = inverse - np.asarray(axis, dtype=np.float64)
    top = float(levels[-1])
    factor = np.ones_like(drainage)
    if top > 0.0:
        unreachable = drainage > top
        factor[unreachable] = drainage[unreachable] / top
    return (
        typing.cast("NumberArray[OneDimension]", shift),
        typing.cast("NumberArray[OneDimension]", factor),
    )


@attrs.frozen(slots=True)
class CarlsonRelPermTable(HysteresisRelPermTable):
    """
    Carlson relative permeability hysteresis for the non-wetting phases: gas in the gas-oil system,
    and oil in the oil-water system when water is the wetting phase.

    A non-wetting phase follows its drainage curve while its saturation rises and, once it falls
    from a peak, the imbibition curve translated to pass through the reversal point (see the
    module notes). Gas peaks at its historical maximum saturation, oil at `1` minus the historical
    minimum water saturation, so the saturation history must carry both. The wetting phases
    follow their drainage curves, or their imbibition curves when `wetting_phase_uses_imbibition`
    is set. The three-phase mixing rule then combines the curves exactly as it does for the
    Killough model.

    The translation is tabulated once, when the table is built, from the two curves, so a
    timestep only reads it. Land trapping is not used: the trapped saturation is where the
    translated curve reaches zero.

    The translated curve can pass through the reversal point only if the imbibition curve reaches
    the drainage value there. Where it tops out lower, the scanning curve is lifted by the factor
    that puts its top on the drainage value, so it still starts on the drainage curve and kr is
    continuous at the reversal.

    The saturation history is passed as keyword arguments, as for the Killough model. A system
    without history, or without an imbibition table, stays on its drainage curves. Oil in an
    oil-wet system is not supported: that needs water as the scanning phase of the oil-water
    system, so an oil-water imbibition table is rejected there unless it only serves the wetting
    phase.
    """

    wetting_phase_uses_imbibition: Boolean = False
    """
    Whether the wetting phase takes its imbibition curve rather than its drainage curve, as
    `EHYSTR` codes 1 and 3 ask. It applies to water in the oil-water system and to oil in the
    gas-oil system.
    """

    gas_shift_axis: NumberArray[OneDimension] = attrs.field(init=False, repr=False, eq=False)
    """Gas saturations at which the translation of the gas imbibition curve is tabulated."""

    gas_shift: NumberArray[OneDimension] = attrs.field(init=False, repr=False, eq=False)
    """Translation `Δ` of the gas imbibition curve at each of `gas_shift_axis`."""

    gas_scale: NumberArray[OneDimension] = attrs.field(init=False, repr=False, eq=False)
    """Vertical factor of the gas imbibition curve at each of `gas_shift_axis`."""

    oil_shift_axis: NumberArray[OneDimension] = attrs.field(init=False, repr=False, eq=False)
    """Oil saturations at which the translation of the oil-water imbibition curve is tabulated."""

    oil_shift: NumberArray[OneDimension] = attrs.field(init=False, repr=False, eq=False)
    """Translation `Δ` of the oil-water imbibition curve at each of `oil_shift_axis`."""

    oil_scale: NumberArray[OneDimension] = attrs.field(init=False, repr=False, eq=False)
    """Vertical factor of the oil-water imbibition curve at each of `oil_shift_axis`."""

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if (
            self.oil_water_imbibition_table is not None
            and not self.wetting_phase_uses_imbibition
            and self.get_oil_water_wetting_phase() != FluidPhase.WATER
        ):
            raise ValidationError(
                "Carlson scanning of water as the non-wetting phase of an oil-wet oil-water "
                "system is not supported. Give no `oil_water_imbibition_table`, or set "
                "`wetting_phase_uses_imbibition` so that it serves the wetting phase."
            )

        for prefix, tabulated in (
            ("gas", self._tabulate_gas_translation()),
            ("oil", self._tabulate_oil_translation()),
        ):
            axis, shift, scale = tabulated
            object.__setattr__(self, f"{prefix}_shift_axis", axis)
            object.__setattr__(self, f"{prefix}_shift", shift)
            object.__setattr__(self, f"{prefix}_scale", scale)

    @staticmethod
    def _no_translation() -> tuple[
        NumberArray[OneDimension], NumberArray[OneDimension], NumberArray[OneDimension]
    ]:
        """Returns the translation of a phase that does not scan: none, at either end of its axis."""
        return (
            typing.cast("NumberArray[OneDimension]", np.array([0.0, 1.0])),
            typing.cast("NumberArray[OneDimension]", np.zeros(2)),
            typing.cast("NumberArray[OneDimension]", np.ones(2)),
        )

    def _tabulate_gas_translation(
        self,
    ) -> tuple[NumberArray[OneDimension], NumberArray[OneDimension], NumberArray[OneDimension]]:
        """Tabulates the translation of the gas imbibition curve, or none when there is none."""
        imbibition_table = self.gas_oil_imbibition_table
        if imbibition_table is None or self.get_gas_oil_wetting_phase() == FluidPhase.GAS:
            return self._no_translation()

        connate_water = self.get_connate_water_saturation()
        axis = typing.cast(
            "NumberArray[OneDimension]", np.linspace(0.0, 1.0 - connate_water, SHIFT_CURVE_POINTS)
        )
        arguments = dict(
            water_saturation=np.full_like(axis, connate_water),
            oil_saturation=1.0 - connate_water - axis,
            gas_saturation=axis,
            connate_water_saturation=connate_water,
            with_derivatives=False,
        )
        _, drainage_krg, _, _ = query_gas_oil_curves(self.gas_oil_drainage_table, **arguments)  # type: ignore[arg-type]
        _, imbibition_krg, _, _ = query_gas_oil_curves(imbibition_table, **arguments)  # type: ignore[arg-type]
        shift, scale = tabulate_translation(
            axis=axis,
            drainage_values=typing.cast("NumberArray[OneDimension]", np.asarray(drainage_krg)),
            imbibition_values=typing.cast("NumberArray[OneDimension]", np.asarray(imbibition_krg)),
        )
        return axis, shift, scale

    def _tabulate_oil_translation(
        self,
    ) -> tuple[NumberArray[OneDimension], NumberArray[OneDimension], NumberArray[OneDimension]]:
        """
        Tabulates the translation of the oil-water imbibition curve along the oil saturation, or
        none when oil is not the non-wetting phase or there is no imbibition curve.
        """
        imbibition_table = self.oil_water_imbibition_table
        if imbibition_table is None or self.get_oil_water_wetting_phase() != FluidPhase.WATER:
            return self._no_translation()

        connate_water = self.get_connate_water_saturation()
        axis = typing.cast(
            "NumberArray[OneDimension]", np.linspace(0.0, 1.0 - connate_water, SHIFT_CURVE_POINTS)
        )
        arguments = dict(
            water_saturation=1.0 - axis,
            oil_saturation=axis,
            gas_saturation=np.zeros_like(axis),
            with_derivatives=False,
        )
        _, drainage_kro, _, _ = query_oil_water_curves(self.oil_water_drainage_table, **arguments)  # type: ignore[arg-type]
        _, imbibition_kro, _, _ = query_oil_water_curves(imbibition_table, **arguments)  # type: ignore[arg-type]
        shift, scale = tabulate_translation(
            axis=axis,
            drainage_values=typing.cast("NumberArray[OneDimension]", np.asarray(drainage_kro)),
            imbibition_values=typing.cast("NumberArray[OneDimension]", np.asarray(imbibition_kro)),
        )
        return axis, shift, scale

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
        Returns relative permeabilities, and optionally their nine derivatives, in one pass.

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
        :param min_gas_saturation: Historical minimum gas saturation per cell. Not used by this
            model.
        :param min_water_saturation: Historical minimum water saturation per cell, where oil, the
            non-wetting phase of a water-wet system, peaked.
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
        # Only a non-wetting phase scans, and only when there is an imbibition curve to translate.
        gas_table = self.gas_oil_imbibition_table
        water_table = self.oil_water_imbibition_table
        use_gas_hysteresis = (
            inputs.use_gas_hysteresis and gas_table is not None and inputs.gas_is_non_wetting
        )
        oil_is_non_wetting = self.get_oil_water_wetting_phase() == FluidPhase.WATER
        use_oil_hysteresis = (
            inputs.use_water_hysteresis
            and water_table is not None
            and oil_is_non_wetting
            and min_water_saturation is not None
        )
        inputs = inputs._replace(
            use_water_hysteresis=use_oil_hysteresis, use_gas_hysteresis=use_gas_hysteresis
        )
        if use_oil_hysteresis:
            # The step weight of the water side compares water with its minimum.
            history = history._replace(water_reversal_saturation=inputs.minimum_water)

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

        # The wetting phases have no scanning curve, only a choice of curve.
        if self.wetting_phase_uses_imbibition:
            if water_table is not None and oil_is_non_wetting:
                krw_d, _, dkrw_d, _ = query_oil_water_curves(
                    water_table, **curve_arguments, **kwargs
                )
            if gas_table is not None and self.get_gas_oil_wetting_phase() != FluidPhase.GAS:
                kro_gas_d, _, dkro_gas_d, _ = query_gas_oil_curves(
                    gas_table, **gas_oil_arguments, **kwargs
                )
        krw_i, kro_water_i, dkrw_i, dkro_water_i = krw_d, kro_water_d, dkrw_d, dkro_water_d
        kro_gas_i, krg_i, dkro_gas_i, dkrg_i = kro_gas_d, krg_d, dkro_gas_d, dkrg_d

        if use_oil_hysteresis:
            translated_water = np.empty_like(sw)
            oil_factor = np.empty_like(sw)
            translate_scanning_saturation(
                size=size,
                saturation=sw,
                reversal_saturation=inputs.minimum_water,
                shift_axis=self.oil_shift_axis,
                shift=self.oil_shift,
                vertical_scale=self.oil_scale,
                reflect=True,
                out_saturation=translated_water,
                out_vertical_scale=oil_factor,
            )
            assert water_table is not None
            _, kro_water_i, _, dkro_water_i = query_oil_water_curves(
                water_table,
                **{
                    **curve_arguments,
                    "water_saturation": translated_water,
                    "oil_saturation": 1.0 - translated_water - sg,
                },
                **kwargs,
            )
            kro_water_i = kro_water_i * oil_factor
            if with_derivatives:
                assert dkro_water_i is not None
                dkro_water_i = typing.cast(
                    "NumberOrArray[OneDimension]", dkro_water_i * oil_factor
                )

        if use_gas_hysteresis:
            translated_gas = np.empty_like(sg)
            gas_factor = np.empty_like(sg)
            translate_scanning_saturation(
                size=size,
                saturation=sg,
                reversal_saturation=inputs.gas_reversal,
                shift_axis=self.gas_shift_axis,
                shift=self.gas_shift,
                vertical_scale=self.gas_scale,
                reflect=False,
                out_saturation=translated_gas,
                out_vertical_scale=gas_factor,
            )
            assert gas_table is not None
            _, krg_i, _, dkrg_i = query_gas_oil_curves(
                gas_table,
                **{
                    **gas_oil_arguments,
                    "gas_saturation": translated_gas,
                    "oil_saturation": 1.0 - sw - translated_gas,
                },
                **kwargs,
            )
            krg_i = krg_i * gas_factor
            if with_derivatives:
                assert dkrg_i is not None
                dkrg_i = typing.cast("NumberOrArray[OneDimension]", dkrg_i * gas_factor)

        curves = BoundingCurves(
            krw_d=typing.cast("NumberOrArray[OneDimension]", krw_d),
            krw_i=typing.cast("NumberOrArray[OneDimension]", krw_i),
            kro_water_d=typing.cast("NumberOrArray[OneDimension]", kro_water_d),
            kro_water_i=typing.cast("NumberOrArray[OneDimension]", kro_water_i),
            kro_gas_d=typing.cast("NumberOrArray[OneDimension]", kro_gas_d),
            kro_gas_i=typing.cast("NumberOrArray[OneDimension]", kro_gas_i),
            krg_d=typing.cast("NumberOrArray[OneDimension]", krg_d),
            krg_i=typing.cast("NumberOrArray[OneDimension]", krg_i),
            dkrw_d=typing.cast("NumberOrArray[OneDimension] | None", dkrw_d),
            dkrw_i=typing.cast("NumberOrArray[OneDimension] | None", dkrw_i),
            dkro_water_d=typing.cast("NumberOrArray[OneDimension] | None", dkro_water_d),
            dkro_water_i=typing.cast("NumberOrArray[OneDimension] | None", dkro_water_i),
            dkro_gas_d=typing.cast("NumberOrArray[OneDimension] | None", dkro_gas_d),
            dkro_gas_i=typing.cast("NumberOrArray[OneDimension] | None", dkro_gas_i),
            dkrg_d=typing.cast("NumberOrArray[OneDimension] | None", dkrg_d),
            dkrg_i=typing.cast("NumberOrArray[OneDimension] | None", dkrg_i),
        )
        return self.scan_bounding_curves(
            inputs=inputs,
            history=history,
            curves=curves,
            scanning_exponent=1.0,
            gas_translated_scanning=True,
            water_translated_scanning=True,
            with_derivatives=with_derivatives,
        )

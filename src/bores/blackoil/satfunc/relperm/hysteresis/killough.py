"""
Killough (1976) scanning curves with Land (1968) trapping.

- Killough, J.E. (1976). "Reservoir Simulation With History-Dependent Saturation Functions".
  SPE 5106.
"""

import typing

import attrs
import numpy as np

from bores.blackoil.satfunc.relperm.hysteresis.base import (
    BoundingCurves,
    HysteresisRelPermTable,
    query_gas_oil_curves,
    query_oil_water_curves,
    uses_tabular_endpoints,
)
from bores.blackoil.satfunc.relperm.hysteresis.land import rescale_scanning_saturation
from bores.blackoil.satfunc.relperm.tables import (
    resolve_residual_oil_saturation_gas,
)
from bores.types import (
    Boolean,
    BooleanArray,
    FluidPhase,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
    RelativePermeabilities,
    RelativePermeabilityDerivatives,
)

__all__ = ["KilloughLandRelPermTable"]


@attrs.frozen(slots=True)
class KilloughLandRelPermTable(HysteresisRelPermTable):
    """
    Implements the Killough relative permeability hysteresis model/table with Land trapping.

    During *primary drainage* the relative permeabilities follow the
    `oil_water_drainage_table` and `gas_oil_drainage_table`.

    When the flow reverses (*imbibition*) two things happen simultaneously.

    **Land trapping**: a portion of the non-wetting phase becomes
    disconnected. The dynamic residual saturation depends on the saturation
    at the reversal point via Land's formula:

        S_r(S_i) = S_i / (1 + C * S_i),    C = 1 / S_r_max - 1 / S_max

    The dynamic residual is handed to the imbibition table as an endpoint override, so it
    changes the curve of an analytical (parametric) imbibition table. A tabular table has
    fixed endpoints, so its saturation axis is rescaled instead, until the curve ends at the
    trapped saturation (see `tabular_land_trapping`).

    **Killough scanning curves**: between the reversal point and the
    maximum historical saturation, kr follows a scanning curve that
    interpolates between the primary drainage and imbibition bounds.

    Both two-phase (`TwoPhaseRelPermTable`) and full three-phase
    (`RelativePermeabilityTable`) backing tables are supported. Wetting
    and non-wetting phase roles are resolved through the canonical API
    (`get_oil_water_wetting_phase` / `get_gas_oil_wetting_phase`) so the
    model is wettability-agnostic.

    The saturation history is passed as keyword arguments to `evaluate`, `derivatives` and
    `evaluate_with_derivatives`. A sub-system (oil-water or gas-oil) whose history is not
    supplied stays on its primary drainage curves, which is the correct behaviour for the first
    drainage cycle. Imbibition curves are only evaluated for a sub-system that has history and an
    imbibition curve that differs from its drainage curve.
    """

    __type__ = "killough_land_relperm_model"

    scanning_interpolation_exponent: Number = 1.0
    """Killough scanning curve interpolation exponent *n* (1 = linear)."""

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

        This is the one implementation behind `evaluate`, `derivatives` and
        `evaluate_with_derivatives`. Each bounding curve is queried once, the scanning weights
        are shared between the curves that scan over the same saturation, and the mixing rule
        and its derivatives are applied in the same loop for the built-in rules.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param with_derivatives: Whether to also compute the derivatives.
        :param max_water_saturation: Historical maximum water saturation per cell.
        :param max_gas_saturation: Historical maximum gas saturation per cell.
        :param water_imbibition_flag: Per-cell flag, true while water saturation increases.
        :param gas_imbibition_flag: Per-cell flag, true while gas saturation increases.
        :param water_reversal_saturation: Water saturation at the last reversal. Defaults to
            `max_water_saturation`.
        :param gas_reversal_saturation: Gas saturation at the last reversal. Defaults to
            `max_gas_saturation`.
        :param kwargs: Extra arguments forwarded to the backing tables, such as residual
            saturation overrides for analytical models.
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
        water_reversal, gas_reversal = inputs.water_reversal, inputs.gas_reversal
        use_water_hysteresis = inputs.use_water_hysteresis
        use_gas_hysteresis = inputs.use_gas_hysteresis
        gas_is_non_wetting = inputs.gas_is_non_wetting
        # Bounding curves
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

        krw_i, kro_water_i, dkrw_i, dkro_water_i = krw_d, kro_water_d, dkrw_d, dkro_water_d
        if use_water_hysteresis:
            imbibition_table = self.oil_water_imbibition_table
            water_table = imbibition_table or drainage_water_system
            land_kwargs = self.land_override_kwargs_water(
                size=size,
                gas_saturation=sg,
                water_reversal=typing.cast("NumberArray[OneDimension]", water_reversal),
                table=water_table,
                kwargs=kwargs,
            )
            water_arguments = curve_arguments
            water_scale = None
            if (
                self.tabular_land_trapping
                and uses_tabular_endpoints(water_table)
                and self.get_oil_water_wetting_phase() == FluidPhase.WATER
            ):
                water_residual = self.land_residual_oil_water(
                    size=size,
                    gas_saturation=sg,
                    water_reversal=typing.cast("NumberArray[OneDimension]", water_reversal),
                )
                if water_residual is not None:
                    rescaled_water = np.empty_like(sw)
                    water_scale = np.empty_like(sw)
                    rescale_scanning_saturation(
                        size=size,
                        saturation=sw,
                        reversal_saturation=typing.cast(
                            "NumberArray[OneDimension]", water_reversal
                        ),
                        trapped_saturation=water_residual,
                        on_wetting_axis=True,
                        table_start=water_table.get_connate_water_saturation(),
                        table_end=1.0 - water_table.get_residual_oil_saturation_water(),
                        saturation_epsilon=1e-12,
                        out_saturation=rescaled_water,
                        out_scale=water_scale,
                    )
                    water_arguments = {
                        **curve_arguments,
                        "water_saturation": rescaled_water,
                        "oil_saturation": 1.0 - rescaled_water - sg,
                    }
            if imbibition_table is not None and imbibition_table is not drainage_water_system:
                krw_i, kro_water_i, dkrw_i, dkro_water_i = query_oil_water_curves(
                    imbibition_table, **water_arguments, **land_kwargs
                )
            elif land_kwargs is not kwargs or water_scale is not None:
                krw_i, kro_water_i, dkrw_i, dkro_water_i = query_oil_water_curves(
                    drainage_water_system, **water_arguments, **land_kwargs
                )
            if water_scale is not None and with_derivatives:
                dkrw_i = dkrw_i * water_scale
                dkro_water_i = dkro_water_i * water_scale

        kro_gas_i, krg_i, dkro_gas_i, dkrg_i = kro_gas_d, krg_d, dkro_gas_d, dkrg_d
        if use_gas_hysteresis:
            imbibition_table = self.gas_oil_imbibition_table
            gas_table = imbibition_table or drainage_gas_system
            land_kwargs = self.land_override_kwargs_gas(
                size=size,
                water_saturation=sw,
                gas_reversal=typing.cast("NumberArray[OneDimension]", gas_reversal),
                table=gas_table,
                kwargs=kwargs,
            )
            gas_residual = None
            if not uses_tabular_endpoints(gas_table) or self.tabular_land_trapping:
                gas_residual = self.land_residual_gas(
                    size=size,
                    water_saturation=sw,
                    gas_reversal=typing.cast("NumberArray[OneDimension]", gas_reversal),
                )
            gas_arguments = gas_oil_arguments
            gas_scale = None
            if gas_residual is not None and uses_tabular_endpoints(gas_table):
                rescaled_gas = np.empty_like(sg)
                gas_scale = np.empty_like(sg)
                connate_water = self.get_connate_water_saturation()
                rescale_scanning_saturation(
                    size=size,
                    saturation=sg,
                    reversal_saturation=typing.cast("NumberArray[OneDimension]", gas_reversal),
                    trapped_saturation=gas_residual,
                    on_wetting_axis=False,
                    table_start=gas_table.get_residual_gas_saturation(),
                    table_end=1.0
                    - connate_water
                    - resolve_residual_oil_saturation_gas(
                        gas_oil_table=gas_table, connate_water_saturation=connate_water
                    ),
                    saturation_epsilon=1e-12,
                    out_saturation=rescaled_gas,
                    out_scale=gas_scale,
                )
                gas_arguments = {
                    **gas_oil_arguments,
                    "gas_saturation": rescaled_gas,
                    "oil_saturation": 1.0 - sw - rescaled_gas,
                }
            if gas_is_non_wetting:
                if gas_residual is not None:
                    trapped_gas = gas_residual
                else:
                    trapped_gas = np.full(
                        1, gas_table.get_residual_gas_saturation(), dtype=sw.dtype
                    )
                history = history._replace(
                    gas_trapped_saturation=typing.cast("NumberArray[OneDimension]", trapped_gas)
                )
            if imbibition_table is not None and imbibition_table is not drainage_gas_system:
                kro_gas_i, krg_i, dkro_gas_i, dkrg_i = query_gas_oil_curves(
                    imbibition_table, **gas_arguments, **land_kwargs
                )
            elif land_kwargs is not kwargs or gas_scale is not None:
                kro_gas_i, krg_i, dkro_gas_i, dkrg_i = query_gas_oil_curves(
                    drainage_gas_system, **gas_arguments, **land_kwargs
                )
            if gas_scale is not None and with_derivatives:
                dkro_gas_i = dkro_gas_i * gas_scale
                dkrg_i = dkrg_i * gas_scale

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
            scanning_exponent=self.scanning_interpolation_exponent,
            gas_translated_scanning=False,
            with_derivatives=with_derivatives,
        )

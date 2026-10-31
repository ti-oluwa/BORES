"""
Killough (1976) scanning curves with Land (1968) trapping.

- Killough, J.E. (1976). "Reservoir Simulation With History-Dependent Saturation Functions".
  SPE 5106.
"""

import math
import typing

import attrs
import numpy as np

from bores.blackoil.satfunc.relperm.hysteresis.base import (
    HysteresisRelPermTable,
    query_gas_oil_curves,
    query_oil_water_curves,
)
from bores.blackoil.satfunc.relperm.hysteresis.kernels import (
    CurveSet,
    HistoryArrays,
    ScanOutputs,
    normalise_saturations,
    normalise_saturations_parallel,
    saturations_need_normalising,
    scan_relative_permeabilities,
    scan_relative_permeabilities_parallel,
)
from bores.blackoil.satfunc.relperm.mixing_rules import get_builtin_rule_id
from bores.blackoil.satfunc.relperm.tables import (
    MixingRule,
    get_mixing_rule_partial_derivatives,
)
from bores.blackoil.satfunc.utils import (
    PARALLEL_SIZE_THRESHOLD,
    flatten_kernel_input,
)
from bores.constants import c
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
    fixed endpoints and ignores it.

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
        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )
        use_water_hysteresis = (
            max_water_saturation is not None and water_imbibition_flag is not None
        )
        use_gas_hysteresis = max_gas_saturation is not None and gas_imbibition_flag is not None

        inputs = [water_saturation, oil_saturation, gas_saturation]
        if use_water_hysteresis:
            inputs += [max_water_saturation, water_imbibition_flag]
            if water_reversal_saturation is not None:
                inputs.append(water_reversal_saturation)
        if use_gas_hysteresis:
            inputs += [max_gas_saturation, gas_imbibition_flag]
            if gas_reversal_saturation is not None:
                inputs.append(gas_reversal_saturation)

        shape = np.broadcast_shapes(*(np.shape(value) for value in inputs))
        flat_shape = shape or (1,)
        size = math.prod(flat_shape)

        sw = flatten_kernel_input(value=water_saturation, shape=flat_shape)
        so = flatten_kernel_input(value=oil_saturation, shape=flat_shape)
        sg = flatten_kernel_input(value=gas_saturation, shape=flat_shape)
        saturation_epsilon = c.SATURATION_EPSILON
        if saturations_need_normalising(
            size=size,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            saturation_epsilon=saturation_epsilon,
        ):
            normalise = (
                normalise_saturations_parallel
                if size >= PARALLEL_SIZE_THRESHOLD
                else normalise_saturations
            )
            sw, so, sg = normalise(
                size=size,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                saturation_epsilon=saturation_epsilon,
            )

        placeholder = np.zeros(1, dtype=sw.dtype)
        no_flag = np.zeros(1, dtype=np.bool_)
        if use_water_hysteresis:
            maximum_water = flatten_kernel_input(value=max_water_saturation, shape=flat_shape)
            water_flag = flatten_kernel_input(
                value=water_imbibition_flag, shape=flat_shape, as_float=False
            )
            water_reversal = (
                maximum_water
                if water_reversal_saturation is None
                else flatten_kernel_input(value=water_reversal_saturation, shape=flat_shape)
            )
        else:
            maximum_water, water_flag, water_reversal = placeholder, no_flag, placeholder
        if use_gas_hysteresis:
            maximum_gas = flatten_kernel_input(value=max_gas_saturation, shape=flat_shape)
            gas_flag = flatten_kernel_input(
                value=gas_imbibition_flag, shape=flat_shape, as_float=False
            )
            gas_reversal = (
                maximum_gas
                if gas_reversal_saturation is None
                else flatten_kernel_input(value=gas_reversal_saturation, shape=flat_shape)
            )
        else:
            maximum_gas, gas_flag, gas_reversal = placeholder, no_flag, placeholder

        history = HistoryArrays(
            max_water_saturation=typing.cast("NumberArray[OneDimension]", maximum_water),
            max_gas_saturation=typing.cast("NumberArray[OneDimension]", maximum_gas),
            water_imbibition_flag=typing.cast("NumberArray[OneDimension]", water_flag),
            gas_imbibition_flag=typing.cast("NumberArray[OneDimension]", gas_flag),
            water_reversal_saturation=typing.cast("NumberArray[OneDimension]", water_reversal),
            gas_reversal_saturation=typing.cast("NumberArray[OneDimension]", gas_reversal),
            gas_trapped_saturation=typing.cast("NumberArray[OneDimension]", placeholder),
        )
        gas_is_non_wetting = self.get_gas_oil_wetting_phase() != FluidPhase.GAS

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
            land_kwargs = self.land_override_kwargs_water(
                size=size,
                gas_saturation=sg,
                water_reversal=typing.cast("NumberArray[OneDimension]", water_reversal),
                table=imbibition_table or drainage_water_system,
                kwargs=kwargs,
            )
            if imbibition_table is not None and imbibition_table is not drainage_water_system:
                krw_i, kro_water_i, dkrw_i, dkro_water_i = query_oil_water_curves(
                    imbibition_table, **curve_arguments, **land_kwargs
                )
            elif land_kwargs is not kwargs:
                krw_i, kro_water_i, dkrw_i, dkro_water_i = query_oil_water_curves(
                    drainage_water_system, **curve_arguments, **land_kwargs
                )

        kro_gas_i, krg_i, dkro_gas_i, dkrg_i = kro_gas_d, krg_d, dkro_gas_d, dkrg_d
        if use_gas_hysteresis:
            imbibition_table = self.gas_oil_imbibition_table
            land_kwargs = self.land_override_kwargs_gas(
                size=size,
                water_saturation=sw,
                gas_reversal=typing.cast("NumberArray[OneDimension]", gas_reversal),
                table=imbibition_table or drainage_gas_system,
                kwargs=kwargs,
            )
            if gas_is_non_wetting:
                if "residual_gas_saturation" in land_kwargs:
                    trapped_gas = land_kwargs["residual_gas_saturation"]
                else:
                    trapped_gas = np.full(
                        1,
                        (imbibition_table or drainage_gas_system).get_residual_gas_saturation(),
                        dtype=sw.dtype,
                    )
                history = history._replace(
                    gas_trapped_saturation=typing.cast("NumberArray[OneDimension]", trapped_gas)
                )
            if imbibition_table is not None and imbibition_table is not drainage_gas_system:
                kro_gas_i, krg_i, dkro_gas_i, dkrg_i = query_gas_oil_curves(
                    imbibition_table, **gas_oil_arguments, **land_kwargs
                )
            elif land_kwargs is not kwargs:
                kro_gas_i, krg_i, dkro_gas_i, dkrg_i = query_gas_oil_curves(
                    drainage_gas_system, **gas_oil_arguments, **land_kwargs
                )

        values = CurveSet(
            krw_drainage=krw_d,
            krw_imbibition=krw_i,
            kro_water_drainage=kro_water_d,
            kro_water_imbibition=kro_water_i,
            kro_gas_drainage=kro_gas_d,
            kro_gas_imbibition=kro_gas_i,
            krg_drainage=krg_d,
            krg_imbibition=krg_i,
        )
        slopes = CurveSet(
            krw_drainage=typing.cast(
                "NumberArray[OneDimension]", dkrw_d if with_derivatives else placeholder
            ),
            krw_imbibition=typing.cast(
                "NumberArray[OneDimension]", dkrw_i if with_derivatives else placeholder
            ),
            kro_water_drainage=typing.cast(
                "NumberArray[OneDimension]", dkro_water_d if with_derivatives else placeholder
            ),
            kro_water_imbibition=typing.cast(
                "NumberArray[OneDimension]", dkro_water_i if with_derivatives else placeholder
            ),
            kro_gas_drainage=typing.cast(
                "NumberArray[OneDimension]", dkro_gas_d if with_derivatives else placeholder
            ),
            kro_gas_imbibition=typing.cast(
                "NumberArray[OneDimension]", dkro_gas_i if with_derivatives else placeholder
            ),
            krg_drainage=typing.cast(
                "NumberArray[OneDimension]", dkrg_d if with_derivatives else placeholder
            ),
            krg_imbibition=typing.cast(
                "NumberArray[OneDimension]", dkrg_i if with_derivatives else placeholder
            ),
        )
        values = CurveSet(
            *(
                typing.cast("NumberArray[OneDimension]", np.asarray(array).reshape(-1))
                for array in values
            )
        )
        slopes = CurveSet(
            *(
                typing.cast("NumberArray[OneDimension]", np.asarray(array).reshape(-1))
                for array in slopes
            )
        )

        dtype = values.krw_drainage.dtype
        if dtype.kind != "f":
            dtype = np.dtype(np.float64)
        rule = typing.cast(MixingRule, self.mixing_rule)
        rule_id = get_builtin_rule_id(rule)
        use_rule_in_kernel = rule_id is not None
        needs_gas_oil = not use_rule_in_kernel
        out = ScanOutputs(
            krw=typing.cast("NumberArray[OneDimension]", np.empty(size, dtype=dtype)),
            kro=typing.cast("NumberArray[OneDimension]", np.empty(size, dtype=dtype)),
            krg=typing.cast("NumberArray[OneDimension]", np.empty(size, dtype=dtype)),
            kro_gas=typing.cast(
                "NumberArray[OneDimension]", np.empty(size if needs_gas_oil else 1, dtype=dtype)
            ),
            dkrw_dsw=typing.cast(
                "NumberArray[OneDimension]", np.empty(size if with_derivatives else 1, dtype=dtype)
            ),
            dkro_dsw=typing.cast(
                "NumberArray[OneDimension]", np.empty(size if with_derivatives else 1, dtype=dtype)
            ),
            dkro_dso=typing.cast(
                "NumberArray[OneDimension]",
                np.empty(size if with_derivatives and use_rule_in_kernel else 1, dtype=dtype),
            ),
            dkro_dsg=typing.cast(
                "NumberArray[OneDimension]", np.empty(size if with_derivatives else 1, dtype=dtype)
            ),
            dkrg_dsg=typing.cast(
                "NumberArray[OneDimension]", np.empty(size if with_derivatives else 1, dtype=dtype)
            ),
        )
        kr_max = self.get_oil_relperm_endpoint()
        irreducible_water_saturation = self.get_connate_water_saturation()
        residual_oil_saturation_water = self.get_residual_oil_saturation_water()
        residual_oil_saturation_gas = self.get_residual_oil_saturation_gas()
        scan = (
            scan_relative_permeabilities_parallel
            if size >= PARALLEL_SIZE_THRESHOLD
            else scan_relative_permeabilities
        )
        scan(
            size=size,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            history=history,
            use_water_hysteresis=use_water_hysteresis,
            use_gas_hysteresis=use_gas_hysteresis,
            gas_is_non_wetting=gas_is_non_wetting,
            scanning_exponent=self.scanning_interpolation_exponent,
            numerical_epsilon=1e-12,
            values=values,
            slopes=slopes,
            with_derivatives=with_derivatives,
            rule_id=-1 if rule_id is None else rule_id,
            kr_max=kr_max,
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
            minimum_mobile_pore_space=c.MINIMUM_MOBILE_PORE_SPACE,
            out=out,
        )

        zeros = np.zeros(size, dtype=dtype) if with_derivatives else None
        krw, krg = out.krw, out.krg
        if use_rule_in_kernel:
            kro = out.kro
            dkro_dsw, dkro_dso, dkro_dsg = out.dkro_dsw, out.dkro_dso, out.dkro_dsg
        else:
            kro_water, kro_gas = out.kro, out.kro_gas
            kro = rule(
                kro_w=kro_water,
                kro_g=kro_gas,
                krw=krw,
                krg=krg,
                kr_max=kr_max,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                irreducible_water_saturation=irreducible_water_saturation,
                residual_oil_saturation_water=residual_oil_saturation_water,
                residual_oil_saturation_gas=residual_oil_saturation_gas,
            )
            dkro_dsw = dkro_dso = dkro_dsg = None
            if with_derivatives:
                partials = get_mixing_rule_partial_derivatives(
                    rule=rule,
                    kro_w=kro_water,
                    kro_g=kro_gas,
                    krw=krw,
                    krg=krg,
                    kr_max=kr_max,
                    water_saturation=sw,
                    oil_saturation=so,
                    gas_saturation=sg,
                    irreducible_water_saturation=irreducible_water_saturation,
                    residual_oil_saturation_water=residual_oil_saturation_water,
                    residual_oil_saturation_gas=residual_oil_saturation_gas,
                    epsilon=c.FINITE_DIFFERENCE_EPSILON,
                )
                dkro_dsw = (
                    partials["dkro_dkro_w"] * out.dkro_dsw
                    + partials["dkro_dkrw"] * out.dkrw_dsw
                    + partials["dkro_dsw_explicit"]
                )
                dkro_dso = partials["dkro_dso_explicit"]
                dkro_dsg = (
                    partials["dkro_dkro_g"] * out.dkro_dsg
                    + partials["dkro_dkrg"] * out.dkrg_dsg
                    + partials["dkro_dsg_explicit"]
                )

        def finish(array: typing.Any) -> typing.Any:
            result = np.asarray(array)
            if result.size == 1 and size > 1:
                result = np.broadcast_to(result, (size,))
            result = result.reshape(shape)
            return result.item() if is_scalar else result

        relperms = RelativePermeabilities(
            water=finish(array=krw),
            oil=finish(array=kro),
            gas=finish(array=krg),
        )
        if not with_derivatives:
            return relperms, None
        assert zeros is not None
        zero = finish(array=zeros)
        return relperms, RelativePermeabilityDerivatives(
            dkrw_dsw=finish(array=out.dkrw_dsw),
            dkro_dsw=finish(array=dkro_dsw),
            dkrg_dsw=zero,
            dkrw_dso=zero,
            dkro_dso=finish(array=dkro_dso),
            dkrg_dso=zero,
            dkrw_dsg=zero,
            dkro_dsg=finish(array=dkro_dsg),
            dkrg_dsg=finish(array=out.dkrg_dsg),
        )

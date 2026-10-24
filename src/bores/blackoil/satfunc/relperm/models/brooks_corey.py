"""Brooks-Corey relative permeability model and table."""

import typing

import attrs
import numpy as np

from bores.blackoil.satfunc.relperm.mixing_rules import (
    MixingRule,
    baker_linear_rule,
    get_mixing_rule,
    get_mixing_rule_partial_derivatives,
)
from bores.blackoil.satfunc.relperm.tables import (
    MinimumRelPerm,
    RelativePermeabilityTable,
    clamp_relperm,
    clamp_relperm_derivative,
    resolve_min_relperm,
    show_invalid_saturation,
)
from bores.constants import c
from bores.errors import ValidationError
from bores.types import (
    FluidPhase,
    NDimension,
    Number,
    NumberOrArray,
    RelativePermeabilities,
    RelativePermeabilityDerivatives,
    Wettability,
)

__all__ = [
    "BrooksCoreyRelPermTable",
    "compute_brookes_corey_relative_permeabilities",
]


def compute_brookes_corey_relative_permeabilities(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    water_exponent: Number,
    oil_exponent: Number,
    gas_exponent: Number,
    maximum_water_relperm: Number = 1.0,
    maximum_oil_relperm: Number = 1.0,
    maximum_gas_relperm: Number = 1.0,
    wettability: Wettability = Wettability.WATER_WET,
    mixed_wet_water_fraction: Number = 0.5,
    mixing_rule: MixingRule = baker_linear_rule,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
    minimum_water_relperm: Number | None = None,
    minimum_oil_relperm: Number | None = None,
    minimum_gas_relperm: Number | None = None,
) -> tuple[NumberOrArray[NDimension], NumberOrArray[NDimension], NumberOrArray[NDimension]]:
    """
    Computes relative permeability for water, oil, and gas in a three-phase system.
    Supports water-wet and oil-wet wettability assumptions.

    Uses Corey-type models for krw, krg, and Stone I rule for kro.

    Supports both scalar and array inputs for saturations.

    :param water_saturation: Current water saturation (fraction, between 0 and 1) - scalar or array.
    :param oil_saturation: Current oil saturation (fraction, between 0 and 1) - scalar or array.
    :param gas_saturation: Current gas saturation (fraction, between 0 and 1) - scalar or array.
    :param irreducible_water_saturation: Irreducible water saturation (swc).
    :param residual_oil_saturation_water: Residual oil saturation after water flood (sorw).
    :param residual_oil_saturation_gas: Residual oil saturation after gas flood (sorg).
    :param residual_gas_saturation: Residual gas saturation (sgr).
    :param water_exponent: Corey exponent for water relative permeability.
    :param oil_exponent: Corey exponent for oil relative permeability (affects Stone I blending).
    :param gas_exponent: Corey exponent for gas relative permeability.
    :param wettability: Wettability type (water-wet or oil-wet).
    :param mixed_wet_water_fraction: Fraction of pore space considered water-wet in mixed-wet systems (0 to 1).
    :param mixing_rule: Mixing rule function for three-phase oil relative permeability.
    :param saturation_epsilon: Tolerance for checking if saturations sum to 1.
    :param minimum_mobile_pore_space: Minimum mobile pore space to avoid division by zero in effective saturation calculations.
    :param minimum_water_relperm: Resolved minimum min_value for water kr (`None` = no min_value).
    :param minimum_oil_relperm: Resolved minimum min_value for oil kr (`None` = no min_value).
    :param minimum_gas_relperm: Resolved minimum min_value for gas kr (`None` = no min_value).
    :return: Tuple of (water_relative_permeability, oil_relative_permeability, gas_relative_permeability)
    """
    # Convert to arrays for vectorized operations
    sw = np.atleast_1d(water_saturation)
    so = np.atleast_1d(oil_saturation)
    sg = np.atleast_1d(gas_saturation)
    is_scalar = (
        np.isscalar(water_saturation)
        and np.isscalar(oil_saturation)
        and np.isscalar(gas_saturation)
        and np.isscalar(irreducible_water_saturation)
        and np.isscalar(residual_oil_saturation_water)
        and np.isscalar(residual_oil_saturation_gas)
        and np.isscalar(residual_gas_saturation)
    )

    # Broadcast all arrays to same shape
    sw, so, sg = np.broadcast_arrays(sw, so, sg)

    # Validate saturations
    if np.any((sw < 0) | (sw > 1) | (so < 0) | (so > 1) | (sg < 0) | (sg > 1)):
        raise ValidationError(
            f"Saturations must be between 0 and 1. Sw: {show_invalid_saturation(sw)}, So: {show_invalid_saturation(so)}, Sg: {show_invalid_saturation(sg)}"
        )

    # Normalize saturations if they do not sum to 1
    total_saturation = sw + so + sg
    needs_norm = (np.abs(total_saturation - 1.0) > saturation_epsilon) & (total_saturation > 0.0)
    if np.any(needs_norm):
        sw = np.where(needs_norm, sw / total_saturation, sw)
        so = np.where(needs_norm, so / total_saturation, so)
        sg = np.where(needs_norm, sg / total_saturation, sg)

    if wettability == Wettability.WATER_WET:
        # 1. Water relperm (wetting phase)
        movable_water_range = (
            1.0 - irreducible_water_saturation - residual_oil_saturation_water  # type: ignore[operator]
        )
        effective_water_saturation = np.where(
            movable_water_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - irreducible_water_saturation) / movable_water_range, 0.0, 1.0),
        )
        krw = maximum_water_relperm * effective_water_saturation**water_exponent

        # 2. Gas relperm (nonwetting)
        movable_gas_range = (  # type: ignore[operator]
            1.0
            - irreducible_water_saturation  # type: ignore[operator]
            - residual_gas_saturation  # type: ignore[operator]
            - residual_oil_saturation_gas
        )
        effective_gas_saturation = np.where(
            movable_gas_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - residual_gas_saturation) / movable_gas_range, 0.0, 1.0),
        )
        krg = maximum_gas_relperm * effective_gas_saturation**gas_exponent

        # 3. Oil relperm (intermediate phase) - mixing rule blending
        # Make sure to apply oil curvature to the two-phase oil kr inputs before mixing,
        # not to the mixed output. (1-krw/krw_max) and (1-krg/krg_max) are the two-phase
        # oil kr approximations; so we shape them with `oil_exponent` before blending.
        kro_w_shaped = (1.0 - krw / maximum_water_relperm) ** oil_exponent
        kro_g_shaped = (1.0 - krg / maximum_gas_relperm) ** oil_exponent
        kro = maximum_oil_relperm * np.clip(
            mixing_rule(
                kro_w=kro_w_shaped,
                kro_g=kro_g_shaped,
                krw=krw,
                krg=krg,
                kr_max=1.0,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                irreducible_water_saturation=irreducible_water_saturation,  # type: ignore[arg-type]
                residual_oil_saturation_water=residual_oil_saturation_water,  # type: ignore[arg-type]
                residual_oil_saturation_gas=residual_oil_saturation_gas,  # type: ignore[arg-type]
            ),
            0.0,
            1.0,
        )

    elif wettability == Wettability.OIL_WET:
        # Oil is wetting, water becomes intermediate
        # 1. Oil relperm (wetting phase)
        movable_oil_range = (
            1.0 - residual_oil_saturation_water - residual_oil_saturation_gas  # type: ignore[operator]
        )
        max_residual = np.minimum(residual_oil_saturation_water, residual_oil_saturation_gas)
        effective_oil_saturation = np.where(
            movable_oil_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - max_residual) / movable_oil_range, 0.0, 1.0),
        )
        kro = maximum_oil_relperm * effective_oil_saturation**oil_exponent

        # 2. Gas relperm (nonwetting phase)
        movable_gas_range = 1.0 - residual_gas_saturation - irreducible_water_saturation  # type: ignore[operator]
        effective_gas_saturation = np.where(
            movable_gas_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - residual_gas_saturation) / movable_gas_range, 0.0, 1.0),
        )
        krg = maximum_gas_relperm * effective_gas_saturation**gas_exponent

        # 3. Water relperm (intermediate phase, use mixing rule style blending)
        kro_proxy_shaped = (1.0 - kro / maximum_oil_relperm) ** water_exponent
        krg_proxy_shaped = (1.0 - krg / maximum_gas_relperm) ** water_exponent
        krw = maximum_water_relperm * np.clip(
            mixing_rule(  # type: ignore[assignment]
                kro_w=kro_proxy_shaped,  # treat oil as wetting
                kro_g=krg_proxy_shaped,  # treat gas as nonwetting
                krw=kro,
                krg=krg,
                kr_max=1.0,
                water_saturation=so,
                oil_saturation=sw,
                gas_saturation=sg,
                irreducible_water_saturation=np.minimum(
                    residual_oil_saturation_water, residual_oil_saturation_gas
                ),
                residual_oil_saturation_water=irreducible_water_saturation,  # type: ignore[arg-type]
                residual_oil_saturation_gas=irreducible_water_saturation,  # type: ignore[arg-type]
            ),
            0.0,
            1.0,
        )

    elif wettability == Wettability.MIXED_WET:
        # Mixed-wet: interpolate between water-wet and oil-wet using mixed_wet_water_fraction.
        # Water-wet contribution
        movable_water_range_ww = (
            1.0 - irreducible_water_saturation - residual_oil_saturation_water  # type: ignore[operator]
        )
        effective_water_saturation_ww = np.where(
            movable_water_range_ww <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - irreducible_water_saturation) / movable_water_range_ww, 0.0, 1.0),
        )
        krw_ww = maximum_water_relperm * effective_water_saturation_ww**water_exponent

        movable_gas_range_ww = (  # type: ignore[operator]
            1.0
            - irreducible_water_saturation  # type: ignore[operator]
            - residual_gas_saturation  # type: ignore[operator]
            - residual_oil_saturation_gas
        )
        effective_gas_saturation_ww = np.where(
            movable_gas_range_ww <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - residual_gas_saturation) / movable_gas_range_ww, 0.0, 1.0),
        )
        krg_ww = maximum_gas_relperm * effective_gas_saturation_ww**gas_exponent
        kro_w_ww = (1.0 - krw_ww / maximum_water_relperm) ** oil_exponent
        kro_g_ww = (1.0 - krg_ww / maximum_gas_relperm) ** oil_exponent
        kro_ww = maximum_oil_relperm * np.clip(
            mixing_rule(
                kro_w=kro_w_ww,
                kro_g=kro_g_ww,
                krw=krw_ww,
                krg=krg_ww,
                kr_max=1.0,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                irreducible_water_saturation=irreducible_water_saturation,  # type: ignore[arg-type]
                residual_oil_saturation_water=residual_oil_saturation_water,  # type: ignore[arg-type]
                residual_oil_saturation_gas=residual_oil_saturation_gas,  # type: ignore[arg-type]
            ),
            0.0,
            1.0,
        )

        # Oil-wet contribution
        movable_oil_range_ow = (
            1.0 - residual_oil_saturation_water - residual_oil_saturation_gas  # type: ignore[operator]
        )
        max_residual_ow = np.minimum(residual_oil_saturation_water, residual_oil_saturation_gas)
        effective_oil_saturation_ow = np.where(
            movable_oil_range_ow <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - max_residual_ow) / movable_oil_range_ow, 0.0, 1.0),
        )
        kro_ow = maximum_oil_relperm * effective_oil_saturation_ow**oil_exponent

        movable_gas_range_ow = (
            1.0 - residual_gas_saturation - irreducible_water_saturation  # type: ignore[operator]
        )
        effective_gas_saturation_ow = np.where(
            movable_gas_range_ow <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - residual_gas_saturation) / movable_gas_range_ow, 0.0, 1.0),
        )
        krg_ow = maximum_gas_relperm * effective_gas_saturation_ow**gas_exponent
        kro_proxy_ow = (1.0 - kro_ow / maximum_oil_relperm) ** water_exponent
        krg_proxy_ow = (1.0 - krg_ow / maximum_gas_relperm) ** water_exponent
        krw_ow = maximum_water_relperm * np.clip(
            mixing_rule(
                kro_w=kro_proxy_ow,
                kro_g=krg_proxy_ow,
                krw=kro_ow,
                krg=krg_ow,
                kr_max=1.0,
                water_saturation=so,
                oil_saturation=sw,
                gas_saturation=sg,
                irreducible_water_saturation=np.minimum(
                    residual_oil_saturation_water, residual_oil_saturation_gas
                ),
                residual_oil_saturation_water=irreducible_water_saturation,  # type: ignore[arg-type]
                residual_oil_saturation_gas=irreducible_water_saturation,  # type: ignore[arg-type]
            ),
            0.0,
            1.0,
        )

        # Weighted blend
        krw = mixed_wet_water_fraction * krw_ww + (1.0 - mixed_wet_water_fraction) * krw_ow
        kro = mixed_wet_water_fraction * kro_ww + (1.0 - mixed_wet_water_fraction) * kro_ow
        krg = mixed_wet_water_fraction * krg_ww + (1.0 - mixed_wet_water_fraction) * krg_ow

    else:
        raise ValidationError(f"Wettability {wettability!r} not implemented.")

    # Clip all results to [0, 1] and apply per-phase minimum
    krw = np.clip(krw, 0.0, 1.0)
    kro = np.clip(kro, 0.0, 1.0)
    krg = np.clip(krg, 0.0, 1.0)
    krw = clamp_relperm(krw, minimum_water_relperm)  # type: ignore[arg-type]
    kro = clamp_relperm(kro, minimum_oil_relperm)  # type: ignore[arg-type]
    krg = clamp_relperm(krg, minimum_gas_relperm)  # type: ignore[arg-type]
    if is_scalar:
        krw = krw.item()  # type: ignore
        kro = kro.item()  # type: ignore
        krg = krg.item()  # type: ignore
    return krw, kro, krg  # type: ignore[return-value]


@attrs.frozen(slots=True)
class BrooksCoreyRelPermTable(RelativePermeabilityTable):
    """
    Implements the Brooks-Corey-type three-phase relative permeability model.

    Supports water-wet and oil-wet wettability assumptions.

    **Minimum relperm min_values** (`minimum_water_relperm`, `minimum_oil_relperm`, `minimum_gas_relperm`):

    `"auto"` - `max(4 * machine_epsilon, 1e-8)` (dtype-aware, same as
    the CMG IMEX/GEM minimum mobility approach). `None` - no min_value
    (default, kr can reach zero exactly). `Number` - explicit user-set
    value.

    The min_value is applied to the computed kr value and the derivative is
    zeroed out in the min_value region, so the Jacobian is always consistent
    with the kr value (no MBE from mismatched kr/derivative pairs).
    """

    irreducible_water_saturation: Number | None = None
    """(Default) Irreducible water saturation (swc)."""

    residual_oil_saturation_water: Number | None = None
    """(Default) Residual oil saturation after water flood (sorw)."""

    residual_oil_saturation_gas: Number | None = None
    """(Default) Residual oil saturation after gas flood (sorg)."""

    residual_gas_saturation: Number | None = None
    """(Default) Residual gas saturation (sgr)."""

    water_exponent: Number = 2.0
    """
    Corey exponent for water relative permeability.

    Higher values make the curve steeper. Meaning slower krw increase with saturation.
    """

    oil_exponent: Number = 2.0
    """
    Corey exponent for oil relative permeability.

    Higher values make the curve steeper. Meaning slower kro increase with saturation.
    """

    gas_exponent: Number = 2.0
    """
    Corey exponent for gas relative permeability. Higher values make the curve steeper.

    Meaning slower krg increase with saturation.
    """

    maximum_water_relperm: Number = 1.0
    """Endpoint (maximum) relative permeability for water (krw at residual oil)."""

    maximum_oil_relperm: Number = 1.0
    """Endpoint (maximum) relative permeability for oil (krocw)."""

    maximum_gas_relperm: Number = 1.0
    """Endpoint (maximum) relative permeability for gas (krg at connate water + residual oil)."""

    wettability: Wettability = Wettability.WATER_WET
    """Wettability type (water-wet or oil-wet)."""

    mixed_wet_water_fraction: Number = 0.5
    """Fraction of pore space that is water-wet in mixed-wet systems (0-1)."""

    mixing_rule: MixingRule | str = baker_linear_rule
    """
    Mixing rule function or name to compute oil relative permeability in three-phase system.

    The function should take the following parameters in order:
    - kro_w: Oil relative permeability from oil-water table
    - kro_g: Oil relative permeability from oil-gas table
    - krw: Water relative permeability from oil-water table
    - krg: Gas relative permeability from gas-oil table
    - kr_max: Oil relative permeability at connate water
    - Sw: Water saturation
    - So: Oil saturation
    - Sg: Gas saturation
    and return the mixed oil relative permeability.
    """

    minimum_water_relperm: MinimumRelPerm = "auto"
    """
    Minimum `min_value` for the water relative permeability.

    `"auto"` - dtype-aware `min_value`; `None` - no `min_value`; `Number` - explicit value.
    """

    minimum_oil_relperm: MinimumRelPerm = "auto"
    """
    Minimum `min_value` for the oil relative permeability.

    `"auto"` - dtype-aware `min_value`; `None` - no `min_value`; `Number` - explicit value.
    """

    minimum_gas_relperm: MinimumRelPerm = "auto"
    """
    Minimum `min_value` for the gas relative permeability.

    `"auto"` - dtype-aware `min_value`; `None` - no `min_value`; `Number` - explicit value.
    """

    supports_vector: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array/vector inputs."""

    def __attrs_post_init__(self) -> None:
        mixing_rule = self.mixing_rule
        if isinstance(mixing_rule, str):
            object.__setattr__(self, "mixing_rule", get_mixing_rule(mixing_rule))

        resolve_min_relperm(self.minimum_water_relperm)
        resolve_min_relperm(self.minimum_oil_relperm)
        resolve_min_relperm(self.minimum_gas_relperm)

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        wettability = self.wettability
        if wettability == Wettability.WATER_WET:
            return FluidPhase.WATER
        elif wettability == Wettability.OIL_WET:
            return FluidPhase.OIL
        elif self.mixed_wet_water_fraction >= 0.5:
            return FluidPhase.WATER
        return FluidPhase.OIL

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        return FluidPhase.OIL

    def get_oil_relperm_endpoint(self) -> Number:
        return self.maximum_oil_relperm

    def get_water_relperm_endpoint(self) -> Number:
        return self.maximum_water_relperm

    def get_gas_relperm_endpoint(self) -> Number:
        return self.maximum_gas_relperm

    def get_connate_water_saturation(self) -> Number:
        """
        This model doesn't distinguish connate from irreducible water
        saturation (a single `irreducible_water_saturation` parameter
        covers both), matching how the rest of this codebase already
        treats the two as coincident by default.
        """
        return self.irreducible_water_saturation or 0.0

    def get_residual_oil_saturation_water(self) -> Number:
        return self.residual_oil_saturation_water or 0.0

    def get_residual_oil_saturation_gas(self) -> Number:
        return self.residual_oil_saturation_gas or 0.0

    def get_residual_gas_saturation(self) -> Number:
        return self.residual_gas_saturation or 0.0

    def evaluate(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        irreducible_water_saturation: NumberOrArray[NDimension] | None = None,
        residual_oil_saturation_water: NumberOrArray[NDimension] | None = None,
        residual_oil_saturation_gas: NumberOrArray[NDimension] | None = None,
        residual_gas_saturation: NumberOrArray[NDimension] | None = None,
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for water, oil, and gas.

        Supports both scalar and array inputs for saturations.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param irreducible_water_saturation: Optional override for irreducible water saturation.
        :param residual_oil_saturation_water: Optional override for residual oil saturation after water flood.
        :param residual_oil_saturation_gas: Optional override for residual oil saturation after gas flood.
        :param residual_gas_saturation: Optional override for residual gas saturation.
        :return: `RelativePermeabilities` dictionary.
        """
        sorw = (
            residual_oil_saturation_water
            if residual_oil_saturation_water is not None
            else self.residual_oil_saturation_water
        )
        sorg = (
            residual_oil_saturation_gas
            if residual_oil_saturation_gas is not None
            else self.residual_oil_saturation_gas
        )
        sgr = (
            residual_gas_saturation
            if residual_gas_saturation is not None
            else self.residual_gas_saturation
        )
        swc = (
            irreducible_water_saturation
            if irreducible_water_saturation is not None
            else self.irreducible_water_saturation
        )
        params_missing = []
        if swc is None:
            params_missing.append("swc")
        if sorw is None:
            params_missing.append("sorw")
        if sorg is None:
            params_missing.append("sorg")
        if sgr is None:
            params_missing.append("sgr")
        if params_missing:
            raise ValidationError(
                f"Residual saturations must be provided either as arguments or set in the model instance. "
                f"Missing: {', '.join(params_missing)}"
            )

        krw, kro, krg = compute_brookes_corey_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
            residual_gas_saturation=sgr,  # type: ignore[arg-type]
            water_exponent=self.water_exponent,
            oil_exponent=self.oil_exponent,
            gas_exponent=self.gas_exponent,
            maximum_water_relperm=self.get_water_relperm_endpoint(),
            maximum_oil_relperm=self.get_oil_relperm_endpoint(),
            maximum_gas_relperm=self.get_gas_relperm_endpoint(),
            wettability=self.wettability,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
            mixing_rule=typing.cast(MixingRule, self.mixing_rule),  # type: ignore[arg-type]
            saturation_epsilon=c.SATURATION_EPSILON,
            minimum_mobile_pore_space=c.MINIMUM_MOBILE_PORE_SPACE,
            minimum_water_relperm=resolve_min_relperm(self.minimum_water_relperm),
            minimum_oil_relperm=resolve_min_relperm(self.minimum_oil_relperm),
            minimum_gas_relperm=resolve_min_relperm(self.minimum_gas_relperm),
        )
        return RelativePermeabilities(water=krw, oil=kro, gas=krg)  # type: ignore[typeddict-item]

    def derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        irreducible_water_saturation: NumberOrArray[NDimension] | None = None,
        residual_oil_saturation_water: NumberOrArray[NDimension] | None = None,
        residual_oil_saturation_gas: NumberOrArray[NDimension] | None = None,
        residual_gas_saturation: NumberOrArray[NDimension] | None = None,
        **kwargs: typing.Any,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute all nine partial derivatives of the three-phase relative
        permeabilities with respect to water saturation, oil saturation, and
        gas saturation using the Brooks-Corey model.

        Returns a dictionary containing:

        ``
        (dkrw/dsw, dkrw/dso, dkrw/dsg,
        dkro/dsw, dkro/dso, dkro/dsg,
        dkrg/dsw, dkrg/dso, dkrg/dsg)
        ``

        For the water-wet case all two-phase Corey power-law derivatives are
        computed analytically via the chain rule through effective saturation.
        The three-phase oil relative permeability derivative is then completed
        by the extended chain rule through the mixing rule (including the
        `dkro/dkrw` and `dkro/dkrg` terms for rules like Stone II that
        use the actual two-phase water/gas kr values).

        Wherever a minimum relperm min_value is active (raw kr ≤ min_value), the
        corresponding derivative is zeroed out, keeping the Jacobian consistent
        with the min_value kr value and preventing MBE.

        :param water_saturation: Water saturation (fraction, 0 to 1).
        :param oil_saturation: Oil saturation (fraction, 0 to 1).
        :param gas_saturation: Gas saturation (fraction, 0 to 1).
        :param irreducible_water_saturation: Optional override for the
            irreducible (connate) water saturation. Uses the model default
            when not provided.
        :param residual_oil_saturation_water: Optional override for the residual
            oil saturation to water flooding. Uses the model default when not
            provided.
        :param residual_oil_saturation_gas: Optional override for the residual
            oil saturation to gas flooding. Uses the model default when not
            provided.
        :param residual_gas_saturation: Optional override for the residual gas
            saturation. Uses the model default when not provided.
        :return: `RelativePermeabilityDerivatives` dictionary.
        """
        swc = (
            irreducible_water_saturation
            if irreducible_water_saturation is not None
            else self.irreducible_water_saturation
        )
        sorw = (
            residual_oil_saturation_water
            if residual_oil_saturation_water is not None
            else self.residual_oil_saturation_water
        )
        sorg = (
            residual_oil_saturation_gas
            if residual_oil_saturation_gas is not None
            else self.residual_oil_saturation_gas
        )
        sgr = (
            residual_gas_saturation
            if residual_gas_saturation is not None
            else self.residual_gas_saturation
        )

        params_missing = []
        if swc is None:
            params_missing.append("swc")
        if sorw is None:
            params_missing.append("sorw")
        if sorg is None:
            params_missing.append("sorg")
        if sgr is None:
            params_missing.append("sgr")
        if params_missing:
            raise ValidationError(
                f"Residual saturations must be provided either as arguments or set in the model instance. "
                f"Missing: {', '.join(params_missing)}"
            )

        water_exponent = self.water_exponent
        oil_exponent = self.oil_exponent
        gas_exponent = self.gas_exponent
        wettability = self.wettability
        mixing_rule = typing.cast(MixingRule, self.mixing_rule)

        krw_min = resolve_min_relperm(self.minimum_water_relperm)
        kro_min = resolve_min_relperm(self.minimum_oil_relperm)
        krg_min = resolve_min_relperm(self.minimum_gas_relperm)

        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
            and np.isscalar(swc)
            and np.isscalar(sorw)
            and np.isscalar(sorg)
            and np.isscalar(sgr)
        )
        sw = np.atleast_1d(water_saturation)
        so = np.atleast_1d(oil_saturation)
        sg = np.atleast_1d(gas_saturation)
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        zeros = np.zeros_like(sw)
        minimum_mobile_pore_space = c.MINIMUM_MOBILE_PORE_SPACE

        krw_max = self.get_water_relperm_endpoint()
        kro_max = self.get_oil_relperm_endpoint()
        krg_max = self.get_gas_relperm_endpoint()

        if wettability == Wettability.OIL_WET:
            # kro (wetting, depends on So)
            movable_oil_range = 1.0 - sorw - sorg  # type: ignore
            max_residual = np.minimum(sorw, sorg)  # type: ignore
            valid_oil = movable_oil_range > minimum_mobile_pore_space
            se_o = np.clip(
                (so - max_residual) / np.where(valid_oil, movable_oil_range, 1.0),
                0.0,
                1.0,
            )
            kro_vals = kro_max * se_o**oil_exponent

            dkro_dso_raw = kro_max * np.where(
                valid_oil & (se_o > 0.0),
                oil_exponent * se_o ** max(oil_exponent - 1.0, 0.0) / movable_oil_range,
                zeros,
            )
            # Apply min_value: zero derivative where raw kro ≤ min_value
            dkro_dso = clamp_relperm_derivative(dkro_dso_raw, kro_vals, kro_min)
            dkro_dsw = zeros.copy()
            dkro_dsg = zeros.copy()

            # krg (non-wetting, depends on Sg)
            movable_gas_range = 1.0 - sgr - swc  # type: ignore
            valid_gas = movable_gas_range > minimum_mobile_pore_space
            se_g = np.clip(
                (sg - sgr) / np.where(valid_gas, movable_gas_range, 1.0),
                0.0,
                1.0,
            )
            krg_vals = krg_max * se_g**gas_exponent

            dkrg_dsg_raw = krg_max * np.where(
                valid_gas & (se_g > 0.0),
                gas_exponent * se_g ** max(gas_exponent - 1.0, 0.0) / movable_gas_range,
                zeros,
            )
            dkrg_dsg = clamp_relperm_derivative(dkrg_dsg_raw, krg_vals, krg_min)  # type: ignore[arg-type]
            dkrg_dsw = zeros.copy()
            dkrg_dso = zeros.copy()

            # krw (intermediate phase, via mixing rule)
            one_minus_kro = np.clip(1.0 - kro_vals / kro_max, 0.0, None)
            one_minus_krg = np.clip(1.0 - krg_vals / krg_max, 0.0, None)
            kro_proxy = one_minus_kro**water_exponent
            krg_proxy = one_minus_krg**water_exponent

            dkro_proxy_dso = np.where(
                one_minus_kro > 0.0,
                water_exponent
                * one_minus_kro ** max(water_exponent - 1.0, 0.0)
                * (-dkro_dso_raw / kro_max),
                zeros,
            )
            dkro_proxy_dsw = zeros.copy()  # kro depends on So only
            dkro_proxy_dsg = zeros.copy()

            dkrg_proxy_dsg = np.where(
                one_minus_krg > 0.0,
                water_exponent
                * one_minus_krg ** max(water_exponent - 1.0, 0.0)
                * (-dkrg_dsg_raw / krg_max),
                zeros,
            )
            dkrg_proxy_dsw = zeros.copy()
            dkrg_proxy_dso = zeros.copy()

            derivatives = get_mixing_rule_partial_derivatives(
                rule=mixing_rule,
                kro_w=kro_proxy,
                kro_g=krg_proxy,  # type: ignore[arg-type]
                krw=kro_vals,
                krg=krg_vals,  # type: ignore[arg-type]
                kr_max=1.0,
                water_saturation=so,
                oil_saturation=sw,
                gas_saturation=sg,
                irreducible_water_saturation=np.minimum(sorw, sorg),  # type: ignore[arg-type]
                residual_oil_saturation_water=swc,  # type: ignore[arg-type]
                residual_oil_saturation_gas=swc,  # type: ignore[arg-type]
                epsilon=c.FINITE_DIFFERENCE_EPSILON,
            )
            dkrw_dkro_proxy = derivatives["dkro_dkro_w"]
            dkrw_dkrg_proxy = derivatives["dkro_dkro_g"]
            dkrw_dsw_explicit = derivatives["dkro_dso_explicit"]
            dkrw_dso_explicit = derivatives["dkro_dsw_explicit"]
            dkrw_dsg_explicit = derivatives["dkro_dsg_explicit"]

            # Chain rule for krw
            krw_raw = krw_max * np.clip(
                mixing_rule(
                    kro_w=kro_proxy,
                    kro_g=krg_proxy,  # type: ignore[arg-type]
                    krw=kro_vals,
                    krg=krg_vals,  # type: ignore[arg-type]
                    kr_max=1.0,
                    water_saturation=so,
                    oil_saturation=sw,
                    gas_saturation=sg,
                    irreducible_water_saturation=np.minimum(sorw, sorg),  # type: ignore[arg-type]
                    residual_oil_saturation_water=swc,  # type: ignore[arg-type]
                    residual_oil_saturation_gas=swc,  # type: ignore[arg-type]
                ),
                0.0,
                1.0,
            )
            dkrw_dsw_raw = krw_max * (
                dkrw_dkro_proxy * dkro_proxy_dsw
                + dkrw_dkrg_proxy * dkrg_proxy_dsw
                + dkrw_dsw_explicit
            )
            dkrw_dso_raw = krw_max * (
                dkrw_dkro_proxy * dkro_proxy_dso
                + dkrw_dkrg_proxy * dkrg_proxy_dso
                + derivatives["dkro_dkrw"] * dkro_dso_raw
                + dkrw_dso_explicit
            )
            dkrw_dsg_raw = krw_max * (
                dkrw_dkro_proxy * dkro_proxy_dsg
                + dkrw_dkrg_proxy * dkrg_proxy_dsg
                + derivatives["dkro_dkrg"] * dkrg_dsg_raw
                + dkrw_dsg_explicit
            )
            # Apply min_value to krw derivatives
            dkrw_dsw = clamp_relperm_derivative(dkrw_dsw_raw, krw_raw, krw_min)
            dkrw_dso = clamp_relperm_derivative(dkrw_dso_raw, krw_raw, krw_min)
            dkrw_dsg = clamp_relperm_derivative(dkrw_dsg_raw, krw_raw, krw_min)

            results = (
                dkrw_dsw,
                dkro_dsw,
                dkrg_dsw,
                dkrw_dso,
                dkro_dso,
                dkrg_dso,
                dkrw_dsg,
                dkro_dsg,
                dkrg_dsg,
            )
            if is_scalar:
                results = tuple(r.item() for r in results)  # type: ignore[union-attr]
            return RelativePermeabilityDerivatives(
                dkrw_dsw=results[0],
                dkro_dsw=results[1],
                dkrg_dsw=results[2],
                dkrw_dso=results[3],
                dkro_dso=results[4],
                dkrg_dso=results[5],
                dkrw_dsg=results[6],
                dkro_dsg=results[7],
                dkrg_dsg=results[8],
            )

        if wettability == Wettability.MIXED_WET:
            f = self.mixed_wet_water_fraction

            # Water-wet sub-system
            mobile_water_range_ww = 1.0 - swc - sorw  # type: ignore[operator]
            valid_water_ww = mobile_water_range_ww > minimum_mobile_pore_space
            se_w_ww = np.clip(
                (sw - swc) / np.where(valid_water_ww, mobile_water_range_ww, 1.0),
                0.0,
                1.0,
            )
            krw_ww = krw_max * se_w_ww**water_exponent
            dkrw_ww_dsw_raw = krw_max * np.where(
                valid_water_ww & (se_w_ww > 0.0),
                water_exponent
                * (se_w_ww ** max(water_exponent - 1.0, 0.0))
                / mobile_water_range_ww,
                zeros,
            )

            mobile_gas_range_ww = 1.0 - swc - sgr - sorg  # type: ignore[operator]
            valid_gas_ww = mobile_gas_range_ww > minimum_mobile_pore_space
            se_g_ww = np.clip(
                (sg - sgr) / np.where(valid_gas_ww, mobile_gas_range_ww, 1.0),
                0.0,
                1.0,
            )
            krg_ww = krg_max * se_g_ww**gas_exponent
            dkrg_ww_dsg_raw = krg_max * np.where(
                valid_gas_ww & (se_g_ww > 0.0),
                gas_exponent * (se_g_ww ** max(gas_exponent - 1.0, 0.0)) / mobile_gas_range_ww,
                zeros,
            )

            one_minus_krw_ww = np.clip(1.0 - krw_ww / krw_max, 0.0, None)
            kro_w_ww = one_minus_krw_ww**oil_exponent
            dkro_w_ww_dsw = np.where(
                one_minus_krw_ww > 0.0,
                oil_exponent
                * (one_minus_krw_ww ** max(oil_exponent - 1.0, 0.0))
                * (-dkrw_ww_dsw_raw / krw_max),
                zeros,
            )

            one_minus_krg_ww = np.clip(1.0 - krg_ww / krg_max, 0.0, None)
            kro_g_ww = one_minus_krg_ww**oil_exponent
            dkro_g_ww_dsg = np.where(
                one_minus_krg_ww > 0.0,
                oil_exponent
                * (one_minus_krg_ww ** max(oil_exponent - 1.0, 0.0))
                * (-dkrg_ww_dsg_raw / krg_max),
                zeros,
            )

            derivs_ww = get_mixing_rule_partial_derivatives(
                rule=mixing_rule,
                kro_w=kro_w_ww,  # type: ignore[arg-type]
                kro_g=kro_g_ww,  # type: ignore[arg-type]
                krw=krw_ww,  # type: ignore[arg-type]
                krg=krg_ww,  # type: ignore[arg-type]
                kr_max=1.0,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                irreducible_water_saturation=swc,  # type: ignore[arg-type]
                residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
                residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
                epsilon=c.FINITE_DIFFERENCE_EPSILON,
            )
            kro_ww_raw = kro_max * np.clip(
                mixing_rule(
                    kro_w=kro_w_ww,  # type: ignore[arg-type]
                    kro_g=kro_g_ww,  # type: ignore[arg-type]
                    krw=krw_ww,  # type: ignore[arg-type]
                    krg=krg_ww,  # type: ignore[arg-type]
                    kr_max=1.0,
                    water_saturation=sw,
                    oil_saturation=so,
                    gas_saturation=sg,
                    irreducible_water_saturation=swc,  # type: ignore[arg-type]
                    residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
                    residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
                ),
                0.0,
                1.0,
            )
            dkro_ww_dsw = kro_max * (
                derivs_ww["dkro_dkro_w"] * dkro_w_ww_dsw
                + derivs_ww["dkro_dkrw"] * dkrw_ww_dsw_raw
                + derivs_ww["dkro_dsw_explicit"]
            )
            dkro_ww_dso = (
                kro_max * derivs_ww["dkro_dso_explicit"]
            )  # neither shaped input depends on So
            dkro_ww_dsg = kro_max * (
                derivs_ww["dkro_dkro_g"] * dkro_g_ww_dsg
                + derivs_ww["dkro_dkrg"] * dkrg_ww_dsg_raw
                + derivs_ww["dkro_dsg_explicit"]
            )

            # Oil-wet sub-system
            movable_oil_range_ow = 1.0 - sorw - sorg  # type: ignore[operator]
            max_residual_ow = np.minimum(sorw, sorg)  # type: ignore[operator]
            valid_oil_ow = movable_oil_range_ow > minimum_mobile_pore_space
            se_o_ow = np.clip(
                (so - max_residual_ow) / np.where(valid_oil_ow, movable_oil_range_ow, 1.0),
                0.0,
                1.0,
            )
            kro_ow = kro_max * se_o_ow**oil_exponent
            dkro_ow_dso = kro_max * np.where(
                valid_oil_ow & (se_o_ow > 0.0),
                oil_exponent * (se_o_ow ** max(oil_exponent - 1.0, 0.0)) / movable_oil_range_ow,
                zeros,
            )

            movable_gas_range_ow = 1.0 - sgr - swc  # type: ignore[operator]
            valid_gas_ow = movable_gas_range_ow > minimum_mobile_pore_space
            se_g_ow = np.clip(
                (sg - sgr) / np.where(valid_gas_ow, movable_gas_range_ow, 1.0),
                0.0,
                1.0,
            )
            krg_ow = krg_max * se_g_ow**gas_exponent
            dkrg_ow_dsg = krg_max * np.where(
                valid_gas_ow & (se_g_ow > 0.0),
                gas_exponent * (se_g_ow ** max(gas_exponent - 1.0, 0.0)) / movable_gas_range_ow,
                zeros,
            )

            one_minus_kro_ow = np.clip(1.0 - kro_ow / kro_max, 0.0, None)
            kro_proxy_ow = one_minus_kro_ow**water_exponent
            dkro_proxy_ow_dso = np.where(
                one_minus_kro_ow > 0.0,
                water_exponent
                * (one_minus_kro_ow ** max(water_exponent - 1.0, 0.0))
                * (-dkro_ow_dso / kro_max),
                zeros,
            )

            one_minus_krg_ow = np.clip(1.0 - krg_ow / krg_max, 0.0, None)
            krg_proxy_ow = one_minus_krg_ow**water_exponent
            dkrg_proxy_ow_dsg = np.where(
                one_minus_krg_ow > 0.0,
                water_exponent
                * (one_minus_krg_ow ** max(water_exponent - 1.0, 0.0))
                * (-dkrg_ow_dsg / krg_max),
                zeros,
            )

            derivs_ow = get_mixing_rule_partial_derivatives(
                rule=mixing_rule,
                kro_w=kro_proxy_ow,
                kro_g=krg_proxy_ow,  # type: ignore[arg-type]
                krw=kro_ow,
                krg=krg_ow,  # type: ignore[arg-type]
                kr_max=1.0,
                water_saturation=so,
                oil_saturation=sw,
                gas_saturation=sg,
                irreducible_water_saturation=np.minimum(sorw, sorg),  # type: ignore[arg-type]
                residual_oil_saturation_water=swc,  # type: ignore[arg-type]
                residual_oil_saturation_gas=swc,  # type: ignore[arg-type]
                epsilon=c.FINITE_DIFFERENCE_EPSILON,
            )
            krw_ow_raw = krw_max * np.clip(
                mixing_rule(
                    kro_w=kro_proxy_ow,
                    kro_g=krg_proxy_ow,  # type: ignore[arg-type]
                    krw=kro_ow,
                    krg=krg_ow,  # type: ignore[arg-type]
                    kr_max=1.0,
                    water_saturation=so,
                    oil_saturation=sw,
                    gas_saturation=sg,
                    irreducible_water_saturation=np.minimum(sorw, sorg),  # type: ignore[arg-type]
                    residual_oil_saturation_water=swc,  # type: ignore[arg-type]
                    residual_oil_saturation_gas=swc,  # type: ignore[arg-type]
                ),
                0.0,
                1.0,
            )
            dkrw_ow_dsw = krw_max * derivs_ow["dkro_dso_explicit"]
            dkrw_ow_dso = krw_max * (
                derivs_ow["dkro_dkro_w"] * dkro_proxy_ow_dso
                + derivs_ow["dkro_dkrw"] * dkro_ow_dso
                + derivs_ow["dkro_dsw_explicit"]
            )
            dkrw_ow_dsg = krw_max * (
                derivs_ow["dkro_dkro_g"] * dkrg_proxy_ow_dsg
                + derivs_ow["dkro_dkrg"] * dkrg_ow_dsg
                + derivs_ow["dkro_dsg_explicit"]
            )

            # Blend raw values for min_value masking
            krw_blend_raw = f * krw_ww + (1.0 - f) * krw_ow_raw
            kro_blend_raw = f * kro_ww_raw + (1.0 - f) * kro_ow
            krg_blend_raw = f * krg_ww + (1.0 - f) * krg_ow

            # Blend derivatives
            dkrw_dsw_raw = f * dkrw_ww_dsw_raw + (1.0 - f) * dkrw_ow_dsw
            dkrw_dso_raw = f * zeros + (1.0 - f) * dkrw_ow_dso
            dkrw_dsg_raw = f * zeros + (1.0 - f) * dkrw_ow_dsg

            dkro_dsw_raw = f * dkro_ww_dsw + (1.0 - f) * zeros
            dkro_dso_raw = f * dkro_ww_dso + (1.0 - f) * dkro_ow_dso
            dkro_dsg_raw = f * dkro_ww_dsg + (1.0 - f) * zeros

            dkrg_dsg_raw = f * dkrg_ww_dsg_raw + (1.0 - f) * dkrg_ow_dsg

            # Apply min_values to blended derivatives
            dkrw_dsw = clamp_relperm_derivative(dkrw_dsw_raw, krw_blend_raw, krw_min)  # type: ignore[arg-type]
            dkrw_dso = clamp_relperm_derivative(dkrw_dso_raw, krw_blend_raw, krw_min)  # type: ignore[arg-type]
            dkrw_dsg = clamp_relperm_derivative(dkrw_dsg_raw, krw_blend_raw, krw_min)  # type: ignore[arg-type]
            dkro_dsw = clamp_relperm_derivative(dkro_dsw_raw, kro_blend_raw, kro_min)
            dkro_dso = clamp_relperm_derivative(dkro_dso_raw, kro_blend_raw, kro_min)
            dkro_dsg = clamp_relperm_derivative(dkro_dsg_raw, kro_blend_raw, kro_min)
            dkrg_dsw = zeros.copy()
            dkrg_dso = zeros.copy()
            dkrg_dsg = clamp_relperm_derivative(dkrg_dsg_raw, krg_blend_raw, krg_min)  # type: ignore[arg-type]

            results = (
                dkrw_dsw,
                dkro_dsw,
                dkrg_dsw,
                dkrw_dso,
                dkro_dso,
                dkrg_dso,
                dkrw_dsg,
                dkro_dsg,
                dkrg_dsg,
            )
            if is_scalar:
                results = tuple(r.item() for r in results)  # type: ignore[union-attr]
            return RelativePermeabilityDerivatives(
                dkrw_dsw=results[0],
                dkro_dsw=results[1],
                dkrg_dsw=results[2],
                dkrw_dso=results[3],
                dkro_dso=results[4],
                dkrg_dso=results[5],
                dkrw_dsg=results[6],
                dkro_dsg=results[7],
                dkrg_dsg=results[8],
            )

        # Water-wet path
        # krw = Se_w ^ nw
        mobile_water_range = 1.0 - swc - sorw  # type: ignore
        valid_water = mobile_water_range > minimum_mobile_pore_space
        se_w = np.clip(
            (sw - swc) / np.where(valid_water, mobile_water_range, 1.0),
            0.0,
            1.0,
        )
        krw_values = krw_max * se_w**water_exponent
        dkrw_dsw_raw = krw_max * np.where(
            valid_water & (se_w > 0.0),
            water_exponent * (se_w ** max(water_exponent - 1.0, 0.0)) / mobile_water_range,
            zeros,
        )
        dkrw_dsw = clamp_relperm_derivative(dkrw_dsw_raw, krw_values, krw_min)  # type: ignore[arg-type]
        dkrw_dso = zeros.copy()
        dkrw_dsg = zeros.copy()

        # krg = Se_g ^ ng
        mobile_gas_range = 1.0 - swc - sgr - sorg  # type: ignore
        valid_gas = mobile_gas_range > minimum_mobile_pore_space
        se_g = np.clip(
            (sg - sgr) / np.where(valid_gas, mobile_gas_range, 1.0),
            0.0,
            1.0,
        )
        krg_values = krg_max * se_g**gas_exponent
        dkrg_dsg_raw = krg_max * np.where(
            valid_gas & (se_g > 0.0),
            gas_exponent * (se_g ** max(gas_exponent - 1.0, 0.0)) / mobile_gas_range,
            zeros,
        )
        dkrg_dsg = clamp_relperm_derivative(dkrg_dsg_raw, krg_values, krg_min)  # type: ignore[arg-type]
        dkrg_dsw = zeros.copy()
        dkrg_dso = zeros.copy()

        # kro_w shaped = (1 - krw)^no
        one_minus_krw = np.clip(1.0 - krw_values / krw_max, 0.0, None)
        kro_w_shaped = one_minus_krw**oil_exponent
        dkro_w_dsw = np.where(
            one_minus_krw > 0.0,
            oil_exponent
            * (one_minus_krw ** max(oil_exponent - 1.0, 0.0))
            * (-dkrw_dsw_raw / krw_max),
            zeros,
        )
        dkro_w_dso = zeros.copy()
        dkro_w_dsg = zeros.copy()

        # kro_g shaped = (1 - krg)^no
        one_minus_krg = np.clip(1.0 - krg_values / krg_max, 0.0, None)
        kro_g_shaped = one_minus_krg**oil_exponent
        dkro_g_dsg = np.where(
            one_minus_krg > 0.0,
            oil_exponent
            * (one_minus_krg ** max(oil_exponent - 1.0, 0.0))
            * (-dkrg_dsg_raw / krg_max),
            zeros,
        )
        dkro_g_dsw = zeros.copy()
        dkro_g_dso = zeros.copy()

        derivatives = get_mixing_rule_partial_derivatives(
            rule=mixing_rule,
            kro_w=kro_w_shaped,  # type: ignore[arg-type]
            kro_g=kro_g_shaped,  # type: ignore[arg-type]
            krw=krw_values,  # type: ignore[arg-type]
            krg=krg_values,  # type: ignore[arg-type]
            kr_max=1.0,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            irreducible_water_saturation=swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
            epsilon=c.FINITE_DIFFERENCE_EPSILON,
        )
        dkro_dkro_w = derivatives["dkro_dkro_w"]
        dkro_dkro_g = derivatives["dkro_dkro_g"]
        dkro_dkrw_mix = derivatives["dkro_dkrw"]
        dkro_dkrg_mix = derivatives["dkro_dkrg"]
        dkro_dsw_explicit = derivatives["dkro_dsw_explicit"]
        dkro_dso_explicit = derivatives["dkro_dso_explicit"]
        dkro_dsg_explicit = derivatives["dkro_dsg_explicit"]

        # Forward evaluate kro for min_value masking
        kro_raw = kro_max * np.clip(
            mixing_rule(
                kro_w=kro_w_shaped,  # type: ignore[arg-type]
                kro_g=kro_g_shaped,  # type: ignore[arg-type]
                krw=krw_values,  # type: ignore[arg-type]
                krg=krg_values,  # type: ignore[arg-type]
                kr_max=1.0,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                irreducible_water_saturation=swc,  # type: ignore[arg-type]
                residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
                residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
            ),
            0.0,
            1.0,
        )

        dkro_dsw_raw = kro_max * (
            dkro_dkro_w * dkro_w_dsw
            + dkro_dkro_g * dkro_g_dsw
            + dkro_dkrw_mix * dkrw_dsw_raw
            + dkro_dkrg_mix * zeros
            + dkro_dsw_explicit
        )
        dkro_dso_raw = kro_max * (
            dkro_dkro_w * dkro_w_dso
            + dkro_dkro_g * dkro_g_dso
            + dkro_dkrw_mix * zeros
            + dkro_dkrg_mix * zeros
            + dkro_dso_explicit
        )
        dkro_dsg_raw = kro_max * (
            dkro_dkro_w * dkro_w_dsg
            + dkro_dkro_g * dkro_g_dsg
            + dkro_dkrw_mix * zeros
            + dkro_dkrg_mix * dkrg_dsg_raw
            + dkro_dsg_explicit
        )

        dkro_dsw = clamp_relperm_derivative(dkro_dsw_raw, kro_raw, kro_min)
        dkro_dso = clamp_relperm_derivative(dkro_dso_raw, kro_raw, kro_min)
        dkro_dsg = clamp_relperm_derivative(dkro_dsg_raw, kro_raw, kro_min)

        results = (
            dkrw_dsw,
            dkro_dsw,
            dkrg_dsw,
            dkrw_dso,
            dkro_dso,
            dkrg_dso,
            dkrw_dsg,
            dkro_dsg,
            dkrg_dsg,
        )
        if is_scalar:
            results = tuple(r.item() for r in results)  # type: ignore[union-attr]
            return RelativePermeabilityDerivatives(
                dkrw_dsw=results[0],
                dkro_dsw=results[1],
                dkrg_dsw=results[2],
                dkrw_dso=results[3],
                dkro_dso=results[4],
                dkrg_dso=results[5],
                dkrw_dsg=results[6],
                dkro_dsg=results[7],
                dkrg_dsg=results[8],
            )
        return RelativePermeabilityDerivatives(
            dkrw_dsw=dkrw_dsw,
            dkro_dsw=dkro_dsw,
            dkrg_dsw=dkrg_dsw,
            dkrw_dso=dkrw_dso,
            dkro_dso=dkro_dso,
            dkrg_dso=dkrg_dso,
            dkrw_dsg=dkrw_dsg,
            dkro_dsg=dkro_dsg,
            dkrg_dsg=dkrg_dsg,
        )

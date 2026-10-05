"""Relative permeability analytical models/tables multi-phase flow simulations."""

import typing

import attrs
import numba
import numpy as np

from bores.blackoil.satfunc.relperm.mixing_rules import (
    MixingRule,
    eclipse_rule,
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
from bores.serde.base import Serializable
from bores.types import (
    FluidPhase,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    RelativePermeabilities,
    RelativePermeabilityDerivatives,
    Wettability,
)

__all__ = [
    "BrooksCoreyRelPermTable",
    "LETParameters",
    "LETThreePhaseRelPermTable",
    "compute_brookes_corey_relative_permeabilities",
    "compute_let_relative_permeabilities",
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
    mixing_rule: MixingRule = eclipse_rule,
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
                kr_max=maximum_oil_relperm,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
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
                kr_max=maximum_water_relperm,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
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
                kr_max=maximum_oil_relperm,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
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
                kr_max=maximum_water_relperm,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
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
    krw = clamp_relperm(krw, minimum_water_relperm)
    kro = clamp_relperm(kro, minimum_oil_relperm)
    krg = clamp_relperm(krg, minimum_gas_relperm)
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

    mixing_rule: MixingRule | str = eclipse_rule
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
            dkrg_dsg = clamp_relperm_derivative(dkrg_dsg_raw, krg_vals, krg_min)
            dkrg_dsw = zeros.copy()
            dkrg_dso = zeros.copy()

            # krw (intermediate phase, via mixing rule)
            one_minus_kro = np.clip(1.0 - kro_vals / kro_max, 0.0, None)
            one_minus_krg = np.clip(1.0 - krg_vals / krg_max, 0.0, None)
            kro_proxy = one_minus_kro**water_exponent
            krg_proxy = one_minus_krg**water_exponent

            dkro_proxy_dso = np.where(
                one_minus_kro > 0.0,
                water_exponent * one_minus_kro ** max(water_exponent - 1.0, 0.0) * (-dkro_dso_raw),
                zeros,
            )
            dkro_proxy_dsw = -dkro_proxy_dso  # So = 1-Sw-Sg
            dkro_proxy_dsg = -dkro_proxy_dso

            dkrg_proxy_dsg = np.where(
                one_minus_krg > 0.0,
                water_exponent * one_minus_krg ** max(water_exponent - 1.0, 0.0) * (-dkrg_dsg_raw),
                zeros,
            )
            dkrg_proxy_dsw = zeros.copy()
            dkrg_proxy_dso = zeros.copy()

            derivatives = get_mixing_rule_partial_derivatives(
                rule=mixing_rule,
                kro_w=kro_proxy,
                kro_g=krg_proxy,  # type: ignore[arg-type]
                krw=kro_vals,
                krg=krg_vals,
                kr_max=krw_max,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                epsilon=c.FINITE_DIFFERENCE_EPSILON,
            )
            dkrw_dkro_proxy = derivatives["dkro_dkro_w"]
            dkrw_dkrg_proxy = derivatives["dkro_dkro_g"]
            dkrw_dsw_explicit = derivatives["dkro_dsw_explicit"]
            dkrw_dso_explicit = derivatives["dkro_dso_explicit"]
            dkrw_dsg_explicit = derivatives["dkro_dsg_explicit"]

            # Chain rule for krw
            krw_raw = krw_max * np.clip(
                mixing_rule(
                    kro_w=kro_proxy,
                    kro_g=krg_proxy,
                    krw=kro_vals,
                    krg=krg_vals,
                    kr_max=krw_max,
                    water_saturation=sw,
                    oil_saturation=so,
                    gas_saturation=sg,
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
                + dkrw_dso_explicit
            )
            dkrw_dsg_raw = krw_max * (
                dkrw_dkro_proxy * dkro_proxy_dsg
                + dkrw_dkrg_proxy * dkrg_proxy_dsg
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
                * (-dkrw_ww_dsw_raw),
                zeros,
            )

            one_minus_krg_ww = np.clip(1.0 - krg_ww / krg_max, 0.0, None)
            kro_g_ww = one_minus_krg_ww**oil_exponent
            dkro_g_ww_dsg = np.where(
                one_minus_krg_ww > 0.0,
                oil_exponent
                * (one_minus_krg_ww ** max(oil_exponent - 1.0, 0.0))
                * (-dkrg_ww_dsg_raw),
                zeros,
            )

            derivs_ww = get_mixing_rule_partial_derivatives(
                rule=mixing_rule,
                kro_w=kro_w_ww,  # type: ignore[arg-type]
                kro_g=kro_g_ww,  # type: ignore[arg-type]
                krw=krw_ww,
                krg=krg_ww,
                kr_max=kro_max,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                epsilon=c.FINITE_DIFFERENCE_EPSILON,
            )
            kro_ww_raw = kro_max * np.clip(
                mixing_rule(
                    kro_w=kro_w_ww,
                    kro_g=kro_g_ww,
                    krw=krw_ww,
                    krg=krg_ww,
                    kr_max=kro_max,
                    water_saturation=sw,
                    oil_saturation=so,
                    gas_saturation=sg,
                ),
                0.0,
                1.0,
            )
            dkro_ww_dsw = kro_max * (
                derivs_ww["dkro_dkro_w"] * dkro_w_ww_dsw
                + derivs_ww["dkro_dkro_g"] * zeros  # kro_g_ww has no Sw dependence
                + derivs_ww["dkro_dsw_explicit"]
            )
            dkro_ww_dso = (
                kro_max * derivs_ww["dkro_dso_explicit"]
            )  # neither shaped input depends on So
            dkro_ww_dsg = kro_max * (
                derivs_ww["dkro_dkro_w"] * zeros  # kro_w_ww has no Sg dependence
                + derivs_ww["dkro_dkro_g"] * dkro_g_ww_dsg
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
                * (-dkro_ow_dso),
                zeros,
            )

            one_minus_krg_ow = np.clip(1.0 - krg_ow / krg_max, 0.0, None)
            krg_proxy_ow = one_minus_krg_ow**water_exponent
            dkrg_proxy_ow_dsg = np.where(
                one_minus_krg_ow > 0.0,
                water_exponent
                * (one_minus_krg_ow ** max(water_exponent - 1.0, 0.0))
                * (-dkrg_ow_dsg),
                zeros,
            )

            derivs_ow = get_mixing_rule_partial_derivatives(
                rule=mixing_rule,
                kro_w=kro_proxy_ow,
                kro_g=krg_proxy_ow,  # type: ignore[arg-type]
                krw=kro_ow,
                krg=krg_ow,
                kr_max=krw_max,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                epsilon=c.FINITE_DIFFERENCE_EPSILON,
            )
            krw_ow_raw = krw_max * np.clip(
                mixing_rule(
                    kro_w=kro_proxy_ow,
                    kro_g=krg_proxy_ow,
                    krw=kro_ow,
                    krg=krg_ow,
                    kr_max=krw_max,
                    water_saturation=sw,
                    oil_saturation=so,
                    gas_saturation=sg,
                ),
                0.0,
                1.0,
            )
            dkrw_ow_dsw = krw_max * derivs_ow["dkro_dsw_explicit"]
            dkrw_ow_dso = krw_max * (
                derivs_ow["dkro_dkro_w"] * dkro_proxy_ow_dso + derivs_ow["dkro_dso_explicit"]
            )
            dkrw_ow_dsg = krw_max * (
                derivs_ow["dkro_dkro_g"] * dkrg_proxy_ow_dsg + derivs_ow["dkro_dsg_explicit"]
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
            dkrw_dsw = clamp_relperm_derivative(dkrw_dsw_raw, krw_blend_raw, krw_min)
            dkrw_dso = clamp_relperm_derivative(dkrw_dso_raw, krw_blend_raw, krw_min)
            dkrw_dsg = clamp_relperm_derivative(dkrw_dsg_raw, krw_blend_raw, krw_min)
            dkro_dsw = clamp_relperm_derivative(dkro_dsw_raw, kro_blend_raw, kro_min)
            dkro_dso = clamp_relperm_derivative(dkro_dso_raw, kro_blend_raw, kro_min)
            dkro_dsg = clamp_relperm_derivative(dkro_dsg_raw, kro_blend_raw, kro_min)
            dkrg_dsw = zeros.copy()
            dkrg_dso = zeros.copy()
            dkrg_dsg = clamp_relperm_derivative(dkrg_dsg_raw, krg_blend_raw, krg_min)

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
        dkrw_dsw = clamp_relperm_derivative(dkrw_dsw_raw, krw_values, krw_min)
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
        dkrg_dsg = clamp_relperm_derivative(dkrg_dsg_raw, krg_values, krg_min)
        dkrg_dsw = zeros.copy()
        dkrg_dso = zeros.copy()

        # kro_w shaped = (1 - krw)^no
        one_minus_krw = np.clip(1.0 - krw_values / krw_max, 0.0, None)
        kro_w_shaped = one_minus_krw**oil_exponent
        dkro_w_dsw = np.where(
            one_minus_krw > 0.0,
            oil_exponent * (one_minus_krw ** max(oil_exponent - 1.0, 0.0)) * (-dkrw_dsw_raw),
            zeros,
        )
        dkro_w_dso = zeros.copy()
        dkro_w_dsg = zeros.copy()

        # kro_g shaped = (1 - krg)^no
        one_minus_krg = np.clip(1.0 - krg_values / krg_max, 0.0, None)
        kro_g_shaped = one_minus_krg**oil_exponent
        dkro_g_dsg = np.where(
            one_minus_krg > 0.0,
            oil_exponent * (one_minus_krg ** max(oil_exponent - 1.0, 0.0)) * (-dkrg_dsg_raw),
            zeros,
        )
        dkro_g_dsw = zeros.copy()
        dkro_g_dso = zeros.copy()

        derivatives = get_mixing_rule_partial_derivatives(
            rule=mixing_rule,
            kro_w=kro_w_shaped,  # type: ignore[arg-type]
            kro_g=kro_g_shaped,  # type: ignore[arg-type]
            krw=krw_values,
            krg=krg_values,
            kr_max=kro_max,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
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
                kro_w=kro_w_shaped,
                kro_g=kro_g_shaped,
                krw=krw_values,
                krg=krg_values,
                kr_max=kro_max,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
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


@attrs.frozen
class LETParameters(Serializable):
    """
    LET curve-shape parameters for a single relative permeability curve.

    The LET correlation computes relative permeability from normalized
    saturation S* as:

        kr = kr_max * S*^L / (S*^L + E * (1 - S*)^T)

    where L, E, and T control different regions of the curve:

    - **L** (low-end): Controls curvature at low normalized saturation.
      Higher values delay the onset of flow (the curve stays near zero longer
      before rising). Analogous to a Corey exponent for the lower end.

    - **E** (elevation): Controls the overall position/elevation of the curve
      between the endpoints. Higher values push the curve downward (lower kr
      at intermediate saturations). E = 1 gives a curve similar to a simple
      power law. E < 1 raises the curve; E > 1 suppresses it.

    - **T** (top-end): Controls curvature at high normalized saturation.
      Higher values make the curve flatten earlier as it approaches kr_max
      (the curve reaches its plateau sooner). Analogous to a Corey exponent
      for the upper end.

    All three parameters must be positive. Typical ranges are L in [0.5, 5],
    E in [0.1, 10], and T in [0.5, 5].
    """

    L: Number = 2.0
    """Low-end shape parameter. Controls curvature near zero normalized saturation."""
    E: Number = 1.0
    """Elevation parameter. Controls overall curve height at intermediate saturations."""
    T: Number = 2.0
    """Top-end shape parameter. Controls curvature near maximum normalized saturation."""

    def __attrs_post_init__(self) -> None:
        if self.L <= 0:
            raise ValidationError(f"LET parameter `L` must be positive, got {self.L}")
        if self.E <= 0:
            raise ValidationError(f"LET parameter `E` must be positive, got {self.E}")
        if self.T <= 0:
            raise ValidationError(f"LET parameter `T` must be positive, got {self.T}")


@numba.njit(cache=True)
def _let_relperm(
    normalized_saturation: NumberArray[NDimension],
    L: Number,
    E: Number,
    T: Number,
) -> NumberArray[NDimension]:
    """
    Core LET relative permeability formula (without endpoint scaling).

    Computes: S*^L / (S*^L + E * (1 - S*)^T)

    Returns 0 when S* = 0, and 1 when S* = 1. For intermediate values the
    curve shape is governed by L, E, and T.

    :param normalized_saturation: Effective (normalized) saturation, clipped to [0, 1].
    :param L: Low-end shape parameter (> 0).
    :param E: Elevation parameter (> 0).
    :param T: Top-end shape parameter (> 0).
    :return: Relative permeability value(s) in [0, 1].
    """
    s = normalized_saturation
    numerator = s**L
    denominator = numerator + E * (1.0 - s) ** T
    # Safe division: denominator is zero only when both s^L = 0 and (1-s)^T = 0,
    # which requires s = 0 and s = 1 simultaneously (impossible). For s = 0,
    # numerator = 0 and denominator = E > 0, so result = 0. For s = 1,
    # numerator = 1 and denominator = 1, so result = 1. No special handling needed
    # for well-formed inputs, but protect against floating-point edge cases.
    return np.where(denominator > 0.0, numerator / denominator, 0.0)  # type: ignore[return-value]


def compute_let_relative_permeabilities(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    residual_gas_saturation: NumberOrArray[NDimension],
    water_L: Number,
    water_E: Number,
    water_T: Number,
    oil_water_L: Number,
    oil_water_E: Number,
    oil_water_T: Number,
    gas_oil_L: Number,
    gas_oil_E: Number,
    gas_oil_T: Number,
    gas_L: Number,
    gas_E: Number,
    gas_T: Number,
    maximum_water_relperm: Number = 1.0,
    maximum_oil_relperm: Number = 1.0,
    maximum_gas_relperm: Number = 1.0,
    wettability: Wettability = Wettability.WATER_WET,
    mixed_wet_water_fraction: Number = 0.5,
    mixing_rule: MixingRule = eclipse_rule,
    saturation_epsilon: Number = 1e-6,
    minimum_mobile_pore_space: Number = 1e-9,
    minimum_water_relperm: Number | None = None,
    minimum_oil_relperm: Number | None = None,
    minimum_gas_relperm: Number | None = None,
) -> tuple[NumberOrArray[NDimension], NumberOrArray[NDimension], NumberOrArray[NDimension]]:
    """
    Compute three-phase relative permeabilities using the LET correlation.

    The LET (Lomeland-Ebeltoft-Thomas) model uses three curve-shape parameters
    (L, E, T) per phase-pair, providing more flexibility than the single Corey
    exponent for fitting laboratory data. Each two-phase kr curve is computed
    independently from normalized saturation, then the three-phase oil kr is
    obtained through a mixing rule.

    Supports both scalar and array inputs for saturations.

    :param water_saturation: Water saturation (fraction, 0 to 1).
    :param oil_saturation: Oil saturation (fraction, 0 to 1).
    :param gas_saturation: Gas saturation (fraction, 0 to 1).
    :param irreducible_water_saturation: Irreducible water saturation (swc).
    :param residual_oil_saturation_water: Residual oil saturation to waterflood (sorw).
    :param residual_oil_saturation_gas: Residual oil saturation to gas flood (sorg).
    :param residual_gas_saturation: Residual gas saturation (sgr).
    :param water_L: Water LET `L` parameter.
    :param water_E: Water LET `E` parameter.
    :param water_T: Water LET `T` parameter.
    :param oil_water_L: Oil (water-oil system) LET `L` parameter.
    :param oil_water_E: Oil (water-oil system) LET `E` parameter.
    :param oil_water_T: Oil (water-oil system) LET `T` parameter.
    :param gas_oil_L: Oil (gas-oil system) LET `L` parameter.
    :param gas_oil_E: Oil (gas-oil system) LET `E` parameter.
    :param gas_oil_T: Oil (gas-oil system) LET `T` parameter.
    :param gas_L: Gas LET `L` parameter.
    :param gas_E: Gas LET `E` parameter.
    :param gas_T: Gas LET `T` parameter.
    :param maximum_water_relperm: Endpoint relative permeability for water (krw_max).
    :param maximum_oil_relperm: Endpoint relative permeability for oil (kro_max).
    :param maximum_gas_relperm: Endpoint relative permeability for gas (krg_max).
    :param wettability: Wettability type (water-wet or oil-wet).
    :param mixed_wet_water_fraction: Fraction of water-wet behavior in mixed-wet case (0 to 1).
    :param mixing_rule: Three-phase mixing rule for oil relative permeability.
    :param minimum_water_relperm: Resolved minimum min_value for water kr (`None` = no min_value).
    :param minimum_oil_relperm: Resolved minimum min_value for oil kr (`None` = no min_value).
    :param minimum_gas_relperm: Resolved minimum min_value for gas kr (`None` = no min_value).
    :return: (krw, kro, krg) tuple of relative permeabilities.
    """
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

    sw, so, sg = np.broadcast_arrays(sw, so, sg)
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

    swc = irreducible_water_saturation
    sorw = residual_oil_saturation_water
    sorg = residual_oil_saturation_gas
    sgr = residual_gas_saturation

    if wettability == Wettability.WATER_WET:
        # Water kr (wetting phase)
        movable_water_range = 1.0 - swc - sorw  # type: ignore[operator]
        sw_star = np.where(
            movable_water_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - swc) / movable_water_range, 0.0, 1.0),
        )
        krw = maximum_water_relperm * _let_relperm(sw_star, water_L, water_E, water_T)

        # Gas kr (non-wetting phase)
        movable_gas_range = 1.0 - swc - sgr - sorg  # type: ignore[operator]
        sg_star = np.where(
            movable_gas_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - sgr) / movable_gas_range, 0.0, 1.0),
        )
        krg = maximum_gas_relperm * _let_relperm(sg_star, gas_L, gas_E, gas_T)

        # Oil kr (intermediate phase, three-phase mixing)
        movable_oil_water_range = 1.0 - swc - sorw  # type: ignore[operator]
        so_star_w = np.where(
            movable_oil_water_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - sorw) / movable_oil_water_range, 0.0, 1.0),
        )
        kro_w = _let_relperm(so_star_w, oil_water_L, oil_water_E, oil_water_T)

        movable_gas_oil_range = 1.0 - swc - sorg - sgr  # type: ignore[operator]
        so_star_g = np.where(
            movable_gas_oil_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - sorg) / movable_gas_oil_range, 0.0, 1.0),
        )
        kro_g = _let_relperm(so_star_g, gas_oil_L, gas_oil_E, gas_oil_T)

        kro_mixed = mixing_rule(
            kro_w=kro_w,
            kro_g=kro_g,
            krw=krw,
            krg=krg,
            kr_max=maximum_oil_relperm,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
        )
        kro = maximum_oil_relperm * np.clip(kro_mixed, 0.0, 1.0)

    elif wettability == Wettability.OIL_WET:
        # Oil is wetting, water becomes intermediate
        movable_oil_range = 1.0 - sorw - sorg  # type: ignore[operator]
        max_residual = np.minimum(sorw, sorg)
        so_star = np.where(
            movable_oil_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - max_residual) / movable_oil_range, 0.0, 1.0),
        )
        kro = maximum_oil_relperm * _let_relperm(so_star, oil_water_L, oil_water_E, oil_water_T)

        movable_gas_range = 1.0 - sgr - swc  # type: ignore[operator]
        sg_star = np.where(
            movable_gas_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - sgr) / movable_gas_range, 0.0, 1.0),
        )
        krg = maximum_gas_relperm * _let_relperm(sg_star, gas_L, gas_E, gas_T)

        movable_water_range_ow = 1.0 - swc - sorw  # type: ignore[operator]
        sw_star_ow = np.where(
            movable_water_range_ow <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - swc) / movable_water_range_ow, 0.0, 1.0),
        )
        krw_ow = _let_relperm(sw_star_ow, water_L, water_E, water_T)

        movable_water_range_gw = 1.0 - swc - sgr  # type: ignore[operator]
        sw_star_gw = np.where(
            movable_water_range_gw <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - swc) / movable_water_range_gw, 0.0, 1.0),
        )
        krw_gw = _let_relperm(sw_star_gw, water_L, water_E, water_T)

        krw_mixed = mixing_rule(
            kro_w=krw_ow,
            kro_g=krw_gw,
            krw=kro,
            krg=krg,
            kr_max=maximum_water_relperm,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
        )
        krw = maximum_water_relperm * np.clip(krw_mixed, 0.0, 1.0)

    elif wettability == Wettability.MIXED_WET:
        # Water-wet sub-system
        movable_water_range = 1.0 - swc - sorw  # type: ignore[operator]
        sw_star_ww = np.where(
            movable_water_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - swc) / movable_water_range, 0.0, 1.0),
        )
        krw_ww = maximum_water_relperm * _let_relperm(sw_star_ww, water_L, water_E, water_T)

        movable_gas_range = 1.0 - swc - sgr - sorg  # type: ignore[operator]
        sg_star_ww = np.where(
            movable_gas_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - sgr) / movable_gas_range, 0.0, 1.0),
        )
        krg_ww = maximum_gas_relperm * _let_relperm(sg_star_ww, gas_L, gas_E, gas_T)

        movable_oil_water_range = 1.0 - swc - sorw  # type: ignore[operator]
        so_star_w_ww = np.where(
            movable_oil_water_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - sorw) / movable_oil_water_range, 0.0, 1.0),
        )
        kro_w_ww = _let_relperm(so_star_w_ww, oil_water_L, oil_water_E, oil_water_T)

        movable_gas_oil_range = 1.0 - swc - sorg - sgr  # type: ignore[operator]
        so_star_g_ww = np.where(
            movable_gas_oil_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - sorg) / movable_gas_oil_range, 0.0, 1.0),
        )
        kro_g_ww = _let_relperm(so_star_g_ww, gas_oil_L, gas_oil_E, gas_oil_T)

        kro_ww = maximum_oil_relperm * np.clip(
            mixing_rule(
                kro_w=kro_w_ww,
                kro_g=kro_g_ww,
                krw=krw_ww,
                krg=krg_ww,
                kr_max=maximum_oil_relperm,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
            ),
            0.0,
            1.0,
        )

        # Oil-wet sub-system
        movable_oil_range_ow = 1.0 - sorw - sorg  # type: ignore[operator]
        max_residual_ow = np.minimum(sorw, sorg)
        so_star_ow = np.where(
            movable_oil_range_ow <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - max_residual_ow) / movable_oil_range_ow, 0.0, 1.0),
        )
        kro_ow = maximum_oil_relperm * _let_relperm(
            so_star_ow, oil_water_L, oil_water_E, oil_water_T
        )

        movable_gas_range_ow = 1.0 - sgr - swc  # type: ignore[operator]
        sg_star_ow = np.where(
            movable_gas_range_ow <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - sgr) / movable_gas_range_ow, 0.0, 1.0),
        )
        krg_ow = maximum_gas_relperm * _let_relperm(sg_star_ow, gas_L, gas_E, gas_T)

        movable_water_range_ow = 1.0 - swc - sorw  # type: ignore[operator]
        sw_star_ow = np.where(
            movable_water_range_ow <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - swc) / movable_water_range_ow, 0.0, 1.0),
        )
        krw_ow_proxy = _let_relperm(sw_star_ow, water_L, water_E, water_T)

        movable_water_range_gw = 1.0 - swc - sgr  # type: ignore[operator]
        sw_star_gw = np.where(
            movable_water_range_gw <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - swc) / movable_water_range_gw, 0.0, 1.0),
        )
        krw_gw_proxy = _let_relperm(sw_star_gw, water_L, water_E, water_T)

        krw_ow = maximum_water_relperm * np.clip(
            mixing_rule(
                kro_w=krw_ow_proxy,
                kro_g=krw_gw_proxy,
                krw=kro_ow,
                krg=krg_ow,
                kr_max=maximum_water_relperm,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
            ),
            0.0,
            1.0,
        )

        # Blend
        krw = mixed_wet_water_fraction * krw_ww + (1.0 - mixed_wet_water_fraction) * krw_ow
        kro = mixed_wet_water_fraction * kro_ww + (1.0 - mixed_wet_water_fraction) * kro_ow
        krg = mixed_wet_water_fraction * krg_ww + (1.0 - mixed_wet_water_fraction) * krg_ow

    else:
        raise ValidationError(f"Wettability {wettability!r} not implemented.")

    krw = np.clip(krw, 0.0, 1.0)
    kro = np.clip(kro, 0.0, 1.0)
    krg = np.clip(krg, 0.0, 1.0)
    krw = clamp_relperm(krw, minimum_water_relperm)
    kro = clamp_relperm(kro, minimum_oil_relperm)
    krg = clamp_relperm(krg, minimum_gas_relperm)
    if is_scalar:
        krw = krw.item()  # type: ignore
        kro = kro.item()  # type: ignore
        krg = krg.item()  # type: ignore
    return krw, kro, krg  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_let_curve_slope_wrt_normalized_saturation(
    normalized_saturation: NumberArray[NDimension],
    L: Number,
    E: Number,
    T: Number,
    kr_max: Number,
) -> NumberArray[NDimension]:
    """
    Analytical derivative of the LET relative permeability curve with
    respect to normalized (effective) saturation.

    The LET curve is:

    ``
    kr = kr_max * S*^L / (S*^L + E * (1-S*)^T)
    ``

    Applying the quotient rule and simplifying:

    ``
    dkr / dS* = kr_max * E * S*^(L-1) * (1-S*)^(T-1)
                * [L*(1-S*) + T*S*] / (S*^L + E*(1-S*)^T)^2
    ``

    The result is zero when the normalized saturation is exactly 0 or 1
    (boundary conditions).

    :param normalized_saturation: Effective saturation in [0, 1], clamped
        internally away from 0 and 1 to avoid power-law singularities.
    :param L: LET low-end curvature parameter (positive).
    :param E: LET elevation parameter (positive).
    :param T: LET high-end curvature parameter (positive).
    :param kr_max: Endpoint relative permeability.
    :return: Derivative array with the same shape as `normalized_saturation`.
    """
    s = np.clip(normalized_saturation, 1e-15, 1.0 - 1e-15)
    denominator = s**L + E * (1.0 - s) ** T
    safe_denominator = np.where(denominator > 1e-30, denominator, 1e-30)
    numerator = E * (s ** (L - 1.0)) * ((1.0 - s) ** (T - 1.0)) * (L * (1.0 - s) + T * s)
    slope = kr_max * numerator / (safe_denominator**2)
    slope = np.where(normalized_saturation <= 0.0, 0.0, slope)
    slope = np.where(normalized_saturation >= 1.0, 0.0, slope)
    return slope  # type: ignore[return-value]


@attrs.frozen(slots=True)
class LETThreePhaseRelPermTable(RelativePermeabilityTable):
    """
    Implements the LET (Lomeland-Ebeltoft-Thomas) three-phase relative permeability model.

    Uses the LET correlation for two-phase relative permeability curves and a
    configurable mixing rule for three-phase oil relative permeability. The LET
    model provides more curve-fitting flexibility than Brooks-Corey by using
    three shape parameters (L, E, T) per phase-pair instead of a single Corey
    exponent.

    Each phase-pair is described by a `LETParameters` instance that groups the
    L, E, and T values:

    - `water`: Parameters for the water kr curve (wetting phase in water-wet).
    - `oil_water`: Parameters for oil kr in the oil-water two-phase system.
    - `gas_oil`: Parameters for oil kr in the gas-oil two-phase system.
    - `gas`: Parameters for the gas kr curve (non-wetting phase).

    Supports water-wet and oil-wet wettability assumptions. Supports both
    scalar and array inputs for saturations (`supports_vector=True`).

    **Minimum relperm min_values** (`minimum_water_relperm`, `minimum_oil_relperm`,
    `minimum_gas_relperm`): same semantics as `BrooksCoreyRelPermTable`.
    """

    irreducible_water_saturation: Number | None = None
    """(Default) Irreducible water saturation (swc)."""

    residual_oil_saturation_water: Number | None = None
    """(Default) Residual oil saturation after water flood (sorw)."""

    residual_oil_saturation_gas: Number | None = None
    """(Default) Residual oil saturation after gas flood (sorg)."""

    residual_gas_saturation: Number | None = None
    """(Default) Residual gas saturation (sgr)."""

    water: LETParameters = LETParameters()
    """LET parameters for the water relative permeability curve."""

    oil_water: LETParameters = LETParameters()
    """LET parameters for oil relative permeability in the water-oil system."""

    gas_oil: LETParameters = LETParameters()
    """LET parameters for oil relative permeability in the gas-oil system."""

    gas: LETParameters = LETParameters()
    """LET parameters for the gas relative permeability curve."""

    maximum_water_relperm: Number = 1.0
    """Endpoint (maximum) relative permeability for water."""

    maximum_oil_relperm: Number = 1.0
    """Endpoint (maximum) relative permeability for oil."""

    maximum_gas_relperm: Number = 1.0
    """Endpoint (maximum) relative permeability for gas."""

    wettability: Wettability = Wettability.WATER_WET
    """Wettability type (water-wet or oil-wet)."""

    mixed_wet_water_fraction: Number = 0.5
    """Fraction of pore space that is water-wet in mixed-wet systems (0-1)."""

    mixing_rule: MixingRule | str = eclipse_rule
    """
    Mixing rule function or name to compute oil relative permeability in
    three-phase system. Accepts a function or a registered name string.
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
        covers both).
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
        Compute relative permeabilities for water, oil, and gas using the
        LET correlation.

        Supports both scalar and array inputs for saturations.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param irreducible_water_saturation: Optional override for swc.
        :param residual_oil_saturation_water: Optional override for sorw.
        :param residual_oil_saturation_gas: Optional override for sorg.
        :param residual_gas_saturation: Optional override for sgr.
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

        krw, kro, krg = compute_let_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
            residual_gas_saturation=sgr,  # type: ignore[arg-type]
            water_L=self.water.L,
            water_E=self.water.E,
            water_T=self.water.T,
            oil_water_L=self.oil_water.L,
            oil_water_E=self.oil_water.E,
            oil_water_T=self.oil_water.T,
            gas_oil_L=self.gas_oil.L,
            gas_oil_E=self.gas_oil.E,
            gas_oil_T=self.gas_oil.T,
            gas_L=self.gas.L,
            gas_E=self.gas.E,
            gas_T=self.gas.T,
            maximum_water_relperm=self.get_water_relperm_endpoint(),
            maximum_oil_relperm=self.get_oil_relperm_endpoint(),
            maximum_gas_relperm=self.get_gas_relperm_endpoint(),
            wettability=self.wettability,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
            mixing_rule=typing.cast(MixingRule, self.mixing_rule),
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
        gas saturation using the LET model.

        Returns a dictionary containing:

        ``
        (dkrw/dsw, dkrw/dso, dkrw/dsg,
        dkro/dsw, dkro/dso, dkro/dsg,
        dkrg/dsw, dkrg/dso, dkrg/dsg)
        ``

        For the water-wet case all LET curve derivatives are computed
        analytically via the closed-form quotient-rule formula (see
        `compute_let_curve_slope_wrt_normalized_saturation`).
        The chain rule propagates these through the effective saturation
        normalisation to give derivatives with respect to physical saturation.
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

        wettability = self.wettability
        mixing_rule = typing.cast(MixingRule, self.mixing_rule)
        water_params = self.water
        oil_water_params = self.oil_water
        gas_oil_params = self.gas_oil
        gas_params = self.gas
        krw_max = self.get_water_relperm_endpoint()
        kro_max = self.get_oil_relperm_endpoint()
        krg_max = self.get_gas_relperm_endpoint()

        # Resolve min_values once up front
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

        if wettability == Wettability.OIL_WET:
            # kro (wetting, depends on So)
            movable_oil_range = 1.0 - sorw - sorg  # type: ignore[operator]
            max_residual = np.minimum(sorw, sorg)  # type: ignore
            valid_oil = movable_oil_range > minimum_mobile_pore_space
            se_o = np.clip(
                (so - max_residual) / np.where(valid_oil, movable_oil_range, 1.0),
                0.0,
                1.0,
            )
            kro_raw = kro_max * _let_relperm(
                se_o,  # type: ignore[arg-type]
                oil_water_params.L,
                oil_water_params.E,
                oil_water_params.T,
            )

            dkro_dso_raw = np.where(
                valid_oil,
                compute_let_curve_slope_wrt_normalized_saturation(
                    se_o,
                    oil_water_params.L,
                    oil_water_params.E,
                    oil_water_params.T,
                    kro_max,
                )
                / movable_oil_range,
                zeros,
            )
            dkro_dso = clamp_relperm_derivative(dkro_dso_raw, kro_raw, kro_min)
            dkro_dsw = zeros.copy()
            dkro_dsg = zeros.copy()

            # krg (non-wetting, depends on Sg)
            movable_gas_range = 1.0 - sgr - swc  # type: ignore[operator]
            valid_gas = movable_gas_range > minimum_mobile_pore_space
            se_g = np.clip(
                (sg - sgr) / np.where(valid_gas, movable_gas_range, 1.0),
                0.0,
                1.0,
            )
            krg_raw = krg_max * _let_relperm(
                se_g,  # type: ignore[arg-type]
                gas_params.L,
                gas_params.E,
                gas_params.T,
            )

            dkrg_dsg_raw = np.where(
                valid_gas,
                compute_let_curve_slope_wrt_normalized_saturation(
                    se_g, gas_params.L, gas_params.E, gas_params.T, krg_max
                )
                / movable_gas_range,
                zeros,
            )
            dkrg_dsg = clamp_relperm_derivative(dkrg_dsg_raw, krg_raw, krg_min)
            dkrg_dsw = zeros.copy()
            dkrg_dso = zeros.copy()

            # krw (intermediate phase, via mixing rule on two-phase water proxies)
            movable_water_range_ow = 1.0 - swc - sorw  # type: ignore[operator]
            valid_water_ow = movable_water_range_ow > minimum_mobile_pore_space
            se_w_ow = np.clip(
                (sw - swc) / np.where(valid_water_ow, movable_water_range_ow, 1.0),
                0.0,
                1.0,
            )
            krw_ow = _let_relperm(
                se_w_ow,  # type: ignore[arg-type]
                water_params.L,
                water_params.E,
                water_params.T,
            )

            movable_water_range_gw = 1.0 - swc - sgr  # type: ignore[operator]
            valid_water_gw = movable_water_range_gw > minimum_mobile_pore_space
            se_w_gw = np.clip(
                (sw - swc) / np.where(valid_water_gw, movable_water_range_gw, 1.0),
                0.0,
                1.0,
            )
            krw_gw = _let_relperm(
                se_w_gw,  # type: ignore[arg-type]
                water_params.L,
                water_params.E,
                water_params.T,
            )

            # d(krw_ow)/dsw - depends only on Sw
            dkrw_ow_dsw = np.where(
                valid_water_ow,
                compute_let_curve_slope_wrt_normalized_saturation(
                    se_w_ow, water_params.L, water_params.E, water_params.T, 1.0
                )
                / movable_water_range_ow,
                zeros,
            )
            # d(krw_gw)/dsw - depends only on Sw
            dkrw_gw_dsw = np.where(
                valid_water_gw,
                compute_let_curve_slope_wrt_normalized_saturation(
                    se_w_gw, water_params.L, water_params.E, water_params.T, 1.0
                )
                / movable_water_range_gw,
                zeros,
            )
            # krw_ow and krw_gw have no So or Sg dependence
            dkrw_ow_dso = zeros.copy()
            dkrw_ow_dsg = zeros.copy()
            dkrw_gw_dso = zeros.copy()
            dkrw_gw_dsg = zeros.copy()

            derivatives = get_mixing_rule_partial_derivatives(
                rule=mixing_rule,
                kro_w=krw_ow,
                kro_g=krw_gw,
                krw=kro_raw,
                krg=krg_raw,
                kr_max=krw_max,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                epsilon=c.FINITE_DIFFERENCE_EPSILON,
            )
            dkrw_dkrw_ow = derivatives["dkro_dkro_w"]
            dkrw_dkrw_gw = derivatives["dkro_dkro_g"]
            dkrw_dsw_explicit = derivatives["dkro_dsw_explicit"]
            dkrw_dso_explicit = derivatives["dkro_dso_explicit"]
            dkrw_dsg_explicit = derivatives["dkro_dsg_explicit"]

            # Forward evaluate krw for min_value masking
            krw_raw = krw_max * np.clip(
                mixing_rule(
                    kro_w=krw_ow,
                    kro_g=krw_gw,
                    krw=kro_raw,
                    krg=krg_raw,
                    kr_max=krw_max,
                    water_saturation=sw,
                    oil_saturation=so,
                    gas_saturation=sg,
                ),
                0.0,
                1.0,
            )

            dkrw_dsw_raw = krw_max * (
                dkrw_dkrw_ow * dkrw_ow_dsw + dkrw_dkrw_gw * dkrw_gw_dsw + dkrw_dsw_explicit
            )
            dkrw_dso_raw = krw_max * (
                dkrw_dkrw_ow * dkrw_ow_dso + dkrw_dkrw_gw * dkrw_gw_dso + dkrw_dso_explicit
            )
            dkrw_dsg_raw = krw_max * (
                dkrw_dkrw_ow * dkrw_ow_dsg + dkrw_dkrw_gw * dkrw_gw_dsg + dkrw_dsg_explicit
            )
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
            mobile_water_range = 1.0 - swc - sorw  # type: ignore[operator]
            valid_water = mobile_water_range > minimum_mobile_pore_space
            se_w = np.clip((sw - swc) / np.where(valid_water, mobile_water_range, 1.0), 0.0, 1.0)
            krw_ww = krw_max * _let_relperm(
                se_w,  # type: ignore[arg-type]
                water_params.L,
                water_params.E,
                water_params.T,
            )
            dkrw_ww_dsw_raw = np.where(
                valid_water,
                compute_let_curve_slope_wrt_normalized_saturation(
                    se_w, water_params.L, water_params.E, water_params.T, krw_max
                )
                / mobile_water_range,
                zeros,
            )

            mobile_gas_range = 1.0 - swc - sgr - sorg  # type: ignore[operator]
            valid_gas = mobile_gas_range > minimum_mobile_pore_space
            se_g = np.clip((sg - sgr) / np.where(valid_gas, mobile_gas_range, 1.0), 0.0, 1.0)
            krg_ww = krg_max * _let_relperm(
                se_g,  # type: ignore[arg-type]
                gas_params.L,
                gas_params.E,
                gas_params.T,
            )
            dkrg_ww_dsg_raw = np.where(
                valid_gas,
                compute_let_curve_slope_wrt_normalized_saturation(
                    se_g, gas_params.L, gas_params.E, gas_params.T, krg_max
                )
                / mobile_gas_range,
                zeros,
            )

            # Two-phase oil inputs (unit-endpoint) and their So-derivatives
            mobile_oil_water_range = 1.0 - swc - sorw  # type: ignore[operator]
            valid_ow = mobile_oil_water_range > minimum_mobile_pore_space
            se_o_w = np.clip(
                (so - sorw) / np.where(valid_ow, mobile_oil_water_range, 1.0), 0.0, 1.0
            )
            kro_w_ww = _let_relperm(
                se_o_w,  # type: ignore[arg-type]
                oil_water_params.L,
                oil_water_params.E,
                oil_water_params.T,
            )
            dkro_w_ww_dso = np.where(
                valid_ow,
                compute_let_curve_slope_wrt_normalized_saturation(
                    se_o_w,
                    oil_water_params.L,
                    oil_water_params.E,
                    oil_water_params.T,
                    1.0,
                )
                / mobile_oil_water_range,
                zeros,
            )

            mobile_gas_oil_range = 1.0 - swc - sorg - sgr  # type: ignore[operator]
            valid_go = mobile_gas_oil_range > minimum_mobile_pore_space
            se_o_g = np.clip((so - sorg) / np.where(valid_go, mobile_gas_oil_range, 1.0), 0.0, 1.0)
            kro_g_ww = _let_relperm(
                se_o_g,  # type: ignore[arg-type]
                gas_oil_params.L,
                gas_oil_params.E,
                gas_oil_params.T,
            )
            dkro_g_ww_dso = np.where(
                valid_go,
                compute_let_curve_slope_wrt_normalized_saturation(
                    se_o_g, gas_oil_params.L, gas_oil_params.E, gas_oil_params.T, 1.0
                )
                / mobile_gas_oil_range,
                zeros,
            )

            derivs_ww = get_mixing_rule_partial_derivatives(
                rule=mixing_rule,
                kro_w=kro_w_ww,
                kro_g=kro_g_ww,
                krw=krw_ww,
                krg=krg_ww,
                kr_max=kro_max,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                epsilon=c.FINITE_DIFFERENCE_EPSILON,
            )
            kro_ww_raw = kro_max * np.clip(
                mixing_rule(
                    kro_w=kro_w_ww,
                    kro_g=kro_g_ww,
                    krw=krw_ww,
                    krg=krg_ww,
                    kr_max=kro_max,
                    water_saturation=sw,
                    oil_saturation=so,
                    gas_saturation=sg,
                ),
                0.0,
                1.0,
            )
            dkro_ww_dsw = kro_max * derivs_ww["dkro_dsw_explicit"]
            dkro_ww_dso = kro_max * (
                derivs_ww["dkro_dkro_w"] * dkro_w_ww_dso
                + derivs_ww["dkro_dkro_g"] * dkro_g_ww_dso
                + derivs_ww["dkro_dso_explicit"]
            )
            dkro_ww_dsg = kro_max * derivs_ww["dkro_dsg_explicit"]

            # Oil-wet sub-system
            movable_oil_range_ow = 1.0 - sorw - sorg  # type: ignore[operator]
            max_residual_ow = np.minimum(sorw, sorg)  # type: ignore[operator]
            valid_oil_ow = movable_oil_range_ow > minimum_mobile_pore_space
            se_o_ow = np.clip(
                (so - max_residual_ow) / np.where(valid_oil_ow, movable_oil_range_ow, 1.0),
                0.0,
                1.0,
            )
            kro_ow = kro_max * _let_relperm(
                se_o_ow,  # type: ignore[arg-type]
                oil_water_params.L,
                oil_water_params.E,
                oil_water_params.T,
            )
            dkro_ow_dso = np.where(
                valid_oil_ow,
                kro_max
                * compute_let_curve_slope_wrt_normalized_saturation(
                    se_o_ow,
                    oil_water_params.L,
                    oil_water_params.E,
                    oil_water_params.T,
                    1.0,
                )
                / movable_oil_range_ow,
                zeros,
            )

            movable_gas_range_ow = 1.0 - sgr - swc  # type: ignore[operator]
            valid_gas_ow = movable_gas_range_ow > minimum_mobile_pore_space
            se_g_ow = np.clip(
                (sg - sgr) / np.where(valid_gas_ow, movable_gas_range_ow, 1.0), 0.0, 1.0
            )
            krg_ow = krg_max * _let_relperm(
                se_g_ow,  # type: ignore[arg-type]
                gas_params.L,
                gas_params.E,
                gas_params.T,
            )
            dkrg_ow_dsg = np.where(
                valid_gas_ow,
                krg_max
                * compute_let_curve_slope_wrt_normalized_saturation(
                    se_g_ow, gas_params.L, gas_params.E, gas_params.T, 1.0
                )
                / movable_gas_range_ow,
                zeros,
            )

            # krw_ow proxies and their Sw-derivatives
            movable_water_range_ow = 1.0 - swc - sorw  # type: ignore[operator]
            valid_w_ow = movable_water_range_ow > minimum_mobile_pore_space
            se_w_ow = np.clip(
                (sw - swc) / np.where(valid_w_ow, movable_water_range_ow, 1.0), 0.0, 1.0
            )
            krw_ow_proxy = _let_relperm(
                se_w_ow,  # type: ignore[arg-type]
                water_params.L,
                water_params.E,
                water_params.T,
            )
            dkrw_ow_proxy_dsw = np.where(
                valid_w_ow,
                compute_let_curve_slope_wrt_normalized_saturation(
                    se_w_ow, water_params.L, water_params.E, water_params.T, 1.0
                )
                / movable_water_range_ow,
                zeros,
            )

            movable_water_range_gw = 1.0 - swc - sgr  # type: ignore[operator]
            valid_w_gw = movable_water_range_gw > minimum_mobile_pore_space
            se_w_gw = np.clip(
                (sw - swc) / np.where(valid_w_gw, movable_water_range_gw, 1.0), 0.0, 1.0
            )
            krw_gw_proxy = _let_relperm(
                se_w_gw,  # type: ignore[arg-type]
                water_params.L,
                water_params.E,
                water_params.T,
            )
            dkrw_gw_proxy_dsw = np.where(
                valid_w_gw,
                compute_let_curve_slope_wrt_normalized_saturation(
                    se_w_gw, water_params.L, water_params.E, water_params.T, 1.0
                )
                / movable_water_range_gw,
                zeros,
            )

            derivs_ow = get_mixing_rule_partial_derivatives(
                rule=mixing_rule,
                kro_w=krw_ow_proxy,
                kro_g=krw_gw_proxy,
                krw=kro_ow,
                krg=krg_ow,
                kr_max=krw_max,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                epsilon=c.FINITE_DIFFERENCE_EPSILON,
            )
            krw_ow_raw = krw_max * np.clip(
                mixing_rule(
                    kro_w=krw_ow_proxy,
                    kro_g=krw_gw_proxy,
                    krw=kro_ow,
                    krg=krg_ow,
                    kr_max=krw_max,
                    water_saturation=sw,
                    oil_saturation=so,
                    gas_saturation=sg,
                ),
                0.0,
                1.0,
            )
            dkrw_ow_dsw = krw_max * (
                derivs_ow["dkro_dkro_w"] * dkrw_ow_proxy_dsw
                + derivs_ow["dkro_dkro_g"] * dkrw_gw_proxy_dsw
                + derivs_ow["dkro_dsw_explicit"]
            )
            dkrw_ow_dso = krw_max * derivs_ow["dkro_dso_explicit"]
            dkrw_ow_dsg = krw_max * derivs_ow["dkro_dsg_explicit"]

            # Blend raw kr values for min_value masking
            krw_blend_raw = f * krw_ww + (1.0 - f) * krw_ow_raw
            kro_blend_raw = f * kro_ww_raw + (1.0 - f) * kro_ow
            krg_blend_raw = f * krg_ww + (1.0 - f) * krg_ow

            # Blend derivatives
            dkrw_dsw_raw = f * dkrw_ww_dsw_raw + (1.0 - f) * dkrw_ow_dsw
            dkrw_dso_raw = (1.0 - f) * dkrw_ow_dso
            dkrw_dsg_raw = (1.0 - f) * dkrw_ow_dsg

            dkro_dsw_raw = f * dkro_ww_dsw
            dkro_dso_raw = f * dkro_ww_dso + (1.0 - f) * dkro_ow_dso
            dkro_dsg_raw = f * dkro_ww_dsg

            dkrg_dsg_raw = f * dkrg_ww_dsg_raw + (1.0 - f) * dkrg_ow_dsg

            # Apply min_values to blended derivatives
            dkrw_dsw = clamp_relperm_derivative(dkrw_dsw_raw, krw_blend_raw, krw_min)
            dkrw_dso = clamp_relperm_derivative(dkrw_dso_raw, krw_blend_raw, krw_min)
            dkrw_dsg = clamp_relperm_derivative(dkrw_dsg_raw, krw_blend_raw, krw_min)
            dkro_dsw = clamp_relperm_derivative(dkro_dsw_raw, kro_blend_raw, kro_min)
            dkro_dso = clamp_relperm_derivative(dkro_dso_raw, kro_blend_raw, kro_min)
            dkro_dsg = clamp_relperm_derivative(dkro_dsg_raw, kro_blend_raw, kro_min)
            dkrg_dsw = zeros.copy()
            dkrg_dso = zeros.copy()
            dkrg_dsg = clamp_relperm_derivative(dkrg_dsg_raw, krg_blend_raw, krg_min)

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
                results = tuple(r.item() for r in results)  # type: ignore
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
        # krw
        mobile_water_range = 1.0 - swc - sorw  # type: ignore[operator]
        valid_water = mobile_water_range > minimum_mobile_pore_space
        se_w_for_krw = np.clip(
            (sw - swc) / np.where(valid_water, mobile_water_range, 1.0),
            0.0,
            1.0,
        )
        krw_raw = krw_max * _let_relperm(
            se_w_for_krw,  # type: ignore[arg-type]
            water_params.L,
            water_params.E,
            water_params.T,
        )
        dkrw_dsw_raw = np.where(
            valid_water,
            compute_let_curve_slope_wrt_normalized_saturation(
                normalized_saturation=se_w_for_krw,
                L=water_params.L,
                E=water_params.E,
                T=water_params.T,
                kr_max=krw_max,
            )
            / mobile_water_range,
            zeros,
        )
        dkrw_dsw = clamp_relperm_derivative(dkrw_dsw_raw, krw_raw, krw_min)
        dkrw_dso = zeros.copy()
        dkrw_dsg = zeros.copy()

        # krg
        mobile_gas_range = 1.0 - swc - sgr - sorg  # type: ignore[operator]
        valid_gas = mobile_gas_range > minimum_mobile_pore_space
        se_g_for_krg = np.clip(
            (sg - sgr) / np.where(valid_gas, mobile_gas_range, 1.0),
            0.0,
            1.0,
        )
        krg_raw = krg_max * _let_relperm(
            se_g_for_krg,  # type: ignore[arg-type]
            gas_params.L,
            gas_params.E,
            gas_params.T,
        )
        dkrg_dsg_raw = np.where(
            valid_gas,
            compute_let_curve_slope_wrt_normalized_saturation(
                normalized_saturation=se_g_for_krg,
                L=gas_params.L,
                E=gas_params.E,
                T=gas_params.T,
                kr_max=krg_max,
            )
            / mobile_gas_range,
            zeros,
        )
        dkrg_dsg = clamp_relperm_derivative(dkrg_dsg_raw, krg_raw, krg_min)
        dkrg_dsw = zeros.copy()
        dkrg_dso = zeros.copy()

        # kro_w (unit-endpoint oil kr from water-oil system, function of So)
        mobile_oil_water_range = 1.0 - swc - sorw  # type: ignore[operator]
        valid_oil_water = mobile_oil_water_range > minimum_mobile_pore_space
        se_o_water_system = np.clip(
            (so - sorw) / np.where(valid_oil_water, mobile_oil_water_range, 1.0),
            0.0,
            1.0,
        )
        kro_w_vals = _let_relperm(
            se_o_water_system,  # type: ignore[arg-type]
            oil_water_params.L,
            oil_water_params.E,
            oil_water_params.T,
        )
        dkro_w_dso = np.where(
            valid_oil_water,
            compute_let_curve_slope_wrt_normalized_saturation(
                normalized_saturation=se_o_water_system,
                L=oil_water_params.L,
                E=oil_water_params.E,
                T=oil_water_params.T,
                kr_max=1.0,
            )
            / mobile_oil_water_range,
            zeros,
        )
        dkro_w_dsw = zeros.copy()
        dkro_w_dsg = zeros.copy()

        # kro_g (unit-endpoint oil kr from gas-oil system, function of So)
        mobile_gas_oil_range = 1.0 - swc - sorg - sgr  # type: ignore
        valid_gas_oil = mobile_gas_oil_range > minimum_mobile_pore_space
        se_o_gas_system = np.clip(
            (so - sorg) / np.where(valid_gas_oil, mobile_gas_oil_range, 1.0),
            0.0,
            1.0,
        )
        kro_g_vals = _let_relperm(
            se_o_gas_system,  # type: ignore[arg-type]
            gas_oil_params.L,
            gas_oil_params.E,
            gas_oil_params.T,
        )
        dkro_g_dso = np.where(
            valid_gas_oil,
            compute_let_curve_slope_wrt_normalized_saturation(
                normalized_saturation=se_o_gas_system,
                L=gas_oil_params.L,
                E=gas_oil_params.E,
                T=gas_oil_params.T,
                kr_max=1.0,
            )
            / mobile_gas_oil_range,
            zeros,
        )
        dkro_g_dsw = zeros.copy()
        dkro_g_dsg = zeros.copy()

        derivatives = get_mixing_rule_partial_derivatives(
            rule=mixing_rule,
            kro_w=kro_w_vals,
            kro_g=kro_g_vals,
            krw=krw_raw,
            krg=krg_raw,
            kr_max=kro_max,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
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
        kro_mixed_raw = kro_max * np.clip(
            mixing_rule(
                kro_w=kro_w_vals,
                kro_g=kro_g_vals,
                krw=krw_raw,
                krg=krg_raw,
                kr_max=kro_max,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
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

        dkro_dsw = clamp_relperm_derivative(dkro_dsw_raw, kro_mixed_raw, kro_min)
        dkro_dso = clamp_relperm_derivative(dkro_dso_raw, kro_mixed_raw, kro_min)
        dkro_dsg = clamp_relperm_derivative(dkro_dsg_raw, kro_mixed_raw, kro_min)

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
            results = tuple(r.item() for r in results)  # type: ignore
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

"""Shared building blocks for the analytical three-phase relative permeability derivatives."""

import typing

import numpy as np

from bores.blackoil.satfunc.relperm.mixing_rules import (
    MixingRule,
    MixingRulePartialDerivatives,
    get_mixing_rule_partial_derivatives,
)
from bores.blackoil.satfunc.relperm.tables import clamp_relperm_derivative, show_invalid_saturation
from bores.errors import ValidationError
from bores.types import (
    Boolean,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    RelativePermeabilityDerivatives,
)

__all__ = [
    "WettabilitySystem",
    "apply_oil_wet_mixing_rule",
    "apply_water_wet_mixing_rule",
    "blend_systems",
    "broadcast_saturations",
    "build_relperm_derivatives",
    "finalize_mixed_wet_derivatives",
    "finalize_oil_wet_derivatives",
    "finalize_water_wet_derivatives",
    "to_scalar",
    "validate_and_normalize_saturations",
]


class WettabilitySystem(typing.NamedTuple):
    """
    Relative permeabilities and their partial derivatives for one wettability
    system, before the minimum relative permeability is applied.

    A water-wet system and an oil-wet system are each evaluated on their own.
    A mixed-wet rock blends the two, and a purely water-wet or oil-wet rock
    uses just one of them.

    Derivatives that are always zero for the system are stored as zero arrays.
    The derivatives of gas relative permeability with respect to water and oil
    saturation are not stored because gas relative permeability depends on gas
    saturation only in every supported model.
    """

    krw: NumberArray[typing.Any]
    """Water relative permeability."""

    kro: NumberArray[typing.Any]
    """Oil relative permeability."""

    krg: NumberArray[typing.Any]
    """Gas relative permeability."""

    dkrw_dsw: NumberArray[typing.Any]
    """Derivative of water relative permeability with respect to water saturation."""

    dkrw_dso: NumberArray[typing.Any]
    """Derivative of water relative permeability with respect to oil saturation."""

    dkrw_dsg: NumberArray[typing.Any]
    """Derivative of water relative permeability with respect to gas saturation."""

    dkro_dsw: NumberArray[typing.Any]
    """Derivative of oil relative permeability with respect to water saturation."""

    dkro_dso: NumberArray[typing.Any]
    """Derivative of oil relative permeability with respect to oil saturation."""

    dkro_dsg: NumberArray[typing.Any]
    """Derivative of oil relative permeability with respect to gas saturation."""

    dkrg_dsg: NumberArray[typing.Any]
    """Derivative of gas relative permeability with respect to gas saturation."""


def broadcast_saturations(
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    *endpoint_saturations: NumberOrArray[NDimension],
) -> tuple[
    NumberArray[NDimension],
    NumberArray[NDimension],
    NumberArray[NDimension],
    NumberArray[NDimension],
    Boolean,
]:
    """
    Bring the saturations to a common array shape.

    :param water_saturation: Water saturation, scalar or array.
    :param oil_saturation: Oil saturation, scalar or array.
    :param gas_saturation: Gas saturation, scalar or array.
    :param endpoint_saturations: Endpoint saturations (connate water, residual oil and
        residual gas). They only decide whether the call is a scalar call.
    :return: Tuple of water saturation, oil saturation, gas saturation, an array of
        zeros with their common shape, and whether every input was a scalar. Callers
        should convert results back to scalars when the last item is `True`.
    """
    is_scalar = (
        np.isscalar(water_saturation)
        and np.isscalar(oil_saturation)
        and np.isscalar(gas_saturation)
        and all(np.isscalar(saturation) for saturation in endpoint_saturations)
    )
    sw = np.atleast_1d(water_saturation)
    so = np.atleast_1d(oil_saturation)
    sg = np.atleast_1d(gas_saturation)
    sw, so, sg = np.broadcast_arrays(sw, so, sg)
    zeros = np.zeros_like(sw)
    return sw, so, sg, zeros, is_scalar  # type: ignore[return-value]


def validate_and_normalize_saturations(
    water_saturation: NumberArray[NDimension],
    oil_saturation: NumberArray[NDimension],
    gas_saturation: NumberArray[NDimension],
    saturation_epsilon: Number,
) -> tuple[NumberArray[NDimension], NumberArray[NDimension], NumberArray[NDimension]]:
    """
    Check that the saturations lie between 0 and 1, and scale them to sum to 1 where they differ
    from 1 by more than `saturation_epsilon`.

    :param water_saturation: Water saturation array.
    :param oil_saturation: Oil saturation array.
    :param gas_saturation: Gas saturation array.
    :param saturation_epsilon: Tolerance on the sum of the saturations.
    :raises ValidationError: If any saturation is outside the range 0 to 1.
    :return: Tuple of the water, oil and gas saturation arrays.
    """
    sw, so, sg = water_saturation, oil_saturation, gas_saturation
    if np.any((sw < 0) | (sw > 1) | (so < 0) | (so > 1) | (sg < 0) | (sg > 1)):
        raise ValidationError(
            f"Saturations must be between 0 and 1. Sw: {show_invalid_saturation(sw)}, "
            f"So: {show_invalid_saturation(so)}, Sg: {show_invalid_saturation(sg)}"
        )
    total_saturation = sw + so + sg
    needs_norm = (np.abs(total_saturation - 1.0) > saturation_epsilon) & (total_saturation > 0.0)
    if np.any(needs_norm):
        sw = np.where(needs_norm, sw / total_saturation, sw)
        so = np.where(needs_norm, so / total_saturation, so)
        sg = np.where(needs_norm, sg / total_saturation, sg)
    return sw, so, sg


def apply_water_wet_mixing_rule(
    *,
    mixing_rule: MixingRule,
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    finite_difference_epsilon: Number,
) -> tuple[NumberArray[NDimension], MixingRulePartialDerivatives]:
    """
    Evaluate the three-phase oil relative permeability mixing rule for a water-wet
    system, together with its partial derivatives.

    Oil is the intermediate phase, so the rule is evaluated with unit endpoint
    inputs and the result is scaled by the oil endpoint relative permeability by
    the caller.

    :param mixing_rule: Three-phase oil relative permeability mixing rule.
    :param kro_w: Oil relative permeability of the oil-water system (unit endpoint).
    :param kro_g: Oil relative permeability of the gas-oil system (unit endpoint).
    :param krw: Water relative permeability.
    :param krg: Gas relative permeability.
    :param water_saturation: Water saturation.
    :param oil_saturation: Oil saturation.
    :param gas_saturation: Gas saturation.
    :param irreducible_water_saturation: Irreducible water saturation.
    :param residual_oil_saturation_water: Residual oil saturation to water flooding.
    :param residual_oil_saturation_gas: Residual oil saturation to gas flooding.
    :param finite_difference_epsilon: Step used when the rule has no analytical derivative.
    :return: Tuple of the mixed oil relative permeability, limited to the range 0 to 1, and
        the partial derivatives of the rule.
    """
    partials = get_mixing_rule_partial_derivatives(
        rule=mixing_rule,
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        kr_max=1.0,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        irreducible_water_saturation=irreducible_water_saturation,
        residual_oil_saturation_water=residual_oil_saturation_water,
        residual_oil_saturation_gas=residual_oil_saturation_gas,
        epsilon=finite_difference_epsilon,
    )
    mixed = np.clip(
        mixing_rule(
            kro_w=kro_w,
            kro_g=kro_g,
            krw=krw,
            krg=krg,
            kr_max=1.0,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
        ),
        0.0,
        1.0,
    )
    return mixed, partials  # type: ignore[return-value]


def apply_oil_wet_mixing_rule(
    *,
    mixing_rule: MixingRule,
    krw_oil_water: NumberOrArray[NDimension],
    krw_gas_water: NumberOrArray[NDimension],
    kro: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    finite_difference_epsilon: Number,
) -> tuple[NumberArray[NDimension], MixingRulePartialDerivatives]:
    """
    Evaluate the mixing rule for an oil-wet system, together with its partial
    derivatives.

    Oil is the wetting phase and water is the intermediate phase, so the rule is
    evaluated with the roles of water and oil exchanged. The returned partial
    derivatives keep the rule's own names, which therefore refer to the exchanged
    roles:

    - `dkro_dkro_w` and `dkro_dkro_g` are the sensitivities of the mixed water
      relative permeability to `krw_oil_water` and `krw_gas_water`.
    - `dkro_dkrw` and `dkro_dkrg` are the sensitivities to oil and gas relative
      permeability.
    - `dkro_dso_explicit` is the explicit dependence on water saturation,
      `dkro_dsw_explicit` the explicit dependence on oil saturation, and
      `dkro_dsg_explicit` the explicit dependence on gas saturation.

    The result is scaled by the water endpoint relative permeability by the caller.

    :param mixing_rule: Three-phase mixing rule.
    :param krw_oil_water: Water relative permeability of the oil-water system (unit endpoint).
    :param krw_gas_water: Water relative permeability of the gas-water system (unit endpoint).
    :param kro: Oil relative permeability.
    :param krg: Gas relative permeability.
    :param water_saturation: Water saturation.
    :param oil_saturation: Oil saturation.
    :param gas_saturation: Gas saturation.
    :param irreducible_water_saturation: Irreducible water saturation.
    :param residual_oil_saturation_water: Residual oil saturation to water flooding.
    :param residual_oil_saturation_gas: Residual oil saturation to gas flooding.
    :param finite_difference_epsilon: Step used when the rule has no analytical derivative.
    :return: Tuple of the mixed water relative permeability, limited to the range 0 to 1,
        and the partial derivatives of the rule.
    """
    swapped_irreducible = np.minimum(residual_oil_saturation_water, residual_oil_saturation_gas)
    partials = get_mixing_rule_partial_derivatives(
        rule=mixing_rule,
        kro_w=krw_oil_water,
        kro_g=krw_gas_water,
        krw=kro,
        krg=krg,
        kr_max=1.0,
        water_saturation=oil_saturation,
        oil_saturation=water_saturation,
        gas_saturation=gas_saturation,
        irreducible_water_saturation=swapped_irreducible,  # type: ignore[arg-type]
        residual_oil_saturation_water=irreducible_water_saturation,
        residual_oil_saturation_gas=irreducible_water_saturation,
        epsilon=finite_difference_epsilon,
    )
    mixed = np.clip(
        mixing_rule(
            kro_w=krw_oil_water,
            kro_g=krw_gas_water,
            krw=kro,
            krg=krg,
            kr_max=1.0,
            water_saturation=oil_saturation,
            oil_saturation=water_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swapped_irreducible,  # type: ignore[arg-type]
            residual_oil_saturation_water=irreducible_water_saturation,
            residual_oil_saturation_gas=irreducible_water_saturation,
        ),
        0.0,
        1.0,
    )
    return mixed, partials  # type: ignore[return-value]


def to_scalar(value: NumberOrArray[NDimension]) -> Number:
    """
    Convert a one-element array to a Python scalar.

    :param value: One-element array or scalar.
    :return: The value as a scalar.
    """
    return np.asarray(value).item()


def blend_systems(
    *,
    water_wet_value: NumberArray[NDimension],
    oil_wet_value: NumberArray[NDimension],
    water_wet_fraction: Number,
) -> NumberArray[NDimension]:
    """
    Blend one quantity of the water-wet and oil-wet systems for a mixed-wet rock.

    :param water_wet_value: The quantity for the water-wet system.
    :param oil_wet_value: The quantity for the oil-wet system.
    :param water_wet_fraction: Fraction of the pore space that is water-wet, from 0 to 1.
    :return: The blended quantity.
    """
    return water_wet_fraction * water_wet_value + (1.0 - water_wet_fraction) * oil_wet_value  # type: ignore[return-value]


def build_relperm_derivatives(
    *,
    dkrw_dsw: NumberOrArray[NDimension],
    dkro_dsw: NumberOrArray[NDimension],
    dkrg_dsw: NumberOrArray[NDimension],
    dkrw_dso: NumberOrArray[NDimension],
    dkro_dso: NumberOrArray[NDimension],
    dkrg_dso: NumberOrArray[NDimension],
    dkrw_dsg: NumberOrArray[NDimension],
    dkro_dsg: NumberOrArray[NDimension],
    dkrg_dsg: NumberOrArray[NDimension],
    is_scalar: Boolean,
) -> RelativePermeabilityDerivatives:
    """
    Package the nine partial derivatives, converting each to a scalar for a scalar call.

    :param dkrw_dsw: Derivative of water relative permeability with respect to water saturation.
    :param dkro_dsw: Derivative of oil relative permeability with respect to water saturation.
    :param dkrg_dsw: Derivative of gas relative permeability with respect to water saturation.
    :param dkrw_dso: Derivative of water relative permeability with respect to oil saturation.
    :param dkro_dso: Derivative of oil relative permeability with respect to oil saturation.
    :param dkrg_dso: Derivative of gas relative permeability with respect to oil saturation.
    :param dkrw_dsg: Derivative of water relative permeability with respect to gas saturation.
    :param dkro_dsg: Derivative of oil relative permeability with respect to gas saturation.
    :param dkrg_dsg: Derivative of gas relative permeability with respect to gas saturation.
    :param is_scalar: Whether the saturations were scalars, in which case each derivative is
        returned as a scalar.
    :return: `RelativePermeabilityDerivatives` dictionary.
    """
    if is_scalar:
        dkrw_dsw = to_scalar(dkrw_dsw)
        dkro_dsw = to_scalar(dkro_dsw)
        dkrg_dsw = to_scalar(dkrg_dsw)
        dkrw_dso = to_scalar(dkrw_dso)
        dkro_dso = to_scalar(dkro_dso)
        dkrg_dso = to_scalar(dkrg_dso)
        dkrw_dsg = to_scalar(dkrw_dsg)
        dkro_dsg = to_scalar(dkro_dsg)
        dkrg_dsg = to_scalar(dkrg_dsg)
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


def finalize_water_wet_derivatives(
    *,
    system: WettabilitySystem,
    minimum_water_relperm: Number | None,
    minimum_oil_relperm: Number | None,
    minimum_gas_relperm: Number | None,
    zeros: NumberArray[NDimension],
    is_scalar: Boolean,
) -> RelativePermeabilityDerivatives:
    """
    Apply the minimum relative permeability to a purely water-wet system and package
    the derivatives.

    Water relative permeability depends on water saturation only and gas relative
    permeability on gas saturation only, so their other derivatives are zero.

    :param system: The water-wet system.
    :param minimum_water_relperm: Resolved minimum water relative permeability, or `None`.
    :param minimum_oil_relperm: Resolved minimum oil relative permeability, or `None`.
    :param minimum_gas_relperm: Resolved minimum gas relative permeability, or `None`.
    :param zeros: Array of zeros with the shape of the saturations.
    :param is_scalar: Whether the saturations were scalars.
    :return: `RelativePermeabilityDerivatives` dictionary.
    """
    return build_relperm_derivatives(
        dkrw_dsw=clamp_relperm_derivative(system.dkrw_dsw, system.krw, minimum_water_relperm),
        dkro_dsw=clamp_relperm_derivative(system.dkro_dsw, system.kro, minimum_oil_relperm),
        dkrg_dsw=zeros.copy(),
        dkrw_dso=zeros.copy(),
        dkro_dso=clamp_relperm_derivative(system.dkro_dso, system.kro, minimum_oil_relperm),
        dkrg_dso=zeros.copy(),
        dkrw_dsg=zeros.copy(),
        dkro_dsg=clamp_relperm_derivative(system.dkro_dsg, system.kro, minimum_oil_relperm),
        dkrg_dsg=clamp_relperm_derivative(system.dkrg_dsg, system.krg, minimum_gas_relperm),
        is_scalar=is_scalar,
    )


def finalize_oil_wet_derivatives(
    *,
    system: WettabilitySystem,
    minimum_water_relperm: Number | None,
    minimum_oil_relperm: Number | None,
    minimum_gas_relperm: Number | None,
    zeros: NumberArray[NDimension],
    is_scalar: Boolean,
) -> RelativePermeabilityDerivatives:
    """
    Apply the minimum relative permeability to a purely oil-wet system and package
    the derivatives.

    Oil relative permeability depends on oil saturation only and gas relative
    permeability on gas saturation only, so their other derivatives are zero.

    :param system: The oil-wet system.
    :param minimum_water_relperm: Resolved minimum water relative permeability, or `None`.
    :param minimum_oil_relperm: Resolved minimum oil relative permeability, or `None`.
    :param minimum_gas_relperm: Resolved minimum gas relative permeability, or `None`.
    :param zeros: Array of zeros with the shape of the saturations.
    :param is_scalar: Whether the saturations were scalars.
    :return: `RelativePermeabilityDerivatives` dictionary.
    """
    return build_relperm_derivatives(
        dkrw_dsw=clamp_relperm_derivative(system.dkrw_dsw, system.krw, minimum_water_relperm),
        dkro_dsw=zeros.copy(),
        dkrg_dsw=zeros.copy(),
        dkrw_dso=clamp_relperm_derivative(system.dkrw_dso, system.krw, minimum_water_relperm),
        dkro_dso=clamp_relperm_derivative(system.dkro_dso, system.kro, minimum_oil_relperm),
        dkrg_dso=zeros.copy(),
        dkrw_dsg=clamp_relperm_derivative(system.dkrw_dsg, system.krw, minimum_water_relperm),
        dkro_dsg=zeros.copy(),
        dkrg_dsg=clamp_relperm_derivative(system.dkrg_dsg, system.krg, minimum_gas_relperm),
        is_scalar=is_scalar,
    )


def finalize_mixed_wet_derivatives(
    *,
    water_wet: WettabilitySystem,
    oil_wet: WettabilitySystem,
    water_wet_fraction: Number,
    minimum_water_relperm: Number | None,
    minimum_oil_relperm: Number | None,
    minimum_gas_relperm: Number | None,
    zeros: NumberArray[NDimension],
    is_scalar: Boolean,
) -> RelativePermeabilityDerivatives:
    """
    Blend a water-wet and an oil-wet system for a mixed-wet rock, apply the minimum
    relative permeability to the blend, and package the derivatives.

    :param water_wet: The water-wet system.
    :param oil_wet: The oil-wet system.
    :param water_wet_fraction: Fraction of the pore space that is water-wet, from 0 to 1.
    :param minimum_water_relperm: Resolved minimum water relative permeability, or `None`.
    :param minimum_oil_relperm: Resolved minimum oil relative permeability, or `None`.
    :param minimum_gas_relperm: Resolved minimum gas relative permeability, or `None`.
    :param zeros: Array of zeros with the shape of the saturations.
    :param is_scalar: Whether the saturations were scalars.
    :return: `RelativePermeabilityDerivatives` dictionary.
    """

    def blend(
        water_wet_value: NumberArray[typing.Any], oil_wet_value: NumberArray[typing.Any]
    ) -> NumberArray[typing.Any]:
        return blend_systems(
            water_wet_value=water_wet_value,
            oil_wet_value=oil_wet_value,
            water_wet_fraction=water_wet_fraction,
        )

    krw = blend(water_wet.krw, oil_wet.krw)
    kro = blend(water_wet.kro, oil_wet.kro)
    krg = blend(water_wet.krg, oil_wet.krg)

    return build_relperm_derivatives(
        dkrw_dsw=clamp_relperm_derivative(
            blend(water_wet.dkrw_dsw, oil_wet.dkrw_dsw), krw, minimum_water_relperm
        ),
        dkro_dsw=clamp_relperm_derivative(
            blend(water_wet.dkro_dsw, oil_wet.dkro_dsw), kro, minimum_oil_relperm
        ),
        dkrg_dsw=zeros.copy(),
        dkrw_dso=clamp_relperm_derivative(
            blend(water_wet.dkrw_dso, oil_wet.dkrw_dso), krw, minimum_water_relperm
        ),
        dkro_dso=clamp_relperm_derivative(
            blend(water_wet.dkro_dso, oil_wet.dkro_dso), kro, minimum_oil_relperm
        ),
        dkrg_dso=zeros.copy(),
        dkrw_dsg=clamp_relperm_derivative(
            blend(water_wet.dkrw_dsg, oil_wet.dkrw_dsg), krw, minimum_water_relperm
        ),
        dkro_dsg=clamp_relperm_derivative(
            blend(water_wet.dkro_dsg, oil_wet.dkro_dsg), kro, minimum_oil_relperm
        ),
        dkrg_dsg=clamp_relperm_derivative(
            blend(water_wet.dkrg_dsg, oil_wet.dkrg_dsg), krg, minimum_gas_relperm
        ),
        is_scalar=is_scalar,
    )

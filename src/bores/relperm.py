"""Relative permeability models and mixing rules for multi-phase flow simulations."""

import threading
import typing

import attrs
import numba
import numpy as np
import numpy.typing as npt

from bores.constants import c
from bores.errors import ValidationError
from bores.grids.base import array as bores_array
from bores.serialization import Serializable, make_serializable_type_registrar
from bores.stores import StoreSerializable
from bores.types import (
    FloatOrArray,
    FluidPhase,
    MixingRule,
    RelativePermeabilities,
    RelativePermeabilityDerivatives,
    T,
    Wettability,
)
from bores.utils import piecewise_linear_slope

__all__ = [
    "BrooksCoreyThreePhaseRelPermModel",
    "LETParameters",
    "LETThreePhaseRelPermModel",
    "ThreePhaseRelPermTable",
    "TwoPhaseRelPermTable",
    "arithmetic_mean_rule",
    "aziz_settari_rule",
    "baker_linear_rule",
    "blunt_rule",
    "compute_corey_three_phase_relative_permeabilities",
    "compute_let_three_phase_relative_permeabilities",
    "eclipse_rule",
    "geometric_mean_rule",
    "get_relperm_table",
    "harmonic_mean_rule",
    "hustad_hansen_rule",
    "linear_interpolation_rule",
    "list_relperm_tables",
    "max_rule",
    "max_rule",
    "mixing_rule",
    "product_saturation_weighted_rule",
    "relperm_table",
    "saturation_weighted_interpolation_rule",
    "stone_II_rule",
    "stone_I_rule",
]


"""
Comparison of common three-phase relative permeability mixing rules:

| Rule                | Conservativeness     | Complexity  | Typical Use Case                            |
|---------------------|----------------------|-------------|---------------------------------------------|
| Min                 | Very conservative    | Simple      | Lower bound, safety factor                  |
| Harmonic Mean       | Very conservative    | Simple      | Series flow, tight rocks                    |
| Geometric Mean      | Conservative         | Simple      | General purpose                             |
| Stone I             | Moderate             | Moderate    | Water-wet systems                           |
| Stone II            | Moderate             | Moderate    | Standard industry practice                  |
| Arithmetic Mean     | Optimistic           | Simple      | Upper bound estimate                        |
| Max                 | Very optimistic      | Simple      | Upper bound, sensitivity                    |
| Saturation Weighted | Moderate             | Moderate    | Varying wettability                         |
| Blunt               | Conservative         | Moderate    | Strong water-wet                            |
| Eclipse             | Moderate             | Moderate    | Commercial simulator standard               |
| Aziz-Settari        | Variable             | Moderate    | Empirical tuning                            |
"""


_MIXING_RULES: typing.Dict[str, MixingRule] = {}
"""Registry of mixing rule functions."""
_MIXING_RULE_SERIALIZERS: typing.Dict[
    MixingRule, typing.Callable[[MixingRule, bool], typing.Any]
] = {}
"""Registry of mixing rule serializers."""
_MIXING_RULE_DESERIALIZERS: typing.Dict[
    str, typing.Callable[[typing.Any], MixingRule]
] = {}
"""Registry of mixing rule deserializers."""
_lock = threading.Lock()


@typing.overload
def mixing_rule(func: MixingRule) -> MixingRule: ...


@typing.overload
def mixing_rule(
    func: None = None,
    name: typing.Optional[str] = None,
    override: bool = False,
    serializer: typing.Optional[typing.Callable[[MixingRule, bool], T]] = None,
    deserializer: typing.Optional[typing.Callable[[T], MixingRule]] = None,
) -> typing.Callable[[MixingRule], MixingRule]: ...


@typing.overload
def mixing_rule(
    func: MixingRule,
    name: typing.Optional[str] = None,
    override: bool = False,
    serializer: typing.Optional[typing.Callable[[MixingRule, bool], T]] = None,
    deserializer: typing.Optional[typing.Callable[[T], MixingRule]] = None,
) -> MixingRule: ...


def mixing_rule(
    func: typing.Optional[MixingRule] = None,
    name: typing.Optional[str] = None,
    override: bool = False,
    serializer: typing.Optional[typing.Callable[[MixingRule, bool], T]] = None,
    deserializer: typing.Optional[typing.Callable[[T], MixingRule]] = None,
) -> typing.Union[MixingRule, typing.Callable[[MixingRule], MixingRule]]:
    """
    Decorator to register a mixing rule function.

    A mixing rule function combines two-phase relative permeabilities
    into a three-phase relative permeability based on saturations.

    :param func: Mixing rule function to register.
    :param name: Optional name to register the function under.
    :param override: Whether to override an existing registration with the same name.
        If `False` (default), raises an error on name conflicts.
    :param serializer: Optional function to serialize the mixing rule.
        This is especially useful for parameterized mixing rules.
    :param deserializer: Optional function to deserialize the mixing rule.
        This is especially useful for parameterized mixing rules.
    :return: The original function.

    Example:
    ```python
    @mixing_rule(name="custom_rule")
    def custom_rule(
        kro_w: FloatOrArray,
        kro_g: FloatOrArray,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
    ) -> FloatOrArray:
        # Custom mixing logic here
        return (kro_w + kro_g) / 2.0

    # Usage
    rule = get_mixing_rule("custom_rule")
    result = rule(kro_w, kro_g, Sw, So, Sg)
    ```
    """

    def decorator(func: MixingRule) -> MixingRule:
        rule_name = name or getattr(func, "__name__", None)
        if rule_name is None:
            raise ValueError(
                "Mixing rule function  must have a `__name__` attribute or a name must be provided."
            )

        with _lock:
            if rule_name in _MIXING_RULES and not override:
                raise ValidationError(
                    f"Mixing rule '{rule_name}' is already registered. Use `override=True` or provide a different name."
                )

            _MIXING_RULES[rule_name] = func
            if serializer is not None:
                _MIXING_RULE_SERIALIZERS[func] = serializer
            if deserializer is not None:
                _MIXING_RULE_DESERIALIZERS[rule_name] = deserializer
        return func

    if func is None:
        return decorator
    return decorator(func)


def serialize_mixing_rule(rule: MixingRule, recurse: bool = True) -> typing.Any:
    """
    Serialize a mixing rule function to its registered name.

    :param rule: Mixing rule function.
    :return: Registered name of the mixing rule.
    """
    with _lock:
        if rule in _MIXING_RULE_SERIALIZERS:
            return _MIXING_RULE_SERIALIZERS[rule](rule, recurse)

        for name, registered_rule in _MIXING_RULES.items():
            if registered_rule == rule:
                return name
    raise ValidationError(
        f"Mixing rule {rule!r} is not registered. Use `@mixing_rule` to register."
    )


def deserialize_mixing_rule(name: str) -> MixingRule:
    """
    Deserialize a mixing rule function from its registered name.

    :param name: Registered name of the mixing rule.
    :return: Mixing rule function.
    """
    with _lock:
        if name in _MIXING_RULE_DESERIALIZERS:
            return _MIXING_RULE_DESERIALIZERS[name](name)
        elif name in _MIXING_RULES:
            return _MIXING_RULES[name]
    raise ValidationError(
        f"Mixing rule '{name}' is not registered. Use `@mixing_rule` to register."
    )


def list_mixing_rules() -> typing.List[str]:
    """
    List all registered mixing rule names.

    :return: List of registered mixing rule names.
    """
    with _lock:
        return list(_MIXING_RULES.keys())


def get_mixing_rule(name: str) -> MixingRule:
    """
    Get a registered mixing rule function by name.

    :param name: Registered name of the mixing rule.
    :return: Mixing rule function.
    :raises ValidationError: If the mixing rule is not registered.
    """
    with _lock:
        if name in _MIXING_RULES:
            return _MIXING_RULES[name]
    raise ValidationError(
        f"Mixing rule '{name}' is not registered. Use `@mixing_rule` to register."
    )


@mixing_rule
@numba.njit(cache=True)
def min_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Conservative rule for 3-phase oil relative permeability.
    kro = min(kro_w, kro_g)
    """
    return np.minimum(kro_w, kro_g)


@mixing_rule
@numba.njit(cache=True)
def stone_I_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Stone I rule (1970) for 3-phase oil relative permeability.
    kro = (kro_w * kro_g) / (kro_w + kro_g - kro_w * kro_g)
    """
    denom = np.maximum(((kro_w + kro_g) - (kro_w * kro_g)), 1e-12)
    result = (kro_w * kro_g) / denom
    # Return 0 if both kro_w and kro_g are zero
    return np.where((kro_w <= 0.0) & (kro_g <= 0.0), 0.0, result)


@mixing_rule
@numba.njit(cache=True)
def stone_II_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Stone II rule (Stone, 1973, JPT) for 3-phase oil relative permeability.

    Original Stone II formula (normalized, krocw = 1.0):
        kro = (krow + krw) * (krog + krg) - krw - krg

    where:
        - krow = oil relperm from oil-water table at current Sw
        - krog = oil relperm from oil-gas table at current Sg
        - krw = water relperm at current Sw
        - krg = gas relperm at current Sg
        - krocw = oil relperm at connate water (typically 1.0 for normalized tables)

    **Approximation Used Here:**
    Since the mixing rule signature only provides kro_w and kro_g (not krw and krg),
    this implementation approximates:
        krw ≈ 1 - kro_w
        krg ≈ 1 - kro_g

    **When This Approximation is Valid:**
    - Normalized tables with unit endpoints (krw_max = krg_max = kro_max = 1.0)
    - Linear or near-linear relative permeability curves
    - Two-phase systems where krw + kro_w ≈ 1 and krg + kro_g ≈ 1

    **When This Approximation Breaks Down:**
    - Non-linear Corey/Brooks-Corey models with exponents ≠ 1
    - Non-normalized tables
    - When krw + kro_w ≠ 1 (which occurs with Corey exponents)

    **Result:**
    With the approximation, the formula simplifies to:
        kro = kro_w + kro_g - 1

    This provides reasonable results for many cases but can be inaccurate with
    highly non-linear curves. For Corey-type models, consider using Stone I or
    geometric mean instead.

    **Reference:**
    Stone, H.L. (1973). "Estimation of Three-Phase Relative Permeability and
    Residual Oil Data." Journal of Canadian Petroleum Technology, 12(4), 53-61.
    """
    # Approximate krw and krg from two-phase oil relperm values
    # Note: This assumes krw + kro_w ≈ 1 (valid for unit-endpoint tables)
    krw_approx = 1.0 - kro_w
    krg_approx = 1.0 - kro_g

    # Stone II formula with approximated water and gas relperm
    result = (kro_w + krw_approx) * (kro_g + krg_approx) - krw_approx - krg_approx
    # This simplifies to: kro = kro_w + kro_g - 1

    # Clamp negative values (can occur when both kro_w and kro_g are small)
    result = np.maximum(result, 0.0)

    # Return zero if either two-phase oil relperm is zero (conservative)
    return np.where((kro_w <= 0.0) | (kro_g <= 0.0), 0.0, result)


@mixing_rule
@numba.njit(cache=True)
def arithmetic_mean_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Simple arithmetic mean of oil-water and oil-gas relative permeabilities.

    kro = (kro_w + kro_g) / 2

    Notes:
        - Simple and conservative
        - Does not account for saturation distribution
        - Tends to overestimate kro compared to other methods
    """
    return (kro_w + kro_g) / 2.0


@mixing_rule
@numba.njit(cache=True)
def geometric_mean_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Geometric mean of oil-water and oil-gas relative permeabilities.

    kro = sqrt(kro_w * kro_g)

    Notes:
        - More conservative than arithmetic mean
        - If either kro_w or kro_g is zero, result is zero
        - Smooth transition between two-phase limits
    """
    return np.sqrt(kro_w * kro_g)


@mixing_rule
@numba.njit(cache=True)
def harmonic_mean_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Harmonic mean of oil-water and oil-gas relative permeabilities.

    kro = 2 / (1/kro_w + 1/kro_g)

    Notes:
    - Most conservative of the mean rules
    - Heavily weighted by the smaller value
    - Useful for series flow paths
    - Returns 0 if either input is zero (conservative)
    """
    # Protect against division by zero with epsilon
    epsilon = 1e-30
    safe_kro_w = np.maximum(kro_w, epsilon)
    safe_kro_g = np.maximum(kro_g, epsilon)

    result = 2.0 / ((1.0 / safe_kro_w) + (1.0 / safe_kro_g))

    # Return 0 if either original value was zero (conservative for flow)
    return np.where((kro_w <= 0.0) | (kro_g <= 0.0), 0.0, result)


@mixing_rule
@numba.njit(cache=True)
def saturation_weighted_interpolation_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Saturation-weighted interpolation between oil-water and oil-gas systems.

    kro = kro_w * (Sw / (Sw + Sg)) + kro_g * (Sg / (Sw + Sg))

    Notes:
        - Interpolates based on ratio of water to gas saturation
        - Reduces to kro_w when Sg=0 (oil-water system)
        - Reduces to kro_g when Sw=0 (oil-gas system)
        - Similar to Stone II but uses different saturation ratios
    """
    total_displacing_phase = water_saturation + gas_saturation

    water_weight = np.where(
        total_displacing_phase > 0.0, water_saturation / total_displacing_phase, 0.0
    )
    gas_weight = np.where(
        total_displacing_phase > 0.0, gas_saturation / total_displacing_phase, 0.0
    )

    result = (kro_w * water_weight) + (kro_g * gas_weight)
    # Return maximum of kro_w and kro_g if total_displacing_phase is zero (pure oil)
    return np.where(total_displacing_phase > 0.0, result, np.maximum(kro_w, kro_g))


@mixing_rule
@numba.njit(cache=True)
def baker_linear_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Baker's linear interpolation rule (1988).

    Interpolates three-phase kro as a saturation-weighted combination
    of the two-phase endpoint values:

        kro = (Sw * kro_w + So * kro_oil_water_contact_angleendpoint + Sg * kro_g) / (Sw + So + Sg)

    where kro_oil_water_contact_angleendpoint is approximated as max(kro_w, kro_g) (the oil kr
    at the oil-water endpoint, i.e. in absence of gas), and kro_g is the
    oil kr from the gas-oil table.

    Simplification used here (standard Baker linear, no separate endpoint table):
        kro = (Sw * kro_w + Sg * kro_g) / (Sw + Sg)   when Sw+Sg > 0
        kro = max(kro_w, kro_g)                         when Sw+Sg = 0 (pure oil)

    This reduces exactly to kro_w at Sg=0 and kro_g at Sw=0.

    Note: This is similar to `saturation_weighted_interpolation_rule`
    """
    total_displacing = water_saturation + gas_saturation

    result = np.where(
        total_displacing > 0.0,
        (kro_w * water_saturation + kro_g * gas_saturation) / total_displacing,
        np.maximum(kro_w, kro_g),
    )
    return result


@mixing_rule
@numba.njit(cache=True)
def blunt_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Blunt's rule for three-phase relative permeability.

    kro = kro_w * kro_g * (2 - kro_w - kro_g)

    Notes:
        - Developed for strongly water-wet systems
        - Accounts for pore-level displacement mechanisms
        - Generally gives conservative estimates
        - Result is clamped to [0, ∞) to handle edge cases with non-normalized tables
    """
    result = kro_w * kro_g * (2.0 - kro_w - kro_g)

    # Clamp to non-negative values (handles edge cases if kro_w + kro_g > 2)
    result = np.maximum(result, 0.0)

    # Return 0 if either kro_w or kro_g is zero (conservative)
    return np.where((kro_w <= 0.0) | (kro_g <= 0.0), 0.0, result)


@mixing_rule
@numba.njit(cache=True)
def hustad_hansen_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Hustad-Hansen rule (1995) for three-phase relative permeability.

    kro = (kro_w * kro_g) / max(kro_w, kro_g)

    Notes:
        - Conservative estimate
        - Ensures kro ≤ min(kro_w, kro_g)
        - Good for intermediate wettability systems
    """
    max_kr = np.maximum(np.maximum(kro_w, kro_g), 1e-12)
    result = (kro_w * kro_g) / max_kr
    # Return 0 if both kro_w and kro_g are zero
    return np.where((kro_w <= 0.0) & (kro_g <= 0.0), 0.0, result)


def aziz_settari_rule(a: float = 0.5, b: float = 0.5) -> MixingRule:
    """
    Aziz-Settari empirical correlation.

    kro = kro_w^a * kro_g^b

    where a and b are empirical exponents (typically a=0.5, b=0.5).

    Notes:
        - Empirical correlation from petroleum engineering textbook
        - Can be tuned with different exponents
        - Generally conservative

    :param a: Exponent for oil-water system (default 0.5).
    :param b: Exponent for oil-gas system (default 0.5).
    :return: A mixing rule function implementing the Aziz-Settari correlation.
    """

    def aziz_settari_serializer(
        rule: MixingRule, recurse: bool = True
    ) -> typing.Dict[str, float]:
        return {"a": a, "b": b}

    def aziz_settari_deserializer(data: typing.Dict[str, float]) -> MixingRule:
        if not isinstance(data, dict) or "a" not in data or "b" not in data:
            raise ValidationError("Invalid data for Aziz-Settari deserialization.")
        return aziz_settari_rule(a=data["a"], b=data["b"])

    @mixing_rule(
        name=f"aziz_settari(a={a},b={b})",
        serializer=aziz_settari_serializer,
        deserializer=aziz_settari_deserializer,
    )
    @numba.njit(cache=True)
    def _rule(
        kro_w: FloatOrArray,
        kro_g: FloatOrArray,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
    ) -> FloatOrArray:
        result = (kro_w**a) * (kro_g**b)
        # Return 0 if either kro_w or kro_g is zero
        return np.where((kro_w <= 0.0) | (kro_g <= 0.0), 0.0, result)

    _rule.__name__ = f"aziz_settari_rule(a={a},b={b})"  # type: ignore[attr-defined]
    return _rule


@mixing_rule
@numba.njit(cache=True)
def eclipse_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    ECLIPSE simulator default three-phase rule.

    Similar to Stone II but with saturation normalization.

    kro = kro_w * f_w + kro_g * f_g

    where f_w and f_g are saturation-dependent factors.

    Notes:
        - Used in commercial ECLIPSE simulator
        - Provides smooth transition between phases
        - Handles edge cases robustly
    """
    total_mobile = oil_saturation + water_saturation + gas_saturation

    # Saturation factors - use np.where to avoid type inconsistency
    denom_w = oil_saturation + gas_saturation
    f_w = np.where(denom_w > 0.0, oil_saturation / denom_w, 0.0)

    denom_g = oil_saturation + water_saturation
    f_g = np.where(denom_g > 0.0, oil_saturation / denom_g, 0.0)

    # Return 0 if total_mobile is zero, otherwise compute kro
    result = (kro_w * f_w) + (kro_g * f_g)
    return np.where(total_mobile > 0.0, result, 0.0)


@mixing_rule
@numba.njit(cache=True)
def max_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Maximum rule - most optimistic estimate.

    kro = max(kro_w, kro_g)

    Notes:
        - Upper bound for oil relative permeability
        - Rarely used in practice (too optimistic)
        - Useful for sensitivity analysis
    """
    return np.maximum(kro_w, kro_g)


@mixing_rule
@numba.njit(cache=True)
def product_saturation_weighted_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Product of two-phase kr values weighted by oil saturation.

    kro = (kro_w * kro_g) * (So / So_max)^n

    where n is an empirical exponent (typically 0.5-2.0).

    Notes:
        - Accounts for reduction in connectivity at lower oil saturations
        - Empirical parameter n can be tuned to match experimental data
        - Conservative for low oil saturations
    """
    n = 1.0  # Empirical exponent

    # Assume maximum oil saturation is 1.0 - Swi - Sgr
    # For simplicity, use total saturation to normalize
    total_sat = water_saturation + oil_saturation + gas_saturation

    so_normalized = np.where(total_sat > 0.0, oil_saturation / total_sat, 0.0)
    result = (kro_w * kro_g) * (so_normalized**n)

    # Return 0 if oil_saturation or total_sat is zero
    return np.where((oil_saturation > 0.0) & (total_sat > 0.0), result, 0.0)


@mixing_rule
@numba.njit(cache=True)
def linear_interpolation_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Simple linear interpolation based on gas-to-water ratio.

    kro = kro_g * R + kro_w * (1 - R)

    where R = Sg / (Sw + Sg)

    Notes:
        - Simple and intuitive
        - Smooth transition between two-phase systems
        - Does not account for three-phase interference effects
    """
    total_displacing = water_saturation + gas_saturation

    gas_fraction = np.where(
        total_displacing > 0.0, gas_saturation / total_displacing, 0.0
    )
    water_fraction = np.where(
        total_displacing > 0.0, water_saturation / total_displacing, 0.0
    )

    result = (kro_g * gas_fraction) + (kro_w * water_fraction)
    # Pure oil - return maximum
    return np.where(total_displacing > 0.0, result, np.maximum(kro_w, kro_g))


def _mixing_rule_partial_derivatives(
    mixing_rule: MixingRule,
    kro_water_two_phase: FloatOrArray,
    kro_gas_two_phase: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
    epsilon: float = 1e-7,
) -> typing.Tuple[
    FloatOrArray,  # d_kro_d_kro_w
    FloatOrArray,  # d_kro_d_kro_g
    FloatOrArray,  # d_kro_d_sw_explicit
    FloatOrArray,  # d_kro_d_so_explicit
    FloatOrArray,  # d_kro_d_sg_explicit
]:
    """
    Compute partial derivatives of the three-phase mixing rule with respect
    to each of its five arguments using central finite differences.

    The mixing rule signature is:

    ```python
    kro = mixing_rule(kro_w, kro_g, water_saturation, oil_saturation, gas_saturation)
    ```

    This function returns five partial derivatives:

    - `d_kro_d_kro_w`: how the mixed oil relative permeability changes when
      the oil-water two-phase oil relative permeability changes.
    - `d_kro_d_kro_g`: how the mixed oil relative permeability changes when
      the gas-oil two-phase oil relative permeability changes.
    - `d_kro_d_sw_explicit`: how the mixed oil relative
      permeability changes through the explicit water saturation argument of
      the mixing rule (e.g. the saturation weighting in `eclipse_rule`).
    - `d_kro_d_so_explicit`: same for oil saturation.
    - `d_kro_d_sg_explicit`: same for gas saturation.

    The cost is ten mixing rule evaluations (five central difference pairs),
    each O(1), regardless of grid size.

    :param mixing_rule: The mixing rule callable.
    :param kro_water_two_phase: Oil relative permeability from the oil-water
        two-phase table at the current saturation iterate.
    :param kro_gas_two_phase: Oil relative permeability from the gas-oil
        two-phase table at the current saturation iterate.
    :param water_saturation: Current water saturation.
    :param oil_saturation: Current oil saturation.
    :param gas_saturation: Current gas saturation.
    :param epsilon: Finite difference step size.
    :return: 5-tuple of partial derivatives described above.
    """
    d_kro_d_kro_w = (
        mixing_rule(
            kro_w=kro_water_two_phase + epsilon,
            kro_g=kro_gas_two_phase,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
        )
        - mixing_rule(
            kro_w=kro_water_two_phase - epsilon,
            kro_g=kro_gas_two_phase,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
        )
    ) / (2.0 * epsilon)

    d_kro_d_kro_g = (
        mixing_rule(
            kro_w=kro_water_two_phase,
            kro_g=kro_gas_two_phase + epsilon,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
        )
        - mixing_rule(
            kro_w=kro_water_two_phase,
            kro_g=kro_gas_two_phase - epsilon,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
        )
    ) / (2.0 * epsilon)

    d_kro_d_sw_explicit = (
        mixing_rule(
            kro_w=kro_water_two_phase,
            kro_g=kro_gas_two_phase,
            water_saturation=water_saturation + epsilon,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
        )
        - mixing_rule(
            kro_w=kro_water_two_phase,
            kro_g=kro_gas_two_phase,
            water_saturation=water_saturation - epsilon,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
        )
    ) / (2.0 * epsilon)

    d_kro_d_so_explicit = (
        mixing_rule(
            kro_w=kro_water_two_phase,
            kro_g=kro_gas_two_phase,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation + epsilon,
            gas_saturation=gas_saturation,
        )
        - mixing_rule(
            kro_w=kro_water_two_phase,
            kro_g=kro_gas_two_phase,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation - epsilon,
            gas_saturation=gas_saturation,
        )
    ) / (2.0 * epsilon)

    d_kro_d_sg_explicit = (
        mixing_rule(
            kro_w=kro_water_two_phase,
            kro_g=kro_gas_two_phase,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation + epsilon,
        )
        - mixing_rule(
            kro_w=kro_water_two_phase,
            kro_g=kro_gas_two_phase,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation - epsilon,
        )
    ) / (2.0 * epsilon)

    return (
        d_kro_d_kro_w,
        d_kro_d_kro_g,
        d_kro_d_sw_explicit,
        d_kro_d_so_explicit,
        d_kro_d_sg_explicit,
    )


class RelativePermeabilityTable(StoreSerializable):
    """
    Protocol for a relative permeability table that computes
    relative permeabilities based on fluid saturations.
    """

    __abstract_serializable__ = True

    def get_relative_permeabilities(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for water, oil, and gas.

        :param water_saturation: Water saturation (fraction) — scalar or array.
        :param oil_saturation: Oil saturation (fraction) — scalar or array.
        :param gas_saturation: Gas saturation (fraction) — scalar or array.
        :param kwargs: Other key word arguments for computing the relative peremabilities.
        :return: `RelativePermeabilities` dictionary with keys "water", "oil", "gas".
        """
        raise NotImplementedError

    def get_relative_permeability_derivatives(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute relative permeability derivatives for water, oil, and gas.

        :param water_saturation: Water saturation (fraction) — scalar or array.
        :param oil_saturation: Oil saturation (fraction) — scalar or array.
        :param gas_saturation: Gas saturation (fraction) — scalar or array.
        :param kwargs: Other key word arguments for computing the derivatives.
        :return: `RelativePermeabilityDerivatives` dictionary.
        """
        raise NotImplementedError

    def __call__(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for water, oil, and gas.

        :param water_saturation: Water saturation (fraction) — scalar or array.
        :param oil_saturation: Oil saturation (fraction) — scalar or array.
        :param gas_saturation: Gas saturation (fraction) — scalar or array.
        :param kwargs: Other key word arguments for computing the relative peremabilities.
        :return: `RelativePermeabilities` dictionary.
        """
        return self.get_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )

    def derivatives(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute relative permeability derivatives for water, oil, and gas.

        :param water_saturation: Water saturation (fraction) — scalar or array.
        :param oil_saturation: Oil saturation (fraction) — scalar or array.
        :param gas_saturation: Gas saturation (fraction) — scalar or array.
        :param kwargs: Other key word arguments for computing the derivatives.
        :return: `RelativePermeabilityDerivatives` dictionary.
        """
        return self.get_relative_permeability_derivatives(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )


_RELPERM_TABLES: typing.Dict[str, typing.Type[RelativePermeabilityTable]] = {}
"""Registry of relative permeability table types."""
_relperm_tables_lock = threading.Lock()
relperm_table = make_serializable_type_registrar(
    base_cls=RelativePermeabilityTable,
    registry=_RELPERM_TABLES,
    key_attr="__type__",
    lock=_relperm_tables_lock,
    override=False,
    auto_register_serializer=True,
    auto_register_deserializer=True,
)


def list_relperm_tables() -> typing.List[str]:
    """
    List all registered relative permeability table types.

    :return: List of registered relative permeability table type names.
    """
    with _relperm_tables_lock:
        return list(_RELPERM_TABLES.keys())


def get_relperm_table(name: str) -> typing.Type[RelativePermeabilityTable]:
    """
    Get a registered relative permeability table type by name.

    :param name: Registered name of the relative permeability table type.
    :return: Relative permeability table class.
    :raises ValidationError: If the relative permeability table type is not registered.
    """
    with _relperm_tables_lock:
        if name not in _RELPERM_TABLES:
            raise ValidationError(
                f"Relative permeability table type '{name}' is not registered. "
                f"Use `@relperm_table` to register it. "
                f"Available types: {list(_RELPERM_TABLES.keys())}"
            )
        return _RELPERM_TABLES[name]


@attrs.frozen
class TwoPhaseRelPermTable(Serializable):
    """
    Two-phase relative permeability lookup table.

    Interpolates relative permeabilities for two fluid phases based on a
    reference saturation value. The reference saturation can be either the
    wetting or non-wetting phase saturation, depending on how the table was
    constructed (e.g. from lab data indexed by Sg vs So).

    Uses `np.interp` for fast vectorized interpolation.
    Supports both scalar and array inputs up to 3D.

    Examples:
    - Oil-Water system (water-wet): reference is Sw (wetting phase), `reference_phase="wetting"`
    - Gas-Oil system indexed by So:  reference is So (wetting phase), `reference_phase="wetting"`
    - Gas-Oil system indexed by Sg:  reference is Sg (non-wetting phase), `reference_phase="non_wetting"`
    """

    wetting_phase: typing.Union[FluidPhase, str] = attrs.field(converter=FluidPhase)
    """The wetting fluid phase, e.g. WATER (oil-water) or OIL (gas-oil)."""

    non_wetting_phase: typing.Union[FluidPhase, str] = attrs.field(converter=FluidPhase)
    """The non-wetting fluid phase, e.g. OIL (oil-water) or GAS (gas-oil)."""

    reference_saturation: npt.NDArray[np.floating] = attrs.field(converter=bores_array)
    """
    Saturation values used as the x-axis for interpolation, monotonically increasing.
    May represent either the wetting or non-wetting phase saturation depending on
    `reference_phase`.
    """

    wetting_phase_relative_permeability: npt.NDArray[np.floating] = attrs.field(
        converter=bores_array
    )
    """Relative permeability values for the wetting phase at each reference saturation."""

    non_wetting_phase_relative_permeability: npt.NDArray[np.floating] = attrs.field(
        converter=bores_array
    )
    """Relative permeability values for the non-wetting phase at each reference saturation."""

    reference_phase: typing.Literal["wetting", "non_wetting"] = attrs.field(
        default="wetting"
    )
    """
    Which phase the `reference_saturation` axis represents.

    - "wetting" - `reference_saturation` holds wetting phase saturation values.
        krw increases and krnw decreases as `reference_saturation` increases.
    - "non_wetting" - `reference_saturation` holds non-wetting phase saturation values.
        krnw increases and krw decreases as `reference_saturation` increases.

    This does not change the interpolation mechanics — it only records which
    physical saturation the caller must supply when querying the table, so that
    `ThreePhaseRelPermTable` (and any other consumer) can dispatch the correct
    saturation grid without hard-coding assumptions.
    """

    def __attrs_post_init__(self) -> None:
        if self.reference_phase not in ("wetting", "non_wetting"):
            raise ValidationError(
                f"`reference_phase` must be 'wetting' or 'non_wetting', got {self.reference_phase!r}"
            )

        if len(self.reference_saturation) != len(
            self.wetting_phase_relative_permeability
        ):
            raise ValidationError(
                f"reference_saturation and wetting phase kr arrays must have same length. "
                f"Got {len(self.reference_saturation)} vs {len(self.wetting_phase_relative_permeability)}"
            )

        if len(self.reference_saturation) != len(
            self.non_wetting_phase_relative_permeability
        ):
            raise ValidationError(
                f"reference_saturation and non-wetting phase kr arrays must have same length. "
                f"Got {len(self.reference_saturation)} vs {len(self.non_wetting_phase_relative_permeability)}"
            )

        if len(self.reference_saturation) < 2:
            raise ValidationError("At least 2 points required for interpolation")

        if not np.all(np.diff(self.reference_saturation) >= 0):
            raise ValidationError(
                "reference_saturation must be monotonically increasing"
            )

    def _resolve_reference(
        self,
        wetting_saturation: FloatOrArray,
        non_wetting_saturation: FloatOrArray,
    ) -> FloatOrArray:
        """
        Return whichever saturation array corresponds to the reference axis.

        :param wetting_saturation: Current wetting phase saturation (scalar or array).
        :param non_wetting_saturation: Current non-wetting phase saturation (scalar or array).
        :return: The saturation to use as the interpolation x-value.
        """
        if self.reference_phase == "non_wetting":
            return non_wetting_saturation
        return wetting_saturation

    def get_wetting_phase_relative_permeability(
        self,
        wetting_saturation: FloatOrArray,
        non_wetting_saturation: typing.Optional[FloatOrArray] = None,
    ) -> FloatOrArray:
        """
        Get wetting phase relative permeability.

        When `reference_phase="wetting"`, only `wetting_saturation` is needed.
        When `reference_phase="non_wetting"`, `non_wetting_saturation` must be supplied.

        :param wetting_saturation: Wetting phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Relative permeability value(s).
        """
        ref = self._resolve_reference(
            wetting_saturation,
            non_wetting_saturation
            if non_wetting_saturation is not None
            else wetting_saturation,
        )
        saturation = np.atleast_1d(ref)
        original_shape = saturation.shape
        kr_flat = np.interp(
            x=saturation.ravel(),
            xp=self.reference_saturation,  # type: ignore[arg-type]
            fp=self.wetting_phase_relative_permeability,  # type: ignore[arg-type]
            left=self.wetting_phase_relative_permeability[0],
            right=self.wetting_phase_relative_permeability[-1],
        )
        return kr_flat.reshape(original_shape)

    def get_non_wetting_phase_relative_permeability(
        self,
        wetting_saturation: FloatOrArray,
        non_wetting_saturation: typing.Optional[FloatOrArray] = None,
    ) -> FloatOrArray:
        """
        Get non-wetting phase relative permeability.

        When `reference_phase="wetting"`, only `wetting_saturation` is needed.
        When `reference_phase="non_wetting"`, `non_wetting_saturation` must be supplied.

        :param wetting_saturation: Wetting phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Relative permeability value(s).
        """
        ref = self._resolve_reference(
            wetting_saturation,
            non_wetting_saturation
            if non_wetting_saturation is not None
            else wetting_saturation,
        )
        saturation = np.atleast_1d(ref)
        original_shape = saturation.shape
        kr_flat = np.interp(
            x=saturation.ravel(),
            xp=self.reference_saturation,  # type: ignore[arg-type]
            fp=self.non_wetting_phase_relative_permeability,  # type: ignore[arg-type]
            left=self.non_wetting_phase_relative_permeability[0],
            right=self.non_wetting_phase_relative_permeability[-1],
        )
        return kr_flat.reshape(original_shape)

    def get_relative_permeabilities(
        self,
        wetting_saturation: FloatOrArray,
        non_wetting_saturation: typing.Optional[FloatOrArray] = None,
    ) -> typing.Tuple[FloatOrArray, FloatOrArray]:
        """
        Get both wetting and non-wetting phase relative permeabilities.

        :param wetting_saturation: Wetting phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Tuple of (wetting_kr, non_wetting_kr).
        """
        kr_wetting = self.get_wetting_phase_relative_permeability(
            wetting_saturation, non_wetting_saturation
        )
        kr_non_wetting = self.get_non_wetting_phase_relative_permeability(
            wetting_saturation, non_wetting_saturation
        )
        return kr_wetting, kr_non_wetting

    def get_wetting_phase_relative_permeability_derivative(
        self,
        wetting_saturation: FloatOrArray,
        non_wetting_saturation: typing.Optional[FloatOrArray] = None,
    ) -> FloatOrArray:
        """
        Derivative of the wetting-phase relative permeability with respect to
        the reference saturation axis of this table.

        The reference axis is determined by the `reference_phase` attribute:

        - `reference_phase="wetting"`: the reference axis is the wetting-phase
        saturation, so this returns `dkrw / d(wetting saturation)`.
        - `reference_phase="non_wetting"`: the reference axis is the
        non-wetting-phase saturation, so this returns
        `dkrw / d(non-wetting saturation)`.

        The derivative is the exact slope of the piecewise-linear interpolant
        defined by the tabulated relative permeability values — the same
        interpolant used by `get_wetting_phase_relative_permeability`.  The
        slope is zero outside the tabulated saturation range because the
        interpolant is constant there.

        :param wetting_saturation: Wetting-phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting-phase saturation (scalar or
            array).  Required when `reference_phase="non_wetting"`; ignored
            when `reference_phase="wetting"`.
        :return: Derivative value(s) with the same shape as the input.
        """
        reference_saturation = self._resolve_reference(
            wetting_saturation,
            non_wetting_saturation
            if non_wetting_saturation is not None
            else wetting_saturation,
        )
        return piecewise_linear_slope(
            query=reference_saturation,
            table_x=self.reference_saturation,
            table_y=self.wetting_phase_relative_permeability,
        )

    def get_non_wetting_phase_relative_permeability_derivative(
        self,
        wetting_saturation: FloatOrArray,
        non_wetting_saturation: typing.Optional[FloatOrArray] = None,
    ) -> FloatOrArray:
        """
        Derivative of the non-wetting-phase relative permeability with respect
        to the reference saturation axis of this table.

        The reference axis is determined by the `reference_phase` attribute:

        - `reference_phase="wetting"`: the reference axis is the wetting-phase
        saturation, so this returns `dkrnw / d(wetting saturation)`.
        - `reference_phase="non_wetting"`: the reference axis is the
        non-wetting-phase saturation, so this returns
        `dkrnw / d(non-wetting saturation)`.

        The derivative is the exact slope of the piecewise-linear interpolant
        defined by the tabulated relative permeability values — the same
        interpolant used by `get_non_wetting_phase_relative_permeability`.
        The slope is zero outside the tabulated saturation range because the
        interpolant is constant there.

        :param wetting_saturation: Wetting-phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting-phase saturation (scalar or
            array).  Required when `reference_phase="non_wetting"`; ignored
            when `reference_phase="wetting"`.
        :return: Derivative value(s) with the same shape as the input.
        """
        reference_saturation = self._resolve_reference(
            wetting_saturation,
            non_wetting_saturation
            if non_wetting_saturation is not None
            else wetting_saturation,
        )
        return piecewise_linear_slope(
            query=reference_saturation,
            table_x=self.reference_saturation,
            table_y=self.non_wetting_phase_relative_permeability,
        )


@relperm_table
@attrs.frozen
class ThreePhaseRelPermTable(
    RelativePermeabilityTable,
    serializers={"mixing_rule": serialize_mixing_rule},
    deserializers={"mixing_rule": deserialize_mixing_rule},
    load_exclude={"supports_arrays"},
    dump_exclude={"supports_arrays"},
):
    """
    Three-phase relative permeability lookup table, with mixing rules.

    Interpolates relative permeabilities for water, oil, and gas based on
    saturation values. Uses two `TwoPhaseRelPermTable` instances (oil-water
    and gas-oil) and a mixing rule for oil in the three-phase system.

    Each two-phase table declares its own `reference_phase` ("wetting" or
    "non_wetting"), so the correct saturation is dispatched automatically —
    no assumptions are hard-coded about whether a table is indexed by So or Sg.

    Supported mixing rules: `max_rule`, `stone_I_rule`, `stone_II_rule`, etc.
    """

    __type__ = "three_phase_relperm_table"

    oil_water_table: TwoPhaseRelPermTable
    """Relative permeability table for the oil-water system."""

    gas_oil_table: TwoPhaseRelPermTable
    """Relative permeability table for the gas-oil system."""

    mixing_rule: typing.Optional[typing.Union[MixingRule, str]] = None
    """
    Mixing rule function or name to compute oil relative permeability in the
    three-phase system. Signature:

        rule(kro_w, kro_g, water_saturation, oil_saturation, gas_saturation) -> kro

    If None, falls back to min(kro_w, kro_g).
    """

    supports_arrays: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array inputs."""

    def __attrs_post_init__(self) -> None:
        if {
            self.oil_water_table.wetting_phase,
            self.oil_water_table.non_wetting_phase,
        } != {FluidPhase.WATER, FluidPhase.OIL}:
            raise ValidationError(
                "`oil_water_table` must be between water and oil phases."
            )

        if {
            self.gas_oil_table.wetting_phase,
            self.gas_oil_table.non_wetting_phase,
        } != {FluidPhase.OIL, FluidPhase.GAS}:
            raise ValidationError("`gas_oil_table` must be between oil and gas phases.")

        mixing_rule = self.mixing_rule
        if isinstance(mixing_rule, str):
            object.__setattr__(self, "mixing_rule", get_mixing_rule(mixing_rule))

    def get_relative_permeabilities(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for water, oil, and gas.

        Each two-phase table is queried using its declared `reference_phase`:
        - `reference_phase="wetting"` - the wetting phase saturation is passed
        - `reference_phase="non_wetting"` - the non-wetting phase saturation is passed

        This means the caller never needs to reverse or reindex tables: build
        the table with whatever saturation axis is natural (So or Sg) and set
        `reference_phase` accordingly.

        :param water_saturation: Water saturation (fraction) — scalar or array.
        :param oil_saturation: Oil saturation (fraction) — scalar or array.
        :param gas_saturation: Gas saturation (fraction) — scalar or array.
        :return: `RelativePermeabilities` dict with keys "water", "oil", "gas".
        """
        sw = np.atleast_1d(water_saturation)
        so = np.atleast_1d(oil_saturation)
        sg = np.atleast_1d(gas_saturation)
        sw, so, sg = np.broadcast_arrays(sw, so, sg)

        if np.any((sw < 0) | (sw > 1) | (so < 0) | (so > 1) | (sg < 0) | (sg > 1)):
            raise ValidationError("Saturations must be between 0 and 1.")

        # Normalize if saturations do not sum to 1
        total_saturation = sw + so + sg
        needs_norm = (np.abs(total_saturation - 1.0) > 1e-6) & (total_saturation > 0.0)
        if np.any(needs_norm):
            sw = np.where(needs_norm, sw / total_saturation, sw)
            so = np.where(needs_norm, so / total_saturation, so)
            sg = np.where(needs_norm, sg / total_saturation, sg)

        # For Oil-water table
        # Wetting phase is WATER, non-wetting is OIL (water-wet convention)
        # or wetting is OIL, non-wetting is WATER (oil-wet convention).
        # In either case we pass both saturations and let the table pick its
        # reference axis.
        if self.oil_water_table.wetting_phase == FluidPhase.WATER:
            krw = self.oil_water_table.get_wetting_phase_relative_permeability(
                sw, non_wetting_saturation=so
            )
            kro_w = self.oil_water_table.get_non_wetting_phase_relative_permeability(
                sw, non_wetting_saturation=so
            )
        else:
            # Oil is wetting phase
            kro_w = self.oil_water_table.get_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sw
            )
            krw = self.oil_water_table.get_non_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sw
            )

        # For Gas-oil table
        if self.gas_oil_table.wetting_phase == FluidPhase.OIL:
            kro_g = self.gas_oil_table.get_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sg
            )
            krg = self.gas_oil_table.get_non_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sg
            )
        else:
            # Gas is wetting phase
            krg = self.gas_oil_table.get_wetting_phase_relative_permeability(
                sg, non_wetting_saturation=so
            )
            kro_g = self.gas_oil_table.get_non_wetting_phase_relative_permeability(
                sg, non_wetting_saturation=so
            )

        # Three-phase oil mixing rule
        if self.mixing_rule is not None:
            kro = self.mixing_rule(  # type: ignore[operator]
                kro_w=kro_w,
                kro_g=kro_g,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
            )
        else:
            kro = np.minimum(kro_w, kro_g)

        return RelativePermeabilities(water=krw, oil=kro, gas=krg)  # type: ignore[typeddict-item]

    def get_relative_permeability_derivatives(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute all nine partial derivatives of the three-phase relative
        permeabilities with respect to water saturation, oil saturation, and
        gas saturation.

        Returns a dictionary containing:
        ```
        (dkrw/dSw, dkrw/dSo, dkrw/dSg,
        dkro/dSw, dkro/dSo, dkro/dSg,
        dkrg/dSw, dkrg/dSo, dkrg/dSg)
        ```

        All output arrays have the same shape as the input saturation arrays
        (or are scalars if the inputs are scalars).

        Water and gas relative permeability derivatives are computed analytically
        from the slopes of the underlying two-phase lookup tables using the same
        reference axis dispatch as the forward evaluation (see the `reference_phase`
        attribute of each two-phase table).

        Oil relative permeability derivatives use the chain rule through the
        three-phase mixing rule:
        ```
        dkro/dSalpha = (d_kro/d_kro_w) * (d_kro_w/d_Salpha)
                    + (d_kro/d_kro_g) * (d_kro_g/d_Salpha)
                    + (d_kro/d_Salpha)_explicit_in_mixing_rule
        ```

        The first two terms use the analytical two-phase table slope derivatives.
        The explicit saturation terms are obtained via a central finite difference
        on the mixing rule call only (ten scalar evaluations, O(1) per cell).

        **Note on oil saturation derivatives**
        Oil saturation is not an independent degree of freedom — it is determined
        by `So = 1 - Sw - Sg`.  Nevertheless `dkralpha/dSo` is included in the
        output because some table configurations (e.g. a gas-oil table whose
        reference axis is oil saturation) produce a non-zero `dkrg/dSo`, and
        the analytical Jacobian assembly needs all nine entries to correctly
        propagate the saturation constraint.

        :param water_saturation: Water saturation (fraction, 0 to 1).
        :param oil_saturation: Oil saturation (fraction, 0 to 1).
        :param gas_saturation: Gas saturation (fraction, 0 to 1).
        :return: `RelativePermeabilityDerivatives` dictionary containing the partial derivatives as described above.
        """
        is_scalar = np.isscalar(water_saturation)
        sw = np.atleast_1d(np.asarray(water_saturation, dtype=np.float64))
        so = np.atleast_1d(np.asarray(oil_saturation, dtype=np.float64))
        sg = np.atleast_1d(np.asarray(gas_saturation, dtype=np.float64))
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        zeros = np.zeros_like(sw)

        oil_water_table = self.oil_water_table
        gas_oil_table = self.gas_oil_table

        # Oil-water table derivatives: krw and kro_w
        # Dispatch based on both wetting_phase and reference_phase

        if oil_water_table.wetting_phase == FluidPhase.WATER:
            # Water is wetting phase
            if oil_water_table.reference_phase == "wetting":
                # Table indexed by Sw (wetting phase) → derivatives w.r.t. Sw
                d_krw_d_sw = (
                    oil_water_table.get_wetting_phase_relative_permeability_derivative(
                        wetting_saturation=sw,
                        non_wetting_saturation=so,
                    )
                )
                d_krw_d_so = zeros.copy()
                d_krw_d_sg = zeros.copy()

                d_kro_w_d_sw = oil_water_table.get_non_wetting_phase_relative_permeability_derivative(
                    wetting_saturation=sw,
                    non_wetting_saturation=so,
                )
                d_kro_w_d_so = zeros.copy()
                d_kro_w_d_sg = zeros.copy()
            else:
                # reference_phase="non_wetting": Table indexed by So (non-wetting phase)
                # → derivatives w.r.t. So
                d_krw_d_sw = zeros.copy()
                d_krw_d_so = (
                    oil_water_table.get_wetting_phase_relative_permeability_derivative(
                        wetting_saturation=sw,
                        non_wetting_saturation=so,
                    )
                )
                d_krw_d_sg = zeros.copy()

                d_kro_w_d_sw = zeros.copy()
                d_kro_w_d_so = oil_water_table.get_non_wetting_phase_relative_permeability_derivative(
                    wetting_saturation=sw,
                    non_wetting_saturation=so,
                )
                d_kro_w_d_sg = zeros.copy()
        else:
            # Oil is wetting phase (oil-wet system)
            if oil_water_table.reference_phase == "wetting":
                # Table indexed by So (wetting phase) → derivatives w.r.t. So
                d_krw_d_sw = zeros.copy()
                d_krw_d_so = oil_water_table.get_non_wetting_phase_relative_permeability_derivative(
                    wetting_saturation=so,
                    non_wetting_saturation=sw,
                )
                d_krw_d_sg = zeros.copy()

                d_kro_w_d_sw = zeros.copy()
                d_kro_w_d_so = (
                    oil_water_table.get_wetting_phase_relative_permeability_derivative(
                        wetting_saturation=so,
                        non_wetting_saturation=sw,
                    )
                )
                d_kro_w_d_sg = zeros.copy()
            else:
                # reference_phase="non_wetting": Table indexed by Sw (non-wetting phase)
                # → derivatives w.r.t. Sw
                d_krw_d_sw = oil_water_table.get_non_wetting_phase_relative_permeability_derivative(
                    wetting_saturation=so,
                    non_wetting_saturation=sw,
                )
                d_krw_d_so = zeros.copy()
                d_krw_d_sg = zeros.copy()

                d_kro_w_d_sw = (
                    oil_water_table.get_wetting_phase_relative_permeability_derivative(
                        wetting_saturation=so,
                        non_wetting_saturation=sw,
                    )
                )
                d_kro_w_d_so = zeros.copy()
                d_kro_w_d_sg = zeros.copy()

        # Gas-oil table derivatives: krg and kro_g
        # Dispatch based on both wetting_phase and reference_phase
        if gas_oil_table.wetting_phase == FluidPhase.OIL:
            # Oil is wetting phase
            if gas_oil_table.reference_phase == "wetting":
                # Table indexed by So (wetting phase) → derivatives w.r.t. So
                d_krg_d_sw = zeros.copy()
                d_krg_d_so = gas_oil_table.get_non_wetting_phase_relative_permeability_derivative(
                    wetting_saturation=so,
                    non_wetting_saturation=sg,
                )
                d_krg_d_sg = zeros.copy()

                d_kro_g_d_sw = zeros.copy()
                d_kro_g_d_so = (
                    gas_oil_table.get_wetting_phase_relative_permeability_derivative(
                        wetting_saturation=so,
                        non_wetting_saturation=sg,
                    )
                )
                d_kro_g_d_sg = zeros.copy()
            else:
                # reference_phase="non_wetting": Table indexed by Sg (non-wetting phase)
                # → derivatives w.r.t. Sg
                d_krg_d_sw = zeros.copy()
                d_krg_d_so = zeros.copy()
                d_krg_d_sg = gas_oil_table.get_non_wetting_phase_relative_permeability_derivative(
                    wetting_saturation=so,
                    non_wetting_saturation=sg,
                )

                d_kro_g_d_sw = zeros.copy()
                d_kro_g_d_so = zeros.copy()
                d_kro_g_d_sg = (
                    gas_oil_table.get_wetting_phase_relative_permeability_derivative(
                        wetting_saturation=so,
                        non_wetting_saturation=sg,
                    )
                )
        else:
            # Gas is wetting phase (uncommon)
            if gas_oil_table.reference_phase == "wetting":
                # Table indexed by Sg (wetting phase) → derivatives w.r.t. Sg
                d_krg_d_sw = zeros.copy()
                d_krg_d_so = zeros.copy()
                d_krg_d_sg = (
                    gas_oil_table.get_wetting_phase_relative_permeability_derivative(
                        wetting_saturation=sg,
                        non_wetting_saturation=so,
                    )
                )

                d_kro_g_d_sw = zeros.copy()
                d_kro_g_d_so = zeros.copy()
                d_kro_g_d_sg = gas_oil_table.get_non_wetting_phase_relative_permeability_derivative(
                    wetting_saturation=sg,
                    non_wetting_saturation=so,
                )
            else:
                # reference_phase="non_wetting": Table indexed by So (non-wetting phase)
                # → derivatives w.r.t. So
                d_krg_d_sw = zeros.copy()
                d_krg_d_so = (
                    gas_oil_table.get_wetting_phase_relative_permeability_derivative(
                        wetting_saturation=sg,
                        non_wetting_saturation=so,
                    )
                )
                d_krg_d_sg = zeros.copy()

                d_kro_g_d_sw = zeros.copy()
                d_kro_g_d_so = gas_oil_table.get_non_wetting_phase_relative_permeability_derivative(
                    wetting_saturation=sg,
                    non_wetting_saturation=so,
                )
                d_kro_g_d_sg = zeros.copy()

        # Forward two-phase oil kr values for mixing rule finite difference
        if oil_water_table.wetting_phase == FluidPhase.WATER:
            kro_w = oil_water_table.get_non_wetting_phase_relative_permeability(
                sw, non_wetting_saturation=so
            )
        else:
            kro_w = oil_water_table.get_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sw
            )

        if gas_oil_table.wetting_phase == FluidPhase.OIL:
            kro_g = gas_oil_table.get_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sg
            )
        else:
            kro_g = gas_oil_table.get_non_wetting_phase_relative_permeability(
                sg, non_wetting_saturation=so
            )

        # Three-phase oil mixing rule derivatives
        mixing_rule = self.mixing_rule
        if mixing_rule is None:
            # max_rule: kro = min(kro_w, kro_g)
            kro_w_arr = np.asarray(kro_w, dtype=np.float64)
            kro_g_arr = np.asarray(kro_g, dtype=np.float64)
            d_kro_d_kro_w = np.where(kro_w_arr <= kro_g_arr, 1.0, 0.0)
            d_kro_d_kro_g = np.where(kro_g_arr <= kro_w_arr, 1.0, 0.0)
            d_kro_d_sw_explicit = zeros.copy()
            d_kro_d_so_explicit = zeros.copy()
            d_kro_d_sg_explicit = zeros.copy()
        else:
            (
                d_kro_d_kro_w,
                d_kro_d_kro_g,
                d_kro_d_sw_explicit,
                d_kro_d_so_explicit,
                d_kro_d_sg_explicit,
            ) = _mixing_rule_partial_derivatives(
                mixing_rule=mixing_rule,  # type: ignore[arg-type]
                kro_water_two_phase=kro_w,
                kro_gas_two_phase=kro_g,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                epsilon=c.FINITE_DIFFERENCE_EPSILON,
            )

        # Chain rule: dkro/dSalpha = (d_kro/d_kro_w) * (d_kro_w/d_Salpha)
        #                           + (d_kro/d_kro_g) * (d_kro_g/d_Salpha)
        #                           + (d_kro/d_Salpha)_explicit
        d_kro_d_sw = (
            d_kro_d_kro_w * d_kro_w_d_sw
            + d_kro_d_kro_g * d_kro_g_d_sw
            + d_kro_d_sw_explicit
        )
        d_kro_d_so = (
            d_kro_d_kro_w * d_kro_w_d_so
            + d_kro_d_kro_g * d_kro_g_d_so
            + d_kro_d_so_explicit
        )
        d_kro_d_sg = (
            d_kro_d_kro_w * d_kro_w_d_sg
            + d_kro_d_kro_g * d_kro_g_d_sg
            + d_kro_d_sg_explicit
        )

        results = (
            d_krw_d_sw,
            d_kro_d_sw,
            d_krg_d_sw,
            d_krw_d_so,
            d_kro_d_so,
            d_krg_d_so,
            d_krw_d_sg,
            d_kro_d_sg,
            d_krg_d_sg,
        )
        if is_scalar:
            results = tuple(float(np.ravel(r)[0]) for r in results)
            return RelativePermeabilityDerivatives(
                dKrw_dSw=results[0],
                dKro_dSw=results[1],
                dKrg_dSw=results[2],
                dKrw_dSo=results[3],
                dKro_dSo=results[4],
                dKrg_dSo=results[5],
                dKrw_dSg=results[6],
                dKro_dSg=results[7],
                dKrg_dSg=results[8],
            )

        return RelativePermeabilityDerivatives(
            dKrw_dSw=d_krw_d_sw,
            dKro_dSw=d_kro_d_sw,
            dKrg_dSw=d_krg_d_sw,
            dKrw_dSo=d_krw_d_so,
            dKro_dSo=d_kro_d_so,
            dKrg_dSo=d_krg_d_so,
            dKrw_dSg=d_krw_d_sg,
            dKro_dSg=d_kro_d_sg,
            dKrg_dSg=d_krg_d_sg,
        )


def compute_corey_three_phase_relative_permeabilities(
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
    irreducible_water_saturation: float,
    residual_oil_saturation_water: float,
    residual_oil_saturation_gas: float,
    residual_gas_saturation: float,
    water_exponent: float,
    oil_exponent: float,
    gas_exponent: float,
    wettability: Wettability = Wettability.WATER_WET,
    mixing_rule: MixingRule = stone_II_rule,
    saturation_epsilon: float = 1e-6,
    minimum_mobile_pore_space: float = 1e-9,
) -> typing.Tuple[FloatOrArray, FloatOrArray, FloatOrArray]:
    """
    Computes relative permeability for water, oil, and gas in a three-phase system.
    Supports water-wet and oil-wet wettability assumptions.

    Uses Corey-type models for krw, krg, and Stone I rule for kro.

    Supports both scalar and array inputs for saturations.

    :param water_saturation: Current water saturation (fraction, between 0 and 1) - scalar or array.
    :param oil_saturation: Current oil saturation (fraction, between 0 and 1) - scalar or array.
    :param gas_saturation: Current gas saturation (fraction, between 0 and 1) - scalar or array.
    :param irreducible_water_saturation: Irreducible water saturation (Swc).
    :param residual_oil_saturation_water: Residual oil saturation after water flood (Sorw).
    :param residual_oil_saturation_gas: Residual oil saturation after gas flood (Sorg).
    :param residual_gas_saturation: Residual gas saturation (Sgr).
    :param water_exponent: Corey exponent for water relative permeability.
    :param oil_exponent: Corey exponent for oil relative permeability (affects Stone I blending).
    :param gas_exponent: Corey exponent for gas relative permeability.
    :param wettability: Wettability type (water-wet or oil-wet).
    :return: (water_relative_permeability, oil_relative_permeability, gas_relative_permeability)
    """
    # Convert to arrays for vectorized operations
    sw = np.atleast_1d(water_saturation)
    so = np.atleast_1d(oil_saturation)
    sg = np.atleast_1d(gas_saturation)

    # Broadcast all arrays to same shape
    sw, so, sg = np.broadcast_arrays(sw, so, sg)

    # Validate saturations
    if np.any((sw < 0) | (sw > 1) | (so < 0) | (so > 1) | (sg < 0) | (sg > 1)):
        raise ValidationError("Saturations must be between 0 and 1.")

    # Normalize saturations if they do not sum to 1
    total_saturation = sw + so + sg
    needs_norm = (np.abs(total_saturation - 1.0) > saturation_epsilon) & (
        total_saturation > 0.0
    )
    if np.any(needs_norm):
        sw = np.where(needs_norm, sw / total_saturation, sw)
        so = np.where(needs_norm, so / total_saturation, so)
        sg = np.where(needs_norm, sg / total_saturation, sg)

    if wettability == Wettability.WATER_WET:
        # 1. Water relperm (wetting phase)
        movable_water_range = (
            1.0 - irreducible_water_saturation - residual_oil_saturation_water
        )
        effective_water_saturation = np.where(
            movable_water_range <= minimum_mobile_pore_space,
            np.zeros_like(sw),
            np.clip(
                (sw - irreducible_water_saturation) / movable_water_range, 0.0, 1.0
            ),
        )
        krw = effective_water_saturation**water_exponent

        # 2. Gas relperm (nonwetting)
        # Mobile gas range must account for connate water + residual oil + residual gas
        movable_gas_range = (
            1.0
            - irreducible_water_saturation
            - residual_gas_saturation
            - residual_oil_saturation_gas
        )
        effective_gas_saturation = np.where(
            movable_gas_range <= minimum_mobile_pore_space,
            np.zeros_like(sg),
            np.clip((sg - residual_gas_saturation) / movable_gas_range, 0.0, 1.0),
        )
        krg = effective_gas_saturation**gas_exponent

        # 3. Oil relperm (intermediate phase) → Stone I blending
        # Make sure to apply oil curvature to the two-phase oil kr inputs before mixing,
        # not to the mixed output. (1-krw) and (1-krg) are the two-phase
        # oil kr approximations; so we shape them with `oil_exponent` before blending.
        kro_w_shaped = (1.0 - krw) ** oil_exponent
        kro_g_shaped = (1.0 - krg) ** oil_exponent
        kro = mixing_rule(
            kro_w=kro_w_shaped,
            kro_g=kro_g_shaped,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
        )

    elif wettability == Wettability.OIL_WET:
        # Oil is wetting, water becomes intermediate
        # 1. Oil relperm (wetting phase)
        movable_oil_range = (
            1.0 - residual_oil_saturation_water - residual_oil_saturation_gas
        )
        max_residual = np.minimum(
            residual_oil_saturation_water, residual_oil_saturation_gas
        )
        effective_oil_saturation = np.where(
            movable_oil_range <= minimum_mobile_pore_space,
            np.zeros_like(so),
            np.clip((so - max_residual) / movable_oil_range, 0.0, 1.0),
        )
        kro = effective_oil_saturation**oil_exponent

        # 2. Gas relperm (nonwetting phase)
        movable_gas_range = 1.0 - residual_gas_saturation - irreducible_water_saturation
        effective_gas_saturation = np.where(
            movable_gas_range <= minimum_mobile_pore_space,
            np.zeros_like(sg),
            np.clip((sg - residual_gas_saturation) / movable_gas_range, 0.0, 1.0),
        )
        krg = effective_gas_saturation**gas_exponent

        # 3. Water relperm (intermediate phase, use Stone I style blending)
        kro_proxy_shaped = (1.0 - kro) ** water_exponent
        krg_proxy_shaped = (1.0 - krg) ** water_exponent
        krw = mixing_rule(  # type: ignore[assignment]
            kro_w=kro_proxy_shaped,  # treat oil as wetting
            kro_g=krg_proxy_shaped,  # treat gas as nonwetting
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
        )

    else:
        raise ValidationError(f"Wettability {wettability!r} not implemented.")

    # Clip all results to [0, 1]
    krw = np.clip(krw, 0.0, 1.0)
    kro = np.clip(kro, 0.0, 1.0)
    krg = np.clip(krg, 0.0, 1.0)
    return krw, kro, krg  # type: ignore[return-value]


@relperm_table
@attrs.frozen
class BrooksCoreyThreePhaseRelPermModel(
    RelativePermeabilityTable,
    serializers={"mixing_rule": serialize_mixing_rule},
    deserializers={"mixing_rule": deserialize_mixing_rule},
    load_exclude={"supports_arrays"},
    dump_exclude={"supports_arrays"},
):
    """
    Brooks-Corey-type three-phase relative permeability model.

    Uses the Brooks-Corey model for two-phase relative permeabilities
    and Stone I mixing rule for oil in three-phase system.

    Supports water-wet and oil-wet wettability assumptions.
    """

    __type__ = "brooks_corey_three_phase_relperm_model"

    irreducible_water_saturation: typing.Optional[float] = None
    """(Default) Irreducible water saturation (Swc)."""
    residual_oil_saturation_water: typing.Optional[float] = None
    """(Default) Residual oil saturation after water flood (Sorw)."""
    residual_oil_saturation_gas: typing.Optional[float] = None
    """(Default) Residual oil saturation after gas flood (Sorg)."""
    residual_gas_saturation: typing.Optional[float] = None
    """(Default) Residual gas saturation (Sgr)."""
    water_exponent: float = 2.0
    """
    Corey exponent for water relative permeability. 
    
    Higher values make the curve steeper. Meaning slower krw increase with saturation.
    """
    oil_exponent: float = 2.0
    """
    Corey exponent for oil relative permeability (affects Stone I blending). 
    
    Higher values make the curve steeper. Meaning slower kro increase with saturation.
    """
    gas_exponent: float = 2.0
    """
    Corey exponent for gas relative permeability. Higher values make the curve steeper. 
    
    Meaning slower krg increase with saturation.
    """
    wettability: Wettability = Wettability.WATER_WET
    """Wettability type (water-wet or oil-wet)."""
    mixing_rule: typing.Union[MixingRule, str] = eclipse_rule
    """
    Mixing rule function or name to compute oil relative permeability in three-phase system.

    The function should take the following parameters in order:
    - kro_w: Oil relative permeability from oil-water table
    - kro_g: Oil relative permeability from oil-gas table
    - Sw: Water saturation
    - So: Oil saturation
    - Sg: Gas saturation
    and return the mixed oil relative permeability.
    """
    supports_arrays: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array inputs."""

    def __attrs_post_init__(self) -> None:
        mixing_rule = self.mixing_rule
        if isinstance(mixing_rule, str):
            object.__setattr__(self, "mixing_rule", get_mixing_rule(mixing_rule))

    def get_relative_permeabilities(  # type: ignore[override]
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        irreducible_water_saturation: typing.Optional[float] = None,
        residual_oil_saturation_water: typing.Optional[float] = None,
        residual_oil_saturation_gas: typing.Optional[float] = None,
        residual_gas_saturation: typing.Optional[float] = None,
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
        0 <= water_saturation, oil_saturation, gas_saturation <= 1
        """
        Sorw = (
            residual_oil_saturation_water
            if residual_oil_saturation_water is not None
            else self.residual_oil_saturation_water
        )
        Sorg = (
            residual_oil_saturation_gas
            if residual_oil_saturation_gas is not None
            else self.residual_oil_saturation_gas
        )
        Sgr = (
            residual_gas_saturation
            if residual_gas_saturation is not None
            else self.residual_gas_saturation
        )
        Swc = (
            irreducible_water_saturation
            if irreducible_water_saturation is not None
            else self.irreducible_water_saturation
        )
        params_missing = []
        if Swc is None:
            params_missing.append("Swc")
        if Sorw is None:
            params_missing.append("Sorw")
        if Sorg is None:
            params_missing.append("Sorg")
        if Sgr is None:
            params_missing.append("Sgr")
        if params_missing:
            raise ValidationError(
                f"Residual saturations must be provided either as arguments or set in the model instance. "
                f"Missing: {', '.join(params_missing)}"
            )

        krw, kro, krg = compute_corey_three_phase_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=Swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=Sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=Sorg,  # type: ignore[arg-type]
            residual_gas_saturation=Sgr,  # type: ignore[arg-type]
            water_exponent=self.water_exponent,
            oil_exponent=self.oil_exponent,
            gas_exponent=self.gas_exponent,
            wettability=self.wettability,
            mixing_rule=self.mixing_rule,  # type: ignore[arg-type]
            saturation_epsilon=c.SATURATION_EPSILON,
            minimum_mobile_pore_space=c.MINIMUM_MOBILE_PORE_SPACE,
        )
        return RelativePermeabilities(water=krw, oil=kro, gas=krg)  # type: ignore[typeddict-item]

    def get_relative_permeability_derivatives(  # type: ignore[override]
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        irreducible_water_saturation: typing.Optional[float] = None,
        residual_oil_saturation_water: typing.Optional[float] = None,
        residual_oil_saturation_gas: typing.Optional[float] = None,
        residual_gas_saturation: typing.Optional[float] = None,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute all nine partial derivatives of the three-phase relative
        permeabilities with respect to water saturation, oil saturation, and
        gas saturation using the Brooks-Corey model.

        Returns a dictionary containing:

        ```
        (dkrw/dSw, dkrw/dSo, dkrw/dSg,
        dkro/dSw, dkro/dSo, dkro/dSg,
        dkrg/dSw, dkrg/dSo, dkrg/dSg)
        ```

        For the water-wet case all two-phase Corey power-law derivatives are
        computed analytically via the chain rule through effective saturation.
        The three-phase oil relative permeability derivative is then completed
        by the chain rule through the mixing rule.

        For the oil-wet case all nine derivatives fall back to central finite
        differences on the full three-phase Corey function to avoid duplicating
        the more complex oil-wet saturation normalisation logic.

        :param water_saturation: Water saturation (fraction, 0 to 1).
        :param oil_saturation: Oil saturation (fraction, 0 to 1).
        :param gas_saturation: Gas saturation (fraction, 0 to 1).
        :param irreducible_water_saturation: Optional override for the
            irreducible (connate) water saturation.  Uses the model default
            when not provided.
        :param residual_oil_saturation_water: Optional override for the residual
            oil saturation to water flooding.  Uses the model default when not
            provided.
        :param residual_oil_saturation_gas: Optional override for the residual
            oil saturation to gas flooding.  Uses the model default when not
            provided.
        :param residual_gas_saturation: Optional override for the residual gas
            saturation.  Uses the model default when not provided.
        :return: `RelativePermeabilityDerivatives` dictionary containing the partial derivatives as described above.
        """
        Swc = (
            irreducible_water_saturation
            if irreducible_water_saturation is not None
            else self.irreducible_water_saturation
        )
        Sorw = (
            residual_oil_saturation_water
            if residual_oil_saturation_water is not None
            else self.residual_oil_saturation_water
        )
        Sorg = (
            residual_oil_saturation_gas
            if residual_oil_saturation_gas is not None
            else self.residual_oil_saturation_gas
        )
        Sgr = (
            residual_gas_saturation
            if residual_gas_saturation is not None
            else self.residual_gas_saturation
        )

        params_missing = []
        if Swc is None:
            params_missing.append("Swc")
        if Sorw is None:
            params_missing.append("Sorw")
        if Sorg is None:
            params_missing.append("Sorg")
        if Sgr is None:
            params_missing.append("Sgr")
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

        is_scalar = np.isscalar(water_saturation)
        sw = np.atleast_1d(water_saturation)
        so = np.atleast_1d(oil_saturation)
        sg = np.atleast_1d(gas_saturation)
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        zeros = np.zeros_like(sw)
        saturation_epsilon = c.SATURATION_EPSILON
        minimum_mobile_pore_space = c.MINIMUM_MOBILE_PORE_SPACE

        if wettability == Wettability.OIL_WET:
            finite_difference_epsilon = c.FINITE_DIFFERENCE_EPSILON
            inv_2_epsilon = 1.0 / (2.0 * finite_difference_epsilon)

            def _corey(sw, so, sg):
                return compute_corey_three_phase_relative_permeabilities(
                    water_saturation=sw,
                    oil_saturation=so,
                    gas_saturation=sg,
                    irreducible_water_saturation=Swc,  # type: ignore[arg-type]
                    residual_oil_saturation_water=Sorw,  # type: ignore[arg-type]
                    residual_oil_saturation_gas=Sorg,  # type: ignore[arg-type]
                    residual_gas_saturation=Sgr,  # type: ignore[arg-type]
                    water_exponent=water_exponent,
                    oil_exponent=oil_exponent,
                    gas_exponent=gas_exponent,
                    wettability=wettability,
                    mixing_rule=mixing_rule,
                    saturation_epsilon=saturation_epsilon,
                    minimum_mobile_pore_space=minimum_mobile_pore_space,
                )

            krw_sw_fwd, kro_sw_fwd, krg_sw_fwd = _corey(
                sw + finite_difference_epsilon, so, sg
            )
            krw_sw_bwd, kro_sw_bwd, krg_sw_bwd = _corey(
                sw - finite_difference_epsilon, so, sg
            )
            krw_so_fwd, kro_so_fwd, krg_so_fwd = _corey(
                sw, so + finite_difference_epsilon, sg
            )
            krw_so_bwd, kro_so_bwd, krg_so_bwd = _corey(
                sw, so - finite_difference_epsilon, sg
            )
            krw_sg_fwd, kro_sg_fwd, krg_sg_fwd = _corey(
                sw, so, sg + finite_difference_epsilon
            )
            krw_sg_bwd, kro_sg_bwd, krg_sg_bwd = _corey(
                sw, so, sg - finite_difference_epsilon
            )

            results = (
                (krw_sw_fwd - krw_sw_bwd) * inv_2_epsilon,
                (krw_so_fwd - krw_so_bwd) * inv_2_epsilon,
                (krw_sg_fwd - krw_sg_bwd) * inv_2_epsilon,
                (kro_sw_fwd - kro_sw_bwd) * inv_2_epsilon,
                (kro_so_fwd - kro_so_bwd) * inv_2_epsilon,
                (kro_sg_fwd - kro_sg_bwd) * inv_2_epsilon,
                (krg_sw_fwd - krg_sw_bwd) * inv_2_epsilon,
                (krg_so_fwd - krg_so_bwd) * inv_2_epsilon,
                (krg_sg_fwd - krg_sg_bwd) * inv_2_epsilon,
            )
            if is_scalar:
                results = tuple(float(np.ravel(r)[0]) for r in results)
            return RelativePermeabilityDerivatives(
                dKrw_dSw=results[0],
                dKro_dSw=results[1],
                dKrg_dSw=results[2],
                dKrw_dSo=results[3],
                dKro_dSo=results[4],
                dKrg_dSo=results[5],
                dKrw_dSg=results[6],
                dKro_dSg=results[7],
                dKrg_dSg=results[8],
            )

        # Water-wet path
        # krw = Se_w ^ nw
        mobile_water_range = 1.0 - Swc - Sorw  # type: ignore
        valid_water = mobile_water_range > minimum_mobile_pore_space
        se_w = np.clip(
            (sw - Swc) / np.where(valid_water, mobile_water_range, 1.0),
            0.0,
            1.0,
        )
        krw_values = se_w**water_exponent
        d_krw_d_sw = np.where(
            valid_water & (se_w > 0.0),
            water_exponent
            * (se_w ** max(water_exponent - 1.0, 0.0))
            / mobile_water_range,
            zeros,
        )
        d_krw_d_so = zeros.copy()
        d_krw_d_sg = zeros.copy()

        # krg = Se_g ^ ng
        mobile_gas_range = 1.0 - Swc - Sgr - Sorg  # type: ignore
        valid_gas = mobile_gas_range > minimum_mobile_pore_space
        se_g = np.clip(
            (sg - Sgr) / np.where(valid_gas, mobile_gas_range, 1.0),
            0.0,
            1.0,
        )
        krg_values = se_g**gas_exponent
        d_krg_d_sg = np.where(
            valid_gas & (se_g > 0.0),
            gas_exponent * (se_g ** max(gas_exponent - 1.0, 0.0)) / mobile_gas_range,
            zeros,
        )
        d_krg_d_sw = zeros.copy()
        d_krg_d_so = zeros.copy()

        # kro_w shaped = (1 - krw)^no
        one_minus_krw = np.clip(1.0 - krw_values, 0.0, None)
        kro_w_shaped = one_minus_krw**oil_exponent
        d_kro_w_d_sw = np.where(
            one_minus_krw > 0.0,
            oil_exponent
            * (one_minus_krw ** max(oil_exponent - 1.0, 0.0))
            * (-d_krw_d_sw),
            zeros,
        )
        d_kro_w_d_so = zeros.copy()
        d_kro_w_d_sg = zeros.copy()

        # kro_g shaped = (1 - krg)^no
        one_minus_krg = np.clip(1.0 - krg_values, 0.0, None)
        kro_g_shaped = one_minus_krg**oil_exponent
        d_kro_g_d_sg = np.where(
            one_minus_krg > 0.0,
            oil_exponent
            * (one_minus_krg ** max(oil_exponent - 1.0, 0.0))
            * (-d_krg_d_sg),
            zeros,
        )
        d_kro_g_d_sw = zeros.copy()
        d_kro_g_d_so = zeros.copy()

        (
            d_kro_d_kro_w,
            d_kro_d_kro_g,
            d_kro_d_water_saturation_explicit,
            d_kro_d_oil_saturation_explicit,
            d_kro_d_gas_saturation_explicit,
        ) = _mixing_rule_partial_derivatives(
            mixing_rule=mixing_rule,
            kro_water_two_phase=kro_w_shaped,
            kro_gas_two_phase=kro_g_shaped,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            epsilon=c.FINITE_DIFFERENCE_EPSILON,
        )

        d_kro_d_sw = (
            d_kro_d_kro_w * d_kro_w_d_sw
            + d_kro_d_kro_g * d_kro_g_d_sw
            + d_kro_d_water_saturation_explicit
        )
        d_kro_d_so = (
            d_kro_d_kro_w * d_kro_w_d_so
            + d_kro_d_kro_g * d_kro_g_d_so
            + d_kro_d_oil_saturation_explicit
        )
        d_kro_d_sg = (
            d_kro_d_kro_w * d_kro_w_d_sg
            + d_kro_d_kro_g * d_kro_g_d_sg
            + d_kro_d_gas_saturation_explicit
        )

        results = (
            d_krw_d_sw,
            d_kro_d_sw,
            d_krg_d_sw,
            d_krw_d_so,
            d_kro_d_so,
            d_krg_d_so,
            d_krw_d_sg,
            d_kro_d_sg,
            d_krg_d_sg,
        )
        if is_scalar:
            results = tuple(float(np.ravel(r)[0]) for r in results)
            return RelativePermeabilityDerivatives(
                dKrw_dSw=results[0],
                dKro_dSw=results[1],
                dKrg_dSw=results[2],
                dKrw_dSo=results[3],
                dKro_dSo=results[4],
                dKrg_dSo=results[5],
                dKrw_dSg=results[6],
                dKro_dSg=results[7],
                dKrg_dSg=results[8],
            )
        return RelativePermeabilityDerivatives(
            dKrw_dSw=d_krw_d_sw,
            dKro_dSw=d_kro_d_sw,
            dKrg_dSw=d_krg_d_sw,
            dKrw_dSo=d_krw_d_so,
            dKro_dSo=d_kro_d_so,
            dKrg_dSo=d_krg_d_so,
            dKrw_dSg=d_krw_d_sg,
            dKro_dSg=d_kro_d_sg,
            dKrg_dSg=d_krg_d_sg,
        )

    def __call__(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for water, oil, and gas.

        Supports both scalar and array inputs for saturations.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :kwarg irreducible_water_saturation: Optional override for irreducible water saturation.
        :kwarg residual_oil_saturation_water: Optional override for residual oil saturation after water flood.
        :kwarg residual_oil_saturation_gas: Optional override for residual oil saturation after gas flood.
        :kwarg residual_gas_saturation: Optional override for residual gas saturation.
        :return: `RelativePermeabilities` dictionary.
        0 <= water_saturation, oil_saturation, gas_saturation <= 1
        """
        return self.get_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=kwargs.get(
                "irreducible_water_saturation", None
            ),
            residual_oil_saturation_water=kwargs.get(
                "residual_oil_saturation_water", None
            ),
            residual_oil_saturation_gas=kwargs.get("residual_oil_saturation_gas", None),
            residual_gas_saturation=kwargs.get("residual_gas_saturation", None),
        )

    def derivatives(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute relative permeability derivatives for water, oil, and gas.

        Supports both scalar and array inputs for saturations.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :kwarg irreducible_water_saturation: Optional override for irreducible water saturation.
        :kwarg residual_oil_saturation_water: Optional override for residual oil saturation after water flood.
        :kwarg residual_oil_saturation_gas: Optional override for residual oil saturation after gas flood.
        :kwarg residual_gas_saturation: Optional override for residual gas saturation.
        :return: `RelativePermeabilityDerivatives` dictionary.
        0 <= water_saturation, oil_saturation, gas_saturation <= 1
        """
        return self.get_relative_permeability_derivatives(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=kwargs.get(
                "irreducible_water_saturation", None
            ),
            residual_oil_saturation_water=kwargs.get(
                "residual_oil_saturation_water", None
            ),
            residual_oil_saturation_gas=kwargs.get("residual_oil_saturation_gas", None),
            residual_gas_saturation=kwargs.get("residual_gas_saturation", None),
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

    L: float = 2.0
    """Low-end shape parameter. Controls curvature near zero normalized saturation."""
    E: float = 1.0
    """Elevation parameter. Controls overall curve height at intermediate saturations."""
    T: float = 2.0
    """Top-end shape parameter. Controls curvature near maximum normalized saturation."""

    def __attrs_post_init__(self) -> None:
        if self.L <= 0:
            raise ValidationError(f"LET parameter `L` must be positive, got {self.L}")
        if self.E <= 0:
            raise ValidationError(f"LET parameter `E` must be positive, got {self.E}")
        if self.T <= 0:
            raise ValidationError(f"LET parameter `T` must be positive, got {self.T}")


def _let_relperm(
    normalized_saturation: FloatOrArray,
    L: float,
    E: float,
    T: float,
) -> FloatOrArray:
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
    return np.where(denominator > 0.0, numerator / denominator, 0.0)


def compute_let_three_phase_relative_permeabilities(
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
    irreducible_water_saturation: float,
    residual_oil_saturation_water: float,
    residual_oil_saturation_gas: float,
    residual_gas_saturation: float,
    water_L: float,
    water_E: float,
    water_T: float,
    oil_water_L: float,
    oil_water_E: float,
    oil_water_T: float,
    gas_oil_L: float,
    gas_oil_E: float,
    gas_oil_T: float,
    gas_L: float,
    gas_E: float,
    gas_T: float,
    max_water_relperm: float = 1.0,
    max_oil_relperm: float = 1.0,
    max_gas_relperm: float = 1.0,
    wettability: Wettability = Wettability.WATER_WET,
    mixing_rule: MixingRule = eclipse_rule,
    saturation_epsilon: float = 1e-6,
    minimum_mobile_pore_space: float = 1e-9,
) -> typing.Tuple[FloatOrArray, FloatOrArray, FloatOrArray]:
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
    :param irreducible_water_saturation: Irreducible water saturation (Swc).
    :param residual_oil_saturation_water: Residual oil saturation to waterflood (Sorw).
    :param residual_oil_saturation_gas: Residual oil saturation to gas flood (Sorg).
    :param residual_gas_saturation: Residual gas saturation (Sgr).
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
    :param max_water_relperm: Endpoint relative permeability for water (krw_max).
    :param max_oil_relperm: Endpoint relative permeability for oil (kro_max).
    :param max_gas_relperm: Endpoint relative permeability for gas (krg_max).
    :param wettability: Wettability type (water-wet or oil-wet).
    :param mixing_rule: Three-phase mixing rule for oil relative permeability.
    :return: (krw, kro, krg) tuple of relative permeabilities.
    """
    sw = np.atleast_1d(water_saturation)
    so = np.atleast_1d(oil_saturation)
    sg = np.atleast_1d(gas_saturation)

    sw, so, sg = np.broadcast_arrays(sw, so, sg)

    if np.any((sw < 0) | (sw > 1) | (so < 0) | (so > 1) | (sg < 0) | (sg > 1)):
        raise ValidationError("Saturations must be between 0 and 1.")

    # Normalize saturations if they do not sum to 1
    total_saturation = sw + so + sg
    needs_norm = (np.abs(total_saturation - 1.0) > saturation_epsilon) & (
        total_saturation > 0.0
    )
    if np.any(needs_norm):
        sw = np.where(needs_norm, sw / total_saturation, sw)
        so = np.where(needs_norm, so / total_saturation, so)
        sg = np.where(needs_norm, sg / total_saturation, sg)

    Swc = irreducible_water_saturation
    Sorw = residual_oil_saturation_water
    Sorg = residual_oil_saturation_gas
    Sgr = residual_gas_saturation

    if wettability == Wettability.WATER_WET:
        # Water kr (wetting phase)
        movable_water_range = 1.0 - Swc - Sorw
        sw_star = np.where(
            movable_water_range <= minimum_mobile_pore_space,
            np.zeros_like(sw),
            np.clip((sw - Swc) / movable_water_range, 0.0, 1.0),
        )
        krw = max_water_relperm * _let_relperm(sw_star, water_L, water_E, water_T)

        # Gas kr (non-wetting phase)
        movable_gas_range = 1.0 - Swc - Sgr - Sorg
        sg_star = np.where(
            movable_gas_range <= minimum_mobile_pore_space,
            np.zeros_like(sg),
            np.clip((sg - Sgr) / movable_gas_range, 0.0, 1.0),
        )
        krg = max_gas_relperm * _let_relperm(sg_star, gas_L, gas_E, gas_T)

        # Oil kr (intermediate phase, three-phase mixing)
        # Compute two-phase oil kr on a unit-endpoint basis (0 to 1) so that
        # mixing rules, which are designed for unit-endpoint curves, produce
        # correct results. The endpoint scaling is applied after mixing.

        # Two-phase oil kr in the water-oil system (unit endpoint)
        movable_oil_water_range = 1.0 - Swc - Sorw
        so_star_w = np.where(
            movable_oil_water_range <= minimum_mobile_pore_space,
            np.zeros_like(so),
            np.clip((so - Sorw) / movable_oil_water_range, 0.0, 1.0),
        )
        kro_w = _let_relperm(so_star_w, oil_water_L, oil_water_E, oil_water_T)

        # Two-phase oil kr in the gas-oil system (unit endpoint)
        movable_gas_oil_range = 1.0 - Swc - Sorg - Sgr
        so_star_g = np.where(
            movable_gas_oil_range <= minimum_mobile_pore_space,
            np.zeros_like(so),
            np.clip((so - Sorg) / movable_gas_oil_range, 0.0, 1.0),
        )
        kro_g = _let_relperm(so_star_g, gas_oil_L, gas_oil_E, gas_oil_T)

        # Combine using mixing rule, then apply endpoint scaling.
        # Clip the mixing rule output to [0, 1] before scaling because some
        # rules (e.g. eclipse, Stone II) can produce values > 1 when both
        # two-phase curves are near their maximum simultaneously.
        kro_mixed = mixing_rule(
            kro_w=kro_w,
            kro_g=kro_g,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
        )
        kro = max_oil_relperm * np.clip(kro_mixed, 0.0, 1.0)

    elif wettability == Wettability.OIL_WET:
        # Oil is wetting, water becomes intermediate
        # Oil kr (wetting phase)
        movable_oil_range = 1.0 - Sorw - Sorg
        max_residual = np.minimum(Sorw, Sorg)
        so_star = np.where(
            movable_oil_range <= minimum_mobile_pore_space,
            np.zeros_like(so),
            np.clip((so - max_residual) / movable_oil_range, 0.0, 1.0),
        )
        kro = max_oil_relperm * _let_relperm(
            so_star, oil_water_L, oil_water_E, oil_water_T
        )

        # Gas kr (non-wetting phase)
        movable_gas_range = 1.0 - Sgr - Swc
        sg_star = np.where(
            movable_gas_range <= minimum_mobile_pore_space,
            np.zeros_like(sg),
            np.clip((sg - Sgr) / movable_gas_range, 0.0, 1.0),
        )
        krg = max_gas_relperm * _let_relperm(sg_star, gas_L, gas_E, gas_T)

        # Water kr (intermediate phase, use mixing rule)
        # Compute two-phase water kr on unit-endpoint basis, apply endpoint
        # scaling after mixing (same rationale as oil in water-wet case).

        # Two-phase water kr from oil-water system (unit endpoint)
        movable_water_range_ow = 1.0 - Swc - Sorw
        sw_star_ow = np.where(
            movable_water_range_ow <= minimum_mobile_pore_space,
            np.zeros_like(sw),
            np.clip((sw - Swc) / movable_water_range_ow, 0.0, 1.0),
        )
        krw_ow = _let_relperm(sw_star_ow, water_L, water_E, water_T)

        # Two-phase water kr proxy from gas-water system (unit endpoint)
        movable_water_range_gw = 1.0 - Swc - Sgr
        sw_star_gw = np.where(
            movable_water_range_gw <= minimum_mobile_pore_space,
            np.zeros_like(sw),
            np.clip((sw - Swc) / movable_water_range_gw, 0.0, 1.0),
        )
        krw_gw = _let_relperm(sw_star_gw, water_L, water_E, water_T)

        krw_mixed = mixing_rule(
            kro_w=krw_ow,
            kro_g=krw_gw,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
        )
        krw = max_water_relperm * np.clip(krw_mixed, 0.0, 1.0)

    else:
        raise ValidationError(f"Wettability {wettability!r} not implemented.")

    krw = np.clip(krw, 0.0, 1.0)
    kro = np.clip(kro, 0.0, 1.0)
    krg = np.clip(krg, 0.0, 1.0)
    return krw, kro, krg  # type: ignore[return-value]


def _let_curve_slope_wrt_normalized_saturation(
    normalized_saturation: npt.NDArray,
    L: float,
    E: float,
    T: float,
    kr_max: float,
) -> npt.NDArray:
    """
    Analytical derivative of the LET relative permeability curve with
    respect to normalized (effective) saturation.

    The LET curve is:

    ```
    kr = kr_max * S*^L / (S*^L + E * (1-S*)^T)
    ```

    Applying the quotient rule and simplifying:

    ```
    dkr / dS* = kr_max * E * S*^(L-1) * (1-S*)^(T-1)
                * [L*(1-S*) + T*S*] / (S*^L + E*(1-S*)^T)^2
    ```

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
    numerator = (
        E * (s ** (L - 1.0)) * ((1.0 - s) ** (T - 1.0)) * (L * (1.0 - s) + T * s)
    )
    slope = kr_max * numerator / (safe_denominator**2)
    slope = np.where(normalized_saturation <= 0.0, 0.0, slope)
    slope = np.where(normalized_saturation >= 1.0, 0.0, slope)
    return slope


@relperm_table
@attrs.frozen
class LETThreePhaseRelPermModel(
    RelativePermeabilityTable,
    serializers={"mixing_rule": serialize_mixing_rule},
    deserializers={"mixing_rule": deserialize_mixing_rule},
    load_exclude={"supports_arrays"},
    dump_exclude={"supports_arrays"},
):
    """
    LET (Lomeland-Ebeltoft-Thomas) three-phase relative permeability model.

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
    scalar and array inputs for saturations (`supports_arrays=True`).
    """

    __type__ = "let_three_phase_relperm_model"

    irreducible_water_saturation: typing.Optional[float] = None
    """(Default) Irreducible water saturation (Swc)."""
    residual_oil_saturation_water: typing.Optional[float] = None
    """(Default) Residual oil saturation after water flood (Sorw)."""
    residual_oil_saturation_gas: typing.Optional[float] = None
    """(Default) Residual oil saturation after gas flood (Sorg)."""
    residual_gas_saturation: typing.Optional[float] = None
    """(Default) Residual gas saturation (Sgr)."""

    water: LETParameters = LETParameters()
    """LET parameters for the water relative permeability curve."""
    oil_water: LETParameters = LETParameters()
    """LET parameters for oil relative permeability in the water-oil system."""
    gas_oil: LETParameters = LETParameters()
    """LET parameters for oil relative permeability in the gas-oil system."""
    gas: LETParameters = LETParameters()
    """LET parameters for the gas relative permeability curve."""

    max_water_relperm: float = 1.0
    """Endpoint (maximum) relative permeability for water."""
    max_oil_relperm: float = 1.0
    """Endpoint (maximum) relative permeability for oil."""
    max_gas_relperm: float = 1.0
    """Endpoint (maximum) relative permeability for gas."""

    wettability: Wettability = Wettability.WATER_WET
    """Wettability type (water-wet or oil-wet)."""
    mixing_rule: typing.Union[MixingRule, str] = eclipse_rule
    """
    Mixing rule function or name to compute oil relative permeability in
    three-phase system. Accepts a function or a registered name string.
    """
    supports_arrays: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array inputs."""

    def __attrs_post_init__(self) -> None:
        mixing_rule = self.mixing_rule
        if isinstance(mixing_rule, str):
            object.__setattr__(self, "mixing_rule", get_mixing_rule(mixing_rule))

    def get_relative_permeabilities(  # type: ignore[override]
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        irreducible_water_saturation: typing.Optional[float] = None,
        residual_oil_saturation_water: typing.Optional[float] = None,
        residual_oil_saturation_gas: typing.Optional[float] = None,
        residual_gas_saturation: typing.Optional[float] = None,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for water, oil, and gas using the
        LET correlation.

        Supports both scalar and array inputs for saturations.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param irreducible_water_saturation: Optional override for Swc.
        :param residual_oil_saturation_water: Optional override for Sorw.
        :param residual_oil_saturation_gas: Optional override for Sorg.
        :param residual_gas_saturation: Optional override for Sgr.
        :return: `RelativePermeabilities` dictionary.
        """
        Sorw = (
            residual_oil_saturation_water
            if residual_oil_saturation_water is not None
            else self.residual_oil_saturation_water
        )
        Sorg = (
            residual_oil_saturation_gas
            if residual_oil_saturation_gas is not None
            else self.residual_oil_saturation_gas
        )
        Sgr = (
            residual_gas_saturation
            if residual_gas_saturation is not None
            else self.residual_gas_saturation
        )
        Swc = (
            irreducible_water_saturation
            if irreducible_water_saturation is not None
            else self.irreducible_water_saturation
        )
        params_missing = []
        if Swc is None:
            params_missing.append("Swc")
        if Sorw is None:
            params_missing.append("Sorw")
        if Sorg is None:
            params_missing.append("Sorg")
        if Sgr is None:
            params_missing.append("Sgr")
        if params_missing:
            raise ValidationError(
                f"Residual saturations must be provided either as arguments or set in the model instance. "
                f"Missing: {', '.join(params_missing)}"
            )

        krw, kro, krg = compute_let_three_phase_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=Swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=Sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=Sorg,  # type: ignore[arg-type]
            residual_gas_saturation=Sgr,  # type: ignore[arg-type]
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
            max_water_relperm=self.max_water_relperm,
            max_oil_relperm=self.max_oil_relperm,
            max_gas_relperm=self.max_gas_relperm,
            wettability=self.wettability,
            mixing_rule=self.mixing_rule,  # type: ignore[arg-type]
            saturation_epsilon=c.SATURATION_EPSILON,
            minimum_mobile_pore_space=c.MINIMUM_MOBILE_PORE_SPACE,
        )
        return RelativePermeabilities(water=krw, oil=kro, gas=krg)  # type: ignore[typeddict-item]

    def get_relative_permeability_derivatives(  # type: ignore[override]
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        irreducible_water_saturation: typing.Optional[float] = None,
        residual_oil_saturation_water: typing.Optional[float] = None,
        residual_oil_saturation_gas: typing.Optional[float] = None,
        residual_gas_saturation: typing.Optional[float] = None,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute all nine partial derivatives of the three-phase relative
        permeabilities with respect to water saturation, oil saturation, and
        gas saturation using the LET model.

        Returns a dictionary containing:

        ```
        (dkrw/dSw, dkrw/dSo, dkrw/dSg,
        dkro/dSw, dkro/dSo, dkro/dSg,
        dkrg/dSw, dkrg/dSo, dkrg/dSg)
        ```

        For the water-wet case all LET curve derivatives are computed
        analytically via the closed-form quotient-rule formula (see
        `_let_curve_slope_wrt_normalized_saturation`).
        The chain rule propagates these through the effective saturation
        normalisation to give derivatives with respect to physical saturation.
        The three-phase oil relative permeability derivative is then completed
        by the chain rule through the mixing rule.

        For the oil-wet case all nine derivatives fall back to central finite
        differences on the full LET function.

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
        :return: `RelativePermeabilityDerivatives` dictionary containing the partial derivatives as described above.
        """
        Swc = (
            irreducible_water_saturation
            if irreducible_water_saturation is not None
            else self.irreducible_water_saturation
        )
        Sorw = (
            residual_oil_saturation_water
            if residual_oil_saturation_water is not None
            else self.residual_oil_saturation_water
        )
        Sorg = (
            residual_oil_saturation_gas
            if residual_oil_saturation_gas is not None
            else self.residual_oil_saturation_gas
        )
        Sgr = (
            residual_gas_saturation
            if residual_gas_saturation is not None
            else self.residual_gas_saturation
        )

        params_missing = []
        if Swc is None:
            params_missing.append("Swc")
        if Sorw is None:
            params_missing.append("Sorw")
        if Sorg is None:
            params_missing.append("Sorg")
        if Sgr is None:
            params_missing.append("Sgr")
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
        krw_max = self.max_water_relperm
        kro_max = self.max_oil_relperm
        krg_max = self.max_gas_relperm

        is_scalar = np.isscalar(water_saturation)
        sw = np.atleast_1d(water_saturation)
        so = np.atleast_1d(oil_saturation)
        sg = np.atleast_1d(gas_saturation)
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        zeros = np.zeros_like(sw)
        saturation_epsilon = c.SATURATION_EPSILON
        minimum_mobile_pore_space = c.MINIMUM_MOBILE_PORE_SPACE

        if wettability == Wettability.OIL_WET:
            finite_difference_epsilon = c.FINITE_DIFFERENCE_EPSILON
            inv_2_epsilon = 1.0 / (2.0 * finite_difference_epsilon)

            def _let(sw, so, sg):
                return compute_let_three_phase_relative_permeabilities(
                    water_saturation=sw,
                    oil_saturation=so,
                    gas_saturation=sg,
                    irreducible_water_saturation=Swc,  # type: ignore[arg-type]
                    residual_oil_saturation_water=Sorw,  # type: ignore[arg-type]
                    residual_oil_saturation_gas=Sorg,  # type: ignore[arg-type]
                    residual_gas_saturation=Sgr,  # type: ignore[arg-type]
                    water_L=water_params.L,
                    water_E=water_params.E,
                    water_T=water_params.T,
                    oil_water_L=oil_water_params.L,
                    oil_water_E=oil_water_params.E,
                    oil_water_T=oil_water_params.T,
                    gas_oil_L=gas_oil_params.L,
                    gas_oil_E=gas_oil_params.E,
                    gas_oil_T=gas_oil_params.T,
                    gas_L=gas_params.L,
                    gas_E=gas_params.E,
                    gas_T=gas_params.T,
                    max_water_relperm=krw_max,
                    max_oil_relperm=kro_max,
                    max_gas_relperm=krg_max,
                    wettability=wettability,
                    mixing_rule=mixing_rule,
                    saturation_epsilon=saturation_epsilon,
                    minimum_mobile_pore_space=minimum_mobile_pore_space,
                )

            krw_sw_fwd, kro_sw_fwd, krg_sw_fwd = _let(
                sw + finite_difference_epsilon, so, sg
            )
            krw_sw_bwd, kro_sw_bwd, krg_sw_bwd = _let(
                sw - finite_difference_epsilon, so, sg
            )
            krw_so_fwd, kro_so_fwd, krg_so_fwd = _let(
                sw, so + finite_difference_epsilon, sg
            )
            krw_so_bwd, kro_so_bwd, krg_so_bwd = _let(
                sw, so - finite_difference_epsilon, sg
            )
            krw_sg_fwd, kro_sg_fwd, krg_sg_fwd = _let(
                sw, so, sg + finite_difference_epsilon
            )
            krw_sg_bwd, kro_sg_bwd, krg_sg_bwd = _let(
                sw, so, sg - finite_difference_epsilon
            )

            results = (
                (krw_sw_fwd - krw_sw_bwd) * inv_2_epsilon,
                (krw_so_fwd - krw_so_bwd) * inv_2_epsilon,
                (krw_sg_fwd - krw_sg_bwd) * inv_2_epsilon,
                (kro_sw_fwd - kro_sw_bwd) * inv_2_epsilon,
                (kro_so_fwd - kro_so_bwd) * inv_2_epsilon,
                (kro_sg_fwd - kro_sg_bwd) * inv_2_epsilon,
                (krg_sw_fwd - krg_sw_bwd) * inv_2_epsilon,
                (krg_so_fwd - krg_so_bwd) * inv_2_epsilon,
                (krg_sg_fwd - krg_sg_bwd) * inv_2_epsilon,
            )
            if is_scalar:
                results = tuple(float(np.ravel(r)[0]) for r in results)
            return RelativePermeabilityDerivatives(
                dKrw_dSw=results[0],
                dKro_dSw=results[1],
                dKrg_dSw=results[2],
                dKrw_dSo=results[3],
                dKro_dSo=results[4],
                dKrg_dSo=results[5],
                dKrw_dSg=results[6],
                dKro_dSg=results[7],
                dKrg_dSg=results[8],
            )

        # Water-wet path
        # krw
        mobile_water_range = 1.0 - Swc - Sorw  # type: ignore
        valid_water = mobile_water_range > minimum_mobile_pore_space
        se_w_for_krw = np.clip(
            (sw - Swc) / np.where(valid_water, mobile_water_range, 1.0),
            0.0,
            1.0,
        )
        d_krw_d_sw = np.where(
            valid_water,
            _let_curve_slope_wrt_normalized_saturation(
                normalized_saturation=se_w_for_krw,
                L=water_params.L,
                E=water_params.E,
                T=water_params.T,
                kr_max=krw_max,
            )
            / mobile_water_range,
            zeros,
        )
        d_krw_d_so = zeros.copy()
        d_krw_d_sg = zeros.copy()

        # krg
        mobile_gas_range = 1.0 - Swc - Sgr - Sorg  # type: ignore
        valid_gas = mobile_gas_range > minimum_mobile_pore_space
        se_g_for_krg = np.clip(
            (sg - Sgr) / np.where(valid_gas, mobile_gas_range, 1.0),
            0.0,
            1.0,
        )
        d_krg_d_sg = np.where(
            valid_gas,
            _let_curve_slope_wrt_normalized_saturation(
                normalized_saturation=se_g_for_krg,
                L=gas_params.L,
                E=gas_params.E,
                T=gas_params.T,
                kr_max=krg_max,
            )
            / mobile_gas_range,
            zeros,
        )
        d_krg_d_sw = zeros.copy()
        d_krg_d_so = zeros.copy()

        # kro_w (unit-endpoint oil kr from water-oil system, function of So)
        mobile_oil_water_range = 1.0 - Swc - Sorw  # type: ignore
        valid_oil_water = mobile_oil_water_range > minimum_mobile_pore_space
        se_o_water_system = np.clip(
            (so - Sorw) / np.where(valid_oil_water, mobile_oil_water_range, 1.0),
            0.0,
            1.0,
        )
        d_kro_w_d_so = np.where(
            valid_oil_water,
            _let_curve_slope_wrt_normalized_saturation(
                normalized_saturation=se_o_water_system,
                L=oil_water_params.L,
                E=oil_water_params.E,
                T=oil_water_params.T,
                kr_max=1.0,
            )
            / mobile_oil_water_range,
            zeros,
        )
        s = np.clip(se_o_water_system, 1e-15, 1.0 - 1e-15)
        kro_w_vals = np.where(
            se_o_water_system > 0.0,
            s**oil_water_params.L
            / (
                s**oil_water_params.L
                + oil_water_params.E * (1.0 - s) ** oil_water_params.T
            ),
            zeros,
        )
        d_kro_w_d_sw = zeros.copy()
        d_kro_w_d_sg = zeros.copy()

        # kro_g (unit-endpoint oil kr from gas-oil system, function of So)
        mobile_gas_oil_range = 1.0 - Swc - Sorg - Sgr  # type: ignore
        valid_gas_oil = mobile_gas_oil_range > minimum_mobile_pore_space
        se_o_gas_system = np.clip(
            (so - Sorg) / np.where(valid_gas_oil, mobile_gas_oil_range, 1.0),
            0.0,
            1.0,
        )
        d_kro_g_d_so = np.where(
            valid_gas_oil,
            _let_curve_slope_wrt_normalized_saturation(
                normalized_saturation=se_o_gas_system,
                L=gas_oil_params.L,
                E=gas_oil_params.E,
                T=gas_oil_params.T,
                kr_max=1.0,
            )
            / mobile_gas_oil_range,
            zeros,
        )
        s = np.clip(se_o_gas_system, 1e-15, 1.0 - 1e-15)
        kro_g_vals = np.where(
            se_o_gas_system > 0.0,
            s**gas_oil_params.L
            / (s**gas_oil_params.L + gas_oil_params.E * (1.0 - s) ** gas_oil_params.T),
            zeros,
        )
        d_kro_g_d_sw = zeros.copy()
        d_kro_g_d_sg = zeros.copy()

        (
            d_kro_d_kro_w,
            d_kro_d_kro_g,
            d_kro_d_water_saturation_explicit,
            d_kro_d_oil_saturation_explicit,
            d_kro_d_gas_saturation_explicit,
        ) = _mixing_rule_partial_derivatives(
            mixing_rule=mixing_rule,
            kro_water_two_phase=kro_w_vals,
            kro_gas_two_phase=kro_g_vals,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            epsilon=c.FINITE_DIFFERENCE_EPSILON,
        )

        d_kro_d_sw = kro_max * (
            d_kro_d_kro_w * d_kro_w_d_sw
            + d_kro_d_kro_g * d_kro_g_d_sw
            + d_kro_d_water_saturation_explicit
        )
        d_kro_d_so = kro_max * (
            d_kro_d_kro_w * d_kro_w_d_so
            + d_kro_d_kro_g * d_kro_g_d_so
            + d_kro_d_oil_saturation_explicit
        )
        d_kro_d_sg = kro_max * (
            d_kro_d_kro_w * d_kro_w_d_sg
            + d_kro_d_kro_g * d_kro_g_d_sg
            + d_kro_d_gas_saturation_explicit
        )

        results = (
            d_krw_d_sw,
            d_kro_d_sw,
            d_krg_d_sw,
            d_krw_d_so,
            d_kro_d_so,
            d_krg_d_so,
            d_krw_d_sg,
            d_kro_d_sg,
            d_krg_d_sg,
        )
        if is_scalar:
            results = tuple(float(np.ravel(r)[0]) for r in results)
            return RelativePermeabilityDerivatives(
                dKrw_dSw=results[0],
                dKro_dSw=results[1],
                dKrg_dSw=results[2],
                dKrw_dSo=results[3],
                dKro_dSo=results[4],
                dKrg_dSo=results[5],
                dKrw_dSg=results[6],
                dKro_dSg=results[7],
                dKrg_dSg=results[8],
            )
        return RelativePermeabilityDerivatives(
            dKrw_dSw=d_krw_d_sw,
            dKro_dSw=d_kro_d_sw,
            dKrg_dSw=d_krg_d_sw,
            dKrw_dSo=d_krw_d_so,
            dKro_dSo=d_kro_d_so,
            dKrg_dSo=d_krg_d_so,
            dKrw_dSg=d_krw_d_sg,
            dKro_dSg=d_kro_d_sg,
            dKrg_dSg=d_krg_d_sg,
        )

    def __call__(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for water, oil, and gas.

        Supports both scalar and array inputs for saturations.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :kwarg irreducible_water_saturation: Optional override for Swc.
        :kwarg residual_oil_saturation_water: Optional override for Sorw.
        :kwarg residual_oil_saturation_gas: Optional override for Sorg.
        :kwarg residual_gas_saturation: Optional override for Sgr.
        :return: `RelativePermeabilities` dictionary.
        """
        return self.get_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=kwargs.get(
                "irreducible_water_saturation", None
            ),
            residual_oil_saturation_water=kwargs.get(
                "residual_oil_saturation_water", None
            ),
            residual_oil_saturation_gas=kwargs.get("residual_oil_saturation_gas", None),
            residual_gas_saturation=kwargs.get("residual_gas_saturation", None),
        )

    def derivatives(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute relative permeability derivatives for water, oil, and gas.

        Supports both scalar and array inputs for saturations.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :kwarg irreducible_water_saturation: Optional override for irreducible water saturation.
        :kwarg residual_oil_saturation_water: Optional override for residual oil saturation after water flood.
        :kwarg residual_oil_saturation_gas: Optional override for residual oil saturation after gas flood.
        :kwarg residual_gas_saturation: Optional override for residual gas saturation.
        :return: `RelativePermeabilityDerivatives` dictionary.
        0 <= water_saturation, oil_saturation, gas_saturation <= 1
        """
        return self.get_relative_permeability_derivatives(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=kwargs.get(
                "irreducible_water_saturation", None
            ),
            residual_oil_saturation_water=kwargs.get(
                "residual_oil_saturation_water", None
            ),
            residual_oil_saturation_gas=kwargs.get("residual_oil_saturation_gas", None),
            residual_gas_saturation=kwargs.get("residual_gas_saturation", None),
        )

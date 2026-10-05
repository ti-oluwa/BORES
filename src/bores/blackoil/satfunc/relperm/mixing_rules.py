"""
Relative permeability mixing rules.

Available three-phase relative permeability mixing rules:

| Rule                | Conservativeness     | Complexity  | Typical Use Case                            |
|---------------------|----------------------|-------------|---------------------------------------------|
| Min                 | Very conservative    | Simple      | Lower bound, safety factor                  |
| Harmonic Mean       | Very conservative    | Simple      | Series flow, tight rocks                    |
| Geometric Mean      | Conservative         | Simple      | General purpose                             |
| Stone I             | Moderate             | Moderate    | Water-wet systems                           |
| Stone II            | Moderate             | Moderate    | Standard industry practice                  |
| Arithmetic Mean     | Optimistic           | Simple      | Upper bound estimate                        |
| Max                 | Very optimistic      | Simple      | Upper bound, sensitivity                    |
| Blunt               | Conservative         | Moderate    | Strong water-wet                            |
| Eclipse             | Moderate             | Moderate    | Commercial simulator standard               |
| Aziz-Settari        | Variable             | Moderate    | Empirical tuning                            |

Note: At points where derivatives are non-unique (e.g., min rule at kro_w == kro_g),
the **subgradient convention** is used: both derivatives set to 0.5.
This ensures the expected behavior in solvers (Newton's method is robust to
subgradient choices as long as they're in the subdifferential).
"""

import threading
import typing
from collections.abc import Mapping

import attrs
import numba
import numpy as np
from numba.extending import overload

from bores.errors import ValidationError
from bores.types import (
    MixingRuleDFunc,
    MixingRuleFunc,
    MixingRulePartialDerivatives,
    NDimension,
    Number,
    NumberOrArray,
)

__all__ = [
    "arithmetic_mean_rule",
    "aziz_settari_rule",
    "baker_linear_rule",
    "blunt_rule",
    "eclipse_rule",
    "geometric_mean_rule",
    "harmonic_mean_rule",
    "hustad_hansen_rule",
    "max_rule",
    "mixing_rule",
    "stone_II_rule",
    "stone_I_rule",
]


@typing.final
@attrs.mutable
class MixingRule:
    """
    Wraps a mixing-rule callable together with an optional analytical
    partial-derivative function.

    **Construction**:

    Normally produced by the `@mixing_rule` decorator, which registers the
    rule and returns a `MixingRule` instance. You can also build one directly:

    ``python
    my_rule = MixingRule(func=my_func)
    ``

    **Attaching an analytical derivative later**:

    Use `MixingRule.dfunc` as a decorator:

    ``python
    @my_rule.dfunc
    def _(kro_w, kro_g, krw, krg, kr_max,
          water_saturation, oil_saturation, gas_saturation):
        ...
        return MixingRulePartialDerivatives(...)
    ``

    **Protocol-compatible objects**:

    If the wrapped callable also has a `partial_derivatives` method (i.e. it
    is a class instance that matches `MixingRuleFunc` and exposes its own
    derivative logic), that method is automatically promoted to `_dfunc`
    during post-initialization.
    """

    func: typing.Union[MixingRuleFunc, "MixingRule"]
    _dfunc: MixingRuleDFunc | None = attrs.field(default=None, alias="dfunc")

    def __attrs_post_init__(self) -> None:
        # If the wrapped callable advertises its own partial_derivatives,
        # promote it automatically (protocol-compatible class pattern).
        if self._dfunc is None:
            dfunc = getattr(self.func, "partial_derivatives", None)
            if callable(dfunc):
                self._dfunc = typing.cast(MixingRuleDFunc, dfunc)

    def dfunc(self, fn: MixingRuleDFunc, /) -> MixingRuleDFunc:
        """
        Decorator that registers an analytical partial-derivative function for
        this mixing rule.

        Usage:

        ``python
        @stone_I_rule.dfunc
        def _(kro_w, kro_g, krw, krg, kr_max,
              water_saturation, oil_saturation, gas_saturation):
            ...
            return MixingRulePartialDerivatives(...)
        ``
        """
        self._dfunc = fn
        return fn

    def __call__(
        self,
        kro_w: NumberOrArray[NDimension],
        kro_g: NumberOrArray[NDimension],
        krw: NumberOrArray[NDimension],
        krg: NumberOrArray[NDimension],
        kr_max: NumberOrArray[NDimension],
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
    ) -> NumberOrArray[NDimension]:
        return self.func(
            kro_w=kro_w,
            kro_g=kro_g,
            krw=krw,
            krg=krg,
            kr_max=kr_max,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
        )

    def partial_derivatives(
        self,
        kro_w: NumberOrArray[NDimension],
        kro_g: NumberOrArray[NDimension],
        krw: NumberOrArray[NDimension],
        krg: NumberOrArray[NDimension],
        kr_max: NumberOrArray[NDimension],
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        epsilon: Number = 1e-7,
    ) -> MixingRulePartialDerivatives:
        """
        Return the seven partial derivatives of this mixing rule.

        If an analytical derivative function has been registered (via
        `@rule.dfunc` or auto-detected from `func.partial_derivatives`),
        it is called directly. Otherwise the derivatives are estimated with
        central finite differences.

        :param kro_w: Two-phase oil relative permeability from the oil-water
            table at the current iterate.
        :param kro_g: Two-phase oil relative permeability from the gas-oil
            table at the current iterate.
        :param krw: Two-phase water relative permeability from the oil-water
            table at the current iterate.
        :param krg: Two-phase gas relative permeability from the gas-oil
            table at the current iterate.
        :param kr_max: Oil relative permeability at connate water
            (normalization reference).
        :param water_saturation: Current water saturation.
        :param oil_saturation: Current oil saturation.
        :param gas_saturation: Current gas saturation.
        :param epsilon: Step size for central differences (used only in the
            fallback path). Defaults to 1e-7.
        :return: A `MixingRulePartialDerivatives` dictionary containing the
            seven partial derivatives.
        """
        if self._dfunc is not None:
            derivatives = self._dfunc(
                kro_w=kro_w,
                kro_g=kro_g,
                krw=krw,
                krg=krg,
                kr_max=kr_max,
                water_saturation=water_saturation,
                oil_saturation=oil_saturation,
                gas_saturation=gas_saturation,
            )
            if isinstance(derivatives, Mapping):
                return derivatives  # type: ignore[return-value]

            # Unpack 7-tuple: (dkro_w, dkro_g, dkrw, dkrg, dsw, dso, dsg)
            return MixingRulePartialDerivatives(
                dkro_dkro_w=derivatives[0],
                dkro_dkro_g=derivatives[1],
                dkro_dkrw=derivatives[2],
                dkro_dkrg=derivatives[3],
                dkro_dsw_explicit=derivatives[4],
                dkro_dso_explicit=derivatives[5],
                dkro_dsg_explicit=derivatives[6],
            )
        return compute_central_difference_partial_derivatives(
            rule=self,
            kro_w=kro_w,
            kro_g=kro_g,
            krw=krw,
            krg=krg,
            kr_max=kr_max,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            epsilon=epsilon,
        )

    def __str__(self) -> str:
        return (
            str(self.func.__name__)  # type: ignore[attr-defined]
            if hasattr(self.func, "__name__")
            else repr(self.func)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(func={self.func!r}, dfunc={self._dfunc!r})"

    def __hash__(self) -> int:
        def resolve_callable_identity(obj: typing.Any):
            # unwrap nested `MixingRule`
            if isinstance(obj, MixingRule):
                return resolve_callable_identity(obj.func)
            return id(obj)

        func_id = resolve_callable_identity(self.func)
        dfunc_id = id(self._dfunc) if self._dfunc is not None else 0
        return hash((func_id, dfunc_id))


def compute_central_difference_partial_derivatives(
    rule: MixingRuleFunc,
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    epsilon: Number = 1e-7,
) -> MixingRulePartialDerivatives:
    """
    Compute all seven partial derivatives of a mixing rule via central finite
    differences.

    This function approximates the partial derivatives numerically when an
    analytical derivative function is not available. It computes central
    differences for `kro_w`, `kro_g`, `krw`, `krg`, `Sw`, `So`,
    and `Sg` - fourteen mixing-rule evaluations in total.

    Uses **relative perturbation** for accuracy across different value scales:

    h_var = epsilon * max(|var|, 1.0)

    Then:

    df/d(var) ≈ (f(var + h_var) - f(var - h_var)) / (2 * h_var)

    **Cost**: Fourteen mixing rule evaluations (seven central difference pairs),
    each O(1), regardless of grid size.

    :param rule: The mixing rule callable to differentiate.
    :param kro_w: Oil relative permeability from the oil-water two-phase table.
    :param kro_g: Oil relative permeability from the gas-oil two-phase table.
    :param krw: Water relative permeability from the oil-water two-phase table.
    :param krg: Gas relative permeability from the gas-oil two-phase table.
    :param kr_max: Oil relative permeability at connate water.
    :param water_saturation: Current water saturation.
    :param oil_saturation: Current oil saturation.
    :param gas_saturation: Current gas saturation.
    :param epsilon: Finite difference step size (default: 1e-7).
    :return: `MixingRulePartialDerivatives` dictionary containing all seven
        partial derivatives.
    """
    # Compute relative step sizes for each variable

    # For kro_w: h scales with magnitude of kro_w
    scale_kro_w = np.maximum(np.abs(kro_w), 1.0)
    h_kro_w = epsilon * scale_kro_w
    two_h_kro_w = 2.0 * h_kro_w

    # For kro_g: h scales with magnitude of kro_g
    scale_kro_g = np.maximum(np.abs(kro_g), 1.0)
    h_kro_g = epsilon * scale_kro_g
    two_h_kro_g = 2.0 * h_kro_g

    # For krw: h scales with magnitude of krw
    scale_krw = np.maximum(np.abs(krw), 1.0)
    h_krw = epsilon * scale_krw
    two_h_krw = 2.0 * h_krw

    # For krg: h scales with magnitude of krg
    scale_krg = np.maximum(np.abs(krg), 1.0)
    h_krg = epsilon * scale_krg
    two_h_krg = 2.0 * h_krg

    # For water_saturation: h scales with magnitude of Sw
    scale_sw = np.maximum(np.abs(water_saturation), 1.0)
    h_sw = epsilon * scale_sw
    two_h_sw = 2.0 * h_sw

    # For oil_saturation: h scales with magnitude of So
    scale_so = np.maximum(np.abs(oil_saturation), 1.0)
    h_so = epsilon * scale_so
    two_h_so = 2.0 * h_so

    # For gas_saturation: h scales with magnitude of Sg
    scale_sg = np.maximum(np.abs(gas_saturation), 1.0)
    h_sg = epsilon * scale_sg
    two_h_sg = 2.0 * h_sg

    # Compute ∂f/∂(kro_w) using relative perturbation
    f_plus = rule(
        kro_w=kro_w + h_kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        kr_max=kr_max,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    f_minus = rule(
        kro_w=kro_w - h_kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        kr_max=kr_max,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    dkro_dkro_w = (f_plus - f_minus) / two_h_kro_w

    # Compute ∂f/∂(kro_g) using relative perturbation
    f_plus = rule(
        kro_w=kro_w,
        kro_g=kro_g + h_kro_g,
        krw=krw,
        krg=krg,
        kr_max=kr_max,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    f_minus = rule(
        kro_w=kro_w,
        kro_g=kro_g - h_kro_g,
        krw=krw,
        krg=krg,
        kr_max=kr_max,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    dkro_dkro_g = (f_plus - f_minus) / two_h_kro_g

    # Compute ∂f/∂(krw) using relative perturbation
    f_plus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw + h_krw,
        krg=krg,
        kr_max=kr_max,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    f_minus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw - h_krw,
        krg=krg,
        kr_max=kr_max,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    dkro_dkrw = (f_plus - f_minus) / two_h_krw

    # Compute ∂f/∂(krg) using relative perturbation
    f_plus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg + h_krg,
        kr_max=kr_max,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    f_minus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg - h_krg,
        kr_max=kr_max,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    dkro_dkrg = (f_plus - f_minus) / two_h_krg

    # Compute ∂f/∂(Sw) using relative perturbation
    f_plus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        kr_max=kr_max,
        water_saturation=water_saturation + h_sw,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    f_minus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        kr_max=kr_max,
        water_saturation=water_saturation - h_sw,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    dkro_dsw_explicit = (f_plus - f_minus) / two_h_sw

    # Compute ∂f/∂(So) using relative perturbation
    f_plus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        kr_max=kr_max,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation + h_so,
        gas_saturation=gas_saturation,
    )
    f_minus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        kr_max=kr_max,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation - h_so,
        gas_saturation=gas_saturation,
    )
    dkro_dso_explicit = (f_plus - f_minus) / two_h_so

    # Compute ∂f/∂(Sg) using relative perturbation
    f_plus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        kr_max=kr_max,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation + h_sg,
    )
    f_minus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        kr_max=kr_max,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation - h_sg,
    )
    dkro_dsg_explicit = (f_plus - f_minus) / two_h_sg

    return MixingRulePartialDerivatives(
        dkro_dkro_w=dkro_dkro_w,
        dkro_dkro_g=dkro_dkro_g,
        dkro_dkrw=dkro_dkrw,
        dkro_dkrg=dkro_dkrg,
        dkro_dsw_explicit=dkro_dsw_explicit,
        dkro_dso_explicit=dkro_dso_explicit,
        dkro_dsg_explicit=dkro_dsg_explicit,
    )


MIXING_RULES: dict[str, MixingRule] = {}
"""Registry of mixing rule functions."""
_lock = threading.Lock()


@typing.overload
def mixing_rule(func: MixingRuleFunc | MixingRule) -> MixingRule: ...
@typing.overload
def mixing_rule(
    func: None = None, name: str | None = None, override: bool = False
) -> typing.Callable[[MixingRuleFunc | MixingRule], MixingRule]: ...
@typing.overload
def mixing_rule(
    func: MixingRuleFunc | MixingRule, name: str | None = None, override: bool = False
) -> MixingRule: ...


def mixing_rule(
    func: MixingRuleFunc | MixingRule | None = None,
    name: str | None = None,
    override: bool = False,
) -> MixingRule | typing.Callable[[MixingRuleFunc | MixingRule], MixingRule]:
    """
    Decorator that registers a mixing rule function or `MixingRule` instance.

    **Behaviour**:

    - If `func` is a plain callable (matching `MixingRuleFunc`), it is wrapped
      in a new `MixingRule` and the wrapper is registered.
    - If `func` is already a `MixingRule`, it is registered as-is.
    - The registered (and possibly newly wrapped) `MixingRule` is returned, so
      the decorated name in the module namespace holds a `MixingRule`, not a
      bare function.

    **Plain function**:

    ```python
    @mixing_rule
    def my_rule(kro_w, kro_g, krw, krg, kr_max, water_saturation, oil_saturation, gas_saturation):
        return (kro_w + kro_g) / 2.0
    ```

    **Protocol-compatible class**:

    ```python
    class MyRule:
        def __call__(self, *, kro_w, kro_g, krw, krg, kr_max, ...): ...
        def partial_derivatives(self, *, kro_w, kro_g, krw, krg, kr_max, ...): ...

    my_rule = mixing_rule(MyRule())
    ```

    **With keyword arguments**:

    ```python
    @mixing_rule(name="custom", override=True)
    def my_rule(...): ...
    ```

    :param func: The function or `MixingRule` to register. When omitted, the
        decorator is called with keyword arguments and returns a one-argument decorator.
    :param name: Registry key. Defaults to `func.__name__`.
    :param override: If `False` (default), raises on duplicate names.
    :return: The registered `MixingRule` instance.
    """

    def register(
        f: MixingRuleFunc | MixingRule,
    ) -> MixingRule:
        # Determine the registry key
        rule_name = name or getattr(f, "__name__", None)
        if rule_name is None:
            raise ValueError(
                "Mixing rule must have a `__name__` attribute or `name` must be provided."
            )

        # Wrap plain callables; pass `MixingRule` instances through
        if isinstance(f, MixingRule):
            rule = f
        else:
            rule = MixingRule(func=f)

        with _lock:
            if rule_name in MIXING_RULES and not override:
                raise ValidationError(
                    f"Mixing rule '{rule_name}' is already registered. "
                    "Use `override=True` or provide a different name."
                )
            MIXING_RULES[rule_name] = rule
        return rule

    if func is None:
        return register
    return register(func)


def list_mixing_rules() -> list[str]:
    """
    List all registered mixing rule names.

    :return: List of registered mixing rule names.
    """
    with _lock:
        return list(MIXING_RULES.keys())


def get_mixing_rule(name: str) -> MixingRule:
    """
    Get a registered mixing rule function by name.

    :param name: Registered name of the mixing rule.
    :return: Mixing rule function.
    :raises ValidationError: If the mixing rule is not registered.
    """
    with _lock:
        if name in MIXING_RULES:
            return MIXING_RULES[name]
    raise ValidationError(
        f"Mixing rule '{name}' is not registered. Use `@mixing_rule` to register."
    )


def get_zeros_like_kro(kro_w: NumberOrArray) -> NumberOrArray[NDimension]:
    """Return an array (or scalar) of zeros with the same shape as kro_w."""
    return np.zeros_like(kro_w) if not np.isscalar(kro_w) else kro_w.dtype.type(0.0)  # type: ignore


@overload(get_zeros_like_kro)
def overload_get_zeros_like_kro(kro_w):
    # Scalar case
    if isinstance(kro_w, numba.types.Number):

        def impl(kro_w):
            return kro_w.dtype.type(0.0)

        return impl

    # Array case
    if isinstance(kro_w, numba.types.Array):

        def impl(kro_w):
            return np.zeros_like(kro_w)

        return impl


@mixing_rule
@numba.njit(cache=True)
def min_rule(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
    """
    Conservative rule for 3-phase oil relative permeability.

    kro = min(kro_w, kro_g)
    """
    return np.minimum(kro_w, kro_g)  # type: ignore[return-value]


@min_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> tuple[
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
]:
    """
    Analytical derivatives for min_rule.

    kro = min(kro_w, kro_g)

    Where kro_w < kro_g  -  ∂kro/∂kro_w = 1, ∂kro/∂kro_g = 0
    Where kro_g ≤ kro_w  -  ∂kro/∂kro_w = 0, ∂kro/∂kro_g = 1
    Tie: split evenly (0.5 each) - subgradient convention.
    No dependence on krw, krg, or explicit saturations.
    """
    kw = np.asarray(kro_w, dtype=np.float64)
    kg = np.asarray(kro_g, dtype=np.float64)
    dkro_dkro_w = np.where(kw < kg, 1.0, np.where(kw > kg, 0.0, 0.5))
    dkro_dkro_g = 1.0 - dkro_dkro_w
    z = get_zeros_like_kro(kro_w)
    return (dkro_dkro_w, dkro_dkro_g, z, z, z, z, z)  # type: ignore[return-value]


@mixing_rule
@numba.njit(cache=True)
def stone_I_rule(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
    """
    Stone I rule (1970) for 3-phase oil relative permeability.
    kro = (kro_w * kro_g) / (kro_w + kro_g - kro_w * kro_g)
    """
    denom = np.maximum(((kro_w + kro_g) - (kro_w * kro_g)), 1e-12)
    result = (kro_w * kro_g) / denom
    # Return 0 if both kro_w and kro_g are zero
    return np.where((kro_w <= 0.0) & (kro_g <= 0.0), 0.0, result)  # type: ignore[return-value]


@stone_I_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> tuple[
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
]:
    """
    Analytical derivatives for Stone I.

    Let D = kro_w + kro_g - kro_w * kro_g  (clamped >= ε)
        N = kro_w * kro_g

    ∂kro/∂kro_w = (kro_g * D - N * (1 - kro_g)) / D²
                = kro_g² / D²           (after simplification)
    ∂kro/∂kro_g = kro_w² / D²          (by symmetry)
    No dependence on krw, krg, or explicit saturations.
    """
    eps = 1e-12
    kw = np.asarray(kro_w, dtype=np.float64)
    kg = np.asarray(kro_g, dtype=np.float64)
    D = np.maximum(kw + kg - kw * kg, eps)
    both_zero = (kw <= 0.0) & (kg <= 0.0)
    dkro_dkro_w = np.where(both_zero, 0.0, kg**2 / D**2)
    dkro_dkro_g = np.where(both_zero, 0.0, kw**2 / D**2)
    z = get_zeros_like_kro(kro_w)
    return (dkro_dkro_w, dkro_dkro_g, z, z, z, z, z)  # type: ignore[return-value]


@mixing_rule
@numba.njit(cache=True)
def stone_II_rule(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
    """
    Stone II rule (Stone, 1973, JPT) for 3-phase oil relative permeability.

    Full normalized formula (Aziz & Settari, 1979):

        kro = krocw * [(krow/krocw + krw) * (krog/krocw + krg) - krw - krg]

    where:
        - krow  = kro_w  (oil relperm from oil-water table at current Sw)
        - krog  = kro_g  (oil relperm from gas-oil table at current Sg)
        - krw            = two-phase water relperm at current Sw
        - krg            = two-phase gas relperm at current Sg
        - krocw = kr_max  (oil relperm at connate water, normalization ref)

    This is the **exact** Stone II formulation, valid for both normalized
    (krocw = 1) and non-normalized tables.  Using the actual krw and krg
    from the two-phase tables removes the approximation `krw ≈ 1 - kro_w`
    that was previously needed when only kro_w and kro_g were available.

    **Reference:**
    Stone, H.L. (1973). "Estimation of Three-Phase Relative Permeability and
    Residual Oil Data." Journal of Canadian Petroleum Technology, 12(4), 53-61.
    Aziz, K. & Settari, A. (1979). Petroleum Reservoir Simulation. Applied
    Science Publishers, London.
    """
    krocw = np.maximum(kr_max, 1e-12)
    result = (
        (kro_w / krocw + krw) * (kro_g / krocw + krg) - krw - krg
    )  # Normalization should occur in caller
    # Clamp negative values (can occur when both kro_w and kro_g are small)
    result = np.maximum(result, 0.0)
    # Return zero if either two-phase oil relperm is zero (conservative)
    return np.where((kro_w <= 0.0) | (kro_g <= 0.0), 0.0, result)  # type: ignore[return-value]


@stone_II_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> tuple[
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
]:
    """
    Analytical derivatives for Stone II (full normalized form).

    Let krocw = kr_max,  A = kro_w/krocw + krw,  B = kro_g/krocw + krg

    kro = krocw * (A * B - krw - krg)

    Active region (kro > 0, both kro_w > 0 and kro_g > 0):

        ∂kro/∂kro_w = B          (from ∂(krocw * A * B)/∂kro_w = B)
        ∂kro/∂kro_g = A          (by symmetry)
        ∂kro/∂krw   = krocw * (B - 1)
        ∂kro/∂krg   = krocw * (A - 1)

    No explicit saturation dependence.
    """
    krocw = np.maximum(kr_max, 1e-12)
    kw = np.asarray(kro_w, dtype=np.float64)
    kg = np.asarray(kro_g, dtype=np.float64)
    w = np.asarray(krw, dtype=np.float64)
    g = np.asarray(krg, dtype=np.float64)

    A = kw / krocw + w
    B = kg / krocw + g
    raw = krocw * (A * B - w - g)
    active = (kw > 0.0) & (kg > 0.0) & (raw > 0.0)

    dkro_dkro_w = np.where(active, B, 0.0)
    dkro_dkro_g = np.where(active, A, 0.0)
    dkro_dkrw = np.where(active, krocw * (B - 1.0), 0.0)
    dkro_dkrg = np.where(active, krocw * (A - 1.0), 0.0)
    z = get_zeros_like_kro(kro_w)
    return (dkro_dkro_w, dkro_dkro_g, dkro_dkrw, dkro_dkrg, z, z, z)  # type: ignore[return-value]


@mixing_rule
@numba.njit(cache=True)
def arithmetic_mean_rule(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
    """
    Simple arithmetic mean of oil-water and oil-gas relative permeabilities.

    kro = (kro_w + kro_g) / 2

    Notes:
    - Simple and conservative
    - Does not account for saturation distribution
    - Tends to overestimate kro compared to other methods
    """
    return (kro_w + kro_g) / 2.0  # type: ignore[return-value]


@arithmetic_mean_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> tuple[
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
]:
    """∂kro/∂kro_w = 0.5, ∂kro/∂kro_g = 0.5, no dependence on krw, krg, or saturations."""
    half = np.full_like(np.asarray(kro_w, dtype=np.float64), 0.5)
    z = get_zeros_like_kro(kro_w)
    return (half, half, z, z, z, z, z)  # type: ignore[return-value]


@mixing_rule
@numba.njit(cache=True)
def geometric_mean_rule(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
    """
    Geometric mean of oil-water and oil-gas relative permeabilities.

    kro = sqrt(kro_w * kro_g)

    Notes:
    - More conservative than arithmetic mean
    - If either kro_w or kro_g is zero, result is zero
    - Smooth transition between two-phase limits
    """
    return np.sqrt(kro_w * kro_g)  # type: ignore[return-value]


@geometric_mean_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> tuple[
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
]:
    """
    kro = sqrt(kw * kg)

    ∂kro/∂kro_w = 0.5 * sqrt(kro_g / kro_w)   (= kro_g / (2 * kro))
    ∂kro/∂kro_g = 0.5 * sqrt(kro_w / kro_g)
    Zero when either input is zero.
    """
    kw = np.asarray(kro_w, dtype=np.float64)
    kg = np.asarray(kro_g, dtype=np.float64)
    kro = np.sqrt(kw * kg)
    safe_kro = np.maximum(kro, 1e-30)
    dkro_dkro_w = np.where(kro > 0.0, 0.5 * kg / safe_kro, 0.0)
    dkro_dkro_g = np.where(kro > 0.0, 0.5 * kw / safe_kro, 0.0)
    z = get_zeros_like_kro(kro_w)
    return (dkro_dkro_w, dkro_dkro_g, z, z, z, z, z)  # type: ignore[return-value]


@mixing_rule
@numba.njit(cache=True)
def harmonic_mean_rule(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
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
    return np.where((kro_w <= 0.0) | (kro_g <= 0.0), 0.0, result)  # type: ignore[return-value]


@harmonic_mean_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> tuple[
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
]:
    """
    kro = 2 kw kg / (kw + kg)

    ∂kro/∂kw = 2 kg² / (kw + kg)²
    ∂kro/∂kg = 2 kw² / (kw + kg)²
    Zero when either input is zero.
    """
    kw = np.asarray(kro_w, dtype=np.float64)
    kg = np.asarray(kro_g, dtype=np.float64)
    both_positive = (kw > 0.0) & (kg > 0.0)
    safe_sum = np.where(both_positive, kw + kg, 1.0)
    dkro_dkro_w = np.where(both_positive, 2.0 * kg**2 / safe_sum**2, 0.0)
    dkro_dkro_g = np.where(both_positive, 2.0 * kw**2 / safe_sum**2, 0.0)
    z = get_zeros_like_kro(kro_w)
    return (dkro_dkro_w, dkro_dkro_g, z, z, z, z, z)  # type: ignore[return-value]


@mixing_rule
@numba.njit(cache=True)
def baker_linear_rule(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
    """
    Baker's linear interpolation rule (1988).

    Interpolates three-phase kro as a saturation-weighted combination
    of the two-phase endpoint values:

        kro = (Sw * kro_w + So * kro_ow_endpoint + Sg * kro_g) / (Sw + So + Sg)

    where kro_ow_endpoint is approximated as max(kro_w, kro_g) (the oil kr
    at the oil-water endpoint, i.e. in absence of gas), and kro_g is the
    oil kr from the gas-oil table.

    Simplification used here (standard Baker linear, no separate endpoint table):

        kro = (Sw * kro_w + Sg * kro_g) / (Sw + Sg)   when Sw+Sg > 0
        kro = max(kro_w, kro_g)                       when Sw+Sg = 0 (pure oil)

    This reduces exactly to kro_w at Sg=0 and kro_g at Sw=0.
    """
    total_displacing = water_saturation + gas_saturation
    result = np.where(
        total_displacing > 0.0,
        (kro_w * water_saturation + kro_g * gas_saturation) / total_displacing,
        np.maximum(kro_w, kro_g),
    )
    return result  # type: ignore[return-value]


@baker_linear_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> tuple[
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
]:
    """
    ∂kro/∂kw  = Sw / T
    ∂kro/∂kg  = Sg / T
    ∂kro/∂Sw  = Sg * (kw - kg) / T²
    ∂kro/∂Sg  = Sw * (kg - kw) / T²
    ∂kro/∂So  = 0
    No dependence on krw or krg.
    """
    kw = np.asarray(kro_w, dtype=np.float64)
    kg = np.asarray(kro_g, dtype=np.float64)
    sw = np.asarray(water_saturation, dtype=np.float64)
    sg = np.asarray(gas_saturation, dtype=np.float64)

    T = sw + sg
    active = T > 0.0
    T_safe = np.where(active, T, 1.0)

    dkro_dkro_w = np.where(active, sw / T_safe, 0.0)
    dkro_dkro_g = np.where(active, sg / T_safe, 0.0)
    dsw = np.where(active, sg * (kw - kg) / T_safe**2, 0.0)
    dsg = np.where(active, sw * (kg - kw) / T_safe**2, 0.0)
    z = get_zeros_like_kro(kro_w)
    return (dkro_dkro_w, dkro_dkro_g, z, z, dsw, z, dsg)  # type: ignore[return-value]


@mixing_rule
@numba.njit(cache=True)
def blunt_rule(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
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
    return np.where((kro_w <= 0.0) | (kro_g <= 0.0), 0.0, result)  # type: ignore[return-value]


@blunt_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> tuple[
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
]:
    """
    kro = kw * kg * (2 - kw - kg)

    ∂kro/∂kw = kg * (2 - kw - kg) + kw * kg * (-1)
             = kg * (2 - 2*kw - kg)
    ∂kro/∂kg = kw * (2 - kw - 2*kg)

    Clamped region (kro ≤ 0) or either input zero - derivative = 0.
    No dependence on krw, krg, or explicit saturations.
    """
    kw = np.asarray(kro_w, dtype=np.float64)
    kg = np.asarray(kro_g, dtype=np.float64)
    active = (kw > 0.0) & (kg > 0.0) & (kw * kg * (2.0 - kw - kg) > 0.0)
    dkro_dkro_w = np.where(active, kg * (2.0 - 2.0 * kw - kg), 0.0)
    dkro_dkro_g = np.where(active, kw * (2.0 - kw - 2.0 * kg), 0.0)
    z = get_zeros_like_kro(kro_w)
    return (dkro_dkro_w, dkro_dkro_g, z, z, z, z, z)  # type: ignore[return-value]


@mixing_rule
@numba.njit(cache=True)
def hustad_hansen_rule(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
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
    return np.where((kro_w <= 0.0) & (kro_g <= 0.0), 0.0, result)  # type: ignore[return-value]


@hustad_hansen_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> tuple[
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
]:
    """
    kro = (kw * kg) / max(kw, kg)

    Case kw > kg:  kro = kw * kg / kw = kg
        ∂kro/∂kw = 0,  ∂kro/∂kg = 1

    Case kg > kw:  kro = kw * kg / kg = kw
        ∂kro/∂kw = 1,  ∂kro/∂kg = 0

    Case kw == kg (tie):  kro = kw = kg,  use subgradient 0.5 each.

    Both zero - 0.
    No dependence on krw, krg, or explicit saturations.
    """
    kw = np.asarray(kro_w, dtype=np.float64)
    kg = np.asarray(kro_g, dtype=np.float64)
    both_zero = (kw <= 0.0) & (kg <= 0.0)
    dkro_dkro_w = np.where(both_zero, 0.0, np.where(kw > kg, 0.0, np.where(kg > kw, 1.0, 0.5)))
    dkro_dkro_g = np.where(both_zero, 0.0, np.where(kg > kw, 0.0, np.where(kw > kg, 1.0, 0.5)))
    z = get_zeros_like_kro(kro_w)
    return (dkro_dkro_w, dkro_dkro_g, z, z, z, z, z)  # type: ignore[return-value]


def aziz_settari_rule(a: Number = 0.5, b: Number = 0.5) -> MixingRule:
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

    @numba.njit(cache=True)
    def _func(
        kro_w: NumberOrArray[NDimension],
        kro_g: NumberOrArray[NDimension],
        krw: NumberOrArray[NDimension],
        krg: NumberOrArray[NDimension],
        kr_max: NumberOrArray[NDimension],
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
    ) -> NumberOrArray[NDimension]:
        result = kro_w**a * kro_g**b
        return np.where((kro_w <= 0.0) | (kro_g <= 0.0), 0.0, result)  # type: ignore[return-value]

    rule: MixingRule = mixing_rule(_func, name=f"aziz_settari(a={a!r}, b={b!r})")

    @rule.dfunc
    @numba.njit(cache=True)
    def _dfunc(
        kro_w: NumberOrArray[NDimension],
        kro_g: NumberOrArray[NDimension],
        krw: NumberOrArray[NDimension],
        krg: NumberOrArray[NDimension],
        kr_max: NumberOrArray[NDimension],
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
    ) -> tuple[
        NumberOrArray[NDimension],
        NumberOrArray[NDimension],
        NumberOrArray[NDimension],
        NumberOrArray[NDimension],
        NumberOrArray[NDimension],
        NumberOrArray[NDimension],
        NumberOrArray[NDimension],
    ]:
        """
        kro = kw^a * kg^b

        ∂kro/∂kw = a * kw^(a-1) * kg^b
        ∂kro/∂kg = b * kw^a    * kg^(b-1)

        Zero when either input is zero or non-positive.
        No dependence on krw, krg, or explicit saturations.
        """
        kw = np.asarray(kro_w, dtype=np.float64)
        kg = np.asarray(kro_g, dtype=np.float64)
        active = (kw > 0.0) & (kg > 0.0)
        safe_kw = np.where(active, kw, 1.0)
        safe_kg = np.where(active, kg, 1.0)
        dkro_dkro_w = np.where(active, a * safe_kw ** (a - 1.0) * safe_kg**b, 0.0)
        dkro_dkro_g = np.where(active, b * safe_kw**a * safe_kg ** (b - 1.0), 0.0)
        z = get_zeros_like_kro(kro_w)
        return (dkro_dkro_w, dkro_dkro_g, z, z, z, z, z)  # type: ignore[return-value]

    return rule


@mixing_rule
@numba.njit(cache=True)
def eclipse_rule(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
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

    # Saturation factors
    denom_w = oil_saturation + gas_saturation
    f_w = np.where(denom_w > 0.0, oil_saturation / denom_w, 0.0)

    denom_g = oil_saturation + water_saturation
    f_g = np.where(denom_g > 0.0, oil_saturation / denom_g, 0.0)

    # Return 0 if `total_mobile` is zero, otherwise compute kro
    result = (kro_w * f_w) + (kro_g * f_g)
    return np.where(total_mobile > 0.0, result, 0.0)  # type: ignore[return-value]


@eclipse_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> tuple[
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
]:
    """
    kro = kw * So/(So+Sg) + kg * So/(So+Sw)

    Let  Dw = So + Sg,   Dg = So + Sw.

    ∂kro/∂kw  = So / Dw
    ∂kro/∂kg  = So / Dg

    ∂kro/∂Sw (explicit):
        Only Dg depends on Sw:  ∂(kg * So/Dg)/∂Sw = -kg * So / Dg²

    ∂kro/∂So (explicit):
        ∂(kw*So/Dw)/∂So = kw * Sg / Dw²
        ∂(kg*So/Dg)/∂So = kg * Sw / Dg²
        - kw * Sg / Dw² + kg * Sw / Dg²

    ∂kro/∂Sg (explicit):
        Only Dw depends on Sg:  ∂(kw*So/Dw)/∂Sg = -kw * So / Dw²

    No dependence on krw or krg.
    """
    kw = np.asarray(kro_w, dtype=np.float64)
    kg = np.asarray(kro_g, dtype=np.float64)
    sw = np.asarray(water_saturation, dtype=np.float64)
    so = np.asarray(oil_saturation, dtype=np.float64)
    sg = np.asarray(gas_saturation, dtype=np.float64)

    total_mobile = so + sw + sg
    active = total_mobile > 0.0

    Dw = so + sg
    Dg = so + sw
    Dw_safe = np.where(Dw > 0.0, Dw, 1.0)
    Dg_safe = np.where(Dg > 0.0, Dg, 1.0)

    dkro_dkro_w = np.where(active & (Dw > 0.0), so / Dw_safe, 0.0)
    dkro_dkro_g = np.where(active & (Dg > 0.0), so / Dg_safe, 0.0)

    dsw = np.where(active & (Dg > 0.0), -kg * so / Dg_safe**2, 0.0)
    dso = np.where(
        active,
        np.where(Dw > 0.0, kw * sg / Dw_safe**2, 0.0)
        + np.where(Dg > 0.0, kg * sw / Dg_safe**2, 0.0),
        0.0,
    )
    dsg = np.where(active & (Dw > 0.0), -kw * so / Dw_safe**2, 0.0)

    z = get_zeros_like_kro(kro_w)
    return (dkro_dkro_w, dkro_dkro_g, z, z, dsw, dso, dsg)  # type: ignore[return-value]


@mixing_rule
@numba.njit(cache=True)
def max_rule(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
    """
    Maximum rule - most optimistic estimate.

    kro = max(kro_w, kro_g)

    Notes:
    - Upper bound for oil relative permeability
    - Rarely used in practice (too optimistic)
    - Useful for sensitivity analysis
    """
    return np.maximum(kro_w, kro_g)  # type: ignore[return-value]


@max_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
) -> tuple[
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
    NumberOrArray[NDimension],
]:
    """
    kro = max(kw, kg)

    kw > kg - ∂/∂kw = 1, ∂/∂kg = 0
    kg > kw - ∂/∂kw = 0, ∂/∂kg = 1
    Tie      - 0.5 each (subgradient).
    No dependence on krw, krg, or explicit saturations.
    """
    kw = np.asarray(kro_w, dtype=np.float64)
    kg = np.asarray(kro_g, dtype=np.float64)
    dkro_dkro_w = np.where(kw > kg, 1.0, np.where(kg > kw, 0.0, 0.5))
    dkro_dkro_g = 1.0 - dkro_dkro_w
    z = get_zeros_like_kro(kro_w)
    return (dkro_dkro_w, dkro_dkro_g, z, z, z, z, z)  # type: ignore[return-value]


def get_mixing_rule_partial_derivatives(
    rule: MixingRule | MixingRuleFunc,
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    epsilon: Number = 1e-7,
) -> MixingRulePartialDerivatives:
    """
    Compute partial derivatives of the three-phase mixing rule with respect
    to each of its saturation-dependent arguments.

    The mixing rule signature is:

    ``python
    kro = mixing_rule(kro_w, kro_g, krw, krg, kr_max,
                      water_saturation, oil_saturation, gas_saturation)
    ``

    This function returns a dictionary containing seven partial derivatives:

    - `dkro_dkro_w`: sensitivity to the oil-water two-phase oil kr.
    - `dkro_dkro_g`: sensitivity to the gas-oil two-phase oil kr.
    - `dkro_dkrw`: sensitivity to the two-phase water kr.
    - `dkro_dkrg`: sensitivity to the two-phase gas kr.
    - `dkro_dsw_explicit`: explicit Sw dependence in the mixing rule.
    - `dkro_dso_explicit`: explicit So dependence in the mixing rule.
    - `dkro_dsg_explicit`: explicit Sg dependence in the mixing rule.

    :param rule: The mixing rule callable.
    :param kro_w: Oil relative permeability from the oil-water two-phase table.
    :param kro_g: Oil relative permeability from the gas-oil two-phase table.
    :param krw: Water relative permeability from the oil-water two-phase table.
    :param krg: Gas relative permeability from the gas-oil two-phase table.
    :param kr_max: Oil relative permeability at connate water.
    :param water_saturation: Current water saturation.
    :param oil_saturation: Current oil saturation.
    :param gas_saturation: Current gas saturation.
    :param epsilon: Finite difference step size.
    :return: A `MixingRulePartialDerivatives` dictionary of 7 partial derivatives.
    """
    if isinstance(rule, MixingRule):
        return rule.partial_derivatives(
            kro_w=kro_w,
            kro_g=kro_g,
            krw=krw,
            krg=krg,
            kr_max=kr_max,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            epsilon=epsilon,
        )
    return compute_central_difference_partial_derivatives(
        rule=rule,
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        kr_max=kr_max,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        epsilon=epsilon,
    )

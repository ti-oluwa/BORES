"""
Three-phase oil relative permeability mixing rules.

A mixing rule builds the three-phase oil relative permeability `kro` from the two
two-phase oil curves, `kro_w` (oil-water system, at the current `Sw`) and `kro_g`
(gas-oil system, at the current `Sg`), together with the water and gas relative
permeabilities and the saturations.

Every registered rule honors the two-phase limits, for any oil endpoint `kr_max`:

- no gas (`Sg = 0`): `kro = kro_w`
- connate water (`Sw = swc`): `kro = kro_g`
- both (`Sw = swc`, `Sg = 0`): `kro = kr_max`

| Rule                  | Character          | Notes                                           |
|-----------------------|--------------------|-------------------------------------------------|
| `baker_linear_rule`   | Moderate (default) | Saturation-weighted interpolation (Baker, 1988). The ECLIPSE default. |
| `stone_I_rule`        | Moderate           | Stone's first model, Aziz-Settari normalization, Fayers-Matthews `Som`. |
| `stone_II_rule`       | Moderate           | Stone's second model, normalized.               |
| `min_rule`            | Very conservative  | Lower bound: `min(kro_w, kro_g)`.               |

**Rule signature**

```python
kro = rule(
    kro_w,
    kro_g,
    krw,
    krg,
    kr_max,
    water_saturation,
    oil_saturation,
    gas_saturation,
    irreducible_water_saturation,
    residual_oil_saturation_water,
    residual_oil_saturation_gas,
)
```

`kro_w`, `kro_g` and `kr_max` must be on the same scale (both raw, or both divided by
`kr_max` with `kr_max = 1`): every rule is positively homogeneous in them,
`rule(c*kro_w, c*kro_g, krw, krg, c*kr_max, ...) = c * rule(kro_w, kro_g, krw, krg, kr_max, ...)`.
`krw` and `krg` are always the raw two-phase values.

At points where a rule is not differentiable (the `min_rule` tie, the cap at `kr_max`),
a one-sided or subgradient convention is used, documented on the rule.

A rule is phase-agnostic. For a system where oil is not the intermediate phase, a
caller applies it to the intermediate phase by passing the phases in the roles the
arguments name: the wetting phase as "water", the intermediate phase as "oil", and the
wetting phase's irreducible saturation as `irreducible_water_saturation`.
"""

import threading
import typing
from collections.abc import Mapping

import attrs
import numpy as np

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
    "MixingRule",
    "baker_linear_rule",
    "compute_central_difference_partial_derivatives",
    "get_mixing_rule",
    "get_mixing_rule_partial_derivatives",
    "list_mixing_rules",
    "min_rule",
    "mixing_rule",
    "stone_II_rule",
    "stone_I_rule",
]

_EPS = 1e-12
"""Guard for divisions by saturation ranges and weights."""


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

    Use `MixingRule.dfunc` as a decorator. The function takes the same keyword
    arguments as the rule and returns the seven partial derivatives, either as a
    `MixingRulePartialDerivatives` dict or as the tuple
    `(dkro_dkro_w, dkro_dkro_g, dkro_dkrw, dkro_dkrg, dkro_dsw, dkro_dso, dkro_dsg)`:

    ``python
    @my_rule.dfunc
    def _(kro_w, kro_g, krw, krg, kr_max,
          water_saturation, oil_saturation, gas_saturation,
          irreducible_water_saturation,
          residual_oil_saturation_water,
          residual_oil_saturation_gas):
        ...
    ``

    **Protocol-compatible objects**:

    If the wrapped callable also has a `partial_derivatives` method (i.e. it
    is a class instance that matches `MixingRuleFunc` and exposes its own
    derivative logic), that method is automatically promoted to `_dfunc`
    during post-initialization.

    Without an analytical derivative, the derivatives are estimated with central
    finite differences.
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
        this mixing rule. See the class docstring for the expected signature.
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
        irreducible_water_saturation: NumberOrArray[NDimension],
        residual_oil_saturation_water: NumberOrArray[NDimension],
        residual_oil_saturation_gas: NumberOrArray[NDimension],
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
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
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
        irreducible_water_saturation: NumberOrArray[NDimension],
        residual_oil_saturation_water: NumberOrArray[NDimension],
        residual_oil_saturation_gas: NumberOrArray[NDimension],
        epsilon: Number = 1e-7,
    ) -> MixingRulePartialDerivatives:
        """
        Return the seven partial derivatives of this mixing rule.

        If an analytical derivative function has been registered (via
        `@rule.dfunc` or auto-detected from `func.partial_derivatives`),
        it is called directly. Otherwise the derivatives are estimated with
        central finite differences.

        The derivatives are with respect to the rule's own arguments, treating the
        three saturations as independent. The chain-rule terms from `kro_w`, `kro_g`,
        `krw` and `krg` back to the physical saturations are assembled by the caller.

        :param kro_w: Two-phase oil relative permeability, oil-water system.
        :param kro_g: Two-phase oil relative permeability, gas-oil system.
        :param krw: Two-phase water relative permeability.
        :param krg: Two-phase gas relative permeability.
        :param kr_max: Oil relative permeability at connate water, on the same scale
            as `kro_w` and `kro_g`.
        :param water_saturation: Current water saturation.
        :param oil_saturation: Current oil saturation.
        :param gas_saturation: Current gas saturation.
        :param irreducible_water_saturation: Connate/irreducible water saturation.
        :param residual_oil_saturation_water: Residual oil saturation to water flood.
        :param residual_oil_saturation_gas: Residual oil saturation to gas flood.
        :param epsilon: Step size for central differences (fallback path only).
        :return: A `MixingRulePartialDerivatives` dictionary.
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
                irreducible_water_saturation=irreducible_water_saturation,
                residual_oil_saturation_water=residual_oil_saturation_water,
                residual_oil_saturation_gas=residual_oil_saturation_gas,
            )
            if isinstance(derivatives, Mapping):
                return derivatives  # type: ignore[return-value]

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
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
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
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    epsilon: Number = 1e-7,
) -> MixingRulePartialDerivatives:
    """
    Compute the seven partial derivatives of a mixing rule by central finite
    differences. Used when a rule has no analytical derivative.

    The step is `epsilon` for saturations and `epsilon * max(1, |x|)` for the relative
    permeabilities. The endpoints and `kr_max` are held fixed.

    :param rule: The mixing rule callable.
    :param kro_w: Two-phase oil relative permeability, oil-water system.
    :param kro_g: Two-phase oil relative permeability, gas-oil system.
    :param krw: Two-phase water relative permeability.
    :param krg: Two-phase gas relative permeability.
    :param kr_max: Oil relative permeability at connate water.
    :param water_saturation: Water saturation.
    :param oil_saturation: Oil saturation.
    :param gas_saturation: Gas saturation.
    :param irreducible_water_saturation: Connate/irreducible water saturation.
    :param residual_oil_saturation_water: Residual oil saturation to water flood.
    :param residual_oil_saturation_gas: Residual oil saturation to gas flood.
    :param epsilon: Base step size.
    :return: A `MixingRulePartialDerivatives` dictionary.
    """
    base = {
        "kro_w": kro_w,
        "kro_g": kro_g,
        "krw": krw,
        "krg": krg,
        "kr_max": kr_max,
        "water_saturation": water_saturation,
        "oil_saturation": oil_saturation,
        "gas_saturation": gas_saturation,
        "irreducible_water_saturation": irreducible_water_saturation,
        "residual_oil_saturation_water": residual_oil_saturation_water,
        "residual_oil_saturation_gas": residual_oil_saturation_gas,
    }
    is_saturation = {"water_saturation", "oil_saturation", "gas_saturation"}

    def partial(name: str) -> NumberOrArray:
        value = np.asarray(base[name], dtype=np.float64)
        step = epsilon if name in is_saturation else epsilon * np.maximum(1.0, np.abs(value))
        plus = dict(base, **{name: value + step})
        minus = dict(base, **{name: value - step})
        return (rule(**plus) - rule(**minus)) / (2.0 * step)  # type: ignore[call-arg]

    return MixingRulePartialDerivatives(
        dkro_dkro_w=partial("kro_w"),
        dkro_dkro_g=partial("kro_g"),
        dkro_dkrw=partial("krw"),
        dkro_dkrg=partial("krg"),
        dkro_dsw_explicit=partial("water_saturation"),
        dkro_dso_explicit=partial("oil_saturation"),
        dkro_dsg_explicit=partial("gas_saturation"),
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
    def my_rule(
        kro_w,
        kro_g,
        krw,
        krg,
        kr_max,
        water_saturation,
        oil_saturation,
        gas_saturation,
        irreducible_water_saturation,
        residual_oil_saturation_water,
        residual_oil_saturation_gas,
    ):
        return np.minimum(kro_w, kro_g)
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

    A custom rule should honor the two-phase limits described in the module docstring.

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
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
    epsilon: Number = 1e-7,
) -> MixingRulePartialDerivatives:
    """
    Compute the partial derivatives of a mixing rule with respect to each of its
    seven varying arguments, using the analytical derivative when the rule has
    one and central differences otherwise.

    The returned dictionary holds:

    - `dkro_dkro_w`, `dkro_dkro_g`: sensitivity to the two-phase oil kr values.
    - `dkro_dkrw`, `dkro_dkrg`: sensitivity to the water and gas kr values.
    - `dkro_dsw_explicit`, `dkro_dso_explicit`, `dkro_dsg_explicit`: the explicit
      dependence on each saturation argument.

    :param rule: The mixing rule callable.
    :param kro_w: Two-phase oil relative permeability, oil-water system.
    :param kro_g: Two-phase oil relative permeability, gas-oil system.
    :param krw: Two-phase water relative permeability.
    :param krg: Two-phase gas relative permeability.
    :param kr_max: Oil relative permeability at connate water.
    :param water_saturation: Water saturation.
    :param oil_saturation: Oil saturation.
    :param gas_saturation: Gas saturation.
    :param irreducible_water_saturation: Connate/irreducible water saturation.
    :param residual_oil_saturation_water: Residual oil saturation to water flood.
    :param residual_oil_saturation_gas: Residual oil saturation to gas flood.
    :param epsilon: Finite difference step size (fallback path only).
    :return: A `MixingRulePartialDerivatives` dictionary.
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
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation_water=residual_oil_saturation_water,
            residual_oil_saturation_gas=residual_oil_saturation_gas,
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
        irreducible_water_saturation=irreducible_water_saturation,
        residual_oil_saturation_water=residual_oil_saturation_water,
        residual_oil_saturation_gas=residual_oil_saturation_gas,
        epsilon=epsilon,
    )


def _broadcast(*values: NumberOrArray) -> tuple[np.ndarray, ...]:
    """Convert to float64 and broadcast to a common shape."""
    return tuple(np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in values)))


def _output(result: np.ndarray, like: NumberOrArray) -> NumberOrArray:
    """Cast a rule result to the floating dtype of `like`, unwrapping 0-d arrays."""
    dtype = (
        like.dtype if hasattr(like, "dtype") and np.issubdtype(like.dtype, np.floating) else None
    )  # type: ignore[union-attr]
    result = result.astype(dtype, copy=False) if dtype is not None else result
    return result[()] if result.ndim == 0 else result


def _safe(denominator: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Denominator with a harmless value substituted where `valid` is False."""
    return np.where(valid, denominator, 1.0)


# --------------------------------------------------------------------------------------
# Minimum rule
# --------------------------------------------------------------------------------------
@mixing_rule
def min_rule(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
    """
    Minimum rule: `kro = min(kro_w, kro_g)`.

    A conservative lower bound on the three-phase oil relative permeability. It
    honors the two-phase limits because `kro_g = kr_max >= kro_w` with no gas and
    `kro_w = kr_max >= kro_g` at connate water.
    """
    return _output(np.minimum(*_broadcast(kro_w, kro_g)), kro_w)


@min_rule.dfunc
def _min_rule_derivatives(
    kro_w,
    kro_g,
    krw,
    krg,
    kr_max,
    water_saturation,
    oil_saturation,
    gas_saturation,
    irreducible_water_saturation,
    residual_oil_saturation_water,
    residual_oil_saturation_gas,
):
    """
    Derivatives of the minimum rule. At a tie (`kro_w == kro_g`) the subgradient
    convention is used: both derivatives are 0.5.
    """
    kw, kg = _broadcast(kro_w, kro_g)
    dkro_dkro_w = np.where(kw < kg, 1.0, np.where(kw == kg, 0.5, 0.0))
    dkro_dkro_g = np.where(kg < kw, 1.0, np.where(kw == kg, 0.5, 0.0))
    zeros = np.zeros_like(kw)
    return (
        dkro_dkro_w,
        dkro_dkro_g,
        zeros,
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
    )


# --------------------------------------------------------------------------------------
# Baker's saturation-weighted rule (the ECLIPSE default)
# --------------------------------------------------------------------------------------
@mixing_rule
def baker_linear_rule(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
    """
    Baker's saturation-weighted interpolation (Baker, 1988), the ECLIPSE default:

        kro = (Sg * kro_g + (Sw - swc) * kro_w) / (Sg + (Sw - swc))

    with `Sw - swc` floored at zero. Honors the two-phase limits exactly. At the
    corner `Sw = swc, Sg = 0` the weights vanish and the rule returns the mean of
    `kro_w` and `kro_g`, which both equal `kr_max` there.
    """
    kw, kg, sw, sg, swc = _broadcast(
        kro_w, kro_g, water_saturation, gas_saturation, irreducible_water_saturation
    )
    sg = np.maximum(sg, 0.0)
    mobile_water = np.maximum(sw - swc, 0.0)
    total = sg + mobile_water
    degenerate = total <= _EPS
    weighted = (sg * kg + mobile_water * kw) / _safe(total, ~degenerate)
    return _output(np.where(degenerate, 0.5 * (kw + kg), weighted), kro_w)


@baker_linear_rule.dfunc
def _baker_linear_rule_derivatives(
    kro_w,
    kro_g,
    krw,
    krg,
    kr_max,
    water_saturation,
    oil_saturation,
    gas_saturation,
    irreducible_water_saturation,
    residual_oil_saturation_water,
    residual_oil_saturation_gas,
):
    """
    Derivatives of Baker's rule. With `W = Sw - swc` (floored at 0) and `T = Sg + W`:

        d/dkro_w = W / T          d/dkro_g = Sg / T
        d/dSw    = Sg (kro_w - kro_g) / T²     (zero when Sw <= swc)
        d/dSg    = W (kro_g - kro_w) / T²
        d/dSo    = 0

    At the degenerate corner (`T ~ 0`) the value is the mean of the two curves, so
    the kr derivatives are 0.5 and the saturation derivatives are 0.
    """
    kw, kg, sw, sg, swc = _broadcast(
        kro_w, kro_g, water_saturation, gas_saturation, irreducible_water_saturation
    )
    sg = np.maximum(sg, 0.0)
    mobile_water = np.maximum(sw - swc, 0.0)
    total = sg + mobile_water
    degenerate = total <= _EPS
    t = _safe(total, ~degenerate)
    zeros = np.zeros_like(kw)
    dkro_dkro_w = np.where(degenerate, 0.5, mobile_water / t)
    dkro_dkro_g = np.where(degenerate, 0.5, sg / t)
    dkro_dsw = np.where(degenerate | (sw <= swc), 0.0, sg * (kw - kg) / t**2)
    dkro_dsg = np.where(degenerate, 0.0, mobile_water * (kg - kw) / t**2)
    return (dkro_dkro_w, dkro_dkro_g, zeros, zeros.copy(), dkro_dsw, zeros.copy(), dkro_dsg)


# --------------------------------------------------------------------------------------
# Stone II
# --------------------------------------------------------------------------------------
@mixing_rule
def stone_II_rule(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
    """
    Stone's second model (Stone, 1973), normalized by the oil endpoint `krocw = kr_max`:

        kro = krocw * [(kro_w / krocw + krw) * (kro_g / krocw + krg) - (krw + krg)]

    The result is bounded to `[0, kr_max]`. Stone II can go negative at low oil
    saturation and can slightly exceed the oil endpoint, and neither is physical.
    """
    kw, kg, rw, rg, km = _broadcast(kro_w, kro_g, krw, krg, kr_max)
    km = np.maximum(km, _EPS)
    raw = km * ((kw / km + rw) * (kg / km + rg) - rw - rg)
    return _output(np.clip(raw, 0.0, km), kro_w)


@stone_II_rule.dfunc
def _stone_II_rule_derivatives(
    kro_w,
    kro_g,
    krw,
    krg,
    kr_max,
    water_saturation,
    oil_saturation,
    gas_saturation,
    irreducible_water_saturation,
    residual_oil_saturation_water,
    residual_oil_saturation_gas,
):
    """
    Derivatives of Stone II. With `A = kro_w/krocw + krw` and `B = kro_g/krocw + krg`:

        d/dkro_w = B     d/dkro_g = A
        d/dkrw   = krocw (B - 1)     d/dkrg = krocw (A - 1)

    All are zero where the result is bounded (at 0 or at `krocw`). There is no explicit
    saturation dependence.
    """
    kw, kg, rw, rg, km = _broadcast(kro_w, kro_g, krw, krg, kr_max)
    km = np.maximum(km, _EPS)
    a = kw / km + rw
    b = kg / km + rg
    raw = km * (a * b - rw - rg)
    active = (raw > 0.0) & (raw < km)
    zeros = np.zeros_like(kw)
    return (
        np.where(active, b, 0.0),
        np.where(active, a, 0.0),
        np.where(active, km * (b - 1.0), 0.0),
        np.where(active, km * (a - 1.0), 0.0),
        zeros,
        zeros.copy(),
        zeros.copy(),
    )


# --------------------------------------------------------------------------------------
# Stone I
# --------------------------------------------------------------------------------------
def _stone_I_state(kw, kg, km, sw, so, sg, swc, sorw, sorg):
    """
    Shared intermediate quantities of Stone I (Aziz-Settari normalization, with the
    Fayers-Matthews minimum oil saturation).

        alpha = 1 - Sg / (1 - swc - sorg)              clipped to [0, 1]
        Som   = alpha * sorw + (1 - alpha) * sorg
        D     = 1 - swc - Som
        So*   = (So - Som) / D,   Sw* = (Sw - swc) / D,   Sg* = Sg / D
        Y     = 1 - Sw*,          Z   = 1 - Sg*
        kro   = (kro_w * kro_g / krocw) * So* / (Y * Z)

    where `Sw - swc` is floored at zero.
    """
    gas_range = np.maximum(1.0 - swc - sorg, _EPS)
    alpha_raw = 1.0 - sg / gas_range
    alpha = np.clip(alpha_raw, 0.0, 1.0)
    alpha_active = (alpha_raw > 0.0) & (alpha_raw < 1.0)
    som = sorg + alpha * (sorw - sorg)
    d = 1.0 - swc - som
    x = so - som
    mobile_water = np.maximum(sw - swc, 0.0)
    valid = (x > 0.0) & (d > _EPS)
    d_safe = _safe(d, valid)
    so_star = x / d_safe
    y = 1.0 - mobile_water / d_safe
    z = 1.0 - sg / d_safe
    usable = valid & (y > _EPS) & (z > _EPS)
    y_safe = _safe(y, usable)
    z_safe = _safe(z, usable)
    c = kw * kg / km
    raw = np.where(usable, c * so_star / (y_safe * z_safe), 0.0)
    return {
        "gas_range": gas_range,
        "alpha_active": alpha_active,
        "d": d_safe,
        "so_star": so_star,
        "mobile_water": mobile_water,
        "y": y_safe,
        "z": z_safe,
        "usable": usable,
        "c": c,
        "raw": raw,
    }


@mixing_rule
def stone_I_rule(
    kro_w: NumberOrArray[NDimension],
    kro_g: NumberOrArray[NDimension],
    krw: NumberOrArray[NDimension],
    krg: NumberOrArray[NDimension],
    kr_max: NumberOrArray[NDimension],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    irreducible_water_saturation: NumberOrArray[NDimension],
    residual_oil_saturation_water: NumberOrArray[NDimension],
    residual_oil_saturation_gas: NumberOrArray[NDimension],
) -> NumberOrArray[NDimension]:
    """
    Stone's first model (Stone, 1970) in the Aziz-Settari normalization, with the
    minimum oil saturation of Fayers and Matthews (1984):

        alpha = 1 - Sg / (1 - swc - sorg)
        Som   = alpha * sorw + (1 - alpha) * sorg
        So*   = (So - Som) / (1 - swc - Som)
        Sw*   = (Sw - swc) / (1 - swc - Som)
        Sg*   = Sg / (1 - swc - Som)
        kro   = krocw * So* * beta_w * beta_g
        beta_w = (kro_w / krocw) / (1 - Sw*)
        beta_g = (kro_g / krocw) / (1 - Sg*)

    with `krocw = kr_max`. Honors the two-phase limits exactly. The result is zero at
    or below `Som` and is bounded to `[0, kr_max]`.
    """
    kw, kg, sw, so, sg, swc, sorw, sorg, km = _broadcast(
        kro_w,
        kro_g,
        water_saturation,
        oil_saturation,
        gas_saturation,
        irreducible_water_saturation,
        residual_oil_saturation_water,
        residual_oil_saturation_gas,
        kr_max,
    )
    km = np.maximum(km, _EPS)
    state = _stone_I_state(kw, kg, km, sw, so, sg, swc, sorw, sorg)
    return _output(np.clip(state["raw"], 0.0, km), kro_w)


@stone_I_rule.dfunc
def _stone_I_rule_derivatives(
    kro_w,
    kro_g,
    krw,
    krg,
    kr_max,
    water_saturation,
    oil_saturation,
    gas_saturation,
    irreducible_water_saturation,
    residual_oil_saturation_water,
    residual_oil_saturation_gas,
):
    """
    Exact gradient of `stone_I_rule` with respect to its arguments, treating the three
    saturations as independent. With `C = kro_w * kro_g / krocw` and `f = C * So* / (Y Z)`:

        d/dkro_w = (kro_g / krocw) So* / (Y Z)
        d/dkro_g = (kro_w / krocw) So* / (Y Z)
        d/dSw    = C So* / (Y² Z D)          (zero when Sw <= swc)
        d/dSo    = C / (D Y Z)
        d/dSg    = C [So*' / (Y Z) - So* Y' / (Y² Z) - So* Z' / (Y Z²)]

    where `Dg = dD/dSg = (sorw - sorg) / (1 - swc - sorg)` while `0 < alpha < 1`
    (else 0), and

        So*' = Dg (1 - swc - So) / D²
        Y'   = (Sw - swc)+ Dg / D²
        Z'   = -1 / D + Sg Dg / D²

    All are zero where the result is zero or bounded at `kr_max`. There is no
    dependence on `krw` or `krg`.
    """
    kw, kg, sw, so, sg, swc, sorw, sorg, km = _broadcast(
        kro_w,
        kro_g,
        water_saturation,
        oil_saturation,
        gas_saturation,
        irreducible_water_saturation,
        residual_oil_saturation_water,
        residual_oil_saturation_gas,
        kr_max,
    )
    km = np.maximum(km, _EPS)
    s = _stone_I_state(kw, kg, km, sw, so, sg, swc, sorw, sorg)
    d, so_star, y, z, c = s["d"], s["so_star"], s["y"], s["z"], s["c"]
    active = s["usable"] & (s["raw"] > 0.0) & (s["raw"] < km)

    d_gas = np.where(s["alpha_active"], (sorw - sorg) / s["gas_range"], 0.0)
    so_star_prime = d_gas * (1.0 - swc - so) / d**2
    y_prime = s["mobile_water"] * d_gas / d**2
    z_prime = -1.0 / d + sg * d_gas / d**2

    base = so_star / (y * z)
    dkro_dkro_w = np.where(active, kg / km * base, 0.0)
    dkro_dkro_g = np.where(active, kw / km * base, 0.0)
    dkro_dsw = np.where(active & (sw > swc), c * so_star / (y**2 * z * d), 0.0)
    dkro_dso = np.where(active, c / (d * y * z), 0.0)
    dkro_dsg = np.where(
        active,
        c
        * (
            so_star_prime / (y * z)
            - so_star * y_prime / (y**2 * z)
            - so_star * z_prime / (y * z**2)
        ),
        0.0,
    )
    zeros = np.zeros_like(kw)
    return (
        dkro_dkro_w,
        dkro_dkro_g,
        zeros,
        zeros.copy(),
        dkro_dsw,
        dkro_dso,
        dkro_dsg,
    )

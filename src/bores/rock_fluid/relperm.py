"""Relative permeability models and mixing rules for multi-phase flow simulations."""

import threading
import typing
from collections.abc import Mapping

import attrs
import numba
import numpy as np
import numpy.typing as npt
from numba.extending import overload
from scipy.interpolate import PchipInterpolator

from bores.constants import c
from bores.errors import ValidationError
from bores.grids.base import array as bores_array
from bores.precision import get_floating_point_info
from bores.serialization import Serializable, make_serializable_type_registrar
from bores.stores import StoreSerializable
from bores.tables.utils import build_pchip_interpolant
from bores.types import (
    FloatOrArray,
    FluidPhase,
    MixingRuleDFunc,
    MixingRuleFunc,
    MixingRulePartialDerivatives,
    RelativePermeabilities,
    RelativePermeabilityDerivatives,
    Spacing,
    T,
    Wettability,
)

__all__ = [
    "BrooksCoreyRelPermModel",
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
    "list_relperm_tables",
    "max_rule",
    "max_rule",
    "mixing_rule",
    "relperm_table",
    "stone_II_rule",
    "stone_I_rule",
]


#: Sentinel type for the `min_*_relperm` attributes on relperm tables and
#: analytical models. `"auto"` derives the floor from the active dtype;
#: `None` disables the floor entirely; a `float` sets an explicit floor.
RelPermFloor = typing.Union[typing.Literal["auto"], None, float]


def _resolve_relperm_floor(floor: RelPermFloor) -> typing.Optional[float]:
    """
    Resolve a `RelPermFloor` sentinel to a concrete float or `None`.

    `"auto"` - `max(4 * machine_epsilon, 1e-8)`, consistent with the
    approach used by CMG IMEX/GEM for minimum mobility floors: tight enough
    that it does not distort the physics, but sufficient to prevent
    near-singular transmissibility rows in the Jacobian.

    `None` - `None` (no floor; kr can reach zero exactly).

    `float` - that value, used as-is after a sign / sanity check.

    :param floor: Sentinel value from a relperm table or analytical model.
    :return: Resolved float floor or `None`.
    """
    if floor is None:
        return None

    if floor == "auto":
        info = get_floating_point_info()
        return float(max(4.0 * float(info.eps), 1e-8))

    if not isinstance(floor, (int, float)):
        raise ValidationError(
            f"`min_*_relperm` must be 'auto', None, or a float. Got {floor!r}."
        )

    if floor < 0.0:
        raise ValidationError(
            f"`min_*_relperm` floor must be non-negative. Got {floor}."
        )
    return float(floor)


@numba.njit(cache=True, inline="always")
def _apply_relperm_floor(
    kr: FloatOrArray,
    floor: typing.Optional[float],
) -> FloatOrArray:
    """
    Clamp `kr` to `[floor, ∞)` in-place-compatible fashion.

    Smoothly transitions from the raw `kr` values to the floored value, avoiding
    non-differentiability and ensuring well-behaved derivatives for solvers.

    When `floor` is `None` the array is returned unchanged.

    :param kr: Relative permeability value(s).
    :param floor: Resolved floor value or `None`.
    :return: Floored relative permeability value(s).
    """
    if floor is None:
        return kr
    return np.sqrt(kr**2 + floor**2).astype(kr.dtype)


@numba.njit(cache=True, inline="always")
def _apply_relperm_floor_to_derivative(
    dkr: FloatOrArray,
    kr_raw: FloatOrArray,
    floor: typing.Optional[float],
) -> FloatOrArray:
    """
    Smoothly clamp the derivative of `kr` to zero in the floored region.
    When `kr` is above the floor, the derivative is unchanged. As `kr` approaches
    and falls below the floor, the derivative transitions smoothly to zero, ensuring
    that the Jacobian remains well-conditioned and solvers remain stable.

    When `floor` is `None` the derivative is returned unchanged.

    The derivative of `max(kr, floor)` w.r.t. any variable is:

    - `dkr / d(var)`  when `kr > floor`  (active region)
    - `kr / np.sqrt(kr**2 + floor**2)`  when `kr <= floor` (floored / flat region)

    :param dkr: Derivative of the raw (pre-floor) kr w.r.t. some variable.
    :param kr_raw: Raw (pre-floor) kr value(s), used to identify floored cells.
    :param floor: Resolved floor value or `None`.
    :return: Derivative with floored cells zeroed out.
    """
    if floor is None:
        return dkr
    return np.where(kr_raw > floor, dkr, kr_raw / np.sqrt(kr_raw**2 + floor**2)).astype(
        dkr.dtype
    )


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
| Blunt               | Conservative         | Moderate    | Strong water-wet                            |
| Eclipse             | Moderate             | Moderate    | Commercial simulator standard               |
| Aziz-Settari        | Variable             | Moderate    | Empirical tuning                            |

Note: At points where derivatives are non-unique (e.g., min rule at kro_w == kro_g),
the **subgradient convention** is used: both derivatives set to 0.5.
This ensures the expected behavior in solvers (Newton's method is robust to
subgradient choices as long as they're in the subdifferential).
"""


@typing.final
@attrs.define
class MixingRule:
    """
    Wraps a mixing-rule callable together with an optional analytical
    partial-derivative function.

    **Construction**:

    Normally produced by the `@mixing_rule` decorator, which registers the
    rule and returns a `MixingRule` instance.  You can also build one
    directly:

    ```python
    my_rule = MixingRule(func=my_func)
    ```

    **Attaching an analytical derivative later**:

    Use `MixingRule.dfunc` as a decorator:

    ```python
    @my_rule.dfunc
    def _(kro_w, kro_g, krw, krg, kro_endpoint,
          water_saturation, oil_saturation, gas_saturation):
        ...
        return MixingRulePartialDerivatives(...)
    ```

    **Protocol-compatible objects**:

    If the wrapped callable also has a `partial_derivatives` method (i.e. it
    is a class instance that matches `MixingRuleFunc` and exposes its own
    derivative logic), that method is automatically promoted to `_dfunc`
    during post-initialization.
    """

    func: typing.Union[MixingRuleFunc, "MixingRule"]
    _dfunc: typing.Optional[MixingRuleDFunc] = attrs.field(default=None, alias="dfunc")

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

        ```python
        @stone_I_rule.dfunc
        def _(kro_w, kro_g, krw, krg, kro_endpoint,
              water_saturation, oil_saturation, gas_saturation):
            ...
            return MixingRulePartialDerivatives(...)
        ```
        """
        self._dfunc = fn
        return fn

    def __call__(
        self,
        kro_w: FloatOrArray,
        kro_g: FloatOrArray,
        krw: FloatOrArray,
        krg: FloatOrArray,
        kro_endpoint: FloatOrArray,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
    ) -> FloatOrArray:
        return self.func(
            kro_w=kro_w,
            kro_g=kro_g,
            krw=krw,
            krg=krg,
            kro_endpoint=kro_endpoint,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
        )

    def partial_derivatives(
        self,
        kro_w: FloatOrArray,
        kro_g: FloatOrArray,
        krw: FloatOrArray,
        krg: FloatOrArray,
        kro_endpoint: FloatOrArray,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        epsilon: float = 1e-7,
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
        :param kro_endpoint: Oil relative permeability at connate water
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
                kro_endpoint=kro_endpoint,
                water_saturation=water_saturation,
                oil_saturation=oil_saturation,
                gas_saturation=gas_saturation,
            )
            if isinstance(derivatives, Mapping):
                return derivatives  # type: ignore[return-value]
            # Unpack 7-tuple: (d_kro_w, d_kro_g, d_krw, d_krg, d_sw, d_so, d_sg)
            return MixingRulePartialDerivatives(
                d_kro_d_kro_w=derivatives[0],
                d_kro_d_kro_g=derivatives[1],
                d_kro_d_krw=derivatives[2],
                d_kro_d_krg=derivatives[3],
                d_kro_d_sw_explicit=derivatives[4],
                d_kro_d_so_explicit=derivatives[5],
                d_kro_d_sg_explicit=derivatives[6],
            )
        return _central_difference_partial_derivatives(
            rule=self,
            kro_w=kro_w,
            kro_g=kro_g,
            krw=krw,
            krg=krg,
            kro_endpoint=kro_endpoint,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            epsilon=epsilon,
        )

    def __str__(self) -> str:
        return self.func.__name__ if hasattr(self.func, "__name__") else repr(self.func)  # type: ignore[union-attr]

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


def _central_difference_partial_derivatives(
    rule: MixingRuleFunc,
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
    epsilon: float = 1e-7,
) -> MixingRulePartialDerivatives:
    """
    Compute all seven partial derivatives of a mixing rule via central finite
    differences.

    This function approximates the partial derivatives numerically when an
    analytical derivative function is not available. It computes central
    differences for `kro_w`, `kro_g`, `krw`, `krg`, `Sw`, `So`,
    and `Sg` — fourteen mixing-rule evaluations in total.

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
    :param kro_endpoint: Oil relative permeability at connate water.
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
        kro_endpoint=kro_endpoint,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    f_minus = rule(
        kro_w=kro_w - h_kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        kro_endpoint=kro_endpoint,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    d_kro_d_kro_w = (f_plus - f_minus) / two_h_kro_w

    # Compute ∂f/∂(kro_g) using relative perturbation
    f_plus = rule(
        kro_w=kro_w,
        kro_g=kro_g + h_kro_g,
        krw=krw,
        krg=krg,
        kro_endpoint=kro_endpoint,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    f_minus = rule(
        kro_w=kro_w,
        kro_g=kro_g - h_kro_g,
        krw=krw,
        krg=krg,
        kro_endpoint=kro_endpoint,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    d_kro_d_kro_g = (f_plus - f_minus) / two_h_kro_g

    # Compute ∂f/∂(krw) using relative perturbation
    f_plus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw + h_krw,
        krg=krg,
        kro_endpoint=kro_endpoint,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    f_minus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw - h_krw,
        krg=krg,
        kro_endpoint=kro_endpoint,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    d_kro_d_krw = (f_plus - f_minus) / two_h_krw

    # Compute ∂f/∂(krg) using relative perturbation
    f_plus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg + h_krg,
        kro_endpoint=kro_endpoint,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    f_minus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg - h_krg,
        kro_endpoint=kro_endpoint,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    d_kro_d_krg = (f_plus - f_minus) / two_h_krg

    # Compute ∂f/∂(Sw) using relative perturbation
    f_plus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        kro_endpoint=kro_endpoint,
        water_saturation=water_saturation + h_sw,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    f_minus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        kro_endpoint=kro_endpoint,
        water_saturation=water_saturation - h_sw,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
    )
    d_kro_d_sw_explicit = (f_plus - f_minus) / two_h_sw

    # Compute ∂f/∂(So) using relative perturbation
    f_plus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        kro_endpoint=kro_endpoint,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation + h_so,
        gas_saturation=gas_saturation,
    )
    f_minus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        kro_endpoint=kro_endpoint,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation - h_so,
        gas_saturation=gas_saturation,
    )
    d_kro_d_so_explicit = (f_plus - f_minus) / two_h_so

    # Compute ∂f/∂(Sg) using relative perturbation
    f_plus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        kro_endpoint=kro_endpoint,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation + h_sg,
    )
    f_minus = rule(
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        kro_endpoint=kro_endpoint,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation - h_sg,
    )
    d_kro_d_sg_explicit = (f_plus - f_minus) / two_h_sg

    return MixingRulePartialDerivatives(
        d_kro_d_kro_w=d_kro_d_kro_w,
        d_kro_d_kro_g=d_kro_d_kro_g,
        d_kro_d_krw=d_kro_d_krw,
        d_kro_d_krg=d_kro_d_krg,
        d_kro_d_sw_explicit=d_kro_d_sw_explicit,
        d_kro_d_so_explicit=d_kro_d_so_explicit,
        d_kro_d_sg_explicit=d_kro_d_sg_explicit,
    )


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
def mixing_rule(func: typing.Union[MixingRuleFunc, MixingRule]) -> MixingRule: ...


@typing.overload
def mixing_rule(
    func: None = None,
    name: typing.Optional[str] = None,
    override: bool = False,
    serializer: typing.Optional[typing.Callable[[MixingRule, bool], T]] = None,
    deserializer: typing.Optional[typing.Callable[[T], MixingRule]] = None,
) -> typing.Callable[[typing.Union[MixingRuleFunc, MixingRule]], MixingRule]: ...


@typing.overload
def mixing_rule(
    func: typing.Union[MixingRuleFunc, MixingRule],
    name: typing.Optional[str] = None,
    override: bool = False,
    serializer: typing.Optional[typing.Callable[[MixingRule, bool], T]] = None,
    deserializer: typing.Optional[typing.Callable[[T], MixingRule]] = None,
) -> MixingRule: ...


def mixing_rule(
    func: typing.Optional[typing.Union[MixingRuleFunc, MixingRule]] = None,
    name: typing.Optional[str] = None,
    override: bool = False,
    serializer: typing.Optional[typing.Callable[[MixingRule, bool], T]] = None,
    deserializer: typing.Optional[typing.Callable[[T], MixingRule]] = None,
) -> typing.Union[
    MixingRule, typing.Callable[[typing.Union[MixingRuleFunc, MixingRule]], MixingRule]
]:
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
    def my_rule(kro_w, kro_g, krw, krg, kro_endpoint,
                water_saturation, oil_saturation, gas_saturation):
        return (kro_w + kro_g) / 2.0
    ```

    **Protocol-compatible class**:

    ```python
    class MyRule:
        def __call__(self, *, kro_w, kro_g, krw, krg, kro_endpoint, ...): ...
        def partial_derivatives(self, *, kro_w, kro_g, krw, krg, kro_endpoint, ...): ...

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
    :param serializer: Optional serializer for parameterised rules (e.g. `aziz_settari_rule`).
    :param deserializer: Optional deserializer for parameterised rules.
    :return: The registered `MixingRule` instance.
    """

    def _register(
        f: typing.Union[MixingRuleFunc, MixingRule],
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
            if rule_name in _MIXING_RULES and not override:
                raise ValidationError(
                    f"Mixing rule '{rule_name}' is already registered. "
                    "Use `override=True` or provide a different name."
                )
            _MIXING_RULES[rule_name] = rule
            if serializer is not None:
                _MIXING_RULE_SERIALIZERS[rule] = serializer
            if deserializer is not None:
                _MIXING_RULE_DESERIALIZERS[rule_name] = deserializer

        return rule

    if func is None:
        return _register
    return _register(func)


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


def _zeros_like_kro(kro_w: FloatOrArray) -> FloatOrArray:
    """Return an array (or scalar) of zeros with the same shape as kro_w."""
    return np.zeros_like(kro_w) if not np.isscalar(kro_w) else kro_w.dtype.type(0.0)  # type: ignore


@overload(_zeros_like_kro)
def _overload_zeros_like_kro(kro_w):
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
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Conservative rule for 3-phase oil relative permeability.

    kro = min(kro_w, kro_g)
    """
    return np.minimum(kro_w, kro_g)


@min_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> typing.Tuple[
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
]:
    """
    Analytical derivatives for min_rule.

    kro = min(kro_w, kro_g)

    Where kro_w < kro_g  -  ∂kro/∂kro_w = 1, ∂kro/∂kro_g = 0
    Where kro_g ≤ kro_w  -  ∂kro/∂kro_w = 0, ∂kro/∂kro_g = 1
    Tie: split evenly (0.5 each) — subgradient convention.
    No dependence on krw, krg, or explicit saturations.
    """
    kw = np.asarray(kro_w, dtype=np.float64)
    kg = np.asarray(kro_g, dtype=np.float64)
    d_kro_d_kro_w = np.where(kw < kg, 1.0, np.where(kw > kg, 0.0, 0.5))
    d_kro_d_kro_g = 1.0 - d_kro_d_kro_w
    z = _zeros_like_kro(kro_w)
    return (d_kro_d_kro_w, d_kro_d_kro_g, z, z, z, z, z)


@mixing_rule
@numba.njit(cache=True)
def stone_I_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
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


@stone_I_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> typing.Tuple[
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
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
    d_kro_d_kro_w = np.where(both_zero, 0.0, kg**2 / D**2)
    d_kro_d_kro_g = np.where(both_zero, 0.0, kw**2 / D**2)
    z = _zeros_like_kro(kro_w)
    return (d_kro_d_kro_w, d_kro_d_kro_g, z, z, z, z, z)


@mixing_rule
@numba.njit(cache=True)
def stone_II_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Stone II rule (Stone, 1973, JPT) for 3-phase oil relative permeability.

    Full normalized formula (Aziz & Settari, 1979):

        kro = krocw * [(krow/krocw + krw) * (krog/krocw + krg) - krw - krg]

    where:
        - krow  = kro_w  (oil relperm from oil-water table at current Sw)
        - krog  = kro_g  (oil relperm from gas-oil table at current Sg)
        - krw            = two-phase water relperm at current Sw
        - krg            = two-phase gas relperm at current Sg
        - krocw = kro_endpoint  (oil relperm at connate water, normalization ref)

    This is the **exact** Stone II formulation, valid for both normalized
    (krocw = 1) and non-normalized tables.  Using the actual krw and krg
    from the two-phase tables removes the approximation ``krw ≈ 1 - kro_w``
    that was previously needed when only kro_w and kro_g were available.

    **Reference:**
    Stone, H.L. (1973). "Estimation of Three-Phase Relative Permeability and
    Residual Oil Data." Journal of Canadian Petroleum Technology, 12(4), 53-61.
    Aziz, K. & Settari, A. (1979). Petroleum Reservoir Simulation. Applied
    Science Publishers, London.
    """
    krocw = np.maximum(kro_endpoint, 1e-12)
    result = krocw * ((kro_w / krocw + krw) * (kro_g / krocw + krg) - krw - krg)
    # Clamp negative values (can occur when both kro_w and kro_g are small)
    result = np.maximum(result, 0.0)
    # Return zero if either two-phase oil relperm is zero (conservative)
    return np.where((kro_w <= 0.0) | (kro_g <= 0.0), 0.0, result)


@stone_II_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> typing.Tuple[
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
]:
    """
    Analytical derivatives for Stone II (full normalized form).

    Let krocw = kro_endpoint,  A = kro_w/krocw + krw,  B = kro_g/krocw + krg

    kro = krocw * (A * B - krw - krg)

    Active region (kro > 0, both kro_w > 0 and kro_g > 0):

        ∂kro/∂kro_w = B          (from ∂(krocw * A * B)/∂kro_w = B)
        ∂kro/∂kro_g = A          (by symmetry)
        ∂kro/∂krw   = krocw * (B - 1)
        ∂kro/∂krg   = krocw * (A - 1)

    No explicit saturation dependence.
    """
    krocw = np.maximum(kro_endpoint, 1e-12)
    kw = np.asarray(kro_w, dtype=np.float64)
    kg = np.asarray(kro_g, dtype=np.float64)
    w = np.asarray(krw, dtype=np.float64)
    g = np.asarray(krg, dtype=np.float64)

    A = kw / krocw + w
    B = kg / krocw + g
    raw = krocw * (A * B - w - g)
    active = (kw > 0.0) & (kg > 0.0) & (raw > 0.0)

    d_kro_d_kro_w = np.where(active, B, 0.0)
    d_kro_d_kro_g = np.where(active, A, 0.0)
    d_kro_d_krw = np.where(active, krocw * (B - 1.0), 0.0)
    d_kro_d_krg = np.where(active, krocw * (A - 1.0), 0.0)
    z = _zeros_like_kro(kro_w)
    return (d_kro_d_kro_w, d_kro_d_kro_g, d_kro_d_krw, d_kro_d_krg, z, z, z)


@mixing_rule
@numba.njit(cache=True)
def arithmetic_mean_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
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


@arithmetic_mean_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> typing.Tuple[
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
]:
    """∂kro/∂kro_w = 0.5, ∂kro/∂kro_g = 0.5, no dependence on krw, krg, or saturations."""
    half = np.full_like(np.asarray(kro_w, dtype=np.float64), 0.5)
    z = _zeros_like_kro(kro_w)
    return (half, half, z, z, z, z, z)


@mixing_rule
@numba.njit(cache=True)
def geometric_mean_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
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


@geometric_mean_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> typing.Tuple[
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
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
    d_kro_d_kro_w = np.where(kro > 0.0, 0.5 * kg / safe_kro, 0.0)
    d_kro_d_kro_g = np.where(kro > 0.0, 0.5 * kw / safe_kro, 0.0)
    z = _zeros_like_kro(kro_w)
    return (d_kro_d_kro_w, d_kro_d_kro_g, z, z, z, z, z)


@mixing_rule
@numba.njit(cache=True)
def harmonic_mean_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
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


@harmonic_mean_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> typing.Tuple[
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
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
    d_kro_d_kro_w = np.where(both_positive, 2.0 * kg**2 / safe_sum**2, 0.0)
    d_kro_d_kro_g = np.where(both_positive, 2.0 * kw**2 / safe_sum**2, 0.0)
    z = _zeros_like_kro(kro_w)
    return (d_kro_d_kro_w, d_kro_d_kro_g, z, z, z, z, z)


@mixing_rule
@numba.njit(cache=True)
def baker_linear_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
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
    return result


@baker_linear_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> typing.Tuple[
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
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

    d_kro_d_kro_w = np.where(active, sw / T_safe, 0.0)
    d_kro_d_kro_g = np.where(active, sg / T_safe, 0.0)
    d_sw = np.where(active, sg * (kw - kg) / T_safe**2, 0.0)
    d_sg = np.where(active, sw * (kg - kw) / T_safe**2, 0.0)
    z = _zeros_like_kro(kro_w)
    return (d_kro_d_kro_w, d_kro_d_kro_g, z, z, d_sw, z, d_sg)


@mixing_rule
@numba.njit(cache=True)
def blunt_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
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


@blunt_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> typing.Tuple[
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
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
    d_kro_d_kro_w = np.where(active, kg * (2.0 - 2.0 * kw - kg), 0.0)
    d_kro_d_kro_g = np.where(active, kw * (2.0 - kw - 2.0 * kg), 0.0)
    z = _zeros_like_kro(kro_w)
    return (d_kro_d_kro_w, d_kro_d_kro_g, z, z, z, z, z)


@mixing_rule
@numba.njit(cache=True)
def hustad_hansen_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
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


@hustad_hansen_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> typing.Tuple[
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
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
    d_kro_d_kro_w = np.where(
        both_zero, 0.0, np.where(kw > kg, 0.0, np.where(kg > kw, 1.0, 0.5))
    )
    d_kro_d_kro_g = np.where(
        both_zero, 0.0, np.where(kg > kw, 0.0, np.where(kw > kg, 1.0, 0.5))
    )
    z = _zeros_like_kro(kro_w)
    return (d_kro_d_kro_w, d_kro_d_kro_g, z, z, z, z, z)


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

    def _aziz_settari_serializer(
        rule: MixingRule, recurse: bool = True
    ) -> typing.Dict[str, float]:
        return {"a": a, "b": b}

    def _aziz_settari_deserializer(data: typing.Any) -> MixingRule:
        if not isinstance(data, dict) or "a" not in data or "b" not in data:
            raise ValidationError("Invalid data for Aziz-Settari deserialization.")
        return aziz_settari_rule(a=data["a"], b=data["b"])

    @numba.njit(cache=True)
    def _func(
        kro_w: FloatOrArray,
        kro_g: FloatOrArray,
        krw: FloatOrArray,
        krg: FloatOrArray,
        kro_endpoint: FloatOrArray,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
    ) -> FloatOrArray:
        result = kro_w**a * kro_g**b
        return np.where((kro_w <= 0.0) | (kro_g <= 0.0), 0.0, result)

    rule: MixingRule = mixing_rule(
        _func,
        name=f"aziz_settari(a={a},b={b})",
        serializer=_aziz_settari_serializer,
        deserializer=_aziz_settari_deserializer,
    )

    @rule.dfunc
    @numba.njit(cache=True)
    def _dfunc(
        kro_w: FloatOrArray,
        kro_g: FloatOrArray,
        krw: FloatOrArray,
        krg: FloatOrArray,
        kro_endpoint: FloatOrArray,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
    ) -> typing.Tuple[
        FloatOrArray,
        FloatOrArray,
        FloatOrArray,
        FloatOrArray,
        FloatOrArray,
        FloatOrArray,
        FloatOrArray,
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
        d_kro_d_kro_w = np.where(active, a * safe_kw ** (a - 1.0) * safe_kg**b, 0.0)
        d_kro_d_kro_g = np.where(active, b * safe_kw**a * safe_kg ** (b - 1.0), 0.0)
        z = _zeros_like_kro(kro_w)
        return (d_kro_d_kro_w, d_kro_d_kro_g, z, z, z, z, z)

    return rule


@mixing_rule
@numba.njit(cache=True)
def eclipse_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
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


@eclipse_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> typing.Tuple[
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
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

    d_kro_d_kro_w = np.where(active & (Dw > 0.0), so / Dw_safe, 0.0)
    d_kro_d_kro_g = np.where(active & (Dg > 0.0), so / Dg_safe, 0.0)

    d_sw = np.where(active & (Dg > 0.0), -kg * so / Dg_safe**2, 0.0)
    d_so = np.where(
        active,
        np.where(Dw > 0.0, kw * sg / Dw_safe**2, 0.0)
        + np.where(Dg > 0.0, kg * sw / Dg_safe**2, 0.0),
        0.0,
    )
    d_sg = np.where(active & (Dw > 0.0), -kw * so / Dw_safe**2, 0.0)

    z = _zeros_like_kro(kro_w)
    return (d_kro_d_kro_w, d_kro_d_kro_g, z, z, d_sw, d_so, d_sg)


@mixing_rule
@numba.njit(cache=True)
def max_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
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


@max_rule.dfunc
@numba.njit(cache=True)
def _(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> typing.Tuple[
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
    FloatOrArray,
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
    d_kro_d_kro_w = np.where(kw > kg, 1.0, np.where(kg > kw, 0.0, 0.5))
    d_kro_d_kro_g = 1.0 - d_kro_d_kro_w
    z = _zeros_like_kro(kro_w)
    return (d_kro_d_kro_w, d_kro_d_kro_g, z, z, z, z, z)


def get_mixing_rule_partial_derivatives(
    rule: typing.Union[MixingRule, MixingRuleFunc],
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    krw: FloatOrArray,
    krg: FloatOrArray,
    kro_endpoint: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
    epsilon: float = 1e-7,
) -> MixingRulePartialDerivatives:
    """
    Compute partial derivatives of the three-phase mixing rule with respect
    to each of its saturation-dependent arguments.

    The mixing rule signature is:

    ```python
    kro = mixing_rule(kro_w, kro_g, krw, krg, kro_endpoint,
                      water_saturation, oil_saturation, gas_saturation)
    ```

    This function returns a dictionary containing seven partial derivatives:

    - `d_kro_d_kro_w`: sensitivity to the oil-water two-phase oil kr.
    - `d_kro_d_kro_g`: sensitivity to the gas-oil two-phase oil kr.
    - `d_kro_d_krw`: sensitivity to the two-phase water kr.
    - `d_kro_d_krg`: sensitivity to the two-phase gas kr.
    - `d_kro_d_sw_explicit`: explicit Sw dependence in the mixing rule.
    - `d_kro_d_so_explicit`: explicit So dependence in the mixing rule.
    - `d_kro_d_sg_explicit`: explicit Sg dependence in the mixing rule.

    :param rule: The mixing rule callable.
    :param kro_w: Oil relative permeability from the oil-water two-phase table.
    :param kro_g: Oil relative permeability from the gas-oil two-phase table.
    :param krw: Water relative permeability from the oil-water two-phase table.
    :param krg: Gas relative permeability from the gas-oil two-phase table.
    :param kro_endpoint: Oil relative permeability at connate water.
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
            kro_endpoint=kro_endpoint,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            epsilon=epsilon,
        )
    return _central_difference_partial_derivatives(
        rule=rule,
        kro_w=kro_w,
        kro_g=kro_g,
        krw=krw,
        krg=krg,
        kro_endpoint=kro_endpoint,
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        epsilon=epsilon,
    )


class RelativePermeabilityTable(StoreSerializable):
    """
    Protocol for a relative permeability table that computes relative permeabilities based on fluid saturations.
    """

    __abstract_serializable__ = True

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        raise NotImplementedError

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        raise NotImplementedError

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
class TwoPhaseRelPermTable(
    Serializable,
    load_exclude={
        "_wetting_pchip",
        "_wetting_dpchip",
        "_non_wetting_pchip",
        "_non_wetting_dpchip",
    },
    dump_exclude={
        "_wetting_pchip",
        "_wetting_dpchip",
        "_non_wetting_pchip",
        "_non_wetting_dpchip",
    },
):
    """
    Two-phase relative permeability lookup table backed by a PCHIP interpolant.

    Interpolates relative permeabilities for two fluid phases based on a
    reference saturation value. The reference saturation can be either the
    wetting or non-wetting phase saturation, depending on how the table was
    constructed (e.g. from lab data indexed by Sg vs So).

    Supports both scalar and array inputs up to 3D.

    Examples:

    - Oil-Water system (water-wet): reference is Sw (wetting phase),
      `reference_phase="wetting"`
    - Gas-Oil system indexed by So: reference is So (wetting phase),
      `reference_phase="wetting"`
    - Gas-Oil system indexed by Sg: reference is Sg (non-wetting phase),
      `reference_phase="non_wetting"`

    **Grid scaling** (`number_of_base_points` / `number_of_endpoint_extra_points`):

    When `number_of_base_points > 0` and the raw knot count is smaller than
    `number_of_base_points`, the table expands the knot grid to `number_of_base_points`
    base points (plus `number_of_endpoint_extra_points` extra knots in each boundary
    decade) before fitting the PCHIP interpolant. This gives smoother curves
    and better derivative accuracy near residual saturations without requiring
    the caller to pre-supply a dense grid.  Pass `number_of_base_points=0` to
    disable scaling entirely and use the raw knots directly.

    **Minimum relperm floors** (`min_wetting_relperm` / `min_non_wetting_relperm`):

    When non-`None`, a floor is applied to the interpolated kr value and
    the derivative is zeroed out wherever the raw (pre-floor) kr is at or
    below the floor. This gives a derivative that is exactly consistent with
    the floored kr, preventing Jacobian-residual mismatches that cause MBE.

    `"auto"` derives the floor from the active floating-point dtype:
    `max(4 * machine_epsilon, 1e-8)`. `None` disables the floor (kr can
    reach zero exactly).
    """

    wetting_phase: typing.Union[FluidPhase, str] = attrs.field(converter=FluidPhase)
    """The wetting fluid phase, e.g. WATER (oil-water) or OIL (gas-oil)."""

    non_wetting_phase: typing.Union[FluidPhase, str] = attrs.field(converter=FluidPhase)
    """The non-wetting fluid phase, e.g. OIL (oil-water) or GAS (gas-oil)."""

    reference_saturation: npt.NDArray[np.floating] = attrs.field(converter=bores_array)
    """
    Saturation values used as the x-axis for interpolation, monotonically
    increasing. May represent either the wetting or non-wetting phase
    saturation depending on `reference_phase`.
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
 
    - `"wetting"` — `reference_saturation` holds wetting phase saturation
      values.  krw increases and krnw decreases as `reference_saturation`
      increases.
    - `"non_wetting"` — `reference_saturation` holds non-wetting phase
      saturation values.  krnw increases and krw decreases as
      `reference_saturation` increases.
 
    This does not change the interpolation mechanics. It only records which
    physical saturation the caller must supply when querying the table, so
    that `ThreePhaseRelPermTable` (and any other consumer) can dispatch the
    correct saturation grid without hard-coding assumptions.
    """

    min_wetting_relperm: RelPermFloor = attrs.field(default=None)
    """
    Minimum floor for the wetting-phase relative permeability.
 
    `"auto"` — `max(4 * machine_epsilon, 1e-8)` (dtype-aware).
    `None` — no floor; kr can reach zero exactly.
    `float` — explicit user-supplied floor value.
 
    The floor is applied to the interpolated kr value, and the derivative is
    zeroed out in the floored region so that the Jacobian is consistent with
    the kr value (no MBE from mismatched kr/derivative pairs).
    """

    min_non_wetting_relperm: RelPermFloor = attrs.field(default=None)
    """
    Minimum floor for the non-wetting-phase relative permeability.
 
    Same semantics as `min_wetting_relperm`.
    """

    number_of_base_points: int = attrs.field(default=200)
    """
    Target number of base knot points used when expanding the raw saturation
    grid before fitting the PCHIP interpolant.
 
    When the number of raw knots is already >= `number_of_base_points`, no
    expansion is performed and the raw knots are used as-is.
    Pass `0` to disable grid scaling entirely.
    """

    number_of_endpoint_extra_points: int = attrs.field(default=20)
    """
    Number of extra knots injected into the first and last 10 % of the
    saturation range during grid expansion (see `number_of_base_points`).
 
    These additional knots improve derivative accuracy near residual
    saturations where kr curves vary most rapidly.
    Pass `0` to disable endpoint enrichment.
    """

    spacing: Spacing = attrs.field(default="cosine")
    """
    Grid spacing mode used when building the expanded knot grid.
 
    Passed directly to `_build_saturation_reference_grid`.  Typical
    values are `"cosine"` (default, clusters points near endpoints) and
    `"linspace"` (uniform).
    """

    _wetting_pchip: PchipInterpolator = attrs.field(init=False, repr=False)
    _wetting_dpchip: PchipInterpolator = attrs.field(init=False, repr=False)
    _non_wetting_pchip: PchipInterpolator = attrs.field(init=False, repr=False)
    _non_wetting_dpchip: PchipInterpolator = attrs.field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if self.reference_phase not in ("wetting", "non_wetting"):
            raise ValidationError(
                f"`reference_phase` must be 'wetting' or 'non_wetting', "
                f"got {self.reference_phase!r}"
            )
        if len(self.reference_saturation) != len(
            self.wetting_phase_relative_permeability
        ):
            raise ValidationError(
                f"`reference_saturation` and wetting phase kr arrays must have same "
                f"length. Got {len(self.reference_saturation)} vs "
                f"{len(self.wetting_phase_relative_permeability)}"
            )
        if len(self.reference_saturation) != len(
            self.non_wetting_phase_relative_permeability
        ):
            raise ValidationError(
                f"`reference_saturation` and non-wetting phase kr arrays must have "
                f"same length. Got {len(self.reference_saturation)} vs "
                f"{len(self.non_wetting_phase_relative_permeability)}"
            )
        if len(self.reference_saturation) < 2:
            raise ValidationError("At least 2 points required for interpolation")
        if not np.all(np.diff(self.reference_saturation) >= 0):
            raise ValidationError(
                "`reference_saturation` must be monotonically increasing"
            )

        # Validate floor sentinels eagerly so errors surface at construction time
        _resolve_relperm_floor(self.min_wetting_relperm)
        _resolve_relperm_floor(self.min_non_wetting_relperm)

        # Build interpolants
        wetting_pchip, wetting_dpchip = build_pchip_interpolant(
            reference_saturation=self.reference_saturation,
            values=self.wetting_phase_relative_permeability,
            number_of_base_points=self.number_of_base_points,
            number_of_endpoint_extra_points=self.number_of_endpoint_extra_points,
            spacing=self.spacing,
        )
        non_wetting_pchip, non_wetting_dpchip = build_pchip_interpolant(
            reference_saturation=self.reference_saturation,
            values=self.non_wetting_phase_relative_permeability,
            number_of_base_points=self.number_of_base_points,
            number_of_endpoint_extra_points=self.number_of_endpoint_extra_points,
            spacing=self.spacing,
        )
        object.__setattr__(self, "_wetting_pchip", wetting_pchip)
        object.__setattr__(self, "_wetting_dpchip", wetting_dpchip)
        object.__setattr__(self, "_non_wetting_pchip", non_wetting_pchip)
        object.__setattr__(self, "_non_wetting_dpchip", non_wetting_dpchip)

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        return self.wetting_phase  # type: ignore[return-value]

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        return self.wetting_phase  # type: ignore[return-value]

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

    def _query_pchip(
        self,
        interpolant: PchipInterpolator,
        reference: FloatOrArray,
        extrapolate_left: float,
        extrapolate_right: float,
    ) -> FloatOrArray:
        """
        Evaluate a pre-built PCHIP interpolant at `reference`, applying
        constant extrapolation at the boundaries.

        :param interpolant: Pre-built `PchipInterpolator` instance.
        :param reference: Query saturation value(s) — scalar or array.
        :param extrapolate_left: Constant returned for values below the knot range.
        :param extrapolate_right: Constant returned for values above the knot range.
        :return: Interpolated value(s) with the same shape as `reference`.
        """
        is_scalar = np.isscalar(reference)
        sat = np.atleast_1d(np.asarray(reference, dtype=np.float64))
        x_min = float(interpolant.x[0])
        x_max = float(interpolant.x[-1])

        result = interpolant(np.clip(sat, x_min, x_max))
        result = np.where(sat < x_min, extrapolate_left, result)
        result = np.where(sat > x_max, extrapolate_right, result)

        if is_scalar:
            return float(result.ravel()[0])
        return result.reshape(sat.shape)

    def _query_dpchip(
        self,
        d_interpolant: PchipInterpolator,
        reference: FloatOrArray,
    ) -> FloatOrArray:
        """
        Evaluate a pre-built PCHIP derivative interpolant at `reference`,
        returning zero outside the knot range (constant extrapolation = zero slope).

        :param d_interpolant: Pre-built derivative `PchipInterpolator`.
        :param reference: Query saturation value(s) — scalar or array.
        :return: Derivative value(s) with the same shape as `reference`.
        """
        is_scalar = np.isscalar(reference)
        sat = np.atleast_1d(np.asarray(reference, dtype=np.float64))
        x_min = float(d_interpolant.x[0])
        x_max = float(d_interpolant.x[-1])

        result = d_interpolant(np.clip(sat, x_min, x_max))
        result = np.where((sat < x_min) | (sat > x_max), 0.0, result)

        if is_scalar:
            return float(result.ravel()[0])
        return result.reshape(sat.shape)

    def get_wetting_phase_relative_permeability(
        self,
        wetting_saturation: FloatOrArray,
        non_wetting_saturation: typing.Optional[FloatOrArray] = None,
    ) -> FloatOrArray:
        """
        Get wetting phase relative permeability.

        When `reference_phase="wetting"`, only `wetting_saturation` is
        needed. When `reference_phase="non_wetting"`, `non_wetting_saturation` must be supplied.

        The `min_wetting_relperm` floor (if set) is applied to the result.

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
        kr = self._query_pchip(
            self._wetting_pchip,
            ref,
            extrapolate_left=float(self.wetting_phase_relative_permeability[0]),
            extrapolate_right=float(self.wetting_phase_relative_permeability[-1]),
        )
        floor = _resolve_relperm_floor(self.min_wetting_relperm)
        return _apply_relperm_floor(kr, floor)

    def get_non_wetting_phase_relative_permeability(
        self,
        wetting_saturation: FloatOrArray,
        non_wetting_saturation: typing.Optional[FloatOrArray] = None,
    ) -> FloatOrArray:
        """
        Get non-wetting phase relative permeability.

        When `reference_phase="wetting"`, only `wetting_saturation` is
        needed. When `reference_phase="non_wetting"`, `non_wetting_saturation` must be supplied.

        The `min_non_wetting_relperm` floor (if set) is applied to the result.

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
        kr = self._query_pchip(
            self._non_wetting_pchip,
            ref,
            extrapolate_left=float(self.non_wetting_phase_relative_permeability[0]),
            extrapolate_right=float(self.non_wetting_phase_relative_permeability[-1]),
        )
        floor = _resolve_relperm_floor(self.min_non_wetting_relperm)
        return _apply_relperm_floor(kr, floor)

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
        :return: Tuple of `(wetting_kr, non_wetting_kr)`.
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
        the reference saturation axis of this table, evaluated from the
        analytical PCHIP derivative.

        The derivative is zero outside the tabulated saturation range
        (constant extrapolation = zero slope).

        Where the `min_wetting_relperm` floor is active (raw kr ≤ floor),
        the derivative is zeroed out to be consistent with the floored
        (constant) kr value, preventing Jacobian-residual mismatches.

        :param wetting_saturation: Wetting-phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting-phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Derivative value(s) with the same shape as the input.
        """
        ref = self._resolve_reference(
            wetting_saturation,
            non_wetting_saturation
            if non_wetting_saturation is not None
            else wetting_saturation,
        )
        dkr = self._query_dpchip(self._wetting_dpchip, ref)
        floor = _resolve_relperm_floor(self.min_wetting_relperm)
        if floor is None:
            return dkr

        # Need raw kr to decide where floor is active
        kr_raw = self._query_pchip(
            self._wetting_pchip,
            ref,
            extrapolate_left=float(self.wetting_phase_relative_permeability[0]),
            extrapolate_right=float(self.wetting_phase_relative_permeability[-1]),
        )
        return _apply_relperm_floor_to_derivative(dkr, kr_raw, floor)

    def get_non_wetting_phase_relative_permeability_derivative(
        self,
        wetting_saturation: FloatOrArray,
        non_wetting_saturation: typing.Optional[FloatOrArray] = None,
    ) -> FloatOrArray:
        """
        Derivative of the non-wetting-phase relative permeability with respect
        to the reference saturation axis of this table, evaluated from the
        analytical PCHIP derivative.

        The derivative is zero outside the tabulated saturation range
        (constant extrapolation = zero slope).

        Where the `min_non_wetting_relperm` floor is active (raw kr ≤
        floor), the derivative is zeroed out to be consistent with the floored
        (constant) kr value, preventing Jacobian-residual mismatches.

        :param wetting_saturation: Wetting-phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting-phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Derivative value(s) with the same shape as the input.
        """
        ref = self._resolve_reference(
            wetting_saturation,
            non_wetting_saturation
            if non_wetting_saturation is not None
            else wetting_saturation,
        )
        dkr = self._query_dpchip(self._non_wetting_dpchip, ref)
        floor = _resolve_relperm_floor(self.min_non_wetting_relperm)
        if floor is None:
            return dkr

        kr_raw = self._query_pchip(
            self._non_wetting_pchip,
            ref,
            extrapolate_left=float(self.non_wetting_phase_relative_permeability[0]),
            extrapolate_right=float(self.non_wetting_phase_relative_permeability[-1]),
        )
        return _apply_relperm_floor_to_derivative(dkr, kr_raw, floor)


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

    Minimum relperm floors on the two-phase tables propagate automatically
    into this three-phase table since the two-phase table methods apply the
    floor before returning.  For the mixing-rule chain rule the floored
    derivative (zeroed in the flat region) is also used consistently, so the
    assembled Jacobian never disagrees with the kr values.

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

        rule(kro_w, kro_g, krw, krg, kro_endpoint,
             water_saturation, oil_saturation, gas_saturation) -> kro

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

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        return self.oil_water_table.wetting_phase  # type:ignore[return-value]

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        return self.gas_oil_table.wetting_phase  # type:ignore[return-value]

    def _get_kro_endpoint(self) -> float:
        """
        Resolve the oil relative permeability at connate water saturation.

        For a normalized table this equals 1.0.  We approximate it as the
        maximum value of the non-wetting (oil) phase kr in the oil-water
        table, which equals kro(Sw=Swc, Sg=0).

        :return: kro endpoint scalar.
        """
        if self.oil_water_table.wetting_phase == FluidPhase.WATER:
            return float(
                np.max(self.oil_water_table.non_wetting_phase_relative_permeability)
            )
        return float(np.max(self.oil_water_table.wetting_phase_relative_permeability))

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

        Minimum relperm floors declared on the two-phase tables are applied
        automatically inside the table query methods and propagate into the
        mixing-rule inputs transparently.

        :param water_saturation: Water saturation (fraction) — scalar or array.
        :param oil_saturation: Oil saturation (fraction) — scalar or array.
        :param gas_saturation: Gas saturation (fraction) — scalar or array.
        :return: `RelativePermeabilities` dict with keys "water", "oil", "gas".
        """
        sw = np.atleast_1d(water_saturation)
        so = np.atleast_1d(oil_saturation)
        sg = np.atleast_1d(gas_saturation)
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )

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

        kro_endpoint = self._get_kro_endpoint()

        # Three-phase oil mixing rule
        if self.mixing_rule is not None:
            kro = self.mixing_rule(  # type: ignore
                kro_w=kro_w,
                kro_g=kro_g,
                krw=krw,
                krg=krg,
                kro_endpoint=kro_endpoint,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
            )
        else:
            kro = np.minimum(kro_w, kro_g)

        if is_scalar:
            return RelativePermeabilities(
                water=krw.item(),  # type: ignore
                oil=kro.item(),  # type: ignore
                gas=krg.item(),  # type: ignore
            )
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
        ``
        (dkrw/dSw, dkrw/dSo, dkrw/dSg,
        dkro/dSw, dkro/dSo, dkro/dSg,
        dkrg/dSw, dkrg/dSo, dkrg/dSg)
        ```

        Water and gas relative permeability derivatives are computed analytically
        from the slopes of the underlying two-phase lookup tables.  Where a
        minimum relperm floor is active on a two-phase table, the derivative
        returned by that table is already zeroed out in the floored region,
        so the assembled nine-derivative Jacobian is automatically consistent
        with the floored kr values.

        Oil relative permeability derivatives use the full extended chain rule
        through the three-phase mixing rule:
        ```
        dkro/dSalpha = (d_kro/d_kro_w) * (d_kro_w/d_Salpha)
                     + (d_kro/d_kro_g) * (d_kro_g/d_Salpha)
                     + (d_kro/d_krw)   * (d_krw/d_Salpha)
                     + (d_kro/d_krg)   * (d_krg/d_Salpha)
                     + (d_kro/d_Salpha)_explicit_in_mixing_rule
        ```

        The ``d_kro/d_krw`` and ``d_kro/d_krg`` terms are non-zero only for
        rules that use the actual two-phase water/gas kr values (e.g. the full
        Stone II rule); they are zero for all other built-in rules.

        :param water_saturation: Water saturation (fraction, 0 to 1).
        :param oil_saturation: Oil saturation (fraction, 0 to 1).
        :param gas_saturation: Gas saturation (fraction, 0 to 1).
        :return: `RelativePermeabilityDerivatives` dictionary.
        """
        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )
        sw = np.atleast_1d(np.asarray(water_saturation, dtype=np.float64))
        so = np.atleast_1d(np.asarray(oil_saturation, dtype=np.float64))
        sg = np.atleast_1d(np.asarray(gas_saturation, dtype=np.float64))
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        zeros = np.zeros_like(sw)

        oil_water_table = self.oil_water_table
        gas_oil_table = self.gas_oil_table

        # Oil-water table derivatives: krw and kro_w
        if oil_water_table.wetting_phase == FluidPhase.WATER:
            if oil_water_table.reference_phase == "wetting":
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
        if gas_oil_table.wetting_phase == FluidPhase.OIL:
            if gas_oil_table.reference_phase == "wetting":
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

        # Forward two-phase oil/water/gas kr values for mixing rule call
        if oil_water_table.wetting_phase == FluidPhase.WATER:
            kro_w = oil_water_table.get_non_wetting_phase_relative_permeability(
                sw, non_wetting_saturation=so
            )
            krw = oil_water_table.get_wetting_phase_relative_permeability(
                sw, non_wetting_saturation=so
            )
        else:
            kro_w = oil_water_table.get_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sw
            )
            krw = oil_water_table.get_non_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sw
            )

        if gas_oil_table.wetting_phase == FluidPhase.OIL:
            kro_g = gas_oil_table.get_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sg
            )
            krg = gas_oil_table.get_non_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sg
            )
        else:
            kro_g = gas_oil_table.get_non_wetting_phase_relative_permeability(
                sg, non_wetting_saturation=so
            )
            krg = gas_oil_table.get_wetting_phase_relative_permeability(
                sg, non_wetting_saturation=so
            )

        kro_endpoint = self._get_kro_endpoint()

        # Three-phase oil mixing rule derivatives
        mixing_rule = self.mixing_rule
        if mixing_rule is None:
            # Fallback: kro = min(kro_w, kro_g)
            kro_w_arr = np.asarray(kro_w, dtype=np.float64)
            kro_g_arr = np.asarray(kro_g, dtype=np.float64)
            d_kro_d_kro_w = np.where(kro_w_arr <= kro_g_arr, 1.0, 0.0)
            d_kro_d_kro_g = np.where(kro_g_arr <= kro_w_arr, 1.0, 0.0)
            d_kro_d_krw = zeros.copy()
            d_kro_d_krg = zeros.copy()
            d_kro_d_sw_explicit = zeros.copy()
            d_kro_d_so_explicit = zeros.copy()
            d_kro_d_sg_explicit = zeros.copy()
        else:
            derivatives = get_mixing_rule_partial_derivatives(
                rule=mixing_rule,  # type: ignore[arg-type]
                kro_w=kro_w,
                kro_g=kro_g,
                krw=krw,
                krg=krg,
                kro_endpoint=kro_endpoint,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                epsilon=c.FINITE_DIFFERENCE_EPSILON,
            )
            d_kro_d_kro_w = derivatives["d_kro_d_kro_w"]
            d_kro_d_kro_g = derivatives["d_kro_d_kro_g"]
            d_kro_d_krw = derivatives["d_kro_d_krw"]
            d_kro_d_krg = derivatives["d_kro_d_krg"]
            d_kro_d_sw_explicit = derivatives["d_kro_d_sw_explicit"]
            d_kro_d_so_explicit = derivatives["d_kro_d_so_explicit"]
            d_kro_d_sg_explicit = derivatives["d_kro_d_sg_explicit"]

        # Extended chain rule:
        # dkro/dSalpha = (d_kro/d_kro_w) * (d_kro_w/d_Salpha)
        #              + (d_kro/d_kro_g) * (d_kro_g/d_Salpha)
        #              + (d_kro/d_krw)   * (d_krw/d_Salpha)
        #              + (d_kro/d_krg)   * (d_krg/d_Salpha)
        #              + (d_kro/d_Salpha)_explicit
        d_kro_d_sw = (
            d_kro_d_kro_w * d_kro_w_d_sw
            + d_kro_d_kro_g * d_kro_g_d_sw
            + d_kro_d_krw * d_krw_d_sw
            + d_kro_d_krg * d_krg_d_sw
            + d_kro_d_sw_explicit
        )
        d_kro_d_so = (
            d_kro_d_kro_w * d_kro_w_d_so
            + d_kro_d_kro_g * d_kro_g_d_so
            + d_kro_d_krw * d_krw_d_so
            + d_kro_d_krg * d_krg_d_so
            + d_kro_d_so_explicit
        )
        d_kro_d_sg = (
            d_kro_d_kro_w * d_kro_w_d_sg
            + d_kro_d_kro_g * d_kro_g_d_sg
            + d_kro_d_krw * d_krw_d_sg
            + d_kro_d_krg * d_krg_d_sg
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
            results = tuple(r.item() for r in results)  # type: ignore
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
    irreducible_water_saturation: FloatOrArray,
    residual_oil_saturation_water: FloatOrArray,
    residual_oil_saturation_gas: FloatOrArray,
    residual_gas_saturation: FloatOrArray,
    water_exponent: float,
    oil_exponent: float,
    gas_exponent: float,
    wettability: Wettability = Wettability.WATER_WET,
    mixed_wet_water_fraction: float = 0.5,
    mixing_rule: MixingRule = eclipse_rule,
    saturation_epsilon: float = 1e-6,
    minimum_mobile_pore_space: float = 1e-9,
    min_water_relperm: typing.Optional[float] = None,
    min_oil_relperm: typing.Optional[float] = None,
    min_gas_relperm: typing.Optional[float] = None,
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
    :param mixed_wet_water_fraction: Fraction of pore space considered water-wet in mixed-wet systems (0 to 1).
    :param mixing_rule: Mixing rule function for three-phase oil relative permeability.
    :param saturation_epsilon: Tolerance for checking if saturations sum to 1.
    :param minimum_mobile_pore_space: Minimum mobile pore space to avoid division by zero in effective saturation calculations.
    :param min_water_relperm: Resolved minimum floor for water kr (`None` = no floor).
    :param min_oil_relperm: Resolved minimum floor for oil kr (`None` = no floor).
    :param min_gas_relperm: Resolved minimum floor for gas kr (`None` = no floor).
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

    # kro_endpoint for mixing rule: 1.0 for unit-endpoint Corey models
    kro_endpoint = 1.0

    if wettability == Wettability.WATER_WET:
        # 1. Water relperm (wetting phase)
        movable_water_range = (
            1.0 - irreducible_water_saturation - residual_oil_saturation_water  # type: ignore[operator]
        )
        effective_water_saturation = np.where(
            movable_water_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip(
                (sw - irreducible_water_saturation) / movable_water_range, 0.0, 1.0
            ),
        )
        krw = effective_water_saturation**water_exponent

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
        krg = effective_gas_saturation**gas_exponent

        # 3. Oil relperm (intermediate phase) - mixing rule blending
        # Make sure to apply oil curvature to the two-phase oil kr inputs before mixing,
        # not to the mixed output. (1-krw) and (1-krg) are the two-phase
        # oil kr approximations; so we shape them with `oil_exponent` before blending.
        kro_w_shaped = (1.0 - krw) ** oil_exponent
        kro_g_shaped = (1.0 - krg) ** oil_exponent
        kro = mixing_rule(
            kro_w=kro_w_shaped,
            kro_g=kro_g_shaped,
            krw=krw,
            krg=krg,
            kro_endpoint=kro_endpoint,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
        )

    elif wettability == Wettability.OIL_WET:
        # Oil is wetting, water becomes intermediate
        # 1. Oil relperm (wetting phase)
        movable_oil_range = (
            1.0 - residual_oil_saturation_water - residual_oil_saturation_gas  # type: ignore[operator]
        )
        max_residual = np.minimum(
            residual_oil_saturation_water, residual_oil_saturation_gas
        )
        effective_oil_saturation = np.where(
            movable_oil_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - max_residual) / movable_oil_range, 0.0, 1.0),
        )
        kro = effective_oil_saturation**oil_exponent

        # 2. Gas relperm (nonwetting phase)
        movable_gas_range = 1.0 - residual_gas_saturation - irreducible_water_saturation  # type: ignore[operator]
        effective_gas_saturation = np.where(
            movable_gas_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - residual_gas_saturation) / movable_gas_range, 0.0, 1.0),
        )
        krg = effective_gas_saturation**gas_exponent

        # 3. Water relperm (intermediate phase, use mixing rule style blending)
        kro_proxy_shaped = (1.0 - kro) ** water_exponent
        krg_proxy_shaped = (1.0 - krg) ** water_exponent
        krw = mixing_rule(  # type: ignore[assignment]
            kro_w=kro_proxy_shaped,  # treat oil as wetting
            kro_g=krg_proxy_shaped,  # treat gas as nonwetting
            krw=kro,
            krg=krg,
            kro_endpoint=kro_endpoint,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
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
            np.clip(
                (sw - irreducible_water_saturation) / movable_water_range_ww, 0.0, 1.0
            ),
        )
        krw_ww = effective_water_saturation_ww**water_exponent

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
        krg_ww = effective_gas_saturation_ww**gas_exponent
        kro_w_ww = (1.0 - krw_ww) ** oil_exponent
        kro_g_ww = (1.0 - krg_ww) ** oil_exponent
        kro_ww = mixing_rule(
            kro_w=kro_w_ww,
            kro_g=kro_g_ww,
            krw=krw_ww,
            krg=krg_ww,
            kro_endpoint=kro_endpoint,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
        )

        # Oil-wet contribution
        movable_oil_range_ow = (
            1.0 - residual_oil_saturation_water - residual_oil_saturation_gas  # type: ignore[operator]
        )
        max_residual_ow = np.minimum(
            residual_oil_saturation_water, residual_oil_saturation_gas
        )
        effective_oil_saturation_ow = np.where(
            movable_oil_range_ow <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - max_residual_ow) / movable_oil_range_ow, 0.0, 1.0),
        )
        kro_ow = effective_oil_saturation_ow**oil_exponent

        movable_gas_range_ow = (
            1.0 - residual_gas_saturation - irreducible_water_saturation  # type: ignore[operator]
        )
        effective_gas_saturation_ow = np.where(
            movable_gas_range_ow <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - residual_gas_saturation) / movable_gas_range_ow, 0.0, 1.0),
        )
        krg_ow = effective_gas_saturation_ow**gas_exponent
        kro_proxy_ow = (1.0 - kro_ow) ** water_exponent
        krg_proxy_ow = (1.0 - krg_ow) ** water_exponent
        krw_ow = mixing_rule(
            kro_w=kro_proxy_ow,
            kro_g=krg_proxy_ow,
            krw=kro_ow,
            krg=krg_ow,
            kro_endpoint=kro_endpoint,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
        )

        # Weighted blend
        krw = (
            mixed_wet_water_fraction * krw_ww
            + (1.0 - mixed_wet_water_fraction) * krw_ow
        )
        kro = (
            mixed_wet_water_fraction * kro_ww
            + (1.0 - mixed_wet_water_fraction) * kro_ow
        )
        krg = (
            mixed_wet_water_fraction * krg_ww
            + (1.0 - mixed_wet_water_fraction) * krg_ow
        )

    else:
        raise ValidationError(f"Wettability {wettability!r} not implemented.")

    # Clip all results to [0, 1] then apply per-phase minimum floors
    krw = np.clip(krw, 0.0, 1.0)
    kro = np.clip(kro, 0.0, 1.0)
    krg = np.clip(krg, 0.0, 1.0)
    krw = _apply_relperm_floor(krw, min_water_relperm)
    kro = _apply_relperm_floor(kro, min_oil_relperm)
    krg = _apply_relperm_floor(krg, min_gas_relperm)
    if is_scalar:
        krw = krw.item()  # type: ignore
        kro = kro.item()  # type: ignore
        krg = krg.item()  # type: ignore
    return krw, kro, krg  # type: ignore[return-value]


@relperm_table
@attrs.frozen
class BrooksCoreyRelPermModel(
    RelativePermeabilityTable,
    serializers={"mixing_rule": serialize_mixing_rule},
    deserializers={"mixing_rule": deserialize_mixing_rule},
    load_exclude={"supports_arrays"},
    dump_exclude={"supports_arrays"},
):
    """
    Brooks-Corey-type three-phase relative permeability model.

    Supports water-wet and oil-wet wettability assumptions.

    **Minimum relperm floors** (`min_water_relperm`, `min_oil_relperm`, `min_gas_relperm`):

    `"auto"` - `max(4 * machine_epsilon, 1e-8)` (dtype-aware, same as
    the CMG IMEX/GEM minimum mobility approach). `None` - no floor
    (default, kr can reach zero exactly). `float` - explicit user-set
    value.

    The floor is applied to the computed kr value and the derivative is
    zeroed out in the floored region, so the Jacobian is always consistent
    with the kr value (no MBE from mismatched kr/derivative pairs).
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
    Corey exponent for oil relative permeability.

    Higher values make the curve steeper. Meaning slower kro increase with saturation.
    """
    gas_exponent: float = 2.0
    """
    Corey exponent for gas relative permeability. Higher values make the curve steeper.

    Meaning slower krg increase with saturation.
    """
    wettability: Wettability = Wettability.WATER_WET
    """Wettability type (water-wet or oil-wet)."""

    mixed_wet_water_fraction: float = 0.5
    """Fraction of pore space that is water-wet in mixed-wet systems (0-1)."""

    mixing_rule: typing.Union[MixingRule, str] = eclipse_rule
    """
    Mixing rule function or name to compute oil relative permeability in three-phase system.

    The function should take the following parameters in order:
    - kro_w: Oil relative permeability from oil-water table
    - kro_g: Oil relative permeability from oil-gas table
    - krw: Water relative permeability from oil-water table
    - krg: Gas relative permeability from gas-oil table
    - kro_endpoint: Oil relative permeability at connate water
    - Sw: Water saturation
    - So: Oil saturation
    - Sg: Gas saturation
    and return the mixed oil relative permeability.
    """

    min_water_relperm: RelPermFloor = "auto"
    """
    Minimum floor for the water relative permeability.

    `"auto"` - dtype-aware floor; `None` - no floor; `float` - explicit value.
    """

    min_oil_relperm: RelPermFloor = "auto"
    """
    Minimum floor for the oil relative permeability.

    `"auto"` - dtype-aware floor; `None` - no floor; `float` - explicit value.
    """

    min_gas_relperm: RelPermFloor = "auto"
    """
    Minimum floor for the gas relative permeability.

    `"auto"` - dtype-aware floor; `None` - no floor; `float` - explicit value.
    """

    supports_arrays: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array inputs."""

    def __attrs_post_init__(self) -> None:
        mixing_rule = self.mixing_rule
        if isinstance(mixing_rule, str):
            object.__setattr__(self, "mixing_rule", get_mixing_rule(mixing_rule))
        # Validate floor sentinels eagerly
        _resolve_relperm_floor(self.min_water_relperm)
        _resolve_relperm_floor(self.min_oil_relperm)
        _resolve_relperm_floor(self.min_gas_relperm)

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

    def get_relative_permeabilities(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        irreducible_water_saturation: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_water: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_gas: typing.Optional[FloatOrArray] = None,
        residual_gas_saturation: typing.Optional[FloatOrArray] = None,
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
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
            mixing_rule=self.mixing_rule,  # type: ignore[arg-type]
            saturation_epsilon=c.SATURATION_EPSILON,
            minimum_mobile_pore_space=c.MINIMUM_MOBILE_PORE_SPACE,
            min_water_relperm=_resolve_relperm_floor(self.min_water_relperm),
            min_oil_relperm=_resolve_relperm_floor(self.min_oil_relperm),
            min_gas_relperm=_resolve_relperm_floor(self.min_gas_relperm),
        )
        return RelativePermeabilities(water=krw, oil=kro, gas=krg)  # type: ignore[typeddict-item]

    def get_relative_permeability_derivatives(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        irreducible_water_saturation: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_water: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_gas: typing.Optional[FloatOrArray] = None,
        residual_gas_saturation: typing.Optional[FloatOrArray] = None,
        **kwargs: typing.Any,
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
        by the extended chain rule through the mixing rule (including the
        `d_kro/d_krw` and `d_kro/d_krg` terms for rules like Stone II that
        use the actual two-phase water/gas kr values).

        Wherever a minimum relperm floor is active (raw kr ≤ floor), the
        corresponding derivative is zeroed out, keeping the Jacobian consistent
        with the floored kr value and preventing MBE.

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

        floor_w = _resolve_relperm_floor(self.min_water_relperm)
        floor_o = _resolve_relperm_floor(self.min_oil_relperm)
        floor_g = _resolve_relperm_floor(self.min_gas_relperm)

        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
            and np.isscalar(Swc)
            and np.isscalar(Sorw)
            and np.isscalar(Sorg)
            and np.isscalar(Sgr)
        )
        sw = np.atleast_1d(water_saturation)
        so = np.atleast_1d(oil_saturation)
        sg = np.atleast_1d(gas_saturation)
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        zeros = np.zeros_like(sw)
        minimum_mobile_pore_space = c.MINIMUM_MOBILE_PORE_SPACE

        if wettability == Wettability.OIL_WET:
            # kro (wetting, depends on So)
            movable_oil_range = 1.0 - Sorw - Sorg  # type: ignore
            max_residual = np.minimum(Sorw, Sorg)  # type: ignore
            valid_oil = movable_oil_range > minimum_mobile_pore_space
            se_o = np.clip(
                (so - max_residual) / np.where(valid_oil, movable_oil_range, 1.0),
                0.0,
                1.0,
            )
            kro_vals = se_o**oil_exponent

            d_kro_d_so_raw = np.where(
                valid_oil & (se_o > 0.0),
                oil_exponent * se_o ** max(oil_exponent - 1.0, 0.0) / movable_oil_range,
                zeros,
            )
            # Apply floor: zero derivative where raw kro ≤ floor
            d_kro_d_so = _apply_relperm_floor_to_derivative(
                d_kro_d_so_raw, kro_vals, floor_o
            )
            d_kro_d_sw = zeros.copy()
            d_kro_d_sg = zeros.copy()

            # krg (non-wetting, depends on Sg)
            movable_gas_range = 1.0 - Sgr - Swc  # type: ignore
            valid_gas = movable_gas_range > minimum_mobile_pore_space
            se_g = np.clip(
                (sg - Sgr) / np.where(valid_gas, movable_gas_range, 1.0),
                0.0,
                1.0,
            )
            krg_vals = se_g**gas_exponent

            d_krg_d_sg_raw = np.where(
                valid_gas & (se_g > 0.0),
                gas_exponent * se_g ** max(gas_exponent - 1.0, 0.0) / movable_gas_range,
                zeros,
            )
            d_krg_d_sg = _apply_relperm_floor_to_derivative(
                d_krg_d_sg_raw, krg_vals, floor_g
            )
            d_krg_d_sw = zeros.copy()
            d_krg_d_so = zeros.copy()

            # krw (intermediate phase, via mixing rule)
            one_minus_kro = np.clip(1.0 - kro_vals, 0.0, None)
            one_minus_krg = np.clip(1.0 - krg_vals, 0.0, None)
            kro_proxy = one_minus_kro**water_exponent
            krg_proxy = one_minus_krg**water_exponent

            d_kro_proxy_d_so = np.where(
                one_minus_kro > 0.0,
                water_exponent
                * one_minus_kro ** max(water_exponent - 1.0, 0.0)
                * (-d_kro_d_so_raw),
                zeros,
            )
            d_kro_proxy_d_sw = -d_kro_proxy_d_so  # So = 1-Sw-Sg
            d_kro_proxy_d_sg = -d_kro_proxy_d_so

            d_krg_proxy_d_sg = np.where(
                one_minus_krg > 0.0,
                water_exponent
                * one_minus_krg ** max(water_exponent - 1.0, 0.0)
                * (-d_krg_d_sg_raw),
                zeros,
            )
            d_krg_proxy_d_sw = zeros.copy()
            d_krg_proxy_d_so = zeros.copy()

            derivatives = get_mixing_rule_partial_derivatives(
                rule=mixing_rule,
                kro_w=kro_proxy,
                kro_g=krg_proxy,  # type: ignore[arg-type]
                krw=kro_vals,
                krg=krg_vals,
                kro_endpoint=1.0,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                epsilon=c.FINITE_DIFFERENCE_EPSILON,
            )
            d_krw_d_kro_proxy = derivatives["d_kro_d_kro_w"]
            d_krw_d_krg_proxy = derivatives["d_kro_d_kro_g"]
            d_krw_d_sw_explicit = derivatives["d_kro_d_sw_explicit"]
            d_krw_d_so_explicit = derivatives["d_kro_d_so_explicit"]
            d_krw_d_sg_explicit = derivatives["d_kro_d_sg_explicit"]

            # Chain rule for krw
            krw_raw = mixing_rule(
                kro_w=kro_proxy,
                kro_g=krg_proxy,
                krw=kro_vals,
                krg=krg_vals,
                kro_endpoint=1.0,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
            )
            d_krw_d_sw_raw = (
                d_krw_d_kro_proxy * d_kro_proxy_d_sw
                + d_krw_d_krg_proxy * d_krg_proxy_d_sw
                + d_krw_d_sw_explicit
            )
            d_krw_d_so_raw = (
                d_krw_d_kro_proxy * d_kro_proxy_d_so
                + d_krw_d_krg_proxy * d_krg_proxy_d_so
                + d_krw_d_so_explicit
            )
            d_krw_d_sg_raw = (
                d_krw_d_kro_proxy * d_kro_proxy_d_sg
                + d_krw_d_krg_proxy * d_krg_proxy_d_sg
                + d_krw_d_sg_explicit
            )
            # Apply floor to krw derivatives
            d_krw_d_sw = _apply_relperm_floor_to_derivative(
                d_krw_d_sw_raw, krw_raw, floor_w
            )
            d_krw_d_so = _apply_relperm_floor_to_derivative(
                d_krw_d_so_raw, krw_raw, floor_w
            )
            d_krw_d_sg = _apply_relperm_floor_to_derivative(
                d_krw_d_sg_raw, krw_raw, floor_w
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
                results = tuple(r.item() for r in results)
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

        if wettability == Wettability.MIXED_WET:
            f = self.mixed_wet_water_fraction

            # Water-wet sub-system
            mobile_water_range_ww = 1.0 - Swc - Sorw  # type: ignore[operator]
            valid_water_ww = mobile_water_range_ww > minimum_mobile_pore_space
            se_w_ww = np.clip(
                (sw - Swc) / np.where(valid_water_ww, mobile_water_range_ww, 1.0),
                0.0,
                1.0,
            )
            krw_ww = se_w_ww**water_exponent
            d_krw_ww_d_sw_raw = np.where(
                valid_water_ww & (se_w_ww > 0.0),
                water_exponent
                * (se_w_ww ** max(water_exponent - 1.0, 0.0))
                / mobile_water_range_ww,
                zeros,
            )

            mobile_gas_range_ww = 1.0 - Swc - Sgr - Sorg  # type: ignore[operator]
            valid_gas_ww = mobile_gas_range_ww > minimum_mobile_pore_space
            se_g_ww = np.clip(
                (sg - Sgr) / np.where(valid_gas_ww, mobile_gas_range_ww, 1.0),
                0.0,
                1.0,
            )
            krg_ww = se_g_ww**gas_exponent
            d_krg_ww_d_sg_raw = np.where(
                valid_gas_ww & (se_g_ww > 0.0),
                gas_exponent
                * (se_g_ww ** max(gas_exponent - 1.0, 0.0))
                / mobile_gas_range_ww,
                zeros,
            )

            one_minus_krw_ww = np.clip(1.0 - krw_ww, 0.0, None)
            kro_w_ww = one_minus_krw_ww**oil_exponent
            d_kro_w_ww_d_sw = np.where(
                one_minus_krw_ww > 0.0,
                oil_exponent
                * (one_minus_krw_ww ** max(oil_exponent - 1.0, 0.0))
                * (-d_krw_ww_d_sw_raw),
                zeros,
            )

            one_minus_krg_ww = np.clip(1.0 - krg_ww, 0.0, None)
            kro_g_ww = one_minus_krg_ww**oil_exponent
            d_kro_g_ww_d_sg = np.where(
                one_minus_krg_ww > 0.0,
                oil_exponent
                * (one_minus_krg_ww ** max(oil_exponent - 1.0, 0.0))
                * (-d_krg_ww_d_sg_raw),
                zeros,
            )

            derivs_ww = get_mixing_rule_partial_derivatives(
                rule=mixing_rule,
                kro_w=kro_w_ww,  # type: ignore[arg-type]
                kro_g=kro_g_ww,  # type: ignore[arg-type]
                krw=krw_ww,
                krg=krg_ww,
                kro_endpoint=1.0,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                epsilon=c.FINITE_DIFFERENCE_EPSILON,
            )
            kro_ww_raw = mixing_rule(
                kro_w=kro_w_ww,
                kro_g=kro_g_ww,
                krw=krw_ww,
                krg=krg_ww,
                kro_endpoint=1.0,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
            )
            d_kro_ww_d_sw = (
                derivs_ww["d_kro_d_kro_w"] * d_kro_w_ww_d_sw
                + derivs_ww["d_kro_d_kro_g"] * zeros  # kro_g_ww has no Sw dependence
                + derivs_ww["d_kro_d_sw_explicit"]
            )
            d_kro_ww_d_so = derivs_ww[
                "d_kro_d_so_explicit"
            ]  # neither shaped input depends on So
            d_kro_ww_d_sg = (
                derivs_ww["d_kro_d_kro_w"] * zeros  # kro_w_ww has no Sg dependence
                + derivs_ww["d_kro_d_kro_g"] * d_kro_g_ww_d_sg
                + derivs_ww["d_kro_d_sg_explicit"]
            )

            # Oil-wet sub-system
            movable_oil_range_ow = 1.0 - Sorw - Sorg  # type: ignore[operator]
            max_residual_ow = np.minimum(Sorw, Sorg)  # type: ignore[operator]
            valid_oil_ow = movable_oil_range_ow > minimum_mobile_pore_space
            se_o_ow = np.clip(
                (so - max_residual_ow)
                / np.where(valid_oil_ow, movable_oil_range_ow, 1.0),
                0.0,
                1.0,
            )
            kro_ow = se_o_ow**oil_exponent
            d_kro_ow_d_so = np.where(
                valid_oil_ow & (se_o_ow > 0.0),
                oil_exponent
                * (se_o_ow ** max(oil_exponent - 1.0, 0.0))
                / movable_oil_range_ow,
                zeros,
            )

            movable_gas_range_ow = 1.0 - Sgr - Swc  # type: ignore[operator]
            valid_gas_ow = movable_gas_range_ow > minimum_mobile_pore_space
            se_g_ow = np.clip(
                (sg - Sgr) / np.where(valid_gas_ow, movable_gas_range_ow, 1.0),
                0.0,
                1.0,
            )
            krg_ow = se_g_ow**gas_exponent
            d_krg_ow_d_sg = np.where(
                valid_gas_ow & (se_g_ow > 0.0),
                gas_exponent
                * (se_g_ow ** max(gas_exponent - 1.0, 0.0))
                / movable_gas_range_ow,
                zeros,
            )

            one_minus_kro_ow = np.clip(1.0 - kro_ow, 0.0, None)
            kro_proxy_ow = one_minus_kro_ow**water_exponent
            d_kro_proxy_ow_d_so = np.where(
                one_minus_kro_ow > 0.0,
                water_exponent
                * (one_minus_kro_ow ** max(water_exponent - 1.0, 0.0))
                * (-d_kro_ow_d_so),
                zeros,
            )

            one_minus_krg_ow = np.clip(1.0 - krg_ow, 0.0, None)
            krg_proxy_ow = one_minus_krg_ow**water_exponent
            d_krg_proxy_ow_d_sg = np.where(
                one_minus_krg_ow > 0.0,
                water_exponent
                * (one_minus_krg_ow ** max(water_exponent - 1.0, 0.0))
                * (-d_krg_ow_d_sg),
                zeros,
            )

            derivs_ow = get_mixing_rule_partial_derivatives(
                rule=mixing_rule,
                kro_w=kro_proxy_ow,
                kro_g=krg_proxy_ow,  # type: ignore[arg-type]
                krw=kro_ow,
                krg=krg_ow,
                kro_endpoint=1.0,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                epsilon=c.FINITE_DIFFERENCE_EPSILON,
            )
            krw_ow_raw = mixing_rule(
                kro_w=kro_proxy_ow,
                kro_g=krg_proxy_ow,
                krw=kro_ow,
                krg=krg_ow,
                kro_endpoint=1.0,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
            )
            d_krw_ow_d_sw = derivs_ow["d_kro_d_sw_explicit"]
            d_krw_ow_d_so = (
                derivs_ow["d_kro_d_kro_w"] * d_kro_proxy_ow_d_so
                + derivs_ow["d_kro_d_so_explicit"]
            )
            d_krw_ow_d_sg = (
                derivs_ow["d_kro_d_kro_g"] * d_krg_proxy_ow_d_sg
                + derivs_ow["d_kro_d_sg_explicit"]
            )

            # Blend raw values for floor masking
            krw_blend_raw = f * krw_ww + (1.0 - f) * krw_ow_raw
            kro_blend_raw = f * kro_ww_raw + (1.0 - f) * kro_ow
            krg_blend_raw = f * krg_ww + (1.0 - f) * krg_ow

            # Blend derivatives
            d_krw_d_sw_raw = f * d_krw_ww_d_sw_raw + (1.0 - f) * d_krw_ow_d_sw
            d_krw_d_so_raw = f * zeros + (1.0 - f) * d_krw_ow_d_so
            d_krw_d_sg_raw = f * zeros + (1.0 - f) * d_krw_ow_d_sg

            d_kro_d_sw_raw = f * d_kro_ww_d_sw + (1.0 - f) * zeros
            d_kro_d_so_raw = f * d_kro_ww_d_so + (1.0 - f) * d_kro_ow_d_so
            d_kro_d_sg_raw = f * d_kro_ww_d_sg + (1.0 - f) * zeros

            d_krg_d_sw_raw = zeros.copy()
            d_krg_d_so_raw = zeros.copy()
            d_krg_d_sg_raw = f * d_krg_ww_d_sg_raw + (1.0 - f) * d_krg_ow_d_sg

            # Apply floors to blended derivatives
            d_krw_d_sw = _apply_relperm_floor_to_derivative(
                d_krw_d_sw_raw, krw_blend_raw, floor_w
            )
            d_krw_d_so = _apply_relperm_floor_to_derivative(
                d_krw_d_so_raw, krw_blend_raw, floor_w
            )
            d_krw_d_sg = _apply_relperm_floor_to_derivative(
                d_krw_d_sg_raw, krw_blend_raw, floor_w
            )
            d_kro_d_sw = _apply_relperm_floor_to_derivative(
                d_kro_d_sw_raw, kro_blend_raw, floor_o
            )
            d_kro_d_so = _apply_relperm_floor_to_derivative(
                d_kro_d_so_raw, kro_blend_raw, floor_o
            )
            d_kro_d_sg = _apply_relperm_floor_to_derivative(
                d_kro_d_sg_raw, kro_blend_raw, floor_o
            )
            d_krg_d_sw = zeros.copy()
            d_krg_d_so = zeros.copy()
            d_krg_d_sg = _apply_relperm_floor_to_derivative(
                d_krg_d_sg_raw, krg_blend_raw, floor_g
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
                results = tuple(r.item() for r in results)
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
        d_krw_d_sw_raw = np.where(
            valid_water & (se_w > 0.0),
            water_exponent
            * (se_w ** max(water_exponent - 1.0, 0.0))
            / mobile_water_range,
            zeros,
        )
        d_krw_d_sw = _apply_relperm_floor_to_derivative(
            d_krw_d_sw_raw, krw_values, floor_w
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
        d_krg_d_sg_raw = np.where(
            valid_gas & (se_g > 0.0),
            gas_exponent * (se_g ** max(gas_exponent - 1.0, 0.0)) / mobile_gas_range,
            zeros,
        )
        d_krg_d_sg = _apply_relperm_floor_to_derivative(
            d_krg_d_sg_raw, krg_values, floor_g
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
            * (-d_krw_d_sw_raw),
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
            * (-d_krg_d_sg_raw),
            zeros,
        )
        d_kro_g_d_sw = zeros.copy()
        d_kro_g_d_so = zeros.copy()

        derivatives = get_mixing_rule_partial_derivatives(
            rule=mixing_rule,
            kro_w=kro_w_shaped,  # type: ignore[arg-type]
            kro_g=kro_g_shaped,  # type: ignore[arg-type]
            krw=krw_values,
            krg=krg_values,
            kro_endpoint=1.0,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            epsilon=c.FINITE_DIFFERENCE_EPSILON,
        )
        d_kro_d_kro_w = derivatives["d_kro_d_kro_w"]
        d_kro_d_kro_g = derivatives["d_kro_d_kro_g"]
        d_kro_d_krw_mix = derivatives["d_kro_d_krw"]
        d_kro_d_krg_mix = derivatives["d_kro_d_krg"]
        d_kro_d_water_saturation_explicit = derivatives["d_kro_d_sw_explicit"]
        d_kro_d_oil_saturation_explicit = derivatives["d_kro_d_so_explicit"]
        d_kro_d_gas_saturation_explicit = derivatives["d_kro_d_sg_explicit"]

        # Forward evaluate kro for floor masking
        kro_raw = mixing_rule(
            kro_w=kro_w_shaped,
            kro_g=kro_g_shaped,
            krw=krw_values,
            krg=krg_values,
            kro_endpoint=1.0,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
        )

        d_kro_d_sw_raw = (
            d_kro_d_kro_w * d_kro_w_d_sw
            + d_kro_d_kro_g * d_kro_g_d_sw
            + d_kro_d_krw_mix * d_krw_d_sw_raw
            + d_kro_d_krg_mix * zeros
            + d_kro_d_water_saturation_explicit
        )
        d_kro_d_so_raw = (
            d_kro_d_kro_w * d_kro_w_d_so
            + d_kro_d_kro_g * d_kro_g_d_so
            + d_kro_d_krw_mix * zeros
            + d_kro_d_krg_mix * zeros
            + d_kro_d_oil_saturation_explicit
        )
        d_kro_d_sg_raw = (
            d_kro_d_kro_w * d_kro_w_d_sg
            + d_kro_d_kro_g * d_kro_g_d_sg
            + d_kro_d_krw_mix * zeros
            + d_kro_d_krg_mix * d_krg_d_sg_raw
            + d_kro_d_gas_saturation_explicit
        )

        d_kro_d_sw = _apply_relperm_floor_to_derivative(
            d_kro_d_sw_raw, kro_raw, floor_o
        )
        d_kro_d_so = _apply_relperm_floor_to_derivative(
            d_kro_d_so_raw, kro_raw, floor_o
        )
        d_kro_d_sg = _apply_relperm_floor_to_derivative(
            d_kro_d_sg_raw, kro_raw, floor_o
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
            results = tuple(r.item() for r in results)
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


@numba.njit(cache=True)
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
    irreducible_water_saturation: FloatOrArray,
    residual_oil_saturation_water: FloatOrArray,
    residual_oil_saturation_gas: FloatOrArray,
    residual_gas_saturation: FloatOrArray,
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
    mixed_wet_water_fraction: float = 0.5,
    mixing_rule: MixingRule = eclipse_rule,
    saturation_epsilon: float = 1e-6,
    minimum_mobile_pore_space: float = 1e-9,
    min_water_relperm: typing.Optional[float] = None,
    min_oil_relperm: typing.Optional[float] = None,
    min_gas_relperm: typing.Optional[float] = None,
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
    :param mixed_wet_water_fraction: Fraction of water-wet behavior in mixed-wet case (0 to 1).
    :param mixing_rule: Three-phase mixing rule for oil relative permeability.
    :param min_water_relperm: Resolved minimum floor for water kr (`None` = no floor).
    :param min_oil_relperm: Resolved minimum floor for oil kr (`None` = no floor).
    :param min_gas_relperm: Resolved minimum floor for gas kr (`None` = no floor).
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

    # kro_endpoint for LET models: max_oil_relperm
    kro_endpoint = max_oil_relperm

    if wettability == Wettability.WATER_WET:
        # Water kr (wetting phase)
        movable_water_range = 1.0 - Swc - Sorw  # type: ignore[operator]
        sw_star = np.where(
            movable_water_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - Swc) / movable_water_range, 0.0, 1.0),
        )
        krw = max_water_relperm * _let_relperm(sw_star, water_L, water_E, water_T)

        # Gas kr (non-wetting phase)
        movable_gas_range = 1.0 - Swc - Sgr - Sorg  # type: ignore[operator]
        sg_star = np.where(
            movable_gas_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - Sgr) / movable_gas_range, 0.0, 1.0),
        )
        krg = max_gas_relperm * _let_relperm(sg_star, gas_L, gas_E, gas_T)

        # Oil kr (intermediate phase, three-phase mixing)
        movable_oil_water_range = 1.0 - Swc - Sorw  # type: ignore[operator]
        so_star_w = np.where(
            movable_oil_water_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - Sorw) / movable_oil_water_range, 0.0, 1.0),
        )
        kro_w = _let_relperm(so_star_w, oil_water_L, oil_water_E, oil_water_T)

        movable_gas_oil_range = 1.0 - Swc - Sorg - Sgr  # type: ignore[operator]
        so_star_g = np.where(
            movable_gas_oil_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - Sorg) / movable_gas_oil_range, 0.0, 1.0),
        )
        kro_g = _let_relperm(so_star_g, gas_oil_L, gas_oil_E, gas_oil_T)

        kro_mixed = mixing_rule(
            kro_w=kro_w,
            kro_g=kro_g,
            krw=krw,
            krg=krg,
            kro_endpoint=kro_endpoint,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
        )
        kro = max_oil_relperm * np.clip(kro_mixed, 0.0, 1.0)

    elif wettability == Wettability.OIL_WET:
        # Oil is wetting, water becomes intermediate
        movable_oil_range = 1.0 - Sorw - Sorg  # type: ignore[operator]
        max_residual = np.minimum(Sorw, Sorg)
        so_star = np.where(
            movable_oil_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - max_residual) / movable_oil_range, 0.0, 1.0),
        )
        kro = max_oil_relperm * _let_relperm(
            so_star, oil_water_L, oil_water_E, oil_water_T
        )

        movable_gas_range = 1.0 - Sgr - Swc  # type: ignore[operator]
        sg_star = np.where(
            movable_gas_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - Sgr) / movable_gas_range, 0.0, 1.0),
        )
        krg = max_gas_relperm * _let_relperm(sg_star, gas_L, gas_E, gas_T)

        movable_water_range_ow = 1.0 - Swc - Sorw  # type: ignore[operator]
        sw_star_ow = np.where(
            movable_water_range_ow <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - Swc) / movable_water_range_ow, 0.0, 1.0),
        )
        krw_ow = _let_relperm(sw_star_ow, water_L, water_E, water_T)

        movable_water_range_gw = 1.0 - Swc - Sgr  # type: ignore[operator]
        sw_star_gw = np.where(
            movable_water_range_gw <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - Swc) / movable_water_range_gw, 0.0, 1.0),
        )
        krw_gw = _let_relperm(sw_star_gw, water_L, water_E, water_T)

        krw_mixed = mixing_rule(
            kro_w=krw_ow,
            kro_g=krw_gw,
            krw=kro,
            krg=krg,
            kro_endpoint=kro_endpoint,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
        )
        krw = max_water_relperm * np.clip(krw_mixed, 0.0, 1.0)

    elif wettability == Wettability.MIXED_WET:
        # Water-wet sub-system
        movable_water_range = 1.0 - Swc - Sorw  # type: ignore[operator]
        sw_star_ww = np.where(
            movable_water_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - Swc) / movable_water_range, 0.0, 1.0),
        )
        krw_ww = max_water_relperm * _let_relperm(sw_star_ww, water_L, water_E, water_T)

        movable_gas_range = 1.0 - Swc - Sgr - Sorg  # type: ignore[operator]
        sg_star_ww = np.where(
            movable_gas_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - Sgr) / movable_gas_range, 0.0, 1.0),
        )
        krg_ww = max_gas_relperm * _let_relperm(sg_star_ww, gas_L, gas_E, gas_T)

        movable_oil_water_range = 1.0 - Swc - Sorw  # type: ignore[operator]
        so_star_w_ww = np.where(
            movable_oil_water_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - Sorw) / movable_oil_water_range, 0.0, 1.0),
        )
        kro_w_ww = _let_relperm(so_star_w_ww, oil_water_L, oil_water_E, oil_water_T)

        movable_gas_oil_range = 1.0 - Swc - Sorg - Sgr  # type: ignore[operator]
        so_star_g_ww = np.where(
            movable_gas_oil_range <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - Sorg) / movable_gas_oil_range, 0.0, 1.0),
        )
        kro_g_ww = _let_relperm(so_star_g_ww, gas_oil_L, gas_oil_E, gas_oil_T)

        kro_ww = max_oil_relperm * np.clip(
            mixing_rule(
                kro_w=kro_w_ww,
                kro_g=kro_g_ww,
                krw=krw_ww,
                krg=krg_ww,
                kro_endpoint=kro_endpoint,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
            ),
            0.0,
            1.0,
        )

        # Oil-wet sub-system
        movable_oil_range_ow = 1.0 - Sorw - Sorg  # type: ignore[operator]
        max_residual_ow = np.minimum(Sorw, Sorg)
        so_star_ow = np.where(
            movable_oil_range_ow <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(so),
            np.clip((so - max_residual_ow) / movable_oil_range_ow, 0.0, 1.0),
        )
        kro_ow = max_oil_relperm * _let_relperm(
            so_star_ow, oil_water_L, oil_water_E, oil_water_T
        )

        movable_gas_range_ow = 1.0 - Sgr - Swc  # type: ignore[operator]
        sg_star_ow = np.where(
            movable_gas_range_ow <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sg),
            np.clip((sg - Sgr) / movable_gas_range_ow, 0.0, 1.0),
        )
        krg_ow = max_gas_relperm * _let_relperm(sg_star_ow, gas_L, gas_E, gas_T)

        movable_water_range_ow = 1.0 - Swc - Sorw  # type: ignore[operator]
        sw_star_ow = np.where(
            movable_water_range_ow <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - Swc) / movable_water_range_ow, 0.0, 1.0),
        )
        krw_ow_proxy = _let_relperm(sw_star_ow, water_L, water_E, water_T)

        movable_water_range_gw = 1.0 - Swc - Sgr  # type: ignore[operator]
        sw_star_gw = np.where(
            movable_water_range_gw <= minimum_mobile_pore_space,  # type: ignore[operator]
            np.zeros_like(sw),
            np.clip((sw - Swc) / movable_water_range_gw, 0.0, 1.0),
        )
        krw_gw_proxy = _let_relperm(sw_star_gw, water_L, water_E, water_T)

        krw_ow = max_water_relperm * np.clip(
            mixing_rule(
                kro_w=krw_ow_proxy,
                kro_g=krw_gw_proxy,
                krw=kro_ow,
                krg=krg_ow,
                kro_endpoint=kro_endpoint,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
            ),
            0.0,
            1.0,
        )

        # Blend
        krw = (
            mixed_wet_water_fraction * krw_ww
            + (1.0 - mixed_wet_water_fraction) * krw_ow
        )
        kro = (
            mixed_wet_water_fraction * kro_ww
            + (1.0 - mixed_wet_water_fraction) * kro_ow
        )
        krg = (
            mixed_wet_water_fraction * krg_ww
            + (1.0 - mixed_wet_water_fraction) * krg_ow
        )

    else:
        raise ValidationError(f"Wettability {wettability!r} not implemented.")

    krw = np.clip(krw, 0.0, 1.0)
    kro = np.clip(kro, 0.0, 1.0)
    krg = np.clip(krg, 0.0, 1.0)
    krw = _apply_relperm_floor(krw, min_water_relperm)
    kro = _apply_relperm_floor(kro, min_oil_relperm)
    krg = _apply_relperm_floor(krg, min_gas_relperm)
    if is_scalar:
        krw = krw.item()  # type: ignore
        kro = kro.item()  # type: ignore
        krg = krg.item()  # type: ignore
    return krw, kro, krg  # type: ignore[return-value]


@numba.njit(cache=True)
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

    **Minimum relperm floors** (``min_water_relperm``, ``min_oil_relperm``,
    ``min_gas_relperm``): same semantics as ``BrooksCoreyRelPermModel``.
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
    mixed_wet_water_fraction: float = 0.5
    """Fraction of pore space that is water-wet in mixed-wet systems (0-1)."""

    mixing_rule: typing.Union[MixingRule, str] = eclipse_rule
    """
    Mixing rule function or name to compute oil relative permeability in
    three-phase system. Accepts a function or a registered name string.
    """

    min_water_relperm: RelPermFloor = "auto"
    """
    Minimum floor for the water relative permeability.

    `"auto"` - dtype-aware floor; `None` - no floor; `float` - explicit value.
    """

    min_oil_relperm: RelPermFloor = "auto"
    """
    Minimum floor for the oil relative permeability.

    `"auto"` - dtype-aware floor; `None` - no floor; `float` - explicit value.
    """

    min_gas_relperm: RelPermFloor = "auto"
    """
    Minimum floor for the gas relative permeability.

    `"auto"` - dtype-aware floor; `None` - no floor; `float` - explicit value.
    """

    supports_arrays: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array inputs."""

    def __attrs_post_init__(self) -> None:
        mixing_rule = self.mixing_rule
        if isinstance(mixing_rule, str):
            object.__setattr__(self, "mixing_rule", get_mixing_rule(mixing_rule))
        # Validate floor sentinels eagerly
        _resolve_relperm_floor(self.min_water_relperm)
        _resolve_relperm_floor(self.min_oil_relperm)
        _resolve_relperm_floor(self.min_gas_relperm)

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

    def get_relative_permeabilities(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        irreducible_water_saturation: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_water: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_gas: typing.Optional[FloatOrArray] = None,
        residual_gas_saturation: typing.Optional[FloatOrArray] = None,
        **kwargs: typing.Any,
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
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
            mixing_rule=self.mixing_rule,  # type: ignore[arg-type]
            saturation_epsilon=c.SATURATION_EPSILON,
            minimum_mobile_pore_space=c.MINIMUM_MOBILE_PORE_SPACE,
            min_water_relperm=_resolve_relperm_floor(self.min_water_relperm),
            min_oil_relperm=_resolve_relperm_floor(self.min_oil_relperm),
            min_gas_relperm=_resolve_relperm_floor(self.min_gas_relperm),
        )
        return RelativePermeabilities(water=krw, oil=kro, gas=krg)  # type: ignore[typeddict-item]

    def get_relative_permeability_derivatives(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        irreducible_water_saturation: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_water: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_gas: typing.Optional[FloatOrArray] = None,
        residual_gas_saturation: typing.Optional[FloatOrArray] = None,
        **kwargs: typing.Any,
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
        by the extended chain rule through the mixing rule (including the
        ``d_kro/d_krw`` and ``d_kro/d_krg`` terms for rules like Stone II that
        use the actual two-phase water/gas kr values).

        Wherever a minimum relperm floor is active (raw kr ≤ floor), the
        corresponding derivative is zeroed out, keeping the Jacobian consistent
        with the floored kr value and preventing MBE.

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

        # Resolve floors once up front
        floor_w = _resolve_relperm_floor(self.min_water_relperm)
        floor_o = _resolve_relperm_floor(self.min_oil_relperm)
        floor_g = _resolve_relperm_floor(self.min_gas_relperm)

        # kro_endpoint for mixing rule calls: max_oil_relperm for LET
        kro_endpoint = kro_max

        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
            and np.isscalar(Swc)
            and np.isscalar(Sorw)
            and np.isscalar(Sorg)
            and np.isscalar(Sgr)
        )
        sw = np.atleast_1d(water_saturation)
        so = np.atleast_1d(oil_saturation)
        sg = np.atleast_1d(gas_saturation)
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        zeros = np.zeros_like(sw)
        minimum_mobile_pore_space = c.MINIMUM_MOBILE_PORE_SPACE

        if wettability == Wettability.OIL_WET:
            # kro (wetting, depends on So)
            movable_oil_range = 1.0 - Sorw - Sorg  # type: ignore[operator]
            max_residual = np.minimum(Sorw, Sorg)  # type: ignore
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

            d_kro_d_so_raw = np.where(
                valid_oil,
                _let_curve_slope_wrt_normalized_saturation(
                    se_o,
                    oil_water_params.L,
                    oil_water_params.E,
                    oil_water_params.T,
                    kro_max,
                )
                / movable_oil_range,
                zeros,
            )
            d_kro_d_so = _apply_relperm_floor_to_derivative(
                d_kro_d_so_raw, kro_raw, floor_o
            )
            d_kro_d_sw = zeros.copy()
            d_kro_d_sg = zeros.copy()

            # krg (non-wetting, depends on Sg)
            movable_gas_range = 1.0 - Sgr - Swc  # type: ignore[operator]
            valid_gas = movable_gas_range > minimum_mobile_pore_space
            se_g = np.clip(
                (sg - Sgr) / np.where(valid_gas, movable_gas_range, 1.0),
                0.0,
                1.0,
            )
            krg_raw = krg_max * _let_relperm(
                se_g,  # type: ignore[arg-type]
                gas_params.L,
                gas_params.E,
                gas_params.T,
            )

            d_krg_d_sg_raw = np.where(
                valid_gas,
                _let_curve_slope_wrt_normalized_saturation(
                    se_g, gas_params.L, gas_params.E, gas_params.T, krg_max
                )
                / movable_gas_range,
                zeros,
            )
            d_krg_d_sg = _apply_relperm_floor_to_derivative(
                d_krg_d_sg_raw, krg_raw, floor_g
            )
            d_krg_d_sw = zeros.copy()
            d_krg_d_so = zeros.copy()

            # krw (intermediate phase, via mixing rule on two-phase water proxies)
            movable_water_range_ow = 1.0 - Swc - Sorw  # type: ignore[operator]
            valid_water_ow = movable_water_range_ow > minimum_mobile_pore_space
            se_w_ow = np.clip(
                (sw - Swc) / np.where(valid_water_ow, movable_water_range_ow, 1.0),
                0.0,
                1.0,
            )
            krw_ow = _let_relperm(
                se_w_ow,  # type: ignore[arg-type]
                water_params.L,
                water_params.E,
                water_params.T,
            )

            movable_water_range_gw = 1.0 - Swc - Sgr  # type: ignore[operator]
            valid_water_gw = movable_water_range_gw > minimum_mobile_pore_space
            se_w_gw = np.clip(
                (sw - Swc) / np.where(valid_water_gw, movable_water_range_gw, 1.0),
                0.0,
                1.0,
            )
            krw_gw = _let_relperm(
                se_w_gw,  # type: ignore[arg-type]
                water_params.L,
                water_params.E,
                water_params.T,
            )

            # d(krw_ow)/dSw — depends only on Sw
            d_krw_ow_d_sw = np.where(
                valid_water_ow,
                _let_curve_slope_wrt_normalized_saturation(
                    se_w_ow, water_params.L, water_params.E, water_params.T, 1.0
                )
                / movable_water_range_ow,
                zeros,
            )
            # d(krw_gw)/dSw — depends only on Sw
            d_krw_gw_d_sw = np.where(
                valid_water_gw,
                _let_curve_slope_wrt_normalized_saturation(
                    se_w_gw, water_params.L, water_params.E, water_params.T, 1.0
                )
                / movable_water_range_gw,
                zeros,
            )
            # krw_ow and krw_gw have no So or Sg dependence
            d_krw_ow_d_so = zeros.copy()
            d_krw_ow_d_sg = zeros.copy()
            d_krw_gw_d_so = zeros.copy()
            d_krw_gw_d_sg = zeros.copy()

            derivatives = get_mixing_rule_partial_derivatives(
                rule=mixing_rule,
                kro_w=krw_ow,
                kro_g=krw_gw,
                krw=kro_raw,
                krg=krg_raw,
                kro_endpoint=kro_endpoint,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                epsilon=c.FINITE_DIFFERENCE_EPSILON,
            )
            d_krw_d_krw_ow = derivatives["d_kro_d_kro_w"]
            d_krw_d_krw_gw = derivatives["d_kro_d_kro_g"]
            d_krw_d_sw_explicit = derivatives["d_kro_d_sw_explicit"]
            d_krw_d_so_explicit = derivatives["d_kro_d_so_explicit"]
            d_krw_d_sg_explicit = derivatives["d_kro_d_sg_explicit"]

            # Forward evaluate krw for floor masking
            krw_raw = mixing_rule(
                kro_w=krw_ow,
                kro_g=krw_gw,
                krw=kro_raw,
                krg=krg_raw,
                kro_endpoint=kro_endpoint,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
            )

            d_krw_d_sw_raw = krw_max * (
                d_krw_d_krw_ow * d_krw_ow_d_sw
                + d_krw_d_krw_gw * d_krw_gw_d_sw
                + d_krw_d_sw_explicit
            )
            d_krw_d_so_raw = krw_max * (
                d_krw_d_krw_ow * d_krw_ow_d_so
                + d_krw_d_krw_gw * d_krw_gw_d_so
                + d_krw_d_so_explicit
            )
            d_krw_d_sg_raw = krw_max * (
                d_krw_d_krw_ow * d_krw_ow_d_sg
                + d_krw_d_krw_gw * d_krw_gw_d_sg
                + d_krw_d_sg_explicit
            )
            d_krw_d_sw = _apply_relperm_floor_to_derivative(
                d_krw_d_sw_raw, krw_raw, floor_w
            )
            d_krw_d_so = _apply_relperm_floor_to_derivative(
                d_krw_d_so_raw, krw_raw, floor_w
            )
            d_krw_d_sg = _apply_relperm_floor_to_derivative(
                d_krw_d_sg_raw, krw_raw, floor_w
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
                results = tuple(r.item() for r in results)
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

        if wettability == Wettability.MIXED_WET:
            f = self.mixed_wet_water_fraction

            # Water-wet sub-system
            mobile_water_range = 1.0 - Swc - Sorw  # type: ignore[operator]
            valid_water = mobile_water_range > minimum_mobile_pore_space
            se_w = np.clip(
                (sw - Swc) / np.where(valid_water, mobile_water_range, 1.0), 0.0, 1.0
            )
            krw_ww = krw_max * _let_relperm(
                se_w,  # type: ignore[arg-type]
                water_params.L,
                water_params.E,
                water_params.T,
            )
            d_krw_ww_d_sw_raw = np.where(
                valid_water,
                _let_curve_slope_wrt_normalized_saturation(
                    se_w, water_params.L, water_params.E, water_params.T, krw_max
                )
                / mobile_water_range,
                zeros,
            )

            mobile_gas_range = 1.0 - Swc - Sgr - Sorg  # type: ignore[operator]
            valid_gas = mobile_gas_range > minimum_mobile_pore_space
            se_g = np.clip(
                (sg - Sgr) / np.where(valid_gas, mobile_gas_range, 1.0), 0.0, 1.0
            )
            krg_ww = krg_max * _let_relperm(
                se_g,  # type: ignore[arg-type]
                gas_params.L,
                gas_params.E,
                gas_params.T,
            )
            d_krg_ww_d_sg_raw = np.where(
                valid_gas,
                _let_curve_slope_wrt_normalized_saturation(
                    se_g, gas_params.L, gas_params.E, gas_params.T, krg_max
                )
                / mobile_gas_range,
                zeros,
            )

            # Two-phase oil inputs (unit-endpoint) and their So-derivatives
            mobile_oil_water_range = 1.0 - Swc - Sorw  # type: ignore[operator]
            valid_ow = mobile_oil_water_range > minimum_mobile_pore_space
            se_o_w = np.clip(
                (so - Sorw) / np.where(valid_ow, mobile_oil_water_range, 1.0), 0.0, 1.0
            )
            kro_w_ww = _let_relperm(
                se_o_w,  # type: ignore[arg-type]
                oil_water_params.L,
                oil_water_params.E,
                oil_water_params.T,
            )
            d_kro_w_ww_d_so = np.where(
                valid_ow,
                _let_curve_slope_wrt_normalized_saturation(
                    se_o_w,
                    oil_water_params.L,
                    oil_water_params.E,
                    oil_water_params.T,
                    1.0,
                )
                / mobile_oil_water_range,
                zeros,
            )

            mobile_gas_oil_range = 1.0 - Swc - Sorg - Sgr  # type: ignore[operator]
            valid_go = mobile_gas_oil_range > minimum_mobile_pore_space
            se_o_g = np.clip(
                (so - Sorg) / np.where(valid_go, mobile_gas_oil_range, 1.0), 0.0, 1.0
            )
            kro_g_ww = _let_relperm(
                se_o_g,  # type: ignore[arg-type]
                gas_oil_params.L,
                gas_oil_params.E,
                gas_oil_params.T,
            )
            d_kro_g_ww_d_so = np.where(
                valid_go,
                _let_curve_slope_wrt_normalized_saturation(
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
                kro_endpoint=kro_endpoint,
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
                    kro_endpoint=kro_endpoint,
                    water_saturation=sw,
                    oil_saturation=so,
                    gas_saturation=sg,
                ),
                0.0,
                1.0,
            )
            d_kro_ww_d_sw = kro_max * derivs_ww["d_kro_d_sw_explicit"]
            d_kro_ww_d_so = kro_max * (
                derivs_ww["d_kro_d_kro_w"] * d_kro_w_ww_d_so
                + derivs_ww["d_kro_d_kro_g"] * d_kro_g_ww_d_so
                + derivs_ww["d_kro_d_so_explicit"]
            )
            d_kro_ww_d_sg = kro_max * derivs_ww["d_kro_d_sg_explicit"]

            # Oil-wet sub-system
            movable_oil_range_ow = 1.0 - Sorw - Sorg  # type: ignore[operator]
            max_residual_ow = np.minimum(Sorw, Sorg)  # type: ignore[operator]
            valid_oil_ow = movable_oil_range_ow > minimum_mobile_pore_space
            se_o_ow = np.clip(
                (so - max_residual_ow)
                / np.where(valid_oil_ow, movable_oil_range_ow, 1.0),
                0.0,
                1.0,
            )
            kro_ow = kro_max * _let_relperm(
                se_o_ow,  # type: ignore[arg-type]
                oil_water_params.L,
                oil_water_params.E,
                oil_water_params.T,
            )
            d_kro_ow_d_so = np.where(
                valid_oil_ow,
                kro_max
                * _let_curve_slope_wrt_normalized_saturation(
                    se_o_ow,
                    oil_water_params.L,
                    oil_water_params.E,
                    oil_water_params.T,
                    1.0,
                )
                / movable_oil_range_ow,
                zeros,
            )

            movable_gas_range_ow = 1.0 - Sgr - Swc  # type: ignore[operator]
            valid_gas_ow = movable_gas_range_ow > minimum_mobile_pore_space
            se_g_ow = np.clip(
                (sg - Sgr) / np.where(valid_gas_ow, movable_gas_range_ow, 1.0), 0.0, 1.0
            )
            krg_ow = krg_max * _let_relperm(
                se_g_ow,  # type: ignore[arg-type]
                gas_params.L,
                gas_params.E,
                gas_params.T,
            )
            d_krg_ow_d_sg = np.where(
                valid_gas_ow,
                krg_max
                * _let_curve_slope_wrt_normalized_saturation(
                    se_g_ow, gas_params.L, gas_params.E, gas_params.T, 1.0
                )
                / movable_gas_range_ow,
                zeros,
            )

            # krw_ow proxies and their Sw-derivatives
            movable_water_range_ow = 1.0 - Swc - Sorw  # type: ignore[operator]
            valid_w_ow = movable_water_range_ow > minimum_mobile_pore_space
            se_w_ow = np.clip(
                (sw - Swc) / np.where(valid_w_ow, movable_water_range_ow, 1.0), 0.0, 1.0
            )
            krw_ow_proxy = _let_relperm(
                se_w_ow,  # type: ignore[arg-type]
                water_params.L,
                water_params.E,
                water_params.T,
            )
            d_krw_ow_proxy_d_sw = np.where(
                valid_w_ow,
                _let_curve_slope_wrt_normalized_saturation(
                    se_w_ow, water_params.L, water_params.E, water_params.T, 1.0
                )
                / movable_water_range_ow,
                zeros,
            )

            movable_water_range_gw = 1.0 - Swc - Sgr  # type: ignore[operator]
            valid_w_gw = movable_water_range_gw > minimum_mobile_pore_space
            se_w_gw = np.clip(
                (sw - Swc) / np.where(valid_w_gw, movable_water_range_gw, 1.0), 0.0, 1.0
            )
            krw_gw_proxy = _let_relperm(
                se_w_gw,  # type: ignore[arg-type]
                water_params.L,
                water_params.E,
                water_params.T,
            )
            d_krw_gw_proxy_d_sw = np.where(
                valid_w_gw,
                _let_curve_slope_wrt_normalized_saturation(
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
                kro_endpoint=kro_endpoint,
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
                    kro_endpoint=kro_endpoint,
                    water_saturation=sw,
                    oil_saturation=so,
                    gas_saturation=sg,
                ),
                0.0,
                1.0,
            )
            d_krw_ow_d_sw = krw_max * (
                derivs_ow["d_kro_d_kro_w"] * d_krw_ow_proxy_d_sw
                + derivs_ow["d_kro_d_kro_g"] * d_krw_gw_proxy_d_sw
                + derivs_ow["d_kro_d_sw_explicit"]
            )
            d_krw_ow_d_so = krw_max * derivs_ow["d_kro_d_so_explicit"]
            d_krw_ow_d_sg = krw_max * derivs_ow["d_kro_d_sg_explicit"]

            # Blend raw kr values for floor masking
            krw_blend_raw = f * krw_ww + (1.0 - f) * krw_ow_raw
            kro_blend_raw = f * kro_ww_raw + (1.0 - f) * kro_ow
            krg_blend_raw = f * krg_ww + (1.0 - f) * krg_ow

            # Blend derivatives
            d_krw_d_sw_raw = f * d_krw_ww_d_sw_raw + (1.0 - f) * d_krw_ow_d_sw
            d_krw_d_so_raw = (1.0 - f) * d_krw_ow_d_so
            d_krw_d_sg_raw = (1.0 - f) * d_krw_ow_d_sg

            d_kro_d_sw_raw = f * d_kro_ww_d_sw
            d_kro_d_so_raw = f * d_kro_ww_d_so + (1.0 - f) * d_kro_ow_d_so
            d_kro_d_sg_raw = f * d_kro_ww_d_sg

            d_krg_d_sw_raw = zeros.copy()
            d_krg_d_so_raw = zeros.copy()
            d_krg_d_sg_raw = f * d_krg_ww_d_sg_raw + (1.0 - f) * d_krg_ow_d_sg

            # Apply floors to blended derivatives
            d_krw_d_sw = _apply_relperm_floor_to_derivative(
                d_krw_d_sw_raw, krw_blend_raw, floor_w
            )
            d_krw_d_so = _apply_relperm_floor_to_derivative(
                d_krw_d_so_raw, krw_blend_raw, floor_w
            )
            d_krw_d_sg = _apply_relperm_floor_to_derivative(
                d_krw_d_sg_raw, krw_blend_raw, floor_w
            )
            d_kro_d_sw = _apply_relperm_floor_to_derivative(
                d_kro_d_sw_raw, kro_blend_raw, floor_o
            )
            d_kro_d_so = _apply_relperm_floor_to_derivative(
                d_kro_d_so_raw, kro_blend_raw, floor_o
            )
            d_kro_d_sg = _apply_relperm_floor_to_derivative(
                d_kro_d_sg_raw, kro_blend_raw, floor_o
            )
            d_krg_d_sw = zeros.copy()
            d_krg_d_so = zeros.copy()
            d_krg_d_sg = _apply_relperm_floor_to_derivative(
                d_krg_d_sg_raw, krg_blend_raw, floor_g
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
                results = tuple(r.item() for r in results)  # type: ignore
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
        mobile_water_range = 1.0 - Swc - Sorw  # type: ignore[operator]
        valid_water = mobile_water_range > minimum_mobile_pore_space
        se_w_for_krw = np.clip(
            (sw - Swc) / np.where(valid_water, mobile_water_range, 1.0),
            0.0,
            1.0,
        )
        krw_raw = krw_max * _let_relperm(
            se_w_for_krw,  # type: ignore[arg-type]
            water_params.L,
            water_params.E,
            water_params.T,
        )
        d_krw_d_sw_raw = np.where(
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
        d_krw_d_sw = _apply_relperm_floor_to_derivative(
            d_krw_d_sw_raw, krw_raw, floor_w
        )
        d_krw_d_so = zeros.copy()
        d_krw_d_sg = zeros.copy()

        # krg
        mobile_gas_range = 1.0 - Swc - Sgr - Sorg  # type: ignore[operator]
        valid_gas = mobile_gas_range > minimum_mobile_pore_space
        se_g_for_krg = np.clip(
            (sg - Sgr) / np.where(valid_gas, mobile_gas_range, 1.0),
            0.0,
            1.0,
        )
        krg_raw = krg_max * _let_relperm(
            se_g_for_krg,  # type: ignore[arg-type]
            gas_params.L,
            gas_params.E,
            gas_params.T,
        )
        d_krg_d_sg_raw = np.where(
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
        d_krg_d_sg = _apply_relperm_floor_to_derivative(
            d_krg_d_sg_raw, krg_raw, floor_g
        )
        d_krg_d_sw = zeros.copy()
        d_krg_d_so = zeros.copy()

        # kro_w (unit-endpoint oil kr from water-oil system, function of So)
        mobile_oil_water_range = 1.0 - Swc - Sorw  # type: ignore[operator]
        valid_oil_water = mobile_oil_water_range > minimum_mobile_pore_space
        se_o_water_system = np.clip(
            (so - Sorw) / np.where(valid_oil_water, mobile_oil_water_range, 1.0),
            0.0,
            1.0,
        )
        kro_w_vals = _let_relperm(
            se_o_water_system,  # type: ignore[arg-type]
            oil_water_params.L,
            oil_water_params.E,
            oil_water_params.T,
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
        kro_g_vals = _let_relperm(
            se_o_gas_system,  # type: ignore[arg-type]
            gas_oil_params.L,
            gas_oil_params.E,
            gas_oil_params.T,
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
        d_kro_g_d_sw = zeros.copy()
        d_kro_g_d_sg = zeros.copy()

        derivatives = get_mixing_rule_partial_derivatives(
            rule=mixing_rule,
            kro_w=kro_w_vals,
            kro_g=kro_g_vals,
            krw=krw_raw,
            krg=krg_raw,
            kro_endpoint=kro_endpoint,
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
            epsilon=c.FINITE_DIFFERENCE_EPSILON,
        )
        d_kro_d_kro_w = derivatives["d_kro_d_kro_w"]
        d_kro_d_kro_g = derivatives["d_kro_d_kro_g"]
        d_kro_d_krw_mix = derivatives["d_kro_d_krw"]
        d_kro_d_krg_mix = derivatives["d_kro_d_krg"]
        d_kro_d_water_saturation_explicit = derivatives["d_kro_d_sw_explicit"]
        d_kro_d_oil_saturation_explicit = derivatives["d_kro_d_so_explicit"]
        d_kro_d_gas_saturation_explicit = derivatives["d_kro_d_sg_explicit"]

        # Forward evaluate kro for floor masking
        kro_mixed_raw = kro_max * np.clip(
            mixing_rule(
                kro_w=kro_w_vals,
                kro_g=kro_g_vals,
                krw=krw_raw,
                krg=krg_raw,
                kro_endpoint=kro_endpoint,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
            ),
            0.0,
            1.0,
        )

        d_kro_d_sw_raw = kro_max * (
            d_kro_d_kro_w * d_kro_w_d_sw
            + d_kro_d_kro_g * d_kro_g_d_sw
            + d_kro_d_krw_mix * d_krw_d_sw_raw
            + d_kro_d_krg_mix * zeros
            + d_kro_d_water_saturation_explicit
        )
        d_kro_d_so_raw = kro_max * (
            d_kro_d_kro_w * d_kro_w_d_so
            + d_kro_d_kro_g * d_kro_g_d_so
            + d_kro_d_krw_mix * zeros
            + d_kro_d_krg_mix * zeros
            + d_kro_d_oil_saturation_explicit
        )
        d_kro_d_sg_raw = kro_max * (
            d_kro_d_kro_w * d_kro_w_d_sg
            + d_kro_d_kro_g * d_kro_g_d_sg
            + d_kro_d_krw_mix * zeros
            + d_kro_d_krg_mix * d_krg_d_sg_raw
            + d_kro_d_gas_saturation_explicit
        )

        d_kro_d_sw = _apply_relperm_floor_to_derivative(
            d_kro_d_sw_raw, kro_mixed_raw, floor_o
        )
        d_kro_d_so = _apply_relperm_floor_to_derivative(
            d_kro_d_so_raw, kro_mixed_raw, floor_o
        )
        d_kro_d_sg = _apply_relperm_floor_to_derivative(
            d_kro_d_sg_raw, kro_mixed_raw, floor_o
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
            results = tuple(r.item() for r in results)
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

"""Base relative permeability tables for multi-phase flow simulations."""

import threading
import typing
import warnings

import attrs
import numba
import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator
from typing_extensions import Self

from bores.blackoil.satfunc.relperm.mixing_rules import (
    MixingRule,
    deserialize_mixing_rule,
    get_mixing_rule,
    get_mixing_rule_partial_derivatives,
    serialize_mixing_rule,
)
from bores.blackoil.satfunc.utils import build_pchip_interpolant
from bores.constants import c
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.precision import get_dtype, get_floating_point_info
from bores.serde.registry import make_serializable_type_registrar
from bores.serde.stores import StoreSerializable
from bores.types import (
    FluidPhase,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
    RelativePermeabilities,
    RelativePermeabilityDerivatives,
    Spacing,
)
from bores.utils import is_scalar_like

__all__ = [
    "RelPermEndpoints",
    "ThreePhaseRelPermTable",
    "TwoPhaseRelPermTable",
    "get_relperm_table",
    "list_relperm_tables",
    "relperm_table",
]


def show_invalid_saturation(val: NumberOrArray[NDimension], *, max_display: int = 20) -> str:
    if is_scalar_like(val) and (val < 0 or val > 1):
        return str(val)

    invalid = val[(val < 0) | (val > 1)]  # type: ignore
    length = len(invalid)
    if length == 0:
        return "OK"
    if length == 1:
        return str(invalid[0])
    return (
        f"{invalid}" if length <= max_display else f"{invalid[:max_display]} ... (count={length})"
    )


#: Sentinel type for the `minimum_*_relperm` attributes on relperm tables and
#: analytical models. `"auto"` derives the min_value from the active dtype;
#: `None` disables the min/floor value entirely; a `Number` sets an explicit min_value.
MinimumRelPerm = typing.Literal["auto"] | None | Number


def resolve_min_relperm(min_value: MinimumRelPerm) -> Number | None:
    """
    Resolve a `MinimumRelPerm` sentinel to a concrete Number or `None`.

    `"auto"` - `max(4 * machine_epsilon, 1e-8)`, consistent with the
    approach used by CMG IMEX/GEM for minimum mobility min_values: tight enough
    that it does not distort the physics, but sufficient to prevent
    near-singular transmissibility rows in the Jacobian.

    `None` - `None` (no min_value; kr can reach zero exactly).

    `Number` - that value, used as-is after a sign / sanity check.

    :param min_value: Sentinel value from a relperm table or analytical model.
    :return: Resolved Number min_value or `None`.
    """
    if min_value is None:
        return None

    if min_value == "auto":
        info = get_floating_point_info()
        return max(4.0 * info.eps, 1e-8)

    if not isinstance(min_value, (int, Number, np.floating, np.integer)):
        raise ValidationError(
            f"`min_*_relperm` must be 'auto', None, or a Number. Got {min_value!r}."
        )

    if min_value < 0.0:
        raise ValidationError(f"`min_*_relperm` min_value must be non-negative. Got {min_value}.")
    return min_value


# TODO: Add numba overload for the Number and NumberArray versions of this function
@numba.njit(cache=True, inline="always")
def clamp_relperm(
    kr: NumberOrArray[NDimension],
    min_value: Number | None,
) -> NumberOrArray[NDimension]:
    """
    Clamp `kr` to `[min_value, ∞)` in-place-compatible fashion.

    Smoothly transitions from the raw `kr` values to the min_value value, avoiding
    non-differentiability and ensuring well-behaved derivatives for solvers.

    When `min_value` is `None` the array is returned unchanged.

    :param kr: Relative permeability value(s).
    :param min_value: Resolved min_value/minimum value or `None`.
    :return: Floored relative permeability value(s).
    """
    if min_value is None:
        return kr
    return np.sqrt(kr**2 + min_value**2).astype(kr.dtype)  # type: ignore[return-value]


@numba.njit(cache=True, inline="always")
def clamp_relperm_derivative(
    dkr: NumberOrArray[NDimension],
    kr_raw: NumberOrArray[NDimension],
    min_value: Number | None,
) -> NumberOrArray[NDimension]:
    """
    Smoothly clamp the derivative of `kr` to zero in the min_value region.
    When `kr` is above the min_value, the derivative is unchanged. As `kr` approaches
    and falls below the min_value, the derivative transitions smoothly to zero, ensuring
    that the Jacobian remains well-conditioned and solvers remain stable.

    When `min_value` is `None` the derivative is returned unchanged.

    The derivative of `max(kr, min_value)` w.row.t. any variable is:

    - `dkr / d(var)`  when `kr > min_value`  (active region)
    - `kr / np.sqrt(kr**2 + min_value**2)`  when `kr <= min_value` (min_value / flat region)

    :param dkr: Derivative of the raw (pre-min_value) kr w.row.t. some variable.
    :param kr_raw: Raw (pre-min_value) kr value(s), used to identify min_value cells.
    :param min_value: Resolved min_value/minimum value or `None`.
    :return: Derivative with min_value cells zeroed out.
    """
    if min_value is None:
        return dkr
    return np.where(  # type: ignore[return-value]
        kr_raw > min_value,
        dkr,
        kr_raw / np.sqrt(kr_raw**2 + min_value**2),  # type: ignore[operator]
    ).astype(dkr.dtype)  # type: ignore[attr-defined]


@attrs.frozen
class RelPermEndpoints:
    """
    Relative permeability endpoint values for all three phases.

    All values are dimensionless and in [0, 1] for normalized tables.
    """

    oil: Number
    """
    Oil relative permeability at connate water saturation with no gas (krocw).
    Reference value for three-phase mixing rule normalization.
    """

    water: Number
    """
    Water relative permeability endpoint (krw at residual oil saturation, Sor).
    Used as the injector mobility min_value reference for water injectors.
    """

    gas: Number
    """
    Gas relative permeability endpoint (krg at connate water + residual oil).
    Used as the injector mobility min_value reference for gas injectors.
    """


@attrs.frozen
class SaturationEndpoints:
    """
    Saturation endpoint values (connate/residual/critical) for all three
    phases - the saturations *at which* mobility starts/stops, as opposed
    to `RelPermEndpoints`, which is the relative permeability *value* at
    those points.

    All values are dimensionless fractions in [0, 1].
    """

    connate_water: Number
    """Connate/critical water saturation (swco) below which water is immobile."""

    residual_oil_water: Number
    """Residual oil saturation after water flood (sorw); oil is immobile below this."""

    residual_oil_gas: Number
    """Residual oil saturation after gas flood (sorg); oil is immobile below this."""

    residual_gas: Number
    """Residual/critical gas saturation (sgr) below which gas is immobile."""


class RelativePermeabilityTable(StoreSerializable):
    """
    Protocol for a relative permeability model/table that
    computes relative permeabilities based on fluid saturations.
    """

    __abstract_serializable__ = True

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        raise NotImplementedError

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        raise NotImplementedError

    def get_oil_relperm_endpoint(self) -> Number:
        """Oil relative permeability at connate water saturation (krocw). Defaults to 1.0."""
        return 1.0

    def get_water_relperm_endpoint(self) -> Number:
        """
        Water relative permeability endpoint (krw at residual oil saturation).

        Returns the maximum krw value from the two-phase oil-water table.
        Defaults to 1.0 for unit-normalized tables and analytical models.
        """
        return 1.0

    def get_gas_relperm_endpoint(self) -> Number:
        """
        Gas relative permeability endpoint (krg at connate water + residual oil).

        Returns the maximum krg value from the two-phase gas-oil table.
        Defaults to 1.0 for unit-normalized tables and analytical models.
        """
        return 1.0

    def get_relperm_endpoints(self) -> RelPermEndpoints:
        """
        Compute all three phase relative permeability endpoints in a single call.

        Returns a `RelPermEndpoints` container holding kro, krw, and krg endpoint
        values. For unit-normalized tables all three are 1.0. For unnormalized
        tabular data the values are read from the stored kr arrays.

        :return: `RelPermEndpoints` with oil, water, and gas endpoint values.
        """
        return RelPermEndpoints(
            oil=self.get_oil_relperm_endpoint(),
            water=self.get_water_relperm_endpoint(),
            gas=self.get_gas_relperm_endpoint(),
        )

    def get_connate_water_saturation(self) -> Number:
        """
        Connate/critical water saturation (swco) below which water is
        immobile.

        Defaults to `0.0` for models/tables that don't carry this
        information (e.g. a unit-normalized table with no known floor).
        """
        return 0.0

    def get_residual_oil_saturation_water(self) -> Number:
        """
        Residual oil saturation after water flood (sorw); oil is immobile
        below this in the oil-water system.

        Defaults to `0.0`.
        """
        return 0.0

    def get_residual_oil_saturation_gas(self) -> Number:
        """
        Residual oil saturation after gas flood (sorg); oil is immobile
        below this in the gas-oil system.

        Defaults to `0.0`.
        """
        return 0.0

    def get_residual_gas_saturation(self) -> Number:
        """
        Residual/critical gas saturation (sgr) below which gas is immobile.

        Defaults to `0.0`.
        """
        return 0.0

    def get_saturation_endpoints(self) -> SaturationEndpoints:
        """
        Compute all four saturation endpoints in a single call.

        :return: `SaturationEndpoints` with connate water and the three
            residual/critical saturations.
        """
        return SaturationEndpoints(
            connate_water=self.get_connate_water_saturation(),
            residual_oil_water=self.get_residual_oil_saturation_water(),
            residual_oil_gas=self.get_residual_oil_saturation_gas(),
            residual_gas=self.get_residual_gas_saturation(),
        )

    def evaluate(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for water, oil, and gas.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param kwargs: Other key word arguments for computing the relative peremabilities.
        :return: `RelativePermeabilities` dictionary with keys "water", "oil", "gas".
        """
        raise NotImplementedError

    def derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute relative permeability derivatives for water, oil, and gas.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param kwargs: Other key word arguments for computing the derivatives.
        :return: `RelativePermeabilityDerivatives` dictionary.
        """
        raise NotImplementedError

    def __call__(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for water, oil, and gas.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :param kwargs: Other key word arguments for computing the relative peremabilities.
        :return: `RelativePermeabilities` dictionary.
        """
        return self.evaluate(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )


RELPERM_TABLES: dict[str, type[RelativePermeabilityTable]] = {}
"""Registry of relative permeability table types."""

_relperm_tables_lock = threading.Lock()
relperm_table = make_serializable_type_registrar(
    base_cls=RelativePermeabilityTable,
    registry=RELPERM_TABLES,
    key_attr="__type__",
    lock=_relperm_tables_lock,
    override=False,
    auto_register_serializer=True,
    auto_register_deserializer=True,
)


def list_relperm_tables() -> list[str]:
    """
    List all registered relative permeability table types.

    :return: List of registered relative permeability table type names.
    """
    with _relperm_tables_lock:
        return list(RELPERM_TABLES.keys())


def get_relperm_table(name: str) -> type[RelativePermeabilityTable]:
    """
    Get a registered relative permeability table type by name.

    :param name: Registered name of the relative permeability table type.
    :return: Relative permeability table class.
    :raises ValidationError: If the relative permeability table type is not registered.
    """
    with _relperm_tables_lock:
        if name not in RELPERM_TABLES:
            raise ValidationError(
                f"Relative permeability table type '{name}' is not registered. "
                f"Use `@relperm_table` to register it. "
                f"Available types: {list(RELPERM_TABLES.keys())}"
            )
        return RELPERM_TABLES[name]


@relperm_table
@attrs.frozen(slots=True)
class TwoPhaseRelPermTable(
    RelativePermeabilityTable,
    load_exclude={
        "_wetting_interp",
        "_wetting_d_interp",
        "_non_wetting_interp",
        "_non_wetting_d_interp",
    },
    dump_exclude={
        "_wetting_interp",
        "_wetting_d_interp",
        "_non_wetting_interp",
        "_non_wetting_d_interp",
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

    **Minimum relperm min_values** (`min_wetting_relperm` / `min_non_wetting_relperm`):

    When non-`None`, a min_value is applied to the interpolated kr value and
    the derivative is zeroed out wherever the raw (pre-min_value) kr is at or
    below the min_value. This gives a derivative that is exactly consistent with
    the min_value kr, preventing Jacobian-residual mismatches that cause MBE.

    `"auto"` derives the min_value from the active floating-point dtype:
    `max(4 * machine_epsilon, 1e-8)`. `None` disables the min_value (kr can
    reach zero exactly).

    **dtype**:

    All stored arrays and all returned scalars / arrays use `dtype`. Query
    methods always cast their output to `dtype` before returning. Defaults to
    `get_dtype()` when not specified.
    """

    __type__ = "two_phase_relperm_table"

    wetting_phase: FluidPhase | str = attrs.field(converter=FluidPhase)
    """The wetting fluid phase, e.g. WATER (oil-water) or OIL (gas-oil)."""

    non_wetting_phase: FluidPhase | str = attrs.field(converter=FluidPhase)
    """The non-wetting fluid phase, e.g. OIL (oil-water) or GAS (gas-oil)."""

    reference_saturation: NumberArray[OneDimension]
    """
    Saturation values used as the x-axis for interpolation, monotonically
    increasing. May represent either the wetting or non-wetting phase
    saturation depending on `reference_phase`.
    """

    wetting_phase_relative_permeability: NumberArray[OneDimension]
    """Relative permeability values for the wetting phase at each reference saturation."""

    non_wetting_phase_relative_permeability: NumberArray[OneDimension]
    """Relative permeability values for the non-wetting phase at each reference saturation."""

    reference_phase: typing.Literal["wetting", "non_wetting"] = attrs.field(default="wetting")
    """
    Which phase the `reference_saturation` axis represents.

    - `"wetting"` - `reference_saturation` holds wetting phase saturation
      values.  krw increases and krnw decreases as `reference_saturation`
      increases.
    - `"non_wetting"` - `reference_saturation` holds non-wetting phase
      saturation values.  krnw increases and krw decreases as
      `reference_saturation` increases.

    This does not change the interpolation mechanics. It only records which
    physical saturation the caller must supply when querying the table, so
    that `ThreePhaseRelPermTable` (and any other consumer) can dispatch the
    correct saturation grid without hard-coding assumptions.
    """

    min_wetting_relperm: MinimumRelPerm = None
    """
    Minimum min_value for the wetting-phase relative permeability.

    `"auto"` - `max(4 * machine_epsilon, 1e-8)` (dtype-aware).
    `None` - no min_value; kr can reach zero exactly.
    `Number` - explicit user-supplied min_value value.

    The min_value is applied to the interpolated kr value, and the derivative is
    zeroed out in the min_value region so that the Jacobian is consistent with
    the kr value (no MBE from mismatched kr/derivative pairs).
    """

    min_non_wetting_relperm: MinimumRelPerm = None
    """
    Minimum min_value for the non-wetting-phase relative permeability.

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

    Typical values are `"cosine"` (default, clusters points near endpoints) and
    `"linear"` (uniform).
    """

    dtype: npt.DTypeLike | None = attrs.field(default=None)
    """
    Array dtype for all stored arrays and all query return values.

    All three saturation/kr arrays are cast to this dtype in `__attrs_post_init__`.
    Query methods (`_query`, `_d_query`) cast their outputs to this
    dtype before returning, including the zero arrays returned for derivatives
    outside the saturation range and for absent phases.

    Defaults to `get_dtype()` when `None`.
    """

    _wetting_interp: PchipInterpolator = attrs.field(init=False, repr=False)
    _wetting_d_interp: PchipInterpolator = attrs.field(init=False, repr=False)
    _non_wetting_interp: PchipInterpolator = attrs.field(init=False, repr=False)
    _non_wetting_d_interp: PchipInterpolator = attrs.field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if self.reference_phase not in ("wetting", "non_wetting"):
            raise ValidationError(
                f"`reference_phase` must be 'wetting' or 'non_wetting', "
                f"got {self.reference_phase!r}"
            )
        if len(self.reference_saturation) != len(self.wetting_phase_relative_permeability):
            raise ValidationError(
                f"`reference_saturation` and wetting phase kr arrays must have same "
                f"length. Got {len(self.reference_saturation)} vs "
                f"{len(self.wetting_phase_relative_permeability)}"
            )
        if len(self.reference_saturation) != len(self.non_wetting_phase_relative_permeability):
            raise ValidationError(
                f"`reference_saturation` and non-wetting phase kr arrays must have "
                f"same length. Got {len(self.reference_saturation)} vs "
                f"{len(self.non_wetting_phase_relative_permeability)}"
            )
        if len(self.reference_saturation) < 2:
            raise ValidationError("At least 2 points required for interpolation")
        if not np.all(np.diff(self.reference_saturation) >= 0):
            raise ValidationError("`reference_saturation` must be monotonically increasing")

        # Resolve and enforce dtype on all stored arrays
        dtype = np.dtype(self.dtype) if self.dtype is not None else get_dtype()
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(
            self,
            "reference_saturation",
            np.asarray(self.reference_saturation, dtype=dtype),
        )
        object.__setattr__(
            self,
            "wetting_phase_relative_permeability",
            np.asarray(self.wetting_phase_relative_permeability, dtype=dtype),
        )
        object.__setattr__(
            self,
            "non_wetting_phase_relative_permeability",
            np.asarray(self.non_wetting_phase_relative_permeability, dtype=dtype),
        )

        # Validate min_value sentinels eagerly so errors surface at construction time
        resolve_min_relperm(self.min_wetting_relperm)
        resolve_min_relperm(self.min_non_wetting_relperm)

        # Build interpolants
        wetting_interp, wetting_d_interp = build_pchip_interpolant(
            reference_saturation=self.reference_saturation,
            values=self.wetting_phase_relative_permeability,
            number_of_base_points=self.number_of_base_points,
            number_of_endpoint_extra_points=self.number_of_endpoint_extra_points,
            spacing=self.spacing,
            dtype=dtype,
        )
        non_wetting_interp, non_wetting_d_interp = build_pchip_interpolant(
            reference_saturation=self.reference_saturation,
            values=self.non_wetting_phase_relative_permeability,
            number_of_base_points=self.number_of_base_points,
            number_of_endpoint_extra_points=self.number_of_endpoint_extra_points,
            spacing=self.spacing,
            dtype=dtype,
        )
        object.__setattr__(self, "_wetting_interp", wetting_interp)
        object.__setattr__(self, "_wetting_d_interp", wetting_d_interp)
        object.__setattr__(self, "_non_wetting_interp", non_wetting_interp)
        object.__setattr__(self, "_non_wetting_d_interp", non_wetting_d_interp)

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        return typing.cast(FluidPhase, self.wetting_phase)

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        return typing.cast(FluidPhase, self.wetting_phase)

    def resolve_reference_saturation(
        self,
        wetting_saturation: NumberOrArray[NDimension],
        non_wetting_saturation: NumberOrArray[NDimension],
    ) -> NumberOrArray[NDimension]:
        """
        Return whichever saturation array corresponds to the reference axis.

        :param wetting_saturation: Current wetting phase saturation (scalar or array).
        :param non_wetting_saturation: Current non-wetting phase saturation (scalar or array).
        :return: The saturation to use as the interpolation x-value.
        """
        if self.reference_phase == "non_wetting":
            return non_wetting_saturation
        return wetting_saturation

    def _query(
        self,
        interpolant: PchipInterpolator,
        reference: NumberOrArray[NDimension],
        extrapolate_left: Number,
        extrapolate_right: Number,
    ) -> NumberOrArray[NDimension]:
        """
        Evaluate a pre-built PCHIP interpolant at `reference`, applying
        constant extrapolation at the boundaries. Result is cast to `self.dtype`.

        :param interpolant: Pre-built `PchipInterpolator` instance.
        :param reference: Query saturation value(s) - scalar or array.
        :param extrapolate_left: Constant returned for values below the knot range.
        :param extrapolate_right: Constant returned for values above the knot range.
        :return: Interpolated value(s) cast to `self.dtype`, with the same shape as `reference`.
        """
        dtype = self.dtype
        is_scalar = np.isscalar(reference)
        sat = np.atleast_1d(reference)
        x_min = interpolant.x[0]
        x_max = interpolant.x[-1]

        result = interpolant(np.clip(sat, x_min, x_max))
        result = np.where(sat < x_min, extrapolate_left, result)
        result = np.where(sat > x_max, extrapolate_right, result)
        result = result.astype(dtype, copy=False)

        if is_scalar:
            return typing.cast(Number, dtype.type(result.item()))  # type: ignore
        return typing.cast(NumberArray[NDimension], result.reshape(sat.shape))

    def _d_query(
        self,
        d_interpolant: PchipInterpolator,
        reference: NumberOrArray[NDimension],
    ) -> NumberOrArray[NDimension]:
        """
        Evaluate a pre-built PCHIP derivative interpolant at `reference`,
        returning zero (in `self.dtype`) outside the knot range.

        :param d_interpolant: Pre-built derivative `PchipInterpolator`.
        :param reference: Query saturation value(s) - scalar or array.
        :return: Derivative value(s) cast to `self.dtype`, with the same shape as `reference`.
        """
        dtype = self.dtype
        is_scalar = np.isscalar(reference)
        sat = np.atleast_1d(reference)
        x_min = d_interpolant.x[0]
        x_max = d_interpolant.x[-1]

        result = d_interpolant(np.clip(sat, x_min, x_max))
        result = np.where(
            (sat < x_min) | (sat > x_max),
            dtype.type(0),  # type: ignore
            result,
        )
        result = result.astype(dtype, copy=False)

        if is_scalar:
            return typing.cast(Number, dtype.type(result.item()))  # type: ignore
        return typing.cast(NumberArray[NDimension], result.reshape(sat.shape))

    def get_wetting_phase_relative_permeability(
        self,
        wetting_saturation: NumberOrArray[NDimension],
        non_wetting_saturation: NumberOrArray[NDimension] | None = None,
    ) -> NumberOrArray[NDimension]:
        """
        Get wetting phase relative permeability.

        When `reference_phase="wetting"`, only `wetting_saturation` is
        needed. When `reference_phase="non_wetting"`, `non_wetting_saturation` must be supplied.

        The `min_wetting_relperm` min_value (if set) is applied to the result.
        The returned value is always cast to `self.dtype`.

        :param wetting_saturation: Wetting phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Relative permeability value(s) in `self.dtype`.
        """
        ref = self.resolve_reference_saturation(
            wetting_saturation,
            non_wetting_saturation if non_wetting_saturation is not None else wetting_saturation,
        )
        kr = self._query(
            self._wetting_interp,
            ref,
            extrapolate_left=self.wetting_phase_relative_permeability[0],
            extrapolate_right=self.wetting_phase_relative_permeability[-1],
        )
        min_value = resolve_min_relperm(self.min_wetting_relperm)
        return clamp_relperm(kr, min_value)

    def get_non_wetting_phase_relative_permeability(
        self,
        wetting_saturation: NumberOrArray[NDimension],
        non_wetting_saturation: NumberOrArray[NDimension] | None = None,
    ) -> NumberOrArray[NDimension]:
        """
        Get non-wetting phase relative permeability.

        When `reference_phase="wetting"`, only `wetting_saturation` is
        needed. When `reference_phase="non_wetting"`, `non_wetting_saturation` must be supplied.

        The `min_non_wetting_relperm` min_value (if set) is applied to the result.
        The returned value is always cast to `self.dtype`.

        :param wetting_saturation: Wetting phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Relative permeability value(s) in `self.dtype`.
        """
        ref = self.resolve_reference_saturation(
            wetting_saturation,
            non_wetting_saturation if non_wetting_saturation is not None else wetting_saturation,
        )
        kr = self._query(
            self._non_wetting_interp,
            ref,
            extrapolate_left=self.non_wetting_phase_relative_permeability[0],
            extrapolate_right=self.non_wetting_phase_relative_permeability[-1],
        )
        min_value = resolve_min_relperm(self.min_non_wetting_relperm)
        return clamp_relperm(kr, min_value)

    def get_two_phase_relative_permeabilities(
        self,
        wetting_saturation: NumberOrArray[NDimension],
        non_wetting_saturation: NumberOrArray[NDimension] | None = None,
    ) -> tuple[NumberOrArray[NDimension], NumberOrArray[NDimension]]:
        """
        Get both wetting and non-wetting phase relative permeabilities.

        :param wetting_saturation: Wetting phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Tuple of `(wetting_kr, non_wetting_kr)` in `self.dtype`.
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
        wetting_saturation: NumberOrArray[NDimension],
        non_wetting_saturation: NumberOrArray[NDimension] | None = None,
    ) -> NumberOrArray[NDimension]:
        """
        Derivative of the wetting-phase relative permeability with respect to
        the reference saturation axis of this table, evaluated from the
        analytical PCHIP derivative.

        The derivative is zero (in `self.dtype`) outside the tabulated saturation
        range (constant extrapolation = zero slope).

        Where the `min_wetting_relperm` min_value is active (raw kr ≤ min_value),
        the derivative is zeroed out to be consistent with the min_value
        (constant) kr value, preventing Jacobian-residual mismatches.

        :param wetting_saturation: Wetting-phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting-phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Derivative value(s) in `self.dtype` with the same shape as the input.
        """
        ref = self.resolve_reference_saturation(
            wetting_saturation,
            non_wetting_saturation if non_wetting_saturation is not None else wetting_saturation,
        )
        dkr = self._d_query(self._wetting_d_interp, ref)
        min_value = resolve_min_relperm(self.min_wetting_relperm)
        if min_value is None:
            return dkr

        kr_raw = self._query(
            self._wetting_interp,
            ref,
            extrapolate_left=self.wetting_phase_relative_permeability[0],
            extrapolate_right=self.wetting_phase_relative_permeability[-1],
        )
        return clamp_relperm_derivative(dkr, kr_raw, min_value)

    def get_non_wetting_phase_relative_permeability_derivative(
        self,
        wetting_saturation: NumberOrArray[NDimension],
        non_wetting_saturation: NumberOrArray[NDimension] | None = None,
    ) -> NumberOrArray[NDimension]:
        """
        Derivative of the non-wetting-phase relative permeability with respect
        to the reference saturation axis of this table, evaluated from the
        analytical PCHIP derivative.

        The derivative is zero (in `self.dtype`) outside the tabulated saturation
        range (constant extrapolation = zero slope).

        Where the `min_non_wetting_relperm` min_value is active (raw kr ≤
        min_value), the derivative is zeroed out to be consistent with the min_value
        (constant) kr value, preventing Jacobian-residual mismatches.

        :param wetting_saturation: Wetting-phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting-phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Derivative value(s) in `self.dtype` with the same shape as the input.
        """
        ref = self.resolve_reference_saturation(
            wetting_saturation,
            non_wetting_saturation if non_wetting_saturation is not None else wetting_saturation,
        )
        dkr = self._d_query(self._non_wetting_d_interp, ref)
        min_value = resolve_min_relperm(self.min_non_wetting_relperm)
        if min_value is None:
            return dkr

        kr_raw = self._query(
            self._non_wetting_interp,
            ref,
            extrapolate_left=self.non_wetting_phase_relative_permeability[0],
            extrapolate_right=self.non_wetting_phase_relative_permeability[-1],
        )
        return clamp_relperm_derivative(dkr, kr_raw, min_value)

    def get_oil_relperm_endpoint(self) -> Number:
        if self.non_wetting_phase == FluidPhase.OIL:
            return np.max(self.non_wetting_phase_relative_permeability)
        elif self.wetting_phase == FluidPhase.OIL:
            return np.max(self.wetting_phase_relative_permeability)
        return 1.0

    def get_water_relperm_endpoint(self) -> Number:
        phases = {self.wetting_phase, self.non_wetting_phase}
        if FluidPhase.WATER not in phases:
            return 1.0
        if self.wetting_phase == FluidPhase.WATER:
            return np.max(self.wetting_phase_relative_permeability)
        return np.max(self.non_wetting_phase_relative_permeability)

    def get_gas_relperm_endpoint(self) -> Number:
        phases = {self.wetting_phase, self.non_wetting_phase}
        if FluidPhase.GAS not in phases:
            return 1.0
        if self.wetting_phase == FluidPhase.GAS:
            return np.max(self.wetting_phase_relative_permeability)
        return np.max(self.non_wetting_phase_relative_permeability)

    def get_reference_axis_phase(self) -> FluidPhase:
        """Returns which physical phase `reference_saturation` represents."""
        return typing.cast(
            FluidPhase,
            self.wetting_phase if self.reference_phase == "wetting" else self.non_wetting_phase,
        )

    def get_connate_water_saturation(self) -> Number:
        """
        Connate/critical water saturation (swco), read directly off this
        table's own saturation axis (its minimum, by construction) when
        that axis represents water. `0.0` if this table doesn't have water
        as one of its two phases at all, or water isn't the axis phase (no
        water-saturation information to read off the axis in that case).
        """
        if self.get_reference_axis_phase() != FluidPhase.WATER:
            return 0.0
        return float(self.reference_saturation[0])

    def get_residual_oil_saturation_water(self) -> Number:
        """
        Residual oil saturation after water flood (sorw), derived from this
        table's own saturation axis - only meaningful for an oil-water
        table (`0.0` otherwise). If the axis is Sw, sorw is `1 -
        Sw_max`; if the axis is So directly, sorw is the axis minimum.
        """
        phases = {self.wetting_phase, self.non_wetting_phase}
        if phases != {FluidPhase.OIL, FluidPhase.WATER}:
            return 0.0
        if self.get_reference_axis_phase() == FluidPhase.WATER:
            return max(0.0, 1.0 - float(self.reference_saturation[-1]))
        return float(self.reference_saturation[0])

    def get_residual_gas_saturation(self) -> Number:
        """
        Residual/critical gas saturation (sgr), read directly off this
        table's own saturation axis (its minimum, by construction) when
        that axis represents gas. `0.0` if this table doesn't have gas as
        one of its two phases, or gas isn't the axis phase.
        """
        if self.get_reference_axis_phase() != FluidPhase.GAS:
            return 0.0
        return float(self.reference_saturation[0])

    def get_residual_oil_saturation_gas(self) -> Number:
        """
        Residual oil saturation after gas flood (sorg), derived from this
        table's own saturation axis - only meaningful for a gas-oil table
        (`0.0` otherwise).

        Standalone two-phase interpretation: this table alone has no
        connate-water information, so this implicitly assumes `swco = 0`
        (`sorg = 1 - Sg_max`). `ThreePhaseRelPermTable.get_residual_oil_saturation_gas`
        overrides this with the more accurate `1 - swco - Sg_max`, combining
        this table with its sibling oil-water table's own connate water.
        """
        phases = {self.wetting_phase, self.non_wetting_phase}
        if phases != {FluidPhase.OIL, FluidPhase.GAS}:
            return 0.0
        if self.get_reference_axis_phase() == FluidPhase.GAS:
            return max(0.0, 1.0 - float(self.reference_saturation[-1]))
        return float(self.reference_saturation[0])

    def evaluate(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for all three phases from a two-phase table.

        The absent phase always returns zero (in `self.dtype`). The reference saturation
        dispatched to the underlying PCHIP interpolant is chosen by inspecting which
        phases this table covers and what `reference_phase` is declared:

        - **Oil-water table** (phases are OIL and WATER): krg = 0. Sw or So is
          forwarded to the interpolant according to `reference_phase`.
        - **Gas-oil table** (phases are GAS and OIL): krw = 0. So or Sg is
          forwarded according to `reference_phase`.

        Minimum relperm min_values declared on the table are applied automatically
        inside the underlying query methods and propagate transparently.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :return: `RelativePermeabilities` dict with keys `"water"`, `"oil"`, `"gas"`.
        """
        dtype = self.dtype
        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )
        sw = np.atleast_1d(water_saturation)
        so = np.atleast_1d(oil_saturation)
        sg = np.atleast_1d(gas_saturation)
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        zeros = np.zeros(sw.shape, dtype=dtype)
        phases = {self.wetting_phase, self.non_wetting_phase}

        if phases == {FluidPhase.OIL, FluidPhase.WATER}:
            if self.wetting_phase == FluidPhase.WATER:
                krw = self.get_wetting_phase_relative_permeability(sw, non_wetting_saturation=so)
                kro = self.get_non_wetting_phase_relative_permeability(
                    sw, non_wetting_saturation=so
                )
            else:
                kro = self.get_wetting_phase_relative_permeability(so, non_wetting_saturation=sw)
                krw = self.get_non_wetting_phase_relative_permeability(
                    so, non_wetting_saturation=sw
                )
            if is_scalar:
                return RelativePermeabilities(
                    water=dtype.type(krw.item()),  # type: ignore
                    oil=dtype.type(kro.item()),  # type: ignore
                    gas=dtype.type(0),  # type: ignore
                )
            return RelativePermeabilities(water=krw, oil=kro, gas=zeros)  # type: ignore[typeddict-item]

        if phases == {FluidPhase.OIL, FluidPhase.GAS}:
            if self.wetting_phase == FluidPhase.OIL:
                kro = self.get_wetting_phase_relative_permeability(so, non_wetting_saturation=sg)
                krg = self.get_non_wetting_phase_relative_permeability(
                    so, non_wetting_saturation=sg
                )
            else:
                krg = self.get_wetting_phase_relative_permeability(sg, non_wetting_saturation=so)
                kro = self.get_non_wetting_phase_relative_permeability(
                    sg, non_wetting_saturation=so
                )
            if is_scalar:
                return RelativePermeabilities(
                    water=dtype.type(0),  # type: ignore
                    oil=dtype.type(kro.item()),  # type: ignore
                    gas=dtype.type(krg.item()),  # type: ignore
                )
            return RelativePermeabilities(water=zeros, oil=kro, gas=krg)  # type: ignore[typeddict-item]

        raise ValidationError(
            f"Cannot dispatch three-phase saturations to a two-phase table with phases "
            f"{self.wetting_phase!r} / {self.non_wetting_phase!r}. "
            f"Expected OIL+WATER or OIL+GAS."
        )

    def derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute the nine partial derivatives of three-phase relative permeabilities
        for a two-phase table.

        The absent phase contributes zero (in `self.dtype`) to every derivative.
        The two active phases contribute only along their natural saturation axis;
        all cross derivatives (e.g. dkrw/dsg for an oil-water table) are zero.

        Where a minimum relperm min_value is active on this table (raw kr ≤ min_value),
        the corresponding derivative is zeroed out by the underlying query method,
        keeping the Jacobian consistent with the min_value kr value.

        Returned layout (same as `ThreePhaseRelPermTable`):

            dkrw_dsw, dkro_dsw, dkrg_dsw,
            dkrw_dso, dkro_dso, dkrg_dso,
            dkrw_dsg, dkro_dsg, dkrg_dsg

        :param water_saturation: Water saturation (fraction, 0 to 1).
        :param oil_saturation: Oil saturation (fraction, 0 to 1).
        :param gas_saturation: Gas saturation (fraction, 0 to 1).
        :return: `RelativePermeabilityDerivatives` dictionary with all values in `self.dtype`.
        """
        dtype = self.dtype
        sw = np.atleast_1d(water_saturation)
        so = np.atleast_1d(oil_saturation)
        sg = np.atleast_1d(gas_saturation)
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        zeros = np.zeros(sw.shape, dtype=dtype)
        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )
        phases = {self.wetting_phase, self.non_wetting_phase}

        if phases == {FluidPhase.OIL, FluidPhase.WATER}:
            if self.wetting_phase == FluidPhase.WATER:
                dkrw = self.get_wetting_phase_relative_permeability_derivative(
                    sw, non_wetting_saturation=so
                )
                dkro = self.get_non_wetting_phase_relative_permeability_derivative(
                    sw, non_wetting_saturation=so
                )
                if self.reference_phase == "wetting":
                    dkrw_dsw, dkrw_dso, dkrw_dsg = dkrw, zeros, zeros
                    dkro_dsw, dkro_dso, dkro_dsg = dkro, zeros, zeros
                else:
                    dkrw_dsw, dkrw_dso, dkrw_dsg = zeros, dkrw, zeros
                    dkro_dsw, dkro_dso, dkro_dsg = zeros, dkro, zeros
            else:
                dkro = self.get_wetting_phase_relative_permeability_derivative(
                    so, non_wetting_saturation=sw
                )
                dkrw = self.get_non_wetting_phase_relative_permeability_derivative(
                    so, non_wetting_saturation=sw
                )
                if self.reference_phase == "wetting":
                    dkro_dsw, dkro_dso, dkro_dsg = zeros, dkro, zeros
                    dkrw_dsw, dkrw_dso, dkrw_dsg = zeros, dkrw, zeros
                else:
                    dkro_dsw, dkro_dso, dkro_dsg = dkro, zeros, zeros
                    dkrw_dsw, dkrw_dso, dkrw_dsg = dkrw, zeros, zeros

            results = (
                dkrw_dsw,
                dkro_dsw,
                zeros,
                dkrw_dso,
                dkro_dso,
                zeros,
                dkrw_dsg,
                dkro_dsg,
                zeros,
            )
            if is_scalar:
                results = tuple(dtype.type(row.item()) for row in results)  # type: ignore
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

        if phases == {FluidPhase.OIL, FluidPhase.GAS}:
            if self.wetting_phase == FluidPhase.OIL:
                dkro = self.get_wetting_phase_relative_permeability_derivative(
                    so, non_wetting_saturation=sg
                )
                dkrg = self.get_non_wetting_phase_relative_permeability_derivative(
                    so, non_wetting_saturation=sg
                )
                if self.reference_phase == "wetting":
                    dkro_dsw, dkro_dso, dkro_dsg = zeros, dkro, zeros
                    dkrg_dsw, dkrg_dso, dkrg_dsg = zeros, dkrg, zeros
                else:
                    dkro_dsw, dkro_dso, dkro_dsg = zeros, zeros, dkro
                    dkrg_dsw, dkrg_dso, dkrg_dsg = zeros, zeros, dkrg
            else:
                dkrg = self.get_wetting_phase_relative_permeability_derivative(
                    sg, non_wetting_saturation=so
                )
                dkro = self.get_non_wetting_phase_relative_permeability_derivative(
                    sg, non_wetting_saturation=so
                )
                if self.reference_phase == "wetting":
                    dkrg_dsw, dkrg_dso, dkrg_dsg = zeros, zeros, dkrg
                    dkro_dsw, dkro_dso, dkro_dsg = zeros, zeros, dkro
                else:
                    dkrg_dsw, dkrg_dso, dkrg_dsg = zeros, dkrg, zeros
                    dkro_dsw, dkro_dso, dkro_dsg = zeros, dkro, zeros

            results = (
                zeros,
                dkro_dsw,
                dkrg_dsw,
                zeros,
                dkro_dso,
                dkrg_dso,
                zeros,
                dkro_dsg,
                dkrg_dsg,
            )
            if is_scalar:
                results = tuple(dtype.type(row.item()) for row in results)  # type: ignore
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

        raise ValidationError(
            f"Cannot dispatch three-phase derivatives to a two-phase table with phases "
            f"{self.wetting_phase!r} / {self.non_wetting_phase!r}. "
            f"Expected OIL+WATER or OIL+GAS."
        )

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        satnum: int = 1,
        *,
        system: typing.Literal["oil_water", "gas_oil"],
        keyword_family: typing.Literal["first", "second"] = "first",
        number_of_base_points: int = 200,
        number_of_endpoint_extra_points: int = 20,
        spacing: Spacing = "cosine",
        min_wetting_relperm: MinimumRelPerm = None,
        min_non_wetting_relperm: MinimumRelPerm = None,
        dtype: npt.DTypeLike = None,
    ) -> Self:
        """
        Build a `TwoPhaseRelPermTable` for one saturation region from a `DeckFile`.

        **First keyword family** (`SWOF` / `SGOF`):

        The first family packs both relperm curves and capillary pressure into a
        single keyword. Only the relperm columns are consumed here; capillary
        pressure is handled by `TwoPhaseCapillaryPressureTable.from_deck`.

        - Oil-water: reads `SWOF` -> `(sw, krw, krow)`. Reference phase is `"wetting"`
          (Sw axis), wetting phase is WATER, non-wetting is OIL.
        - Gas-oil:   reads `SGOF` -> `(sg, krg, krog)`. Reference phase is `"non_wetting"`
          (Sg axis), wetting phase is OIL, non-wetting is GAS.

        **Second keyword family** (`SWFN` / `SGFN` + `SOF2` / `SOF3`):

        The second family separates water/gas curves from the oil curve.

        - Oil-water: reads `SWFN` -> `(sw, krw)` for krw. For krow, reads `SOF2`
          (two-phase) or `SOF3` (three-phase, takes `krow` column). The So axis of
          `SOF2`/`SOF3` is inverted to Sw using `Sw = 1 - So` before building the
          table, so the final table is indexed by Sw matching the `SWFN` convention.
        - Gas-oil:   reads `SGFN` -> `(sg, krg)` for krg. For krog, reads `SOF3`
          (takes `krog` column). The So axis of `SOF3` is inverted to Sg using
          `Sg = 1 - Sw_connate - So`.

        When `SOF3` is absent but `SOF2` is present and `system="gas_oil"`, a
        warning is raised and `krog` defaults to the `kro` column of `SOF2`.

        :param deck_file: Parsed `DeckFile` containing `PROPS`-section keywords.
        :param satnum: 1-based saturation region number (default region = 1).
            Region index is given as `region_index = max(satnum - 1, 0)`.
        :param system: `"oil_water"` or `"gas_oil"`.
        :param keyword_family: `"first"` (`SWOF`/`SGOF`) or `"second"` (`SWFN`/`SGFN`+`SOF`).
        :param number_of_base_points: Passed to PCHIP grid scaling.
        :param number_of_endpoint_extra_points: Passed to PCHIP endpoint enrichment.
        :param spacing: Grid spacing mode for PCHIP scaling.
        :param min_wetting_relperm: Min-value sentinel for wetting phase kr.
        :param min_non_wetting_relperm: Min-value sentinel for non-wetting phase kr.
        :param dtype: Array dtype for all stored arrays and query returns.
        :returns: `TwoPhaseRelPermTable` for the specified region and system.
        :raises ValidationError: When required keywords are missing or the region
            index is out of range.
        """
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        region_index = max(satnum - 1, 0)

        def get_rows(keyword: str) -> list[dict[str, typing.Any]]:
            """Extract rows for `region_index` from a deck keyword, or return []."""
            all_regions = deck_file.get(keyword)
            if all_regions is None:
                return []
            if region_index >= len(all_regions):
                return []
            return all_regions[region_index]

        def require(keyword: str) -> list[dict[str, typing.Any]]:
            rows = get_rows(keyword)
            if not rows:
                raise ValidationError(
                    f"Keyword `{keyword}` not found or region index {region_index} "
                    f"is out of range in the provided `DeckFile`."
                )
            return rows

        if system == "oil_water":
            if keyword_family == "first":
                # SWOF: (sw, krw, krow, pcow) - take sw, krw, krow
                rows = require("SWOF")
                sw = np.array([row["sw"] for row in rows], dtype=dtype)
                krw = np.array([row["krw"] for row in rows], dtype=dtype)
                krow = np.array([row["krow"] for row in rows], dtype=dtype)
                return cls(
                    wetting_phase=FluidPhase.WATER,
                    non_wetting_phase=FluidPhase.OIL,
                    reference_saturation=sw,  # type: ignore[arg-type]
                    wetting_phase_relative_permeability=krw,  # type: ignore[arg-type]
                    non_wetting_phase_relative_permeability=krow,  # type: ignore[arg-type]
                    reference_phase="wetting",
                    min_wetting_relperm=min_wetting_relperm,
                    min_non_wetting_relperm=min_non_wetting_relperm,
                    number_of_base_points=number_of_base_points,
                    number_of_endpoint_extra_points=number_of_endpoint_extra_points,
                    spacing=spacing,
                    dtype=dtype,
                )
            else:
                # Second family: SWFN for krw, SOF3 (preferred) or SOF2 for krow
                swfn_rows = require("SWFN")
                sw = np.array([row["sw"] for row in swfn_rows], dtype=dtype)
                krw = np.array([row["krw"] for row in swfn_rows], dtype=dtype)

                sof3_rows = get_rows("SOF3")
                sof2_rows = get_rows("SOF2")

                if sof3_rows:
                    # SOF3: (so, krow, krog) - take krow column, invert So -> Sw
                    so = np.array([row["so"] for row in sof3_rows], dtype=dtype)
                    krow = np.array([row["krow"] for row in sof3_rows], dtype=dtype)
                elif sof2_rows:
                    so = np.array([row["so"] for row in sof2_rows], dtype=dtype)
                    krow = np.array([row["kro"] for row in sof2_rows], dtype=dtype)
                else:
                    raise ValidationError(
                        "Second saturation-function family requires `SOF2` or `SOF3` for "
                        "the oil relative permeability column (krow). "
                        "Neither was found in the `DeckFile`."
                    )

                # Invert So -> Sw: Sw = 1 - So, then reverse arrays so Sw is ascending
                sw_from_so = (dtype.type(1) - so)[::-1]  # type: ignore[attr-defined]
                krow_on_sw = krow[::-1]

                # Interpolate krow onto the SWFN Sw axis
                krow_interp = np.interp(sw, sw_from_so, krow_on_sw).astype(dtype, copy=False)
                return cls(
                    wetting_phase=FluidPhase.WATER,
                    non_wetting_phase=FluidPhase.OIL,
                    reference_saturation=sw,  # type: ignore[arg-type]
                    wetting_phase_relative_permeability=krw,  # type: ignore[arg-type]
                    non_wetting_phase_relative_permeability=krow_interp,  # type: ignore[arg-type]
                    reference_phase="wetting",
                    min_wetting_relperm=min_wetting_relperm,
                    min_non_wetting_relperm=min_non_wetting_relperm,
                    number_of_base_points=number_of_base_points,
                    number_of_endpoint_extra_points=number_of_endpoint_extra_points,
                    spacing=spacing,
                    dtype=dtype,
                )

        elif system == "gas_oil":
            if keyword_family == "first":
                # SGOF: (sg, krg, krog, pcog) - take sg, krg, krog
                rows = require("SGOF")
                sg = np.array([row["sg"] for row in rows], dtype=dtype)
                krg = np.array([row["krg"] for row in rows], dtype=dtype)
                krog = np.array([row["krog"] for row in rows], dtype=dtype)
                # SGOF is indexed by Sg: wetting is OIL, non-wetting is GAS,
                # reference_phase="non_wetting" because the axis is Sg
                return cls(
                    wetting_phase=FluidPhase.OIL,
                    non_wetting_phase=FluidPhase.GAS,
                    reference_saturation=sg,  # type: ignore[arg-type]
                    wetting_phase_relative_permeability=krog,  # type: ignore[arg-type]
                    non_wetting_phase_relative_permeability=krg,  # type: ignore[arg-type]
                    reference_phase="non_wetting",
                    min_wetting_relperm=min_wetting_relperm,
                    min_non_wetting_relperm=min_non_wetting_relperm,
                    number_of_base_points=number_of_base_points,
                    number_of_endpoint_extra_points=number_of_endpoint_extra_points,
                    spacing=spacing,
                    dtype=dtype,
                )
            else:
                # Second family: SGFN for krg, SOF3 (preferred) or SOF2 for krog
                sgfn_rows = require("SGFN")
                sg = np.array([row["sg"] for row in sgfn_rows], dtype=dtype)
                krg = np.array([row["krg"] for row in sgfn_rows], dtype=dtype)

                sof3_rows = get_rows("SOF3")
                sof2_rows = get_rows("SOF2")

                if sof3_rows:
                    so = np.array([row["so"] for row in sof3_rows], dtype=dtype)
                    krog = np.array([row["krog"] for row in sof3_rows], dtype=dtype)
                elif sof2_rows:
                    warnings.warn(
                        "`SOF3` not found; using `SOF2` `kro` column for `krog` in gas-oil "
                        "table. This is only appropriate for two-phase (gas-oil) runs.",
                        UserWarning,
                        stacklevel=2,
                    )
                    so = np.array([row["so"] for row in sof2_rows], dtype=dtype)
                    krog = np.array([row["kro"] for row in sof2_rows], dtype=dtype)
                else:
                    raise ValidationError(
                        "Second saturation-function family requires `SOF3` for the gas-oil "
                        "krog column. `SOF3` (and fallback `SOF2`) not found in the `DeckFile`."
                    )

                # Invert So -> Sg: Sg = 1 - swc - So
                # Without swc available here, use Sg = 1 - So as an approximation.
                # Callers that know swc should build the table manually or pass connate
                # water saturation as a column offset.
                sg_from_so = (dtype.type(1) - so)[::-1]  # type: ignore[attr-defined]
                krog_on_sg = krog[::-1]

                krog_interp = np.interp(sg, sg_from_so, krog_on_sg).astype(dtype, copy=False)
                return cls(
                    wetting_phase=FluidPhase.OIL,
                    non_wetting_phase=FluidPhase.GAS,
                    reference_saturation=sg,  # type: ignore[arg-type]
                    wetting_phase_relative_permeability=krog_interp,  # type: ignore[arg-type]
                    non_wetting_phase_relative_permeability=krg,  # type: ignore[arg-type]
                    reference_phase="non_wetting",
                    min_wetting_relperm=min_wetting_relperm,
                    min_non_wetting_relperm=min_non_wetting_relperm,
                    number_of_base_points=number_of_base_points,
                    number_of_endpoint_extra_points=number_of_endpoint_extra_points,
                    spacing=spacing,
                    dtype=dtype,
                )
        raise ValidationError(f"`system` must be 'oil_water' or 'gas_oil'; got {system!r}.")


@relperm_table
@attrs.frozen(slots=True)
class ThreePhaseRelPermTable(
    RelativePermeabilityTable,
    serializers={"mixing_rule": serialize_mixing_rule},
    deserializers={"mixing_rule": deserialize_mixing_rule},
    load_exclude={"supports_vector"},
    dump_exclude={"supports_vector"},
):
    """
    Three-phase relative permeability lookup table, with mixing rules.

    Interpolates relative permeabilities for water, oil, and gas based on
    saturation values. Uses two `TwoPhaseRelPermTable` instances (oil-water
    and gas-oil) and a mixing rule for oil in the three-phase system.

    Each two-phase table declares its own `reference_phase` ("wetting" or
    "non_wetting"), so the correct saturation is dispatched automatically -
    no assumptions are hard-coded about whether a table is indexed by So or Sg.

    **dtype**:

    The `oil_water_table` and `gas_oil_table` must share the same dtype.
    This is validated at construction time. All outputs (kr values and
    derivatives) are cast to that shared dtype. The mixing rule result is
    also cast to the shared dtype before being returned.

    Minimum relperm `min_values` on the two-phase tables propagate automatically
    into this three-phase table since the two-phase table methods apply the
    min_value before returning. For the mixing-rule chain rule the min_value
    derivative (zeroed in the flat region) is also used consistently, so the
    assembled Jacobian never disagrees with the kr values.

    Supported mixing rules: `max_rule`, `stone_I_rule`, `stone_II_rule`, etc.
    """

    __type__ = "three_phase_relperm_table"

    oil_water_table: TwoPhaseRelPermTable
    """Relative permeability table for the oil-water system."""

    gas_oil_table: TwoPhaseRelPermTable
    """Relative permeability table for the gas-oil system."""

    mixing_rule: MixingRule | str | None = None
    """
    Mixing rule function or name to compute oil relative permeability in the
    three-phase system. Signature:

        rule(kro_w, kro_g, krw, krg, kr_max,
             water_saturation, oil_saturation, gas_saturation) -> kro

    If `None`, falls back to `min(kro_w, kro_g)`.
    """

    supports_vector: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array inputs."""

    def __attrs_post_init__(self) -> None:
        if {
            self.oil_water_table.wetting_phase,
            self.oil_water_table.non_wetting_phase,
        } != {FluidPhase.WATER, FluidPhase.OIL}:
            raise ValidationError("`oil_water_table` must be between water and oil phases.")

        if {
            self.gas_oil_table.wetting_phase,
            self.gas_oil_table.non_wetting_phase,
        } != {FluidPhase.OIL, FluidPhase.GAS}:
            raise ValidationError("`gas_oil_table` must be between oil and gas phases.")

        # Validate matching dtype between the two sub-tables
        ow_dtype = np.dtype(self.oil_water_table.dtype)
        go_dtype = np.dtype(self.gas_oil_table.dtype)
        if ow_dtype != go_dtype:
            raise ValidationError(
                f"`oil_water_table` dtype ({ow_dtype}) and `gas_oil_table` dtype "
                f"({go_dtype}) must match. Convert one of the tables before combining."
            )

        mixing_rule = self.mixing_rule
        if isinstance(mixing_rule, str):
            object.__setattr__(self, "mixing_rule", get_mixing_rule(mixing_rule))

    @property
    def dtype(self) -> np.dtype:
        """Shared `dtype` of the two underlying two-phase tables."""
        return np.dtype(self.oil_water_table.dtype)

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        return self.oil_water_table.wetting_phase  # type: ignore[return-value]

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        return self.gas_oil_table.wetting_phase  # type: ignore[return-value]

    def get_oil_relperm_endpoint(self) -> Number:
        return self.oil_water_table.get_oil_relperm_endpoint()

    def get_water_relperm_endpoint(self) -> Number:
        return self.oil_water_table.get_water_relperm_endpoint()

    def get_gas_relperm_endpoint(self) -> Number:
        return self.gas_oil_table.get_gas_relperm_endpoint()

    def get_connate_water_saturation(self) -> Number:
        return self.oil_water_table.get_connate_water_saturation()

    def get_residual_oil_saturation_water(self) -> Number:
        return self.oil_water_table.get_residual_oil_saturation_water()

    def get_residual_gas_saturation(self) -> Number:
        return self.gas_oil_table.get_residual_gas_saturation()

    def get_residual_oil_saturation_gas(self) -> Number:
        """
        Residual oil saturation after gas flood (sorg), combining the
        gas-oil table's own gas-saturation axis with the oil-water table's
        connate water saturation: `sorg = 1 - swco - Sg_max`. This is more
        accurate than `gas_oil_table.get_residual_oil_saturation_gas()`
        alone, which (used standalone) has no connate-water information and
        implicitly assumes `swco = 0`.
        """
        gas_oil = self.gas_oil_table
        phases = {gas_oil.wetting_phase, gas_oil.non_wetting_phase}  # type: ignore[union-attr]
        if FluidPhase.GAS not in phases or not isinstance(gas_oil, TwoPhaseRelPermTable):
            return gas_oil.get_residual_oil_saturation_gas()
        axis_phase = (
            gas_oil.wetting_phase
            if gas_oil.reference_phase == "wetting"
            else gas_oil.non_wetting_phase
        )
        if axis_phase != FluidPhase.GAS:
            return gas_oil.get_residual_oil_saturation_gas()
        swco = self.oil_water_table.get_connate_water_saturation()
        sg_max = float(gas_oil.reference_saturation[-1])
        return max(0.0, 1.0 - swco - sg_max)

    def evaluate(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for water, oil, and gas.

        Each two-phase table is queried using its declared `reference_phase`:

        - `reference_phase="wetting"` - the wetting phase saturation is passed.
        - `reference_phase="non_wetting"` - the non-wetting phase saturation is passed.

        Minimum relperm `min_values` declared on the two-phase tables are applied
        automatically inside the table query methods and propagate into the
        mixing-rule inputs transparently. All returned values are cast to the
        shared `self.dtype`.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :return: `RelativePermeabilities` dict with keys `"water"`, `"oil"`, `"gas"` in `self.dtype`.
        """
        dtype = self.dtype
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
            raise ValidationError(
                f"Saturations must be between 0 and 1. "
                f"Sw: {show_invalid_saturation(sw)}, "
                f"So: {show_invalid_saturation(so)}, "
                f"Sg: {show_invalid_saturation(sg)}"
            )

        total_saturation = sw + so + sg
        needs_norm = (np.abs(total_saturation - 1.0) > 1e-6) & (total_saturation > 0.0)
        if np.any(needs_norm):
            sw = np.where(needs_norm, sw / total_saturation, sw)
            so = np.where(needs_norm, so / total_saturation, so)
            sg = np.where(needs_norm, sg / total_saturation, sg)

        if self.oil_water_table.wetting_phase == FluidPhase.WATER:
            krw = self.oil_water_table.get_wetting_phase_relative_permeability(
                sw, non_wetting_saturation=so
            )
            kro_w = self.oil_water_table.get_non_wetting_phase_relative_permeability(
                sw, non_wetting_saturation=so
            )
        else:
            kro_w = self.oil_water_table.get_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sw
            )
            krw = self.oil_water_table.get_non_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sw
            )

        if self.gas_oil_table.wetting_phase == FluidPhase.OIL:
            kro_g = self.gas_oil_table.get_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sg
            )
            krg = self.gas_oil_table.get_non_wetting_phase_relative_permeability(
                so, non_wetting_saturation=sg
            )
        else:
            krg = self.gas_oil_table.get_wetting_phase_relative_permeability(
                sg, non_wetting_saturation=so
            )
            kro_g = self.gas_oil_table.get_non_wetting_phase_relative_permeability(
                sg, non_wetting_saturation=so
            )

        kr_max = self.get_oil_relperm_endpoint()
        if self.mixing_rule is not None:
            kro = self.mixing_rule(  # type: ignore
                kro_w=kro_w,
                kro_g=kro_g,
                krw=krw,
                krg=krg,
                kr_max=kr_max,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
            )
        else:
            kro = np.minimum(kro_w, kro_g)

        # Cast mixing rule output to shared dtype
        kro = kro.astype(dtype, copy=False)  # type: ignore[attr-defined]

        if is_scalar:
            return RelativePermeabilities(
                water=dtype.type(krw.item()),  # type: ignore[attr-defined]
                oil=dtype.type(kro.item()),
                gas=dtype.type(krg.item()),  # type: ignore[attr-defined]
            )
        return RelativePermeabilities(
            water=krw.astype(dtype, copy=False),  # type: ignore[attr-defined]
            oil=kro,
            gas=krg.astype(dtype, copy=False),  # type: ignore[attr-defined]
        )

    def derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute all nine partial derivatives of the three-phase relative
        permeabilities with respect to water saturation, oil saturation, and
        gas saturation.

        Returns a dictionary containing:

            dkrw_dsw, dkro_dsw, dkrg_dsw,
            dkrw_dso, dkro_dso, dkrg_dso,
            dkrw_dsg, dkro_dsg, dkrg_dsg

        Water and gas relative permeability derivatives are computed analytically
        from the slopes of the underlying two-phase lookup tables. Where a
        minimum relperm min_value is active on a two-phase table, the derivative
        returned by that table is already zeroed out in the min_value region,
        so the assembled nine-derivative Jacobian is automatically consistent
        with the min_value kr values.

        Oil relative permeability derivatives use the full extended chain rule
        through the three-phase mixing rule:

            dkro/dSalpha = (dkro/dkro_w) * (dkro_w/d_Salpha)
                         + (dkro/dkro_g) * (dkro_g/d_Salpha)
                         + (dkro/dkrw)   * (dkrw/d_Salpha)
                         + (dkro/dkrg)   * (dkrg/d_Salpha)
                         + (dkro/d_Salpha)_explicit_in_mixing_rule

        All returned values are cast to `self.dtype`.

        :param water_saturation: Water saturation (fraction, 0 to 1).
        :param oil_saturation: Oil saturation (fraction, 0 to 1).
        :param gas_saturation: Gas saturation (fraction, 0 to 1).
        :return: `RelativePermeabilityDerivatives` dictionary with all values in `self.dtype`.
        """
        dtype = self.dtype
        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )
        sw = np.atleast_1d(water_saturation)
        so = np.atleast_1d(oil_saturation)
        sg = np.atleast_1d(gas_saturation)
        sw, so, sg = np.broadcast_arrays(sw, so, sg)
        zeros = np.zeros(sw.shape, dtype=dtype)

        oil_water_table = self.oil_water_table
        gas_oil_table = self.gas_oil_table

        # Oil-water table derivatives: krw and kro_w
        if oil_water_table.wetting_phase == FluidPhase.WATER:
            if oil_water_table.reference_phase == "wetting":
                dkrw_dsw = oil_water_table.get_wetting_phase_relative_permeability_derivative(
                    sw, non_wetting_saturation=so
                )
                dkrw_dso, dkrw_dsg = zeros.copy(), zeros.copy()
                dkro_w_dsw = (
                    oil_water_table.get_non_wetting_phase_relative_permeability_derivative(
                        sw, non_wetting_saturation=so
                    )
                )
                dkro_w_dso, dkro_w_dsg = zeros.copy(), zeros.copy()
            else:
                dkrw_dsw, dkrw_dsg = zeros.copy(), zeros.copy()
                dkrw_dso = oil_water_table.get_wetting_phase_relative_permeability_derivative(
                    sw, non_wetting_saturation=so
                )
                dkro_w_dsw, dkro_w_dsg = zeros.copy(), zeros.copy()
                dkro_w_dso = (
                    oil_water_table.get_non_wetting_phase_relative_permeability_derivative(
                        sw, non_wetting_saturation=so
                    )
                )
        else:
            if oil_water_table.reference_phase == "wetting":
                dkrw_dsw, dkrw_dsg = zeros.copy(), zeros.copy()
                dkrw_dso = oil_water_table.get_non_wetting_phase_relative_permeability_derivative(
                    so, non_wetting_saturation=sw
                )
                dkro_w_dsw, dkro_w_dsg = zeros.copy(), zeros.copy()
                dkro_w_dso = oil_water_table.get_wetting_phase_relative_permeability_derivative(
                    so, non_wetting_saturation=sw
                )
            else:
                dkrw_dso, dkrw_dsg = zeros.copy(), zeros.copy()
                dkrw_dsw = oil_water_table.get_non_wetting_phase_relative_permeability_derivative(
                    so, non_wetting_saturation=sw
                )
                dkro_w_dso, dkro_w_dsg = zeros.copy(), zeros.copy()
                dkro_w_dsw = oil_water_table.get_wetting_phase_relative_permeability_derivative(
                    so, non_wetting_saturation=sw
                )

        # Gas-oil table derivatives: krg and kro_g
        if gas_oil_table.wetting_phase == FluidPhase.OIL:
            if gas_oil_table.reference_phase == "wetting":
                dkrg_dsw, dkrg_dsg = zeros.copy(), zeros.copy()
                dkrg_dso = gas_oil_table.get_non_wetting_phase_relative_permeability_derivative(
                    so, non_wetting_saturation=sg
                )
                dkro_g_dsw, dkro_g_dsg = zeros.copy(), zeros.copy()
                dkro_g_dso = gas_oil_table.get_wetting_phase_relative_permeability_derivative(
                    so, non_wetting_saturation=sg
                )
            else:
                dkrg_dsw, dkrg_dso = zeros.copy(), zeros.copy()
                dkrg_dsg = gas_oil_table.get_non_wetting_phase_relative_permeability_derivative(
                    so, non_wetting_saturation=sg
                )
                dkro_g_dsw, dkro_g_dso = zeros.copy(), zeros.copy()
                dkro_g_dsg = gas_oil_table.get_wetting_phase_relative_permeability_derivative(
                    so, non_wetting_saturation=sg
                )
        else:
            if gas_oil_table.reference_phase == "wetting":
                dkrg_dsw, dkrg_dso = zeros.copy(), zeros.copy()
                dkrg_dsg = gas_oil_table.get_wetting_phase_relative_permeability_derivative(
                    sg, non_wetting_saturation=so
                )
                dkro_g_dsw, dkro_g_dso = zeros.copy(), zeros.copy()
                dkro_g_dsg = gas_oil_table.get_non_wetting_phase_relative_permeability_derivative(
                    sg, non_wetting_saturation=so
                )
            else:
                dkrg_dsw, dkrg_dsg = zeros.copy(), zeros.copy()
                dkrg_dso = gas_oil_table.get_wetting_phase_relative_permeability_derivative(
                    sg, non_wetting_saturation=so
                )
                dkro_g_dsw, dkro_g_dsg = zeros.copy(), zeros.copy()
                dkro_g_dso = gas_oil_table.get_non_wetting_phase_relative_permeability_derivative(
                    sg, non_wetting_saturation=so
                )

        # Forward two-phase kr values for mixing rule
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

        kr_max = self.get_oil_relperm_endpoint()
        # Mixing rule partial derivatives
        mixing_rule = typing.cast(MixingRule | None, self.mixing_rule)
        if mixing_rule is None:
            dkro_dkro_w = np.where(kro_w <= kro_g, 1.0, 0.0)
            dkro_dkro_g = np.where(kro_g <= kro_w, 1.0, 0.0)
            dkro_dkrw = zeros.copy()
            dkro_dkrg = zeros.copy()
            dkro_dsw_explicit = zeros.copy()
            dkro_dso_explicit = zeros.copy()
            dkro_dsg_explicit = zeros.copy()
        else:
            derivatives = get_mixing_rule_partial_derivatives(
                rule=mixing_rule,  # type: ignore[arg-type]
                kro_w=kro_w,
                kro_g=kro_g,
                krw=krw,
                krg=krg,
                kr_max=kr_max,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
                epsilon=c.FINITE_DIFFERENCE_EPSILON,
            )
            dkro_dkro_w = derivatives["dkro_dkro_w"]
            dkro_dkro_g = derivatives["dkro_dkro_g"]
            dkro_dkrw = derivatives["dkro_dkrw"]
            dkro_dkrg = derivatives["dkro_dkrg"]
            dkro_dsw_explicit = derivatives["dkro_dsw_explicit"]
            dkro_dso_explicit = derivatives["dkro_dso_explicit"]
            dkro_dsg_explicit = derivatives["dkro_dsg_explicit"]

        # Extended chain rule - cast all results to shared dtype
        dkro_dsw = (
            dkro_dkro_w * dkro_w_dsw
            + dkro_dkro_g * dkro_g_dsw
            + dkro_dkrw * dkrw_dsw
            + dkro_dkrg * dkrg_dsw
            + dkro_dsw_explicit
        ).astype(dtype, copy=False)  # type: ignore[attr-defined]
        dkro_dso = (
            dkro_dkro_w * dkro_w_dso
            + dkro_dkro_g * dkro_g_dso
            + dkro_dkrw * dkrw_dso
            + dkro_dkrg * dkrg_dso
            + dkro_dso_explicit
        ).astype(dtype, copy=False)  # type: ignore[attr-defined]
        dkro_dsg = (
            dkro_dkro_w * dkro_w_dsg
            + dkro_dkro_g * dkro_g_dsg
            + dkro_dkrw * dkrw_dsg
            + dkro_dkrg * dkrg_dsg
            + dkro_dsg_explicit
        ).astype(dtype, copy=False)  # type: ignore[attr-defined]

        results = (
            dkrw_dsw.astype(dtype, copy=False),  # type: ignore[attr-defined]
            dkro_dsw,
            dkrg_dsw.astype(dtype, copy=False),  # type: ignore[attr-defined]
            dkrw_dso.astype(dtype, copy=False),  # type: ignore[attr-defined]
            dkro_dso,
            dkrg_dso.astype(dtype, copy=False),  # type: ignore[attr-defined]
            dkrw_dsg.astype(dtype, copy=False),  # type: ignore[attr-defined]
            dkro_dsg,
            dkrg_dsg.astype(dtype, copy=False),  # type: ignore[attr-defined]
        )

        if is_scalar:
            results = tuple(dtype.type(row.item()) for row in results)  # type: ignore
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

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        satnum: int = 1,
        *,
        mixing_rule: MixingRule | str | None = None,
        keyword_family: typing.Literal["first", "second", "auto"] = "auto",
        number_of_base_points: int = 200,
        number_of_endpoint_extra_points: int = 20,
        spacing: Spacing = "cosine",
        min_wetting_relperm: MinimumRelPerm = None,
        min_non_wetting_relperm: MinimumRelPerm = None,
        dtype: npt.DTypeLike = None,
    ) -> "ThreePhaseRelPermTable":
        """
        Build a `ThreePhaseRelPermTable` for one saturation region from a `DeckFile`.

        Detects which Eclipse saturation-function keyword family is present and
        builds the oil-water and gas-oil two-phase sub-tables automatically:

        **First family** (detected when `SWOF` or `SGOF` is present):

        `SWOF` supplies the oil-water system `(sw, krw, krow)`.
        `SGOF` supplies the gas-oil system `(sg, krg, krog)`.

        Both keywords must be present for a three-phase table. If only one
        is found a warning is issued and a `TwoPhaseRelPermTable` should be
        used instead.

        **Second family** (detected when `SWFN` or `SGFN` is present):

        `SWFN` + `SOF3` supplies the oil-water system.
        `SGFN` + `SOF3` supplies the gas-oil system.

        `SOF3` must be present and provide both `krow` and `krog` columns.
        Falls back to `SOF2` for the oil-water krow column with a warning.

        **`auto` detection** (default):

        Checks for `SWOF`/`SGOF` first (first family), then `SWFN`/`SGFN`
        (second family). Raises `ValidationError` when neither is found.

        When the deck does not contain enough keywords for a full three-phase
        table (e.g. only `SWOF` without `SGOF`), a warning is emitted
        explaining which keyword is missing and how to build a two-phase table
        instead.

        :param deck_file: Parsed `DeckFile` containing `PROPS`-section keywords.
        :param region_index: 1-based saturation region number (default region = 1).
            Region index is given as `region_index = max(satnum - 1, 0)`.
        :param mixing_rule: Three-phase oil mixing rule (name or callable).
            `None` falls back to `min(kro_w, kro_g)`.
        :param keyword_family: `"first"`, `"second"`, or `"auto"` (default).
        :param number_of_base_points: Passed to PCHIP grid scaling.
        :param number_of_endpoint_extra_points: Passed to PCHIP endpoint enrichment.
        :param spacing: Grid spacing mode for PCHIP scaling.
        :param min_wetting_relperm: Min-value sentinel for wetting phase kr in
            both sub-tables.
        :param min_non_wetting_relperm: Min-value sentinel for non-wetting phase kr
            in both sub-tables.
        :param dtype: Array dtype shared by both sub-tables and all query returns.
        :returns: `ThreePhaseRelPermTable` for the specified region.
        :raises ValidationError: When required keywords are missing.
        """
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        region_index = max(satnum - 1, 0)

        def has(keyword: str) -> bool:
            if not deck_file.has(keyword):
                return False

            all_regions = deck_file.get(keyword)
            assert all_regions is not None
            return region_index < len(all_regions) and bool(all_regions[region_index])

        shared_kwargs: dict[str, typing.Any] = dict(
            satnum=satnum,
            number_of_base_points=number_of_base_points,
            number_of_endpoint_extra_points=number_of_endpoint_extra_points,
            spacing=spacing,
            min_wetting_relperm=min_wetting_relperm,
            min_non_wetting_relperm=min_non_wetting_relperm,
            dtype=dtype,
        )

        # Resolve keyword family
        family: typing.Literal["first", "second"]
        if keyword_family == "auto":
            if has("SWOF") or has("SGOF"):
                family = "first"
            elif has("SWFN") or has("SGFN"):
                family = "second"
            else:
                raise ValidationError(
                    "No recognised saturation-function keywords found in the DeckFile "
                    f"for `SATNUM` {satnum}. Expected one of: "
                    "`SWOF`, `SGOF` (first family) or `SWFN`, `SGFN` (second family)."
                )
        else:
            family = keyword_family  # type: ignore[assignment]

        # Check availability of both systems and warn clearly when one is missing
        if family == "first":
            has_oil_water_table = has("SWOF")
            has_gas_oil_table = has("SGOF")
        else:
            has_oil_water_table = has("SWFN") and (has("SOF3") or has("SOF2"))
            has_gas_oil_table = has("SGFN") and has("SOF3")

        if not has_oil_water_table:
            oil_water_keyword = "`SWOF`" if family == "first" else "`SWFN` + `SOF3`/`SOF2`"
            warnings.warn(
                f"Oil-water keyword(s) `{oil_water_keyword}` not found for `SATNUM` "
                f"{satnum}. Cannot build a three-phase table. "
                "Use `TwoPhaseRelPermTable.from_deck(..., system='gas_oil')` "
                "to build a gas-oil only table.",
                UserWarning,
                stacklevel=2,
            )
            raise ValidationError(
                f"Oil-water keyword(s) `{oil_water_keyword}` required for {cls.__name__} "
                f"not found at SATNUM {satnum}."
            )

        if not has_gas_oil_table:
            gas_oil_keyword = "`SGOF`" if family == "first" else "`SGFN` + `SOF3`"
            warnings.warn(
                f"Gas-oil keyword(s) `{gas_oil_keyword}` not found for `SATNUM` "
                f"{satnum}. Cannot build a three-phase table. "
                "Use `TwoPhaseRelPermTable.from_deck(..., system='oil_water')` "
                "to build an oil-water only table.",
                UserWarning,
                stacklevel=2,
            )
            raise ValidationError(
                f"Gas-oil keyword(s) `{gas_oil_keyword}` required for {cls.__name__} "
                f"not found at `SATNUM` {satnum}."
            )

        oil_water_table = TwoPhaseRelPermTable.from_deck(
            deck_file=deck_file,
            system="oil_water",
            keyword_family=family,
            **shared_kwargs,
        )
        gas_oil_table = TwoPhaseRelPermTable.from_deck(
            deck_file=deck_file,
            system="gas_oil",
            keyword_family=family,
            **shared_kwargs,
        )
        return cls(
            oil_water_table=oil_water_table, gas_oil_table=gas_oil_table, mixing_rule=mixing_rule
        )

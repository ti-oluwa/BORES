"""Base capillary pressure tables for multi-phase flow simulations."""

import threading
import typing
import warnings

import attrs
import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator
from typing_extensions import Self

from bores.blackoil.satfunc.utils import build_pchip_interpolant
from bores.constants import UnitConversionTable, get_conversion_factors
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.precision import get_dtype
from bores.serde.registry import make_serializable_type_registrar
from bores.serde.stores import StoreSerializable
from bores.types import (
    CapillaryPressureDerivatives,
    CapillaryPressures,
    FluidPhase,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
    Spacing,
    UnitSystem,
)

__all__ = [
    "ThreePhaseCapillaryPressureTable",
    "TwoPhaseCapillaryPressureTable",
    "capillary_pressure_table",
]


class CapillaryPressureTable(StoreSerializable):
    """
    Protocol for a capillary pressure model that computes
    capillary pressures based on fluid saturations.
    """

    __abstract_serializable__ = True

    unit_system: UnitSystem

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        return FluidPhase.WATER

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        return FluidPhase.OIL

    def evaluate(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures for three-phase system.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :return: `CapillaryPressures` dictionary.
        """
        raise NotImplementedError

    def derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> CapillaryPressureDerivatives:
        """
        Compute capillary pressure derivatives for three-phase system.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :return: `CapillaryPressureDerivatives` dictionary containing the partial derivatives as described above.
        """
        raise NotImplementedError

    def __call__(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Computes capillary pressures based on fluid saturations.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :return: `CapillaryPressures` dictionary.
        """
        return self.evaluate(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        raise NotImplementedError


CAPILLARY_PRESSURE_TABLES: dict[str, type[CapillaryPressureTable]] = {}
"""Registry for capillary pressure table types."""
_capillary_pressure_table_lock = threading.Lock()
capillary_pressure_table = make_serializable_type_registrar(
    base_cls=CapillaryPressureTable,
    registry=CAPILLARY_PRESSURE_TABLES,
    key_attr="__type__",
    lock=_capillary_pressure_table_lock,
    override=False,
    auto_register_serializer=True,
    auto_register_deserializer=True,
)


def list_capillary_pressure_tables() -> list[str]:
    """
    List all registered capillary pressure table types.

    :return: List of capillary pressure table type names.
    """
    with _capillary_pressure_table_lock:
        return list(CAPILLARY_PRESSURE_TABLES.keys())


def get_capillary_pressure_table(name: str) -> type[CapillaryPressureTable]:
    """
    Get a registered capillary pressure table type by name.

    :param name: Name of the capillary pressure table type.
    :return: Capillary pressure table class.
    :raises KeyError: If the type name is not registered.
    """
    with _capillary_pressure_table_lock:
        if name not in CAPILLARY_PRESSURE_TABLES:
            raise ValidationError(
                f"Capillary pressure table type '{name}' is not registered. "
                f"Use `@capillary_pressure_table` to register it. "
                f"Available types: {list(CAPILLARY_PRESSURE_TABLES.keys())}"
            )
        return CAPILLARY_PRESSURE_TABLES[name]


@capillary_pressure_table
@attrs.frozen(slots=True)
class TwoPhaseCapillaryPressureTable(
    CapillaryPressureTable,
    load_exclude={"_interp", "_d_interp"},
    dump_exclude={"_interp", "_d_interp"},
):
    """
    Two-phase capillary pressure lookup table backed by a PCHIP interpolant.

    Interpolates capillary pressure for two fluid phases based on a
    **reference saturation** value. The reference saturation can be either
    the wetting or non-wetting phase saturation, depending on how the table
    was constructed, e.g. a gas-oil table may be indexed by oil saturation
    (wetting) or by gas saturation (non-wetting).

    Supports both scalar and array inputs.

    **Grid scaling** (`number_of_base_points` / `number_of_endpoint_extra_points`):

    Identical semantics to `TwoPhaseRelPermTable`. The default
    `number_of_endpoint_extra_points=30` (vs 20 for relperm) reflects that Pc curves are
    typically unbounded near residual saturation, making endpoint fidelity
    especially important for implicit convergence. Pass `number_of_base_points=0`
    to disable scaling.

    **dtype**:

    All stored arrays and all returned scalars / arrays use `dtype`. Query
    methods always cast their output to `dtype` before returning. Defaults to
    `get_dtype()` when not specified.

    **unit_system**:

    `capillary_pressure` is dimensional (pressure units) and is stored in
    `unit_system`. `reference_saturation` is always dimensionless and is
    unaffected by unit conversion. Use `convert(target)` to produce a copy of
    this table rescaled to a different `UnitSystem`.

    **Standalone three-phase use**:

    Subclasses `CapillaryPressureTable` and is registered under
    `__type__ = "two_phase_capillary_pressure_table"`, so it can be used
    anywhere a full `CapillaryPressureTable` is expected - e.g. directly as
    `RockFluidRegion.capillary_pressure` - without wrapping it in a
    `ThreePhaseCapillaryPressureTable` first. `evaluate`/`derivatives` take
    the standard three-phase `(water_saturation, oil_saturation,
    gas_saturation)` signature and always return both `"oil_water"` and
    `"gas_oil"` entries, zeroing whichever phase pair this table doesn't
    cover (mirrors `TwoPhaseRelPermTable`).

    `__call__` is inherited from `CapillaryPressureTable` and therefore
    takes that same three-phase signature (forwarding to `evaluate`), not
    the two-argument `(wetting_saturation, non_wetting_saturation)` form.
    For direct single-value queries against this table's own reference
    axis, use `get_capillary_pressure`/`get_capillary_pressure_derivative`
    instead.
    """

    __type__ = "two_phase_capillary_pressure_table"

    wetting_phase: FluidPhase | str = attrs.field(converter=FluidPhase)
    """The wetting fluid phase, e.g. WATER (oil-water system) or OIL (gas-oil system)."""

    non_wetting_phase: FluidPhase | str = attrs.field(converter=FluidPhase)
    """The non-wetting fluid phase, e.g. OIL (oil-water system) or GAS (gas-oil system)."""

    reference_saturation: NumberArray[OneDimension]
    """
    Saturation values used as the x-axis for interpolation, monotonically
    increasing. May represent either the wetting or non-wetting phase
    saturation depending on `reference_phase`. Dimensionless.
    """

    capillary_pressure: NumberArray[OneDimension]
    """
    Capillary pressure values `Pc = P_non_wetting - P_wetting` corresponding
    to each `reference_saturation` point. Units follow `unit_system`
    (psi / bar / atm / Pa).
    """

    reference_phase: typing.Literal["wetting", "non_wetting"] = attrs.field(default="wetting")
    """
    Which phase the `reference_saturation` axis represents.

    - `"wetting"` - the x-axis holds wetting-phase saturation values.
      This is the standard convention for oil-water tables (Sw axis) and for
      gas-oil tables indexed by So.
    - `"non_wetting"` - the x-axis holds non-wetting-phase saturation
      values.  Use this for gas-oil tables indexed by Sg.

    This attribute does not change the interpolation mechanics. It only
    records which physical saturation must be supplied by the caller so that
    `ThreePhaseCapillaryPressureTable` (and any other consumer) can dispatch
    the correct saturation without hard-coding assumptions.
    """

    number_of_base_points: int = attrs.field(default=200)
    """
    Target number of base knot points used when expanding the raw saturation
    grid before fitting the PCHIP interpolant.

    Pass `0` to disable grid scaling and use the raw knots directly.
    """

    number_of_endpoint_extra_points: int = attrs.field(default=30)
    """
    Number of extra knots injected into the first and last 10 % of the
    saturation range during grid expansion (see `number_of_base_points`).

    The higher default of 30 (vs 20 for relperm) reflects that Pc curves vary
    most steeply near residual saturations. Pass `0` to disable.
    """

    spacing: Spacing = attrs.field(default="cosine")
    """Grid spacing mode used when building the expanded knot grid."""

    unit_system: UnitSystem = attrs.field(default=UnitSystem.FIELD)
    """
    Unit system in which `capillary_pressure` is expressed.

    `reference_saturation` is dimensionless and is unaffected by unit
    conversion. Use `convert(target)` to rescale `capillary_pressure`
    (and the derivative interpolant) to another `UnitSystem`.
    """

    dtype: npt.DTypeLike | None = attrs.field(default=None)
    """
    Array dtype for all stored arrays and all query return values.

    Both `reference_saturation` and `capillary_pressure` are cast to this
    dtype in `__attrs_post_init__`. Query methods (`_query`,
    `_d_query`) cast their outputs to this dtype before returning.

    Defaults to `get_dtype()` when `None`.
    """

    _interp: PchipInterpolator = attrs.field(init=False, repr=False)
    _d_interp: PchipInterpolator = attrs.field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if self.reference_phase not in ("wetting", "non_wetting"):
            raise ValidationError(
                f"`reference_phase` must be 'wetting' or 'non_wetting', "
                f"got {self.reference_phase!r}"
            )
        if len(self.reference_saturation) != len(self.capillary_pressure):
            raise ValidationError(
                f"`reference_saturation` and `capillary_pressure` arrays must have "
                f"the same length.  Got {len(self.reference_saturation)} vs "
                f"{len(self.capillary_pressure)}"
            )
        if len(self.reference_saturation) < 2:
            raise ValidationError("At least 2 points required for interpolation.")
        if not np.all(np.diff(self.reference_saturation) >= 0):
            raise ValidationError("`reference_saturation` must be monotonically increasing.")

        # Resolve and enforce dtype on both stored arrays
        dtype = np.dtype(self.dtype) if self.dtype is not None else get_dtype()
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(
            self,
            "reference_saturation",
            np.asarray(self.reference_saturation, dtype=dtype),
        )
        object.__setattr__(
            self,
            "capillary_pressure",
            np.asarray(self.capillary_pressure, dtype=dtype),
        )

        # Build interpolant
        interp, d_interp = build_pchip_interpolant(
            reference_saturation=self.reference_saturation,
            values=self.capillary_pressure,
            number_of_base_points=self.number_of_base_points,
            number_of_endpoint_extra_points=self.number_of_endpoint_extra_points,
            spacing=self.spacing,
            dtype=dtype,
        )
        object.__setattr__(self, "_interp", interp)
        object.__setattr__(self, "_d_interp", d_interp)

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

        :param wetting_saturation: Current wetting-phase saturation.
        :param non_wetting_saturation: Current non-wetting-phase saturation.
        :return: The saturation to use as the interpolation x-value.
        """
        if self.reference_phase == "non_wetting":
            return non_wetting_saturation
        return wetting_saturation

    def _query(
        self,
        reference: NumberOrArray[NDimension],
    ) -> NumberOrArray[NDimension]:
        """
        Evaluate the capillary pressure PCHIP interpolant at `reference`,
        applying constant extrapolation at the boundaries. Result is cast
        to `self.dtype`.

        :param reference: Query saturation value(s) - scalar or array.
        :return: Capillary pressure value(s) cast to `self.dtype`.
        """
        dtype = self.dtype
        is_scalar = np.isscalar(reference)
        sat = np.atleast_1d(reference)
        x_min = self._interp.x[0]
        x_max = self._interp.x[-1]

        result = self._interp(np.clip(sat, x_min, x_max))
        result = np.where(sat < x_min, self.capillary_pressure[0], result)
        result = np.where(sat > x_max, self.capillary_pressure[-1], result)
        result = result.astype(dtype, copy=False)

        if is_scalar:
            return typing.cast(Number, dtype.type(result.item()))  # type: ignore
        return typing.cast(NumberOrArray[NDimension], result.reshape(sat.shape))

    def _d_query(self, reference: NumberOrArray[NDimension]) -> NumberOrArray[NDimension]:
        """
        Evaluate the analytical PCHIP derivative at `reference`, returning
        zero (in `self.dtype`) outside the knot range.

        :param reference: Query saturation value(s) - scalar or array.
        :return: Derivative value(s) cast to `self.dtype`.
        """
        dtype = self.dtype
        is_scalar = np.isscalar(reference)
        sat = np.atleast_1d(reference)
        x_min = self._d_interp.x[0]
        x_max = self._d_interp.x[-1]

        result = self._d_interp(np.clip(sat, x_min, x_max))
        result = np.where(
            (sat < x_min) | (sat > x_max),
            dtype.type(0),  # type: ignore
            result,
        )
        result = result.astype(dtype, copy=False)

        if is_scalar:
            return typing.cast(Number, dtype.type(result.item()))  # type: ignore
        return typing.cast(NumberOrArray[NDimension], result.reshape(sat.shape))

    def get_capillary_pressure(
        self,
        wetting_saturation: NumberOrArray[NDimension],
        non_wetting_saturation: NumberOrArray[NDimension] | None = None,
    ) -> NumberOrArray[NDimension]:
        """
        Get capillary pressure at the given saturation(s).

        When `reference_phase="wetting"`, only `wetting_saturation` is
        needed. When `reference_phase="non_wetting"`, `non_wetting_saturation` must be supplied.

        :param wetting_saturation: Wetting-phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting-phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Capillary pressure value(s) in `self.dtype`, matching the input shape.
        """
        ref = self.resolve_reference_saturation(
            wetting_saturation,
            non_wetting_saturation if non_wetting_saturation is not None else wetting_saturation,
        )
        return self._query(ref)

    def get_capillary_pressure_derivative(
        self,
        wetting_saturation: NumberOrArray[NDimension],
        non_wetting_saturation: NumberOrArray[NDimension] | None = None,
    ) -> NumberOrArray[NDimension]:
        """
        Derivative of capillary pressure with respect to the reference
        saturation axis of this table: `dPc / d(reference_saturation)`.

        Evaluated from the analytical PCHIP derivative. Zero (in `self.dtype`)
        outside the tabulated range (constant extrapolation = zero slope).

        :param wetting_saturation: Wetting-phase saturation (scalar or array).
        :param non_wetting_saturation: Non-wetting-phase saturation (scalar or array).
            Required when `reference_phase="non_wetting"`.
        :return: Derivative value(s) in `self.dtype` with the same shape as the input.
        """
        ref = self.resolve_reference_saturation(
            wetting_saturation,
            non_wetting_saturation if non_wetting_saturation is not None else wetting_saturation,
        )
        return self._d_query(ref)

    def evaluate(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures for the three-phase system from this
        two-phase table alone, so it can be used standalone anywhere a full
        `CapillaryPressureTable` is expected.

        The saturation for the phase pair this table doesn't cover always
        returns zero (in `self.dtype`):

        - **Oil-water table** (phases are OIL and WATER): `gas_oil` = 0.
        - **Gas-oil table** (phases are GAS and OIL): `oil_water` = 0.

        Which of `water_saturation`/`oil_saturation`/`gas_saturation` is
        forwarded as this table's interpolation reference is resolved
        internally by `get_capillary_pressure` via `reference_phase`; only
        `wetting_phase` needs to be inspected here to route the right pair
        of saturations to it.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :return: `CapillaryPressures` dict with keys `"oil_water"`, `"gas_oil"`.
        :raises ValidationError: If this table's phases are neither
            `{OIL, WATER}` nor `{OIL, GAS}`.
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

        if phases == {FluidPhase.WATER, FluidPhase.OIL}:
            if self.wetting_phase == FluidPhase.WATER:
                pcow = self.get_capillary_pressure(
                    wetting_saturation=sw, non_wetting_saturation=so
                )
            else:
                pcow = self.get_capillary_pressure(
                    wetting_saturation=so, non_wetting_saturation=sw
                )
            if is_scalar:
                return CapillaryPressures(
                    oil_water=dtype.type(np.asarray(pcow).item()),  # type: ignore
                    gas_oil=dtype.type(0),  # type: ignore
                )
            return CapillaryPressures(oil_water=pcow, gas_oil=zeros)  # type: ignore[typeddict-item]

        if phases == {FluidPhase.OIL, FluidPhase.GAS}:
            if self.wetting_phase == FluidPhase.OIL:
                pcgo = self.get_capillary_pressure(
                    wetting_saturation=so, non_wetting_saturation=sg
                )
            else:
                pcgo = self.get_capillary_pressure(
                    wetting_saturation=sg, non_wetting_saturation=so
                )
            if is_scalar:
                return CapillaryPressures(
                    oil_water=dtype.type(0),  # type: ignore
                    gas_oil=dtype.type(np.asarray(pcgo).item()),  # type: ignore
                )
            return CapillaryPressures(oil_water=zeros, gas_oil=pcgo)  # type: ignore[typeddict-item]

        raise ValidationError(
            f"Cannot dispatch three-phase saturations to a two-phase capillary "
            f"pressure table with phases {self.wetting_phase!r} / "
            f"{self.non_wetting_phase!r}. Expected OIL+WATER or OIL+GAS."
        )

    def derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> CapillaryPressureDerivatives:
        """
        Compute the four three-phase capillary-pressure derivatives from
        this two-phase table alone, so it can be used standalone anywhere a
        full `CapillaryPressureTable` is expected.

        Only the single derivative along this table's own
        `reference_saturation` axis is non-zero; the other three, including
        both derivatives for the phase pair this table doesn't cover are
        zero (in `self.dtype`). Unlike `evaluate`, both `wetting_phase` and
        `reference_phase` must be inspected here: `get_capillary_pressure_derivative`
        returns `dPc / d(reference_saturation)`, so which saturation that
        slope is actually with respect to depends on which physical
        saturation `reference_phase` currently points at.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :return: `CapillaryPressureDerivatives` dict, all four keys, in `self.dtype`.
        :raises ValidationError: If this table's phases are neither
            `{OIL, WATER}` nor `{OIL, GAS}`.
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

        if phases == {FluidPhase.WATER, FluidPhase.OIL}:
            if self.wetting_phase == FluidPhase.WATER:
                if self.reference_phase == "wetting":
                    dpcow_dsw = self.get_capillary_pressure_derivative(
                        wetting_saturation=sw, non_wetting_saturation=so
                    )
                    dpcow_dso = zeros
                else:
                    dpcow_dsw = zeros
                    dpcow_dso = self.get_capillary_pressure_derivative(
                        wetting_saturation=sw, non_wetting_saturation=so
                    )
            else:
                # Oil is the wetting phase.
                if self.reference_phase == "wetting":
                    dpcow_dsw = zeros
                    dpcow_dso = self.get_capillary_pressure_derivative(
                        wetting_saturation=so, non_wetting_saturation=sw
                    )
                else:
                    dpcow_dsw = self.get_capillary_pressure_derivative(
                        wetting_saturation=so, non_wetting_saturation=sw
                    )
                    dpcow_dso = zeros
            results = (dpcow_dsw, dpcow_dso, zeros, zeros)
            if is_scalar:
                results = tuple(
                    dtype.type(np.asarray(row).item())  # type: ignore
                    for row in results
                )
            return CapillaryPressureDerivatives(
                dpcow_dsw=results[0],
                dpcow_dso=results[1],
                dpcgo_dso=results[2],
                dpcgo_dsg=results[3],
            )

        if phases == {FluidPhase.OIL, FluidPhase.GAS}:
            if self.wetting_phase == FluidPhase.OIL:
                if self.reference_phase == "wetting":
                    dpcgo_dso = self.get_capillary_pressure_derivative(
                        wetting_saturation=so, non_wetting_saturation=sg
                    )
                    dpcgo_dsg = zeros
                else:
                    dpcgo_dso = zeros
                    dpcgo_dsg = self.get_capillary_pressure_derivative(
                        wetting_saturation=so, non_wetting_saturation=sg
                    )
            else:
                # Gas is the wetting phase (uncommon but supported).
                if self.reference_phase == "wetting":
                    dpcgo_dso = zeros
                    dpcgo_dsg = self.get_capillary_pressure_derivative(
                        wetting_saturation=sg, non_wetting_saturation=so
                    )
                else:
                    dpcgo_dso = self.get_capillary_pressure_derivative(
                        wetting_saturation=sg, non_wetting_saturation=so
                    )
                    dpcgo_dsg = zeros
            results = (zeros, zeros, dpcgo_dso, dpcgo_dsg)
            if is_scalar:
                results = tuple(
                    dtype.type(np.asarray(row).item())  # type: ignore
                    for row in results
                )
            return CapillaryPressureDerivatives(
                dpcow_dsw=results[0],
                dpcow_dso=results[1],
                dpcgo_dso=results[2],
                dpcgo_dsg=results[3],
            )

        raise ValidationError(
            f"Cannot dispatch three-phase saturations to a two-phase capillary "
            f"pressure table with phases {self.wetting_phase!r} / "
            f"{self.non_wetting_phase!r}. Expected OIL+WATER or OIL+GAS."
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `TwoPhaseCapillaryPressureTable` with `capillary_pressure`
        rescaled to *target*.

        `reference_saturation` is dimensionless and is copied unchanged.
        The PCHIP interpolant is rebuilt from the rescaled values at
        construction time of the new instance.

        :param target: Target `UnitSystem`.
        :returns: New `TwoPhaseCapillaryPressureTable` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        pressure_factor = factors["pressure"]
        return self.__class__(
            wetting_phase=self.wetting_phase,
            non_wetting_phase=self.non_wetting_phase,
            reference_saturation=self.reference_saturation.copy(),
            capillary_pressure=(self.capillary_pressure * pressure_factor).astype(  # type: ignore[arg-type]
                self.dtype, copy=False
            ),
            reference_phase=self.reference_phase,
            number_of_base_points=self.number_of_base_points,
            number_of_endpoint_extra_points=self.number_of_endpoint_extra_points,
            spacing=self.spacing,
            unit_system=target,
            dtype=self.dtype,
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
        number_of_endpoint_extra_points: int = 30,
        spacing: Spacing = "cosine",
        dtype: npt.DTypeLike = None,
    ) -> Self:
        """
        Build a `TwoPhaseCapillaryPressureTable` for one saturation region from a `DeckFile`.

        Capillary pressure is only available in the **first** saturation-function
        family (`SWOF` / `SGOF`); the **second** family (`SWFN` / `SGFN`) also
        carries a Pc column, so both families are supported here, unlike
        `TwoPhaseRelPermTable.from_deck` which needs `SOF2`/`SOF3` for the
        second family's oil curve.

        **First keyword family** (`SWOF` / `SGOF`):

        - Oil-water: reads `SWOF` -> `(sw, pcow)`. Reference phase `"wetting"`
          (Sw axis), wetting phase WATER, non-wetting OIL.
        - Gas-oil:   reads `SGOF` -> `(sg, pcog)`. Reference phase `"non_wetting"`
          (Sg axis), wetting phase OIL, non-wetting GAS.

        **Second keyword family** (`SWFN` / `SGFN`):

        - Oil-water: reads `SWFN` -> `(sw, pcow)`. Same axis/phase convention
          as the first family.
        - Gas-oil:   reads `SGFN` -> `(sg, pcog)`. Same axis/phase convention
          as the first family.

        The unit system is read from `deck_file.unit_system` automatically.

        :param deck_file: Parsed `DeckFile` containing `PROPS`-section keywords.
        :param satnum: 1-based saturation region index (default region = 1).
            Region index is given by `region_index = max(satnum - 1, 0)`.
        :param system: `"oil_water"` or `"gas_oil"`.
        :param keyword_family: `"first"` (`SWOF`/`SGOF`) or `"second"` (`SWFN`/`SGFN`).
        :param number_of_base_points: Passed to PCHIP grid scaling.
        :param number_of_endpoint_extra_points: Passed to PCHIP endpoint enrichment.
        :param spacing: Grid spacing mode for PCHIP scaling.
        :param dtype: Array dtype for all stored arrays and query returns.
        :returns: `TwoPhaseCapillaryPressureTable` for the specified region and system.
        :raises ValidationError: When the required keyword is missing or the region
            index is out of range.
        """
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        region_index = max(satnum - 1, 0)
        unit_system = deck_file.unit_system

        def require(keyword: str) -> list[dict[str, typing.Any]]:
            all_regions = deck_file.get(keyword)
            if all_regions is None or region_index >= len(all_regions):
                raise ValidationError(
                    f"Keyword `{keyword}` not found or region index {region_index} "
                    f"is out of range in the provided `DeckFile`."
                )

            rows = all_regions[region_index]
            if not rows:
                raise ValidationError(
                    f"Keyword `{keyword}` region index {region_index} has no rows."
                )
            return rows

        if system == "oil_water":
            keyword = "SWOF" if keyword_family == "first" else "SWFN"
            rows = require(keyword)
            sw = np.array([row["sw"] for row in rows], dtype=dtype)
            pcow = np.array([row["pcow"] for row in rows], dtype=dtype)
            return cls(
                wetting_phase=FluidPhase.WATER,
                non_wetting_phase=FluidPhase.OIL,
                reference_saturation=sw,  # type: ignore[arg-type]
                capillary_pressure=pcow,  # type: ignore[arg-type]
                reference_phase="wetting",
                number_of_base_points=number_of_base_points,
                number_of_endpoint_extra_points=number_of_endpoint_extra_points,
                spacing=spacing,
                unit_system=unit_system,
                dtype=dtype,
            )

        elif system == "gas_oil":
            keyword = "SGOF" if keyword_family == "first" else "SGFN"
            rows = require(keyword)
            sg = np.array([row["sg"] for row in rows], dtype=dtype)
            pcog = np.array([row["pcog"] for row in rows], dtype=dtype)
            return cls(
                wetting_phase=FluidPhase.OIL,
                non_wetting_phase=FluidPhase.GAS,
                reference_saturation=sg,  # type: ignore[arg-type]
                capillary_pressure=pcog,  # type: ignore[arg-type]
                reference_phase="non_wetting",
                number_of_base_points=number_of_base_points,
                number_of_endpoint_extra_points=number_of_endpoint_extra_points,
                spacing=spacing,
                unit_system=unit_system,
                dtype=dtype,
            )

        raise ValidationError(f"`system` must be 'oil_water' or 'gas_oil'; got {system!r}.")


@capillary_pressure_table
@attrs.frozen(slots=True)
class ThreePhaseCapillaryPressureTable(
    CapillaryPressureTable,
    load_exclude={"supports_vector"},
    dump_exclude={"supports_vector"},
):
    """
    Three-phase capillary pressure lookup table.

    Uses two two-phase tables (oil-water and gas-oil) to compute capillary pressures
    in a three-phase system (water, oil, gas).

    Pcow = Po - Pw (oil-water capillary pressure)
    Pcgo = Pg - Po (gas-oil capillary pressure)

    **dtype**:

    The `oil_water_table` and `gas_oil_table` must share the same dtype.
    This is validated at construction time. All returned derivative values
    are cast to that shared dtype.

    **unit_system**:

    The `oil_water_table` and `gas_oil_table` must share the same
    `unit_system`. This is validated at construction time. Use `convert(target)`
    to produce a copy of this table (and both sub-tables) rescaled to another
    `UnitSystem`.
    """

    __type__ = "three_phase_capillary_pressure_table"

    oil_water_table: TwoPhaseCapillaryPressureTable
    """
    Capillary pressure table for oil-water system (wetting phase = water or oil).

    A table of Pcow against wetting phase saturation (water saturation if water is wetting phase,
    oil saturation if oil is wetting phase).
    """

    gas_oil_table: TwoPhaseCapillaryPressureTable
    """
    Capillary pressure table for gas-oil system (wetting phase = oil).

    A table of Pcgo against oil saturation.
    """

    supports_vector: bool = attrs.field(init=False, repr=False, default=True)
    """Flag indicating support for array inputs."""

    def __attrs_post_init__(self) -> None:
        """Validate that the tables are set up correctly for three-phase flow."""
        if {
            self.oil_water_table.wetting_phase,
            self.oil_water_table.non_wetting_phase,
        } != {FluidPhase.WATER, FluidPhase.OIL}:
            raise ValidationError("`oil_water_table` must be between water and oil phases.")
        if {self.gas_oil_table.wetting_phase, self.gas_oil_table.non_wetting_phase} != {
            FluidPhase.OIL,
            FluidPhase.GAS,
        }:
            raise ValidationError("`gas_oil_table` must be between oil and gas phases.")

        if self.oil_water_table.wetting_phase == self.gas_oil_table.non_wetting_phase:
            raise ValidationError(
                "Wetting phase of `oil_water_table` cannot be the same as non-wetting phase of `gas_oil_table`."
            )

        # Validate matching dtype between the two sub-tables
        ow_dtype = np.dtype(self.oil_water_table.dtype)
        go_dtype = np.dtype(self.gas_oil_table.dtype)
        if ow_dtype != go_dtype:
            raise ValidationError(
                f"`oil_water_table` dtype ({ow_dtype}) and `gas_oil_table` dtype "
                f"({go_dtype}) must match. Convert one of the tables before combining."
            )

        # Validate matching unit_system between the two sub-tables
        if self.oil_water_table.unit_system != self.gas_oil_table.unit_system:
            raise ValidationError(
                f"`oil_water_table` unit_system ({self.oil_water_table.unit_system.value!r}) "
                f"and `gas_oil_table` unit_system ({self.gas_oil_table.unit_system.value!r}) "
                f"must match. Convert one of the tables before combining."
            )

    @property
    def dtype(self) -> np.dtype:
        """Shared `dtype` of the two underlying two-phase tables."""
        return np.dtype(self.oil_water_table.dtype)

    @property
    def unit_system(self) -> UnitSystem:  # type: ignore[override]
        """Shared `unit_system` of the two underlying two-phase tables."""
        return self.oil_water_table.unit_system

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        return typing.cast(FluidPhase, self.oil_water_table.wetting_phase)

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        return typing.cast(FluidPhase, self.gas_oil_table.wetting_phase)

    def evaluate(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures for the three-phase system.

        Each sub-table is queried using its declared `reference_phase`:

        - `reference_phase="wetting"` - the wetting-phase saturation is passed.
        - `reference_phase="non_wetting"` - the non-wetting-phase saturation is
        passed.

        For the oil-water table the wetting phase is either WATER or OIL.
        For the gas-oil table the wetting phase is always OIL, but the table may
        be indexed by So (`reference_phase="wetting"`) or by Sg
        (`reference_phase="non_wetting"`).

        Returned values are in the shared `self.dtype` and `self.unit_system`.

        :param water_saturation: Water saturation (fraction, 0-1).
        :param oil_saturation: Oil saturation (fraction, 0-1).
        :param gas_saturation: Gas saturation (fraction, 0-1).
        :return: `CapillaryPressures` dictionary.
        """
        oil_water_table = self.oil_water_table
        gas_oil_table = self.gas_oil_table

        # Oil-water capillary pressure. Dispatch to the wetting-phase saturation
        if oil_water_table.wetting_phase == FluidPhase.WATER:
            pcow = oil_water_table.get_capillary_pressure(
                wetting_saturation=water_saturation,
                non_wetting_saturation=oil_saturation,
            )
        else:
            # Oil is the wetting phase
            pcow = oil_water_table.get_capillary_pressure(
                wetting_saturation=oil_saturation,
                non_wetting_saturation=water_saturation,
            )

        # Gas-oil capillary pressure. Dispatch to the correct saturation
        # depending on the table's reference_phase axis.
        if gas_oil_table.wetting_phase == FluidPhase.OIL:
            pcgo = gas_oil_table.get_capillary_pressure(
                wetting_saturation=oil_saturation,
                non_wetting_saturation=gas_saturation,
            )
        else:
            # Gas is the wetting phase (uncommon but supported)
            pcgo = gas_oil_table.get_capillary_pressure(
                wetting_saturation=gas_saturation,
                non_wetting_saturation=oil_saturation,
            )

        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo)  # type: ignore[typeddict-item]

    def derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        **kwargs: typing.Any,
    ) -> CapillaryPressureDerivatives:
        """
        Compute the partial derivatives of the oil-water and gas-oil capillary
        pressures with respect to saturation.

        Returns a `CapillaryPressureDerivatives` dictionary with four entries:

        - `dpcow_dsw`: non-zero when the oil-water table's reference axis is
        water saturation (`wetting_phase=WATER, reference_phase="wetting"`).
        - `dpcow_dso`: non-zero when the oil-water table's reference axis is oil
        saturation (`wetting_phase=OIL, reference_phase="wetting"`).
        - `dpcgo_dso`: non-zero when the gas-oil table's reference axis is oil
        saturation (`reference_phase="wetting"`, the wetting phase being OIL).
        - `dpcgo_dsg`: non-zero when the gas-oil table's reference axis is gas
        saturation (`reference_phase="non_wetting"`, or gas is the wetting
        phase with `reference_phase="wetting"`).

        At most one of `dpcow_dsw` / `dpcow_dso` is non-zero, and at most one
        of `dpcgo_dso` / `dpcgo_dsg` is non-zero, for a given table
        configuration. All derivatives are exact PCHIP slopes from the
        underlying two-phase tables, cast to the shared `self.dtype`.

        :param water_saturation: Water saturation (scalar or array).
        :param oil_saturation: Oil saturation (scalar or array).
        :param gas_saturation: Gas saturation (scalar or array).
        :return: `CapillaryPressureDerivatives` dictionary in `self.dtype`.
        """
        dtype = self.dtype
        oil_water_table = self.oil_water_table
        gas_oil_table = self.gas_oil_table

        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )
        sw = np.atleast_1d(water_saturation)
        zero = dtype.type(0) if is_scalar else np.zeros(sw.shape, dtype=dtype)  # type: ignore

        # Oil-water derivatives
        if oil_water_table.wetting_phase == FluidPhase.WATER:
            if oil_water_table.reference_phase == "wetting":
                # Table indexed by Sw (wetting phase) -> derivative is dpcow/dsw
                dpcow_dsw = oil_water_table.get_capillary_pressure_derivative(
                    wetting_saturation=water_saturation,  # type: ignore[arg-type]
                    non_wetting_saturation=oil_saturation,  # type: ignore[arg-type]
                )
                dpcow_dso = zero
            else:
                # reference_phase="non_wetting" and wetting_phase=WATER means
                # table is indexed by So (non-wetting phase) -> derivative is dpcow/dso
                dpcow_dsw = zero
                dpcow_dso = oil_water_table.get_capillary_pressure_derivative(
                    wetting_saturation=water_saturation,  # type: ignore[arg-type]
                    non_wetting_saturation=oil_saturation,  # type: ignore[arg-type]
                )
        else:
            # Oil is the wetting phase.  The reference_phase attribute then
            # controls whether the table is indexed by So ("wetting") or Sw
            # ("non_wetting").  Either way the derivative is with respect to
            # whichever saturation is the reference axis.
            if oil_water_table.reference_phase == "wetting":
                # Table indexed by So -> derivative is dpcow/dso
                dpcow_dsw = zero
                dpcow_dso = oil_water_table.get_capillary_pressure_derivative(
                    wetting_saturation=oil_saturation,  # type: ignore[arg-type]
                    non_wetting_saturation=water_saturation,  # type: ignore[arg-type]
                )
            else:
                # reference_phase="non_wetting" and wetting_phase=OIL means the
                # table is indexed by water saturation (the non-wetting phase here
                # is water) -> derivative is dpcow/dsw
                dpcow_dsw = oil_water_table.get_capillary_pressure_derivative(
                    wetting_saturation=oil_saturation,  # type: ignore[arg-type]
                    non_wetting_saturation=water_saturation,  # type: ignore[arg-type]
                )
                dpcow_dso = zero

        # Gas-oil derivatives
        if gas_oil_table.wetting_phase == FluidPhase.OIL:
            if gas_oil_table.reference_phase == "wetting":
                # Table indexed by So -> derivative is dpcgo/dso
                dpcgo_dso = gas_oil_table.get_capillary_pressure_derivative(
                    wetting_saturation=oil_saturation,  # type: ignore[arg-type]
                    non_wetting_saturation=gas_saturation,  # type: ignore[arg-type]
                )
                dpcgo_dsg = zero
            else:
                # reference_phase="non_wetting" -> table indexed by Sg
                dpcgo_dso = zero
                dpcgo_dsg = gas_oil_table.get_capillary_pressure_derivative(
                    wetting_saturation=oil_saturation,  # type: ignore[arg-type]
                    non_wetting_saturation=gas_saturation,  # type: ignore[arg-type]
                )
        else:
            # Gas is the wetting phase (uncommon). reference_phase="wetting"
            # means indexed by Sg, "non_wetting" means indexed by So.
            if gas_oil_table.reference_phase == "wetting":
                dpcgo_dso = zero
                dpcgo_dsg = gas_oil_table.get_capillary_pressure_derivative(
                    wetting_saturation=gas_saturation,  # type: ignore[arg-type]
                    non_wetting_saturation=oil_saturation,  # type: ignore[arg-type]
                )
            else:
                dpcgo_dso = gas_oil_table.get_capillary_pressure_derivative(
                    wetting_saturation=gas_saturation,  # type: ignore[arg-type]
                    non_wetting_saturation=oil_saturation,  # type: ignore[arg-type]
                )
                dpcgo_dsg = zero

        return CapillaryPressureDerivatives(
            dpcow_dsw=dpcow_dsw,
            dpcow_dso=dpcow_dso,
            dpcgo_dso=dpcgo_dso,
            dpcgo_dsg=dpcgo_dsg,
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `ThreePhaseCapillaryPressureTable` with both sub-tables
        rescaled to *target*.

        :param target: Target `UnitSystem`.
        :returns: New `ThreePhaseCapillaryPressureTable` in *target* units.
        """
        if target == self.unit_system:
            return self

        return self.__class__(
            oil_water_table=self.oil_water_table.convert(target, table=table),
            gas_oil_table=self.gas_oil_table.convert(target, table=table),
        )

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        satnum: int = 1,
        *,
        keyword_family: typing.Literal["first", "second", "auto"] = "auto",
        number_of_base_points: int = 200,
        number_of_endpoint_extra_points: int = 30,
        spacing: Spacing = "cosine",
        dtype: npt.DTypeLike = None,
    ) -> Self:
        """
        Build a `ThreePhaseCapillaryPressureTable` for one saturation region from
        a `DeckFile`.

        Detects which Eclipse saturation-function keyword family is present and
        builds the oil-water and gas-oil two-phase sub-tables automatically:

        **First family** (detected when `SWOF` or `SGOF` is present):

        `SWOF` supplies the oil-water capillary pressure `(sw, pcow)`.
        `SGOF` supplies the gas-oil capillary pressure `(sg, pcog)`.

        **Second family** (detected when `SWFN` or `SGFN` is present):

        `SWFN` supplies the oil-water capillary pressure `(sw, pcow)`.
        `SGFN` supplies the gas-oil capillary pressure `(sg, pcog)`.

        Unlike `ThreePhaseRelPermTable.from_deck`, the second family does
        not need `SOF2`/`SOF3` here since capillary pressure has no oil-relative
        component — both `SWFN` and `SGFN` already carry their own Pc column.

        Both keywords (oil-water and gas-oil) must be present for a three-phase
        table. If only one is found a warning is issued and a
        `TwoPhaseCapillaryPressureTable` should be used instead.

        The unit system is read from `deck_file.unit_system` automatically and
        is shared by both sub-tables.

        :param deck_file: Parsed `DeckFile` containing PRO`PS-section keywords.
        :param satnum: 1-based saturation region index (default region = 1).
            Region index is given as `region_index = max(satnum - 1, 0)`.
        :param keyword_family: `"first"`, `"second"`, or `"auto"` (default).
        :param number_of_base_points: Passed to PCHIP grid scaling.
        :param number_of_endpoint_extra_points: Passed to PCHIP endpoint enrichment.
        :param spacing: Grid spacing mode for PCHIP scaling.
        :param dtype: Array dtype shared by both sub-tables and all query returns.
        :returns: `ThreePhaseCapillaryPressureTable` for the specified region.
        :raises ValidationError: When required keywords are missing.
        """
        dtype = np.dtype(dtype) if dtype is not None else get_dtype()
        region_index = max(satnum - 1, 0)

        def has(keyword: str) -> bool:
            all_regions = deck_file.get(keyword)
            if all_regions is None:
                return False
            return region_index < len(all_regions) and bool(all_regions[region_index])

        shared_kwargs: dict[str, typing.Any] = dict(
            satnum=satnum,
            number_of_base_points=number_of_base_points,
            number_of_endpoint_extra_points=number_of_endpoint_extra_points,
            spacing=spacing,
            dtype=dtype,
        )

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

        oil_water_keyword = "SWOF" if family == "first" else "SWFN"
        gas_oil_keyword = "SGOF" if family == "first" else "SGFN"
        has_oil_water_table = has(oil_water_keyword)
        has_gas_oil_table = has(gas_oil_keyword)

        if not has_oil_water_table:
            warnings.warn(
                f"Oil-water keyword `{oil_water_keyword}` not found for `SATNUM` "
                f"{satnum}. Cannot build a three-phase capillary pressure table. "
                "Use `TwoPhaseCapillaryPressureTable.from_deck(..., system='gas_oil')` "
                "to build a gas-oil only table.",
                UserWarning,
                stacklevel=2,
            )
            raise ValidationError(
                f"Oil-water keyword `{oil_water_keyword}` required for "
                f"`ThreePhaseCapillaryPressureTable` not found at `SATNUM` {satnum}."
            )

        if not has_gas_oil_table:
            warnings.warn(
                f"Gas-oil keyword `{gas_oil_keyword}` not found for `SATNUM` "
                f"{satnum}. Cannot build a three-phase capillary pressure table. "
                "Use `TwoPhaseCapillaryPressureTable.from_deck(..., system='oil_water')` "
                "to build an oil-water only table.",
                UserWarning,
                stacklevel=2,
            )
            raise ValidationError(
                f"Gas-oil keyword `{gas_oil_keyword}` required for "
                f"`ThreePhaseCapillaryPressureTable` not found at `SATNUM` {satnum}."
            )

        oil_water_table = TwoPhaseCapillaryPressureTable.from_deck(
            deck_file=deck_file,
            system="oil_water",
            keyword_family=family,
            **shared_kwargs,
        )
        gas_oil_table = TwoPhaseCapillaryPressureTable.from_deck(
            deck_file=deck_file,
            system="gas_oil",
            keyword_family=family,
            **shared_kwargs,
        )
        return cls(oil_water_table=oil_water_table, gas_oil_table=gas_oil_table)

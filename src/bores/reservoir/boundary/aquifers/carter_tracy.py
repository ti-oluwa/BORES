import typing
import warnings

import attrs
import numba
import numpy as np
from scipy.optimize import root_scalar
from scipy.special import j1, y1
from typing_extensions import Self

from bores.constants import c, get_conversion_factors
from bores.deck.core import DeckParseError
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.reservoir.boundary.base import (
    BoundaryCondition,
    BoundaryConditionType,
    boundary_condition,
)
from bores.types import Number, NumberArray, OneDimension, UnitConversionTable, UnitSystem

__all__ = [
    "CarterTracyAquifer",
    "compute_incremental_influx",
    "from_deck",
    "load_carter_tracy_aquifer_from_record",
    "load_carter_tracy_aquifers_from_records",
]


def compute_bessel_roots(r_ed: Number, n_max: int) -> NumberArray[OneDimension]:
    """
    Roots of `J1(β·r_eD)·Y1(β) - J1(β)·Y1(β·r_eD) = 0` - the `β_n` needed by
    the bounded-aquifer Bessel series.

    Ported from `pywaterflood.aquifer.get_bessel_roots` (Frank Male,
    https://github.com/frank1010111/pywaterflood, MIT licensed), which cites
    Klins, Bouchard & Cable (1988), eq. 9.

    :param r_ed: Dimensionless radius `r_e / r_w`. Must be `> 1`.
    :param n_max: Number of roots to find.
    :returns: Shape `(n_max,)` array of roots, ascending.
    """

    def root_func(beta: NumberArray[OneDimension]) -> NumberArray[OneDimension]:
        return typing.cast(
            NumberArray[OneDimension], j1(beta * r_ed) * y1(beta) - j1(beta) * y1(beta * r_ed)
        )

    sample = typing.cast(
        NumberArray[OneDimension], np.linspace(1e-9, 8 * n_max / r_ed, n_max * 400)
    )
    zero_crossings = np.array([])
    while len(zero_crossings) < n_max:
        sample = sample * 2
        zero_crossings = np.where(np.diff(np.sign(root_func(sample))))[0]
    zero_crossings = zero_crossings[:n_max]
    roots = [root_scalar(root_func, x0=sample[zc]).root for zc in zero_crossings]  # type: ignore[arg-type]
    return typing.cast(NumberArray[OneDimension], np.asarray(roots, dtype=np.float64))


def compute_bessel_series_coefficients(
    r_ed: Number, betas: NumberArray[OneDimension]
) -> tuple[NumberArray[OneDimension], NumberArray[OneDimension]]:
    """
    Precomputes the Bessel-dependent part of every term in the bounded
    aquifer's `pD`/`pD'` series (Klins, Bouchard & Cable, 1988, eqs. 6-9),
    since `J1(β_n·r_eD)` and `J1(β_n)` depend only on `β_n` and `r_eD`and not
    on `t_D`. Only `exp(-β_n²·t_D)` actually varies at evaluation time,
    so the hot-path series reduces to `Σ 2·exp(-β_n²·t_D)·coefficient_n`.

    `pD`'s per-term coefficient: `J1(β_n·r_eD)² / (β_n²·(J1(β_n·r_eD)² - J1(β_n)²))`.
    `pD'`'s per-term coefficient: `-J1(β_n·r_eD)² / (J1(β_n·r_eD)² - J1(β_n)²)`
    (`= -β_n² · pD`'s coefficient`, computed directly here rather than
    derived, so a reader never has to reconstruct that relationship).

    :param r_ed: Dimensionless radius `r_e / r_w`.
    :param betas: `compute_bessel_roots(r_ed, n)`.
    :returns: `(pd_coefficients, pd_prime_coefficients)`, each shape `(n,)`,
        same order as `betas`.
    """
    j1_beta_red = j1(betas * r_ed)
    j1_beta = j1(betas)
    denominator = j1_beta_red**2 - j1_beta**2
    pd_coefficients = j1_beta_red**2 / (betas**2 * denominator)
    pd_prime_coefficients = -(j1_beta_red**2) / denominator
    return (
        typing.cast(NumberArray[OneDimension], np.asarray(pd_coefficients, dtype=np.float64)),
        typing.cast(
            NumberArray[OneDimension], np.asarray(pd_prime_coefficients, dtype=np.float64)
        ),
    )


@numba.njit(cache=True)
def compute_infinite_dimensionless_pressure(t_d: Number) -> Number:
    """
    Dimensionless pressure `pD(tD)` for an infinite-acting radial aquifer.

    Edwardson et al. (1962) polynomial for `tD <= 100`, logarithmic
    approximation for `tD > 100`. Both from Carter & Tracy (1960).

    :param t_d: Dimensionless time.
    :returns: Dimensionless pressure.
    """
    if t_d <= 0.0:
        return 0.0
    if t_d > 100.0:
        return 0.5 * (np.log(t_d) + 0.80907)
    sqrt_td = np.sqrt(t_d)
    td_15 = t_d**1.5
    numerator = 370.529 * sqrt_td + 137.582 * t_d + 5.69549 * td_15
    denominator = 328.834 + 265.488 * sqrt_td + 45.2157 * t_d + td_15
    return numerator / denominator


@numba.njit(cache=True)
def compute_infinite_dimensionless_pressure_derivative(t_d: Number) -> Number:
    """
    Derivative `pD'(tD)` for an infinite-acting radial aquifer.

    Edwardson et al. (1962) polynomial ratio for `tD <= 100`, analytical
    derivative of the logarithmic approximation for `tD > 100`.

    :param t_d: Dimensionless time.
    :returns: Dimensionless pressure derivative.
    """
    if t_d <= 0.0:
        return 0.0
    if t_d > 100.0:
        return 1.0 / (2.0 * t_d)
    sqrt_td = np.sqrt(t_d)
    td_15 = t_d**1.5
    td_2 = t_d**2.0
    td_25 = t_d**2.5
    e = 716.441 + 46.7984 * sqrt_td + 270.038 * t_d + 71.0098 * td_15
    f = 1296.86 * sqrt_td + 1204.73 * t_d + 618.618 * td_15 + 538.072 * td_2 + 142.41 * td_25
    if abs(f) < 1e-30:
        return 0.0
    return e / f


@numba.njit(cache=True)
def compute_bounded_aquifer_threshold(r_ed: Number) -> Number:
    """
    Dimensionless time below which an infinite-acting aquifer is an
    accurate, much cheaper stand-in for a bounded one of radius ratio `r_ed`.

    `0.4 * (r_eD^2 - 1)` (`pywaterflood.aquifer.water_dimensionless`):
    below this, the pressure transient hasn't reached the aquifer's outer
    edge yet.

    :param r_ed: Dimensionless radius `r_e / r_w`.
    :returns: Threshold dimensionless time.
    """
    return 0.4 * (r_ed**2 - 1.0)


@numba.njit(cache=True)
def compute_finite_dimensionless_pressure(
    t_d: Number,
    betas: NumberArray[OneDimension],
    pd_coefficients: NumberArray[OneDimension],
    linear_coefficient: Number,
    constant_coefficient: Number,
) -> Number:
    """
    Dimensionless pressure `pD(tD, r_eD)` for a bounded (finite) radial
    aquifer, from precomputed per-root coefficients
    (`compute_bessel_series_coefficients`).

    :param t_d: Dimensionless time.
    :param betas: This aquifer's Bessel roots.
    :param pd_coefficients: Matching `pD` coefficients, same order as `betas`.
    :param linear_coefficient: `2/(r_eD^2-1)`, this aquifer's own r_eD-dependent slope.
    :param constant_coefficient: The series' r_eD-only constant term.
    :returns: Dimensionless pressure.
    """
    series = 0.0
    for n in range(betas.shape[0]):
        series += 2.0 * np.exp(-(betas[n] ** 2) * t_d) * pd_coefficients[n]
    return constant_coefficient + linear_coefficient * t_d + series


@numba.njit(cache=True)
def compute_finite_dimensionless_pressure_derivative(
    t_d: Number,
    betas: NumberArray[OneDimension],
    pd_prime_coefficients: NumberArray[OneDimension],
    linear_coefficient: Number,
) -> Number:
    """
    Derivative `pD'(tD, r_eD)` of `compute_finite_dimensionless_pressure`,
    from precomputed per-root coefficients.

    :param t_d: Dimensionless time.
    :param betas: This aquifer's Bessel roots.
    :param pd_prime_coefficients: Matching `pD'` coefficients, same order as `betas`.
    :param linear_coefficient: `2/(r_eD^2-1)`, same value passed to `compute_finite_dimensionless_pressure`.
    :returns: Dimensionless pressure derivative.
    """
    series = 0.0
    for n in range(betas.shape[0]):
        series += 2.0 * np.exp(-(betas[n] ** 2) * t_d) * pd_prime_coefficients[n]
    return linear_coefficient + series


@numba.njit(cache=True)
def compute_incremental_influx(
    previous_cumulative_influx: Number,
    previous_dimensionless_time: Number,
    current_dimensionless_time: Number,
    current_pressure_drop: Number,
    aquifer_constant: Number,
    bounded: bool,
    dimensionless_radius_ratio: Number,
    betas: NumberArray[OneDimension],
    pd_coefficients: NumberArray[OneDimension],
    pd_prime_coefficients: NumberArray[OneDimension],
    linear_coefficient: Number,
    constant_coefficient: Number,
) -> Number:
    """
    One Carter-Tracy (1960) recursive step - Eq. 3:

        (We)_n = (We)_{n-1}
                 + [(tD)_n - (tD)_{n-1}]
                   * [aquifer_constant*ΔP_n - (We)_{n-1}*pD'_n]
                   / [pD_n - (tD)_{n-1}*pD'_n]

    A pure function that reads no workspace state and writes nothing. Safe
    to call every Newton/Picard iteration within a single not yet accepted
    timestep; the caller decides separately whether to treat this as a
    trial value or to write the result into `AquiferWorkspace` as the new
    committed state.

    :param previous_cumulative_influx: `(We)_{n-1}` as of the last committed state.
    :param previous_dimensionless_time: `(tD)_{n-1}` as of the last committed state.
    :param current_dimensionless_time: `(tD)_n` at the time being evaluated.
    :param current_pressure_drop: `ΔP_n = p_initial - p_boundary`, using
        the current trial (or accepted) boundary pressure.
    :param aquifer_constant: This aquifer's own resolved `aquifer_constant`.
    :param bounded: Whether to switch to the finite-aquifer Bessel series
        once `current_dimensionless_time` passes
        `compute_bounded_aquifer_threshold(dimensionless_radius_ratio)`.
    :param dimensionless_radius_ratio: `r_e/r_w`. Only used when `bounded`.
    :param betas: This aquifer's Bessel roots. Pass a zero-length array when `bounded` is `False`.
    :param pd_coefficients: Matching `pD` coefficients. Zero-length when `bounded` is `False`.
    :param pd_prime_coefficients: Matching `pD'` coefficients. Zero-length when `bounded` is `False`.
    :param linear_coefficient: `2/(r_eD^2-1)`. Unused when `bounded` is `False`.
    :param constant_coefficient: The bounded series' constant term. Unused when `bounded` is `False`.
    :returns: `(We)_n`, computed from the last committed state, not written anywhere.
    """
    delta_t_d = current_dimensionless_time - previous_dimensionless_time
    if delta_t_d <= 0.0:
        return previous_cumulative_influx

    if bounded and current_dimensionless_time >= compute_bounded_aquifer_threshold(
        r_ed=dimensionless_radius_ratio
    ):
        current_p_d = compute_finite_dimensionless_pressure(
            t_d=current_dimensionless_time,
            betas=betas,
            pd_coefficients=pd_coefficients,
            linear_coefficient=linear_coefficient,
            constant_coefficient=constant_coefficient,
        )
        current_p_d_prime = compute_finite_dimensionless_pressure_derivative(
            t_d=current_dimensionless_time,
            betas=betas,
            pd_prime_coefficients=pd_prime_coefficients,
            linear_coefficient=linear_coefficient,
        )
    else:
        current_p_d = compute_infinite_dimensionless_pressure(current_dimensionless_time)
        current_p_d_prime = compute_infinite_dimensionless_pressure_derivative(
            current_dimensionless_time
        )

    denominator = current_p_d - previous_dimensionless_time * current_p_d_prime
    if abs(denominator) < 1e-30:
        return previous_cumulative_influx

    numerator_bracket = (
        aquifer_constant * current_pressure_drop - previous_cumulative_influx * current_p_d_prime
    )
    return previous_cumulative_influx + delta_t_d * (numerator_bracket / denominator)


@boundary_condition
@attrs.frozen(slots=True)
class CarterTracyAquifer(BoundaryCondition):
    """
    Parameters for a water influx aquifer using the Carter-Tracy (1960)
    recursive approximation to the Van Everdingen-Hurst transient solution.

    A pure parameter container that resolves its physical or calibrated
    inputs, and (when `bounded_aquifer=True`) the Bessel roots and their
    series coefficients, once at construction, and carries no other
    state. The recurrence itself (`compute_incremental_influx`) is a free
    function operating on `AquiferWorkspace` arrays.

    This class never runs on the hot path, only compiles into one
    row of `CompiledAquifers`.

    Computes cumulative and incremental water influx from a finite radial
    aquifer using the Carter-Tracy recurrence, which avoids the
    superposition convolution of the original Van Everdingen-Hurst method
    while preserving its physical basis. The recurrence runs in O(1)
    memory and O(1) CPU per time step regardless of simulation length.

    **Carter-Tracy recurrence** (Carter & Tracy, 1960, Eq. 3) - see
    `compute_incremental_influx`.

    **Dimensionless time** (FIELD units, Carter & Tracy 1960, Eq. 1):

        tD = 6.328e-3 * k * t / (φ * μ_w * ct * r_w²)

    where `r_w` is the inner (reservoir-aquifer contact) radius in ft and
    `t` is in days.

    **Aquifer constant** (FIELD units, Carter & Tracy 1960, Eq. 2):

        aquifer_constant = 1.119 * φ * ct * (r_e² - r_w²) * h * f

    where `f = θ/360` is the encroachment angle fraction and `r_e` is the
    outer aquifer radius in ft.

    **pD and pD' approximations**: Edwardson et al. (1962) polynomial for
    `tD <= 100`, logarithmic approximation for `tD > 100` - see
    `compute_infinite_dimensionless_pressure`.

    **Two construction modes**:

    *Physical-properties mode*: supply `aquifer_permeability`,
    `aquifer_porosity`, `aquifer_compressibility`, `water_viscosity`,
    `inner_radius`, `outer_radius`, `aquifer_thickness`. `aquifer_constant`
    and hydraulic diffusivity are derived automatically in FIELD units
    then stored in `unit_system` units.

    *Calibrated-constant mode*: supply `aquifer_constant` and,
    optionally, `dimensionless_time_scale` (recommended) and
    `dimensionless_radius_ratio` (record-keeping only unless
    `bounded_aquifer=True`). Useful when parameters are history-matched
    rather than measured directly.

    **Bounded aquifers**: by default, always uses the infinite-acting
    `pD`/`pD'` approximation, regardless of `dimensionless_radius_ratio`/
    `outer_radius`. Set `bounded_aquifer=True` to switch to the Klins,
    Bouchard & Cable (1988) finite-aquifer solution once dimensionless
    time passes `compute_bounded_aquifer_threshold(r_eD)`.

    **Unit system**: all user-supplied dimensional inputs must be in
    `unit_system`. Internally, the FIELD-unit constants (1.119, 6.328e-3)
    are applied after converting inputs to FIELD; `aquifer_constant` and
    hydraulic diffusivity are then converted back to `unit_system` for
    storage.

    **References**:

    - Carter, R.D. & Tracy, G.W. (1960). *An Improved Method for
      Calculating Water Influx.* Trans. AIME, 219, 415-417.
    - Edwardson, M.J. et al. (1962). *Calculation of Formation
      Temperature Disturbances Caused by Mud Circulation.* JPT, 14(4),
      416-426. (source of the pD polynomial approximations)
    - Klins, M.A., Bouchard, A.J. & Cable, C.L. (1988). *A Polynomial
      Approach to the Van Everdingen-Hurst Dimensionless Variables.*
      SPE Reservoir Engineering, 3(1), 320-326.
    - Ahmed, T. (2010). *Reservoir Engineering Handbook*, 4th ed.
      Gulf Professional Publishing. (Carter-Tracy chapter.)
    """

    __type__: typing.ClassVar[str] = "carter_tracy_aquifer"

    condition_type: typing.ClassVar[BoundaryConditionType] = BoundaryConditionType.FLUX

    initial_pressure: Number
    """Initial aquifer / reservoir pressure in `unit_system` pressure units."""

    aquifer_permeability: Number | None = attrs.field(default=None)
    """Aquifer permeability. Physical mode only."""

    aquifer_porosity: Number | None = attrs.field(default=None)
    """Aquifer porosity (fraction). Physical mode only."""

    aquifer_compressibility: Number | None = attrs.field(default=None)
    """Total aquifer compressibility. Physical mode only."""

    water_viscosity: Number | None = attrs.field(default=None)
    """Water viscosity at reservoir conditions. Physical mode only."""

    inner_radius: Number | None = attrs.field(default=None)
    """Reservoir-aquifer contact radius. Physical mode only."""

    outer_radius: Number | None = attrs.field(default=None)
    """
    Outer aquifer extent. Physical mode only. Always sets total aquifer
    storage capacity (via `r_e^2 - r_w^2` in `aquifer_constant`); also
    sets the transient response's `r_eD = r_e/r_w` when `bounded_aquifer=True`.
    """

    aquifer_thickness: Number | None = attrs.field(default=None)
    """Aquifer thickness. Physical mode only."""

    aquifer_constant: Number | None = attrs.field(default=None)
    """
    Pre-computed or history-matched aquifer constant (reservoir volume /
    pressure in `unit_system`). Calibrated-constant mode only.
    """

    dimensionless_radius_ratio: Number = attrs.field(default=10.0)
    """
    `r_e / r_w`. Calibrated-constant mode only - physical mode derives
    its own `r_e / r_w` from `outer_radius`/`inner_radius` instead and
    ignores this field. Affects the transient response shape only when
    `bounded_aquifer=True`; otherwise stored for the record only.
    """

    bounded_aquifer: bool = attrs.field(default=False)
    """
    Opt-in: use the Klins, Bouchard & Cable (1988) finite/bounded-aquifer
    `pD(tD, r_eD)` once `tD` passes `compute_bounded_aquifer_threshold(r_eD)`,
    instead of always treating the aquifer as infinite-acting. Defaults
    to `False` for backward compatibility - see the full discussion in
    the class docstring history (prior handoff notes) if reviving this
    default is ever considered.
    """

    dimensionless_time_scale: Number | None = attrs.field(default=None)
    """
    `tD / t` - dimensionless time per unit of `unit_system` time.
    Calibrated-constant mode only, optional but recommended. When set,
    `tD = dimensionless_time_scale * t`. When left `None`, `tD` falls
    back to raw elapsed `time` - dimensionally meaningless and dependent
    on `unit_system`'s time unit - and `__attrs_post_init__` warns about it.
    """

    angle: Number = attrs.field(default=360.0)
    """Aquifer encroachment angle in degrees."""

    unit_system: UnitSystem = attrs.field(default=UnitSystem.FIELD)
    """Unit system for all dimensional parameters and returned flux values."""

    # Resolved scalars to be compiled into `CompiledAquifers`

    resolved_aquifer_constant: Number = attrs.field(default=0.0, init=False, repr=False)
    """Resolved aquifer constant in `unit_system` units. Set on initialization."""

    resolved_dimensionless_radius_ratio: Number = attrs.field(default=10.0, init=False, repr=False)
    """Resolved `r_e / r_w`. Set on initialization."""

    hydraulic_diffusivity: Number | None = attrs.field(default=None, init=False, repr=False)
    """
    Hydraulic diffusivity in `[length^2 / time]` in `unit_system` units.
    Used to compute dimensionless time: `tD = diffusivity * t / r_w^2`.
    `None` in calibrated-constant mode.
    """

    bessel_roots: NumberArray[OneDimension] = attrs.field(
        factory=lambda: np.empty(0, dtype=np.float64), init=False, repr=False
    )
    """
    `compute_bessel_roots(resolved_dimensionless_radius_ratio, AQUIFER_BESSEL_SERIES_TERMS)`,
    computed once at construction when `bounded_aquifer=True`. Empty when
    `bounded_aquifer=False`.
    """

    pd_coefficients: NumberArray[OneDimension] = attrs.field(
        factory=lambda: np.empty(0, dtype=np.float64), init=False, repr=False
    )
    """Precomputed `pD` series coefficients matching `bessel_roots` - see `compute_bessel_series_coefficients`."""

    pd_prime_coefficients: NumberArray[OneDimension] = attrs.field(
        factory=lambda: np.empty(0, dtype=np.float64), init=False, repr=False
    )
    """Precomputed `pD'` series coefficients matching `bessel_roots`."""

    linear_coefficient: Number = attrs.field(default=0.0, init=False, repr=False)
    """`2/(r_eD^2-1)`, the bounded series' r_eD-dependent slope. `0.0` when `bounded_aquifer=False`."""

    constant_coefficient: Number = attrs.field(default=0.0, init=False, repr=False)
    """The bounded series' r_eD-only constant term. `0.0` when `bounded_aquifer=False`."""

    def __attrs_post_init__(self) -> None:
        has_physical = all(
            v is not None
            for v in (
                self.aquifer_permeability,
                self.aquifer_porosity,
                self.aquifer_compressibility,
                self.water_viscosity,
                self.inner_radius,
                self.outer_radius,
                self.aquifer_thickness,
            )
        )
        has_calibrated = self.aquifer_constant is not None

        if not (has_physical or has_calibrated):
            raise ValidationError(
                f"{type(self).__name__!r} requires either:\n"
                "  Physical-properties mode: aquifer_permeability, aquifer_porosity,\n"
                "    aquifer_compressibility, water_viscosity, inner_radius,\n"
                "    outer_radius, aquifer_thickness.\n"
                "  Calibrated-constant mode: aquifer_constant."
            )

        if has_physical:
            assert self.inner_radius is not None
            assert self.outer_radius is not None
            assert self.aquifer_permeability is not None
            assert self.aquifer_porosity is not None
            assert self.aquifer_compressibility is not None
            assert self.water_viscosity is not None
            assert self.aquifer_thickness is not None

            if self.inner_radius <= 0:
                raise ValidationError("`inner_radius` must be positive.")
            if self.outer_radius <= self.inner_radius:
                raise ValidationError("`outer_radius` must be greater than `inner_radius`.")

            if self.unit_system != UnitSystem.FIELD:
                to_field = get_conversion_factors(self.unit_system, UnitSystem.FIELD)
                r_w_ft = self.inner_radius * to_field["length"]
                r_e_ft = self.outer_radius * to_field["length"]
                height_ft = self.aquifer_thickness * to_field["length"]
                compressibility_psi = self.aquifer_compressibility * to_field["compressibility"]
                permeability_md = self.aquifer_permeability * to_field["permeability"]
                viscosity_cp = self.water_viscosity * to_field["viscosity"]
                from_field = get_conversion_factors(UnitSystem.FIELD, self.unit_system)
            else:
                r_w_ft = self.inner_radius
                r_e_ft = self.outer_radius
                height_ft = self.aquifer_thickness
                compressibility_psi = self.aquifer_compressibility
                permeability_md = self.aquifer_permeability
                viscosity_cp = self.water_viscosity
                from_field = None

            r_d = r_e_ft / r_w_ft
            object.__setattr__(self, "resolved_dimensionless_radius_ratio", r_d)

            angle_fraction = self.angle / 360.0

            aquifer_constant_bbl_per_psi = (
                1.119
                * self.aquifer_porosity
                * compressibility_psi
                * (r_e_ft**2 - r_w_ft**2)
                * height_ft
                * angle_fraction
            )
            aquifer_constant_ft3_per_psi = aquifer_constant_bbl_per_psi * c.BARRELS_TO_CUBIC_FEET

            if from_field is not None:
                aquifer_constant = (
                    aquifer_constant_ft3_per_psi * from_field["volume"] / from_field["pressure"]
                )
            else:
                aquifer_constant = aquifer_constant_ft3_per_psi

            object.__setattr__(self, "resolved_aquifer_constant", aquifer_constant)

            hydraulic_diffusivity_ft2_per_day = (
                6.328e-3
                * permeability_md
                / (self.aquifer_porosity * viscosity_cp * compressibility_psi)
            )
            if from_field is not None:
                hydraulic_diffusivity = (
                    hydraulic_diffusivity_ft2_per_day
                    * (from_field["length"] ** 2)
                    / from_field["time"]
                )
            else:
                hydraulic_diffusivity = hydraulic_diffusivity_ft2_per_day

            object.__setattr__(self, "hydraulic_diffusivity", hydraulic_diffusivity)

        else:
            object.__setattr__(self, "resolved_aquifer_constant", self.aquifer_constant)
            object.__setattr__(
                self, "resolved_dimensionless_radius_ratio", self.dimensionless_radius_ratio
            )
            object.__setattr__(self, "hydraulic_diffusivity", None)

            if self.dimensionless_time_scale is None:
                warnings.warn(
                    f"{type(self).__name__!r} is in calibrated-constant mode "
                    "without `dimensionless_time_scale` set, so `tD` falls back "
                    "to raw elapsed `time`, and its scale depends on `unit_system`'s time unit. Supply "
                    "`dimensionless_time_scale` (tD per unit time, from your "
                    "history match) for physically meaningful transient "
                    "behaviour, or use physical-properties mode instead.",
                    stacklevel=2,
                )

            if not self.bounded_aquifer and self.dimensionless_radius_ratio != 10:
                warnings.warn(
                    f"{type(self).__name__!r} has a non-default "
                    f"`dimensionless_radius_ratio={self.dimensionless_radius_ratio!r}` "
                    "but `bounded_aquifer=False`, so it has no effect as the "
                    "aquifer is treated as infinite-acting regardless.",
                    stacklevel=2,
                )

        if self.bounded_aquifer:
            r_d = self.resolved_dimensionless_radius_ratio
            if r_d <= 1.0:
                raise ValidationError(
                    "`bounded_aquifer=True` requires a dimensionless radius "
                    f"ratio (r_e/r_w) greater than 1; resolved to {r_d!r}."
                )
            betas = compute_bessel_roots(r_d, c.AQUIFER_BESSEL_SERIES_TERMS)
            pd_coefficients, pd_prime_coefficients = compute_bessel_series_coefficients(r_d, betas)
            object.__setattr__(self, "bessel_roots", betas)
            object.__setattr__(self, "pd_coefficients", pd_coefficients)
            object.__setattr__(self, "pd_prime_coefficients", pd_prime_coefficients)
            object.__setattr__(self, "linear_coefficient", 2.0 / (r_d**2 - 1.0))
            second_term = -(3 * r_d**4 - 4 * r_d**4 * np.log(r_d) - 2 * r_d**2 - 1) / (
                4 * (r_d**2 - 1) ** 2
            )
            object.__setattr__(
                self, "constant_coefficient", self.linear_coefficient * 0.25 + second_term
            )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `CarterTracyAquifer` with all dimensional parameters
        rescaled to *target*.

        :param target: Target `UnitSystem`.
        :param table: Optional custom conversion table.
        :returns: New `CarterTracyAquifer` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        pressure_factor = factors["pressure"]
        length_factor = factors["length"]
        permeability_factor = factors["permeability"]
        viscosity_factor = factors["viscosity"]
        volume_factor = factors["volume"]
        compressibility_factor = factors["compressibility"]
        time_factor = factors["time"]

        return attrs.evolve(
            self,
            initial_pressure=self.initial_pressure * pressure_factor,
            aquifer_permeability=(
                self.aquifer_permeability * permeability_factor
                if self.aquifer_permeability is not None
                else None
            ),
            aquifer_porosity=self.aquifer_porosity,
            aquifer_compressibility=(
                self.aquifer_compressibility * compressibility_factor
                if self.aquifer_compressibility is not None
                else None
            ),
            water_viscosity=(
                self.water_viscosity * viscosity_factor
                if self.water_viscosity is not None
                else None
            ),
            inner_radius=(
                self.inner_radius * length_factor if self.inner_radius is not None else None
            ),
            outer_radius=(
                self.outer_radius * length_factor if self.outer_radius is not None else None
            ),
            aquifer_thickness=(
                self.aquifer_thickness * length_factor
                if self.aquifer_thickness is not None
                else None
            ),
            aquifer_constant=(
                self.aquifer_constant * volume_factor / pressure_factor
                if self.aquifer_constant is not None
                else None
            ),
            dimensionless_radius_ratio=self.dimensionless_radius_ratio,
            dimensionless_time_scale=(
                self.dimensionless_time_scale / time_factor
                if self.dimensionless_time_scale is not None
                else None
            ),
            bounded_aquifer=self.bounded_aquifer,
            angle=self.angle,
            unit_system=target,
        )


def load_carter_tracy_aquifer_from_record(
    record: typing.Mapping[str, typing.Any],
    unit_system: UnitSystem,
    *,
    outer_radius: Number,
    water_viscosity: Number,
) -> CarterTracyAquifer:
    """
    Build a `CarterTracyAquifer` from one `AQUCT` record, in
    physical-properties mode.

    `AQUCT` does not fully specify a physical-mode `CarterTracyAquifer` on
    its own, so two inputs have to come from elsewhere and are required
    keyword arguments here rather than silently defaulted:

    - `outer_radius` - `AQUCT`'s own `radius` item is the aquifer's inner
      (reservoir-contact) radius only; `resolved_aquifer_constant`'s
      `r_e^2 - r_w^2` term needs the aquifer's outer extent too, which
      `AQUCT` never supplies as a number. Eclipse instead controls
      boundedness through a referenced `AQUTAB` influence-function table,
      which this codebase does not parse yet - see the
      `aquifer_influence_table_number` warning below.
    - `water_viscosity` - `AQUCT` gives a `pvt_table_number` (available
      on `record["pvt_table_number"]`) rather than a viscosity value
      directly; resolving it means evaluating that water PVT table at
      `record["initial_pressure"]`, which this function does not do, so
      the resolved viscosity has to be supplied directly.

    :param record: One parsed `AQUCT` record.
    :param unit_system: The deck's unit system.
    :param outer_radius: Aquifer outer radius, `unit_system` units.
    :param water_viscosity: Water viscosity at aquifer conditions, `unit_system` units.
    :returns: Constructed `CarterTracyAquifer`, always `bounded_aquifer=False`
        - see the note on `aquifer_influence_table_number` above.
    """
    influence_table = record["aquifer_influence_table_number"]
    if influence_table != 1:
        warnings.warn(
            f"AQUCT aquifer {record['aquifer_id']!r} references AQUTAB table "
            f"{influence_table!r}, but custom AQUTAB influence functions are "
            "not parsed by this codebase yet. Building it as infinite-acting "
            "(bounded_aquifer=False) instead of honouring that table.",
            stacklevel=2,
        )

    return CarterTracyAquifer(
        initial_pressure=record["initial_pressure"],
        aquifer_permeability=record["permeability"],
        aquifer_porosity=record["porosity"],
        aquifer_compressibility=record["total_compressibility"],
        water_viscosity=water_viscosity,
        inner_radius=record["radius"],
        outer_radius=outer_radius,
        aquifer_thickness=record["thickness"],
        bounded_aquifer=False,
        angle=record["influence_angle"],
        unit_system=unit_system,
    )


def load_carter_tracy_aquifers_from_records(
    records: typing.Sequence[typing.Mapping[str, typing.Any]],
    unit_system: UnitSystem,
    *,
    outer_radii: typing.Mapping[int, Number],
    water_viscosities: typing.Mapping[int, Number],
) -> dict[int, CarterTracyAquifer]:
    """
    Build every `CarterTracyAquifer` a deck's `AQUCT` records define.

    :param records: Every `AQUCT` record in the deck.
    :param unit_system: The deck's unit system.
    :param outer_radii: Aquifer outer radius per `aquifer_id` - see
        `load_carter_tracy_aquifer_from_record` for why this can't be
        read from the record itself.
    :param water_viscosities: Water viscosity at aquifer conditions per
        `aquifer_id` - see `load_carter_tracy_aquifer_from_record`.
    :returns: `CarterTracyAquifer`s keyed by their deck `aquifer_id`,
        ready to be matched against `AQUANCON` connections by that same id.
    :raises DeckParseError: If an aquifer's id is missing from `outer_radii`
        or `water_viscosities`.
    """
    aquifers: dict[int, CarterTracyAquifer] = {}
    for record in records:
        aquifer_id = record["aquifer_id"]
        if aquifer_id not in outer_radii:
            raise DeckParseError(
                f"AQUCT aquifer {aquifer_id!r}: no `outer_radius` supplied in "
                "`outer_radii`. `AQUCT` doesn't specify the aquifer's outer "
                "extent - see `load_carter_tracy_aquifer_from_record`."
            )
        if aquifer_id not in water_viscosities:
            raise DeckParseError(
                f"AQUCT aquifer {aquifer_id!r}: no `water_viscosity` supplied "
                "in `water_viscosities`. `AQUCT` gives a PVT table reference "
                "(`pvt_table_number`), not a viscosity value directly - see "
                "`load_carter_tracy_aquifer_from_record`."
            )
        aquifers[aquifer_id] = load_carter_tracy_aquifer_from_record(
            record,
            unit_system,
            outer_radius=outer_radii[aquifer_id],
            water_viscosity=water_viscosities[aquifer_id],
        )
    return aquifers


def from_deck(
    deck_file: DeckFile,
    *,
    outer_radii: typing.Mapping[int, Number],
    water_viscosities: typing.Mapping[int, Number],
) -> dict[int, CarterTracyAquifer]:
    """
    Build every `CarterTracyAquifer` a deck defines, from its `AQUCT` keyword.

    :param deck_file: Parsed deck, read for its `AQUCT` records and unit system.
    :param outer_radii: Aquifer outer radius per `aquifer_id` - see
        `load_carter_tracy_aquifer_from_record` for why this can't be
        read from the deck.
    :param water_viscosities: Water viscosity at aquifer conditions per
        `aquifer_id` - see `load_carter_tracy_aquifer_from_record`.
    :returns: `CarterTracyAquifer`s keyed by their deck `aquifer_id`. Empty
        if the deck has no `AQUCT` keyword.
    :raises DeckParseError: If an aquifer's id is missing from `outer_radii`
        or `water_viscosities`.
    """
    records = deck_file.get("AQUCT") or []
    return load_carter_tracy_aquifers_from_records(
        records,
        unit_system=deck_file.unit_system,
        outer_radii=outer_radii,
        water_viscosities=water_viscosities,
    )

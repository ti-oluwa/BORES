import typing
import warnings

import attrs
import numba
import numpy as np
from scipy.optimize import root_scalar
from scipy.special import j1, y1
from typing_extensions import Self

from bores.constants import c, get_conversion_factors
from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.reservoir.boundary.base import (
    BoundaryCondition,
    BoundaryConditionType,
    InitialPressureFactory,
)
from bores.types import (
    Integer,
    Number,
    NumberArray,
    OneDimension,
    UnitConversionTable,
    UnitSystem,
)

if typing.TYPE_CHECKING:
    from bores.blackoil.pvt.regions import PVT

BUILT_IN_INFLUENCE_TABLE_NUMBER = 1
"""The influence table number that selects the built-in influence function."""

FIRST_INFLUENCE_TABLE_NUMBER = 2
"""The number of the first `AQUTAB` table, since table 1 is the built-in one."""

__all__ = [
    "AquiferInfluenceTable",
    "CarterTracyAquifer",
    "compute_incremental_influx",
    "compute_tabulated_dimensionless_pressure",
    "load_aquifer_influence_table",
    "load_aquifer_influence_table_from_records",
    "load_carter_tracy_aquifer",
    "load_carter_tracy_aquifer_from_record",
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


@attrs.frozen(slots=True)
class AquiferInfluenceTable:
    """
    A user-supplied dimensionless influence function `pD(tD)` for a Carter-Tracy aquifer,
    as given by the deck's `AQUTAB` keyword.
    """

    times: NumberArray[OneDimension]
    """Dimensionless times `tD`, not negative and strictly increasing."""

    pressures: NumberArray[OneDimension]
    """Dimensionless pressures `pD`, one for each entry of `times`."""

    def __attrs_post_init__(self) -> None:
        if self.times.ndim != 1 or self.times.shape != self.pressures.shape:
            raise ValidationError("An influence table needs one `pD` for each `tD`.")
        if self.times.shape[0] < 2:
            raise ValidationError("An influence table needs at least two rows.")
        if not (np.all(np.isfinite(self.times)) and np.all(np.isfinite(self.pressures))):
            raise ValidationError("An influence table can only hold finite values.")
        if self.times[0] < 0.0 or np.any(np.diff(self.times) <= 0.0):
            raise ValidationError(
                "The `tD` column of an influence table must not be negative and must "
                "strictly increase."
            )

    @typing.overload
    @classmethod
    def from_deck(cls, deck_file: DeckFile, *, table_number: Integer) -> Self: ...
    @typing.overload
    @classmethod
    def from_deck(cls, deck_file: DeckFile, *, table_number: None = None) -> dict[int, Self]: ...

    @classmethod
    def from_deck(
        cls, deck_file: DeckFile, *, table_number: Integer | None = None
    ) -> Self | dict[int, Self]:
        """
        Construct one or all `AquiferInfluenceTable` objects from a parsed `DeckFile`.

        Reads the `AQUTAB` keyword (see `load_aquifer_influence_table`).

        :param deck_file: Parsed `bores.deck.file.DeckFile`.
        :param table_number: Number of a specific table to extract, or `None` for all.
            `AQUTAB` tables are numbered from 2 in the order they appear.
        :returns: A single `AquiferInfluenceTable` if `table_number` is given; a
            `dict[int, AquiferInfluenceTable]` keyed by table number otherwise, empty
            if the deck has no `AQUTAB` keyword.
        :raises ValidationError: If `table_number` is given but not found, or a table
            is not valid.
        """
        return typing.cast(
            Self | dict[int, Self],
            load_aquifer_influence_table(deck_file, table_number=table_number),
        )


@numba.njit(cache=True)
def compute_tabulated_dimensionless_pressure(
    t_d: Number,
    times: NumberArray[OneDimension],
    pressures: NumberArray[OneDimension],
) -> tuple[Number, Number]:
    """
    `pD` and `pD'` at `tD`, read from a table by linear interpolation.

    Before the first row the table is joined to the origin by a straight line, or if the
    first row is at `tD = 0`, `pD` stays at its value there. After the last row the slope
    of the last two rows is continued.

    :param t_d: Dimensionless time.
    :param times: Table `tD` column, increasing.
    :param pressures: Table `pD` column.
    :returns: `(pD, pD')`, where `pD'` is the slope of the segment `tD` falls in.
    """
    n_rows = times.shape[0]
    if t_d <= times[0]:
        if times[0] > 0.0:
            slope = pressures[0] / times[0]
            return slope * t_d, slope
        slope = (pressures[1] - pressures[0]) / (times[1] - times[0])
        return pressures[0], slope

    if t_d >= times[n_rows - 1]:
        slope = (pressures[n_rows - 1] - pressures[n_rows - 2]) / (
            times[n_rows - 1] - times[n_rows - 2]
        )
        return pressures[n_rows - 1] + slope * (t_d - times[n_rows - 1]), slope

    upper = np.searchsorted(times, t_d)
    lower = upper - 1
    slope = (pressures[upper] - pressures[lower]) / (times[upper] - times[lower])
    return pressures[lower] + slope * (t_d - times[lower]), slope


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
    influence_times: NumberArray[OneDimension],
    influence_pressures: NumberArray[OneDimension],
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
    :param influence_times: The aquifer's `AQUTAB` table `tD` column. Zero-length when the
        aquifer uses the built-in influence function. A table takes precedence over `bounded`.
    :param influence_pressures: Matching `pD` column, zero-length with `influence_times`.
    :returns: `(We)_n`, computed from the last committed state, not written anywhere.
    """
    delta_t_d = current_dimensionless_time - previous_dimensionless_time
    if delta_t_d <= 0.0:
        return previous_cumulative_influx

    if influence_times.shape[0] > 0:
        current_p_d, current_p_d_prime = compute_tabulated_dimensionless_pressure(
            current_dimensionless_time, influence_times, influence_pressures
        )
    elif bounded and current_dimensionless_time >= compute_bounded_aquifer_threshold(
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

    **Aquifer constant** (FIELD units, Carter & Tracy 1960, Eq. 2; Ahmed,
    Reservoir Engineering Handbook):

        aquifer_constant = 1.119 * φ * ct * r_w² * h * f

    where `f = θ/360` is the encroachment angle fraction and `r_w` is the
    reservoir/aquifer contact radius (`inner_radius`) in ft. This is the
    same radius `tD` uses - the aquifer's own outer extent (`outer_radius`)
    plays no part in either formula; it only sets `r_eD = r_e/r_w` for the
    bounded-aquifer Bessel series below.

    **pD and pD' approximations**: Edwardson et al. (1962) polynomial for
    `tD <= 100`, logarithmic approximation for `tD > 100` - see
    `compute_infinite_dimensionless_pressure`.

    **Two construction modes**:

    *Physical-properties mode*: supply `aquifer_permeability`,
    `aquifer_porosity`, `aquifer_compressibility`, `water_viscosity`,
    `inner_radius`, `aquifer_thickness`. `aquifer_constant` and hydraulic
    diffusivity are derived automatically in FIELD units then stored in
    `unit_system` units. Also supply `outer_radius` if `bounded_aquifer=True`
    (it isn't needed otherwise).

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

    **Influence tables**: set `influence_table` to an `AquiferInfluenceTable` (the deck's
    `AQUTAB` keyword) to use a user-supplied `pD(tD)` and its slope instead of the
    built-in infinite or bounded functions. The table takes precedence over
    `bounded_aquifer`. It is dimensionless, so unit conversion leaves it unchanged.

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
    Outer aquifer extent. Physical mode only, and only needed when
    `bounded_aquifer=True`, to set the transient response's
    `r_eD = r_e/r_w`. Plays no part in `aquifer_constant`, which depends
    on `inner_radius` alone. Ignored (with a warning if set) when
    `bounded_aquifer=False`.
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
    `r_e / r_w`. Calibrated-constant mode, and physical mode when
    `outer_radius` is left unset, both use this directly; physical mode
    with `outer_radius` given derives `r_e / r_w` from
    `outer_radius`/`inner_radius` instead and ignores this field. Affects
    the transient response shape only when `bounded_aquifer=True`;
    otherwise stored for the record only.
    """

    bounded_aquifer: bool = attrs.field(default=False)
    """
    Opt-in: use the Klins, Bouchard & Cable (1988) finite/bounded-aquifer
    `pD(tD, r_eD)` once `tD` passes `compute_bounded_aquifer_threshold(r_eD)`,
    instead of always treating the aquifer as infinite-acting. Defaults
    to `False` for backward compatibility.
    """

    dimensionless_time_scale: Number | None = attrs.field(default=None)
    """
    `tD / t` - dimensionless time per unit of `unit_system` time.
    Calibrated-constant mode only, optional but recommended. When set,
    `tD = dimensionless_time_scale * t`. When left `None`, `tD` falls
    back to raw elapsed `time`. Dimensionally meaningless and dependent
    on `unit_system`'s time unit.
    """

    angle: Number = attrs.field(default=360.0)
    """Aquifer encroachment angle in degrees."""

    unit_system: UnitSystem = attrs.field(default=UnitSystem.FIELD)
    """Unit system for all dimensional parameters and returned flux values."""

    aquifer_id: int | None = attrs.field(default=None)
    """
    Aquifer identification number, when built `from_deck`. The `AQUCT`
    record's own `aquifer_id`, referenced by `AQUANCON` to attach this
    aquifer to grid connections. Record-keeping only; `None` when built
    directly rather than from a deck.
    """

    influence_table: AquiferInfluenceTable | None = attrs.field(default=None)
    """
    Dimensionless influence function from the deck's `AQUTAB` keyword. When set it is used
    in place of the built-in infinite or bounded aquifer functions. `None` uses those.
    """

    pvt_table_number: int | None = attrs.field(default=None)
    """
    Water PVT table number, usually when built `from_deck`. The `AQUCT` record's
    own `pvt_table_number`, used to resolve `water_viscosity` from a `PVT`
    object at load time. Record-keeping only afterwards; not read anywhere
    in the class or the recurrence itself.
    """

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
                self.aquifer_thickness,
            )
        )
        has_calibrated = self.aquifer_constant is not None

        if not (has_physical or has_calibrated):
            raise ValidationError(
                f"{type(self).__name__!r} requires either:\n"
                "  Physical-properties mode: aquifer_permeability, aquifer_porosity,\n"
                "    aquifer_compressibility, water_viscosity, inner_radius,\n"
                "    aquifer_thickness. `outer_radius` is also needed if\n"
                "    `bounded_aquifer=True`.\n"
                "  Calibrated-constant mode: aquifer_constant."
            )

        if has_physical:
            assert self.inner_radius is not None
            assert self.aquifer_permeability is not None
            assert self.aquifer_porosity is not None
            assert self.aquifer_compressibility is not None
            assert self.water_viscosity is not None
            assert self.aquifer_thickness is not None

            if self.inner_radius <= 0:
                raise ValidationError("`inner_radius` must be positive.")
            if self.outer_radius is not None and self.outer_radius <= self.inner_radius:
                raise ValidationError("`outer_radius` must be greater than `inner_radius`.")

            if self.unit_system != UnitSystem.FIELD:
                to_field = get_conversion_factors(self.unit_system, UnitSystem.FIELD)
                r_w_ft = self.inner_radius * to_field["length"]
                r_e_ft = (
                    self.outer_radius * to_field["length"]
                    if self.outer_radius is not None
                    else None
                )
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

            # `r_eD = r_e/r_w` only matters for the bounded-aquifer Bessel
            # series (see the `if self.bounded_aquifer:` block below), not
            # for `aquifer_constant`. Fall back to `dimensionless_radius_ratio`
            # (same as calibrated-constant mode) when `outer_radius` isn't given.
            r_d = r_e_ft / r_w_ft if r_e_ft is not None else self.dimensionless_radius_ratio
            object.__setattr__(self, "resolved_dimensionless_radius_ratio", r_d)

            if not self.bounded_aquifer and self.outer_radius is not None:
                warnings.warn(
                    f"{type(self).__name__!r} was given `outer_radius`, but "
                    "`bounded_aquifer=False`, so it has no effect as the "
                    "aquifer is treated as infinite-acting regardless.",
                    stacklevel=2,
                )

            angle_fraction = self.angle / 360.0

            # Carter & Tracy (1960), Eq. 2 / Ahmed, Reservoir Engineering
            # Handbook: B = 1.119 * phi * ct * r_w^2 * h * f - one radius
            # (the reservoir/aquifer contact radius), not `r_e^2 - r_w^2`.
            # Matches the Handbook's own worked example to 3 s.f.:
            # 1.119*0.2*1e-6*2000**2*25*1.0 = 22.38 ~ 22.4 bbl/psi.
            aquifer_constant_bbl_per_psi = (
                1.119
                * self.aquifer_porosity
                * compressibility_psi
                * (r_w_ft**2)
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

    @typing.overload
    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        pvt: "PVT",
        aquifer_id: Integer,
        get_initial_pressure: InitialPressureFactory | None = None,
    ) -> Self: ...
    @typing.overload
    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        pvt: "PVT",
        aquifer_id: None = None,
        get_initial_pressure: InitialPressureFactory | None = None,
    ) -> dict[int, Self]: ...

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        pvt: "PVT",
        aquifer_id: Integer | None = None,
        get_initial_pressure: InitialPressureFactory | None = None,
    ) -> Self | dict[int, Self]:
        """
        Construct one or all `CarterTracyAquifer` objects from a parsed `DeckFile`.

        Reads the `AQUCT` keyword, and the `AQUTAB` keyword for any aquifer that uses
        a custom influence table (see `load_carter_tracy_aquifer`).

        :param deck_file: Parsed `bores.deck.file.DeckFile`.
        :param pvt: The model's PVT tables, from which `water_viscosity` is resolved.
        :param aquifer_id: Id of a specific aquifer to extract, or `None` for all.
        :param get_initial_pressure: Called with an `AQUCT` record whose initial
            pressure was defaulted (`1*`) and returns the pressure to use. Without it,
            a defaulted initial pressure is an error.
        :returns: A single `CarterTracyAquifer` if `aquifer_id` is given;
            a `dict[int, CarterTracyAquifer]` otherwise.
        :raises ValidationError: If the deck has no `AQUCT` keyword, `aquifer_id` is
            given but not found in it, or an aquifer cannot be built.
        """
        return typing.cast(
            Self | dict[int, Self],
            load_carter_tracy_aquifer(
                deck_file,
                pvt=pvt,
                aquifer_id=aquifer_id,
                get_initial_pressure=get_initial_pressure,
            ),
        )


def load_aquifer_influence_table_from_records(
    records: typing.Sequence[typing.Mapping[str, typing.Any]],
) -> AquiferInfluenceTable:
    """
    Build an `AquiferInfluenceTable` from the rows of one `AQUTAB` table.

    :param records: The table's rows, each with `dimensionless_time` and
        `dimensionless_pressure`.
    :returns: The influence table.
    :raises ValidationError: If the rows do not make a valid table.
    """
    return AquiferInfluenceTable(
        times=typing.cast(
            NumberArray[OneDimension],
            np.asarray([record["dimensionless_time"] for record in records], dtype=np.float64),
        ),
        pressures=typing.cast(
            NumberArray[OneDimension],
            np.asarray([record["dimensionless_pressure"] for record in records], dtype=np.float64),
        ),
    )


@typing.overload
def load_aquifer_influence_table(
    deck_file: DeckFile, *, table_number: Integer
) -> AquiferInfluenceTable: ...
@typing.overload
def load_aquifer_influence_table(
    deck_file: DeckFile, *, table_number: None = None
) -> dict[int, AquiferInfluenceTable]: ...


def load_aquifer_influence_table(
    deck_file: DeckFile, *, table_number: Integer | None = None
) -> AquiferInfluenceTable | dict[int, AquiferInfluenceTable]:
    """
    Load one or all `AQUTAB` influence tables of a deck.

    Table 1 is the built-in influence function, so `AQUTAB` tables are numbered from 2 in
    the order they appear.

    :param deck_file: Parsed `bores.deck.file.DeckFile`.
    :param table_number: Number of a specific table to load, or `None` for all.
    :returns: A single `AquiferInfluenceTable` if `table_number` is given; otherwise a
        `dict[int, AquiferInfluenceTable]` keyed by table number, empty if the deck has
        no `AQUTAB` keyword.
    :raises ValidationError: If `table_number` is given but not found, or a table is not
        valid.
    """
    deck_tables = deck_file.get("AQUTAB") or []
    tables = {
        number: load_aquifer_influence_table_from_records(records)
        for number, records in enumerate(deck_tables, start=FIRST_INFLUENCE_TABLE_NUMBER)
    }
    if table_number is None:
        return tables
    if int(table_number) not in tables:
        raise ValidationError(
            f"Influence table {table_number!r} not found in `AQUTAB`. Available: "
            f"{sorted(tables)}. Table 1 is the built-in influence function and has no "
            "`AQUTAB` data."
        )
    return tables[int(table_number)]


def load_carter_tracy_aquifer_from_record(
    record: typing.Mapping[str, typing.Any],
    unit_system: UnitSystem,
    *,
    pvt: "PVT",
    get_initial_pressure: InitialPressureFactory | None = None,
    influence_tables: typing.Mapping[int, AquiferInfluenceTable] | None = None,
) -> CarterTracyAquifer:
    """
    Build a `CarterTracyAquifer` from one `AQUCT` record, in
    physical-properties mode.

    `AQUCT` gives every physical-mode input directly except water
    viscosity. Item 10 (`pvt_table_number`) is a water PVT table
    reference, not a value, so it's resolved here as `pvt.region(
    pvt_table_number).static.water_reference_viscosity`, the `PVTW`
    reference viscosity for that region, evaluated once rather than
    re-interpolated per timestep, matching this class's own
    constant-viscosity assumption. `AQUCT`'s single `radius` item maps to
    `inner_radius`; `outer_radius` is left unset.

    :param record: One parsed `AQUCT` record.
    :param unit_system: The deck's unit system.
    :param pvt: The model's PVT tables, covering at least
        `record["pvt_table_number"]`'s region (e.g. `PVT.from_deck(deck_file, ...)`).
    :param get_initial_pressure: Called with `record` to get the initial pressure
        when the record defaults it (`1*`).
    :param influence_tables: The deck's `AQUTAB` tables by table number, which starts at 2.
        Needed when the record's `aquifer_influence_table_number` is not 1.
    :returns: Constructed `CarterTracyAquifer`, always `bounded_aquifer=False`. It uses
        the record's `AQUTAB` influence table when it names one, and the built-in
        infinite-acting influence function otherwise.
    :raises ValidationError: If `pvt`'s region for `record["pvt_table_number"]`
        has no `PVTW`-derived reference viscosity, the record defaults its initial
        pressure and `get_initial_pressure` is not given, or the record names an
        influence table that `influence_tables` does not have.
    """
    aquifer_id = record["aquifer_id"]
    pvt_table_number = record["pvt_table_number"]

    influence_table_number = record["aquifer_influence_table_number"]
    influence_table: AquiferInfluenceTable | None = None
    if influence_table_number != BUILT_IN_INFLUENCE_TABLE_NUMBER:
        if influence_tables is None or influence_table_number not in influence_tables:
            raise ValidationError(
                f"`AQUCT` aquifer {aquifer_id!r} uses influence table {influence_table_number!r}, "
                "but the deck has no such `AQUTAB` table. Tables are numbered from "
                f"{FIRST_INFLUENCE_TABLE_NUMBER} in the order they appear, and table "
                f"{BUILT_IN_INFLUENCE_TABLE_NUMBER} is the built-in influence function."
            )
        influence_table = influence_tables[typing.cast(int, influence_table_number)]

    water_viscosity = pvt.region(pvt_table_number).static.water_reference_viscosity
    if water_viscosity is None:
        raise ValidationError(
            f"`AQUCT` aquifer {aquifer_id!r}: PVT region {pvt_table_number!r} "
            "has no `PVTW`-derived `water_reference_viscosity`. `AQUCT` "
            "references this region as its water PVT table but the deck's "
            "`PVTW` keyword doesn't cover it."
        )

    initial_pressure = record["initial_pressure"]
    if initial_pressure is None:
        if get_initial_pressure is None:
            raise ValidationError(
                f"`AQUCT` aquifer {aquifer_id!r} defaults its initial pressure (`1*`), "
                "which is taken from the initial reservoir pressure. Load it through "
                "`load_boundary_conditions` with `reservoir_pressure`, or give the "
                "pressure in the deck."
            )
        initial_pressure = get_initial_pressure(record)

    return CarterTracyAquifer(
        aquifer_id=aquifer_id,
        initial_pressure=initial_pressure,
        aquifer_permeability=record["permeability"],
        aquifer_porosity=record["porosity"],
        aquifer_compressibility=record["total_compressibility"],
        water_viscosity=water_viscosity,
        inner_radius=record["radius"],
        aquifer_thickness=record["thickness"],
        bounded_aquifer=False,
        angle=record["influence_angle"],
        influence_table=influence_table,
        pvt_table_number=pvt_table_number,
        unit_system=unit_system,
    )


@typing.overload
def load_carter_tracy_aquifer(
    deck_file: DeckFile,
    *,
    pvt: "PVT",
    aquifer_id: Integer,
    get_initial_pressure: InitialPressureFactory | None = None,
) -> CarterTracyAquifer: ...
@typing.overload
def load_carter_tracy_aquifer(
    deck_file: DeckFile,
    *,
    pvt: "PVT",
    aquifer_id: None = None,
    get_initial_pressure: InitialPressureFactory | None = None,
) -> dict[int, CarterTracyAquifer]: ...


def load_carter_tracy_aquifer(
    deck_file: DeckFile,
    *,
    pvt: "PVT",
    aquifer_id: Integer | None = None,
    get_initial_pressure: InitialPressureFactory | None = None,
) -> CarterTracyAquifer | dict[int, CarterTracyAquifer]:
    """
    Load one or all `AQUCT` aquifers of a deck as `CarterTracyAquifer`s.

    An aquifer whose record names an influence table other than 1 takes it from the
    deck's `AQUTAB` keyword (see `load_aquifer_influence_table`).

    :param deck_file: Parsed `bores.deck.file.DeckFile`.
    :param pvt: The model's PVT tables, from which `water_viscosity` is resolved (see
        `load_carter_tracy_aquifer_from_record`).
    :param aquifer_id: Id of a specific aquifer to load, or `None` for all.
    :param get_initial_pressure: Called with an `AQUCT` record whose initial pressure was
        defaulted (`1*`) and returns the pressure to use. Without it, a defaulted initial
        pressure is an error.
    :returns: A single `CarterTracyAquifer` if `aquifer_id` is given; otherwise a
        `dict[int, CarterTracyAquifer]` keyed by aquifer id.
    :raises ValidationError: If the deck has no `AQUCT` keyword, `aquifer_id` is given but
        not found in it, or an aquifer cannot be built (see
        `load_carter_tracy_aquifer_from_record`).
    """
    records = deck_file.get("AQUCT")
    if not records:
        raise ValidationError("No `AQUCT` keyword found in the provided deck.")

    influence_tables = load_aquifer_influence_table(deck_file)
    if aquifer_id is not None:
        matching = [record for record in records if record["aquifer_id"] == aquifer_id]
        if not matching:
            available = sorted(record["aquifer_id"] for record in records)
            raise ValidationError(
                f"Aquifer {aquifer_id!r} not found in `AQUCT`. Available: {available}."
            )
        return load_carter_tracy_aquifer_from_record(
            matching[0],
            deck_file.unit_system,
            pvt=pvt,
            get_initial_pressure=get_initial_pressure,
            influence_tables=influence_tables,
        )

    return {
        record["aquifer_id"]: load_carter_tracy_aquifer_from_record(
            record,
            deck_file.unit_system,
            pvt=pvt,
            get_initial_pressure=get_initial_pressure,
            influence_tables=influence_tables,
        )
        for record in records
    }

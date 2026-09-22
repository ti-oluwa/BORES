import typing

import attrs
import numba
import numpy as np
from typing_extensions import Self

from bores.constants import c, get_conversion_factors
from bores.errors import ValidationError
from bores.reservoir.boundary.base import (
    BoundaryCondition,
    BoundaryConditionType,
    boundary_condition,
)
from bores.types import Number, UnitConversionTable, UnitSystem

__all__ = ["FetkovichAquifer", "compute_incremental_influx"]


@numba.njit(cache=True)
def compute_incremental_influx(
    previous_cumulative_influx: Number,
    previous_aquifer_pressure: Number,
    boundary_pressure: Number,
    productivity_index: Number,
    initial_encroachable_water: Number,
    initial_pressure: Number,
    elapsed_time: Number,
) -> tuple[Number, Number]:
    """
    One Fetkovich (1971) recursive step.

    A pure function of its arguments that reads no workspace state and
    writes nothing. Safe to call every Newton/Picard iteration within a
    single not yet accepted timestep (each call with the same arguments
    returns the same result); the caller decides separately whether to
    treat this as a trial value or to write the result into
    `AquiferWorkspace` as the new committed state.

        ΔWe = (We_i/p_i) · (p̄_a - p̄_R) · [1 - exp(-J·p_i·Δt / We_i)]
        We_new = We_previous + ΔWe
        p̄_a,new = p_i · (1 - We_new/We_i)

    where `p̄_a` is the aquifer's own average pressure (declining as it
    depletes) and `p̄_R` is the current reservoir boundary pressure driving
    the influx.

    :param previous_cumulative_influx: `We` as of the last committed
        state (reservoir volume).
    :param previous_aquifer_pressure: The aquifer's own average pressure
        as of the last committed state (pressure).
    :param boundary_pressure: Current reservoir boundary pressure driving
        the influx this step (pressure) - a trial value mid-iteration, or
        the accepted value when committing.
    :param productivity_index: Aquifer productivity index `J` (volume/time/pressure).
    :param initial_encroachable_water: `We_i`, the aquifer's total
        encroachable capacity (reservoir volume).
    :param initial_pressure: `p_i`, the aquifer's initial pressure (pressure).
    :param elapsed_time: Time since the last committed state (time).
    :returns: `(new_cumulative_influx, new_aquifer_pressure)`, computed
        from the last committed state and `boundary_pressure`, not
        written anywhere.
    """
    pressure_drop = previous_aquifer_pressure - boundary_pressure
    exponent = -productivity_index * initial_pressure * elapsed_time / initial_encroachable_water
    incremental_influx = (
        (initial_encroachable_water / initial_pressure) * pressure_drop * (1.0 - np.exp(exponent))
    )
    new_cumulative_influx = previous_cumulative_influx + incremental_influx
    new_aquifer_pressure = initial_pressure * (
        1.0 - new_cumulative_influx / initial_encroachable_water
    )
    return new_cumulative_influx, new_aquifer_pressure


@boundary_condition
@attrs.frozen(slots=True)
class FetkovichAquifer(BoundaryCondition):
    """
    Parameters for a water influx aquifer using the Fetkovich (1971)
    simplified material-balance approximation for a finite aquifer.

    A pure parameter container that resolves its physical or calibrated
    inputs into `J` (productivity index) and `We_i` (initial encroachable
    water) once at construction, and carries no other state. The
    recurrence itself (`compute_incremental_influx`) is a free function
    operating on `AquiferWorkspace` arrays.

    This class never runs on the hot path, only compiles into one row of
    `CompiledAquifers`.

    Models the aquifer as a finite tank with a fixed total encroachable
    water capacity, draining into the reservoir through a productivity
    index in the same way a well drains a reservoir. Unlike Carter-Tracy
    (a transient solution to the diffusivity equation), Fetkovich is a
    pseudo-steady-state approximation. It is simpler, cheaper, and depletes
    exactly once its capacity is exhausted, at the cost of some accuracy
    early in an aquifer's transient response.

    **Fetkovich recurrence** (Fetkovich, 1971, Eqs. 8-10; see also Ahmed,
    Reservoir Engineering Handbook, Fetkovich chapter):

        ΔWe_n = (We_i/p_i) · (p̄_a,n-1 - p̄_R,n) · [1 - exp(-J·p_i·Δt_n / We_i)]
        We_n = We_n-1 + ΔWe_n
        p̄_a,n = p_i · (1 - We_n/We_i)

    where:

    - `We_n` - cumulative influx at step n (reservoir volume).
    - `p̄_a,n` - the aquifer's own average pressure at step n, declining
      linearly with cumulative fractional depletion.
    - `p̄_R,n` - current reservoir boundary pressure.
    - `J` - aquifer productivity index (volume/time/pressure).
    - `We_i` - initial encroachable water (the aquifer's total capacity).
    - `p_i` - initial aquifer pressure.

    The influx rate for a timestep is the incremental influx divided by
    the elapsed time: `q_n = ΔWe_n / Δt_n`.

    **Initial encroachable water** (Fetkovich, 1971, Eq. 3):

        We_i = c_t * W_i * p_i

    where `W_i` is the initial volume of water in the aquifer (reservoir
    volume) and `c_t` is total aquifer compressibility.

    **Aquifer productivity index**, physical-properties mode only
    (pseudo-steady-state radial flow, FIELD units, bbl/day/psi; Ahmed,
    Reservoir Engineering Handbook):

        J = 0.00708 * k * h * f / [μ_w * (ln(r_e/r_w) - 0.75)]

    where `f = θ/360` is the encroachment angle fraction. The `-0.75`
    pseudo-steady-state regime constant matches this codebase's own
    `compute_peaceman_well_index`'s `regime_constant=-3/4` convention for
    a no-flow-outer-boundary drainage volume - physically, a finite
    aquifer draining into the reservoir is the same problem as a well
    draining a bounded reservoir, just with the roles of "well" and
    "reservoir" reversed.

    **Three construction modes**:

    *Physical-properties mode*: supply `aquifer_permeability`,
    `aquifer_porosity`, `aquifer_compressibility`, `water_viscosity`,
    `inner_radius`, `outer_radius`, `aquifer_thickness`. Both `J` and
    `We_i` are derived automatically in FIELD units then stored in
    `unit_system` units.

    *Calibrated mode, direct*: supply `aquifer_productivity_index` and
    `initial_encroachable_water` directly. Useful when both are already
    known from a history match.

    *Calibrated mode, from deck*: supply `aquifer_productivity_index`,
    `aquifer_compressibility`, and `initial_aquifer_water_volume`; `We_i`
    is derived as `c_t * W_i * p_i`. This matches Eclipse's own `AQUFETP`
    record exactly, which gives `J`, `c_t`, and `W_i` directly rather than
    `We_i` or aquifer geometry.

    **Unit system**: all user-supplied dimensional inputs must be in
    `unit_system`. Internally, the FIELD-unit constant (0.00708) is
    applied after converting physical-mode inputs to FIELD; `J` and
    `We_i` are then converted back to `unit_system` for storage.

    **References**:

    - Fetkovich, M.J. (1971). *A Simplified Approach to Water Influx
      Calculations - Finite Aquifer Systems.* JPT, 23(7), 814-828.
    - Ahmed, T. (2010). *Reservoir Engineering Handbook*, 4th ed.
      Gulf Professional Publishing. (Fetkovich chapter.)
    """

    __type__: typing.ClassVar[str] = "fetkovich_aquifer"

    condition_type: typing.ClassVar[BoundaryConditionType] = BoundaryConditionType.FLUX

    initial_pressure: Number
    """Initial aquifer pressure in `unit_system` pressure units."""

    aquifer_permeability: Number | None = attrs.field(default=None)
    """Aquifer permeability. Physical mode only."""

    aquifer_porosity: Number | None = attrs.field(default=None)
    """Aquifer porosity (fraction). Physical mode only."""

    aquifer_compressibility: Number | None = attrs.field(default=None)
    """
    Total aquifer compressibility.

    Required in physical mode (derives `J`) and in calibrated
    mode-from-deck (derives `We_i` alongside `initial_aquifer_water_volume`).
    """

    water_viscosity: Number | None = attrs.field(default=None)
    """Water viscosity at reservoir conditions. Physical mode only."""

    inner_radius: Number | None = attrs.field(default=None)
    """Reservoir-aquifer contact radius. Physical mode only."""

    outer_radius: Number | None = attrs.field(default=None)
    """Outer aquifer extent. Physical mode only."""

    aquifer_thickness: Number | None = attrs.field(default=None)
    """Aquifer thickness. Physical mode only."""

    aquifer_productivity_index: Number | None = attrs.field(default=None)
    """
    Pre-computed or history-matched aquifer productivity index `J`
    (volume/time/pressure in `unit_system`). Either calibrated mode only.
    """

    initial_encroachable_water: Number | None = attrs.field(default=None)
    """
    Pre-computed or history-matched `We_i` (reservoir volume in
    `unit_system`). Calibrated mode, direct, only. When `None` in
    calibrated mode, derived from `aquifer_compressibility` and
    `initial_aquifer_water_volume` instead.
    """

    initial_aquifer_water_volume: Number | None = attrs.field(default=None)
    """
    `W_i`, the aquifer's initial water volume (reservoir volume in
    `unit_system`). Calibrated mode-from-deck only - used with
    `aquifer_compressibility` to derive `We_i = c_t * W_i * p_i`. Ignored
    if `initial_encroachable_water` is given directly.
    """

    angle: Number = attrs.field(default=360.0)
    """Aquifer encroachment angle in degrees. Physical mode only."""

    unit_system: UnitSystem = attrs.field(default=UnitSystem.FIELD)
    """Unit system for all dimensional parameters."""

    productivity_index: Number = attrs.field(default=0.0, init=False, repr=False)
    """Resolved `J` in `unit_system` units. Set on initialization. Compiles into `CompiledAquifers`."""

    encroachable_water: Number = attrs.field(default=0.0, init=False, repr=False)
    """Resolved `We_i` in `unit_system` units. Set on initialization. Compiles into `CompiledAquifers`."""

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
        has_calibrated_direct = (
            self.aquifer_productivity_index is not None
            and self.initial_encroachable_water is not None
        )
        has_calibrated_from_deck = (
            self.aquifer_productivity_index is not None
            and self.aquifer_compressibility is not None
            and self.initial_aquifer_water_volume is not None
        )

        if not (has_physical or has_calibrated_direct or has_calibrated_from_deck):
            raise ValidationError(
                f"{type(self).__name__!r} requires one of:\n"
                "  Physical-properties mode: aquifer_permeability, aquifer_porosity,\n"
                "    aquifer_compressibility, water_viscosity, inner_radius,\n"
                "    outer_radius, aquifer_thickness.\n"
                "  Calibrated mode, direct: aquifer_productivity_index,\n"
                "    initial_encroachable_water.\n"
                "  Calibrated mode, from deck: aquifer_productivity_index,\n"
                "    aquifer_compressibility, initial_aquifer_water_volume."
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
                initial_pressure_psi = self.initial_pressure * to_field["pressure"]
                from_field = get_conversion_factors(UnitSystem.FIELD, self.unit_system)
            else:
                r_w_ft = self.inner_radius
                r_e_ft = self.outer_radius
                height_ft = self.aquifer_thickness
                compressibility_psi = self.aquifer_compressibility
                permeability_md = self.aquifer_permeability
                viscosity_cp = self.water_viscosity
                initial_pressure_psi = self.initial_pressure
                from_field = None

            angle_fraction = self.angle / 360.0

            # J = 0.00708*k*h*f / [μ*(ln(re/rw) - 0.75)]  [bbl/day/psi, FIELD]
            productivity_index_bbl_per_day_psi = (
                0.00708 * permeability_md * height_ft * angle_fraction
            ) / (viscosity_cp * (np.log(r_e_ft / r_w_ft) - 0.75))

            # W_i = pi*(re^2-rw^2)*h*phi*f / BARRELS_TO_CUBIC_FEET  [bbl, FIELD]
            initial_water_volume_bbl = (
                np.pi
                * (r_e_ft**2 - r_w_ft**2)
                * height_ft
                * self.aquifer_porosity
                * angle_fraction
                / c.BARRELS_TO_CUBIC_FEET
            )
            # We_i = ct * W_i * p_i  [bbl, FIELD]
            encroachable_water_bbl = (
                compressibility_psi * initial_water_volume_bbl * initial_pressure_psi
            )

            if from_field is not None:
                productivity_index = (
                    productivity_index_bbl_per_day_psi
                    * c.BARRELS_TO_CUBIC_FEET
                    * from_field["volume"]
                    / from_field["time"]
                    / from_field["pressure"]
                )
                encroachable_water = (
                    encroachable_water_bbl * c.BARRELS_TO_CUBIC_FEET * from_field["volume"]
                )
            else:
                productivity_index = productivity_index_bbl_per_day_psi
                encroachable_water = encroachable_water_bbl

            object.__setattr__(self, "productivity_index", productivity_index)
            object.__setattr__(self, "encroachable_water", encroachable_water)

        else:
            assert self.aquifer_productivity_index is not None
            object.__setattr__(self, "productivity_index", self.aquifer_productivity_index)

            if self.initial_encroachable_water is not None:
                object.__setattr__(self, "encroachable_water", self.initial_encroachable_water)
            else:
                assert self.aquifer_compressibility is not None
                assert self.initial_aquifer_water_volume is not None
                object.__setattr__(
                    self,
                    "encroachable_water",
                    self.aquifer_compressibility
                    * self.initial_aquifer_water_volume
                    * self.initial_pressure,
                )

        if self.encroachable_water <= 0:
            raise ValidationError(
                "Resolved `encroachable_water` (We_i) must be positive; got "
                f"{self.encroachable_water!r}. Check aquifer_compressibility/"
                "aquifer_porosity/initial_pressure are all positive."
            )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `FetkovichAquifer` with every dimensional field
        rescaled to *target*.

        :param target: Target `UnitSystem`.
        :returns: New `FetkovichAquifer` in *target* units.
        """
        if target == self.unit_system:
            return self
        factors = get_conversion_factors(self.unit_system, target, table=table)

        def convert_optional(value: Number | None, factor: Number) -> Number | None:
            return value * factor if value is not None else None

        rate_factor = factors["reservoir_rate"] / factors["pressure"]
        return attrs.evolve(
            self,
            initial_pressure=self.initial_pressure * factors["pressure"],
            aquifer_permeability=convert_optional(
                self.aquifer_permeability, factors["permeability"]
            ),
            aquifer_porosity=self.aquifer_porosity,
            aquifer_compressibility=convert_optional(
                self.aquifer_compressibility, factors["compressibility"]
            ),
            water_viscosity=convert_optional(self.water_viscosity, factors["viscosity"]),
            inner_radius=convert_optional(self.inner_radius, factors["length"]),
            outer_radius=convert_optional(self.outer_radius, factors["length"]),
            aquifer_thickness=convert_optional(self.aquifer_thickness, factors["length"]),
            aquifer_productivity_index=convert_optional(
                self.aquifer_productivity_index, rate_factor
            ),
            initial_encroachable_water=convert_optional(
                self.initial_encroachable_water, factors["reservoir_volume"]
            ),
            initial_aquifer_water_volume=convert_optional(
                self.initial_aquifer_water_volume, factors["reservoir_volume"]
            ),
            angle=self.angle,
            unit_system=target,
        )

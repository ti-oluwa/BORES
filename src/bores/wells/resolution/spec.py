import enum

import attrs
from typing_extensions import Self

from bores.constants import UnitConversionTable, c, get_conversion_factors
from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.types import Integer, Number, UnitSystem
from bores.utils import scale

__all__ = ["ConnectionPressureMode", "WellControlSpec"]


class ConnectionPressureMode(enum.Enum):
    """
    How a well's per-connection flowing pressure is derived when it has
    no `WellBoreModel`/`VFPTable` assigned.

    Only affects the connection-to-connection distribution below a
    well's reference depth. A well with no hydraulics model still can't
    resolve a THP control mode, a `THPLimit`, or THP reporting so those
    always require one, regardless of this setting.
    """

    HYDRAULIC = "hydraulic"
    """
    Require a real `WellBoreModel`/`VFPTable` for any well whose
    connections need pressure distributed across them. Raises clearly
    if one isn't assigned. The rigorous default.
    """

    UNIFORM_BHP = "uniform_bhp"
    """
    Apply the reference pressure unchanged at every connection, with
    no hydrostatic or friction correction between them. Matches how a
    simulator with no hydraulics model assigned treats a well's
    connections when nothing else is available.
    """


@attrs.frozen(kw_only=True, slots=True)
class WellControlSpec(Serializable):
    """
    Numerical tuning for well-control resolution.

    Solver tunables for well control resolution.
    """

    max_fixed_point_iterations: Integer = attrs.field(
        factory=lambda: c.CONTROL_MAX_FIXED_POINT_ITERATIONS
    )
    rate_convergence_tolerance: Number = attrs.field(
        factory=lambda: c.CONTROL_RATE_CONVERGENCE_TOLERANCE
    )
    max_bisection_iterations: Integer = attrs.field(
        factory=lambda: c.CONTROL_MAX_BISECTION_ITERATIONS
    )
    producer_bhp_floor: Number = attrs.field(factory=lambda: c.MINIMUM_VALID_PRESSURE)
    injector_bhp_bracket_multiplier: Number = attrs.field(
        factory=lambda: c.CONTROL_INJECTOR_BHP_BRACKET_MULTIPLIER
    )
    group_rate_cutback_factor: Number = attrs.field(
        factory=lambda: c.CONTROL_GROUP_RATE_CUTBACK_FACTOR
    )
    """
    Fraction a group's target rate is multiplied by, per call, when a
    `GECON` limit with `WorkoverAction.RATE` is breached.
    """
    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system for pressure-valued control limits."""

    connection_pressure_mode: ConnectionPressureMode = ConnectionPressureMode.HYDRAULIC
    """How to treat a well with no `WellBoreModel`/`VFPTable` assigned."""

    def __attrs_post_init__(self) -> None:
        if self.max_fixed_point_iterations < 1:
            raise ValidationError(
                "`max_fixed_point_iterations` must be >= 1; got "
                f"{self.max_fixed_point_iterations}."
            )

        if self.rate_convergence_tolerance <= 0:
            raise ValidationError(
                "`rate_convergence_tolerance` must be positive; got "
                f"{self.rate_convergence_tolerance}."
            )

        if self.max_bisection_iterations < 1:
            raise ValidationError(
                f"`max_bisection_iterations` must be >= 1; got {self.max_bisection_iterations}."
            )

        if self.producer_bhp_floor <= 0:
            raise ValidationError(
                f"`producer_bhp_floor` must be positive; got {self.producer_bhp_floor}."
            )

        if self.injector_bhp_bracket_multiplier <= 1.0:
            raise ValidationError(
                "`injector_bhp_bracket_multiplier` must be > 1.0; got "
                f"{self.injector_bhp_bracket_multiplier}."
            )

        if not (0.0 < self.group_rate_cutback_factor < 1.0):
            raise ValidationError(
                "`group_rate_cutback_factor` must be strictly between 0.0 "
                f"and 1.0; got {self.group_rate_cutback_factor}."
            )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `WellControlSpec` with pressure-valued limits rescaled
        to *target*.

        `producer_bhp_floor` is converted using the pressure factor for the
        source and target unit systems. Iteration limits, tolerances, and
        dimensionless multipliers are copied unchanged.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `WellControlSpec` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        return attrs.evolve(
            self,
            producer_bhp_floor=scale(self.producer_bhp_floor, factors["pressure"]),
            unit_system=target,
        )

import attrs
from typing_extensions import Self

from bores.constants import UnitConversionTable, c, get_conversion_factors
from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.types import Integer, Number, UnitSystem
from bores.utils import scale

__all__ = ["WellControlSpec"]


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
    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system for pressure-valued control limits."""

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

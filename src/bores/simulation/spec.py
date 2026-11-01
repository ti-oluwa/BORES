"""Run-level configuration, passed through `ScheduleContext.runspec`."""

import datetime

import attrs
from typing_extensions import Self

from bores.constants import Constants
from bores.errors import ValidationError
from bores.simulation.options import EquilibrationOptions, ModelDimensions, SaturationOptions
from bores.simulation.run_options import (
    NewtonOptions,
    OutputOptions,
    PhaseOptions,
    PhysicsOptions,
    SolverOptions,
    StepLimitOptions,
)
from bores.timestep.control import Timer
from bores.types import (
    DiscretizationMethod,
    Phases,
    UnitConversionTable,
    UnitSystem,
)
from bores.units import get_conversion_factors, resolve_unit_conversion_table
from bores.wells.resolution.spec import WellControlSpec

__all__ = ["RunSpec"]


@attrs.frozen(kw_only=True, slots=True, frozen=True)
class RunSpec:
    """Run-level configuration for a simulation."""

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system for the run."""

    constants: Constants = attrs.field(factory=Constants)
    """Physical and conversion constants for the run."""

    timer: Timer | None = None
    """The run's time step manager, if one has been attached yet."""

    start_date: datetime.datetime | None = None
    """The run's calendar start date, if the run is calendar-anchored."""

    phases: Phases | None = None
    """
    The fluid phases the model has and which of them exchange components, as its deck declared
    them. `None` when they were not declared, so they are to be inferred from the data supplied.
    """

    saturation_options: SaturationOptions = attrs.field(factory=SaturationOptions)
    """The optional saturation function features the deck switched on with `SATOPTS`."""

    equilibration_options: EquilibrationOptions = attrs.field(factory=EquilibrationOptions)
    """The optional equilibration features the deck switched on with `EQLOPTS`."""

    dimensions: ModelDimensions | None = None
    """The sizes the deck declared for its tables, regions, wells and VFP tables. `None` if it declared none."""

    solver_options: SolverOptions = attrs.field(factory=SolverOptions)
    """Iterative linear solver settings for the pressure and transport systems."""

    newton_options: NewtonOptions = attrs.field(factory=NewtonOptions)
    """Newton-Raphson settings for the implicit solvers."""

    step_limit_options: StepLimitOptions = attrs.field(factory=StepLimitOptions)
    """How much a time step may change the state before it is rejected or cut."""

    physics_options: PhysicsOptions = attrs.field(factory=PhysicsOptions)
    """Which physical effects the model includes and how the state is kept physically consistent."""

    phase_options: PhaseOptions = attrs.field(factory=PhaseOptions)
    """The thresholds that decide whether a phase is present in a cell."""

    output_options: OutputOptions = attrs.field(factory=OutputOptions)
    """What a run reports while it advances."""

    well_control: WellControlSpec = attrs.field(factory=WellControlSpec)
    """Numerical tuning for well-control resolution. Defaults to `WellControlSpec`'s own defaults."""

    data_check_only: bool = False
    """
    Whether the run only loads and checks its input without simulating (`NOSIM`).

    A run in this mode stops after the model, schedule and settings have been built and validated.
    """

    discretization_method: DiscretizationMethod = "impes"
    """Evolution scheme to use for the simulation ('impes' or 'implicit')."""

    def __attrs_post_init__(self) -> None:
        # The minimum injector saturations must exceed the phase appearance tolerance, or the
        # seeded phase would count as absent again straight away.
        tolerance = self.phase_options.phase_appearance_tolerance
        for name in ("minimum_injector_gas_saturation", "minimum_injector_water_saturation"):
            seeded = getattr(self.physics_options, name)
            if seeded is not None and seeded <= tolerance:
                raise ValidationError(
                    f"`physics_options.{name}` must be greater than "
                    "`phase_options.phase_appearance_tolerance` to avoid numerical issues with "
                    "phase appearance."
                )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Converts the run spec to a different unit system.

        :param target: Target unit system.
        :param table: Optional conversion table. If not provided, a default table will be used.
        :returns: A new `RunSpec` instance with properties converted to the target unit system.
        """
        if self.unit_system == target:
            return self
        table = resolve_unit_conversion_table(table)

        factors = get_conversion_factors(self.unit_system, target, table=table)
        pressure_factor = factors["pressure"]
        return attrs.evolve(
            self,
            well_control=self.well_control.convert(target, table=table),
            timer=self.timer.convert(target, table=table) if self.timer is not None else None,
            step_limit_options=self.step_limit_options.convert(pressure_factor=pressure_factor),
            physics_options=self.physics_options.convert(volume_factor=factors["volume"]),
            unit_system=target,
        )

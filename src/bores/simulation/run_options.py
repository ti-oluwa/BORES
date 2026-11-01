"""
The settings of a run that no deck describes, grouped by what they configure.

Each group is a frozen set of related settings with its own defaults, held on `RunSpec` under the
name given in its docstring. The ones a deck can set (`SATOPTS`, `EQLOPTS`, `MINPV`) are in
`bores.simulation.options` instead.
"""

import typing
from collections.abc import Iterable

import attrs

from bores.types import Boolean, Integer, MiscibilityModel, Number, PreconditionerStr, SolverStr
from bores.utils import scale

__all__ = [
    "NewtonOptions",
    "OutputOptions",
    "PhaseOptions",
    "PhysicsOptions",
    "SolverOptions",
    "StepLimitOptions",
]


@attrs.frozen(kw_only=True, slots=True)
class SolverOptions:
    """
    Iterative linear solver settings for the pressure and transport systems.

    Held on `RunSpec.solver_options`.
    """

    pressure_solver: SolverStr | Iterable[SolverStr] = "bicgstab"
    """Pressure system solver(s) (can be a list of solver to use in sequence)."""
    transport_solver: SolverStr | Iterable[SolverStr] = "bicgstab"
    """Transport system solver(s) (can be a list of solver to use in sequence)."""
    pressure_preconditioner: PreconditionerStr | None = "ilu"
    """Preconditioner to use for pressure system solvers."""
    transport_preconditioner: PreconditionerStr | None = "ilu"
    """Preconditioner to use for transport system solvers."""
    pressure_convergence_tolerance: Number = attrs.field(
        default=1e-6, validator=attrs.validators.le(1e-2)
    )
    """Relative convergence tolerance for pressure iterative solvers (default is 1e-6)."""
    transport_convergence_tolerance: Number = attrs.field(
        default=1e-4, validator=attrs.validators.le(1e-2)
    )
    """Relative convergence tolerance for saturation iterative solvers (default is 1e-4). Transport matrix tend to be more well conditioned."""
    maximum_solver_iterations: Integer = attrs.field(
        default=250,
        validator=attrs.validators.and_(
            attrs.validators.ge(1),  # type: ignore[arg-type]
            attrs.validators.le(500),  # type: ignore[arg-type]
        ),
    )
    """
    Maximum number of iterations allowed per time step for all iterative solvers.
    
    Capped at 500 to prevent excessive computation time in case of non-convergence.
    If the solver does not converge within this limit, the matrix is most likely
    ill-conditioned or the problem setup needs to be reviewed. Use a stronger
    preconditioner, try another solver, or adjust simulation parameters accordingly.
    """
    use_nonlinear_pressure_solve: Boolean = False


@attrs.frozen(kw_only=True, slots=True)
class NewtonOptions:
    """
    Newton-Raphson settings for the implicit solvers.

    Held on `RunSpec.newton_options`.
    """

    maximum_newton_iterations: Integer = attrs.field(
        default=10,
        validator=attrs.validators.and_(
            attrs.validators.ge(1),  # type: ignore[arg-type]
            attrs.validators.le(50),  # type: ignore[arg-type]
        ),
    )
    """Maximum Newton-Raphson non-linear iterations for implicit solvers."""
    newton_tolerance: Number = attrs.field(default=1e-5, validator=attrs.validators.le(1e-2))
    """Relative residual tolerance for Newton convergence in implicit solvers."""
    maximum_line_search_cuts: Integer = attrs.field(
        default=4,
        validator=attrs.validators.and_(
            attrs.validators.ge(0),  # type: ignore[arg-type]
            attrs.validators.le(10),  # type: ignore[arg-type]
        ),
    )
    """Maximum line search cuts/bisections per Newton step."""
    saturation_jacobian_assembly_method: typing.Literal["numerical", "analytical"] = "analytical"
    """
    Method used to assemble the Jacobian matrix in the implicit saturation
    Newton loop.

    - `"numerical"`: column-wise forward finite differences exploiting the
      7-point stencil sparsity (the default, always available).
    - `"analytical"`: exact derivatives from the relperm and capillary
      pressure table derivative methods, assembled by a Numba-parallel kernel.
      Faster and more accurate than the numerical Jacobian, but requires that
      the relperm and capillary pressure tables implement the `derivatives(...)` API.
    """
    maximum_newton_saturation_change: Number = attrs.field(
        default=0.2,
        validator=attrs.validators.and_(
            attrs.validators.gt(0.0),
            attrs.validators.le(1.0),
        ),
    )
    """
    Maximum per-cell saturation change per Newton iteration. Damps the Newton
    update to prevent upwind direction flipping that causes limit-cycle oscillation.
    """
    newton_stagnation_patience: Integer = attrs.field(
        default=4,
        validator=attrs.validators.ge(1),
    )
    """
    Number of consecutive Newton iterations allowed with insufficient
    residual improvement before the solver declares stagnation.

    During the Newton-Raphson solve, the relative residual is monitored
    between iterations. If the reduction in residual falls below
    `newton_stagnation_improvement_threshold` for this many consecutive
    iterations, the solver assumes it is no longer making meaningful
    progress.

    This typically indicates:
    - Poor Jacobian quality (e.g., inaccurate analytical derivatives)
    - Strong nonlinearities (e.g., phase appearance, sharp fronts)
    - Ill-conditioned system

    When stagnation is detected, the solver may terminate early or trigger
    fallback strategies (e.g., timestep reduction or switching assembly method).

    Lower values means faster detection, more aggressive termination  
    Higher values means more tolerance for slow convergence
    """
    newton_stagnation_improvement_threshold: Number = attrs.field(
        default=0.005,
        validator=attrs.validators.gt(0),
    )
    """
    Minimum relative improvement in residual required between successive
    Newton iterations to be considered as making progress.

    Defined as the fractional reduction in the nonlinear residual norm:

        improvement = (R_k - R_{k+1}) / R_k

    If the computed improvement falls below this threshold, the iteration
    is classified as "non-improving". When this occurs for
    `newton_stagnation_patience` consecutive iterations, stagnation is declared.

    Typical interpretation:
    - 0.005 means that we require at least 0.5% residual reduction per iteration
    - Smaller values means more tolerant of slow convergence
    - Larger values means stricter, may trigger stagnation earlier

    This parameter is critical for preventing wasted iterations in cases
    where Newton updates oscillate or make negligible progress.
    """


@attrs.frozen(kw_only=True, slots=True)
class StepLimitOptions:
    """
    How much a time step may change the state before it is rejected or cut.

    Held on `RunSpec.step_limit_options`.
    """

    cfl_threshold: Number = 0.7
    """
    Maximum allowable transport CFL number for the 'explicit' evolution scheme to ensure numerical stability.

    Typically kept below 1.0 to prevent instability in explicit transport updates.

    Lowering this value increases stability but may require smaller time steps.
    Raising them can improve performance but risks instability. Use with caution and monitor simulation behavior.
    """
    maximum_pressure_change: Number = attrs.field(default=1000.0, validator=attrs.validators.ge(0))
    """
    Maximum allowable pressure change (in psi) per time step.

    Controls time step size by limiting pressure variations to maintain numerical stability
    and physical accuracy. When exceeded, the time step is reduced or rejected.

    Default: 500 psi (~35 bar). This is suitable for most field-scale simulations with typical
    reservoir pressures of 1000-5000 psi.

    Adjust based on simulation characteristics:

    **Tighten to 50-75 psi when:**
    - Simulating high-rate wells with large near-wellbore pressure gradients
    - Reservoir pressure is low (<1000 psi) to maintain relative accuracy
    - Using highly compressible fluids (gas reservoirs)
    - Fine-grid simulations (<10m cells) where local variations are significant
    - Observing pressure oscillations or convergence issues

    **Relax to 150-300 psi when:**
    - Depletion studies with slow, uniform pressure decline
    - Field-scale models (>100m cells) where averaging reduces local variations
    - Reservoir pressure is high (>5000 psi) making relative changes small
    - Simulation is stable and material balance errors are acceptable
    - Computational efficiency is critical and accuracy requirements are relaxed

    **Guidelines by reservoir pressure:**
    - Low pressure (<1000 psi): 25-50 psi (2.5-5% relative change)
    - Moderate pressure (1000-3000 psi): 50-100 psi (2-5% relative change)
    - High pressure (3000-6000 psi): 100-200 psi (2-4% relative change)
    - Very high pressure (>6000 psi): 150-300 psi (2-5% relative change)

    Note: Larger changes can cause density/viscosity jumps and well control issues.
    """
    maximum_oil_saturation_change: Number = attrs.field(
        default=0.6, validator=attrs.validators.ge(0)
    )
    """
    Maximum allowable oil saturation change (absolute, fractional) per time step.

    Controls time step size by limiting saturation variations to prevent numerical
    instabilities and maintain material balance accuracy. When exceeded, the time
    step is reduced or rejected.
    """
    maximum_water_saturation_change: Number = attrs.field(
        default=0.6, validator=attrs.validators.ge(0)
    )
    """
    Maximum allowable water saturation change (absolute, fractional) per time step.

    Controls time step size by limiting saturation variations to prevent numerical
    instabilities and maintain material balance accuracy. When exceeded, the time
    step is reduced or rejected.
    """
    maximum_gas_saturation_change: Number = attrs.field(
        default=0.5, validator=attrs.validators.ge(0)
    )
    """
    Maximum allowable gas saturation change (absolute, fractional) per time step.

    Controls time step size by limiting saturation variations to prevent numerical
    instabilities and maintain material balance accuracy. When exceeded, the time
    step is reduced or rejected.
    """
    saturation_change_violation_tolerance: Number = attrs.field(
        default=5e-2, validator=attrs.validators.ge(0)
    )

    def convert(self, *, pressure_factor: Number) -> "StepLimitOptions":
        """
        The limits in another unit system.

        :param pressure_factor: What to multiply a pressure by to express it in the new unit system.
        :returns: The limits, with the pressure change converted. The rest are dimensionless.
        """
        return attrs.evolve(
            self, maximum_pressure_change=scale(self.maximum_pressure_change, pressure_factor)
        )


@attrs.frozen(kw_only=True, slots=True)
class PhysicsOptions:
    """
    Which physical effects the model includes and how the state is kept physically consistent.

    Held on `RunSpec.physics_options`.
    """

    capillary_effects_enabled: Boolean = True
    """Whether to include capillary pressure effects in the simulation."""
    capillary_strength_factor: Number = attrs.field(
        default=1.0,
        validator=attrs.validators.ge(0),  # type: ignore[arg-type]
    )
    """
    Factor to scale capillary flow for numerical stability. Reduce to dampen capillary effects.
    Increase to enhance capillary effects.

    Capillary gradients can become numerically dominant in fine meshes or sharp saturation fronts.
    Damping avoids overshoot/undershoot by reducing their contribution without removing them.

    Set to 0 to disable capillary effects entirely (not recommended).
    """
    gravity_enabled: Boolean = True
    """Whether to disable structural dip effects in reservoir simulation."""
    miscibility: MiscibilityModel = "immiscible"
    """Miscibility model: 'immiscible', 'todd-longstaff'"""
    freeze_saturation_pressure: Boolean = False
    """
    If True, keeps oil bubble point pressure (Pb) constant at its initial value throughout the simulation.

    You would sometimes want this set to True, if you are not modelling complex conditions like 
    miscible injection, Waterflooding, etc., to adhere to standard black-oil model assumptions.

    This is appropriate for:
    - Natural depletion with no compositional changes
    - Waterflooding where oil composition remains constant
    - Simplified black-oil models without compositional tracking

    If False (default), Pb is recomputed each timestep based on current solution GOR.

    **Properties affected when Pb is frozen:**
    - Bubble point pressure (Pb) - directly frozen
    - Solution GOR (Rs) - computed using frozen Pb as reference
    - Oil FVF (Bo) - uses frozen Pb for undersaturated calculations
    - Oil compressibility (Co) - switches at frozen Pb
    - Oil viscosity (μo) - indirectly through Rs
    - Oil density (ρo) - indirectly through Rs and Bo
    """
    normalize_saturations: Boolean = False
    """
    Whether to normalize saturations so that `So + Sw + Sg = 1.0` after each timestep.

    When True (default), the simulator rescales all three phase saturations at the end
    of each saturation update so their sum is exactly 1.0. This corrects small
    numerical drift from the explicit transport solve and maintains strict pore-volume
    conservation. The normalization uses safe division to avoid issues in cells with
    near-zero total saturation.

    Set to False only when debugging saturation solver mass balance, since the raw
    (unnormalized) values reveal how much drift the transport step actually produces.
    For production simulations, always leave this enabled.
    """
    minimum_injector_gas_saturation: Number | None = attrs.field(
        default=None, validator=attrs.validators.optional(attrs.validators.ge(0))
    )
    """
    Minimum gas saturation enforced in active gas injector wellblocks after
    each pressure solve and rate computation.

    When gas is injected into a cell with zero initial gas saturation, the
    in-situ relative permeability to gas (krg) is zero, which prevents the
    injected gas from being transported to neighbouring cells in the saturation
    solve. This parameter seeds a small but non-zero gas saturation in those
    cells so that krg > 0 and transport can proceed.

    The value should be above `phase_appearance_tolerance` and `residual_gas_saturation` to guarantee that
    the relative permeability model returns a non-zero krg. Setting too large a value
    introduces an artificial initial gas saturation that may affect early-time
    results and MBE. Set to `None` to disable gas saturation seeding entirely,
    which is appropriate when the initial gas saturation is already non-zero in
    injector cells.
    """
    minimum_injector_water_saturation: Number | None = attrs.field(
        default=None, validator=attrs.validators.optional(attrs.validators.ge(0))
    )
    """
    Minimum water saturation enforced in active water injector wellblocks after
    each pressure solve and rate computation.

    Analogous to `minimum_injector_gas_saturation` but for water injectors.
    In practice, connate water saturation is almost always non-zero so this
    seeding is rarely needed for water injection. However, in synthetic models
    or edge cases where initial water saturation is exactly zero in injector
    cells, this parameter prevents krw from being zero in the transport step.

    The value should be above `phase_appearance_tolerance` and `irreducible_water_saturation` to guarantee that
    the relative permeability model returns a non-zero krw. Set to `None`
    to disable water saturation seeding, which is the recommended setting for
    most realistic models where connate water is already present.
    """
    minimum_pore_volume: Number | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.ge(0)),  # type: ignore[arg-type]
    )
    """
    Pore volume below which a cell takes no part in the flow (`MINPV`), in the pore volume unit of
    `unit_system` (m³, ft³ or cm³). `None` when the deck sets none.

    Pore volumes here are the ones at the rock's reference pressure, so the cells this excludes
    are fixed for the whole run and do not change as pressure moves the pore volume.
    """

    def convert(self, *, volume_factor: Number) -> "PhysicsOptions":
        """
        The settings in another unit system.

        :param volume_factor: What to multiply a volume by to express it in the new unit system.
        :returns: The settings, with the minimum pore volume converted. The rest are dimensionless.
        """
        if self.minimum_pore_volume is None:
            return self
        return attrs.evolve(
            self, minimum_pore_volume=scale(self.minimum_pore_volume, volume_factor)
        )


@attrs.frozen(kw_only=True, slots=True)
class PhaseOptions:
    """
    The thresholds that decide whether a phase is present in a cell.

    Held on `RunSpec.phase_options`.
    """

    phase_appearance_tolerance: Number = attrs.field(default=1e-6, validator=attrs.validators.ge(0))
    """
    Tolerance for determining phase appearance/disappearance based on saturation levels.

    Used to avoid numerical issues when a phase's saturation approaches zero. This helps
    maintain stability in relative permeability and mobility calculations by treating phases
    with saturations below this threshold as absent from the system.
    """
    oil_presence_saturation: Number | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.ge(0)),  # type: ignore[arg-type]
    )
    """
    Oil saturation above which a cell is treated as holding oil when the initial phase state is
    checked for consistency. `None` uses the `OIL_PRESENCE_SATURATION` constant of `constants`.
    """
    gas_presence_saturation: Number | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.ge(0)),  # type: ignore[arg-type]
    )
    """
    Gas saturation above which a cell is treated as holding gas when the initial phase state is
    checked for consistency. `None` uses the `GAS_PRESENCE_SATURATION` constant of `constants`.
    """
    pressure_consistency_tolerance: Number | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.ge(0)),  # type: ignore[arg-type]
    )
    """
    Fraction of the bubble point or dew point by which a cell's pressure may stray from it before
    the initial phase state is reported as inconsistent. `None` uses the
    `PRESSURE_CONSISTENCY_TOLERANCE` constant of `constants`. The saturation at or below which a
    phase counts as absent in that check is `phase_appearance_tolerance`.
    """


@attrs.frozen(kw_only=True, slots=True)
class OutputOptions:
    """
    What a run reports while it advances.

    Held on `RunSpec.output_options`.
    """

    output_frequency: Integer = attrs.field(default=1, validator=attrs.validators.ge(1))
    """Frequency, in time steps, at which model states are output."""
    log_interval: Integer = attrs.field(default=5, validator=attrs.validators.ge(0))
    """Interval (in time steps) at which to log simulation progress."""
    capture_timer_state: Boolean = True
    """
    Whether to capture and include the timer state in the yielded model states during simulation monitoring.

    When True (default), the `Timer` state (current time, step, etc.) is included in the `ModelState` objects yielded by the simulator. 
    This allows for more informative logging and analysis of simulation progress.
    When False, the timer state is not included in the yielded states, which may reduce memory usage if the timer state is 
    large or if many states are captured.
    """
    warn_well_anomalies: Boolean = True
    """Whether to warn about anomalous flow rates during the simulation."""

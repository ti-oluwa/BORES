import threading
import typing
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import attrs
from typing_extensions import Self

from bores.boundary_conditions import BoundaryConditions
from bores.constants import Constants
from bores.stores import StoreSerializable
from bores.tables.pvt import PVTTables
from bores.tables.rock_fluid import RockFluidTables
from bores.timing import Timer
from bores.types import (
    EvolutionScheme,
    MiscibilityModel,
    PreconditionerStr,
    Range,
    RelativeMobilityRange,
    SolverStr,
    ThreeDimensions,
)
from bores.wells import Wells, WellSchedules

__all__ = ["Config", "new_task_pool"]


@typing.final
@attrs.frozen(kw_only=True, auto_attribs=True)
class Config(
    StoreSerializable,
    load_exclude={"pvt_tables", "_lock", "task_pool"},
    dump_exclude={"pvt_tables", "_lock", "task_pool"},
):
    """Simulation run configuration and parameters."""

    timer: Timer
    """Simulation time manager to control time steps and simulation time."""

    rock_fluid_tables: RockFluidTables
    """Rock and fluid property tables for the simulation."""

    wells: typing.Optional[Wells[ThreeDimensions]] = None
    """Well configuration for the simulation."""

    well_schedules: typing.Optional[WellSchedules[ThreeDimensions]] = None
    """Well schedules for dynamic well control during the simulation."""

    boundary_conditions: typing.Optional[BoundaryConditions[ThreeDimensions]] = None
    """Boundary conditions for the simulation."""

    pvt_tables: typing.Optional[PVTTables] = None
    """PVT tables for fluid property lookups during the simulation."""

    pressure_convergence_tolerance: float = attrs.field(
        default=1e-6, validator=attrs.validators.le(1e-2)
    )
    """Relative convergence tolerance for pressure iterative solvers (default is 1e-6)."""

    saturation_convergence_tolerance: float = attrs.field(
        default=1e-4, validator=attrs.validators.le(1e-2)
    )
    """Relative convergence tolerance for saturation iterative solvers (default is 1e-4). Transport matrix tend to be more well conditioned."""

    max_iterations: int = attrs.field(  # type: ignore
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

    output_frequency: int = attrs.field(default=1, validator=attrs.validators.ge(1))
    """Frequency at which model states are yielded/outputted during the simulation."""

    scheme: EvolutionScheme = "impes"
    """Evolution scheme to use for the simulation ('impes', 'explicit', 'implicit', 'sequential_implicit')."""

    use_pseudo_pressure: bool = True
    """Whether to use pseudo-pressure for gas (when applicable)."""

    relative_mobility_range: RelativeMobilityRange = attrs.field(
        default=RelativeMobilityRange(
            oil=Range(min=1e-12, max=1e6),
            water=Range(min=1e-12, max=1e6),
            gas=Range(min=1e-12, max=1e6),
        )
    )
    """
    Relative mobility ranges for oil, water, and gas phases.

    Each phase has a `Range` object defining its minimum and maximum relative mobility.
    Adjust minimum or maximum values to constrain phase mobilities during simulation.

    Minimum values should not be exactly zero for the best numerical stability.
    """

    total_compressibility_range: Range = attrs.field(default=Range(min=1e-24, max=1e-2))
    """Range to constrain total compressibility for the simulation. This is usually necessary for numerical stability."""

    capillary_strength_factor: float = attrs.field(  # type: ignore
        default=1.0,
        validator=attrs.validators.and_(attrs.validators.ge(0), attrs.validators.le(1)),  # type: ignore[arg-type]
    )
    """
    Factor to scale capillary flow for numerical stability. Reduce to dampen capillary effects.
    Increase to enhance capillary effects.

    Capillary gradients can become numerically dominant in fine meshes or sharp saturation fronts.
    Damping avoids overshoot/undershoot by reducing their contribution without removing them.

    Set to 0 to disable capillary effects entirely (not recommended).
    """

    disable_capillary_effects: bool = False
    """Whether to include capillary pressure effects in the simulation."""

    disable_structural_dip: bool = False
    """Whether to disable structural dip effects in reservoir modeling/simulation."""

    miscibility_model: MiscibilityModel = "immiscible"
    """Miscibility model: 'immiscible', 'todd_longstaff'"""

    saturation_cfl_threshold: float = 0.7
    """
    Maximum allowable saturation CFL number for the 'explicit' evolution scheme to ensure numerical stability.

    Typically kept below 1.0 to prevent instability in explicit saturation updates.

    Lowering this value increases stability but may require smaller time steps.
    Raising them can improve performance but risks instability. Use with caution and monitor simulation behavior.
    """

    pressure_cfl_threshold: float = 0.9
    """
    Maximum allowable pressure CFL number for the 'explicit' evolution scheme to ensure numerical stability.

    Typically kept below 1.0 to prevent instability in explicit pressure updates.
    
    Lowering this value increases stability but may require smaller time steps.
    Raising them can improve performance but risks instability. Use with caution and monitor simulation behavior.
    """

    constants: Constants = attrs.field(factory=Constants)
    """Physical and conversion constants used in the simulation."""

    warn_well_anomalies: bool = True
    """Whether to warn about anomalous flow rates during the simulation."""

    log_interval: int = attrs.field(default=5, validator=attrs.validators.ge(0))  # type: ignore
    """Interval (in time steps) at which to log simulation progress."""

    pressure_solver: typing.Union[SolverStr, typing.Iterable[SolverStr]] = "bicgstab"
    """Pressure matrix system solver(s) (can be a list of solver to use in sequence) to use for solving linear systems."""

    saturation_solver: typing.Union[SolverStr, typing.Iterable[SolverStr]] = "bicgstab"
    """Saturation matrix system solver(s) (can be a list of solver to use in sequence) to use for solving linear systems."""

    pressure_preconditioner: typing.Optional[PreconditionerStr] = "ilu"
    """Preconditioner to use for pressure solvers."""

    saturation_preconditioner: typing.Optional[PreconditionerStr] = "ilu"
    """Preconditioner to use for saturation solvers."""

    phase_appearance_tolerance: float = attrs.field(  # type: ignore
        default=1e-6,
        validator=attrs.validators.ge(0),
    )
    """
    Tolerance for determining phase appearance/disappearance based on saturation levels.

    Used to avoid numerical issues when a phase's saturation approaches zero. This helps
    maintain stability in relative permeability and mobility calculations by treating phases
    with saturations below this threshold as absent from the system.
    """

    residual_oil_drainage_ratio_water_flood: float = 0.6
    """Ratio to compute oil drainage residual from imbibition value during water flooding."""

    residual_oil_drainage_ratio_gas_flood: float = 0.6
    """Ratio to compute oil drainage residual from imbibition value during gas flooding."""

    residual_gas_drainage_ratio: float = 0.5
    """Ratio to compute gas drainage residual from imbibition value."""

    max_oil_saturation_change: float = 0.6
    """
    Maximum allowable oil saturation change (absolute, fractional) per time step.

    Controls time step size by limiting saturation variations to prevent numerical
    instabilities and maintain material balance accuracy. When exceeded, the time
    step is reduced or rejected.
    """

    max_water_saturation_change: float = attrs.field(  # type: ignore
        default=0.6, validator=attrs.validators.ge(0)
    )
    """
    Maximum allowable water saturation change (absolute, fractional) per time step.

    Controls time step size by limiting saturation variations to prevent numerical
    instabilities and maintain material balance accuracy. When exceeded, the time
    step is reduced or rejected.
    """

    max_gas_saturation_change: float = attrs.field(  # type: ignore
        default=0.5, validator=attrs.validators.ge(0)
    )
    """
    Maximum allowable gas saturation change (absolute, fractional) per time step.

    Controls time step size by limiting saturation variations to prevent numerical
    instabilities and maintain material balance accuracy. When exceeded, the time
    step is reduced or rejected.
    """

    max_pressure_change: float = attrs.field(  # type: ignore
        default=500.0, validator=attrs.validators.ge(0)
    )
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

    max_newton_iterations: int = attrs.field(  # type: ignore
        default=15,
        validator=attrs.validators.and_(
            attrs.validators.ge(1),  # type: ignore[arg-type]
            attrs.validators.le(50),  # type: ignore[arg-type]
        ),
    )
    """Maximum Newton-Raphson iterations for implicit solvers."""

    newton_tolerance: float = attrs.field(
        default=1e-6, validator=attrs.validators.le(1e-2)
    )
    """Relative residual tolerance for Newton convergence in implicit solvers."""

    line_search_max_cuts: int = attrs.field(  # type: ignore
        default=4,
        validator=attrs.validators.and_(
            attrs.validators.ge(0),  # type: ignore[arg-type]
            attrs.validators.le(10),  # type: ignore[arg-type]
        ),
    )
    """Maximum line search bisections per Newton step."""

    max_saturation_step: float = attrs.field(
        default=0.05,
        validator=attrs.validators.and_(
            attrs.validators.gt(0.0),
            attrs.validators.le(1.0),
        ),
    )
    """
    Maximum per-cell saturation change per Newton iteration. Damps the Newton
    update to prevent upwind direction flipping that causes limit-cycle oscillation.
    """

    newton_saturation_change_tolerance: float = attrs.field(
        default=1e-4,
        validator=attrs.validators.le(1e-2),
    )
    """
    Saturation change tolerance for dual convergence check. If the maximum
    saturation change per iteration falls below this and the relative residual
    is below 1e-3, Newton is declared converged.
    """

    newton_stagnation_patience: int = attrs.field(
        default=3,
        validator=attrs.validators.ge(1),
    )
    
    newton_stagnation_improvement_threshold: float = attrs.field(
        default=0.01,
        validator=attrs.validators.gt(0),
    )

    jacobian_assembly_method: typing.Literal["numerical", "analytical"] = "numerical"
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

    normalize_saturations: bool = True
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

    freeze_saturation_pressure: bool = False
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

    task_pool: typing.Optional[ThreadPoolExecutor] = attrs.field(
        default=None,
        eq=False,
        hash=False,
    )
    """
    Optional thread pool for concurrent matrix assembly during simulation.

    When provided, the three independent assembly stages in the pressure solver
    (accumulation, face transmissibilities, well contributions) and the two
    independent stages in the saturation solver (flux contributions, well rate
    grids) are submitted concurrently rather than run sequentially. The calling
    thread blocks only until all submitted stages complete, so the effective
    assembly time approaches the duration of the slowest stage rather than
    their sum.

    When None (default), all assembly stages run sequentially on the calling
    thread with zero threading overhead. This is the correct choice for small
    grids where threading bookkeeping exceeds the parallelism gain.

    **When to provide a pool**

    The break-even point is approximately 10,000 interior cells. Below this
    threshold the overhead of thread synchronisation, future creation, and
    queue operations exceeds the time saved by concurrent execution. At 50,000
    cells the concurrent path reduces assembly time by roughly 30-50%, which
    translates to approximately 7-10% reduction in total per-step wall time
    (the linear solve is unaffected and typically dominates at this scale).
    At 200,000+ cells the benefit is clearly measurable.

    A rough guide by grid size:

    - < 10,000 cells  → leave as `None`
    - 10,000-50,000   → marginal benefit, profile before committing
    - 50,000-200,000  → noticeable benefit, 3 workers recommended
    - > 200,000       → clearly beneficial, assembly cost approaches solve cost

    **Lifecycle**

    The pool is not created or shut down by `Config`. The caller is responsible
    for managing its lifetime. The recommended pattern is to create the pool
    once for the entire simulation run using `new_task_pool()` and pass it in
    at `Config` construction time:

    ```python
    with new_task_pool(concurrency=3) as pool:
        config = Config(..., task_pool=pool)
        for state in run(model, config):
            process(state)
    # Pool shuts down cleanly here
    ```

    Do not share a pool between concurrent simulation runs unless the pool has
    sufficient workers to service both simultaneously.
    """

    _lock: threading.Lock = attrs.field(
        factory=threading.Lock, init=False, repr=False, hash=False
    )
    """Internal lock for thread-safe operations."""

    def copy(self, **kwargs: typing.Any) -> Self:
        """Create a deep copy of the `Config` instance."""
        with self._lock:
            return attrs.evolve(self, **kwargs)

    def with_updates(self, **kwargs: typing.Any) -> Self:
        """
        Return a new `Config` with updated parameters (immutable pattern).

        :param kwargs: Keyword arguments for fields to update
        :return: New `Config` instance with updated values
        :raises AttributeError: If any key is not a valid `Config` attribute
        """
        for key in kwargs:
            if not hasattr(self, key):
                raise AttributeError(f"Config has no attribute '{key}'")
        with self._lock:
            return attrs.evolve(self, **kwargs)


@contextmanager
def new_task_pool(
    concurrency: typing.Optional[int] = None,
) -> typing.Generator[ThreadPoolExecutor, None, None]:
    """
    Context manager that creates a `ThreadPoolExecutor` for concurrent
    simulation assembly and shuts it down cleanly on exit.

    Intended as the standard way to supply a `task_pool` to `Config`.
    The pool is created once, used for the entire simulation run, and
    gracefully shut down (waiting for any in-flight work to complete)
    when the `with` block exits - whether normally or due to an exception.

    :param concurrency: Maximum number of tasks that may run concurrently.
        If `None`, Python defaults to `min(32, os.cpu_count() + 4)`,
        which is almost always too large for simulation assembly. Pass an
        explicit value instead:

        - `3` for IMPES (pressure assembly has 3 independent stages,
          saturation assembly has 2 - 3 workers covers both without waste).
        - `2` if the machine has only 2 physical cores available to the
          process, or if memory bandwidth is the bottleneck rather than
          compute.
        - Higher values provide no additional benefit for the current
          assembly design, which submits at most 3 tasks per solver call.

    :yields: The configured `ThreadPoolExecutor`.

    Example: Standard IMPES run with concurrent assembly:

    ```python
    from bores.config import Config, new_task_pool

    with new_task_pool(concurrency=3) as pool:
        config = Config(
            timer=timer,
            rock_fluid_tables=tables,
            task_pool=pool,
        )
        for state in run(model, config):
            process(state)
    # Pool shuts down here; all in-flight writes complete before exit
    ```

    Example: Conditional pool based on grid size:

    ```python
    cell_count = nx * ny * nz

    if cell_count > 10_000:
        with new_task_pool(concurrency=3) as pool:
            config = Config(..., task_pool=pool)
            for state in run(model, config):
                process(state)
    else:
        config = Config(...)   # no pool, sequential assembly
        for state in run(model, config):
            process(state)
    ```

    Note: Do not pass the pool to more than one `Config` instance that
    will be used concurrently. Each simulation run submits up to 3 tasks
    per solver call; two concurrent runs would require 6 workers to avoid
    queuing, and the assembly functions are not designed for that usage.
    """
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        yield pool

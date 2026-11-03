"""Simulation run options/configurations, grouped by what they configure."""

import typing
import warnings
from collections.abc import Iterable

import attrs

from bores.constants import c
from bores.deck.file import DeckFile
from bores.types import (
    Boolean,
    Integer,
    MiscibilityModel,
    Number,
    PreconditionerStr,
    SolverStr,
    UnitSystem,
)
from bores.utils import scale

__all__ = [
    "EquilibrationOptions",
    "MessageLimits",
    "ModelDimensions",
    "NewtonOptions",
    "OutputOptions",
    "PhaseOptions",
    "PhysicsOptions",
    "ReportRequest",
    "SaturationOptions",
    "SolverOptions",
    "StepLimitOptions",
    "get_scheduled_at_start",
    "load_dissolution_limits",
    "load_equilibration_options",
    "load_minimum_pore_volume",
    "load_model_dimensions",
    "load_output_options",
    "load_saturation_options",
]


@attrs.frozen(kw_only=True, slots=True)
class SaturationOptions:
    """
    The saturation function features of a run, which a deck switches on with `SATOPTS`.

    Held on `RunSpec.saturation_options`.
    """

    hysteresis_enabled: Boolean = False
    """
    Enable saturation hysteresis tracking for residual saturations.

    When True, residual oil and gas saturations are updated each time step
    based on the saturation reversal history of each cell (drainage vs.
    imbibition regime). This is required when using hysteresis-aware
    relative permeability or capillary pressure models (e.g. Killough,
    Land trapping) where the scanning curves depend on the maximum
    historical saturation reached in each cell.

    When False (default), residual saturations are treated as fixed rock
    properties and the hysteresis state is not updated during the
    simulation. This is appropriate for standard black-oil simulations
    without hysteresis models, and avoids the overhead of per-step
    history tracking.

    This flag should only be enabled when the configured relative
    permeability and/or capillary pressure tables implement hysteresis-
    aware scanning curves. Enabling it with non-hysteretic tables has no
    physical effect and incurs unnecessary computation.
    """

    hysteresis_saturation_tolerance: Number | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.ge(0)),  # type: ignore[arg-type]
    )
    """
    Smallest saturation change between time steps that counts as a phase moving up or down
    when tracking relative permeability hysteresis. `None` uses the
    `HYSTERESIS_SATURATION_TOLERANCE` constant of `constants`.
    """

    endpoint_scaling_enabled: Boolean | None = None
    """
    Whether the saturation function end points are scaled cell by cell, so the deck's end point
    arrays (`SWL`, `SGCR`, ...) are used (`ENDSCALE`). `None` leaves it to the deck, which turns
    it on by having `ENDSCALE`; `False` or `True` overrides the deck.
    """

    directional_endpoint_scaling: Boolean = False
    """Whether end point scaling depends on the flow direction (`ENDSCALE` with `DIRECT`)."""

    irreversible_endpoint_scaling: Boolean = False
    """Whether end point scaling is irreversible, so one set applies to flow in each direction (`ENDSCALE` with `IRREVERS`)."""

    scale_critical_saturations: Boolean = False
    """Whether the critical saturations are scaled consistently with the other end points (`SCALECRS YES`)."""

    directional_relative_permeability: Boolean = False
    """Whether relative permeabilities depend on the flow direction (`DIRECT`)."""

    irreversible_directional_relative_permeability: Boolean = False
    """Whether the directional relative permeabilities are irreversible (`IRREVERS`)."""

    surface_tension_dependence: Boolean = False
    """Whether relative permeability and capillary pressure depend on surface tension (`SURFTENS`)."""

    unrecognized: frozenset[str] = frozenset()
    """Any other option the deck listed, upper-cased, since the options Eclipse accepts depend on its version."""


@attrs.frozen(kw_only=True, slots=True)
class EquilibrationOptions:
    """The optional equilibration features a deck switches on with `EQLOPTS`."""

    threshold_pressures: Boolean = False
    """Whether the deck defines pressure differences that flow between regions must exceed (`THPRES`)."""

    irreversible_threshold_pressures: Boolean = False
    """Whether a threshold pressure applies in one direction only (`IRREVERS`)."""

    quiescent_start: Boolean = False
    """Whether the initial state is adjusted so that nothing flows at the start (`QUIESC`)."""

    mobile_fluid_endpoint_correction: Boolean = False
    """Whether saturation end points are corrected so the initial fluids are mobile (`MOBILE`)."""

    unrecognized: frozenset[str] = frozenset()
    """Any other option the deck listed, upper-cased."""


@attrs.frozen(kw_only=True, slots=True)
class ModelDimensions:
    """
    The sizes a deck declares for its own tables, regions, wells and VFP tables.

    Every field is `None` when the deck does not write the keyword it comes from. They are
    capacities Eclipse reserves rather than anything a calculation needs, which is why they are
    checked against the deck (`bores.deck.dimensions`) and otherwise just carried.
    """

    saturation_tables: Integer | None = None
    """Number of saturation function regions (`TABDIMS`)."""

    pvt_tables: Integer | None = None
    """Number of PVT regions (`TABDIMS`)."""

    saturation_table_nodes: Integer | None = None
    """Most rows in a saturation function table (`TABDIMS`)."""

    pvt_pressure_nodes: Integer | None = None
    """Most pressure rows in a PVT table (`TABDIMS`)."""

    dissolved_gas_ratio_nodes: Integer | None = None
    """Most solution gas-oil ratio rows in a live oil table (`TABDIMS`)."""

    vaporized_oil_ratio_nodes: Integer | None = None
    """Most vaporized oil-gas ratio rows in a wet gas table (`TABDIMS`)."""

    equilibration_regions: Integer | None = None
    """Number of equilibration regions (`EQLDIMS`)."""

    pressure_depth_nodes: Integer | None = None
    """Most depth nodes in an equilibrium pressure table (`EQLDIMS`)."""

    ratio_depth_nodes: Integer | None = None
    """Most depth rows in a table of solution gas-oil or vaporized oil-gas ratio against depth (`EQLDIMS`)."""

    fluid_in_place_regions: Integer | None = None
    """Number of fluid-in-place regions in a region set (`REGDIMS`)."""

    fluid_in_place_region_sets: Integer | None = None
    """Number of fluid-in-place region sets, that is `FIPNUM` and the other `FIP<name>` arrays (`REGDIMS`)."""

    reservoir_regions: Integer | None = None
    """Number of independent reservoir regions (`REGDIMS`)."""

    flux_regions: Integer | None = None
    """Number of flux regions, `FLUXNUM` (`REGDIMS`)."""

    wells: Integer | None = None
    """Most wells in the model (`WELLDIMS`)."""

    connections_per_well: Integer | None = None
    """Most connections in one well (`WELLDIMS`)."""

    groups: Integer | None = None
    """Most well groups in the model (`WELLDIMS`)."""

    wells_per_group: Integer | None = None
    """Most wells or groups directly under one group (`WELLDIMS`)."""

    endpoint_depth_tables: Integer | None = None
    """Number of tables of end points against depth (`ENDSCALE`)."""

    endpoint_depth_nodes: Integer | None = None
    """Most depth nodes in a table of end points against depth (`ENDSCALE`)."""

    fault_segments: Integer | None = None
    """Most fault segments (`FAULTDIM`)."""

    multiplier_tables: Integer | None = None
    """Number of productivity index multiplier tables (`PIMTDIMS`)."""

    multiplier_table_rows: Integer | None = None
    """Most rows in a productivity index multiplier table (`PIMTDIMS`)."""

    vfp_flow_rate_values: Integer | None = None
    """Most flow rate values on a production VFP table's axis (`VFPPDIMS`)."""

    vfp_tubing_head_pressure_values: Integer | None = None
    """Most tubing head pressure values on a production VFP table's axis (`VFPPDIMS`)."""

    vfp_water_fraction_values: Integer | None = None
    """Most water fraction values on a production VFP table's axis (`VFPPDIMS`)."""

    vfp_gas_fraction_values: Integer | None = None
    """Most gas fraction values on a production VFP table's axis (`VFPPDIMS`)."""

    vfp_artificial_lift_values: Integer | None = None
    """Most artificial lift values on a production VFP table's axis (`VFPPDIMS`)."""

    production_vfp_tables: Integer | None = None
    """Number of production VFP tables (`VFPPDIMS`)."""

    injection_vfp_tables: Integer | None = None
    """Number of injection VFP tables (`VFPIDIMS`)."""


DIMENSION_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("TABDIMS", "num_saturation_tables", "saturation_tables"),
    ("TABDIMS", "num_pvt_tables", "pvt_tables"),
    ("TABDIMS", "max_saturation_table_nodes", "saturation_table_nodes"),
    ("TABDIMS", "max_pvt_pressure_nodes", "pvt_pressure_nodes"),
    ("TABDIMS", "max_dissolved_gas_ratio_nodes", "dissolved_gas_ratio_nodes"),
    ("TABDIMS", "max_vaporized_oil_ratio_nodes", "vaporized_oil_ratio_nodes"),
    ("EQLDIMS", "max_equilibration_regions", "equilibration_regions"),
    ("EQLDIMS", "max_pressure_depth_nodes", "pressure_depth_nodes"),
    ("EQLDIMS", "max_rs_rv_depth_nodes", "ratio_depth_nodes"),
    ("REGDIMS", "max_fluid_in_place_regions", "fluid_in_place_regions"),
    ("REGDIMS", "max_fluid_in_place_region_sets", "fluid_in_place_region_sets"),
    ("REGDIMS", "max_reservoir_regions", "reservoir_regions"),
    ("REGDIMS", "max_flux_regions", "flux_regions"),
    ("WELLDIMS", "max_wells", "wells"),
    ("WELLDIMS", "max_connections_per_well", "connections_per_well"),
    ("WELLDIMS", "max_groups", "groups"),
    ("WELLDIMS", "max_wells_per_group", "wells_per_group"),
    ("ENDSCALE", "num_endpoint_depth_tables", "endpoint_depth_tables"),
    ("ENDSCALE", "num_endpoint_depth_nodes", "endpoint_depth_nodes"),
    ("FAULTDIM", "max_fault_segments", "fault_segments"),
    ("PIMTDIMS", "max_multiplier_tables", "multiplier_tables"),
    ("PIMTDIMS", "max_multiplier_table_rows", "multiplier_table_rows"),
    ("VFPPDIMS", "max_flow_rate_values", "vfp_flow_rate_values"),
    ("VFPPDIMS", "max_tubing_head_pressure_values", "vfp_tubing_head_pressure_values"),
    ("VFPPDIMS", "max_water_fraction_values", "vfp_water_fraction_values"),
    ("VFPPDIMS", "max_gas_fraction_values", "vfp_gas_fraction_values"),
    ("VFPPDIMS", "max_artificial_lift_values", "vfp_artificial_lift_values"),
    ("VFPPDIMS", "max_production_vfp_tables", "production_vfp_tables"),
    ("VFPIDIMS", "max_injection_vfp_tables", "injection_vfp_tables"),
)
"""Where each `ModelDimensions` field comes from: the keyword, the item in its record, and the field."""


def load_saturation_options(deck_file: DeckFile) -> SaturationOptions:
    """
    Read `SATOPTS`, `ENDSCALE` and `SCALECRS`.

    :param deck_file: The deck to read.
    :returns: The options, all off for whichever of the keywords the deck does not have.
    """
    listed = frozenset(str(option).upper() for option in deck_file.get("SATOPTS") or ())
    endpoint_scaling = deck_file.get_shared("ENDSCALE")
    critical_scaling = deck_file.get_shared("SCALECRS")
    return SaturationOptions(
        endpoint_scaling_enabled=endpoint_scaling is not None,
        directional_endpoint_scaling=bool(endpoint_scaling)
        and endpoint_scaling["directional"] == "DIRECT",
        irreversible_endpoint_scaling=bool(endpoint_scaling)
        and endpoint_scaling["reversibility"] == "IRREVERS",
        scale_critical_saturations=bool(critical_scaling) and bool(critical_scaling["enabled"]),
        hysteresis_enabled="HYSTER" in listed,
        directional_relative_permeability="DIRECT" in listed,
        irreversible_directional_relative_permeability="IRREVERS" in listed,
        surface_tension_dependence="SURFTENS" in listed,
        unrecognized=listed - {"HYSTER", "DIRECT", "IRREVERS", "SURFTENS"},
    )


def load_equilibration_options(deck_file: DeckFile) -> EquilibrationOptions:
    """
    Read `EQLOPTS`.

    :param deck_file: The deck to read.
    :returns: The options, all off if the deck has no `EQLOPTS`.
    """
    listed = frozenset(str(option).upper() for option in deck_file.get("EQLOPTS") or ())
    return EquilibrationOptions(
        threshold_pressures="THPRES" in listed,
        irreversible_threshold_pressures="IRREVERS" in listed,
        quiescent_start="QUIESC" in listed,
        mobile_fluid_endpoint_correction="MOBILE" in listed,
        unrecognized=listed - {"THPRES", "IRREVERS", "QUIESC", "MOBILE"},
    )


def load_model_dimensions(deck_file: DeckFile) -> ModelDimensions | None:
    """
    Read the deck's dimension keywords.

    :param deck_file: The deck to read.
    :returns: The dimensions, or `None` if the deck has none of the dimension keywords.
    """
    keywords = {keyword for keyword, _, _ in DIMENSION_SOURCES}
    if not deck_file.has_any(*keywords):
        return None
    records = {keyword: deck_file.get_shared(keyword) for keyword in keywords}
    values: dict[str, typing.Any] = {}
    for keyword, item, field in DIMENSION_SOURCES:
        record = records[keyword]
        if record is not None and record.get(item) is not None:
            values[field] = int(record[item])
    return ModelDimensions(**values)


def load_minimum_pore_volume(deck_file: DeckFile) -> Number | None:
    """
    Read `MINPV`, the pore volume below which a cell takes no part in the flow.

    The deck gives it in reservoir volume: cubic metres (`METRIC`), reservoir barrels (`FIELD`) or
    cubic centimetres (`LAB`). It is returned in the volume unit pore volumes have here (m³, ft³
    or cm³), so the two can be compared directly.

    :param deck_file: The deck to read.
    :returns: The minimum pore volume, or `None` if the deck has no `MINPV` (the simulator's own
        default then applies).
    """
    record = deck_file.get("MINPV")
    if record is None:
        return None
    minimum = float(record["minpv"])
    if deck_file.unit_system == UnitSystem.FIELD:
        minimum *= c.BARRELS_TO_CUBIC_FEET
    return minimum


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

    maximum_solution_gor_increase_rate: Number | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.ge(0)),  # type: ignore[arg-type]
    )
    """
    The fastest the solution gas-oil ratio of a cell may rise (`DRSDT`), in the unit of the gas-oil
    ratio per time unit of `unit_system`. `None` when the deck sets no limit.

    Oil that is under-saturated with gas does not take up gas from a free gas phase at once. This
    caps how much it can take up per unit of time, so gas moving into contact with oil dissolves
    gradually.
    """

    solution_gor_limit_applies_to_all_cells: Boolean = True
    """
    Whether the solution gas-oil ratio limit holds in every cell (`DRSDT` option `ALL`) or only in
    cells that hold free gas (`FREE`).
    """

    maximum_vaporized_oil_gas_ratio_increase_rate: Number | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.ge(0)),  # type: ignore[arg-type]
    )
    """
    The fastest the vaporized oil-gas ratio of a cell may rise (`DRVDT`), in the unit of that ratio
    per time unit of `unit_system`. `None` when the deck sets no limit.

    The counterpart of `maximum_solution_gor_increase_rate`: how fast gas that is under-saturated
    with oil can take up oil.
    """

    well_index_regime_constant: Number | None = None
    """
    The constant added to `ln(re / rw) + skin` in the well index, which selects the flow regime
    the index assumes: 0 for steady state, -3/4 for pseudo steady state, 1/2 for transient. `None`
    uses the `WELL_INDEX_REGIME_CONSTANT` constant of `constants`. It is handed to the wells API
    that builds the well indices.

    A deck's own well index (Eclipse's) has no such constant, so a deck loads as 0.
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

    phase_appearance_tolerance: Number = attrs.field(
        default=1e-6, validator=attrs.validators.ge(0)
    )
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
class MessageLimits:
    """How many messages of each kind are printed, and how many of each stop the run (`MESSAGES`)."""

    message_print_limit: Integer = 1_000_000
    """Most ordinary messages printed."""
    comment_print_limit: Integer = 1_000_000
    """Most comments printed."""
    warning_print_limit: Integer = 10_000
    """Most warnings printed."""
    problem_print_limit: Integer = 100
    """Most problems printed."""
    error_print_limit: Integer = 100
    """Most errors printed."""
    bug_print_limit: Integer = 100
    """Most bugs printed."""
    message_stop_limit: Integer = 1_000_000
    """Ordinary messages after which the run stops."""
    comment_stop_limit: Integer = 1_000_000
    """Comments after which the run stops."""
    warning_stop_limit: Integer = 10_000
    """Warnings after which the run stops."""
    problem_stop_limit: Integer = 100
    """Problems after which the run stops."""
    error_stop_limit: Integer = 10
    """Errors after which the run stops."""
    bug_stop_limit: Integer = 1
    """Bugs after which the run stops."""
    group_print_limit: Integer = 10
    """Most messages of one group printed."""


@attrs.frozen(kw_only=True, slots=True)
class ReportRequest:
    """One request in the deck for something to be reported, and from when it applies."""

    keyword: str
    """The keyword that made the request (`RPTPROPS`, `RPTSOL`, `RPTRST` or `RPTSCHED`)."""

    time: Number
    """When the request takes effect, in the time unit of the unit system (0 for the start)."""

    mnemonics: tuple[tuple[str, Number | None], ...]
    """The mnemonics asked for with their level, `None` where the mnemonic has no level, in the order written."""

    flags: tuple[str, ...] = ()
    """Bare flags written alongside the mnemonics."""


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

    echo_input: Boolean = True
    """Whether the input is echoed to the log as it is read (`NOECHO` turns it off)."""

    formatted_input: Boolean = False
    """Whether input files are formatted rather than binary (`FMTIN`)."""

    formatted_output: Boolean = False
    """Whether output files are formatted rather than binary (`FMTOUT`)."""

    unified_input: Boolean = False
    """Whether restart input comes from one unified file (`UNIFIN`)."""

    unified_output: Boolean = False
    """Whether restart output goes to one unified file (`UNIFOUT`)."""

    write_initial_file: Boolean = False
    """Whether the initial file of static properties is written (`INIT`)."""

    write_grid_file: Boolean = False
    """Whether the grid geometry file is written (`GRIDFILE` with its first item set)."""

    write_extended_grid_file: Boolean = False
    """Whether the extended grid geometry file is written (`GRIDFILE` with its second item set)."""

    write_run_summary: Boolean = False
    """Whether a run summary is written (`RUNSUM`)."""

    message_limits: "MessageLimits | None" = None
    """How many of each kind of message are printed and how many stop the run (`MESSAGES`). `None` when the deck sets none."""

    report_requests: tuple["ReportRequest", ...] = ()
    """What the deck asks to have reported and when (`RPTPROPS`, `RPTSOL`, `RPTRST`, `RPTSCHED`), ordered by the time they take effect."""


def get_scheduled_at_start(deck_file: DeckFile, keyword: str) -> dict[str, typing.Any] | None:
    """
    The record of a scheduled keyword that applies from the start of the run.

    A scheduled keyword changes its setting from the time it appears at. This is the one in effect
    at time zero. The later ones are warned about, as a setting held for the whole run cannot
    follow them.

    :param deck_file: The deck to read.
    :param keyword: The scheduled keyword, such as `DRSDT`.
    :returns: The record (the last written if several are at time zero), or `None` if there is none
        at time zero.
    """
    occurrences = deck_file.get_shared(keyword) or []
    later = sorted({item["schedule_time"] for item in occurrences if item["schedule_time"] != 0})
    if later:
        warnings.warn(
            f"`{keyword}` appears again at {', '.join(f'{time:g}' for time in later)} "
            f"({deck_file.unit_system.name} time unit); only the one in effect at the start of the "
            "run is used.",
            stacklevel=2,
        )
    at_start = [item for item in occurrences if item["schedule_time"] == 0]
    return at_start[-1] if at_start else None


def load_dissolution_limits(deck_file: DeckFile) -> dict[str, typing.Any]:
    """
    Read how fast gas may dissolve in oil and oil vaporize in gas, from `DRSDT` and `DRVDT`.

    :param deck_file: The deck to read.
    :returns: Settings for `PhysicsOptions`, holding only what the deck sets.
    """
    settings: dict[str, typing.Any] = {}
    solution = get_scheduled_at_start(deck_file, "DRSDT")
    if solution is not None:
        if solution["max_rate"] is not None:
            settings["maximum_solution_gor_increase_rate"] = float(solution["max_rate"])
        settings["solution_gor_limit_applies_to_all_cells"] = (
            str(solution["option"]).upper() == "ALL"
        )

    vaporized = get_scheduled_at_start(deck_file, "DRVDT")
    if vaporized is not None and vaporized["max_rate"] is not None:
        settings["maximum_vaporized_oil_gas_ratio_increase_rate"] = float(vaporized["max_rate"])
    return settings


def load_output_options(deck_file: DeckFile) -> OutputOptions:
    """
    Read the keywords that decide what a run writes and reports.

    These are `NOECHO`, `FMTIN`, `FMTOUT`, `UNIFIN`, `UNIFOUT`, `INIT`, `GRIDFILE`, `RUNSUM`,
    `MESSAGES` and the report requests `RPTPROPS`, `RPTSOL`, `RPTRST` and `RPTSCHED`. The other
    output settings have no deck keyword and keep their defaults.

    :param deck_file: The deck to read.
    :returns: The output options.
    """
    present = deck_file.has_many(
        "NOECHO", "FMTIN", "FMTOUT", "UNIFIN", "UNIFOUT", "INIT", "RUNSUM"
    )
    settings: dict[str, typing.Any] = {
        "echo_input": not present["NOECHO"],
        "formatted_input": present["FMTIN"],
        "formatted_output": present["FMTOUT"],
        "unified_input": present["UNIFIN"],
        "unified_output": present["UNIFOUT"],
        "write_initial_file": present["INIT"],
        "write_run_summary": present["RUNSUM"],
    }
    grid_file = deck_file.get_shared("GRIDFILE")
    if grid_file is not None:
        settings["write_grid_file"] = bool(grid_file["grid"])
        settings["write_extended_grid_file"] = bool(grid_file["egrid"])

    messages = deck_file.get_shared("MESSAGES")
    if messages is not None:
        settings["message_limits"] = MessageLimits(**{
            name: int(value) for name, value in messages.items()
        })

    requests: list[ReportRequest] = []
    for keyword in ("RPTPROPS", "RPTSOL", "RPTRST", "RPTSCHED"):
        for record in deck_file.get_shared(keyword) or []:
            requests.append(
                ReportRequest(
                    keyword=keyword,
                    time=float(record.get("schedule_time", 0.0)),
                    mnemonics=tuple(
                        (name, None if level is None else float(level))
                        for name, level in record["mnemonics"].items()
                    ),
                    flags=tuple(record["flags"]),
                )
            )
    settings["report_requests"] = tuple(sorted(requests, key=lambda request: request.time))
    return OutputOptions(**settings)

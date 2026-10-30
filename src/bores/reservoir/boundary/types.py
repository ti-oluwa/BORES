import typing

import attrs
from typing_extensions import Self

from bores.reservoir.boundary.base import (
    BoundaryCondition,
    BoundaryConditionType,
    FluxBasis,
)
from bores.types import Number, UnitConversionTable, UnitSystem
from bores.units import get_conversion_factors
from bores.utils import scale

__all__ = [
    "ConstantFluxBoundary",
    "ConstantPressureBoundary",
    "ProductivityIndexBoundary",
]


@attrs.frozen(slots=True)
class ConstantFluxBoundary(BoundaryCondition):
    """
    Constant-flux (Neumann) boundary condition.

    A fixed flow rate through the faces of a region. `basis` says what the rate is measured
    against: the whole region (the default), each face, or each unit of face area. This is a
    pure parameter container that compiles into the static flux of each face.

    The default `flux=0.0` gives a **sealed (no-flow) boundary**, the most common boundary
    condition for reservoir flanks, top, and base. A positive `flux` is flow into the
    reservoir (injection or aquifer influx) and a negative `flux` is production or efflux.

    **Unit system**

    A `TOTAL` or `PER_FACE` `flux` is a *volume-per-time* in the unit system:

    - `FIELD` - ft³/day
    - `METRIC` - m³/day
    - `LAB` - cm³/hour
    - `SI` - m³/s

    A `PER_AREA` `flux` is that per unit of area, for example ft³/day/ft² in `FIELD`.

    :param flux: The flow rate, measured against what `basis` says. Default is 0 (no-flow).
    :param basis: Whether `flux` is the rate of the whole region, of each face, or per unit of
        face area. See `FluxBasis`.
    :param unit_system: Unit system for `flux`. Default `FIELD`.
    """

    condition_type: typing.ClassVar[BoundaryConditionType] = BoundaryConditionType.FLUX

    flux: Number = 0.0
    """The flow rate in `unit_system`, measured against `basis`. Positive = into reservoir."""

    basis: FluxBasis = FluxBasis.TOTAL
    """What `flux` is measured against: the whole region, each face, or each unit of face area."""

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system for `flux`."""

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `ConstantFluxBoundary` with `flux` rescaled to
        *target*.

        :param target: Target `UnitSystem`.
        :returns: New `ConstantFluxBoundary` in `target` units.
        """
        if target == self.unit_system:
            return self
        factors = get_conversion_factors(self.unit_system, target, table=table)
        flux_factor = factors["reservoir_rate"]
        if self.basis is FluxBasis.PER_AREA:
            flux_factor = flux_factor / factors["length"] ** 2
        return attrs.evolve(self, flux=scale(self.flux, flux_factor), unit_system=target)

    def is_no_flow(self) -> bool:
        """Return `True` if `flux == 0` (sealed boundary)."""
        return self.flux == 0


@attrs.frozen(slots=True)
class ConstantPressureBoundary(BoundaryCondition):
    """
    Constant-pressure (Dirichlet) boundary condition.

    A fixed pressure at every face in the region. This is a pure parameter
    container, compiles directly into `CompiledBoundaryConditions.static_values`.
    The solver uses this as a ghost-cell pressure to compute the face flux
    via the half-transmissibility:

        q_face = T_half * (pressure_boundary - pressure_interior)

    Typical applications:

    - **Strong aquifer**: an aquifer so large that its pressure does not
      change over the simulation period. Set `pressure` to the initial
      aquifer pressure.
    - **Constant-pressure producer**: a producing boundary held at
      abandonment pressure.
    - **Injection at manifold pressure**: an injection flank held at the
      injection pump delivery pressure.

    **Unit system**

    `pressure` is in the *pressure* unit of `unit_system`:

    - `FIELD` - psi
    - `METRIC` - bar
    - `LAB` - atm
    - `SI` - Pa

    :param pressure: Prescribed pressure at all faces in the region.
    :param unit_system: Unit system for `pressure`.
    """

    condition_type: typing.ClassVar[BoundaryConditionType] = BoundaryConditionType.PRESSURE

    pressure: Number
    """Prescribed boundary pressure in `unit_system` units."""

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system for `pressure`."""

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `ConstantPressureBoundary` with `pressure` rescaled
        to *target*.

        :param target: Target `UnitSystem`.
        :returns: New `ConstantPressureBoundary` in `target` units.
        """
        if target == self.unit_system:
            return self
        factors = get_conversion_factors(self.unit_system, target, table=table)
        return attrs.evolve(
            self, pressure=scale(self.pressure, factors["pressure"]), unit_system=target
        )


@attrs.frozen(slots=True)
class ProductivityIndexBoundary(BoundaryCondition):
    """
    Robin (mixed) boundary condition using a productivity-index formulation.

    A pure parameter container that compiles into `CompiledBoundaryConditions.productivity_indices`.
    The actual per-face flux formula:

        q_face[i] = productivity_index * (pressure_boundary - p_interior[i])

    is a free function operating on `CompiledBoundaryConditions` arrays and
    a plain cell-pressure array (`boundary/cache.py`).

    The productivity index is uniform across every face in the region.

    This is physically equivalent to a **well-index / productivity-index**
    formulation applied at the grid boundary, useful for:

    - **Partial aquifer**: an aquifer whose influx is approximately
      proportional to the pressure difference at the boundary.
    - **Leaky boundary**: a boundary face with a known transmissibility-like
      coefficient connecting the reservoir to an external pressure source.
    - **Injection manifold**: injection driven by a manifold pressure with a
      known injectivity per face.

    **Sign convention**

    A positive result means flow *into* the reservoir (net influx):
    `pressure_boundary > p_interior` -> positive flux.
    A negative result means flow *out of* the reservoir (net production):
    `pressure_boundary < p_interior` -> negative flux.

    **Unit system**

    - `pressure_boundary` - pressure in `unit_system`.
    - `productivity_index` - volume/(time·pressure) in `unit_system`
      (ft³/day/psi, m³/day/bar, etc.).
    """

    condition_type: typing.ClassVar[BoundaryConditionType] = BoundaryConditionType.FLUX

    pressure_boundary: Number
    """Reference boundary pressure in `unit_system` units."""

    productivity_index: Number = 1.0
    """Uniform productivity index (volume/time/pressure in `unit_system`)."""

    unit_system: UnitSystem = UnitSystem.FIELD
    """Unit system for `pressure_boundary` and `productivity_index`."""

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `ProductivityIndexBoundary` with `pressure_boundary`
        and `productivity_index` rescaled to *target*.

        :param target: Target `UnitSystem`.
        :returns: New `ProductivityIndexBoundary` in *target* units.
        """
        if target == self.unit_system:
            return self
        factors = get_conversion_factors(self.unit_system, target, table=table)
        productivity_index_factor = factors["reservoir_rate"] / factors["pressure"]
        return attrs.evolve(
            self,
            pressure_boundary=scale(self.pressure_boundary, factors["pressure"]),
            productivity_index=scale(self.productivity_index, productivity_index_factor),
            unit_system=target,
        )

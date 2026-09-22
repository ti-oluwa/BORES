import typing

import attrs
from typing_extensions import Self

from bores.constants import get_conversion_factors
from bores.reservoir.boundary.base import (
    BoundaryCondition,
    BoundaryConditionType,
    boundary_condition,
)
from bores.types import Number, UnitConversionTable, UnitSystem

__all__ = [
    "ConstantFluxBoundary",
    "ConstantPressureBoundary",
    "ProductivityIndexBoundary",
]


@boundary_condition
@attrs.frozen(slots=True)
class ConstantFluxBoundary(BoundaryCondition):
    """
    Constant-flux (Neumann) boundary condition.

    A uniform volumetric flow rate at every face in the region. This is a pure
    parameter container, compiles directly into `CompiledBoundaryConditions.static_values`.

    The default `flux=0.0` gives a **sealed (no-flow) boundary**; the most
    common boundary condition for reservoir flanks, top, and base. A positive
    `flux` represents flow into the reservoir (injection or aquifer
    influx); a negative `flux` represents production or efflux.

    **Unit system**

    The `flux` value is interpreted in the *volume-per-time* unit of
    `unit_system`:

    - `FIELD` - ft³/day
    - `METRIC` - m³/day
    - `LAB` - cm³/hour
    - `SI` - m³/s

    :param flux: Volumetric flow rate applied uniformly across the region
        (volume/time in `unit_system`). Default is 0 (no-flow).
    :param unit_system: Unit system for `flux`. Default `FIELD` (ft³/day).
    """

    __type__: typing.ClassVar[str] = "constant_flux_boundary"
    condition_type: typing.ClassVar[BoundaryConditionType] = BoundaryConditionType.FLUX

    flux: Number = 0.0
    """Volumetric flux (volume/time in `unit_system`). Positive = into reservoir."""

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
        return self.__class__(flux=self.flux * factors["reservoir_rate"], unit_system=target)

    def is_no_flow(self) -> bool:
        """Return `True` if `flux == 0` (sealed boundary)."""
        return self.flux == 0


@boundary_condition
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

    __type__: typing.ClassVar[str] = "constant_pressure_boundary"
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
        return self.__class__(pressure=self.pressure * factors["pressure"], unit_system=target)


@boundary_condition
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

    __type__: typing.ClassVar[str] = "productivity_index_boundary"
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
        return self.__class__(
            pressure_boundary=self.pressure_boundary * factors["pressure"],
            productivity_index=self.productivity_index * productivity_index_factor,
            unit_system=target,
        )

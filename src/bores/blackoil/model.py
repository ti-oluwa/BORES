"""Top-level black-oil simulation model: reservoir, fluid, wells, boundary conditions."""

import typing

import numpy.typing as npt
from typing_extensions import Self

from bores.blackoil.fluids.model import BlackOil
from bores.constants import UnitConversionTable, build_unit_conversion_table
from bores.errors import ValidationError
from bores.reservoir.boundary.conditions import BoundaryConditions
from bores.reservoir.model import Reservoir
from bores.serde.stores import StoreSerializable
from bores.types import UnitSystem
from bores.wells.model import WellSystem

if typing.TYPE_CHECKING:
    from bores.blackoil.compile import CompiledBlackOilModel

__all__ = ["BlackOilModel"]


class BlackOilModel(
    StoreSerializable,
    fields={
        "reservoir": Reservoir,
        "fluid": BlackOil,
        "wells": WellSystem | None,
        "boundary_conditions": BoundaryConditions | None,
        "unit_system": UnitSystem,
    },
):
    """
    Black-oil model characterization.

    Bundles a reservoir, its fluid physics, and optionally wells and
    boundary conditions. All components are normalised to `unit_system`
    at construction.
    """

    def __init__(
        self,
        reservoir: Reservoir,
        fluid: BlackOil,
        wells: WellSystem | None = None,
        boundary_conditions: BoundaryConditions | None = None,
        unit_system: UnitSystem | None = None,
    ) -> None:
        """
        Initialize the black-oil model.

        :param reservoir: Reservoir characterization.
        :param fluid: Black-oil fluid physics.
        :param wells: Optional wells sub-system.
        :param boundary_conditions: Optional boundary conditions.
        :param unit_system: Target unit system for every component. When
            `None`, all supplied components must share the same unit
            system. If they do not, a `ValidationError` is raised. When
            provided, each component is converted to `unit_system` as
            needed.
        :raises ValidationError: If `unit_system` is `None` and the
            supplied components do not all share the same unit system.
        """
        unit_conversion_table = build_unit_conversion_table()

        reservoir_unit_system = reservoir.unit_system
        fluid_unit_system = fluid.unit_system
        wells_unit_system = wells.unit_system if wells is not None else None
        boundary_conditions_unit_system = (
            boundary_conditions.unit_system if boundary_conditions is not None else None
        )

        if unit_system is None:
            systems = {reservoir_unit_system, fluid_unit_system}
            if wells_unit_system is not None:
                systems.add(wells_unit_system)
            if boundary_conditions_unit_system is not None:
                systems.add(boundary_conditions_unit_system)

            if len(systems) > 1:
                raise ValidationError(
                    "All components must share the same unit system when "
                    "`unit_system` is not explicitly provided. "
                    f"Found: {sorted(s.value for s in systems)}. "
                    "Pass `unit_system` explicitly to convert all components "
                    "to a common system."
                )
            unit_system = reservoir_unit_system

        if reservoir_unit_system != unit_system:
            reservoir = reservoir.convert(unit_system, table=unit_conversion_table)

        if fluid_unit_system != unit_system:
            fluid = fluid.convert(unit_system, table=unit_conversion_table)

        if wells is not None and wells_unit_system != unit_system:
            wells = wells.convert(unit_system, table=unit_conversion_table)

        if boundary_conditions is not None and boundary_conditions_unit_system != unit_system:
            boundary_conditions = boundary_conditions.convert(
                unit_system, table=unit_conversion_table
            )

        self.reservoir = reservoir
        """Reservoir characterization, in `unit_system`."""

        self.fluid = fluid
        """Black-oil fluid physics, in `unit_system`."""

        self.wells = wells
        """Optional wells sub-system, in `unit_system`. `None` if not supplied."""

        self.boundary_conditions = boundary_conditions
        """Optional boundary conditions, in `unit_system`. `None` if not supplied."""

        self.unit_system = unit_system
        """Unit system in which every component is expressed."""

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Convert the model to the *target* unit system.

        :param target: Target unit system.
        :param table: Optional custom conversion table.
        :returns: New `BlackOilModel` with every component converted to `target`.
        """
        if target == self.unit_system:
            return self
        return self.__class__(
            reservoir=self.reservoir.convert(target, table=table),
            fluid=self.fluid.convert(target, table=table),
            wells=(self.wells.convert(target, table=table) if self.wells is not None else None),
            boundary_conditions=(
                self.boundary_conditions.convert(target, table=table)
                if self.boundary_conditions is not None
                else None
            ),
            unit_system=target,
        )

    def compile(
        self, *, dtype: npt.DTypeLike = None, **resolve_kwargs: typing.Any
    ) -> "CompiledBlackOilModel":
        """
        Compiles this model into a `CompiledBlackOilModel`.

        :param dtype: Forwarded to `compile_model`.
        :param resolve_kwargs: Forwarded to `compile_model`.
        :returns: The compiled model.
        :raises BoundaryConditionCompilationError: If `boundary_conditions` is set.
        :raises WellCompilationError: If well compilation fails.
        """
        from bores.blackoil.compile import compile_model

        return compile_model(self, dtype=dtype, **resolve_kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"reservoir={self.reservoir!r}, "
            f"fluid={self.fluid!r}, "
            f"has_wells={self.wells is not None}, "
            f"has_boundary_conditions={self.boundary_conditions is not None}, "
            f"unit_system={self.unit_system.value!r}"
            f")"
        )

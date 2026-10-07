import enum
import typing

from typing_extensions import Self

from bores.types import Number, UnitConversionTable, UnitSystem

__all__ = ["BoundaryCondition", "BoundaryConditionType", "InitialPressureFactory"]

InitialPressureFactory: typing.TypeAlias = typing.Callable[
    [typing.Mapping[str, typing.Any]], Number
]
"""
Callable that works out the initial pressure of an aquifer whose deck record leaves it
defaulted (`1*`), given that record.
"""


class BoundaryConditionType(enum.Enum):
    """
    Discriminator that controls how the solver uses a boundary condition's output.

    `PRESSURE`
        A prescribed pressure (psi / bar / atm / Pa depending on
        `unit_system`) at each boundary face. The solver applies a
        Dirichlet constraint: it adds `T_face * p_boundary` to the RHS
        and `T_face` to the diagonal of the flow equation for each face.

    `FLUX`
        A volumetric flow rate into the reservoir (`ft³/day` / `m³/day` /
        etc.) at each boundary face. The solver applies a Neumann source
        term: it adds the flux directly to the RHS of the cell owning the
        face. A flux of zero is a sealed (no-flow) boundary.
    """

    PRESSURE = "pressure"
    FLUX = "flux"


class BoundaryCondition:
    """
    Base class for every boundary-condition parameter container.

    Subclasses declare:

    - `condition_type: typing.ClassVar[BoundaryConditionType]` - static
      per concrete class, not computed per instance.
    - `unit_system: UnitSystem` field.
    - `convert` method: returns a unit-rescaled copy.

    **Unit system contract**

    Every concrete subclass that carries dimensional parameters (pressures,
    rates, permeabilities) must implement the `SupportsUnitSystem` protocol.
    """

    unit_system: UnitSystem
    condition_type: typing.ClassVar[BoundaryConditionType]

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        raise NotImplementedError

import enum
import threading
import typing

from typing_extensions import Self

from bores.serde.registry import make_serializable_type_registrar
from bores.serde.stores import StoreSerializable
from bores.types import UnitConversionTable, UnitSystem

__all__ = [
    "BoundaryCondition",
    "BoundaryConditionType",
]


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


class BoundaryCondition(StoreSerializable):
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

    __abstract_serializable__ = True

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


# Registry of concrete boundary condition classes
BOUNDARY_CONDITIONS: dict[str, type[BoundaryCondition]] = {}
boundary_condition = make_serializable_type_registrar(
    base_cls=BoundaryCondition,
    registry=BOUNDARY_CONDITIONS,
    lock=threading.Lock(),
    key_attr="__type__",
    override=False,
    auto_register_serializer=True,
    auto_register_deserializer=True,
)
"""
Class decorator that registers a `BoundaryCondition` subclass for
serialisation. Must be applied to every concrete condition class that needs
to survive a `dump` / `load` cycle.
"""

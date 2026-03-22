import enum
import typing
from typing import TypeAlias

import attrs
import numpy as np
from scipy.sparse import csr_array, csr_matrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import LinearOperator  # type: ignore[import-untyped]
from typing_extensions import TypedDict

from bores.errors import ValidationError
from bores.serialization import Serializable

__all__ = [
    "ArrayLike",
    "CapillaryPressures",
    "Coordinates",
    "EvolutionScheme",
    "FluidPhase",
    "Interpolator",
    "MiscibilityModel",
    "MixingRule",
    "NDimension",
    "OneDimension",
    "OneDimensionalGrid",
    "Orientation",
    "Preconditioner",
    "Range",
    "RelativeMobilityRange",
    "RelativePermeabilities",
    "Solver",
    "SolverFunc",
    "ThreeDimensionalGrid",
    "ThreeDimensions",
    "TwoDimensionalGrid",
    "TwoDimensions",
    "WellFluidType",
    "Wettability",
    "Wettability",
]

T = typing.TypeVar("T")
Tco = typing.TypeVar("Tco", covariant=True)

S = typing.TypeVar("S")


NDimension = typing.TypeVar("NDimension", bound=typing.Tuple[int, ...])
Coordinates = typing.TypeVar("Coordinates", bound=typing.Tuple[int, ...])

ThreeDimensions: TypeAlias = typing.Tuple[int, int, int]
"""3D indices"""
TwoDimensions: TypeAlias = typing.Tuple[int, int]
"""2D indices"""
OneDimension: TypeAlias = typing.Tuple[int]
"""1D index"""

Numeric = typing.Union[int, float, np.floating, np.integer]
NDimensionalGrid = np.ndarray[NDimension, np.dtype[np.floating]]
FloatOrArray = typing.Union[float, np.typing.NDArray[np.floating]]


ThreeDimensionalGrid = NDimensionalGrid[ThreeDimensions]
"""3D grid type for simulation data, represented as a 3D NumPy array of floats"""
TwoDimensionalGrid = NDimensionalGrid[TwoDimensions]
"""2D grid type for simulation data, represented as a 2D NumPy array of floats"""
OneDimensionalGrid = NDimensionalGrid[OneDimension]
"""1D grid type for simulation data, represented as a 1D NumPy array of floats"""


class Orientation(enum.Enum):
    """
    Enum representing directional orientation in a 2D/3D simulation.
    """

    X = "x"
    Y = "y"
    Z = "z"
    UNSET = "unset"

    def __str__(self) -> str:
        return self.value


class FluidPhase(enum.Enum):
    """Enum representing the phase of the fluid in the reservoir."""

    WATER = "water"
    GAS = "gas"
    OIL = "oil"

    def __str__(self) -> str:
        return self.value


WellFluidType = typing.Literal["water", "oil", "gas"]
"""Types of fluids that can be injected in the simulation"""

EvolutionScheme = typing.Literal["impes", "explicit", "implicit", "sequential_implicit"]
"""
Discretization methods for numerical simulations

- "impes": Implicit pressure, Explicit saturation
- "explicit": Both pressure and saturation are treated explicitly
- "implicit": Both pressure and saturation are treated implicitly
- "sequential_implicit": Implicit pressure followed by implicit saturation (Newton-Raphson)
"""

MiscibilityModel = typing.Literal["immiscible", "todd_longstaff"]
"""Miscibility models for fluid interactions in the simulation"""


class ArrayLike(typing.Protocol[Tco]):
    """
    Protocol for an array-like object that supports
    basic operations like length, indexing, iteration, and containment checks.
    """

    def __len__(self) -> int:
        """Returns the length of the array-like object."""
        ...

    def __getitem__(self, index: int, /) -> Tco:
        """Returns the item at the specified index."""
        ...

    def __iter__(self) -> typing.Iterator[Tco]:
        """Returns an iterator over the items in the array-like object."""
        ...

    def __contains__(self, obj: typing.Any, /) -> bool:
        """Checks if the object is in the array-like object."""
        ...


Interpolator = typing.Callable[[float], float]


PreconditionerStr = typing.Union[
    typing.Literal["cpr", "ilu", "amg", "block_jacobi", "polynomial", "diagonal"], str
]
PreconditionerFactory = typing.Callable[
    [typing.Union[csr_array, csr_matrix]], LinearOperator
]
Preconditioner = typing.Union[LinearOperator, PreconditionerStr, PreconditionerFactory]

SolverStr = typing.Union[
    typing.Literal["gmres", "lgmres", "bicgstab", "tfqmr", "cg", "cgs", "direct"], str
]


class SolverFunc(typing.Protocol):
    """
    Protocol for a solver function compatible with SciPy's linear solvers.
    """

    def __call__(
        self,
        A: typing.Any,
        b: typing.Any,
        x0: typing.Optional[typing.Any],
        *,
        rtol: float,
        atol: float,
        maxiter: typing.Optional[int],
        M: typing.Optional[typing.Any],
        callback: typing.Optional[typing.Callable[[np.typing.NDArray], None]],
    ) -> np.typing.NDArray: ...


Solver = typing.Union[SolverFunc, SolverStr]


class MixingRule(typing.Protocol):
    """
    Protocol for a mixing rule function that combines two properties
    based on their saturations.
    """

    def __call__(
        self,
        *,
        kro_w: FloatOrArray,
        kro_g: FloatOrArray,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
    ) -> FloatOrArray:
        """
        Combines two properties based on their saturations.

        :param kro_w: Property value for water phase.
        :param kro_g: Property value for gas phase.
        :param water_saturation: Saturation of the water phase.
        :param oil_saturation: Saturation of the oil phase.
        :param gas_saturation: Saturation of the gas phase.
        :return: Combined property value.
        """
        ...


class RelativePermeabilities(TypedDict):
    """Dictionary holding relative permeabilities for different phases."""

    water: FloatOrArray
    oil: FloatOrArray
    gas: FloatOrArray


class RelativePermeabilityDerivatives(TypedDict):
    """Dictionary holding relative permeabilities derivatives."""

    # w.r.t water
    dKrw_dSw: FloatOrArray
    dKro_dSw: FloatOrArray
    dKrg_dSw: FloatOrArray
    # w.r.t oil
    dKrw_dSo: FloatOrArray
    dKro_dSo: FloatOrArray
    dKrg_dSo: FloatOrArray
    # w.r.t gas
    dKrw_dSg: FloatOrArray
    dKro_dSg: FloatOrArray
    dKrg_dSg: FloatOrArray


class CapillaryPressures(typing.TypedDict):
    """Dictionary containing capillary pressures for different phase pairs."""

    oil_water: FloatOrArray  # Pcow = Po - Pw
    gas_oil: FloatOrArray  # Pcgo = Pg - Po


class CapillaryPressureDerivatives(TypedDict):
    """Dictionary containing capillary pressure derivatives for different phase pairs."""

    dPcow_dSw: FloatOrArray
    dPcow_dSo: FloatOrArray
    dPcgo_dSg: FloatOrArray


class Wettability(enum.Enum):
    """Enum representing the wettability type of the reservoir rock."""

    WATER_WET = "water_wet"
    OIL_WET = "oil_wet"
    MIXED_WET = "mixed_wet"

    def __str__(self) -> str:
        return self.value


@attrs.frozen
class Range(Serializable):
    """
    Class representing minimum and maximum values.
    """

    min: float
    """Minimum value."""
    max: float
    """Maximum value."""

    def __attrs_post_init__(self) -> None:
        if self.min > self.max:
            raise ValidationError("Minimum value cannot be greater than maximum value.")

    def clip(self, value: T) -> T:
        """
        Clips the given value between the minimum and maximum values.

        :param value: The value to be clipped.
        :return: The clipped value.
        """
        from bores.utils import clip

        return clip(value, self.min, self.max)

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> typing.Iterator[float]:
        yield self.min
        yield self.max

    def __contains__(self, item: float) -> bool:
        return self.min <= item <= self.max

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.min
        elif index == 1:
            return self.max
        else:
            raise IndexError("Index out of range for Range. Valid indices are 0 and 1.")


class RelativeMobilityRange(TypedDict):
    """
    Dictionary holding relative mobility ranges for different phases.
    """

    oil: Range
    water: Range
    gas: Range


Kcon = typing.TypeVar("Kcon", contravariant=True)
Vcon = typing.TypeVar("Vcon", contravariant=True)


class SupportsSetItem(typing.Protocol, typing.Generic[Kcon, Vcon]):
    """
    Protocol for objects that support item assignment.
    """

    def __setitem__(self, key: Kcon, value: Vcon, /) -> None:
        """Sets the item at the specified key to the given value."""
        ...


GasZFactorMethod = typing.Literal["papay", "hall-yarborough", "dak"]

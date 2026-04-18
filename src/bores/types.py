import enum
import typing
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array, csr_matrix  # type: ignore[import-untyped]
from scipy.sparse.linalg import LinearOperator  # type: ignore[import-untyped]
from typing_extensions import TypedDict

__all__ = [
    "ArrayLike",
    "CapillaryPressures",
    "Coordinates",
    "EvolutionScheme",
    "FluidPhase",
    "Interpolator",
    "MiscibilityModel",
    "MixingRuleFunc",
    "NDimension",
    "OneDimension",
    "OneDimensionalGrid",
    "Orientation",
    "Preconditioner",
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
FloatOrArray = typing.Union[float, npt.NDArray[np.floating]]


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

EvolutionScheme = typing.Literal[
    "impes",
    "explicit",
    "sequential-implicit",
    "full-sequential-implicit",
    "si",
    "full-si",
]
"""
Discretization methods for numerical simulations

- `"impes"`: Implicit pressure, Explicit saturation
- `"explicit"`: Both pressure and saturation are treated explicitly
- `"sequential-implicit"` or `"si"`: Both pressure and saturation are treated (sequentially) implicitly
- `"full-sequential-implicit"` or `"full-si"`: Both pressure and saturation are treated (sequentially) implicitly
"""

MiscibilityModel = typing.Literal["immiscible", "todd-longstaff"]
"""Miscibility models for fluid interactions in the simulation"""

Spacing = typing.Literal["cosine", "linspace"]

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
    typing.Literal[
        "gmres",
        "lgmres",
        "bicgstab",
        "tfqmr",
        "cg",
        "cgs",
        "minres",
        "bicg",
        "qmr",
        "gcrotmk",
        "direct",
    ],
    str,
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
        callback: typing.Optional[typing.Callable[[npt.NDArray], None]],
    ) -> npt.NDArray: ...


Solver = typing.Union[SolverFunc, SolverStr]


class MixingRuleFunc(typing.Protocol):
    """
    Protocol for a mixing rule function that combines two-phase oil relative
    permeabilities into a three-phase estimate.

    The extended signature provides every quantity a mixing rule could need,
    including the actual two-phase water and gas relative permeabilities and
    the oil endpoint, so that rules requiring the full Stone II normalization
    or saturation-weighted interpolations can be implemented without
    approximation.

    Parameters that a simple rule (e.g. geometric mean) does not need can
    simply be ignored in the implementation.
    """

    def __call__(
        self,
        *,
        kro_w: FloatOrArray,
        kro_g: FloatOrArray,
        krw: FloatOrArray,
        krg: FloatOrArray,
        kro_endpoint: FloatOrArray,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
    ) -> FloatOrArray:
        """
        Compute the three-phase oil relative permeability.

        :param kro_w: Two-phase oil relative permeability from the oil-water
            table at the current water saturation.
        :param kro_g: Two-phase oil relative permeability from the gas-oil
            table at the current gas (or oil) saturation.
        :param krw: Two-phase water relative permeability from the oil-water
            table at the current water saturation.
        :param krg: Two-phase gas relative permeability from the gas-oil
            table at the current gas saturation.
        :param kro_endpoint: Oil relative permeability at connate water
            saturation (kro at Sw=Swc, Sg=0).  Equal to ``max_oil_relperm``
            for normalized tables; 1.0 for unit-endpoint tables.
        :param water_saturation: Current water saturation.
        :param oil_saturation: Current oil saturation.
        :param gas_saturation: Current gas saturation.
        :return: Three-phase oil relative permeability.
        """
        ...


class MixingRulePartialDerivatives(TypedDict):
    """
    The seven partial derivatives of a three-phase oil relative permeability
    mixing rule with respect to each of its seven saturation-dependent
    arguments.

    The mixing rule signature is:

        kro = rule(kro_w, kro_g, krw, krg, kro_endpoint,
                   water_saturation, oil_saturation, gas_saturation)

    The derivatives here are taken with respect to the *inputs that vary
    with saturation* in a way that is explicit in the mixing rule itself.
    The chain-rule terms through `kro_w`, `kro_g`, `krw`, and `krg`
    back to physical saturations are assembled at the call site in the table
    derivative methods.

    Fields:

    d_kro_d_kro_w :
        ∂kro / ∂kro_w  - sensitivity to the oil-water two-phase oil kr.
    d_kro_d_kro_g :
        ∂kro / ∂kro_g  - sensitivity to the gas-oil two-phase oil kr.
    d_kro_d_krw :
        ∂kro / ∂krw  - sensitivity to the two-phase water kr.
        Zero for rules that do not use `krw` directly (most simple rules).
    d_kro_d_krg :
        ∂kro / ∂krg  - sensitivity to the two-phase gas kr.
        Zero for rules that do not use `krg` directly (most simple rules).
    d_kro_d_sw_explicit :
        ∂kro / ∂Sw  through the explicit water-saturation argument of the
        mixing rule (e.g. saturation weighting in `eclipse_rule`).
        Zero for rules that do not depend directly on saturation.
    d_kro_d_so_explicit :
        ∂kro / ∂So  through the explicit oil-saturation argument.
    d_kro_d_sg_explicit :
        ∂kro / ∂Sg  through the explicit gas-saturation argument.
    """

    d_kro_d_kro_w: FloatOrArray
    d_kro_d_kro_g: FloatOrArray
    d_kro_d_krw: FloatOrArray
    d_kro_d_krg: FloatOrArray
    d_kro_d_sw_explicit: FloatOrArray
    d_kro_d_so_explicit: FloatOrArray
    d_kro_d_sg_explicit: FloatOrArray


class MixingRuleDFunc(typing.Protocol):
    """
    Protocol for a mixing rule partial derivatives function.
    """

    def __call__(
        self,
        *,
        kro_w: FloatOrArray,
        kro_g: FloatOrArray,
        krw: FloatOrArray,
        krg: FloatOrArray,
        kro_endpoint: FloatOrArray,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
    ) -> typing.Union[
        MixingRulePartialDerivatives,
        typing.Tuple[
            FloatOrArray,
            FloatOrArray,
            FloatOrArray,
            FloatOrArray,
            FloatOrArray,
            FloatOrArray,
            FloatOrArray,
        ],
    ]:
        """
        Return the seven partial derivatives of the mixing rule.

        The tuple order (when not returning a `MixingRulePartialDerivatives` dict) is:

        `(d_kro_d_kro_w, d_kro_d_kro_g, d_kro_d_krw, d_kro_d_krg,
           d_kro_d_sw_explicit, d_kro_d_so_explicit, d_kro_d_sg_explicit)`

        :param kro_w: Two-phase oil kr from oil-water table.
        :param kro_g: Two-phase oil kr from gas-oil table.
        :param krw: Two-phase water kr from oil-water table.
        :param krg: Two-phase gas kr from gas-oil table.
        :param kro_endpoint: Oil kr at connate water (normalization reference).
        :param water_saturation: Current water saturation.
        :param oil_saturation: Current oil saturation.
        :param gas_saturation: Current gas saturation.
        :return: The seven partial derivatives, either as a
            `MixingRulePartialDerivatives` dict or a 7-tuple.
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


class CapillaryPressures(TypedDict):
    """Dictionary containing capillary pressures for different phase pairs."""

    oil_water: FloatOrArray  # Pcow = Po - Pw
    gas_oil: FloatOrArray  # Pcgo = Pg - Po


class CapillaryPressureDerivatives(TypedDict):
    """Dictionary containing capillary pressure derivatives for different phase pairs."""

    dPcow_dSw: FloatOrArray
    dPcow_dSo: FloatOrArray
    dPcgo_dSg: FloatOrArray
    dPcgo_dSo: FloatOrArray


class Wettability(enum.Enum):
    """Enum representing the wettability type of the reservoir rock."""

    WATER_WET = "water_wet"
    OIL_WET = "oil_wet"
    MIXED_WET = "mixed_wet"

    def __str__(self) -> str:
        return self.value


Kcon = typing.TypeVar("Kcon", contravariant=True)
Vcon = typing.TypeVar("Vcon", contravariant=True)


GasZFactorMethod = typing.Literal["papay", "hall-yarborough", "dak"]
